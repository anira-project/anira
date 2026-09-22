/*
 * The ExecuTorch adapter: one loaded program per instance with its entry method, the slots
 * bound to the method's inputs and outputs by position (the method meta carries no tensor
 * names) and checked against its tensor metas at prepare (the meta reports the planned upper
 * bound of every axis and marks no axis dynamic, so an extent at or below the bound matches),
 * every input of a call copied into the instance's staging tensors (the program's inputs are
 * the runtime's own) and every memory-planned output copied out into the descriptor's memory.
 * File-local over anira::backend::Model: nothing of ExecuTorch enters a public header.
 */
#ifdef USE_EXECUTORCH

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

// IWYU pragma: begin_keep — the ExecuTorch headers are compiled as SYSTEM includes,
// where misc-include-cleaner cannot attribute the used symbols to their providers.
#include "executorch/extension/data_loader/buffer_data_loader.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor_ptr.h"
#include "executorch/extension/tensor/tensor_ptr_maker.h"
#include "executorch/extension/threadpool/threadpool.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/evalue.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/result.h"
#include "executorch/runtime/core/tag.h"
#include "executorch/runtime/executor/method_meta.h"
// IWYU pragma: end_keep

namespace anira::backend {

namespace {

constexpr const char* k_engine = "executorch";

// Every fallible ExecuTorch call returns a runtime::Error (or a Result carrying one). A failure
// here means a setup or runtime problem, so it becomes a StatusError with the failing call and
// the error named (executorch::runtime::to_string, e.g. "InvalidProgram"); the status is
// ANIRA_ERROR_ENGINE unless the caller passes another.
void executorch_check(executorch::runtime::Error error,
                      const char* what,
                      anira_status failure = ANIRA_ERROR_ENGINE,
                      const std::string& where = "") {
    if (error != executorch::runtime::Error::Ok) {
        const std::string text = std::string(what) +
                                 " failed with Error::" + executorch::runtime::to_string(error) +
                                 " (" + std::to_string(static_cast<uint32_t>(error)) + ")";
        throw StatusError(failure,
                          model_file::message(k_engine, where.empty() ? what : where, text));
    }
}

// Pin ExecuTorch's process-wide XNNPACK threadpool to a single thread, to match the
// other backends (anira gets its parallelism from running multiple adapter instances, and
// worker-pool fan-out is unwanted on real-time audio systems). The threadpool is a
// process-global singleton, so this is done once, not per instance.
void pin_threadpool_to_one_thread() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto* threadpool = executorch::extension::threadpool::get_threadpool();
        // _unsafe_reset_threadpool is deprecated but remains the only exported way to
        // size the pool permanently: the suggested UseNThreadsThreadPoolGuard is a
        // scoped, Meta-internal API. Resizing here is safe — no inference has run yet,
        // so no threadpool pointer is held anywhere.
        // NOLINTNEXTLINE(clang-diagnostic-deprecated-declarations)
        if (threadpool == nullptr || !threadpool->_unsafe_reset_threadpool(1)) {
            ANIRA_LOG_WARNING(log_group::k_backend_executorch,
                              "could not pin the XNNPACK threadpool to "
                              "a single thread.");
        }
    });
}

// The anira dtype of an ExecuTorch scalar type; 0 for one anira has no code for.
anira_dtype dtype_of(executorch::aten::ScalarType type) {
    switch (type) {
        case executorch::aten::ScalarType::Float: return ANIRA_DTYPE_F32;
        case executorch::aten::ScalarType::Double: return ANIRA_DTYPE_F64;
        case executorch::aten::ScalarType::Half: return ANIRA_DTYPE_F16;
        case executorch::aten::ScalarType::BFloat16: return ANIRA_DTYPE_BF16;
        case executorch::aten::ScalarType::Char: return ANIRA_DTYPE_I8;
        case executorch::aten::ScalarType::Byte: return ANIRA_DTYPE_U8;
        case executorch::aten::ScalarType::Short: return ANIRA_DTYPE_I16;
        case executorch::aten::ScalarType::Int: return ANIRA_DTYPE_I32;
        case executorch::aten::ScalarType::Long: return ANIRA_DTYPE_I64;
        default: return 0;
    }
}

// One tensor of the method as its meta describes it: no name, the planned sizes, the type.
EngineTensor engine_tensor_of(
    const executorch::runtime::Result<executorch::runtime::TensorInfo>& info,
    bool is_tensor) {
    EngineTensor tensor;
    if (!is_tensor || !info.ok()) {
        tensor.m_type_word = "not a tensor";
        return tensor;
    }
    for (const int32_t size : info->sizes()) { tensor.m_dims.push_back(size); }
    tensor.m_dtype = dtype_of(info->scalar_type());
    tensor.m_type_word =
        "ExecuTorch scalar type " + std::to_string(static_cast<int>(info->scalar_type()));
    return tensor;
}

/// One loaded program with its entry method: what one instance of the adapter runs on. Loads
/// at construction (a file or the bytes of the record, which the record's owner keeps alive
/// for the adapter's life); binds afterwards; runs every inference over the context's
/// descriptors through its own staging tensors.
class Instance {
public:
    explicit Instance(const Model& model);
    ~Instance() = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The method's tensors of either side as the meta describes them, in the method's order.
    std::vector<EngineTensor> inputs();
    std::vector<EngineTensor> outputs();

    /// Builds the staging tensors once, one per input slot wrapping instance-owned host memory
    /// of the record's element count and engine dims, in the method's positional order.
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The FIXED warm-up of the record over the staging tensors (zeros); a failing execute is
    /// StatusError(ANIRA_ERROR_ENGINE) with the error named.
    void warm_up(uint32_t iterations);

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing
    /// execute or a result of another shape than the record's.
    anira_status process(const anira_engine_ctx& ctx) noexcept;

private:
    /// Copies every input descriptor into its staging tensor, executes and copies every result
    /// into its output descriptor; throws StatusError.
    void run(const anira_engine_ctx& ctx);

    std::unique_ptr<executorch::extension::Module> m_module;  ///< the loaded .pte program
    std::string m_method;  ///< the method executed per inference: the record's entry, else forward
    std::string m_where;   ///< the model path or "memory", for the failure messages

    std::vector<std::vector<float>> m_input_data;  ///< instance-owned host memory behind the
                                                   ///< input tensors, per input slot
    std::vector<executorch::extension::TensorPtr> m_input_tensors;  ///< wrapping m_input_data
    std::vector<executorch::runtime::EValue> m_input_values;        ///< the reusable argument list,
                                                                    ///< in the method's order
    std::vector<size_t> m_input_elements;
    std::vector<size_t> m_result_of_slot;  ///< per output slot: the result's index
    std::vector<size_t> m_output_elements;
};

Instance::Instance(const Model& model) {
    pin_threadpool_to_one_thread();

    if (model.m_bytes != nullptr) {
        // BufferDataLoader keeps a pointer into the caller's buffer; the record's owner keeps
        // the bytes alive for the adapter's life.
        m_where = model_file::k_memory;
        m_module = std::make_unique<executorch::extension::Module>(
            std::make_unique<executorch::extension::BufferDataLoader>(model.m_bytes,
                                                                      model.m_num_bytes));
    } else {
        m_where = model_file::require_readable(model.m_path, k_engine);
        m_module = std::make_unique<executorch::extension::Module>(m_where);
    }

    // Load the program and the selected method up front: this parses the .pte, initializes
    // the delegates and allocates the planned memory, none of which may happen lazily on
    // the inference thread.
    m_method = model.m_entry.empty() ? "forward" : model.m_entry;
    executorch_check(m_module->load_method(m_method),
                     ("Module::load_method(\"" + m_method + "\")").c_str(),
                     ANIRA_ERROR_MODEL_LOAD,
                     m_where);
}

std::vector<EngineTensor> Instance::inputs() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(), "Module::method_meta", ANIRA_ERROR_MODEL_LOAD, m_where);
    std::vector<EngineTensor> tensors;
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
        const auto tag = meta->input_tag(i);
        tensors.push_back(engine_tensor_of(meta->input_tensor_meta(i),
                                           tag.ok() && *tag == executorch::runtime::Tag::Tensor));
    }
    return tensors;
}

std::vector<EngineTensor> Instance::outputs() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(), "Module::method_meta", ANIRA_ERROR_MODEL_LOAD, m_where);
    std::vector<EngineTensor> tensors;
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
        const auto tag = meta->output_tag(i);
        tensors.push_back(engine_tensor_of(meta->output_tensor_meta(i),
                                           tag.ok() && *tag == executorch::runtime::Tag::Tensor));
    }
    return tensors;
}

void Instance::bind(const Model& model,
                    const std::vector<SlotBinding>& inputs,
                    const std::vector<SlotBinding>& outputs) {
    // Positional on both sides: the exactly-once rule made slot i the method's input i.
    m_input_data.assign(inputs.size(), {});
    m_input_tensors.clear();
    m_input_values.clear();
    m_input_elements.clear();
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        const std::vector<int64_t>& dims = model.m_inputs[slot].m_dims;
        std::vector<executorch::aten::SizesType> sizes(dims.begin(), dims.end());
        m_input_data[slot].assign(model.m_inputs[slot].m_num_elements, 0.F);
        m_input_tensors.push_back(
            executorch::extension::from_blob(m_input_data[slot].data(),
                                             std::move(sizes),
                                             executorch::aten::ScalarType::Float));
        m_input_values.emplace_back(*m_input_tensors.back());
        m_input_elements.push_back(model.m_inputs[slot].m_num_elements);
    }
    m_result_of_slot.clear();
    m_output_elements.clear();
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        m_result_of_slot.push_back(outputs[slot].m_index);
        m_output_elements.push_back(model.m_outputs[slot].m_num_elements);
    }
}

void Instance::warm_up(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; ++i) {
        const auto result = m_module->execute(m_method, m_input_values);
        executorch_check(result.error(),
                         ("Module::execute(\"" + m_method + "\") (warm-up)").c_str());
    }
}

void Instance::run(const anira_engine_ctx& ctx) {
    if (ctx.inputs == nullptr || ctx.outputs == nullptr ||
        ctx.num_inputs != m_input_elements.size() || ctx.num_outputs != m_result_of_slot.size()) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          std::string(k_engine) + ": the context carries " +
                              std::to_string(ctx.num_inputs) + " inputs and " +
                              std::to_string(ctx.num_outputs) + " outputs for a model of " +
                              std::to_string(m_input_elements.size()) + " and " +
                              std::to_string(m_result_of_slot.size()));
    }
    // Staging in: the program's inputs are the runtime's own tensors, so every input block is
    // copied from its descriptor's memory, read on this call.
    for (size_t slot = 0; slot < m_input_elements.size(); ++slot) {
        const float* data = host_f32_packed(ctx.inputs[slot], m_input_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_input_elements[slot]) + " elements");
        }
        std::memcpy(m_input_data[slot].data(), data, m_input_elements[slot] * sizeof(float));
    }

    const auto result = m_module->execute(m_method, m_input_values);
    executorch_check(result.error(), ("Module::execute(\"" + m_method + "\")").c_str());
    const std::vector<executorch::runtime::EValue>& outputs = result.get();

    // Out of the memory-planned results, one copy each into the descriptor's memory.
    for (size_t slot = 0; slot < m_result_of_slot.size(); ++slot) {
        const size_t index = m_result_of_slot[slot];
        if (index >= outputs.size() || !outputs[index].isTensor()) {
            throw StatusError(ANIRA_ERROR_ENGINE,
                              std::string(k_engine) +
                                  ": the method returned no tensor for output slot " +
                                  std::to_string(slot) + " (result " + std::to_string(index) +
                                  " of " + std::to_string(outputs.size()) + ")");
        }
        const executorch::aten::Tensor& tensor = outputs[index].toTensor();
        if (tensor.scalar_type() != executorch::aten::ScalarType::Float ||
            std::cmp_not_equal(tensor.numel(), m_output_elements[slot])) {
            throw StatusError(ANIRA_ERROR_ENGINE,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  ": the method returned " + std::to_string(tensor.numel()) +
                                  " elements of scalar type " +
                                  std::to_string(static_cast<int>(tensor.scalar_type())) +
                                  " for the spec's " + std::to_string(m_output_elements[slot]) +
                                  " float32 elements");
        }
        float* data = host_f32_packed(ctx.outputs[slot], m_output_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_output_elements[slot]) + " elements");
        }
        std::memcpy(data, tensor.const_data_ptr<float>(), m_output_elements[slot] * sizeof(float));
    }
}

anira_status Instance::process(const anira_engine_ctx& ctx) noexcept {
    try {
        run(ctx);
        return ANIRA_OK;
    } catch (const StatusError& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_executorch, "%s", e.what());
        return e.status();
    } catch (const std::exception& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_executorch, "%s", e.what());
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_executorch,
                           "executorch threw a non-std exception out of execute");
        return ANIRA_ERROR_ENGINE;
    }
}

class ExecuTorchAdapter final : public Adapter {
protected:
    void do_prepare(const Model& model) override {
        require_f32(model, k_engine);
        m_instances.clear();
        const uint32_t count = model.m_instances > 0 ? model.m_instances : 1U;
        for (uint32_t i = 0; i < count; ++i) {
            m_instances.push_back(std::make_unique<Instance>(model));
        }
        // Positional on both sides (the meta carries no names), checked against the metas:
        // the planned upper bound of every axis, an extent at or below it matching.
        Instance& first = *m_instances.front();
        const std::vector<EngineTensor> inputs = first.inputs();
        const std::vector<EngineTensor> outputs = first.outputs();
        const std::vector<SlotBinding> input_bindings = bind_side(model.m_inputs,
                                                                  inputs,
                                                                  inputs.size(),
                                                                  ExtentRule::UpperBound,
                                                                  k_engine,
                                                                  "input");
        const std::vector<SlotBinding> output_bindings = bind_side(model.m_outputs,
                                                                   outputs,
                                                                   outputs.size(),
                                                                   ExtentRule::UpperBound,
                                                                   k_engine,
                                                                   "output");
        for (const std::unique_ptr<Instance>& instance : m_instances) {
            instance->bind(model, input_bindings, output_bindings);
            instance->warm_up(model.m_warm_up);
        }
        set_bindings(bindings_of(input_bindings, output_bindings));
    }

    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept override {
        if (ctx.instance >= m_instances.size()) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        return m_instances[ctx.instance]->process(ctx);
    }

private:
    std::vector<std::unique_ptr<Instance>> m_instances;
};

}  // namespace

std::shared_ptr<Adapter> make_executorch_adapter() {
    return std::make_shared<ExecuTorchAdapter>();
}

}  // namespace anira::backend

#endif  // USE_EXECUTORCH
