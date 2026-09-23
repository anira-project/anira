/*
 * The ExecuTorch adapter: one data loader and one program (the .pte parsed once) per loaded
 * model, shared by every executor, and one Module over that program with its own loaded entry
 * method per executor (the runtime's Module is not thread-safe; its constructor over a shared
 * program is the documented way to run one program from several), the slots bound to the
 * method's inputs and outputs by position (the method meta carries no tensor names) and
 * checked against its tensor metas at load (the meta reports the planned upper bound of every
 * axis and marks no axis dynamic, so an extent at or below the bound matches), every input of
 * a call copied into the executor's staging tensors (the program's inputs are the runtime's
 * own) and every memory-planned output copied out into the descriptor's memory. The XNNPACK
 * delegate of every method is loaded with its runtime options set to no shared workspace and
 * no weight cache (the process-wide mutexes the prebuilt runtime otherwise takes around every
 * call), and it captures its thread pool at that load under NoThreadPoolGuard, so every call
 * runs inline on the inference thread that made it. File-local over anira::backend::Model:
 * nothing of ExecuTorch enters a public header.
 */
#ifdef USE_EXECUTORCH

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../capi/words.h"
#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

// IWYU pragma: begin_keep — the ExecuTorch headers are compiled as SYSTEM includes,
// where misc-include-cleaner cannot attribute the used symbols to their providers.
#include "executorch/extension/data_loader/buffer_data_loader.h"
#include "executorch/extension/data_loader/file_data_loader.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor_ptr.h"
#include "executorch/extension/tensor/tensor_ptr_maker.h"
#include "executorch/extension/threadpool/threadpool_guard.h"
#include "executorch/runtime/backend/backend_options_map.h"
#include "executorch/runtime/backend/interface.h"
#include "executorch/runtime/backend/options.h"
#include "executorch/runtime/core/data_loader.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/evalue.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/result.h"
#include "executorch/runtime/core/tag.h"
#include "executorch/runtime/executor/method_meta.h"
#include "executorch/runtime/executor/program.h"
// IWYU pragma: end_keep

namespace anira::backend {

namespace {

constexpr const char* k_engine = "executorch";

// The XNNPACK delegate's backend id and its workspace sharing mode "disabled", as
// backends/xnnpack/runtime/XNNPACKBackend.h of the pinned release spells them (the prebuilt
// package ships no header of the delegate; the option keys are string literals below, since
// BackendOptions::set_option takes an array). The delegate reads both options at its init,
// from the map the method is loaded with: the sharing mode (0: a workspace per delegate
// instance; 1: one per model; 2: one per process) and whether the weights are unpacked into
// the process-wide cache. Both off, two executors never meet on the mutexes the shared
// workspace and the cache take around every call, at the price of a workspace and an unpacked
// copy of the weights per executor.
constexpr const char* k_xnnpack_backend = "XnnpackBackend";
constexpr int k_xnnpack_workspace_per_instance = 0;

// The names ExecuTorch's backends register under that spell a provider of anira's enum (the
// delegate an export is lowered to, as the .pte names it); every other registered backend
// travels as it is, the runtime's own word, in provider_id.
constexpr std::array<std::pair<const char*, anira_provider>, 3> k_backend_names{{
    {k_xnnpack_backend, ANIRA_PROVIDER_XNNPACK},
    {"CoreMLBackend", ANIRA_PROVIDER_COREML},
    {"VulkanBackend", ANIRA_PROVIDER_VULKAN},
}};

// The registered name of a provider: the table's for one of the enum, the provider_id itself
// (a registered name the capabilities listed) for a custom one; empty for the default provider
// and for a provider of the enum no backend spells.
std::string backend_name_of(anira_provider provider, std::string_view provider_id) {
    if (!provider_id.empty()) { return std::string(provider_id); }
    for (const auto& [name, value] : k_backend_names) {
        if (value == provider) { return name; }
    }
    return "";
}

// The backends registered to this runtime and available, by their registered names.
std::vector<std::string> registered_backends() {
    std::vector<std::string> names;
    const size_t count = executorch::runtime::get_num_registered_backends();
    for (size_t i = 0; i < count; ++i) {
        const auto name = executorch::runtime::get_backend_name(i);
        if (!name.ok() || name.get() == nullptr) { continue; }
        const executorch::runtime::BackendInterface* backend =
            executorch::runtime::get_backend_class(name.get());
        if (backend == nullptr || !backend->is_available()) { continue; }
        names.emplace_back(name.get());
    }
    return names;
}

std::string backends_list(const std::vector<std::string>& names) {
    std::string text;
    for (const std::string& name : names) {
        if (!text.empty()) { text += ", "; }
        text += name;
    }
    return text.empty() ? "none" : text;
}

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

/// What one load shares between its executors: the data loader over the file or the bytes of
/// the record (which the record's owner keeps alive for the loaded model's life) and the
/// program parsed from it, immutable once loaded; the runtime asks that the loader outlive the
/// program, and every Module over the program (Module's constructor over a shared program)
/// dies before this does. Held by the loaded model and by every executor, freed with the last
/// of them.
struct SharedProgram {
    explicit SharedProgram(const Model& model);
    ~SharedProgram() = default;
    SharedProgram(const SharedProgram&) = delete;
    SharedProgram& operator=(const SharedProgram&) = delete;
    SharedProgram(SharedProgram&&) = delete;
    SharedProgram& operator=(SharedProgram&&) = delete;

    std::string m_where;  ///< the model path or "memory", for the failure messages
    std::unique_ptr<executorch::runtime::DataLoader> m_loader;  ///< declared first: dies last
    std::shared_ptr<executorch::runtime::Program> m_program;
};

SharedProgram::SharedProgram(const Model& model) {
    if (model.m_bytes != nullptr) {
        // BufferDataLoader keeps a pointer into the caller's buffer.
        m_where = model_file::k_memory;
        m_loader = std::make_unique<executorch::extension::BufferDataLoader>(model.m_bytes,
                                                                             model.m_num_bytes);
    } else {
        m_where = model_file::require_readable(model.m_path, k_engine);
        auto loader = executorch::extension::FileDataLoader::from(m_where.c_str());
        executorch_check(loader.error(), "FileDataLoader::from", ANIRA_ERROR_MODEL_LOAD, m_where);
        m_loader = std::make_unique<executorch::extension::FileDataLoader>(std::move(loader.get()));
    }
    // The program: parsed and verified once, shared by every Module over it.
    auto program = executorch::runtime::Program::load(m_loader.get());
    executorch_check(program.error(), "Program::load", ANIRA_ERROR_MODEL_LOAD, m_where);
    m_program = std::make_shared<executorch::runtime::Program>(std::move(program.get()));
}

/// One Module over the shared program with its own loaded entry method: what one executor of
/// the adapter runs on. Loads the method at construction (the delegates initialised with the
/// runtime options above, under the thread-pool guard; the planned memory allocated); binds
/// afterwards; runs every inference over the context's descriptors through its own staging
/// tensors.
class Instance final : public Executor {
public:
    Instance(std::shared_ptr<const SharedProgram> shared, const Model& model);
    ~Instance() override = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The method's tensors of either side as the meta describes them, in the method's order.
    std::vector<EngineTensor> inputs();
    std::vector<EngineTensor> outputs();
    /// The backends the method uses (the delegates the export was lowered to), by their
    /// registered names; empty for a portable export.
    std::vector<std::string> backends();
    /// The method executed per inference.
    const std::string& method() const noexcept { return m_method; }

    /// Builds the staging tensors once, one per input slot wrapping instance-owned host memory
    /// of the record's element count and engine dims, in the method's positional order.
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The FIXED warm-up of the record over the staging tensors (zeros); a failing execute is
    /// StatusError(ANIRA_ERROR_ENGINE) with the error named.
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing
    /// execute or a result of another shape than the record's.
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;

private:
    /// Copies every input descriptor into its staging tensor, executes and copies every result
    /// into its output descriptor; throws StatusError.
    void run(const anira_engine_ctx& ctx);

    std::shared_ptr<const SharedProgram> m_shared;            ///< the program, shared
    std::unique_ptr<executorch::extension::Module> m_module;  ///< this executor's, over it
    std::string m_method;  ///< the method executed per inference: the record's entry, else forward
    std::string m_execute_label;  ///< what a failing execute is named after, built once
    /// The XNNPACK delegate's runtime options and the map that hands them to the method's
    /// load, keyed by the delegate's id (the map keeps a view of the options: both members).
    executorch::runtime::BackendOptions<2> m_xnnpack_options;
    executorch::runtime::LoadBackendOptionsMap m_backend_options;

    std::vector<std::vector<float>> m_input_data;  ///< instance-owned host memory behind the
                                                   ///< input tensors, per input slot
    std::vector<executorch::extension::TensorPtr> m_input_tensors;  ///< wrapping m_input_data
    std::vector<executorch::runtime::EValue> m_input_values;        ///< the reusable argument list,
                                                                    ///< in the method's order
    std::vector<size_t> m_input_elements;
    std::vector<size_t> m_result_of_slot;  ///< per output slot: the result's index
    std::vector<size_t> m_output_elements;
};

Instance::Instance(std::shared_ptr<const SharedProgram> shared, const Model& model)
    : m_shared(std::move(shared))
    , m_module(std::make_unique<executorch::extension::Module>(m_shared->m_program)) {
    m_method = model.m_entry.empty() ? "forward" : model.m_entry;
    m_execute_label = "Module::execute(\"" + m_method + "\")";

    // The delegate's runtime options, read at its init (the method's load below): no shared
    // workspace, no weight cache.
    executorch_check(
        m_xnnpack_options.set_option("workspace_sharing_mode", k_xnnpack_workspace_per_instance),
        "BackendOptions::set_option(\"workspace_sharing_mode\")");
    executorch_check(m_xnnpack_options.set_option("weight_cache_enabled", false),
                     "BackendOptions::set_option(\"weight_cache_enabled\")");
    executorch_check(m_backend_options.set_options(k_xnnpack_backend, m_xnnpack_options.view()),
                     "LoadBackendOptionsMap::set_options");

    // The selected method up front: this initialises the delegates (the XNNPACK runtime
    // captures the thread pool it will run on here, and under the guard it captures none, so
    // its calls run inline on the calling thread, as the other engines' do) and allocates the
    // planned memory, none of which may happen lazily on the inference thread.
    const executorch::extension::threadpool::NoThreadPoolGuard no_threadpool;
    executorch_check(m_module->load_method(m_method, nullptr, nullptr, &m_backend_options),
                     ("Module::load_method(\"" + m_method + "\")").c_str(),
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
}

std::vector<EngineTensor> Instance::inputs() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
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
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    std::vector<EngineTensor> tensors;
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
        const auto tag = meta->output_tag(i);
        tensors.push_back(engine_tensor_of(meta->output_tensor_meta(i),
                                           tag.ok() && *tag == executorch::runtime::Tag::Tensor));
    }
    return tensors;
}

std::vector<std::string> Instance::backends() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    std::vector<std::string> names;
    for (size_t i = 0; i < meta->num_backends(); ++i) {
        const auto name = meta->get_backend_name(i);
        if (name.ok() && name.get() != nullptr) { names.emplace_back(name.get()); }
    }
    return names;
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
    const executorch::extension::threadpool::NoThreadPoolGuard no_threadpool;
    for (uint32_t i = 0; i < iterations; ++i) {
        const auto result = m_module->execute(m_method, m_input_values);
        executorch_check(result.error(), (m_execute_label + " (warm-up)").c_str());
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

    // Inline on this thread: the guard keeps every operator that asks for the thread pool
    // per call on it (the delegate captured none at its load).
    const executorch::extension::threadpool::NoThreadPoolGuard no_threadpool;
    const auto result = m_module->execute(m_method, m_input_values);
    executorch_check(result.error(), m_execute_label.c_str());
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

anira_status Instance::process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept {
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

/// The loaded model: the program every executor shares (SharedProgram, parsed once), positional
/// on both sides (the meta carries no names), checked against a probe's metas: the planned
/// upper bound of every axis, an extent at or below it matching; the probe is the executor of
/// shared slot 0 or, for a record without a shared slot, the spare of the first exclusive
/// session.
class ExecuTorchLoaded final : public ExecutorLoaded {
public:
    /// The default provider (a portable export, or whatever delegates the runtime has for the
    /// export's), or a backend registered to this runtime and available, by the name the
    /// enum's spelling maps to or the registered name itself.
    bool serves(anira_provider provider, std::string_view provider_id) const noexcept override {
        if (provider == ANIRA_PROVIDER_DEFAULT && provider_id.empty()) { return true; }
        try {
            const std::string wanted = backend_name_of(provider, provider_id);
            if (wanted.empty()) { return false; }
            const std::vector<std::string> registered = registered_backends();
            return std::ranges::find(registered, wanted) != registered.end();
        } catch (...) {  // NOLINT(bugprone-empty-catch) an unlisted backend is the answer
            return false;
        }
    }

    std::string provider_reason() const override {
        std::string reason =
            "ExecuTorch runs the backends compiled into its runtime (registered "
            "and available here: ";
        try {
            reason += backends_list(registered_backends());
        } catch (...) {  // NOLINT(bugprone-empty-catch) the list stays as far as it got
        }
        reason += "); an entry pinned to a provider names the backend its export was lowered to";
        return reason;
    }

protected:
    void do_load(const Model& model) override {
        require_f32(model, k_engine);
        m_shared = std::make_shared<const SharedProgram>(model);
        auto probe = std::make_unique<Instance>(m_shared, model);
        // A pinned entry names the backend its export was lowered to: the method must use it,
        // or the export is mislabeled and refused here rather than at its first call.
        if (model.m_provider != ANIRA_PROVIDER_DEFAULT || !model.m_provider_id.empty()) {
            const std::string wanted = backend_name_of(model.m_provider, model.m_provider_id);
            const std::vector<std::string> used = probe->backends();
            if (std::ranges::find(used, wanted) == used.end()) {
                throw StatusError(
                    ANIRA_ERROR_CONFIG,
                    "executorch: " + m_shared->m_where + ": the entry is pinned to provider '" +
                        anira::capi::provider_label(model.m_provider, model.m_provider_id) +
                        "' (backend '" + wanted + "'), but its method '" + probe->method() +
                        "' uses " +
                        (used.empty() ? "no backend" : "the backends " + backends_list(used)) +
                        "; a pin names the backend the export was lowered to");
            }
        }
        const std::vector<EngineTensor> inputs = probe->inputs();
        const std::vector<EngineTensor> outputs = probe->outputs();
        m_input_bindings = bind_side(model.m_inputs,
                                     inputs,
                                     inputs.size(),
                                     ExtentRule::UpperBound,
                                     k_engine,
                                     "input");
        m_output_bindings = bind_side(model.m_outputs,
                                      outputs,
                                      outputs.size(),
                                      ExtentRule::UpperBound,
                                      k_engine,
                                      "output");
        probe->bind(model, m_input_bindings, m_output_bindings);
        set_bindings(bindings_of(m_input_bindings, m_output_bindings));
        adopt(std::move(probe));
    }

    std::unique_ptr<Executor> make_executor() override {
        auto executor = std::make_unique<Instance>(m_shared, model());
        executor->bind(model(), m_input_bindings, m_output_bindings);
        return executor;
    }

private:
    std::shared_ptr<const SharedProgram> m_shared;  ///< freed with the last executor over it
    std::vector<SlotBinding> m_input_bindings;
    std::vector<SlotBinding> m_output_bindings;
};

}  // namespace

std::shared_ptr<Loaded> make_executorch_loaded() {
    return std::make_shared<ExecuTorchLoaded>();
}

std::vector<ProviderInfo> executorch_providers() {
    std::vector<ProviderInfo> providers;
    for (const std::string& name : registered_backends()) {
        ProviderInfo info;
        info.m_provider_id = name;
        for (const auto& [registered, value] : k_backend_names) {
            if (name == registered) {
                info.m_provider = value;
                info.m_provider_id.clear();
                break;
            }
        }
        providers.push_back(std::move(info));
    }
    return providers;
}

}  // namespace anira::backend

#endif  // USE_EXECUTORCH
