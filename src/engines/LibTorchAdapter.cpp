/*
 * The LibTorch adapter: one TorchScript module per executor (a module is not shareable between
 * threads, so every executor loads its own, with its own copy of the weights; what one load
 * shares is the record and the binding tables), the entry method resolved at load, the input
 * slots bound to the method's arguments by name where the entry's tensors record or the
 * canonical name matches one and by argument position otherwise, the outputs to the method's
 * returns by position (a method returns unnamed tensors). Every input of a call is a
 * torch::from_blob view over the descriptor's memory (no deleter: the view owns nothing and
 * the struct's memory is never swapped away); the result tensors are the engine's own and are
 * copied into the output descriptors' memory. File-local over anira::engine::Model: nothing
 * of LibTorch enters a public header.
 */
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../capi/words.h"
#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

// Avoid min/max macro conflicts on Windows for LibTorch compatibility
#ifdef _WIN32
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
#endif

// LibTorch headers trigger many warnings; disabling for cleaner build logs
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#endif

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/jit_type_base.h>
#include <c10/core/ScalarType.h>  // IWYU pragma: keep - the c10 alias of the headeronly type
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/method.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/utils.h>

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace anira::engine {

namespace {

constexpr const char* k_engine = anira::capi::engine_word(ANIRA_ENGINE_LIBTORCH);

// The anira dtype of a torch scalar type; 0 for one anira has no code for.
anira_dtype dtype_of(c10::ScalarType type) {
    switch (type) {
        case c10::ScalarType::Float: return ANIRA_DTYPE_F32;
        case c10::ScalarType::Double: return ANIRA_DTYPE_F64;
        case c10::ScalarType::Half: return ANIRA_DTYPE_F16;
        case c10::ScalarType::BFloat16: return ANIRA_DTYPE_BF16;
        case c10::ScalarType::Char: return ANIRA_DTYPE_I8;
        case c10::ScalarType::Byte: return ANIRA_DTYPE_U8;
        case c10::ScalarType::Short: return ANIRA_DTYPE_I16;
        case c10::ScalarType::Int: return ANIRA_DTYPE_I32;
        case c10::ScalarType::Long: return ANIRA_DTYPE_I64;
        default: return 0;
    }
}

// "[1, 1, 512]" of a tensor's sizes.
std::string sizes_text(c10::IntArrayRef sizes) {
    std::string text = "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) { text += ", "; }
        text += std::to_string(sizes[i]);
    }
    return text + "]";
}

// One argument of the entry method after self: its name, whether its type is Tensor, whether
// it carries a default value (the call may leave it out).
struct Argument {
    std::string m_name;
    bool m_tensor = false;
    bool m_defaulted = false;
};

/// One module of the model: what one executor of the adapter runs on. Loads at construction
/// (a file or the bytes of the record) and resolves the entry method; binds afterwards; runs
/// every inference over the context's descriptors.
class Instance final : public Executor {
public:
    explicit Instance(const Model& model);
    ~Instance() override = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The method's arguments after self, in argument order.
    const std::vector<Argument>& arguments() const noexcept { return m_arguments; }

    /// How many tensors the method returns, when its schema says (a tuple of tensors or one
    /// tensor); nothing for a list, whose length only a result tells.
    std::optional<size_t> num_returns() const noexcept { return m_num_returns; }

    /// Keeps, per input slot, the argument it feeds and the engine dims of the record; per
    /// output slot, the result it copies. The bound arguments are the method's leading ones
    /// (the call passes them in argument order; a trailing argument with a default value stays
    /// with its default).
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The FIXED warm-up of the record over this instance's own scratch memory, through the
    /// process path; the first result is checked against the record (the schema carries no
    /// shapes: this is where an output's dtype and shape are known at load), a mismatch
    /// StatusError(ANIRA_ERROR_CONFIG) naming the slot and both shapes, a failing inference
    /// StatusError(ANIRA_ERROR_ENGINE) with LibTorch's text. Nothing before bind.
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing run or
    /// a result of another dtype or shape than the record's.
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;

private:
    /// Binds every input descriptor and runs the method; throws StatusError for a descriptor
    /// it cannot bind and c10::Error for a failing run.
    c10::IValue execute(const anira_engine_ctx& ctx);
    /// The result tensor of output slot `slot`, from a tuple, a tensor list or one tensor;
    /// throws StatusError(ANIRA_ERROR_ENGINE) when the result has no such tensor.
    at::Tensor result_of(const c10::IValue& result, size_t slot) const;
    /// Copies every result tensor into its output descriptor's memory, after checking its
    /// dtype and element count against the record; throws StatusError.
    void deliver(const c10::IValue& result, const anira_engine_ctx& ctx) const;

    torch::jit::Module m_module;
    std::optional<torch::jit::Method> m_method;
    std::vector<Argument> m_arguments;
    std::optional<size_t> m_num_returns;
    torch::TensorOptions m_tensor_options;
    /// The record bind was given (the loaded model's, which outlives this executor): what the
    /// warm-up checks the first result against.
    const Model* m_record = nullptr;

    std::vector<c10::IValue> m_inputs;               ///< the call's arguments, in argument order
    std::vector<size_t> m_argument_of_slot;          ///< per input slot: the argument it feeds
    std::vector<std::vector<int64_t>> m_input_dims;  ///< per input slot: the engine dims
    std::vector<size_t> m_input_elements;
    std::vector<size_t> m_result_of_slot;  ///< per output slot: the result tensor's index
    std::vector<std::vector<int64_t>> m_output_dims;
    std::vector<size_t> m_output_elements;
};

Instance::Instance(const Model& model) {
    m_tensor_options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(false);
    std::string where = model_file::k_memory;
    if (model.m_bytes != nullptr) {
        try {
            std::istringstream stream(
                std::string(static_cast<const char*>(model.m_bytes), model.m_num_bytes));
            m_module = torch::jit::load(stream);
        } catch (const c10::Error& e) {
            // A model that will not load fails prepare with the engine's own text in the
            // status message; carrying on would call eval() on an empty module.
            throw StatusError(ANIRA_ERROR_MODEL_LOAD,
                              model_file::message(k_engine, where, e.what_without_backtrace()));
        }
    } else {
        where = model_file::require_readable(model.m_path, k_engine);
        try {
            m_module = torch::jit::load(where);
        } catch (const c10::Error& e) {
            throw StatusError(ANIRA_ERROR_MODEL_LOAD,
                              model_file::message(k_engine, where, e.what_without_backtrace()));
        }
    }
    m_module.eval();

    // The entry method: the record's, else forward. A module without it cannot run.
    const std::string entry = model.m_entry.empty() ? "forward" : model.m_entry;
    m_method = m_module.find_method(entry);
    if (!m_method.has_value()) {
        throw StatusError(
            ANIRA_ERROR_MODEL_LOAD,
            model_file::message(k_engine, where, "the module has no method '" + entry + "'"));
    }
    const c10::FunctionSchema& schema = m_method->function().getSchema();
    // The arguments after self: what the slots bind to.
    const std::vector<c10::Argument>& arguments = schema.arguments();
    for (size_t i = arguments.empty() ? 0 : 1; i < arguments.size(); ++i) {
        const c10::Argument& argument = arguments[i];
        m_arguments.push_back(Argument{
            .m_name = argument.name(),
            .m_tensor =
                argument.type() != nullptr && argument.type()->kind() == c10::TypeKind::TensorType,
            .m_defaulted = argument.default_value().has_value()});
    }
    // The returns: one tensor, a tuple of tensors, or a list (the length a result tells).
    const std::vector<c10::Argument>& returns = schema.returns();
    if (returns.size() == 1 && returns[0].type() != nullptr) {
        const c10::TypePtr& type = returns[0].type();
        if (type->kind() == c10::TypeKind::TensorType) {
            m_num_returns = 1;
        } else if (type->kind() == c10::TypeKind::TupleType) {
            m_num_returns = type->expect<c10::TupleType>()->elements().size();
        }
    } else if (returns.size() > 1) {
        m_num_returns = returns.size();
    }
}

void Instance::bind(const Model& model,
                    const std::vector<SlotBinding>& inputs,
                    const std::vector<SlotBinding>& outputs) {
    m_record = &model;
    m_inputs.assign(inputs.size(), c10::IValue());
    m_argument_of_slot.clear();
    m_input_dims.clear();
    m_input_elements.clear();
    m_result_of_slot.clear();
    m_output_dims.clear();
    m_output_elements.clear();
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        m_argument_of_slot.push_back(inputs[slot].m_index);
        m_input_dims.push_back(model.m_inputs[slot].m_dims);
        m_input_elements.push_back(model.m_inputs[slot].m_num_elements);
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        m_result_of_slot.push_back(outputs[slot].m_index);
        m_output_dims.push_back(model.m_outputs[slot].m_dims);
        m_output_elements.push_back(model.m_outputs[slot].m_num_elements);
    }
}

c10::IValue Instance::execute(const anira_engine_ctx& ctx) {
    if (ctx.inputs == nullptr || ctx.outputs == nullptr ||
        ctx.num_inputs != m_argument_of_slot.size() || ctx.num_outputs != m_result_of_slot.size()) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          std::string(k_engine) + ": the context carries " +
                              std::to_string(ctx.num_inputs) + " inputs and " +
                              std::to_string(ctx.num_outputs) + " outputs for a model of " +
                              std::to_string(m_argument_of_slot.size()) + " and " +
                              std::to_string(m_result_of_slot.size()));
    }
    // Every argument a view over its descriptor's memory, read on this call: nothing is
    // copied in, nothing is owned, and the pointers are read anew every time.
    for (size_t slot = 0; slot < m_argument_of_slot.size(); ++slot) {
        float* data = host_f32_packed(ctx.inputs[slot], m_input_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_input_elements[slot]) + " elements");
        }
        m_inputs[m_argument_of_slot[slot]] =
            torch::from_blob(data, c10::IntArrayRef(m_input_dims[slot]), m_tensor_options);
    }
    if (!m_method.has_value()) {
        throw StatusError(ANIRA_ERROR_INVALID_STATE,
                          std::string(k_engine) + ": no entry method was resolved at load");
    }
    const torch::NoGradGuard no_grad;
    return (*m_method)(m_inputs);
}

at::Tensor Instance::result_of(const c10::IValue& result, size_t slot) const {
    const size_t index = m_result_of_slot[slot];
    if (result.isTuple()) {
        const auto& elements = result.toTuple()->elements();
        if (index < elements.size() && elements[index].isTensor()) {
            return elements[index].toTensor();
        }
    } else if (result.isTensorList()) {
        const auto list = result.toTensorList();
        if (index < list.size()) { return list.get(index); }
    } else if (result.isTensor() && index == 0) {
        return result.toTensor();
    }
    throw StatusError(ANIRA_ERROR_ENGINE,
                      std::string(k_engine) + ": the method returned no tensor for output slot " +
                          std::to_string(slot) + " (result " + std::to_string(index) + ")");
}

void Instance::deliver(const c10::IValue& result, const anira_engine_ctx& ctx) const {
    for (size_t slot = 0; slot < m_result_of_slot.size(); ++slot) {
        const at::Tensor tensor = result_of(result, slot);
        if (tensor.scalar_type() != c10::ScalarType::Float ||
            std::cmp_not_equal(tensor.numel(), m_output_elements[slot])) {
            throw StatusError(ANIRA_ERROR_ENGINE,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  ": the method returned a " +
                                  std::string(c10::toString(tensor.scalar_type())) + " tensor " +
                                  sizes_text(tensor.sizes()) + " for the spec's engine dims " +
                                  sizes_text(c10::IntArrayRef(m_output_dims[slot])));
        }
        float* data = host_f32_packed(ctx.outputs[slot], m_output_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_output_elements[slot]) + " elements");
        }
        // The result is the engine's own tensor: one copy into the descriptor's memory,
        // straight when the result is contiguous, through a view over the memory else.
        if (tensor.is_contiguous()) {
            std::memcpy(data,
                        tensor.const_data_ptr<float>(),
                        m_output_elements[slot] * sizeof(float));
        } else {
            torch::from_blob(data, c10::IntArrayRef(m_output_dims[slot]), m_tensor_options)
                .copy_(tensor.reshape(c10::IntArrayRef(m_output_dims[slot])));
        }
    }
}

void Instance::warm_up(uint32_t iterations) {
    if (iterations == 0 || m_record == nullptr) { return; }
    const Model& model = *m_record;
    std::vector<std::vector<float>> input_scratch(m_argument_of_slot.size());
    std::vector<std::vector<float>> output_scratch(m_result_of_slot.size());
    std::vector<anira_tensor> inputs(input_scratch.size());
    std::vector<anira_tensor> outputs(output_scratch.size());
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        input_scratch[slot].assign(m_input_elements[slot], 0.F);
        anira_tensor_init_host(&inputs[slot],
                               input_scratch[slot].data(),
                               ANIRA_DTYPE_F32,
                               static_cast<uint32_t>(m_input_dims[slot].size()),
                               m_input_dims[slot].data());
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        output_scratch[slot].assign(m_output_elements[slot], 0.F);
        anira_tensor_init_host(&outputs[slot],
                               output_scratch[slot].data(),
                               ANIRA_DTYPE_F32,
                               static_cast<uint32_t>(m_output_dims[slot].size()),
                               m_output_dims[slot].data());
    }
    anira_engine_ctx ctx{};
    ctx.num_inputs = static_cast<uint32_t>(inputs.size());
    ctx.num_outputs = static_cast<uint32_t>(outputs.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.inputs = inputs.data();
    ctx.outputs = outputs.data();
    for (uint32_t i = 0; i < iterations; ++i) {
        c10::IValue result;
        try {
            result = execute(ctx);
        } catch (const c10::Error& e) {
            // A warm-up that fails fails prepare: the model cannot run.
            throw StatusError(ANIRA_ERROR_ENGINE,
                              model_file::message(k_engine, "warm-up", e.what_without_backtrace()));
        }
        if (i == 0) {
            // The check of the binding rule on the output side, on the first result: the
            // schema carries no shapes, so this is where a result's dtype and shape meet the
            // record, at load.
            for (size_t slot = 0; slot < m_result_of_slot.size(); ++slot) {
                const at::Tensor tensor = result_of(result, slot);
                const std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
                EngineTensor engine_tensor;
                engine_tensor.m_dims = sizes;
                engine_tensor.m_dtype = dtype_of(tensor.scalar_type());
                engine_tensor.m_type_word = c10::toString(tensor.scalar_type());
                check_engine_tensor(model.m_outputs[slot],
                                    engine_tensor,
                                    ExtentRule::Exact,
                                    k_engine,
                                    "output");
            }
        }
        deliver(result, ctx);
    }
}

anira_status Instance::process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept {
    try {
        deliver(execute(ctx), ctx);
        m_failures.succeeded();
        return ANIRA_OK;
    } catch (const StatusError& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_libtorch, "%s", e.what());
        }
        return e.status();
    } catch (const c10::Error& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_libtorch, "%s", e.what_without_backtrace());
        }
        return ANIRA_ERROR_ENGINE;
    } catch (const std::exception& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_libtorch, "%s", e.what());
        }
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_libtorch,
                               "libtorch threw a non-std exception out of the method");
        }
        return ANIRA_ERROR_ENGINE;
    }
}

/// The engine object: what LibTorch keeps per process, set at the engine's init: one intra-op
/// thread (anira's inference threads are the parallelism) and c10's glog-style log level from
/// anira's (INFO=0, WARNING=1, ERROR=2, FATAL=3; c10 has no severity below INFO, so Debug and
/// Info both map to INFO). The core keeps one object per engine (Core::builtin_engine); every
/// loaded model holds it.
class LibTorchEngine final : public BuiltinEngine {
public:
    LibTorchEngine() : BuiltinEngine(ANIRA_ENGINE_LIBTORCH) {}

protected:
    void do_init(const anira_init_info& info) override {
        torch::set_num_threads(1);
        FLAGS_caffe2_log_level = std::max(static_cast<int>(info.log_level) - 1, 0);
    }
};

/// The loaded model: the binding read off a probe module (every module loads the same file),
/// the probe the executor of shared slot 0 or, for a record without a shared slot, the spare of
/// the first exclusive session; every executor loads its own module (its own functions and
/// graph executors, and its own copy of the weights).
class LibTorchLoaded final : public ExecutorLoaded {
public:
    explicit LibTorchLoaded(std::shared_ptr<LibTorchEngine> engine)
        : ExecutorLoaded(std::move(engine)) {}

    std::string provider_reason() const override {
        return "a LibTorch device needs the device-domain edges of a later pre-release (a "
               ".to(device) per call allocates), so the adapter runs the default provider "
               "alone";
    }

protected:
    void do_load(const Model& model) override {
        require_f32(model, k_engine);
        auto probe = std::make_unique<Instance>(model);
        const Instance& first = *probe;

        // The input side binds to the method's arguments after self: by name (the record's,
        // else the canonical) where one matches, by argument position otherwise; every
        // argument without a default value must be bound, and the bound ones must be the
        // method's leading arguments, since the call passes them in order and leaves a
        // trailing defaulted argument to its default.
        std::vector<std::string> names;
        size_t required = 0;
        for (const Argument& argument : first.arguments()) {
            names.push_back(argument.m_name);
            if (!argument.m_defaulted) { required = names.size(); }
        }
        m_input_bindings = bind_slots(model.m_inputs, names, required, k_engine, "input");
        for (size_t slot = 0; slot < m_input_bindings.size(); ++slot) {
            const Argument& argument = first.arguments()[m_input_bindings[slot].m_index];
            if (!argument.m_tensor) {
                throw StatusError(ANIRA_ERROR_CONFIG,
                                  std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                      " '" + model.m_inputs[slot].m_name +
                                      "' binds the method's argument '" + argument.m_name +
                                      "', which is not a tensor");
            }
            if (m_input_bindings[slot].m_index >= m_input_bindings.size()) {
                throw StatusError(ANIRA_ERROR_CONFIG,
                                  std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                      " '" + model.m_inputs[slot].m_name +
                                      "' binds the method's argument '" + argument.m_name +
                                      "' (index " + std::to_string(m_input_bindings[slot].m_index) +
                                      "), beyond the " + std::to_string(m_input_bindings.size()) +
                                      " leading arguments the model config's inputs fill; "
                                      "LibTorch is called with the method's leading arguments");
            }
        }
        // The output side binds by position to the method's returns; their count is the
        // schema's where it says (a tuple, one tensor), else the first result's.
        const size_t num_outputs = first.num_returns().value_or(model.m_outputs.size());
        m_output_bindings = bind_slots(model.m_outputs,
                                       std::vector<std::string>(num_outputs),
                                       num_outputs,
                                       k_engine,
                                       "output");
        probe->bind(model, m_input_bindings, m_output_bindings);
        // The schema carries no output shapes: the first run of the warm-up checks them
        // against the record, and a record without a warm-up gets that one run here, on the
        // probe, so a mismatch is refused at load, not at the first chunk.
        if (model.m_warm_up == 0) { probe->warm_up(1); }
        set_bindings(bindings_of(m_input_bindings, m_output_bindings));
        adopt(std::move(probe));
    }

    std::unique_ptr<Executor> make_executor() override {
        auto executor = std::make_unique<Instance>(model());
        executor->bind(model(), m_input_bindings, m_output_bindings);
        return executor;
    }

private:
    std::vector<SlotBinding> m_input_bindings;
    std::vector<SlotBinding> m_output_bindings;
};

}  // namespace

std::shared_ptr<BuiltinEngine> make_libtorch_engine() {
    return std::make_shared<LibTorchEngine>();
}

std::shared_ptr<Loaded> make_libtorch_loaded(const std::shared_ptr<BuiltinEngine>& engine) {
    std::shared_ptr<LibTorchEngine> own = std::dynamic_pointer_cast<LibTorchEngine>(engine);
    if (own == nullptr) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          "libtorch: the engine object is not this adapter's");
    }
    return std::make_shared<LibTorchLoaded>(std::move(own));
}

}  // namespace anira::engine
