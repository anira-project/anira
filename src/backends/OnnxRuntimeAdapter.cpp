/*
 * The ONNX Runtime adapter: one environment per process, on the engine object, created at
 * the engine's init with anira's log level (ONNX Runtime keeps one environment per process
 * behind every handle; the web build creates it with a global thread pool of one thread,
 * since a WebAssembly session cannot spawn its own), one session-options object per loaded
 * model, shared by every executor, and one Ort::Session per executor (kept per
 * executor, so that no two executors share a session's allocator), the slots bound to the
 * graph's inputs and outputs by name where the entry's tensors record or the canonical name
 * matches one and by graph position otherwise, checked against the graph's shapes at load,
 * and every tensor of a call bound over anira's memory: Ort::Value tensors created per call
 * over the descriptors of the context, the outputs pre-created over the output descriptors'
 * memory and handed to Session::Run, so no result is copied. File-local over
 * anira::backend::Model: nothing of ONNX Runtime enters a public header.
 */
#include <anira/CoreConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/Logger.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../capi/words.h"
#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

namespace anira::backend {

namespace {

constexpr const char* k_engine = anira::capi::engine_word(ANIRA_ENGINE_ONNXRUNTIME);

// Maps anira's log level to the severity of the ONNX Runtime environment.
// Debug maps to VERBOSE, ONNX Runtime's most detailed severity.
OrtLoggingLevel to_ort_logging_level(anira::LogLevel log_level) {
    switch (log_level) {
        case anira::LogLevel::Debug: return ORT_LOGGING_LEVEL_VERBOSE;
        case anira::LogLevel::Info: return ORT_LOGGING_LEVEL_INFO;
        case anira::LogLevel::Warning: return ORT_LOGGING_LEVEL_WARNING;
        case anira::LogLevel::Error: return ORT_LOGGING_LEVEL_ERROR;
    }
    return ORT_LOGGING_LEVEL_WARNING;
}

// If ONNX Runtime's symbols leak out of the module embedding anira (misconfigured
// visibility) and the host process has loaded a different ONNX Runtime, the
// dynamic linker can bind OrtGetApiBase to the host's runtime. GetApi() with
// our (newer) ORT_API_VERSION then returns null and the first Ort:: call
// crashes the host. Detect that here and fail with a diagnosable error
// instead; the throw propagates out of prepare.
void throw_if_foreign_onnxruntime() {
    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == nullptr || api_base->GetApi(ORT_API_VERSION) == nullptr) {
        throw std::runtime_error(
            "anira: OrtGetApiBase resolved to an ONNX Runtime that does not "
            "support the API version anira was built against. A different "
            "ONNX Runtime is already loaded in this process (e.g. shipped by "
            "the host application) and ONNX Runtime's symbols were not kept private "
            "to the module embedding anira. Link ONNX Runtime only through "
            "anira::onnxruntime and compile the translation units that include "
            "its headers with hidden visibility (see the troubleshooting guide).");
    }
}

// The anira dtype of an ONNX element type; 0 for one anira has no code for.
anira_dtype dtype_of(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return ANIRA_DTYPE_F32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ANIRA_DTYPE_F64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return ANIRA_DTYPE_F16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return ANIRA_DTYPE_BF16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return ANIRA_DTYPE_I8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return ANIRA_DTYPE_U8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return ANIRA_DTYPE_I16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return ANIRA_DTYPE_I32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return ANIRA_DTYPE_I64;
        default: return 0;
    }
}

// The tensors of one side of the graph as the session describes them: the name, the shape
// with -1 for a symbolic (dynamic) extent, the element type.
std::vector<EngineTensor> side_of(const Ort::Session& session,
                                  bool inputs,
                                  Ort::AllocatorWithDefaultOptions& allocator) {
    const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<EngineTensor> tensors;
    tensors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        EngineTensor tensor;
        const Ort::AllocatedStringPtr name = inputs ? session.GetInputNameAllocated(i, allocator)
                                                    : session.GetOutputNameAllocated(i, allocator);
        tensor.m_name = name.get();
        // The type info owns what its tensor view reads: it lives for the whole block.
        const Ort::TypeInfo info =
            inputs ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        if (info.GetONNXType() == ONNX_TYPE_TENSOR) {
            const Ort::ConstTensorTypeAndShapeInfo shape = info.GetTensorTypeAndShapeInfo();
            tensor.m_dims = shape.GetShape();
            const ONNXTensorElementDataType type = shape.GetElementType();
            tensor.m_dtype = dtype_of(type);
            tensor.m_type_word = "ONNX element type " + std::to_string(static_cast<int>(type));
        } else {
            tensor.m_type_word = "not a tensor (ONNX type " +
                                 std::to_string(static_cast<int>(info.GetONNXType())) + ")";
        }
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

// The registered names of ONNX Runtime's execution providers (its constants.h) that spell a
// provider of anira's enum; every other name travels as it is, the runtime's own word, in
// provider_id. Vulkan has no ONNX Runtime execution provider.
constexpr std::array<std::pair<const char*, anira_provider>, 6> k_provider_names{{
    {"CPUExecutionProvider", ANIRA_PROVIDER_DEFAULT},
    {"CUDAExecutionProvider", ANIRA_PROVIDER_CUDA},
    {"DmlExecutionProvider", ANIRA_PROVIDER_DIRECTML},
    {"CoreMLExecutionProvider", ANIRA_PROVIDER_COREML},
    {"WebGpuExecutionProvider", ANIRA_PROVIDER_WEBGPU},
    {"XnnpackExecutionProvider", ANIRA_PROVIDER_XNNPACK},
}};

// The registered name of the record's provider: the table's for a provider of the enum, the
// provider_id itself (a registered name the capabilities listed) for a custom one; empty for
// the default provider and for Vulkan.
std::string registered_name(anira_provider provider, std::string_view provider_id) {
    if (!provider_id.empty()) { return std::string(provider_id); }
    for (const auto& [name, value] : k_provider_names) {
        if (value == provider && value != ANIRA_PROVIDER_DEFAULT) { return name; }
    }
    return "";
}

// Appends the record's provider to the session options, ahead of the CPU provider ONNX Runtime
// keeps last: CUDA through its own entry, every other through the generic one under its
// registered name (the generic entry takes the registered names as well as the short ones).
// A provider the runtime refuses (not in this build, no device) is ANIRA_ERROR_NOT_SUPPORTED
// with the runtime's text.
void append_provider(Ort::SessionOptions& options, const Model& model) {
    if (model.m_provider == ANIRA_PROVIDER_DEFAULT && model.m_provider_id.empty()) { return; }
    const std::string label = anira::capi::provider_label(model.m_provider, model.m_provider_id);
    const std::string name = registered_name(model.m_provider, model.m_provider_id);
    if (name.empty()) {
        throw StatusError(ANIRA_ERROR_NOT_SUPPORTED,
                          "onnxruntime: no execution provider serves '" + label + "'");
    }
    // The options of the record (the "provider_options" context extension's set for this
    // backend), in the runtime's vocabulary: CUDA's V2 options take them by key, the generic
    // entry as its map.
    std::vector<const char*> keys;
    std::vector<const char*> values;
    std::unordered_map<std::string, std::string> map;
    for (const auto& [key, value] : model.m_options) {
        keys.push_back(key.c_str());
        values.push_back(value.c_str());
        map[key] = value;
    }
    try {
        if (model.m_provider == ANIRA_PROVIDER_CUDA) {
            const OrtApi& api = Ort::GetApi();
            OrtCUDAProviderOptionsV2* cuda = nullptr;
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
            const std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)>
                guard(cuda, api.ReleaseCUDAProviderOptions);
            if (!keys.empty()) {
                Ort::ThrowOnError(
                    api.UpdateCUDAProviderOptions(cuda, keys.data(), values.data(), keys.size()));
            }
            options.AppendExecutionProvider_CUDA_V2(*cuda);
        } else {
            options.AppendExecutionProvider(name, map);
        }
    } catch (const Ort::Exception& e) {
        throw StatusError(ANIRA_ERROR_NOT_SUPPORTED,
                          "onnxruntime: the runtime refused execution provider '" + label + "' (" +
                              name + "): " + e.what());
    }
}

/// The engine object: the environment every session of every loaded model is created over,
/// made at the engine's init with anira's log level as its logger's severity (ONNX Runtime
/// keeps one environment per process behind every handle, and the first handle's logger is
/// the process's, so the level in effect at init is the one the runtime logs at; the web
/// build gives it a global thread pool of one intra-op thread, since a WebAssembly session
/// cannot spawn its own). Every loaded model holds the object, so the environment outlives
/// every session; the core keeps one object per engine (Core::builtin_engine).
class OnnxRuntimeEngine final : public BuiltinEngine {
public:
    OnnxRuntimeEngine() : BuiltinEngine(ANIRA_ENGINE_ONNXRUNTIME), m_env(nullptr) {}

    /// The environment init created. Throws ANIRA_ERROR_INVALID_STATE before init.
    const Ort::Env& env() const {
        if (!initialised()) {
            throw StatusError(ANIRA_ERROR_INVALID_STATE,
                              "onnxruntime: the engine was never initialised (init runs before "
                              "load)");
        }
        return m_env;
    }

protected:
    void do_init(const anira_init_info& info) override {
        throw_if_foreign_onnxruntime();
        const OrtLoggingLevel severity =
            to_ort_logging_level(static_cast<anira::LogLevel>(info.log_level));
        try {
#ifdef USE_ANIRA_WEB
            // No session-owned threads in WebAssembly: the environment carries a global thread
            // pool of one intra-op thread the sessions run on. The environment copies the
            // threading options, which die with this scope.
            Ort::ThreadingOptions threading;
            threading.SetGlobalIntraOpNumThreads(1);
            m_env = Ort::Env(threading, severity, "Default");
#else
            m_env = Ort::Env(severity, "Default");
#endif
        } catch (const Ort::Exception& e) {
            throw StatusError(
                ANIRA_ERROR_ENGINE,
                std::string("onnxruntime: the environment could not be created: ") + e.what());
        }
    }

    std::vector<ProviderInfo> runtime_providers() const override { return onnxruntime_providers(); }

private:
    Ort::Env m_env;  ///< null until init
};

/// What one load shares between its executors: the engine object, whose environment every
/// session is created over, and the session options (one intra-op thread, the record's
/// provider appended), which a session reads at its creation on the control thread and never
/// again. Immutable after load, held by the loaded model and by every executor, freed with the
/// last of them; the object outlives them all.
struct SharedEnvironment {
    SharedEnvironment(std::shared_ptr<const OnnxRuntimeEngine> engine, const Model& model);

    /// The environment of the engine object.
    const Ort::Env& env() const { return m_engine->env(); }

    std::shared_ptr<const OnnxRuntimeEngine> m_engine;
    Ort::SessionOptions m_options;
};

SharedEnvironment::SharedEnvironment(std::shared_ptr<const OnnxRuntimeEngine> engine,
                                     const Model& model)
    : m_engine(std::move(engine)) {
    m_options.SetIntraOpNumThreads(1);
    append_provider(m_options, model);
}

/// One session of the model over the shared environment: what one executor of the adapter
/// runs on. Loads at construction (a file or the bytes of the record), binds afterwards, and
/// runs every inference over the context's descriptors.
class Instance final : public Executor {
public:
    Instance(std::shared_ptr<const SharedEnvironment> shared, const Model& model);
    ~Instance() override;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The graph's tensors of either side, in graph order.
    std::vector<EngineTensor> inputs() { return side_of(*m_session, true, m_allocator); }
    std::vector<EngineTensor> outputs() { return side_of(*m_session, false, m_allocator); }

    /// Keeps, per slot, the name of the graph tensor it binds to and the engine dims of the
    /// record: Session::Run matches every value to the name beside it, so the call keeps the
    /// slot order on both sides.
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The FIXED warm-up of the record, over this instance's own scratch memory and through
    /// the process path; a failing inference is StatusError(ANIRA_ERROR_ENGINE) with ONNX
    /// Runtime's text.
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing run.
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;

private:
    /// Binds every descriptor of the context and runs; throws StatusError for a descriptor
    /// it cannot bind and Ort::Exception for a failing run.
    void run(const anira_engine_ctx& ctx);

    Ort::MemoryInfo m_memory_info;
    std::shared_ptr<const SharedEnvironment> m_shared;  ///< the environment and the options
    Ort::AllocatorWithDefaultOptions m_allocator;
    std::unique_ptr<Ort::Session> m_session;

    std::vector<Ort::AllocatedStringPtr> m_name_store;  ///< every graph name, owned
    std::vector<const char*> m_input_names;             ///< per input slot: its graph name
    std::vector<const char*> m_output_names;            ///< per output slot
    std::vector<std::vector<int64_t>> m_input_dims;     ///< per input slot: the engine dims
    std::vector<std::vector<int64_t>> m_output_dims;
    std::vector<size_t> m_input_elements;
    std::vector<size_t> m_output_elements;
    std::vector<Ort::Value> m_inputs;   ///< per input slot, created over the descriptor per call
    std::vector<Ort::Value> m_outputs;  ///< per output slot, likewise
};

Instance::Instance(std::shared_ptr<const SharedEnvironment> shared, const Model& model)
    : m_memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
    , m_shared(std::move(shared)) {
    if (model.m_bytes != nullptr) {
        try {
            m_session = std::make_unique<Ort::Session>(m_shared->env(),
                                                       model.m_bytes,
                                                       model.m_num_bytes,
                                                       m_shared->m_options);
        } catch (const Ort::Exception& e) {
            throw StatusError(ANIRA_ERROR_MODEL_LOAD,
                              model_file::message(k_engine, model_file::k_memory, e.what()));
        }
    } else {
        const std::string modelpath = model_file::require_readable(model.m_path, k_engine);
#ifdef _WIN32
        // ORT's Windows API takes a wide path; the message keeps the UTF-8 one.
        const std::wstring ort_path(modelpath.begin(), modelpath.end());
#else
        const std::string& ort_path = modelpath;
#endif
        try {
            m_session = std::make_unique<Ort::Session>(m_shared->env(),
                                                       ort_path.c_str(),
                                                       m_shared->m_options);
        } catch (const Ort::Exception& e) {
            throw StatusError(ANIRA_ERROR_MODEL_LOAD,
                              model_file::message(k_engine, modelpath, e.what()));
        }
    }
}

Instance::~Instance() {
    // Reseting the session here is very important otherwise new models might not be loaded
    // correctly
    m_session.reset();
}

void Instance::bind(const Model& model,
                    const std::vector<SlotBinding>& inputs,
                    const std::vector<SlotBinding>& outputs) {
    m_name_store.clear();
    m_input_names.clear();
    m_output_names.clear();
    m_input_dims.clear();
    m_output_dims.clear();
    m_input_elements.clear();
    m_output_elements.clear();
    m_inputs.clear();
    m_outputs.clear();
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        m_name_store.push_back(m_session->GetInputNameAllocated(inputs[slot].m_index, m_allocator));
        m_input_names.push_back(m_name_store.back().get());
        m_input_dims.push_back(model.m_inputs[slot].m_dims);
        m_input_elements.push_back(model.m_inputs[slot].m_num_elements);
        m_inputs.emplace_back(nullptr);
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        m_name_store.push_back(
            m_session->GetOutputNameAllocated(outputs[slot].m_index, m_allocator));
        m_output_names.push_back(m_name_store.back().get());
        m_output_dims.push_back(model.m_outputs[slot].m_dims);
        m_output_elements.push_back(model.m_outputs[slot].m_num_elements);
        m_outputs.emplace_back(nullptr);
    }
}

void Instance::run(const anira_engine_ctx& ctx) {
    if (ctx.inputs == nullptr || ctx.outputs == nullptr || ctx.num_inputs != m_inputs.size() ||
        ctx.num_outputs != m_outputs.size()) {
        throw StatusError(
            ANIRA_ERROR_INVALID_ARGUMENT,
            std::string(k_engine) + ": the context carries " + std::to_string(ctx.num_inputs) +
                " inputs and " + std::to_string(ctx.num_outputs) + " outputs for a model of " +
                std::to_string(m_inputs.size()) + " and " + std::to_string(m_outputs.size()));
    }
    // Every tensor over the memory of its descriptor, read on this call: the inputs as the
    // engine reads them, the outputs as the values the engine writes into.
    for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
        float* data = host_f32_packed(ctx.inputs[slot], m_input_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_input_elements[slot]) + " elements");
        }
        m_inputs[slot] = Ort::Value::CreateTensor<float>(m_memory_info,
                                                         data,
                                                         m_input_elements[slot],
                                                         m_input_dims[slot].data(),
                                                         m_input_dims[slot].size());
    }
    for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
        float* data = host_f32_packed(ctx.outputs[slot], m_output_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_output_elements[slot]) + " elements");
        }
        m_outputs[slot] = Ort::Value::CreateTensor<float>(m_memory_info,
                                                          data,
                                                          m_output_elements[slot],
                                                          m_output_dims[slot].data(),
                                                          m_output_dims[slot].size());
    }
    m_session->Run(Ort::RunOptions{nullptr},
                   m_input_names.data(),
                   m_inputs.data(),
                   m_inputs.size(),
                   m_output_names.data(),
                   m_outputs.data(),
                   m_outputs.size());
}

void Instance::warm_up(uint32_t iterations) {
    if (iterations == 0) { return; }
    // Scratch of the record's element counts, described as the engine dims: zeros in, the
    // results discarded.
    std::vector<std::vector<float>> input_scratch(m_inputs.size());
    std::vector<std::vector<float>> output_scratch(m_outputs.size());
    std::vector<anira_tensor> inputs(m_inputs.size());
    std::vector<anira_tensor> outputs(m_outputs.size());
    for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
        input_scratch[slot].assign(m_input_elements[slot], 0.F);
        anira_tensor_init_host(&inputs[slot],
                               input_scratch[slot].data(),
                               ANIRA_DTYPE_F32,
                               static_cast<uint32_t>(m_input_dims[slot].size()),
                               m_input_dims[slot].data());
    }
    for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
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
        try {
            run(ctx);
        } catch (const Ort::Exception& e) {
            // A warm-up that fails fails prepare: the model cannot run.
            throw StatusError(ANIRA_ERROR_ENGINE,
                              model_file::message(k_engine, "warm-up", e.what()));
        }
    }
}

anira_status Instance::process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept {
    try {
        run(ctx);
        m_failures.succeeded();
        return ANIRA_OK;
    } catch (const StatusError& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_backend_onnx, "%s", e.what());
        }
        return e.status();
    } catch (const std::exception& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_backend_onnx, "%s", e.what());
        }
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_backend_onnx,
                               "onnxruntime threw a non-std exception out of Session::Run");
        }
        return ANIRA_ERROR_ENGINE;
    }
}

/// The loaded model: the environment handle and the options every executor shares
/// (SharedEnvironment), the binding read off a probe session (every session loads the same
/// file), the probe the executor of shared slot 0 or, for a record without a shared slot, the
/// spare of the first exclusive session; every executor a session of its own.
class OnnxRuntimeLoaded final : public ExecutorLoaded {
public:
    explicit OnnxRuntimeLoaded(std::shared_ptr<OnnxRuntimeEngine> engine)
        : ExecutorLoaded(engine), m_engine(std::move(engine)) {}

    /// The default provider always; a provider of the enum or a registered name when the
    /// runtime lists it among its available execution providers (the engine object's list;
    /// never Vulkan, which ONNX Runtime has no provider for).
    bool serves(anira_provider provider, std::string_view provider_id) const noexcept override {
        if (provider == ANIRA_PROVIDER_DEFAULT && provider_id.empty()) { return true; }
        if (provider == ANIRA_PROVIDER_VULKAN) { return false; }
        try {
            for (const ProviderInfo& info : m_engine->providers()) {
                if (info.m_provider == provider && info.m_provider_id == provider_id) {
                    return true;
                }
            }
        } catch (...) {  // NOLINT(bugprone-empty-catch) an unlisted provider is the answer
        }
        return false;
    }

    std::string provider_reason() const override {
        std::string reason =
            "the ONNX Runtime of this build reports these execution providers "
            "beyond the CPU: ";
        std::string listed;
        try {
            for (const ProviderInfo& info : m_engine->providers()) {
                if (info.m_provider == ANIRA_PROVIDER_DEFAULT && info.m_provider_id.empty()) {
                    continue;  // the default provider, listed first
                }
                if (!listed.empty()) { listed += ", "; }
                listed += anira::capi::provider_label(info.m_provider, info.m_provider_id);
            }
        } catch (...) {  // NOLINT(bugprone-empty-catch) the list stays as far as it got
        }
        reason += listed.empty() ? "none" : listed;
        reason += " (and it has no Vulkan provider)";
        return reason;
    }

protected:
    void do_load(const Model& model) override {
        require_f32(model, k_engine);
        m_shared = std::make_shared<const SharedEnvironment>(m_engine, model);
        auto probe = std::make_unique<Instance>(m_shared, model);
        const std::vector<EngineTensor> inputs = probe->inputs();
        const std::vector<EngineTensor> outputs = probe->outputs();
        m_input_bindings =
            bind_side(model.m_inputs, inputs, inputs.size(), ExtentRule::Exact, k_engine, "input");
        m_output_bindings = bind_side(model.m_outputs,
                                      outputs,
                                      outputs.size(),
                                      ExtentRule::Exact,
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
    std::shared_ptr<OnnxRuntimeEngine> m_engine;        ///< the core's, held for the model's life
    std::shared_ptr<const SharedEnvironment> m_shared;  ///< freed with the last executor over it
    std::vector<SlotBinding> m_input_bindings;
    std::vector<SlotBinding> m_output_bindings;
};

}  // namespace

std::shared_ptr<BuiltinEngine> make_onnxruntime_engine() {
    return std::make_shared<OnnxRuntimeEngine>();
}

std::shared_ptr<Loaded> make_onnxruntime_loaded(const std::shared_ptr<BuiltinEngine>& engine) {
    std::shared_ptr<OnnxRuntimeEngine> own = std::dynamic_pointer_cast<OnnxRuntimeEngine>(engine);
    if (own == nullptr) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          "onnxruntime: the engine object is not this adapter's");
    }
    return std::make_shared<OnnxRuntimeLoaded>(std::move(own));
}

std::vector<ProviderInfo> onnxruntime_providers() {
    std::vector<ProviderInfo> providers;
    try {
        throw_if_foreign_onnxruntime();
        for (const std::string& name : Ort::GetAvailableProviders()) {
            ProviderInfo info;
            info.m_provider_id = name;
            for (const auto& [registered, value] : k_provider_names) {
                if (name == registered) {
                    info.m_provider = value;
                    info.m_provider_id.clear();
                    break;
                }
            }
            if (info.m_provider == ANIRA_PROVIDER_DEFAULT && info.m_provider_id.empty()) {
                continue;  // the CPU provider: the default provider, listed by the caller
            }
            providers.push_back(std::move(info));
        }
    } catch (const std::exception& error) {
        ANIRA_LOG_WARNING(anira::log_group::k_backend_onnx,
                          "onnxruntime: the runtime could not be asked for its execution "
                          "providers (%s); the capabilities list the default provider alone",
                          error.what());
        providers.clear();
    }
    return providers;
}

}  // namespace anira::backend
