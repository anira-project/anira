/*
 * The ONNX Runtime adapter: one Ort::Session per instance in its own environment, the slots
 * bound to the graph's inputs and outputs by name where the entry's tensors record or the
 * canonical name matches one and by graph position otherwise, checked against the graph's
 * shapes at prepare, and every tensor of a call bound over anira's memory: Ort::Value tensors
 * created per call over the descriptors of the context, the outputs pre-created over the
 * output descriptors' memory and handed to Session::Run, so no result is copied. File-local
 * over anira::backend::Model: nothing of ONNX Runtime enters a public header.
 */
#include <anira/CoreConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/Logger.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

namespace anira::backend {

namespace {

constexpr const char* k_engine = "onnxruntime";

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

// If backend symbols leak out of the module embedding anira (misconfigured
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
            "the host application) and backend symbols were not kept private "
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

/// One session of the model in its own environment: what one instance of the adapter runs
/// on. Loads at construction (a file or the bytes of the record), binds afterwards, and runs
/// every inference over the context's descriptors.
class Instance {
public:
    explicit Instance(const Model& model);
    ~Instance();
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
    void warm_up(uint32_t iterations);

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing run.
    anira_status process(const anira_engine_ctx& ctx) noexcept;

private:
    /// Binds every descriptor of the context and runs; throws StatusError for a descriptor
    /// it cannot bind and Ort::Exception for a failing run.
    void run(const anira_engine_ctx& ctx);

    Ort::MemoryInfo m_memory_info;
    Ort::Env m_env;
    Ort::AllocatorWithDefaultOptions m_allocator;
    Ort::SessionOptions m_session_options;
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

Instance::Instance(const Model& model)
    : m_memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
#ifdef USE_ANIRA_WEB
    , m_env(nullptr) {
    // Create threading options
    OrtThreadingOptions* threading_options = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateThreadingOptions(&threading_options));
    Ort::ThrowOnError(Ort::GetApi().SetGlobalIntraOpNumThreads(threading_options, 1));

    // Create environment with global threadpools
    OrtEnv* raw_env = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateEnvWithGlobalThreadPools(
        to_ort_logging_level(model.m_log_level),  // Logging level
        "Default",                                // Log ID
        threading_options,                        // Threading options
        &raw_env                                  // Out parameter for the raw environment
        ));

    m_env = Ort::Env(raw_env);  // Wrap the raw environment in a C++ object
#else
    , m_env(to_ort_logging_level(model.m_log_level), "Default") {
#endif
    m_session_options.SetIntraOpNumThreads(1);

    if (model.m_bytes != nullptr) {
        try {
            m_session = std::make_unique<Ort::Session>(m_env,
                                                       model.m_bytes,
                                                       model.m_num_bytes,
                                                       m_session_options);
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
            m_session = std::make_unique<Ort::Session>(m_env, ort_path.c_str(), m_session_options);
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

anira_status Instance::process(const anira_engine_ctx& ctx) noexcept {
    try {
        run(ctx);
        return ANIRA_OK;
    } catch (const StatusError& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_onnx, "%s", e.what());
        return e.status();
    } catch (const std::exception& e) {
        // Ort::Exception among them: the engine's text is logged here, unlatched; the
        // scheduler latches the status.
        ANIRA_LOG_RT_ERROR(log_group::k_backend_onnx, "%s", e.what());
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_onnx,
                           "onnxruntime threw a non-std exception out of Session::Run");
        return ANIRA_ERROR_ENGINE;
    }
}

class OnnxRuntimeAdapter final : public Adapter {
protected:
    void do_prepare(const Model& model) override {
        require_f32(model, k_engine);
        throw_if_foreign_onnxruntime();
        m_instances.clear();
        const uint32_t count = model.m_instances > 0 ? model.m_instances : 1U;
        for (uint32_t i = 0; i < count; ++i) {
            m_instances.push_back(std::make_unique<Instance>(model));
        }
        // The binding rule and the check over the first instance's graph (every instance
        // loaded the same file), then the same binding on every instance.
        Instance& first = *m_instances.front();
        const std::vector<EngineTensor> inputs = first.inputs();
        const std::vector<EngineTensor> outputs = first.outputs();
        const std::vector<SlotBinding> input_bindings =
            bind_side(model.m_inputs, inputs, inputs.size(), ExtentRule::Exact, k_engine, "input");
        const std::vector<SlotBinding> output_bindings = bind_side(model.m_outputs,
                                                                   outputs,
                                                                   outputs.size(),
                                                                   ExtentRule::Exact,
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

std::shared_ptr<Adapter> make_onnxruntime_adapter() {
    return std::make_shared<OnnxRuntimeAdapter>();
}

}  // namespace anira::backend
