#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/backends/BackendBase.h>
#include <anira/backends/OnnxRuntimeProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/MemoryBlock.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace anira {

namespace {

// Maps anira's log level to the severity of the ONNX Runtime environment.
// Debug maps to VERBOSE, ONNX Runtime's most detailed severity.
OrtLoggingLevel to_ort_logging_level(LogLevel log_level) {
    switch (log_level) {
        case LogLevel::Debug: return ORT_LOGGING_LEVEL_VERBOSE;
        case LogLevel::Info: return ORT_LOGGING_LEVEL_INFO;
        case LogLevel::Warning: return ORT_LOGGING_LEVEL_WARNING;
        case LogLevel::Error: return ORT_LOGGING_LEVEL_ERROR;
    }
    return ORT_LOGGING_LEVEL_WARNING;
}

// If backend symbols leak out of the module embedding anira (misconfigured
// visibility) and the host process has loaded a different ONNX Runtime, the
// dynamic linker can bind OrtGetApiBase to the host's runtime. GetApi() with
// our (newer) ORT_API_VERSION then returns null and the first Ort:: call
// crashes the host. Detect that here and fail with a diagnosable error
// instead; the throw propagates out of the InferenceHandler constructor.
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

}  // namespace

// Defined here, not in the header: it owns ONNX Runtime objects, and the engine
// headers stay out of anira's public headers (see the note on BackendBase).
struct OnnxRuntimeProcessor::Instance {
    Instance(InferenceConfig& inference_config);
    ~Instance();

    void prepare();
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 const std::shared_ptr<SessionElement>& session);

    Ort::MemoryInfo m_memory_info;                 ///< Memory information for tensor allocation
    Ort::Env m_env;                                ///< ONNX Runtime environment
    Ort::AllocatorWithDefaultOptions m_ort_alloc;  ///< Default allocator for ONNX Runtime
    Ort::SessionOptions m_session_options;         ///< Session configuration options

    std::unique_ptr<Ort::Session> m_session;  ///< ONNX Runtime inference session

    std::vector<MemoryBlock<float>> m_input_data;  ///< Pre-allocated input data buffers
    std::vector<Ort::Value> m_inputs;              ///< ONNX Runtime input tensors
    std::vector<Ort::Value> m_outputs;             ///< ONNX Runtime output tensors

    std::vector<Ort::AllocatedStringPtr> m_input_name;   ///< Input tensor names (allocated
                                                         ///< strings)
    std::vector<Ort::AllocatedStringPtr> m_output_name;  ///< Output tensor names (allocated
                                                         ///< strings)

    std::vector<const char*> m_output_names;  ///< Output tensor name pointers for API calls
    std::vector<const char*> m_input_names;   ///< Input tensor name pointers for API calls

    InferenceConfig& m_inference_config;    ///< Reference to inference configuration
    std::atomic<bool> m_processing{false};  ///< Flag indicating if instance is currently
                                            ///< processing
};

OnnxRuntimeProcessor::OnnxRuntimeProcessor(InferenceConfig& inference_config)
    : BackendBase(inference_config) {
    throw_if_foreign_onnxruntime();
    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

OnnxRuntimeProcessor::~OnnxRuntimeProcessor() = default;

void OnnxRuntimeProcessor::prepare() {
    for (auto& instance : m_instances) { instance->prepare(); }
}

void OnnxRuntimeProcessor::process(std::vector<BufferF>& input,
                                   std::vector<BufferF>& output,
                                   std::shared_ptr<SessionElement> session) {
    while (true) {
        for (auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                instance->process(input, output, session);
                instance->m_processing.exchange(false);
                return;
            }
        }
    }
}

OnnxRuntimeProcessor::Instance::Instance(InferenceConfig& inference_config)
    : m_memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
#ifdef USE_ANIRA_WEB
    , m_env(nullptr)
    , m_inference_config(inference_config) {
    // Create threading options
    OrtThreadingOptions* threading_options = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateThreadingOptions(&threading_options));
    Ort::ThrowOnError(Ort::GetApi().SetGlobalIntraOpNumThreads(threading_options, 1));

    // Create environment with global threadpools
    OrtEnv* raw_env = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateEnvWithGlobalThreadPools(
        to_ort_logging_level(get_log_level()),  // Logging level
        "Default",                              // Log ID
        threading_options,                      // Threading options
        &raw_env                                // Out parameter for the raw environment
        ));

    m_env = Ort::Env(raw_env);  // Wrap the raw environment in a C++ object
#else
    , m_env(to_ort_logging_level(get_log_level()), "Default")
    , m_inference_config(inference_config) {
#endif
    m_session_options.SetIntraOpNumThreads(1);

    // Check if the model is binary
    if (m_inference_config.is_model_binary(anira::InferenceBackend::ONNX)) {
        const anira::ModelData* model_data =
            m_inference_config.get_model_data(anira::InferenceBackend::ONNX);
        assert(model_data && "Model data not found for binary model!");

        // Load model from binary data
        m_session = std::make_unique<Ort::Session>(m_env,
                                                   model_data->m_data,
                                                   model_data->m_size,
                                                   m_session_options);
    } else {
        // Load model from file path
#ifdef _WIN32
        std::string modelpath_str =
            m_inference_config.get_model_path(anira::InferenceBackend::ONNX);
        std::wstring modelpath = std::wstring(modelpath_str.begin(), modelpath_str.end());
#else
        std::string const modelpath =
            m_inference_config.get_model_path(anira::InferenceBackend::ONNX);
#endif
        m_session = std::make_unique<Ort::Session>(m_env, modelpath.c_str(), m_session_options);
    }

    m_input_names.resize(m_session->GetInputCount());
    m_output_names.resize(m_session->GetOutputCount());
    m_input_name.clear();
    m_output_name.clear();

    for (size_t i = 0; i < m_session->GetInputCount(); ++i) {
        m_input_name.emplace_back(m_session->GetInputNameAllocated(i, m_ort_alloc));
        m_input_names[i] = m_input_name[i].get();
    }
    for (size_t i = 0; i < m_session->GetOutputCount(); ++i) {
        m_output_name.emplace_back(m_session->GetOutputNameAllocated(i, m_ort_alloc));
        m_output_names[i] = m_output_name[i].get();
    }

    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());
    m_inputs.clear();
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_input_data[i].data(),
            m_input_data[i].size(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].data(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].size()));
    }

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        try {
            m_outputs = m_session->Run(Ort::RunOptions{nullptr},
                                       m_input_names.data(),
                                       m_inputs.data(),
                                       m_input_names.size(),
                                       m_output_names.data(),
                                       m_output_names.size());
        } catch (Ort::Exception& e) { LOG_ERROR << e.what() << '\n'; }
    }
}

OnnxRuntimeProcessor::Instance::~Instance() {
    // Reseting the session here is very important otherwise new models might not be loaded
    // correctly
    m_session.reset();
}

void OnnxRuntimeProcessor::Instance::prepare() {
    for (auto& i : m_input_data) { i.clear(); }
}

void OnnxRuntimeProcessor::Instance::process(std::vector<BufferF>& input,
                                             std::vector<BufferF>& output,
                                             const std::shared_ptr<SessionElement>&) {
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_inputs[i] = Ort::Value::CreateTensor<float>(
            m_memory_info,
            input[i].data(),
            input[i].get_num_samples() * input[i].get_num_channels(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].data(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].size());
    }

    try {
        m_outputs = m_session->Run(Ort::RunOptions{nullptr},
                                   m_input_names.data(),
                                   m_inputs.data(),
                                   m_input_names.size(),
                                   m_output_names.data(),
                                   m_output_names.size());
    } catch (Ort::Exception& e) { LOG_ERROR << e.what() << '\n'; }

    for (size_t i = 0; i < m_outputs.size(); i++) {
        const auto output_read_ptr = m_outputs[i].GetTensorMutableData<float>();
        for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
            output[i].get_memory_block()[j] = output_read_ptr[j];
        }
    }
}

}  // namespace anira
