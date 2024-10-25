#include <anira/backends/TFLiteProcessor.h>

#ifdef _WIN32
#include <comdef.h>
#endif

namespace anira {

TFLiteProcessor::TFLiteProcessor(InferenceConfig& config) : BackendBase(config)
{
    for (size_t i = 0; i < m_inference_config.m_num_threads; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

TFLiteProcessor::~TFLiteProcessor() {
}

void TFLiteProcessor::prepare() {
    for(auto& instance : m_instances) {
        instance->prepare();
    }
}

void TFLiteProcessor::process(AudioBufferF& input, AudioBufferF& output) {
    while (true) {
        for(auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                instance->process(input, output);
                instance->m_processing.exchange(false);
                return;
            }
        }
    }
}

TFLiteProcessor::Instance::Instance(InferenceConfig& config) : m_inference_config(config)
{
#ifdef _WIN32
    std::string modelpath_str = m_inference_config.m_model_path_tflite;
    std::wstring modelpath = std::wstring(modelpath_str.begin(), modelpath_str.end());
#else
    std::string modelpath = m_inference_config.m_model_path_tflite;
#endif

#ifdef _WIN32
    _bstr_t modelPathChar (modelpath.c_str());
    m_model = TfLiteModelCreateFromFile(modelPathChar);
#else
    m_model = TfLiteModelCreateFromFile(modelpath.c_str());
#endif

    m_options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(m_options, 1);
    m_interpreter = TfLiteInterpreterCreate(m_model, m_options);
    // This is necessary when we have dynamic input shapes, it should be done before allocating tensors obviously
    std::vector<int> input_shape(m_inference_config.m_model_input_shape_tflite.begin(), m_inference_config.m_model_input_shape_tflite.end());
    TfLiteInterpreterResizeInputTensor(m_interpreter, 0, input_shape.data(), input_shape.size());
}

TFLiteProcessor::Instance::~Instance() {
    TfLiteInterpreterDelete(m_interpreter);
    TfLiteInterpreterOptionsDelete(m_options);
    TfLiteModelDelete(m_model);
}

void TFLiteProcessor::Instance::prepare() {
    TfLiteInterpreterAllocateTensors(m_interpreter);
    m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
    m_output_tensor = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);

    if (m_inference_config.m_warm_up) {
        AudioBufferF input(1, m_inference_config.m_new_model_input_size);
        AudioBufferF output(1, m_inference_config.m_new_model_output_size);
        process(input, output);
    }
}

void TFLiteProcessor::Instance::process(AudioBufferF& input, AudioBufferF& output) {
    TfLiteTensorCopyFromBuffer(m_input_tensor, input.get_raw_data(), input.get_num_samples() * sizeof(float)); //TODO: Multichannel support
    TfLiteInterpreterInvoke(m_interpreter);
    TfLiteTensorCopyToBuffer(m_output_tensor, output.get_raw_data(), output.get_num_samples() * sizeof(float)); //TODO: Multichannel support
}

} // namespace anira