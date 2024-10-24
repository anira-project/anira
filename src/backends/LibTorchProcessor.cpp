#include <anira/backends/LibTorchProcessor.h>

namespace anira {

LibtorchProcessor::LibtorchProcessor(InferenceConfig& config) : BackendBase(config) {
    torch::set_num_threads(1);
    
    try {
        m_module = torch::jit::load(m_inference_config.m_model_path_torch);
    }
    catch (const c10::Error& e) {
        std::cerr << "[ERROR] error loading the model\n";
        std::cerr << e.what() << std::endl;
    }
}

LibtorchProcessor::~LibtorchProcessor() {
}

void LibtorchProcessor::prepare() {
    m_inputs.clear();
    m_inputs.push_back(torch::zeros(m_inference_config.m_model_input_shape_torch));

    if (m_inference_config.m_warm_up) {
        AudioBufferF input(1, m_inference_config.m_new_model_input_size);
        AudioBufferF output(1, m_inference_config.m_new_model_output_size);
        process(input, output);
    }
}

void LibtorchProcessor::process(AudioBufferF& input, AudioBufferF& output) { 
    // Create input tensor object from input data values and shape
    m_input_tensor = torch::from_blob(input.get_raw_data(), (const long long) input.get_num_samples()).reshape(m_inference_config.m_model_input_shape_torch); // TODO: Multichannel support

    m_inputs[0] = m_input_tensor;

    // Run inference
    m_output_tensor = m_module.forward(m_inputs).toTensor();

    // Flatten the output tensor
    m_output_tensor = m_output_tensor.view({-1});

    // Extract the output tensor data
    for (size_t i = 0; i < m_inference_config.m_new_model_output_size; i++) {
        output.set_sample(0, i, m_output_tensor[(int64_t) i].item<float>());
    }
}

} // namespace anira