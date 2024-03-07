#include <anira/backends/LibTorchProcessor.h>

namespace anira {

LibtorchProcessor::LibtorchProcessor(InferenceConfig& config) : BackendBase(config) {
    torch::set_num_threads(1);
    
    try {
        module = torch::jit::load(inferenceConfig.m_model_path_torch);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cout << e.what() << std::endl;
    }
}

LibtorchProcessor::~LibtorchProcessor() {
}

void LibtorchProcessor::prepareToPlay() {
    inputs.clear();
    inputs.push_back(torch::zeros(inferenceConfig.m_model_input_shape_torch));

    if (inferenceConfig.m_warm_up) {
        AudioBufferF input(1, inferenceConfig.m_batch_size * inferenceConfig.m_model_input_size_backend);
        AudioBufferF output(1, inferenceConfig.m_batch_size * inferenceConfig.m_model_output_size_backend);
        processBlock(input, output);
    }
}

void LibtorchProcessor::processBlock(AudioBufferF& input, AudioBufferF& output) { 
    // Create input tensor object from input data values and shape
    inputTensor = torch::from_blob(input.getRawData(), (const long long) input.getNumSamples()).reshape(inferenceConfig.m_model_input_shape_torch); // TODO: Multichannel support

    inputs[0] = inputTensor;

    // Run inference
    outputTensor = module.forward(inputs).toTensor();

    // Flatten the output tensor
    outputTensor = outputTensor.view({-1});

    // Extract the output tensor data
    for (size_t i = 0; i < inferenceConfig.m_batch_size * inferenceConfig.m_model_output_size_backend; i++) {
        output.setSample(0, i, outputTensor[(int64_t) i].item<float>());
    }
}

} // namespace anira