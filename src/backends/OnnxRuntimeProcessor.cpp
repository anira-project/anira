#include <aria/backends/OnnxRuntimeProcessor.h>

OnnxRuntimeProcessor::OnnxRuntimeProcessor(InferenceConfig& config) :
    memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
    inferenceConfig(config)
{
}

OnnxRuntimeProcessor::~OnnxRuntimeProcessor()
{
}

void OnnxRuntimeProcessor::prepareToPlay() {
#ifdef _WIN32
    std::string modelpathStr = inferenceConfig.m_model_path_onnx;
    std::wstring modelpath = std::wstring(modelpathStr.begin(), modelpathStr.end());
#else
    std::string modelpath = inferenceConfig.m_model_path_onnx;
#endif

    session_options.SetIntraOpNumThreads(1);
    session = std::make_unique<Ort::Session>(env, modelpath.c_str(), session_options);
    // Define the shape of input tensor

    inputShape.clear();
    for (long long i : inferenceConfig.m_model_input_shape_onnx) {
        inputShape.push_back(i);
    }

    if (inferenceConfig.m_warm_up) {
        AudioBufferF input(1, inferenceConfig.m_batch_size * inferenceConfig.m_model_input_size_backend);
        AudioBufferF output(1, inferenceConfig.m_batch_size  * inferenceConfig.m_model_output_size_backend);
        processBlock(input, output);
    }
}

void OnnxRuntimeProcessor::processBlock(AudioBufferF& input, AudioBufferF& output) {
    // Create input tensor object from input data values and shape
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>  (memory_info,
                                                                    input.getRawData(),
                                                                    input.getNumSamples(), // TODO: Multichannel support
                                                                    inputShape.data(),
                                                                    inputShape.size());


    // Get input and output names from model
    Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, ort_alloc);
    inputNames = {(char*) inputName.get()};
    outputNames = {(char*) outputName.get()};

    try {
        // Run inference
        outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, inputNames.size(), outputNames.data(), outputNames.size());
    }
    catch (Ort::Exception &e) {
        std::cout << e.what() << std::endl;
    }

    // Extract the output tensor dat
    for (size_t i = 0; i < inferenceConfig.m_batch_size * inferenceConfig.m_model_output_size_backend; i++) {
        output.setSample(0, i, outputTensors[0].GetTensorMutableData<float>()[i]); // TODO: Multichannel support
    }
}