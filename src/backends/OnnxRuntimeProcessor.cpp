#include <anira/backends/OnnxRuntimeProcessor.h>

namespace anira {

OnnxRuntimeProcessor::OnnxRuntimeProcessor(InferenceConfig& config) :
    memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
    BackendBase(config)
{
#ifdef _WIN32
    std::string modelpathStr = inferenceConfig.m_model_path_onnx;
    std::wstring modelpath = std::wstring(modelpathStr.begin(), modelpathStr.end());
#else
    std::string modelpath = inferenceConfig.m_model_path_onnx;
#endif

    session_options.SetIntraOpNumThreads(1);
    session = std::make_unique<Ort::Session>(env, modelpath.c_str(), session_options);

    inputName = std::make_unique<Ort::AllocatedStringPtr>(session->GetInputNameAllocated(0, ort_alloc));
    outputName = std::make_unique<Ort::AllocatedStringPtr>(session->GetOutputNameAllocated(0, ort_alloc));
    inputNames = {(char*) inputName->get()};
    outputNames = {(char*) outputName->get()};

    inputSize = config.m_batch_size * config.m_model_input_size;
    outputSize = config.m_batch_size * config.m_model_output_size;

    std::vector<int64_t> inputShape = config.m_model_input_shape_onnx;
    inputData.resize(inputSize, 0.0f);

    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            inputData.data(),
            inputSize,
            inputShape.data(),
            inputShape.size()
    ));
}

OnnxRuntimeProcessor::~OnnxRuntimeProcessor()
{
}

void OnnxRuntimeProcessor::prepareToPlay() {
    if (inferenceConfig.m_warm_up) {
        AudioBufferF input(1, inputSize);
        AudioBufferF output(1, outputSize);
        processBlock(input, output);
    }
}

void OnnxRuntimeProcessor::processBlock(AudioBufferF& input, AudioBufferF& output) {
    auto inputWritePtr = inputTensor[0].GetTensorMutableData<float>();
    auto inputReadPtr = input.getReadPointer(0);
    for (size_t i = 0; i < inputSize; i++) {
        inputWritePtr[i] = inputReadPtr[i];
    }

    try {
        outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensor.data(), inputNames.size(), outputNames.data(), outputNames.size());
    }
    catch (Ort::Exception &e) {
        std::cout << e.what() << std::endl;
    }

    auto outputWritePtr = output.getWritePointer(0);
    auto outputReadPtr = outputTensor[0].GetTensorMutableData<float>();
    for (size_t i = 0; i < outputSize; ++i) {
        outputWritePtr[i] = outputReadPtr[i];
    }
}

} // namespace anira