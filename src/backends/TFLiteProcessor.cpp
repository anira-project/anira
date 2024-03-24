#include <anira/backends/TFLiteProcessor.h>

#ifdef _WIN32
#include <comdef.h>
#endif

namespace anira {

TFLiteProcessor::TFLiteProcessor(InferenceConfig& config) : BackendBase(config)
{
#ifdef _WIN32
    std::string modelpathStr = inferenceConfig.m_model_path_tflite;
    std::wstring modelpath = std::wstring(modelpathStr.begin(), modelpathStr.end());
#else
    std::string modelpath = inferenceConfig.m_model_path_tflite;
#endif

#ifdef _WIN32
    _bstr_t modelPathChar (modelpath.c_str());
    model = TfLiteModelCreateFromFile(modelPathChar);
#else
    model = TfLiteModelCreateFromFile(modelpath.c_str());
#endif

    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    interpreter = TfLiteInterpreterCreate(model, options);
    // This is necessary when we have dynamic input shapes, it should be done before allocating tensors obviously
    TfLiteInterpreterResizeInputTensor(interpreter, 0, inferenceConfig.m_model_input_shape_tflite.data(), inferenceConfig.m_model_input_shape_tflite.size());
}

TFLiteProcessor::~TFLiteProcessor()
{
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}

void TFLiteProcessor::prepareToPlay() {
    TfLiteInterpreterAllocateTensors(interpreter);
    inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    if (inferenceConfig.m_warm_up) {
        AudioBufferF input(1, inferenceConfig.m_batch_size * inferenceConfig.m_model_input_size);
        AudioBufferF output(1, inferenceConfig.m_batch_size * inferenceConfig.m_model_output_size);
        processBlock(input, output);
    }
}

void TFLiteProcessor::processBlock(AudioBufferF& input, AudioBufferF& output) {
    TfLiteTensorCopyFromBuffer(inputTensor, input.getRawData(), input.getNumSamples() * sizeof(float)); //TODO: Multichannel support
    TfLiteInterpreterInvoke(interpreter);
    TfLiteTensorCopyToBuffer(outputTensor, output.getRawData(), output.getNumSamples() * sizeof(float)); //TODO: Multichannel support
}

} // namespace anira