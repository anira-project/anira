# How to use anira in a real-time audio application

# Step 1: Define your model configuration
Start by specifying your model configuration using anira::InferenceConfig. This includes the model path, input/output shapes, batch size, and other critical settings that match your model's requirements.
```cpp
anira::InferenceConfig hybridNNConfig(
    // Model path and shapes for different backends
#ifdef USE_LIBTORCH
    "path/to/your/model.pt", // LibTorch model path
    {2048, 1, 150}, // Input shape for LibTorch
    {2048, 1}, // Output shape for LibTorch
#endif
#ifdef USE_ONNXRUNTIME
    "path/to/your/model.onnx", // ONNX model path
    {2048, 1, 150}, // Input shape for ONNX
    {2048, 1}, // Output shape for ONNX
#endif
#ifdef USE_TFLITE
    "path/to/your/model.tflite", // TensorFlow Lite model path
    {2048, 150, 1}, // Input shape for TensorFlow Lite
    {2048, 1}, // Output shape for TensorFlow Lite
#endif
    2048, // Batch size
    150, // Model input size
    1, // Model output size
    HYBRIDNN_MAX_INFERENCE_TIME, // Max inference time
    false, // Warm-up flag
    0.5f // Wait in process block to reduce latency
);
```

# Step 2: Implement custom pre- and post-processing
Implement your model-specific pre- and post-processing by extending anira::PrePostProcessor. This is crucial for preparing the input and handling the output correctly.
```cpp
class HybridNNPrePostProcessor : public anira::PrePostProcessor {
public:
    void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        for (size_t batch = 0; batch < config.m_batch_size; ++batch) {
            size_t baseIdx = batch * config.m_model_input_size_backend;
            popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size, baseIdx);
        }
    };
    
    anira::InferenceConfig config = hybridNNConfig;
};

```

# Step 3: Integrate anira into your application
In your application, you'll need to create instances of anira::InferenceHandler and your custom pre-post processor. The InferenceHandler is responsible for managing the inference process, including threading and real-time constraints.

```cpp
// Example initialization in your application's setup function
HybridNNPrePostProcessor prePostProcessor;
anira::InferenceHandler inferenceHandler(prePostProcessor, hybridNNConfig);
```

# Step 4: Allocate memory before processing
Before processing audio data, ensure that all necessary memory allocations are done upfront. 

```cpp
void prepareAudioProcessing(double sampleRate, int bufferSize, int numChannels) {
    anira::HostAudioConfig audioConfig {
        1, // currently only mono is supported
        bufferSize,
        sampleRate
    };

    inferenceHandler.prepare(audioConfig);
    
    // anira will add latency  
    int latencyInSamples = inferenceHandler.getLatency();
}
```

# Step 5: Real-time audio processing
In your audio processing callback, use the anira::InferenceHandler to process your audio data. This involves converting your audio data to the format expected by anira, calling the process method, and then handling the output.
```cpp
void processBlock(AudioBuffer& audioBuffer) {
    inferenceHandler.process(audioBuffer.getArrayOfWritePointers(), audioBuffer.getNumSamples());
}
```
