# How to use anira in a real-time audio application

## Step 1: Define your model configuration

Start by specifying your model configuration using ```anira::InferenceConfig```. This includes the model path, input/output shapes, batch size, and other critical settings that match your model's requirements.

```cpp
anira::InferenceConfig hybridNNConfig(
    // Model path and shapes for different backends
#ifdef USE_LIBTORCH
    "path/to/your/model.pt", // LibTorch model path (required, when -DANIRA_WITH_LIBTORCH=ON)
    {2048, 1, 150}, // Input shape for LibTorch (required, when -DANIRA_WITH_LIBTORCH=ON)
    {2048, 1}, // Output shape for LibTorch (required, when -DANIRA_WITH_LIBTORCH=ON)
#endif
#ifdef USE_ONNXRUNTIME
    "path/to/your/model.onnx", // ONNX model path (required, when -DANIRA_WITH_ONNX=ON)
    {2048, 1, 150}, // Input shape for ONNX (required, when -DANIRA_WITH_ONNX=ON)
    {2048, 1}, // Output shape for ONNX (required, when -DANIRA_WITH_ONNX=ON)
#endif
#ifdef USE_TFLITE
    "path/to/your/model.tflite", // TensorFlow Lite model path (required, when -DANIRA_WITH_TFLITE=ON)
    {2048, 150, 1}, // Input shape for TensorFlow Lite (required, when -DANIRA_WITH_TFLITE=ON)
    {2048, 1}, // Output shape for TensorFlow Lite (required, when -DANIRA_WITH_TFLITE=ON)
#endif
    2048, // Batch size
    150, // Model input size
    1, // Model output size
    false, // Warm-up flag (optional: default = false)
    0.5f, // Wait for inference in process block to reduce latency, 0.f is no waiting and 0.5f is wait for half a buffertime. Example buffer size 512 and sample rate 48000 Hz, a value of 0.5f = 5.33 ms (optional: default = 0.5f)
    false, // Bind one instance (plugin or jack client) to one thread (optional: default = false), this needs to be set to true if you use a stateful model 
    8 // Number of threads for parallel inference (optional: default = ((int) std::thread::hardware_concurrency() - 1 > 0) ? (int) std::thread::hardware_concurrency() - 1 : 1)), when bind_session_to_thread is true, this value is ignored and for every new instance a new thread is created
);
```

## Step 2: Implement custom pre- and post-processing

Implement your model-specific pre- and post-processing by inheriting from the ```anira::PrePostProcessor``` class. This is crucial for preparing the input and handling the output correctly. Default pre- and post-processing functions are implemented in the ```anira::PrePostProcessor``` class, so when you have the same amount of input and output samples in your model and need no specific pre- or post-processing you don't need to inherit from the class.

```cpp
class HybridNNPrePostProcessor : public anira::PrePostProcessor {
public:
    void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        for (size_t batch = 0; batch < config.m_batch_size; ++batch) {
            size_t baseIdx = batch * config.m_model_input_size;
            popSamplesFromBuffer(input, output, config.m_model_output_size, config.m_model_input_size-config.m_model_output_size, baseIdx);
        }
    };
    
    anira::InferenceConfig config = hybridNNConfig;
};

```

## Step 3: Integrate anira into your application

In your application, you'll need to create instances of anira::InferenceHandler and your custom pre-post processor. The InferenceHandler is responsible for managing the inference process, including threading and real-time constraints.

```cpp
// Example initialization in your application's setup function
HybridNNPrePostProcessor prePostProcessor;
anira::InferenceHandler inferenceHandler(prePostProcessor, hybridNNConfig);
```

## Step 4: Allocate memory before processing

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

## Step 5: Real-time audio processing

In your audio processing callback, use the ```anira::InferenceHandler``` to process your audio data. This involves converting your audio data to the format expected by anira, calling the process method, and then handling the output.

```cpp
void processBlock(AudioBuffer& audioBuffer) {
    inferenceHandler.process(audioBuffer.getArrayOfWritePointers(), audioBuffer.getNumSamples());
}
```
