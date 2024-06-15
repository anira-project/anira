# anira User Guide

## Preface

To use anira in your real-time audio application, you need to create instances for the following classes:

| Class | Description |
|-|-|
| `InferenceConfig` | A configuration struct for defining model specifics such as input/output shape, model details, batch size, and more. |
| `PrePostProcessor` | Enables pre- and post-processing steps before and after inference. Either use the default PrePostProcessor or inherit from this class for custom processing. |
| `InferenceHandler` | Manages audio processing/inference for the real-time thread by offloading inference to a thread pool and updating the real-time thread's buffers with processed audio. |
| `HostAudioConfig` | A configuration struct for defining the host audio configs, number of channels, buffer size, and sample rate. |

## Using anira for Real-time Audio Inference

### Step 1: Define your Model Configuration

Start by specifying your model configuration using ``anira::InferenceConfig``. This includes the model path, input/output shapes, and other critical settings that match the requirements of your model. When using a single backend, you define the model path and input/output shapes only once.

```cpp
#include <anira/anira.h>

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
    42.66f, // Maximum inference time in ms for processing of all batches (required)

    0, // Internal model latency in samples for processing of all batches (optional: default = 0)
    false, // Warm-up the inference engine with a null inference run in prepare method (optional: default = false)
    0.5f, // Wait for the next processed buffer from the thread pool in the real-time thread's process block
          // method to reduce latency. 0.f is no waiting and 0.5f is wait for half a buffertime. Example
          // buffer size 512 and sample rate 48000 Hz, a value of 0.5f = 5.33 ms of maximum waiting time
          // (optional: default = 0.5f)
    false, // Bind one instance of the InferenceHandler to one thread (optional: default = false), this needs
           // to be set to true if you use a stateful model 
    8 // Number of threads for parallel inference
      // (optional: default = ((int) std::thread::hardware_concurrency() - 1 > 0) ?
      // (int) std::thread::hardware_concurrency() - 1 : 1)), when bind_session_to_thread is true,
      // this value is ignored and for every new instance a new thread is created
);
```

### Step 2: Create a PrePostProcessor Instance

If your model does not require any specific pre- or post-processing, you can use the default ``anira::PrePostProcessor``. This is likely to be the case if the input and output shapes of the model are the same, the batchsize is 1, and your model operates in the time domain.

```cpp
anira::PrePostProcessor myPrePostProcessor;
```

If your model requires costum pre- or post-processing, you can inherit from the ```anira::PrePostProcessor``` class and overwrite the ```preProcess``` and ```postProcess``` methods so that they match your model's requirements. In the ```preProcess``` method, we get the input samples from the audio application through an ``anira::RingBuffer`` and push them into the output buffer, which is an ``anira::AudioBufferF``. This output buffer is then used for inference. In the ```postProcess``` method we get the input samples through an ``anira::AudioBufferF`` and push them into the output buffer, which is an ``anira::RingBuffer``. The samples from this output buffer are then returned to the audio application by the ``anira::InferenceHandler``.

When your pre- and post-processing requires to access values from the ```anira::InferenceConfig``` struct, you can store the config as a member in your custom pre- and post-processor class.  Here is an example of a custom pre- and post-processor. The ```anira::InferenceConfig``` myConfig is supposed to be provided in the "MyConfig.h" file.

```cpp
#include <anira/anira.h>
#include "MyConfig.h"

class MyPrePostProcessor : public anira::PrePostProcessor {
public:
    virtual void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        int64_t num_batches;
        int64_t num_input_samples;
        int64_t num_output_samples;
        if (currentInferenceBackend == anira::LIBTORCH) {
            num_batches = config.m_model_input_shape_torch[0];
            num_input_samples = config.m_model_input_shape_torch[2];
            num_output_samples = config.m_model_output_shape_torch[1];
        } else if (currentInferenceBackend == anira::ONNX) {
            num_batches = config.m_model_input_shape_onnx[0];
            num_input_samples = config.m_model_input_shape_onnx[2];
            num_output_samples = config.m_model_output_shape_onnx[1];
        } else if (currentInferenceBackend == anira::TFLITE) {
            num_batches = config.m_model_input_shape_tflite[0];
            num_input_samples = config.m_model_input_shape_tflite[1];
            num_output_samples = config.m_model_output_shape_tflite[1];
        } else {
            throw std::runtime_error("Invalid inference backend");
        }
            
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t baseIdx = batch * num_input_samples;
            popSamplesFromBuffer(input, output, num_output_samples, num_input_samples-num_output_samples, baseIdx);
        }
    };
    
    anira::InferenceConfig config = myConfig;
};
```

Note: The ```anira::PrePostProcessor``` class provides some methods to help you implement your own pre- and post-processing.  The following methods are provided up until now:

- ```void popSamplesFromBuffer(anira::RingBuffer& input, anira::AudioBufferF& output)``` - Pop output.size() samples from the input buffer and push them into the output buffer.
- ```void popSamplesFromBuffer(anira::RingBuffer& input, anira::AudioBufferF& output, int numNewSamples, int numOldSamples)``` - Pop numNewSamples new samples from the input buffer and get numOldSamples already poped samples from the input buffer and push them into the output buffer. The order of the samples in the output buffer is from oldest to newest. This can be useful for models that have a large receptive field that requires acces to past samples.
- ```void popSamplesFromBuffer(anira::RingBuffer& input, anira::AudioBufferF& output, int numNewSamples, int numOldSamples, int offset)``` - Same as the above method, but starts writing to the output buffer at the offset.
- ```void pushSamplesToBuffer(anira::AudioBufferF& input, anira::RingBuffer& output)``` - Pushes input.size() samples from the input buffer into the output buffer.

### Step 3: Create an InferenceHandler Instance

In your application, you will need to create an instance of the ``anira::InferenceHandler`` class. This class is responsible for managing the inference process, including threading and real-time constraints. The constructor takes as arguments an instance of the default or custom ``anira::PrePostProcessor`` and an instance of the ``anira::InferenceConfig`` structure.

```cpp
// Sample initialization in your application's initialization function

// Default PrePostProcessor
anira::PrePostProcessor myPrePostProcessor;
// or custom PrePostProcessor
MyPrePostProcessor myPrePostProcessor;

// Create an InferenceHandler instance
anira::InferenceHandler myInferenceHandler(myPrePostProcessor, myConfig);
```

### Step 4: Allocate Memory Before Processing

Before processing audio data, the `prepare` method of the ``anira::InferenceHandler`` instance must be called. This allocates all necessary memory in advance. The `prepare` method needs an instance of ``anira::HostAudioConfig`` which defines the number of channels, buffer size and sample rate of the host audio application. We also need to select the inference backend we want to use. Depending on the backends you enabled during the build process, you can choose amongst `anira::LIBTORCH`, `anira::ONNX`, `anira::TFLITE` and `anira::NONE`. After preparing the `anira::InferenceHandler`, you can get the latency of the inference process in samples by calling the `getLatency` method and use this information to compensate for the latency in your real-time audio application.

```cpp
void prepareAudioProcessing(double sampleRate, int bufferSize, int numChannels) {

    // Create an instance of anira::HostAudioConfig
    anira::HostAudioConfig audioConfig {
        1, // currently only mono is supported
        bufferSize,
        sampleRate
    };

    inferenceHandler.prepare(audioConfig);

    // Select the inference backend
    inferenceHandler.selectInferenceBackend(anira::LIBTORCH);
    
    // Get the latency of the inference process in samples
    int latencyInSamples = inferenceHandler.getLatency();
}
```

Note: Up until now, anira only supports mono audio processing. Stereo audio processing will be supported soon.

### Step 5: Real-time Audio Processing

Now we are ready to process audio in the process callback of our real-time audio application. The process method of the ``anira::InferenceHandler`` instance takes the input samples for all channels as an array of float pointers - ``float**``, and after calling the process method, the data is overwritten with the processed output.

```cpp
// Real-time safe audio processing in the process callback of your application
processBlock(float** audioData, int numSamples) {
    inferenceHandler.process(audioData, numSamples)
}
// audioData now contains the processed audio samples
```

## anira Roundtrip

To use the `anira::NONE` backend and get a continuous audio signal, you may need to define a custom backend processor that does not perform any inference and is activated when the `anira::NONE` backend is selected. To do this, you need to inherit from the `anira::BackendBase` class and override the `processBlock` method and in some cases the `prepareToPlay` method as well. Here is an example of a custom backend processor that does not perform any inference and just does a roundtrip for the respective myPrePostProcessor that we defined in the previous steps.

```cpp
#include <anira/anira.h>

class MyNoneProcessor : public anira::BackendBase {
public:
    MyNoneProcessor(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void processBlock(anira::AudioBufferF &input, anira::AudioBufferF &output) override {
        auto equalChannels = input.getNumChannels() == output.getNumChannels();
        auto sampleDiff = input.getNumSamples() - output.getNumSamples();
        int64_t num_batches;
        int64_t num_input_samples;
#if USE_LIBTORCH
        num_batches = inferenceConfig.m_model_input_shape_torch[0];
        num_input_samples = inferenceConfig.m_model_input_shape_torch[2];
#elif USE_ONNXRUNTIME
        num_batches = inferenceConfig.m_model_input_shape_onnx[0];
        num_input_samples = inferenceConfig.m_model_input_shape_onnx[2];
#elif USE_TFLITE
        num_batches = inferenceConfig.m_model_input_shape_tflite[0];
        num_input_samples = inferenceConfig.m_model_input_shape_tflite[1];
#endif

        if (equalChannels && sampleDiff >= 0) {
            for (size_t channel = 0; channel < input.getNumChannels(); ++channel) {
                auto writePtr = output.getWritePointer(channel);
                auto readPtr = input.getReadPointer(channel);

                for (size_t batch = 0; batch < num_batches; ++batch) {
                    size_t baseIdx = batch * num_input_samples;
                    writePtr[batch] = readPtr[num_input_samples - 1 + baseIdx];
                }
            }
        }
    }
};
```

After defining the custom backend processor, you can create an instance of the `MyNoneProcessor` class and pass it to the `anira::InferenceHandler` instance as an additional argument in the constructor. The `anira::InferenceHandler` will then use the `MyNoneProcessor` instance when the `anira::NONE` backend is selected, instead of the default roundtrip processor.

```cpp
// In Step 3: Create an InferenceHandler Instance
anira::InferenceHandler myInferenceHandler(myPrePostProcessor, myConfig, myNoneProcessor);
```