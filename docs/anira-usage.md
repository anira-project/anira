# anira User Guide

## Preface

Anira provides the following structures and classes to help you integrate real-time audio processing with your machine learning models:

| Class | Description |
| - | - |
| `ContextConfig` | **Optional:** The configuration structure that defines the context across all anira instances. Here you can define the behaviour of the thread pool, such as specifying the number of threads or enabling host-provided threads. |
| `InferenceHandler` | Manages audio processing/inference for the real-time thread, offloading inference to the thread pool and updating the real-time thread buffers with processed audio. This class provides the main interface for interacting with the library. |
| `InferenceConfig`  | A configuration structure for defining model specifics such as input/output shape, model details such as maximum inference time, and more. Each InferenceHandler instance must be constructed with this configuration. |
| `PrePostProcessor` | Enables pre- and post-processing steps before and after inference. Either use the default PrePostProcessor or inherit from this class for custom processing. |
| `HostAudioConfig` | A structure for defining the host audio configuration: buffer size and sample rate. |

## Using anira for Real-time Audio Inference

### Step 1: Define your Model Configuration

Start by specifying your model configuration using `anira::InferenceConfig`. This includes the model path, input/output shapes, and other critical settings that match the requirements of your model.

#### Step 1.1: Define the model information and the corresponding inference backend

First pass the model information and the corresponding inference backend in a `std::vector<anira::ModelData>`. `anira::ModelData` offers two ways to define the model information:

1. Pass the model path as a string:
```cpp
{std::string model_path, anira::InferenceBackend backend}
```
2. Pass the model data as binary information:
```cpp
{void* model_data, size_t model_size, anira::InferenceBackend backend}
```

Now define your model information in a `std::vector<anira::ModelData>`.

```cpp
std::vector<anira::ModelData> model_data = {
    {"path/to/your/model.pt", anira::InferenceBackend::LIBTORCH},
    {"path/to/your/model.onnx", anira::InferenceBackend::ONNX},
    {"path/to/your/model.tflite", anira::InferenceBackend::TFLITE}
};
```

**Note:** It is not necessary to submit a model for each backend anira was built with, only the one you want to use.


#### Step 1.2: Define the input and output shapes of the model

In the next step, define the input and output shapes of the model for each backend in a `std::vector<anira::TensorShape>`. The `anira::TensorShape` is defined as follows:

```cpp
{std::vector<int64_t> input_shape, std::vector<int64_t> output_shape, (optional) anira::InferenceBackend}
```

Now define the input and output shapes of your model for each backend used in the `std::vector<anira::ModelData>`.

```cpp
std::vector<anira::TensorShape> tensor_shapes = {
    {{{1, 1, 15380}}, {{1, 1, 2048}}, anira::InferenceBackend::LIBTORCH},
    {{{1, 1, 15380}}, {{1, 1, 2048}}, anira::InferenceBackend::ONNX},
    {{{1, 15380, 1}}, {{1, 2048, 1}}, anira::InferenceBackend::TFLITE}
};
```
**Note:** If the input and output shapes of the model are the same for all backends, you can also define only one `anira::TensorShape` without a specific `anira::InferenceBackend`.


#### Step 1.3: Define the `anira::InferenceConfig`

Finally, define the necessary `anira::InferenceConfig` with the model information, input/output shapes and the maximum inference time in ms. The maximum inference time is the measured worst case inference time. If the inference time during execution exceeds this value, it is likely that the audio signal will contain artifacts.

```cpp
anira::InferenceConfig inference_config (
    model_data, // std::vector<anira::ModelData>
    tensor_shapes, // std::vector<anira::TensorShape>
    42.66f // Maximum inference time in ms
);
```

There are also some optional parameters that can be set in the `anira::InferenceConfig`:

| Parameter | Description                                                                                                                                                                                                                                                                                                                                                                                        |
| - |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `internal_latency` | Type: `unsigned int`, default: `0`. Submit if your model has an internal latency. This allows the latency calculation to take it into account.                                                                                                                                                                                                                                                     |
| `warm_up` | Type: `unsigned int`, default: `0`. Defines the number of warm-up iterations before starting the inference process.                                                                                                                                                                                                                                                                                |
| `index_audio_data` | Type: `std::array<size_t, 2>` default: `{0, 0}`. Defines the input and output index of the audio data vector of tensors                                                                                                                                                                                                                                                                            |
| `num_audio_channels` | Type `std::array<size_t, 2>` default: `{1, 1}`. Defines the number of audio channels used for the input and output audio tensors.                                                                                                                                                                                                                                                                  |
| `session_exclusive_processor` | Type: `bool`, default: `false`. If set to `true`, the session will use an exclusive processor for inference and therefore cannot be processed parallel. Necessary for e.g. stateful models.                                                                                                                                                                                                        |
| `num_parallel_processors` | Type: `unsigned int`, default: `std::thread::hardware_concurrency() / 2`. Defines the number of parallel processors that can be used for the inference.                                                                                                                                                                                                                                            |
| `wait_in_process_block` | Type: `float`, default: `0.0f`. This parameter can only be set, if anira was build with `ANIRA_WITH_CONTROLLED_BLOCKING=ON`. This should be a value between `0.f` and `1.f`. It specifies the proportion of available processing time that the library will try to acquire new data from the inference threads on the real-time thread. This is a controversial parameter and should be used with caution. |

### Step 2: Create a PrePostProcessor Instance

If your model does not require any specific pre- or post-processing, you can use the default `anira::PrePostProcessor`. This is likely to be the case if the input and output shapes of the model are the same, the batchsize is 1, and your model operates in the time domain.

```cpp
anira::PrePostProcessor pp_processor;
```

If your model requires custom pre- or post-processing, you can inherit from the `anira::PrePostProcessor` class and overwrite the ```pre_process``` and ```post_process``` methods so that they match your model's requirements. In the ```pre_process``` method, we get the input samples from the audio application through an `anira::RingBuffer` and push them into the output buffer, which is an `anira::AudioBufferF`. This output buffer is then used for inference. In the ```post_process``` method we get the input samples through an `anira::AudioBufferF` and push them into the output buffer, which is an `anira::RingBuffer`. The samples from this output buffer are then returned to the audio application by the `anira::InferenceHandler`.

When your pre- and post-processing requires to access values from the `anira::InferenceConfig` struct, you can store the config as a member in your custom pre- and post-processor class.  Here is an example of a custom pre- and post-processor. The `anira::InferenceConfig` inference_config is supposed to be provided in the "MyConfig.h" file.

```cpp
#include <anira/anira.h>
#include "MyConfig.h"

class CustomPrePostProcessor : public anira::PrePostProcessor {
public:
    virtual void pre_process(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        pop_samples_from_buffer(input, output, inference_config.m_output_sizes[inference_config.m_index_audio_data[anira::IndexAudioData::Output]], inference_config.m_input_sizes[inference_config.m_index_audio_data[anira::IndexAudioData::Input]]-inference_config.m_output_sizes[inference_config.m_index_audio_data[anira::IndexAudioData::Output]]);
    };
    
    anira::InferenceConfig config = inference_config;
};
```

**Note:** The `anira::PrePostProcessor` class provides some methods to help you implement your own pre- and post-processing.  The following methods are provided up until now:

| Method                                                                                                                                     | Description |
|--------------------------------------------------------------------------------------------------------------------------------------------| - |
| `void pop_samples_from_buffer(anira::RingBuffer& input, anira::AudioBufferF& output)`                                                      | Pop output.size() samples from the input buffer and push them into the output buffer. |
| `void pop_samples_from_buffer(anira::RingBuffer& input, anira::AudioBufferF& output, int num_new_samples, int num_old_samples)`            | Pop num_new_samples new samples from the input buffer and get num_old_samples already poped samples from the input buffer and push them into the output buffer. The order of the samples in the output buffer is from oldest to newest. This can be useful for models that have a large receptive field that requires acces to past samples. |
| `void pop_samples_from_buffer(anira::RingBuffer& input, anira::AudioBufferF& output, int num_new_samples, int num_old_samples, int offset)` | Same as the above method, but starts writing to the output buffer at the offset. |
| `void push_samples_to_buffer(anira::AudioBufferF& input, anira::RingBuffer& output)`                                                       | Pushes input.size() samples from the input buffer into the output buffer. |

#### Optional Step: Submit and Retrieve Additional Tensor Values

Some neural networks not only require audio data as input and output tensors. For example, some models require additional input parameters or output values, like e.g. a prediction of the model's confidence. In this case you can use the `anira::PrePostProcessor` to submit or retrieve additional values. For this purpose the following public thread safe functions are provided:

| Method | Description |
| - | - |
| `void set_input(const float& input, size_t i, size_t j)` | Sets the input value at position i, j in the input tensor. |
| `void set_output(const float& output, size_t i, size_t j)` | Sets the output value at position i, j in the output tensor. |
| `float get_input(size_t i, size_t j)` | Returns the input value at position i, j in the input tensor. |
| `float get_output(size_t i, size_t j)` | Returns the output value at position i, j in the output tensor. |

### Step 3: Create an InferenceHandler Instance

In your application, you will need to create an instance of the `anira::InferenceHandler` class. This class is responsible for managing the inference process, including threading and real-time constraints. The constructor takes as arguments an instance of the default or custom `anira::PrePostProcessor` and an instance of the `anira::InferenceConfig` structure.

```cpp
// Sample initialization in your application's initialization function

// Default PrePostProcessor
anira::PrePostProcessor pp_processor;
// or custom PrePostProcessor
CustomPrePostProcessor pp_processor;

// Create an InferenceHandler instance
anira::InferenceHandler inference_handler(pp_processor, inference_config);
```

#### Optional Step: Define the Context Configuration

If you want to define a custom context configuration, you can do so by creating an instance of the `anira::ContextConfig` structure. This structure allows you to define the behaviour of the thread pool, by specifying the number of threads or allowing to use host-provided threads.

```cpp
// Use the existing anira::InferenceConfig and anira::PrePostProcessor instances

// Create an instance of anira::ContextConfig
anira::ContextConfig context_config {
    4, // Number of threads
    false // Use host-provided threads
};

// Create an InferenceHandler instance
anira::InferenceHandler inference_handler(pp_processor, inference_config, context_config);
```

### Step 4: Allocate Memory Before Processing

Before processing audio data, the `prepare` method of the `anira::InferenceHandler` instance must be called. This allocates all necessary memory in advance. The `prepare` method needs an instance of `anira::HostAudioConfig` which defines the buffer size and sample rate of the host audio application. We also need to select the inference backend we want to use. Depending on the backends you enabled during the build process, you can choose amongst `anira::LIBTORCH`, `anira::ONNX`, `anira::TFLITE` and `anira::CUSTOM`. After preparing the `anira::InferenceHandler`, you can get the latency of the inference process in samples by calling the `get_latency` method and use this information to compensate for the latency in your real-time audio application.

```cpp
void prepareAudioProcessing(double sample_rate, int buffer_size, int num_channels) {

    // Create an instance of anira::HostAudioConfig
    anira::HostAudioConfig host_config {
        buffer_size,
        sample_rate
    };

    inference_handler.prepare(host_config);

    // Select the inference backend
    inference_handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);
    
    // Get the latency of the inference process in samples
    int latency_in_samples = inference_handler.get_latency();
}
```

#### Optional Step: Enable host provided threads (for CLAP-Plugins)

This step is only supported for CLAP-Plugins. If you want to use host-provided threads, ensure that the `anira::ContextConfig` structure is created with the `use_host_provided_threads` flag set to `true`.

In a next step we try to get the clap_host_thread_pool extension from the host.

```
const clap_host_thread_pool* clap_thread_pool{nullptr};

// Clap extensions should be available from the .init() call
bool init() noexcept override {
    clap_thread_pool = static_cast<clap_host_thread_pool const*>(_host.host()->get_extension(_host.host(), CLAP_EXT_THREAD_POOL));

    return true;
}
```

Now, when we prepare the audio processing, we check, if the clap_thread_pool is available and if we can send execution request to the host thread pool. If the request is successful, we define a lambda function that allows `anira` to submit tasks to the host thread pool. The lambda must return a boolean value, if the task was successfully submitted to the host thread pool. If it fails, `anira` will start its own threads as a fallback.

```cpp
bool activate(double sampleRate, uint32_t minFrameCount,
                             uint32_t maxFrameCount) noexcept override
{
    anira::HostAudioConfig config ((size_t) maxFrameCount, sampleRate);
    
    if (clap_thread_pool != nullptr && clap_thread_pool->request_exec) {
        config.submit_task_to_host_thread = [this](int number_of_tasks) -> bool {
            if (clap_thread_pool->request_exec(_host.host(), number_of_tasks)) {
                return true;
            } else {
                return false;
            }
        };
    }
    
    inference_handler.prepare(config);
    
    ... 
    
    return true;
}
```

If `anira` submits now task to the host thread pool, `threadPoolExec(uint32_t taskIndex)` will be called in the host thread pool. This method should be overwritten in the application and should call the `exec_inference` method of the `anira::InferenceHandler` instance.

```cpp
void threadPoolExec(uint32_t taskIndex) noexcept override {
    inference_handler.exec_inference();
}
```

### Step 5: Real-time Audio Processing

Now we are ready to process audio in the process callback of our real-time audio application. The process method of the `anira::InferenceHandler` instance takes the input samples for all channels as an array of float pointers - ``float**``, and after calling the process method, the data is overwritten with the processed output.

```cpp
// Real-time safe audio processing in the process callback of your application
void process(float** audio_data, int num_samples) {
    inference_handler.process(audio_data, num_samples)
}
// audio_data now contains the processed audio samples
```

### Optional Step 6: Define a Custom::InferenceBackend

To use a custom backend processor, inherit from the `anira::BackendBase` class and overwrite the `process` and  `prepare` methods. The `process` method is called when the `anira::InferenceBackend::CUSTOM` backend is selected. The `process` method takes two `anira::AudioBufferF` instances as input and output buffers and a `std::shared_ptr<anira::SessionElement>` session element. The session element is necessary to e.g. send or retrieve additional values submitted by the pre- and post-processor.

The custom backend enables the integration of additional inference engines, customization of existing engines, or the implementation of a simple roundtrip/bypass backend that directly returns input samples, bypassing the inference stage.

The following example will demonstrate how to implement a custom bypass backend for the CNN model, where 15380 past samples are used as input and 2048 samples are returned as output. In order to bypass the inference stage, we just have to return the last 2048 samples of the input buffer.

**Note:** If you want to implement a custom inference backend use the existing backend implementations as a reference.

```cpp
#include <anira/anira.h>

class BypassProcessor : public anira::BackendBase {
public:
    BypassProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(anira::AudioBufferF &input, anira::AudioBufferF &output, [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        auto equal_channels = input.get_num_channels() == output.get_num_channels();
        auto sample_diff = input.get_num_samples() - output.get_num_samples();

        if (equal_channels && sample_diff >= 0) {
            for (size_t channel = 0; channel < input.get_num_channels(); ++channel) {
                auto write_ptr = output.get_write_pointer(channel);
                auto read_ptr = input.get_read_pointer(channel);

                for (size_t i = 0; i < output.get_num_samples(); ++i) {
                    write_ptr[i] = read_ptr[i+sample_diff];
                }
            }
        }
    }
};
```

After defining the custom backend processor, you can create an instance of the `BypassProcessor` class and pass it to the `anira::InferenceHandler` instance as an additional argument in the constructor. The `anira::InferenceHandler` will then use the `BypassProcessor` instance when the `anira::CUSTOM` backend is selected, instead of the default roundtrip processor.

```cpp
// Create an instance of the custom CustomProcessor
BypassProcessor bypass_processor(inference_config);
// In Step 3: Create an InferenceHandler Instance
anira::InferenceHandler inference_handler(pp_processor, inference_config, bypass_processor);
```