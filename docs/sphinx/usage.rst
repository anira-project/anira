Usage Guide
===========

Overview
--------

Anira provides the following structures and classes to help you integrate real-time audio processing with your machine learning models:

+---------------------------------------+--------------------------------------------------------------------------+
| Class                                 | Description                                                              |
+=======================================+==========================================================================+
| :cpp:class:`anira::InferenceHandler`  | Manages audio processing/inference for the real-time thread,             |
|                                       | offloading inference to the thread pool and updating the real-time       |
|                                       | thread buffers with processed audio. This class provides the main        |
|                                       | interface for interacting with the library.                              |
+---------------------------------------+--------------------------------------------------------------------------+
| :cpp:struct:`anira::InferenceConfig`  | A configuration structure for defining model specifics such as           |
|                                       | input/output shape, model details such as maximum inference time,        |
|                                       | and more. Each InferenceHandler instance must be constructed with        |
|                                       | this configuration.                                                      |
+---------------------------------------+--------------------------------------------------------------------------+
| :cpp:class:`anira::PrePostProcessor`  | Enables pre- and post-processing steps before and after inference.       |
|                                       | Either use the default PrePostProcessor or inherit from this class       |
|                                       | for custom processing.                                                   |
+---------------------------------------+--------------------------------------------------------------------------+
| :cpp:class:`anira::HostConfig`        | A structure for defining the host configuration: buffer size             |
|                                       | and sample rate.                                                         |
+---------------------------------------+--------------------------------------------------------------------------+
| :cpp:struct:`anira::ContextConfig`    | **Optional:** The configuration structure that defines the context       |
|                                       | across all anira instances. Here you can define the behaviour of the     |
|                                       | thread pool, such as specifying the number of threads.                   |
+---------------------------------------+--------------------------------------------------------------------------+

1. Inference Configuration
--------------------------

Start by specifying your model configuration using :cpp:struct:`anira::InferenceConfig`. This includes the model path, input/output shapes, and other critical settings that match the requirements of your model.

1.1. ModelData
~~~~~~~~~~~~~~

First define the model information and the corresponding inference backend in a :cpp:struct:`anira::ModelData`. There are two ways to define the model information:

Pass the model path as a string:

.. code-block:: cpp

    {std::string model_path, anira::InferenceBackend backend}

Pass the model data as binary information:

.. code-block:: cpp

    {void* model_data, size_t model_size, anira::InferenceBackend backend}

.. note::
    Defining the model data as binary information is only possible for the ``anira::InferenceBackend::ONNX`` until now.

The :cpp:struct:`anira::InferenceConfig` requires a vector of :cpp:struct:`anira::ModelData`.

.. code-block:: cpp

    std::vector<anira::ModelData> model_data = {
        {"path/to/your/model.pt", anira::InferenceBackend::LIBTORCH},
        {"path/to/your/model.onnx", anira::InferenceBackend::ONNX},
        {"path/to/your/model.tflite", anira::InferenceBackend::TFLITE}
    };

.. note::
    It is not necessary to submit a model for each backend anira was built with, only the one you want to use.

1.2. TensorShape
~~~~~~~~~~~~~~~~

In the next step, define the input and output shapes of the model in an :cpp:struct:`anira::TensorShape`. The input and output_shapes are defined as :cpp:type:`anira::TensorShapeList`, where each inner vector represents the shape of a tensor.

.. code-block:: cpp

    {anira::TensorShapeList input_shape, anira::TensorShapeList output_shape, (optional) anira::InferenceBackend}

The input and output shapes are defined as a vector of integers, where each integer represents the size of a dimension in the tensor. The optional :cpp:enum:`anira::InferenceBackend` parameter allows you to specify which backend this shape corresponds to. If you do not specify the backend, the shape is used for all backends that do not have a specific shape defined.

The :cpp:struct:`anira::InferenceConfig` requires a vector of :cpp:struct:`anira::TensorShape`.

.. code-block:: cpp

    std::vector<anira::TensorShape> tensor_shape = {
        {{{1, 4, 15380}, {1, 1}}, {{1, 1, 2048}, {1, 1}}, anira::InferenceBackend::LIBTORCH},
        {{{1, 4, 15380}, {1, 1}}, {{1, 1, 2048}, {1, 1}}, anira::InferenceBackend::ONNX},
        {{{1, 15380, 4}, {1, 1}}, {{1, 2048, 1}, {1, 1}}, anira::InferenceBackend::TFLITE}
    };

.. note::
    If the input and output shapes of the model are the same for all backends, you can also define only one :cpp:struct:`anira::TensorShape` without a specific :cpp:enum:`anira::InferenceBackend`:

1.3. (Optional) ProcessingSpec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, you may want to define a processing specification that describes how the model should be processed. This is optional and can be used to specify additional parameters for the inference process. Here you can define the number of input and output channels and also whether tensors shall be streamable or non-streamable.

The following parameters can be defined in the :cpp:struct:`anira::ProcessingSpec`:

+-------------------------------+------------------------------------------------------------------------------------------------+
| Parameter                     | Description                                                                                    |
+===============================+================================================================================================+
| preprocess_input_channels     | Type: ``std::vector<size_t>``, default: ``std::vector<size_t>{input_tensor_shape.size(), 1}``  |
|                               | Defines the number of input channels for the model. Only streamable tensors can have input     |
|                               | channels != 1.                                                                                 |
+-------------------------------+------------------------------------------------------------------------------------------------+
| postprocess_output_channels   | Type: ``std::vector<size_t>``, default: ``std::vector<size_t>{output_tensor_shape.size(), 1}`` |
|                               | Defines the number of output channels for the model. Only streamable tensors can have output   |
|                               | channels != 1.                                                                                 |
+-------------------------------+------------------------------------------------------------------------------------------------+
| preprocess_input_size         | Type: ``std::vector<size_t>``, default: ``input_tensor_sizes``. Specifies the minimum number   |
|                               | of samples required per tensor before triggering preprocessing and inference. For streamable   |
|                               | tensors, this determines how many samples must accumulate before processing begins. Set to     |
|                               | ``0`` for non-streamable tensors to start processing immediately without waiting for samples.  |
+-------------------------------+------------------------------------------------------------------------------------------------+
| postprocess_output_size       | Type: ``std::vector<size_t>``, default: ``output_tensor_sizes``. Defines the number of samples |
|                               | that will be returned after the postprocessing step. Set to ``0`` for non-streamable tensors.  |
+-------------------------------+------------------------------------------------------------------------------------------------+
| internal_model_latency        | Type: ``std::vector<size_t>``, default: ``std::vector<size_t>{input_tensor_shape.size(), 0}``. |
|                               | Submit if your model has an internal latency. This allows for the latency calculation to take  |
|                               | it into account.                                                                               |
+-------------------------------+------------------------------------------------------------------------------------------------+

You only need to define the parameters that are relevant for your model. If you do not define an :cpp:struct:`anira::ProcessingSpec`, the default values will be used. Here is an example of how to define the :cpp:struct:`anira::ProcessingSpec` with all parameters:

.. code-block:: cpp

    std::vector<anira::ProcessingSpec> processing_spec = {
        {4, 1}, // Input tensor 0 has 4 input channels, and tensor 1 has 1 input channel
        {1, 1}, // Output tensor 0 has 1 output channel, and tensor 1 has 1 output channel
        {2048, 0}, // Preprocess input size is 2048 for tensor 0 and 0 for tensor 1
        {2048, 0}, // Postprocess output size is 2048 for tensor 0 and 0 for tensor 1
        {0, 0}  // Internal model latency is 0 for both tensors, meaning no internal latency
    };

1.4. InferenceConfig
~~~~~~~~~~~~~~~~~~~~

Finally, define the necessary :cpp:struct:`anira::InferenceConfig` with the vector of :cpp:struct:`anira::ModelData`, vector of :cpp:struct:`anira::TensorShape`, the optional :cpp:struct:`anira::ProcessingSpec`, and the maximum inference time. The maximum inference time is the measured worst case inference time. If the inference time during execution exceeds this value, it is likely that the audio signal will contain dropouts. There are also some other optional parameters that can be set in the :cpp:struct:`anira::InferenceConfig` to further customize the inference process.

.. code-block:: cpp

    anira::InferenceConfig inference_config (
        model_data, // std::vector<anira::ModelData>
        tensor_shape, // std::vector<anira::TensorShape>
        processing_spec, // anira::ProcessingSpec (optional)
        42.66f // Maximum inference time in ms
    );

These are the other optional parameters that can be set in the :cpp:struct:`anira::InferenceConfig`:

+-----------------------------+--------------------------------------------------------+
| Parameter                   | Description                                            |
+=============================+========================================================+
| warm_up                     | Type: ``unsigned int``, default: ``0``. Defines the    |
|                             | number of warm-up iterations before starting the       |
|                             | inference process.                                     |
+-----------------------------+--------------------------------------------------------+
| session_exclusive_processor | Type: ``bool``, default: ``false``. If set to          |
|                             | ``true``, the session will use an exclusive processor  |
|                             | for inference and therefore cannot be processed        |
|                             | parallel. Necessary for e.g. stateful models.          |
+-----------------------------+--------------------------------------------------------+
| blocking_ratio              | Type: ``float``, default: ``0.0f``. Defines the        |
|                             | proportion of available processing time (0.0-0.99)     |
|                             | that the library will use to acquire new data from     |
|                             | inference threads on the real-time thread. Use with    |
|                             | caution as this affects real-time performance.         |
+-----------------------------+--------------------------------------------------------+
| num_parallel_processors     | Type: ``unsigned int``, default:                       |
|                             | ``std::thread::hardware_concurrency() / 2``. Defines   |
|                             | the number of parallel processors that can be used     |
|                             | for the inference.                                     |
+-----------------------------+--------------------------------------------------------+

2. Pre and Post Processing
--------------------------

For most use cases, you can use the default :cpp:class:`anira::PrePostProcessor` without modification. This is suitable when your model operates in the time domain with straightforward input/output tensor shapes.

.. code-block:: cpp

    // Create an instance of anira::PrePostProcessor
    anira::PrePostProcessor pp_processor(inference_config);

If your model requires custom pre- or post-processing (such as frequency domain transforms, custom windowing, or multi-tensor operations), you can create a custom preprocessor by inheriting from the :cpp:class:`anira::PrePostProcessor` class. For detailed information on implementing custom preprocessing and postprocessing, see the :doc:`custom_preprocessing` chapter.

3. Inference Handler
--------------------

In your application, you will need to create an instance of the :cpp:class:`anira::InferenceHandler` class. This class is responsible for managing the inference process, including threading and real-time constraints. The constructor takes as arguments an instance of the default or custom :cpp:class:`anira::PrePostProcessor` and an instance of the :cpp:class:`anira::InferenceConfig` structure.

.. code-block:: cpp

    // Sample initialization in your application's initialization function

    // Default PrePostProcessor
    anira::PrePostProcessor pp_processor(inference_config);
    // or custom PrePostProcessor
    CustomPrePostProcessor pp_processor(inference_config);

    // Create an InferenceHandler instance
    anira::InferenceHandler inference_handler(pp_processor, inference_config);

3.1. (Optional) ContextConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to define a custom context configuration, you can do so by creating an instance of the :cpp:struct:`anira::ContextConfig` structure. This structure allows you to define the behaviour of the thread pool, by specifying the number of threads.

.. code-block:: cpp

    // Use the existing anira::InferenceConfig and anira::PrePostProcessor instances

    // Create an instance of anira::ContextConfig
    anira::ContextConfig context_config {
        4 // Number of threads
    };

    // Create an InferenceHandler instance
    anira::InferenceHandler inference_handler(pp_processor, inference_config, context_config);

4. Get ready for Processing
---------------------------

Before processing audio data, the :cpp:func:`anira::InferenceHandler::prepare` method of the :cpp:class:`anira::InferenceHandler` instance must be called. This allocates all necessary memory in advance. The :cpp:func:`anira::InferenceHandler::prepare` method needs an instance of :cpp:struct:`anira::HostConfig` which defines the buffer size and sample rate of the host application. Also an inference backend must be selected, which is done by calling the :cpp:func:`anira::InferenceHandler::set_inference_backend` method.

4.1. HostConfig
~~~~~~~~~~~~~~~

The :cpp:struct:`anira::HostConfig` structure defines the host application's configuration, including buffer size and sample rate. This configuration is essential for the :cpp:class:`anira::InferenceHandler` to allocate appropriate memory and calculate processing latency.

To construct :cpp:struct:`anira::HostConfig`, provide the buffer size and sample rate for a specific streamable input tensor. By default, tensor index 0 is used. For models with multiple input tensors, specify the desired tensor index.

The structure also includes an optional parameter that controls whether the buffer size is seen as static or as the maximum buffer size. When this parameter is set to true, variable buffer sizes smaller than the specified maximum are allowed, which is useful for real-time applications with dynamic buffer sizes. However, this may increase the latency that anira calculates, since it needs to compensate for all possible size variations.

**Create HostConfig with static buffer size for input tensor 0:**

.. code-block:: cpp

    anira::HostConfig host_config {
        2048.f, // Buffer size in samples
        44100.f // Sample rate in Hz
    };

**Create HostConfig with maximum buffer size for input tensor 1:**

.. code-block:: cpp

    anira::HostConfig host_config {
        2048.f, // Buffer size in samples
        44100.f, // Sample rate in Hz
        true, // Allow smaller buffer sizes (optional, default is false)
        1 // Tensor index (optional, default is 0)
    };

..  note::
    The buffer size parameter accepts floating-point values, allowing you to specify fractional relationships between the host buffer and the model processing buffer. For example, setting a buffer size of 0.5f means the :cpp:class:`anira::InferenceHandler` will receive one sample for the specified input tensor every two host buffer cycles. The latency calculation in anira accounts for this, assuming the sample is provided during the second host buffer cycle (worst-case scenario). If your model produces output at twice the input rate, the :cpp:class:`anira::InferenceHandler` can return one sample per host buffer cycle.

4.2. Prepare
~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::prepare` method is called with an instance of :cpp:struct:`anira::HostConfig` to allocate the necessary memory for the inference process. This method must be called before processing audio data. You can optionally specify the latency compensation for the inference process by passing a latency value in samples for a specific output tensor or a vector of latency values for all output tensors. If you do not specify a latency value, anira will calculate a minimal latency based on the information in the :cpp:struct:`anira::HostConfig` and the :cpp:struct:`anira::InferenceConfig`. This latency calculation is quite sophisticated and you can read more about it in the :doc:`latency` section.

**Preparing without custom latency (automatic latency calculation):**

.. code-block:: cpp

    // Prepare the :cpp with automatic latency calculation
    inference_handler.prepare(host_config);

**Preparing with custom latency for a specific output tensor:**

.. code-block:: cpp

    // Prepare with custom latency for the first output tensor (index 0)
    size_t custom_latency_samples = 1024;
    size_t output_tensor_index = 0;
    inference_handler.prepare(host_config, custom_latency_samples, output_tensor_index);

**Preparing with custom latency for all output tensors:**

.. code-block:: cpp

    // Prepare with custom latency values for all output tensors
    std::vector<size_t> custom_latency_values = {1024, 512}; // Different latency for each tensor
    inference_handler.prepare(host_config, custom_latency_values);

.. note::
    Only streamable tensors can have a latency != 0. Non-streamable tensors are available via the :cpp:func:`anira::PrePostProcessor::get_output` method and do not require a latency value.

4.3. Select Backend
~~~~~~~~~~~~~~~~~~~

Before processing audio, you must select which inference backend to use. The available backends depend on which ones were enabled during the build process. You can choose from:

- ``anira::InferenceBackend::LIBTORCH`` - PyTorch/LibTorch models
- ``anira::InferenceBackend::ONNX`` - ONNX Runtime models  
- ``anira::InferenceBackend::TFLITE`` - TensorFlow Lite models
- ``anira::InferenceBackend::CUSTOM`` - Custom backend implementations

Select the backend that corresponds to your model format:

.. code-block:: cpp

    // Select the inference backend
    inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);

.. note::
    Please refer to the :doc:`custom_backends` section for more information on how to implement your own custom backend.

5. Real-time Processing
-----------------------

Now we are ready to process audio in the process callback of our real-time audio application. For streamable as well as non-streamable tensors, the :cpp:func:`anira::InferenceHandler::process` or the :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods can be used to process audio data. All methods can be used in the real-time thread. Each function is overloaded so it can be used with a single tensor or with a vector of tensors.

5.1. Process Method
~~~~~~~~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::process` method is the most straightforward approach for real-time audio processing when input and output happen simultaneously.

**Simple In-Place Processing:**

For models where input and output have the same shape and only one tensor is streamable:

.. code-block:: cpp

    // In your real-time audio callback
    void processBlock(float** audio_data, int num_samples) {
        // Process audio in-place - input is overwritten with output
        size_t processed_samples = inference_handler.process(
            audio_data, 
            num_samples
        );
        // audio_data now contains the processed audio samples
    }

**Separate Input/Output Buffers:**

For models where the input and output shapes differ or when you want to keep input and output separate:

.. code-block:: cpp

    void processBlock(float** input_audio, float** output_audio, int num_samples) {
        size_t output_samples = inference_handler.process(
            input_audio,                // const float* const* - input data
            num_samples,                // number of input samples
            output_audio,               // float* const* - output buffer
            output_buffer_size          // maximum output buffer size
        );
        // output_samples contains the actual number of samples written
    }

**Multi-Tensor Processing:**

For models with multiple input and output tensors (e.g., audio + control parameters):

.. code-block:: cpp

    // Prepare input and output data for multiple tensors in initialization
    const float* const* const* input_data = new const float* const*[2];
    float* const* const* output_data = new float* const*[2];

    void processBlock(float** audio_input, float* control_params, 
                     float** audio_output, float* confidence_output, 
                     int num_audio_samples) {
        
        input_data[0] = audio_input;                           // Tensor 0: audio data
        input_data[1] = (const float* const*) &control_params; // Tensor 1: control parameters
        
        output_data[0] = audio_output;                        // Tensor 0: processed audio
        output_data[1] = (float* const*) &confidence_output;  // Tensor 1: confidence scores
        
        // Specify number of samples for each tensor
        size_t input_samples[] = {num_audio_samples, 4};      // Audio samples, 4 control values
        size_t output_samples[] = {num_audio_samples, 1};     // Audio samples, 1 confidence value
        
        // Process all tensors simultaneously
        size_t* processed_samples = inference_handler.process(
            input_data, input_samples,
            output_data, output_samples
        );
    }

    // Clean up
    delete[] input_data;
    delete[] output_data;

5.2. Push/Pop Data Method
~~~~~~~~~~~~~~~~~~~~~~~~~

The :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods enable decoupled processing where input and output operations are separated. This is particularly useful for:

- Models with different input/output timing requirements
- Buffered processing scenarios

.. warning::
    The :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` methods should only be called from the same thread. Otherwise you may run into race conditions or other threading issues.

**Basic Decoupled Processing:**

.. code-block:: cpp

    void processBlock(float** input_audio, float** output_audio, int num_samples) {
        // Push input data to the inference pipeline
        inference_handler.push_data(
            input_audio,                // const float* const* - input data
            num_samples,                // number of input samples
            0                          // tensor index (optional, defaults to 0)
        );
        
        // Pop processed output data from the pipeline
        size_t received_samples = inference_handler.pop_data(
            output_audio,              // float* const* - output buffer
            num_samples,               // maximum number of output samples
            0                          // tensor index (optional, defaults to 0)
        );
        
        // received_samples contains the actual number of samples retrieved
    }

**Multi-Tensor Decoupled Processing:**

.. code-block:: cpp

    // Prepare input and output data for multiple tensors in initialization
    const float* const* const* input_data = new const float* const*[2];
    float* const* const* output_data = new float* const*[2];

    void processBlock(float** audio_input, float* control_params,
                     float** audio_output, float* confidence_output,
                     int num_audio_samples) {
        
        // Push data for multiple tensors
        input_data[0] = audio_input;
        input_data[1] = (const float* const*) &control_params;
        
        size_t input_samples[] = {num_audio_samples, 4};
        inference_handler.push_data(input_data, input_samples);
        
        // Pop data for multiple tensors
        output_data[0] = audio_output;
        output_data[1] = (float* const*) &confidence_output;
        
        size_t output_samples[] = {num_audio_samples, 1};
        size_t* received_samples = inference_handler.pop_data(output_data, output_samples);
    }
    
    // Clean up
    delete[] input_data;
    delete[] output_data;

5.3. Processing Non-Streamable Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some neural networks require additional input parameters or output values that do not need to be time-aligned and can therefore be updated asynchronously with the host buffers. For non-streamable tensors (those with ``preprocess_input_size`` or ``postprocess_output_size`` set to 0), you can use the :cpp:class:`anira::PrePostProcessor` methods to submit or retrieve additional values.

**Setting and Getting Non-Streamable Values:**

.. code-block:: cpp

    // In your custom PrePostProcessor or directly via the :cpp
    
    // Set input values for non-streamable tensors
    pp_processor.set_input(gain_value, tensor_index, sample_index);
    pp_processor.set_input(threshold_value, tensor_index, sample_index + 1);
    
    // Get output values from non-streamable tensors  
    float confidence_score = pp_processor.get_output(tensor_index, sample_index);
    float peak_gain = pp_processor.get_output(tensor_index, sample_index + 1);

**Example: Audio Effect with Control Parameters:**

.. code-block:: cpp

    void processBlock(float** audio_data, int num_samples, 
                     float gain_param, float threshold_param) {
        
        // Set control parameters for non-streamable tensor (tensor index 1)
        pp_processor.set_input(gain_param, 1, 0);
        pp_processor.set_input(threshold_param, 1, 1);
        
        // Process audio (tensor index 0 is streamable audio data)
        inference_handler.process(audio_data, num_samples);
        
        // Retrieve computed values from non-streamable output tensor (tensor index 1)
        float computed_peak_gain = pp_processor.get_output(1, 0);
        float signal_energy = pp_processor.get_output(1, 1);
    }

..  note::
    The functions :cpp:func:`anira::PrePostProcessor::set_input` and :cpp:func:`anira::PrePostProcessor::get_output` can be called from any thread, allowing you to update control parameters or retrieve additional values asynchronously without blocking the real-time audio processing thread.
