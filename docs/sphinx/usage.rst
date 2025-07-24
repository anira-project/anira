Usage Guide
===========

This guide provides comprehensive instructions for integrating anira into your real-time audio processing applications.

Class Reference Examples
------------------------

Here are different ways to reference classes in Sphinx:

**Basic class reference:**
The :cpp:class:`anira::PrePostProcessor` class enables pre- and post-processing steps.

**Namespace reference:**
All anira classes are in the :cpp:class:`anira` namespace.

**Method reference:**
Use the :cpp:func:`anira::InferenceHandler::process` method for real-time processing.

**Advanced Code Block Referencing:**

You can reference classes and methods within code comments:

.. code-block:: cpp

   // The anira::InferenceHandler manages the inference process
   anira::InferenceHandler handler(processor, config);
   
   // Call the process() method for real-time audio processing
   handler.process(audio_data, num_samples);

**Highlighted Code with References:**

.. code-block:: cpp
   :emphasize-lines: 2, 5
   :linenos:
   :caption: Example using anira::InferenceHandler

   // Initialize the inference system
   anira::InferenceHandler handler(processor, config);  // <- Main class
   handler.prepare(host_config);
   
   handler.process(audio_data, num_samples);  // <- Key method

**Code with Cross-references:**

The following example shows how to use :cpp:class:`anira::InferenceHandler`:

.. code-block:: cpp
   :name: inference-handler-example

   void setup_inference() {
       // Create configuration - see anira::InferenceConfig documentation
       anira::InferenceConfig config(model_data, tensor_shapes, 42.66f);
       
       // Initialize processor - inherits from anira::PrePostProcessor
       CustomProcessor processor(config);
       
       // Create handler - the main anira::InferenceHandler instance
       anira::InferenceHandler handler(processor, config);
   }

**Full class documentation (uncomment to use):**

..
   .. doxygenclass:: anira::PrePostProcessor
      :members:

**Specific method documentation:**

..
   .. doxygenfunction:: anira::InferenceHandler::process

Overview
--------

Anira provides the following structures and classes to help you integrate real-time audio processing with your machine learning models:

+------------------+--------------------------------------------------------------------------+
| Class            | Description                                                              |
+==================+==========================================================================+
| ContextConfig    | **Optional:** The configuration structure that defines the context       |
|                  | across all anira instances. Here you can define the behaviour of the     |
|                  | thread pool, such as specifying the number of threads.                   |
+------------------+--------------------------------------------------------------------------+
| InferenceHandler | Manages audio processing/inference for the real-time thread,             |
|                  | offloading inference to the thread pool and updating the real-time       |
|                  | thread buffers with processed audio. This class provides the main        |
|                  | interface for interacting with the library.                              |
+------------------+--------------------------------------------------------------------------+
| InferenceConfig  | A configuration structure for defining model specifics such as           |
|                  | input/output shape, model details such as maximum inference time,        |
|                  | and more. Each InferenceHandler instance must be constructed with        |
|                  | this configuration.                                                      |
+------------------+--------------------------------------------------------------------------+
| PrePostProcessor | Enables pre- and post-processing steps before and after inference.       |
|                  | Either use the default PrePostProcessor or inherit from this class       |
|                  | for custom processing.                                                   |
+------------------+--------------------------------------------------------------------------+
| HostConfig       | A structure for defining the host configuration: buffer size             |
|                  | and sample rate.                                                         |
+------------------+--------------------------------------------------------------------------+

Step 1: Define your Model Configuration
---------------------------------------

Start by specifying your model configuration using ``anira::InferenceConfig``. This includes the model path, input/output shapes, and other critical settings that match the requirements of your model.

Step 1.1: Define the model information and the corresponding inference backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First pass the model information and the corresponding inference backend in a ``std::vector<anira::ModelData>``. ``anira::ModelData`` offers two ways to define the model information:

1. Pass the model path as a string:

.. code-block:: cpp

   {std::string model_path, anira::InferenceBackend backend}

2. Pass the model data as binary information:

.. code-block:: cpp

   {void* model_data, size_t model_size, anira::InferenceBackend backend}

.. note::
   Defining the model data as binary information is only possible for the ``anira::ONNX`` until now.

Now define your model information in a ``std::vector<anira::ModelData>``.

.. code-block:: cpp

   std::vector<anira::ModelData> model_data = {
       {"path/to/your/model.pt", anira::InferenceBackend::LIBTORCH},
       {"path/to/your/model.onnx", anira::InferenceBackend::ONNX},
       {"path/to/your/model.tflite", anira::InferenceBackend::TFLITE}
   };

.. note::
   It is not necessary to submit a model for each backend anira was built with, only the one you want to use.

Step 1.2: Define the input and output shapes of the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the next step, define the input and output shapes of the model for each backend in a ``std::vector<anira::TensorShape>``. The ``anira::TensorShape`` is defined as follows:

.. code-block:: cpp

   {std::vector<int64_t> input_shape, std::vector<int64_t> output_shape, (optional) anira::InferenceBackend}

Now define the input and output shapes of your model for each backend used in the ``std::vector<anira::ModelData>``.

.. code-block:: cpp

   std::vector<anira::TensorShape> tensor_shapes = {
       {{{1, 1, 15380}}, {{1, 1, 2048}}, anira::InferenceBackend::LIBTORCH},
       {{{1, 1, 15380}}, {{1, 1, 2048}}, anira::InferenceBackend::ONNX},
       {{{1, 15380, 1}}, {{1, 2048, 1}}, anira::InferenceBackend::TFLITE}
   };

.. note::
   If the input and output shapes of the model are the same for all backends, you can also define only one ``anira::TensorShape`` without a specific ``anira::InferenceBackend``.

Step 1.3: Define the anira::InferenceConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, define the necessary ``anira::InferenceConfig`` with the model information, input/output shapes and the maximum inference time in ms. The maximum inference time is the measured worst case inference time. If the inference time during execution exceeds this value, it is likely that the audio signal will contain artifacts.

.. code-block:: cpp

   anira::InferenceConfig inference_config (
       model_data, // std::vector<anira::ModelData>
       tensor_shapes, // std::vector<anira::TensorShape>
       42.66f // Maximum inference time in ms
   );

There are also some optional parameters that can be set in the ``anira::InferenceConfig``:

+---------------------------+--------------------------------------------------------+
| Parameter                 | Description                                            |
+===========================+========================================================+
| internal_latency          | Type: ``unsigned int``, default: ``0``. Submit if      |
|                           | your model has an internal latency. This allows the    |
|                           | latency calculation to take it into account.           |
+---------------------------+--------------------------------------------------------+
| warm_up                   | Type: ``unsigned int``, default: ``0``. Defines the    |
|                           | number of warm-up iterations before starting the       |
|                           | inference process.                                     |
+---------------------------+--------------------------------------------------------+
| session_exclusive_processor | Type: ``bool``, default: ``false``. If set to        |
|                           | ``true``, the session will use an exclusive processor  |
|                           | for inference and therefore cannot be processed        |
|                           | parallel. Necessary for e.g. stateful models.          |
+---------------------------+--------------------------------------------------------+
| num_parallel_processors   | Type: ``unsigned int``, default:                       |
|                           | ``std::thread::hardware_concurrency() / 2``. Defines   |
|                           | the number of parallel processors that can be used     |
|                           | for the inference.                                     |
+---------------------------+--------------------------------------------------------+
| blocking_ratio            | Type: ``float``, default: ``0.0f``. This should be a   |
|                           | value between ``0.f`` and ``1.f``. It specifies the    |
|                           | proportion of available processing time that the       |
|                           | library will try to acquire new data from the          |
|                           | inference threads on the real-time thread. This is a   |
|                           | controversial parameter and should be used with        |
|                           | caution.                                               |
+---------------------------+--------------------------------------------------------+

Step 2: Create a PrePostProcessor Instance
------------------------------------------

If your model does not require any specific pre- or post-processing, you can use the default :cpp:class:`anira::PrePostProcessor`. This is likely to be the case if the input and output shapes of the model are the same, the batchsize is 1, and your model operates in the time domain.

.. code-block:: cpp

   // Create an instance of anira::PrePostProcessor
   anira::PrePostProcessor pp_processor(inference_config);

Custom Pre/Post Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your model requires custom pre- or post-processing, you can inherit from the :cpp:class:`anira::PrePostProcessor` class and overwrite the :cpp:func:`anira::PrePostProcessor::pre_process` and :cpp:func:`anira::PrePostProcessor::post_process` methods so that they match your model's requirements.

In the ``pre_process`` method, we get the input samples from the audio application through an ``std::vector<anira::RingBuffer>`` and push them into the output buffer, which is an ``std::vector<anira::BufferF>``. This output buffer is then used for inference.

In the ``post_process`` method we get the input samples through an ``std::vector<anira::BufferF>`` and push them into the output buffer, which is an ``std::vector<anira::RingBuffer>``. The samples from this output buffer are then returned to the audio application by the :cpp:class:`anira::InferenceHandler`.

.. code-block:: cpp

   #include <anira/anira.h>

   class CustomPrePostProcessor : public anira::PrePostProcessor {
   public:
       using anira::PrePostProcessor::PrePostProcessor;

       virtual void pre_process(std::vector<anira::RingBuffer>& input, 
                               std::vector<anira::BufferF>& output, 
                               [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
           pop_samples_from_buffer(input[0], output[0], 
                                 m_inference_config.get_tensor_output_size()[0], 
                                 m_inference_config.get_tensor_input_size()[0]-m_inference_config.get_tensor_output_size()[0]);
       };
   };

.. note::
   The ``anira::PrePostProcessor`` class provides some methods to help you implement your own pre- and post-processing.

Available Helper Methods
~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------------------+
| Method                            | Description                                   |
+===================================+===============================================+
| pop_samples_from_buffer           | Pop output.size() samples from the input      |
| (input, output)                   | buffer and push them into the output buffer.  |
+-----------------------------------+-----------------------------------------------+
| pop_samples_from_buffer           | Pop num_new_samples new samples from the      |
| (input, output, num_new_samples,  | input buffer and get num_old_samples already  |
| num_old_samples)                  | poped samples from the input buffer and push  |
|                                   | them into the output buffer. The order of     |
|                                   | the samples in the output buffer is from      |
|                                   | oldest to newest.                             |
+-----------------------------------+-----------------------------------------------+
| pop_samples_from_buffer           | Same as the above method, but starts writing  |
| (input, output, num_new_samples,  | to the output buffer at the offset.           |
| num_old_samples, offset)          |                                               |
+-----------------------------------+-----------------------------------------------+
| push_samples_to_buffer            | Pushes input.size() samples from the input    |
| (input, output)                   | buffer into the output buffer.                |
+-----------------------------------+-----------------------------------------------+

Additional Tensor Values
~~~~~~~~~~~~~~~~~~~~~~~~

Some neural networks not only require audio data as input and output tensors. For example, some models require additional input parameters or output values, like e.g. a prediction of the model's confidence. In this case you can use the ``anira::PrePostProcessor`` to submit or retrieve additional values.

+-------------------------------+-----------------------------------------------+
| Method                        | Description                                   |
+===============================+===============================================+
| set_input(input, i, j)        | Sets the input value at position i, j in the  |
|                               | input tensor.                                 |
+-------------------------------+-----------------------------------------------+
| set_output(output, i, j)      | Sets the output value at position i, j in     |
|                               | the output tensor.                            |
+-------------------------------+-----------------------------------------------+
| get_input(i, j)               | Returns the input value at position i, j in   |
|                               | the input tensor.                             |
+-------------------------------+-----------------------------------------------+
| get_output(i, j)              | Returns the output value at position i, j in  |
|                               | the output tensor.                            |
+-------------------------------+-----------------------------------------------+

Step 3: Create an InferenceHandler Instance
-------------------------------------------

In your application, you will need to create an instance of the :cpp:class:`anira::InferenceHandler` class. This class is responsible for managing the inference process, including threading and real-time constraints. The constructor takes as arguments an instance of the default or custom :cpp:class:`anira::PrePostProcessor` and an instance of the :cpp:class:`anira::InferenceConfig` structure.

.. code-block:: cpp

   // Sample initialization in your application's initialization function

   // Default PrePostProcessor
   anira::PrePostProcessor pp_processor(inference_config);
   // or custom PrePostProcessor
   CustomPrePostProcessor pp_processor(inference_config);

   // Create an InferenceHandler instance
   anira::InferenceHandler inference_handler(pp_processor, inference_config);

Optional: Define the Context Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to define a custom context configuration, you can do so by creating an instance of the ``anira::ContextConfig`` structure. This structure allows you to define the behaviour of the thread pool, by specifying the number of threads.

.. code-block:: cpp

   // Use the existing anira::InferenceConfig and anira::PrePostProcessor instances

   // Create an instance of anira::ContextConfig
   anira::ContextConfig context_config {
       4 // Number of threads
   };

   // Create an InferenceHandler instance
   anira::InferenceHandler inference_handler(pp_processor, inference_config, context_config);

Step 4: Allocate Memory Before Processing
-----------------------------------------

Before processing audio data, the ``prepare`` method of the ``anira::InferenceHandler`` instance must be called. This allocates all necessary memory in advance. The ``prepare`` method needs an instance of ``anira::HostConfig`` which defines the buffer size and sample rate of the host application.

We also need to select the inference backend we want to use. Depending on the backends you enabled during the build process, you can choose amongst ``anira::LIBTORCH``, ``anira::ONNX``, ``anira::TFLITE`` and ``anira::CUSTOM``.

After preparing the ``anira::InferenceHandler``, you can get the latency of the inference process in samples by calling the ``get_latency`` method and use this information to compensate for the latency in your real-time audio application.

.. code-block:: cpp

   void prepare_audio_processing(double sample_rate, int buffer_size) {

       // Create an instance of anira::HostConfig
       anira::HostConfig host_config {
           buffer_size,
           sample_rate
       };

       inference_handler.prepare(host_config);

       // Select the inference backend
       inference_handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);
       
       // Get the latency of the inference process in samples
       int latency_in_samples = inference_handler.get_latency();
   }

Step 5: Real-time Audio Processing
----------------------------------

Now we are ready to process audio in the process callback of our real-time audio application. The process method of the ``anira::InferenceHandler`` instance takes the input samples for all channels as an array of float pointers - ``float**``, and after calling the process method, the data is overwritten with the processed output.

.. code-block:: cpp

   // Real-time safe audio processing in the process callback of your application
   void process(float** audio_data, int num_samples) {
       inference_handler.process(audio_data, num_samples)
   }
   // audio_data now contains the processed audio samples

Custom Backend Processors
-------------------------

To use a custom backend processor, inherit from the ``anira::BackendBase`` class and overwrite the ``process`` and ``prepare`` methods. The ``process`` method is called when the ``anira::InferenceBackend::CUSTOM`` backend is selected.

The ``process`` method takes two ``anira::BufferF`` instances as input and output buffers and a ``std::shared_ptr<anira::SessionElement>`` session element. The session element is necessary to e.g. send or retrieve additional values submitted by the pre- and post-processor.

The custom backend enables the integration of additional inference engines, customization of existing engines, or the implementation of a simple roundtrip/bypass backend that directly returns input samples, bypassing the inference stage.

Example: Bypass Backend
~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to implement a custom bypass backend for a CNN model, where 15380 past samples are used as input and 2048 samples are returned as output. In order to bypass the inference stage, we just have to return the last 2048 samples of the input buffer.

.. code-block:: cpp

   #include <anira/anira.h>

   class BypassProcessor : public anira::BackendBase {
   public:
       BypassProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

       void process(anira::BufferF &input, anira::BufferF &output, [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
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

After defining the custom backend processor, you can create an instance of the ``BypassProcessor`` class and pass it to the ``anira::InferenceHandler`` instance as an additional argument in the constructor. The ``anira::InferenceHandler`` will then use the ``BypassProcessor`` instance when the ``anira::CUSTOM`` backend is selected.

.. code-block:: cpp

   // Create an instance of the custom CustomProcessor
   BypassProcessor bypass_processor(inference_config);
   // In Step 3: Create an InferenceHandler Instance
   anira::InferenceHandler inference_handler(pp_processor, inference_config, bypass_processor);

.. note::
   If you want to implement a custom inference backend use the existing backend implementations as a reference.
