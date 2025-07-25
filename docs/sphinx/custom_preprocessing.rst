Custom Pre/Post Processing
===========================

If your model requires custom pre- or post-processing, you can inherit from the :cpp:class:`anira::PrePostProcessor` class and override the :cpp:func:`anira::PrePostProcessor::pre_process` and :cpp:func:`anira::PrePostProcessor::post_process` methods to match your model's specific requirements.

The :cpp:func:`anira::PrePostProcessor::pre_process` method receives input data from the application through a vector of :cpp:class:`anira::RingBuffer` instances and transforms them into output buffers (a vector of :cpp:type:`anira::BufferF`). These output buffers are then fed directly to the inference engine.

The :cpp:func:`anira::PrePostProcessor::post_process` method receives inference results through a vector of :cpp:type:`anira::BufferF` instances and writes them to output ring buffers (a vector of :cpp:class:`anira::RingBuffer`). The :cpp:class:`anira::InferenceHandler` then retrieves samples from these ring buffers and returns them to the audio application.

Non-streamable tensors, such as control parameters or static values, can be handled using the :cpp:func:`anira::PrePostProcessor::get_input` and :cpp:func:`anira::PrePostProcessor::set_input` methods for input data, and :cpp:func:`anira::PrePostProcessor::get_output` and :cpp:func:`anira::PrePostProcessor::set_output` methods for output data. These methods allow you to store and retrieve non-streamable tensor values in a thread-safe manner.

Understanding Streamable vs Non-Streamable Tensors
--------------------------------------------------

Anira supports two types of tensors that require different handling in custom preprocessors:

**Streamable Tensors:**
- Data that flows continuously (time-varying signals)
- Have ``preprocess_input_size > 0`` and ``postprocess_output_size > 0``
- Data comes from :cpp:class:`anira::RingBuffer` instances via the ``input`` parameter
- Use helper methods like ``pop_samples_from_buffer()`` to extract data

**Non-Streamable Tensors:**
- Control parameters, static values, or metadata (non-time-varying)
- Have ``preprocess_input_size == 0`` and ``postprocess_output_size == 0``  
- Data comes from the preprocessor's internal storage via ``get_input()`` and ``set_input()`` methods
- Must be manually written to and read from :cpp:type:`anira::BufferF` tensors
- Note: Non-streamable tensors have no channel count (always use channel 0)

Basic Custom PrePostProcessor Implementation
--------------------------------------------

.. code-block:: cpp

    #include <anira/anira.h>

    class CustomPrePostProcessor : public anira::PrePostProcessor {
    public:
        // Inherit constructor from base class
        using anira::PrePostProcessor::PrePostProcessor;

        virtual void pre_process(std::vector<anira::RingBuffer>& input, 
                                std::vector<anira::BufferF>& output, 
                                [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
            for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
                if (m_inference_config.get_preprocess_input_size()[i] > 0) {
                    // Streamable tensor: extract audio data from ring buffer
                    pop_samples_from_buffer(input[i], output[i], 
                                          m_inference_config.get_preprocess_input_size()[i]);
                } else {
                    // Non-streamable tensor: get data from internal storage
                    // Note: Non-streamable tensors always use channel 0
                    for (size_t sample = 0; sample < m_inference_config.get_tensor_input_size()[i]; ++sample) {
                        output[i].set_sample(0, sample, get_input(i, sample));
                    }
                }
            }
        }

        virtual void post_process(std::vector<anira::BufferF>& input, 
                                 std::vector<anira::RingBuffer>& output, 
                                 [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
            for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
                if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                    // Streamable tensor: write audio data to ring buffer
                    push_samples_to_buffer(input[i], output[i], 
                                         m_inference_config.get_postprocess_output_size()[i]);
                } else {
                    // Non-streamable tensor: store data in internal storage
                    // Note: Non-streamable tensors always use channel 0
                    for (size_t sample = 0; sample < m_inference_config.get_tensor_output_size()[i]; ++sample) {
                        set_output(input[i].get_sample(0, sample), i, sample);
                    }
                }
            }
        }
    };

Available Helper Methods
~~~~~~~~~~~~~~~~~~~~~~~~

The :cpp:class:`anira::PrePostProcessor` provides several helper methods to facilitate data handling between audio buffers and neural network tensors. Here are the key methods you can use:

+-----------------------------------------------------------------------+------------------------------------------------+
| Method                                                                | Description                                    |
+=======================================================================+================================================+
| :cpp:func:`anira::PrePostProcessor::pop_samples_from_buffer`          | Extracts samples from input ring buffer and    |
|                                                                       | writes them to output buffer. Multiple         |
|                                                                       | overloads support different windowing modes.   |
+-----------------------------------------------------------------------+------------------------------------------------+
| :cpp:func:`anira::PrePostProcessor::push_samples_to_buffer`           | Writes samples from input buffer to output     |
|                                                                       | ring buffer.                                   |
+-----------------------------------------------------------------------+------------------------------------------------+
| :cpp:func:`anira::PrePostProcessor::get_input`                        | Retrieves non-streamable input values from     |
|                                                                       | internal storage (thread-safe).                |
+-----------------------------------------------------------------------+------------------------------------------------+
| :cpp:func:`anira::PrePostProcessor::set_input`                        | Sets non-streamable input values to internal   |
|                                                                       | storage (thread-safe).                         |
+-----------------------------------------------------------------------+------------------------------------------------+
| :cpp:func:`anira::PrePostProcessor::get_output`                       | Retrieves non-streamable output values from    |
|                                                                       | internal storage (thread-safe).                |
+-----------------------------------------------------------------------+------------------------------------------------+
| :cpp:func:`anira::PrePostProcessor::set_output`                       | Sets non-streamable output values to internal  |
|                                                                       | storage (thread-safe).                         |
+-----------------------------------------------------------------------+------------------------------------------------+

Integration with InferenceHandler
---------------------------------

Once you've implemented your custom preprocessor, integrate it with the inference system:

.. code-block:: cpp

    // First create your inference configuration
    anira::InferenceConfig inference_config(/* your config parameters */);
    
    // Create your custom preprocessor instance
    // Note: The preprocessor requires an InferenceConfig reference
    CustomPrePostProcessor pp_processor(inference_config);
    
    // Create InferenceHandler with custom preprocessor
    anira::InferenceHandler inference_handler(pp_processor, inference_config);

.. note::
    The preprocess and postprocess methods are called from the audio thread and must be real-time safe. Avoid operations that could cause blocking, memory allocation, or other non-deterministic behavior that could introduce audio dropouts or latency issues.