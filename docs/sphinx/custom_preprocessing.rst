Custom Pre/Post Processing
===========================

If your model requires custom pre- or post-processing, you can inherit from the :cpp:class:`anira::PrePostProcessor` class and override the :cpp:func:`anira::PrePostProcessor::pre_process` and :cpp:func:`anira::PrePostProcessor::post_process` methods to match your model's specific requirements.

The :cpp:func:`anira::PrePostProcessor::pre_process` method receives input data from the application through a vector of :cpp:class:`anira::RingBuffer` instances and transforms them into output buffers (a vector of :cpp:type:`anira::BufferF`). These output buffers are then fed directly to the inference engine.

The :cpp:func:`anira::PrePostProcessor::post_process` method receives inference results through a vector of :cpp:type:`anira::BufferF` instances and writes them to output ring buffers (a vector of :cpp:class:`anira::RingBuffer`). The :cpp:class:`anira::InferenceHandler` then retrieves samples from these ring buffers and returns them to the audio application.

In addition to :cpp:func:`anira::PrePostProcessor::pre_process` and :cpp:func:`anira::PrePostProcessor::post_process`, the :cpp:class:`anira::PrePostProcessor` provides two optional hooks, :cpp:func:`anira::PrePostProcessor::before_inference` and :cpp:func:`anira::PrePostProcessor::after_inference`, which are called on the inference thread immediately before and after the backend runs. See `Inference Thread Hooks`_ below.

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

Inference Thread Hooks
----------------------

The :cpp:class:`anira::PrePostProcessor` provides two additional virtual methods that can be overridden: :cpp:func:`anira::PrePostProcessor::before_inference` and :cpp:func:`anira::PrePostProcessor::after_inference`. Unlike :cpp:func:`anira::PrePostProcessor::pre_process` and :cpp:func:`anira::PrePostProcessor::post_process`, which are called on the audio thread, these hooks are called on the inference thread — immediately before and immediately after the inference engine runs. Their default implementations do nothing.

:cpp:func:`anira::PrePostProcessor::before_inference` receives the input tensors (a vector of :cpp:type:`anira::BufferF`) right before they are fed to the backend. :cpp:func:`anira::PrePostProcessor::after_inference` receives the output tensors right after the backend has produced them.

These hooks are the correct place to handle state that must flow from one inference to the next, such as recurrent (hidden) state feedback in stateful models. This cannot be done reliably in :cpp:func:`anira::PrePostProcessor::pre_process`, because it is called at submission time: when more than one inference is queued, all of their input tensors have already been filled before the first inference runs, so any cross-inference state written there would be stale. In contrast, :cpp:func:`anira::PrePostProcessor::after_inference` runs before the inference is marked as done and before the next session-exclusive inference can be dispatched, so state captured there is guaranteed to be visible to the next :cpp:func:`anira::PrePostProcessor::before_inference` call.

.. code-block:: cpp

    class StatefulPrePostProcessor : public anira::PrePostProcessor {
    public:
        using anira::PrePostProcessor::PrePostProcessor;

        virtual void before_inference(std::vector<anira::BufferF>& input,
                                      [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
            // Splice the hidden state captured from the previous inference
            // into the corresponding input tensor, e.g. input[1]
        }

        virtual void after_inference(std::vector<anira::BufferF>& output,
                                     [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
            // Capture the updated hidden state from the corresponding
            // output tensor, e.g. output[1], for the next inference
        }
    };

.. note::
    For stateful models, give the model config the stateful state (``"state": "stateful"`` in the model file, ``cfg.state(ANIRA_MODEL_STATEFUL)`` on the builder). This guarantees that inferences execute strictly in submission order and never concurrently, which makes these hooks the only safe place to splice in cross-inference state.

Integration with InferenceHandler
---------------------------------

Once you've implemented your custom preprocessor, integrate it with the inference system:

.. code-block:: cpp

    // First create your inference configuration: the model and contract files (section 1 of
    // the usage guide), bridged to the InferenceConfig the runtime of this pre-release takes
    anira::ModelConfig model_config = anira::ModelConfig::from_file("model.json");
    anira::ContractHandle contract = anira::ContractHandle::from_file("contract.json");
    anira::InferenceConfig inference_config =
        anira::v3compat::to_inference_config(model_config, contract);
    
    // Create your custom preprocessor instance
    // Note: The preprocessor requires an InferenceConfig reference
    CustomPrePostProcessor pp_processor(inference_config);
    
    // Create InferenceHandler with custom preprocessor
    anira::InferenceHandler inference_handler(pp_processor, inference_config);

.. note::
    The preprocess and postprocess methods are called from the audio thread and must be real-time safe. Avoid operations that could cause blocking, memory allocation, or other non-deterministic behavior that could introduce audio dropouts or latency issues.