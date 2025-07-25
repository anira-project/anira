Custom Backend Definition
=========================

Anira provides the flexibility to implement custom inference backends through the :cpp:class:`anira::BackendBase` class. This enables integration of additional inference engines, customization of existing engines, or implementation of specialized processing logic such as bypass or roundtrip backends.

Understanding the Backend Interface
-----------------------------------

The :cpp:class:`anira::BackendBase` class provides the foundation for all inference backends in anira. When creating a custom backend, you need to inherit from this class and implement the required virtual methods.

**Core Virtual Methods:**

+---------------------------+--------------------------------------------------------------------------+
| Method                    | Description                                                              |
+===========================+==========================================================================+
| ``prepare()``             | Called once during initialization to set up the backend. Use this        |
|                           | method to load models, allocate memory, or configure the inference       |
|                           | engine.                                                                  |
+---------------------------+--------------------------------------------------------------------------+
| ``process()``             | Called for each inference operation. This method receives input          |
|                           | buffers, performs the actual inference, and writes results to output     |
|                           | buffers.                                                                 |
+---------------------------+--------------------------------------------------------------------------+

**Method Signatures:**

.. code-block:: cpp

    class CustomBackend : public anira::BackendBase {
    public:
        CustomBackend(anira::InferenceConfig& inference_config) 
            : anira::BackendBase(inference_config) {}

        // Initialize the backend (called once)
        void prepare() override {
            // Load models, allocate memory, configure inference engine
        }

        // Process inference (called repeatedly in real-time)
        void process(std::vector<anira::BufferF>& input, std::vector<anira::BufferF>& output, 
                    std::shared_ptr<anira::SessionElement> session) override {
            // Perform inference and write results to output buffers
        }
    };

Implementing the Backend
------------------------

**Constructor Requirements:**

Your custom backend must accept an :cpp:struct:`anira::InferenceConfig` reference in its constructor and pass it to the base class. This configuration provides access to model information, tensor shapes, and other inference parameters.

.. code-block:: cpp

    class CustomBackend : public anira::BackendBase {
    public:
        CustomBackend(anira::InferenceConfig& inference_config) 
            : anira::BackendBase(inference_config) {
            // Optional: Store additional configuration or initialize members
        }
    };

**Implementing prepare():**

The :cpp:func:`anira::BackendBase::prepare` method is called once during the initialization phase. Use this method to:

- Load neural network models
- Initialize inference engines or libraries
- Allocate persistent memory buffers
- Configure backend-specific settings

.. code-block:: cpp

    void prepare() override {
        // Example: Load a custom inference engine
        auto& model_data = m_inference_config.get_model_data();
        if (!model_data.empty()) {
            std::string model_path = model_data[0].model_path;
            // Load your model here
            custom_engine.load_model(model_path);
        }
        
        // Example: Pre-allocate inference buffers
        auto input_shape = m_inference_config.get_tensor_input_shape();
        auto output_shape = m_inference_config.get_tensor_output_shape();
        
        inference_input_buffer.resize(input_shape[0]);
        inference_output_buffer.resize(output_shape[0]);
    }

**Implementing process():**

The :cpp:func:`anira::BackendBase::process` method is called for each inference operation. This method receives:

- ``input``: vector of :cpp:type:`anira::BufferF` containing input data from the pre-processor for all input tensors
- ``output``: vector of :cpp:type:`anira::BufferF` where results should be written for all output tensors
- ``session``: shared pointer of :cpp:type:`anira::SessionElement` for accessing additional session data

The vectors contain one :cpp:type:`anira::BufferF` for each tensor defined in your model. Most audio processing models have a single input and single output tensor (both at index 0), but some models may have multiple tensors for different purposes (e.g., audio data, control parameters, confidence outputs).

.. code-block:: cpp

    void process(std::vector<anira::BufferF>& input, std::vector<anira::BufferF>& output, 
                std::shared_ptr<anira::SessionElement> session) override {
        // Process each tensor - typically there's one input and one output tensor
        for (size_t tensor_idx = 0; tensor_idx < input.size() && tensor_idx < output.size(); ++tensor_idx) {
            auto& input_buffer = input[tensor_idx];
            auto& output_buffer = output[tensor_idx];
            
            // Copy input data to inference engine format
            for (size_t channel = 0; channel < input_buffer.get_num_channels(); ++channel) {
                auto input_ptr = input_buffer.get_read_pointer(channel);
                // Copy to your inference engine's input format
                std::copy(input_ptr, input_ptr + input_buffer.get_num_samples(), 
                         inference_input_buffer.begin());
            }
            
            // Perform inference
            custom_engine.infer(inference_input_buffer, inference_output_buffer);
            
            // Copy results to output buffer
            for (size_t channel = 0; channel < output_buffer.get_num_channels(); ++channel) {
                auto output_ptr = output_buffer.get_write_pointer(channel);
                std::copy(inference_output_buffer.begin(), 
                         inference_output_buffer.begin() + output_buffer.get_num_samples(),
                         output_ptr);
            }
        }
    }

Backend Integration
-------------------

Once your custom backend is implemented, integrate it with the :cpp:class:`anira::InferenceHandler`:

.. code-block:: cpp

    // Create your custom backend instance
    CustomBackend custom_backend(inference_config);
    
    // Create InferenceHandler with custom backend
    anira::InferenceHandler inference_handler(pp_processor, inference_config, custom_backend);
    
    // Select the custom backend
    inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);

Example: Bypass Backend
-----------------------

The following example demonstrates a simple bypass backend that returns the last portion of the input buffer as output, effectively bypassing the inference stage:

.. code-block:: cpp

    #include <anira/anira.h>

    class BypassProcessor : public anira::BackendBase {
    public:
        BypassProcessor(anira::InferenceConfig& inference_config) 
            : anira::BackendBase(inference_config) {}

        void prepare() override {
            // No preparation needed for bypass
        }

        void process(std::vector<anira::BufferF>& input, std::vector<anira::BufferF>& output, 
                    [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
            // Process each tensor pair
            for (size_t tensor_idx = 0; tensor_idx < input.size() && tensor_idx < output.size(); ++tensor_idx) {
                auto& input_buffer = input[tensor_idx];
                auto& output_buffer = output[tensor_idx];
                
                auto equal_channels = input_buffer.get_num_channels() == output_buffer.get_num_channels();
                auto sample_diff = input_buffer.get_num_samples() - output_buffer.get_num_samples();

                if (equal_channels && sample_diff >= 0) {
                    for (size_t channel = 0; channel < input_buffer.get_num_channels(); ++channel) {
                        auto write_ptr = output_buffer.get_write_pointer(channel);
                        auto read_ptr = input_buffer.get_read_pointer(channel);

                        // Copy the last output.get_num_samples() from input to output
                        for (size_t i = 0; i < output_buffer.get_num_samples(); ++i) {
                            write_ptr[i] = read_ptr[i + sample_diff];
                        }
                    }
                } else {
                    // Clear output if dimensions don't match
                    output_buffer.clear();
                }
            }
        }
    };

.. note::
    When implementing custom inference backends, refer to the existing backend implementations in the anira source code (``src/backends/``) for additional guidance and best practices. Each backend demonstrates different approaches to handling model loading, memory management, and inference execution.
