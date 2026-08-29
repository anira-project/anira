#include <anira/InferenceConfig.h>
#include <anira/backends/BackendBase.h>
#include <anira/backends/LibTorchProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/MemoryBlock.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <sstream>
#include <vector>

// Avoid min/max macro conflicts on Windows for LibTorch compatibility
#ifdef _WIN32
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
#endif

// LibTorch headers trigger many warnings; disabling for cleaner build logs
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#endif

#include <ATen/core/ATen_fwd.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/utils.h>

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace anira {

// Defined here, not in the header: it owns LibTorch objects, and the engine headers
// stay out of anira's public headers (see the note on BackendBase).
struct LibtorchProcessor::Instance {
    Instance(InferenceConfig& inference_config);

    void prepare();
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 const std::shared_ptr<SessionElement>& session);

    torch::jit::script::Module m_module;  ///< Loaded TorchScript model for inference

    std::vector<MemoryBlock<float>> m_input_data;  ///< Pre-allocated input data buffers

    std::vector<c10::IValue> m_inputs;      ///< PyTorch input tensor values
    c10::IValue m_outputs;                  ///< PyTorch output tensor values
    torch::TensorOptions m_tensor_options;  ///< Tensor options for device, dtype and grad
                                            ///< settings

    InferenceConfig& m_inference_config;    ///< Reference to inference configuration
    std::atomic<bool> m_processing{false};  ///< Flag indicating if instance is currently
                                            ///< processing
};

LibtorchProcessor::LibtorchProcessor(InferenceConfig& inference_config)
    : BackendBase(inference_config) {
    torch::set_num_threads(1);

    // Forward anira's log level to c10's glog-style logging (INFO=0, WARNING=1,
    // ERROR=2, FATAL=3). c10 has no severity below INFO, so both Debug and Info
    // map to INFO.
    FLAGS_caffe2_log_level = std::max(static_cast<int>(get_log_level()) - 1, 0);

    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

LibtorchProcessor::~LibtorchProcessor() = default;

void LibtorchProcessor::prepare() {
    for (auto& instance : m_instances) { instance->prepare(); }
}

void LibtorchProcessor::process(std::vector<BufferF>& input,
                                std::vector<BufferF>& output,
                                std::shared_ptr<SessionElement> session) {
    while (true) {
        for (auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                instance->process(input, output, session);
                instance->m_processing.exchange(false);
                return;
            }
        }
    }
}

LibtorchProcessor::Instance::Instance(InferenceConfig& inference_config)
    : m_inference_config(inference_config) {
    m_tensor_options = torch::TensorOptions().requires_grad(false);

    if (m_inference_config.is_model_binary(anira::InferenceBackend::LIBTORCH)) {
        try {
            const anira::ModelData* model_data =
                m_inference_config.get_model_data(anira::InferenceBackend::LIBTORCH);
            std::istringstream stream(
                std::string(static_cast<const char*>(model_data->m_data), model_data->m_size));
            m_module = torch::jit::load(stream);
        } catch (const c10::Error& e) {
            LOG_ERROR << "[ERROR] error loading the model\n";
            LOG_ERROR << e.what() << '\n';
        }
    } else {
        try {
            m_module = torch::jit::load(
                m_inference_config.get_model_path(anira::InferenceBackend::LIBTORCH));
        } catch (const c10::Error& e) {
            LOG_ERROR << "[ERROR] error loading the model\n";
            LOG_ERROR << e.what() << '\n';
        }
    }
    m_module.eval();

    m_inputs.resize(m_inference_config.get_tensor_input_shape().size());
    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());

    // Create tensors with requires_grad disabled from the start through tensor options
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        m_inputs[i] = torch::from_blob(
            m_input_data[i].data(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[i],
            m_tensor_options);
    }

    // No gradient calculation for inference
    torch::NoGradGuard const no_grad;
    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        if (!m_inference_config.get_model_function(InferenceBackend::LIBTORCH).empty()) {
            auto method = m_module.get_method(
                m_inference_config.get_model_function(InferenceBackend::LIBTORCH));
            m_outputs = method(m_inputs);
        } else {
            // Run inference
            m_outputs = m_module.forward(m_inputs);
        }
    }
}

void LibtorchProcessor::Instance::prepare() {
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].clear();
    }
}

void LibtorchProcessor::Instance::process(std::vector<BufferF>& input,
                                          std::vector<BufferF>& output,
                                          const std::shared_ptr<SessionElement>&) {
    // No gradient calculation for inference
    torch::NoGradGuard const no_grad;
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].swap_data(input[i].get_memory_block());
        input[i].reset_channel_ptr();
        // This is necessary because the tensor data pointers seem to change from inference to
        // inference
        m_inputs[i] = torch::from_blob(
            m_input_data[i].data(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[i],
            m_tensor_options);
    }

    // Run inference
    if (!m_inference_config.get_model_function(InferenceBackend::LIBTORCH).empty()) {
        auto method =
            m_module.get_method(m_inference_config.get_model_function(InferenceBackend::LIBTORCH));
        m_outputs = method(m_inputs);
    } else {
        m_outputs = m_module.forward(m_inputs);
    }

    // We need to copy the data because we cannot access the data pointer ref of the tensor directly
    if (m_outputs.isTuple()) {
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                output[i].get_memory_block()[j] =
                    m_outputs.toTuple()->elements()[i].toTensor().view({-1}).data_ptr<float>()[j];
            }
        }
    } else if (m_outputs.isTensorList()) {
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                output[i].get_memory_block()[j] =
                    m_outputs.toTensorList().get(i).view({-1}).data_ptr<float>()[j];
            }
        }
    } else if (m_outputs.isTensor()) {
        for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[0]; j++) {
            output[0].get_memory_block()[j] = m_outputs.toTensor().view({-1}).data_ptr<float>()[j];
        }
    }
}

}  // namespace anira