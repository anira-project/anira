#ifdef USE_LITERT

#include <anira/InferenceConfig.h>
#include <anira/backends/BackendBase.h>
#include <anira/backends/LiteRtProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace anira {

namespace {

// Every LiteRT C API call returns a LiteRtStatus. A failure here means a setup or
// runtime problem, so we throw with the failing call + status — this keeps a broken
// state from silently producing zeros (and avoids using the result of a failed call).
inline void litert_check(LiteRtStatus status, const char* what) {
    if (status != kLiteRtStatusOk) {
        throw std::runtime_error(std::string("[anira][LiteRT] ") + what + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Build a ranked float32 tensor type from an anira tensor shape.
LiteRtRankedTensorType make_float32_type(const std::vector<int64_t>& shape) {
    LiteRtRankedTensorType type{};
    type.element_type = kLiteRtElementTypeFloat32;
    type.layout.rank = static_cast<unsigned int>(shape.size());
    type.layout.has_strides = false;
    for (size_t d = 0; d < shape.size() && d < LITERT_TENSOR_MAX_RANK; ++d) {
        type.layout.dimensions[d] = static_cast<int32_t>(shape[d]);
    }
    return type;
}

}  // namespace

LiteRtProcessor::LiteRtProcessor(InferenceConfig& inference_config)
    : BackendBase(inference_config) {
    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

LiteRtProcessor::~LiteRtProcessor() = default;

void LiteRtProcessor::prepare() {
    for (auto& instance : m_instances) { instance->prepare(); }
}

void LiteRtProcessor::process(std::vector<BufferF>& input,
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

LiteRtProcessor::Instance::Instance(InferenceConfig& inference_config)
    : m_inference_config(inference_config) {
    litert_check(LiteRtCreateEnvironment(0, nullptr, &m_env), "LiteRtCreateEnvironment");

    if (inference_config.is_model_binary(anira::InferenceBackend::LITERT)) {
        const anira::ModelData* model_data =
            m_inference_config.get_model_data(anira::InferenceBackend::LITERT);
        assert(model_data && "Model data not found for binary model!");
        litert_check(
            LiteRtCreateModelFromBuffer(m_env, model_data->m_data, model_data->m_size, &m_model),
            "LiteRtCreateModelFromBuffer");
    } else {
        std::string const modelpath =
            m_inference_config.get_model_path(anira::InferenceBackend::LITERT);
        litert_check(LiteRtCreateModelFromFile(m_env, modelpath.c_str(), &m_model),
                     "LiteRtCreateModelFromFile");
    }

    // CPU compilation, pinned to a single thread to match the other backends (anira gets
    // its parallelism from running multiple processor instances). The prebuilt LiteRt
    // runtime does not export the LrtCpuOptions helper symbols, so we build the payload it
    // would emit directly: an "xnnpack"-identified opaque-options blob carrying num_threads.
    // This depends only on the core exported API (LiteRtCreateOpaqueOptions / AddOpaqueOptions).
    litert_check(LiteRtCreateOptions(&m_options), "LiteRtCreateOptions");
    litert_check(LiteRtSetOptionsHardwareAccelerators(m_options, kLiteRtHwAcceleratorCpu),
                 "LiteRtSetOptionsHardwareAccelerators");

    LiteRtOpaqueOptions cpu_opaque = nullptr;
    const char* const cpu_opts_toml = "num_threads = 1\n";  // freed by the deleter below
    const size_t cpu_opts_len = std::strlen(cpu_opts_toml) + 1;
    char* cpu_payload = static_cast<char*>(std::malloc(cpu_opts_len));
    std::memcpy(cpu_payload, cpu_opts_toml, cpu_opts_len);
    litert_check(LiteRtCreateOpaqueOptions(
                     "xnnpack",
                     cpu_payload,
                     [](void* p) { std::free(p); },
                     &cpu_opaque),
                 "LiteRtCreateOpaqueOptions");
    litert_check(LiteRtAddOpaqueOptions(m_options, cpu_opaque), "LiteRtAddOpaqueOptions");

    litert_check(LiteRtCreateCompiledModel(m_env, m_model, m_options, &m_compiled_model),
                 "LiteRtCreateCompiledModel");

    // Create managed host-memory tensor buffers from the configured shapes.
    const size_t num_inputs = m_inference_config.get_tensor_input_shape().size();
    m_input_buffers.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        const std::vector<int64_t>& shape =
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LITERT)[i];
        const LiteRtRankedTensorType type = make_float32_type(shape);
        const size_t bytes = m_inference_config.get_tensor_input_size()[i] * sizeof(float);
        litert_check(LiteRtCreateManagedTensorBuffer(m_env,
                                                     kLiteRtTensorBufferTypeHostMemory,
                                                     &type,
                                                     bytes,
                                                     &m_input_buffers[i]),
                     "LiteRtCreateManagedTensorBuffer (input)");
    }

    const size_t num_outputs = m_inference_config.get_tensor_output_shape().size();
    m_output_buffers.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        const std::vector<int64_t>& shape =
            m_inference_config.get_tensor_output_shape(anira::InferenceBackend::LITERT)[i];
        const LiteRtRankedTensorType type = make_float32_type(shape);
        const size_t bytes = m_inference_config.get_tensor_output_size()[i] * sizeof(float);
        litert_check(LiteRtCreateManagedTensorBuffer(m_env,
                                                     kLiteRtTensorBufferTypeHostMemory,
                                                     &type,
                                                     bytes,
                                                     &m_output_buffers[i]),
                     "LiteRtCreateManagedTensorBuffer (output)");
    }

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        litert_check(LiteRtRunCompiledModel(m_compiled_model,
                                            /*signature_index=*/0,
                                            m_input_buffers.size(),
                                            m_input_buffers.data(),
                                            m_output_buffers.size(),
                                            m_output_buffers.data()),
                     "LiteRtRunCompiledModel (warm-up)");
    }
}

LiteRtProcessor::Instance::~Instance() {
    for (auto& buffer : m_input_buffers) {
        if (buffer) { LiteRtDestroyTensorBuffer(buffer); }
    }
    for (auto& buffer : m_output_buffers) {
        if (buffer) { LiteRtDestroyTensorBuffer(buffer); }
    }
    if (m_compiled_model) { LiteRtDestroyCompiledModel(m_compiled_model); }
    if (m_options) { LiteRtDestroyOptions(m_options); }
    if (m_model) { LiteRtDestroyModel(m_model); }
    if (m_env) { LiteRtDestroyEnvironment(m_env); }
}

void LiteRtProcessor::Instance::prepare() {
    // Reset input buffers to a known (zero) state between sessions.
    for (size_t i = 0; i < m_input_buffers.size(); ++i) {
        void* host = nullptr;
        litert_check(
            LiteRtLockTensorBuffer(m_input_buffers[i], &host, kLiteRtTensorBufferLockModeWrite),
            "LiteRtLockTensorBuffer (prepare)");
        std::memset(host, 0, m_inference_config.get_tensor_input_size()[i] * sizeof(float));
        litert_check(LiteRtUnlockTensorBuffer(m_input_buffers[i]),
                     "LiteRtUnlockTensorBuffer (prepare)");
    }
}

void LiteRtProcessor::Instance::process(std::vector<BufferF>& input,
                                        std::vector<BufferF>& output,
                                        const std::shared_ptr<SessionElement>&) {
    for (size_t i = 0; i < m_input_buffers.size(); ++i) {
        void* host = nullptr;
        litert_check(
            LiteRtLockTensorBuffer(m_input_buffers[i], &host, kLiteRtTensorBufferLockModeWrite),
            "LiteRtLockTensorBuffer (input)");
        std::memcpy(host,
                    input[i].get_memory_block().data(),
                    m_inference_config.get_tensor_input_size()[i] * sizeof(float));
        litert_check(LiteRtUnlockTensorBuffer(m_input_buffers[i]),
                     "LiteRtUnlockTensorBuffer (input)");
    }

    litert_check(LiteRtRunCompiledModel(m_compiled_model,
                                        /*signature_index=*/0,
                                        m_input_buffers.size(),
                                        m_input_buffers.data(),
                                        m_output_buffers.size(),
                                        m_output_buffers.data()),
                 "LiteRtRunCompiledModel");

    for (size_t i = 0; i < m_output_buffers.size(); ++i) {
        void* host = nullptr;
        litert_check(
            LiteRtLockTensorBuffer(m_output_buffers[i], &host, kLiteRtTensorBufferLockModeRead),
            "LiteRtLockTensorBuffer (output)");
        std::memcpy(output[i].get_memory_block().data(),
                    host,
                    m_inference_config.get_tensor_output_size()[i] * sizeof(float));
        litert_check(LiteRtUnlockTensorBuffer(m_output_buffers[i]),
                     "LiteRtUnlockTensorBuffer (output)");
    }
}

}  // namespace anira

#endif  // USE_LITERT
