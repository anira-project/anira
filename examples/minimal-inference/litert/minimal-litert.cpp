/* ==========================================================================

Minimal LiteRT example using the native LiteRt CompiledModel C API.
Licence: Apache 2.0

========================================================================== */

#include <anira/compat/v3_to_v2.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include "../../../extras/models/model_files.h"
#include "../../../include/anira/utils/Buffer.h"
#include "../../../include/anira/utils/MemoryBlock.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#define LITERT_MINIMAL_CHECK(x)                                  \
    if ((x) != kLiteRtStatusOk) {                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

static LiteRtRankedTensorType make_float32_type(const std::vector<int64_t>& shape) {
    LiteRtRankedTensorType type{};
    type.element_type = kLiteRtElementTypeFloat32;
    type.layout.rank = static_cast<unsigned int>(shape.size());
    type.layout.has_strides = false;
    for (size_t d = 0; d < shape.size() && d < LITERT_TENSOR_MAX_RANK; ++d) {
        type.layout.dimensions[d] = static_cast<int32_t>(shape[d]);
    }
    return type;
}

static void print_shape(const char* label, size_t i, const std::vector<int64_t>& shape) {
    std::cout << label << " shape " << i << ": [";
    for (size_t j = 0; j < shape.size(); ++j) {
        std::cout << shape[j];
        if (j < shape.size() - 1) { std::cout << ", "; }
    }
    std::cout << "]" << std::endl;
}

void minimal_inference(anira::InferenceConfig m_inference_config) {
    std::cout << "Minimal LiteRT example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: "
              << m_inference_config.get_model_path(anira::InferenceBackend::LITERT) << std::endl;

    // Create environment, load model and compile it for the CPU
    LiteRtEnvironment env = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateEnvironment(0, nullptr, &env));

    LiteRtModel model = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateModelFromFile(
        env,
        m_inference_config.get_model_path(anira::InferenceBackend::LITERT).c_str(),
        &model));

    LiteRtOptions options = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateOptions(&options));
    LITERT_MINIMAL_CHECK(LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu));

    LiteRtCompiledModel compiled_model = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateCompiledModel(env, model, options, &compiled_model));

    // Fill a Buffer with some data
    anira::BufferF input(1, m_inference_config.get_tensor_input_size()[0]);
    for (size_t i = 0; i < m_inference_config.get_tensor_input_size()[0]; ++i) {
        input.set_sample(0, static_cast<int>(i), static_cast<float>(i) * 0.000001f);
    }

    // Create the input tensor buffers and copy the input data into the first one
    std::vector<LiteRtTensorBuffer> input_buffers(
        m_inference_config.get_tensor_input_shape().size());
    for (size_t i = 0; i < input_buffers.size(); ++i) {
        const std::vector<int64_t>& shape =
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LITERT)[i];
        LiteRtRankedTensorType type = make_float32_type(shape);
        const size_t bytes = m_inference_config.get_tensor_input_size()[i] * sizeof(float);
        LITERT_MINIMAL_CHECK(LiteRtCreateManagedTensorBuffer(env,
                                                             kLiteRtTensorBufferTypeHostMemory,
                                                             &type,
                                                             bytes,
                                                             &input_buffers[i]));

        void* host = nullptr;
        LITERT_MINIMAL_CHECK(
            LiteRtLockTensorBuffer(input_buffers[i], &host, kLiteRtTensorBufferLockModeWrite));
        if (i == 0) {
            std::memcpy(host, input.get_memory_block().data(), bytes);
        } else {
            std::memset(host, 0, bytes);
        }
        LITERT_MINIMAL_CHECK(LiteRtUnlockTensorBuffer(input_buffers[i]));

        print_shape("Input", i, shape);
    }

    // Create the output tensor buffers
    std::vector<LiteRtTensorBuffer> output_buffers(
        m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < output_buffers.size(); ++i) {
        const std::vector<int64_t>& shape =
            m_inference_config.get_tensor_output_shape(anira::InferenceBackend::LITERT)[i];
        LiteRtRankedTensorType type = make_float32_type(shape);
        const size_t bytes = m_inference_config.get_tensor_output_size()[i] * sizeof(float);
        LITERT_MINIMAL_CHECK(LiteRtCreateManagedTensorBuffer(env,
                                                             kLiteRtTensorBufferTypeHostMemory,
                                                             &type,
                                                             bytes,
                                                             &output_buffers[i]));

        print_shape("Output", i, shape);
    }

    // Execute inference
    LITERT_MINIMAL_CHECK(LiteRtRunCompiledModel(compiled_model,
                                                /*signature_index=*/0,
                                                input_buffers.size(),
                                                input_buffers.data(),
                                                output_buffers.size(),
                                                output_buffers.data()));

    // Read back and print the output data
    for (size_t i = 0; i < output_buffers.size(); ++i) {
        void* host = nullptr;
        LITERT_MINIMAL_CHECK(
            LiteRtLockTensorBuffer(output_buffers[i], &host, kLiteRtTensorBufferLockModeRead));
        const float* output_data = static_cast<const float*>(host);
        for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; ++j) {
            std::cout << "Output data [" << i << "][" << j << "]: " << output_data[j] << std::endl;
        }
        LITERT_MINIMAL_CHECK(LiteRtUnlockTensorBuffer(output_buffers[i]));
    }

    // Dispose of all the LiteRT objects
    for (auto& buffer : input_buffers) { LiteRtDestroyTensorBuffer(buffer); }
    for (auto& buffer : output_buffers) { LiteRtDestroyTensorBuffer(buffer); }
    LiteRtDestroyCompiledModel(compiled_model);
    LiteRtDestroyOptions(options);
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(env);
}

int main(int argc, const char* argv[]) {
    // The bundled models: a model file and a contract file each (extras/models/model_files.h),
    // loaded with the 3.x API and bridged to the 2.x InferenceConfig this example reads its
    // model path and tensor shapes from. The candidate list keeps the entries of this build's
    // engines only.
    const std::array<std::pair<const char*, const char*>, 5> models_to_inference = {{
        {k_hybridnn_model_json, k_hybridnn_contract_json},
        {k_cnn_model_json, k_cnn_contract_json},
        {k_rnn_model_json, k_rnn_contract_json},
        {k_gain_model_json, k_gain_contract_json},
        {k_stereo_gain_model_json, k_stereo_gain_contract_json},
    }};

    for (const auto& [model_json, contract_json] : models_to_inference) {
        const anira::ModelConfig model_config = anira::ModelConfig::from_file(model_json);
        const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
        minimal_inference(anira::v3compat::to_inference_config(model_config,
                                                               contract,
                                                               anira::v3compat::enabled_engines()));
    }

    return 0;
}
