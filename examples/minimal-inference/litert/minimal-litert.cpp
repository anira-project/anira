/* ==========================================================================

Minimal LiteRT example using the native LiteRt CompiledModel C API.
Licence: Apache 2.0

========================================================================== */

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

#include "../../../extras/models/model_files.h"
#include "../minimal_model.h"
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

void minimal_inference(const MinimalModel& info) {
    std::cout << "Minimal LiteRT example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << info.m_path << std::endl;

    // Create environment, load model and compile it for the CPU
    LiteRtEnvironment env = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateEnvironment(0, nullptr, &env));

    LiteRtModel model = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateModelFromFile(env, info.m_path.c_str(), &model));

    LiteRtOptions options = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateOptions(&options));
    LITERT_MINIMAL_CHECK(LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu));

    LiteRtCompiledModel compiled_model = nullptr;
    LITERT_MINIMAL_CHECK(LiteRtCreateCompiledModel(env, model, options, &compiled_model));

    // Fill a Buffer with some data
    std::vector<float> input(info.m_input_sizes[0]);
    for (size_t i = 0; i < info.m_input_sizes[0]; ++i) {
        input[i] = static_cast<float>(i) * 0.000001f;
    }

    // Create the input tensor buffers and copy the input data into the first one
    std::vector<LiteRtTensorBuffer> input_buffers(info.m_input_shapes.size());
    for (size_t i = 0; i < input_buffers.size(); ++i) {
        const std::vector<int64_t>& shape = info.m_input_shapes[i];
        LiteRtRankedTensorType type = make_float32_type(shape);
        const size_t bytes = info.m_input_sizes[i] * sizeof(float);
        LITERT_MINIMAL_CHECK(LiteRtCreateManagedTensorBuffer(env,
                                                             kLiteRtTensorBufferTypeHostMemory,
                                                             &type,
                                                             bytes,
                                                             &input_buffers[i]));

        void* host = nullptr;
        LITERT_MINIMAL_CHECK(
            LiteRtLockTensorBuffer(input_buffers[i], &host, kLiteRtTensorBufferLockModeWrite));
        if (i == 0) {
            std::memcpy(host, input.data(), bytes);
        } else {
            std::memset(host, 0, bytes);
        }
        LITERT_MINIMAL_CHECK(LiteRtUnlockTensorBuffer(input_buffers[i]));

        print_shape("Input", i, shape);
    }

    // Create the output tensor buffers
    std::vector<LiteRtTensorBuffer> output_buffers(info.m_output_shapes.size());
    for (size_t i = 0; i < output_buffers.size(); ++i) {
        const std::vector<int64_t>& shape = info.m_output_shapes[i];
        LiteRtRankedTensorType type = make_float32_type(shape);
        const size_t bytes = info.m_output_sizes[i] * sizeof(float);
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
        for (size_t j = 0; j < info.m_output_sizes[i]; ++j) {
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
    // The bundled model files (extras/models/model_files.h), read back with the 3.x API
    // (minimal_model.h): the path of the entry for this example's engine and the shapes the
    // engine is fed. A file without an entry for the engine is skipped.
    const std::array<const char*, 5> models_to_inference = {
        k_hybridnn_model_json,
        k_cnn_model_json,
        k_rnn_model_json,
        k_gain_model_json,
        k_stereo_gain_model_json,
    };

    for (const char* model_json : models_to_inference) {
        const std::optional<MinimalModel> model = minimal_model(model_json, ANIRA_ENGINE_LITERT);
        if (!model.has_value()) {
            std::cout << model_json << ": no entry for this engine" << std::endl;
            continue;
        }
        minimal_inference(*model);
    }

    return 0;
}
