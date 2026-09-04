// The 2.x configuration documents of four bundled models, as text: what a 2.x deployment wrote
// by hand (an inference_config block, optionally a core_config block, absolute model paths).
// The 2.x JsonConfigLoader reads them as they are and the 3.x loaders upgrade them; the tests
// compare both against the 3.x files of extras/models. Built at run time because the 2.x
// format takes absolute paths: @MODELS@ stands for ANIRA_EXTRAS_MODELS_DIR.
#ifndef ANIRA_TEST_SUPPORT_V2_DOCUMENTS_H
#define ANIRA_TEST_SUPPORT_V2_DOCUMENTS_H

#include <cstddef>
#include <string>
#include <string_view>

namespace anira_test {

inline std::string with_models_dir(std::string_view text) {
    constexpr std::string_view placeholder = "@MODELS@";
    std::string out;
    size_t pos = 0;
    while (true) {
        const size_t hit = text.find(placeholder, pos);
        if (hit == std::string_view::npos) {
            out.append(text.substr(pos));
            return out;
        }
        out.append(text.substr(pos, hit - pos));
        out.append(ANIRA_EXTRAS_MODELS_DIR);
        pos = hit + placeholder.size();
    }
}

/// SimpleGainNetwork, mono: five entries, a 512-sample stream plus a static gain scalar.
inline std::string gain_v2_document() {
    return with_models_dir(R"({
  "context_config": { "num_threads": 1 },
  "inference_config": {
    "model_data": [
      { "model_path": "@MODELS@/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.pt",
        "inference_backend": "LIBTORCH" },
      { "model_path": "@MODELS@/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.onnx",
        "inference_backend": "ONNX" },
      { "model_path": "@MODELS@/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.tflite",
        "inference_backend": "TFLITE" },
      { "model_path": "@MODELS@/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.tflite",
        "inference_backend": "LITERT" },
      { "model_path": "@MODELS@/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.pte",
        "inference_backend": "EXECUTORCH" }
    ],
    "tensor_shape": [ { "input_shape": [[1, 1, 512], [1]], "output_shape": [[1, 1, 512], [1]] } ],
    "processing_spec": {
      "preprocess_input_channels": [1, 1],
      "postprocess_output_channels": [1, 1],
      "preprocess_input_size": [512, 0],
      "postprocess_output_size": [512, 0]
    },
    "max_inference_time": 5.0,
    "warm_up": 1
  }
})");
}

/// RAVE funk drum, the whole model: a stateful 2048-sample stream with 2048 samples of latency.
inline std::string rave_funk_drum_v2_document() {
    return with_models_dir(R"({
  "context_config": { "num_threads": 1 },
  "inference_config": {
    "model_data": [
      { "model_path": "@MODELS@/third-party/ircam-acids/RAVE/rave_funk_drum.ts",
        "inference_backend": "LIBTORCH" }
    ],
    "tensor_shape": [ { "input_shape": [1, 1, 2048], "output_shape": [1, 1, 2048] } ],
    "processing_spec": {
      "preprocess_input_channels": [1],
      "postprocess_output_channels": [1],
      "preprocess_input_size": [2048],
      "postprocess_output_size": [2048],
      "internal_model_latency": [2048]
    },
    "max_inference_time": 42.66,
    "warm_up": 5,
    "session_exclusive_processor": true
  }
})");
}

/// The encoder entry point of the same file: 2048 audio samples in, one four-channel latent
/// frame out.
inline std::string rave_funk_drum_encoder_v2_document() {
    return with_models_dir(R"({
  "inference_config": {
    "model_data": [
      { "model_path": "@MODELS@/third-party/ircam-acids/RAVE/rave_funk_drum.ts",
        "inference_backend": "LIBTORCH", "model_function": "encode" }
    ],
    "tensor_shape": [ { "input_shape": [1, 1, 2048], "output_shape": [1, 4, 1] } ],
    "processing_spec": {
      "preprocess_input_channels": [1],
      "postprocess_output_channels": [4]
    },
    "max_inference_time": 42.66,
    "warm_up": 5,
    "session_exclusive_processor": true
  }
})");
}

/// The decoder entry point: one latent frame in, 2048 audio samples out.
inline std::string rave_funk_drum_decoder_v2_document() {
    return with_models_dir(R"({
  "inference_config": {
    "model_data": [
      { "model_path": "@MODELS@/third-party/ircam-acids/RAVE/rave_funk_drum.ts",
        "inference_backend": "LIBTORCH", "model_function": "decode" }
    ],
    "tensor_shape": [ { "input_shape": [1, 4, 1], "output_shape": [1, 1, 2048] } ],
    "processing_spec": {
      "preprocess_input_channels": [4],
      "postprocess_output_channels": [1],
      "preprocess_input_size": [1],
      "postprocess_output_size": [2048],
      "internal_model_latency": [2048]
    },
    "max_inference_time": 42.66,
    "warm_up": 5,
    "session_exclusive_processor": true
  }
})");
}

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_V2_DOCUMENTS_H
