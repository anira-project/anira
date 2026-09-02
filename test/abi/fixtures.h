#ifndef ANIRA_TEST_ABI_FIXTURES_H
#define ANIRA_TEST_ABI_FIXTURES_H

// JSON documents for the loader tests: the three v3 files of section 8 (the
// architecture document's own examples) and version 2 documents shaped like the shipped
// configs. Paths are relative on purpose; the tests pass a base_dir where it matters.

namespace anira_test {

inline constexpr const char* k_model_v3 = R"({
  "models": [
    { "engine": "onnxruntime", "path": "model.onnx",
      "tensors": { "audio_in": "input_0", "mask_out": "output_0" } },
    { "engine": "libtorch", "path": "model.pt",
      "tensors": { "audio_in": { "name": "x" }, "gain": { "layout": [0, "insert"] } },
      "entry": { "name": "forward_streaming" } },
    { "engine": "de.tu-berlin.coreml", "path": "/abs/model.mlpackage" }
  ],
  "default_engine": "onnxruntime",
  "state": "stateless",
  "max_instances": 4,
  "anchor": "mask_out",
  "inputs": [
    { "name": "audio_in", "dtype": "float32", "role": "streamed",
      "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
      "window": { "min": 2048, "max": 8192 }, "context": 1024 },
    { "name": "gain", "role": "static", "axes": [ ["any", 1] ] }
  ],
  "outputs": [
    { "name": "mask_out", "dtype": "float32", "role": "streamed",
      "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
      "window": { "min": 2048, "max": "unbounded" }, "context": 1024, "latency": 512,
      "time_ratio": [1, 2] }
  ]
})";

inline constexpr const char* k_machine_v3 = R"({
  "num_threads": 0,
  "wait_strategy": "spin_backoff",
  "log": { "level": "warning", "drain": "thread", "queue_capacity": 512, "drain_interval_ms": 10 },
  "cuda":   { "device": 1, "pinned_pool_limit": 67108864 },
  "vulkan": { "device": 2, "queue_family": 3 },
  "metal":  { },
  "gl":     { "threads": "caller_thread" }
})";

inline constexpr const char* k_contract_hard_v3 = R"({ "hard": {
    "block_min": 512, "block_max": 512, "rate": 48000,
    "budget": "measured", "warmup": "until_stable", "on_miss": "bypass", "wait_ratio": 0
} })";

inline constexpr const char* k_contract_async_v3 = R"({ "async": {
    "deadline_ms": 33.3, "on_late": "drop", "priority": "auto",
    "lanes": 0, "max_in_flight": 0, "delivery": "polled"
}, "edge_cost": "strict" })";

// The SimpleGain config as the tree generates it (two inputs, the second static).
inline constexpr const char* k_simple_gain_v2 = R"({
  "context_config": { "num_threads": 1 },
  "inference_config": {
    "model_data": [
      { "model_path": "models/simple_gain_network_mono.pt",     "inference_backend": "LIBTORCH" },
      { "model_path": "models/simple_gain_network_mono.onnx",   "inference_backend": "ONNX" },
      { "model_path": "models/simple_gain_network_mono.tflite", "inference_backend": "TFLITE" },
      { "model_path": "models/simple_gain_network_mono.tflite", "inference_backend": "LITERT" },
      { "model_path": "models/simple_gain_network_mono.pte",    "inference_backend": "EXECUTORCH" }
    ],
    "tensor_shape": [ { "input_shape": [[1, 1, 512], [1]], "output_shape": [[1, 1, 512], [1]] } ],
    "processing_spec": {
      "preprocess_input_channels": [1, 1], "postprocess_output_channels": [1, 1],
      "preprocess_input_size": [512, 0], "postprocess_output_size": [512, 0]
    },
    "max_inference_time": 5.0,
    "warm_up": 1
  }
})";

// A RAVE-decoder-shaped document: stateful, a named entry point, an output latency.
inline constexpr const char* k_rave_v2 = R"({
  "context_config": { "num_threads": 2, "wait_strategy": "blocking", "log_level": "error" },
  "inference_config": {
    "model_data": [ { "model_path": "rave.ts", "inference_backend": "LIBTORCH", "model_function": "decode" } ],
    "tensor_shape": [ { "input_shape": [1, 4, 1], "output_shape": [1, 1, 2048] } ],
    "processing_spec": {
      "preprocess_input_channels": [4], "postprocess_output_channels": [1],
      "preprocess_input_size": [1], "postprocess_output_size": [2048],
      "internal_model_latency": [2048]
    },
    "max_inference_time": 42.66,
    "warm_up": 5,
    "session_exclusive_processor": true,
    "num_parallel_processors": 3,
    "blocking_ratio": 0.5
  }
})";

// A HybridNN-shaped document: a batched window over a Time axis of 150.
inline constexpr const char* k_hybrid_v2 = R"({
  "inference_config": {
    "model_data": [ { "model_path": "hybrid.onnx", "inference_backend": "ONNX" } ],
    "tensor_shape": [ { "input_shape": [[256, 1, 150]], "output_shape": [[256, 1]] } ],
    "processing_spec": {
      "preprocess_input_channels": [1], "postprocess_output_channels": [1],
      "preprocess_input_size": [256], "postprocess_output_size": [256]
    },
    "max_inference_time": 5.33
  }
})";

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_FIXTURES_H
