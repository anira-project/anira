#ifndef ANIRA_INFERENCECONFIG_H
#define ANIRA_INFERENCECONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>

namespace anira {

struct InferenceConfig {
    InferenceConfig(
#ifdef USE_LIBTORCH
            const std::string model_path_torch,
            const std::vector<int64_t> model_input_shape_torch,
            const std::vector<int64_t> model_output_shape_torch,
#endif
#ifdef USE_ONNXRUNTIME
            const std::string model_path_onnx,
            const std::vector<int64_t> model_input_shape_onnx,
            const std::vector<int64_t> model_output_shape_onnx,
#endif
#ifdef USE_TFLITE
            const std::string model_path_tflite,
            const std::vector<int64_t> model_input_shape_tflite,
            const std::vector<int64_t> model_output_shape_tflite,
#endif
            size_t batch_size,
            size_t model_input_size,
            size_t model_input_size_backend,
            size_t model_output_size_backend,
            size_t max_inference_time,
            int model_latency,
            bool warm_up = false,
            int numberOfThreads = std::thread::hardware_concurrency() - 1) :
#ifdef USE_LIBTORCH
            m_model_path_torch(model_path_torch),
            m_model_input_shape_torch(model_input_shape_torch),
            m_model_output_shape_torch(model_output_shape_torch),
#endif
#ifdef USE_ONNXRUNTIME
            m_model_path_onnx(model_path_onnx),
            m_model_input_shape_onnx(model_input_shape_onnx),
            m_model_output_shape_onnx(model_output_shape_onnx),
#endif
#ifdef USE_TFLITE
            m_model_path_tflite(model_path_tflite),
            m_model_input_shape_tflite(model_input_shape_tflite),
            m_model_output_shape_tflite(model_output_shape_tflite),
#endif
            m_batch_size(batch_size),
            m_model_input_size(model_input_size),
            m_model_input_size_backend(model_input_size_backend),
            m_model_output_size_backend(model_output_size_backend),
            m_max_inference_time(max_inference_time),
            m_model_latency(model_latency),
            m_warm_up(warm_up),
            m_number_of_threads(numberOfThreads)
    {}

    const size_t m_batch_size;
    const size_t m_model_input_size;
    const size_t m_model_input_size_backend;
    const size_t m_model_output_size_backend;
    const size_t m_max_inference_time;
    const int m_model_latency;
    const bool m_warm_up;

    const int m_number_of_threads;

#ifdef USE_LIBTORCH
    const std::string m_model_path_torch;
    const std::vector<int64_t> m_model_input_shape_torch;
    const std::vector<int64_t> m_model_output_shape_torch;
#endif

#ifdef USE_ONNXRUNTIME
    const std::string m_model_path_onnx;
    const std::vector<int64_t> m_model_input_shape_onnx;
    const std::vector<int64_t> m_model_output_shape_onnx;
#endif

#ifdef USE_TFLITE
    const std::string m_model_path_tflite;
    const std::vector<int64_t> m_model_input_shape_tflite;
    const std::vector<int64_t> m_model_output_shape_tflite;
#endif
};

} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H