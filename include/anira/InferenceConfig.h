#ifndef ANIRA_INFERENCECONFIG_H
#define ANIRA_INFERENCECONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>
#include "anira/system/AnriaConfig.h"

namespace anira {

struct ANIRA_API InferenceConfig {
    InferenceConfig(
#ifdef USE_LIBTORCH
            std::string model_path_torch = "",
            std::vector<int64_t> model_input_shape_torch = {},
            std::vector<int64_t> model_output_shape_torch = {},
#endif
#ifdef USE_ONNXRUNTIME
            std::string model_path_onnx = "",
            std::vector<int64_t> model_input_shape_onnx = {},
            std::vector<int64_t> model_output_shape_onnx = {},
#endif
#ifdef USE_TFLITE
            std::string model_path_tflite = "",
            std::vector<int> model_input_shape_tflite = {},
            std::vector<int> model_output_shape_tflite = {},
#endif
            size_t batch_size = 0,
            size_t model_input_size = 0,
            size_t model_output_size = 0,
            size_t max_inference_time = 0,
            int model_latency = 0,
            bool warm_up = false,
            float wait_in_process_block = 0.5f,
            bool bind_session_to_thread = false,
            int numberOfThreads = ((int) std::thread::hardware_concurrency() - 1 > 0) ? (int) std::thread::hardware_concurrency() - 1 : 1) :
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
            m_model_output_size(model_output_size),
            m_max_inference_time(max_inference_time),
            m_model_latency(model_latency),
            m_warm_up(warm_up),
            m_wait_in_process_block(wait_in_process_block),
            m_bind_session_to_thread(bind_session_to_thread),
            m_number_of_threads(numberOfThreads)
    {}

#ifdef USE_LIBTORCH
    std::string m_model_path_torch;
    std::vector<int64_t> m_model_input_shape_torch;
    std::vector<int64_t> m_model_output_shape_torch;
#endif

#ifdef USE_ONNXRUNTIME
    std::string m_model_path_onnx;
    std::vector<int64_t> m_model_input_shape_onnx;
    std::vector<int64_t> m_model_output_shape_onnx;
#endif

#ifdef USE_TFLITE
    std::string m_model_path_tflite;
    std::vector<int> m_model_input_shape_tflite;
    std::vector<int> m_model_output_shape_tflite;
#endif

    size_t m_batch_size;
    size_t m_model_input_size;
    size_t m_model_output_size;
    size_t m_max_inference_time;
    int m_model_latency;
    bool m_warm_up;

    float m_wait_in_process_block;
    bool m_bind_session_to_thread;
    int m_number_of_threads;

    bool operator==(const InferenceConfig& other) const {
        return
#ifdef USE_LIBTORCH
            m_model_path_torch == other.m_model_path_torch &&
            m_model_input_shape_torch == other.m_model_input_shape_torch &&
            m_model_output_shape_torch == other.m_model_output_shape_torch &&
#endif
#ifdef USE_ONNXRUNTIME
            m_model_path_onnx == other.m_model_path_onnx &&
            m_model_input_shape_onnx == other.m_model_input_shape_onnx &&
            m_model_output_shape_onnx == other.m_model_output_shape_onnx &&
#endif
#ifdef USE_TFLITE
            m_model_path_tflite == other.m_model_path_tflite &&
            m_model_input_shape_tflite == other.m_model_input_shape_tflite &&
            m_model_output_shape_tflite == other.m_model_output_shape_tflite &&
#endif
            m_batch_size == other.m_batch_size &&
            m_model_input_size == other.m_model_input_size &&
            m_model_output_size == other.m_model_output_size &&
            m_max_inference_time == other.m_max_inference_time &&
            m_model_latency == other.m_model_latency &&
            m_warm_up == other.m_warm_up &&
            m_wait_in_process_block == other.m_wait_in_process_block &&
            m_bind_session_to_thread == other.m_bind_session_to_thread &&
            m_number_of_threads == other.m_number_of_threads;
    }

    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

};


} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H