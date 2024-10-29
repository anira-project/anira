#ifndef ANIRA_INFERENCECONFIG_H
#define ANIRA_INFERENCECONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include "anira/system/AniraWinExports.h"

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
            std::vector<int64_t> model_input_shape_tflite = {},
            std::vector<int64_t> model_output_shape_tflite = {},
#endif
            float max_inference_time = 0, // in ms per input of batch_size
            int model_latency = 0, // in samples per input of batch_size
            bool warm_up = false,
            bool bind_session_to_processor = false,
            int num_parallel_processors = ((int) std::thread::hardware_concurrency() / 2 > 0) ? (int) std::thread::hardware_concurrency() / 2 : 1
#ifdef USE_SEMAPHORE
            , float wait_in_process_block = 0.f
#endif
            )
 :
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
            m_max_inference_time(max_inference_time),
            m_model_latency(model_latency),
            m_warm_up(warm_up),
            m_bind_session_to_processor(bind_session_to_processor),
            m_num_parallel_processors(num_parallel_processors)
#ifdef USE_SEMAPHORE
            , m_wait_in_process_block(wait_in_process_block)
#endif
    {
#ifdef USE_LIBTORCH
        if (m_model_input_shape_torch.size() > 0) {
            m_new_model_input_size = 1;
            for (int i = 0; i < m_model_input_shape_torch.size(); ++i) {
                m_new_model_input_size *= (int) m_model_input_shape_torch[i];
            }
        }
        if (m_model_output_shape_torch.size() > 0) {
            m_new_model_output_size = 1;
            for (int i = 0; i < m_model_output_shape_torch.size(); ++i) {
                m_new_model_output_size *= (int) m_model_output_shape_torch[i];
            }
        }
#elif USE_ONNXRUNTIME
        if (m_model_input_shape_onnx.size() > 0) {
            m_new_model_input_size = 1;
            for (int i = 0; i < m_model_input_shape_onnx.size(); ++i) {
                m_new_model_input_size *= (int) m_model_input_shape_onnx[i];
            }
        }
        if (m_model_output_shape_onnx.size() > 0) {
            m_new_model_output_size = 1;
            for (int i = 0; i < m_model_output_shape_onnx.size(); ++i) {
                m_new_model_output_size *= (int) m_model_output_shape_onnx[i];
            }
        }
#elif USE_TFLITE
        if (m_model_input_shape_tflite.size() > 0) {
            m_new_model_input_size = 1;
            for (int i = 0; i < m_model_input_shape_tflite.size(); ++i) {
                m_new_model_input_size *= m_model_input_shape_tflite[i];
            }
        }
        if (m_model_output_shape_tflite.size() > 0) {
            m_new_model_output_size = 1;
            for (int i = 0; i < m_model_output_shape_tflite.size(); ++i) {
                m_new_model_output_size *= m_model_output_shape_tflite[i];
            }
        }
#endif
        if (m_bind_session_to_processor) {
            m_num_parallel_processors = 1;
        }
        if (m_num_parallel_processors < 1) {
            m_num_parallel_processors = 1;
            std::cout << "[WARNING] Number of parellel processors must be at least 1. Setting to 1." << std::endl;
        }
    }

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
    std::vector<int64_t> m_model_input_shape_tflite; // tflite requires int but for compatibility with other backends we use int64_t
    std::vector<int64_t> m_model_output_shape_tflite;
#endif

    float m_max_inference_time;
    int m_model_latency;
    bool m_warm_up;
    bool m_bind_session_to_processor;
    int m_num_parallel_processors;

#ifdef USE_SEMAPHORE
    float m_wait_in_process_block;
#endif
    
    int m_new_model_input_size;
    int m_new_model_output_size;

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
            m_max_inference_time == other.m_max_inference_time &&
            m_model_latency == other.m_model_latency &&
            m_warm_up == other.m_warm_up &&
            m_bind_session_to_processor == other.m_bind_session_to_processor &&
            m_num_parallel_processors == other.m_num_parallel_processors &&
#ifdef USE_SEMAPHORE
            m_wait_in_process_block == other.m_wait_in_process_block &&
#endif
            m_new_model_input_size == other.m_new_model_input_size &&
            m_new_model_output_size == other.m_new_model_output_size;
    }

    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

};


} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H