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
    InferenceConfig() = default;

    InferenceConfig(
#ifdef USE_LIBTORCH
            std::string model_path_torch,
            std::vector<std::vector<int64_t>> model_input_shape_torch,
            std::vector<std::vector<int64_t>> model_output_shape_torch,
#endif
#ifdef USE_ONNXRUNTIME
            std::string model_path_onnx,
            std::vector<std::vector<int64_t>> model_input_shape_onnx,
            std::vector<std::vector<int64_t>> model_output_shape_onnx,
#endif
#ifdef USE_TFLITE
            std::string model_path_tflite,
            std::vector<std::vector<int64_t>> model_input_shape_tflite,
            std::vector<std::vector<int64_t>> model_output_shape_tflite,
#endif
            float max_inference_time, // in ms per input of batch_size
            unsigned int model_latency = 0, // in samples per input of batch_size
            unsigned int warm_up = 0, // TODO: change accordingly to int warm up in constructor not prepare to play
            bool session_exclusive_processor = false,
            std::array<size_t, 2> index_audio_data = {0, 0}, // input and output index of audio data vector of tensors
            unsigned int num_parallel_processors = ((int) std::thread::hardware_concurrency() / 2 > 0) ? (int) std::thread::hardware_concurrency() / 2 : 1
#ifdef USE_SEMAPHORE
            , float wait_in_process_block = 0.f
#endif
            ) :
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
            m_session_exclusive_processor(session_exclusive_processor),
            m_num_parallel_processors(num_parallel_processors),
            m_index_audio_data(index_audio_data)
#ifdef USE_SEMAPHORE
            , m_wait_in_process_block(wait_in_process_block)
#endif
    {
#ifdef USE_LIBTORCH
        m_input_sizes.resize(m_model_input_shape_torch.size());
        for (int i = 0; i < m_model_input_shape_torch.size(); ++i) {
            m_input_sizes[i] = 1;
            for (int j = 0; j < m_model_input_shape_torch[i].size(); ++j) {
                m_input_sizes[i] *= (int) m_model_input_shape_torch[i][j];
            }
        }
        m_output_sizes.resize(m_model_output_shape_torch.size());
        for (int i = 0; i < m_model_output_shape_torch.size(); ++i) {
            m_output_sizes[i] = 1;
            for (int j = 0; j < m_model_output_shape_torch[i].size(); ++j) {
                m_output_sizes[i] *= (int) m_model_output_shape_torch[i][j];
            }
        }
#elif USE_ONNXRUNTIME
        m_input_sizes.resize(m_model_input_shape_onnx.size());
        for (int i = 0; i < m_model_input_shape_onnx.size(); ++i) {
            m_input_sizes[i] = 1;
            for (int j = 0; j < m_model_input_shape_onnx[i].size(); ++j) {
                m_input_sizes[i] *= (int) m_model_input_shape_onnx[i][j];
            }
        }
        m_output_sizes.resize(m_model_output_shape_onnx.size());
        for (int i = 0; i < m_model_output_shape_onnx.size(); ++i) {
            m_output_sizes[i] = 1;
            for (int j = 0; j < m_model_output_shape_onnx[i].size(); ++j) {
                m_output_sizes[i] *= (int) m_model_output_shape_onnx[i][j];
            }
        }
#elif USE_TFLITE
        m_input_sizes.resize(m_model_input_shape_tflite.size());
        for (int i = 0; i < m_model_input_shape_tflite.size(); ++i) {
            m_input_sizes[i] = 1;
            for (int j = 0; j < m_model_input_shape_tflite[i].size(); ++j) {
                m_input_sizes[i] *= (int) m_model_input_shape_tflite[i][j];
            }
        }
        m_output_sizes.resize(m_model_output_shape_tflite.size());
        for (int i = 0; i < m_model_output_shape_tflite.size(); ++i) {
            m_output_sizes[i] = 1;
            for (int j = 0; j < m_model_output_shape_tflite[i].size(); ++j) {
                m_output_sizes[i] *= (int) m_model_output_shape_tflite[i][j];
            }
        }
#endif
        if (m_session_exclusive_processor) {
            m_num_parallel_processors = 1;
        }
        if (m_num_parallel_processors < 1) {
            m_num_parallel_processors = 1;
            std::cout << "[WARNING] Number of parellel processors must be at least 1. Setting to 1." << std::endl;
        }
    }

    ~InferenceConfig() = default;

    void set_input_sizes(const std::vector<size_t>& input_sizes) {
        m_input_sizes = input_sizes;
    }

    void set_output_sizes(const std::vector<size_t>& output_sizes) {
        m_output_sizes = output_sizes;
    }

#ifdef USE_LIBTORCH
    std::string m_model_path_torch;
    std::vector<std::vector<int64_t>> m_model_input_shape_torch;
    std::vector<std::vector<int64_t>> m_model_output_shape_torch;
#endif

#ifdef USE_ONNXRUNTIME
    std::string m_model_path_onnx;
    std::vector<std::vector<int64_t>> m_model_input_shape_onnx;
    std::vector<std::vector<int64_t>> m_model_output_shape_onnx;
#endif

#ifdef USE_TFLITE
    std::string m_model_path_tflite;
    std::vector<std::vector<int64_t>> m_model_input_shape_tflite; // tflite requires int but for compatibility with other backends we use int64_t
    std::vector<std::vector<int64_t>> m_model_output_shape_tflite;
#endif

    float m_max_inference_time;
    unsigned int m_model_latency;
    bool m_warm_up;
    bool m_session_exclusive_processor;
    std::array<size_t, 2> m_index_audio_data;
    size_t m_num_parallel_processors;

#ifdef USE_SEMAPHORE
    float m_wait_in_process_block;
#endif
    
    std::vector<size_t> m_input_sizes;
    std::vector<size_t> m_output_sizes;

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
            std::abs(m_max_inference_time - other.m_max_inference_time) < 1e-6 &&
            m_model_latency == other.m_model_latency &&
            m_warm_up == other.m_warm_up &&
            m_session_exclusive_processor == other.m_session_exclusive_processor &&
            m_num_parallel_processors == other.m_num_parallel_processors &&
#ifdef USE_SEMAPHORE
            std::abs(m_wait_in_process_block - other.m_wait_in_process_block) < 1e-6 &&
#endif
            m_input_sizes == other.m_input_sizes &&
            m_output_sizes == other.m_output_sizes;
    }

    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

};


} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H