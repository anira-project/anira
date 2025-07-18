#include <thread>
#include <stdint.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include "gtest/gtest.h"
#include <anira/anira.h>
#include <anira/utils/helperFunctions.h>

#include "../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../extras/models/hybrid-nn/HybridNNBypassProcessor.h" // Only needed for round trip test
#include "WavReader.h"

#define INFERENCE_TIMEOUT_S 2
using namespace anira;

struct InferenceTestParams{
    InferenceBackend backend;
    HostAudioConfig host_config;
    std::string input_data_path;
    std::string reference_data_path;
    size_t reference_data_offset;
    float epsilon_rel = 1e-6f;
    float epsilon_abs = 1e-7f;
};

std::ostream& operator<<(std::ostream& stream, const InferenceTestParams& params)
{
    std::string backend;
    switch (params.backend){
    #ifdef USE_LIBTORCH
    case anira::LIBTORCH:
        backend = "libtorch";
        break;
    #endif
    #ifdef USE_ONNXRUNTIME
    case anira::ONNX:
        backend = "ONNX";
        break;
    #endif
    #ifdef USE_TFLITE
    case anira::TFLITE:
        backend = "TFlite";
        break;
    #endif
    case anira::CUSTOM:
        backend = "custom";
        break;
    
    default:
        backend = "unknown";
        break;
    }
    
    stream << "{ ";
    stream << "backend = " << backend;
    stream << ", host_buffer_size = " << params.host_config.m_buffer_size;
    stream << ", host_sample_rate = " << params.host_config.m_sample_rate;
    stream << " }";

    return stream;
}



// Test fixture for paramterized inference tests
class InferenceTest: public ::testing::TestWithParam<InferenceTestParams>{        
};



TEST_P(InferenceTest, Simple){

    auto const& test_params = GetParam();
    auto const& buffer_size = test_params.host_config.m_buffer_size;
    auto const& reference_offset = test_params.reference_data_offset;

    // read reference data
    std::vector<float> data_input;
    std::vector<float> data_reference;

    read_wav(test_params.input_data_path, data_input);
    read_wav(test_params.reference_data_path, data_reference);

    ASSERT_TRUE(data_input.size() > 0);
    ASSERT_TRUE(data_reference.size() > 0);

    // setup inference
    ContextConfig anira_context_config;
    InferenceConfig inference_config = hybridnn_config;
    HybridNNPrePostProcessor pp_processor(inference_config);
    HybridNNBypassProcessor bypass_processor(inference_config);

    // Create an InferenceHandler instance
    InferenceHandler inference_handler(pp_processor, inference_config,bypass_processor, anira_context_config);


    // Allocate memory for audio processing
    inference_handler.prepare(test_params.host_config);
    // Select the inference backend
    inference_handler.set_inference_backend(test_params.backend);

    int latency_offset = inference_handler.get_latency(); // The 0th tensor is the audio data tensor, so we only need the first element of the latency vector

    BufferF test_buffer(1, buffer_size);
    RingBuffer ring_buffer;
    ring_buffer.initialize_with_positions(1, latency_offset + buffer_size + reference_offset);
    
    //fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < latency_offset + reference_offset; i++){
        ring_buffer.push_sample(0, 0);
    }    

    // TODO better default value for repeat
    for (size_t repeat = 0; repeat < 150; repeat++){

        for (size_t i = 0; i < buffer_size; i++){
            test_buffer.set_sample(0, i, data_input.at((repeat*buffer_size)+i));
            ring_buffer.push_sample(0, data_reference.at((repeat*buffer_size)+i));
        }
        
        size_t prev_samples = inference_handler.get_num_received_samples(0);

        inference_handler.process(test_buffer.get_array_of_write_pointers(), buffer_size);
        
        // wait until the block was properly processed
        auto start = std::chrono::system_clock::now();
        while (!(inference_handler.get_num_received_samples(0) >= prev_samples)){
            if (std::chrono::system_clock::now() >  start + std::chrono::duration<long int>(INFERENCE_TIMEOUT_S)){
                FAIL() << "Timeout while waiting for block to be processed";
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }        

        for (size_t i = 0; i < buffer_size; i++){
            float reference = ring_buffer.pop_sample(0);
            float processed = test_buffer.get_sample(0, i);
                        
            if (repeat*buffer_size + i < latency_offset + reference_offset){
                ASSERT_FLOAT_EQ(reference, 0);
            } else {
                // calculate epsilon on the fly
                float epsilon = max(abs(reference), abs(processed)) * test_params.epsilon_rel + test_params.epsilon_abs; 
                ASSERT_NEAR(reference, processed, epsilon) << "repeat=" << repeat << ", i=" << i << ", total sample nr: " << repeat*buffer_size + i  << std::endl;
            }
        }
    }
}

std::string build_test_name(const testing::TestParamInfo<InferenceTest::ParamType>& info){
    std::stringstream ss_sample_rate, ss_buffer_size;

    // Set precision to 4 decimal places for cleaner names
    ss_sample_rate << std::fixed << std::setprecision(4) << info.param.host_config.m_sample_rate;
    ss_buffer_size << std::fixed << std::setprecision(4) << info.param.host_config.m_buffer_size;
    
    std::string sample_rate_str = ss_sample_rate.str();
    std::string buffer_size_str = ss_buffer_size.str();

    // Replace decimal points with underscores to make valid test names
    std::replace(sample_rate_str.begin(), sample_rate_str.end(), '.', '_');
    std::replace(buffer_size_str.begin(), buffer_size_str.end(), '.', '_');

    return sample_rate_str + "x" + buffer_size_str;
}

INSTANTIATE_TEST_SUITE_P(
    InferenceBypass, InferenceTest, ::testing::Values(
        InferenceTestParams{
            anira::CUSTOM,
            HostAudioConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::CUSTOM,
            HostAudioConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::CUSTOM,
            HostAudioConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::CUSTOM,
            HostAudioConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        }
    ),
    build_test_name
);

#ifdef USE_LIBTORCH
INSTANTIATE_TEST_SUITE_P(
    InferenceLibtorch, InferenceTest, ::testing::Values(
        InferenceTestParams{
            anira::LIBTORCH,
            HostAudioConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::LIBTORCH,
            HostAudioConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::LIBTORCH,
            HostAudioConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::LIBTORCH,
            HostAudioConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        }
    ),
    build_test_name
);
#endif

#ifdef USE_ONNXRUNTIME
INSTANTIATE_TEST_SUITE_P(
    InferenceOnnx, InferenceTest, ::testing::Values(
        InferenceTestParams{
            anira::ONNX,
            HostAudioConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::ONNX,
            HostAudioConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::ONNX,
            HostAudioConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::ONNX,
            HostAudioConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        }
    ),
    build_test_name
);
#endif

#ifdef USE_TFLITE
INSTANTIATE_TEST_SUITE_P(
    InferenceTflite, InferenceTest, ::testing::Values(
        InferenceTestParams{
            anira::TFLITE,
            HostAudioConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::TFLITE,
            HostAudioConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::TFLITE,
            HostAudioConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::TFLITE,
            HostAudioConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        }
    ),
    build_test_name
);
#endif
