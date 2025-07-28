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
    HostConfig host_config;
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
    case anira::InferenceBackend::LIBTORCH:
        backend = "libtorch";
        break;
    #endif
    #ifdef USE_ONNXRUNTIME
    case anira::InferenceBackend::ONNX:
        backend = "ONNX";
        break;
    #endif
    #ifdef USE_TFLITE
    case anira::InferenceBackend::TFLITE:
        backend = "TFlite";
        break;
    #endif
    case anira::InferenceBackend::CUSTOM:
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

    // This test requires the buffer size to be a multiple of the preprocess input size
    if (static_cast<size_t>(buffer_size) % inference_config.get_preprocess_input_size()[0] != 0){
        GTEST_SKIP() << "Test requires the preprocess_input_size to be a multiple of the buffer size.";
        return;
    }

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
        
        size_t prev_samples = inference_handler.get_available_samples(0);

        inference_handler.process(test_buffer.get_array_of_write_pointers(), buffer_size);
        
        // wait until the block was properly processed
        auto start = std::chrono::system_clock::now();
        while (inference_handler.get_available_samples(0) != prev_samples){
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

TEST_P(InferenceTest, WithCustomLatency){

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

    // This test requires the buffer size to be a multiple of the preprocess input size
    if (static_cast<size_t>(buffer_size) % inference_config.get_preprocess_input_size()[0] != 0){
        GTEST_SKIP() << "Test requires the preprocess_input_size to be a multiple of the buffer size.";
        return;
    }


    // Create an InferenceHandler instance
    InferenceHandler inference_handler(pp_processor, inference_config,bypass_processor, anira_context_config);


    // Allocate memory for audio processing
    inference_handler.prepare(test_params.host_config, 0);
    // Select the inference backend
    inference_handler.set_inference_backend(test_params.backend);

    int latency = inference_handler.get_latency(); // The 0th tensor is the audio data tensor, so we only need the first element of the latency vector

    ASSERT_EQ(latency, 0) << "Custom latency should be set to 0 for this test";

    BufferF test_buffer(1, buffer_size);
    RingBuffer ring_buffer;
    ring_buffer.initialize_with_positions(1, buffer_size + reference_offset);
    
    //fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < reference_offset; i++){
        ring_buffer.push_sample(0, 0);
    }    

    // TODO better default value for repeat
    for (size_t repeat = 0; repeat < 150; repeat++){

        for (size_t i = 0; i < buffer_size; i++){
            test_buffer.set_sample(0, i, data_input.at((repeat*buffer_size)+i));
            ring_buffer.push_sample(0, data_reference.at((repeat*buffer_size)+i));
        }
        
        size_t prev_samples = inference_handler.get_available_samples(0);

        inference_handler.push_data(test_buffer.get_array_of_read_pointers(), buffer_size);
        
        // wait until the block was properly processed
        auto start = std::chrono::system_clock::now();
        while (inference_handler.get_available_samples(0) != prev_samples + buffer_size){
            if (std::chrono::system_clock::now() >  start + std::chrono::duration<long int>(INFERENCE_TIMEOUT_S)){
                FAIL() << "Timeout while waiting for block to be processed";
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }

        inference_handler.pop_data(test_buffer.get_array_of_write_pointers(), buffer_size);

        for (size_t i = 0; i < buffer_size; i++){
            float reference = ring_buffer.pop_sample(0);
            float processed = test_buffer.get_sample(0, i);
                        
            if (repeat*buffer_size + i < reference_offset){
                ASSERT_FLOAT_EQ(reference, 0);
            } else {
                // calculate epsilon on the fly
                float epsilon = max(abs(reference), abs(processed)) * test_params.epsilon_rel + test_params.epsilon_abs; 
                ASSERT_NEAR(reference, processed, epsilon) << "repeat=" << repeat << ", i=" << i << ", total sample nr: " << repeat*buffer_size + i  << std::endl;
            }
        }
    }
}

TEST_P(InferenceTest, Reset){

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

    // This test requires the buffer size to be a multiple of the preprocess input size
    if (static_cast<size_t>(buffer_size) % inference_config.get_preprocess_input_size()[0] != 0){
        GTEST_SKIP() << "Test requires the preprocess_input_size to be a multiple of the buffer size.";
        return;
    }

    // Create an InferenceHandler instance
    InferenceHandler inference_handler(pp_processor, inference_config, bypass_processor, anira_context_config);

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

    // First, process some data to "contaminate" the internal state
    for (size_t repeat = 0; repeat < 50; repeat++){
        for (size_t i = 0; i < buffer_size; i++){
            test_buffer.set_sample(0, i, data_input.at((repeat*buffer_size)+i));
            ring_buffer.push_sample(0, data_reference.at((repeat*buffer_size)+i));
        }
        
        size_t prev_samples = inference_handler.get_available_samples(0);
        inference_handler.process(test_buffer.get_array_of_write_pointers(), buffer_size);
        
        // wait until the block was properly processed
        auto start = std::chrono::system_clock::now();
        while (inference_handler.get_available_samples(0) != prev_samples){
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

    // Now reset the inference handler
    inference_handler.reset();

    // Verify that the available samples count is reset
    EXPECT_EQ(inference_handler.get_available_samples(0), latency_offset) << "Available samples should be " << latency_offset << " after reset";

    // Reset the ring buffer to restart from the beginning of reference data
    ring_buffer.clear_with_positions();
    ring_buffer.initialize_with_positions(1, latency_offset + buffer_size + reference_offset);
    
    // Fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < latency_offset + reference_offset; i++){
        ring_buffer.push_sample(0, 0);
    }

    // Process data again and verify that output matches reference from the beginning
    for (size_t repeat = 0; repeat < 150; repeat++){

        for (size_t i = 0; i < buffer_size; i++){
            test_buffer.set_sample(0, i, data_input.at((repeat*buffer_size)+i));
            ring_buffer.push_sample(0, data_reference.at((repeat*buffer_size)+i));
        }
        
        size_t prev_samples = inference_handler.get_available_samples(0);

        inference_handler.process(test_buffer.get_array_of_write_pointers(), buffer_size);
        
        // wait until the block was properly processed
        auto start = std::chrono::system_clock::now();
        while (inference_handler.get_available_samples(0) != prev_samples){
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
                ASSERT_NEAR(reference, processed, epsilon) << "After reset: repeat=" << repeat << ", i=" << i << ", total sample nr: " << repeat*buffer_size + i  << std::endl;
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
            anira::InferenceBackend::CUSTOM,
            HostConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::InferenceBackend::CUSTOM,
            HostConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::InferenceBackend::CUSTOM,
            HostConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::InferenceBackend::CUSTOM,
            HostConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            0,
            FLT_EPSILON,
            0
        },
        InferenceTestParams{
            anira::InferenceBackend::CUSTOM,
            HostConfig(300, 44100),
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
            anira::InferenceBackend::LIBTORCH,
            HostConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::LIBTORCH,
            HostConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::LIBTORCH,
            HostConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::LIBTORCH,
            HostConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::LIBTORCH,
            HostConfig(300, 44100),
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
            anira::InferenceBackend::ONNX,
            HostConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::ONNX,
            HostConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::ONNX,
            HostConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::ONNX,
            HostConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::ONNX,
            HostConfig(300, 44100),
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
            anira::InferenceBackend::TFLITE,
            HostConfig(1024, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::TFLITE,
            HostConfig(2048, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::TFLITE,
            HostConfig(512, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::TFLITE,
            HostConfig(256, 44100),
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav",
            std::string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav",
            149,
            1e-6f,
            2e-7f
        },
        InferenceTestParams{
            anira::InferenceBackend::TFLITE,
            HostConfig(300, 44100),
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
