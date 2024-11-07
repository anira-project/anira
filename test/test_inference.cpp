#include "gtest/gtest.h"
#include <anira/anira.h>

#include "../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/desktop/models/hybrid-nn/HybridNNBypassProcessor.h" // Only needed for round trip test
#include <chrono>
#include <thread>
#include "WavReader.h"
#ifndef RANDOMDOUBLE
#define RANDOMDOUBLE

static double randomDouble(int LO, int HI){
  return (double) LO + (double) (std::rand()) / ((double) (RAND_MAX/(HI-LO)));
}

#endif // RANDOMDOUBLE



using namespace anira;

static void fill_buffer(AudioBufferF &buffer){
    for (size_t i = 0; i < buffer.get_num_samples(); i++){
        double new_val = randomDouble(-1, 1);
        buffer.set_sample(0, i, new_val);
    }
}

static void push_buffer_to_ringbuffer(AudioBufferF const &buffer, RingBuffer &ringbuffer){
    for (size_t i = 0; i < buffer.get_num_samples(); i++){
        ringbuffer.push_sample(0, buffer.get_sample(0, i));
    }
}

TEST(Test_Inference, passthrough){

    size_t bufferSize = 2048;
    double sampleRate = 48000;

    InferenceConfig inferenceConfig = hybridnn_config;
    anira::AniraContextConfig anira_context_config;

    // Create a pre- and post-processor instance
    HybridNNPrePostProcessor myPrePostProcessor;
    HybridNNBypassProcessor bypass_processor(inferenceConfig);
    // Create an InferenceHandler instance
    anira::InferenceHandler inferenceHandler(myPrePostProcessor, inferenceConfig, bypass_processor, anira_context_config);

    // Create a HostAudioConfig instance containing the host config infos
    anira::HostAudioConfig audioConfig {
        bufferSize,
        sampleRate
    };  



    // Allocate memory for audio processing
    inferenceHandler.prepare(audioConfig);
    // Select the inference backend
    inferenceHandler.set_inference_backend(anira::CUSTOM);

    int latency_offset = inferenceHandler.get_latency();

    RingBuffer ring_buffer;
    ring_buffer.initialize_with_positions(1, latency_offset+bufferSize);

    //fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < latency_offset; i++){
        ring_buffer.push_sample(0, 0);
    }    

    AudioBufferF test_buffer(1, bufferSize);

    std::cout << "starting test" << std::endl;
    for (size_t repeat = 0; repeat < 50; repeat++)
    {
        fill_buffer(test_buffer);
        push_buffer_to_ringbuffer(test_buffer, ring_buffer);
        
        inferenceHandler.process(test_buffer.get_array_of_write_pointers(), bufferSize);

        for (size_t i = 0; i < bufferSize; i++){
            EXPECT_FLOAT_EQ(
                ring_buffer.pop_sample(0),
                test_buffer.get_sample(0, i )
            );
        }
    }
}


TEST(Test_Inference, inference){

    size_t bufferSize = 1024;
    double sampleRate = 44100;

    // read reference data
    std::vector<float> data_input;
    std::vector<float> data_predicted;

    read_wav(string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/x_test.wav", data_input);
    read_wav(string(GUITARLSTM_MODELS_PATH_PYTORCH) + "/model_0/y_pred.wav", data_predicted);

    assert(data_input.size() > 0);
    assert(data_predicted.size() > 0);


    InferenceConfig inferenceConfig = hybridnn_config;
    anira::AniraContextConfig anira_context_config;

    // Create a pre- and post-processor instance
    HybridNNPrePostProcessor myPrePostProcessor;
    HybridNNBypassProcessor bypass_processor(inferenceConfig);
    // Create an InferenceHandler instance
    anira::InferenceHandler inferenceHandler(myPrePostProcessor, inferenceConfig, bypass_processor, anira_context_config);

    // Create a HostAudioConfig instance containing the host config infos
    anira::HostAudioConfig audioConfig {
        bufferSize,
        sampleRate
    };  



    // Allocate memory for audio processing
    inferenceHandler.prepare(audioConfig);
    // Select the inference backend
    inferenceHandler.set_inference_backend(anira::LIBTORCH);

    int latency_offset = inferenceHandler.get_latency();
    std::cout << "latency in samples: " << inferenceHandler.get_latency() << std::endl;

    auto output_file_processed = std::fstream("/home/leto/ak/random_shit/audio_file_compare/processed_output.bin", std::ios::out | std::ios::binary);
    auto input_file = std::fstream("/home/leto/ak/random_shit/audio_file_compare/input.bin", std::ios::out | std::ios::binary);
    auto output_file_reference = std::fstream("/home/leto/ak/random_shit/audio_file_compare/reference_output.bin", std::ios::out | std::ios::binary);

    RingBuffer ring_buffer;
    ring_buffer.initialize_with_positions(1, latency_offset+bufferSize);
    
    //fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < latency_offset; i++){
        ring_buffer.push_sample(0, 0);
    }    

    AudioBufferF test_buffer(1, bufferSize);

    std::cout << "starting test" << std::endl;
    for (size_t repeat = 0; repeat < 150; repeat++)
    {
        for (size_t i = 0; i < bufferSize; i++)
        {
            test_buffer.set_sample(0, i, data_input.at((repeat*bufferSize)+i));
            ring_buffer.push_sample(0, data_predicted.at((repeat*bufferSize)+i));
            float input_value = test_buffer.get_sample(0, i);
            input_file.write(reinterpret_cast<const char*>(&input_value), sizeof(float));

        }
        
        size_t prev_samples = inferenceHandler.get_inference_manager().get_num_received_samples();

        inferenceHandler.process(test_buffer.get_array_of_write_pointers(), bufferSize);
        
        // wait until the block was properly processed
        while (!(inferenceHandler.get_inference_manager().get_num_received_samples() >= prev_samples)){
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }        

        for (size_t i = 0; i < bufferSize; i++){
            float reference = ring_buffer.pop_sample(0);
            float processed = test_buffer.get_sample(0, i);
            
            output_file_reference.write(reinterpret_cast<const char*>(&reference), sizeof(float));
            output_file_processed.write(reinterpret_cast<const char*>(&processed), sizeof(float));

            ASSERT_FLOAT_EQ(
                reference,
                processed
            ) << "repeat=" << repeat << ", i=" << i << ", total sample nr: " << repeat*bufferSize + i << std::endl;
        }
    }
    input_file.close();
    output_file_processed.close();
    output_file_reference.close();

}