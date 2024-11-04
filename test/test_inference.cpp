#include "gtest/gtest.h"
#include <anira/anira.h>

#include "../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../extras/desktop/models/hybrid-nn/advanced-configs/HybridNNNoneProcessor.h" // Only needed for round trip test

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

static void push_buffer_to_ringbuffer(AudioBufferF &buffer, RingBuffer &ringbuffer){
    for (size_t i = 0; i < buffer.get_num_samples(); i++){
        ringbuffer.push_sample(0, buffer.get_sample(0, i));
    }
}

TEST(Test_Inference, passthrough){

    size_t bufferSize = 2048;
    double sampleRate = 48000;

    InferenceConfig inferenceConfig = hybridnn_config;

    // Create a pre- and post-processor instance
    HybridNNPrePostProcessor myPrePostProcessor;
    HybridNNNoneProcessor noneProcessor(inferenceConfig);
    // Create an InferenceHandler instance
    anira::InferenceHandler inferenceHandler(myPrePostProcessor, inferenceConfig, noneProcessor);

    // Create a HostAudioConfig instance containing the host config infos
    anira::HostAudioConfig audioConfig {
        1, // currently only mono is supported
        bufferSize,
        sampleRate
    };  



    // Allocate memory for audio processing
    inferenceHandler.prepare(audioConfig);
    // Select the inference backend
    inferenceHandler.set_inference_backend(anira::NONE);

    
    size_t latency_offset = inferenceHandler.get_latency();

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

    // Create a pre- and post-processor instance
    HybridNNPrePostProcessor myPrePostProcessor;
    HybridNNNoneProcessor noneProcessor(inferenceConfig);
    // Create an InferenceHandler instance
    anira::InferenceHandler inferenceHandler(myPrePostProcessor, inferenceConfig, noneProcessor);

    // Create a HostAudioConfig instance containing the host config infos
    anira::HostAudioConfig audioConfig {
        1, // currently only mono is supported
        bufferSize,
        sampleRate
    };  



    // Allocate memory for audio processing
    inferenceHandler.prepare(audioConfig);
    // Select the inference backend
    inferenceHandler.set_inference_backend(anira::LIBTORCH);

    size_t latency_offset = inferenceHandler.get_latency();

    auto output_file_processed = std::fstream("/home/leto/ak/random_shit/audio_file_compare/processed_output.bin", std::ios::out | std::ios::binary);
    auto output_file_reference = std::fstream("/home/leto/ak/random_shit/audio_file_compare/reference_output.bin", std::ios::out | std::ios::binary);

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
        for (size_t i = 0; i < bufferSize; i++)
        {
            test_buffer.set_sample(0, 1, data_input.at((repeat*bufferSize)+i));
            ring_buffer.push_sample(0, data_predicted.at((repeat*bufferSize)+i));
        }
                
        inferenceHandler.process(test_buffer.get_array_of_write_pointers(), bufferSize);

        for (size_t i = 0; i < bufferSize; i++){
            float reference = ring_buffer.pop_sample(0);
            float processed = test_buffer.get_sample(0, i);
            
            output_file_reference.write(reinterpret_cast<const char*>(&reference), sizeof(float));
            output_file_processed.write(reinterpret_cast<const char*>(&processed), sizeof(float));

            EXPECT_FLOAT_EQ(
                reference,
                processed
            );
        }
    }
    output_file_processed.close();
    output_file_reference.close();

}