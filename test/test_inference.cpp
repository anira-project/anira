#include "gtest/gtest.h"
#include <anira/anira.h>


#include "../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../extras/desktop/models/hybrid-nn/advanced-configs/HybridNNNoneProcessor.h" // Only needed for round trip test

#include "wav_reader.h"
#ifndef RANDOMDOUBLE
#define RANDOMDOUBLE

static double randomDouble(int LO, int HI){
  return (double) LO + (double) (std::rand()) / ((double) (RAND_MAX/(HI-LO)));
}

#endif // RANDOMDOUBLE



using namespace anira;

static void fill_buffer(AudioBufferF &buffer){
    for (size_t i = 0; i < buffer.getNumSamples(); i++){
        double new_val = randomDouble(-1, 1);
        buffer.setSample(0, i, new_val);
    }
}

static void push_buffer_to_ringbuffer(AudioBufferF &buffer, RingBuffer &ringbuffer){
    for (size_t i = 0; i < buffer.getNumSamples(); i++){
        ringbuffer.pushSample(0, buffer.getSample(0, i));
    }
}

TEST(Test_Inference, passthrough){

    size_t bufferSize = 2048;
    double sampleRate = 48000;

    InferenceConfig inferenceConfig = hybridNNConfig;

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
    inferenceHandler.setInferenceBackend(anira::NONE);

    
    size_t latency_offset = inferenceHandler.getLatency();

    RingBuffer ring_buffer;
    ring_buffer.initializeWithPositions(1, latency_offset+bufferSize);

    //fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < latency_offset; i++){
        ring_buffer.pushSample(0, 0);
    }    

    AudioBufferF test_buffer(1, bufferSize);

    std::cout << "starting test" << std::endl;
    for (size_t repeat = 0; repeat < 50; repeat++)
    {
        fill_buffer(test_buffer);
        push_buffer_to_ringbuffer(test_buffer, ring_buffer);
        
        inferenceHandler.process(test_buffer.getArrayOfWritePointers(), bufferSize);

        for (size_t i = 0; i < bufferSize; i++){
            EXPECT_FLOAT_EQ(
                ring_buffer.popSample(0),
                test_buffer.getSample(0, i )
            );
        }
    }
}


TEST(Test_Inference, inference){

    size_t bufferSize = 1024;
    double sampleRate = 48000;

    std::vector<float> input_file;
    std::vector<float> output_file;
    // read_wav(string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/../../data/ts9_test1_in_FP32.wav", input_file);
    // read_wav(string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/../../data/ts9_test1_out_FP32.wav", output_file);
    read_wav(string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav", input_file);
    read_wav(string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/y_pred_tflite.wav", output_file);
    // read_wav("../extras/desktop/models/hybrid-nn/GuitarLSTM/data/ts9_test1_out_FP32.wav", output_file);
    assert(input_file.size() > 0);
    assert(output_file.size() > 0);
    InferenceConfig inferenceConfig = hybridNNConfig;

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
    inferenceHandler.setInferenceBackend(anira::TFLITE);

    
    size_t latency_offset = inferenceHandler.getLatency();

    RingBuffer ring_buffer;
    ring_buffer.initializeWithPositions(1, latency_offset+bufferSize);

    //fill the buffer with zeroes to compensate for the latency
    for (size_t i = 0; i < latency_offset; i++){
        ring_buffer.pushSample(0, 0);
    }    

    AudioBufferF test_buffer(1, bufferSize);

    std::cout << "starting test" << std::endl;
    for (size_t repeat = 0; repeat < 2; repeat++)
    {
        for (size_t i = 0; i < bufferSize; i++)
        {
            test_buffer.setSample(0, 1, input_file.at((repeat*bufferSize)+i));
            ring_buffer.pushSample(0, output_file.at((repeat*bufferSize)+i));
        }
                
        inferenceHandler.process(test_buffer.getArrayOfWritePointers(), bufferSize);

        for (size_t i = 0; i < bufferSize; i++){
            ASSERT_FLOAT_EQ(
                ring_buffer.popSample(0),
                test_buffer.getSample(0, i )
            );
        }
    }
}