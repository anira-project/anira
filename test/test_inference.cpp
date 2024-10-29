#include "gtest/gtest.h"
#include <anira/anira.h>


#include "../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../extras/desktop/models/hybrid-nn/advanced-configs/HybridNNNoneProcessor.h" // Only needed for round trip test

#ifndef RANDOMDOUBLE
#define RANDOMDOUBLE

static double randomDouble(int LO, int HI){
  return (double) LO + (double) (std::rand()) / ((double) (RAND_MAX/(HI-LO)));
}

#endif // RANDOMDOUBLE



using namespace anira;

static void fill_buffers(AudioBufferF &buffer1, AudioBufferF &buffer2, size_t buffersize){
    for (size_t i = 0; i < buffersize; i++){
        double new_val = randomDouble(-1, 1);
        buffer1.setSample(0, i, new_val);
        buffer2.setSample(0, i, new_val);
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

    AudioBufferF buffer(1, bufferSize);
    AudioBufferF passBuffer(1, bufferSize);
    
    size_t latency_offset = inferenceHandler.getLatency();
    std::cout << "latency in samples: " << latency_offset << std::endl;
    ASSERT_EQ(latency_offset, bufferSize);

    fill_buffers(buffer, passBuffer, bufferSize);

    for (size_t i = 0; i < bufferSize; i++){
        ASSERT_FLOAT_EQ(
            buffer.getSample(0, i),
            passBuffer.getSample(0, i)
        );
    }

    for (size_t repeat = 0; repeat < 50; repeat++)
    {
            
        for (size_t i = 0; i < bufferSize; i++){
            passBuffer.setSample(0, i, buffer.getSample(0, i));
        }
        
        inferenceHandler.process(passBuffer.getArrayOfWritePointers(), bufferSize);

        // ignore first cycle, as the output will be all 0
        if (repeat == 0){
            for (size_t i = 0; i < bufferSize; i++){
                EXPECT_FLOAT_EQ(
                    0,
                    passBuffer.getSample(0, i )
                );
            }
            continue;
        }

        for (size_t i = 0; i < bufferSize; i++){
            EXPECT_FLOAT_EQ(
                buffer.getSample(0, i),
                passBuffer.getSample(0, i )
            );
        }
    }

    // fill_buffers(buffer, passBuffer, bufferSize);
    // inferenceHandler.process(passBuffer.getArrayOfWritePointers(), bufferSize);

    // for (size_t i = 0; i < bufferSize; i++){
    //     EXPECT_FLOAT_EQ(
    //         buffer.getSample(0, i),
    //         passBuffer.getSample(0, i)
    //     );
    // }
}