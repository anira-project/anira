#include "gtest/gtest.h"
#include <anira/anira.h>

#ifndef RANDOMDOUBLE
#define RANDOMDOUBLE

static double randomDouble(int LO, int HI){
  return (double) LO + (double) (std::rand()) / ((double) (RAND_MAX/(HI-LO)));
}

#endif // RANDOMDOUBLE

using namespace anira;
TEST(Test_Audiobuffer, simple_write){
    AudioBufferF buffer = AudioBufferF(1,10);
    for (size_t i = 0; i < buffer.getNumSamples(); i++){
        EXPECT_FLOAT_EQ(0.f, buffer.getSample(0,i));
    }

    buffer.setSample(0,5, 0.9f);
    
    for (size_t i = 0; i < buffer.getNumSamples(); i++){
        float expected = i == 5 ? 0.9f : 0.f;
        EXPECT_FLOAT_EQ(expected, buffer.getSample(0,i));
    }

    buffer.clear();
    for (size_t i = 0; i < buffer.getNumSamples(); i++){
        EXPECT_FLOAT_EQ(0.f, buffer.getSample(0,i));
    }
}