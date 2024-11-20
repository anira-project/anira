#include "gtest/gtest.h"
#include <anira/anira.h>

using namespace anira;
TEST(Test_Audiobuffer, simple_write){
    AudioBufferF buffer = AudioBufferF(1,10);
    for (size_t i = 0; i < buffer.get_num_samples(); i++){
        EXPECT_FLOAT_EQ(0.f, buffer.get_sample(0,i));
    }

    buffer.set_sample(0,5, 0.9f);
    
    for (size_t i = 0; i < buffer.get_num_samples(); i++){
        float expected = i == 5 ? 0.9f : 0.f;
        EXPECT_FLOAT_EQ(expected, buffer.get_sample(0,i));
    }

    buffer.clear();
    for (size_t i = 0; i < buffer.get_num_samples(); i++){
        EXPECT_FLOAT_EQ(0.f, buffer.get_sample(0,i));
    }
}