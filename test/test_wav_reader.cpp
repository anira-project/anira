#include "WavReader.h"
#include "gtest/gtest.h"



TEST(WavReader, read){
    std::vector<float> data_input;
    std::vector<float> reference = {-0.008862195, -0.007644168, -0.006510143, -0.004956109, -0.002772061, 0.0001680037, 0.0034440758, 0.0063841403, 0.008064178, 0.007770171, 0.0058801295, 0.0029820655, 4.2000924e-05, -0.0023520517, -0.004284094, -0.0059221303, -0.007098156, -0.007224159, -0.005502121, -0.0020580452};
    read_wav(string(GUITARLSTM_MODELS_PATH_TENSORFLOW) + "/model_0/x_test.wav", data_input);

    for (size_t i = 0; i < reference.size(); i++)
    {
        EXPECT_FLOAT_EQ(data_input[i], reference[i]);
    }
    

}