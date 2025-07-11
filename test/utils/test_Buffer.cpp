#include "gtest/gtest.h"
#include <anira/anira.h>

using namespace anira;
TEST(Buffer, SimpleWrite){
    BufferF buffer = BufferF(1,10);
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

TEST(Buffer, BlockSwap){
    int block_size = 10;

    MemoryBlock<int> block;
    anira::Buffer<int> buffer(1, block_size);

    // fill blocks
    block.resize(block_size);
    for (int i = 0; i < block_size; i++){
        block[i] = i;
        buffer.set_sample(0, i, i+block_size);
    }

    // check that buffers were filled corerctly
    for (int i = 0; i < block_size; i++)
    {
        ASSERT_EQ(block[i], i);
        ASSERT_EQ(buffer.get_sample(0,i), i+block_size);
    }
    
    int* block_ptr = block.data();
    int* buffer_ptr = buffer.data();
    
    // Do the swap
    block.swap_data(buffer.get_memory_block());
    buffer.reset_channel_ptr();

    // check that the blocks were actually swapped
    ASSERT_EQ(block_ptr, buffer.data());
    ASSERT_EQ(buffer_ptr, block.data());

    // check that buffer values were correctly swapped
    for (int i = 0; i < block_size; i++)
    {
        ASSERT_EQ(block[i], i+block_size);
        ASSERT_EQ(buffer.get_sample(0,i), i);
    }
}

TEST(Buffer, BufferSwap){
    int block_size = 10;

    anira::Buffer<int> buffer1(1, block_size);
    anira::Buffer<int> buffer2(1, block_size);

    // fill buffers
    for (int i = 0; i < block_size; i++){
        buffer1.set_sample(0, i, i);
        buffer2.set_sample(0, i, i+block_size);
    }

    // check that buffers were filled corerctly
    for (int i = 0; i < block_size; i++)
    {
        ASSERT_EQ(buffer1.get_sample(0,i), i);
        ASSERT_EQ(buffer2.get_sample(0,i), i+block_size);
    }
    
    int* buffer1_ptr = buffer1.data();
    int* buffer2_ptr = buffer2.data();
    
    // Do the swap
    buffer1.swap_data(buffer2);

    // check that the blocks were actually swapped
    ASSERT_EQ(buffer1_ptr, buffer2.data());
    ASSERT_EQ(buffer2_ptr, buffer1.data());

    // check that buffer values were correctly swapped
    for (int i = 0; i < block_size; i++)
    {
        ASSERT_EQ(buffer1.get_sample(0,i), i+block_size);
        ASSERT_EQ(buffer2.get_sample(0,i), i);
    }
}
TEST(Buffer, InvalidSizeSwap){
    anira::Buffer<int> buffer1(1, 5);
    anira::Buffer<int> buffer2(1, 6);
    int* buffer1_ptr = buffer1.data();
    int* buffer2_ptr = buffer2.data();
    
    testing::internal::CaptureStderr();
    buffer1.swap_data(buffer2);

    std::string output = testing::internal::GetCapturedStderr();

        // check that the blocks were actually swapped
    ASSERT_EQ(buffer1_ptr, buffer1.data());
    ASSERT_EQ(buffer2_ptr, buffer2.data());
    ASSERT_EQ(output, std::string("Cannot swap data, buffers have different number of channels or sizes!\n"));
}

TEST(Buffer, InvalidChannelsSwap){
    anira::Buffer<int> buffer1(2, 5);
    anira::Buffer<int> buffer2(1, 5);
    int* buffer1_ptr = buffer1.data();
    int* buffer2_ptr = buffer2.data();
    
    testing::internal::CaptureStderr();
    buffer1.swap_data(buffer2);

    std::string output = testing::internal::GetCapturedStderr();

        // check that the blocks were actually swapped
    ASSERT_EQ(buffer1_ptr, buffer1.data());
    ASSERT_EQ(buffer2_ptr, buffer2.data());
    ASSERT_EQ(output, std::string("Cannot swap data, buffers have different number of channels or sizes!\n"));
}

// TEST(Buffer, BlockOfBlocks){
//     int block_size = 10;
//     MemoryBlock<std::atomic<int>> block_of_atomics(block_size);
//     for (int i = 0; i < block_size; i++) {
//         block_of_atomics[i].store(i);
//     }
//     std::atomic<int> const* block_ptr = block_of_atomics.data();

//     // TODO assert something here?

// }