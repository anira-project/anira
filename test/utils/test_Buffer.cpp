#include <anira/utils/Buffer.h>
#include <anira/utils/MemoryBlock.h>

#include <cstddef>

#include "gtest/gtest.h"

using namespace anira;
TEST(Buffer, SimpleWrite) {
    BufferF buffer = BufferF(1, 10);
    for (size_t i = 0; i < buffer.get_num_samples(); i++) {
        EXPECT_FLOAT_EQ(0.f, buffer.get_sample(0, i));
    }

    buffer.set_sample(0, 5, 0.9f);

    for (size_t i = 0; i < buffer.get_num_samples(); i++) {
        float const expected = i == 5 ? 0.9f : 0.f;
        EXPECT_FLOAT_EQ(expected, buffer.get_sample(0, i));
    }

    buffer.clear();
    for (size_t i = 0; i < buffer.get_num_samples(); i++) {
        EXPECT_FLOAT_EQ(0.f, buffer.get_sample(0, i));
    }
}

TEST(Buffer, BlockSwap) {
    int const block_size = 10;

    MemoryBlock<int> block;
    anira::Buffer<int> buffer(1, block_size);

    // fill blocks
    block.resize(block_size);
    for (int i = 0; i < block_size; i++) {
        block[i] = i;
        buffer.set_sample(0, i, i + block_size);
    }

    // check that buffers were filled corerctly
    for (int i = 0; i < block_size; i++) {
        ASSERT_EQ(block[i], i);
        ASSERT_EQ(buffer.get_sample(0, i), i + block_size);
    }

    const int* block_ptr = block.data();
    const int* buffer_ptr = buffer.data();

    // Do the swap
    block.swap_data(buffer.get_memory_block());
    buffer.reset_channel_ptr();

    // check that the blocks were actually swapped
    ASSERT_EQ(block_ptr, buffer.data());
    ASSERT_EQ(buffer_ptr, block.data());

    // check that buffer values were correctly swapped
    for (int i = 0; i < block_size; i++) {
        ASSERT_EQ(block[i], i + block_size);
        ASSERT_EQ(buffer.get_sample(0, i), i);
    }
}
