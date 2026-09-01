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

    int* block_ptr = block.data();
    int* buffer_ptr = buffer.data();

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

TEST(Buffer, BufferSwap) {
    int const block_size = 10;

    anira::Buffer<int> buffer1(1, block_size);
    anira::Buffer<int> buffer2(1, block_size);

    // fill buffers
    for (int i = 0; i < block_size; i++) {
        buffer1.set_sample(0, i, i);
        buffer2.set_sample(0, i, i + block_size);
    }

    // check that buffers were filled corerctly
    for (int i = 0; i < block_size; i++) {
        ASSERT_EQ(buffer1.get_sample(0, i), i);
        ASSERT_EQ(buffer2.get_sample(0, i), i + block_size);
    }

    int* buffer1_ptr = buffer1.data();
    int* buffer2_ptr = buffer2.data();

    // Do the swap
    buffer1.swap_data(buffer2);

    // check that the blocks were actually swapped
    ASSERT_EQ(buffer1_ptr, buffer2.data());
    ASSERT_EQ(buffer2_ptr, buffer1.data());

    // check that buffer values were correctly swapped
    for (int i = 0; i < block_size; i++) {
        ASSERT_EQ(buffer1.get_sample(0, i), i + block_size);
        ASSERT_EQ(buffer2.get_sample(0, i), i);
    }
}
// Mismatched dimensions violate swap_data()'s contract: the containers never
// log, so the call asserts in debug builds and is a silent no-op in release.
#ifdef NDEBUG
// Both mismatch shapes hit the same "different dimensions" guard; one test
// keeps both input shapes (audit, docs/ci-overhaul.md step 9a).
TEST(Buffer, InvalidSwap) {
    anira::Buffer<int> size_mismatch1(1, 5), size_mismatch2(1, 6);
    int* size_ptr1 = size_mismatch1.data();
    int* size_ptr2 = size_mismatch2.data();
    size_mismatch1.swap_data(size_mismatch2);
    ASSERT_EQ(size_ptr1, size_mismatch1.data()) << "size mismatch swapped";
    ASSERT_EQ(size_ptr2, size_mismatch2.data()) << "size mismatch swapped";

    anira::Buffer<int> channel_mismatch1(2, 5), channel_mismatch2(1, 5);
    int* channel_ptr1 = channel_mismatch1.data();
    int* channel_ptr2 = channel_mismatch2.data();
    channel_mismatch1.swap_data(channel_mismatch2);
    ASSERT_EQ(channel_ptr1, channel_mismatch1.data()) << "channel mismatch swapped";
    ASSERT_EQ(channel_ptr2, channel_mismatch2.data()) << "channel mismatch swapped";
}
#elif GTEST_HAS_DEATH_TEST
TEST(BufferDeathTest, InvalidSwap) {
    anira::Buffer<int> size_mismatch1(1, 5), size_mismatch2(1, 6);
    EXPECT_DEATH(size_mismatch1.swap_data(size_mismatch2), "different dimensions");

    anira::Buffer<int> channel_mismatch1(2, 5), channel_mismatch2(1, 5);
    EXPECT_DEATH(channel_mismatch1.swap_data(channel_mismatch2), "different dimensions");
}
#endif
