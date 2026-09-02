#include <anira/utils/RingBuffer.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"

using namespace anira;

namespace {
class RingBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a 2-channel, 5-sample ring buffer for most tests
        m_ring_buffer.initialize_with_positions(2, 5);
    }

    void TearDown() override {
        // Clean up after each test
    }

    RingBuffer m_ring_buffer;
};
}  // namespace

// Test basic initialization
// RingBufferTest.Initialization was removed: anira::RingBuffer is an alias over
// thl::core::RingBuffer, whose own suite asserts the same post-init state
// (RingBuffer.InitialiseWithPositions: dimensions + available samples;
// ClearWithPositions: zero past samples) — audit, docs/ci-overhaul.md step 9a.
// Test single channel push and pop operations
TEST_F(RingBufferTest, SingleChannelPushPop) {
    const size_t channel = 0;
    const std::array<float, 3> test_values = {1.0f, 2.0f, 3.0f};

    // Push some samples
    for (float const value : test_values) { m_ring_buffer.push_sample(channel, value); }

    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 3);

    // Pop samples and verify they come out in FIFO order
    for (float const expected_value : test_values) {
        float const popped_value = m_ring_buffer.pop_sample(channel);
        EXPECT_FLOAT_EQ(popped_value, expected_value);
    }

    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 0);
}

// The ring buffer no longer inherits Buffer<float> (thl::core composition):
// dimensions are exposed directly, and element types beyond float are
// available through anira::RingBufferT<T>.
TEST_F(RingBufferTest, DimensionsAndTypedVariant) {
    EXPECT_EQ(m_ring_buffer.get_num_channels(), 2);
    EXPECT_EQ(m_ring_buffer.get_num_samples(), 5);

    RingBufferT<int64_t> tokens;
    tokens.initialize_with_positions(1, 4);
    const int64_t big = (1LL << 40) + 7;  // exact beyond float32's 2^24 range
    tokens.push_sample(0, big);
    EXPECT_EQ(tokens.pop_sample(0), big);
}