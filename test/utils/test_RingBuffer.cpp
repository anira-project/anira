#include <anira/utils/Buffer.h>
#include <anira/utils/RingBuffer.h>

#include <array>
#include <cstddef>
#include <string>

#include "gtest/gtest.h"

using namespace anira;

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

// Test basic initialization
TEST_F(RingBufferTest, Initialization) {
    EXPECT_EQ(m_ring_buffer.get_num_channels(), 2);
    EXPECT_EQ(m_ring_buffer.get_num_samples(), 5);

    // All channels should start empty
    for (size_t channel = 0; channel < m_ring_buffer.get_num_channels(); ++channel) {
        EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 0);
        // Past samples are actually-written history, none yet after init
        EXPECT_EQ(m_ring_buffer.get_available_past_samples(channel), 0);
    }
}

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

// Test multi-channel operations
TEST_F(RingBufferTest, MultiChannelOperations) {
    const std::array<float, 3> channel0_values = {1.0f, 2.0f, 3.0f};
    const std::array<float, 3> channel1_values = {10.0f, 20.0f, 30.0f};

    // Push samples to both channels
    for (size_t i = 0; i < 3; ++i) {
        m_ring_buffer.push_sample(0, channel0_values[i]);
        m_ring_buffer.push_sample(1, channel1_values[i]);
    }

    // Verify available samples for both channels
    EXPECT_EQ(m_ring_buffer.get_available_samples(0), 3);
    EXPECT_EQ(m_ring_buffer.get_available_samples(1), 3);

    // Pop from channel 0 and verify channel 1 is unaffected
    float popped = m_ring_buffer.pop_sample(0);
    EXPECT_FLOAT_EQ(popped, 1.0f);
    EXPECT_EQ(m_ring_buffer.get_available_samples(0), 2);
    EXPECT_EQ(m_ring_buffer.get_available_samples(1), 3);

    // Pop from channel 1 and verify
    popped = m_ring_buffer.pop_sample(1);
    EXPECT_FLOAT_EQ(popped, 10.0f);
    EXPECT_EQ(m_ring_buffer.get_available_samples(0), 2);
    EXPECT_EQ(m_ring_buffer.get_available_samples(1), 2);
}

// Test buffer overflow behavior
TEST_F(RingBufferTest, BufferOverflow) {
    const size_t channel = 0;

    // Fill the buffer completely (5 samples)
    for (int i = 1; i <= 5; ++i) { m_ring_buffer.push_sample(channel, static_cast<float>(i)); }

    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 5);

    // Push one more sample: silently overwrites the oldest (delay-line
    // semantics of the thl::core ring buffer)
    m_ring_buffer.push_sample(channel, 6.0f);

    // Buffer should still be full
    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 5);

    // The oldest sample (1.0f) should have been overwritten
    // So we should get 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    std::array<float, 5> const expected_values = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (float const expected : expected_values) {
        float const popped = m_ring_buffer.pop_sample(channel);
        EXPECT_FLOAT_EQ(popped, expected);
    }
}

// Test popping from empty buffer
TEST_F(RingBufferTest, PopFromEmptyBuffer) {
    const size_t channel = 0;

    // Popping an empty channel returns a value-initialized sample
    float const popped = m_ring_buffer.pop_sample(channel);
    EXPECT_FLOAT_EQ(popped, 0.0f);
}

// Test get_future_sample with offset
TEST_F(RingBufferTest, GetSampleWithOffset) {
    const size_t channel = 0;
    const std::array<float, 3> test_values = {1.0f, 2.0f, 3.0f};

    // Push some samples
    for (float const value : test_values) { m_ring_buffer.push_sample(channel, value); }

    // Test getting samples with different offsets
    EXPECT_FLOAT_EQ(m_ring_buffer.get_future_sample(channel, 0), 1.0f);  // First sample
    EXPECT_FLOAT_EQ(m_ring_buffer.get_future_sample(channel, 1), 2.0f);  // Second sample
    EXPECT_FLOAT_EQ(m_ring_buffer.get_future_sample(channel, 2), 3.0f);  // Third sample
}

// Test get_future_sample with invalid offset
TEST_F(RingBufferTest, GetSampleInvalidOffset) {
    const size_t channel = 0;

    // Push only one sample
    m_ring_buffer.push_sample(channel, 1.0f);

    // Offsets are taken modulo the capacity (the caller is responsible for
    // checking get_available_samples()); offset 5 on a 5-slot ring wraps to
    // the pushed sample
    float const sample = m_ring_buffer.get_future_sample(channel, 5);
    EXPECT_FLOAT_EQ(sample, 1.0f);
}

// Test get_past_sample functionality
TEST_F(RingBufferTest, GetPastSample) {
    const size_t channel = 0;
    const std::array<float, 5> test_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Push and then pop some samples to create past samples
    for (float const value : test_values) { m_ring_buffer.push_sample(channel, value); }

    // Pop first two samples
    m_ring_buffer.pop_sample(channel);  // Pop 1.0f
    m_ring_buffer.pop_sample(channel);  // Pop 2.0f

    // Now we should have 3.0f, 4.0f, 5.0f in the buffer
    EXPECT_FLOAT_EQ(m_ring_buffer.get_past_sample(channel, 0), 3.0f);  // Current sample
    EXPECT_FLOAT_EQ(m_ring_buffer.get_past_sample(channel, 1), 2.0f);
    EXPECT_FLOAT_EQ(m_ring_buffer.get_past_sample(channel, 2), 1.0f);

    // Beyond the written history the offset wraps modulo the capacity (the
    // caller guards via get_available_past_samples()); offset 3 lands on the
    // newest unread sample here
    EXPECT_FLOAT_EQ(m_ring_buffer.get_past_sample(channel, 3), 5.0f);
}

// Test circular buffer wrap-around
TEST_F(RingBufferTest, CircularWrapAround) {
    const size_t channel = 0;

    // Fill buffer completely
    for (int i = 1; i <= 5; ++i) { m_ring_buffer.push_sample(channel, static_cast<float>(i)); }

    // Pop some samples
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 1.0f);
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 2.0f);

    // Push more samples (should wrap around)
    m_ring_buffer.push_sample(channel, 6.0f);
    m_ring_buffer.push_sample(channel, 7.0f);

    // Verify the remaining samples
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 3.0f);
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 4.0f);
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 5.0f);
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 6.0f);
    EXPECT_FLOAT_EQ(m_ring_buffer.pop_sample(channel), 7.0f);
}

// Test clear_with_positions
TEST_F(RingBufferTest, ClearWithPositions) {
    const size_t channel = 0;

    // Add some samples
    m_ring_buffer.push_sample(channel, 1.0f);
    m_ring_buffer.push_sample(channel, 2.0f);
    m_ring_buffer.push_sample(channel, 3.0f);

    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 3);

    // Clear the buffer
    m_ring_buffer.clear_with_positions();

    // Buffer should be empty, history reset
    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 0);
    EXPECT_EQ(m_ring_buffer.get_available_past_samples(channel), 0);

    // All slots should be zero again
    for (size_t i = 0; i < m_ring_buffer.get_num_samples(); ++i) {
        EXPECT_FLOAT_EQ(m_ring_buffer.get_future_sample(channel, i), 0.0f);
    }
}

// Test available samples calculation
TEST_F(RingBufferTest, AvailableSamplesCalculation) {
    const size_t channel = 0;

    // Initially empty, no history
    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 0);
    EXPECT_EQ(m_ring_buffer.get_available_past_samples(channel), 0);

    // Add samples one by one: unread count grows, nothing is history yet
    for (int i = 1; i <= 5; ++i) {
        m_ring_buffer.push_sample(channel, static_cast<float>(i));
        EXPECT_EQ(m_ring_buffer.get_available_samples(channel), i);
        EXPECT_EQ(m_ring_buffer.get_available_past_samples(channel), 0);
    }

    // Buffer is now full
    EXPECT_EQ(m_ring_buffer.get_available_samples(channel), 5);

    // Pop samples: unread count decreases and consumed samples become history
    for (int i = 4; i >= 0; --i) {
        m_ring_buffer.pop_sample(channel);
        EXPECT_EQ(m_ring_buffer.get_available_samples(channel), i);
        EXPECT_EQ(m_ring_buffer.get_available_past_samples(channel), 5 - i);
    }
}

// Test edge cases with single sample buffer
TEST(RingBufferSingleSample, EdgeCases) {
    RingBuffer small_buffer;
    small_buffer.initialize_with_positions(1, 1);

    const size_t channel = 0;

    // Test single sample operations
    small_buffer.push_sample(channel, 42.0f);
    EXPECT_EQ(small_buffer.get_available_samples(channel), 1);

    EXPECT_FLOAT_EQ(small_buffer.pop_sample(channel), 42.0f);
    EXPECT_EQ(small_buffer.get_available_samples(channel), 0);

    // Test overflow with single sample
    small_buffer.push_sample(channel, 1.0f);

    small_buffer.push_sample(channel, 2.0f);  // Silently overwrites

    // Should get the newer value
    EXPECT_FLOAT_EQ(small_buffer.pop_sample(channel), 2.0f);
}

// Test zero-sized buffer (edge case)
TEST(RingBufferZeroSize, EdgeCase) {
    RingBuffer zero_buffer;
    zero_buffer.initialize_with_positions(1, 0);

    const size_t channel = 0;

    // Operations on zero-sized buffer should handle gracefully
    EXPECT_EQ(zero_buffer.get_available_samples(channel), 0);
    EXPECT_EQ(zero_buffer.get_available_past_samples(channel), 0);

    // Pushing to a zero-sized buffer is a no-op
    zero_buffer.push_sample(channel, 1.0f);
    EXPECT_EQ(zero_buffer.get_available_samples(channel), 0);
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