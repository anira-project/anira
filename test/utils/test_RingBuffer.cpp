#include "gtest/gtest.h"
#include <anira/anira.h>

using namespace anira;

class RingBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a 2-channel, 5-sample ring buffer for most tests
        ring_buffer.initialize_with_positions(2, 5);
    }

    void TearDown() override {
        // Clean up after each test
    }

    RingBuffer ring_buffer;
};

// Test basic initialization
TEST_F(RingBufferTest, Initialization) {
    EXPECT_EQ(ring_buffer.get_num_channels(), 2);
    EXPECT_EQ(ring_buffer.get_num_samples(), 5);
    
    // All channels should start empty
    for (size_t channel = 0; channel < ring_buffer.get_num_channels(); ++channel) {
        EXPECT_EQ(ring_buffer.get_available_samples(channel), 0);
        EXPECT_EQ(ring_buffer.get_available_past_samples(channel), 5);
    }
}

// Test single channel push and pop operations
TEST_F(RingBufferTest, SingleChannelPushPop) {
    const size_t channel = 0;
    const float test_values[] = {1.0f, 2.0f, 3.0f};
    
    // Push some samples
    for (float value : test_values) {
        ring_buffer.push_sample(channel, value);
    }
    
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 3);
    
    // Pop samples and verify they come out in FIFO order
    for (float expected_value : test_values) {
        float popped_value = ring_buffer.pop_sample(channel);
        EXPECT_FLOAT_EQ(popped_value, expected_value);
    }
    
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 0);
}

// Test multi-channel operations
TEST_F(RingBufferTest, MultiChannelOperations) {
    const float channel0_values[] = {1.0f, 2.0f, 3.0f};
    const float channel1_values[] = {10.0f, 20.0f, 30.0f};
    
    // Push samples to both channels
    for (size_t i = 0; i < 3; ++i) {
        ring_buffer.push_sample(0, channel0_values[i]);
        ring_buffer.push_sample(1, channel1_values[i]);
    }
    
    // Verify available samples for both channels
    EXPECT_EQ(ring_buffer.get_available_samples(0), 3);
    EXPECT_EQ(ring_buffer.get_available_samples(1), 3);
    
    // Pop from channel 0 and verify channel 1 is unaffected
    float popped = ring_buffer.pop_sample(0);
    EXPECT_FLOAT_EQ(popped, 1.0f);
    EXPECT_EQ(ring_buffer.get_available_samples(0), 2);
    EXPECT_EQ(ring_buffer.get_available_samples(1), 3);
    
    // Pop from channel 1 and verify
    popped = ring_buffer.pop_sample(1);
    EXPECT_FLOAT_EQ(popped, 10.0f);
    EXPECT_EQ(ring_buffer.get_available_samples(0), 2);
    EXPECT_EQ(ring_buffer.get_available_samples(1), 2);
}

// Test buffer overflow behavior
TEST_F(RingBufferTest, BufferOverflow) {
    const size_t channel = 0;
    
    // Fill the buffer completely (5 samples)
    for (int i = 1; i <= 5; ++i) {
        ring_buffer.push_sample(channel, static_cast<float>(i));
    }
    
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 5);
    
    // Capture stderr to check for overflow error
    testing::internal::CaptureStderr();
    
    // Push one more sample (should cause overflow)
    ring_buffer.push_sample(channel, 6.0f);
    
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("Buffer overflow detected") != std::string::npos);
    
    // Buffer should still be full
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 5);
    
    // The oldest sample (1.0f) should have been overwritten
    // So we should get 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    float expected_values[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (float expected : expected_values) {
        float popped = ring_buffer.pop_sample(channel);
        EXPECT_FLOAT_EQ(popped, expected);
    }
}

// Test popping from empty buffer
TEST_F(RingBufferTest, PopFromEmptyBuffer) {
    const size_t channel = 0;
    
    // Capture stderr to check for empty buffer error
    testing::internal::CaptureStderr();
    
    float popped = ring_buffer.pop_sample(channel);
    
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("Attempted to pop sample from empty buffer") != std::string::npos);
    EXPECT_FLOAT_EQ(popped, 0.0f);
}

// Test get_future_sample with offset
TEST_F(RingBufferTest, GetSampleWithOffset) {
    const size_t channel = 0;
    const float test_values[] = {1.0f, 2.0f, 3.0f};
    
    // Push some samples
    for (float value : test_values) {
        ring_buffer.push_sample(channel, value);
    }
    
    // Test getting samples with different offsets
    EXPECT_FLOAT_EQ(ring_buffer.get_future_sample(channel, 0), 1.0f);  // First sample
    EXPECT_FLOAT_EQ(ring_buffer.get_future_sample(channel, 1), 2.0f);  // Second sample
    EXPECT_FLOAT_EQ(ring_buffer.get_future_sample(channel, 2), 3.0f);  // Third sample
}

// Test get_future_sample with invalid offset
TEST_F(RingBufferTest, GetSampleInvalidOffset) {
    const size_t channel = 0;
    
    // Push only one sample
    ring_buffer.push_sample(channel, 1.0f);
    
    // Capture stderr to check for invalid offset error
    testing::internal::CaptureStderr();
    
    // Try to get sample with offset beyond available samples
    float sample = ring_buffer.get_future_sample(channel, 5);
    
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("Attempted to get sample with offset") != std::string::npos);
    EXPECT_FLOAT_EQ(sample, 0.0f);
}

// Test get_past_sample functionality
TEST_F(RingBufferTest, GetPastSample) {
    const size_t channel = 0;
    const float test_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // Push and then pop some samples to create past samples
    for (float value : test_values) {
        ring_buffer.push_sample(channel, value);
    }
    
    // Pop first two samples
    ring_buffer.pop_sample(channel);  // Pop 1.0f
    ring_buffer.pop_sample(channel);  // Pop 2.0f
    
    // Now we should have 3.0f, 4.0f, 5.0f in the buffer
    EXPECT_FLOAT_EQ(ring_buffer.get_past_sample(channel, 0), 3.0f);  // Current sample
    EXPECT_FLOAT_EQ(ring_buffer.get_past_sample(channel, 1), 2.0f);
    EXPECT_FLOAT_EQ(ring_buffer.get_past_sample(channel, 2), 1.0f);

    // Capture stderr to check for past sample retrieval
    testing::internal::CaptureStderr();
    EXPECT_FLOAT_EQ(ring_buffer.get_past_sample(channel, 3), 0.0f);  // No sample available

    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("RingBuffer: Attempted to get past sample with offset") != std::string::npos);
}

// Test circular buffer wrap-around
TEST_F(RingBufferTest, CircularWrapAround) {
    const size_t channel = 0;
    
    // Fill buffer completely
    for (int i = 1; i <= 5; ++i) {
        ring_buffer.push_sample(channel, static_cast<float>(i));
    }
    
    // Pop some samples
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 1.0f);
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 2.0f);
    
    // Push more samples (should wrap around)
    ring_buffer.push_sample(channel, 6.0f);
    ring_buffer.push_sample(channel, 7.0f);
    
    // Verify the remaining samples
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 3.0f);
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 4.0f);
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 5.0f);
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 6.0f);
    EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel), 7.0f);
}

// Test clear_with_positions
TEST_F(RingBufferTest, ClearWithPositions) {
    const size_t channel = 0;
    
    // Add some samples
    ring_buffer.push_sample(channel, 1.0f);
    ring_buffer.push_sample(channel, 2.0f);
    ring_buffer.push_sample(channel, 3.0f);
    
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 3);
    
    // Clear the buffer
    ring_buffer.clear_with_positions();
    
    // Buffer should be empty
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 0);
    EXPECT_EQ(ring_buffer.get_available_past_samples(channel), 5);
    
    // All samples should be zero
    for (size_t i = 0; i < ring_buffer.get_num_samples(); ++i) {
        EXPECT_FLOAT_EQ(ring_buffer.Buffer<float>::get_sample(channel, i), 0.0f);
    }
}

// Test available samples calculation
TEST_F(RingBufferTest, AvailableSamplesCalculation) {
    const size_t channel = 0;
    
    // Initially empty
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 0);
    EXPECT_EQ(ring_buffer.get_available_past_samples(channel), 5);
    
    // Add samples one by one and check available count
    for (int i = 1; i <= 5; ++i) {
        ring_buffer.push_sample(channel, static_cast<float>(i));
        EXPECT_EQ(ring_buffer.get_available_samples(channel), i);
        EXPECT_EQ(ring_buffer.get_available_past_samples(channel), 5 - i);
    }
    
    // Buffer is now full
    EXPECT_EQ(ring_buffer.get_available_samples(channel), 5);
    EXPECT_EQ(ring_buffer.get_available_past_samples(channel), 0);
    
    // Pop samples and check count decreases
    for (int i = 4; i >= 0; --i) {
        ring_buffer.pop_sample(channel);
        EXPECT_EQ(ring_buffer.get_available_samples(channel), i);
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
    
    testing::internal::CaptureStderr();
    small_buffer.push_sample(channel, 2.0f);  // Should overflow
    std::string output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(output.find("Buffer overflow detected") != std::string::npos);
    
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
    
    // Pushing to zero-sized buffer should be handled
    testing::internal::CaptureStderr();
    zero_buffer.push_sample(channel, 1.0f);
    std::string output = testing::internal::GetCapturedStderr();
    // May or may not produce an error, depends on implementation
}

// Test inheritance from Buffer<float>
TEST_F(RingBufferTest, BufferInheritance) {
    // Test that RingBuffer still provides Buffer functionality
    EXPECT_EQ(ring_buffer.get_num_channels(), 2);
    EXPECT_EQ(ring_buffer.get_num_samples(), 5);
    
    // Test direct access to underlying buffer (inherited methods)
    ring_buffer.Buffer<float>::set_sample(0, 0, 99.0f);
    EXPECT_FLOAT_EQ(ring_buffer.Buffer<float>::get_sample(0, 0), 99.0f);
    
    // Test that we can get pointers to the data
    float* write_ptr = ring_buffer.get_write_pointer(0);
    const float* read_ptr = ring_buffer.get_read_pointer(0);
    EXPECT_NE(write_ptr, nullptr);
    EXPECT_NE(read_ptr, nullptr);
    EXPECT_EQ(write_ptr, read_ptr);  // Should point to same location
}