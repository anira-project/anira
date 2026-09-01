// anira/utils/helperFunctions.h is a public header, but its only includers are
// benchmark.h and the benchmark fixture — both excluded from the coverage
// report — so it never reached a measured translation unit and dropped out of
// the report entirely rather than showing up uncovered. These tests put it back
// in, and pin the statistics helpers the benchmark output is built from.

#include <anira/utils/Buffer.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/helperFunctions.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

namespace {

// Deliberately unsorted, and with a known order statistic at every index.
std::vector<double> samples() {
    return {5.0, 1.0, 4.0, 2.0, 3.0};
}

}  // namespace

TEST(HelperFunctions, RandomSampleStaysInAudioRange) {
    for (int i = 0; i < 1000; ++i) {
        const float sample = anira::random_sample();
        EXPECT_GE(sample, -1.0F);
        EXPECT_LE(sample, 1.0F);
    }
}

TEST(HelperFunctions, PercentileOfAnEmptyVectorThrows) {
    EXPECT_THROW((void)anira::calculate_percentile({}, 0.5), std::invalid_argument);
}

// The index is truncated, so the result is always an element of the input —
// never a value interpolated between two of them.
TEST(HelperFunctions, PercentilePicksAnExistingElement) {
    EXPECT_DOUBLE_EQ(anira::calculate_percentile(samples(), 0.0), 1.0);
    EXPECT_DOUBLE_EQ(anira::calculate_percentile(samples(), 0.5), 3.0);
    EXPECT_DOUBLE_EQ(anira::calculate_percentile(samples(), 1.0), 5.0);
    // 0.99 * (5 - 1) = 3.96, truncated to index 3.
    EXPECT_DOUBLE_EQ(anira::calculate_percentile(samples(), 0.99), 4.0);
}

TEST(HelperFunctions, PercentileOfASingleValue) {
    EXPECT_DOUBLE_EQ(anira::calculate_percentile({7.5}, 0.0), 7.5);
    EXPECT_DOUBLE_EQ(anira::calculate_percentile({7.5}, 1.0), 7.5);
}

TEST(HelperFunctions, MinAndMax) {
    EXPECT_DOUBLE_EQ(anira::calculate_min(samples()), 1.0);
    EXPECT_DOUBLE_EQ(anira::calculate_max(samples()), 5.0);
    EXPECT_DOUBLE_EQ(anira::calculate_min({-2.5}), -2.5);
    EXPECT_DOUBLE_EQ(anira::calculate_max({-2.5}), -2.5);
}

TEST(HelperFunctions, FillBufferTouchesEveryChannelAndSample) {
    anira::BufferF buffer(3, 16);
    for (size_t channel = 0; channel < buffer.get_num_channels(); ++channel) {
        for (size_t sample = 0; sample < buffer.get_num_samples(); ++sample) {
            buffer.set_sample(channel, sample, 2.0F);  // outside random_sample()'s range
        }
    }

    anira::fill_buffer(buffer);

    for (size_t channel = 0; channel < buffer.get_num_channels(); ++channel) {
        for (size_t sample = 0; sample < buffer.get_num_samples(); ++sample) {
            const float value = buffer.get_sample(channel, sample);
            EXPECT_GE(value, -1.0F) << "channel " << channel << " sample " << sample;
            EXPECT_LE(value, 1.0F) << "channel " << channel << " sample " << sample;
        }
    }
}

TEST(HelperFunctions, PushBufferToRingBufferTransfersEveryChannel) {
    anira::BufferF buffer(2, 4);
    for (size_t channel = 0; channel < buffer.get_num_channels(); ++channel) {
        for (size_t sample = 0; sample < buffer.get_num_samples(); ++sample) {
            buffer.set_sample(channel, sample, static_cast<float>((channel * 10) + sample));
        }
    }

    anira::RingBuffer ring_buffer;
    ring_buffer.initialize_with_positions(2, 4);
    anira::push_buffer_to_ringbuffer(buffer, ring_buffer);

    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t sample = 0; sample < 4; ++sample) {
            EXPECT_FLOAT_EQ(ring_buffer.pop_sample(channel),
                            static_cast<float>((channel * 10) + sample))
                << "channel " << channel << " sample " << sample;
        }
    }
}

TEST(HelperFunctions, PushBufferToRingBufferRejectsEmptyOperands) {
    anira::RingBuffer ring_buffer;
    ring_buffer.initialize_with_positions(1, 4);

    const anira::BufferF empty_buffer(0, 0);
    EXPECT_THROW(anira::push_buffer_to_ringbuffer(empty_buffer, ring_buffer),
                 std::invalid_argument);

    const anira::BufferF buffer(1, 4);
    anira::RingBuffer uninitialized;
    EXPECT_THROW(anira::push_buffer_to_ringbuffer(buffer, uninitialized), std::invalid_argument);
}
