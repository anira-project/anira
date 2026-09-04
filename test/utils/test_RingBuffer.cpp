#include <anira/abi/enums.h>
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
// ---- The typed rings: one instantiation per scalar dtype behind anira_ring -------------------

namespace {

// Push, pop, peek, fill, discard and the batched window pop over one channel of a ring of
// `dtype`, stored as `T`; a call with another dtype moves nothing.
template <typename T>
void round_trip(anira_dtype dtype) {
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(2, 8, dtype));
    EXPECT_EQ(ring.dtype(), dtype);
    EXPECT_EQ(ring.num_channels(), 2U);
    EXPECT_EQ(ring.capacity(), 8U);
    EXPECT_EQ(ring.available(1), 0U);

    const std::array<T, 5> in{T(1), T(2), T(3), T(4), T(5)};
    EXPECT_EQ(ring.push_block(1, in.data(), dtype, in.size()), 5U);
    EXPECT_EQ(ring.available(1), 5U);
    EXPECT_EQ(ring.available(0), 0U) << "channels are independent";

    std::array<T, 3> out{};
    EXPECT_EQ(ring.pop_block(1, out.data(), dtype, out.size()), 3U);
    EXPECT_EQ(out[0], T(1));
    EXPECT_EQ(out[2], T(3));
    EXPECT_EQ(ring.available(1), 2U);
    EXPECT_EQ(ring.available_past(1), 3U);

    std::array<T, 3> past{};
    EXPECT_EQ(ring.peek_past_block(1, past.data(), dtype, past.size()), 3U);
    EXPECT_EQ(past[0], T(1)) << "oldest first";
    EXPECT_EQ(past[2], T(3)) << "the sample popped last";

    const T seven = T(7);
    EXPECT_EQ(ring.push_fill(1, &seven, dtype, 2), 2U);
    EXPECT_EQ(ring.available(1), 4U);
    EXPECT_EQ(ring.discard(1, 1), 1U);  // drops T(4)
    EXPECT_EQ(ring.available(1), 3U);   // T(5), T(7), T(7)

    // Two windows of one old and one new element each: [4 5] then [5 7].
    std::array<T, 4> windows{};
    EXPECT_EQ(ring.pop_windows(1, windows.data(), dtype, 1, 1, 0, 2), 4U);
    EXPECT_EQ(windows[0], T(4));
    EXPECT_EQ(windows[1], T(5));
    EXPECT_EQ(windows[2], T(5));
    EXPECT_EQ(windows[3], T(7));
    EXPECT_EQ(ring.available(1), 1U);

    EXPECT_EQ(ring.push_zeros(0, 3), 3U);
    std::array<T, 3> zeros{T(9), T(9), T(9)};
    EXPECT_EQ(ring.pop_block(0, zeros.data(), dtype, zeros.size()), 3U);
    EXPECT_EQ(zeros[0], T{});
    EXPECT_EQ(zeros[2], T{});

    // Another dtype moves nothing and writes nothing.
    const anira_dtype other = dtype == ANIRA_DTYPE_F32 ? ANIRA_DTYPE_I16 : ANIRA_DTYPE_F32;
    std::array<T, 2> untouched{T(42), T(42)};
    EXPECT_EQ(ring.pop_block(1, untouched.data(), other, untouched.size()), 0U);
    EXPECT_EQ(untouched[0], T(42));
    EXPECT_EQ(ring.push_block(1, in.data(), other, 2), 0U);
    EXPECT_EQ(ring.peek_past_block(1, untouched.data(), other, 1), 0U);
    EXPECT_EQ(ring.push_fill(1, &seven, other, 2), 0U);
    EXPECT_EQ(ring.pop_windows(1, untouched.data(), other, 1, 1, 0, 1), 0U);
    EXPECT_EQ(untouched[1], T(42));
    EXPECT_EQ(ring.available(1), 1U);

    ring.clear_with_positions();
    EXPECT_EQ(ring.dtype(), dtype) << "clear keeps the dtype";
    EXPECT_EQ(ring.capacity(), 8U) << "clear keeps the capacity";
    EXPECT_EQ(ring.available(1), 0U);
}

}  // namespace

TEST(RingBufferTyped, Float32) {
    round_trip<float>(ANIRA_DTYPE_F32);
}
TEST(RingBufferTyped, Float64) {
    round_trip<double>(ANIRA_DTYPE_F64);
}
TEST(RingBufferTyped, Float16IsStoredAsItsBits) {
    round_trip<uint16_t>(ANIRA_DTYPE_F16);
}
TEST(RingBufferTyped, BFloat16IsStoredAsItsBits) {
    round_trip<uint16_t>(ANIRA_DTYPE_BF16);
}
TEST(RingBufferTyped, Int8) {
    round_trip<int8_t>(ANIRA_DTYPE_I8);
}
TEST(RingBufferTyped, UInt8) {
    round_trip<uint8_t>(ANIRA_DTYPE_U8);
}
TEST(RingBufferTyped, Bool8) {
    round_trip<uint8_t>(ANIRA_DTYPE_BOOL8);
}
TEST(RingBufferTyped, Int16) {
    round_trip<int16_t>(ANIRA_DTYPE_I16);
}
TEST(RingBufferTyped, Int32) {
    round_trip<int32_t>(ANIRA_DTYPE_I32);
}
TEST(RingBufferTyped, Int64) {
    round_trip<int64_t>(ANIRA_DTYPE_I64);
}

TEST(RingBufferTyped, EveryDtypeIsItsOwnRing) {
    // uint8 and bool8 store the same bytes, as do float16 and bfloat16, but no two dtypes share
    // a ring: the ring reports the dtype it was initialised with and refuses the sibling.
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(1, 4, ANIRA_DTYPE_BOOL8));
    EXPECT_EQ(ring.dtype(), ANIRA_DTYPE_BOOL8);
    const uint8_t yes = 1;
    EXPECT_EQ(ring.push_fill(0, &yes, ANIRA_DTYPE_U8, 1), 0U);
    EXPECT_EQ(ring.push_fill(0, &yes, ANIRA_DTYPE_BOOL8, 1), 1U);
    ASSERT_TRUE(ring.initialize_with_positions(1, 4, ANIRA_DTYPE_BF16));
    EXPECT_EQ(ring.dtype(), ANIRA_DTYPE_BF16);
    const uint16_t bits = 0x3f80;  // 1.0 as bfloat16
    EXPECT_EQ(ring.push_fill(0, &bits, ANIRA_DTYPE_F16, 1), 0U);
    EXPECT_EQ(ring.push_fill(0, &bits, ANIRA_DTYPE_BF16, 1), 1U);
    uint16_t back = 0;
    EXPECT_EQ(ring.pop_block(0, &back, ANIRA_DTYPE_BF16, 1), 1U);
    EXPECT_EQ(back, bits) << "no arithmetic happens on a ring element";
}

TEST(RingBufferTyped, DtypesTheRingsCannotStoreAreRefusedAtInitialize) {
    anira::RingBuffer ring;
    ring.initialize_with_positions(1, 4);
    ring.push_sample(0, 1.0F);
    EXPECT_FALSE(ring.initialize_with_positions(1, 4, ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 32, 4)))
        << "lanes > 1";
    EXPECT_FALSE(ring.initialize_with_positions(1, 4, ANIRA_MAKE_DTYPE(ANIRA_DTYPE_OPAQUE, 64, 1)));
    EXPECT_FALSE(
        ring.initialize_with_positions(1, 4, ANIRA_MAKE_DTYPE(ANIRA_DTYPE_COMPLEX, 64, 1)));
    EXPECT_FALSE(ring.initialize_with_positions(1, 4, ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 128, 1)))
        << "a width no dtype has";
    EXPECT_EQ(ring.dtype(), ANIRA_DTYPE_F32) << "the ring is left as it was";
    EXPECT_EQ(ring.available(0), 1U);
    EXPECT_FLOAT_EQ(ring.pop_sample(0), 1.0F);
}

TEST(RingBufferTyped, TheFloatFaceIsTheFloat32Arm) {
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(1, 4, ANIRA_DTYPE_I16));
    const int16_t value = 3;
    EXPECT_EQ(ring.push_block(0, &value, ANIRA_DTYPE_I16, 1), 1U);

    // The 2.x float calls do nothing on an int16 ring; the counts are dtype-independent.
    ring.push_sample(0, 1.0F);
    EXPECT_EQ(ring.get_available_samples(0), 1U);
    EXPECT_FLOAT_EQ(ring.pop_sample(0), 0.0F);
    EXPECT_EQ(ring.get_available_samples(0), 1U) << "nothing was popped";
    std::array<float, 2> floats{5.0F, 5.0F};
    ring.pop_block(0, floats.data(), floats.size());
    EXPECT_FLOAT_EQ(floats[0], 5.0F) << "nothing was written";
    EXPECT_FLOAT_EQ(ring.get_future_sample(0, 0), 0.0F);
    EXPECT_FLOAT_EQ(ring.get_past_sample(0, 1), 0.0F);
    EXPECT_EQ(ring.get_num_channels(), 1U);
    EXPECT_EQ(ring.get_num_samples(), 4U);

    // The two-argument initialise is the float face: the ring is float32 again.
    ring.initialize_with_positions(1, 4);
    EXPECT_EQ(ring.dtype(), ANIRA_DTYPE_F32);
    ring.push_sample(0, 2.0F);
    ring.push_sample(0, 3.0F);
    EXPECT_FLOAT_EQ(ring.get_future_sample(0, 1), 3.0F);
    EXPECT_FLOAT_EQ(ring.pop_sample(0), 2.0F);
    EXPECT_FLOAT_EQ(ring.get_past_sample(0, 1), 2.0F);
}

TEST(RingBufferTyped, ADefaultConstructedRingIsAnEmptyFloat32Ring) {
    // std::vector<anira::RingBuffer> input(1) and the like rely on it.
    const anira::RingBuffer ring;
    EXPECT_EQ(ring.dtype(), ANIRA_DTYPE_F32);
    EXPECT_EQ(ring.capacity(), 0U);
    EXPECT_EQ(ring.num_channels(), 0U);
}
