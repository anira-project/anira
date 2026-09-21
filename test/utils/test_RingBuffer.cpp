#include <anira/abi/enums.h>
#include <anira/utils/RingBuffer.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

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
    EXPECT_EQ(ring.element_size(), sizeof(T));
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
    EXPECT_EQ(ring.element_size(), sizeof(float));
    EXPECT_EQ(ring.capacity(), 0U);
    EXPECT_EQ(ring.num_channels(), 0U);
}

// ---- The strided block calls: one channel of an interleaved block per call --------------------

namespace {

constexpr size_t k_strided_capacity = 8;

// Every channel its own sequence, and no value twice within what a ring can hold; below 100, so
// that it is exact in every one of the ten element types.
template <typename T>
T strided_value(size_t frame, size_t channel) {
    return static_cast<T>(((frame * 7) + (channel * 13) + 1) % 100);
}

// The strided calls of a ring of `dtype` (stored as `T`) against the per-sample calls of a
// reference instantiation of the same storage, for a block of `num_channels` interleaved
// channels: channel `c` is pushed from `block + c` and popped into `block + c`, stride
// `num_channels`. The rounds wrap the ring, fill it exactly, pop beyond what is available and
// push more than the capacity.
template <typename T>
void strided_differential(anira_dtype dtype, size_t num_channels) {
    SCOPED_TRACE(testing::Message() << "interleaved channels: " << num_channels);
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(num_channels, k_strided_capacity, dtype));
    RingBufferT<T> reference;
    reference.initialize_with_positions(num_channels, k_strided_capacity);

    struct Round {
        size_t m_push;
        size_t m_pop;
    };
    // available after each round: 2 (wrapped), 2 (filled exactly to 8 in between), 0 (the pop
    // asks for 2 more than the 5 available), 0 (11 pushed into 8: the tail survives).
    const std::array<Round, 4> rounds{{{5, 3}, {6, 6}, {3, 7}, {11, 8}}};
    const T sentinel = static_cast<T>(101);
    size_t frame_offset = 0;
    for (const Round& round : rounds) {
        SCOPED_TRACE(testing::Message() << "push " << round.m_push << ", pop " << round.m_pop);
        std::vector<T> block(round.m_push * num_channels);
        for (size_t frame = 0; frame < round.m_push; ++frame) {
            for (size_t channel = 0; channel < num_channels; ++channel) {
                block[(frame * num_channels) + channel] =
                    strided_value<T>(frame_offset + frame, channel);
            }
        }
        frame_offset += round.m_push;
        for (size_t channel = 0; channel < num_channels; ++channel) {
            EXPECT_EQ(
                ring.push_block(channel, block.data() + channel, dtype, round.m_push, num_channels),
                round.m_push);
            for (size_t frame = 0; frame < round.m_push; ++frame) {
                reference.push_sample(channel, block[(frame * num_channels) + channel]);
            }
            EXPECT_EQ(ring.available(channel), reference.get_available_samples(channel));
        }

        std::vector<T> popped(round.m_pop * num_channels, sentinel);
        std::vector<T> expected(round.m_pop * num_channels, sentinel);
        for (size_t channel = 0; channel < num_channels; ++channel) {
            EXPECT_EQ(
                ring.pop_block(channel, popped.data() + channel, dtype, round.m_pop, num_channels),
                round.m_pop);
            for (size_t frame = 0; frame < round.m_pop; ++frame) {
                // An empty channel pops a value-initialised element: the tail of a pop beyond
                // the available count.
                expected[(frame * num_channels) + channel] = reference.pop_sample(channel);
            }
            EXPECT_EQ(ring.available(channel), reference.get_available_samples(channel));
            EXPECT_EQ(ring.available_past(channel), reference.get_available_past_samples(channel));
        }
        EXPECT_EQ(popped, expected);

        // The history, oldest first, through the same stride.
        const size_t num_past = std::min<size_t>(round.m_pop, 4);
        std::vector<T> past(num_past * num_channels, sentinel);
        std::vector<T> expected_past(num_past * num_channels, sentinel);
        for (size_t channel = 0; channel < num_channels; ++channel) {
            EXPECT_EQ(
                ring.peek_past_block(channel, past.data() + channel, dtype, num_past, num_channels),
                num_past);
            for (size_t k = 0; k < num_past; ++k) {
                expected_past[(k * num_channels) + channel] =
                    reference.get_past_sample(channel, num_past - k);
            }
        }
        EXPECT_EQ(past, expected_past);
    }
}

// What a strided call leaves alone: the elements between the strided ones, and everything when
// the dtype is not the ring's own.
template <typename T>
void strided_writes_only_its_run(anira_dtype dtype) {
    constexpr size_t k_stride = 3;
    constexpr size_t k_frames = 4;
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(1, k_strided_capacity, dtype));
    std::array<T, k_frames * k_stride> block{};
    for (size_t i = 0; i < block.size(); ++i) { block[i] = static_cast<T>(i + 1); }
    const T sentinel = static_cast<T>(101);

    // Another dtype moves nothing and writes nothing, under a stride too.
    const anira_dtype other = dtype == ANIRA_DTYPE_F32 ? ANIRA_DTYPE_I16 : ANIRA_DTYPE_F32;
    EXPECT_EQ(ring.push_block(0, block.data(), other, k_frames, k_stride), 0U);
    EXPECT_EQ(ring.available(0), 0U) << "nothing was pushed";

    ASSERT_EQ(ring.push_block(0, block.data() + 1, dtype, k_frames, k_stride), k_frames);
    std::array<T, k_frames * k_stride> untouched{};
    untouched.fill(sentinel);
    EXPECT_EQ(ring.pop_block(0, untouched.data(), other, k_frames, k_stride), 0U);
    EXPECT_EQ(ring.peek_past_block(0, untouched.data(), other, k_frames, k_stride), 0U);
    EXPECT_EQ(ring.available(0), k_frames) << "nothing was popped";
    for (const T& element : untouched) { EXPECT_EQ(element, sentinel); }

    // The pop asks for two more than the available four: six strided elements are written, the
    // last two value-initialised, and no element between them.
    constexpr size_t k_popped = k_frames + 2;
    std::array<T, k_popped * k_stride> popped{};
    popped.fill(sentinel);
    EXPECT_EQ(ring.pop_block(0, popped.data() + 2, dtype, k_popped, k_stride), k_popped);
    for (size_t i = 0; i < popped.size(); ++i) {
        const size_t frame = i / k_stride;
        if (i % k_stride != 2) {
            EXPECT_EQ(popped[i], sentinel) << "element " << i << " is not part of the run";
        } else if (frame < k_frames) {
            EXPECT_EQ(popped[i], block[(frame * k_stride) + 1]) << "frame " << frame;
        } else {
            EXPECT_EQ(popped[i], T{}) << "frame " << frame << " is beyond the available count";
        }
    }

    std::array<T, k_frames * k_stride> past{};
    past.fill(sentinel);
    EXPECT_EQ(ring.peek_past_block(0, past.data(), dtype, k_frames, k_stride), k_frames);
    for (size_t i = 0; i < past.size(); ++i) {
        const size_t frame = i / k_stride;
        if (i % k_stride != 0) {
            EXPECT_EQ(past[i], sentinel) << "element " << i << " is not part of the run";
        } else {
            EXPECT_EQ(past[i], block[(frame * k_stride) + 1]) << "frame " << frame;
        }
    }
}

// A stride of 1, spelled or defaulted, is the unit-stride call of the storage.
template <typename T>
void unit_stride_is_the_unit_call(anira_dtype dtype) {
    anira::RingBuffer defaulted;
    anira::RingBuffer spelled;
    ASSERT_TRUE(defaulted.initialize_with_positions(1, k_strided_capacity, dtype));
    ASSERT_TRUE(spelled.initialize_with_positions(1, k_strided_capacity, dtype));
    RingBufferT<T> reference;
    reference.initialize_with_positions(1, k_strided_capacity);

    const T sentinel = static_cast<T>(101);
    size_t frame_offset = 0;
    // 6 of 8, then 11 (wraps, and more than the capacity), popped 5 and then 9 (beyond the 8
    // available).
    const std::array<std::array<size_t, 2>, 2> rounds{{{6, 5}, {11, 9}}};
    for (const auto& [num_push, num_pop] : rounds) {
        std::vector<T> block(num_push);
        for (size_t frame = 0; frame < num_push; ++frame) {
            block[frame] = strided_value<T>(frame_offset + frame, 0);
        }
        frame_offset += num_push;
        EXPECT_EQ(defaulted.push_block(0, block.data(), dtype, num_push), num_push);
        EXPECT_EQ(spelled.push_block(0, block.data(), dtype, num_push, 1), num_push);
        reference.push_block(0, block.data(), num_push);

        std::vector<T> from_defaulted(num_pop, sentinel);
        std::vector<T> from_spelled(num_pop, sentinel);
        std::vector<T> expected(num_pop, sentinel);
        EXPECT_EQ(defaulted.pop_block(0, from_defaulted.data(), dtype, num_pop), num_pop);
        EXPECT_EQ(spelled.pop_block(0, from_spelled.data(), dtype, num_pop, 1), num_pop);
        reference.pop_block(0, expected.data(), num_pop);
        EXPECT_EQ(from_defaulted, expected);
        EXPECT_EQ(from_spelled, expected);

        std::vector<T> past_defaulted(3, sentinel);
        std::vector<T> past_spelled(3, sentinel);
        std::vector<T> expected_past(3, sentinel);
        EXPECT_EQ(defaulted.peek_past_block(0, past_defaulted.data(), dtype, 3), 3U);
        EXPECT_EQ(spelled.peek_past_block(0, past_spelled.data(), dtype, 3, 1), 3U);
        reference.peek_past_block(0, expected_past.data(), 3);
        EXPECT_EQ(past_defaulted, expected_past);
        EXPECT_EQ(past_spelled, expected_past);
    }
}

template <typename T>
void strided(anira_dtype dtype) {
    for (const size_t num_channels : {size_t{1}, size_t{2}, size_t{3}, size_t{6}}) {
        strided_differential<T>(dtype, num_channels);
    }
    strided_writes_only_its_run<T>(dtype);
    unit_stride_is_the_unit_call<T>(dtype);
}

}  // namespace

TEST(RingBufferStrided, Float32) {
    strided<float>(ANIRA_DTYPE_F32);
}
TEST(RingBufferStrided, Float64) {
    strided<double>(ANIRA_DTYPE_F64);
}
TEST(RingBufferStrided, Float16IsStoredAsItsBits) {
    strided<uint16_t>(ANIRA_DTYPE_F16);
}
TEST(RingBufferStrided, BFloat16IsStoredAsItsBits) {
    strided<uint16_t>(ANIRA_DTYPE_BF16);
}
TEST(RingBufferStrided, Int8) {
    strided<int8_t>(ANIRA_DTYPE_I8);
}
TEST(RingBufferStrided, UInt8) {
    strided<uint8_t>(ANIRA_DTYPE_U8);
}
TEST(RingBufferStrided, Bool8) {
    strided<uint8_t>(ANIRA_DTYPE_BOOL8);
}
TEST(RingBufferStrided, Int16) {
    strided<int16_t>(ANIRA_DTYPE_I16);
}
TEST(RingBufferStrided, Int32) {
    strided<int32_t>(ANIRA_DTYPE_I32);
}
TEST(RingBufferStrided, Int64) {
    strided<int64_t>(ANIRA_DTYPE_I64);
}

// The float face is untouched by the stride: a float pointer with three arguments is still the
// 2.x call, never the typed one with a defaulted stride.
TEST(RingBufferStrided, TheFloatFaceKeepsItsThreeArguments) {
    anira::RingBuffer ring;
    ring.initialize_with_positions(1, 4);
    const std::array<float, 3> in{1.0F, 2.0F, 3.0F};
    ring.push_block(0, in.data(), in.size());
    EXPECT_EQ(ring.available(0), 3U);
    // Interleaved stereo, the right channel: 10 and 20.
    const std::array<float, 4> interleaved{-1.0F, 10.0F, -1.0F, 20.0F};
    EXPECT_EQ(ring.push_block(0, interleaved.data() + 1, ANIRA_DTYPE_F32, 2, 2), 2U);
    std::array<float, 4> out{};
    ring.pop_block(0, out.data(), out.size());
    EXPECT_FLOAT_EQ(out[0], 2.0F) << "5 into 4: the oldest element was overwritten";
    EXPECT_FLOAT_EQ(out[1], 3.0F);
    EXPECT_FLOAT_EQ(out[2], 10.0F);
    EXPECT_FLOAT_EQ(out[3], 20.0F);
}
