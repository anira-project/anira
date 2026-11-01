#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include "capi/validate.h"
#include "scheduler/SessionElement.h"

#if defined(ANIRA_WITH_ASAN) || defined(ANIRA_WITH_LSAN)
#include <sanitizer/lsan_interface.h>
#endif

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"
#include "../support/session_cases.h"
#include "gtest/gtest.h"

using namespace anira;
using anira_test::SizingHostBlocks;
using anira_test::SizingResult;
using anira_test::SizingSession;
using anira_test::SizingSlot;

// The sizing of a session at prepare, read off a C handler's session (test/support/
// session_cases.h): the latency per output, the struct pool and the ring sizes, for the
// host blocks and models below, and the ring dtypes; then the one-sided models. Three cases
// drive a SessionElement of their own, unregistered: clear() and prepare() on a thread that may
// hang, and the reference rule of the 2.x HostConfig.

namespace {
struct SessionElementTestParams {
    SizingHostBlocks m_host_blocks;
    std::vector<SizingSlot> m_inputs;
    std::vector<SizingSlot> m_outputs;
    double m_budget_ms = 0.0;
    double m_wait_ratio = 0.0;
    std::vector<unsigned int> m_expected_latency;
    size_t m_expected_num_structs = 0;
    std::vector<size_t> m_expected_send_buffer_sizes;
    std::vector<size_t> m_expected_receive_buffer_sizes;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[ ";
    for (const auto& item : vec) { os << item << " "; }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& stream, const SessionElementTestParams& params) {
    stream << "{ block = " << params.m_host_blocks.m_block
           << ", host_rate = " << params.m_host_blocks.m_host_rate
           << ", smaller = " << params.m_host_blocks.m_allow_smaller
           << ", budget = " << params.m_budget_ms << " ms, wait ratio = " << params.m_wait_ratio
           << ", expected latency = " << params.m_expected_latency
           << ", expected num_structs = " << params.m_expected_num_structs
           << ", expected send buffer sizes = " << params.m_expected_send_buffer_sizes
           << ", expected receive buffer sizes = " << params.m_expected_receive_buffer_sizes
           << " }";
    return stream;
}

class SessionElementTest : public ::testing::TestWithParam<SessionElementTestParams> {};

/// A case prepared and sized; a failed prepare fails the test.
SizingResult prepare_and_measure(const anira::ModelConfig& model,
                                 const SizingHostBlocks& blocks,
                                 double budget_ms = 10.0,
                                 double wait_ratio = 0.0) {
    const SizingSession session(model, anira_test::sizing_contract(blocks, budget_ms, wait_ratio));
    EXPECT_EQ(session.prepared(), ANIRA_OK) << session.message();
    return session.measure();
}
}  // namespace

TEST_P(SessionElementTest, LatencyStructAndRingbuffers) {
    const SessionElementTestParams& params = GetParam();
    const SizingResult result =
        prepare_and_measure(anira_test::sizing_model(params.m_inputs,
                                                     params.m_outputs,
                                                     2,
                                                     params.m_host_blocks.m_anchor),
                            params.m_host_blocks,
                            params.m_budget_ms,
                            params.m_wait_ratio);
    ASSERT_EQ(result.m_latency.size(), params.m_expected_latency.size());
    for (size_t i = 0; i < params.m_expected_latency.size(); ++i) {
        ASSERT_EQ(result.m_latency[i], params.m_expected_latency[i])
            << "Latency mismatch at index " << i;
    }
    ASSERT_EQ(result.m_num_structs, params.m_expected_num_structs) << "Number of structs mismatch";
    for (size_t i = 0; i < params.m_expected_send_buffer_sizes.size(); ++i) {
        ASSERT_EQ(result.m_send[i], params.m_expected_send_buffer_sizes[i])
            << "Send buffer size mismatch at index " << i;
    }
    for (size_t i = 0; i < params.m_expected_receive_buffer_sizes.size(); ++i) {
        ASSERT_EQ(result.m_receive[i], params.m_expected_receive_buffer_sizes[i])
            << "Receive buffer size mismatch at index " << i;
    }
}

namespace {
// The processor of the ring dtype case: it fills the two ring-moving phases (with the default
// bodies), so the validator lets a ring hold another dtype than its float32 spec. No inference
// runs.
anira_status ANIRA_CALL default_pre_process(const anira_processor_ctx* ctx,
                                            void* /*prepared*/,
                                            void* /*user_data*/) noexcept {
    return anira_processor_default_pre_process(ctx);
}

anira_status ANIRA_CALL default_post_process(const anira_processor_ctx* ctx,
                                             void* /*prepared*/,
                                             void* /*user_data*/) noexcept {
    return anira_processor_default_post_process(ctx);
}

anira_processor_desc ring_moving_processor() {
    anira_processor_desc processor = ANIRA_PROCESSOR_DESC_INIT;
    processor.flags = ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST;  // what the default bodies are
    processor.pre_process = default_pre_process;
    processor.post_process = default_post_process;
    return processor;
}

/// The mono 512 pass-through with the rings of `in` and `out` in the dtypes given, prepared
/// with the ring-moving processor; the status of the prepare in `status`.
struct RingDtypeSession {
    RingDtypeSession(anira_dtype in, anira_dtype out) {
        anira::ContractHandle contract = anira_test::explicit_contract(512,
                                                                       anira_test::k_rate,
                                                                       ANIRA_MISS_ZEROS,
                                                                       0.0,
                                                                       /*budget_ms=*/5.0);
        contract.hard_ring_dtype("in", in).hard_ring_dtype("out", out);
        m_status = m_handler.prepare(contract);
    }

    anira::SessionElement& session() const { return *anira_test::session_of(m_handler.m_handler); }

    anira_test::Context m_context{2};
    std::vector<anira_processor_desc> m_processors{ring_moving_processor()};
    anira_test::Handler m_handler{m_context, anira_test::mono_stream_model(512), {}, m_processors};
    anira_status m_status = ANIRA_ERROR_INTERNAL;
};
}  // namespace

// The rings store the element type of their slot's ring dtype: a slot named int16 gets an int16
// ring of the same element count as the float run (the sizes are element counts), a slot not
// named stays float32, the latency priming is zeros of the ring's own type, and a dtype the
// rings cannot store is refused like an unresolvable reference stream.
TEST(SessionElementRingDtypes, ASlotsRingHoldsItsRingDtype) {
    const RingDtypeSession floats(ANIRA_DTYPE_F32, ANIRA_DTYPE_F32);
    ASSERT_EQ(floats.m_status, ANIRA_OK) << floats.m_handler.m_err.message;
    ASSERT_EQ(floats.session().send_ring(0).dtype(), ANIRA_DTYPE_F32);
    ASSERT_EQ(floats.session().receive_ring(0).dtype(), ANIRA_DTYPE_F32);

    const RingDtypeSession int16_in(ANIRA_DTYPE_I16, ANIRA_DTYPE_F32);
    ASSERT_EQ(int16_in.m_status, ANIRA_OK) << int16_in.m_handler.m_err.message;
    EXPECT_EQ(int16_in.session().send_ring(0).dtype(), ANIRA_DTYPE_I16);
    EXPECT_EQ(int16_in.session().receive_ring(0).dtype(), ANIRA_DTYPE_F32) << "not named: float32";
    EXPECT_EQ(int16_in.session().m_send_buffer_size, floats.session().m_send_buffer_size);
    EXPECT_EQ(int16_in.session().m_receive_buffer_size, floats.session().m_receive_buffer_size);
    EXPECT_EQ(int16_in.session().send_ring(0).capacity(), floats.session().send_ring(0).capacity());
    EXPECT_EQ(int16_in.session().send_ring(0).num_channels(),
              floats.session().send_ring(0).num_channels());

    const RingDtypeSession int16_out(ANIRA_DTYPE_F32, ANIRA_DTYPE_I16);
    ASSERT_EQ(int16_out.m_status, ANIRA_OK) << int16_out.m_handler.m_err.message;
    EXPECT_EQ(int16_out.session().receive_ring(0).dtype(), ANIRA_DTYPE_I16);
    EXPECT_EQ(int16_out.session().receive_ring(0).capacity(),
              floats.session().receive_ring(0).capacity());
    EXPECT_EQ(int16_out.session().receive_ring(0).available(0),
              floats.session().receive_ring(0).available(0))
        << "the latency priming is the same element count, typed";
    int16_t primed = 7;
    ASSERT_EQ(int16_out.session().receive_ring(0).pop_block(0, &primed, ANIRA_DTYPE_I16, 1), 1U);
    EXPECT_EQ(primed, 0);

    const RingDtypeSession lanes(ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 32, 4), ANIRA_DTYPE_F32);
    EXPECT_EQ(lanes.m_status, ANIRA_ERROR_CONFIG) << lanes.m_handler.m_err.message;
}

namespace {
/// A slot's buffer as an address.
uintptr_t address_of(const anira_tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.handle.host.ptr);
}
}  // namespace

// A chunk stores every slot in its spec's dtype: the descriptor names the spec's dtype, the
// buffer holds the elements of that dtype, every buffer starts on a 64-byte boundary, and an
// all-float32 chunk keeps the sizes and the offsets it always had (elements * 4, each padded to
// 64), so a float32 model's chunks are byte for byte what they were.
TEST(SessionElementChunk, ASlotHoldsItsSpecsDtype) {
    using capi::DerivedSpec;
    const std::vector<DerivedSpec> inputs{
        DerivedSpec{.m_dims = {1, 1, 512}, .m_dtype = ANIRA_DTYPE_I16},
        DerivedSpec{.m_dims = {1}, .m_dtype = ANIRA_DTYPE_F32},
        DerivedSpec{.m_dims = {1, 3}, .m_dtype = ANIRA_DTYPE_I8},
    };
    const std::vector<DerivedSpec> outputs{
        DerivedSpec{.m_dims = {1, 1, 512}, .m_dtype = ANIRA_DTYPE_F16},
        DerivedSpec{.m_dims = {1}, .m_dtype = ANIRA_DTYPE_I64},
    };
    const SessionElement::ThreadSafeStruct chunk(inputs, outputs, {}, {});
    ASSERT_EQ(chunk.m_input_tensors.size(), 3U);
    ASSERT_EQ(chunk.m_output_tensors.size(), 2U);
    const std::array<anira_dtype, 3> in_dtypes{ANIRA_DTYPE_I16, ANIRA_DTYPE_F32, ANIRA_DTYPE_I8};
    const std::array<anira_dtype, 2> out_dtypes{ANIRA_DTYPE_F16, ANIRA_DTYPE_I64};
    for (size_t slot = 0; slot < in_dtypes.size(); ++slot) {
        EXPECT_EQ(chunk.m_input_tensors[slot].dtype, in_dtypes[slot]) << "input " << slot;
        EXPECT_EQ(address_of(chunk.m_input_tensors[slot]) % 64, 0U) << "input " << slot;
    }
    for (size_t slot = 0; slot < out_dtypes.size(); ++slot) {
        EXPECT_EQ(chunk.m_output_tensors[slot].dtype, out_dtypes[slot]) << "output " << slot;
        EXPECT_EQ(address_of(chunk.m_output_tensors[slot]) % 64, 0U) << "output " << slot;
    }
    EXPECT_EQ(anira_tensor_num_elements(&chunk.m_input_tensors[0]), 512U);
    // 512 int16 elements are 1024 bytes; a float32 slot would have taken 2048.
    EXPECT_EQ(address_of(chunk.m_input_tensors[1]) - address_of(chunk.m_input_tensors[0]), 1024U);
    // One float32 element padded to 64 bytes.
    EXPECT_EQ(address_of(chunk.m_input_tensors[2]) - address_of(chunk.m_input_tensors[1]), 64U);
    // 3 int8 elements padded to 64, then the first output: 512 float16 elements, 1024 bytes.
    EXPECT_EQ(address_of(chunk.m_output_tensors[0]) - address_of(chunk.m_input_tensors[2]), 64U);
    EXPECT_EQ(address_of(chunk.m_output_tensors[1]) - address_of(chunk.m_output_tensors[0]), 1024U);
    auto* last = static_cast<int64_t*>(chunk.m_output_tensors[1].handle.host.ptr);
    EXPECT_EQ(*last, 0) << "the buffers start at zeros";

    const std::vector<DerivedSpec> floats{
        DerivedSpec{.m_dims = {1, 1, 512}},
        DerivedSpec{.m_dims = {1}},
    };
    const SessionElement::ThreadSafeStruct float_chunk(floats, floats, {}, {});
    EXPECT_EQ(float_chunk.m_input_tensors[0].dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(
        address_of(float_chunk.m_input_tensors[1]) - address_of(float_chunk.m_input_tensors[0]),
        2048U);
    EXPECT_EQ(
        address_of(float_chunk.m_output_tensors[0]) - address_of(float_chunk.m_input_tensors[1]),
        64U);
}

namespace {
std::string build_test_name(const testing::TestParamInfo<SessionElementTest::ParamType>& info) {
    return "case_" + std::to_string(info.index);
}

// A one-tensor-per-side model of `elements` samples in and out, mono, every sample a hop.
std::vector<SizingSlot> mono(int64_t elements) {
    return {SizingSlot{.m_elements = elements}};
}

SessionElementTestParams row(SizingHostBlocks blocks,
                             std::vector<SizingSlot> inputs,
                             std::vector<SizingSlot> outputs,
                             double budget_ms,
                             double wait_ratio,
                             std::vector<unsigned int> latency,
                             size_t num_structs,
                             std::vector<size_t> send,
                             std::vector<size_t> receive) {
    return SessionElementTestParams{.m_host_blocks = blocks,
                                    .m_inputs = std::move(inputs),
                                    .m_outputs = std::move(outputs),
                                    .m_budget_ms = budget_ms,
                                    .m_wait_ratio = wait_ratio,
                                    .m_expected_latency = std::move(latency),
                                    .m_expected_num_structs = num_structs,
                                    .m_expected_send_buffer_sizes = std::move(send),
                                    .m_expected_receive_buffer_sizes = std::move(receive)};
}

/// Host blocks anchored on input `index`.
SizingHostBlocks anchored(double block, double host_rate, bool smaller, size_t index) {
    return SizingHostBlocks{
        .m_block = block,
        .m_host_rate = host_rate,
        .m_allow_smaller = smaller,
        .m_anchor = anira_test::SizingAnchor{.m_is_input = true, .m_index = index}};
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    LatencyStructAndRingbuffers,
    SessionElementTest,
    ::testing::Values(
        // Basic test cases similar to InferenceManager tests
        row({2048, 48000}, mono(2048), mono(2048), 40.0, 0.0, {2048}, 4, {2048}, {10240}),
        // collected inside the blocking wait: one slot in flight
        row({2048, 48000}, mono(2048), mono(2048), 17.0, 0.5, {0}, 2, {2048}, {4096}),
        // send buffer: one block plus hop - 1
        row({2048, 48000, true}, mono(2048), mono(2048), 20.0, 0.5, {3966}, 4, {4095}, {12158}),
        row({2048, 48000, true}, mono(2048), mono(2048), 10.0, 0.5, {3006}, 2, {4095}, {7102}),
        row({2048, 48000, true}, mono(2048), mono(2048), 19.0, 0.0, {4095}, 4, {4095}, {12287}),
        row({1, 48000.0 / 2048, true}, mono(1), mono(2048), 20.0, 0.0, {4095}, 4, {1}, {12287}),
        row({1, 48000.0 / 2048}, mono(1), mono(2048), 50.0, 0.0, {4096}, 6, {1}, {16384}),
        row({1, 48000.0 / 2048, true}, mono(1), mono(2048), 51.0, 0.0, {6143}, 6, {1}, {18431}),
        row({256, 48000.0},
            mono(2048),
            {{.m_elements = 4, .m_channels = 4}},
            40.0,
            0.0,
            {1},
            4,
            {2048},
            {5}),
        row({1. / 256., 48000. / 2048., true},
            {{.m_elements = 4, .m_channels = 4}},
            mono(2048),
            40.0,
            0.0,
            {3972},
            2,
            {1},
            {8068}),
        row({1., 48000. / 2048.},
            {{.m_elements = 16, .m_channels = 16}},
            {{.m_elements = 2048}, {.m_elements = 512, .m_channels = 2}},
            40.0,
            0.0,
            {2048, 256},
            4,
            {1},
            {10240, 1280}),
        row(anchored(256., 48000. / 8, true, 1),
            {{.m_elements = 16, .m_channels = 16}, {.m_elements = 512, .m_channels = 2}},
            {{.m_elements = 2048}, {.m_elements = 384, .m_channels = 3}},
            5.0,
            0.0,
            {4096, 256},
            4,
            {1, 511},
            {12288, 768}),
        row(anchored(600., 48000. / 8, false, 1),
            {{.m_elements = 16, .m_channels = 16}, {.m_elements = 512, .m_channels = 2}},
            {{.m_elements = 2048}, {.m_elements = 384, .m_channels = 3}},
            50.0,
            0.0,
            {8192, 512},
            10,
            {3, 848},
            {28672, 1792}),
        // Non-power-of-two buffer size tests
        row({100, 48000, true}, mono(2048), mono(2048), 13.0, 0.0, {2759}, 2, {2147}, {6855}),
        row({300, 44100}, mono(1024), mono(1024), 40.0, 0.0, {2820}, 6, {1320}, {8964}),
        row({2.5, 48000. / 2048., true},
            {{.m_elements = 8, .m_channels = 8}},
            mono(1024),
            12.0,
            0.0,
            {3583},
            10,
            {3},
            {13823}),
        row({2.5, 48000. / 1024., true},
            {{.m_elements = 8, .m_channels = 8}},
            mono(1024),
            4.0,
            0.5,
            {1406},
            6,
            {3},
            {7550}),
        // A window of 10000 samples moving 2048 per inference.
        row({2048, 48000, true},
            {{.m_elements = 10000, .m_hop = 2048}},
            mono(2048),
            21.0,
            0.0,
            {4095},
            4,
            {12047},
            {12287}),
        // Edge cases with very small buffer sizes. 30 ms of inference per 11.6 ms hop on
        // two processors exceeds the pool's capacity (LatencyCalculator::is_feasible);
        // the values size one host block on an idle pool.
        row({1, 44100, true}, mono(512), mono(512), 30.0, 0.0, {1834}, 6, {512}, {4906}),
        // Test with large buffer sizes
        row({4096, 96000}, mono(1024), mono(1024), 20.0, 0.0, {4096}, 16, {4096}, {20480}),
        // Test with very short inference times
        row({512, 48000, true}, mono(256), mono(256), 1.0, 0.0, {767}, 8, {767}, {2815}),
        // Test with very long inference times: 100 ms per 5.3 ms hop on two processors is
        // far beyond the pool's capacity; sized for one host block on an idle pool.
        row({512, 48000}, mono(256), mono(256), 100.0, 0.0, {5120}, 44, {512}, {16384})),
    build_test_name);

namespace {
// The two fixtures below outlive their tests on purpose (a detached thread may still be
// inside the code under test). Tell LeakSanitizer so, or it fails the binary at exit
// over a leak the tests intend.
template <typename T>
T* leak_intentionally(T* p) {
#if defined(ANIRA_WITH_ASAN) || defined(ANIRA_WITH_LSAN)
    __lsan_ignore_object(p);
#endif
    return p;
}
}  // namespace

// =============================================================================
// Regression: clear() must drain the done-semaphores without blocking
// =============================================================================

namespace {
// Everything the detached worker thread touches lives here, heap-allocated: if clear()
// deadlocks (the bug under test), the thread stays blocked inside it forever and must not
// reference destroyed objects while the remaining tests run, so a failing test leaks the
// fixture (process teardown reclaims it); a passing one destroys it. A C handler over a
// 2048-sample stream with a wait ratio, so that clear() takes the semaphore branch.
struct ClearDeadlockFixture {
    anira_test::SizingSession m_handler{
        anira_test::sizing_model({{.m_elements = 2048}}, {{.m_elements = 2048}}, 2),
        anira_test::sizing_contract(anira_test::sizing_host_blocks(2048, 48000), 40.0, 0.5)};
    std::atomic<bool> m_cleared{false};
};
}  // namespace

TEST(SessionElementClearTest, ClearWithBlockingRatioDoesNotFreeze) {
    auto fixture = std::make_unique<ClearDeadlockFixture>();
    ASSERT_EQ(fixture->m_handler.prepared(), ANIRA_OK) << fixture->m_handler.message();
    SessionElement* const session = anira_test::session_of(fixture->m_handler.handler()).get();
    ASSERT_NE(session, nullptr);

    // Mixed signal state: one struct carries a stale unconsumed completion
    // signal, the others none. clear() must consume what is there and never
    // wait — a blind acquire() on a count-0 semaphore blocks forever, since
    // the callers' drain_inference_queue() guarantees no inference thread will
    // signal this session again.
    session->m_inference_queue[0]->m_done_semaphore.release();

    ClearDeadlockFixture* const raw = fixture.get();
    std::thread([raw, session] {
        session->clear();
        raw->m_cleared.store(true, std::memory_order_release);
    }).detach();

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fixture->m_cleared.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!fixture->m_cleared.load(std::memory_order_acquire)) {
        leak_intentionally(fixture.release());  // the thread may still be inside clear()
        FAIL() << "SessionElement::clear() deadlocked draining the done-semaphores "
                  "(blind acquire on a semaphore with count 0).";
    }
}

// =============================================================================
// One-sided streaming regression tests (redo of the reverted PR #101, see #110)
//
// A generator has no streamable input (its inputs are all control parameters);
// an analyser has no streamable output (its results leave as non-streamable
// values). Both were broken at prepare() time: the generator hung / divided by
// zero on the reference input's size, the analyser segfaulted in
// sync_latencies() on an empty adjusted-latency vector. The reference stream is
// now first-class (an input or an output), so both prepare() correctly.
//
// Expected values are derived by comparison with an equivalent two-sided model
// prepared by the same code -- never hard-coded numbers -- so they are immune to
// float vs double rounding.
// =============================================================================

namespace {
// Generator: 4 control parameters in, a 2048-sample audio stream out.
anira::ModelConfig generator_model() {
    return anira_test::sizing_model({{.m_elements = 4, .m_hop = 0}}, {{.m_elements = 2048}}, 2);
}

// The generator's two-sided twin: a 2048-sample stream in and out. Same reference
// size (2048), so the driving-side inference count and the output side match.
anira::ModelConfig generator_twin_model(
    const std::optional<anira_test::SizingAnchor>& anchor = std::nullopt) {
    return anira_test::sizing_model({{.m_elements = 2048}}, {{.m_elements = 2048}}, 2, anchor);
}

// Analyser: a 2048-sample audio stream in plus one control parameter in, one
// non-streamable scalar out.
anira::ModelConfig analyser_model() {
    return anira_test::sizing_model({{.m_elements = 2048}, {.m_elements = 1, .m_hop = 0}},
                                    {{.m_elements = 1, .m_hop = 0}},
                                    2);
}

// The analyser's two-sided twin: identical input side, a 2048-sample streamable
// output instead of the scalar.
anira::ModelConfig analyser_twin_model() {
    return anira_test::sizing_model({{.m_elements = 2048}, {.m_elements = 1, .m_hop = 0}},
                                    {{.m_elements = 2048}},
                                    2);
}

SizingHostBlocks host_blocks(double block, double host_rate, bool smaller) {
    return SizingHostBlocks{.m_block = block, .m_host_rate = host_rate, .m_allow_smaller = smaller};
}

class OneSidedStreamingPrepareTest : public ::testing::TestWithParam<bool> {};
}  // namespace

TEST_P(OneSidedStreamingPrepareTest, GeneratorEqualsTwoSidedTwin) {
    bool const smaller = GetParam();
    SizingResult const gen =
        prepare_and_measure(generator_model(), host_blocks(512, 48000, smaller));
    SizingResult const twin =
        prepare_and_measure(generator_twin_model(), host_blocks(512, 48000, smaller));

    EXPECT_EQ(gen.m_latency, twin.m_latency);
    EXPECT_EQ(gen.m_num_structs, twin.m_num_structs);
    EXPECT_EQ(gen.m_receive, twin.m_receive);
    ASSERT_EQ(gen.m_send.size(), 1u);
    EXPECT_EQ(gen.m_send[0], 0u) << "A generator has no streamable input, so no send ring.";
    ASSERT_EQ(gen.m_latency.size(), 1u);
    EXPECT_GT(gen.m_latency[0], 0u) << "The streamable output still has a latency.";
}

TEST_P(OneSidedStreamingPrepareTest, AnalyserHasNoStreamLatency) {
    bool const smaller = GetParam();
    SizingResult const ana =
        prepare_and_measure(analyser_model(), host_blocks(512, 48000, smaller));
    SizingResult const twin =
        prepare_and_measure(analyser_twin_model(), host_blocks(512, 48000, smaller));

    ASSERT_EQ(ana.m_latency.size(), 1u);
    EXPECT_EQ(ana.m_latency[0], 0u) << "A non-streamable output carries no stream latency.";
    ASSERT_EQ(ana.m_receive.size(), 1u);
    EXPECT_EQ(ana.m_receive[0], 0u) << "A non-streamable output has no receive ring.";
    EXPECT_EQ(ana.m_send, twin.m_send) << "The input side is identical to the twin.";
    EXPECT_EQ(ana.m_num_structs, twin.m_num_structs);
}

// Issue #98: with allow_smaller_buffers the adjusted-latency pass collected entries for
// streamable outputs only and then indexed the synced vector by output tensor index --
// an empty vector when every output is non-streamable (crash in sync_latencies()), a
// short one for a mixed output list (out-of-bounds read into m_latency). Both passes
// now build one entry per output tensor. The models are the issue's (Silero-VAD
// shaped: probability + recurrent-state outputs); the oracle for the mixed case is the
// non-adjusted pass, which always handled non-streamable outputs correctly.
TEST_P(OneSidedStreamingPrepareTest, NonStreamableOutputsKeepLatencyIndexAligned) {
    bool const smaller = GetParam();
    // 576 samples moving 512 per inference beside a [2, 1, 128] state.
    const std::vector<SizingSlot> inputs{{.m_elements = 576, .m_hop = 512},
                                         {.m_elements = 256, .m_hop = 0}};
    SizingHostBlocks const host = host_blocks(512, 16000, smaller);

    // Case 1: every output non-streamable.
    SizingResult const all_static = prepare_and_measure(
        anira_test::sizing_model(inputs,
                                 {{.m_elements = 1, .m_hop = 0}, {.m_elements = 256, .m_hop = 0}},
                                 2),
        host);
    ASSERT_EQ(all_static.m_latency.size(), 2u);
    EXPECT_EQ(all_static.m_latency, (std::vector<unsigned int>{0, 0}))
        << "Non-streamable outputs carry no stream latency.";
    EXPECT_EQ(all_static.m_receive, (std::vector<size_t>{0, 0}))
        << "Non-streamable outputs have no receive rings.";

    // Case 2: a streamable result plus a non-streamable state tensor.
    anira::ModelConfig const mixed =
        anira_test::sizing_model(inputs,
                                 {{.m_elements = 1, .m_hop = 1}, {.m_elements = 256, .m_hop = 0}},
                                 2);
    SizingResult const adjusted = prepare_and_measure(mixed, host);
    SizingResult const plain = prepare_and_measure(mixed, host_blocks(512, 16000, false));
    ASSERT_EQ(adjusted.m_latency.size(), 2u);
    ASSERT_EQ(plain.m_latency.size(), 2u);
    EXPECT_EQ(adjusted.m_latency[1], 0u) << "The state tensor keeps a latency of 0.";
    EXPECT_EQ(adjusted.m_receive[1], 0u);
    EXPECT_GE(adjusted.m_latency[0], plain.m_latency[0])
        << "The smaller-buffer pass may only raise the streamed output's latency.";
    EXPECT_LE(adjusted.m_latency[0], plain.m_latency[0] + 512u)
        << "and by at most one host block -- anything else is an out-of-bounds read.";
    if (!smaller) { EXPECT_EQ(adjusted.m_latency, plain.m_latency); }
}

INSTANTIATE_TEST_SUITE_P(OneSidedStreaming,
                         OneSidedStreamingPrepareTest,
                         ::testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "allow_smaller_buffers" : "static_buffer";
                         });

// The generator hang (#101): with allow_smaller_buffers the buffer-ratio countdown
// never terminated. Run the prepare on a detached thread against a deadline. A failing test
// leaks the fixture -- if prepare ever hangs again the detached thread must not touch a
// destroyed fixture (same pattern as SessionElementClearTest); a passing one destroys it.
namespace {
struct GeneratorPrepareFixture {
    std::optional<anira_test::SizingSession> m_handler;
    std::atomic<bool> m_prepared{false};
};
}  // namespace

TEST(OneSidedStreamingPrepareStandalone, GeneratorPrepareTerminates) {
    auto fixture = std::make_unique<GeneratorPrepareFixture>();
    GeneratorPrepareFixture* const raw = fixture.get();
    std::thread([raw] {
        raw->m_handler.emplace(
            generator_model(),
            anira_test::sizing_contract(host_blocks(512, 48000, true), 10.0, 0.0));
        raw->m_prepared.store(true, std::memory_order_release);
    }).detach();

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fixture->m_prepared.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!fixture->m_prepared.load(std::memory_order_acquire)) {
        leak_intentionally(fixture.release());  // the thread may still be inside prepare
        FAIL() << "prepare did not terminate for a generator with smaller blocks allowed (the "
                  "buffer-ratio countdown divided by the non-streamable reference input's size "
                  "0).";
    }
    ASSERT_TRUE(fixture->m_handler.has_value());
    if (fixture->m_handler.has_value()) {
        EXPECT_EQ(fixture->m_handler->prepared(), ANIRA_OK) << fixture->m_handler->message();
    }
}

TEST(OneSidedStreamingPrepareStandalone, ExplicitOutputReferenceEqualsInputReference) {
    // A two-sided model with equal input/output stream sizes: anchoring input 0 or output 0
    // gives the same ratios and hence the same result.
    SizingResult const by_input =
        prepare_and_measure(generator_twin_model(), host_blocks(2048, 48000, true));
    SizingResult const by_output = prepare_and_measure(
        generator_twin_model(anira_test::SizingAnchor{.m_is_input = false, .m_index = 0}),
        host_blocks(2048, 48000, true));

    EXPECT_EQ(by_input.m_latency, by_output.m_latency);
    EXPECT_EQ(by_input.m_num_structs, by_output.m_num_structs);
    EXPECT_EQ(by_input.m_send, by_output.m_send);
    EXPECT_EQ(by_input.m_receive, by_output.m_receive);
}

namespace {
// The status of a handler's create over the model, beside the pass-through on the custom row:
// an anchor the model config cannot have is refused there (ANIRA_ERROR_CONFIG), before any
// prepare.
anira_status create_status(const anira::ModelConfig& model) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    anira_status status = anira_pipeline_create(&pipeline, &err);
    if (status != ANIRA_OK) { return status; }
    anira_test::add_passthrough(pipeline);
    const std::array<const anira_model_config*, 1> variants{model.native()};
    status = anira_pipeline_add_inference(pipeline, variants.data(), 1, nullptr, 0, &err);
    if (status == ANIRA_OK) {
        const anira_test::Context context(2);
        anira_handler* handler = nullptr;
        status = anira_handler_create(context.m_context, pipeline, &handler, &err);
        anira_handler_destroy(handler);
    }
    anira_pipeline_destroy(pipeline);
    return status;
}

anira::ModelConfig anchored_at(anira::ModelConfig model, const std::string& name) {
    model.anchor(name);
    return model;
}
}  // namespace

TEST(OneSidedStreamingPrepareStandalone, InvalidReferenceIsRefused) {
    // An explicit non-streamable input anchor on a generator.
    EXPECT_EQ(create_status(anchored_at(generator_model(), "in0")), ANIRA_ERROR_CONFIG);
    // An explicit non-streamable output anchor on an analyser.
    EXPECT_EQ(create_status(anchored_at(analyser_model(), "out0")), ANIRA_ERROR_CONFIG);
    // A name that is no tensor, both directions.
    EXPECT_EQ(create_status(anchored_at(generator_twin_model(), "in5")), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(create_status(anchored_at(generator_twin_model(), "out5")), ANIRA_ERROR_CONFIG);
    // No streamable tensor anywhere.
    EXPECT_EQ(create_status(anira_test::sizing_model({{.m_elements = 4, .m_hop = 0}},
                                                     {{.m_elements = 1, .m_hop = 0}},
                                                     2)),
              ANIRA_ERROR_CONFIG);
}
