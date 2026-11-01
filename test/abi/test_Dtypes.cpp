// Model tensors of any dtype anira names, end to end through a registered engine (the
// pass-through of handler_support.h, which copies each input slot to the output slot of the
// same dtype): the chunk stores each tensor in its spec's dtype, a Streamed tensor's ring
// streams the spec's dtype unless the contract names another, a Static tensor travels in its
// spec's dtype, and nothing in anira converts. A host of another sample type names its type in
// the contract's ring dtypes and converts in its own processor; a host tensor of another dtype
// than the ring's is refused per call and leaves the stream untouched.
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "capi/words.h"
#include "handler_support.h"
#include "scheduler/SessionElement.h"

namespace {

using anira::ContractHandle;
using anira::ModelConfig;
using anira::Tensor;
using anira::TensorSpec;
using anira_test::k_block;
using anira_test::k_rate;

constexpr size_t k_blocks = 4;

/// A mono Streamed tensor [Time] of `dtype` that moves one host block per inference.
TensorSpec stream(std::string_view name, anira_dtype dtype) {
    TensorSpec spec(name, dtype, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_TIME, static_cast<int64_t>(k_block))
        .window(static_cast<int64_t>(k_block), static_cast<int64_t>(k_block), 0);
    return spec;
}

/// One value of `dtype` for the integer `i`, as the element's bytes: distinct small values,
/// a normal number for the float types, 0 or 1 for bool.
void put(std::byte* at, anira_dtype dtype, size_t i) {
    const auto n = static_cast<int64_t>(i);
    const auto store = [at](const auto value) { std::memcpy(at, &value, sizeof(value)); };
    switch (dtype) {
        case ANIRA_DTYPE_F32: store(static_cast<float>(n % 1000)); break;
        case ANIRA_DTYPE_F64: store(static_cast<double>(n % 1000) / 8.0); break;
        case ANIRA_DTYPE_F16: store(static_cast<uint16_t>(0x3c00 + (n % 1024))); break;
        case ANIRA_DTYPE_BF16: store(static_cast<uint16_t>(0x3f80 + (n % 128))); break;
        case ANIRA_DTYPE_I8: store(static_cast<int8_t>((n % 200) - 100)); break;
        case ANIRA_DTYPE_U8: store(static_cast<uint8_t>(n % 256)); break;
        case ANIRA_DTYPE_I16: store(static_cast<int16_t>((n % 60000) - 30000)); break;
        case ANIRA_DTYPE_I32: store(static_cast<int32_t>((n * 3) - 7)); break;
        case ANIRA_DTYPE_I64: store(static_cast<int64_t>((n * 5) - 11)); break;
        case ANIRA_DTYPE_BOOL8: store(static_cast<uint8_t>(n % 2)); break;
        default: FAIL() << "no value for dtype " << dtype;
    }
}

size_t size_of(anira_dtype dtype) {
    return static_cast<size_t>(ANIRA_DTYPE_BITS(dtype)) * ANIRA_DTYPE_LANES(dtype) / 8;
}

/// Block `k` of the typed stream.
std::vector<std::byte> block_of(anira_dtype dtype, size_t k) {
    std::vector<std::byte> bytes(k_block * size_of(dtype));
    for (size_t n = 0; n < k_block; ++n) {
        put(&bytes[n * size_of(dtype)], dtype, (k * k_block) + n);
    }
    return bytes;
}

anira_backend_id custom_backend() {
    return anira_backend_id{.struct_size = sizeof(anira_backend_id),
                            .engine = ANIRA_ENGINE_CUSTOM,
                            .provider = ANIRA_PROVIDER_CPU,
                            .engine_id = anira_test::k_custom};
}

/// Runs `k_blocks` blocks of the typed stream through a prepared handler, each waited for, and
/// returns the output stream; what process returned is checked to be ANIRA_OK.
std::vector<std::byte> run_stream(anira_handler* h, anira_dtype dtype) {
    std::vector<std::byte> out_stream;
    const std::array<int64_t, 2> extents{1, static_cast<int64_t>(k_block)};
    for (size_t k = 0; k < k_blocks; ++k) {
        std::vector<std::byte> in = block_of(dtype, k);
        std::vector<std::byte> out(in.size());
        const Tensor source = Tensor::from_host(in.data(), dtype, extents);
        const Tensor sink = Tensor::from_host(out.data(), dtype, extents);
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_process(h, &source, 0, &sink, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_block);
        anira_test::wait_for_block(h, prev);
        out_stream.insert(out_stream.end(), out.begin(), out.end());
    }
    return out_stream;
}

/// Slot 0 of one side of the selected plan's report row.
anira_plan_slot slot_row(const anira_handler* h, bool inputs) {
    std::array<anira_plan_slot, 4> rows{};
    auto count = static_cast<uint32_t>(rows.size());
    EXPECT_EQ(anira_plan_report_slots(anira_handler_plan_report(h),
                                      anira_handler_get_plan(h),
                                      inputs ? 1U : 0U,
                                      sizeof(anira_plan_slot),
                                      &count,
                                      rows.data()),
              ANIRA_OK);
    return rows[0];
}

/// The input stream delayed by `latency` samples, zeros first: what a pass-through delivers.
std::vector<std::byte> delayed(anira_dtype dtype, size_t latency) {
    const size_t element = size_of(dtype);
    std::vector<std::byte> in;
    for (size_t k = 0; k < k_blocks; ++k) {
        const std::vector<std::byte> block = block_of(dtype, k);
        in.insert(in.end(), block.begin(), block.end());
    }
    std::vector<std::byte> out(in.size(), std::byte{0});
    if (latency * element < in.size()) {
        std::memcpy(out.data() + (latency * element), in.data(), in.size() - (latency * element));
    }
    return out;
}

}  // namespace

// A model of each dtype anira names streams through a registered engine with no ring dtype set:
// the ring streams the spec's dtype, the host passes tensors of it, the chunk hands the engine
// descriptors of it, and the output is the input delayed by the reported latency, bit for bit.
TEST(AbiDtypes, AModelOfEveryDtypeStreamsWithoutARingSetting) {
    const anira_test::Context context(2);
    const std::vector<anira_backend_id> candidates{custom_backend()};
    for (const anira_dtype dtype : {ANIRA_DTYPE_F32,
                                    ANIRA_DTYPE_F64,
                                    ANIRA_DTYPE_F16,
                                    ANIRA_DTYPE_BF16,
                                    ANIRA_DTYPE_I8,
                                    ANIRA_DTYPE_U8,
                                    ANIRA_DTYPE_I16,
                                    ANIRA_DTYPE_I32,
                                    ANIRA_DTYPE_I64,
                                    ANIRA_DTYPE_BOOL8}) {
        SCOPED_TRACE(anira::capi::dtype_word(dtype));
        ModelConfig model;
        model.add_model_path(anira_test::k_custom, "custom-processor");
        model.input(stream("in", dtype));
        model.output(stream("out", dtype));
        anira_test::Handler handler(context, model, candidates);
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS)),
                  ANIRA_OK)
            << handler.m_err.message;
        anira_handler* h = handler.m_handler;
        const anira::SessionElement& session = *anira_test::session_of(h);
        EXPECT_EQ(session.send_ring(0).dtype(), dtype) << "the ring streams the spec's dtype";
        EXPECT_EQ(session.receive_ring(0).dtype(), dtype);
        // The report names both dtypes; a registered engine is handed the chunk's buffer.
        for (const bool inputs : {true, false}) {
            const anira_plan_slot row = slot_row(h, inputs);
            EXPECT_EQ(row.dtype, dtype);
            EXPECT_EQ(row.ring_dtype, dtype);
            EXPECT_EQ(row.edge_class, ANIRA_EDGE_ZERO_COPY);
        }
        const std::vector<std::byte> out = run_stream(h, dtype);
        EXPECT_EQ(out, delayed(dtype, anira_handler_get_latency(h, 0)));
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    }
}

// The spec's create refuses no dtype but 0; a spec of a dtype anira does not name is refused
// where it is used (the validator's NOT_SUPPORTED) and has no word in a model file: to_json
// refuses it with NOT_SUPPORTED instead of writing another dtype's word.
TEST(AbiDtypes, ADtypeAniraDoesNotNameHasNoWordInAModelFile) {
    anira_tensor_spec* spec = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_tensor_spec_create("in", 0, ANIRA_ROLE_STREAMED, &spec, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(spec, nullptr);

    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(stream("in", ANIRA_MAKE_DTYPE(ANIRA_DTYPE_UINT, 16, 1)));
    model.output(stream("out", ANIRA_DTYPE_I16));
    size_t length = 0;
    EXPECT_EQ(anira_model_config_to_json(model.native(), nullptr, 0, &length),
              ANIRA_ERROR_NOT_SUPPORTED);

    ModelConfig named;
    named.add_model_path(anira_test::k_custom, "custom-processor");
    named.input(stream("in", ANIRA_DTYPE_BF16));
    named.output(stream("out", ANIRA_DTYPE_BOOL8));
    EXPECT_EQ(anira_model_config_to_json(named.native(), nullptr, 0, &length),
              ANIRA_ERROR_BUFFER_TOO_SMALL);
    std::string text(length + 1, '\0');  // the text and its terminator
    ASSERT_EQ(anira_model_config_to_json(named.native(), text.data(), text.size(), &length),
              ANIRA_OK);
    text.resize(length);
    EXPECT_NE(text.find(R"("dtype": "bfloat16")"), std::string::npos) << text;
    EXPECT_NE(text.find(R"("dtype": "bool")"), std::string::npos) << text;
}

// A Static tensor travels in its spec's dtype: an int16 Static input set by the host reaches
// the engine whole (the pass-through copies it to the int16 Static output) and comes back
// through get_static_output; another dtype is refused on both sides.
TEST(AbiDtypes, AStaticTensorTravelsInItsSpecsDtype) {
    const anira_test::Context context(2);
    const std::vector<anira_backend_id> candidates{custom_backend()};
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(stream("in", ANIRA_DTYPE_I16));
    TensorSpec gain("gain", ANIRA_DTYPE_I16, ANIRA_ROLE_STATIC);
    gain.axis(0, ANIRA_AXIS_ANY, 3);
    model.input(gain);
    model.output(stream("out", ANIRA_DTYPE_I16));
    TensorSpec peak("peak", ANIRA_DTYPE_I16, ANIRA_ROLE_STATIC);
    peak.axis(0, ANIRA_AXIS_ANY, 3);
    model.output(peak);
    anira_test::Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(anira_test::explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS)),
              ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    std::array<int16_t, 3> value{-1234, 7, 32767};
    const std::array<int64_t, 1> static_extents{3};
    const Tensor static_in = Tensor::from_host(value.data(), ANIRA_DTYPE_I16, static_extents);
    ASSERT_EQ(anira_handler_set_static_input(h, 1, &static_in), ANIRA_OK);
    std::array<float, 3> wrong{};
    const Tensor float_in = Tensor::from_host(wrong.data(), ANIRA_DTYPE_F32, static_extents);
    EXPECT_EQ(anira_handler_set_static_input(h, 1, &float_in), ANIRA_ERROR_CONFIG);

    static_cast<void>(run_stream(h, ANIRA_DTYPE_I16));
    std::array<int16_t, 3> got{};
    const Tensor static_out = Tensor::from_host(got.data(), ANIRA_DTYPE_I16, static_extents);
    ASSERT_EQ(anira_handler_get_static_output(h, 1, &static_out), ANIRA_OK);
    EXPECT_EQ(got, value);
    const Tensor float_out = Tensor::from_host(wrong.data(), ANIRA_DTYPE_F32, static_extents);
    EXPECT_EQ(anira_handler_get_static_output(h, 1, &float_out), ANIRA_ERROR_CONFIG);
}

// A float32 host tensor on an int16 model's slot (the ring at its default, the spec's int16)
// is refused by every Hard entry that takes a tensor: ANIRA_ERROR_CONFIG, recorded, nothing
// delivered, one record that names both remedies, and nothing reaches the ring: a twin handler
// fed only the valid calls produces the same output stream, bit for bit.
TEST(AbiDtypes, AHostTensorOfAnotherDtypeIsRefusedPerCallAndLeavesTheStream) {
    const anira_test::Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_DEBUG);
    anira_drain_log();
    anira_test::RecordCollector collector;
    const std::vector<anira_backend_id> candidates{custom_backend()};
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(stream("in", ANIRA_DTYPE_I16));
    model.output(stream("out", ANIRA_DTYPE_I16));
    anira_test::Handler refused(context, model, candidates);
    anira_test::Handler twin(context, model, candidates);
    const ContractHandle contract =
        anira_test::explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS);
    ASSERT_EQ(refused.prepare(contract), ANIRA_OK) << refused.m_err.message;
    ASSERT_EQ(twin.prepare(contract), ANIRA_OK) << twin.m_err.message;

    const std::array<int64_t, 2> extents{1, static_cast<int64_t>(k_block)};
    std::vector<float> floats(k_block, 0.5F);
    const Tensor float_block = Tensor::from_host(floats.data(), ANIRA_DTYPE_F32, extents);
    std::vector<int16_t> sink(k_block);
    const Tensor int16_sink = Tensor::from_host(sink.data(), ANIRA_DTYPE_I16, extents);
    size_t delivered = 7;
    EXPECT_EQ(anira_handler_process(refused.m_handler, &float_block, 0, &int16_sink, 0, &delivered),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_push_data(refused.m_handler, &float_block, 0), ANIRA_ERROR_CONFIG);
    const std::array<anira_tensor, 1> multi{float_block};
    EXPECT_EQ(
        anira_handler_process_multi(refused.m_handler, multi.data(), 1, &int16_sink, 1, nullptr),
        ANIRA_ERROR_CONFIG);
    const Tensor float_sink = Tensor::from_host(floats.data(), ANIRA_DTYPE_F32, extents);
    EXPECT_EQ(anira_handler_pop_data(refused.m_handler, &float_sink, 0, nullptr),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_handler_rt_error(refused.m_handler), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_test::session_of(refused.m_handler)->send_ring(0).available(0), 0U);

    EXPECT_EQ(run_stream(refused.m_handler, ANIRA_DTYPE_I16),
              run_stream(twin.m_handler, ANIRA_DTYPE_I16));
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::find_record(collector, "nothing converts", "rt").m_message,
              "anira_handler_process: the tensor of input slot 0 'in' is float32, the slot's ring "
              "dtype is int16; nothing converts: pass a tensor of the ring dtype, or name float32 "
              "in the contract's ring_dtypes and add a processor that converts in pre_process");
#endif
}

namespace {

/// float32 to float16 (round to nearest, ties to even; finite inputs whose half is normal), the
/// processor's own helper: anira converts nothing.
uint16_t to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude == 0) { return static_cast<uint16_t>(sign); }
    const int exponent = static_cast<int>(magnitude >> 23) - 127 + 15;
    uint32_t mantissa = (magnitude & 0x7fffffU) | 0x800000U;
    // 24 bits of mantissa to 11: round the 13 dropped bits to nearest, ties to even.
    const uint32_t dropped = mantissa & 0x1fffU;
    mantissa >>= 13;
    if (dropped > 0x1000U || (dropped == 0x1000U && (mantissa & 1U) != 0)) { ++mantissa; }
    auto half_exponent = static_cast<uint32_t>(exponent);
    if (mantissa == 0x800U) {
        mantissa >>= 1;
        ++half_exponent;
    }
    return static_cast<uint16_t>(sign | (half_exponent << 10) | (mantissa & 0x3ffU));
}

float from_half(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000U) << 16;
    const uint32_t exponent = (half >> 10) & 0x1fU;
    const uint32_t mantissa = half & 0x3ffU;
    uint32_t bits = sign;
    if (exponent != 0) { bits |= ((exponent - 15 + 127) << 23) | (mantissa << 13); }
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// What a host's converting processor does between its float32 rings and a model of another
/// dtype, with its own scaling: float16 (a value conversion), int8 (x * 127, rounded and
/// clamped, the model's convention). Both phases, one hop per call, the scratch its own.
class ConvertingPrepared final : public anira::Processor::Prepared {
public:
    explicit ConvertingPrepared(anira_dtype model_dtype) : m_model_dtype(model_dtype) {}

    anira_status pre_process(anira::ProcessorContext& ctx) noexcept override {
        Tensor tensor{};
        anira::RingView ring;
        if (ctx.input_tensor(0, tensor) != ANIRA_OK || ctx.input_ring(0, ring) != ANIRA_OK) {
            return ANIRA_ERROR_CONFIG;
        }
        void* model = tensor.data(m_model_dtype);
        if (model == nullptr || ring.pop_block(0, std::span<float>(m_scratch)) != k_block) {
            return ANIRA_ERROR_CONFIG;
        }
        for (size_t n = 0; n < k_block; ++n) {
            if (m_model_dtype == ANIRA_DTYPE_F16) {
                static_cast<uint16_t*>(model)[n] = to_half(m_scratch.at(n));
            } else {
                const float scaled = std::nearbyint(m_scratch.at(n) * 127.0F);
                static_cast<int8_t*>(model)[n] = static_cast<int8_t>(
                    scaled < -128.0F ? -128.0F : (scaled > 127.0F ? 127.0F : scaled));
            }
        }
        return ANIRA_OK;
    }

    anira_status post_process(anira::ProcessorContext& ctx) noexcept override {
        Tensor tensor{};
        anira::RingView ring;
        if (ctx.output_tensor(0, tensor) != ANIRA_OK || ctx.output_ring(0, ring) != ANIRA_OK) {
            return ANIRA_ERROR_CONFIG;
        }
        const void* model = tensor.data(m_model_dtype);
        if (model == nullptr) { return ANIRA_ERROR_CONFIG; }
        for (size_t n = 0; n < k_block; ++n) {
            m_scratch.at(n) =
                m_model_dtype == ANIRA_DTYPE_F16
                    ? from_half(static_cast<const uint16_t*>(model)[n])
                    : static_cast<float>(static_cast<const int8_t*>(model)[n]) / 127.0F;
        }
        return ring.push_block(0, std::span<const float>(m_scratch)) == k_block
                   ? ANIRA_OK
                   : ANIRA_ERROR_CONFIG;
    }

private:
    anira_dtype m_model_dtype;
    std::array<float, k_block> m_scratch{};
};

class ConvertingProcessor final : public anira::Processor {
public:
    explicit ConvertingProcessor(anira_dtype model_dtype) : m_model_dtype(model_dtype) {}
    uint32_t phases() const noexcept override { return k_pre_process | k_post_process; }
    uint32_t flags() const noexcept override { return ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST; }
    std::unique_ptr<anira::Processor::Prepared> prepare(
        const anira::PrepareInfo& /*info*/) override {
        return std::make_unique<ConvertingPrepared>(m_model_dtype);
    }

private:
    anira_dtype m_model_dtype;
};

}  // namespace

// A float32 host on a float16 and on an int8 model: the contract names float32 for both rings,
// the host's processor converts in pre_process and post_process with its own scaling, and the
// values the model can hold come back as they went in (the float16 ones exactly, the int8 ones
// k / 127 exactly), delayed by the latency. Without the processor the same contract is CONFIG
// at prepare: nothing in anira converts.
TEST(AbiDtypes, AFloatHostOnAnotherModelDtypeConvertsInItsProcessor) {
    const anira_test::Context context(2);
    for (const anira_dtype dtype : {ANIRA_DTYPE_F16, ANIRA_DTYPE_I8}) {
        SCOPED_TRACE(anira::capi::dtype_word(dtype));
        ModelConfig model;
        model.add_model_path(anira_test::k_custom, "custom-processor");
        model.input(stream("in", dtype));
        model.output(stream("out", dtype));
        ContractHandle contract = anira_test::explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS);
        contract.hard_ring_dtype("in", ANIRA_DTYPE_F32).hard_ring_dtype("out", ANIRA_DTYPE_F32);

        anira::Pipeline bare;
        anira_test::add_passthrough(bare.native());
        bare.add_inference(model, {anira::BackendId(custom_backend())});
        anira_handler* without = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_create(context.m_context, bare.native(), &without, &err), ANIRA_OK)
            << err.message;
        EXPECT_EQ(anira_handler_prepare(without, contract.native(), &err), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(err.message).find("nothing converts"), std::string::npos)
            << err.message;
        anira_handler_destroy(without);

        anira::Pipeline pipe;
        anira_test::add_passthrough(pipe.native());
        pipe.add_inference(model, {anira::BackendId(custom_backend())});
        pipe.add_processor(std::make_shared<ConvertingProcessor>(dtype));
        anira_handler* h = nullptr;
        ASSERT_EQ(anira_handler_create(context.m_context, pipe.native(), &h, &err), ANIRA_OK)
            << err.message;
        ASSERT_EQ(anira_handler_prepare(h, contract.native(), &err), ANIRA_OK) << err.message;
        const size_t latency = anira_handler_get_latency(h, 0);
        for (const bool inputs : {true, false}) {
            const anira_plan_slot row = slot_row(h, inputs);
            EXPECT_EQ(row.dtype, dtype) << "the model tensor's";
            EXPECT_EQ(row.ring_dtype, ANIRA_DTYPE_F32) << "the host's, named in the contract";
        }

        std::vector<float> in_stream;
        std::vector<float> out_stream;
        const std::array<int64_t, 2> extents{1, static_cast<int64_t>(k_block)};
        for (size_t k = 0; k < k_blocks; ++k) {
            std::vector<float> in(k_block);
            for (size_t n = 0; n < k_block; ++n) {
                const auto i = static_cast<int64_t>((k * k_block) + n);
                // float16: eighths in [-4, 4); int8: k / 127 for k in [-100, 100).
                in[n] = dtype == ANIRA_DTYPE_F16 ? static_cast<float>((i % 64) - 32) / 8.0F
                                                 : static_cast<float>((i % 200) - 100) / 127.0F;
            }
            std::vector<float> out(k_block, -9.0F);
            const Tensor source = Tensor::from_host(in.data(), ANIRA_DTYPE_F32, extents);
            const Tensor sink = Tensor::from_host(out.data(), ANIRA_DTYPE_F32, extents);
            const size_t prev = anira_test::available(h);
            ASSERT_EQ(anira_handler_process(h, &source, 0, &sink, 0, nullptr), ANIRA_OK);
            anira_test::wait_for_block(h, prev);
            in_stream.insert(in_stream.end(), in.begin(), in.end());
            out_stream.insert(out_stream.end(), out.begin(), out.end());
        }
        for (size_t n = 0; n < out_stream.size(); ++n) {
            const float want = n < latency ? 0.0F : in_stream[n - latency];
            ASSERT_EQ(out_stream[n], want) << "sample " << n;
        }
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        anira_handler_destroy(h);
    }
}

// ============================================================================================
// The built-in engines: the typed fixtures through the Hard entries
// ============================================================================================

namespace {

/// The built-in engines of this build whose adapters bind typed tensors.
std::vector<anira_engine> typed_engines() {
    std::vector<anira_engine> engines;
#ifdef USE_ONNXRUNTIME
    engines.push_back(ANIRA_ENGINE_ONNXRUNTIME);
#endif
#ifdef USE_LIBTORCH
    engines.push_back(ANIRA_ENGINE_LIBTORCH);
#endif
#ifdef USE_LITERT
    engines.push_back(ANIRA_ENGINE_LITERT);
#endif
    return engines;
}

/// The built-in engines of this build that read a model tensor's quantisation parameters.
std::vector<anira_engine> quantised_engines() {
    std::vector<anira_engine> engines;
#ifdef USE_LITERT
    engines.push_back(ANIRA_ENGINE_LITERT);
#endif
    return engines;
}

/// One candidate: the engine, on the CPU path.
std::vector<anira_backend_id> one_engine(anira_engine engine) {
    return {anira_backend_id{.struct_size = sizeof(anira_backend_id),
                             .engine = static_cast<uint32_t>(engine),
                             .provider = ANIRA_PROVIDER_CPU,
                             .engine_id = nullptr}};
}

/// A small integer as an element of `dtype` (exact in every dtype anira names; bool: nonzero).
void store_value(std::byte* at, anira_dtype dtype, float value) {
    const auto store = [at](const auto element) { std::memcpy(at, &element, sizeof(element)); };
    switch (dtype) {
        case ANIRA_DTYPE_F32: store(value); break;
        case ANIRA_DTYPE_F64: store(static_cast<double>(value)); break;
        case ANIRA_DTYPE_F16: store(to_half(value)); break;
        case ANIRA_DTYPE_BF16: {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            store(static_cast<uint16_t>(bits >> 16));
        } break;
        case ANIRA_DTYPE_I8: store(static_cast<int8_t>(value)); break;
        case ANIRA_DTYPE_U8: store(static_cast<uint8_t>(value)); break;
        case ANIRA_DTYPE_I16: store(static_cast<int16_t>(value)); break;
        case ANIRA_DTYPE_I32: store(static_cast<int32_t>(value)); break;
        case ANIRA_DTYPE_I64: store(static_cast<int64_t>(value)); break;
        case ANIRA_DTYPE_BOOL8: store(static_cast<uint8_t>(value != 0.0F ? 1 : 0)); break;
        default: FAIL() << "no store for dtype " << dtype;
    }
}

float load_value(const std::byte* at, anira_dtype dtype) {
    const auto load = [at](auto element) {
        std::memcpy(&element, at, sizeof(element));
        return element;
    };
    switch (dtype) {
        case ANIRA_DTYPE_F32: return load(0.0F);
        case ANIRA_DTYPE_F64: return static_cast<float>(load(0.0));
        case ANIRA_DTYPE_F16: return from_half(load(uint16_t{0}));
        case ANIRA_DTYPE_BF16: {
            const uint32_t bits = static_cast<uint32_t>(load(uint16_t{0})) << 16;
            float value = 0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        case ANIRA_DTYPE_I8: return load(int8_t{0});
        case ANIRA_DTYPE_U8: return load(uint8_t{0});
        case ANIRA_DTYPE_I16: return load(int16_t{0});
        case ANIRA_DTYPE_I32: return static_cast<float>(load(int32_t{0}));
        case ANIRA_DTYPE_I64: return static_cast<float>(load(int64_t{0}));
        case ANIRA_DTYPE_BOOL8: return load(uint8_t{0}) != 0 ? 1.0F : 0.0F;
        default: ADD_FAILURE() << "no load for dtype " << dtype; return 0.0F;
    }
}

/// Streams `k_blocks` blocks of `values(k, n)` (a small integer per sample) in `dtype` through a
/// prepared handler, each waited for, and returns the output stream as floats.
template <class Values>
std::vector<float> run_values(anira_handler* h, anira_dtype dtype, const Values& values) {
    const size_t element = size_of(dtype);
    const std::array<int64_t, 2> extents{1, static_cast<int64_t>(k_block)};
    std::vector<float> out_stream;
    for (size_t k = 0; k < k_blocks; ++k) {
        std::vector<std::byte> in(k_block * element);
        std::vector<std::byte> out(k_block * element, std::byte{0x5a});
        for (size_t n = 0; n < k_block; ++n) { store_value(&in[n * element], dtype, values(k, n)); }
        const Tensor source = Tensor::from_host(in.data(), dtype, extents);
        const Tensor sink = Tensor::from_host(out.data(), dtype, extents);
        const size_t prev = anira_test::available(h);
        EXPECT_EQ(anira_handler_process(h, &source, 0, &sink, 0, nullptr), ANIRA_OK);
        anira_test::wait_for_block(h, prev);
        for (size_t n = 0; n < k_block; ++n) {
            out_stream.push_back(load_value(&out[n * element], dtype));
        }
    }
    return out_stream;
}

/// The typed gain stream: block k, sample n, a small integer in [-8, 8] (every block holds both
/// ends, so every block's peak is 8).
float gain_sample(size_t k, size_t n) {
    return static_cast<float>(static_cast<int>(((k * k_block) + n) % 17) - 8);
}

}  // namespace

// The bundled TypedGainNetwork at int16, int8 and float16 on every built-in engine that binds
// typed tensors: no ring setting (the rings stream the spec's dtype), the Static gain of 2 in
// the file's dtype, every block waited for. The stream comes out as the input times 2 delayed by
// the latency, exactly, and the Static peak is the doubled largest magnitude, 16.
TEST(AbiDtypes, TheTypedGainFixturesRunOnEveryTypedEngine) {
    const anira_test::Context context(2);
    const std::array<std::pair<const char*, anira_dtype>, 3> sets{{
        {k_typed_gain_int16_model_json, ANIRA_DTYPE_I16},
        {k_typed_gain_int8_model_json, ANIRA_DTYPE_I8},
        {k_typed_gain_float16_model_json, ANIRA_DTYPE_F16},
    }};
    for (const anira_engine engine : typed_engines()) {
        for (const auto& [model_json, dtype] : sets) {
            SCOPED_TRACE(std::string(anira::capi::engine_word(engine)) + " " +
                         anira::capi::dtype_word(dtype));
            const ModelConfig model = ModelConfig::from_file(model_json);
            const std::vector<anira_backend_id> candidates = one_engine(engine);
            anira_test::Handler handler(context, model, candidates);
            ASSERT_EQ(handler.prepare(anira_test::file_contract(k_gain_contract_json, k_block)),
                      ANIRA_OK)
                << handler.m_err.message;
            anira_handler* h = handler.m_handler;
            const std::array<int64_t, 1> scalar{1};
            std::array<std::byte, 8> gain{};
            store_value(gain.data(), dtype, 2.0F);
            const Tensor gain_in = Tensor::from_host(gain.data(), dtype, scalar);
            ASSERT_EQ(anira_handler_set_static_input(h, 1, &gain_in), ANIRA_OK);

            const std::vector<float> out = run_values(h, dtype, gain_sample);
            const size_t latency = anira_handler_get_latency(h, 0);
            for (size_t n = 0; n < out.size(); ++n) {
                const float want = n < latency ? 0.0F
                                               : 2.0F * gain_sample((n - latency) / k_block,
                                                                    (n - latency) % k_block);
                ASSERT_EQ(out[n], want) << "sample " << n;
            }
            std::array<std::byte, 8> peak{};
            const Tensor peak_out = Tensor::from_host(peak.data(), dtype, scalar);
            ASSERT_EQ(anira_handler_get_static_output(h, 1, &peak_out), ANIRA_OK);
            EXPECT_EQ(load_value(peak.data(), dtype), 16.0F);
            EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        }
    }
}

// The fully integer TFLite model: its int8 input and output carry a scale and a zero point
// (quantised I/O, which anira does not support), so the load refuses it with NOT_SUPPORTED
// saying so, rather than handing the engine's integers on as plain int8 samples.
TEST(AbiDtypes, AQuantisedModelIsRefusedAtLoad) {
    const anira_test::Context context(2);
    const auto spec = [](std::string_view name) {
        TensorSpec tensor(name, ANIRA_DTYPE_I8, ANIRA_ROLE_STREAMED);
        tensor.axis(0, ANIRA_AXIS_BATCH, 1)
            .axis(1, ANIRA_AXIS_CHANNEL, 1)
            .axis(2, ANIRA_AXIS_TIME, 64)
            .window(64, 64, 0);
        return tensor;
    };
    for (const anira_engine engine : quantised_engines()) {
        SCOPED_TRACE(anira::capi::engine_word(engine));
        ModelConfig model;
        model.add_model_path(engine, k_quantized_gain_int8_tflite);
        model.input(spec("x"));
        model.output(spec("output_0"));
        const std::vector<anira_backend_id> candidates = one_engine(engine);
        anira_test::Handler handler(context, model, candidates);
        EXPECT_EQ(handler.prepare(anira_test::explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS)),
                  ANIRA_ERROR_NOT_SUPPORTED)
            << handler.m_err.message;
        EXPECT_NE(std::string_view(handler.m_err.message).find("quantised I/O is not supported"),
                  std::string_view::npos)
            << handler.m_err.message;
    }
}

#ifdef USE_ONNXRUNTIME
// The ONNX dtype matrix: for each of the ten dtypes anira names, a model whose one input and
// one output are of that dtype (Cast to float32, times 2, Cast back) through the ONNX Runtime
// engine and the Hard entries, with no ring setting. The output is the input doubled (a bool
// comes back as it went in), delayed by the latency, exactly.
TEST(AbiDtypes, TheOnnxDtypeMatrixRunsThroughOnnxRuntime) {
    const anira_test::Context context(2);
    const std::vector<anira_backend_id> candidates = one_engine(ANIRA_ENGINE_ONNXRUNTIME);
    for (const auto& [word, dtype] : anira::capi::k_dtype_words) {
        SCOPED_TRACE(word);
        const auto spec = [dtype](std::string_view name) {
            TensorSpec tensor(name, dtype, ANIRA_ROLE_STREAMED);
            tensor.axis(0, ANIRA_AXIS_BATCH, 1)
                .axis(1, ANIRA_AXIS_CHANNEL, 1)
                .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_block))
                .window(static_cast<int64_t>(k_block), static_cast<int64_t>(k_block), 0);
            return tensor;
        };
        ModelConfig model;
        model.add_model_path(ANIRA_ENGINE_ONNXRUNTIME,
                             std::string(k_dtype_matrix_onnx_prefix) + word + ".onnx");
        model.input(spec("data"));
        model.output(spec("processed_data"));
        anira_test::Handler handler(context, model, candidates);
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS)),
                  ANIRA_OK)
            << handler.m_err.message;
        const bool boolean = dtype == ANIRA_DTYPE_BOOL8;
        const auto values = [boolean](size_t k, size_t n) {
            return static_cast<float>(((k * k_block) + n) % (boolean ? 2 : 8));
        };
        const std::vector<float> out = run_values(handler.m_handler, dtype, values);
        const size_t latency = anira_handler_get_latency(handler.m_handler, 0);
        for (size_t n = 0; n < out.size(); ++n) {
            const float in =
                n < latency ? 0.0F : values((n - latency) / k_block, (n - latency) % k_block);
            ASSERT_EQ(out[n], boolean ? in : 2.0F * in) << "sample " << n;
        }
    }
}
#endif  // USE_ONNXRUNTIME
