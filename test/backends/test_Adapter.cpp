// The engine room's interface (src/backends/Adapter.h) and the legacy adapter over the 2.x
// virtual, driven directly: no core, no session. The claim loop under several threads, the
// float32 helpers of the built-in adapters, the 2.x plan table as requests, and the copies the
// legacy adapter makes around a descriptor that names other memory than the struct's buffer.

#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "backends/Adapter.h"
#include "backends/Adapters.h"
#include "backends/LegacyAdapter.h"
#include "gtest/gtest.h"
#include "utils/StatusError.h"

namespace {

using anira::backend::Adapter;
using anira::backend::ChunkBuffers;
using anira::backend::Model;
using anira::backend::PlanRequest;
using anira::backend::Source;
using anira::backend::TensorInfo;

constexpr size_t k_block = 512;

// A record of one float32 tensor of `dims`.
TensorInfo f32_tensor(const std::string& name, std::vector<int64_t> dims) {
    TensorInfo tensor;
    tensor.m_name = name;
    tensor.m_dims = std::move(dims);
    size_t elements = 1;
    for (const int64_t extent : tensor.m_dims) { elements *= static_cast<size_t>(extent); }
    tensor.m_num_elements = elements;
    return tensor;
}

// A model of one [1, 1, k_block] float32 tensor per side.
Model gain_model(uint32_t instances = 1) {
    Model model;
    model.m_engine = ANIRA_ENGINE_NONE;
    model.m_path = "placeholder";
    model.m_inputs.push_back(f32_tensor("audio_in", {1, 1, k_block}));
    model.m_outputs.push_back(f32_tensor("audio_out", {1, 1, k_block}));
    model.m_instances = instances;
    return model;
}

// An adapter that records what run hands its process: the instance of every call, whether two
// calls ever ran at once on one instance, and the reset calls.
class RecordingAdapter final : public Adapter {
public:
    static constexpr uint32_t k_max_instances = 8;

    std::atomic<uint32_t> m_calls{0};
    std::atomic<uint32_t> m_overlaps{0};
    std::atomic<uint32_t> m_out_of_range{0};
    std::atomic<uint32_t> m_resets{0};
    std::atomic<uint32_t> m_reset_instance{k_max_instances};
    std::array<std::atomic<bool>, k_max_instances> m_in_use{};
    std::array<std::atomic<uint32_t>, k_max_instances> m_per_instance{};
    anira_status m_status = ANIRA_OK;
    std::chrono::microseconds m_hold{200};

protected:
    void do_prepare(const Model& model) override { static_cast<void>(model); }

    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override {
        static_cast<void>(chunk);
        if (ctx.instance >= model().m_instances || ctx.instance >= k_max_instances) {
            m_out_of_range.fetch_add(1);
            return ANIRA_ERROR_INTERNAL;
        }
        if (m_in_use.at(ctx.instance).exchange(true)) { m_overlaps.fetch_add(1); }
        std::this_thread::sleep_for(m_hold);
        m_per_instance.at(ctx.instance).fetch_add(1);
        m_calls.fetch_add(1);
        m_in_use.at(ctx.instance).store(false);
        return m_status;
    }

    void reset(const anira_engine_ctx& ctx) noexcept override {
        m_resets.fetch_add(1);
        m_reset_instance.store(ctx.instance);
    }
};

// A 2.x backend that records the first sample it was handed and passes the block through
// (BackendBase::process), or throws.
class RecordingBackend final : public anira::BackendBase {
public:
    explicit RecordingBackend(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void prepare() override { ++m_prepares; }

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 std::shared_ptr<anira::SessionElement> session) override {
        if (m_throws) { throw std::runtime_error("the 2.x backend refused"); }
        m_first_sample = input.at(0).get_sample(0, 0);
        m_input_seen = input.at(0).data();
        m_output_seen = output.at(0).data();
        anira::BackendBase::process(input, output, std::move(session));
        ++m_calls;
    }

    int m_prepares = 0;
    int m_calls = 0;
    bool m_throws = false;
    float m_first_sample = -1.F;
    const float* m_input_seen = nullptr;
    const float* m_output_seen = nullptr;
};

// A 2.x backend that waits inside process until `arrivals` callers are inside at once, or a
// deadline passes: whether the adapter lets a 2.x backend run concurrently.
class MeetingBackend final : public anira::BackendBase {
public:
    MeetingBackend(anira::InferenceConfig& config, uint32_t arrivals)
        : anira::BackendBase(config), m_arrivals(arrivals) {}

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 std::shared_ptr<anira::SessionElement> session) override {
        static_cast<void>(input);
        static_cast<void>(output);
        static_cast<void>(session);
        m_inside.fetch_add(1);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (m_inside.load() < m_arrivals && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (m_inside.load() >= m_arrivals) { m_met.store(true); }
    }

    uint32_t m_arrivals;
    std::atomic<uint32_t> m_inside{0};
    std::atomic<bool> m_met{false};
};

anira::InferenceConfig custom_only_config() {
    return anira::InferenceConfig(
        std::vector<anira::ModelData>{
            anira::ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<anira::TensorShape>{anira::TensorShape({{1, 1, k_block}}, {{1, 1, k_block}})},
        anira::ProcessingSpec({1}, {1}, {k_block}, {k_block}),
        10.F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/false,
        /*blocking_ratio=*/0.F,
        /*num_parallel_processors=*/2);
}

// Every backend of this build, in enum order, CUSTOM last: the tail of the 2.x plan table.
std::vector<anira::InferenceBackend> every_backend() {
    return {
#ifdef USE_LIBTORCH
        anira::InferenceBackend::LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
        anira::InferenceBackend::ONNX,
#endif
#ifdef USE_TFLITE
        anira::InferenceBackend::TFLITE,
#endif
#ifdef USE_LITERT
        anira::InferenceBackend::LITERT,
#endif
#ifdef USE_EXECUTORCH
        anira::InferenceBackend::EXECUTORCH,
#endif
        anira::InferenceBackend::CUSTOM,
    };
}

// The context of one call over `inputs` and `outputs`, instance 0, no ticket.
anira_engine_ctx context_of(const std::vector<anira_tensor>& inputs,
                            std::vector<anira_tensor>& outputs) {
    anira_engine_ctx ctx{};
    ctx.num_inputs = static_cast<uint32_t>(inputs.size());
    ctx.num_outputs = static_cast<uint32_t>(outputs.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.inputs = inputs.data();
    ctx.outputs = outputs.data();
    return ctx;
}

// One descriptor per buffer of `buffers`, over its memory, as [1, elements].
std::vector<anira_tensor> descriptors_over(std::vector<anira::BufferF>& buffers) {
    std::vector<anira_tensor> tensors(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        const std::array<int64_t, 2> shape{
            1,
            static_cast<int64_t>(buffers[i].get_num_channels() * buffers[i].get_num_samples())};
        anira_tensor_init_host(&tensors[i], buffers[i].data(), ANIRA_DTYPE_F32, 2, shape.data());
    }
    return tensors;
}

}  // namespace

// ============================================================================================
// The claim loop
// ============================================================================================

TEST(Adapter, RunRefusesBeforePrepare) {
    RecordingAdapter adapter;
    EXPECT_FALSE(adapter.prepared());
    const anira_engine_ctx ctx{};
    EXPECT_EQ(adapter.run(ctx, nullptr, false), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(adapter.m_calls.load(), 0U);
}

TEST(Adapter, PrepareKeepsTheRecordAndSizesTheInstances) {
    RecordingAdapter adapter;
    const Model model = gain_model(3);
    adapter.prepare(model);
    EXPECT_TRUE(adapter.prepared());
    EXPECT_EQ(adapter.model(), model);
    EXPECT_EQ(adapter.model().m_instances, 3U);
    EXPECT_EQ(adapter.model().m_inputs.at(0).m_num_elements, k_block);
}

// Every call gets an instance below the record's count, never two calls at once on one, and no
// call starves: N threads x M calls on 3 instances all complete.
TEST(Adapter, TheClaimLoopHandsOutDistinctInstancesUnderThreads) {
    RecordingAdapter adapter;
    adapter.prepare(gain_model(3));
    constexpr uint32_t k_threads = 6;
    constexpr uint32_t k_calls = 40;
    std::atomic<uint32_t> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(k_threads);
    for (uint32_t t = 0; t < k_threads; ++t) {
        threads.emplace_back([&adapter, &failures]() {
            anira_engine_ctx ctx{};
            ctx.instance = 77;  // the caller's copy stays as it is
            for (uint32_t call = 0; call < k_calls; ++call) {
                if (adapter.run(ctx, nullptr, false) != ANIRA_OK) { failures.fetch_add(1); }
            }
            if (ctx.instance != 77) { failures.fetch_add(1); }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    EXPECT_EQ(failures.load(), 0U);
    EXPECT_EQ(adapter.m_calls.load(), k_threads * k_calls);
    EXPECT_EQ(adapter.m_overlaps.load(), 0U) << "two calls at once on one instance";
    EXPECT_EQ(adapter.m_out_of_range.load(), 0U) << "an instance at or above the count";
    uint32_t seen = 0;
    for (uint32_t i = 0; i < 3; ++i) { seen += adapter.m_per_instance.at(i).load(); }
    EXPECT_EQ(seen, k_threads * k_calls);
    EXPECT_GT(adapter.m_per_instance.at(1).load() + adapter.m_per_instance.at(2).load(), 0U)
        << "six threads on three instances never used a second one";
}

// reset runs on the claimed instance right before that call's process, and only when asked;
// the status process returns is run's.
TEST(Adapter, ResetFirstRunsOnTheClaimedInstanceAndTheStatusIsProcesss) {
    RecordingAdapter adapter;
    adapter.prepare(gain_model(1));
    const anira_engine_ctx ctx{};
    EXPECT_EQ(adapter.run(ctx, nullptr, false), ANIRA_OK);
    EXPECT_EQ(adapter.m_resets.load(), 0U);
    EXPECT_EQ(adapter.run(ctx, nullptr, true), ANIRA_OK);
    EXPECT_EQ(adapter.m_resets.load(), 1U);
    EXPECT_EQ(adapter.m_reset_instance.load(), 0U);
    adapter.m_status = ANIRA_ERROR_ENGINE;
    EXPECT_EQ(adapter.run(ctx, nullptr, false), ANIRA_ERROR_ENGINE);
    EXPECT_EQ(adapter.m_calls.load(), 3U);
}

// ============================================================================================
// The float32 helpers of the built-in adapters
// ============================================================================================

TEST(Adapter, HostF32PackedAcceptsPackedAndAllZeroStrides) {
    std::array<float, 8> data{};
    const std::array<int64_t, 3> shape{1, 2, 4};
    anira_tensor tensor;
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 3, shape.data());
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), data.data()) << "all-zero strides";

    tensor.strides[0] = 8;
    tensor.strides[1] = 4;
    tensor.strides[2] = 1;
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), data.data()) << "row-major strides";

    tensor.strides[0] = 999;  // an axis of extent 1 is never stepped
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), data.data());

    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 3, shape.data());
    tensor.byte_offset = sizeof(float);
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), data.data() + 1) << "the offset";

    anira_tensor_init_pinned(&tensor, data.data(), ANIRA_DTYPE_F32, 3, shape.data());
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), data.data()) << "page-locked host";
}

TEST(Adapter, HostF32PackedRefusals) {
    std::array<float, 8> data{};
    const std::array<int64_t, 3> shape{1, 2, 4};
    anira_tensor tensor;

    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_I16, 3, shape.data());
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), nullptr) << "dtype";

    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 3, shape.data());
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 7), nullptr) << "wrong count";
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 9), nullptr) << "wrong count";

    tensor.strides[0] = 8;
    tensor.strides[1] = 1;
    tensor.strides[2] = 2;
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), nullptr) << "strided";

    const std::array<const float*, 1> planes{data.data()};
    const std::array<int64_t, 2> planar_shape{1, 8};
    anira_tensor_init_host_planar(&tensor,
                                  static_cast<const void*>(planes.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  planar_shape.data());
    ASSERT_EQ(tensor.dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), nullptr) << "planar";

    anira_tensor_init_host(&tensor, nullptr, ANIRA_DTYPE_F32, 3, shape.data());
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), nullptr) << "no memory";

    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 3, shape.data());
    tensor.domain = static_cast<uint32_t>(ANIRA_DOMAIN_CUDA);
    EXPECT_EQ(anira::backend::host_f32_packed(tensor, 8), nullptr) << "a device domain";

    const anira_tensor zero{};
    EXPECT_EQ(anira::backend::host_f32_packed(zero, 0), nullptr) << "a refused factory's record";
}

TEST(Adapter, RequireF32NamesTheEngineAndTheFirstOtherTensor) {
    Model model = gain_model();
    EXPECT_NO_THROW(anira::backend::require_f32(model, "onnxruntime"));

    model.m_inputs.push_back(f32_tensor("gain", {1}));
    model.m_inputs.back().m_dtype = ANIRA_DTYPE_I16;
    model.m_outputs.push_back(f32_tensor("peak", {1}));
    model.m_outputs.back().m_dtype = ANIRA_DTYPE_I32;
    try {
        anira::backend::require_f32(model, "onnxruntime");
        FAIL() << "an int16 input passed";
    } catch (const anira::StatusError& e) {
        EXPECT_EQ(e.status(), ANIRA_ERROR_CONFIG);
        const std::string message = e.what();
        EXPECT_NE(message.find("onnxruntime"), std::string::npos) << message;
        EXPECT_NE(message.find("input tensor 'gain'"), std::string::npos) << message;
        EXPECT_EQ(message.find("peak"), std::string::npos) << "the first offender is named";
    }

    model.m_inputs.back().m_dtype = ANIRA_DTYPE_F32;
    try {
        anira::backend::require_f32(model, "libtorch");
        FAIL() << "an int32 output passed";
    } catch (const anira::StatusError& e) {
        EXPECT_EQ(e.status(), ANIRA_ERROR_CONFIG);
        const std::string message = e.what();
        EXPECT_NE(message.find("libtorch"), std::string::npos) << message;
        EXPECT_NE(message.find("output tensor 'peak'"), std::string::npos) << message;
    }
}

// ============================================================================================
// The record
// ============================================================================================

TEST(Adapter, ModelsCompareByWhatAnAdapterReads) {
    const Model a = gain_model(2);
    Model b = gain_model(2);
    EXPECT_EQ(a, b);
    b.m_log_level = anira::LogLevel::Debug;
    b.m_bytes_owner = std::shared_ptr<const void>(std::make_shared<int>(1));
    EXPECT_EQ(a, b) << "the level and the owner are no part of the identity";
    b = gain_model(3);
    EXPECT_NE(a, b) << "the instances are";
    b = gain_model(2);
    b.m_inputs.at(0).m_dims.back() = 256;
    EXPECT_NE(a, b) << "the dims are";
    b = gain_model(2);
    b.m_entry = "encode";
    EXPECT_NE(a, b) << "the entry is";
    b = gain_model(2);
    b.m_warm_up = 3;
    EXPECT_NE(a, b) << "the warm-up is";
    b = gain_model(2);
    b.m_session_exclusive = true;
    EXPECT_NE(a, b) << "the exclusivity is";
}

// ============================================================================================
// The 2.x plan table as requests
// ============================================================================================

// A CUSTOM-only configuration: the custom row first, then every other backend of the build
// (the roundtrip without a model), as Core::create_session always built the table.
TEST(Adapter, LegacyPlanRequestsOfACustomOnlyConfigMatchTheDefaultTable) {
    anira::InferenceConfig config = custom_only_config();
    std::vector<anira::InferenceBackend> expected{anira::InferenceBackend::CUSTOM};
    for (const anira::InferenceBackend backend : every_backend()) {
        if (backend != anira::InferenceBackend::CUSTOM) { expected.push_back(backend); }
    }

    const std::vector<PlanRequest> without = anira::backend::legacy_plan_requests(config, nullptr);
    ASSERT_EQ(without.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        SCOPED_TRACE("plan " + std::to_string(i));
        EXPECT_EQ(without[i].m_legacy_backend, expected[i]);
        EXPECT_EQ(without[i].m_source, Source::Roundtrip);
        EXPECT_EQ(without[i].m_backend, nullptr);
        EXPECT_EQ(without[i].m_missing_model, i > 0) << "the custom row has its model";
        EXPECT_EQ(without[i].m_model.m_engine, anira::backend::engine_of(expected[i]));
        EXPECT_EQ(without[i].m_model.m_instances, 2U);
        EXPECT_FALSE(without[i].m_model.m_session_exclusive);
        ASSERT_EQ(without[i].m_model.m_inputs.size(), 1U);
        EXPECT_EQ(without[i].m_model.m_inputs.at(0).m_dims, (std::vector<int64_t>{1, 1, k_block}));
        EXPECT_EQ(without[i].m_model.m_inputs.at(0).m_num_elements, k_block);
        EXPECT_TRUE(without[i].m_model.m_inputs.at(0).m_engine_name.empty());
    }
    EXPECT_EQ(without[0].m_model.m_path, "placeholder");
    EXPECT_EQ(without[0].m_model.m_engine, ANIRA_ENGINE_NONE);

    RecordingBackend custom(config);
    const std::vector<PlanRequest> with = anira::backend::legacy_plan_requests(config, &custom);
    ASSERT_EQ(with.size(), expected.size());
    EXPECT_EQ(with[0].m_source, Source::Legacy);
    EXPECT_EQ(with[0].m_backend, &custom);
    for (size_t i = 1; i < with.size(); ++i) {
        EXPECT_EQ(with[i].m_source, Source::Roundtrip);
        EXPECT_EQ(with[i].m_backend, nullptr);
    }
}

// A configuration with a model for every engine of the build: one BuiltIn row per model, in
// its order, then CUSTOM (the roundtrip, or the caller's backend).
TEST(Adapter, LegacyPlanRequestsOfEveryEngineOfTheBuildMatchTheDefaultTable) {
    std::vector<anira::ModelData> rows;
    for (const anira::InferenceBackend backend : every_backend()) {
        if (backend == anira::InferenceBackend::CUSTOM) { continue; }
        rows.emplace_back("model-" + std::to_string(static_cast<int>(backend)), backend);
    }
    if (rows.empty()) { GTEST_SKIP() << "an engine-free build: the custom-only case covers it"; }
    anira::InferenceConfig config(
        rows,
        std::vector<anira::TensorShape>{anira::TensorShape({{1, 1, k_block}}, {{1, 1, k_block}})},
        anira::ProcessingSpec({1}, {1}, {k_block}, {k_block}),
        10.F,
        /*warm_up=*/2,
        /*session_exclusive_processor=*/false,
        /*blocking_ratio=*/0.F,
        /*num_parallel_processors=*/3);

    const std::vector<PlanRequest> requests = anira::backend::legacy_plan_requests(config, nullptr);
    ASSERT_EQ(requests.size(), rows.size() + 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        SCOPED_TRACE("plan " + std::to_string(i));
        EXPECT_EQ(requests[i].m_legacy_backend, rows[i].m_backend);
        EXPECT_EQ(requests[i].m_source, Source::BuiltIn);
        EXPECT_FALSE(requests[i].m_missing_model);
        EXPECT_EQ(requests[i].m_model.m_engine, anira::backend::engine_of(rows[i].m_backend));
        EXPECT_NE(requests[i].m_model.m_engine, ANIRA_ENGINE_NONE);
        EXPECT_EQ(requests[i].m_model.m_path,
                  "model-" + std::to_string(static_cast<int>(rows[i].m_backend)));
        EXPECT_EQ(requests[i].m_model.m_bytes, nullptr);
        EXPECT_EQ(requests[i].m_model.m_instances, 3U);
        EXPECT_EQ(requests[i].m_model.m_warm_up, 2U);
    }
    EXPECT_EQ(requests.back().m_legacy_backend, anira::InferenceBackend::CUSTOM);
    EXPECT_EQ(requests.back().m_source, Source::Roundtrip);
    EXPECT_FALSE(requests.back().m_missing_model);
}

// ============================================================================================
// The legacy adapter
// ============================================================================================

TEST(Adapter, LegacyAdapterPreparesItsBackendAndPassesTheStructsBuffersThrough) {
    anira::InferenceConfig config = custom_only_config();
    RecordingBackend backend(config);
    anira::backend::LegacyAdapter adapter(backend);
    EXPECT_EQ(&adapter.backend(), &backend);
    adapter.prepare(gain_model());
    EXPECT_EQ(backend.m_prepares, 1);

    std::vector<anira::BufferF> inputs;
    inputs.emplace_back(1, 4);
    std::vector<anira::BufferF> outputs;
    outputs.emplace_back(1, 4);
    for (size_t n = 0; n < 4; ++n) { inputs[0].set_sample(0, n, static_cast<float>(n + 1)); }
    outputs[0].clear();
    const std::vector<anira_tensor> input_tensors = descriptors_over(inputs);
    std::vector<anira_tensor> output_tensors = descriptors_over(outputs);
    ChunkBuffers chunk{.m_inputs = &inputs, .m_outputs = &outputs};

    EXPECT_EQ(adapter.run(context_of(input_tensors, output_tensors), &chunk, false), ANIRA_OK);
    EXPECT_EQ(backend.m_calls, 1);
    EXPECT_EQ(backend.m_first_sample, 1.F);
    EXPECT_EQ(backend.m_input_seen, inputs[0].data()) << "the struct's own buffer";
    EXPECT_EQ(backend.m_output_seen, outputs[0].data());
    for (size_t n = 0; n < 4; ++n) {
        EXPECT_EQ(outputs[0].get_sample(0, n), static_cast<float>(n + 1));
    }
}

// A descriptor over other memory than the struct's buffer: the input is copied into the
// buffer ahead of the 2.x call, the output out of it behind the call; a descriptor over the
// buffer itself moves nothing.
TEST(Adapter, LegacyAdapterCopiesAForeignSlotInAndOutAroundTheCall) {
    anira::InferenceConfig config = custom_only_config();
    RecordingBackend backend(config);
    anira::backend::LegacyAdapter adapter(backend);
    adapter.prepare(gain_model());

    std::vector<anira::BufferF> inputs;
    inputs.emplace_back(1, 4);
    std::vector<anira::BufferF> outputs;
    outputs.emplace_back(1, 4);
    inputs[0].clear();
    outputs[0].clear();
    std::array<float, 4> foreign_in{5.F, 6.F, 7.F, 8.F};
    std::array<float, 4> foreign_out{};
    const std::array<int64_t, 2> shape{1, 4};
    std::vector<anira_tensor> input_tensors(1);
    std::vector<anira_tensor> output_tensors(1);
    anira_tensor_init_host(&input_tensors[0], foreign_in.data(), ANIRA_DTYPE_F32, 2, shape.data());
    anira_tensor_init_host(&output_tensors[0],
                           foreign_out.data(),
                           ANIRA_DTYPE_F32,
                           2,
                           shape.data());
    ChunkBuffers chunk{.m_inputs = &inputs, .m_outputs = &outputs};

    EXPECT_EQ(adapter.run(context_of(input_tensors, output_tensors), &chunk, false), ANIRA_OK);
    EXPECT_EQ(backend.m_first_sample, 5.F) << "the foreign input reached the struct's buffer";
    EXPECT_EQ(backend.m_input_seen, inputs[0].data()) << "the 2.x call ran over the struct";
    for (size_t n = 0; n < 4; ++n) {
        EXPECT_EQ(inputs[0].get_sample(0, n), foreign_in.at(n)) << "copied in";
        EXPECT_EQ(outputs[0].get_sample(0, n), foreign_in.at(n)) << "the pass-through";
        EXPECT_EQ(foreign_out.at(n), foreign_in.at(n)) << "copied out";
    }

    // The struct's own buffers again: the input keeps what the buffer holds.
    for (size_t n = 0; n < 4; ++n) { inputs[0].set_sample(0, n, 1.F); }
    const std::vector<anira_tensor> own_inputs = descriptors_over(inputs);
    std::vector<anira_tensor> own_outputs = descriptors_over(outputs);
    foreign_out.fill(0.F);
    EXPECT_EQ(adapter.run(context_of(own_inputs, own_outputs), &chunk, false), ANIRA_OK);
    EXPECT_EQ(backend.m_first_sample, 1.F);
    EXPECT_EQ(foreign_out.at(0), 0.F) << "nothing was copied out";
}

// A descriptor the adapter cannot read as the buffer's block is anira's own bug: the chunk
// fails instead of running on stale data; a throwing 2.x backend fails the chunk with ENGINE.
TEST(Adapter, LegacyAdapterFailsAChunkItCannotBindAndAThrowingBackend) {
    anira::InferenceConfig config = custom_only_config();
    RecordingBackend backend(config);
    anira::backend::LegacyAdapter adapter(backend);
    adapter.prepare(gain_model());

    std::vector<anira::BufferF> inputs;
    inputs.emplace_back(1, 4);
    std::vector<anira::BufferF> outputs;
    outputs.emplace_back(1, 4);
    std::vector<anira_tensor> input_tensors = descriptors_over(inputs);
    std::vector<anira_tensor> output_tensors = descriptors_over(outputs);
    ChunkBuffers chunk{.m_inputs = &inputs, .m_outputs = &outputs};

    input_tensors[0].dtype = ANIRA_DTYPE_I16;
    EXPECT_EQ(adapter.run(context_of(input_tensors, output_tensors), &chunk, false),
              ANIRA_ERROR_INTERNAL);
    EXPECT_EQ(backend.m_calls, 0) << "the 2.x call never ran";
    input_tensors[0].dtype = ANIRA_DTYPE_F32;

    // No buffers at all: the legacy adapter has nothing to call over.
    EXPECT_EQ(adapter.run(context_of(input_tensors, output_tensors), nullptr, false),
              ANIRA_ERROR_INVALID_STATE);

    backend.m_throws = true;
    EXPECT_EQ(adapter.run(context_of(input_tensors, output_tensors), &chunk, false),
              ANIRA_ERROR_ENGINE);
}

// The roundtrip: a BackendBase built from the configuration, the 2.x default processor, owned
// by the adapter.
TEST(Adapter, LegacyAdapterOverTheRoundtripPassesTheBlockThrough) {
    anira::InferenceConfig config = custom_only_config();
    anira::backend::LegacyAdapter adapter(config);
    adapter.prepare(gain_model());
    std::vector<anira::BufferF> inputs;
    inputs.emplace_back(1, 4);
    std::vector<anira::BufferF> outputs;
    outputs.emplace_back(1, 4);
    for (size_t n = 0; n < 4; ++n) { inputs[0].set_sample(0, n, static_cast<float>(n + 1)); }
    outputs[0].clear();
    const std::vector<anira_tensor> input_tensors = descriptors_over(inputs);
    std::vector<anira_tensor> output_tensors = descriptors_over(outputs);
    ChunkBuffers chunk{.m_inputs = &inputs, .m_outputs = &outputs};
    EXPECT_EQ(adapter.run(context_of(input_tensors, output_tensors), &chunk, false), ANIRA_OK);
    for (size_t n = 0; n < 4; ++n) {
        EXPECT_EQ(outputs[0].get_sample(0, n), static_cast<float>(n + 1));
    }
}

// A 2.x backend is called as concurrently as the scheduler dispatches, as it always was: the
// legacy adapter takes no instance claim, so two callers meet inside process on a record of
// one instance.
TEST(Adapter, LegacyAdapterTakesNoInstanceClaim) {
    anira::InferenceConfig config = custom_only_config();
    MeetingBackend backend(config, 2);
    anira::backend::LegacyAdapter adapter(backend);
    adapter.prepare(gain_model(1));
    std::vector<anira::BufferF> inputs;
    inputs.emplace_back(1, 4);
    std::vector<anira::BufferF> outputs;
    outputs.emplace_back(1, 4);
    const std::vector<anira_tensor> input_tensors = descriptors_over(inputs);
    std::vector<anira_tensor> output_tensors = descriptors_over(outputs);
    ChunkBuffers chunk{.m_inputs = &inputs, .m_outputs = &outputs};
    const anira_engine_ctx ctx = context_of(input_tensors, output_tensors);
    std::thread other([&adapter, &ctx, &chunk]() { adapter.run(ctx, &chunk, false); });
    EXPECT_EQ(adapter.run(ctx, &chunk, false), ANIRA_OK);
    other.join();
    EXPECT_TRUE(backend.m_met.load()) << "the second caller waited outside";
}

// ============================================================================================
// The built-in adapters of this commit
// ============================================================================================

TEST(Adapter, NoBuiltInAdapterOfTheDescriptorShapeYet) {
    EXPECT_EQ(anira::backend::make_builtin_adapter(ANIRA_ENGINE_NONE), nullptr);
    EXPECT_EQ(anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME), nullptr);
    EXPECT_EQ(anira::backend::engine_of(anira::InferenceBackend::CUSTOM), ANIRA_ENGINE_NONE);
}
