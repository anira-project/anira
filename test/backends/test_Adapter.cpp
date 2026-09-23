// The engine room's interface (src/backends/Adapter.h) and the legacy adapter over the 2.x
// virtual, driven directly: no core, no session. The claim loop under several threads, the
// float32 helpers of the built-in adapters, the 2.x plan table as requests, and the copies the
// legacy adapter makes around a descriptor that names other memory than the struct's buffer.

#include <anira/CoreConfig.h>
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
#include <utility>
#include <vector>

#include "backends/Adapter.h"
#include "backends/Adapters.h"
#include "backends/LegacyAdapter.h"
#include "gtest/gtest.h"
#include "utils/StatusError.h"

namespace {

using anira::backend::Adapter;
using anira::backend::Bindings;
using anira::backend::ChunkBuffers;
using anira::backend::EngineTensor;
using anira::backend::ExtentRule;
using anira::backend::Model;
using anira::backend::PlanRequest;
using anira::backend::SlotBinding;
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

// A record of one float32 tensor of `dims` that the entry's tensors record names `engine_name`.
TensorInfo named_tensor(const std::string& name,
                        const std::string& engine_name,
                        std::vector<int64_t> dims) {
    TensorInfo tensor = f32_tensor(name, std::move(dims));
    tensor.m_engine_name = engine_name;
    return tensor;
}

// One float32 tensor of an engine's side.
EngineTensor engine_f32(const std::string& name, std::vector<int64_t> dims) {
    EngineTensor tensor;
    tensor.m_name = name;
    tensor.m_dims = std::move(dims);
    tensor.m_dtype = ANIRA_DTYPE_F32;
    tensor.m_type_word = "float32";
    return tensor;
}

// The StatusError a call throws; a failure of the case when it throws nothing else.
template <typename Call>
anira::StatusError status_error_of(Call&& call) {
    try {
        call();
    } catch (const anira::StatusError& error) { return error; }
    ADD_FAILURE() << "no StatusError was thrown";
    return {ANIRA_OK, ""};
}

// A descriptor over `data` as the packed float32 block of `dims`.
anira_tensor descriptor_over(std::vector<float>& data, const std::vector<int64_t>& dims) {
    anira_tensor tensor{};
    anira_tensor_init_host(&tensor,
                           data.data(),
                           ANIRA_DTYPE_F32,
                           static_cast<uint32_t>(dims.size()),
                           dims.data());
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
    const anira::InferenceConfig config(
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
    EXPECT_EQ(adapter.wrapped(), &backend);
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
// The binding rule and the check
// ============================================================================================

// A slot the entry's tensors record names binds to that tensor; a slot without a record binds
// to the tensor of its canonical name where the side has one, else to the tensor at its own
// position; the engine's order does not matter to a named slot.
TEST(Adapter, BindSlotsByRecordNameByCanonicalNameAndByPosition) {
    const std::vector<TensorInfo> slots{named_tensor("audio_in", "data", {1, 1, k_block}),
                                        f32_tensor("gain", {1}),
                                        f32_tensor("extra", {1})};
    const std::vector<std::string> engine{"data", "gain", "other"};
    const std::vector<SlotBinding> bindings =
        anira::backend::bind_slots(slots, engine, engine.size(), "test", "input");
    ASSERT_EQ(bindings.size(), 3U);
    EXPECT_EQ(bindings[0].m_index, 0U);
    EXPECT_EQ(bindings[0].m_binding, ANIRA_BINDING_NAME) << "the record's name";
    EXPECT_EQ(bindings[1].m_index, 1U);
    EXPECT_EQ(bindings[1].m_binding, ANIRA_BINDING_NAME) << "the canonical name";
    EXPECT_EQ(bindings[2].m_index, 2U);
    EXPECT_EQ(bindings[2].m_binding, ANIRA_BINDING_POSITION) << "neither";

    const std::vector<std::string> reordered{"gain", "data"};
    const std::vector<TensorInfo> two{named_tensor("audio_in", "data", {1, 1, k_block}),
                                      f32_tensor("gain", {1})};
    const std::vector<SlotBinding> by_name =
        anira::backend::bind_slots(two, reordered, 2, "test", "input");
    EXPECT_EQ(by_name[0].m_index, 1U);
    EXPECT_EQ(by_name[1].m_index, 0U);

    // A side without names: every slot by position.
    const std::vector<std::string> unnamed{"", ""};
    const std::vector<SlotBinding> positional =
        anira::backend::bind_slots({f32_tensor("a", {1}), f32_tensor("b", {1})},
                                   unnamed,
                                   2,
                                   "test",
                                   "output");
    EXPECT_EQ(positional[0].m_index, 0U);
    EXPECT_EQ(positional[0].m_binding, ANIRA_BINDING_POSITION);
    EXPECT_EQ(positional[1].m_index, 1U);
    EXPECT_EQ(positional[1].m_binding, ANIRA_BINDING_POSITION);

    // The 2.x path: no names on either side.
    const std::vector<SlotBinding> legacy =
        anira::backend::bind_slots({f32_tensor("", {1})}, unnamed, 1, "test", "input");
    EXPECT_EQ(legacy[0].m_index, 0U);
    EXPECT_EQ(legacy[0].m_binding, ANIRA_BINDING_POSITION);
}

TEST(Adapter, BindSlotsRefusals) {
    // A record name the side lacks: CONFIG listing the names.
    {
        const anira::StatusError error = status_error_of([] {
            anira::backend::bind_slots({named_tensor("in", "ghost", {1})},
                                       {"data", "gain"},
                                       2,
                                       "onnxruntime",
                                       "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("onnxruntime: input slot 0 'in'"), std::string::npos) << message;
        EXPECT_NE(message.find("'ghost'"), std::string::npos) << message;
        EXPECT_NE(message.find("'data', 'gain'"), std::string::npos) << message;
    }
    // A record name on a side without names.
    {
        const anira::StatusError error = status_error_of([] {
            anira::backend::bind_slots({named_tensor("out", "y", {1})},
                                       {""},
                                       1,
                                       "libtorch",
                                       "output");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("the side binds by position alone"),
                  std::string::npos)
            << error.what();
    }
    // A position the engine has no tensor at.
    {
        const anira::StatusError error = status_error_of([] {
            anira::backend::bind_slots({f32_tensor("a", {1}), f32_tensor("b", {1})},
                                       {"x"},
                                       1,
                                       "test",
                                       "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("binds by position, and the test model has 1"),
                  std::string::npos)
            << error.what();
    }
    // An engine tensor no slot binds.
    {
        const anira::StatusError error = status_error_of([] {
            anira::backend::bind_slots({f32_tensor("a", {1})}, {"x", "y"}, 2, "test", "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("'y' (index 1) is bound by no slot"),
                  std::string::npos)
            << error.what();
    }
    // The same engine tensor at or above `required` may stay unbound (a LibTorch argument
    // with a default value).
    EXPECT_NO_THROW(
        anira::backend::bind_slots({f32_tensor("a", {1})}, {"x", "y"}, 1, "test", "input"));
    // Two slots on one engine tensor: one by name, one by position.
    {
        const anira::StatusError error = status_error_of([] {
            anira::backend::bind_slots({f32_tensor("p", {1}), f32_tensor("x", {1})},
                                       {"x", "y"},
                                       2,
                                       "test",
                                       "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("slots 0 'p' (by position) and 1 'x' (by name) both bind"),
                  std::string::npos)
            << message;
    }
}

TEST(Adapter, CheckEngineTensorComparesDtypeRankAndEveryStaticExtent) {
    const TensorInfo slot = f32_tensor("audio_in", {1, 1, k_block});
    EXPECT_NO_THROW(anira::backend::check_engine_tensor(slot,
                                                        engine_f32("data", {1, 1, k_block}),
                                                        ExtentRule::Exact,
                                                        "test",
                                                        "input"));
    EXPECT_NO_THROW(anira::backend::check_engine_tensor(slot,
                                                        engine_f32("data", {1, 1, -1}),
                                                        ExtentRule::Exact,
                                                        "test",
                                                        "input"))
        << "a dynamic extent matches anything";
    {
        const anira::StatusError error = status_error_of([&slot] {
            anira::backend::check_engine_tensor(slot,
                                                engine_f32("data", {1, k_block}),
                                                ExtentRule::Exact,
                                                "test",
                                                "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("input tensor 'audio_in' bound to the model's 'data'"),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find("the ranks differ"), std::string::npos) << message;
    }
    {
        const anira::StatusError error = status_error_of([&slot] {
            anira::backend::check_engine_tensor(slot,
                                                engine_f32("data", {1, 1, 256}),
                                                ExtentRule::Exact,
                                                "test",
                                                "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("[1, 1, 256]"), std::string::npos) << message;
        EXPECT_NE(message.find("[1, 1, 512]"), std::string::npos) << message;
        EXPECT_NE(message.find("axis 2: 256 against 512"), std::string::npos) << message;
    }
    {
        EngineTensor int16 = engine_f32("data", {1, 1, k_block});
        int16.m_dtype = ANIRA_DTYPE_I16;
        int16.m_type_word = "int16";
        const anira::StatusError error = status_error_of([&slot, &int16] {
            anira::backend::check_engine_tensor(slot, int16, ExtentRule::Exact, "test", "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("element type is int16"), std::string::npos)
            << error.what();
    }
    // The upper-bound rule of a planned engine: at or below the bound matches, above does not.
    EXPECT_NO_THROW(anira::backend::check_engine_tensor(slot,
                                                        engine_f32("", {1, 1, 65536}),
                                                        ExtentRule::UpperBound,
                                                        "executorch",
                                                        "input"));
    {
        const anira::StatusError error = status_error_of([&slot] {
            anira::backend::check_engine_tensor(slot,
                                                engine_f32("", {1, 1, 256}),
                                                ExtentRule::UpperBound,
                                                "executorch",
                                                "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("bound to the model's unnamed input"), std::string::npos) << message;
        EXPECT_NE(message.find("256 is the planned upper bound of 512"), std::string::npos)
            << message;
    }
    {
        const anira::StatusError error = status_error_of([&slot] {
            anira::backend::check_engine_tensor(slot,
                                                engine_f32("data", {1, 1, 65536}),
                                                ExtentRule::Exact,
                                                "test",
                                                "input");
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG) << "exact means exact";
    }
}

TEST(Adapter, BindSideBindsThenChecksAndBindingsOfReports) {
    const std::vector<TensorInfo> slots{f32_tensor("audio_in", {1, 1, k_block}),
                                        f32_tensor("gain", {1})};
    const std::vector<EngineTensor> engine{engine_f32("data", {1, 1, -1}), engine_f32("gain", {1})};
    const std::vector<SlotBinding> inputs =
        anira::backend::bind_side(slots, engine, 2, ExtentRule::Exact, "test", "input");
    EXPECT_EQ(inputs[0].m_binding, ANIRA_BINDING_POSITION);
    EXPECT_EQ(inputs[1].m_binding, ANIRA_BINDING_NAME);
    const std::vector<EngineTensor> wrong{engine_f32("data", {1, 1, 256}), engine_f32("gain", {1})};
    const anira::StatusError error = status_error_of([&slots, &wrong] {
        anira::backend::bind_side(slots, wrong, 2, ExtentRule::Exact, "test", "input");
    });
    EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
    const Bindings report =
        anira::backend::bindings_of(inputs,
                                    {SlotBinding{.m_index = 0, .m_binding = ANIRA_BINDING_NAME}});
    EXPECT_EQ(report.m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_NAME}));
    EXPECT_EQ(report.m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_NAME}));
}

// Every slot reports position until the adapter's prepare says otherwise: the legacy adapter's
// 2.x backend binds by position.
TEST(Adapter, PrepareReportsPositionForEverySlotUntilTheAdapterSaysOtherwise) {
    RecordingAdapter adapter;
    EXPECT_TRUE(adapter.bindings().m_inputs.empty());
    adapter.prepare(gain_model());
    EXPECT_EQ(adapter.bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter.bindings().m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    anira::InferenceConfig config = custom_only_config();
    RecordingBackend backend(config);
    anira::backend::LegacyAdapter legacy(backend);
    legacy.prepare(gain_model());
    EXPECT_EQ(legacy.bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
}

// ============================================================================================
// The built-in adapters
// ============================================================================================

// Every engine of the build has an adapter (unprepared until prepare loads a model); an engine
// the build does not carry, and the custom engine, have none.
TEST(Adapter, MakeBuiltInAdapterAnswersTheEnginesOfTheBuild) {
    EXPECT_EQ(anira::backend::make_builtin_adapter(ANIRA_ENGINE_NONE), nullptr);
    EXPECT_EQ(anira::backend::engine_of(anira::InferenceBackend::CUSTOM), ANIRA_ENGINE_NONE);
    for (const anira::InferenceBackend backend : every_backend()) {
        if (backend == anira::InferenceBackend::CUSTOM) { continue; }
        const anira_engine engine = anira::backend::engine_of(backend);
        ASSERT_NE(engine, ANIRA_ENGINE_NONE);
        const std::shared_ptr<Adapter> adapter = anira::backend::make_builtin_adapter(engine);
        ASSERT_NE(adapter, nullptr) << "engine " << static_cast<int>(engine);
        EXPECT_FALSE(adapter->prepared());
    }
#ifndef USE_ONNXRUNTIME
    EXPECT_EQ(anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME), nullptr);
#endif
}

// ============================================================================================
// ONNX Runtime
// ============================================================================================

#ifdef USE_ONNXRUNTIME

namespace {

std::string gain_onnx() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.onnx";
}

std::string accumulator_onnx() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/StatefulAccumulatorNetwork/models/"
        "stateful_accumulator_network_stereo.onnx";
}

// The bundled gain model as gain.model.json describes it: the canonical names, no records.
// The graph's names are data / gain / processed_data / peak.
Model onnx_gain_model(uint32_t warm_up = 1) {
    Model model;
    model.m_engine = ANIRA_ENGINE_ONNXRUNTIME;
    model.m_path = gain_onnx();
    model.m_inputs.push_back(f32_tensor("audio_in", {1, 1, k_block}));
    model.m_inputs.push_back(f32_tensor("gain", {1}));
    model.m_outputs.push_back(f32_tensor("audio_out", {1, 1, k_block}));
    model.m_outputs.push_back(f32_tensor("gain_out", {1}));
    model.m_warm_up = warm_up;
    return model;
}

// The bundled accumulator as stateful_accumulator.model.json describes it: the canonical names
// are the graph's (state_in / data / processed_data / state_out).
Model onnx_accumulator_model() {
    Model model;
    model.m_engine = ANIRA_ENGINE_ONNXRUNTIME;
    model.m_path = accumulator_onnx();
    model.m_inputs.push_back(f32_tensor("state_in", {1, 2, 2}));
    model.m_inputs.push_back(f32_tensor("data", {1, 2, 64}));
    model.m_outputs.push_back(f32_tensor("processed_data", {1, 2, 64}));
    model.m_outputs.push_back(f32_tensor("state_out", {1, 2, 2}));
    model.m_warm_up = 1;
    return model;
}

}  // namespace

// The gain: audio_in and the two outputs bind by position (the graph names them data,
// processed_data and peak), the gain input by its canonical name; the check passes on the
// graph's [1, 1, -1] and [1]; the run applies the gain over the descriptors' memory, in place.
TEST(AdapterOnnxRuntime, TheGainBindsByPositionAndByNameAndRunsOverTheDescriptors) {
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
    ASSERT_NE(adapter, nullptr);
    adapter->prepare(onnx_gain_model());
    ASSERT_TRUE(adapter->prepared());
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_NAME}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));

    std::vector<float> audio_in(k_block);
    for (size_t n = 0; n < k_block; ++n) { audio_in[n] = static_cast<float>(n) / k_block; }
    std::vector<float> gain{0.5F};
    std::vector<float> audio_out(k_block, -1.F);
    std::vector<float> gain_out{-1.F};
    const std::vector<anira_tensor> inputs{descriptor_over(audio_in, {1, 1, k_block}),
                                           descriptor_over(gain, {1})};
    std::vector<anira_tensor> outputs{descriptor_over(audio_out, {1, 1, k_block}),
                                      descriptor_over(gain_out, {1})};
    for (int run = 0; run < 2; ++run) {
        ASSERT_EQ(adapter->run(context_of(inputs, outputs), nullptr, false), ANIRA_OK);
        for (size_t n = 0; n < k_block; ++n) {
            ASSERT_EQ(audio_out[n], 0.5F * audio_in[n]) << "sample " << n;
        }
        EXPECT_GT(gain_out[0], 0.F) << "the peak landed in the caller's memory";
    }
}

// The accumulator's canonical names are the graph's: every slot binds by name, whatever order
// the slots are declared in, and the closed form of one block holds on a zero state.
TEST(AdapterOnnxRuntime, TheAccumulatorBindsByNameOnEverySlotInAnyOrder) {
    Model model = onnx_accumulator_model();
    std::swap(model.m_inputs[0], model.m_inputs[1]);    // data first, state_in second
    std::swap(model.m_outputs[0], model.m_outputs[1]);  // state_out first
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
    adapter->prepare(model);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));

    std::vector<float> data(128);
    for (size_t n = 0; n < data.size(); ++n) { data[n] = static_cast<float>(n % 7); }
    std::vector<float> state_in(4, 0.F);
    std::vector<float> state_out(4, -1.F);
    std::vector<float> processed(128, -1.F);
    const std::vector<anira_tensor> inputs{descriptor_over(data, {1, 2, 64}),
                                           descriptor_over(state_in, {1, 2, 2})};
    std::vector<anira_tensor> outputs{descriptor_over(state_out, {1, 2, 2}),
                                      descriptor_over(processed, {1, 2, 64})};
    ASSERT_EQ(adapter->run(context_of(inputs, outputs), nullptr, false), ANIRA_OK);
    for (size_t n = 0; n < data.size(); ++n) { ASSERT_EQ(processed[n], data[n]) << "sample " << n; }
    float sum0 = 0.F;
    float sum1 = 0.F;
    for (size_t n = 0; n < 64; ++n) {
        sum0 += data[n];
        sum1 += data[64 + n];
    }
    EXPECT_EQ(state_out[0], sum0);
    EXPECT_EQ(state_out[1], 1.F);
    EXPECT_EQ(state_out[2], sum1);
    EXPECT_EQ(state_out[3], 1.F);
}

TEST(AdapterOnnxRuntime, PrepareRefusals) {
    // A record naming a tensor the graph lacks: CONFIG listing the graph's names.
    {
        Model model = onnx_gain_model();
        model.m_inputs[0].m_engine_name = "ghost";
        const std::shared_ptr<Adapter> adapter =
            anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("onnxruntime: input slot 0 'audio_in'"), std::string::npos)
            << message;
        EXPECT_NE(message.find("'ghost'"), std::string::npos) << message;
        EXPECT_NE(message.find("'data', 'gain'"), std::string::npos) << message;
        EXPECT_FALSE(adapter->prepared());
    }
    // A shape the graph does not have: CONFIG naming the slot, the graph tensor and both shapes.
    {
        Model model = onnx_gain_model();
        model.m_inputs[1] = f32_tensor("gain", {2});
        const std::shared_ptr<Adapter> adapter =
            anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("input tensor 'gain' bound to the model's 'gain'"),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find("axis 0: 1 against 2"), std::string::npos) << message;
    }
    // A rank the graph does not have.
    {
        Model model = onnx_gain_model();
        model.m_outputs[0] = f32_tensor("audio_out", {k_block});
        const std::shared_ptr<Adapter> adapter =
            anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("the ranks differ"), std::string::npos)
            << error.what();
    }
    // Bytes ONNX Runtime cannot parse: MODEL_LOAD with its text and the location.
    {
        const std::string junk = "definitely not a model";
        Model model = onnx_gain_model();
        model.m_path.clear();
        model.m_bytes = junk.data();
        model.m_num_bytes = junk.size();
        const std::shared_ptr<Adapter> adapter =
            anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_MODEL_LOAD);
        EXPECT_NE(std::string(error.what()).find("onnxruntime: memory: "), std::string::npos)
            << error.what();
    }
    // A missing file: NO_SUCH_FILE, before ONNX Runtime sees the path.
    {
        Model model = onnx_gain_model();
        model.m_path = "this/model/does/not/exist.onnx";
        const std::shared_ptr<Adapter> adapter =
            anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_NO_SUCH_FILE);
    }
    // An int16 record: CONFIG naming the engine and the tensor, before the file is opened.
    {
        Model model = onnx_gain_model();
        model.m_path = "this/model/does/not/exist.onnx";
        model.m_inputs[0].m_dtype = ANIRA_DTYPE_I16;
        const std::shared_ptr<Adapter> adapter =
            anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("onnxruntime: input tensor 'audio_in'"),
                  std::string::npos)
            << error.what();
    }
}

// A descriptor the adapter cannot hand the engine fails the call with INVALID_ARGUMENT, and
// nothing is written; a shorter or a foreign-typed block are the cases.
TEST(AdapterOnnxRuntime, ADescriptorTheAdapterCannotBindFailsTheCall) {
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_ONNXRUNTIME);
    adapter->prepare(onnx_gain_model(0));
    std::vector<float> audio_in(k_block, 0.25F);
    std::vector<float> gain{0.5F};
    std::vector<float> audio_out(k_block, -1.F);
    std::vector<float> gain_out{-1.F};
    std::vector<anira_tensor> inputs{descriptor_over(audio_in, {1, 1, k_block / 2}),
                                     descriptor_over(gain, {1})};
    std::vector<anira_tensor> outputs{descriptor_over(audio_out, {1, 1, k_block}),
                                      descriptor_over(gain_out, {1})};
    EXPECT_EQ(adapter->run(context_of(inputs, outputs), nullptr, false),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "half the elements";
    EXPECT_EQ(audio_out[0], -1.F) << "nothing ran";
    inputs[0] = descriptor_over(audio_in, {1, 1, k_block});
    inputs[0].dtype = ANIRA_DTYPE_I16;
    EXPECT_EQ(adapter->run(context_of(inputs, outputs), nullptr, false),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "not float32";
    inputs[0].dtype = ANIRA_DTYPE_F32;
    anira_engine_ctx short_ctx = context_of(inputs, outputs);
    short_ctx.num_inputs = 1;
    EXPECT_EQ(adapter->run(short_ctx, nullptr, false), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a context of another slot count";
    EXPECT_EQ(adapter->run(context_of(inputs, outputs), nullptr, false), ANIRA_OK);
    EXPECT_EQ(audio_out[0], 0.125F);
}

#endif  // USE_ONNXRUNTIME

// ============================================================================================
// TensorFlow Lite and LiteRT: the bundled exports through their signatures
// ============================================================================================

#if defined(USE_TFLITE) || defined(USE_LITERT)

namespace {

std::string gain_tflite() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.tflite";
}

std::string accumulator_tflite() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/StatefulAccumulatorNetwork/models/"
        "stateful_accumulator_network_stereo.tflite";
}

std::string guitarlstm_tflite() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/hybrid-nn/GuitarLSTM/tensorflow-version/models/model_0/GuitarLSTM-256.tflite";
}

// The bundled gain export as gain.model.json describes it for its TensorFlow rows: the static
// gain re-viewed to the export's rank 3 by the row's layout (the engine dims here), no names.
// The signature's keys are args_0 / args_0_1 / output_0 / output_1.
Model tensorflow_gain_model(anira_engine engine, uint32_t warm_up = 1) {
    Model model;
    model.m_engine = engine;
    model.m_path = gain_tflite();
    model.m_inputs.push_back(f32_tensor("audio_in", {1, 1, k_block}));
    model.m_inputs.push_back(f32_tensor("gain", {1, 1, 1}));
    model.m_outputs.push_back(f32_tensor("audio_out", {1, 1, k_block}));
    model.m_outputs.push_back(f32_tensor("gain_out", {1}));
    model.m_warm_up = warm_up;
    return model;
}

// The bundled accumulator export as stateful_accumulator.model.json describes it: the
// canonical names (none a signature key), the state first in the inputs and last in the
// outputs.
Model tensorflow_accumulator_model(anira_engine engine) {
    Model model;
    model.m_engine = engine;
    model.m_path = accumulator_tflite();
    model.m_inputs.push_back(f32_tensor("state_in", {1, 2, 2}));
    model.m_inputs.push_back(f32_tensor("data", {1, 2, 64}));
    model.m_outputs.push_back(f32_tensor("processed_data", {1, 2, 64}));
    model.m_outputs.push_back(f32_tensor("state_out", {1, 2, 2}));
    model.m_warm_up = 1;
    return model;
}

// One block of the gain through `adapter`: the ramp halved lands in the caller's memory.
void expect_gain_of_one_half(Adapter& adapter) {
    std::vector<float> audio_in(k_block);
    for (size_t n = 0; n < k_block; ++n) { audio_in[n] = static_cast<float>(n) / k_block; }
    std::vector<float> gain{0.5F};
    std::vector<float> audio_out(k_block, -1.F);
    std::vector<float> gain_out{-1.F};
    const std::vector<anira_tensor> inputs{descriptor_over(audio_in, {1, 1, k_block}),
                                           descriptor_over(gain, {1, 1, 1})};
    std::vector<anira_tensor> outputs{descriptor_over(audio_out, {1, 1, k_block}),
                                      descriptor_over(gain_out, {1})};
    ASSERT_EQ(adapter.run(context_of(inputs, outputs), nullptr, false), ANIRA_OK);
    for (size_t n = 0; n < k_block; ++n) {
        ASSERT_EQ(audio_out[n], 0.5F * audio_in[n]) << "sample " << n;
    }
    EXPECT_GT(gain_out[0], 0.F) << "the peak landed in the caller's memory";
}

// One block of the accumulator on a zero state through `adapter`: the closed form of one block.
void expect_accumulator_closed_form(Adapter& adapter) {
    std::vector<float> data(128);
    for (size_t n = 0; n < data.size(); ++n) { data[n] = static_cast<float>(n % 7); }
    std::vector<float> state_in(4, 0.F);
    std::vector<float> state_out(4, -1.F);
    std::vector<float> processed(128, -1.F);
    const std::vector<anira_tensor> inputs{descriptor_over(state_in, {1, 2, 2}),
                                           descriptor_over(data, {1, 2, 64})};
    std::vector<anira_tensor> outputs{descriptor_over(processed, {1, 2, 64}),
                                      descriptor_over(state_out, {1, 2, 2})};
    ASSERT_EQ(adapter.run(context_of(inputs, outputs), nullptr, true), ANIRA_OK)
        << "the first inference of a stream: reset, then process";
    for (size_t n = 0; n < data.size(); ++n) { ASSERT_EQ(processed[n], data[n]) << "sample " << n; }
    float sum0 = 0.F;
    float sum1 = 0.F;
    for (size_t n = 0; n < 64; ++n) {
        sum0 += data[n];
        sum1 += data[64 + n];
    }
    EXPECT_EQ(state_out[0], sum0);
    EXPECT_EQ(state_out[1], 1.F);
    EXPECT_EQ(state_out[2], sum1);
    EXPECT_EQ(state_out[3], 1.F);
}

}  // namespace

#endif  // USE_TFLITE || USE_LITERT

#ifdef USE_TFLITE

// The gain through the signature runner: audio_in and gain bind by position to args_0 and
// args_0_1 (the runner lists the keys in key order), the gain at the export's rank 3 through
// the row's layout, args_0 resized to the pinned window; the outputs bind by position to
// output_0 (the stream) and output_1 (the peak), the file's own order [peak, stream]
// notwithstanding. Without the layout the rank-1 gain is refused against the export's rank 3.
TEST(AdapterTFLite, TheGainBindsByPositionThroughTheSignatureRunner) {
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE);
    ASSERT_NE(adapter, nullptr);
    adapter->prepare(tensorflow_gain_model(ANIRA_ENGINE_TFLITE));
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    expect_gain_of_one_half(*adapter);
    expect_gain_of_one_half(*adapter);

    Model rank_one = tensorflow_gain_model(ANIRA_ENGINE_TFLITE);
    rank_one.m_inputs[1] = f32_tensor("gain", {1});
    const anira::StatusError error = status_error_of([&rank_one] {
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE)->prepare(rank_one);
    });
    EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
    const std::string message = error.what();
    EXPECT_NE(message.find("tflite: input tensor 'gain' bound to the model's 'args_0_1'"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("[1, 1, 1]"), std::string::npos) << message;
    EXPECT_NE(message.find("the ranks differ"), std::string::npos) << message;
}

// The accumulator through the signature runner binds by position in the runner's key order,
// which is the declared one, and follows the closed form; reset before the first inference
// of a stream re-initialises the variable tensors, of which the export has none.
TEST(AdapterTFLite, TheAccumulatorBindsByPositionThroughTheSignatureRunner) {
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE);
    adapter->prepare(tensorflow_accumulator_model(ANIRA_ENGINE_TFLITE));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    expect_accumulator_closed_form(*adapter);

    // A record naming the keys binds the same slots by name.
    Model named = tensorflow_accumulator_model(ANIRA_ENGINE_TFLITE);
    named.m_outputs[0].m_engine_name = "output_0";
    named.m_outputs[1].m_engine_name = "output_1";
    const std::shared_ptr<Adapter> by_name =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE);
    by_name->prepare(named);
    EXPECT_EQ(by_name->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    expect_accumulator_closed_form(*by_name);

    // A record naming a key the signature lacks: CONFIG listing the keys.
    named.m_outputs[0].m_engine_name = "ghost";
    const anira::StatusError error = status_error_of(
        [&named] { anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE)->prepare(named); });
    EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::string(error.what()).find("'output_0', 'output_1'"), std::string::npos)
        << error.what();
}

// A file without a signature (the GuitarLSTM export) runs through the interpreter itself: the
// slots bind by position in the interpreter's order, the tensor names (args_0, Identity) being
// the names a record could use.
TEST(AdapterTFLite, AFileWithoutASignatureRunsThroughTheInterpreter) {
    Model model;
    model.m_engine = ANIRA_ENGINE_TFLITE;
    model.m_path = guitarlstm_tflite();
    model.m_inputs.push_back(f32_tensor("audio_in", {256, 150, 1}));
    model.m_outputs.push_back(f32_tensor("audio_out", {256, 1}));
    model.m_warm_up = 1;
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE);
    adapter->prepare(model);
    EXPECT_EQ(adapter->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    std::vector<float> audio_in(256 * 150, 0.1F);
    std::vector<float> audio_out(256, 0.F);
    const std::vector<anira_tensor> inputs{descriptor_over(audio_in, {256, 150, 1})};
    std::vector<anira_tensor> outputs{descriptor_over(audio_out, {256, 1})};
    ASSERT_EQ(adapter->run(context_of(inputs, outputs), nullptr, false), ANIRA_OK);
    bool written = false;
    for (const float sample : audio_out) { written = written || sample != 0.F; }
    EXPECT_TRUE(written);

    Model named = model;
    named.m_inputs[0].m_engine_name = "args_0";
    named.m_outputs[0].m_engine_name = "Identity";
    const std::shared_ptr<Adapter> by_name =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_TFLITE);
    by_name->prepare(named);
    EXPECT_EQ(by_name->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_NAME}));
    EXPECT_EQ(by_name->bindings().m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_NAME}));
}

#endif  // USE_TFLITE

#ifdef USE_LITERT

// LiteRT lists the gain export's signature outputs in the file's order, [output_1 (the peak),
// output_0 (the stream)]: a positional binding of the declared order would put the stream on
// the peak, which the shape check refuses at prepare with both shapes; the names of the file's
// litert row bind the two outputs to their keys, and the gain runs at the export's rank 3.
TEST(AdapterLiteRt, TheGainNeedsItsOutputsNamedAndRunsAtTheExportsRank) {
    Model positional = tensorflow_gain_model(ANIRA_ENGINE_LITERT);
    const anira::StatusError error = status_error_of([&positional] {
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_LITERT)->prepare(positional);
    });
    EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
    const std::string message = error.what();
    EXPECT_NE(message.find("litert: output tensor 'audio_out' bound to the model's 'output_1'"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("[1]"), std::string::npos) << message;
    EXPECT_NE(message.find("[1, 1, 512]"), std::string::npos) << message;

    Model named = tensorflow_gain_model(ANIRA_ENGINE_LITERT);
    named.m_outputs[0].m_engine_name = "output_0";
    named.m_outputs[1].m_engine_name = "output_1";
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_LITERT);
    ASSERT_NE(adapter, nullptr);
    adapter->prepare(named);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    expect_gain_of_one_half(*adapter);
    expect_gain_of_one_half(*adapter);
}

// The accumulator on LiteRT: the same reversal on the output side, refused by position and
// bound by the keys of the file's litert row; the closed form holds.
TEST(AdapterLiteRt, TheAccumulatorBindsItsOutputsByName) {
    Model positional = tensorflow_accumulator_model(ANIRA_ENGINE_LITERT);
    const anira::StatusError error = status_error_of([&positional] {
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_LITERT)->prepare(positional);
    });
    EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::string(error.what()).find("'processed_data' bound to the model's 'output_1'"),
              std::string::npos)
        << error.what();

    Model named = tensorflow_accumulator_model(ANIRA_ENGINE_LITERT);
    named.m_outputs[0].m_engine_name = "output_0";
    named.m_outputs[1].m_engine_name = "output_1";
    const std::shared_ptr<Adapter> adapter =
        anira::backend::make_builtin_adapter(ANIRA_ENGINE_LITERT);
    adapter->prepare(named);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    expect_accumulator_closed_form(*adapter);

    // A record naming a key the signature lacks: CONFIG listing the keys in LiteRT's order.
    named.m_outputs[1].m_engine_name = "ghost";
    const anira::StatusError missing = status_error_of(
        [&named] { anira::backend::make_builtin_adapter(ANIRA_ENGINE_LITERT)->prepare(named); });
    EXPECT_EQ(missing.status(), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::string(missing.what()).find("'output_1', 'output_0'"), std::string::npos)
        << missing.what();
}

#endif  // USE_LITERT
