// The engine room's interface (src/backends/Adapter.h: Loaded, Prepared, Executor) and the
// legacy adapter over the 2.x virtual, driven directly: no core, no session. The claim loop
// under several threads, the exclusive path, the float32 helpers of the built-in adapters, the
// 2.x plan table as requests, and the copies the legacy adapter makes around a descriptor that
// names other memory than the struct's buffer.

#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
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

using anira::StatusError;
using anira::backend::Bindings;
using anira::backend::BuiltinEngine;
using anira::backend::ChunkBuffers;
using anira::backend::EngineTensor;
using anira::backend::Executor;
using anira::backend::ExecutorLoaded;
using anira::backend::ExtentRule;
using anira::backend::Loaded;
using anira::backend::Model;
using anira::backend::PlanRequest;
using anira::backend::Prepared;
using anira::backend::PrepareRequest;
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
    tensor.m_export_name = engine_name;
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

/// The init record the core hands an engine: the level in effect, no thread pool, no context.
anira_init_info init_info() {
    anira_init_info info = ANIRA_INIT_INFO_INIT;
    info.log_level = static_cast<uint32_t>(anira::get_log_level());
    return info;
}

/// What every executor of a RecordingLoaded records: the calls, whether two calls ever ran at
/// once on one executor, the calls per shared slot, the calls out of range, the resets and the
/// flags of the last call.
struct Recording {
    static constexpr uint32_t k_max_instances = 8;

    std::atomic<uint32_t> m_calls{0};
    std::atomic<uint32_t> m_overlaps{0};
    std::atomic<uint32_t> m_out_of_range{0};
    std::atomic<uint32_t> m_resets{0};
    std::atomic<uint32_t> m_reset_instance{k_max_instances};
    std::atomic<uint32_t> m_last_flags{0};
    std::atomic<uint32_t> m_own_calls{0};
    std::atomic<uint32_t> m_executors_made{0};
    std::array<std::atomic<uint32_t>, k_max_instances> m_per_instance{};
    anira_status m_status = ANIRA_OK;
    std::chrono::microseconds m_hold{200};
};

/// One executor: records into the shared Recording; its own busy flag tells an overlap.
class RecordingExecutor final : public Executor {
public:
    explicit RecordingExecutor(Recording& recording) : m_recording(&recording) {
        m_recording->m_executors_made.fetch_add(1);
    }

    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override {
        static_cast<void>(chunk);
        if (m_in_use.exchange(true)) { m_recording->m_overlaps.fetch_add(1); }
        std::this_thread::sleep_for(m_recording->m_hold);
        m_recording->m_last_flags.store(ctx.flags);
        if ((ctx.flags & ANIRA_ENGINE_CALL_EXCLUSIVE) != 0U) {
            m_recording->m_own_calls.fetch_add(1);
        } else if (ctx.instance >= Recording::k_max_instances) {
            m_recording->m_out_of_range.fetch_add(1);
        } else {
            m_recording->m_per_instance.at(ctx.instance).fetch_add(1);
        }
        m_recording->m_calls.fetch_add(1);
        m_in_use.store(false);
        return m_recording->m_status;
    }

    void reset(const anira_engine_ctx& ctx) noexcept override {
        m_recording->m_resets.fetch_add(1);
        m_recording->m_reset_instance.store(ctx.instance);
    }

private:
    Recording* m_recording;
    std::atomic<bool> m_in_use{false};
};

/// The engine object of the recording doubles: counts its inits, refuses them when told to.
class RecordingEngine final : public BuiltinEngine {
public:
    /// Under any engine value: a double that says it is a built-in engine's object is one an
    /// adapter refuses.
    explicit RecordingEngine(anira_engine engine = ANIRA_ENGINE_NONE) : BuiltinEngine(engine) {}

    uint32_t m_inits = 0;
    anira_status m_refuse = ANIRA_OK;  ///< what do_init throws while not ANIRA_OK

protected:
    void do_init(const anira_init_info& /*info*/) override {
        ++m_inits;
        if (m_refuse != ANIRA_OK) { throw StatusError(m_refuse, "the recording engine refused"); }
    }
};

/// The engine object of a recording loaded model, initialised as the core has it before a
/// load.
std::shared_ptr<RecordingEngine> initialised_engine() {
    auto engine = std::make_shared<RecordingEngine>();
    engine->ensure_init(init_info());
    return engine;
}

/// A loaded model of recording executors: the probe and every further one made over it, over
/// an initialised engine object of its own unless one is given.
class RecordingLoaded final : public ExecutorLoaded {
public:
    RecordingLoaded() : ExecutorLoaded(initialised_engine()) {}
    explicit RecordingLoaded(std::shared_ptr<BuiltinEngine> engine)
        : ExecutorLoaded(std::move(engine)) {}

    Recording m_recording;

protected:
    void do_load(const Model& model) override {
        static_cast<void>(model);
        adopt(std::make_unique<RecordingExecutor>(m_recording));
    }
    std::unique_ptr<Executor> make_executor() override {
        return std::make_unique<RecordingExecutor>(m_recording);
    }
};

/// A shared session's handle over `loaded` (no record: the 2.x path's shape).
std::unique_ptr<Prepared> shared_session(Loaded& loaded) {
    return loaded.prepare(PrepareRequest{.m_exclusive = false, .m_info = nullptr});
}

/// An exclusive session's handle over `loaded`.
std::unique_ptr<Prepared> exclusive_session(Loaded& loaded) {
    return loaded.prepare(PrepareRequest{.m_exclusive = true, .m_info = nullptr});
}

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
// The lifecycle and the claim loop
// ============================================================================================

TEST(Adapter, PrepareRefusesBeforeLoad) {
    RecordingLoaded loaded;
    EXPECT_FALSE(loaded.loaded());
    EXPECT_EQ(loaded.num_instances(), 0U);
    const anira::StatusError error = status_error_of([&] { shared_session(loaded); });
    EXPECT_EQ(error.status(), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(loaded.m_recording.m_calls.load(), 0U);
}

// load keeps the record, makes one executor per shared slot (the probe first) and warms each up
// once; a record without a shared slot keeps its probe as the spare of the first exclusive
// session, and a second exclusive session gets one more executor.
TEST(Adapter, LoadKeepsTheRecordAndSizesTheSharedSlots) {
    RecordingLoaded loaded;
    const Model model = gain_model(3);
    loaded.load(model);
    EXPECT_TRUE(loaded.loaded());
    EXPECT_EQ(loaded.model(), model);
    EXPECT_EQ(loaded.num_instances(), 3U);
    EXPECT_EQ(loaded.model().m_inputs.at(0).m_num_elements, k_block);
    EXPECT_EQ(loaded.m_recording.m_executors_made.load(), 3U) << "one executor per shared slot";

    RecordingLoaded stateful;
    stateful.load(gain_model(0));
    EXPECT_EQ(stateful.num_instances(), 0U);
    EXPECT_EQ(stateful.m_recording.m_executors_made.load(), 1U) << "the probe, kept as the spare";
    const std::unique_ptr<Prepared> first = exclusive_session(stateful);
    EXPECT_EQ(stateful.m_recording.m_executors_made.load(), 1U) << "the first session takes it";
    const std::unique_ptr<Prepared> second = exclusive_session(stateful);
    EXPECT_EQ(stateful.m_recording.m_executors_made.load(), 2U) << "the second gets one more";
    EXPECT_TRUE(first->exclusive() && second->exclusive());
    EXPECT_EQ(&first->loaded(), &stateful);
}

// Every shared call gets a slot below the record's count, never two calls at once on one, and
// no call starves: N threads x M calls of one session on 3 slots all complete; the caller's
// context stays as it is.
TEST(Adapter, TheClaimLoopHandsOutDistinctInstancesUnderThreads) {
    RecordingLoaded loaded;
    loaded.load(gain_model(3));
    const std::unique_ptr<Prepared> session = shared_session(loaded);
    EXPECT_FALSE(session->exclusive());
    constexpr uint32_t k_threads = 6;
    constexpr uint32_t k_calls = 40;
    std::atomic<uint32_t> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(k_threads);
    for (uint32_t t = 0; t < k_threads; ++t) {
        threads.emplace_back([&session, &failures]() {
            anira_engine_ctx ctx{};
            ctx.instance = 77;  // the caller's copy stays as it is
            for (uint32_t call = 0; call < k_calls; ++call) {
                if (session->run(ctx, nullptr, false) != ANIRA_OK) { failures.fetch_add(1); }
            }
            if (ctx.instance != 77 || ctx.flags != 0U) { failures.fetch_add(1); }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    const Recording& recording = loaded.m_recording;
    EXPECT_EQ(failures.load(), 0U);
    EXPECT_EQ(recording.m_calls.load(), k_threads * k_calls);
    EXPECT_EQ(recording.m_overlaps.load(), 0U) << "two calls at once on one executor";
    EXPECT_EQ(recording.m_out_of_range.load(), 0U) << "a slot at or above the count";
    EXPECT_EQ(recording.m_own_calls.load(), 0U) << "a shared session claims";
    uint32_t seen = 0;
    for (uint32_t i = 0; i < 3; ++i) { seen += recording.m_per_instance.at(i).load(); }
    EXPECT_EQ(seen, k_threads * k_calls);
    EXPECT_GT(recording.m_per_instance.at(1).load() + recording.m_per_instance.at(2).load(), 0U)
        << "six threads on three slots never used a second one";
}

// An exclusive session claims nothing: its calls carry ANIRA_ENGINE_CALL_EXCLUSIVE with
// instance 0 and run on its own executor, reset runs on it right before the call when asked,
// and the status process returns is run's. A shared session is never reset.
TEST(Adapter, AnExclusiveSessionRunsOnItsOwnExecutorAndResetsWhenAsked) {
    RecordingLoaded loaded;
    loaded.load(gain_model(0));
    const std::unique_ptr<Prepared> session = exclusive_session(loaded);
    Recording& recording = loaded.m_recording;
    const anira_engine_ctx ctx{};
    EXPECT_EQ(session->run(ctx, nullptr, false), ANIRA_OK);
    EXPECT_EQ(recording.m_resets.load(), 0U);
    EXPECT_EQ(recording.m_own_calls.load(), 1U);
    EXPECT_EQ(recording.m_last_flags.load() & ANIRA_ENGINE_CALL_EXCLUSIVE, 1U);
    EXPECT_EQ(session->run(ctx, nullptr, true), ANIRA_OK);
    EXPECT_EQ(recording.m_resets.load(), 1U);
    EXPECT_EQ(recording.m_reset_instance.load(), 0U);
    recording.m_status = ANIRA_ERROR_ENGINE;
    EXPECT_EQ(session->run(ctx, nullptr, false), ANIRA_ERROR_ENGINE);
    EXPECT_EQ(recording.m_calls.load(), 3U);

    // A shared session over a model with slots: reset_first is never honoured (a shared
    // model keeps nothing per stream), and a shared call on a model without a slot refuses.
    RecordingLoaded shared;
    shared.load(gain_model(1));
    const std::unique_ptr<Prepared> shared_one = shared_session(shared);
    EXPECT_EQ(shared_one->run(ctx, nullptr, true), ANIRA_OK);
    EXPECT_EQ(shared.m_recording.m_resets.load(), 0U);
    const std::unique_ptr<Prepared> shared_none = shared_session(loaded);
    EXPECT_EQ(shared_none->run(ctx, nullptr, false), ANIRA_ERROR_INVALID_STATE)
        << "no shared slot to claim";
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
    b.m_bytes_owner = std::shared_ptr<const void>(std::make_shared<int>(1));
    EXPECT_EQ(a, b) << "the owner is no part of the identity";
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
    b = gain_model(0);
    EXPECT_NE(a, b) << "a record without a shared slot (an exclusive session's) is another";
    b = gain_model(2);
    b.m_provider = ANIRA_PROVIDER_CUDA;
    EXPECT_NE(a, b) << "the provider is";
    b = gain_model(2);
    b.m_provider_id = "QNNExecutionProvider";
    EXPECT_NE(a, b) << "a custom provider is";
    b = gain_model(2);
    b.m_options = {{"device_id", "1"}};
    EXPECT_NE(a, b) << "the provider's options are";
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
        ASSERT_EQ(without[i].m_model.m_inputs.size(), 1U);
        EXPECT_EQ(without[i].m_model.m_inputs.at(0).m_dims, (std::vector<int64_t>{1, 1, k_block}));
        EXPECT_EQ(without[i].m_model.m_inputs.at(0).m_num_elements, k_block);
        EXPECT_TRUE(without[i].m_model.m_inputs.at(0).m_export_name.empty());
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
    anira::backend::LegacyLoaded loaded(backend);
    EXPECT_EQ(loaded.wrapped(), &backend);
    loaded.load(gain_model());
    EXPECT_EQ(backend.m_prepares, 1);
    const std::unique_ptr<Prepared> session = shared_session(loaded);
    Prepared& adapter = *session;

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
    anira::backend::LegacyLoaded loaded(backend);
    loaded.load(gain_model());
    const std::unique_ptr<Prepared> session = shared_session(loaded);
    Prepared& adapter = *session;

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
    anira::backend::LegacyLoaded loaded(backend);
    loaded.load(gain_model());
    const std::unique_ptr<Prepared> session = shared_session(loaded);
    Prepared& adapter = *session;

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
    anira::backend::LegacyLoaded loaded(config);
    loaded.load(gain_model());
    const std::unique_ptr<Prepared> session = shared_session(loaded);
    Prepared& adapter = *session;
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
// legacy adapter takes no slot claim, so two callers meet inside process on a record of one
// slot.
TEST(Adapter, LegacyAdapterTakesNoInstanceClaim) {
    anira::InferenceConfig config = custom_only_config();
    MeetingBackend backend(config, 2);
    anira::backend::LegacyLoaded loaded(backend);
    loaded.load(gain_model(1));
    const std::unique_ptr<Prepared> session = shared_session(loaded);
    Prepared& adapter = *session;
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

// Every slot reports position until the engine's load says otherwise: the legacy adapter's
// 2.x backend binds by position.
TEST(Adapter, LoadReportsPositionForEverySlotUntilTheEngineSaysOtherwise) {
    RecordingLoaded loaded;
    EXPECT_TRUE(loaded.bindings().m_inputs.empty());
    loaded.load(gain_model());
    EXPECT_EQ(loaded.bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    EXPECT_EQ(loaded.bindings().m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    anira::InferenceConfig config = custom_only_config();
    RecordingBackend backend(config);
    anira::backend::LegacyLoaded legacy(backend);
    legacy.load(gain_model());
    EXPECT_EQ(legacy.bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
}

// ============================================================================================
// The built-in adapters
// ============================================================================================

// Every engine of the build has an engine object (uninitialised until a loaded model's init)
// and an adapter over it (unloaded until load loads a model); an engine the build does not
// carry, and the custom engine, have neither.
TEST(Adapter, MakeBuiltInEngineAndLoadedAnswerTheEnginesOfTheBuild) {
    EXPECT_EQ(anira::backend::make_builtin_engine(ANIRA_ENGINE_NONE), nullptr);
    EXPECT_EQ(anira::backend::make_builtin_loaded(nullptr), nullptr);
    EXPECT_EQ(anira::backend::engine_of(anira::InferenceBackend::CUSTOM), ANIRA_ENGINE_NONE);
    for (const anira::InferenceBackend backend : every_backend()) {
        if (backend == anira::InferenceBackend::CUSTOM) { continue; }
        const anira_engine engine = anira::backend::engine_of(backend);
        ASSERT_NE(engine, ANIRA_ENGINE_NONE);
        const std::shared_ptr<BuiltinEngine> object = anira::backend::make_builtin_engine(engine);
        ASSERT_NE(object, nullptr) << "engine " << static_cast<int>(engine);
        EXPECT_EQ(object->engine(), engine);
        EXPECT_FALSE(object->initialised());
        const std::shared_ptr<Loaded> loaded = anira::backend::make_builtin_loaded(object);
        ASSERT_NE(loaded, nullptr) << "engine " << static_cast<int>(engine);
        EXPECT_FALSE(loaded->loaded());
        EXPECT_EQ(&dynamic_cast<ExecutorLoaded&>(*loaded).engine(), object.get())
            << "the loaded model holds the object it was made over";
        // The query needs no init and lists the default provider first.
        const std::vector<anira::backend::ProviderInfo> providers = object->providers();
        ASSERT_FALSE(providers.empty());
        EXPECT_EQ(providers.front(), anira::backend::ProviderInfo{});
        EXPECT_EQ(providers, anira::backend::builtin_providers(engine));
        EXPECT_FALSE(object->initialised()) << "the query initialises nothing";
        // An object that claims the engine but is not the adapter's own is refused.
        const auto foreign = std::make_shared<RecordingEngine>(engine);
        try {
            static_cast<void>(anira::backend::make_builtin_loaded(foreign));
            ADD_FAILURE() << "a recording engine object passed as engine "
                          << static_cast<int>(engine);
        } catch (const StatusError& e) { EXPECT_EQ(e.status(), ANIRA_ERROR_INVALID_ARGUMENT); }
    }
#ifndef USE_ONNXRUNTIME
    EXPECT_EQ(anira::backend::make_builtin_engine(ANIRA_ENGINE_ONNXRUNTIME), nullptr);
#endif
}

// A built-in engine's init is its object's, once: the first loaded model's init runs it with
// the record, every later one finds it done, and load refuses to run before it (what load
// builds runs over what init built). A refused init is not remembered, so the next init runs
// it again.
TEST(Adapter, ALoadedModelInitialisesItsEngineObjectOnceAndLoadsAfterItAlone) {
    const auto engine = std::make_shared<RecordingEngine>();
    RecordingLoaded first(engine);
    RecordingLoaded second(engine);
    EXPECT_EQ(&first.engine(), engine.get());
    EXPECT_FALSE(engine->initialised());
    try {
        first.load(gain_model());
        ADD_FAILURE() << "a load before init";
    } catch (const StatusError& e) {
        EXPECT_EQ(e.status(), ANIRA_ERROR_INVALID_STATE);
        EXPECT_NE(std::string(e.what()).find("init runs before load"), std::string::npos)
            << e.what();
    }
    EXPECT_FALSE(first.loaded());
    EXPECT_EQ(engine->m_inits, 0U);

    engine->m_refuse = ANIRA_ERROR_ENGINE;
    try {
        first.init(init_info());
        ADD_FAILURE() << "a refused init";
    } catch (const StatusError& e) { EXPECT_EQ(e.status(), ANIRA_ERROR_ENGINE); }
    EXPECT_FALSE(engine->initialised()) << "a refused init is not remembered";
    EXPECT_EQ(engine->m_inits, 1U);

    engine->m_refuse = ANIRA_OK;
    first.init(init_info());
    EXPECT_TRUE(engine->initialised());
    EXPECT_EQ(engine->m_inits, 2U);
    second.init(init_info());
    EXPECT_EQ(engine->m_inits, 2U) << "once per object";
    first.load(gain_model());
    second.load(gain_model());
    EXPECT_TRUE(first.loaded());
    EXPECT_TRUE(second.loaded());
}

namespace {

/// A built-in engine's loaded model and one session's handle over it, driven directly: what
/// the inference thread runs. prepare loads the record and prepares one session, exclusive when
/// the record has no shared slot.
class Rig {
public:
    explicit Rig(std::shared_ptr<Loaded> loaded) : m_loaded(std::move(loaded)) {}

    void prepare(const Model& model) {
        m_prepared.reset();
        m_loaded->init(init_info());
        m_loaded->load(model);
        m_prepared = m_loaded->prepare(
            PrepareRequest{.m_exclusive = model.m_instances == 0, .m_info = nullptr});
    }
    bool prepared() const noexcept { return m_loaded->loaded() && m_prepared != nullptr; }
    const Model& model() const noexcept { return m_loaded->model(); }
    const Bindings& bindings() const noexcept { return m_loaded->bindings(); }
    Loaded& loaded() const noexcept { return *m_loaded; }
    anira_status run(const anira_engine_ctx& ctx, ChunkBuffers* chunk, bool reset_first) noexcept {
        return m_prepared->run(ctx, chunk, reset_first);
    }

private:
    std::shared_ptr<Loaded> m_loaded;
    std::unique_ptr<Prepared> m_prepared;
};

/// The rig of a built-in engine of the build; a null loaded model for one it does not carry.
std::shared_ptr<Rig> builtin_rig(anira_engine engine) {
    std::shared_ptr<Loaded> loaded =
        anira::backend::make_builtin_loaded(anira::backend::make_builtin_engine(engine));
    return loaded == nullptr ? nullptr : std::make_shared<Rig>(std::move(loaded));
}

}  // namespace

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

// The provider of the record: the default one always; a provider of the enum or a registered
// name exactly when the runtime lists it among its available execution providers, never
// Vulkan; a record on a provider the adapter does not serve is refused at load with the
// runtime's list, before anything of the model is read.
TEST(AdapterOnnxRuntime, TheProviderIsServedWhenTheRuntimeListsIt) {
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
    ASSERT_NE(adapter, nullptr);
    const Loaded& loaded = adapter->loaded();
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_DEFAULT, ""));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_VULKAN, ""));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_DEFAULT, "com.example.nobody"));
    const std::vector<anira::backend::ProviderInfo> listed =
        anira::backend::onnxruntime_providers();
    for (const anira::backend::ProviderInfo& info : listed) {
        EXPECT_TRUE(loaded.serves(info.m_provider, info.m_provider_id)) << info.m_provider_id;
    }
    const bool has_cuda = std::ranges::any_of(listed, [](const anira::backend::ProviderInfo& p) {
        return p.m_provider == ANIRA_PROVIDER_CUDA;
    });
    EXPECT_EQ(loaded.serves(ANIRA_PROVIDER_CUDA, ""), has_cuda);
    EXPECT_NE(loaded.provider_reason().find("execution providers"), std::string::npos)
        << loaded.provider_reason();

    Model nobody = onnx_gain_model();
    nobody.m_provider_id = "com.example.nobody";
    const anira::StatusError refused =
        status_error_of([&nobody] { builtin_rig(ANIRA_ENGINE_ONNXRUNTIME)->prepare(nobody); });
    EXPECT_EQ(refused.status(), ANIRA_ERROR_NOT_SUPPORTED);
    const std::string message = refused.what();
    EXPECT_NE(message.find("engine 'onnxruntime' does not serve provider 'com.example.nobody'"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("execution providers"), std::string::npos) << message;
    if (!has_cuda) {
        Model cuda = onnx_gain_model();
        cuda.m_provider = ANIRA_PROVIDER_CUDA;
        cuda.m_options = {{"device_id", "0"}};  // travel to the V2 options, refused with them
        const anira::StatusError no_cuda =
            status_error_of([&cuda] { builtin_rig(ANIRA_ENGINE_ONNXRUNTIME)->prepare(cuda); });
        EXPECT_EQ(no_cuda.status(), ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_NE(std::string(no_cuda.what()).find("provider 'cuda'"), std::string::npos)
            << no_cuda.what();
    }
}

// The gain: audio_in and the two outputs bind by position (the graph names them data,
// processed_data and peak), the gain input by its canonical name; the check passes on the
// graph's [1, 1, -1] and [1]; the run applies the gain over the descriptors' memory, in place.
TEST(AdapterOnnxRuntime, TheGainBindsByPositionAndByNameAndRunsOverTheDescriptors) {
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
        model.m_inputs[0].m_export_name = "ghost";
        const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
        const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
        const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
        const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_MODEL_LOAD);
        EXPECT_NE(std::string(error.what()).find("onnxruntime: memory: "), std::string::npos)
            << error.what();
    }
    // A missing file: NO_SUCH_FILE, before ONNX Runtime sees the path.
    {
        Model model = onnx_gain_model();
        model.m_path = "this/model/does/not/exist.onnx";
        const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
        const anira::StatusError error = status_error_of([&] { adapter->prepare(model); });
        EXPECT_EQ(error.status(), ANIRA_ERROR_NO_SUCH_FILE);
    }
    // An int16 record: CONFIG naming the engine and the tensor, before the file is opened.
    {
        Model model = onnx_gain_model();
        model.m_path = "this/model/does/not/exist.onnx";
        model.m_inputs[0].m_dtype = ANIRA_DTYPE_I16;
        const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_ONNXRUNTIME);
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
void expect_gain_of_one_half(Rig& adapter) {
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
void expect_accumulator_closed_form(Rig& adapter) {
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
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_TFLITE);
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
    const anira::StatusError error =
        status_error_of([&rank_one] { builtin_rig(ANIRA_ENGINE_TFLITE)->prepare(rank_one); });
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
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_TFLITE);
    adapter->prepare(tensorflow_accumulator_model(ANIRA_ENGINE_TFLITE));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    expect_accumulator_closed_form(*adapter);

    // A record naming the keys binds the same slots by name.
    Model named = tensorflow_accumulator_model(ANIRA_ENGINE_TFLITE);
    named.m_outputs[0].m_export_name = "output_0";
    named.m_outputs[1].m_export_name = "output_1";
    const std::shared_ptr<Rig> by_name = builtin_rig(ANIRA_ENGINE_TFLITE);
    by_name->prepare(named);
    EXPECT_EQ(by_name->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    expect_accumulator_closed_form(*by_name);

    // A record naming a key the signature lacks: CONFIG listing the keys.
    named.m_outputs[0].m_export_name = "ghost";
    const anira::StatusError error =
        status_error_of([&named] { builtin_rig(ANIRA_ENGINE_TFLITE)->prepare(named); });
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
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_TFLITE);
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
    named.m_inputs[0].m_export_name = "args_0";
    named.m_outputs[0].m_export_name = "Identity";
    const std::shared_ptr<Rig> by_name = builtin_rig(ANIRA_ENGINE_TFLITE);
    by_name->prepare(named);
    EXPECT_EQ(by_name->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_NAME}));
    EXPECT_EQ(by_name->bindings().m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_NAME}));
}

#endif  // USE_TFLITE

#ifdef USE_LITERT

// LiteRT lists the gain export's signature outputs in the file's order, [output_1 (the peak),
// output_0 (the stream)], and the adapter binds a slot without a name by the KEY order of the
// signature (the names sorted, as TensorFlow Lite's signature runner lists them), so the
// declared order puts the stream on output_0 and the peak on output_1, as it does on TFLite,
// and the gain runs at the export's rank 3; the names of the file's litert row bind the same.
TEST(AdapterLiteRt, TheGainBindsByPositionInTheSignaturesKeyOrder) {
    const std::shared_ptr<Rig> positional = builtin_rig(ANIRA_ENGINE_LITERT);
    ASSERT_NE(positional, nullptr);
    positional->prepare(tensorflow_gain_model(ANIRA_ENGINE_LITERT));
    EXPECT_EQ(positional->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(positional->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    expect_gain_of_one_half(*positional);
    expect_gain_of_one_half(*positional);

    Model named = tensorflow_gain_model(ANIRA_ENGINE_LITERT);
    named.m_outputs[0].m_export_name = "output_0";
    named.m_outputs[1].m_export_name = "output_1";
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_LITERT);
    ASSERT_NE(adapter, nullptr);
    adapter->prepare(named);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    expect_gain_of_one_half(*adapter);
    expect_gain_of_one_half(*adapter);
}

// The provider of the record on LiteRT: an accelerator by its hardware's name beside DEFAULT
// ("gpu", "npu"), never a provider of the enum; a name whose hardware no registered
// accelerator supports here is refused at load naming the registered ones (the environment's
// automatic registration loads the accelerator libraries it finds; a LiteRT library that does
// not export its accelerator query, the Windows DLL, knows the CPU alone and says so).
TEST(AdapterLiteRt, AnAcceleratorIsNamedByItsHardware) {
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_LITERT);
    ASSERT_NE(adapter, nullptr);
    const Loaded& loaded = adapter->loaded();
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_DEFAULT, ""));
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_DEFAULT, "gpu"));
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_DEFAULT, "npu"));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_DEFAULT, "tpu"));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_CUDA, ""));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_XNNPACK, ""));
    EXPECT_NE(loaded.provider_reason().find("'gpu'"), std::string::npos)
        << loaded.provider_reason();

    Model enum_provider = tensorflow_gain_model(ANIRA_ENGINE_LITERT);
    enum_provider.m_provider = ANIRA_PROVIDER_CUDA;
    const anira::StatusError refused = status_error_of(
        [&enum_provider] { builtin_rig(ANIRA_ENGINE_LITERT)->prepare(enum_provider); });
    EXPECT_EQ(refused.status(), ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(std::string(refused.what()).find("engine 'litert' does not serve provider 'cuda'"),
              std::string::npos)
        << refused.what();

    const std::vector<anira::backend::ProviderInfo> listed = anira::backend::litert_providers();
    const bool has_gpu = std::ranges::any_of(listed, [](const anira::backend::ProviderInfo& p) {
        return p.m_provider_id == "gpu";
    });
    if (!has_gpu) {
        Model gpu = tensorflow_gain_model(ANIRA_ENGINE_LITERT);
        gpu.m_outputs[0].m_export_name = "output_0";
        gpu.m_outputs[1].m_export_name = "output_1";
        gpu.m_provider_id = "gpu";
        const anira::StatusError no_gpu =
            status_error_of([&gpu] { builtin_rig(ANIRA_ENGINE_LITERT)->prepare(gpu); });
        EXPECT_EQ(no_gpu.status(), ANIRA_ERROR_NOT_SUPPORTED);
        const std::string message = no_gpu.what();
        EXPECT_NE(message.find("no registered accelerator supports 'gpu'"), std::string::npos)
            << message;
        EXPECT_NE(message.find("cpu"), std::string::npos) << message;
    }
}

// The accumulator on LiteRT: the same file order on the output side, bound by position in the
// key order (the closed form holds, as on TFLite) and by the keys of the file's litert row.
TEST(AdapterLiteRt, TheAccumulatorBindsByPositionInTheSignaturesKeyOrder) {
    const std::shared_ptr<Rig> positional = builtin_rig(ANIRA_ENGINE_LITERT);
    positional->prepare(tensorflow_accumulator_model(ANIRA_ENGINE_LITERT));
    EXPECT_EQ(positional->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    expect_accumulator_closed_form(*positional);

    Model named = tensorflow_accumulator_model(ANIRA_ENGINE_LITERT);
    named.m_outputs[0].m_export_name = "output_0";
    named.m_outputs[1].m_export_name = "output_1";
    const std::shared_ptr<Rig> adapter = builtin_rig(ANIRA_ENGINE_LITERT);
    adapter->prepare(named);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));
    expect_accumulator_closed_form(*adapter);

    // A record naming a key the signature lacks: CONFIG listing the keys in the key order.
    named.m_outputs[1].m_export_name = "ghost";
    const anira::StatusError missing =
        status_error_of([&named] { builtin_rig(ANIRA_ENGINE_LITERT)->prepare(named); });
    EXPECT_EQ(missing.status(), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::string(missing.what()).find("'output_0', 'output_1'"), std::string::npos)
        << missing.what();
}

#endif  // USE_LITERT
