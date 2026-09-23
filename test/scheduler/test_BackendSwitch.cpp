// A plan switch while audio runs. The session's whole selection is one atomic, the dense
// plan index (SessionElement::m_current_plan). A chunk is stamped with it when it is
// submitted (Core::pre_process), and its pre_process, before_inference, engine call,
// after_inference and post_process all resolve that stamp. A switch that lands while the
// chunk is queued or in flight therefore applies from the next chunk on, and exactly one
// processor runs per chunk. The 2.x selection by backend is a lookup in the same table.
//
// Before the stamp, every one of those steps loaded the session's backend on its own, and
// InferenceThread::inference() loaded it once per engine block: a switch landing between
// two loads ran two engines for one chunk, or none, and told the hooks one backend while
// another ran. The stamp is the index and not the backend, because two plans may run on one
// backend (two variants of a model, two providers of an engine).

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "backends/Adapters.h"
#include "capi/handles.h"  // IWYU pragma: keep - defines anira_context_config
#include "gtest/gtest.h"

using namespace anira;

namespace {

constexpr size_t k_block = 512;
constexpr float k_sample_rate = 48000.F;
constexpr auto k_wait = std::chrono::seconds(20);

// A CUSTOM-only model: a compiled-in engine backend has no processor behind it and falls
// back to the default round-trip processor, so the switch needs no model file.
InferenceConfig make_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, k_block}}, {{1, 1, k_block}})},
        ProcessingSpec({1}, {1}, {k_block}, {k_block}),
        10.F,
        0,
        false,
        0.F,
        2);
}

// The first engine backend this build carries; none on an engine-free build.
std::optional<InferenceBackend> first_engine_backend() {
#if defined(USE_LIBTORCH)
    return InferenceBackend::LIBTORCH;
#elif defined(USE_ONNXRUNTIME)
    return InferenceBackend::ONNX;
#elif defined(USE_TFLITE)
    return InferenceBackend::TFLITE;
#elif defined(USE_LITERT)
    return InferenceBackend::LITERT;
#elif defined(USE_EXECUTORCH)
    return InferenceBackend::EXECUTORCH;
#else
    return std::nullopt;
#endif
}

// Records the backend every step of a chunk is told, and holds a chunk inside
// before_inference while the gate is closed: between its stamp and its engine call.
class RecordingProcessor : public PrePostProcessor {
public:
    // pre_process and post_process run inside the handler's real-time process call, so the
    // records must not allocate there (RTSan): the room is reserved here, and a record past
    // it is dropped, which the size checks of the tests would show.
    static constexpr size_t k_record_capacity = 64;

    explicit RecordingProcessor(InferenceConfig& inference_config)
        : PrePostProcessor(inference_config) {
        m_pre.reserve(k_record_capacity);
        m_post.reserve(k_record_capacity);
    }
    ~RecordingProcessor() override { m_gate_open.store(true); }
    RecordingProcessor(const RecordingProcessor&) = delete;
    RecordingProcessor& operator=(const RecordingProcessor&) = delete;
    RecordingProcessor(RecordingProcessor&&) = delete;
    RecordingProcessor& operator=(RecordingProcessor&&) = delete;

    void pre_process(std::vector<RingBuffer>& input,
                     std::vector<BufferF>& output,
                     InferenceBackend backend) override {
        if (m_pre.size() < k_record_capacity) { m_pre.push_back(backend); }  // driving thread only
        PrePostProcessor::pre_process(input, output, backend);
    }

    void post_process(std::vector<BufferF>& input,
                      std::vector<RingBuffer>& output,
                      InferenceBackend backend) override {
        if (m_post.size() < k_record_capacity) {
            m_post.push_back(backend);
        }  // driving thread only
        PrePostProcessor::post_process(input, output, backend);
    }

    void before_inference(std::vector<BufferF>& /*input*/, InferenceBackend backend) override {
        m_before.store(backend);
        m_in_hook.store(true);
        while (!m_gate_open.load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
    }

    void after_inference(std::vector<BufferF>& /*output*/, InferenceBackend backend) override {
        m_after.store(backend);
    }

    std::vector<InferenceBackend> m_pre;
    std::vector<InferenceBackend> m_post;
    std::atomic<InferenceBackend> m_before{InferenceBackend::CUSTOM};
    std::atomic<InferenceBackend> m_after{InferenceBackend::CUSTOM};
    std::atomic<bool> m_in_hook{false};
    std::atomic<bool> m_gate_open{true};
};

// The CUSTOM processor: the pass-through, counted.
class CountingBackend : public BackendBase {
public:
    explicit CountingBackend(InferenceConfig& config) : BackendBase(config) {}

    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 std::shared_ptr<SessionElement> session) override {
        BackendBase::process(input, output, std::move(session));
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
};

// Polls get_available_samples (which collects the completed chunk and so runs its
// post_process on this thread) until `count` chunks have been post-processed.
bool wait_for_post(InferenceHandler& handler, const RecordingProcessor& pp, size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + k_wait;
    while (std::chrono::steady_clock::now() < deadline) {
        static_cast<void>(handler.get_available_samples(0));
        if (pp.m_post.size() >= count) { return true; }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return false;
}

// The session a manager holds, found through the core's list (white-box).
std::shared_ptr<SessionElement> session_of(const InferenceManager& manager) {
    for (const auto& session : Core::get_sessions()) {
        if (session->m_session_id == manager.get_session_id()) { return session; }
    }
    return nullptr;
}

// The context config of CoreConfig(2): two inference threads.
anira_context_config two_threads() {
    anira_context_config config;
    config.m_num_threads = 2;
    return config;
}

// Two plans on one backend, in the manager's own words: the 2.x table's row for `backend`
// (the custom backend, or the roundtrip without one) twice. The table is the session's from
// create, so a test that wants two plans on one backend asks for them there.
std::vector<backend::PlanRequest> two_plans_on(InferenceConfig& config,
                                               InferenceBackend backend,
                                               BackendBase* custom) {
    const std::vector<backend::PlanRequest> table = backend::legacy_plan_requests(config, custom);
    std::vector<backend::PlanRequest> requests;
    for (const backend::PlanRequest& request : table) {
        if (request.m_legacy_backend == backend) {
            requests.push_back(request);
            requests.push_back(request);
            break;
        }
    }
    return requests;
}

}  // namespace

TEST(BackendSwitch, AChunkInFlightFinishesOnTheBackendItWasSubmittedUnder) {
    const std::optional<InferenceBackend> engine = first_engine_backend();
    if (!engine.has_value()) { GTEST_SKIP() << "no engine backend in this build to switch from"; }

    InferenceConfig config = make_config();
    RecordingProcessor pp(config);
    CountingBackend custom(config);
    {
        InferenceHandler handler(pp, config, custom, CoreConfig(2));
        handler.prepare(HostConfig(k_block, k_sample_rate));
        handler.set_inference_backend(*engine);

        // Chunk 1 is submitted under the engine backend and held before its engine call.
        pp.m_gate_open.store(false);
        std::vector<float> io(k_block, 0.5F);
        const std::array<float*, 1> channels = {io.data()};
        handler.process(channels.data(), k_block);
        const auto deadline = std::chrono::steady_clock::now() + k_wait;
        while (!pp.m_in_hook.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        ASSERT_TRUE(pp.m_in_hook.load()) << "chunk 1 never reached before_inference";

        // The switch lands while chunk 1 is in flight.
        handler.set_inference_backend(InferenceBackend::CUSTOM);
        EXPECT_EQ(handler.get_inference_backend(), InferenceBackend::CUSTOM);
        pp.m_gate_open.store(true);
        ASSERT_TRUE(wait_for_post(handler, pp, 1)) << "chunk 1 never completed";

        // Chunk 1 ran on the engine backend end to end (its default-processor fallback):
        // the custom processor was not called, and every step was told the same backend.
        EXPECT_EQ(custom.m_calls.load(), 0) << "the switch reached the in-flight chunk";
        ASSERT_EQ(pp.m_pre.size(), 1U);
        EXPECT_EQ(pp.m_pre[0], *engine);
        EXPECT_EQ(pp.m_before.load(), *engine);
        EXPECT_EQ(pp.m_after.load(), *engine);
        ASSERT_EQ(pp.m_post.size(), 1U);
        EXPECT_EQ(pp.m_post[0], *engine) << "post_process was told the session's new backend";

        // Chunk 2 is submitted after the switch and runs on CUSTOM, exactly once.
        handler.process(channels.data(), k_block);
        ASSERT_TRUE(wait_for_post(handler, pp, 2)) << "chunk 2 never completed";
        EXPECT_EQ(custom.m_calls.load(), 1);
        ASSERT_EQ(pp.m_pre.size(), 2U);
        EXPECT_EQ(pp.m_pre[1], InferenceBackend::CUSTOM);
        EXPECT_EQ(pp.m_before.load(), InferenceBackend::CUSTOM);
        EXPECT_EQ(pp.m_after.load(), InferenceBackend::CUSTOM);
        EXPECT_EQ(pp.m_post[1], InferenceBackend::CUSTOM);
    }
}

// The reverse direction, which needs no engine: a chunk submitted under CUSTOM is stamped
// CUSTOM, whatever the build carries.
TEST(BackendSwitch, EveryStepOfAChunkIsToldTheStampedBackend) {
    InferenceConfig config = make_config();
    RecordingProcessor pp(config);
    CountingBackend custom(config);
    {
        InferenceHandler handler(pp, config, custom, CoreConfig(2));
        handler.prepare(HostConfig(k_block, k_sample_rate));
        ASSERT_EQ(handler.get_inference_backend(), InferenceBackend::CUSTOM);

        std::vector<float> io(k_block, 0.5F);
        const std::array<float*, 1> channels = {io.data()};
        handler.process(channels.data(), k_block);
        ASSERT_TRUE(wait_for_post(handler, pp, 1)) << "the chunk never completed";

        EXPECT_EQ(custom.m_calls.load(), 1) << "exactly one processor runs per chunk";
        ASSERT_EQ(pp.m_pre.size(), 1U);
        EXPECT_EQ(pp.m_pre[0], InferenceBackend::CUSTOM);
        EXPECT_EQ(pp.m_before.load(), InferenceBackend::CUSTOM);
        EXPECT_EQ(pp.m_after.load(), InferenceBackend::CUSTOM);
        ASSERT_EQ(pp.m_post.size(), 1U);
        EXPECT_EQ(pp.m_post[0], InferenceBackend::CUSTOM);
    }
}

// What the index buys: two plans on one backend are two selections. A selection stored as a
// backend could not tell them apart (it would read back as the first).
TEST(BackendSwitch, TwoPlansOnOneBackendStayDistinct) {
    InferenceConfig config = make_config();
    RecordingProcessor pp(config);
    CountingBackend custom(config);
    {
        InferenceManager manager(pp,
                                 config,
                                 two_plans_on(config, InferenceBackend::CUSTOM, &custom),
                                 two_threads());
        EXPECT_EQ(manager.get_plan(), 0U) << "a new table selects plan 0";
        manager.prepare(HostConfig(k_block, k_sample_rate));

        ASSERT_TRUE(manager.set_plan(1));
        EXPECT_EQ(manager.get_plan(), 1U);
        EXPECT_EQ(manager.get_backend(), InferenceBackend::CUSTOM);
        // The 2.x selection by backend is the first plan on it.
        manager.set_backend(InferenceBackend::CUSTOM);
        EXPECT_EQ(manager.get_plan(), 0U);
        ASSERT_TRUE(manager.set_plan(1));

        // A chunk submitted under plan 1 carries the index 1, not "the CUSTOM plan".
        // The manager takes host tensors: one planar block, in place (the same tensor on
        // both sides).
        std::vector<float> io(k_block, 0.5F);
        const std::array<float*, 1> channels = {io.data()};
        const std::array<int64_t, 2> shape = {1, static_cast<int64_t>(k_block)};
        anira_tensor block{};
        anira_tensor_init_host_planar(&block,
                                      static_cast<const void*>(channels.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());
        manager.process(&block, &block);
        const auto deadline = std::chrono::steady_clock::now() + k_wait;
        while (pp.m_post.empty() && std::chrono::steady_clock::now() < deadline) {
            static_cast<void>(manager.get_available_samples(0, 0));
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        ASSERT_EQ(pp.m_post.size(), 1U) << "the chunk never completed";
        EXPECT_EQ(custom.m_calls.load(), 1);

        const std::shared_ptr<SessionElement> session = session_of(manager);
        ASSERT_NE(session, nullptr);
        size_t stamped = 0;
        for (const auto& chunk : session->m_inference_queue) {
            if (chunk->m_plan == 1U) { ++stamped; }
        }
        EXPECT_EQ(stamped, 1U) << "exactly the submitted chunk carries plan 1";
    }
}

TEST(BackendSwitch, AnIndexOutOfRangeLeavesTheSelection) {
    InferenceConfig config = make_config();
    PrePostProcessor pp(config);
    InferenceManager manager(pp,
                             config,
                             two_plans_on(config, InferenceBackend::CUSTOM, nullptr),
                             two_threads());
    ASSERT_TRUE(manager.set_plan(1));
    EXPECT_FALSE(manager.set_plan(2));
    EXPECT_EQ(manager.get_plan(), 1U);
    // An empty table is refused at create: a session always has a plan.
    EXPECT_THROW(InferenceManager(pp, config, std::vector<backend::PlanRequest>{}, two_threads()),
                 std::invalid_argument);
}

// The table is read without synchronization once chunks exist, so it is the session's from
// create and never replaced: the plans asked for are the plans the prepared session holds.
TEST(BackendSwitch, ThePlanTableIsTheSessionsFromCreate) {
    InferenceConfig config = make_config();
    PrePostProcessor pp(config);
    CountingBackend custom(config);
    InferenceManager manager(pp,
                             config,
                             two_plans_on(config, InferenceBackend::CUSTOM, &custom),
                             two_threads());
    ASSERT_TRUE(manager.set_plan(1));
    manager.prepare(HostConfig(k_block, k_sample_rate));
    EXPECT_EQ(manager.get_plan(), 1U) << "prepare keeps the selection";
    const std::shared_ptr<SessionElement> session = session_of(manager);
    ASSERT_NE(session, nullptr);
    ASSERT_EQ(session->m_plans.size(), 2U);
    EXPECT_EQ(session->m_plans[0].m_legacy_backend, InferenceBackend::CUSTOM);
    EXPECT_EQ(session->m_plans[1].m_legacy_backend, InferenceBackend::CUSTOM);
    EXPECT_NE(session->m_plans[0].m_adapter, nullptr);
    EXPECT_NE(session->m_plans[0].m_adapter, session->m_plans[1].m_adapter)
        << "a 2.x backend gets a legacy adapter per plan, never pooled";
}

// The 2.x selection by backend on a table that names no plan for it (a 3.x handler's table
// holds exactly its plans): the selection stays, nothing else happens.
TEST(BackendSwitch, ABackendWithoutAPlanLeavesTheSelection) {
    const std::optional<InferenceBackend> engine = first_engine_backend();
    if (!engine.has_value()) {
        GTEST_SKIP() << "needs an engine in the build: CUSTOM is the only backend here";
    }
    InferenceConfig config = make_config();
    PrePostProcessor pp(config);
    // No row runs on CUSTOM: two plans on the engine's backend (the roundtrip without a model).
    InferenceManager manager(pp, config, two_plans_on(config, *engine, nullptr), two_threads());
    ASSERT_TRUE(manager.set_plan(1));
    manager.set_backend(InferenceBackend::CUSTOM);
    EXPECT_EQ(manager.get_plan(), 1U) << "no plan runs on CUSTOM: the selection is unchanged";
    EXPECT_EQ(manager.get_backend(), *engine);
    manager.set_backend(*engine);
    EXPECT_EQ(manager.get_plan(), 0U) << "the first plan on that backend";
}

// A 2.x session: one row per configured model, in order, then every other backend of the
// build, so set_backend() finds a row for whatever a 2.x caller names.
TEST(BackendSwitch, TheDefaultTableNamesEveryBackendOfTheBuild) {
    InferenceConfig config = make_config();
    PrePostProcessor pp(config);
    InferenceManager manager(pp, config, nullptr, CoreConfig(2));
    const std::shared_ptr<SessionElement> session = session_of(manager);
    ASSERT_NE(session, nullptr);

    ASSERT_FALSE(session->m_plans.empty());
    EXPECT_EQ(session->m_plans[0].m_legacy_backend, InferenceBackend::CUSTOM)
        << "the configured model";
    EXPECT_EQ(manager.get_plan(), 0U);
    EXPECT_EQ(manager.get_backend(), InferenceBackend::CUSTOM);

    const std::optional<InferenceBackend> engine = first_engine_backend();
    if (engine.has_value()) {
        const std::optional<uint32_t> plan = session->plan_of_backend(*engine);
        // An explicit return: the optional-access check does not see through ASSERT_TRUE.
        if (!plan.has_value()) { FAIL() << "an engine of the build without a model has a row"; }
        manager.set_backend(*engine);
        EXPECT_EQ(manager.get_plan(), *plan);
        EXPECT_EQ(manager.get_backend(), *engine);
    }
}
