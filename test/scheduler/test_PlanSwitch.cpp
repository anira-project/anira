// A plan switch while audio runs. The session's whole selection is one atomic, the dense
// plan index (SessionElement::m_current_plan). A chunk is stamped with it when it is
// submitted (Core::pre_process), and its pre_process, before_inference, engine call,
// after_inference and post_process all resolve that stamp. A switch that lands while the
// chunk is queued or in flight therefore applies from the next chunk on, and exactly one
// engine runs per chunk.
//
// Before the stamp, every one of those steps loaded the session's selection on its own, and
// InferenceThread::inference() loaded it once per engine block: a switch landing between
// two loads ran two engines for one chunk, or none, and told the hooks one plan while
// another ran. The stamp is the index and not the engine, because two plans may run on one
// engine (two variants of a model, two providers of an engine).
//
// The sessions are C handlers' over two custom rows on two counting gate engines (or one row on
// one engine under two providers), with a processor that records the engine every phase of a
// chunk is told (its context, anira_processor_engine). A chunk is held in flight inside engine
// A's call while its gate is closed (the processor's hooks are real-time code and do not wait).

#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "../abi/handler_support.h"
#include "gtest/gtest.h"
#include "scheduler/InferenceManager.h"
#include "scheduler/SessionElement.h"

using namespace anira;

namespace {

constexpr size_t k_block = 512;
constexpr auto k_wait = std::chrono::seconds(20);
constexpr const char* k_engine_a = "org.example.plan_a";
constexpr const char* k_engine_b = "org.example.plan_b";
/// What engine A serves: the CPU path and a custom provider, two plans of one row.
constexpr const char* k_npu = "com.example.npu";
constexpr std::array<const char*, 2> k_served{"cpu", k_npu};

// Which of the two engines a phase was told: 0 for A, 1 for B, -1 for anything else.
int engine_index(const anira_processor_ctx* ctx) noexcept {
    anira_engine engine = ANIRA_ENGINE_NONE;
    const char* id = nullptr;
    if (anira_processor_engine(ctx, &engine, &id) != ANIRA_OK || id == nullptr) { return -1; }
    if (std::strcmp(id, k_engine_a) == 0) { return 0; }
    if (std::strcmp(id, k_engine_b) == 0) { return 1; }
    return -1;
}

// Records the engine every step of a chunk is told. pre_process and post_process run inside
// the handler's real-time process call, so the records must not allocate there (RTSan): fixed
// room, and a record past it is dropped, which the size checks of the tests would show.
struct Recorder {
    static constexpr size_t k_record_capacity = 64;

    std::array<int, k_record_capacity> m_pre{};
    size_t m_num_pre = 0;  // driving thread only
    std::array<int, k_record_capacity> m_post{};
    size_t m_num_post = 0;  // driving thread only
    std::atomic<int> m_before{-1};
    std::atomic<int> m_after{-1};
    std::atomic<bool> m_in_hook{false};
};

anira_status ANIRA_CALL record_pre(const anira_processor_ctx* ctx,
                                   void* /*prepared*/,
                                   void* user_data) noexcept {
    auto& recorder = *static_cast<Recorder*>(user_data);
    if (recorder.m_num_pre < Recorder::k_record_capacity) {
        recorder.m_pre[recorder.m_num_pre++] = engine_index(ctx);
    }
    return anira_processor_default_pre_process(ctx);
}

anira_status ANIRA_CALL record_post(const anira_processor_ctx* ctx,
                                    void* /*prepared*/,
                                    void* user_data) noexcept {
    auto& recorder = *static_cast<Recorder*>(user_data);
    if (recorder.m_num_post < Recorder::k_record_capacity) {
        recorder.m_post[recorder.m_num_post++] = engine_index(ctx);
    }
    return anira_processor_default_post_process(ctx);
}

anira_status ANIRA_CALL record_before(const anira_processor_ctx* ctx,
                                      void* /*prepared*/,
                                      void* user_data) noexcept {
    auto& recorder = *static_cast<Recorder*>(user_data);
    recorder.m_before.store(engine_index(ctx));
    recorder.m_in_hook.store(true);
    return ANIRA_OK;
}

anira_status ANIRA_CALL record_after(const anira_processor_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) noexcept {
    static_cast<Recorder*>(user_data)->m_after.store(engine_index(ctx));
    return ANIRA_OK;
}

anira_processor_desc recording_processor(Recorder& recorder) {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.user_data = &recorder;
    desc.flags = ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST;
    desc.pre_process = record_pre;
    desc.post_process = record_post;
    desc.before_inference = record_before;
    desc.after_inference = record_after;
    return desc;
}

// A counting pass-through engine object under `id`, over a gate engine's descriptor, serving
// `providers` (none: the CPU path alone); the pipelines it is added to hold it.
anira_custom_engine* engine_under(anira_test::GateEngine& gate,
                                  const char* id,
                                  std::span<const char* const> providers = {}) {
    anira_engine_desc desc = gate.desc();
    desc.providers = providers.empty() ? nullptr : providers.data();
    desc.num_providers = static_cast<uint32_t>(providers.size());
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_custom_engine_create(id, &desc, &engine, &err), ANIRA_OK) << err.message;
    return engine;
}

// The mono 512 pass-through over rows on the engines named, in order (one plan per row under
// the default set).
anira::ModelConfig rows_on(const std::vector<const char*>& engines) {
    anira::ModelConfig model;
    for (size_t i = 0; i < engines.size(); ++i) {
        model.add_model_path(engines[i], "row-" + std::to_string(i));
    }
    model.input(anira_test::stream_tensor("in", 1, k_block, k_block));
    model.output(anira_test::stream_tensor("out", 1, k_block, k_block));
    model.max_instances(2);
    return model;
}

// Engine A on the CPU path and on k_npu: two plans of one row.
std::vector<anira_backend_id> two_providers_of_a() {
    return {{.struct_size = sizeof(anira_backend_id),
             .engine = ANIRA_ENGINE_CUSTOM,
             .provider = ANIRA_PROVIDER_CPU,
             .engine_id = k_engine_a},
            {.struct_size = sizeof(anira_backend_id),
             .engine = ANIRA_ENGINE_CUSTOM,
             .provider = ANIRA_PROVIDER_CUSTOM,
             .engine_id = k_engine_a,
             .provider_id = k_npu}};
}

// A handler over the model with the two counting engines A and B and the recording
// processor on its pipeline, prepared under `candidates` (none: the default set). The plans
// are the rows in order, each row's in candidate order.
class PlanRig {
public:
    PlanRig(const anira::ModelConfig& model,
            Recorder& recorder,
            const std::vector<anira_backend_id>& candidates = {})
        : m_context(2) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &err), ANIRA_OK) << err.message;
        anira_custom_engine* const a = engine_under(m_a, k_engine_a, k_served);
        anira_custom_engine* const b = engine_under(m_b, k_engine_b);
        anira_test::add_engine(m_pipeline, a);
        anira_test::add_engine(m_pipeline, b);
        anira_custom_engine_destroy(a);
        anira_custom_engine_destroy(b);
        const std::array<const anira_model_config*, 1> variants{model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               variants.data(),
                                               1,
                                               candidates.empty() ? nullptr : candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &err),
                  ANIRA_OK)
            << err.message;
        const anira_processor_desc processor = recording_processor(recorder);
        EXPECT_EQ(anira_pipeline_add_processor(m_pipeline, &processor, &err), ANIRA_OK)
            << err.message;
        EXPECT_EQ(anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &err), ANIRA_OK)
            << err.message;
        EXPECT_EQ(anira_handler_prepare(m_handler,
                                        anira_test::explicit_contract(k_block,
                                                                      anira_test::k_rate,
                                                                      ANIRA_MISS_ZEROS,
                                                                      0.0,
                                                                      /*budget_ms=*/10.0)
                                            .native(),
                                        &err),
                  ANIRA_OK)
            << err.message;
    }
    ~PlanRig() {
        m_a.m_open.store(true);
        m_b.m_open.store(true);
        anira_handler_destroy(m_handler);
        anira_pipeline_destroy(m_pipeline);
    }
    PlanRig(const PlanRig&) = delete;
    PlanRig& operator=(const PlanRig&) = delete;
    PlanRig(PlanRig&&) = delete;
    PlanRig& operator=(PlanRig&&) = delete;

    anira_handler* handler() const { return m_handler; }
    InferenceManager& manager() const { return *m_handler->m_manager; }

    /// One block through anira_handler_process, in place.
    void process_block() {
        std::vector<float> io(k_block, 0.5F);
        const std::array<float*, 1> channels{io.data()};
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_block)};
        anira_tensor block{};
        anira_tensor_init_host_planar(&block,
                                      static_cast<const void*>(channels.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());
        size_t delivered = 0;
        static_cast<void>(anira_handler_process(m_handler, &block, 0, &block, 0, &delivered));
    }

    anira_test::GateEngine m_a;  ///< before the handler, which dies first
    anira_test::GateEngine m_b;

private:
    anira_test::Context m_context;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
};

// Polls the available samples (which collects the completed chunk and so runs its
// post_process on this thread) until `count` chunks have been post-processed.
bool wait_for_post(const PlanRig& rig, const Recorder& recorder, size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + k_wait;
    while (std::chrono::steady_clock::now() < deadline) {
        static_cast<void>(anira_test::available(rig.handler()));
        if (recorder.m_num_post >= count) { return true; }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return false;
}

}  // namespace

TEST(PlanSwitch, AChunkInFlightFinishesOnThePlanItWasSubmittedUnder) {
    Recorder recorder;
    PlanRig rig(rows_on({k_engine_a, k_engine_b}), recorder);
    ASSERT_EQ(anira_handler_get_plan(rig.handler()), 0U) << "plan 0 runs on engine A";

    // Chunk 1 is submitted under plan 0 and held inside engine A's call.
    rig.m_a.m_open.store(false);
    rig.process_block();
    const auto deadline = std::chrono::steady_clock::now() + k_wait;
    while (!recorder.m_in_hook.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_TRUE(recorder.m_in_hook.load()) << "chunk 1 never reached before_inference";

    // The switch lands while chunk 1 is in flight.
    ASSERT_EQ(anira_handler_set_plan(rig.handler(), 1), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(rig.handler()), 1U);
    rig.m_a.m_open.store(true);
    ASSERT_TRUE(wait_for_post(rig, recorder, 1)) << "chunk 1 never completed";

    // Chunk 1 ran on engine A end to end: engine B was not called, and every step was told
    // the same engine.
    EXPECT_EQ(rig.m_a.m_calls.load(), 1);
    EXPECT_EQ(rig.m_b.m_calls.load(), 0) << "the switch reached the in-flight chunk";
    ASSERT_EQ(recorder.m_num_pre, 1U);
    EXPECT_EQ(recorder.m_pre[0], 0);
    EXPECT_EQ(recorder.m_before.load(), 0);
    EXPECT_EQ(recorder.m_after.load(), 0);
    ASSERT_EQ(recorder.m_num_post, 1U);
    EXPECT_EQ(recorder.m_post[0], 0) << "post_process was told the session's new plan";

    // Chunk 2 is submitted after the switch and runs on engine B, exactly once.
    rig.process_block();
    ASSERT_TRUE(wait_for_post(rig, recorder, 2)) << "chunk 2 never completed";
    EXPECT_EQ(rig.m_b.m_calls.load(), 1);
    EXPECT_EQ(rig.m_a.m_calls.load(), 1);
    ASSERT_EQ(recorder.m_num_pre, 2U);
    EXPECT_EQ(recorder.m_pre[1], 1);
    EXPECT_EQ(recorder.m_before.load(), 1);
    EXPECT_EQ(recorder.m_after.load(), 1);
    EXPECT_EQ(recorder.m_post[1], 1);
}

// Without a switch: every step of a chunk is told the plan it was stamped with, and exactly
// one engine runs.
TEST(PlanSwitch, EveryStepOfAChunkIsToldTheStampedPlan) {
    Recorder recorder;
    PlanRig rig(rows_on({k_engine_a, k_engine_b}), recorder);
    ASSERT_EQ(anira_handler_set_plan(rig.handler(), 1), ANIRA_OK);

    rig.process_block();
    ASSERT_TRUE(wait_for_post(rig, recorder, 1)) << "the chunk never completed";

    EXPECT_EQ(rig.m_b.m_calls.load(), 1) << "exactly one engine runs per chunk";
    EXPECT_EQ(rig.m_a.m_calls.load(), 0);
    ASSERT_EQ(recorder.m_num_pre, 1U);
    EXPECT_EQ(recorder.m_pre[0], 1);
    EXPECT_EQ(recorder.m_before.load(), 1);
    EXPECT_EQ(recorder.m_after.load(), 1);
    ASSERT_EQ(recorder.m_num_post, 1U);
    EXPECT_EQ(recorder.m_post[0], 1);
}

// What the index buys: two plans on one engine are two selections. A selection stored as an
// engine could not tell them apart (it would read back as the first).
TEST(PlanSwitch, TwoPlansOnOneEngineStayDistinct) {
    Recorder recorder;
    PlanRig rig(rows_on({k_engine_a}), recorder, two_providers_of_a());
    EXPECT_EQ(anira_handler_get_plan(rig.handler()), 0U) << "a new table selects plan 0";
    ASSERT_EQ(anira_handler_set_plan(rig.handler(), 1), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(rig.handler()), 1U);

    // A chunk submitted under plan 1 carries the index 1, not "the plan of engine A".
    rig.process_block();
    ASSERT_TRUE(wait_for_post(rig, recorder, 1)) << "the chunk never completed";
    EXPECT_EQ(rig.m_a.m_calls.load(), 1);
    EXPECT_EQ(recorder.m_post[0], 0) << "both plans run on engine A";

    const std::shared_ptr<SessionElement> session = anira_test::session_of(rig.handler());
    ASSERT_NE(session, nullptr);
    size_t stamped = 0;
    for (const auto& chunk : session->m_inference_queue) {
        if (chunk->m_plan == 1U) { ++stamped; }
    }
    EXPECT_EQ(stamped, 1U) << "exactly the submitted chunk carries plan 1";
}

TEST(PlanSwitch, AnIndexOutOfRangeLeavesTheSelection) {
    Recorder recorder;
    const PlanRig rig(rows_on({k_engine_a}), recorder, two_providers_of_a());
    ASSERT_TRUE(rig.manager().set_plan(1));
    EXPECT_FALSE(rig.manager().set_plan(2));
    EXPECT_EQ(rig.manager().get_plan(), 1U);
}

// The table is read without synchronization once chunks exist, so it is the session's from
// create and never replaced: the plans of the handler's table are the plans the prepared
// session holds, each on a loaded model of its own.
TEST(PlanSwitch, ThePlanTableIsTheSessionsFromCreate) {
    Recorder recorder;
    const PlanRig rig(rows_on({k_engine_a}), recorder, two_providers_of_a());
    const std::shared_ptr<SessionElement> session = anira_test::session_of(rig.handler());
    ASSERT_NE(session, nullptr);
    ASSERT_EQ(session->m_plans.size(), 2U);
    ASSERT_NE(session->m_plans[0].m_loaded, nullptr);
    ASSERT_NE(session->m_plans[1].m_loaded, nullptr);
    EXPECT_NE(session->m_plans[0].m_loaded, session->m_plans[1].m_loaded)
        << "two providers of a row are two records, so two loaded models";
}
