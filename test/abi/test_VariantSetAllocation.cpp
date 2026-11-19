// What anira_handler_add_variant and anira_handler_remove_variant ask the allocator for, and
// what an allocation that fails does to them (src/capi/handler.cpp, "the variants of a
// handler"). Either entry does everything that can fail first, the allocations of its two
// publications included (thl::RCU::stage_replace), and ends in a part that cannot fail and
// allocates nothing (commit_added_plans, finish_removal). What the suite pins:
//
//   the last part     asks for no memory, counted between the two calls of the handler's probe
//                     (anira_handler::m_allocation_probe): for an add, also with a read lease
//                     out on either published table, and for a removal, also when it finds its
//                     variant selected by an entry that raced it;
//   a failed call     with each allocation of the call failing in turn, the entry returns
//                     ANIRA_ERROR_OUT_OF_MEMORY and its last part never started; what a host and
//                     the handler's other threads read is as before the call (the report object,
//                     the published rows and the published table by their addresses, the counts,
//                     the latency, the selection, rt_error); what the call built is released
//                     (every load has its unload, every prepare its unprepare, the pool has the
//                     models it had); the stream goes on; the same call then succeeds; and once
//                     everything is torn down, as many blocks are alive in the process as before
//                     the handler existed.
//
// One part of an add is left out of "each allocation": the JSON record of the variant, which a
// registered engine's plans are pooled by (anira::capi::model_config_json). It is made with
// nlohmann::json, whose destructor allocates when it destroys a non-empty array or object and
// is noexcept, so a request that fails there terminates the process, whatever anira does
// around it. The handler's probe names that part too, and no request made in it is failed.
//
// This file replaces the global operator new and operator delete, which is why it is an
// executable of its own (test_abi_allocation; test/CMakeLists.txt says where it is built and
// why not everywhere): every request a thread makes through them is counted for that thread,
// the n-th request of a thread can be made to fail, and the blocks alive in the process are
// counted. The requests are per thread because the control thread is not the only one that
// runs during an add: the inference threads run its warm-up beside it. The binary is white-box,
// so the library's objects are part of it and every allocation they make goes through the
// operators below.
//
// Engine-free: every plan runs on one registered engine whose loaded model is a gain, the
// factor the number its entry's path spells, and which allocates nothing itself, so every
// request of a call is the library's.

#include <anira/abi/config.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "capi/handler.h"
#include "handler_support.h"
#include "scheduler/Core.h"
#include "scheduler/SessionElement.h"

namespace {

// ---- the allocator ---------------------------------------------------------------------------

/// What the calling thread did through operator new. Thread-local, so that what an inference
/// thread allocates meanwhile is not counted and never fails, and constant-initialised, so that
/// reading it allocates nothing.
struct AllocationProbe {
    uint64_t m_requests = 0;  ///< calls of operator new, failed ones included
    uint64_t m_fail_in = 0;   ///< above 0: the request that many from now fails, once
    bool m_exempt = false;    ///< inside a part of a call whose requests are not failed
    uint64_t m_exempted = 0;  ///< the requests made there
};

constinit thread_local AllocationProbe t_probe;

/// The blocks operator new gave out and operator delete did not get back yet, of every thread.
constinit std::atomic<int64_t> g_live_blocks{0};

bool request_fails() noexcept {
    ++t_probe.m_requests;
    if (t_probe.m_exempt) {
        ++t_probe.m_exempted;
        return false;
    }
    if (t_probe.m_fail_in == 0) { return false; }
    --t_probe.m_fail_in;
    return t_probe.m_fail_in == 0;
}

void* counted(void* block) noexcept {
    if (block != nullptr) { g_live_blocks.fetch_add(1, std::memory_order_relaxed); }
    return block;
}

void* allocate(std::size_t size) noexcept {
    if (request_fails()) { return nullptr; }
    return counted(std::malloc(std::max<std::size_t>(size, 1)));
}

void* allocate(std::size_t size, std::align_val_t alignment) noexcept {
    if (request_fails()) { return nullptr; }
    void* block = nullptr;
    const std::size_t at_least_a_pointer =
        std::max(static_cast<std::size_t>(alignment), sizeof(void*));
    if (posix_memalign(&block, at_least_a_pointer, std::max<std::size_t>(size, 1)) != 0) {
        return nullptr;
    }
    return counted(block);
}

void release(void* block) noexcept {
    if (block == nullptr) { return; }
    g_live_blocks.fetch_sub(1, std::memory_order_relaxed);
    std::free(block);
}

void* or_bad_alloc(void* block) {
    if (block == nullptr) { throw std::bad_alloc(); }
    return block;
}

}  // namespace

// Every replaceable form, so that no allocation of this executable goes around the probe and no
// block is freed by another allocator than the one it is from.
void* operator new(std::size_t size) {
    return or_bad_alloc(allocate(size));
}
void* operator new[](std::size_t size) {
    return or_bad_alloc(allocate(size));
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
    return allocate(size);
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
    return allocate(size);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    return or_bad_alloc(allocate(size, alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return or_bad_alloc(allocate(size, alignment));
}
void* operator new(std::size_t size,
                   std::align_val_t alignment,
                   const std::nothrow_t& /*tag*/) noexcept {
    return allocate(size, alignment);
}
void* operator new[](std::size_t size,
                     std::align_val_t alignment,
                     const std::nothrow_t& /*tag*/) noexcept {
    return allocate(size, alignment);
}

void operator delete(void* block) noexcept {
    release(block);
}
void operator delete[](void* block) noexcept {
    release(block);
}
void operator delete(void* block, std::size_t /*size*/) noexcept {
    release(block);
}
void operator delete[](void* block, std::size_t /*size*/) noexcept {
    release(block);
}
void operator delete(void* block, const std::nothrow_t& /*tag*/) noexcept {
    release(block);
}
void operator delete[](void* block, const std::nothrow_t& /*tag*/) noexcept {
    release(block);
}
void operator delete(void* block, std::align_val_t /*alignment*/) noexcept {
    release(block);
}
void operator delete[](void* block, std::align_val_t /*alignment*/) noexcept {
    release(block);
}
void operator delete(void* block, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    release(block);
}
void operator delete[](void* block, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    release(block);
}
void operator delete(void* block,
                     std::align_val_t /*alignment*/,
                     const std::nothrow_t& /*tag*/) noexcept {
    release(block);
}
void operator delete[](void* block,
                       std::align_val_t /*alignment*/,
                       const std::nothrow_t& /*tag*/) noexcept {
    release(block);
}

namespace {

int64_t live_blocks() noexcept {
    return g_live_blocks.load(std::memory_order_relaxed);
}

// ---- the parts of a call the handler names ---------------------------------------------------

/// What the calling thread asked for inside the part of a control entry that cannot fail
/// (anira::capi::AllocationScope::Commit).
struct CommitCount {
    uint64_t m_commits = 0;   ///< the parts that ran to their end
    uint64_t m_requests = 0;  ///< the calls of operator new inside them
    uint64_t m_entered_at = 0;
};

constinit thread_local CommitCount t_commit;

/// The handler's probe (anira_handler::m_allocation_probe), which is told on the thread of the
/// entry when a part starts and when it is done. The last part of an add or a removal: its
/// requests are counted. The JSON record of the variant: its requests are not failed, since a
/// request that fails inside nlohmann::json's destructor terminates the process (the library's
/// one known place of an add where an allocation that fails is no status; AllocationScope says
/// why).
void probe(anira::capi::AllocationScope scope, bool entered) noexcept {
    if (scope == anira::capi::AllocationScope::ModelRecord) {
        t_probe.m_exempt = entered;
        return;
    }
    if (entered) {
        t_commit.m_entered_at = t_probe.m_requests;
        return;
    }
    t_commit.m_requests += t_probe.m_requests - t_commit.m_entered_at;
    ++t_commit.m_commits;
}

// ---- the engine and the models ---------------------------------------------------------------

constexpr size_t k_hop = 64;       // the window of the stream, and the contract's block
constexpr double k_rate = 6400.0;  // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr double k_budget_ms = 1.0;
constexpr uint32_t k_warmup = 2;  // the inferences of a plan's pass (ANIRA_WARMUP_FIXED)
constexpr const char* k_engine = "org.example.variant_set_allocation";

/// The gains a model of the suite can be: the factor is the index, which an entry's path spells.
constexpr std::array<float, 8> k_factors{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};

/// A registered C engine whose loaded model is a gain: output 0 is input 0 times the factor its
/// entry's path spells ("2" is 2). It allocates nothing: its loaded model and its prepared
/// handle are the address of the factor in k_factors. Every slot of the lifecycle is counted.
struct GainEngine {
    std::atomic<int> m_loads{0};
    std::atomic<int> m_unloads{0};
    std::atomic<int> m_prepares{0};
    std::atomic<int> m_unprepares{0};
};

anira_status ANIRA_CALL gain_load(const anira_engine_load_info* info,
                                  void* user_data,
                                  void** out_loaded) {
    const char* path = anira_model_config_model_path(info->model, info->row);
    if (path == nullptr) { return ANIRA_ERROR_CONFIG; }
    const auto factor = static_cast<size_t>(std::atoi(path));
    if (factor == 0 || factor >= k_factors.size()) { return ANIRA_ERROR_MODEL_LOAD; }
    static_cast<GainEngine*>(user_data)->m_loads.fetch_add(1);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) the handle is never written through
    *out_loaded = const_cast<float*>(&k_factors.at(factor));
    return ANIRA_OK;
}

void ANIRA_CALL gain_unload(void* /*loaded*/, void* user_data) {
    static_cast<GainEngine*>(user_data)->m_unloads.fetch_add(1);
}

anira_status ANIRA_CALL gain_prepare(const anira_prepare_info* /*info*/,
                                     void* loaded,
                                     void* user_data,
                                     void** out_prepared) {
    if (loaded == nullptr) { return ANIRA_ERROR_INTERNAL; }
    static_cast<GainEngine*>(user_data)->m_prepares.fetch_add(1);
    *out_prepared = loaded;
    return ANIRA_OK;
}

void ANIRA_CALL gain_unprepare(void* /*prepared*/, void* user_data) {
    static_cast<GainEngine*>(user_data)->m_unprepares.fetch_add(1);
}

anira_status ANIRA_CALL gain_process(const anira_engine_ctx* ctx,
                                     void* prepared,
                                     void* /*user_data*/) {
    if (prepared == nullptr) { return ANIRA_ERROR_INTERNAL; }
    const float gain = *static_cast<const float*>(prepared);
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    for (size_t i = 0; i < anira_test::elements_of(ctx->outputs[0]); ++i) { out[i] = in[i] * gain; }
    return ANIRA_OK;
}

/// A mono stream of k_hop samples per inference on either side ("in", "out") over one entry of
/// the engine whose path spells the model's factor.
anira::ModelConfig gain_model(const char* factor) {
    const auto hop = static_cast<int64_t>(k_hop);
    anira::ModelConfig model;
    model.add_model_path(k_engine, factor);
    model.input(anira_test::stream_tensor("in", 1, hop, hop));
    model.output(anira_test::stream_tensor("out", 1, hop, hop));
    model.max_instances(2);
    return model;
}

// ---- the rig ---------------------------------------------------------------------------------

/// A context, a pipeline with the engine and an inference stage over `variants` under the
/// default candidate set, and the handler over it, prepared under an explicit budget with a
/// warm-up pass of k_warmup inferences per plan (an add then runs a pass of its own on the
/// inference threads) and with the probe of this file on its last parts.
class Rig {
public:
    Rig(std::initializer_list<const anira::ModelConfig*> variants,
        anira_log_level level = ANIRA_LOG_ERROR)
        : m_context(2, ANIRA_WAIT_SPIN_BACKOFF, level) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &err), ANIRA_OK) << err.message;
        anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
        desc.user_data = &m_engine;
        desc.load = gain_load;
        desc.unload = gain_unload;
        desc.prepare = gain_prepare;
        desc.unprepare = gain_unprepare;
        desc.process = gain_process;
        anira_custom_engine* object = nullptr;
        EXPECT_EQ(anira_custom_engine_create(k_engine, &desc, &object, &err), ANIRA_OK)
            << err.message;
        anira_test::add_engine(m_pipeline, object);
        anira_custom_engine_destroy(object);
        std::vector<const anira_model_config*> natives;
        natives.reserve(variants.size());
        for (const anira::ModelConfig* variant : variants) { natives.push_back(variant->native()); }
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               natives.data(),
                                               static_cast<uint32_t>(natives.size()),
                                               nullptr,
                                               0,
                                               &err),
                  ANIRA_OK)
            << err.message;
        EXPECT_EQ(anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &err), ANIRA_OK)
            << err.message;
        const anira::ContractHandle contract =
            anira_test::explicit_contract(static_cast<uint32_t>(k_hop),
                                          k_rate,
                                          ANIRA_MISS_ZEROS,
                                          0.0,
                                          k_budget_ms,
                                          k_warmup);
        EXPECT_EQ(anira_handler_prepare(m_handler, contract.native(), &err), ANIRA_OK)
            << err.message;
        if (m_handler != nullptr) { m_handler->m_allocation_probe = probe; }
        t_commit = CommitCount{};
    }
    ~Rig() {
        anira_handler_destroy(m_handler);
        anira_pipeline_destroy(m_pipeline);
    }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;
    Rig(Rig&&) = delete;
    Rig& operator=(Rig&&) = delete;

    anira_handler* h() const { return m_handler; }
    anira::SessionElement& session() const { return m_handler->m_manager->session(); }

    /// anira_handler_add_variant with the calling thread's `fail_in`-th request from now
    /// failing (0: none does). The status, the message kept, and whether that request was
    /// reached, which is whether it failed.
    anira_status add(const anira::ModelConfig& model, uint32_t* variant, uint64_t fail_in = 0) {
        anira_error err = ANIRA_ERROR_INIT;
        t_probe.m_fail_in = fail_in;
        const anira_status status =
            anira_handler_add_variant(m_handler, model.native(), variant, &err);
        note_failure(fail_in);
        m_message = err.message;
        return status;
    }

    /// anira_handler_remove_variant, likewise.
    anira_status remove(uint32_t variant, uint64_t fail_in = 0) {
        anira_error err = ANIRA_ERROR_INIT;
        t_probe.m_fail_in = fail_in;
        const anira_status status = anira_handler_remove_variant(m_handler, variant, &err);
        note_failure(fail_in);
        m_message = err.message;
        return status;
    }

    GainEngine m_engine;  ///< before the handler, which dies first
    std::string m_message;
    bool m_failed = false;  ///< the last call reached the request that was to fail

private:
    void note_failure(uint64_t fail_in) {
        m_failed = fail_in != 0 && t_probe.m_fail_in == 0;
        t_probe.m_fail_in = 0;
    }

    anira_test::Context m_context;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
};

/// The states of the rows of a plan report, in order.
std::vector<uint32_t> states_of(const anira_plan_report* report) {
    uint32_t count = anira_plan_report_num_plans(report);
    std::vector<anira_plan_info> rows(count);
    if (count != 0) {
        EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
                  ANIRA_OK);
    }
    std::vector<uint32_t> states;
    states.reserve(rows.size());
    for (const anira_plan_info& row : rows) { states.push_back(row.state); }
    return states;
}

constexpr auto k_ready = static_cast<uint32_t>(ANIRA_PLAN_READY);
constexpr auto k_removed = static_cast<uint32_t>(ANIRA_PLAN_REMOVED);

/// What a call that failed must leave as it was: what a host reads of the handler, what its
/// other threads read (the two published versions, by their addresses: a publication makes a
/// new object), what the handler and its session hold behind that, and what the call's build
/// must have given back (the engine's loaded models and prepared handles, the pool's models).
struct Snapshot {
    explicit Snapshot(const Rig& rig)
        : m_report(anira_handler_plan_report(rig.h()))
        , m_states(states_of(m_report))
        , m_latency(anira_handler_get_latency(rig.h(), 0))
        , m_entries(anira_handler_num_entries(rig.h()))
        , m_rt_error(anira_handler_rt_error(rig.h()))
        , m_selected(anira_handler_get_plan(rig.h()))
        , m_rows(published_rows(rig))
        , m_table(published_table(rig))
        , m_variants(rig.h()->m_pipeline.m_variants.size())
        , m_removed(rig.h()->m_pipeline.m_removed.size())
        , m_candidates(rig.h()->m_pipeline.m_candidates.size())
        , m_derived(rig.h()->m_variant_derived.size())
        , m_plans(rig.h()->m_plans.size())
        , m_reports(rig.h()->m_reports.size())
        , m_strings(rig.h()->m_plan_strings.size())
        , m_session_plans(rig.session().m_plans.size())
        , m_session_slots(static_cast<size_t>(std::ranges::count_if(
              rig.session().m_plans,
              [](const std::shared_ptr<anira::SessionElement::PlanSlot>& slot) {
                  return slot != nullptr;
              })))
        , m_pool(anira::Core::get_num_loaded_models())
        , m_loaded(rig.m_engine.m_loads.load() - rig.m_engine.m_unloads.load())
        , m_prepared(rig.m_engine.m_prepares.load() - rig.m_engine.m_unprepares.load()) {}

    /// The version of the handler's plan rows a selection entry would read now.
    static const void* published_rows(const Rig& rig) {
        const auto lease = rig.h()->m_plan_row_readers.claim();
        return lease.has_value() ? &lease->data() : nullptr;
    }
    /// The version of the session's plan table the driving thread would stamp from now.
    static const void* published_table(const Rig& rig) {
        const auto lease = rig.session().claim_plans();
        return lease.has_value() ? &lease->data() : nullptr;
    }

    const anira_plan_report* m_report;
    std::vector<uint32_t> m_states;
    uint32_t m_latency;
    uint32_t m_entries;
    anira_status m_rt_error;
    uint32_t m_selected;
    const void* m_rows;
    const void* m_table;
    size_t m_variants;
    size_t m_removed;
    size_t m_candidates;
    size_t m_derived;
    size_t m_plans;
    size_t m_reports;
    size_t m_strings;
    size_t m_session_plans;
    size_t m_session_slots;
    size_t m_pool;
    int m_loaded;
    int m_prepared;
};

void expect_unchanged(const Rig& rig, const Snapshot& before) {
    const Snapshot now(rig);
    EXPECT_EQ(now.m_report, before.m_report) << "a failed call published a report";
    EXPECT_EQ(now.m_states, before.m_states);
    EXPECT_EQ(now.m_latency, before.m_latency);
    EXPECT_EQ(now.m_entries, before.m_entries);
    EXPECT_EQ(now.m_rt_error, before.m_rt_error);
    EXPECT_EQ(now.m_selected, before.m_selected);
    EXPECT_EQ(now.m_rows, before.m_rows) << "a failed call published the handler's plan rows";
    EXPECT_EQ(now.m_table, before.m_table) << "a failed call published the session's plan table";
    EXPECT_EQ(now.m_variants, before.m_variants) << "the pipeline copy kept or lost a variant";
    EXPECT_EQ(now.m_removed, before.m_removed);
    EXPECT_EQ(now.m_candidates, before.m_candidates) << "the candidate set kept its growth";
    EXPECT_EQ(now.m_derived, before.m_derived);
    EXPECT_EQ(now.m_plans, before.m_plans);
    EXPECT_EQ(now.m_reports, before.m_reports);
    EXPECT_EQ(now.m_strings, before.m_strings);
    EXPECT_EQ(now.m_session_plans, before.m_session_plans) << "the session's list kept a slot";
    EXPECT_EQ(now.m_session_slots, before.m_session_slots) << "the session's list lost a slot";
    EXPECT_EQ(now.m_pool, before.m_pool) << "the pool kept or lost a loaded model";
    EXPECT_EQ(now.m_loaded, before.m_loaded) << "a load of the failed call has no unload";
    EXPECT_EQ(now.m_prepared, before.m_prepared) << "a prepare of it has no unprepare";
}

// ---- the stream ------------------------------------------------------------------------------

/// The driving thread of a rig, played by the test's own thread between two control calls: one
/// block of a ramp per call of anira_handler_process, in place, each block's inference waited
/// for, so the stream is complete and block k of the output is the chunk of call k minus the
/// latency. Every block is checked against it: the ramp of that call times the factor of the
/// variant that was selected then, silence for the blocks of the latency.
class Stream {
public:
    explicit Stream(anira_handler* handler)
        : m_handler(handler), m_latency(anira_handler_get_latency(handler, 0) / k_hop) {
        EXPECT_EQ(anira_handler_get_latency(handler, 0) % k_hop, 0U)
            << "the latency is no whole number of blocks";
        m_factors.reserve(64);
    }

    /// The blocks until a chunk formed now comes out: the latency and the block itself.
    size_t round_trip() const { return m_latency + 1; }

    /// `count` blocks while the variant of `factor` is selected.
    void run(size_t count, float factor) {
        for (size_t i = 0; i < count; ++i) { one_block(factor); }
    }

    /// The factor the last block of the output carried (0 for silence).
    float last_factor() const { return m_last_factor; }

private:
    static float sample(size_t block, size_t i) {
        // Block 0 of the ramp starts at sample 0, which no factor shows on: it starts at block 1.
        return static_cast<float>((block + 1) * k_hop + i) / 65536.0F;
    }

    void one_block(float factor) {
        const size_t index = m_factors.size();
        std::array<float, k_hop> block{};
        for (size_t i = 0; i < k_hop; ++i) { block.at(i) = sample(index, i); }
        const std::array<float*, 1> planes{block.data()};
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_hop)};
        anira_tensor io{};
        anira_tensor_init_host_planar(&io,
                                      static_cast<const void*>(planes.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());
        const size_t before = anira_test::available(m_handler);
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_process(m_handler, &io, 0, &io, 0, &delivered), ANIRA_OK)
            << "block " << index;
        m_factors.push_back(factor);
        // The block's inference, collected, before the next call.
        anira_test::wait_for_available(m_handler, before);
        const bool silence = index < m_latency;
        m_last_factor = silence ? 0.0F : m_factors.at(index - m_latency);
        for (size_t i = 0; i < k_hop; ++i) {
            const float expected = silence ? 0.0F : sample(index - m_latency, i) * m_last_factor;
            ASSERT_EQ(block.at(i), expected) << "block " << index << ", sample " << i;
        }
    }

    anira_handler* m_handler;
    size_t m_latency;              ///< in blocks
    std::vector<float> m_factors;  ///< per call: the factor of the variant selected then
    float m_last_factor = 0.0F;
};

// ---- the last part allocates nothing ---------------------------------------------------------

// The part of an add that makes the new plans visible asks for no memory: with nothing out,
// and with a read lease out on the session's plan table and on the handler's plan rows, which
// keeps the versions the commit replaces from being freed (they go into the room the stage
// made for them).
TEST(VariantSetAllocation, TheCommitOfAnAddAllocatesNothing) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig three = gain_model("3");
    const anira::ModelConfig four = gain_model("4");
    Rig rig({&two});
    Stream stream(rig.h());
    stream.run(stream.round_trip(), 2.0F);

    uint32_t variant = 0;
    const uint64_t requests = t_probe.m_requests;
    ASSERT_EQ(rig.add(three, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_GT(t_probe.m_requests, requests) << "the operators of this file do not see the call";
    EXPECT_EQ(variant, 1U);
    EXPECT_EQ(t_commit.m_commits, 1U);
    EXPECT_EQ(t_commit.m_requests, 0U) << "the commit of an add asked for memory";

    {
        const auto table = rig.session().claim_plans();
        const auto rows = rig.h()->m_plan_row_readers.claim();
        if (!table.has_value() || !rows.has_value()) {
            FAIL() << "no free reader slot on an idle handler";
            return;
        }
        ASSERT_EQ(rig.add(four, &variant), ANIRA_OK) << rig.m_message;
        EXPECT_EQ(table->data().m_slots.size(), 2U) << "a lease keeps the version of its claim";
        EXPECT_EQ(rows->data().m_rows.size(), 2U);
    }
    EXPECT_EQ(variant, 2U);
    EXPECT_EQ(t_commit.m_commits, 2U);
    EXPECT_EQ(t_commit.m_requests, 0U) << "the commit of an add asked for memory, a lease out";

    // What was committed is what the stream runs.
    EXPECT_EQ(states_of(anira_handler_plan_report(rig.h())),
              (std::vector<uint32_t>{k_ready, k_ready, k_ready}));
    ASSERT_EQ(anira_handler_set_model(rig.h(), 2), ANIRA_OK);
    stream.run(stream.round_trip(), 4.0F);
    EXPECT_EQ(stream.last_factor(), 4.0F);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);
}

// The part of a removal behind its first publication asks for no memory.
TEST(VariantSetAllocation, TheTailOfARemovalAllocatesNothing) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig three = gain_model("3");
    Rig rig({&two, &three});
    Stream stream(rig.h());
    stream.run(stream.round_trip(), 2.0F);

    const uint64_t requests = t_probe.m_requests;
    ASSERT_EQ(rig.remove(1), ANIRA_OK) << rig.m_message;
    EXPECT_GT(t_probe.m_requests, requests) << "the operators of this file do not see the call";
    EXPECT_EQ(t_commit.m_commits, 1U);
    EXPECT_EQ(t_commit.m_requests, 0U) << "the tail of a removal asked for memory";

    EXPECT_EQ(states_of(anira_handler_plan_report(rig.h())),
              (std::vector<uint32_t>{k_ready, k_removed}));
    EXPECT_EQ(rig.m_engine.m_unprepares.load(), 1);
    EXPECT_EQ(rig.m_engine.m_unloads.load(), 1);
    EXPECT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_ERROR_NOT_FOUND);
    stream.run(stream.round_trip(), 2.0F);
    EXPECT_EQ(stream.last_factor(), 2.0F);
}

// Nor does it when it finds its variant selected: a selection entry that read the plan rows
// while the variant's plan was ready may store it when the removal is past its refusal, and the
// removal moves that selection to the first ready plan and says so in one Warning, which is
// logged behind the tail (a log record allocates). The racing entry is played by this thread:
// it holds a read lease on the rows, lets the removal publish the marked rows and wait, stores
// the removed plan the way anira_handler_set_plan stores inside its lease, and gives the lease
// back. The removal runs on a thread of its own, whose requests are the ones counted.
TEST(VariantSetAllocation, TheTailOfARemovalThatMovesASelectionAllocatesNothing) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig three = gain_model("3");
    Rig rig({&two, &three}, ANIRA_LOG_WARNING);

    anira_status status = ANIRA_ERROR_INTERNAL;
    CommitCount counted;
    std::thread remover;
    {
        const auto lease = rig.h()->m_plan_row_readers.claim();
        if (!lease.has_value()) {
            FAIL() << "no free reader slot on an idle handler";
            return;
        }
        remover = std::thread([&] {
            t_commit = CommitCount{};
            status = anira_handler_remove_variant(rig.h(), 1, nullptr);
            counted = t_commit;
        });
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(anira_test::k_wait_s);
        while (states_of(anira_handler_plan_report(rig.h())) !=
               std::vector<uint32_t>{k_ready, k_removed}) {
            if (std::chrono::steady_clock::now() > deadline) { break; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        // The store of the entry that raced: plan 1 was ready in the rows this lease holds.
        EXPECT_EQ(lease->data().m_rows.at(1).state, k_ready);
        rig.h()->m_selected_plan.store(1, std::memory_order_relaxed);
    }
    remover.join();
    EXPECT_EQ(status, ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(rig.h()), 0U) << "the first ready plan";
    EXPECT_EQ(counted.m_commits, 1U);
    EXPECT_EQ(counted.m_requests, 0U) << "the tail of a removal that moved a selection asked "
                                         "for memory";
}

// ---- a call that cannot allocate changes nothing ---------------------------------------------

/// What one run of a call with a failing request showed.
struct Run {
    bool m_failed = false;             ///< the request that was to fail was reached
    anira_status m_status = ANIRA_OK;  ///< of the call with the failing request
    uint64_t m_requests = 0;           ///< the calling thread's requests during that call
    uint64_t m_exempted = 0;           ///< those of them that are not failed (the model's record)
};

/// The calling thread's requests during one call, for a Run.
class RequestCount {
public:
    void into(Run& run) const {
        run.m_requests = t_probe.m_requests - m_requests;
        run.m_exempted = t_probe.m_exempted - m_exempted;
    }

private:
    uint64_t m_requests = t_probe.m_requests;
    uint64_t m_exempted = t_probe.m_exempted;
};

// One add on a rig of its own, with the calling thread's `fail_in`-th request of the call
// failing (0: none). A call that reports the failure must have changed nothing, and the same
// call then goes through; a call that went through must have added the variant.
Run add_with_failing_request(uint64_t fail_in) {
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig three = gain_model("3");
    Rig rig({&two});
    Stream stream(rig.h());
    stream.run(stream.round_trip(), 2.0F);
    const Snapshot before(rig);

    Run run;
    uint32_t variant = 99;
    const RequestCount requests;
    run.m_status = rig.add(three, &variant, fail_in);
    requests.into(run);
    run.m_failed = rig.m_failed;
    if (run.m_status != ANIRA_OK) {
        EXPECT_TRUE(run.m_failed) << "the call failed for another reason: " << rig.m_message;
        EXPECT_EQ(run.m_status, ANIRA_ERROR_OUT_OF_MEMORY) << rig.m_message;
        EXPECT_EQ(variant, 99U) << "a failed add wrote the index";
        EXPECT_EQ(t_commit.m_commits, 0U) << "the commit ran in a call that failed";
        expect_unchanged(rig, before);
        stream.run(stream.round_trip(), 2.0F);
        EXPECT_EQ(stream.last_factor(), 2.0F);
        // The same call, with memory.
        EXPECT_EQ(rig.add(three, &variant), ANIRA_OK) << rig.m_message;
    }
    EXPECT_EQ(variant, 1U);
    EXPECT_EQ(t_commit.m_commits, 1U);
    EXPECT_EQ(t_commit.m_requests, 0U) << "the commit of an add asked for memory";
    EXPECT_EQ(states_of(anira_handler_plan_report(rig.h())),
              (std::vector<uint32_t>{k_ready, k_ready}));
    EXPECT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    stream.run(stream.round_trip(), 3.0F);
    EXPECT_EQ(stream.last_factor(), 3.0F);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);
    return run;
}

// One removal on a rig of its own, likewise. A call that reports the failure must have changed
// nothing: the variant is still there and still runs, and the same call then goes through.
Run remove_with_failing_request(uint64_t fail_in) {
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig three = gain_model("3");
    Rig rig({&two, &three});
    Stream stream(rig.h());
    stream.run(stream.round_trip(), 2.0F);
    const Snapshot before(rig);

    Run run;
    const RequestCount requests;
    run.m_status = rig.remove(1, fail_in);
    requests.into(run);
    run.m_failed = rig.m_failed;
    if (run.m_status != ANIRA_OK) {
        EXPECT_TRUE(run.m_failed) << "the call failed for another reason: " << rig.m_message;
        EXPECT_EQ(run.m_status, ANIRA_ERROR_OUT_OF_MEMORY) << rig.m_message;
        EXPECT_EQ(t_commit.m_commits, 0U) << "the tail ran in a call that failed";
        expect_unchanged(rig, before);
        // The variant the call did not remove runs.
        EXPECT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
        stream.run(stream.round_trip(), 3.0F);
        EXPECT_EQ(stream.last_factor(), 3.0F);
        EXPECT_EQ(anira_handler_set_model(rig.h(), 0), ANIRA_OK);
        // The same call, with memory.
        EXPECT_EQ(rig.remove(1), ANIRA_OK) << rig.m_message;
    }
    EXPECT_EQ(t_commit.m_commits, 1U);
    EXPECT_EQ(t_commit.m_requests, 0U) << "the tail of a removal asked for memory";
    EXPECT_EQ(states_of(anira_handler_plan_report(rig.h())),
              (std::vector<uint32_t>{k_ready, k_removed}));
    EXPECT_EQ(rig.m_engine.m_unprepares.load(), 1);
    EXPECT_EQ(rig.m_engine.m_unloads.load(), 1);
    stream.run(stream.round_trip(), 2.0F);
    EXPECT_EQ(stream.last_factor(), 2.0F);
    return run;
}

/// What fail_each_request() found.
struct Sweep {
    uint64_t m_failed = 0;  ///< the requests that were failed, one per run
    Run m_last;             ///< the run that no longer reached the request that was to fail
};

// Runs `call` (one of the two above) with its first request failing, then with its second, and
// so on until a run no longer reaches the request that was to fail: by then every request of
// the call that can be failed has failed once. Behind every run the core must go (nothing of
// the run is alive) and the process must hold as many blocks as before the first one.
template <typename Call>
Sweep fail_each_request(Call&& call) {
    const auto traced = [&call](uint64_t nth) {
        SCOPED_TRACE("request " + std::to_string(nth) + " of the call fails");
        return call(nth);
    };
    // Once without a failure and once with one: what the process allocates once and keeps (the
    // first use of a status text, of the log, of this thread's trace) exists before the blocks
    // are counted.
    traced(0);
    traced(1);
    EXPECT_TRUE(anira_test::fresh_core());
    const int64_t before = live_blocks();
    Sweep sweep;
    while (!::testing::Test::HasFailure()) {
        const uint64_t nth = sweep.m_failed + 1;
        sweep.m_last = traced(nth);
        EXPECT_TRUE(anira_test::fresh_core()) << "request " << nth;
        EXPECT_EQ(live_blocks(), before) << "request " << nth << " failed: blocks left behind";
        if (!sweep.m_last.m_failed) {
            EXPECT_EQ(sweep.m_last.m_status, ANIRA_OK);
            break;
        }
        EXPECT_EQ(sweep.m_last.m_status, ANIRA_ERROR_OUT_OF_MEMORY)
            << "request " << nth << " failed and the call went on";
        sweep.m_failed = nth;
    }
    return sweep;
}

// anira_handler_add_variant on a playing handler, each of its allocations failing in turn: the
// model's validation, the build's session with its load, its prepare and its warm-up pass, the
// rows, the report, and what its commit publishes. Every one but those of the model's JSON
// record, which nlohmann::json makes (see the head of this file).
TEST(VariantSetAllocation, AnAddThatCannotAllocateChangesNothing) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Sweep sweep = fail_each_request(add_with_failing_request);
    ASSERT_FALSE(HasFailure());
    // The two published versions, the report and the build's session alone are more than this.
    EXPECT_GE(sweep.m_failed, 32U) << "the operators of this file do not see the call";
    // The call asks for the same memory every time, so every request it makes was failed once
    // or is one of the record's.
    EXPECT_EQ(sweep.m_failed + sweep.m_last.m_exempted, sweep.m_last.m_requests);
    EXPECT_GT(sweep.m_last.m_exempted, 0U)
        << "the model's record makes no request any more: fail them like every other";
    RecordProperty("requests", std::to_string(sweep.m_last.m_requests));
    RecordProperty("requests_failed", std::to_string(sweep.m_failed));
    RecordProperty("requests_of_the_model_record", std::to_string(sweep.m_last.m_exempted));
}

// anira_handler_remove_variant on a playing handler, each of its allocations failing in turn:
// the plan list, the report, and what its tail publishes.
TEST(VariantSetAllocation, ARemovalThatCannotAllocateChangesNothing) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Sweep sweep = fail_each_request(remove_with_failing_request);
    ASSERT_FALSE(HasFailure());
    // The plan list, the report, the two published versions and the room of the lists.
    EXPECT_GE(sweep.m_failed, 8U) << "the operators of this file do not see the call";
    EXPECT_EQ(sweep.m_failed, sweep.m_last.m_requests) << "a request of the call was not failed";
    EXPECT_EQ(sweep.m_last.m_exempted, 0U);
    RecordProperty("requests", std::to_string(sweep.m_last.m_requests));
}

}  // namespace
