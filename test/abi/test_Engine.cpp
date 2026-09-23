// anira/abi/engine.h: the registration of a custom engine on a pipeline
// (anira_pipeline_register_engine) and the lifetime of its carrier, the stage's twin. What runs
// a registered engine arrives with the engine room of src/backends: in this build of the tree a
// registered engine is a registration alone, and a model entry naming its id is refused at
// anira_handler_create as it was before (LegalBeforeAndAfterAddInference says so); the runtime
// cases follow that commit.
#include <anira/abi/build_info.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "capi/engine.h"
#include "handler_support.h"

namespace {

using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::Context;
using anira_test::explicit_contract;
using anira_test::k_custom;

constexpr size_t k_hop = anira_test::k_block;

/// What an engine's callbacks count on its registration.
struct EngineLife {
    int m_released = 0;
    int m_prepared = 0;
    int m_unprepared = 0;
    int m_processed = 0;
};

anira_status ANIRA_CALL engine_process(const anira_engine_ctx* /*ctx*/,
                                       void* /*prepared*/,
                                       void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_processed;
    return ANIRA_OK;
}

void ANIRA_CALL engine_reset(const anira_engine_ctx* /*ctx*/,
                             void* /*prepared*/,
                             void* /*user_data*/) {}

anira_status ANIRA_CALL engine_prepare(const anira_engine_prepare_info* /*info*/,
                                       void* user_data,
                                       void** /*out_prepared*/) {
    ++static_cast<EngineLife*>(user_data)->m_prepared;
    return ANIRA_OK;
}

void ANIRA_CALL engine_unprepare(void* /*prepared*/, void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_unprepared;
}

void ANIRA_CALL engine_release(void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_released;
}

/// A descriptor with every slot filled and no promise, over `life`.
anira_engine_desc full_engine(EngineLife& life) {
    anira_engine_desc engine = ANIRA_ENGINE_DESC_INIT;
    engine.user_data = &life;
    engine.process = engine_process;
    engine.reset = engine_reset;
    engine.prepare = engine_prepare;
    engine.unprepare = engine_unprepare;
    engine.release = engine_release;
    return engine;
}

/// A mono stream, [1, 1, hop] in and out, on one model entry that names `engine_id`: the
/// engine-free custom row (anira.v2.custom) runs it as an exact pass-through; another id is
/// what a registered engine would run.
ModelConfig stream_model(const char* engine_id = k_custom) {
    ModelConfig model;
    model.add_model_path(engine_id, "custom-processor");
    TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_hop));
    in.window(static_cast<int64_t>(k_hop), static_cast<int64_t>(k_hop), 0);
    model.input(in);
    TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_hop));
    out.window(static_cast<int64_t>(k_hop), static_cast<int64_t>(k_hop), 0);
    model.output(out);
    return model;
}

/// A pipeline handle with its lifetime; the inference stage is added on request.
struct Pipe {
    Pipe() { EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message; }
    ~Pipe() { destroy(); }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    Pipe(Pipe&&) = delete;
    Pipe& operator=(Pipe&&) = delete;

    void add_inference(const ModelConfig& model) {
        const std::array<const anira_model_config*, 1> variants{model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline, variants.data(), 1, nullptr, 0, &m_err),
                  ANIRA_OK)
            << m_err.message;
    }
    anira_status register_engine(const char* id, const anira_engine_desc& engine) {
        m_err = ANIRA_ERROR_INIT;
        return anira_pipeline_register_engine(m_pipeline, id, &engine, &m_err);
    }
    anira_status create_handler(const Context& context, anira_handler** out) {
        m_err = ANIRA_ERROR_INIT;
        return anira_handler_create(context.m_context, m_pipeline, out, &m_err);
    }
    void destroy() {
        anira_pipeline_destroy(m_pipeline);
        m_pipeline = nullptr;
    }

    anira_pipeline* m_pipeline = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

}  // namespace

// Every ANIRA_ERROR_INVALID_ARGUMENT cause of the registration, named in the message: the three
// NULL arguments, an id that is no reverse-URI name or anira's own (anira.v2.custom among them),
// a short struct_size, a flags bit the header does not define, a NULL process and a NULL kinds
// array or entry with a count. A refused call creates no carrier and never calls release.
TEST(AbiEngine, RegisterRefusals) {
    Pipe pipe;
    EngineLife life;
    const anira_engine_desc engine = full_engine(life);
    const auto refused =
        [&pipe, &life](const char* id, const anira_engine_desc* desc, const char* fragment) {
            anira_error err = ANIRA_ERROR_INIT;
            EXPECT_EQ(anira_pipeline_register_engine(pipe.m_pipeline, id, desc, &err),
                      ANIRA_ERROR_INVALID_ARGUMENT)
                << id;
            EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
            EXPECT_TRUE(pipe.m_pipeline->m_engines.empty());
            EXPECT_EQ(life.m_released, 0);
        };
    {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_register_engine(nullptr, "org.example.gain", &engine, &err),
                  ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::strstr(err.message, "NULL pipeline"), nullptr) << err.message;
    }
    refused(nullptr, &engine, "NULL engine_id");
    refused("org.example.gain", nullptr, "NULL desc");
    // The id: a reverse-URI name, and not one of anira's own.
    refused("gain", &engine, "'.'");
    refused("anira.gain", &engine, "anira.");
    refused(k_custom, &engine, "anira.");
    // The descriptor.
    anira_engine_desc bad = engine;
    bad.struct_size = static_cast<uint32_t>(offsetof(anira_engine_desc, user_data));
    refused("org.example.gain", &bad, "struct_size");
    bad = engine;
    bad.flags = 8U;  // the reserved alias bit is not defined in this pre-release
    refused("org.example.gain", &bad, "flags");
    bad.flags = 0x80000000U | ANIRA_ENGINE_FLAG_REALTIME_SAFE;
    refused("org.example.gain", &bad, "flags");
    bad = engine;
    bad.process = nullptr;
    refused("org.example.gain", &bad, "process");
    bad = engine;
    bad.num_consumed_kinds = 1;
    refused("org.example.gain", &bad, "consumed_kinds");
    const std::array<const char*, 1> null_kind{nullptr};
    bad.consumed_kinds = null_kind.data();
    refused("org.example.gain", &bad, "consumed_kinds[0]");
    // The same descriptor, well formed, registers: the refusals were the arguments'.
    EXPECT_EQ(pipe.register_engine("org.example.gain", engine), ANIRA_OK) << pipe.m_err.message;
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);
}

// A caller compiled against a header that ends after process hands over the slots up to it:
// the rest reads as ANIRA_ENGINE_DESC_INIT, so the carrier's engine has no reset, no prepare,
// no unprepare and no release, whatever the memory beyond struct_size holds; the kinds are
// copied, the id owned.
TEST(AbiEngine, AShortDescriptorReadsAsTheDefaults) {
    Pipe pipe;
    EngineLife life;
    anira_engine_desc engine = full_engine(life);  // release is set: beyond struct_size, not read
    const std::array<const char*, 1> kinds{"model:entry"};
    engine.consumed_kinds = kinds.data();
    engine.num_consumed_kinds = 1;
    engine.flags = ANIRA_ENGINE_FLAG_DYNAMIC_TIME;
    engine.struct_size = static_cast<uint32_t>(offsetof(anira_engine_desc, process) +
                                               sizeof(anira_engine_process_fn));
    {
        const std::string id = "org.example.short";  // dies before the carrier does
        ASSERT_EQ(pipe.register_engine(id.c_str(), engine), ANIRA_OK) << pipe.m_err.message;
    }
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    const anira::capi::EngineCarrier& carrier = *pipe.m_pipeline->m_engines[0];
    EXPECT_EQ(carrier.id(), "org.example.short");
    const anira_engine_desc& kept = carrier.desc();
    EXPECT_EQ(kept.struct_size, sizeof(anira_engine_desc));
    EXPECT_EQ(kept.abi_version, ANIRA_ABI_VERSION);
    EXPECT_EQ(kept.user_data, &life);
    EXPECT_EQ(kept.flags, ANIRA_ENGINE_FLAG_DYNAMIC_TIME);
    EXPECT_EQ(kept.process, engine_process);
    EXPECT_EQ(kept.reset, nullptr);
    EXPECT_EQ(kept.prepare, nullptr);
    EXPECT_EQ(kept.unprepare, nullptr);
    EXPECT_EQ(kept.release, nullptr);
    ASSERT_EQ(carrier.consumed_kinds().size(), 1U);
    EXPECT_EQ(carrier.consumed_kinds()[0], "model:entry");
    ASSERT_NE(kept.consumed_kinds, nullptr);
    EXPECT_NE(kept.consumed_kinds, kinds.data()) << "copied into the carrier";
    EXPECT_STREQ(kept.consumed_kinds[0], "model:entry");
    pipe.destroy();
    EXPECT_EQ(life.m_released, 0) << "the release slot lay beyond struct_size";
}

// The abi_version of the descriptor is checked as anira_check_abi checks it: another major is
// ANIRA_ERROR_ABI_VERSION, no carrier, no release; the header's own registers.
TEST(AbiEngine, AbiVersionIsChecked) {
    Pipe pipe;
    EngineLife life;
    anira_engine_desc engine = full_engine(life);
    engine.abi_version = ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1U, 0U);
    EXPECT_EQ(pipe.register_engine("org.example.gain", engine), ANIRA_ERROR_ABI_VERSION);
    EXPECT_NE(std::strstr(pipe.m_err.message, "ABI"), nullptr) << pipe.m_err.message;
    EXPECT_TRUE(pipe.m_pipeline->m_engines.empty());
    engine.abi_version = ANIRA_ABI_VERSION;
    EXPECT_EQ(pipe.register_engine("org.example.gain", engine), ANIRA_OK) << pipe.m_err.message;
    EXPECT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);
}

// One carrier per id: a second registration of an id is ANIRA_ERROR_INVALID_STATE naming it,
// creates no carrier and never calls its release, while a malformed second descriptor reports
// its own fault first (the duplicate is checked last); another id registers beside the first.
TEST(AbiEngine, ADuplicateIdIsInvalidState) {
    Pipe pipe;
    EngineLife first;
    EngineLife second;
    EngineLife third;
    ASSERT_EQ(pipe.register_engine("org.example.twice", full_engine(first)), ANIRA_OK)
        << pipe.m_err.message;
    EXPECT_EQ(pipe.register_engine("org.example.twice", full_engine(second)),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(std::strstr(pipe.m_err.message, "org.example.twice"), nullptr) << pipe.m_err.message;
    EXPECT_NE(std::strstr(pipe.m_err.message, "already"), nullptr) << pipe.m_err.message;
    // The same descriptor a second time is a duplicate as well.
    EXPECT_EQ(pipe.register_engine("org.example.twice", full_engine(first)),
              ANIRA_ERROR_INVALID_STATE);
    // A malformed descriptor under the taken id is reported as such, not as a duplicate.
    anira_engine_desc bad = full_engine(second);
    bad.flags = 8U;
    EXPECT_EQ(pipe.register_engine("org.example.twice", bad), ANIRA_ERROR_INVALID_ARGUMENT);
    bad = full_engine(second);
    bad.process = nullptr;
    EXPECT_EQ(pipe.register_engine("org.example.twice", bad), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(pipe.register_engine("org.example.thrice", full_engine(third)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 2U);
    EXPECT_EQ(pipe.m_pipeline->m_engines[0]->id(), "org.example.twice");
    EXPECT_EQ(pipe.m_pipeline->m_engines[1]->id(), "org.example.thrice");
    pipe.destroy();
    EXPECT_EQ(first.m_released, 1);
    EXPECT_EQ(second.m_released, 0);
    EXPECT_EQ(third.m_released, 1);
}

// A registration is legal before and after anira_pipeline_add_inference, and a handler created
// from the pipeline carries the same carriers; a registered engine no entry names is not a
// plan and not an error, so the handler prepares and runs on the custom row as before. A row
// naming a registered id is not yet runnable in this build of the tree: the create refuses it
// as it did before the registration existed (the engine room lifts this).
TEST(AbiEngine, LegalBeforeAndAfterAddInference) {
    const Context context;
    Pipe pipe;
    EngineLife before;
    EngineLife after;
    ASSERT_EQ(pipe.register_engine("org.example.before", full_engine(before)), ANIRA_OK)
        << pipe.m_err.message;
    pipe.add_inference(stream_model());
    ASSERT_EQ(pipe.register_engine("org.example.after", full_engine(after)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 2U);

    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(handler->m_pipeline.m_engines.size(), 2U);
    EXPECT_EQ(handler->m_pipeline.m_engines[0].get(), pipe.m_pipeline->m_engines[0].get());
    EXPECT_EQ(handler->m_pipeline.m_engines[1].get(), pipe.m_pipeline->m_engines[1].get());
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler);
    ASSERT_NE(report, nullptr);
    EXPECT_EQ(anira_plan_report_num_plans(report), 1U) << "the custom row alone is a plan";
    EXPECT_EQ(before.m_prepared + after.m_prepared, 0) << "no row names either engine";
    anira_handler_destroy(handler);
    EXPECT_EQ(before.m_released + after.m_released, 0) << "the pipeline still holds them";

    // A row naming a registered id: refused at create in this build of the tree, with the
    // refusal of an unregistered custom id.
    Pipe named;
    EngineLife life;
    ASSERT_EQ(named.register_engine("org.example.after", full_engine(life)), ANIRA_OK)
        << named.m_err.message;
    named.add_inference(stream_model("org.example.after"));
    anira_handler* refused = nullptr;
    EXPECT_EQ(named.create_handler(context, &refused), ANIRA_ERROR_NOT_SUPPORTED)
        << named.m_err.message;
    EXPECT_NE(std::strstr(named.m_err.message, "org.example.after"), nullptr)
        << named.m_err.message;
    EXPECT_EQ(refused, nullptr);
    named.destroy();
    EXPECT_EQ(life.m_released, 1);
    pipe.destroy();
    EXPECT_EQ(before.m_released, 1);
    EXPECT_EQ(after.m_released, 1);
}

// The carrier is shared by the pipeline and every handler created from it: release fires
// exactly once, with the last of them, whatever the destroy order, after the handlers' sessions
// are gone.
TEST(AbiEngine, ReleaseFiresOnceWithTheLastCarrier) {
    const Context context;
    const ModelConfig model = stream_model();
    const anira::ContractHandle contract = explicit_contract();
    // The three orders: the pipeline first, between the handlers, last.
    const std::array<std::array<int, 3>, 3> orders{{{0, 1, 2}, {1, 0, 2}, {1, 2, 0}}};
    for (const std::array<int, 3>& order : orders) {
        SCOPED_TRACE("order " + std::to_string(order[0]) + std::to_string(order[1]) +
                     std::to_string(order[2]));
        EngineLife life;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine("org.example.shared", full_engine(life)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(model);
        anira_handler* first = nullptr;
        anira_handler* second = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &first), ANIRA_OK) << pipe.m_err.message;
        ASSERT_EQ(pipe.create_handler(context, &second), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
        const std::array<std::function<void()>, 3> destroy{
            [&pipe] { pipe.destroy(); },
            [&first] { anira_handler_destroy(first); },
            [&second] { anira_handler_destroy(second); }};
        for (size_t step = 0; step < 3; ++step) {
            EXPECT_EQ(life.m_released, 0) << "step " << step;
            destroy.at(static_cast<size_t>(order.at(step)))();
        }
        EXPECT_EQ(life.m_released, 1);
    }
}

// A handler copies the pipeline at create: an engine registered afterwards reaches the pipeline
// and not the handler, and its release fires with the pipeline, the handler sharing nothing of it.
TEST(AbiEngine, RegisteredAfterCreateDoesNotReachTheHandler) {
    const Context context;
    Pipe pipe;
    pipe.add_inference(stream_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    EngineLife life;
    ASSERT_EQ(pipe.register_engine("org.example.late", full_engine(life)), ANIRA_OK)
        << pipe.m_err.message;
    EXPECT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    EXPECT_TRUE(handler->m_pipeline.m_engines.empty());
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1) << "the handler holds no share of the late carrier";
    anira_handler_destroy(handler);
    EXPECT_EQ(life.m_released, 1);
}
