#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>

#include <anira/anira.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <thread>

#include "../abi/handler_support.h"
#include "gtest/gtest.h"

using namespace anira;

// Regression tests for issue #106: Core::create_session() used to count the
// session before doing the work that can throw (loading the model in a backend
// processor's constructor, a custom processor's prepare()) and registered it only
// afterwards. A throw therefore left the active-session counter permanently
// incremented with nothing registered, and since the thread-pool teardown was gated
// on that counter reaching zero, the inference threads (and the context) lived on for
// the rest of the process. Registration is now the last step of create_session(),
// and everything before it is rolled back on failure. The sessions are C handlers'
// (anira_handler_prepare creates the session); a failed prepare returns the status.

namespace {

// The pool's threads count themselves as active when they enter their loop, i.e.
// asynchronously after prepare(); only the join side (count 0) is synchronous.
bool wait_for_num_inference_threads(unsigned int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (Core::get_num_inference_threads() != expected) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// An engine whose load fails the way an engine's load does for an unloadable model: inside
// create_session(), after the session id was handed out.
anira_status ANIRA_CALL refuse_load(const anira_engine_load_info* /*info*/,
                                    void* /*user_data*/,
                                    void** /*out_loaded*/) noexcept {
    return ANIRA_ERROR_MODEL_LOAD;
}

anira_status ANIRA_CALL never_runs(const anira_engine_ctx* /*ctx*/,
                                   void* /*prepared*/,
                                   void* /*user_data*/) noexcept {
    return ANIRA_ERROR_INTERNAL;
}

// The refusing engine object under the custom row's id; the pipelines that add it hold it.
struct RefusingEngine {
    RefusingEngine() {
        anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
        desc.load = refuse_load;
        desc.process = never_runs;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_custom_engine_create(anira_test::k_custom, &desc, &m_engine, &err),
                  ANIRA_OK)
            << err.message;
    }
    ~RefusingEngine() { anira_custom_engine_destroy(m_engine); }
    RefusingEngine(const RefusingEngine&) = delete;
    RefusingEngine& operator=(const RefusingEngine&) = delete;
    RefusingEngine(RefusingEngine&&) = delete;
    RefusingEngine& operator=(RefusingEngine&&) = delete;

    anira_custom_engine* m_engine = nullptr;
};

void expect_clean_slate() {
    EXPECT_EQ(Core::get_num_sessions(), 0) << "a failed session was left registered";
    EXPECT_EQ(Core::get_num_inference_threads(), 0u)
        << "inference threads survived a failed session creation";
    EXPECT_FALSE(Core::has_inference_threads())
        << "a thread pool was built for a session that never existed";
}

// A later, valid handler must get a fresh pool and tear it down again.
void expect_next_handler_lifecycle_intact() {
    {
        const anira_test::Context context(2);
        anira_test::Handler handler(context, anira_test::mono_stream_model());
        EXPECT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK)
            << handler.m_err.message;
        EXPECT_EQ(Core::get_num_sessions(), 1);
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    expect_clean_slate();
}

// One prepare over the refusing engine: its status, and the record names the refused load.
void expect_refused_prepare(const anira_test::Context& context) {
    const RefusingEngine engine;
    anira_test::Handler handler(context, anira_test::mono_stream_model(), {}, {}, engine.m_engine);
    EXPECT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_ERROR_MODEL_LOAD)
        << handler.m_err.message;
    EXPECT_NE(std::string(handler.m_err.message).find("refused load"), std::string::npos)
        << handler.m_err.message;
}

}  // namespace

// Several failures in a row must not accumulate anything either (the original
// report: "each failed load adds another increment").
TEST(CreateSessionFailureTest, RepeatedFailuresLeakNothing) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    for (int i = 0; i < 3; ++i) {
        {
            const anira_test::Context context(2);
            expect_refused_prepare(context);
        }
        expect_clean_slate();
    }
    expect_next_handler_lifecycle_intact();
}

// A failure while other sessions exist must leave those sessions' pool alone.
TEST(CreateSessionFailureTest, FailureBesideLiveSessionKeepsItsPool) {
    ASSERT_EQ(Core::get_num_sessions(), 0);

    const anira_test::Context context(2);
    anira_test::Handler live_handler(context, anira_test::mono_stream_model());
    ASSERT_EQ(live_handler.prepare(anira_test::explicit_contract()), ANIRA_OK)
        << live_handler.m_err.message;
    ASSERT_TRUE(wait_for_num_inference_threads(2));

    expect_refused_prepare(context);

    EXPECT_EQ(Core::get_num_sessions(), 1);
    EXPECT_TRUE(wait_for_num_inference_threads(2));
}

namespace {

// A model of one engine over a mono stream of 2048 samples.
ModelConfig engine_model(anira_engine engine, const std::filesystem::path& path) {
    ModelConfig model;
    model.add_model_path(engine, path);
    model.input(anira_test::stream_tensor("in", 1, 2048, 2048));
    model.output(anira_test::stream_tensor("out", 1, 2048, 2048));
    return model;
}

// The 2.x configuration's 42.66 ms budget, a block of one hop.
ContractHandle engine_contract() {
    return anira_test::explicit_contract(2048, anira_test::k_rate, ANIRA_MISS_BYPASS, 0.0, 42.66);
}

}  // namespace

#ifdef USE_ONNXRUNTIME
// The reproduction from the issue: a file ONNX Runtime cannot parse throws from the
// ONNX Runtime adapter's load, i.e. from inside create_session()'s plan setup — the
// same path as the refusing engine above, but through a real engine.
TEST(CreateSessionFailureTest, UnloadableOnnxModelLeaksNothing) {
    ASSERT_EQ(Core::get_num_sessions(), 0);

    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "anira_create_session_failure_not_a_model.onnx";
    {
        std::ofstream file(model_path, std::ios::binary);
        file << "definitely not a model";
    }

    {
        const anira_test::Context context(2);
        anira_test::Handler handler(context, engine_model(ANIRA_ENGINE_ONNXRUNTIME, model_path));
        EXPECT_NE(handler.prepare(engine_contract()), ANIRA_OK);
    }
    std::filesystem::remove(model_path);

    expect_clean_slate();
    expect_next_handler_lifecycle_intact();
}
#endif

// ============================================================================
// The error strategy: a model file that is not there is ANIRA_ERROR_NO_SUCH_FILE with the
// path and the engine name, on every backend, before any engine sees the path; a file the
// engine refuses is ANIRA_ERROR_MODEL_LOAD with the engine's own text.
// ============================================================================

namespace {

void expect_missing_file_is_no_such_file(anira_engine engine, const char* engine_name) {
    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "anira_create_session_failure_missing_model";
    std::filesystem::remove(model_path);
    {
        const anira_test::Context context(2);
        anira_test::Handler handler(context, engine_model(engine, model_path));
        EXPECT_EQ(handler.prepare(engine_contract()), ANIRA_ERROR_NO_SUCH_FILE)
            << handler.m_err.message;
        const std::string what = handler.m_err.message;
        EXPECT_NE(what.find(engine_name), std::string::npos) << what;
        EXPECT_NE(what.find(model_path.filename().string()), std::string::npos) << what;
        EXPECT_NE(what.find("no such file"), std::string::npos) << what;
    }
    expect_clean_slate();
}

}  // namespace

TEST(CreateSessionFailureTest, MissingModelFileIsNoSuchFileOnEveryBackend) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
#ifdef USE_ONNXRUNTIME
    expect_missing_file_is_no_such_file(ANIRA_ENGINE_ONNXRUNTIME, "onnxruntime");
#endif
#ifdef USE_LIBTORCH
    expect_missing_file_is_no_such_file(ANIRA_ENGINE_LIBTORCH, "libtorch");
#endif
#ifdef USE_TFLITE
    expect_missing_file_is_no_such_file(ANIRA_ENGINE_TFLITE, "tflite");
#endif
#ifdef USE_LITERT
    expect_missing_file_is_no_such_file(ANIRA_ENGINE_LITERT, "litert");
#endif
#ifdef USE_EXECUTORCH
    expect_missing_file_is_no_such_file(ANIRA_ENGINE_EXECUTORCH, "executorch");
#endif
    expect_next_handler_lifecycle_intact();
}

#ifdef USE_ONNXRUNTIME
TEST(CreateSessionFailureTest, UnloadableModelIsModelLoadWithTheEngineText) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "anira_create_session_failure_engine_text.onnx";
    {
        std::ofstream file(model_path, std::ios::binary);
        file << "definitely not a model";
    }
    {
        const anira_test::Context context(2);
        anira_test::Handler handler(context, engine_model(ANIRA_ENGINE_ONNXRUNTIME, model_path));
        EXPECT_EQ(handler.prepare(engine_contract()), ANIRA_ERROR_MODEL_LOAD)
            << handler.m_err.message;
        const std::string what = handler.m_err.message;
        EXPECT_EQ(what.rfind("onnxruntime: ", 0), 0) << "the engine name leads: " << what;
        EXPECT_NE(what.find(model_path.filename().string()), std::string::npos) << what;
    }
    std::filesystem::remove(model_path);
    expect_clean_slate();
    expect_next_handler_lifecycle_intact();
}
#endif
