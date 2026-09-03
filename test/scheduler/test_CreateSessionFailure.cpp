#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Context.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

// Regression tests for issue #106: Context::create_session() used to count the
// session before doing the work that can throw (loading the model in a backend
// processor's constructor, a custom processor's prepare()) and registered it only
// afterwards. A throw therefore left the active-session counter permanently
// incremented with nothing registered, and since the thread-pool teardown was gated
// on that counter reaching zero, the inference threads (and the context) lived on for
// the rest of the process. Registration is now the last step of create_session(),
// and everything before it is rolled back on failure.

namespace {

// The pool's threads count themselves as active when they enter their loop, i.e.
// asynchronously after prepare(); only the join side (count 0) is synchronous.
bool wait_for_num_inference_threads(unsigned int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (Context::get_num_inference_threads() != expected) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

InferenceConfig make_custom_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        1.f,
        0,
        false,
        0.f,
        2);
}

// A backend whose prepare() fails the way a backend constructor does for an
// unloadable model: inside create_session(), after the session id was handed out.
struct ThrowingProcessor : public BackendBase {
    using BackendBase::BackendBase;
    void prepare() override { throw std::runtime_error("test backend: cannot load model"); }
};

void expect_clean_slate() {
    EXPECT_EQ(Context::get_num_sessions(), 0) << "a failed session was left registered";
    EXPECT_EQ(Context::get_num_inference_threads(), 0u)
        << "inference threads survived a failed session creation";
    EXPECT_FALSE(Context::has_inference_threads())
        << "a thread pool was built for a session that never existed";
}

// A later, valid handler must get a fresh pool and tear it down again.
void expect_next_handler_lifecycle_intact() {
    {
        InferenceConfig inference_config = make_custom_config();
        PrePostProcessor pp_processor(inference_config);
        InferenceHandler handler(pp_processor, inference_config, ContextConfig(2));
        handler.prepare(HostConfig(512, 48000));
        EXPECT_EQ(Context::get_num_sessions(), 1);
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    expect_clean_slate();
}

}  // namespace

// Several failures in a row must not accumulate anything either (the original
// report: "each failed load adds another increment").
TEST(CreateSessionFailureTest, RepeatedFailuresLeakNothing) {
    ASSERT_EQ(Context::get_num_sessions(), 0);

    InferenceConfig inference_config = make_custom_config();
    PrePostProcessor pp_processor(inference_config);
    ThrowingProcessor throwing_processor(inference_config);

    for (int i = 0; i < 3; ++i) {
        // The first iteration also pins the exception type (the standalone
        // single-failure test it absorbed — audit, docs/ci-overhaul.md step 9a).
        if (i == 0) {
            EXPECT_THROW(
                {
                    const InferenceHandler handler(pp_processor,
                                                   inference_config,
                                                   throwing_processor,
                                                   ContextConfig(2));
                },
                std::runtime_error);
        } else {
            EXPECT_ANY_THROW({
                const InferenceHandler handler(pp_processor,
                                               inference_config,
                                               throwing_processor,
                                               ContextConfig(2));
            });
        }
        expect_clean_slate();
    }
    expect_next_handler_lifecycle_intact();
}

// A failure while other sessions exist must leave those sessions' pool alone.
TEST(CreateSessionFailureTest, FailureBesideLiveSessionKeepsItsPool) {
    ASSERT_EQ(Context::get_num_sessions(), 0);

    InferenceConfig live_config = make_custom_config();
    PrePostProcessor live_pp(live_config);
    InferenceHandler live_handler(live_pp, live_config, ContextConfig(2));
    live_handler.prepare(HostConfig(512, 48000));
    ASSERT_TRUE(wait_for_num_inference_threads(2));

    InferenceConfig inference_config = make_custom_config();
    PrePostProcessor pp_processor(inference_config);
    ThrowingProcessor throwing_processor(inference_config);
    EXPECT_ANY_THROW({
        const InferenceHandler handler(pp_processor,
                                       inference_config,
                                       throwing_processor,
                                       ContextConfig(2));
    });

    EXPECT_EQ(Context::get_num_sessions(), 1);
    EXPECT_TRUE(wait_for_num_inference_threads(2));
}

#ifdef USE_ONNXRUNTIME
// The reproduction from the issue: a file ONNX Runtime cannot parse throws from the
// OnnxRuntimeProcessor constructor, i.e. from inside create_session()'s processor
// setup — the same path as the custom processor above, but through a real backend.
TEST(CreateSessionFailureTest, UnloadableOnnxModelLeaksNothing) {
    ASSERT_EQ(Context::get_num_sessions(), 0);

    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "anira_create_session_failure_not_a_model.onnx";
    {
        std::ofstream file(model_path, std::ios::binary);
        file << "definitely not a model";
    }

    {
        InferenceConfig inference_config(
            std::vector<ModelData>{ModelData(model_path.string(), InferenceBackend::ONNX)},
            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
            42.66f);
        PrePostProcessor pp_processor(inference_config);

        EXPECT_ANY_THROW(
            { const InferenceHandler handler(pp_processor, inference_config, ContextConfig(2)); });
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

#include <anira/abi/status.h>

#include "../../src/utils/StatusError.h"

namespace {

template <class Body>
anira::StatusError status_error_of(Body&& body) {
    try {
        body();
    } catch (const anira::StatusError& e) { return e; }
    ADD_FAILURE() << "expected an anira::StatusError";
    return {ANIRA_OK, ""};
}

void expect_missing_file_is_no_such_file(InferenceBackend backend, const char* engine) {
    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "anira_create_session_failure_missing_model";
    std::filesystem::remove(model_path);
    InferenceConfig inference_config(
        std::vector<ModelData>{ModelData(model_path.string(), backend)},
        std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
        42.66f);
    PrePostProcessor pp_processor(inference_config);
    const anira::StatusError error = status_error_of(
        [&] { const InferenceHandler handler(pp_processor, inference_config, ContextConfig(2)); });
    EXPECT_EQ(error.status(), ANIRA_ERROR_NO_SUCH_FILE) << error.what();
    const std::string what = error.what();
    EXPECT_NE(what.find(engine), std::string::npos) << what;
    EXPECT_NE(what.find(model_path.filename().string()), std::string::npos) << what;
    EXPECT_NE(what.find("no such file"), std::string::npos) << what;
    expect_clean_slate();
}

}  // namespace

TEST(CreateSessionFailureTest, MissingModelFileIsNoSuchFileOnEveryBackend) {
    ASSERT_EQ(Context::get_num_sessions(), 0);
#ifdef USE_ONNXRUNTIME
    expect_missing_file_is_no_such_file(InferenceBackend::ONNX, "onnxruntime");
#endif
#ifdef USE_LIBTORCH
    expect_missing_file_is_no_such_file(InferenceBackend::LIBTORCH, "libtorch");
#endif
#ifdef USE_TFLITE
    expect_missing_file_is_no_such_file(InferenceBackend::TFLITE, "tflite");
#endif
#ifdef USE_LITERT
    expect_missing_file_is_no_such_file(InferenceBackend::LITERT, "litert");
#endif
#ifdef USE_EXECUTORCH
    expect_missing_file_is_no_such_file(InferenceBackend::EXECUTORCH, "executorch");
#endif
    expect_next_handler_lifecycle_intact();
}

#ifdef USE_ONNXRUNTIME
TEST(CreateSessionFailureTest, UnloadableModelIsModelLoadWithTheEngineText) {
    ASSERT_EQ(Context::get_num_sessions(), 0);
    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "anira_create_session_failure_engine_text.onnx";
    {
        std::ofstream file(model_path, std::ios::binary);
        file << "definitely not a model";
    }
    {
        InferenceConfig inference_config(
            std::vector<ModelData>{ModelData(model_path.string(), InferenceBackend::ONNX)},
            std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
            42.66f);
        PrePostProcessor pp_processor(inference_config);
        const anira::StatusError error = status_error_of([&] {
            const InferenceHandler handler(pp_processor, inference_config, ContextConfig(2));
        });
        EXPECT_EQ(error.status(), ANIRA_ERROR_MODEL_LOAD) << error.what();
        const std::string what = error.what();
        EXPECT_EQ(what.rfind("onnxruntime: ", 0), 0) << "the engine name leads: " << what;
        EXPECT_NE(what.find(model_path.filename().string()), std::string::npos) << what;
    }
    std::filesystem::remove(model_path);
    expect_clean_slate();
    expect_next_handler_lifecycle_intact();
}
#endif
