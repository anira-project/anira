// Every rejection path of JsonConfigLoader. The happy paths live in
// test_JsonConfigLoader.cpp and need real model files; these cases are pure
// JSON -> config and run in any build, backends enabled or not.
//
// The contract under test throughout: a malformed value is reported and
// *skipped*, never fatal. A config that still has model data, a tensor shape
// and max_inference_time yields an InferenceConfig; anything less yields
// nullptr. The CoreConfig is always produced, falling back to defaults.

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/utils/JsonConfigLoader.h>

#include <array>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

// A model_data / tensor_shape / max_inference_time triple that parses cleanly,
// so a test can vary exactly one key and attribute the outcome to it.
constexpr const char* k_valid_model_data =
    R"([{ "model_path": "x", "inference_backend": "CUSTOM" }])";
constexpr const char* k_valid_tensor_shape =
    R"([{ "input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]] }])";

std::unique_ptr<anira::InferenceConfig> load_inference_config(const std::string& json) {
    std::istringstream stream(json);
    anira::JsonConfigLoader loader(stream);
    return loader.get_inference_config();
}

std::unique_ptr<anira::CoreConfig> load_context_config(const std::string& json) {
    std::istringstream stream(json);
    anira::JsonConfigLoader loader(stream);
    return loader.get_core_config();
}

// Wraps an "inference_config" body around the three keys, overriding whichever
// the caller passes.
std::string inference_json(const std::string& model_data = k_valid_model_data,
                           const std::string& tensor_shape = k_valid_tensor_shape,
                           const std::string& extra_keys = R"("max_inference_time": 5.0)") {
    return R"({"inference_config": { "model_data": )" + model_data + R"(, "tensor_shape": )" +
           tensor_shape + ", " + extra_keys + "}}";
}

// Every backend spelling the loader accepts, paired with whether this build
// compiled that backend in. Both arms of each #if are exercised across the
// coverage legs, which differ in enabled backends.
struct BackendCase {
    const char* m_name;
    bool m_compiled_in;
};

// The USE_* macros are defined exactly for the backends this build compiled in.
#ifdef USE_ONNXRUNTIME
constexpr bool k_onnx_compiled_in = true;
#else
constexpr bool k_onnx_compiled_in = false;
#endif
#ifdef USE_TFLITE
constexpr bool k_tflite_compiled_in = true;
#else
constexpr bool k_tflite_compiled_in = false;
#endif
#ifdef USE_LITERT
constexpr bool k_litert_compiled_in = true;
#else
constexpr bool k_litert_compiled_in = false;
#endif
#ifdef USE_EXECUTORCH
constexpr bool k_executorch_compiled_in = true;
#else
constexpr bool k_executorch_compiled_in = false;
#endif
#ifdef USE_LIBTORCH
constexpr bool k_libtorch_compiled_in = true;
#else
constexpr bool k_libtorch_compiled_in = false;
#endif

constexpr std::array<BackendCase, 6> k_backend_cases = {{
    {.m_name = "ONNX", .m_compiled_in = k_onnx_compiled_in},
    {.m_name = "TFLITE", .m_compiled_in = k_tflite_compiled_in},
    {.m_name = "LITERT", .m_compiled_in = k_litert_compiled_in},
    {.m_name = "EXECUTORCH", .m_compiled_in = k_executorch_compiled_in},
    {.m_name = "LIBTORCH", .m_compiled_in = k_libtorch_compiled_in},
    {.m_name = "CUSTOM", .m_compiled_in = true},
}};

std::string model_data_for(const char* backend) {
    return std::string(R"([{"model_path": "x", "inference_backend": ")") + backend + R"("}])";
}

}  // namespace

// ============================================================================
// Stream and document level
// ============================================================================

// A path that cannot be opened is reported, and the empty stream then fails to
// parse — so neither config is produced, exactly as for malformed JSON.
TEST(JsonConfigLoaderErrors, UnopenableFileYieldsNoConfigs) {
    anira::JsonConfigLoader loader("this/path/does/not/exist.json");
    EXPECT_EQ(loader.get_core_config(), nullptr);
    EXPECT_EQ(loader.get_inference_config(), nullptr);
}

TEST(JsonConfigLoaderErrors, MalformedJsonIsReportedNotThrown) {
    std::istringstream stream(R"({"inference_config": )");
    anira::JsonConfigLoader loader(stream);
    // parse() never ran, so neither config was built.
    EXPECT_EQ(loader.get_core_config(), nullptr);
    EXPECT_EQ(loader.get_inference_config(), nullptr);
}

TEST(JsonConfigLoaderErrors, MissingInferenceConfigKey) {
    const auto core_config = load_context_config(R"({"context_config": {"num_threads": 1}})");
    ASSERT_NE(core_config, nullptr);
    EXPECT_EQ(core_config->m_num_threads, 1U);
    EXPECT_EQ(load_inference_config(R"({"context_config": {"num_threads": 1}})"), nullptr);
}

TEST(JsonConfigLoaderErrors, MissingContextConfigKeyKeepsDefaults) {
    const auto config = load_context_config(inference_json());
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_num_threads, anira::default_num_threads());
    EXPECT_EQ(config->m_wait_strategy, anira::WaitStrategy::SpinBackoff);
    EXPECT_EQ(config->m_log.m_level, anira::default_log_level());
}

// ============================================================================
// core_config
// ============================================================================

TEST(JsonConfigLoaderErrors, NumThreadsWrongType) {
    const auto config = load_context_config(R"({"context_config": {"num_threads": "many"}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_num_threads, anira::default_num_threads());
}

TEST(JsonConfigLoaderErrors, NumThreadsNegativeIsNotUnsigned) {
    const auto config = load_context_config(R"({"context_config": {"num_threads": -2}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_num_threads, anira::default_num_threads());
}

TEST(JsonConfigLoaderErrors, WaitStrategySpinBackoffAndBlocking) {
    const auto spin =
        load_context_config(R"({"context_config": {"wait_strategy": "spin_backoff"}})");
    ASSERT_NE(spin, nullptr);
    EXPECT_EQ(spin->m_wait_strategy, anira::WaitStrategy::SpinBackoff);

    const auto blocking =
        load_context_config(R"({"context_config": {"wait_strategy": "blocking"}})");
    ASSERT_NE(blocking, nullptr);
#ifdef __EMSCRIPTEN__
    EXPECT_EQ(blocking->m_wait_strategy, anira::WaitStrategy::SpinBackoff);
#else
    EXPECT_EQ(blocking->m_wait_strategy, anira::WaitStrategy::Blocking);
#endif
}

TEST(JsonConfigLoaderErrors, WaitStrategyUnknownStringFallsBackToSpinBackoff) {
    const auto config = load_context_config(R"({"context_config": {"wait_strategy": "yield"}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_wait_strategy, anira::WaitStrategy::SpinBackoff);
}

TEST(JsonConfigLoaderErrors, WaitStrategyWrongTypeFallsBackToSpinBackoff) {
    const auto config = load_context_config(R"({"context_config": {"wait_strategy": 7}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_wait_strategy, anira::WaitStrategy::SpinBackoff);
}

TEST(JsonConfigLoaderErrors, AllLogLevelSpellings) {
    struct LevelCase {
        const char* m_text;
        anira::LogLevel m_expected;
    };
    constexpr std::array<LevelCase, 4> k_cases = {{
        {.m_text = "debug", .m_expected = anira::LogLevel::Debug},
        {.m_text = "info", .m_expected = anira::LogLevel::Info},
        {.m_text = "warning", .m_expected = anira::LogLevel::Warning},
        {.m_text = "error", .m_expected = anira::LogLevel::Error},
    }};
    for (const auto& test_case : k_cases) {
        const auto config = load_context_config(std::string(R"({"context_config": {"log": )") +
                                                R"({"level": ")" + test_case.m_text + R"("}}})");
        ASSERT_NE(config, nullptr) << test_case.m_text;
        EXPECT_EQ(config->m_log.m_level, test_case.m_expected) << test_case.m_text;
    }
}

TEST(JsonConfigLoaderErrors, LogLevelUnknownStringKeepsDefault) {
    const auto config = load_context_config(R"({"context_config": {"log": {"level": "trace"}}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_level, anira::default_log_level());
}

TEST(JsonConfigLoaderErrors, LogLevelWrongTypeKeepsDefault) {
    const auto config = load_context_config(R"({"context_config": {"log": {"level": 0}}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_level, anira::default_log_level());
}

TEST(JsonConfigLoaderErrors, LegacyLogLevelKeyWrongTypeKeepsDefault) {
    const auto config = load_context_config(R"({"context_config": {"log_level": []}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_level, anira::default_log_level());
}

TEST(JsonConfigLoaderErrors, LogBlockNotAnObjectAbortsTheWholeLogBlock) {
    const auto config = load_context_config(R"({"context_config": {"log": "debug"}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_level, anira::default_log_level());
    EXPECT_EQ(config->m_log.m_drain, anira::default_log_drain());
}

TEST(JsonConfigLoaderErrors, LogDrainThreadAndManual) {
    const auto thread = load_context_config(R"({"context_config": {"log": {"drain": "thread"}}})");
    ASSERT_NE(thread, nullptr);
#ifdef __EMSCRIPTEN__
    EXPECT_EQ(thread->m_log.m_drain, anira::LogDrain::Manual);
#else
    EXPECT_EQ(thread->m_log.m_drain, anira::LogDrain::Thread);
#endif

    const auto manual = load_context_config(R"({"context_config": {"log": {"drain": "manual"}}})");
    ASSERT_NE(manual, nullptr);
    EXPECT_EQ(manual->m_log.m_drain, anira::LogDrain::Manual);
}

TEST(JsonConfigLoaderErrors, LogDrainUnknownAndWrongTypeKeepDefault) {
    for (const char* body : {R"({"drain": "poll"})", R"({"drain": 1})"}) {
        const auto config =
            load_context_config(std::string(R"({"context_config": {"log": )") + body + "}}");
        ASSERT_NE(config, nullptr) << body;
        EXPECT_EQ(config->m_log.m_drain, anira::default_log_drain()) << body;
    }
}

TEST(JsonConfigLoaderErrors, LogQueueCapacityAndDrainIntervalWrongTypeKeepDefaults) {
    const auto config = load_context_config(
        R"({"context_config": {"log": {"queue_capacity": -1, "drain_interval_ms": "fast"}}})");
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_log.m_queue_capacity, 512U);
    EXPECT_EQ(config->m_log.m_drain_interval_ms, 10U);
}

// ============================================================================
// model_data
// ============================================================================

TEST(JsonConfigLoaderErrors, ModelDataNotAnArray) {
    EXPECT_EQ(load_inference_config(inference_json(R"({"model_path": "x"})")), nullptr);
}

TEST(JsonConfigLoaderErrors, ModelDataEmptyArray) {
    EXPECT_EQ(load_inference_config(inference_json("[]")), nullptr);
}

TEST(JsonConfigLoaderErrors, ModelDataEntryMissingRequiredKeys) {
    EXPECT_EQ(load_inference_config(inference_json(R"([{"model_path": "x"}])")), nullptr);
    EXPECT_EQ(load_inference_config(inference_json(R"([{"inference_backend": "CUSTOM"}])")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, ModelDataEntryWrongValueTypes) {
    EXPECT_EQ(load_inference_config(
                  inference_json(R"([{"model_path": 3, "inference_backend": "CUSTOM"}])")),
              nullptr);
    EXPECT_EQ(
        load_inference_config(inference_json(R"([{"model_path": "x", "inference_backend": 3}])")),
        nullptr);
}

TEST(JsonConfigLoaderErrors, ModelDataUnknownBackendName) {
    EXPECT_EQ(load_inference_config(
                  inference_json(R"([{"model_path": "x", "inference_backend": "COREML"}])")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, EveryModelDataBackendSpelling) {
    for (const auto& backend : k_backend_cases) {
        const auto config = load_inference_config(inference_json(model_data_for(backend.m_name)));
        if (backend.m_compiled_in) {
            ASSERT_NE(config, nullptr) << backend.m_name;
            EXPECT_EQ(config->m_model_data.size(), 1U) << backend.m_name;
        } else {
            EXPECT_EQ(config, nullptr) << backend.m_name;
        }
    }
}

// model_function is only read for the two backends that support it, and only
// as a string. A non-string value drops the entry either way (the check is
// inside the enabled arm; the disabled arm drops it for being disabled).
TEST(JsonConfigLoaderErrors, ModelFunctionOnLibTorch) {
    const auto config = load_inference_config(inference_json(
        R"([{"model_path": "x", "inference_backend": "LIBTORCH", "model_function": "forward"}])"));
#if USE_LIBTORCH
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_model_data[0].m_model_function, "forward");
#else
    EXPECT_EQ(config, nullptr);
#endif
    EXPECT_EQ(load_inference_config(inference_json(
                  R"([{"model_path": "x", "inference_backend": "LIBTORCH",
                       "model_function": 42}])")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, ModelFunctionOnExecuTorch) {
    const auto config = load_inference_config(inference_json(
        R"([{"model_path": "x", "inference_backend": "EXECUTORCH",
             "model_function": "forward"}])"));
#if USE_EXECUTORCH
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_model_data[0].m_model_function, "forward");
#else
    EXPECT_EQ(config, nullptr);
#endif
    EXPECT_EQ(load_inference_config(inference_json(
                  R"([{"model_path": "x", "inference_backend": "EXECUTORCH",
                       "model_function": 42}])")),
              nullptr);
}

// ============================================================================
// tensor_shape
// ============================================================================

TEST(JsonConfigLoaderErrors, TensorShapeNotAnArrayOrEmpty) {
    EXPECT_EQ(load_inference_config(inference_json(k_valid_model_data, R"({"input_shape": []})")),
              nullptr);
    EXPECT_EQ(load_inference_config(inference_json(k_valid_model_data, "[]")), nullptr);
}

TEST(JsonConfigLoaderErrors, TensorShapeEntryMissingShapeKeys) {
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data, R"([{"input_shape": [[1, 512]]}])")),
              nullptr);
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data, R"([{"output_shape": [[1, 512]]}])")),
              nullptr);
}

// A flat [1, 1, 512] is accepted as shorthand for a single tensor and must
// produce the same config as the nested [[1, 1, 512]].
TEST(JsonConfigLoaderErrors, FlatShapeShorthandMatchesNestedForm) {
    const auto flat = load_inference_config(
        inference_json(k_valid_model_data,
                       R"([{"input_shape": [1, 1, 512], "output_shape": [1, 1, 512]}])"));
    const auto nested = load_inference_config(inference_json());
    ASSERT_NE(flat, nullptr);
    ASSERT_NE(nested, nullptr);
    EXPECT_EQ(*flat, *nested);
}

TEST(JsonConfigLoaderErrors, ShapeNotAnArrayYieldsNoTensorShape) {
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data,
                                 R"([{"input_shape": 512, "output_shape": [[1, 1, 512]]}])")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, ShapeEmptyArrayYieldsNoTensorShape) {
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data,
                                 R"([{"input_shape": [], "output_shape": [[1, 1, 512]]}])")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, ShapeOfStringsIsRejected) {
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data,
                                 R"([{"input_shape": ["a"], "output_shape": [[1, 1, 512]]}])")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, TensorShapeBackendWrongTypeFallsBackToUniversal) {
    const auto config = load_inference_config(
        inference_json(k_valid_model_data,
                       R"([{"input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]],
             "inference_backend": 5}])"));
    ASSERT_NE(config, nullptr);
    EXPECT_TRUE(config->m_tensor_shape[0].is_universal());
}

TEST(JsonConfigLoaderErrors, TensorShapeExplicitUniversalBackend) {
    const auto config = load_inference_config(
        inference_json(k_valid_model_data,
                       R"([{"input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]],
             "inference_backend": "UNIVERSAL"}])"));
    ASSERT_NE(config, nullptr);
    EXPECT_TRUE(config->m_tensor_shape[0].is_universal());
}

TEST(JsonConfigLoaderErrors, TensorShapeUnknownBackendNameDropsTheEntry) {
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data,
                                 R"([{"input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]],
                       "inference_backend": "COREML"}])")),
              nullptr);
}

// The same table on the tensor_shape side. Each entry is paired with a matching
// model_data entry so the config is constructible.
TEST(JsonConfigLoaderErrors, EveryTensorShapeBackendSpelling) {
    for (const auto& backend : k_backend_cases) {
        const std::string tensor_shape =
            std::string(R"([{"input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]],
                             "inference_backend": ")") +
            backend.m_name + R"("}])";
        const auto config =
            load_inference_config(inference_json(model_data_for(backend.m_name), tensor_shape));
        if (backend.m_compiled_in) {
            ASSERT_NE(config, nullptr) << backend.m_name;
            EXPECT_FALSE(config->m_tensor_shape[0].is_universal()) << backend.m_name;
        } else {
            EXPECT_EQ(config, nullptr) << backend.m_name;
        }
    }
}

// ============================================================================
// processing_spec
// ============================================================================

TEST(JsonConfigLoaderErrors, ProcessingSpecAllKeysAreRead) {
    const auto config = load_inference_config(inference_json(k_valid_model_data,
                                                             k_valid_tensor_shape,
                                                             R"("max_inference_time": 5.0,
            "processing_spec": {
                "preprocess_input_channels": [2],
                "postprocess_output_channels": [2],
                "preprocess_input_size": [256],
                "postprocess_output_size": [256],
                "internal_model_latency": [8]
            })"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->get_preprocess_input_channels(), std::vector<size_t>{2});
    EXPECT_EQ(config->get_postprocess_output_channels(), std::vector<size_t>{2});
    EXPECT_EQ(config->get_preprocess_input_size(), std::vector<size_t>{256});
    EXPECT_EQ(config->get_postprocess_output_size(), std::vector<size_t>{256});
    EXPECT_EQ(config->get_internal_model_latency(), std::vector<size_t>{8});
}

// A rejected processing_spec value degrades to the empty vector, which
// update_processing_spec() then fills with the derived defaults — so the config
// is still built, just without the requested override.
TEST(JsonConfigLoaderErrors, ProcessingSpecRejectedValuesFallBackToDerivedDefaults) {
    for (const char* bad : {R"("preprocess_input_channels": 2)",
                            R"("preprocess_input_channels": [])",
                            R"("preprocess_input_channels": ["two"])",
                            R"("preprocess_input_channels": [-2])"}) {
        const auto config = load_inference_config(inference_json(
            k_valid_model_data,
            k_valid_tensor_shape,
            std::string(R"("max_inference_time": 5.0, "processing_spec": {)") + bad + "}"));
        ASSERT_NE(config, nullptr) << bad;
        EXPECT_EQ(config->get_preprocess_input_channels(), std::vector<size_t>{1}) << bad;
    }
}

// ============================================================================
// single parameters
// ============================================================================

TEST(JsonConfigLoaderErrors, MissingMaxInferenceTime) {
    EXPECT_EQ(load_inference_config(
                  inference_json(k_valid_model_data, k_valid_tensor_shape, R"("warm_up": 1)")),
              nullptr);
}

// max_inference_time must be a float: an integer literal is not
// is_number_float() and is rejected, which leaves the config unbuildable.
TEST(JsonConfigLoaderErrors, MaxInferenceTimeMustBeFloat) {
    EXPECT_EQ(
        load_inference_config(
            inference_json(k_valid_model_data, k_valid_tensor_shape, R"("max_inference_time": 5)")),
        nullptr);
    EXPECT_EQ(load_inference_config(inference_json(k_valid_model_data,
                                                   k_valid_tensor_shape,
                                                   R"("max_inference_time": "5.0")")),
              nullptr);
}

TEST(JsonConfigLoaderErrors, OptionalSingleParametersAreRead) {
    const auto config =
        load_inference_config(inference_json(k_valid_model_data,
                                             k_valid_tensor_shape,
                                             R"("max_inference_time": 5.0, "warm_up": 3,
                           "session_exclusive_processor": false, "blocking_ratio": 0.25,
                           "num_parallel_processors": 4)"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_warm_up, 3U);
    EXPECT_FALSE(config->m_session_exclusive_processor);
    EXPECT_FLOAT_EQ(config->m_blocking_ratio, 0.25F);
    EXPECT_EQ(config->m_num_parallel_processors, 4U);
}

TEST(JsonConfigLoaderErrors, SessionExclusiveProcessorOverridesParallelProcessors) {
    const auto config = load_inference_config(inference_json(k_valid_model_data,
                                                             k_valid_tensor_shape,
                                                             R"("max_inference_time": 5.0,
                           "session_exclusive_processor": true, "num_parallel_processors": 4)"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_num_parallel_processors, 1U);
}

TEST(JsonConfigLoaderErrors, OptionalSingleParametersWrongTypeKeepDefaults) {
    const auto reference = load_inference_config(inference_json());
    ASSERT_NE(reference, nullptr);

    const auto config =
        load_inference_config(inference_json(k_valid_model_data,
                                             k_valid_tensor_shape,
                                             R"("max_inference_time": 5.0, "warm_up": "three",
                           "session_exclusive_processor": 1, "blocking_ratio": "quarter",
                           "num_parallel_processors": -4)"));
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->m_warm_up, reference->m_warm_up);
    EXPECT_EQ(config->m_session_exclusive_processor, reference->m_session_exclusive_processor);
    EXPECT_FLOAT_EQ(config->m_blocking_ratio, reference->m_blocking_ratio);
    EXPECT_EQ(config->m_num_parallel_processors, reference->m_num_parallel_processors);
    EXPECT_EQ(*config, *reference);
}
