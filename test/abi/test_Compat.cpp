// anira/compat/v2.hpp against the live 2.x classes, which stay public until the cut-over: the
// same arguments on both sides, every getter compared. Suite AbiCompatConfig is the
// configuration half (the enum and its conversions, ModelData, TensorShape, ProcessingSpec,
// InferenceConfig, JsonConfigLoader, ContextConfig): the upgrade of what the constructors
// write, the getters of the live class on the same arguments, the normalisation table row by
// row, the loader over the version-2 documents of test/support/v2_documents.h. Suite AbiCompat
// is the runtime half: the PassthroughEngine against the 2.x BackendBase and the
// LegacyProcessorStage on C handlers built through anira::Pipeline, against the live 2.x
// handler where both run. A 2.x value and a shim value are compared by name through
// same_backend, never by a cast: the 2.x enumerators depend on the engines of the build, the
// shim's do not.
#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/JsonConfigLoader.h>
#include <anira/utils/Logger.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <anira/compat/v2.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../support/log_record_collector.h"
#include "../support/v2_documents.h"
#include "fixtures.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

namespace v2 = anira::v2;

/// A value outside the enum, as a cast from an int of a stale 2.x build would give.
// NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) the out-of-range value is the point
const auto k_no_backend = static_cast<v2::InferenceBackend>(17);

// ---- the two backend enums -----------------------------------------------------------------

/// The 2.x enumerator of a shim backend, where this build has one.
std::optional<anira::InferenceBackend> old_backend(v2::InferenceBackend backend) {
    switch (backend) {
#ifdef USE_LIBTORCH
        case v2::LIBTORCH: return anira::InferenceBackend::LIBTORCH;
#endif
#ifdef USE_ONNXRUNTIME
        case v2::ONNX: return anira::InferenceBackend::ONNX;
#endif
#ifdef USE_TFLITE
        case v2::TFLITE: return anira::InferenceBackend::TFLITE;
#endif
#ifdef USE_LITERT
        case v2::LITERT: return anira::InferenceBackend::LITERT;
#endif
#ifdef USE_EXECUTORCH
        case v2::EXECUTORCH: return anira::InferenceBackend::EXECUTORCH;
#endif
        case v2::CUSTOM: return anira::InferenceBackend::CUSTOM;
        default: return std::nullopt;
    }
}

/// The 2.x enumerator of a backend the build has (the argument lists hold no other).
anira::InferenceBackend required_old_backend(v2::InferenceBackend backend) {
    const std::optional<anira::InferenceBackend> mapped = old_backend(backend);
    if (!mapped.has_value()) { throw std::logic_error("a backend this build does not have"); }
    return *mapped;
}

/// Whether a 2.x value and a shim value name one backend.
bool same_backend(anira::InferenceBackend old, v2::InferenceBackend shim) {
    const std::optional<anira::InferenceBackend> mapped = old_backend(shim);
    return mapped.has_value() && *mapped == old;
}

constexpr std::array<v2::InferenceBackend, 6> k_backends{v2::LIBTORCH,
                                                         v2::ONNX,
                                                         v2::TFLITE,
                                                         v2::LITERT,
                                                         v2::EXECUTORCH,
                                                         v2::CUSTOM};

// ---- the arguments, spelled once for both sides ----------------------------------------------

/// One model entry: a path (or bytes) for a backend, with an entry point.
struct Row {
    Row(v2::InferenceBackend backend, std::string path, std::string function = "")
        : m_backend(backend), m_path(std::move(path)), m_function(std::move(function)) {}

    v2::InferenceBackend m_backend;
    std::string m_path;
    std::string m_function;
};

/// One tensor-shape row: universal, or of a backend.
struct Shapes {
    Shapes(v2::TensorShapeList inputs,
           v2::TensorShapeList outputs,
           std::optional<v2::InferenceBackend> backend = std::nullopt)
        : m_inputs(std::move(inputs)), m_outputs(std::move(outputs)), m_backend(backend) {}

    v2::TensorShapeList m_inputs;
    v2::TensorShapeList m_outputs;
    std::optional<v2::InferenceBackend> m_backend;
};

/// Everything the two constructors take.
struct Arguments {
    std::vector<Row> m_rows;
    std::vector<Shapes> m_shapes;
    std::vector<size_t> m_in_channels;
    std::vector<size_t> m_out_channels;
    std::vector<size_t> m_in_sizes;
    std::vector<size_t> m_out_sizes;
    std::vector<size_t> m_latencies;
    float m_max_inference_time = 5.F;
    unsigned m_warm_up = 0;
    bool m_exclusive = false;
    float m_blocking_ratio = 0.F;
    unsigned m_processors = 2;
};

/// The rows of this build only: the 2.x side cannot name an engine the build lacks.
std::vector<Row> rows_of_build(const std::vector<Row>& rows) {
    std::vector<Row> kept;
    for (const Row& row : rows) {
        if (old_backend(row.m_backend).has_value()) { kept.push_back(row); }
    }
    return kept;
}

/// A shape row survives on the 2.x side only for a backend the build has.
std::vector<Shapes> shapes_of_build(const std::vector<Shapes>& shapes) {
    std::vector<Shapes> kept;
    for (const Shapes& row : shapes) {
        if (!row.m_backend.has_value() || old_backend(*row.m_backend).has_value()) {
            kept.push_back(row);
        }
    }
    return kept;
}

anira::InferenceConfig make_old(const Arguments& args) {
    std::vector<anira::ModelData> rows;
    rows.reserve(args.m_rows.size());
    for (const Row& row : args.m_rows) {
        rows.emplace_back(row.m_path, required_old_backend(row.m_backend), row.m_function);
    }
    std::vector<anira::TensorShape> shapes;
    shapes.reserve(args.m_shapes.size());
    for (const Shapes& row : args.m_shapes) {
        if (row.m_backend.has_value()) {
            shapes.emplace_back(row.m_inputs, row.m_outputs, required_old_backend(*row.m_backend));
        } else {
            shapes.emplace_back(row.m_inputs, row.m_outputs);
        }
    }
    return {rows,
            shapes,
            anira::ProcessingSpec(args.m_in_channels,
                                  args.m_out_channels,
                                  args.m_in_sizes,
                                  args.m_out_sizes,
                                  args.m_latencies),
            args.m_max_inference_time,
            args.m_warm_up,
            args.m_exclusive,
            args.m_blocking_ratio,
            args.m_processors};
}

v2::InferenceConfig make_shim(const Arguments& args) {
    std::vector<v2::ModelData> rows;
    rows.reserve(args.m_rows.size());
    for (const Row& row : args.m_rows) {
        rows.emplace_back(row.m_path, row.m_backend, row.m_function);
    }
    std::vector<v2::TensorShape> shapes;
    shapes.reserve(args.m_shapes.size());
    for (const Shapes& row : args.m_shapes) {
        if (row.m_backend.has_value()) {
            shapes.emplace_back(row.m_inputs, row.m_outputs, row.m_backend.value());
        } else {
            shapes.emplace_back(row.m_inputs, row.m_outputs);
        }
    }
    return {rows,
            shapes,
            v2::ProcessingSpec(args.m_in_channels,
                               args.m_out_channels,
                               args.m_in_sizes,
                               args.m_out_sizes,
                               args.m_latencies),
            args.m_max_inference_time,
            args.m_warm_up,
            args.m_exclusive,
            args.m_blocking_ratio,
            args.m_processors};
}

std::string model_path(const char* relative) {
    return std::string(ANIRA_EXTRAS_MODELS_DIR) + "/" + relative;
}

/// SimpleGainNetwork, mono: a 512-sample stream and a static gain, both ways; the TensorFlow
/// exports hold the gain at rank 3.
Arguments gain_arguments() {
    const std::string stem =
        model_path("model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono");
    Arguments args;
    args.m_rows = rows_of_build({{v2::LIBTORCH, stem + ".pt", ""},
                                 {v2::ONNX, stem + ".onnx", ""},
                                 {v2::TFLITE, stem + ".tflite", ""},
                                 {v2::LITERT, stem + ".tflite", ""},
                                 {v2::EXECUTORCH, stem + ".pte", ""},
                                 {v2::CUSTOM, "placeholder", ""}});
    args.m_shapes = shapes_of_build({{{{1, 1, 512}, {1, 1, 1}}, {{1, 1, 512}, {1}}, v2::TFLITE},
                                     {{{1, 1, 512}, {1, 1, 1}}, {{1, 1, 512}, {1}}, v2::LITERT},
                                     {{{1, 1, 512}, {1}}, {{1, 1, 512}, {1}}, std::nullopt}});
    args.m_in_channels = {1, 1};
    args.m_out_channels = {1, 1};
    args.m_in_sizes = {512, 0};
    args.m_out_sizes = {512, 0};
    args.m_warm_up = 1;
    return args;
}

/// GuitarLSTM: 256 windows of 150 samples in, 256 samples out; the TensorFlow exports hold the
/// input channels last.
Arguments hybridnn_arguments() {
    const std::string dir = model_path("hybrid-nn/GuitarLSTM/");
    Arguments args;
    args.m_rows = rows_of_build(
        {{v2::LIBTORCH, dir + "pytorch-version/models/model_0/GuitarLSTM-dynamic.pt", ""},
         {v2::ONNX, dir + "pytorch-version/models/model_0/GuitarLSTM-libtorch-dynamic.onnx", ""},
         {v2::TFLITE, dir + "tensorflow-version/models/model_0/GuitarLSTM-256.tflite", ""},
         {v2::LITERT, dir + "tensorflow-version/models/model_0/GuitarLSTM-256.tflite", ""}});
    args.m_shapes = shapes_of_build({{{{256, 1, 150}}, {{256, 1}}, std::nullopt},
                                     {{{256, 150, 1}}, {{256, 1}}, v2::TFLITE},
                                     {{{256, 150, 1}}, {{256, 1}}, v2::LITERT}});
    args.m_in_channels = {1};
    args.m_out_channels = {1};
    args.m_in_sizes = {256};
    args.m_out_sizes = {256};
    return args;
}

/// The steerable-nafx CNN: a 15380-sample window of which 2048 are new, 2048 out; the
/// TensorFlow exports channels last on both sides.
Arguments cnn_arguments() {
    const std::string dir = model_path("cnn/steerable-nafx/models/model_0/steerable-nafx");
    Arguments args;
    args.m_rows = rows_of_build({{v2::LIBTORCH, dir + "-dynamic.pt", ""},
                                 {v2::ONNX, dir + "-libtorch-dynamic.onnx", ""},
                                 {v2::TFLITE, dir + "-dynamic.tflite", ""},
                                 {v2::LITERT, dir + "-dynamic.tflite", ""}});
    args.m_shapes = shapes_of_build({{{{1, 1, 15380}}, {{1, 1, 2048}}, std::nullopt},
                                     {{{1, 15380, 1}}, {{1, 2048, 1}}, v2::TFLITE},
                                     {{{1, 15380, 1}}, {{1, 2048, 1}}, v2::LITERT}});
    args.m_in_channels = {1};
    args.m_out_channels = {1};
    args.m_in_sizes = {2048};
    args.m_out_sizes = {2048};
    args.m_max_inference_time = 10.F;
    return args;
}

/// RAVE, the whole model: stateful, 2048 samples of latency, a fractional budget.
Arguments rave_arguments() {
    Arguments args;
    args.m_rows = rows_of_build(
        {{v2::LIBTORCH, model_path("third-party/ircam-acids/RAVE/rave_funk_drum.ts"), ""},
         {v2::CUSTOM, "placeholder", ""}});
    args.m_shapes = {{{{1, 1, 2048}}, {{1, 1, 2048}}, std::nullopt}};
    args.m_in_channels = {1};
    args.m_out_channels = {1};
    args.m_in_sizes = {2048};
    args.m_out_sizes = {2048};
    args.m_latencies = {2048};
    args.m_max_inference_time = 42.66F;
    args.m_warm_up = 5;
    args.m_exclusive = true;
    args.m_blocking_ratio = 0.5F;
    args.m_processors = 4;
    return args;
}

/// A stereo stream with a Static output and the default processing sizes.
Arguments stereo_arguments() {
    Arguments args;
    args.m_rows = {{v2::CUSTOM, "placeholder", ""}};
    args.m_shapes = {{{{1, 2, 256}}, {{1, 2, 256}, {1, 4}}, std::nullopt}};
    args.m_in_channels = {2};
    args.m_out_channels = {2, 1};
    args.m_out_sizes = {128, 0};
    args.m_latencies = {64, 0};
    return args;
}

// ---- the comparison ----------------------------------------------------------------------------

/// Every getter of the live class against the shim on the same arguments, for every backend
/// this build has.
void expect_getters_equal(anira::InferenceConfig& old, const v2::InferenceConfig& shim) {
    EXPECT_EQ(old.get_tensor_input_shape(), shim.get_tensor_input_shape());
    EXPECT_EQ(old.get_tensor_output_shape(), shim.get_tensor_output_shape());
    EXPECT_EQ(old.get_tensor_input_size(), shim.get_tensor_input_size());
    EXPECT_EQ(old.get_tensor_output_size(), shim.get_tensor_output_size());
    EXPECT_EQ(old.get_preprocess_input_channels(), shim.get_preprocess_input_channels());
    EXPECT_EQ(old.get_postprocess_output_channels(), shim.get_postprocess_output_channels());
    EXPECT_EQ(old.get_preprocess_input_size(), shim.get_preprocess_input_size());
    EXPECT_EQ(old.get_postprocess_output_size(), shim.get_postprocess_output_size());
    EXPECT_EQ(old.get_internal_model_latency(), shim.get_internal_model_latency());
    EXPECT_FLOAT_EQ(old.m_max_inference_time, shim.m_max_inference_time);
    EXPECT_EQ(old.m_warm_up, shim.m_warm_up);
    EXPECT_EQ(old.m_session_exclusive_processor, shim.m_session_exclusive_processor);
    EXPECT_FLOAT_EQ(old.m_blocking_ratio, shim.m_blocking_ratio);
    EXPECT_EQ(old.m_num_parallel_processors, shim.m_num_parallel_processors);
    for (const v2::InferenceBackend backend : k_backends) {
        const std::optional<anira::InferenceBackend> mapped = old_backend(backend);
        if (!mapped.has_value()) { continue; }
        SCOPED_TRACE(v2::to_engine(backend).kind);
        EXPECT_EQ(old.get_tensor_input_shape(*mapped), shim.get_tensor_input_shape(backend));
        EXPECT_EQ(old.get_tensor_output_shape(*mapped), shim.get_tensor_output_shape(backend));
        const anira::ModelData* old_entry = old.get_model_data(*mapped);
        const v2::ModelData* shim_entry = shim.get_model_data(backend);
        ASSERT_EQ(old_entry != nullptr, shim_entry != nullptr);
        EXPECT_EQ(old.get_model_function(*mapped), shim.get_model_function(backend));
        EXPECT_EQ(old.is_model_binary(*mapped), shim.is_model_binary(backend));
        if (old_entry != nullptr) {
            // 2.x asserts on a backend without an entry: asked only where one exists.
            EXPECT_EQ(old.get_model_path(*mapped), shim.get_model_path(backend));
            EXPECT_TRUE(same_backend(old_entry->m_backend, shim_entry->m_backend));
            EXPECT_EQ(old_entry->m_size, shim_entry->m_size);
            EXPECT_EQ(old_entry->m_is_binary, shim_entry->m_is_binary);
        }
    }
    ASSERT_EQ(old.m_model_data.size(), shim.m_model_data.size());
    for (size_t i = 0; i < old.m_model_data.size(); ++i) {
        EXPECT_TRUE(same_backend(old.m_model_data[i].m_backend, shim.m_model_data[i].m_backend))
            << "model_data[" << i << "]";
        EXPECT_EQ(old.m_model_data[i].m_model_function, shim.m_model_data[i].m_model_function);
    }
}

/// Raises the log level to Warning for a scope, so that the shim's warnings are delivered in
/// every build type, and puts the previous level back.
struct WarningLevel {
    WarningLevel() : m_previous(anira::get_log_level()) {
        anira::set_log_level(anira::LogLevel::Warning);
    }
    ~WarningLevel() { anira::set_log_level(m_previous); }
    WarningLevel(const WarningLevel&) = delete;
    WarningLevel& operator=(const WarningLevel&) = delete;
    WarningLevel(WarningLevel&&) = delete;
    WarningLevel& operator=(WarningLevel&&) = delete;

    anira::LogLevel m_previous;
};

/// The status of the anira::Error a call throws; ANIRA_OK when it throws none.
template <class Call>
anira_status status_of(Call&& call) {
    try {
        call();
    } catch (const anira::Error& error) { return error.status; }
    return ANIRA_OK;
}

}  // namespace

// ---- the enum --------------------------------------------------------------------------------

// The shim's values are the anira_engine values under the 2.x names, and CUSTOM is the pair on
// anira.v2.custom: an entry of another custom id is CUSTOM to a processor but not the shim's.
TEST(AbiCompatConfig, TheBackendsConvertByName) {
    const std::array<std::pair<v2::InferenceBackend, anira_engine>, 5> built_in{{
        {v2::LIBTORCH, ANIRA_ENGINE_LIBTORCH},
        {v2::ONNX, ANIRA_ENGINE_ONNXRUNTIME},
        {v2::TFLITE, ANIRA_ENGINE_TFLITE},
        {v2::LITERT, ANIRA_ENGINE_LITERT},
        {v2::EXECUTORCH, ANIRA_ENGINE_EXECUTORCH},
    }};
    for (const auto& [backend, engine] : built_in) {
        EXPECT_EQ(static_cast<uint32_t>(backend), static_cast<uint32_t>(engine))
            << "the value is the engine's";
        EXPECT_EQ(v2::to_engine(backend).kind, engine);
        EXPECT_TRUE(v2::to_engine(backend).id.empty());
        EXPECT_EQ(v2::to_backend(anira::EngineRef{.kind = engine, .id = {}}), backend);
        EXPECT_TRUE(v2::names(anira::EngineRef{.kind = engine, .id = {}}, backend));
        EXPECT_FALSE(v2::names(anira::EngineRef{.kind = engine, .id = {}}, v2::CUSTOM));
        EXPECT_EQ(v2::is_available(backend), old_backend(backend).has_value());
    }
    EXPECT_EQ(v2::to_engine(v2::CUSTOM).kind, ANIRA_ENGINE_CUSTOM);
    EXPECT_EQ(v2::to_engine(v2::CUSTOM).id, "anira.v2.custom");
    EXPECT_EQ(v2::to_backend(v2::to_engine(v2::CUSTOM)), v2::CUSTOM);
    const anira::EngineRef other{.kind = ANIRA_ENGINE_CUSTOM, .id = "org.example.gain"};
    EXPECT_EQ(v2::to_backend(other), v2::CUSTOM);
    EXPECT_FALSE(v2::names(other, v2::CUSTOM));
    EXPECT_EQ(v2::to_backend(anira::EngineRef{}), v2::CUSTOM);
    EXPECT_TRUE(v2::is_available(v2::CUSTOM));
    EXPECT_EQ(v2::to_engine(k_no_backend).kind, ANIRA_ENGINE_NONE);
    EXPECT_FALSE(v2::is_available(k_no_backend));
    EXPECT_TRUE(same_backend(anira::InferenceBackend::CUSTOM, v2::CUSTOM));
}

// ---- InferenceConfig
// -----------------------------------------------------------------------------

// What the constructor writes is a version-2 document: its upgrade equals the upgrade of the
// same configuration spelled as a 2.x file, the model file and the legacy contract alike.
TEST(AbiCompatConfig, TheConstructorUpgradesLikeTheDocument) {
    const std::string stem =
        model_path("model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono");
    const v2::InferenceConfig shim(
        {v2::ModelData(stem + ".onnx", v2::ONNX), v2::ModelData(stem + ".tflite", v2::LITERT)},
        {v2::TensorShape({{1, 1, 512}, {1, 1, 1}}, {{1, 1, 512}, {1}}, v2::LITERT),
         v2::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {1}})},
        v2::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F,
        1,
        false,
        0.25F,
        3);
    const std::string document = R"({ "inference_config": {
    "model_data": [ { "model_path": ")" +
                                 stem + R"(.onnx", "inference_backend": "ONNX" },
                    { "model_path": ")" +
                                 stem + R"(.tflite", "inference_backend": "LITERT" } ],
    "tensor_shape": [ { "input_shape": [[1, 1, 512], [1, 1, 1]], "output_shape": [[1, 1, 512], [1]],
                        "inference_backend": "LITERT" },
                      { "input_shape": [[1, 1, 512], [1]], "output_shape": [[1, 1, 512], [1]] } ],
    "processing_spec": { "preprocess_input_channels": [1, 1], "postprocess_output_channels": [1, 1],
                         "preprocess_input_size": [512, 0], "postprocess_output_size": [512, 0] },
    "max_inference_time": 5.0, "warm_up": 1, "blocking_ratio": 0.25, "num_parallel_processors": 3 } })";
    anira::ModelConfig upgraded = anira::ModelConfig::from_json(document);
    ASSERT_TRUE(upgraded.upgraded());
    EXPECT_EQ(shim.model_config().to_json(), upgraded.to_json());
    const std::optional<anira::ContractHandle> legacy = upgraded.take_legacy_contract();
    if (!legacy.has_value()) { FAIL() << "the upgrade holds no legacy contract"; }
    const anira::Hard expected = legacy->hard();
    const anira::Hard actual = shim.hard();
    EXPECT_EQ(actual.budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(actual.budget_value, expected.budget_value);
    EXPECT_EQ(actual.warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(actual.warmup_iterations, 1U);
    EXPECT_EQ(actual.on_miss, ANIRA_MISS_ZEROS);
    EXPECT_DOUBLE_EQ(actual.wait_ratio, expected.wait_ratio);
    EXPECT_EQ(actual.block_max, 0U) << "no geometry: the handler's prepare sets it";
    EXPECT_TRUE(actual.ring_dtypes.empty());
    EXPECT_TRUE(actual.latencies.empty());
}

// Every getter of the live class on the same arguments: the gain (a Static tensor, a unit axis
// inserted on the TensorFlow rows), GuitarLSTM and the CNN (channels-last rows, a window longer
// than its hop), RAVE (stateful, latency, a fractional budget) and a stereo stream with a
// Static output and the default sizes.
TEST(AbiCompatConfig, TheGettersMatchTheLiveClass) {
    for (const auto& [name, args] : std::vector<std::pair<const char*, Arguments>>{
             {"gain", gain_arguments()},
             {"hybridnn", hybridnn_arguments()},
             {"cnn", cnn_arguments()},
             {"rave", rave_arguments()},
             {"stereo", stereo_arguments()},
         }) {
        SCOPED_TRACE(name);
        anira::InferenceConfig old = make_old(args);
        const v2::InferenceConfig shim = make_shim(args);
        expect_getters_equal(old, shim);
    }
}

// The backend-qualified shape getter answers the entry's layout over the canonical shape (what
// the 2.x HybridNN processor reads as its window length) and the canonical shape for a backend
// without an entry or without a layout.
TEST(AbiCompatConfig, TheBackendShapeGetterAppliesTheLayout) {
    const v2::InferenceConfig hybridnn = make_shim(hybridnn_arguments());
    const v2::TensorShapeList canonical{{256, 1, 150}};
    EXPECT_EQ(hybridnn.get_tensor_input_shape(), canonical);
    for (const v2::InferenceBackend backend : {v2::TFLITE, v2::LITERT}) {
        if (hybridnn.get_model_data(backend) == nullptr) {
            EXPECT_EQ(hybridnn.get_tensor_input_shape(backend), canonical);
            continue;
        }
        EXPECT_EQ(hybridnn.get_tensor_input_shape(backend)[0][1], 150);
        EXPECT_EQ(hybridnn.get_tensor_input_shape(backend), (v2::TensorShapeList{{256, 150, 1}}));
    }
    EXPECT_EQ(hybridnn.get_tensor_input_shape(v2::CUSTOM), canonical);
    EXPECT_EQ(hybridnn.get_tensor_input_shape(k_no_backend), canonical);

    // The shim keeps the rows of every backend, present in this build or not.
    const v2::InferenceConfig gain(
        {v2::ModelData("gain.tflite", v2::TFLITE), v2::ModelData("gain.onnx", v2::ONNX)},
        {v2::TensorShape({{1, 1, 512}, {1, 1, 1}}, {{1, 1, 512}, {1}}, v2::TFLITE),
         v2::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {1}})},
        v2::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F);
    EXPECT_EQ(gain.get_tensor_input_shape(v2::TFLITE),
              (v2::TensorShapeList{{1, 1, 512}, {1, 1, 1}}));
    EXPECT_EQ(gain.get_tensor_input_shape(v2::ONNX), (v2::TensorShapeList{{1, 1, 512}, {1}}));
    EXPECT_EQ(gain.get_tensor_output_shape(v2::TFLITE), (v2::TensorShapeList{{1, 1, 512}, {1}}));
    // The rows: the universal one, TFLITE's own, a universal clone for ONNX (the 2.x set).
    ASSERT_EQ(gain.m_tensor_shape.size(), 3U);
    EXPECT_TRUE(gain.m_tensor_shape[0].is_universal());
    EXPECT_FALSE(gain.m_tensor_shape[1].is_universal());
    EXPECT_EQ(gain.m_tensor_shape[1].m_backend, v2::TFLITE);
    EXPECT_TRUE(gain.m_tensor_shape[2].is_universal());
    EXPECT_EQ(gain.m_tensor_shape[2].m_backend, v2::ONNX);
}

// A bytes entry is borrowed: the handle, the read-back and a copy all point at the caller's
// buffer; the path getter answers the bytes as a string, as 2.x did.
TEST(AbiCompatConfig, BytesEntriesAreBorrowed) {
    const std::string bytes = "not really a model, but bytes";
    std::vector<char> buffer(bytes.begin(), bytes.end());
    Arguments args = stereo_arguments();
    args.m_rows.clear();
    anira::InferenceConfig old(
        {anira::ModelData(buffer.data(), buffer.size(), anira::InferenceBackend::CUSTOM)},
        {anira::TensorShape(args.m_shapes[0].m_inputs, args.m_shapes[0].m_outputs)},
        anira::ProcessingSpec(args.m_in_channels,
                              args.m_out_channels,
                              args.m_in_sizes,
                              args.m_out_sizes,
                              args.m_latencies),
        5.F);
    const v2::InferenceConfig shim(
        {v2::ModelData(buffer.data(), buffer.size(), v2::CUSTOM),
         v2::ModelData("entry.pt", v2::LIBTORCH, "forward_streaming")},
        {v2::TensorShape(args.m_shapes[0].m_inputs, args.m_shapes[0].m_outputs)},
        v2::ProcessingSpec(args.m_in_channels,
                           args.m_out_channels,
                           args.m_in_sizes,
                           args.m_out_sizes,
                           args.m_latencies),
        5.F);
    EXPECT_TRUE(shim.is_model_binary(v2::CUSTOM));
    EXPECT_EQ(shim.get_model_data(v2::CUSTOM)->m_data, buffer.data());
    EXPECT_EQ(old.get_model_data(anira::InferenceBackend::CUSTOM)->m_data, buffer.data());
    EXPECT_EQ(shim.get_model_path(v2::CUSTOM), bytes);
    EXPECT_EQ(old.get_model_path(anira::InferenceBackend::CUSTOM), bytes);
    EXPECT_EQ(shim.model_config().model_bytes(0).data(),
              reinterpret_cast<const std::byte*>(buffer.data()));
    EXPECT_FALSE(shim.is_model_binary(v2::LIBTORCH));
    EXPECT_EQ(shim.get_model_path(v2::LIBTORCH), "entry.pt");
    EXPECT_EQ(shim.get_model_function(v2::LIBTORCH), "forward_streaming");

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) the copy is the subject
    const v2::InferenceConfig copy = shim;
    EXPECT_EQ(copy, shim);
    EXPECT_NE(copy.model_config().native(), shim.model_config().native());
    EXPECT_EQ(copy.get_model_data(v2::CUSTOM)->m_data, buffer.data());
    EXPECT_EQ(copy.model_config().model_bytes(0).data(),
              reinterpret_cast<const std::byte*>(buffer.data()));
    const anira::ModelConfig own = shim.model_config_copy();
    EXPECT_EQ(own.model_bytes(0).data(), reinterpret_cast<const std::byte*>(buffer.data()));
    EXPECT_EQ(own.to_json(), shim.model_config().to_json());

    // Another buffer with the same bytes is another configuration (the 2.x rule: by pointer).
    std::vector<char> other(bytes.begin(), bytes.end());
    const v2::InferenceConfig elsewhere(
        {v2::ModelData(other.data(), other.size(), v2::CUSTOM),
         v2::ModelData("entry.pt", v2::LIBTORCH, "forward_streaming")},
        {v2::TensorShape(args.m_shapes[0].m_inputs, args.m_shapes[0].m_outputs)},
        v2::ProcessingSpec(args.m_in_channels,
                           args.m_out_channels,
                           args.m_in_sizes,
                           args.m_out_sizes,
                           args.m_latencies),
        5.F);
    EXPECT_NE(elsewhere, shim);
}

// A copy loads its document again and equals the original; equality is the configuration's,
// not the spelling's (an explicit default equals the default); an empty configuration answers
// empty and has no model config; the handler's private copy takes an anchor the original never
// sees.
TEST(AbiCompatConfig, CopiesAndEquality) {
    const v2::InferenceConfig rave = make_shim(rave_arguments());
    v2::InferenceConfig copy = rave;
    EXPECT_EQ(copy, rave);
    EXPECT_NE(copy.model_config().native(), rave.model_config().native());
    const v2::InferenceConfig moved = std::move(copy);
    EXPECT_EQ(moved, rave);

    Arguments spelled = stereo_arguments();
    Arguments defaulted = spelled;
    spelled.m_in_sizes = {256};  // what the default computes
    EXPECT_EQ(make_shim(spelled), make_shim(defaulted));
    defaulted.m_warm_up = 3;
    EXPECT_NE(make_shim(spelled), make_shim(defaulted));

    v2::InferenceConfig assigned = make_shim(stereo_arguments());
    assigned = rave;
    EXPECT_EQ(assigned, rave);

    const v2::InferenceConfig empty;
    EXPECT_TRUE(empty.get_tensor_input_shape().empty());
    EXPECT_TRUE(empty.get_tensor_input_shape(v2::ONNX).empty());
    EXPECT_TRUE(empty.get_preprocess_input_size().empty());
    EXPECT_TRUE(empty.m_model_data.empty());
    EXPECT_EQ(empty.get_model_data(v2::ONNX), nullptr);
    EXPECT_EQ(empty.get_model_path(v2::ONNX), "");
    EXPECT_EQ(status_of([&] { static_cast<void>(empty.model_config()); }),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(status_of([&] { static_cast<void>(empty.model_config_copy()); }),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(status_of([&] { static_cast<void>(empty.hard()); }), ANIRA_ERROR_INVALID_STATE);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) the copy is the subject
    const v2::InferenceConfig empty_copy = empty;
    EXPECT_EQ(empty_copy, empty);
    EXPECT_NE(empty, rave);

    anira::ModelConfig own = rave.model_config_copy();
    own.anchor("output_0");
    EXPECT_EQ(own.anchor(), "output_0");
    EXPECT_TRUE(rave.model_config().anchor().empty());
}

// The normalisation table, row by row: what the 2.x constructor did that the upgrade does not,
// and what anira 3 refuses where 2.x ran.
TEST(AbiCompatConfig, TheNormalisationTable) {
    const WarningLevel level;
    anira_test::RecordCollector collector;

    // max_inference_time <= 0: refused first, by both (2.x: std::invalid_argument).
    Arguments args = stereo_arguments();
    args.m_max_inference_time = 0.F;
    EXPECT_EQ(status_of([&] { make_shim(args); }), ANIRA_ERROR_CONFIG);
    EXPECT_THROW(make_old(args), std::invalid_argument);

    // Session-exclusive: one processor on both.
    args = stereo_arguments();
    args.m_exclusive = true;
    args.m_processors = 4;
    EXPECT_EQ(make_shim(args).m_num_parallel_processors, 1U);
    EXPECT_EQ(make_old(args).m_num_parallel_processors, 1U);

    // A count below 1: 1 on both, with the 2.x warning.
    args = stereo_arguments();
    args.m_processors = 0;
    EXPECT_EQ(make_shim(args).m_num_parallel_processors, 1U);
    EXPECT_EQ(make_old(args).m_num_parallel_processors, 1U);
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.has("parallel processors must be at least 1", "native"));
#endif

    // A spec vector with the wrong entry count: dropped and defaulted on both, with a warning
    // on the shim's side.
    args = stereo_arguments();
    args.m_in_channels = {2, 2, 2};
    args.m_in_sizes = {256, 256};
    args.m_latencies = {1};
    {
        anira::InferenceConfig old = make_old(args);
        const v2::InferenceConfig shim = make_shim(args);
        EXPECT_EQ(shim.get_preprocess_input_channels(), (std::vector<size_t>{1}));
        expect_getters_equal(old, shim);
    }
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.has("preprocess_input_channels has 3 entries", "native"));
    EXPECT_TRUE(collector.has("internal_model_latency has 1 entries", "native"));
#endif

    // A non-streamable tensor with more than one channel: refused by both.
    args = stereo_arguments();
    args.m_out_channels = {2, 2};
    EXPECT_EQ(status_of([&] { make_shim(args); }), ANIRA_ERROR_CONFIG);
    EXPECT_THROW(make_old(args), std::invalid_argument);

    // A model_function on a backend that has none: the shim drops it with the 2.x message.
    {
        const v2::InferenceConfig shim({v2::ModelData("gain.onnx", v2::ONNX, "forward")},
                                       {v2::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
                                       5.F);
        EXPECT_EQ(shim.get_model_function(v2::ONNX), "");
        EXPECT_FALSE(shim.model_config().model_ext<anira::ext::Entry>(0).has_value());
    }
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.has("Model function is only applicable", "native"));
#endif

    // A backend row that reshapes (the same element count, other extents): 2.x ran it, anira 3
    // permutes axes per engine and cannot reshape.
    EXPECT_EQ(status_of([] {
                  const v2::InferenceConfig shim(
                      {v2::ModelData("gain.onnx", v2::ONNX)},
                      {v2::TensorShape({{1, 1, 512}}, {{1, 1, 512}}),
                       v2::TensorShape({{1, 2, 256}}, {{1, 1, 512}}, v2::ONNX)},
                      5.F);
              }),
              ANIRA_ERROR_JSON);

    // A backend row of a backend without an entry is dropped, whatever it holds.
    {
        const v2::InferenceConfig shim({v2::ModelData("gain.onnx", v2::ONNX)},
                                       {v2::TensorShape({{1, 1, 512}}, {{1, 1, 512}}),
                                        v2::TensorShape({{1, 2, 256}}, {{1, 1, 512}}, v2::TFLITE)},
                                       5.F);
        EXPECT_EQ(shim.get_tensor_input_shape(v2::TFLITE), (v2::TensorShapeList{{1, 1, 512}}));
    }

    // A streamed tensor with two channels and no axis of extent 2: refused.
    args = stereo_arguments();
    args.m_shapes = {{{{1, 1, 512}}, {{1, 2, 256}, {1, 4}}, std::nullopt}};
    EXPECT_EQ(status_of([&] { make_shim(args); }), ANIRA_ERROR_JSON);

    // The value types refuse what 2.x asserted.
    EXPECT_EQ(status_of([] { const v2::ModelData data(nullptr, 4, v2::ONNX); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(status_of([] { const v2::ModelData data("", v2::ONNX); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(status_of([] { const v2::TensorShape shape({}, {{1}}); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(status_of([] {
                  const v2::InferenceConfig shim({v2::ModelData("gain.onnx", k_no_backend)},
                                                 {v2::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
                                                 5.F);
              }),
              ANIRA_ERROR_INVALID_ARGUMENT);
}

// A path entry owns its characters: copies and moves point at their own, equal by content.
TEST(AbiCompatConfig, ModelDataOwnsItsPath) {
    std::string path = "a/path/to/model.onnx";
    const v2::ModelData entry(path, v2::ONNX);
    path.assign(path.size(), 'x');
    EXPECT_EQ(std::string(static_cast<const char*>(entry.m_data), entry.m_size),
              "a/path/to/model.onnx");
    v2::ModelData copy = entry;
    EXPECT_NE(copy.m_data, entry.m_data);
    EXPECT_EQ(copy, entry);
    const v2::ModelData moved = std::move(copy);
    EXPECT_EQ(moved, entry);
    v2::ModelData assigned("other.pt", v2::LIBTORCH, "forward");
    assigned = entry;
    EXPECT_EQ(assigned, entry);
    EXPECT_NE(assigned.m_data, entry.m_data);
    EXPECT_NE(v2::ModelData("a/path/to/model.onnx", v2::LITERT), entry);
    std::vector<char> bytes(8);
    const v2::ModelData binary(bytes.data(), bytes.size(), v2::ONNX);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) the copy is the subject
    const v2::ModelData binary_copy = binary;
    EXPECT_EQ(binary_copy.m_data, bytes.data()) << "bytes are borrowed";
}

// ---- ContextConfig ----------------------------------------------------------------------------

// The 2.x fields mint an anira::ContextConfig and read back from one; the default thread count
// is the sentinel anira resolves.
TEST(AbiCompatConfig, TheContextConfigMintsAndReadsBack) {
    const v2::ContextConfig defaults;
    EXPECT_EQ(defaults.m_num_threads, v2::ContextConfig::k_threads_auto);
    EXPECT_EQ(defaults.m_wait_strategy, v2::WaitStrategy::SpinBackoff);
    EXPECT_EQ(defaults.m_log, v2::LogConfig());
    EXPECT_EQ(defaults.m_log.m_level, v2::default_log_level());

    v2::ContextConfig config(3, v2::WaitStrategy::Blocking, v2::LogLevel::Debug);
    config.m_log.m_drain = v2::LogDrain::Manual;
    config.m_log.m_queue_capacity = 1024;
    config.m_log.m_drain_interval_ms = 5;
    const v2::ContextConfig read(config.to_context_config());
    EXPECT_EQ(read.m_num_threads, 3U);
    EXPECT_EQ(read.m_wait_strategy, v2::WaitStrategy::Blocking);
    EXPECT_EQ(read.m_log, config.m_log);

    const v2::ContextConfig two = 2;  // the 2.x implicit form
    EXPECT_EQ(two.m_num_threads, 2U);
    EXPECT_STREQ(v2::to_string(v2::WaitStrategy::Blocking), "blocking");
    EXPECT_STREQ(v2::to_string(v2::LogLevel::Warning), "warning");
    EXPECT_STREQ(v2::to_string(v2::LogDrain::Manual), "manual");
    const v2::HostConfig host(512.F, 48000.F);
    EXPECT_EQ(host,
              v2::HostConfig(512.F, 48000.F, false, v2::HostConfig::k_first_streamable, true));
    EXPECT_NE(host, v2::HostConfig(512.F, 44100.F));
}

// ---- JsonConfigLoader ------------------------------------------------------------------------

// The four version-2 documents through the istream constructor, against the live loader: the
// context's thread count where the document sets it, and every getter of the configuration.
TEST(AbiCompatConfig, TheLoaderMatchesTheLiveLoader) {
    for (const auto& [name, text] : std::vector<std::pair<const char*, std::string>>{
             {"gain", anira_test::gain_v2_document()},
             {"rave", anira_test::rave_funk_drum_v2_document()},
             {"encoder", anira_test::rave_funk_drum_encoder_v2_document()},
             {"decoder", anira_test::rave_funk_drum_decoder_v2_document()},
         }) {
        SCOPED_TRACE(name);
        std::istringstream old_stream(text);
        anira::JsonConfigLoader old_loader(old_stream);
        std::istringstream shim_stream(text);
        v2::JsonConfigLoader shim_loader(shim_stream);

        const std::unique_ptr<anira::CoreConfig> old_context = old_loader.get_core_config();
        const std::unique_ptr<v2::ContextConfig> shim_context = shim_loader.get_context_config();
        ASSERT_NE(old_context, nullptr);
        ASSERT_NE(shim_context, nullptr);
        if (text.find("\"context_config\"") != std::string::npos) {
            EXPECT_EQ(shim_context->m_num_threads, old_context->m_num_threads);
        } else {
            EXPECT_EQ(shim_context->m_num_threads, v2::ContextConfig::k_threads_auto);
        }
        EXPECT_EQ(shim_loader.get_context_config(), nullptr) << "once, as in 2.x";

        const std::unique_ptr<anira::InferenceConfig> old = old_loader.get_inference_config();
        const std::unique_ptr<v2::InferenceConfig> shim = shim_loader.get_inference_config();
        if (old == nullptr) {
            // The document's only engine is not in this build: 2.x dropped the row and gave up.
            ASSERT_NE(shim, nullptr);
            continue;
        }
        ASSERT_NE(shim, nullptr);
        EXPECT_EQ(shim_loader.get_inference_config(), nullptr) << "once, as in 2.x";
        // Only the rows of this build: 2.x dropped the others, the shim keeps them as entries.
        for (const v2::InferenceBackend backend : k_backends) {
            const std::optional<anira::InferenceBackend> mapped = old_backend(backend);
            if (!mapped.has_value()) { continue; }
            EXPECT_EQ(old->get_model_function(*mapped), shim->get_model_function(backend));
            EXPECT_EQ(old->get_tensor_input_shape(*mapped), shim->get_tensor_input_shape(backend));
            EXPECT_EQ(old->get_tensor_output_shape(*mapped),
                      shim->get_tensor_output_shape(backend));
            if (old->get_model_data(*mapped) != nullptr) {
                EXPECT_EQ(old->get_model_path(*mapped), shim->get_model_path(backend));
            }
        }
        EXPECT_EQ(old->get_tensor_input_shape(), shim->get_tensor_input_shape());
        EXPECT_EQ(old->get_tensor_input_size(), shim->get_tensor_input_size());
        EXPECT_EQ(old->get_tensor_output_size(), shim->get_tensor_output_size());
        EXPECT_EQ(old->get_preprocess_input_channels(), shim->get_preprocess_input_channels());
        EXPECT_EQ(old->get_postprocess_output_channels(), shim->get_postprocess_output_channels());
        EXPECT_EQ(old->get_preprocess_input_size(), shim->get_preprocess_input_size());
        EXPECT_EQ(old->get_postprocess_output_size(), shim->get_postprocess_output_size());
        EXPECT_EQ(old->get_internal_model_latency(), shim->get_internal_model_latency());
        EXPECT_FLOAT_EQ(old->m_max_inference_time, shim->m_max_inference_time);
        EXPECT_EQ(old->m_warm_up, shim->m_warm_up);
        EXPECT_EQ(old->m_session_exclusive_processor, shim->m_session_exclusive_processor);
        EXPECT_FLOAT_EQ(old->m_blocking_ratio, shim->m_blocking_ratio);
        EXPECT_EQ(old->m_num_parallel_processors, shim->m_num_parallel_processors);
    }
}

// The path constructor: a relative model_path resolves against the document's directory (the
// 3.x rule), and every entry of the document is kept, present in this build or not.
TEST(AbiCompatConfig, TheLoaderReadsAFile) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "anira_test_compat_loader";
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / "gain.json";
    {
        std::ofstream out(file);
        out << R"({ "context_config": { "num_threads": 2, "wait_strategy": "blocking",
                  "log": { "level": "debug", "drain": "manual", "queue_capacity": 1024,
                           "drain_interval_ms": 7 } },
  "inference_config": {
    "model_data": [ { "model_path": "models/gain.onnx", "inference_backend": "ONNX" },
                    { "model_path": "models/gain.pte", "inference_backend": "EXECUTORCH",
                      "model_function": "forward" } ],
    "tensor_shape": [ { "input_shape": [1, 1, 512], "output_shape": [1, 1, 512] } ],
    "max_inference_time": 2.5 } })";
    }
    v2::JsonConfigLoader loader(file.string());
    const std::unique_ptr<v2::ContextConfig> context = loader.get_context_config();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->m_num_threads, 2U);
    EXPECT_EQ(context->m_wait_strategy, v2::WaitStrategy::Blocking);
    EXPECT_EQ(context->m_log.m_level, v2::LogLevel::Debug);
    EXPECT_EQ(context->m_log.m_drain, v2::LogDrain::Manual);
    EXPECT_EQ(context->m_log.m_queue_capacity, 1024U);
    EXPECT_EQ(context->m_log.m_drain_interval_ms, 7U);
    const std::unique_ptr<v2::InferenceConfig> config = loader.get_inference_config();
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->get_model_path(v2::ONNX),
              (dir / "models/gain.onnx").lexically_normal().generic_string());
    EXPECT_EQ(config->get_model_function(v2::EXECUTORCH), "forward");
    EXPECT_FLOAT_EQ(config->m_max_inference_time, 2.5F);
    EXPECT_EQ(config->get_preprocess_input_size(), (std::vector<size_t>{512}));
    std::filesystem::remove_all(dir);
}

// No leniency: what the 2.x loader logged and answered with nullptr is an anira::Error, from
// the constructor; a document with a context_config alone has no InferenceConfig to give.
TEST(AbiCompatConfig, TheLoaderRefuses) {
    EXPECT_EQ(status_of([] { const v2::JsonConfigLoader loader("/no/such/anira/config.json"); }),
              ANIRA_ERROR_NO_SUCH_FILE);
    const auto from_text = [](const std::string& text) {
        return status_of([&text] {
            std::istringstream stream(text);
            const v2::JsonConfigLoader loader(stream);
        });
    };
    EXPECT_EQ(from_text("{ \"inference_config\": "), ANIRA_ERROR_JSON);
    EXPECT_EQ(from_text(anira_test::k_model_v3), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(from_text(anira_test::k_context_v3), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(from_text(R"({ "inference_config": { "model_data": [],
        "tensor_shape": [ { "input_shape": [1, 512], "output_shape": [1, 512] } ],
        "max_inference_time": 5.0, "warm_up": "five" } })"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(from_text(R"({ "context_config": { "num_threads": "two" } })"), ANIRA_ERROR_JSON);

    std::istringstream context_only(R"({ "context_config": { "num_threads": 3 } })");
    v2::JsonConfigLoader loader(context_only);
    const std::unique_ptr<v2::ContextConfig> context = loader.get_context_config();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->m_num_threads, 3U);
    EXPECT_EQ(status_of([&loader] { static_cast<void>(loader.get_inference_config()); }),
              ANIRA_ERROR_CONFIG);
}

// ---- the runtime half: the pass-through and the stage (suite AbiCompat) ----------------------

namespace {

/// A C handler created from a C++ Pipeline, destroyed with the object.
struct PipelineHandler {
    PipelineHandler(const anira_test::Context& context, const anira::Pipeline& pipeline) {
        m_status = anira_handler_create(context.m_context, pipeline.native(), &m_handler, &m_err);
    }
    ~PipelineHandler() { anira_handler_destroy(m_handler); }
    PipelineHandler(const PipelineHandler&) = delete;
    PipelineHandler& operator=(const PipelineHandler&) = delete;
    PipelineHandler(PipelineHandler&&) = delete;
    PipelineHandler& operator=(PipelineHandler&&) = delete;

    anira_status prepare(const anira::ContractHandle& contract) {
        m_err = ANIRA_ERROR_INIT;
        return anira_handler_prepare(m_handler, contract.native(), &m_err);
    }

    anira_handler* m_handler = nullptr;
    anira_status m_status = ANIRA_OK;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// The 2.x side's core config: what anira_test::Context sets on the C side, so the two users
/// of the core reconcile without a mismatch.
anira::CoreConfig oracle_core_config() {
    anira::CoreConfig config(2, anira::WaitStrategy::SpinBackoff, anira::LogLevel::Error);
    config.m_log.m_drain = anira::LogDrain::Manual;
    return config;
}

/// The legacy contract of a shim configuration with the host geometry of a block.
anira::ContractHandle contract_of(const v2::InferenceConfig& config, uint32_t block) {
    anira::Hard hard = config.hard();
    hard.block_min = block;
    hard.block_max = block;
    hard.rate = anira_test::k_rate;
    return anira::ContractHandle(hard);
}

/// The plan rows of a prepared handler.
std::vector<anira_plan_info> plans_of(const anira_handler* handler) {
    const anira_plan_report* report = anira_handler_plan_report(handler);
    EXPECT_NE(report, nullptr);
    uint32_t count = anira_plan_report_num_plans(report);
    std::vector<anira_plan_info> rows(count);
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
              ANIRA_OK);
    return rows;
}

/// The backend a plan row runs on, as the stage context hands it to a processor.
v2::InferenceBackend backend_of(const anira_plan_info& info) {
    return v2::to_backend(anira::EngineRef{
        .kind = static_cast<anira_engine>(info.engine),
        .id = info.engine_id != nullptr ? std::string_view(info.engine_id) : std::string_view()});
}

/// The pass-through's three outputs against its two inputs: a copy, a pair of other sizes, an
/// output without an input.
constexpr size_t k_stream = 512;
constexpr size_t k_small_in = 2;
constexpr size_t k_small_out = 4;

}  // namespace

// The pass-through is 2.x's BackendBase::process: output i a copy of input i of the same size,
// zeros for a pair of other sizes and for an output beyond the inputs. Once on the kernel
// (the engine's three levels against BackendBase over the same data), once through a C handler
// built on an anira::Pipeline with the LegacyProcessorStage against the live 2.x handler, whose
// CUSTOM row runs the 2.x base pass-through: the same blocks, bit for bit.
TEST(AbiCompat, ThePassthroughEngineIsTheBackendBase) {
    const std::shared_ptr<v2::PassthroughEngine> engine = std::make_shared<v2::PassthroughEngine>();
    EXPECT_EQ(engine->id(), v2::k_custom_engine_id);
    EXPECT_EQ(engine->flags(), ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL);
    EXPECT_TRUE(engine->providers().empty()) << "the CPU path alone";

    // The kernel: 64 and 16 elements in, 64, 32 and 64 out.
    {
        std::vector<float> in0 = anira_test::ramp(1, 64);
        std::vector<float> in1(16, 0.25F);
        std::vector<float> out0(64, 7.F);
        std::vector<float> out1(32, 7.F);
        std::vector<float> out2(64, 7.F);
        const std::array<int64_t, 2> wide{1, 64};
        const std::array<int64_t, 2> narrow{1, 16};
        const std::array<int64_t, 2> middle{1, 32};
        const std::array<anira::Tensor, 2> inputs{
            anira::Tensor::from_host(in0.data(), ANIRA_DTYPE_F32, wide),
            anira::Tensor::from_host(in1.data(), ANIRA_DTYPE_F32, narrow)};
        std::array<anira::Tensor, 3> outputs{
            anira::Tensor::from_host(out0.data(), ANIRA_DTYPE_F32, wide),
            anira::Tensor::from_host(out1.data(), ANIRA_DTYPE_F32, middle),
            anira::Tensor::from_host(out2.data(), ANIRA_DTYPE_F32, wide)};
        const anira_engine_load_info load_info{};
        const anira_prepare_info prepare_info{};
        const std::unique_ptr<anira::Engine::Loaded> loaded =
            engine->load(anira::EngineLoadInfo(&load_info));
        ASSERT_NE(loaded, nullptr);
        const std::unique_ptr<anira::Engine::Prepared> prepared =
            loaded->prepare(anira::PrepareInfo(&prepare_info));
        ASSERT_NE(prepared, nullptr);
        anira_engine_ctx record{};
        record.num_inputs = static_cast<uint32_t>(inputs.size());
        record.num_outputs = static_cast<uint32_t>(outputs.size());
        record.inputs = inputs.data();
        record.outputs = outputs.data();
        anira::EngineContext context(&record);
        EXPECT_EQ(prepared->process(context), ANIRA_OK);

        anira::InferenceConfig config_2x(
            {anira::ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
            {anira::TensorShape({{1, 64}, {1, 16}}, {{1, 64}, {1, 32}, {1, 64}})},
            5.F);
        anira::BackendBase base(config_2x);
        std::vector<anira::BufferF> old_in;
        old_in.emplace_back(1, 64);
        old_in.emplace_back(1, 16);
        std::ranges::copy(in0, old_in[0].get_write_pointer(0));
        std::ranges::copy(in1, old_in[1].get_write_pointer(0));
        std::vector<anira::BufferF> old_out;
        old_out.emplace_back(1, 64);
        old_out.emplace_back(1, 32);
        old_out.emplace_back(1, 64);
        for (anira::BufferF& buffer : old_out) {
            std::fill_n(buffer.get_write_pointer(0), buffer.get_num_samples(), 7.F);
        }
        base.process(old_in, old_out, nullptr);

        anira_test::expect_same_block(out0, in0, 0);
        anira_test::expect_same_block(out0,
                                      std::span<const float>(old_out[0].get_read_pointer(0), 64),
                                      0);
        anira_test::expect_all(out1, 0.F, "a pair of other sizes");
        anira_test::expect_same_block(out1,
                                      std::span<const float>(old_out[1].get_read_pointer(0), 32),
                                      1);
        anira_test::expect_all(out2, 0.F, "an output without an input");
        anira_test::expect_same_block(out2,
                                      std::span<const float>(old_out[2].get_read_pointer(0), 64),
                                      2);
    }

    // Through the handlers: a stream and a Static pair of other sizes in, a stream, a Static
    // output and a stream without an input out.
    const anira_test::Context context;
    v2::InferenceConfig config(
        {v2::ModelData("placeholder", v2::CUSTOM)},
        {v2::TensorShape({{1, 1, k_stream}, {1, k_small_in}},
                         {{1, 1, k_stream}, {1, k_small_out}, {1, 1, k_stream}})},
        v2::ProcessingSpec({1, 1}, {1, 1, 1}, {k_stream, 0}, {k_stream, 0, k_stream}),
        5.F,
        0,
        false,
        0.F,
        2);
    v2::PrePostProcessor pp(config);
    anira::Pipeline pipeline;
    pipeline.register_engine(engine);
    pipeline.inference(config.model_config());
    pipeline.add(anira::stage::Custom(std::make_shared<v2::LegacyProcessorStage>(pp)));
    PipelineHandler handler(context, pipeline);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(contract_of(config, k_stream)), ANIRA_OK) << handler.m_err.message;
    const std::vector<anira_plan_info> plans = plans_of(handler.m_handler);
    ASSERT_EQ(plans.size(), 1U);
    EXPECT_EQ(plans[0].engine, ANIRA_ENGINE_CUSTOM);
    EXPECT_STREQ(plans[0].engine_id, v2::k_custom_engine_id);
    EXPECT_EQ(plans[0].provider, ANIRA_PROVIDER_CPU);
    EXPECT_EQ(plans[0].engine_flags,
              ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL);
    anira_test::FloatFace face(handler.m_handler);

    anira::InferenceConfig config_2x(
        {anira::ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        {anira::TensorShape({{1, 1, k_stream}, {1, k_small_in}},
                            {{1, 1, k_stream}, {1, k_small_out}, {1, 1, k_stream}})},
        anira::ProcessingSpec({1, 1}, {1, 1, 1}, {k_stream, 0}, {k_stream, 0, k_stream}),
        5.F,
        0,
        false,
        0.F,
        2);
    anira::PrePostProcessor pp_2x(config_2x);
    anira::InferenceHandler handler_2x(pp_2x, config_2x, oracle_core_config());
    handler_2x.prepare(
        anira::HostConfig(static_cast<float>(k_stream), static_cast<float>(anira_test::k_rate)));

    bool delivered_signal = false;
    for (size_t k = 1; k <= 12; ++k) {
        const std::vector<float> in = anira_test::ramp(k, k_stream);
        for (size_t j = 0; j < k_small_in; ++j) {
            pp.set_input(static_cast<float>(k + j), 1, j);
            pp_2x.set_input(static_cast<float>(k + j), 1, j);
        }
        std::vector<float> c0(k_stream, -1.F);
        std::vector<float> c2(k_stream, -1.F);
        const std::array<const float*, 1> c_in_ch{in.data()};
        const std::array<const float* const*, 2> c_in{c_in_ch.data(), nullptr};
        const std::array<size_t, 2> num_in{k_stream, 0};
        const std::array<float*, 1> c_out0{c0.data()};
        const std::array<float*, 1> c_out2{c2.data()};
        const std::array<float* const*, 3> c_out{c_out0.data(), nullptr, c_out2.data()};
        const std::array<size_t, 3> num_out{k_stream, 0, k_stream};
        std::array<size_t, 3> delivered{};
        const size_t prev = anira_test::available(handler.m_handler);
        ASSERT_EQ(face.process_multi(c_in.data(),
                                     num_in.data(),
                                     c_out.data(),
                                     num_out.data(),
                                     delivered.data()),
                  ANIRA_OK);
        anira_test::wait_for_block(handler.m_handler, prev);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());

        std::vector<float> v0(k_stream, -1.F);
        std::vector<float> v2_out(k_stream, -1.F);
        const std::array<const float*, 1> v_in_ch{in.data()};
        const std::array<const float* const*, 2> v_in{v_in_ch.data(), nullptr};
        std::array<size_t, 2> v_num_in{k_stream, 0};
        const std::array<float*, 1> v_out0{v0.data()};
        const std::array<float*, 1> v_out2{v2_out.data()};
        const std::array<float* const*, 3> v_out{v_out0.data(), nullptr, v_out2.data()};
        std::array<size_t, 3> v_num_out{k_stream, 0, k_stream};
        const size_t prev_2x = handler_2x.get_available_samples(0);
        handler_2x.process(v_in.data(), v_num_in.data(), v_out.data(), v_num_out.data());
        anira_test::wait_for_block(handler_2x, prev_2x);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());

        EXPECT_EQ(delivered[0], v_num_out[0]) << "block " << k;
        anira_test::expect_same_block(c0, v0, k);
        anira_test::expect_same_block(c2, v2_out, k);
        anira_test::expect_all(c2, 0.F, "an output without an input");
        for (size_t j = 0; j < k_small_out; ++j) {
            EXPECT_EQ(pp.get_output(1, j), 0.F) << "a pair of other sizes, block " << k;
            EXPECT_EQ(pp.get_output(1, j), pp_2x.get_output(1, j)) << "block " << k;
        }
        delivered_signal = delivered_signal || c0.back() != 0.F;
    }
    EXPECT_TRUE(delivered_signal) << "the stream came through within twelve blocks";
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
}

namespace {

/// A 2.x processor that records the backend each of the four phases was called with.
class RecordingProcessor : public v2::PrePostProcessor {
public:
    explicit RecordingProcessor(v2::InferenceConfig& config) : v2::PrePostProcessor(config) {
        forget();
    }

    void pre_process(std::vector<v2::RingBuffer>& input,
                     std::vector<v2::BufferF>& output,
                     v2::InferenceBackend backend) override {
        m_pre.store(static_cast<int>(backend));
        v2::PrePostProcessor::pre_process(input, output, backend);
    }
    void post_process(std::vector<v2::BufferF>& input,
                      std::vector<v2::RingBuffer>& output,
                      v2::InferenceBackend backend) override {
        m_post.store(static_cast<int>(backend));
        v2::PrePostProcessor::post_process(input, output, backend);
    }
    void before_inference(std::vector<v2::BufferF>& /*input*/,
                          v2::InferenceBackend backend) override {
        m_before.store(static_cast<int>(backend));
    }
    void after_inference(std::vector<v2::BufferF>& /*output*/,
                         v2::InferenceBackend backend) override {
        m_after.store(static_cast<int>(backend));
    }

    void forget() {
        m_pre.store(-1);
        m_post.store(-1);
        m_before.store(-1);
        m_after.store(-1);
    }

    std::atomic<int> m_pre{-1};
    std::atomic<int> m_post{-1};
    std::atomic<int> m_before{-1};
    std::atomic<int> m_after{-1};
};

}  // namespace

// The processor sees the backend of the chunk's plan in all four phases: CUSTOM on the custom
// plan (the stage saw ANIRA_ENGINE_CUSTOM with anira.v2.custom), the engine's backend on a
// built-in plan. Every plan of the default candidate set is on the CPU path.
TEST(AbiCompat, TheStageReceivesTheBackendOfThePlan) {
    const std::string stem =
        model_path("model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono");
    std::vector<v2::ModelData> rows;
    for (const anira_engine engine : anira_test::oracle_engines()) {
        const v2::InferenceBackend backend =
            v2::to_backend(anira::EngineRef{.kind = engine, .id = {}});
        const char* extension = backend == v2::ONNX         ? ".onnx"
                                : backend == v2::LIBTORCH   ? ".pt"
                                : backend == v2::EXECUTORCH ? ".pte"
                                                            : ".tflite";
        rows.emplace_back(stem + extension, backend);
    }
    rows.emplace_back("placeholder", v2::CUSTOM);
    v2::InferenceConfig config(
        rows,
        {v2::TensorShape({{1, 1, 512}, {1, 1, 1}}, {{1, 1, 512}, {1}}, v2::TFLITE),
         v2::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {1}})},
        v2::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F,
        0,
        false,
        0.F,
        2);
    RecordingProcessor pp(config);
    pp.set_input(1.F, 1, 0);
    anira::Pipeline pipeline;
    pipeline.register_engine(std::make_shared<v2::PassthroughEngine>());
    pipeline.inference(config.model_config());
    pipeline.add(anira::stage::Custom(std::make_shared<v2::LegacyProcessorStage>(pp)));
    const anira_test::Context context;
    PipelineHandler handler(context, pipeline);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(contract_of(config, 512)), ANIRA_OK) << handler.m_err.message;
    const std::vector<anira_plan_info> plans = plans_of(handler.m_handler);
    ASSERT_EQ(plans.size(), rows.size()) << "one plan per entry";
    anira_test::FloatFace face(handler.m_handler);

    bool custom_seen = false;
    for (uint32_t plan = 0; plan < plans.size(); ++plan) {
        SCOPED_TRACE(plan);
        EXPECT_EQ(plans[plan].provider, ANIRA_PROVIDER_CPU);
        const v2::InferenceBackend expected = backend_of(plans[plan]);
        custom_seen = custom_seen || expected == v2::CUSTOM;
        ASSERT_EQ(anira_handler_set_plan(handler.m_handler, plan), ANIRA_OK);
        // The chunks in flight when the plan switched ran on the previous plan: drive until the
        // switch is through, then record from a clean slate.
        for (size_t pass = 0; pass < 2; ++pass) {
            if (pass == 1) { pp.forget(); }
            for (size_t k = 0; k < 4; ++k) {
                std::vector<float> block = anira_test::ramp(k, 512);
                const std::array<float*, 1> channels{block.data()};
                size_t delivered = 0;
                const size_t prev = anira_test::available(handler.m_handler);
                ASSERT_EQ(face.process_inplace(channels.data(), 512, 0, &delivered), ANIRA_OK);
                anira_test::wait_for_block(handler.m_handler, prev);
                ASSERT_FALSE(::testing::Test::HasFatalFailure());
            }
        }
        EXPECT_EQ(pp.m_pre.load(), static_cast<int>(expected));
        EXPECT_EQ(pp.m_before.load(), static_cast<int>(expected));
        EXPECT_EQ(pp.m_after.load(), static_cast<int>(expected));
        EXPECT_EQ(pp.m_post.load(), static_cast<int>(expected));
    }
    EXPECT_TRUE(custom_seen);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
}

namespace {

/// A 2.x processor whose post_process throws on demand: an anira::Error, or anything else.
class ThrowingProcessor : public v2::PrePostProcessor {
public:
    explicit ThrowingProcessor(v2::InferenceConfig& config) : v2::PrePostProcessor(config) {}

    void post_process(std::vector<v2::BufferF>& input,
                      std::vector<v2::RingBuffer>& output,
                      v2::InferenceBackend backend) override {
        if (m_throw_error.load()) { throw anira::Error(ANIRA_ERROR_CONFIG, "post_process"); }
        if (m_throw_other.load()) { throw std::runtime_error("post_process"); }
        v2::PrePostProcessor::post_process(input, output, backend);
    }

    std::atomic<bool> m_throw_error{false};
    std::atomic<bool> m_throw_other{false};
};

}  // namespace

// A throw never crosses the C boundary: the stage's trampoline turns an anira::Error into its
// status and anything else into ANIRA_ERROR_ENGINE, which fails the chunk (zeros at its stream
// position) and lands in anira_handler_rt_error; the handler keeps running (in 2.x the throw
// reached the host).
TEST(AbiCompat, AThrowingProcessorFailsItsChunk) {
    v2::InferenceConfig config({v2::ModelData("placeholder", v2::CUSTOM)},
                               {v2::TensorShape({{1, 1, k_stream}}, {{1, 1, k_stream}})},
                               5.F,
                               0,
                               false,
                               0.F,
                               2);
    ThrowingProcessor pp(config);
    anira::Pipeline pipeline;
    pipeline.register_engine(std::make_shared<v2::PassthroughEngine>());
    pipeline.inference(config.model_config());
    pipeline.add(anira::stage::Custom(std::make_shared<v2::LegacyProcessorStage>(pp)));
    const anira_test::Context context;
    PipelineHandler handler(context, pipeline);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(contract_of(config, k_stream)), ANIRA_OK) << handler.m_err.message;
    anira_test::FloatFace face(handler.m_handler);
    const auto run_blocks = [&](size_t count) {
        for (size_t k = 0; k < count; ++k) {
            std::vector<float> block = anira_test::ramp(k, k_stream);
            const std::array<float*, 1> channels{block.data()};
            size_t delivered = 0;
            const size_t prev = anira_test::available(handler.m_handler);
            EXPECT_EQ(face.process_inplace(channels.data(), k_stream, 0, &delivered), ANIRA_OK);
            anira_test::wait_for_block(handler.m_handler, prev);
        }
    };
    run_blocks(4);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);

    pp.m_throw_other.store(true);
    run_blocks(4);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_ENGINE);
    pp.m_throw_other.store(false);

    anira_handler_reset(handler.m_handler);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK) << "reset clears it";
    pp.m_throw_error.store(true);
    run_blocks(4);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_CONFIG);
    pp.m_throw_error.store(false);
}

// ---- the handler: the differential oracle against the live 2.x handler (suite AbiCompat) ----

namespace {

/// The shim side's context: what anira_test::Context and oracle_core_config() set, so the users
/// of the one core reconcile without a mismatch.
v2::ContextConfig shim_context_config(unsigned threads = 2) {
    v2::ContextConfig config(threads, v2::WaitStrategy::SpinBackoff, v2::LogLevel::Error);
    config.m_log.m_drain = v2::LogDrain::Manual;
    return config;
}

/// Waits until the output ring of slot 0 holds `expected` samples; fails after k_wait_s.
template <class Handler>
void wait_ring(const Handler& handler, size_t expected) {
    const auto start = std::chrono::steady_clock::now();
    while (handler.get_available_samples(0) != expected) {
        if (std::chrono::steady_clock::now() > start + std::chrono::seconds(anira_test::k_wait_s)) {
            FAIL() << "timeout while waiting for " << expected << " available samples (have "
                   << handler.get_available_samples(0) << ")";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

/// Polls a condition for up to four seconds.
bool eventually(const std::function<bool()>& condition) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

/// The call form both sides are driven through.
enum class Form { InPlace, Separate, Multi, PushPop, PushPopMulti };

constexpr std::array<Form, 5> k_forms{Form::InPlace,
                                      Form::Separate,
                                      Form::Multi,
                                      Form::PushPop,
                                      Form::PushPopMulti};

/// One block of a gain-shaped handler (a stream and a Static value in, a stream and a Static
/// value out) through `form`; the same code drives the 2.x class and the shim, whose members
/// have the same signatures. `out` receives the stream, `static_out` the Static output of the
/// multi forms (untouched by the others). Returns the delivered count of slot 0.
template <class Handler>
size_t run_gain_block(Handler& handler,
                      Form form,
                      size_t k,
                      float gain,
                      std::vector<float>& out,
                      float& static_out) {
    const size_t block = anira_test::k_block;
    const std::vector<float> in = anira_test::ramp(k, block);
    out.assign(block, -1.F);
    const size_t prev = handler.get_available_samples(0);
    const std::array<const float*, 1> in_ch{in.data()};
    const std::array<float*, 1> out_ch{out.data()};
    const std::array<const float*, 1> gain_ch{&gain};
    const std::array<float*, 1> static_ch{&static_out};
    const std::array<const float* const*, 2> ins{in_ch.data(), gain_ch.data()};
    const std::array<float* const*, 2> outs{out_ch.data(), static_ch.data()};
    std::array<size_t, 2> num_in{block, 1};
    std::array<size_t, 2> num_out{block, 1};
    size_t delivered = 0;
    switch (form) {
        case Form::InPlace: {
            out = in;
            const std::array<float*, 1> io{out.data()};
            delivered = handler.process(io.data(), block);
            wait_ring(handler, prev);
            break;
        }
        case Form::Separate:
            delivered = handler.process(in_ch.data(), block, out_ch.data(), block);
            wait_ring(handler, prev);
            break;
        case Form::Multi:
            delivered = handler.process(ins.data(), num_in.data(), outs.data(), num_out.data())[0];
            wait_ring(handler, prev);
            break;
        case Form::PushPop:
            handler.push_data(in_ch.data(), block);
            wait_ring(handler, prev + block);
            delivered = handler.pop_data(out_ch.data(), block);
            break;
        case Form::PushPopMulti:
            handler.push_data(ins.data(), num_in.data());
            wait_ring(handler, prev + block);
            delivered = handler.pop_data(outs.data(), num_out.data())[0];
            break;
    }
    return delivered;
}

/// The gain oracle's rows: every engine of oracle_engines() (LiteRT is left out, as in
/// test_Handler: its gain export lists the outputs in another order, which a configuration
/// without tensor names binds wrongly on both sides) and the CUSTOM row.
Arguments oracle_gain_arguments() {
    Arguments args = gain_arguments();
    std::erase_if(args.m_rows, [](const Row& row) { return row.m_backend == v2::LITERT; });
    std::erase_if(args.m_shapes, [](const Shapes& row) { return row.m_backend == v2::LITERT; });
    args.m_warm_up = 0;
    return args;
}

/// The gain shape with the CUSTOM row alone.
Arguments gain_custom_arguments(float blocking_ratio = 0.F) {
    Arguments args;
    args.m_rows = {{v2::CUSTOM, "placeholder"}};
    args.m_shapes = {{{{1, 1, 512}, {1}}, {{1, 1, 512}, {1}}}};
    args.m_in_channels = {1, 1};
    args.m_out_channels = {1, 1};
    args.m_in_sizes = {512, 0};
    args.m_out_sizes = {512, 0};
    args.m_blocking_ratio = blocking_ratio;
    return args;
}

/// One stream in, two out (test_InferenceHandlerApi's two-output model), with the given
/// internal latency per output.
Arguments two_output_arguments(size_t latency = 0) {
    Arguments args;
    args.m_rows = {{v2::CUSTOM, "placeholder"}};
    args.m_shapes = {{{{1, 1, 512}}, {{1, 1, 512}, {1, 1, 512}}}};
    args.m_in_channels = {1};
    args.m_out_channels = {1, 1};
    args.m_in_sizes = {512};
    args.m_out_sizes = {512, 512};
    args.m_latencies = {latency, latency};
    args.m_max_inference_time = 10.F;
    return args;
}

/// A generator: four Static parameters in, a 2048-sample stream out.
Arguments generator_arguments() {
    Arguments args;
    args.m_rows = {{v2::CUSTOM, "placeholder"}};
    args.m_shapes = {{{{1, 4}}, {{1, 2048}}}};
    args.m_in_channels = {1};
    args.m_out_channels = {1};
    args.m_in_sizes = {0};
    args.m_out_sizes = {2048};
    args.m_max_inference_time = 10.F;
    return args;
}

/// An analyser: a 2048-sample stream and a Static parameter in, a Static scalar out.
Arguments analyser_arguments() {
    Arguments args;
    args.m_rows = {{v2::CUSTOM, "placeholder"}};
    args.m_shapes = {{{{1, 2048}, {1, 1}}, {{1, 1}}}};
    args.m_in_channels = {1, 1};
    args.m_out_channels = {1};
    args.m_in_sizes = {2048, 0};
    args.m_out_sizes = {0};
    args.m_max_inference_time = 10.F;
    return args;
}

// ---- kernels: one arithmetic for the 2.x BackendBase and the anira::Engine ------------------

/// The model ends of one inference as float pointers, however they came.
struct Io {
    static constexpr size_t k_max = 4;
    std::array<const float*, k_max> m_in{};
    std::array<size_t, k_max> m_in_size{};
    std::array<float*, k_max> m_out{};
    std::array<size_t, k_max> m_out_size{};
    size_t m_num_in = 0;
    size_t m_num_out = 0;
};

Io io_of(const anira::EngineContext& ctx) noexcept {
    Io io;
    io.m_num_in = std::min(ctx.inputs().size(), Io::k_max);
    io.m_num_out = std::min(ctx.outputs().size(), Io::k_max);
    for (size_t i = 0; i < io.m_num_in; ++i) {
        io.m_in[i] = ctx.inputs()[i].data_f32();
        io.m_in_size[i] = ctx.inputs()[i].num_elements();
    }
    for (size_t i = 0; i < io.m_num_out; ++i) {
        io.m_out[i] = ctx.outputs()[i].data_f32();
        io.m_out_size[i] = ctx.outputs()[i].num_elements();
    }
    return io;
}

Io io_of(std::vector<anira::BufferF>& input, std::vector<anira::BufferF>& output) {
    Io io;
    io.m_num_in = std::min(input.size(), Io::k_max);
    io.m_num_out = std::min(output.size(), Io::k_max);
    for (size_t i = 0; i < io.m_num_in; ++i) {
        io.m_in[i] = input[i].get_read_pointer(0);
        io.m_in_size[i] = input[i].get_num_samples();
    }
    for (size_t i = 0; i < io.m_num_out; ++i) {
        io.m_out[i] = output[i].get_write_pointer(0);
        io.m_out_size[i] = output[i].get_num_samples();
    }
    return io;
}

/// The gain model's arithmetic: the stream times the Static gain, the gain mirrored.
struct GainKernel {
    void operator()(const Io& io) const noexcept {
        const float gain = io.m_in[1][0];
        for (size_t i = 0; i < io.m_out_size[0]; ++i) { io.m_out[0][i] = io.m_in[0][i] * gain; }
        io.m_out[1][0] = gain;
    }
};

/// GainKernel behind a gate both sides share: the inference waits while the gate is closed.
struct GateKernel {
    std::atomic<bool>* m_open = nullptr;
    void operator()(const Io& io) const noexcept {
        while (!m_open->load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
        GainKernel{}(io);
    }
};

/// The generator: every output sample the value of parameter 0.
struct ParamFillKernel {
    void operator()(const Io& io) const noexcept {
        for (size_t i = 0; i < io.m_out_size[0]; ++i) { io.m_out[0][i] = io.m_in[0][0]; }
    }
};

/// The analyser: the mean of the window plus the parameter.
struct MeanPlusParamKernel {
    void operator()(const Io& io) const noexcept {
        double sum = 0.0;
        for (size_t i = 0; i < io.m_in_size[0]; ++i) { sum += static_cast<double>(io.m_in[0][i]); }
        const float mean = io.m_in_size[0] > 0
                               ? static_cast<float>(sum / static_cast<double>(io.m_in_size[0]))
                               : 0.F;
        io.m_out[0][0] = mean + io.m_in[1][0];
    }
};

/// The 2.x side: a BackendBase over a kernel.
template <class Kernel>
class KernelBackend : public anira::BackendBase {
public:
    KernelBackend(anira::InferenceConfig& config, Kernel kernel)
        : anira::BackendBase(config), m_kernel(kernel) {}
    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 std::shared_ptr<anira::SessionElement> /*session*/) override {
        m_kernel(io_of(input, output));
        m_calls.fetch_add(1);
    }

    Kernel m_kernel;
    std::atomic<int> m_calls{0};
};

/// The shim side: an anira::Engine under anira.v2.custom over a kernel, counting its levels.
template <class Kernel>
class KernelEngine : public anira::Engine {
public:
    explicit KernelEngine(Kernel kernel = {})
        : anira::Engine(v2::k_custom_engine_id), m_kernel(kernel) {}

    std::span<const char* const> providers() const noexcept override { return m_providers; }
    std::uint64_t query(const anira::InitInfo& /*info*/) const override { return m_query; }
    void init(const anira::InitInfo& /*info*/) override { m_inits.fetch_add(1); }
    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& /*info*/) override {
        m_loads.fetch_add(1);
        return std::make_unique<Loaded>(*this);
    }
    void release() noexcept override { m_releases.fetch_add(1); }

    Kernel m_kernel;
    std::vector<const char*> m_providers;
    std::uint64_t m_query = ~std::uint64_t{0};
    std::atomic<int> m_inits{0};
    std::atomic<int> m_loads{0};
    std::atomic<int> m_prepares{0};
    std::atomic<int> m_releases{0};
    std::atomic<int> m_calls{0};

private:
    class Prepared : public anira::Engine::Prepared {
    public:
        explicit Prepared(KernelEngine& owner) : m_owner(owner) {}
        anira_status process(anira::EngineContext& ctx) noexcept override {
            m_owner.m_kernel(io_of(ctx));
            m_owner.m_calls.fetch_add(1);
            return ANIRA_OK;
        }

    private:
        KernelEngine& m_owner;
    };
    class Loaded : public anira::Engine::Loaded {
    public:
        explicit Loaded(KernelEngine& owner) : m_owner(owner) {}
        std::unique_ptr<anira::Engine::Prepared> prepare(
            const anira::PrepareInfo& /*info*/) override {
            m_owner.m_prepares.fetch_add(1);
            return std::make_unique<Prepared>(m_owner);
        }

    private:
        KernelEngine& m_owner;
    };
};

/// A 2.x handler and a shim handler over the same arguments, the CUSTOM row on the default
/// pass-through on both sides (the 2.x roundtrip, the PassthroughEngine) and the built-in rows
/// on their engines.
struct Pair {
    explicit Pair(const Arguments& args)
        : m_old(make_old(args))
        , m_old_pp(m_old)
        , m_old_handler(m_old_pp, m_old, oracle_core_config())
        , m_shim(make_shim(args))
        , m_pp(m_shim)
        , m_handler(m_pp, m_shim, shim_context_config()) {}

    void prepare(const anira::HostConfig& old_host, const v2::HostConfig& host) {
        m_old_handler.prepare(old_host);
        m_handler.prepare(host);
    }
    void prepare(size_t block = anira_test::k_block) {
        prepare(
            anira::HostConfig(static_cast<float>(block), static_cast<float>(anira_test::k_rate)),
            v2::HostConfig(static_cast<float>(block), static_cast<float>(anira_test::k_rate)));
    }
    void set_gain(float gain) {
        m_old_pp.set_input(gain, 1, 0);
        m_pp.set_input(gain, 1, 0);
    }

    anira::InferenceConfig m_old;
    anira::PrePostProcessor m_old_pp;
    anira::InferenceHandler m_old_handler;
    v2::InferenceConfig m_shim;
    v2::PrePostProcessor m_pp;
    v2::InferenceHandler m_handler;
};

/// The same over a kernel: a KernelBackend on the 2.x side, a KernelEngine on the shim side.
template <class Kernel>
struct KernelPair {
    KernelPair(const Arguments& args, Kernel kernel)
        : m_old(make_old(args))
        , m_old_pp(m_old)
        , m_backend(m_old, kernel)
        , m_old_handler(m_old_pp, m_old, m_backend, oracle_core_config())
        , m_shim(make_shim(args))
        , m_pp(m_shim)
        , m_engine(kernel)
        , m_handler(m_pp, m_shim, m_engine, shim_context_config()) {}

    void prepare(size_t block = anira_test::k_block) {
        m_old_handler.prepare(
            anira::HostConfig(static_cast<float>(block), static_cast<float>(anira_test::k_rate)));
        m_handler.prepare(
            v2::HostConfig(static_cast<float>(block), static_cast<float>(anira_test::k_rate)));
    }

    anira::InferenceConfig m_old;
    anira::PrePostProcessor m_old_pp;
    KernelBackend<Kernel> m_backend;
    anira::InferenceHandler m_old_handler;
    v2::InferenceConfig m_shim;
    v2::PrePostProcessor m_pp;
    KernelEngine<Kernel> m_engine;
    v2::InferenceHandler m_handler;
};

/// The backends both sides of the gain oracle run: the oracle engines and CUSTOM, as pairs.
std::vector<std::pair<anira::InferenceBackend, v2::InferenceBackend>> oracle_backends() {
    std::vector<std::pair<anira::InferenceBackend, v2::InferenceBackend>> out;
    for (const anira_engine engine : anira_test::oracle_engines()) {
        const v2::InferenceBackend backend =
            v2::to_backend(anira::EngineRef{.kind = engine, .id = {}});
        out.emplace_back(required_old_backend(backend), backend);
    }
    out.emplace_back(anira::InferenceBackend::CUSTOM, v2::CUSTOM);
    return out;
}

class AbiCompatForm : public ::testing::TestWithParam<Form> {};

}  // namespace

// Case 1: every plan, every form, 20 blocks: the counts and the samples equal the 2.x class's.
TEST_P(AbiCompatForm, GainMatchesOnEveryPlan) {
    Pair pair(oracle_gain_arguments());
    pair.prepare();
    pair.set_gain(0.5F);
    size_t k = 1;
    for (const auto& [old_backend_value, backend] : oracle_backends()) {
        SCOPED_TRACE(static_cast<uint32_t>(backend));
        pair.m_old_handler.set_inference_backend(old_backend_value);
        pair.m_handler.set_inference_backend(backend);
        EXPECT_EQ(pair.m_handler.get_inference_backend(), backend);
        for (size_t i = 0; i < 20; ++i, ++k) {
            std::vector<float> c;
            std::vector<float> v;
            float c_static = -1.F;
            float v_static = -1.F;
            const size_t n_c = run_gain_block(pair.m_handler, GetParam(), k, 0.5F, c, c_static);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            const size_t n_v = run_gain_block(pair.m_old_handler, GetParam(), k, 0.5F, v, v_static);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            EXPECT_EQ(n_c, n_v) << "block " << k;
            EXPECT_EQ(n_c, anira_test::k_block) << "block " << k;
            anira_test::expect_same_block(c, v, k);
            EXPECT_EQ(c_static, v_static) << "block " << k;
        }
    }
    EXPECT_EQ(pair.m_handler.rt_error(), ANIRA_OK);
}

INSTANTIATE_TEST_SUITE_P(Forms,
                         AbiCompatForm,
                         ::testing::ValuesIn(k_forms),
                         [](const ::testing::TestParamInfo<Form>& info) {
                             switch (info.param) {
                                 case Form::InPlace: return std::string("InPlace");
                                 case Form::Separate: return std::string("Separate");
                                 case Form::Multi: return std::string("Multi");
                                 case Form::PushPop: return std::string("PushPop");
                                 case Form::PushPopMulti: return std::string("PushPopMulti");
                             }
                             return std::string("Unknown");
                         });

// Case 2: a switch of backend before every block, on both sides: the same backend reads back
// (compared by name) and the same samples come out.
TEST(AbiCompat, SetInferenceBackendSwitchesLikeTheOracle) {
    Pair pair(oracle_gain_arguments());
    pair.prepare();
    pair.set_gain(1.F);
    EXPECT_TRUE(same_backend(pair.m_old_handler.get_inference_backend(),
                             pair.m_handler.get_inference_backend()))
        << "the 2.x initial rule";
    const auto backends = oracle_backends();
    for (size_t k = 1; k <= 24; ++k) {
        const auto& [old_backend_value, backend] = backends[k % backends.size()];
        pair.m_old_handler.set_inference_backend(old_backend_value);
        pair.m_handler.set_inference_backend(backend);
        EXPECT_TRUE(same_backend(pair.m_old_handler.get_inference_backend(),
                                 pair.m_handler.get_inference_backend()))
            << "block " << k;
        std::vector<float> c;
        std::vector<float> v;
        float c_static = -1.F;
        float v_static = -1.F;
        const size_t n_c = run_gain_block(pair.m_handler, Form::InPlace, k, 1.F, c, c_static);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const size_t n_v = run_gain_block(pair.m_old_handler, Form::InPlace, k, 1.F, v, v_static);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(n_c, n_v);
        anira_test::expect_same_block(c, v, k);
    }
}

// Case 5: a custom kernel as a 2.x BackendBase and as an anira::Engine under anira.v2.custom,
// driven through the multi form with the gain changing through the Static count: the same
// streams, the same Static outputs, the same counts.
TEST(AbiCompat, ACustomEngineMatchesACustomBackend) {
    KernelPair<GainKernel> pair(gain_custom_arguments(), GainKernel{});
    pair.prepare();
    EXPECT_EQ(pair.m_handler.get_inference_backend(), v2::CUSTOM);
    for (size_t k = 1; k <= 20; ++k) {
        const float gain = 0.25F * static_cast<float>(k % 5);
        std::vector<float> c;
        std::vector<float> v;
        float c_static = -1.F;
        float v_static = -1.F;
        const size_t n_c = run_gain_block(pair.m_handler, Form::Multi, k, gain, c, c_static);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const size_t n_v = run_gain_block(pair.m_old_handler, Form::Multi, k, gain, v, v_static);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(n_c, n_v);
        anira_test::expect_same_block(c, v, k);
        EXPECT_EQ(c_static, v_static) << "block " << k;
    }
    EXPECT_GT(pair.m_engine.m_calls.load(), 0);
    EXPECT_EQ(pair.m_engine.m_calls.load(), pair.m_backend.m_calls.load());
}

// Case 6: the Static values travel through the processor's atomics on both sides: set with
// set_input, read with get_output after every block, and through the multi form's value count.
TEST(AbiCompat, StaticInputsAndOutputsAgree) {
    KernelPair<GainKernel> pair(gain_custom_arguments(), GainKernel{});
    pair.prepare();
    EXPECT_EQ(pair.m_pp.get_output(1, 0), 0.F) << "zero until an inference is collected";
    for (size_t k = 1; k <= 12; ++k) {
        const auto gain = static_cast<float>(k);
        pair.m_pp.set_input(gain, 1, 0);
        pair.m_old_pp.set_input(gain, 1, 0);
        std::vector<float> c;
        std::vector<float> v;
        float unused = 0.F;
        run_gain_block(pair.m_handler, Form::InPlace, k, gain, c, unused);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        run_gain_block(pair.m_old_handler, Form::InPlace, k, gain, v, unused);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        anira_test::expect_same_block(c, v, k);
        EXPECT_EQ(pair.m_pp.get_output(1, 0), pair.m_old_pp.get_output(1, 0)) << "block " << k;
    }
    EXPECT_NE(pair.m_pp.get_output(1, 0), 0.F);
}

// Case 7: a starved block on both sides (the gate both kernels share held closed): the stream
// count is 0, the Static request is zeroed and reports 0, the processor's atomics keep the
// last collected value.
TEST(AbiCompat, AMissedBlockZeroesTheStaticRequest) {
    std::atomic<bool> open{true};
    KernelPair<GateKernel> pair(gain_custom_arguments(), GateKernel{.m_open = &open});
    struct OpenAtExit {
        std::atomic<bool>& m_open;
        ~OpenAtExit() { m_open.store(true); }
    };
    const OpenAtExit open_at_exit{open};
    pair.prepare();
    const auto starve = [&open](auto& handler, auto& pp, const char* side) {
        SCOPED_TRACE(side);
        open.store(true);
        for (size_t k = 1; k <= 4; ++k) {
            std::vector<float> out;
            float static_out = -1.F;
            run_gain_block(handler, Form::Multi, k, 2.F, out, static_out);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
        }
        const float held = pp.get_output(1, 0);
        EXPECT_EQ(held, 2.F);
        open.store(false);
        bool missed = false;
        for (size_t k = 5; k <= 16 && !missed; ++k) {
            const std::vector<float> in = anira_test::ramp(k);
            std::vector<float> out(anira_test::k_block, -1.F);
            const float gain = 3.F;
            float static_out = -1.F;
            const std::array<const float*, 1> in_ch{in.data()};
            const std::array<float*, 1> out_ch{out.data()};
            const std::array<const float*, 1> gain_ch{&gain};
            const std::array<float*, 1> static_ch{&static_out};
            const std::array<const float* const*, 2> ins{in_ch.data(), gain_ch.data()};
            const std::array<float* const*, 2> outs{out_ch.data(), static_ch.data()};
            std::array<size_t, 2> num_in{anira_test::k_block, 1};
            std::array<size_t, 2> num_out{anira_test::k_block, 1};
            handler.process(ins.data(), num_in.data(), outs.data(), num_out.data());
            if (num_out[0] == 0) {
                missed = true;
                EXPECT_EQ(num_out[1], 0U) << "a missed block reports 0 for the Static slot";
                EXPECT_EQ(static_out, 0.F) << "the Static request is zeroed";
                EXPECT_EQ(pp.get_output(1, 0), held) << "the atomics keep the last value";
            }
        }
        EXPECT_TRUE(missed) << "a closed gate starves a block within twelve";
        open.store(true);
    };
    starve(pair.m_old_handler, pair.m_old_pp, "2.x");
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    starve(pair.m_handler, pair.m_pp, "shim");
}

// Case 8: the samples waiting in the output ring agree after prepare, after every block and
// after reset.
TEST(AbiCompat, AvailableSamplesAgree) {
    Pair pair(gain_custom_arguments());
    pair.prepare();
    pair.set_gain(1.F);
    EXPECT_EQ(pair.m_handler.get_available_samples(0), pair.m_old_handler.get_available_samples(0));
    EXPECT_EQ(pair.m_handler.get_available_samples(0), pair.m_handler.get_latency(0));
    for (size_t k = 1; k <= 6; ++k) {
        std::vector<float> out;
        float unused = 0.F;
        run_gain_block(pair.m_handler, Form::InPlace, k, 1.F, out, unused);
        run_gain_block(pair.m_old_handler, Form::InPlace, k, 1.F, out, unused);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(pair.m_handler.get_available_samples(0),
                  pair.m_old_handler.get_available_samples(0))
            << "block " << k;
    }
    pair.m_handler.reset();
    pair.m_old_handler.reset();
    EXPECT_EQ(pair.m_handler.get_available_samples(0), pair.m_old_handler.get_available_samples(0));
    EXPECT_EQ(pair.m_handler.get_available_samples(1), 0U) << "a Static output has no ring";
    EXPECT_EQ(pair.m_handler.get_available_samples(9), 0U) << "out of range";
}

// Case 9: the latency figures agree under blocking ratio 0 and 1: per index and as the vector,
// index-aligned, 0 for the Static output.
TEST(AbiCompat, LatencyFiguresAgree) {
    for (const float ratio : {0.F, 1.F}) {
        SCOPED_TRACE(ratio);
        Pair pair(gain_custom_arguments(ratio));
        pair.prepare();
        const std::vector<unsigned> c = pair.m_handler.get_latency_vector();
        const std::vector<unsigned> v = pair.m_old_handler.get_latency_vector();
        EXPECT_EQ(c, v);
        ASSERT_EQ(c.size(), 2U);
        EXPECT_EQ(pair.m_handler.get_latency(0), c[0]);
        EXPECT_EQ(pair.m_handler.get_latency(1), 0U);
        EXPECT_EQ(c[1], 0U);
        EXPECT_EQ(pair.m_handler.get_latency(7), 0U) << "out of range";
    }
}

// Case 10: reset restarts the stream on both sides alike.
TEST(AbiCompat, ResetReSeedsTheStreamOnBothSides) {
    Pair pair(gain_custom_arguments());
    pair.prepare();
    pair.set_gain(1.F);
    for (size_t round = 0; round < 2; ++round) {
        for (size_t k = 1; k <= 5; ++k) {
            std::vector<float> c;
            std::vector<float> v;
            float unused = 0.F;
            run_gain_block(pair.m_handler, Form::Separate, k, 1.F, c, unused);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            run_gain_block(pair.m_old_handler, Form::Separate, k, 1.F, v, unused);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            anira_test::expect_same_block(c, v, (round * 10) + k);
        }
        pair.m_handler.reset();
        pair.m_old_handler.reset();
    }
    EXPECT_EQ(pair.m_handler.rt_error(), ANIRA_OK);
}

// Case 11: set_non_realtime makes the generator deterministic on both sides: 40 blocks, no
// wait between them, equal samples.
TEST(AbiCompat, NonRealtimeIsDeterministic) {
    KernelPair<ParamFillKernel> pair(generator_arguments(), ParamFillKernel{});
    pair.prepare();
    pair.m_handler.set_non_realtime(true);
    pair.m_old_handler.set_non_realtime(true);
    for (size_t k = 0; k < 40; ++k) {
        const std::array<float, 4> params{1.F + static_cast<float>(k), 0.F, 0.F, 0.F};
        const auto pull = [&params](auto& handler) {
            std::vector<float> out(anira_test::k_block, -1.F);
            const std::array<const float*, 1> param_ch{params.data()};
            const std::array<float*, 1> out_ch{out.data()};
            const std::array<const float* const*, 1> ins{param_ch.data()};
            const std::array<float* const*, 1> outs{out_ch.data()};
            std::array<size_t, 1> num_in{4};
            std::array<size_t, 1> num_out{anira_test::k_block};
            const size_t delivered =
                handler.process(ins.data(), num_in.data(), outs.data(), num_out.data())[0];
            EXPECT_EQ(delivered, anira_test::k_block);
            return out;
        };
        const std::vector<float> c = pull(pair.m_handler);
        const std::vector<float> v = pull(pair.m_old_handler);
        anira_test::expect_same_block(c, v, k);
    }
    EXPECT_EQ(pair.m_engine.m_calls.load(), pair.m_backend.m_calls.load());
    EXPECT_EQ(pair.m_handler.rt_error(), ANIRA_OK);
}

// Case 12: the deadline pop returns by its deadline, under ratio 0 (the shim polls, 2.x neither
// waited nor collected) and ratio 1 (a semaphore on both sides, the same count).
TEST(AbiCompat, PopDataWithDeadlineReturnsByTheDeadline) {
    for (const float ratio : {0.F, 1.F}) {
        SCOPED_TRACE(ratio);
        Pair pair(gain_custom_arguments(ratio));
        pair.prepare();
        std::vector<float> out(anira_test::k_block, -1.F);
        const std::array<float*, 1> out_ch{out.data()};
        const auto started = std::chrono::steady_clock::now();
        const size_t c = pair.m_handler.pop_data(out_ch.data(),
                                                 anira_test::k_block,
                                                 started + std::chrono::milliseconds(50));
        EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(2));
        EXPECT_LE(c, anira_test::k_block);
        if (ratio > 0.F) {
            const size_t v = pair.m_old_handler.pop_data(
                out_ch.data(),
                anira_test::k_block,
                std::chrono::steady_clock::now() + std::chrono::milliseconds(50));
            EXPECT_EQ(c, v);
        }
    }
}

// Case 13: a push-only analyser updates its Static output: the processor's atomics hold the
// latest collected value, polled through push_data(nullptr, 0), which collects.
TEST(AbiCompat, PushOnlyAnalyserUpdatesGetOutput) {
    KernelPair<MeanPlusParamKernel> pair(analyser_arguments(), MeanPlusParamKernel{});
    pair.prepare();
    pair.m_pp.set_input(2.F, 1, 0);
    pair.m_old_pp.set_input(2.F, 1, 0);
    const std::vector<float> audio(anira_test::k_block, 3.F);
    const std::array<const float*, 1> audio_ch{audio.data()};
    for (size_t block = 0; block < 2048 / anira_test::k_block; ++block) {
        pair.m_handler.push_data(audio_ch.data(), anira_test::k_block, 0);
        pair.m_old_handler.push_data(audio_ch.data(), anira_test::k_block, 0);
    }
    EXPECT_TRUE(eventually([&pair] {
        pair.m_handler.push_data(nullptr, 0, 0);
        return pair.m_pp.get_output(0, 0) == 5.F;
    })) << "mean 3 plus parameter 2";
    EXPECT_TRUE(eventually([&pair] {
        pair.m_old_handler.push_data(nullptr, 0, 0);
        return pair.m_old_pp.get_output(0, 0) == 5.F;
    }));
    EXPECT_EQ(pair.m_engine.m_calls.load(), 1);
    EXPECT_EQ(pair.m_handler.rt_error(), ANIRA_OK);
}

// Case 16: the two custom-latency prepare forms: the declared figures replace the computed ones
// on both sides, the receive ring is primed with them (zeros first), a request below the
// model's internal latency is raised to it (with the Warning on the shim's side), an index out
// of range throws on both (std::invalid_argument there, anira::Error here), a non-streamable
// index throws on the shim, and a plain prepare returns to the computed figures.
TEST(AbiCompat, CustomLatencyMatchesTheOracle) {
    Pair pair(two_output_arguments());
    const anira::HostConfig old_host(512.F, static_cast<float>(anira_test::k_rate));
    const v2::HostConfig host(512.F, static_cast<float>(anira_test::k_rate));
    const auto run_two_outputs = [&pair](size_t blocks) {
        for (size_t k = 1; k <= blocks; ++k) {
            const std::vector<float> in = anira_test::ramp(k);
            const auto drive =
                [&in](auto& handler, std::vector<float>& out0, std::vector<float>& out1) {
                    out0.assign(anira_test::k_block, -1.F);
                    out1.assign(anira_test::k_block, -1.F);
                    const std::array<const float*, 1> in_ch{in.data()};
                    const std::array<float*, 1> out0_ch{out0.data()};
                    const std::array<float*, 1> out1_ch{out1.data()};
                    const std::array<const float* const*, 1> ins{in_ch.data()};
                    const std::array<float* const*, 2> outs{out0_ch.data(), out1_ch.data()};
                    std::array<size_t, 1> num_in{anira_test::k_block};
                    std::array<size_t, 2> num_out{anira_test::k_block, anira_test::k_block};
                    const size_t prev = handler.get_available_samples(0);
                    handler.process(ins.data(), num_in.data(), outs.data(), num_out.data());
                    wait_ring(handler, prev);
                };
            std::vector<float> c0;
            std::vector<float> c1;
            std::vector<float> v0;
            std::vector<float> v1;
            drive(pair.m_handler, c0, c1);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            drive(pair.m_old_handler, v0, v1);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            anira_test::expect_same_block(c0, v0, k);
            anira_test::expect_same_block(c1, v1, k);
            if (k <= 2) { anira_test::expect_all(c1, 0.F, "the primed latency of output 1"); }
        }
    };

    pair.m_old_handler.prepare(old_host, 1024, 1);
    pair.m_handler.prepare(host, 1024, 1);
    EXPECT_EQ(pair.m_handler.get_latency(1), 1024U);
    EXPECT_EQ(pair.m_handler.get_latency_vector(), pair.m_old_handler.get_latency_vector());
    run_two_outputs(6);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    pair.m_old_handler.prepare(old_host, std::vector<unsigned>{512, 1024});
    pair.m_handler.prepare(host, std::vector<unsigned>{512, 1024});
    EXPECT_EQ(pair.m_handler.get_latency(0), 512U);
    EXPECT_EQ(pair.m_handler.get_latency(1), 1024U);
    EXPECT_EQ(pair.m_handler.get_latency_vector(), pair.m_old_handler.get_latency_vector());
    run_two_outputs(6);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    pair.m_old_handler.prepare(old_host);
    pair.m_handler.prepare(host);
    EXPECT_EQ(pair.m_handler.get_latency_vector(), pair.m_old_handler.get_latency_vector())
        << "a plain prepare returns to the computed figures";
    EXPECT_NE(pair.m_handler.get_latency(1), 1024U);

    EXPECT_THROW(pair.m_old_handler.prepare(old_host, 128U, 5), std::invalid_argument);
    EXPECT_EQ(status_of([&] { pair.m_handler.prepare(host, 128U, 5); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(status_of([&] { pair.m_handler.prepare(host, std::vector<unsigned>{1, 2, 3}); }),
              ANIRA_ERROR_INVALID_ARGUMENT);

    // Below the model's internal latency: raised to it on both sides.
    Pair floor_pair(two_output_arguments(256));
    // After the contexts exist: each reconciles the level of this copy of anira at its create.
    const WarningLevel level;
    anira_test::RecordCollector collector;
    floor_pair.m_old_handler.prepare(old_host, 100U, 0);
    floor_pair.m_handler.prepare(host, 100U, 0);
    EXPECT_EQ(floor_pair.m_handler.get_latency(0), 256U);
    EXPECT_EQ(floor_pair.m_handler.get_latency_vector(),
              floor_pair.m_old_handler.get_latency_vector());
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.has("is below the internal model latency 256; clamping", "native"));
#endif

    // A non-streamable output has no stream latency.
    Pair gain(gain_custom_arguments());
    EXPECT_EQ(status_of([&] { gain.m_handler.prepare(host, 100U, 1); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(status_of([&] { gain.m_handler.prepare(host, std::vector<unsigned>{0, 5}); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    gain.m_handler.prepare(host, std::vector<unsigned>{600, 0});
    EXPECT_EQ(gain.m_handler.get_latency(0), 600U);
}

// Case 17: a block size of none and a rate of 0 are refused; a fractional block size is not
// (2.x took it as a float, the shim passes it through to the double Hard geometry).
TEST(AbiCompat, ANonPositiveHostConfigIsRefused) {
    const Arguments args = gain_custom_arguments();
    v2::InferenceConfig config = make_shim(args);
    v2::PrePostProcessor pp(config);
    v2::InferenceHandler handler(pp, config, shim_context_config());
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(0.F, 48000.F)); }),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(-512.F, 48000.F)); }),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(512.F, 0.F)); }), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(512.5F, 48000.F)); }), ANIRA_OK);
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(512.F, 48000.F)); }), ANIRA_OK);
}

/// The decoder shape of a control-rate model: one latent sample in, k_upsample audio samples
/// out, each the latent value (the RAVE decoder of the JUCE example, in miniature).
constexpr size_t k_upsample = 8;
struct UpsampleKernel {
    void operator()(const Io& io) const noexcept {
        for (size_t i = 0; i < io.m_out_size[0]; ++i) {
            io.m_out[0][i] = io.m_in[0][i / k_upsample];
        }
    }
};

Arguments upsample_arguments() {
    Arguments args;
    args.m_rows = {{v2::CUSTOM, "placeholder"}};
    args.m_shapes = {{{{1, 1, 1}}, {{1, 1, static_cast<int64_t>(k_upsample)}}}};
    args.m_in_channels = {1};
    args.m_out_channels = {1};
    args.m_in_sizes = {1};
    args.m_out_sizes = {k_upsample};
    return args;
}

/// The samples of a stream whose block is `block` that the host moves in call k (1-based): the
/// whole ones accumulated since call k - 1, floor(k b) - floor((k - 1) b) -- the accounting of
/// LatencyCalculator. `block` is a double: the host's own count, not the float the 2.x
/// HostConfig rounds it to.
size_t whole_samples_of_call(double block, size_t k) {
    const auto upto = [block](size_t calls) {
        return static_cast<size_t>(std::floor(static_cast<double>(calls) * block + 1e-9));
    };
    return upto(k) - upto(k - 1);
}

// Case 31: a fractional host block on a control-rate reference (the latent input of a
// decoder, prepared with samplesPerBlock / 2048.f in the JUCE example): the 2.x class and the
// shim, prepared with the same HostConfig, agree on the latency, the available samples and
// every output sample over 96 calls (24 and 32 latent samples), the host moving a latent
// sample only once a whole one has accumulated (0 or 1 per call) and popping its audio block
// of 2 (0.25 x 8) or 8/3 (1/3 x 8: 2 or 3) samples per call.
TEST(AbiCompat, AFractionalBlockMatchesTheOracle) {
    for (const double block : {0.25, 1.0 / 3.0}) {
        SCOPED_TRACE(block);
        KernelPair<UpsampleKernel> pair(upsample_arguments(), UpsampleKernel{});
        // 100 latent samples per second: 5 ms of inference is half a hop, the pool keeps up.
        pair.m_old_handler.prepare(anira::HostConfig(static_cast<float>(block), 100.F));
        pair.m_handler.prepare(v2::HostConfig(static_cast<float>(block), 100.F));
        const std::vector<unsigned> latency = pair.m_handler.get_latency_vector();
        ASSERT_EQ(latency, pair.m_old_handler.get_latency_vector());
        ASSERT_EQ(latency.size(), 1U);
        EXPECT_GT(latency[0], 0U);
        ASSERT_EQ(pair.m_handler.get_available_samples(0),
                  pair.m_old_handler.get_available_samples(0));

        size_t pushed = 0;
        size_t popped = 0;
        size_t latent_out = 0;
        for (size_t k = 1; k <= 96; ++k) {
            const size_t num_in = whole_samples_of_call(block, k);
            const size_t num_out = whole_samples_of_call(block * k_upsample, k);
            ASSERT_LE(num_in, 1U);
            const float latent = static_cast<float>(pushed + 1);
            const std::array<const float*, 1> in_ch{&latent};
            std::vector<float> out_new(num_out + 1, -1.F);
            std::vector<float> out_old(num_out + 1, -1.F);
            const std::array<float*, 1> new_ch{out_new.data()};
            const std::array<float*, 1> old_ch{out_old.data()};
            pushed += num_in;
            popped += num_out;
            // Settled: every inference the call submitted has landed in the output ring.
            const size_t settled = latency[0] + k_upsample * pushed - popped;
            EXPECT_EQ(pair.m_handler.process(in_ch.data(), num_in, new_ch.data(), num_out), num_out)
                << "call " << k;
            EXPECT_EQ(pair.m_old_handler.process(in_ch.data(), num_in, old_ch.data(), num_out),
                      num_out)
                << "call " << k;
            wait_ring(pair.m_handler, settled);
            wait_ring(pair.m_old_handler, settled);
            ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "call " << k;
            EXPECT_EQ(out_new, out_old) << "call " << k;
            EXPECT_EQ(out_new[num_out], -1.F) << "nothing past the block";
            latent_out +=
                static_cast<size_t>(std::count_if(out_new.begin(),
                                                  out_new.begin() + static_cast<ptrdiff_t>(num_out),
                                                  [](float value) { return value > 0.F; }));
            EXPECT_EQ(pair.m_handler.get_available_samples(0),
                      pair.m_old_handler.get_available_samples(0))
                << "call " << k;
        }
        EXPECT_EQ(pushed, block == 0.25 ? 24U : 32U);
        // Every latent value was inferred, and the stream carried them out behind the latency.
        EXPECT_EQ(pair.m_engine.m_calls.load(), static_cast<int>(pushed));
        EXPECT_EQ(latent_out, popped - latency[0]);
    }
}

// Case 18: a reference tensor that is out of range or not streamable is refused.
TEST(AbiCompat, AnUnstreamableReferenceIsRefused) {
    v2::InferenceConfig config = make_shim(gain_custom_arguments());
    v2::PrePostProcessor pp(config);
    v2::InferenceHandler handler(pp, config, shim_context_config());
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(512.F, 48000.F, false, 1, true)); }),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "input 1 is the Static gain";
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(512.F, 48000.F, false, 1, false)); }),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "output 1 is Static";
    EXPECT_EQ(status_of([&] { handler.prepare(v2::HostConfig(512.F, 48000.F, false, 5, true)); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_THROW(handler.prepare(v2::HostConfig(512.F, 48000.F, false, 5, true)),
                 std::runtime_error)
        << "a catch of the standard base still sees it";
}

// Case 19: another reference tensor rebuilds the C handler on the private model config; the
// caller's config is untouched and the stream still matches 2.x prepared the same way.
TEST(AbiCompat, ReAnchoringRebuildsTheHandler) {
    Pair pair(gain_custom_arguments());
    pair.prepare();
    const anira_handler* first = pair.m_handler.native();
    pair.prepare(anira::HostConfig(512.F, static_cast<float>(anira_test::k_rate), false, 0, false),
                 v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate), false, 0, false));
    EXPECT_NE(pair.m_handler.native(), first) << "a new C handler";
    EXPECT_TRUE(pair.m_shim.model_config().anchor().empty()) << "the caller's config is untouched";
    pair.set_gain(1.F);
    for (size_t k = 1; k <= 8; ++k) {
        std::vector<float> c;
        std::vector<float> v;
        float unused = 0.F;
        run_gain_block(pair.m_handler, Form::InPlace, k, 1.F, c, unused);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        run_gain_block(pair.m_old_handler, Form::InPlace, k, 1.F, v, unused);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        anira_test::expect_same_block(c, v, k);
    }
}

// Case 20: a backend of the build without a row keeps the selection and logs once through the
// real-time queue; rt_error() is untouched.
TEST(AbiCompat, SetBackendWithoutAPlanKeepsTheSelection) {
    const Arguments args = oracle_gain_arguments();
    // An engine of the build without a row where there is one, else any engine without a row
    // (an engine-free build).
    std::optional<v2::InferenceBackend> missing;
    for (const bool of_the_build : {true, false}) {
        for (const v2::InferenceBackend backend : k_backends) {
            const bool has_row = std::ranges::any_of(args.m_rows, [backend](const Row& row) {
                return row.m_backend == backend;
            });
            if (backend == v2::CUSTOM || has_row || (of_the_build && !v2::is_available(backend))) {
                continue;
            }
            missing = backend;
            break;
        }
        if (missing.has_value()) { break; }
    }
    if (!missing.has_value()) { FAIL() << "the gain rows leave LiteRT out at least"; }
    v2::InferenceConfig config = make_shim(args);
    v2::PrePostProcessor pp(config);
    v2::InferenceHandler handler(pp, config, shim_context_config());
    handler.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
    anira_drain_log();
    anira_test::RecordCollector collector;
    const v2::InferenceBackend before = handler.get_inference_backend();
    handler.set_inference_backend(*missing);
    EXPECT_EQ(handler.get_inference_backend(), before) << "the selection is unchanged";
    EXPECT_EQ(handler.rt_error(), ANIRA_OK);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.has("set_inference_backend: no plan runs on this backend", "rt"));
#endif
}

// Case 21: without an inference loop a waiting form runs the nonblocking stem, returns its
// count and leaves ANIRA_ERROR_INVALID_STATE in rt_error(); nothing is thrown.
TEST(AbiCompat, NoLoopReturnsTheStemCount) {
    const Arguments args = generator_arguments();
    v2::InferenceConfig config = make_shim(args);
    v2::PrePostProcessor pp(config);
    KernelEngine<ParamFillKernel> engine;
    v2::InferenceHandler handler(pp, config, engine, shim_context_config(0));
    handler.prepare(v2::HostConfig(2048.F, static_cast<float>(anira_test::k_rate)));
    EXPECT_EQ(v2::InferenceHandler::get_num_inference_threads(), 0U);
    handler.set_non_realtime(true);
    const size_t priming = handler.get_latency(0) > 0 ? 2048 : 0;
    std::vector<float> out(2048, -1.F);
    const std::array<float, 4> params{1.F, 0.F, 0.F, 0.F};
    const std::array<const float*, 1> param_ch{params.data()};
    const std::array<float*, 1> out_ch{out.data()};
    const std::array<const float* const*, 1> ins{param_ch.data()};
    const std::array<float* const*, 1> outs{out_ch.data()};
    std::array<size_t, 1> num_in{4};
    std::array<size_t, 1> num_out{2048};
    const auto started = std::chrono::steady_clock::now();
    const size_t delivered =
        handler.process(ins.data(), num_in.data(), outs.data(), num_out.data())[0];
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(500))
        << "refused, not waited for";
    EXPECT_EQ(delivered, priming) << "the stem's count";
    EXPECT_EQ(handler.rt_error(), ANIRA_ERROR_INVALID_STATE);
}

// Case 22: nothing loads at construction; a model that does not load fails prepare.
TEST(AbiCompat, AModelThatDoesNotLoadFailsAtPrepare) {
    const std::vector<anira_engine> engines = anira_test::oracle_engines();
    if (engines.empty()) { GTEST_SKIP() << "no built-in engine in this build"; }
    const v2::InferenceBackend backend =
        v2::to_backend(anira::EngineRef{.kind = engines.front(), .id = {}});
    Arguments args = gain_custom_arguments();
    args.m_rows = {{backend, "/no/such/anira/model.file"}};
    v2::InferenceConfig config = make_shim(args);
    v2::PrePostProcessor pp(config);
    std::optional<v2::InferenceHandler> handler;
    ASSERT_NO_THROW(handler.emplace(pp, config, shim_context_config()));
    if (!handler.has_value()) { FAIL() << "the constructor threw"; }
    const anira_status status =
        status_of([&] { handler->prepare(v2::HostConfig(512.F, 48000.F)); });
    EXPECT_TRUE(status == ANIRA_ERROR_NO_SUCH_FILE || status == ANIRA_ERROR_MODEL_LOAD)
        << anira_status_string(status);
    // Unprepared: every real-time form returns 0 and records ANIRA_ERROR_NOT_PREPARED.
    std::vector<float> block(512, 0.F);
    const std::array<float*, 1> channels{block.data()};
    EXPECT_EQ(handler->process(channels.data(), 512), 0U);
    EXPECT_EQ(handler->rt_error(), ANIRA_ERROR_NOT_PREPARED);
}

namespace {

/// Case 23's routes: every form that never waits on a handler without a blocking ratio, with
/// the non-realtime flag off, and the members that never wait, called from a nonblocking
/// function. The forms carry no attribute (a waiting route may reach them), so the
/// compile-time analysis is silenced here; under ANIRA_WITH_RTSAN the runtime check stands: an
/// allocation, a lock or a syscall inside this extent aborts the test.
struct Buffers {
    std::vector<float> m_in = std::vector<float>(anira_test::k_block, 0.25F);
    std::vector<float> m_out = std::vector<float>(anira_test::k_block, 0.F);
    float m_gain = 1.F;
    float m_static = 0.F;
};

#if defined(__clang__) && __clang_major__ >= 20
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfunction-effects"
#endif
size_t drive_nonblocking(v2::InferenceHandler& handler,
                         v2::PrePostProcessor& pp,
                         Buffers& buffers) noexcept ANIRA_NONBLOCKING {
    size_t delivered = 0;
    const std::array<const float*, 1> in_ch{buffers.m_in.data()};
    const std::array<float*, 1> out_ch{buffers.m_out.data()};
    const std::array<const float*, 1> gain_ch{&buffers.m_gain};
    const std::array<float*, 1> static_ch{&buffers.m_static};
    const std::array<const float* const*, 2> ins{in_ch.data(), gain_ch.data()};
    const std::array<float* const*, 2> outs{out_ch.data(), static_ch.data()};
    std::array<size_t, 2> num_in{anira_test::k_block, 1};
    std::array<size_t, 2> num_out{anira_test::k_block, 1};
    delivered += handler.process(out_ch.data(), anira_test::k_block);
    delivered +=
        handler.process(in_ch.data(), anira_test::k_block, out_ch.data(), anira_test::k_block);
    delivered += handler.process(ins.data(), num_in.data(), outs.data(), num_out.data())[0];
    handler.push_data(in_ch.data(), anira_test::k_block);
    handler.push_data(ins.data(), num_in.data());
    delivered += handler.pop_data(out_ch.data(), anira_test::k_block);
    delivered += handler.pop_data(outs.data(), num_out.data())[0];
    delivered += handler.get_available_samples(0);
    delivered += handler.get_latency(0);
    handler.set_inference_backend(v2::CUSTOM);
    delivered += handler.get_inference_backend() == v2::CUSTOM ? 1 : 0;
    handler.reset();
    delivered += handler.rt_error() == ANIRA_OK ? 1 : 0;
    pp.set_input(2.F, 1, 0);
    delivered += pp.get_output(1, 0) >= 0.F ? 1 : 0;
    return delivered;
}
#if defined(__clang__) && __clang_major__ >= 20
#pragma clang diagnostic pop
#endif

}  // namespace

// Case 23: the routes that never wait run inside a nonblocking extent (the RTSan proof on the
// RTSan leg; elsewhere they run and are checked for their results).
TEST(AbiCompat, NonblockingRoutesUnderRtsan) {
    v2::InferenceConfig config = make_shim(gain_custom_arguments());
    v2::PrePostProcessor pp(config);
    v2::InferenceHandler handler(pp, config, shim_context_config());
    handler.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
    Buffers buffers;
    for (size_t round = 0; round < 4; ++round) {
        EXPECT_GT(drive_nonblocking(handler, pp, buffers), 0U);
    }
    EXPECT_EQ(handler.rt_error(), ANIRA_OK);
}

// Case 24: the 2.x statics forward to the core entries.
TEST(AbiCompat, ContextStaticsForward) {
    {
        v2::InferenceConfig config = make_shim(gain_custom_arguments());
        v2::PrePostProcessor pp(config);
        v2::InferenceHandler handler(pp, config, shim_context_config());
        handler.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
        EXPECT_TRUE(v2::Context::has_core());
        EXPECT_EQ(v2::Context::get_num_inference_threads(), 2U);
        EXPECT_EQ(v2::InferenceHandler::get_num_inference_threads(), 2U);
        EXPECT_EQ(v2::Context::shutdown(), ANIRA_ERROR_INVALID_STATE)
            << "refused while a handler lives";
        EXPECT_FALSE(v2::Context::release_core_if_idle());
        static_cast<void>(v2::Context::drain_log());
        static_cast<void>(handler.drain_log());
    }
    EXPECT_EQ(v2::Context::shutdown(), ANIRA_OK);
}

// Case 25: the caller's engine outlives the handlers: two living handlers on one Engine& with
// equal configurations share one C engine (one init, one load, two prepares); once both are
// gone its release runs once and the object lives on; a third handler gets a fresh C engine and
// a second init.
TEST(AbiCompat, TheEngineOutlivesTheHandler) {
    KernelEngine<GainKernel> engine;
    const Arguments args = gain_custom_arguments();
    {
        v2::InferenceConfig first_config = make_shim(args);
        v2::InferenceConfig second_config = make_shim(args);
        v2::PrePostProcessor first_pp(first_config);
        v2::PrePostProcessor second_pp(second_config);
        v2::InferenceHandler first(first_pp, first_config, engine, shim_context_config());
        v2::InferenceHandler second(second_pp, second_config, engine, shim_context_config());
        first.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
        second.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
        EXPECT_EQ(engine.m_inits.load(), 1);
        EXPECT_EQ(engine.m_loads.load(), 1) << "equal configurations load once";
        EXPECT_EQ(engine.m_prepares.load(), 2);
        EXPECT_EQ(engine.m_releases.load(), 0);
    }
    EXPECT_EQ(engine.m_releases.load(), 1);
    {
        v2::InferenceConfig config = make_shim(args);
        v2::PrePostProcessor pp(config);
        v2::InferenceHandler third(pp, config, engine, shim_context_config());
        third.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
        EXPECT_EQ(engine.m_inits.load(), 2) << "a fresh C engine";
    }
    EXPECT_EQ(engine.m_releases.load(), 2);

    // An engine of another id, and a configuration without a CUSTOM row, are refused.
    class Other : public anira::Engine {
    public:
        Other() : anira::Engine("org.example.other") {}
        std::unique_ptr<anira::Engine::Loaded> load(
            const anira::EngineLoadInfo& /*info*/) override {
            return nullptr;
        }
    };
    Other other;
    v2::InferenceConfig config = make_shim(args);
    v2::PrePostProcessor pp(config);
    EXPECT_EQ(status_of([&] { const v2::InferenceHandler refused(pp, config, other); }),
              ANIRA_ERROR_INVALID_ARGUMENT);
    Arguments built_in = oracle_gain_arguments();
    std::erase_if(built_in.m_rows, [](const Row& row) { return row.m_backend == v2::CUSTOM; });
    if (!built_in.m_rows.empty()) {
        v2::InferenceConfig no_custom = make_shim(built_in);
        v2::PrePostProcessor no_custom_pp(no_custom);
        EXPECT_EQ(
            status_of([&] { const v2::InferenceHandler refused(no_custom_pp, no_custom, engine); }),
            ANIRA_ERROR_CONFIG);
    }
}

// Case 28: a prepare with the settings of the last one resets the stream and loads nothing;
// another geometry loads again; another reference tensor rebuilds the handler and loads again.
TEST(AbiCompat, AnUnchangedPrepareResetsWithoutReloading) {
    KernelEngine<GainKernel> engine;
    v2::InferenceConfig config = make_shim(gain_custom_arguments());
    v2::PrePostProcessor pp(config);
    v2::InferenceHandler handler(pp, config, engine, shim_context_config());
    const v2::HostConfig host(512.F, static_cast<float>(anira_test::k_rate));
    handler.prepare(host);
    EXPECT_EQ(engine.m_loads.load(), 1);
    pp.set_input(1.F, 1, 0);
    for (size_t k = 1; k <= 3; ++k) {
        std::vector<float> out;
        float unused = 0.F;
        run_gain_block(handler, Form::InPlace, k, 1.F, out, unused);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
    const anira_handler* native = handler.native();
    handler.prepare(host);
    EXPECT_EQ(engine.m_loads.load(), 1) << "unchanged: nothing loads";
    EXPECT_EQ(handler.native(), native);
    EXPECT_EQ(handler.get_available_samples(0), handler.get_latency(0)) << "the stream restarts";

    handler.prepare(v2::HostConfig(256.F, static_cast<float>(anira_test::k_rate)));
    EXPECT_EQ(engine.m_loads.load(), 2) << "another geometry loads again";
    EXPECT_EQ(handler.native(), native);

    handler.prepare(v2::HostConfig(256.F, static_cast<float>(anira_test::k_rate), false, 0, false));
    EXPECT_EQ(engine.m_loads.load(), 3) << "another anchor rebuilds and loads again";
    EXPECT_NE(handler.native(), native);
    EXPECT_EQ(engine.m_inits.load(), 1) << "one C engine throughout";
    EXPECT_EQ(handler.get_inference_backend(), v2::CUSTOM);
}

// Case 29: a custom engine off the CPU path runs on its first listed provider: one custom plan
// on that provider, CUSTOM to the host, the output of the 2.x twin.
TEST(AbiCompat, ACustomEngineOffTheCpuPathRunsOnItsFirstProvider) {
    const Arguments args = gain_custom_arguments();
    anira::InferenceConfig old = make_old(args);
    anira::PrePostProcessor old_pp(old);
    KernelBackend<GainKernel> backend(old, GainKernel{});
    anira::InferenceHandler old_handler(old_pp, old, backend, oracle_core_config());
    v2::InferenceConfig config = make_shim(args);
    v2::PrePostProcessor pp(config);
    KernelEngine<GainKernel> engine;
    engine.m_providers = {"com.example.npu"};
    v2::InferenceHandler handler(pp, config, engine, shim_context_config());
    old_handler.prepare(anira::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
    handler.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
    const std::vector<anira_plan_info> plans = plans_of(handler.native());
    ASSERT_EQ(plans.size(), 1U);
    EXPECT_EQ(plans[0].engine, ANIRA_ENGINE_CUSTOM);
    EXPECT_EQ(plans[0].provider, ANIRA_PROVIDER_CUSTOM);
    EXPECT_STREQ(plans[0].provider_id, "com.example.npu");
    EXPECT_EQ(handler.get_inference_backend(), v2::CUSTOM);
    for (size_t k = 1; k <= 8; ++k) {
        std::vector<float> c;
        std::vector<float> v;
        float c_static = -1.F;
        float v_static = -1.F;
        run_gain_block(handler, Form::Multi, k, 0.5F, c, c_static);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        run_gain_block(old_handler, Form::Multi, k, 0.5F, v, v_static);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        anira_test::expect_same_block(c, v, k);
        EXPECT_EQ(c_static, v_static);
    }
}

// Case 30: the same engine whose query clears its one provider leaves the CUSTOM row without a
// plan; as the only row, that is ANIRA_ERROR_CONFIG at construction.
TEST(AbiCompat, ACustomEngineWithoutAUsableProviderIsConfig) {
    v2::InferenceConfig config = make_shim(gain_custom_arguments());
    v2::PrePostProcessor pp(config);
    KernelEngine<GainKernel> engine;
    engine.m_providers = {"com.example.npu"};
    engine.m_query = 0;
    EXPECT_EQ(status_of([&] {
                  const v2::InferenceHandler handler(pp, config, engine, shim_context_config());
              }),
              ANIRA_ERROR_CONFIG);
}

// ---- the 2.x processors, copied (cases 3, 4), and the forms' own rules (cases 14, 15) ---------

// The extras' 2.x processors are included as they are for the 2.x side; the shim side compiles
// the same class bodies, pasted verbatim, against anira::v2 through a namespace alias, so one
// text runs on both type sets (a diff of the bodies against the extras headers is the review).
// NOLINTBEGIN: verbatim copies of extras/models/cnn/CNNPrePostProcessor.h and
// extras/models/hybrid-nn/HybridNNPrePostProcessor.h
namespace compat_copy {
namespace anira = ::anira::v2;

class CNNPrePostProcessor : public anira::PrePostProcessor {
public:
    using anira::PrePostProcessor::PrePostProcessor;

    virtual void pre_process(
        std::vector<anira::RingBuffer>& input,
        std::vector<anira::BufferF>& output,
        [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        pop_samples_from_buffer(input[0],
                                output[0],
                                m_inference_config.get_tensor_output_size()[0],
                                m_inference_config.get_tensor_input_size()[0] -
                                    m_inference_config.get_tensor_output_size()[0]);
    }
};

class HybridNNPrePostProcessor : public anira::PrePostProcessor {
public:
    using anira::PrePostProcessor::PrePostProcessor;

    virtual void pre_process(
        std::vector<anira::RingBuffer>& input,
        std::vector<anira::BufferF>& output,
        [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        int64_t num_batches = 0;
        int64_t num_input_samples = 0;
        int64_t num_output_samples = 0;

        bool channels_last = false;
        anira::InferenceBackend channels_last_backend = anira::InferenceBackend::CUSTOM;
#ifdef USE_TFLITE
        if (current_inference_backend == anira::InferenceBackend::TFLITE) {
            channels_last = true;
            channels_last_backend = anira::InferenceBackend::TFLITE;
        }
#endif
#ifdef USE_LITERT
        if (current_inference_backend == anira::InferenceBackend::LITERT) {
            channels_last = true;
            channels_last_backend = anira::InferenceBackend::LITERT;
        }
#endif
        if (channels_last) {
            num_batches = m_inference_config.get_tensor_input_shape(channels_last_backend)[0][0];
            num_input_samples =
                m_inference_config.get_tensor_input_shape(channels_last_backend)[0][1];
            num_output_samples =
                m_inference_config.get_tensor_output_shape(channels_last_backend)[0][1];
        } else {
            num_batches = m_inference_config.get_tensor_input_shape()[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape()[0][2];
            num_output_samples = m_inference_config.get_tensor_output_shape()[0][1];
        }
        if (
#ifdef USE_LIBTORCH
            current_inference_backend != anira::InferenceBackend::LIBTORCH &&
#endif
#ifdef USE_ONNXRUNTIME
            current_inference_backend != anira::InferenceBackend::ONNX &&
#endif
#ifdef USE_TFLITE
            current_inference_backend != anira::InferenceBackend::TFLITE &&
#endif
#ifdef USE_LITERT
            current_inference_backend != anira::InferenceBackend::LITERT &&
#endif
#ifdef USE_EXECUTORCH
            current_inference_backend != anira::InferenceBackend::EXECUTORCH &&
#endif
            current_inference_backend != anira::InferenceBackend::CUSTOM) {
            throw std::runtime_error("Invalid inference backend");
        }

        for (size_t batch = 0; batch < (size_t)num_batches; batch++) {
            size_t base_index = batch * (size_t)num_input_samples;
            pop_samples_from_buffer(input[0],
                                    output[0],
                                    (size_t)num_output_samples,
                                    (size_t)(num_input_samples - num_output_samples),
                                    base_index);
        }
    }
};

}  // namespace compat_copy
// NOLINTEND

namespace {

/// A 2.x handler and a shim handler, each with its own processor class, over the same
/// arguments.
template <class OldProcessor, class Processor>
struct ProcessorPair {
    explicit ProcessorPair(const Arguments& args)
        : m_old(make_old(args))
        , m_old_pp(m_old)
        , m_old_handler(m_old_pp, m_old, oracle_core_config())
        , m_shim(make_shim(args))
        , m_pp(m_shim)
        , m_handler(m_pp, m_shim, shim_context_config()) {}

    anira::InferenceConfig m_old;
    OldProcessor m_old_pp;
    anira::InferenceHandler m_old_handler;
    v2::InferenceConfig m_shim;
    Processor m_pp;
    v2::InferenceHandler m_handler;
};

/// The rows of a bundled model on the oracle engines (LiteRT left out, as for the gain).
Arguments without_litert(Arguments args) {
    std::erase_if(args.m_rows, [](const Row& row) { return row.m_backend == v2::LITERT; });
    std::erase_if(args.m_shapes, [](const Shapes& row) { return row.m_backend == v2::LITERT; });
    return args;
}

/// One block in place on a mono handler of `block` samples; returns the delivered count.
template <class Handler>
size_t run_in_place(Handler& handler, size_t k, size_t block, std::vector<float>& out) {
    out = anira_test::ramp(k, block);
    const size_t prev = handler.get_available_samples(0);
    const std::array<float*, 1> io{out.data()};
    const size_t delivered = handler.process(io.data(), block);
    wait_ring(handler, prev);
    return delivered;
}

/// Every plan of a processor pair in place, `blocks` blocks each, bit for bit.
template <class Pair>
void expect_every_plan_matches(Pair& pair, size_t block, size_t blocks) {
    std::vector<std::pair<anira::InferenceBackend, v2::InferenceBackend>> backends;
    for (const v2::InferenceBackend backend : k_backends) {
        if (backend == v2::CUSTOM || pair.m_shim.get_model_data(backend) == nullptr) { continue; }
        const std::optional<anira::InferenceBackend> mapped = old_backend(backend);
        if (mapped.has_value()) { backends.emplace_back(*mapped, backend); }
    }
    ASSERT_FALSE(backends.empty()) << "no engine of this build runs the model";
    size_t k = 1;
    for (const auto& [old_backend_value, backend] : backends) {
        SCOPED_TRACE(static_cast<uint32_t>(backend));
        pair.m_old_handler.set_inference_backend(old_backend_value);
        pair.m_handler.set_inference_backend(backend);
        for (size_t i = 0; i < blocks; ++i, ++k) {
            std::vector<float> c;
            std::vector<float> v;
            const size_t n_c = run_in_place(pair.m_handler, k, block, c);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            const size_t n_v = run_in_place(pair.m_old_handler, k, block, v);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            EXPECT_EQ(n_c, n_v) << "block " << k;
            anira_test::expect_same_block(c, v, k);
        }
    }
    EXPECT_EQ(pair.m_handler.rt_error(), ANIRA_OK);
}

}  // namespace

// Case 3: the steerable-nafx CNN through the copied 2.x processor (a window of 15380 samples,
// 2048 of them new) on every engine of the oracle, in place: the samples of the 2.x class.
TEST(AbiCompat, CnnMatchesThroughTheCopiedProcessor) {
    if (anira_test::oracle_engines().empty()) {
        GTEST_SKIP() << "no built-in engine in this build";
    }
    ProcessorPair<CNNPrePostProcessor, compat_copy::CNNPrePostProcessor> pair(
        without_litert(cnn_arguments()));
    pair.m_old_handler.prepare(anira::HostConfig(2048.F, static_cast<float>(anira_test::k_rate)));
    pair.m_handler.prepare(v2::HostConfig(2048.F, static_cast<float>(anira_test::k_rate)));
    expect_every_plan_matches(pair, 2048, 4);
}

// Case 4: GuitarLSTM through the copied 2.x processor, which reads the backend-qualified shape
// (the channels-last TensorFlow rows read [256, 150, 1] through the entry's layout, decision
// 8.1) and pops 256 windows of 150 samples per inference: the samples of the 2.x class.
TEST(AbiCompat, HybridNnMatchesThroughTheCopiedProcessor) {
    if (anira_test::oracle_engines().empty()) {
        GTEST_SKIP() << "no built-in engine in this build";
    }
    ProcessorPair<HybridNNPrePostProcessor, compat_copy::HybridNNPrePostProcessor> pair(
        without_litert(hybridnn_arguments()));
    pair.m_old_handler.prepare(anira::HostConfig(256.F, static_cast<float>(anira_test::k_rate)));
    pair.m_handler.prepare(v2::HostConfig(256.F, static_cast<float>(anira_test::k_rate)));
    expect_every_plan_matches(pair, 256, 6);
}

// Case 14: a single form carries its one slot per side: after a pop of both outputs, a single
// pop of output 0 leaves output 1's memory and ring alone (2.x resent the last pointers of
// every slot, and re-popped output 1 into the previous call's memory). A shim rule, no 2.x
// comparison.
TEST(AbiCompat, ASingleFormCarriesOneSlotPerSide) {
    v2::InferenceConfig config = make_shim(two_output_arguments());
    v2::PrePostProcessor pp(config);
    v2::InferenceHandler handler(pp, config, shim_context_config());
    handler.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
    // One block in, so both output rings hold more than the pops below take.
    const std::vector<float> in = anira_test::ramp(1);
    const std::array<const float*, 1> in_ch{in.data()};
    const size_t prev = handler.get_available_samples(0);
    handler.push_data(in_ch.data(), 512);
    wait_ring(handler, prev + 512);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    std::vector<float> out0(256, -1.F);
    std::vector<float> out1(256, -1.F);
    const std::array<float*, 1> out0_ch{out0.data()};
    const std::array<float*, 1> out1_ch{out1.data()};
    const std::array<float* const*, 2> outs{out0_ch.data(), out1_ch.data()};
    std::array<size_t, 2> num_out{256, 256};
    handler.pop_data(outs.data(), num_out.data());
    EXPECT_EQ(num_out[0], 256U);
    EXPECT_EQ(num_out[1], 256U);
    const size_t ring1 = handler.get_available_samples(1);
    std::ranges::fill(out1, 7.F);
    EXPECT_EQ(handler.pop_data(out0_ch.data(), 128, 0), 128U);
    anira_test::expect_all(out1, 7.F, "output 1's memory is not written");
    EXPECT_EQ(handler.get_available_samples(1), ring1) << "output 1's ring is not popped";
    EXPECT_EQ(handler.rt_error(), ANIRA_OK);
}

// Case 15: the in-place single form on a generator (its input slot 0 is the Static parameter
// tensor, its output slot 0 the stream) stores min(count, 4) values into the processor and
// pulls the stream; the stream-sized count writes nothing past the parameters.
TEST(AbiCompat, AGeneratorSingleFormClampsTheStaticCount) {
    v2::InferenceConfig config = make_shim(generator_arguments());
    v2::PrePostProcessor pp(config);
    KernelEngine<ParamFillKernel> engine;
    v2::InferenceHandler handler(pp, config, engine, shim_context_config());
    handler.prepare(v2::HostConfig(512.F, static_cast<float>(anira_test::k_rate)));
    for (size_t block = 0; block < 8; ++block) {
        std::vector<float> buffer(512);
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = static_cast<float>(block * 1000 + i);
        }
        const std::vector<float> sent = buffer;
        const std::array<float*, 1> channels{buffer.data()};
        const size_t received = handler.process(channels.data(), 512, 0);
        EXPECT_TRUE(received == 0 || received == 512U) << "received " << received;
        for (size_t j = 0; j < 4; ++j) { EXPECT_EQ(pp.get_input(0, j), sent[j]); }
        EXPECT_EQ(pp.get_input(0, 4), 0.F) << "nothing past the tensor's four values";
    }
    EXPECT_EQ(handler.rt_error(), ANIRA_OK);
}
