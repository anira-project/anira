// InferenceConfig's own surface: the accessors and setters hosts call, the
// value semantics of ModelData / TensorShape / ProcessingSpec, and every
// rejection update_processing_spec() performs. All of it is backend-agnostic —
// the CUSTOM backend needs no engine — so these run in every build.

#include <anira/InferenceConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

constexpr float k_max_inference_time = 5.0F;

// The InferenceBackend enumerators are compile-time conditional (USE_*), so
// CUSTOM is the only one guaranteed to exist. Tests that need a *second*,
// distinct backend pick whichever engine this build has and are skipped when
// the build has none.
#if defined(USE_ONNXRUNTIME)
#define ANIRA_TEST_OTHER_BACKEND anira::InferenceBackend::ONNX
#elif defined(USE_LIBTORCH)
#define ANIRA_TEST_OTHER_BACKEND anira::InferenceBackend::LIBTORCH
#elif defined(USE_LITERT)
#define ANIRA_TEST_OTHER_BACKEND anira::InferenceBackend::LITERT
#elif defined(USE_TFLITE)
#define ANIRA_TEST_OTHER_BACKEND anira::InferenceBackend::TFLITE
#elif defined(USE_EXECUTORCH)
#define ANIRA_TEST_OTHER_BACKEND anira::InferenceBackend::EXECUTORCH
#endif

// One CUSTOM model and one universal 512-sample shape: the smallest valid
// configuration, reused by every case that varies exactly one other thing.
std::vector<anira::ModelData> custom_model_data() {
    return {anira::ModelData("model.custom", anira::InferenceBackend::CUSTOM)};
}

std::vector<anira::TensorShape> universal_shape() {
    return {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}})};
}

anira::InferenceConfig make_config() {
    return {custom_model_data(), universal_shape(), k_max_inference_time};
}

}  // namespace

// ============================================================================
// ModelData value semantics
// ============================================================================

// A non-binary ModelData owns a copy of the bytes: the copy constructor,
// the assignment operator and the destructor must each allocate/free it, and
// equality must compare contents rather than pointers.
TEST(InferenceConfigValues, NonBinaryModelDataCopiesItsBytes) {
    std::string path = "some/model/path.pt";
    const anira::ModelData original(path.data(),
                                    path.size(),
                                    anira::InferenceBackend::CUSTOM,
                                    "",
                                    /*is_binary=*/false);
    ASSERT_NE(original.m_data, path.data()) << "non-binary data must be copied, not aliased";

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) the copy is the subject
    const anira::ModelData copy(original);
    EXPECT_NE(copy.m_data, original.m_data);
    EXPECT_EQ(copy, original);

    anira::ModelData assigned("other", anira::InferenceBackend::CUSTOM);
    assigned = original;
    EXPECT_EQ(assigned, original);
    EXPECT_FALSE(assigned != original);

    // Mutating the source buffer must not change the copies.
    path[0] = 'X';
    EXPECT_EQ(copy, original);
}

// A binary ModelData points at externally owned bytes, so equality is pointer
// identity and no copying happens.
TEST(InferenceConfigValues, BinaryModelDataAliasesItsBytes) {
    static constexpr std::array<char, 19> k_bytes = {"binary-model-bytes"};
    const anira::ModelData first(const_cast<char*>(k_bytes.data()),
                                 k_bytes.size(),
                                 anira::InferenceBackend::CUSTOM);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) the copy is the subject
    const anira::ModelData copy(first);
    EXPECT_EQ(copy.m_data, first.m_data);
    EXPECT_EQ(copy, first);

    // Assigning a binary entry rebinds the pointer instead of copying.
    static constexpr std::array<char, 12> k_other = {"other-bytes"};
    anira::ModelData assigned(const_cast<char*>(k_other.data()),
                              k_other.size(),
                              anira::InferenceBackend::CUSTOM);
    assigned = first;
    EXPECT_EQ(assigned.m_data, first.m_data);
    EXPECT_EQ(assigned, first);

    static constexpr std::array<char, 19> k_same_content = {"binary-model-bytes"};
    const anira::ModelData other(const_cast<char*>(k_same_content.data()),
                                 k_same_content.size(),
                                 anira::InferenceBackend::CUSTOM);
    EXPECT_TRUE(other != first) << "binary data compares by pointer, not by content";
}

TEST(InferenceConfigValues, ModelDataInequalityDimensions) {
    const anira::ModelData reference("model", anira::InferenceBackend::CUSTOM);

    const anira::ModelData different_size("model-longer", anira::InferenceBackend::CUSTOM);
    EXPECT_TRUE(reference != different_size);

    // The string constructor defaults to is_binary = false; the raw-pointer one
    // to true. Entries that disagree on it are never equal.
    static constexpr std::array<char, 6> k_bytes = {"model"};
    const anira::ModelData different_binary_flag(const_cast<char*>(k_bytes.data()),
                                                 std::strlen(k_bytes.data()),
                                                 anira::InferenceBackend::CUSTOM);
    EXPECT_TRUE(reference != different_binary_flag);
}

// model_function is only meaningful for LIBTORCH and EXECUTORCH; on any other
// backend it is reported and otherwise carried along verbatim.
TEST(InferenceConfigValues, ModelFunctionOnAnUnsupportedBackendIsStillStored) {
    const anira::ModelData data("model", anira::InferenceBackend::CUSTOM, "forward");
    EXPECT_EQ(data.m_model_function, "forward");
}

// ============================================================================
// TensorShape / ProcessingSpec value semantics
// ============================================================================

TEST(InferenceConfigValues, TensorShapeEquality) {
    const anira::TensorShape universal({{1, 512}}, {{1, 512}});
    const anira::TensorShape same_universal({{1, 512}}, {{1, 512}});
    EXPECT_TRUE(universal == same_universal);
    EXPECT_FALSE(universal != same_universal);
    EXPECT_TRUE(universal.is_universal());

    const anira::TensorShape other_universal({{1, 256}}, {{1, 512}});
    EXPECT_TRUE(universal != other_universal);

    const anira::TensorShape custom({{1, 512}}, {{1, 512}}, anira::InferenceBackend::CUSTOM);
    EXPECT_FALSE(custom.is_universal());
    // A universal shape and a backend-specific one are never equal, whatever
    // their dimensions.
    EXPECT_TRUE(universal != custom);

    const anira::TensorShape same_custom({{1, 512}}, {{1, 512}}, anira::InferenceBackend::CUSTOM);
    EXPECT_TRUE(custom == same_custom);
}

TEST(InferenceConfigValues, ProcessingSpecEquality) {
    const anira::ProcessingSpec reference({1}, {1}, {512}, {512}, {0});
    const anira::ProcessingSpec same({1}, {1}, {512}, {512}, {0});
    EXPECT_TRUE(reference == same);
    EXPECT_FALSE(reference != same);

    EXPECT_TRUE(reference != anira::ProcessingSpec({2}, {1}, {512}, {512}, {0}));
    EXPECT_TRUE(reference != anira::ProcessingSpec({1}, {2}, {512}, {512}, {0}));
    EXPECT_TRUE(reference != anira::ProcessingSpec({1}, {1}, {256}, {512}, {0}));
    EXPECT_TRUE(reference != anira::ProcessingSpec({1}, {1}, {512}, {256}, {0}));
    EXPECT_TRUE(reference != anira::ProcessingSpec({1}, {1}, {512}, {512}, {8}));
}

TEST(InferenceConfigValues, InferenceConfigEquality) {
    const anira::InferenceConfig reference = make_config();
    EXPECT_TRUE(reference == make_config());
    EXPECT_FALSE(reference != make_config());

    anira::InferenceConfig different_time = make_config();
    different_time.m_max_inference_time = k_max_inference_time + 1.0F;
    EXPECT_TRUE(reference != different_time);

    anira::InferenceConfig different_warm_up = make_config();
    different_warm_up.m_warm_up = reference.m_warm_up + 1;
    EXPECT_TRUE(reference != different_warm_up);

    anira::InferenceConfig different_ratio = make_config();
    different_ratio.m_blocking_ratio = 0.9F;
    EXPECT_TRUE(reference != different_ratio);
}

// ============================================================================
// Accessors
// ============================================================================

TEST(InferenceConfigAccessors, ModelLookupsByBackend) {
    anira::InferenceConfig config = make_config();

    EXPECT_EQ(config.get_model_path(anira::InferenceBackend::CUSTOM), "model.custom");
    EXPECT_EQ(config.get_model_function(anira::InferenceBackend::CUSTOM), "");
    // A string literal model path is not binary data.
    EXPECT_FALSE(config.is_model_binary(anira::InferenceBackend::CUSTOM));

    const anira::ModelData* data = config.get_model_data(anira::InferenceBackend::CUSTOM);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->m_backend, anira::InferenceBackend::CUSTOM);
}

#ifdef ANIRA_TEST_OTHER_BACKEND
// Lookups for a backend that has no model entry must report "not found" rather
// than return a neighbouring entry.
TEST(InferenceConfigAccessors, ModelLookupsForAnAbsentBackend) {
    const anira::InferenceConfig config = make_config();
    EXPECT_EQ(config.get_model_function(ANIRA_TEST_OTHER_BACKEND), "");
    EXPECT_FALSE(config.is_model_binary(ANIRA_TEST_OTHER_BACKEND));
    EXPECT_EQ(config.get_model_data(ANIRA_TEST_OTHER_BACKEND), nullptr);
}
#endif  // ANIRA_TEST_OTHER_BACKEND

TEST(InferenceConfigAccessors, SetModelPathReplacesTheBytes) {
    anira::InferenceConfig config = make_config();
    config.set_model_path("another/model.custom", anira::InferenceBackend::CUSTOM);
    EXPECT_EQ(config.get_model_path(anira::InferenceBackend::CUSTOM), "another/model.custom");
}

// A binary model's bytes are owned by the caller, so set_model_path() must
// leave them alone.
TEST(InferenceConfigAccessors, SetModelPathLeavesBinaryModelsUntouched) {
    static constexpr std::array<char, 19> k_bytes = {"binary-model-bytes"};
    anira::InferenceConfig config({anira::ModelData(const_cast<char*>(k_bytes.data()),
                                                    k_bytes.size(),
                                                    anira::InferenceBackend::CUSTOM)},
                                  {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
                                  k_max_inference_time);
    config.set_model_path("ignored", anira::InferenceBackend::CUSTOM);
    EXPECT_EQ(config.get_model_data(anira::InferenceBackend::CUSTOM)->m_data,
              static_cast<const void*>(k_bytes.data()));
}

TEST(InferenceConfigAccessors, DerivedProcessingSpecDefaults) {
    const anira::InferenceConfig config = make_config();
    EXPECT_EQ(config.get_tensor_input_size(), std::vector<size_t>{512});
    EXPECT_EQ(config.get_tensor_output_size(), std::vector<size_t>{512});
    EXPECT_EQ(config.get_preprocess_input_channels(), std::vector<size_t>{1});
    EXPECT_EQ(config.get_postprocess_output_channels(), std::vector<size_t>{1});
    EXPECT_EQ(config.get_preprocess_input_size(), std::vector<size_t>{512});
    EXPECT_EQ(config.get_postprocess_output_size(), std::vector<size_t>{512});
    EXPECT_EQ(config.get_internal_model_latency(), std::vector<size_t>{0});
}

// get_tensor_shape() prefers an exact backend match, falls back to a universal
// shape, and only then reports and returns the first entry.
TEST(InferenceConfigAccessors, TensorShapeLookupPrecedence) {
    const anira::InferenceConfig config(
        custom_model_data(),
        {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}}, anira::InferenceBackend::CUSTOM)},
        k_max_inference_time);

    // Exact match.
    EXPECT_EQ(config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM),
              anira::TensorShapeList({{1, 1, 512}}));
    EXPECT_EQ(config.get_tensor_output_shape(anira::InferenceBackend::CUSTOM),
              anira::TensorShapeList({{1, 1, 512}}));

#ifdef ANIRA_TEST_OTHER_BACKEND
    // No match and no universal shape: reported, then falls back to the first entry.
    EXPECT_EQ(config.get_tensor_input_shape(ANIRA_TEST_OTHER_BACKEND),
              anira::TensorShapeList({{1, 1, 512}}));
#endif
}

TEST(InferenceConfigAccessors, ShapeSettersRederiveTheProcessingSpec) {
    anira::InferenceConfig config = make_config();

    config.set_tensor_input_shape({{1, 1, 256}});
    EXPECT_EQ(config.get_tensor_input_size(), std::vector<size_t>{256});
    EXPECT_EQ(config.get_preprocess_input_size(), std::vector<size_t>{256});

    config.set_tensor_output_shape({{1, 1, 256}});
    EXPECT_EQ(config.get_tensor_output_size(), std::vector<size_t>{256});
    EXPECT_EQ(config.get_postprocess_output_size(), std::vector<size_t>{256});
}

TEST(InferenceConfigAccessors, ProcessingSpecSetters) {
    anira::InferenceConfig config = make_config();

    config.set_preprocess_input_channels({2});
    EXPECT_EQ(config.get_preprocess_input_channels(), std::vector<size_t>{2});

    config.set_preprocess_output_channels({4});
    EXPECT_EQ(config.get_postprocess_output_channels(), std::vector<size_t>{4});

    config.set_preprocess_input_size({128});
    EXPECT_EQ(config.get_preprocess_input_size(), std::vector<size_t>{128});

    config.set_postprocess_output_size({64});
    EXPECT_EQ(config.get_postprocess_output_size(), std::vector<size_t>{64});

    config.set_internal_model_latency({16});
    EXPECT_EQ(config.get_internal_model_latency(), std::vector<size_t>{16});
}

// ============================================================================
// Construction-time validation
// ============================================================================

TEST(InferenceConfigValidation, MaxInferenceTimeMustBePositive) {
    const std::vector<anira::ModelData> model_data = custom_model_data();
    const std::vector<anira::TensorShape> tensor_shape = universal_shape();

    EXPECT_THROW(anira::InferenceConfig(model_data, tensor_shape, 0.0F), std::invalid_argument);
    EXPECT_THROW(anira::InferenceConfig(model_data, tensor_shape, -1.0F), std::invalid_argument);
}

// The processor count is clamped to at least one, and pinned to exactly one for
// a session-exclusive processor whatever the requested count.
TEST(InferenceConfigValidation, ParallelProcessorCountIsClamped) {
    struct ClampCase {
        bool m_session_exclusive;
        unsigned int m_requested;
    };
    constexpr std::array<ClampCase, 3> k_cases = {{
        {.m_session_exclusive = true, .m_requested = 8},
        {.m_session_exclusive = false, .m_requested = 0},
        {.m_session_exclusive = true, .m_requested = 0},
    }};

    for (const auto& test_case : k_cases) {
        const anira::InferenceConfig config(custom_model_data(),
                                            universal_shape(),
                                            k_max_inference_time,
                                            /*warm_up=*/0,
                                            test_case.m_session_exclusive,
                                            /*blocking_ratio=*/0.5F,
                                            test_case.m_requested);
        EXPECT_EQ(config.m_num_parallel_processors, 1U)
            << "session_exclusive=" << test_case.m_session_exclusive
            << " requested=" << test_case.m_requested;
    }

    // A plain count above one is left alone.
    const anira::InferenceConfig config(custom_model_data(),
                                        universal_shape(),
                                        k_max_inference_time,
                                        /*warm_up=*/0,
                                        /*session_exclusive_processor=*/false,
                                        /*blocking_ratio=*/0.5F,
                                        /*num_parallel_processors=*/4);
    EXPECT_EQ(config.m_num_parallel_processors, 4U);
}

#ifdef NDEBUG
// TensorShape asserts on an empty shape list, so this InferenceConfig rejection
// is only reachable in builds where NDEBUG compiles the assert out — which is
// what the Release coverage build does.
TEST(InferenceConfigValidation, EmptyInputOrOutputShapeIsRejected) {
    const std::vector<anira::ModelData> model_data = custom_model_data();

    EXPECT_THROW(anira::InferenceConfig(model_data,
                                        {anira::TensorShape({}, {{1, 1, 512}})},
                                        k_max_inference_time),
                 std::invalid_argument);
    EXPECT_THROW(anira::InferenceConfig(model_data,
                                        {anira::TensorShape({{1, 1, 512}}, {})},
                                        k_max_inference_time),
                 std::invalid_argument);
}
#endif  // NDEBUG

TEST(InferenceConfigValidation, NonPositiveDimensionsAreRejected) {
    const std::vector<anira::ModelData> model_data = custom_model_data();

    EXPECT_THROW(anira::InferenceConfig(model_data,
                                        {anira::TensorShape({{1, 1, 0}}, {{1, 1, 512}})},
                                        k_max_inference_time),
                 std::invalid_argument);
    EXPECT_THROW(anira::InferenceConfig(model_data,
                                        {anira::TensorShape({{1, 1, 512}}, {{1, -1, 512}})},
                                        k_max_inference_time),
                 std::invalid_argument);
}

#ifdef ANIRA_TEST_OTHER_BACKEND
// Every backend-specific shape must describe the same flattened tensor sizes;
// otherwise a backend switch would silently change the buffer geometry.
TEST(InferenceConfigValidation, MismatchedSizesAcrossBackendsAreRejected) {
    const std::vector<anira::ModelData> model_data = custom_model_data();

    EXPECT_THROW(
        anira::InferenceConfig(
            model_data,
            {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}}, anira::InferenceBackend::CUSTOM),
             anira::TensorShape({{1, 1, 256}}, {{1, 1, 512}}, ANIRA_TEST_OTHER_BACKEND)},
            k_max_inference_time),
        std::invalid_argument);

    EXPECT_THROW(
        anira::InferenceConfig(
            model_data,
            {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}}, anira::InferenceBackend::CUSTOM),
             anira::TensorShape({{1, 1, 512}}, {{1, 1, 256}}, ANIRA_TEST_OTHER_BACKEND)},
            k_max_inference_time),
        std::invalid_argument);
}
#endif  // ANIRA_TEST_OTHER_BACKEND

// A ProcessingSpec whose vectors do not have one entry per tensor is silently
// normalised to the derived defaults rather than rejected: update_processing_spec()
// rebuilds any vector whose arity does not match the tensor count. (The explicit
// arity checks further down that function are therefore unreachable.)
TEST(InferenceConfigValidation, ProcessingSpecArityMismatchesAreNormalised) {
    const std::vector<anira::ModelData> model_data = custom_model_data();
    const std::vector<anira::TensorShape> tensor_shape = universal_shape();

    const std::array<anira::ProcessingSpec, 5> specs = {
        anira::ProcessingSpec({1, 1}, {1}, {512}, {512}, {0}),
        anira::ProcessingSpec({1}, {1, 1}, {512}, {512}, {0}),
        anira::ProcessingSpec({1}, {1}, {512, 512}, {512}, {0}),
        anira::ProcessingSpec({1}, {1}, {512}, {512, 512}, {0}),
        anira::ProcessingSpec({1}, {1}, {512}, {512}, {0, 0}),
    };
    for (const auto& spec : specs) {
        const anira::InferenceConfig config(model_data, tensor_shape, spec, k_max_inference_time);
        EXPECT_EQ(config.get_preprocess_input_channels().size(), 1U);
        EXPECT_EQ(config.get_postprocess_output_channels().size(), 1U);
        EXPECT_EQ(config.get_preprocess_input_size().size(), 1U);
        EXPECT_EQ(config.get_postprocess_output_size().size(), 1U);
        EXPECT_EQ(config.get_internal_model_latency().size(), 1U);
    }
}

// A non-streamable tensor (size 0) carries no channel dimension, so anything
// but one channel is a configuration error.
TEST(InferenceConfigValidation, NonStreamableTensorsMustHaveOneChannel) {
    const std::vector<anira::ModelData> model_data = custom_model_data();
    const std::vector<anira::TensorShape> tensor_shape = universal_shape();

    EXPECT_THROW(anira::InferenceConfig(model_data,
                                        tensor_shape,
                                        anira::ProcessingSpec({2}, {1}, {0}, {512}, {0}),
                                        k_max_inference_time),
                 std::invalid_argument);
    EXPECT_THROW(anira::InferenceConfig(model_data,
                                        tensor_shape,
                                        anira::ProcessingSpec({1}, {2}, {512}, {0}, {0}),
                                        k_max_inference_time),
                 std::invalid_argument);

    // One channel on a non-streamable tensor is the accepted form.
    EXPECT_NO_THROW(anira::InferenceConfig(model_data,
                                           tensor_shape,
                                           anira::ProcessingSpec({1}, {1}, {0}, {0}, {0}),
                                           k_max_inference_time));
}

// A universal shape is cloned once per model backend, so every model has an
// entry whose m_backend matches it. The clone stays flagged universal.
TEST(InferenceConfigValidation, UniversalShapeIsClonedForEachModelBackend) {
    const anira::InferenceConfig config(custom_model_data(),
                                        universal_shape(),
                                        k_max_inference_time);
    ASSERT_EQ(config.m_tensor_shape.size(), 2U);
    EXPECT_TRUE(config.m_tensor_shape[0].is_universal());
    EXPECT_EQ(config.m_tensor_shape[1].m_backend, anira::InferenceBackend::CUSTOM);
}

TEST(InferenceConfigValidation, BackendSpecificShapeIsNotCloned) {
    const anira::InferenceConfig config(
        custom_model_data(),
        {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}}, anira::InferenceBackend::CUSTOM)},
        k_max_inference_time);
    EXPECT_EQ(config.m_tensor_shape.size(), 1U);
}
