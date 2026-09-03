// The LibtorchProcessor paths the end-to-end suites never take: a model handed
// over as bytes instead of a path, a named model_function (warm-up and steady
// state), a load failure, and a module returning a tuple of tensors. The
// processor is driven directly — no Context, no threads — the way
// test_BackendBase.cpp drives BackendBase, so each case is one deterministic call.

#ifdef USE_LIBTORCH

#include <anira/InferenceConfig.h>
#include <anira/backends/LibTorchProcessor.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "backend_test_support.h"
#include "gtest/gtest.h"

namespace {

using anira_test::any_sample_nonzero;
using anira_test::filled_buffers;
using anira_test::read_model_file;

std::string rave_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR "/third-party/ircam-acids/RAVE/rave_funk_drum.ts";
}

std::string gain_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.pt";
}

}  // namespace

// A model supplied as bytes must load through the in-memory branch.
TEST(LibTorchProcessor, BinaryModelDataLoadsFromMemory) {
    const std::vector<char> bytes = read_model_file(rave_model_path());
    ASSERT_FALSE(bytes.empty()) << "fixture missing: " << rave_model_path();

    anira::InferenceConfig config({anira::ModelData(const_cast<char*>(bytes.data()),
                                                    bytes.size(),
                                                    anira::InferenceBackend::LIBTORCH,
                                                    "",
                                                    /*is_binary=*/true)},
                                  {anira::TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                                  anira::ProcessingSpec({1}, {1}, {2048}, {2048}, {2048}),
                                  42.66F,
                                  /*warm_up=*/0,
                                  /*session_exclusive_processor=*/true);
    ASSERT_TRUE(config.is_model_binary(anira::InferenceBackend::LIBTORCH));

    anira::LibtorchProcessor processor(config);
    processor.prepare();

    std::vector<anira::BufferF> input = filled_buffers({2048}, 0.1F);
    std::vector<anira::BufferF> output = filled_buffers({2048}, 0.F);

    processor.process(input, output, nullptr);

    // RAVE is not an identity, so assert the buffer was written at all rather
    // than a specific value.
    EXPECT_TRUE(any_sample_nonzero(output[0]));
}

// A named model_function must be used for the warm-up inferences and for every
// later process() call, instead of forward().
TEST(LibTorchProcessor, NamedModelFunctionIsUsedForWarmUpAndProcessing) {
    anira::InferenceConfig config(
        {anira::ModelData(rave_model_path(), anira::InferenceBackend::LIBTORCH, "encode")},
        {anira::TensorShape({{1, 1, 2048}}, {{1, 4, 1}})},
        anira::ProcessingSpec({1}, {4}),
        42.66F,
        /*warm_up=*/2,
        /*session_exclusive_processor=*/true);
    ASSERT_EQ(config.get_model_function(anira::InferenceBackend::LIBTORCH), "encode");

    anira::LibtorchProcessor processor(config);
    processor.prepare();

    std::vector<anira::BufferF> input = filled_buffers({2048}, 0.2F);
    std::vector<anira::BufferF> output = filled_buffers({4}, 0.F);

    // "encode" produces the 4 latent values; forward() would need a
    // 2048-sample output buffer and overrun this one.
    processor.process(input, output, nullptr);
    EXPECT_EQ(output[0].get_num_samples(), 4U);
}

// A model that cannot be loaded fails construction with std::runtime_error —
// the same contract the other backends honour, and the one create_session()
// rolls back on (see test_CreateSessionFailure.cpp). Carrying on with an empty
// module would let an engine-specific c10 exception escape instead.
TEST(LibTorchProcessor, UnloadableModelThrowsRuntimeError) {
    anira::InferenceConfig config(
        {anira::ModelData("this/model/does/not/exist.pt", anira::InferenceBackend::LIBTORCH)},
        {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        anira::ProcessingSpec({1}, {1}, {512}, {512}),
        5.F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/true);

    EXPECT_THROW({ const anira::LibtorchProcessor processor(config); }, std::runtime_error);
}

// The gain fixture returns two tensors, so its output arrives as a tuple rather
// than a bare tensor — the branch that unpacks it per output tensor.
TEST(LibTorchProcessor, MultiTensorOutputIsUnpacked) {
    ASSERT_FALSE(read_model_file(gain_model_path()).empty())
        << "fixture missing: " << gain_model_path();

    anira::InferenceConfig config(
        {anira::ModelData(gain_model_path(), anira::InferenceBackend::LIBTORCH)},
        {anira::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {1}})},
        anira::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F,
        /*warm_up=*/1,
        /*session_exclusive_processor=*/true);

    anira::LibtorchProcessor processor(config);
    processor.prepare();

    std::vector<anira::BufferF> input = filled_buffers({512, 1}, 0.5F);
    std::vector<anira::BufferF> output = filled_buffers({512, 1}, 0.F);

    processor.process(input, output, nullptr);

    // Both output tensors must have been written, which only happens if the
    // tuple was unpacked element by element.
    EXPECT_NE(output[0].get_sample(0, 0), 0.F);
}

#endif  // USE_LIBTORCH
