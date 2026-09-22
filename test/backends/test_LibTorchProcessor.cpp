// The LibTorch adapter, driven directly through the engine room's interface (the built-in
// adapter of the engine, the record of a 2.x configuration, one run over a context built by
// hand: no Context, no threads), on the paths the end-to-end suites never take: a model handed
// over as bytes instead of a path, a named entry (warm-up and steady state), a load failure, a
// module returning a tuple of tensors, the inputs bound to the method's arguments by name, a
// method with a defaulted argument, and an output of another shape than the record's.

#ifdef USE_LIBTORCH

#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend_test_support.h"
#include "backends/Adapter.h"
#include "backends/Adapters.h"
#include "gtest/gtest.h"
#include "utils/StatusError.h"

namespace {

using anira::backend::Adapter;
using anira::backend::Model;
using anira_test::any_sample_nonzero;
using anira_test::context_of;
using anira_test::descriptors_of;
using anira_test::filled_buffers;
using anira_test::read_model_file;

std::string rave_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR "/third-party/ircam-acids/RAVE/rave_funk_drum.ts";
}

std::string gain_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.pt";
}

std::shared_ptr<Adapter> libtorch_adapter() {
    std::shared_ptr<Adapter> adapter = anira::backend::make_builtin_adapter(ANIRA_ENGINE_LIBTORCH);
    EXPECT_NE(adapter, nullptr);
    return adapter;
}

// The gain model's 2.x configuration: the block and the static gain in, the block and the
// peak out; a warm-up of one.
anira::InferenceConfig gain_config(uint32_t warm_up = 1) {
    return anira::InferenceConfig(
        {anira::ModelData(gain_model_path(), anira::InferenceBackend::LIBTORCH)},
        {anira::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {1}})},
        anira::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F,
        warm_up,
        /*session_exclusive_processor=*/true);
}

// One run of `adapter` over the buffers, with descriptors in the record's shapes.
anira_status run_once(Adapter& adapter,
                      std::vector<anira::BufferF>& inputs,
                      std::vector<anira::BufferF>& outputs) {
    const std::vector<anira_tensor> input_tensors =
        descriptors_of(inputs, adapter.model().m_inputs);
    std::vector<anira_tensor> output_tensors = descriptors_of(outputs, adapter.model().m_outputs);
    return adapter.run(context_of(input_tensors, output_tensors), nullptr, false);
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
    const Model model = anira::backend::model_of(config, anira::InferenceBackend::LIBTORCH);
    ASSERT_NE(model.m_bytes, nullptr);

    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    adapter->prepare(model);
    ASSERT_TRUE(adapter->prepared());

    std::vector<anira::BufferF> input = filled_buffers({2048}, 0.1F);
    std::vector<anira::BufferF> output = filled_buffers({2048}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);

    // RAVE is not an identity, so assert the buffer was written at all rather
    // than a specific value.
    EXPECT_TRUE(any_sample_nonzero(output[0]));
}

// A named entry must be used for the warm-up inferences and for every later run, instead of
// forward().
TEST(LibTorchProcessor, NamedModelFunctionIsUsedForWarmUpAndProcessing) {
    anira::InferenceConfig config(
        {anira::ModelData(rave_model_path(), anira::InferenceBackend::LIBTORCH, "encode")},
        {anira::TensorShape({{1, 1, 2048}}, {{1, 4, 1}})},
        anira::ProcessingSpec({1}, {4}),
        42.66F,
        /*warm_up=*/2,
        /*session_exclusive_processor=*/true);
    const Model model = anira::backend::model_of(config, anira::InferenceBackend::LIBTORCH);
    ASSERT_EQ(model.m_entry, "encode");
    ASSERT_EQ(model.m_warm_up, 2U);

    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    adapter->prepare(model);

    std::vector<anira::BufferF> input = filled_buffers({2048}, 0.2F);
    std::vector<anira::BufferF> output = filled_buffers({4}, 0.F);
    // "encode" produces the 4 latent values; forward() would return a 2048-sample block,
    // which the delivery would refuse against the record's [1, 4, 1].
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);
    EXPECT_EQ(output[0].get_num_samples(), 4U);
}

// A model that cannot be loaded fails prepare with std::runtime_error (a StatusError of
// MODEL_LOAD or NO_SUCH_FILE) — the contract the other engines honour, and the one
// create_session() rolls back on (see test_CreateSessionFailure.cpp). Carrying on with an
// empty module would let an engine-specific c10 exception escape instead.
TEST(LibTorchProcessor, UnloadableModelThrowsRuntimeError) {
    anira::InferenceConfig config(
        {anira::ModelData("this/model/does/not/exist.pt", anira::InferenceBackend::LIBTORCH)},
        {anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        anira::ProcessingSpec({1}, {1}, {512}, {512}),
        5.F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/true);
    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    EXPECT_THROW(
        adapter->prepare(anira::backend::model_of(config, anira::InferenceBackend::LIBTORCH)),
        std::runtime_error);
    EXPECT_FALSE(adapter->prepared());

    // A method the module does not have fails the same way.
    anira::InferenceConfig no_method(
        {anira::ModelData(gain_model_path(), anira::InferenceBackend::LIBTORCH, "no_such_method")},
        {anira::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {1}})},
        anira::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/true);
    try {
        adapter->prepare(anira::backend::model_of(no_method, anira::InferenceBackend::LIBTORCH));
        FAIL() << "a missing method loaded";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_MODEL_LOAD);
        EXPECT_NE(std::string(error.what()).find("no method 'no_such_method'"), std::string::npos)
            << error.what();
    }
}

// The gain fixture returns two tensors, so its result arrives as a tuple rather than a bare
// tensor — the branch that unpacks it per output slot. On the 2.x path the record has no
// names: the slots bind the method's arguments (data, gain) by position.
TEST(LibTorchProcessor, MultiTensorOutputIsUnpacked) {
    ASSERT_FALSE(read_model_file(gain_model_path()).empty())
        << "fixture missing: " << gain_model_path();
    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    adapter->prepare(anira::backend::model_of(gain_config(), anira::InferenceBackend::LIBTORCH));
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));

    std::vector<anira::BufferF> input = filled_buffers({512, 1}, 0.5F);
    std::vector<anira::BufferF> output = filled_buffers({512, 1}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);

    // Both output tensors must have been written, which only happens if the tuple was
    // unpacked element by element: the block times the gain of one half, and its peak.
    for (size_t n = 0; n < 512; ++n) { ASSERT_EQ(output[0].get_sample(0, n), 0.25F) << n; }
    EXPECT_EQ(output[1].get_sample(0, 0), 0.25F);
}

// The method's arguments have names (forward(self, data, gain) here): a record naming the
// slots crosswise binds by name, so the scalar still lands in `gain` and the block in `data`
// whatever the slot order; a record naming an argument the method lacks lists the arguments;
// a record on an output side, which has no names, is refused too.
TEST(LibTorchProcessor, InputsBindToTheMethodsArgumentsByName) {
    anira::InferenceConfig config(
        {anira::ModelData(gain_model_path(), anira::InferenceBackend::LIBTORCH)},
        {anira::TensorShape({{1}, {1, 1, 512}}, {{1, 1, 512}, {1}})},
        anira::ProcessingSpec({1, 1}, {1, 1}, {0, 512}, {512, 0}),
        5.F,
        /*warm_up=*/1,
        /*session_exclusive_processor=*/true);
    Model model = anira::backend::model_of(config, anira::InferenceBackend::LIBTORCH);
    model.m_inputs[0].m_engine_name = "gain";
    model.m_inputs[1].m_engine_name = "data";
    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    adapter->prepare(model);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));

    std::vector<anira::BufferF> input = filled_buffers({1, 512}, 0.5F);
    for (size_t n = 0; n < 512; ++n) { input[1].set_sample(0, n, static_cast<float>(n) / 512.F); }
    std::vector<anira::BufferF> output = filled_buffers({512, 1}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);
    for (size_t n = 0; n < 512; ++n) {
        ASSERT_EQ(output[0].get_sample(0, n), 0.5F * (static_cast<float>(n) / 512.F)) << n;
    }

    model.m_inputs[0].m_engine_name = "ghost";
    try {
        libtorch_adapter()->prepare(model);
        FAIL() << "a missing argument bound";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("'ghost'"), std::string::npos) << message;
        EXPECT_NE(message.find("'data', 'gain'"), std::string::npos) << message;
    }

    model.m_inputs[0].m_engine_name = "gain";
    model.m_outputs[0].m_engine_name = "processed";
    try {
        libtorch_adapter()->prepare(model);
        FAIL() << "a named output bound";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("binds by position alone"), std::string::npos)
            << error.what();
    }
}

// RAVE's decode(z, from_forward: bool = False): one model input feeds the leading argument and
// the defaulted one stays with its default, so the count check does not ask for it.
TEST(LibTorchProcessor, AMethodWithADefaultedArgumentBindsItsLeadingArgument) {
    anira::InferenceConfig config(
        {anira::ModelData(rave_model_path(), anira::InferenceBackend::LIBTORCH, "decode")},
        {anira::TensorShape({{1, 4, 1}}, {{1, 1, 2048}})},
        anira::ProcessingSpec({4}, {1}),
        42.66F,
        /*warm_up=*/1,
        /*session_exclusive_processor=*/true);
    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    adapter->prepare(anira::backend::model_of(config, anira::InferenceBackend::LIBTORCH));
    EXPECT_EQ(adapter->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));

    std::vector<anira::BufferF> input = filled_buffers({4}, 0.1F);
    std::vector<anira::BufferF> output = filled_buffers({2048}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);
    EXPECT_TRUE(any_sample_nonzero(output[0]));
}

// The schema carries no shapes: an output of another shape than the record's is caught on the
// warm-up's first result at prepare (CONFIG naming both shapes), and, without a warm-up, at
// the first run (the chunk fails with ENGINE; nothing is written).
TEST(LibTorchProcessor, AnOutputOfAnotherShapeIsRefusedAtTheWarmUpOrFailsTheRun) {
    anira::InferenceConfig config(
        {anira::ModelData(gain_model_path(), anira::InferenceBackend::LIBTORCH)},
        {anira::TensorShape({{1, 1, 512}, {1}}, {{1, 1, 512}, {2}})},
        anira::ProcessingSpec({1, 1}, {1, 1}, {512, 0}, {512, 0}),
        5.F,
        /*warm_up=*/1,
        /*session_exclusive_processor=*/true);
    const Model warmed = anira::backend::model_of(config, anira::InferenceBackend::LIBTORCH);
    try {
        libtorch_adapter()->prepare(warmed);
        FAIL() << "a [2] output passed for a [1] return";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("libtorch: output tensor"), std::string::npos) << message;
        EXPECT_NE(message.find("[1]"), std::string::npos) << message;
        EXPECT_NE(message.find("[2]"), std::string::npos) << message;
    }

    Model unwarmed = warmed;
    unwarmed.m_warm_up = 0;
    const std::shared_ptr<Adapter> adapter = libtorch_adapter();
    adapter->prepare(unwarmed);
    std::vector<anira::BufferF> input = filled_buffers({512, 1}, 0.5F);
    std::vector<anira::BufferF> output = filled_buffers({512, 2}, -1.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_ERROR_ENGINE);
    EXPECT_EQ(output[1].get_sample(0, 0), -1.F) << "nothing was written";
}

#endif  // USE_LIBTORCH
