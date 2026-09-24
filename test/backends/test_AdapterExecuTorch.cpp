// The ExecuTorch adapter, driven directly through the engine room's interface (the built-in
// adapter of the engine, the record of a 2.x configuration, one run over a context built by
// hand: no Context, no threads), on the paths test_ExecuTorchModelFunction.cpp does not take:
// a program handed over as bytes rather than a path, a program that cannot be loaded at all,
// a method the program lacks, and the check against the method meta's planned bounds.

#ifdef USE_EXECUTORCH

#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend_test_support.h"
#include "backends/Adapter.h"
#include "backends/Adapters.h"
#include "gtest/gtest.h"
#include "utils/StatusError.h"

namespace {

constexpr size_t k_size = 64;

using anira::backend::Model;
using anira_test::context_of;
using anira_test::descriptors_of;
using anira_test::filled_buffers;
using anira_test::read_model_file;

std::string multifunction_model_path() {
    return std::string(ANIRA_EXTRAS_MODELS_DIR
                       "/model-pool/example-models/SimpleGainNetwork/models") +
           "/simple_gain_network_multifunction.pte";
}

std::string gain_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.pte";
}

/// A loaded model and one prepared handle over it, driven directly (no core, no session): the
/// shape the inference thread runs. prepare loads the record and prepares one session on it,
/// exclusive when the record has no shared slot (a session-exclusive configuration).
class Rig {
public:
    explicit Rig(std::shared_ptr<anira::backend::Loaded> loaded) : m_loaded(std::move(loaded)) {}

    void prepare(const anira::backend::Model& model) {
        m_prepared.reset();
        // The engine object's init, as the core runs it before a load: the level in effect.
        anira_init_info info = ANIRA_INIT_INFO_INIT;
        info.log_level = static_cast<uint32_t>(anira::get_log_level());
        m_loaded->init(info);
        m_loaded->load(model);
        m_prepared =
            m_loaded->prepare(anira::backend::PrepareRequest{.m_exclusive = model.m_instances == 0,
                                                             .m_info = nullptr});
    }
    bool prepared() const noexcept { return m_loaded->loaded() && m_prepared != nullptr; }
    const anira::backend::Model& model() const noexcept { return m_loaded->model(); }
    const anira::backend::Bindings& bindings() const noexcept { return m_loaded->bindings(); }
    anira_status run(const anira_engine_ctx& ctx,
                     anira::backend::ChunkBuffers* chunk,
                     bool reset_first) noexcept {
        return m_prepared->run(ctx, chunk, reset_first);
    }

private:
    std::shared_ptr<anira::backend::Loaded> m_loaded;
    std::unique_ptr<anira::backend::Prepared> m_prepared;
};

std::shared_ptr<Rig> executorch_adapter() {
    std::shared_ptr<anira::backend::Loaded> loaded = anira::backend::make_builtin_loaded(
        anira::backend::make_builtin_engine(ANIRA_ENGINE_EXECUTORCH));
    EXPECT_NE(loaded, nullptr);
    return std::make_shared<Rig>(std::move(loaded));
}

}  // namespace

// A .pte supplied as bytes loads through the BufferDataLoader branch and runs the same named
// method as the same program supplied as a path. "gain2" multiplies by two, so the output
// identifies which program ran.
TEST(AdapterExecuTorch, BinaryModelDataLoadsFromMemory) {
    const std::vector<char> bytes = read_model_file(multifunction_model_path());
    ASSERT_FALSE(bytes.empty()) << "fixture missing: " << multifunction_model_path();

    const anira::InferenceConfig config(
        {anira::ModelData(const_cast<char*>(bytes.data()),
                          bytes.size(),
                          anira::InferenceBackend::EXECUTORCH,
                          "gain2",
                          /*is_binary=*/true)},
        {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                            {{1, 1, static_cast<int64_t>(k_size)}})},
        5.0F,
        /*warm_up=*/1,
        /*session_exclusive_processor=*/true);
    ASSERT_TRUE(config.is_model_binary(anira::InferenceBackend::EXECUTORCH));
    const Model model = anira::backend::model_of(config, anira::InferenceBackend::EXECUTORCH);
    ASSERT_NE(model.m_bytes, nullptr);
    ASSERT_EQ(model.m_entry, "gain2");

    const std::shared_ptr<Rig> adapter = executorch_adapter();
    adapter->prepare(model);
    EXPECT_EQ(adapter->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));

    std::vector<anira::BufferF> input = filled_buffers({k_size}, 0.25F);
    std::vector<anira::BufferF> output = filled_buffers({k_size}, 0.F);
    const std::vector<anira_tensor> input_tensors = descriptors_of(input, model.m_inputs);
    std::vector<anira_tensor> output_tensors = descriptors_of(output, model.m_outputs);
    EXPECT_EQ(adapter->run(context_of(input_tensors, output_tensors), nullptr, false), ANIRA_OK);
    for (size_t i = 0; i < k_size; ++i) {
        EXPECT_FLOAT_EQ(output[0].get_sample(0, i), 0.5F) << "sample " << i;
    }
}

// The provider of the record on ExecuTorch is the delegate (ExecuTorch's backend) an export
// was lowered for: the adapter serves the default provider and every delegate registered to
// the runtime and available (this package registers XNNPACK), the capabilities' query lists
// the same, and a pinned entry whose method does not use the pinned delegate (the bundled gain
// is a portable export) is a mislabeled export, refused at load naming the pin, the delegate
// and the method.
TEST(AdapterExecuTorch, TheProviderIsTheExportsDelegate) {
    // The package registers the XNNPACK backend (the adapter loads every method with its
    // options): the query lists it as the enum's provider, and the adapter serves every
    // provider the query lists.
    const std::vector<anira::backend::ProviderInfo> listed = anira::backend::executorch_providers();
    const bool lists_xnnpack =
        std::ranges::any_of(listed, [](const anira::backend::ProviderInfo& p) {
            return p.m_provider == ANIRA_PROVIDER_XNNPACK && p.m_provider_id.empty();
        });
    EXPECT_TRUE(lists_xnnpack);

    const std::shared_ptr<Rig> adapter = executorch_adapter();
    const std::shared_ptr<anira::backend::Loaded> fresh = anira::backend::make_builtin_loaded(
        anira::backend::make_builtin_engine(ANIRA_ENGINE_EXECUTORCH));
    ASSERT_NE(fresh, nullptr);
    const anira::backend::Loaded& loaded = *fresh;
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_DEFAULT, ""));
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_XNNPACK, ""));
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_DEFAULT, "XnnpackBackend"))
        << "the registered name serves as the custom spelling";
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_CUDA, ""));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_DEFAULT, "NobodysBackend"));
    for (const anira::backend::ProviderInfo& info : listed) {
        EXPECT_TRUE(loaded.serves(info.m_provider, info.m_provider_id)) << info.m_provider_id;
    }
    EXPECT_NE(loaded.provider_reason().find("XnnpackBackend"), std::string::npos)
        << loaded.provider_reason();

    // The multifunction export's "gain2" (one input, one output), a portable method.
    const anira::InferenceConfig config(
        {anira::ModelData(multifunction_model_path(),
                          anira::InferenceBackend::EXECUTORCH,
                          "gain2")},
        {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                            {{1, 1, static_cast<int64_t>(k_size)}})},
        5.0F,
        /*warm_up=*/1,
        /*session_exclusive_processor=*/true);
    Model neutral = anira::backend::model_of(config, anira::InferenceBackend::EXECUTORCH);
    ASSERT_EQ(neutral.m_entry, "gain2");
    adapter->prepare(neutral);
    EXPECT_TRUE(adapter->prepared()) << "a portable export on the default provider";

    Model pinned = neutral;
    pinned.m_provider = ANIRA_PROVIDER_XNNPACK;
    try {
        executorch_adapter()->prepare(pinned);
        FAIL() << "a portable export pinned to XNNPACK loaded";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(
            message.find("pinned to provider 'xnnpack' (ExecuTorch backend 'XnnpackBackend')"),
            std::string::npos)
            << message;
        EXPECT_NE(message.find("method 'gain2' uses no delegate"), std::string::npos) << message;
    }
    Model cuda = neutral;
    cuda.m_provider = ANIRA_PROVIDER_CUDA;
    try {
        executorch_adapter()->prepare(cuda);
        FAIL() << "a provider no delegate spells loaded";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_NE(std::string(error.what()).find("does not serve provider 'cuda'"),
                  std::string::npos)
            << error.what();
    }
}

// The contract create_session() rolls back on: a StatusError, which is a std::runtime_error.
TEST(AdapterExecuTorch, UnloadableModelThrowsRuntimeError) {
    const anira::InferenceConfig config(
        {anira::ModelData("this/model/does/not/exist.pte", anira::InferenceBackend::EXECUTORCH)},
        {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                            {{1, 1, static_cast<int64_t>(k_size)}})},
        5.0F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/true);
    const std::shared_ptr<Rig> adapter = executorch_adapter();
    EXPECT_THROW(
        adapter->prepare(anira::backend::model_of(config, anira::InferenceBackend::EXECUTORCH)),
        std::runtime_error);
    EXPECT_FALSE(adapter->prepared());
}

// A method name the program does not carry must fail the same way rather than silently
// falling back to forward().
TEST(AdapterExecuTorch, UnknownModelFunctionThrowsRuntimeError) {
    const anira::InferenceConfig config(
        {anira::ModelData(multifunction_model_path(),
                          anira::InferenceBackend::EXECUTORCH,
                          "no_such_method")},
        {anira::TensorShape({{1, 1, static_cast<int64_t>(k_size)}},
                            {{1, 1, static_cast<int64_t>(k_size)}})},
        5.0F,
        /*warm_up=*/0,
        /*session_exclusive_processor=*/true);
    const std::shared_ptr<Rig> adapter = executorch_adapter();
    try {
        adapter->prepare(anira::backend::model_of(config, anira::InferenceBackend::EXECUTORCH));
        FAIL() << "a missing method loaded";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_MODEL_LOAD);
        EXPECT_NE(std::string(error.what()).find("no_such_method"), std::string::npos)
            << error.what();
    }
}

// The method meta reports the planned upper bound of every axis and marks no axis dynamic:
// the bundled gain program plans [1, 1, 65536] for its block, so a record of 512 binds and one
// above the bound, or of another rank, is CONFIG at prepare naming both shapes.
TEST(AdapterExecuTorch, TheMethodMetasPlannedBoundsAreTheCheck) {
    const auto config_of = [](const std::vector<int64_t>& block) {
        return anira::InferenceConfig(
            {anira::ModelData(gain_model_path(), anira::InferenceBackend::EXECUTORCH)},
            {anira::TensorShape({block, {1}}, {block, {1}})},
            5.0F,
            /*warm_up=*/0,
            /*session_exclusive_processor=*/true);
    };
    const std::shared_ptr<Rig> adapter = executorch_adapter();
    adapter->prepare(
        anira::backend::model_of(config_of({1, 1, 512}), anira::InferenceBackend::EXECUTORCH));
    EXPECT_TRUE(adapter->prepared());
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));

    try {
        executorch_adapter()->prepare(
            anira::backend::model_of(config_of({1, 1, 70000}),
                                     anira::InferenceBackend::EXECUTORCH));
        FAIL() << "a block above the planned bound bound";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("executorch: input tensor"), std::string::npos) << message;
        EXPECT_NE(message.find("65536 is the planned upper bound of 70000"), std::string::npos)
            << message;
    }
    try {
        executorch_adapter()->prepare(
            anira::backend::model_of(config_of({1, 512}), anira::InferenceBackend::EXECUTORCH));
        FAIL() << "a block of another rank bound";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string(error.what()).find("the ranks differ"), std::string::npos)
            << error.what();
    }
}

// The engine object's init sets the XNNPACK delegate's runtime options for the process (a
// workspace per delegate instance, no weight cache) and reads them back through the
// registered backend, so a release that spells the option keys otherwise is caught at init,
// not by the mutexes.
TEST(AdapterExecuTorch, TheEngineInitSetsTheXnnpackOptionsOfTheProcess) {
    const std::shared_ptr<anira::backend::BuiltinEngine> engine =
        anira::backend::make_builtin_engine(ANIRA_ENGINE_EXECUTORCH);
    ASSERT_NE(engine, nullptr);
    anira_init_info info = ANIRA_INIT_INFO_INIT;
    info.log_level = static_cast<uint32_t>(anira::get_log_level());
    engine->ensure_init(info);
    int workspace_sharing_mode = -1;
    bool weight_cache_enabled = true;
    ASSERT_TRUE(
        anira::backend::executorch_xnnpack_options(workspace_sharing_mode, weight_cache_enabled))
        << "the registered backend answers";
    EXPECT_EQ(workspace_sharing_mode, 0) << "a workspace per delegate instance";
    EXPECT_FALSE(weight_cache_enabled);
}

#endif  // USE_EXECUTORCH
