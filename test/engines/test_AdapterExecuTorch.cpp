// The ExecuTorch adapter, driven directly through the engine room's interface (the built-in
// adapter of the engine, a model record written out, one run over a context built by
// hand: no Context, no threads), on the paths test_ExecuTorchModelFunction.cpp does not take:
// a program handed over as bytes rather than a path, a program that cannot be loaded at all,
// a method the program lacks, and the check against the method meta's planned bounds.

#ifdef USE_EXECUTORCH

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine_test_support.h"
#include "engines/Adapter.h"
#include "engines/Adapters.h"
#include "gtest/gtest.h"
#include "utils/Logger.h"
#include "utils/StatusError.h"

namespace {

constexpr size_t k_size = 64;

using anira::engine::Model;
using anira_test::context_of;
using anira_test::descriptors_of;
using anira_test::filled_buffers;
using anira_test::model_record;
using anira_test::read_model_file;
using anira_test::with_bytes;

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
/// exclusive when the record has no shared slot (a stateful model's).
class Rig {
public:
    explicit Rig(std::shared_ptr<anira::engine::Loaded> loaded) : m_loaded(std::move(loaded)) {}

    void prepare(const anira::engine::Model& model) {
        m_prepared.reset();
        // The engine object's init, as the core runs it before a load: the level in effect.
        anira_init_info info = ANIRA_INIT_INFO_INIT;
        info.log_level = static_cast<uint32_t>(anira::get_log_level());
        m_loaded->init(info);
        m_loaded->load(model);
        m_prepared =
            m_loaded->prepare(anira::engine::PrepareRequest{.m_exclusive = model.m_instances == 0,
                                                            .m_info = nullptr});
    }
    bool prepared() const noexcept { return m_loaded->loaded() && m_prepared != nullptr; }
    const anira::engine::Model& model() const noexcept { return m_loaded->model(); }
    const anira::engine::Bindings& bindings() const noexcept { return m_loaded->bindings(); }
    anira_status run(const anira_engine_ctx& ctx, bool reset_first) noexcept {
        return m_prepared->run(ctx, reset_first);
    }
    const anira::engine::Loaded& loaded() const noexcept { return *m_loaded; }
    /// The executor the session runs on: its own for an exclusive session, else shared slot 0.
    const anira::engine::Executor& executor() const {
        const auto* prepared =
            dynamic_cast<const anira::engine::ExecutorPrepared*>(m_prepared.get());
        if (prepared != nullptr && prepared->own() != nullptr) { return *prepared->own(); }
        return dynamic_cast<const anira::engine::ExecutorLoaded&>(*m_loaded).executor(0);
    }

private:
    std::shared_ptr<anira::engine::Loaded> m_loaded;
    std::unique_ptr<anira::engine::Prepared> m_prepared;
};

std::shared_ptr<Rig> executorch_adapter() {
    std::shared_ptr<anira::engine::Loaded> loaded = anira::engine::make_builtin_loaded(
        anira::engine::make_builtin_engine(ANIRA_ENGINE_EXECUTORCH));
    EXPECT_NE(loaded, nullptr);
    return std::make_shared<Rig>(std::move(loaded));
}

}  // namespace

// The int16 TypedGainNetwork exports, the memory plan's two cases: the planned export (every
// input and output the plan's) costs one copy per slot each way (set_input's into the plan,
// counted, and the result's out of it) and reports HOST_COPY; the unplanned one at its static
// 512-sample window aliases the inputs (set_input) and binds the outputs to the caller's memory
// per call (set_output_data_ptr): no copy, the results in the caller's memory, ZERO_COPY on
// every slot. Two memories alternated, new values each call: the results follow the memory.
TEST(AdapterExecuTorch, TheMemoryPlanDecidesWhatIsBoundInPlace) {
    for (const bool unplanned : {false, true}) {
        SCOPED_TRACE(unplanned ? "unplanned" : "planned");
        Model model = model_record(
            ANIRA_ENGINE_EXECUTORCH,
            std::string(
                ANIRA_EXTRAS_MODELS_DIR
                "/model-pool/example-models/TypedGainNetwork/models/typed_gain_network_int16") +
                (unplanned ? "_unplanned.pte" : ".pte"),
            {{1, 1, 512}, {1}},
            {{1, 1, 512}, {1}},
            1);
        for (auto* side : {&model.m_inputs, &model.m_outputs}) {
            for (anira::engine::TensorInfo& tensor : *side) { tensor.m_dtype = ANIRA_DTYPE_I16; }
        }
        const std::shared_ptr<Rig> adapter = executorch_adapter();
        adapter->prepare(model);
        ASSERT_TRUE(adapter->prepared());
        const anira_edge_class edge = unplanned ? ANIRA_EDGE_ZERO_COPY : ANIRA_EDGE_HOST_COPY;
        for (const bool is_input : {true, false}) {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                EXPECT_EQ(adapter->loaded().slot_edge(is_input, slot).m_class, edge);
            }
        }
        struct Memory {
            std::vector<int16_t> m_data = std::vector<int16_t>(512);
            std::array<int16_t, 1> m_gain{2};
            std::vector<int16_t> m_out = std::vector<int16_t>(512);
            std::array<int16_t, 1> m_peak{};
        };
        std::array<Memory, 2> memories;
        const std::array<int64_t, 3> block{1, 1, 512};
        const std::array<int64_t, 1> scalar{1};
        for (int call = 0; call < 4; ++call) {
            Memory& memory = memories.at(static_cast<size_t>(call % 2));
            const int scale = call + 1;
            for (size_t n = 0; n < 512; ++n) {
                memory.m_data[n] = static_cast<int16_t>(scale * (static_cast<int>(n % 17) - 8));
            }
            std::ranges::fill(memory.m_out, int16_t{0x5a5a});
            std::vector<anira_tensor> inputs(2);
            std::vector<anira_tensor> outputs(2);
            anira_tensor_init_host(&inputs[0],
                                   memory.m_data.data(),
                                   ANIRA_DTYPE_I16,
                                   3,
                                   block.data());
            anira_tensor_init_host(&inputs[1],
                                   memory.m_gain.data(),
                                   ANIRA_DTYPE_I16,
                                   1,
                                   scalar.data());
            anira_tensor_init_host(&outputs[0],
                                   memory.m_out.data(),
                                   ANIRA_DTYPE_I16,
                                   3,
                                   block.data());
            anira_tensor_init_host(&outputs[1],
                                   memory.m_peak.data(),
                                   ANIRA_DTYPE_I16,
                                   1,
                                   scalar.data());
            const uint64_t copies_in = adapter->executor().copies_in();
            const uint64_t copies_out = adapter->executor().copies_out();
            ASSERT_EQ(adapter->run(context_of(inputs, outputs), false), ANIRA_OK);
            const uint64_t copies = unplanned ? 0U : 2U;
            EXPECT_EQ(adapter->executor().copies_in() - copies_in, copies);
            EXPECT_EQ(adapter->executor().copies_out() - copies_out, copies);
            for (size_t n = 0; n < 512; ++n) {
                ASSERT_EQ(memory.m_out[n], 2 * memory.m_data[n])
                    << "call " << call << " sample " << n;
            }
            EXPECT_EQ(memory.m_peak[0], 16 * scale);
        }
    }
}

// A .pte supplied as bytes loads through the BufferDataLoader branch and runs the same named
// method as the same program supplied as a path. "gain2" multiplies by two, so the output
// identifies which program ran.
TEST(AdapterExecuTorch, BinaryModelDataLoadsFromMemory) {
    const std::vector<char> bytes = read_model_file(multifunction_model_path());
    ASSERT_FALSE(bytes.empty()) << "fixture missing: " << multifunction_model_path();

    const Model model = with_bytes(model_record(ANIRA_ENGINE_EXECUTORCH,
                                                "",
                                                {{1, 1, static_cast<int64_t>(k_size)}},
                                                {{1, 1, static_cast<int64_t>(k_size)}},
                                                1,
                                                "gain2"),
                                   bytes);
    ASSERT_NE(model.m_bytes, nullptr);
    ASSERT_EQ(model.m_entry, "gain2");

    const std::shared_ptr<Rig> adapter = executorch_adapter();
    adapter->prepare(model);
    EXPECT_EQ(adapter->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));

    anira_test::Buffers input = filled_buffers({k_size}, 0.25F);
    anira_test::Buffers output = filled_buffers({k_size}, 0.F);
    const std::vector<anira_tensor> input_tensors = descriptors_of(input, model.m_inputs);
    std::vector<anira_tensor> output_tensors = descriptors_of(output, model.m_outputs);
    EXPECT_EQ(adapter->run(context_of(input_tensors, output_tensors), false), ANIRA_OK);
    for (size_t i = 0; i < k_size; ++i) { EXPECT_FLOAT_EQ(output[0][i], 0.5F) << "sample " << i; }
}

// The provider of the record on ExecuTorch is the delegate (ExecuTorch's backend) an export
// was lowered for: the adapter serves the CPU path and every delegate registered to
// the runtime and available (this package registers XNNPACK), the capabilities' query lists
// the same, and a pinned entry whose method does not use the pinned delegate (the bundled gain
// is a portable export) is a mislabeled export, refused at load naming the pin, the delegate
// and the method.
TEST(AdapterExecuTorch, TheProviderIsTheExportsDelegate) {
    // The package registers the XNNPACK backend (the adapter loads every method with its
    // options): the query lists it as the enum's provider, and the adapter serves every
    // provider the query lists.
    const std::vector<anira::engine::ProviderInfo> listed = anira::engine::executorch_providers();
    const bool lists_xnnpack =
        std::ranges::any_of(listed, [](const anira::engine::ProviderInfo& p) {
            return p.m_provider == ANIRA_PROVIDER_XNNPACK && p.m_provider_id.empty();
        });
    EXPECT_TRUE(lists_xnnpack);

    const std::shared_ptr<Rig> adapter = executorch_adapter();
    const std::shared_ptr<anira::engine::Loaded> fresh = anira::engine::make_builtin_loaded(
        anira::engine::make_builtin_engine(ANIRA_ENGINE_EXECUTORCH));
    ASSERT_NE(fresh, nullptr);
    const anira::engine::Loaded& loaded = *fresh;
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_CPU, ""));
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_XNNPACK, ""));
    EXPECT_TRUE(loaded.serves(ANIRA_PROVIDER_CUSTOM, "XnnpackBackend"))
        << "the registered name serves as the custom spelling";
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_CUDA, ""));
    EXPECT_FALSE(loaded.serves(ANIRA_PROVIDER_CUSTOM, "NobodysBackend"));
    for (const anira::engine::ProviderInfo& info : listed) {
        EXPECT_TRUE(loaded.serves(info.m_provider, info.m_provider_id)) << info.m_provider_id;
    }
    EXPECT_NE(loaded.provider_reason().find("XnnpackBackend"), std::string::npos)
        << loaded.provider_reason();

    // The multifunction export's "gain2" (one input, one output), a portable method.
    const Model neutral = model_record(ANIRA_ENGINE_EXECUTORCH,
                                       multifunction_model_path(),
                                       {{1, 1, static_cast<int64_t>(k_size)}},
                                       {{1, 1, static_cast<int64_t>(k_size)}},
                                       1,
                                       "gain2");
    ASSERT_EQ(neutral.m_entry, "gain2");
    adapter->prepare(neutral);
    EXPECT_TRUE(adapter->prepared()) << "a portable export on the CPU path";

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
    const std::shared_ptr<Rig> adapter = executorch_adapter();
    EXPECT_THROW(adapter->prepare(model_record(ANIRA_ENGINE_EXECUTORCH,
                                               "this/model/does/not/exist.pte",
                                               {{1, 1, static_cast<int64_t>(k_size)}},
                                               {{1, 1, static_cast<int64_t>(k_size)}})),
                 std::runtime_error);
    EXPECT_FALSE(adapter->prepared());
}

// A method name the program does not carry must fail the same way rather than silently
// falling back to forward().
TEST(AdapterExecuTorch, UnknownModelFunctionThrowsRuntimeError) {
    const std::shared_ptr<Rig> adapter = executorch_adapter();
    try {
        adapter->prepare(model_record(ANIRA_ENGINE_EXECUTORCH,
                                      multifunction_model_path(),
                                      {{1, 1, static_cast<int64_t>(k_size)}},
                                      {{1, 1, static_cast<int64_t>(k_size)}},
                                      0,
                                      "no_such_method"));
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
    const auto record_of = [](const std::vector<int64_t>& block) {
        return model_record(ANIRA_ENGINE_EXECUTORCH, gain_model_path(), {block, {1}}, {block, {1}});
    };
    const std::shared_ptr<Rig> adapter = executorch_adapter();
    adapter->prepare(record_of({1, 1, 512}));
    EXPECT_TRUE(adapter->prepared());
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));

    try {
        executorch_adapter()->prepare(record_of({1, 1, 70000}));
        FAIL() << "a block above the planned bound bound";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("executorch: input tensor"), std::string::npos) << message;
        EXPECT_NE(message.find("65536 is the planned upper bound of 70000"), std::string::npos)
            << message;
    }
    try {
        executorch_adapter()->prepare(record_of({1, 512}));
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
    const std::shared_ptr<anira::engine::BuiltinEngine> engine =
        anira::engine::make_builtin_engine(ANIRA_ENGINE_EXECUTORCH);
    ASSERT_NE(engine, nullptr);
    anira_init_info info = ANIRA_INIT_INFO_INIT;
    info.log_level = static_cast<uint32_t>(anira::get_log_level());
    engine->ensure_init(info);
    int workspace_sharing_mode = -1;
    bool weight_cache_enabled = true;
    ASSERT_TRUE(
        anira::engine::executorch_xnnpack_options(workspace_sharing_mode, weight_cache_enabled))
        << "the registered backend answers";
    EXPECT_EQ(workspace_sharing_mode, 0) << "a workspace per delegate instance";
    EXPECT_FALSE(weight_cache_enabled);
}

#endif  // USE_EXECUTORCH
