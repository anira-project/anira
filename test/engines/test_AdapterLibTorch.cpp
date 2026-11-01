// The LibTorch adapter, driven directly through the engine room's interface (the built-in
// adapter of the engine, a model record written out, one run over a context built by
// hand: no Context, no threads), on the paths the end-to-end suites never take: a model handed
// over as bytes instead of a path, a named entry (warm-up and steady state), a load failure, a
// module returning a tuple of tensors, the inputs bound to the method's arguments by name, a
// method with a defaulted argument, an output of another shape than the record's, and the
// engine's init pinning OpenBLAS (the Linux aarch64 package's BLAS) to one thread.

#ifdef USE_LIBTORCH

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils/Logger.h"

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include "engine_test_support.h"
#include "engines/Adapter.h"
#include "engines/Adapters.h"
#include "gtest/gtest.h"
#include "utils/StatusError.h"

namespace {

using anira::engine::Model;
using anira_test::any_sample_nonzero;
using anira_test::context_of;
using anira_test::descriptors_of;
using anira_test::filled_buffers;
using anira_test::model_record;
using anira_test::read_model_file;
using anira_test::with_bytes;

std::string rave_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR "/third-party/ircam-acids/RAVE/rave_funk_drum.ts";
}

/// The float16 bits of a small integer (|value| < 2048, exact), and back: the test's own helpers
/// (no vendor header in a test).
uint16_t half_of(int value) {
    if (value == 0) { return 0; }
    const uint16_t sign = value < 0 ? 0x8000U : 0U;
    auto magnitude = static_cast<uint32_t>(value < 0 ? -value : value);
    int exponent = 0;
    while ((magnitude >> exponent) > 1U) { ++exponent; }
    const uint32_t mantissa = (magnitude << (10 - exponent)) & 0x3ffU;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent + 15) << 10) | mantissa);
}

float float_of_half(uint16_t half) {
    const uint32_t exponent = (half >> 10) & 0x1fU;
    if (exponent == 0) { return 0.0F; }
    const float magnitude = std::ldexp(1.0F + static_cast<float>(half & 0x3ffU) / 1024.0F,
                                       static_cast<int>(exponent) - 15);
    return (half & 0x8000U) != 0 ? -magnitude : magnitude;
}

std::string gain_model_path() {
    return ANIRA_EXTRAS_MODELS_DIR
        "/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.pt";
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

std::shared_ptr<Rig> libtorch_adapter() {
    std::shared_ptr<anira::engine::Loaded> loaded = anira::engine::make_builtin_loaded(
        anira::engine::make_builtin_engine(ANIRA_ENGINE_LIBTORCH));
    EXPECT_NE(loaded, nullptr);
    return std::make_shared<Rig>(std::move(loaded));
}

// The gain model's record: the block and the static gain in, the block and the peak out; a
// warm-up of one.
Model gain_model(uint32_t warm_up = 1) {
    return model_record(ANIRA_ENGINE_LIBTORCH,
                        gain_model_path(),
                        {{1, 1, 512}, {1}},
                        {{1, 1, 512}, {1}},
                        warm_up);
}

// One run of `adapter` over the buffers, with descriptors in the record's shapes.
anira_status run_once(Rig& adapter, anira_test::Buffers& inputs, anira_test::Buffers& outputs) {
    const std::vector<anira_tensor> input_tensors =
        descriptors_of(inputs, adapter.model().m_inputs);
    std::vector<anira_tensor> output_tensors = descriptors_of(outputs, adapter.model().m_outputs);
    return adapter.run(context_of(input_tensors, output_tensors), false);
}

}  // namespace

// The TypedGainNetwork TorchScript exports at int16, int8 and float16: the record's dtype on
// every slot, the inputs bound in place (from_blob over the descriptors, no copy counted), the
// outputs, which a method returns as tensors of its own, copied once each; the stream comes out
// doubled exactly and the peak is 16.
TEST(AdapterLibTorch, ATypedModelBindsItsInputsAndCopiesItsOutputs) {
    struct Case {
        const char* m_word;
        anira_dtype m_dtype;
    };
    for (const Case& c : {Case{.m_word = "int16", .m_dtype = ANIRA_DTYPE_I16},
                          Case{.m_word = "int8", .m_dtype = ANIRA_DTYPE_I8},
                          Case{.m_word = "float16", .m_dtype = ANIRA_DTYPE_F16}}) {
        SCOPED_TRACE(c.m_word);
        Model model = model_record(
            ANIRA_ENGINE_LIBTORCH,
            ANIRA_EXTRAS_MODELS_DIR
                "/model-pool/example-models/TypedGainNetwork/models/typed_gain_network_" +
                std::string(c.m_word) + ".pt",
            {{1, 1, 512}, {1}},
            {{1, 1, 512}, {1}},
            1);
        for (auto* side : {&model.m_inputs, &model.m_outputs}) {
            for (anira::engine::TensorInfo& tensor : *side) { tensor.m_dtype = c.m_dtype; }
        }
        const std::shared_ptr<Rig> adapter = libtorch_adapter();
        adapter->prepare(model);
        ASSERT_TRUE(adapter->prepared());

        const size_t element = c.m_dtype == ANIRA_DTYPE_I8 ? 1 : 2;
        const auto put = [&](std::byte* at, int value) {
            if (c.m_dtype == ANIRA_DTYPE_I8) {
                const auto v = static_cast<int8_t>(value);
                std::memcpy(at, &v, 1);
            } else if (c.m_dtype == ANIRA_DTYPE_I16) {
                const auto v = static_cast<int16_t>(value);
                std::memcpy(at, &v, 2);
            } else {
                const uint16_t v = half_of(value);
                std::memcpy(at, &v, 2);
            }
        };
        const auto get = [&](const std::byte* at) {
            if (c.m_dtype == ANIRA_DTYPE_I8) {
                int8_t v = 0;
                std::memcpy(&v, at, 1);
                return static_cast<float>(v);
            }
            if (c.m_dtype == ANIRA_DTYPE_I16) {
                int16_t v = 0;
                std::memcpy(&v, at, 2);
                return static_cast<float>(v);
            }
            uint16_t v = 0;
            std::memcpy(&v, at, 2);
            return float_of_half(v);
        };
        std::vector<std::byte> data(512 * element);
        std::vector<std::byte> gain(element);
        std::vector<std::byte> out(512 * element, std::byte{0x5a});
        std::vector<std::byte> peak(element, std::byte{0x5a});
        for (size_t n = 0; n < 512; ++n) { put(&data[n * element], static_cast<int>(n % 17) - 8); }
        put(gain.data(), 2);
        const std::array<int64_t, 3> block{1, 1, 512};
        const std::array<int64_t, 1> scalar{1};
        std::vector<anira_tensor> inputs(2);
        std::vector<anira_tensor> outputs(2);
        anira_tensor_init_host(&inputs[0], data.data(), c.m_dtype, 3, block.data());
        anira_tensor_init_host(&inputs[1], gain.data(), c.m_dtype, 1, scalar.data());
        anira_tensor_init_host(&outputs[0], out.data(), c.m_dtype, 3, block.data());
        anira_tensor_init_host(&outputs[1], peak.data(), c.m_dtype, 1, scalar.data());

        const uint64_t copies_in = adapter->executor().copies_in();
        const uint64_t copies_out = adapter->executor().copies_out();
        ASSERT_EQ(adapter->run(context_of(inputs, outputs), false), ANIRA_OK);
        EXPECT_EQ(adapter->executor().copies_in() - copies_in, 0U) << "the inputs are bound";
        EXPECT_EQ(adapter->executor().copies_out() - copies_out, 2U) << "one copy per output";
        for (size_t n = 0; n < 512; ++n) {
            ASSERT_EQ(get(&out[n * element]),
                      2.0F * static_cast<float>(static_cast<int>(n % 17) - 8))
                << n;
        }
        EXPECT_EQ(get(peak.data()), 16.0F);
    }
}

// A model supplied as bytes must load through the in-memory branch.
TEST(AdapterLibTorch, BinaryModelDataLoadsFromMemory) {
    const std::vector<char> bytes = read_model_file(rave_model_path());
    ASSERT_FALSE(bytes.empty()) << "fixture missing: " << rave_model_path();

    const Model model =
        with_bytes(model_record(ANIRA_ENGINE_LIBTORCH, "", {{1, 1, 2048}}, {{1, 1, 2048}}), bytes);
    ASSERT_NE(model.m_bytes, nullptr);

    const std::shared_ptr<Rig> adapter = libtorch_adapter();
    adapter->prepare(model);
    ASSERT_TRUE(adapter->prepared());

    anira_test::Buffers input = filled_buffers({2048}, 0.1F);
    anira_test::Buffers output = filled_buffers({2048}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);

    // RAVE is not an identity, so assert the buffer was written at all rather
    // than a specific value.
    EXPECT_TRUE(any_sample_nonzero(output[0]));
}

// A named entry must be used for the warm-up inferences and for every later run, instead of
// forward().
TEST(AdapterLibTorch, NamedModelFunctionIsUsedForWarmUpAndProcessing) {
    const Model model = model_record(ANIRA_ENGINE_LIBTORCH,
                                     rave_model_path(),
                                     {{1, 1, 2048}},
                                     {{1, 4, 1}},
                                     2,
                                     "encode");
    ASSERT_EQ(model.m_entry, "encode");
    ASSERT_EQ(model.m_warm_up, 2U);

    const std::shared_ptr<Rig> adapter = libtorch_adapter();
    adapter->prepare(model);

    anira_test::Buffers input = filled_buffers({2048}, 0.2F);
    anira_test::Buffers output = filled_buffers({4}, 0.F);
    // "encode" produces the 4 latent values; forward() would return a 2048-sample block,
    // which the delivery would refuse against the record's [1, 4, 1].
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);
    EXPECT_EQ(output[0].size(), 4U);
}

// A model that cannot be loaded fails prepare with std::runtime_error (a StatusError of
// MODEL_LOAD or NO_SUCH_FILE) — the contract the other engines honour, and the one
// create_session() rolls back on (see test_CreateSessionFailure.cpp). Carrying on with an
// empty module would let an engine-specific c10 exception escape instead.
TEST(AdapterLibTorch, UnloadableModelThrowsRuntimeError) {
    const std::shared_ptr<Rig> adapter = libtorch_adapter();
    EXPECT_THROW(adapter->prepare(model_record(ANIRA_ENGINE_LIBTORCH,
                                               "this/model/does/not/exist.pt",
                                               {{1, 1, 512}},
                                               {{1, 1, 512}})),
                 std::runtime_error);
    EXPECT_FALSE(adapter->prepared());

    // A method the module does not have fails the same way.
    Model no_method = gain_model(0);
    no_method.m_entry = "no_such_method";
    try {
        adapter->prepare(no_method);
        FAIL() << "a missing method loaded";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_MODEL_LOAD);
        EXPECT_NE(std::string(error.what()).find("no method 'no_such_method'"), std::string::npos)
            << error.what();
    }
}

// The gain fixture returns two tensors, so its result arrives as a tuple rather than a bare
// tensor — the branch that unpacks it per output slot. A record without names binds the
// method's arguments (data, gain) by position.
TEST(AdapterLibTorch, MultiTensorOutputIsUnpacked) {
    ASSERT_FALSE(read_model_file(gain_model_path()).empty())
        << "fixture missing: " << gain_model_path();
    const std::shared_ptr<Rig> adapter = libtorch_adapter();
    adapter->prepare(gain_model());
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));
    EXPECT_EQ(adapter->bindings().m_outputs,
              (std::vector<anira_binding>{ANIRA_BINDING_POSITION, ANIRA_BINDING_POSITION}));

    anira_test::Buffers input = filled_buffers({512, 1}, 0.5F);
    anira_test::Buffers output = filled_buffers({512, 1}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);

    // Both output tensors must have been written, which only happens if the tuple was
    // unpacked element by element: the block times the gain of one half, and its peak.
    for (size_t n = 0; n < 512; ++n) { ASSERT_EQ(output[0][n], 0.25F) << n; }
    EXPECT_EQ(output[1][0], 0.25F);
}

// The method's arguments have names (forward(self, data, gain) here): a record naming the
// slots crosswise binds by name, so the scalar still lands in `gain` and the block in `data`
// whatever the slot order; a record naming an argument the method lacks lists the arguments;
// a record on an output side, which has no names, is refused too.
TEST(AdapterLibTorch, InputsBindToTheMethodsArgumentsByName) {
    Model model = model_record(ANIRA_ENGINE_LIBTORCH,
                               gain_model_path(),
                               {{1}, {1, 1, 512}},
                               {{1, 1, 512}, {1}},
                               1);
    model.m_inputs[0].m_export_name = "gain";
    model.m_inputs[1].m_export_name = "data";
    const std::shared_ptr<Rig> adapter = libtorch_adapter();
    adapter->prepare(model);
    EXPECT_EQ(adapter->bindings().m_inputs,
              (std::vector<anira_binding>{ANIRA_BINDING_NAME, ANIRA_BINDING_NAME}));

    anira_test::Buffers input = filled_buffers({1, 512}, 0.5F);
    for (size_t n = 0; n < 512; ++n) { input[1][n] = static_cast<float>(n) / 512.F; }
    anira_test::Buffers output = filled_buffers({512, 1}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);
    for (size_t n = 0; n < 512; ++n) {
        ASSERT_EQ(output[0][n], 0.5F * (static_cast<float>(n) / 512.F)) << n;
    }

    model.m_inputs[0].m_export_name = "ghost";
    try {
        libtorch_adapter()->prepare(model);
        FAIL() << "a missing argument bound";
    } catch (const anira::StatusError& error) {
        EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
        const std::string message = error.what();
        EXPECT_NE(message.find("'ghost'"), std::string::npos) << message;
        EXPECT_NE(message.find("'data', 'gain'"), std::string::npos) << message;
    }

    model.m_inputs[0].m_export_name = "gain";
    model.m_outputs[0].m_export_name = "processed";
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
TEST(AdapterLibTorch, AMethodWithADefaultedArgumentBindsItsLeadingArgument) {
    const std::shared_ptr<Rig> adapter = libtorch_adapter();
    adapter->prepare(model_record(ANIRA_ENGINE_LIBTORCH,
                                  rave_model_path(),
                                  {{1, 4, 1}},
                                  {{1, 1, 2048}},
                                  1,
                                  "decode"));
    EXPECT_EQ(adapter->bindings().m_inputs, (std::vector<anira_binding>{ANIRA_BINDING_POSITION}));

    anira_test::Buffers input = filled_buffers({4}, 0.1F);
    anira_test::Buffers output = filled_buffers({2048}, 0.F);
    EXPECT_EQ(run_once(*adapter, input, output), ANIRA_OK);
    EXPECT_TRUE(any_sample_nonzero(output[0]));
}

// The schema carries no shapes: an output of another shape than the record's is caught on the
// warm-up's first result at prepare (CONFIG naming both shapes), and, without a warm-up, at
// the first run (the chunk fails with ENGINE; nothing is written).
TEST(AdapterLibTorch, AnOutputOfAnotherShapeIsRefusedAtLoadWithOrWithoutAWarmUp) {
    const Model warmed = model_record(ANIRA_ENGINE_LIBTORCH,
                                      gain_model_path(),
                                      {{1, 1, 512}, {1}},
                                      {{1, 1, 512}, {2}},
                                      1);
    const Model unwarmed = [&warmed] {
        Model record = warmed;
        record.m_warm_up = 0;
        return record;
    }();
    // The schema carries no output shapes, so the check is the first forward's: the warm-up's
    // where the record has one, one run on the probe at load where it has none.
    const auto expect_refused_at_load = [](const Model& record) {
        try {
            libtorch_adapter()->prepare(record);
            ADD_FAILURE() << "a [2] output passed for a [1] return (warm_up " << record.m_warm_up
                          << ")";
        } catch (const anira::StatusError& error) {
            EXPECT_EQ(error.status(), ANIRA_ERROR_CONFIG);
            const std::string message = error.what();
            EXPECT_NE(message.find("libtorch: output tensor"), std::string::npos) << message;
            EXPECT_NE(message.find("[1]"), std::string::npos) << message;
            EXPECT_NE(message.find("[2]"), std::string::npos) << message;
        }
    };
    expect_refused_at_load(warmed);
    expect_refused_at_load(unwarmed);
}

// The engine's init pins OpenBLAS to one thread where the LibTorch build carries it (the Linux
// aarch64 package): torch::set_num_threads(1) leaves OpenBLAS's own pool alone, whose workers
// spin beside anira's inference threads on every core. The pool is widened first, so the test
// does not pass on an OPENBLAS_NUM_THREADS=1 environment alone; a build without OpenBLAS skips.
TEST(AdapterLibTorch, InitPinsOpenBlasToOneThread) {
#if defined(__linux__)
    using SetNumThreads = void (*)(int);
    using GetNumThreads = int (*)();
    void* const set_symbol = dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
    void* const get_symbol = dlsym(RTLD_DEFAULT, "openblas_get_num_threads");
    if (set_symbol == nullptr || get_symbol == nullptr) {
        GTEST_SKIP() << "this LibTorch build carries no OpenBLAS";
    }
    const auto set_num_threads = reinterpret_cast<SetNumThreads>(set_symbol);
    const auto get_num_threads = reinterpret_cast<GetNumThreads>(get_symbol);
    set_num_threads(2);
    ASSERT_EQ(get_num_threads(), 2);

    // A fresh engine object, so its init runs (the object remembers a successful one).
    const std::shared_ptr<anira::engine::BuiltinEngine> engine =
        anira::engine::make_builtin_engine(ANIRA_ENGINE_LIBTORCH);
    ASSERT_NE(engine, nullptr);
    anira_init_info info = ANIRA_INIT_INFO_INIT;
    info.log_level = static_cast<uint32_t>(anira::get_log_level());
    engine->ensure_init(info);
    EXPECT_EQ(get_num_threads(), 1);
#else
    GTEST_SKIP() << "OpenBLAS ships with the Linux aarch64 LibTorch package alone";
#endif
}

#endif  // USE_LIBTORCH
