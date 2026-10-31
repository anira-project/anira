// The bundled configuration files, extras/models/**/*.model.json and *.contract.json: each one
// loads as a 3.x document, and what the validator derives for it under its contract equals what
// the 2.x fixture header of the model used to spell out by hand (the literals below are those
// headers: the tensor shapes, the TensorFlow layout, the channels, the hops, the latencies, the
// budget, the warm-up, whether the model is stateful); the three builders the benchmarks sweep
// with equal their file at the default size.

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../../extras/models/cnn/CNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/model_files.h"
#include "../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "capi/ext_registry.h"
#include "capi/handles.h"
#include "capi/validate.h"
#include "handler_support.h"

namespace {

using Shapes = std::vector<std::vector<int64_t>>;

constexpr std::array<anira_engine, 5> k_every_engine{ANIRA_ENGINE_LIBTORCH,
                                                     ANIRA_ENGINE_ONNXRUNTIME,
                                                     ANIRA_ENGINE_TFLITE,
                                                     ANIRA_ENGINE_LITERT,
                                                     ANIRA_ENGINE_EXECUTORCH};
constexpr std::array<anira_engine, 1> k_libtorch_only{ANIRA_ENGINE_LIBTORCH};

/// What the 2.x fixture header of a model said.
struct Oracle {
    Shapes m_in;  ///< the universal shapes (the PyTorch export's order)
    Shapes m_out;
    std::optional<std::pair<Shapes, Shapes>> m_tensorflow;  ///< when they differ
    std::vector<size_t> m_in_channels;
    std::vector<size_t> m_out_channels;
    std::vector<size_t> m_in_size;
    std::vector<size_t> m_out_size;
    std::vector<size_t> m_latency;
    float m_max_ms = 0.0F;
    unsigned m_warm_up = 0;
    bool m_stateful = false;
    std::string m_model_function;
    std::span<const anira_engine> m_engines;  ///< the engines the file names
};

/// One model entry of this build that survived the candidates: its engine, the engine's
/// extents of either side (its layout applied), its file, its entry point and whether it is
/// bytes.
struct RowFacts {
    anira_engine m_engine = ANIRA_ENGINE_NONE;
    Shapes m_in;
    Shapes m_out;
    std::string m_path;
    std::string m_entry;
    bool m_bytes = false;

    bool operator==(const RowFacts& other) const = default;
};

/// What the validator derives for a model under its contract, one candidate per engine of
/// this build on the CPU path.
struct Facts {
    Shapes m_in;
    Shapes m_out;
    std::vector<RowFacts> m_rows;
    std::vector<size_t> m_in_channels;
    std::vector<size_t> m_out_channels;
    std::vector<size_t> m_in_size;
    std::vector<size_t> m_out_size;
    std::vector<size_t> m_latency;
    double m_max_ms = 0.0;
    unsigned m_warm_up = 0;
    bool m_stateful = false;
    uint32_t m_instances = 0;

    bool operator==(const Facts& other) const = default;
};

Facts facts_of(const anira::ModelConfig& model, const anira::ContractHandle& contract) {
    std::vector<anira_backend_id> candidates;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        candidates.push_back({.struct_size = sizeof(anira_backend_id),
                              .engine = static_cast<uint32_t>(engine),
                              .provider = ANIRA_PROVIDER_CPU,
                              .engine_id = nullptr});
    }
    const anira_model_config& cfg = *model.native();
    const anira::capi::EngineFacts no_engines;
    anira::capi::Derived derived;
    anira::capi::validate(cfg,
                          contract.native(),
                          candidates.data(),
                          static_cast<uint32_t>(candidates.size()),
                          derived,
                          nullptr,
                          &no_engines);
    Facts facts;
    for (size_t i = 0; i < cfg.m_inputs.size(); ++i) {
        const anira::capi::DerivedSpec& spec = derived.m_inputs[i];
        facts.m_in.push_back(spec.m_dims);
        facts.m_in_channels.push_back(static_cast<size_t>(spec.m_channels));
        facts.m_in_size.push_back(static_cast<size_t>(spec.m_hop));
    }
    for (size_t i = 0; i < cfg.m_outputs.size(); ++i) {
        const anira::capi::DerivedSpec& spec = derived.m_outputs[i];
        facts.m_out.push_back(spec.m_dims);
        facts.m_out_channels.push_back(static_cast<size_t>(spec.m_channels));
        facts.m_out_size.push_back(static_cast<size_t>(spec.m_hop));
        facts.m_latency.push_back(static_cast<size_t>(cfg.m_outputs[i].m_latency));
    }
    for (const size_t index : derived.m_rows) {
        const anira::capi::ModelEntry& row = cfg.m_models[index];
        RowFacts row_facts{.m_engine = row.m_engine,
                           .m_path = row.m_path,
                           .m_bytes = row.has_bytes()};
        const auto dims_of = [&row](const std::vector<anira_tensor_spec>& specs,
                                    const std::vector<anira::capi::DerivedSpec>& derived_specs) {
            Shapes shapes;
            for (size_t i = 0; i < specs.size(); ++i) {
                const auto binding = row.m_tensors.find(specs[i].m_name);
                shapes.push_back(anira::capi::engine_dims_of(specs[i],
                                                             derived_specs[i],
                                                             binding == row.m_tensors.end()
                                                                 ? std::vector<uint32_t>{}
                                                                 : binding->second.m_layout));
            }
            return shapes;
        };
        row_facts.m_in = dims_of(cfg.m_inputs, derived.m_inputs);
        row_facts.m_out = dims_of(cfg.m_outputs, derived.m_outputs);
        if (const auto* entry = row.m_ext.payload<anira::capi::EntryPayload>("entry")) {
            row_facts.m_entry = entry->m_name;
        }
        facts.m_rows.push_back(std::move(row_facts));
    }
    const anira::capi::HardContract* hard = contract.native()->hard();
    if (hard != nullptr) {
        facts.m_max_ms = hard->m_budget_ms;
        facts.m_warm_up = hard->m_warmup == ANIRA_WARMUP_FIXED ? hard->m_warmup_iterations : 0U;
    }
    facts.m_stateful = cfg.m_state == ANIRA_MODEL_STATEFUL ||
                       std::ranges::any_of(cfg.m_inputs, [](const anira_tensor_spec& spec) {
                           return spec.m_role == ANIRA_ROLE_STATE;
                       });
    facts.m_instances = cfg.m_max_instances;
    return facts;
}

/// The engines of this build the file names; empty means the validator refuses the file here.
std::vector<anira_engine> usable(std::span<const anira_engine> named) {
    std::vector<anira_engine> out;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        if (std::ranges::find(named, engine) != named.end()) { out.push_back(engine); }
    }
    return out;
}

bool is_tensorflow(anira_engine engine) {
    return engine == ANIRA_ENGINE_TFLITE || engine == ANIRA_ENGINE_LITERT;
}

void expect_matches(const Facts& facts, const Oracle& oracle) {
    EXPECT_EQ(facts.m_in, oracle.m_in);
    EXPECT_EQ(facts.m_out, oracle.m_out);
    EXPECT_EQ(facts.m_rows.size(), usable(oracle.m_engines).size())
        << "one entry per engine of this build the file names";
    for (const RowFacts& row : facts.m_rows) {
        const bool tensorflow = is_tensorflow(row.m_engine) && oracle.m_tensorflow.has_value();
        EXPECT_EQ(row.m_in, tensorflow ? oracle.m_tensorflow->first : oracle.m_in)
            << "engine " << static_cast<int>(row.m_engine);
        EXPECT_EQ(row.m_out, tensorflow ? oracle.m_tensorflow->second : oracle.m_out)
            << "engine " << static_cast<int>(row.m_engine);
        EXPECT_FALSE(row.m_bytes);
        EXPECT_EQ(row.m_entry, oracle.m_model_function);
    }
    EXPECT_EQ(facts.m_in_channels, oracle.m_in_channels);
    EXPECT_EQ(facts.m_out_channels, oracle.m_out_channels);
    EXPECT_EQ(facts.m_in_size, oracle.m_in_size);
    EXPECT_EQ(facts.m_out_size, oracle.m_out_size);
    EXPECT_EQ(facts.m_latency, oracle.m_latency);
    EXPECT_FLOAT_EQ(static_cast<float>(facts.m_max_ms), oracle.m_max_ms);
    EXPECT_EQ(facts.m_warm_up, oracle.m_warm_up);
    EXPECT_EQ(facts.m_stateful, oracle.m_stateful);
    // The files set no instance ceiling: the pool's thread count, resolved by the core at
    // prepare, and 1 for a stateful model whatever the count says (the session's exclusive
    // dispatch). ThePrepareResolvesTheFilesInstanceCount pins both resolutions.
    EXPECT_EQ(facts.m_instances, ANIRA_INSTANCES_AUTO);
}

/// Loads and derives one model and checks it against its oracle; skips when this build has no
/// engine the file names (the validator refuses such a file, which test_Translate covers).
void check_files(const char* model_json, const char* contract_json, const Oracle& oracle) {
    if (usable(oracle.m_engines).empty()) {
        GTEST_SKIP() << "this build has no engine " << model_json << " names";
    }
    const anira::ModelConfig loaded = anira::ModelConfig::from_file(model_json);
    EXPECT_FALSE(loaded.upgraded()) << "a 3.x document";
    expect_matches(facts_of(loaded, anira::ContractHandle::from_file(contract_json)), oracle);
}

/// The builder at its default size equals the file, so the benchmarks sweep the same model.
void check_builder(const char* model_json,
                   const char* contract_json,
                   const anira::ModelConfig& built,
                   std::span<const anira_engine> engines) {
    if (usable(engines).empty()) {
        GTEST_SKIP() << "this build has no engine " << model_json << " names";
    }
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    EXPECT_EQ(facts_of(anira::ModelConfig::from_file(model_json), contract),
              facts_of(built, contract));
}

Oracle cnn_oracle(int64_t window, unsigned warm_up) {
    return Oracle{.m_in = {{1, 1, window}},
                  .m_out = {{1, 1, 2048}},
                  .m_tensorflow = std::pair{Shapes{{1, window, 1}}, Shapes{{1, 2048, 1}}},
                  .m_in_channels = {1},
                  .m_out_channels = {1},
                  .m_in_size = {2048},
                  .m_out_size = {2048},
                  .m_latency = {0},
                  .m_max_ms = 42.66F,
                  .m_warm_up = warm_up,
                  .m_engines = k_every_engine};
}

// The TensorFlow export holds the static gain at rank 3 ([1, 1, 1]): the tflite and litert
// rows carry a layout for it, which the bridge turns into the backend-qualified shape.
Oracle gain_oracle(int64_t channels) {
    return Oracle{.m_in = {{1, channels, 512}, {1}},
                  .m_out = {{1, channels, 512}, {1}},
                  .m_tensorflow = std::pair{Shapes{{1, channels, 512}, {1, 1, 1}},
                                            Shapes{{1, channels, 512}, {1}}},
                  .m_in_channels = {static_cast<size_t>(channels), 1},
                  .m_out_channels = {static_cast<size_t>(channels), 1},
                  .m_in_size = {512, 0},
                  .m_out_size = {512, 0},
                  .m_latency = {0, 0},
                  .m_max_ms = 5.0F,
                  .m_warm_up = 1,
                  .m_engines = k_every_engine};
}

}  // namespace

TEST(ExtrasFixtures, Cnn) {
    check_files(k_cnn_model_json, k_cnn_contract_json, cnn_oracle(15380, 2));
}
TEST(ExtrasFixtures, MediumCnn) {
    check_files(k_medium_cnn_model_json, k_medium_cnn_contract_json, cnn_oracle(3380, 0));
}
TEST(ExtrasFixtures, SmallCnn) {
    check_files(k_small_cnn_model_json, k_small_cnn_contract_json, cnn_oracle(2180, 0));
}

TEST(ExtrasFixtures, HybridNn) {
    check_files(k_hybridnn_model_json,
                k_hybridnn_contract_json,
                Oracle{.m_in = {{256, 1, 150}},
                       .m_out = {{256, 1}},
                       .m_tensorflow = std::pair{Shapes{{256, 150, 1}}, Shapes{{256, 1}}},
                       .m_in_channels = {1},
                       .m_out_channels = {1},
                       .m_in_size = {256},
                       .m_out_size = {256},
                       .m_latency = {0},
                       .m_max_ms = 5.33F,
                       .m_warm_up = 3,
                       .m_engines = k_every_engine});
}

TEST(ExtrasFixtures, StatefulRnn) {
    check_files(k_rnn_model_json,
                k_rnn_contract_json,
                Oracle{.m_in = {{2048, 1, 1}},
                       .m_out = {{2048, 1, 1}},
                       .m_tensorflow = std::pair{Shapes{{1, 2048, 1}}, Shapes{{1, 2048, 1}}},
                       .m_in_channels = {1},
                       .m_out_channels = {1},
                       .m_in_size = {2048},
                       .m_out_size = {2048},
                       .m_latency = {0},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 2,
                       .m_stateful = true,
                       .m_engines = k_every_engine});
}

// The instance count the files leave to anira (ANIRA_INSTANCES_AUTO, expect_matches), as a
// prepare resolves it on a pool of three threads: the stateful RNN's session is exclusive, one
// inference at a time on one instance; the stateless gain model's session runs as many
// instances as the pool has threads.
TEST(ExtrasFixtures, ThePrepareResolvesTheFilesInstanceCount) {
    if (usable(k_every_engine).empty()) { GTEST_SKIP() << "no engine of this build"; }
    ASSERT_EQ(anira::Core::get_num_sessions(), 0) << "a previous test left a session";
    static constexpr uint32_t k_threads = 3;
    const anira_test::Context context(k_threads);
    const auto resolved =
        [&context](const char* model_json, const char* contract_json, uint32_t block) {
            anira_test::Handler handler(context, anira::ModelConfig::from_file(model_json));
            EXPECT_EQ(handler.prepare(anira_test::file_contract(contract_json, block)), ANIRA_OK)
                << handler.m_err.message;
            const std::shared_ptr<anira::SessionElement> session =
                anira_test::session_of(handler.m_handler);
            EXPECT_NE(session, nullptr) << model_json;
            EXPECT_EQ(anira::Core::get_thread_pool_size(), k_threads);
            return session == nullptr ? std::pair{false, 0U}
                                      : std::pair{session->exclusive(), session->m_num_instances};
        };
    EXPECT_EQ(resolved(k_rnn_model_json, k_rnn_contract_json, 2048), std::pair(true, 1U))
        << "the stateful RNN: exclusive, one instance";
    EXPECT_EQ(resolved(k_gain_model_json, k_gain_contract_json, 512), std::pair(false, k_threads))
        << "the gain model: one instance per pool thread";
}

TEST(ExtrasFixtures, Gain) {
    check_files(k_gain_model_json, k_gain_contract_json, gain_oracle(1));
}
TEST(ExtrasFixtures, StereoGain) {
    check_files(k_stereo_gain_model_json, k_stereo_gain_contract_json, gain_oracle(2));
}

// The bundled StatefulAccumulatorNetwork is a 3.x-only file: the 2.x bridge refuses a State
// spec, so there is no 2.x oracle, and the file is checked field by field. A row for every
// engine: the TFLite export orders its outputs [state_out, processed_data] in the file while
// its signature keeps the declared mapping under the keys output_0 / output_1, so the tflite
// row binds by position through the signature runner's key order and the litert row names the
// two outputs (LiteRT lists them in the file's order); every row's file in the fetched tree,
// the State pair around the stereo stream, and the contract file's budget and warm-up.
TEST(ExtrasFixtures, StatefulAccumulator) {
    const anira::ModelConfig loaded =
        anira::ModelConfig::from_file(k_stateful_accumulator_model_json);
    EXPECT_FALSE(loaded.upgraded()) << "a 3.x document";
    const anira_model_config& cfg = *loaded.native();
    ASSERT_EQ(cfg.m_models.size(), 5U);
    EXPECT_EQ(cfg.m_models[0].m_engine, ANIRA_ENGINE_LIBTORCH);
    EXPECT_EQ(cfg.m_models[1].m_engine, ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_EQ(cfg.m_models[2].m_engine, ANIRA_ENGINE_EXECUTORCH);
    EXPECT_EQ(cfg.m_models[3].m_engine, ANIRA_ENGINE_TFLITE);
    EXPECT_EQ(cfg.m_models[4].m_engine, ANIRA_ENGINE_LITERT);
    EXPECT_TRUE(cfg.m_models[3].m_tensors.empty()) << "the runner's key order is the declared one";
    ASSERT_EQ(cfg.m_models[4].m_tensors.size(), 2U);
    EXPECT_EQ(cfg.m_models[4].m_tensors.at("processed_data").m_name, "output_0");
    EXPECT_EQ(cfg.m_models[4].m_tensors.at("state_out").m_name, "output_1");
    for (const anira::capi::ModelEntry& row : cfg.m_models) {
        EXPECT_NE(row.m_path.find("StatefulAccumulatorNetwork/models/"
                                  "stateful_accumulator_network_stereo."),
                  std::string::npos)
            << row.m_path;
        EXPECT_TRUE(std::filesystem::exists(row.m_path)) << row.m_path;
    }

    ASSERT_EQ(cfg.m_inputs.size(), 2U);
    ASSERT_EQ(cfg.m_outputs.size(), 2U);
    const auto expect_axes =
        [](const anira_tensor_spec& spec, anira_axis_tag last, int64_t extent) {
            ASSERT_EQ(spec.m_ndim, 3U);
            EXPECT_EQ(spec.m_axes[0].m_tag, ANIRA_AXIS_BATCH);
            EXPECT_EQ(spec.m_axes[0].m_extent, 1);
            EXPECT_EQ(spec.m_axes[1].m_tag, ANIRA_AXIS_CHANNEL);
            EXPECT_EQ(spec.m_axes[1].m_extent, 2);
            EXPECT_EQ(spec.m_axes[2].m_tag, last);
            EXPECT_EQ(spec.m_axes[2].m_extent, extent);
            EXPECT_EQ(spec.m_dtype, ANIRA_DTYPE_F32);
        };
    const auto expect_state =
        [&](const anira_tensor_spec& spec, const char* name, const char* source) {
            EXPECT_EQ(spec.m_name, name);
            EXPECT_EQ(spec.m_role, ANIRA_ROLE_STATE);
            expect_axes(spec, ANIRA_AXIS_ANY, 2);
            EXPECT_EQ(spec.m_window_max, 0) << "no window on a State spec";
            EXPECT_EQ(spec.m_state_source, source);
        };
    const auto expect_stream = [&](const anira_tensor_spec& spec, const char* name) {
        EXPECT_EQ(spec.m_name, name);
        EXPECT_EQ(spec.m_role, ANIRA_ROLE_STREAMED);
        expect_axes(spec, ANIRA_AXIS_TIME, 64);
        EXPECT_EQ(spec.m_window_min, 64);
        EXPECT_EQ(spec.m_window_max, 64);
        EXPECT_EQ(spec.m_overlap, 0);
        EXPECT_TRUE(spec.m_state_source.empty());
    };
    expect_state(cfg.m_inputs[0], "state_in", "state_out");
    expect_stream(cfg.m_inputs[1], "data");
    expect_stream(cfg.m_outputs[0], "processed_data");
    expect_state(cfg.m_outputs[1], "state_out", "");

    const anira::ContractHandle contract =
        anira::ContractHandle::from_file(k_stateful_accumulator_contract_json);
    EXPECT_FALSE(contract.upgraded());
    const anira::capi::HardContract* hard = contract.native()->hard();
    ASSERT_NE(hard, nullptr);
    EXPECT_EQ(hard->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(hard->m_budget_ms, 5.0);
    EXPECT_EQ(hard->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(hard->m_warmup_iterations, 1U);
}

TEST(ExtrasFixtures, RaveFunkDrum) {
    check_files(k_rave_funk_drum_model_json,
                k_rave_funk_drum_contract_json,
                Oracle{.m_in = {{1, 1, 2048}},
                       .m_out = {{1, 1, 2048}},
                       .m_in_channels = {1},
                       .m_out_channels = {1},
                       .m_in_size = {2048},
                       .m_out_size = {2048},
                       .m_latency = {2048},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 5,
                       .m_stateful = true,
                       .m_engines = k_libtorch_only});
}

TEST(ExtrasFixtures, RaveFunkDrumEncoder) {
    check_files(k_rave_funk_drum_encoder_model_json,
                k_rave_funk_drum_encoder_contract_json,
                Oracle{.m_in = {{1, 1, 2048}},
                       .m_out = {{1, 4, 1}},
                       .m_in_channels = {1},
                       .m_out_channels = {4},
                       .m_in_size = {2048},
                       .m_out_size = {1},
                       .m_latency = {0},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 5,
                       .m_stateful = true,
                       .m_model_function = "encode",
                       .m_engines = k_libtorch_only});
}

TEST(ExtrasFixtures, RaveFunkDrumDecoder) {
    check_files(k_rave_funk_drum_decoder_model_json,
                k_rave_funk_drum_decoder_contract_json,
                Oracle{.m_in = {{1, 4, 1}},
                       .m_out = {{1, 1, 2048}},
                       .m_in_channels = {4},
                       .m_out_channels = {1},
                       .m_in_size = {1},
                       .m_out_size = {2048},
                       .m_latency = {2048},
                       .m_max_ms = 42.66F,
                       .m_warm_up = 5,
                       .m_stateful = true,
                       .m_model_function = "decode",
                       .m_engines = k_libtorch_only});
}

// The builders the benchmarks sweep with, at their default size.
TEST(ExtrasFixtures, CnnBuildersEqualTheFiles) {
    check_builder(k_cnn_model_json, k_cnn_contract_json, cnn_model_config(), k_every_engine);
}
TEST(ExtrasFixtures, MediumCnnBuilderEqualsTheFile) {
    check_builder(k_medium_cnn_model_json,
                  k_medium_cnn_contract_json,
                  cnn_model_config(2048, CnnSize::Medium),
                  k_every_engine);
}
TEST(ExtrasFixtures, SmallCnnBuilderEqualsTheFile) {
    check_builder(k_small_cnn_model_json,
                  k_small_cnn_contract_json,
                  cnn_model_config(2048, CnnSize::Small),
                  k_every_engine);
}
TEST(ExtrasFixtures, HybridNnBuilderEqualsTheFile) {
    check_builder(k_hybridnn_model_json,
                  k_hybridnn_contract_json,
                  hybridnn_model_config(),
                  k_every_engine);
}
TEST(ExtrasFixtures, RnnBuilderEqualsTheFile) {
    check_builder(k_rnn_model_json, k_rnn_contract_json, rnn_model_config(), k_every_engine);
}

// A swept builder changes the hop and nothing else.
TEST(ExtrasFixtures, SweptBuildersFollowTheHop) {
    if (usable(k_every_engine).empty()) { GTEST_SKIP() << "no engine in this build"; }
    const anira::ContractHandle cnn_contract =
        anira::ContractHandle::from_file(k_cnn_contract_json);
    const Facts cnn = facts_of(cnn_model_config(512), cnn_contract);
    EXPECT_EQ(cnn.m_in, (Shapes{{1, 1, 512 + 13332}}));
    EXPECT_EQ(cnn.m_in_size, (std::vector<size_t>{512}));
    EXPECT_EQ(cnn.m_out_size, (std::vector<size_t>{512}));

    const anira::ContractHandle hybrid_contract =
        anira::ContractHandle::from_file(k_hybridnn_contract_json);
    const Facts hybrid = facts_of(hybridnn_model_config(1024), hybrid_contract);
    EXPECT_EQ(hybrid.m_in, (Shapes{{1024, 1, 150}}));
    EXPECT_EQ(hybrid.m_in_size, (std::vector<size_t>{1024}));

    const anira::ContractHandle rnn_contract =
        anira::ContractHandle::from_file(k_rnn_contract_json);
    const Facts rnn = facts_of(rnn_model_config(256), rnn_contract);
    EXPECT_EQ(rnn.m_in, (Shapes{{256, 1, 1}}));
    EXPECT_EQ(rnn.m_in_size, (std::vector<size_t>{256}));
}
