// The C++ face of anira/compat/v3_to_v2.h: the overloads over the anira.hpp handles return
// the 2.x object and throw anira::Error with the translator's message. The rules themselves
// are test_Translate's; here each overload is exercised once, plus what only the C++ face
// adds (the Hard aggregate minted on the way, the thrown message).

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using anira::ContractHandle;
using anira::Hard;
using anira::MachineConfig;
using anira::ModelConfig;
using anira::TensorSpec;

constexpr const char* k_custom = "anira.v2.custom";

TensorSpec streamed(std::string_view name, int64_t time = 512) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, time);
    spec.window(time, time, 0);
    return spec;
}

ModelConfig minimal() {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    return model;
}

Hard explicit_hard() {
    Hard hard;
    hard.budget = ANIRA_BUDGET_EXPLICIT;
    hard.budget_value = std::chrono::milliseconds(5);
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = 1;
    return hard;
}

struct Thrown {
    bool m_thrown = false;
    anira_status m_status = ANIRA_OK;
    std::string m_what;
};

template <class Body>
Thrown thrown_by(Body&& body) {
    try {
        body();
    } catch (const anira::Error& error) {
        return Thrown{.m_thrown = true, .m_status = error.status, .m_what = error.what()};
    }
    return Thrown{};
}

}  // namespace

TEST(AbiBridge, InferenceConfigFromAHardAggregateAndFromAHandle) {
    const ModelConfig model = minimal();
    const anira::InferenceConfig from_aggregate =
        anira::v3compat::to_inference_config(model, explicit_hard());
    EXPECT_FLOAT_EQ(from_aggregate.m_max_inference_time, 5.0F);
    EXPECT_EQ(from_aggregate.m_warm_up, 1U);
    EXPECT_EQ(from_aggregate.get_tensor_input_shape(), (anira::TensorShapeList{{1, 1, 512}}));

    const ContractHandle contract(explicit_hard());
    const anira::InferenceConfig from_handle =
        anira::v3compat::to_inference_config(model, contract);
    EXPECT_EQ(from_handle, from_aggregate);
}

TEST(AbiBridge, CandidatesNarrowTheEntries) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    model.input(streamed("in"));
    model.output(streamed("out"));
    const std::array<anira::Engine, 1> custom_only{ANIRA_ENGINE_NONE};
    const anira::InferenceConfig cfg =
        anira::v3compat::to_inference_config(model, explicit_hard(), custom_only);
    EXPECT_EQ(cfg.m_model_data.size(), 1U);
    EXPECT_EQ(cfg.m_model_data[0].m_backend, anira::InferenceBackend::CUSTOM);
}

TEST(AbiBridge, TheDefaultHardIsNotSupportedAndTheErrorCarriesTheReason) {
    const ModelConfig model = minimal();
    const Thrown thrown = thrown_by([&] { anira::v3compat::to_inference_config(model, Hard{}); });
    ASSERT_TRUE(thrown.m_thrown);
    EXPECT_EQ(thrown.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(thrown.m_what.find("MEASURED budget"), std::string::npos) << thrown.m_what;
}

TEST(AbiBridge, AConfigRuleThrowsConfigNamingTheTensor) {
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in").latency(4));
    model.output(streamed("out"));
    const Thrown thrown =
        thrown_by([&] { anira::v3compat::to_inference_config(model, explicit_hard()); });
    ASSERT_TRUE(thrown.m_thrown);
    EXPECT_EQ(thrown.m_status, ANIRA_ERROR_CONFIG);
    EXPECT_NE(thrown.m_what.find("tensor 'in'"), std::string::npos) << thrown.m_what;
}

TEST(AbiBridge, AForeignCustomEngineIsNotSupported) {
    ModelConfig model;
    model.add_model_path("de.tu-berlin.coreml", "model.mlmodelc");
    model.input(streamed("in"));
    model.output(streamed("out"));
    const Thrown thrown =
        thrown_by([&] { anira::v3compat::to_inference_config(model, explicit_hard()); });
    ASSERT_TRUE(thrown.m_thrown);
    EXPECT_EQ(thrown.m_status, ANIRA_ERROR_NOT_SUPPORTED);
}

TEST(AbiBridge, ALayoutYieldsABackendShape) {
    ModelConfig model;
    const uint32_t row = model.add_model_path(k_custom, "model.custom");
    model.tensor_layout(row, "in", std::array{0U, 2U, 1U});
    model.input(streamed("in", 15380));
    model.output(streamed("out", 2048));
    const anira::InferenceConfig cfg = anira::v3compat::to_inference_config(model, explicit_hard());
    EXPECT_EQ(cfg.get_tensor_input_shape(anira::InferenceBackend::CUSTOM),
              (anira::TensorShapeList{{1, 15380, 1}}));
    EXPECT_EQ(cfg.get_tensor_input_shape(), (anira::TensorShapeList{{1, 1, 15380}}));
}

TEST(AbiBridge, BorrowedBytesKeepTheirPointer) {
    static constexpr std::array<std::byte, 8> k_bytes{};
    ModelConfig model;
    model.add_model_bytes(k_custom, k_bytes, ANIRA_BYTES_BORROW);
    model.input(streamed("in"));
    model.output(streamed("out"));
    const anira::InferenceConfig cfg = anira::v3compat::to_inference_config(model, explicit_hard());
    ASSERT_EQ(cfg.m_model_data.size(), 1U);
    EXPECT_EQ(cfg.m_model_data[0].m_data, static_cast<const void*>(k_bytes.data()));
    EXPECT_TRUE(cfg.m_model_data[0].m_is_binary);
}

TEST(AbiBridge, ContextConfigFromTheMachineHandle) {
    MachineConfig machine;
    machine.threads(2, ANIRA_WAIT_BLOCKING)
        .log_level(ANIRA_LOG_ERROR)
        .log_drain(ANIRA_LOG_DRAIN_MANUAL, 25)
        .log_queue_capacity(1024);
    const anira::ContextConfig context = anira::v3compat::to_context_config(machine);
    EXPECT_EQ(context.m_num_threads, 2U);
    EXPECT_EQ(context.m_wait_strategy, anira::WaitStrategy::Blocking);
    EXPECT_EQ(context.m_log.m_level, anira::LogLevel::Error);
    EXPECT_EQ(context.m_log.m_drain, anira::LogDrain::Manual);
    EXPECT_EQ(context.m_log.m_drain_interval_ms, 25U);
    EXPECT_EQ(context.m_log.m_queue_capacity, 1024U);

    MachineConfig unknown;
    unknown.ext_json("de.example.unknown", R"({"version": 1})");
    const Thrown thrown = thrown_by([&] { anira::v3compat::to_context_config(unknown); });
    ASSERT_TRUE(thrown.m_thrown);
    EXPECT_EQ(thrown.m_status, ANIRA_ERROR_EXTENSION_UNKNOWN);
}

TEST(AbiBridge, HostConfigFromTheHardGeometryAndFromTheHostsOwn) {
    const ModelConfig model = minimal();
    Hard hard = explicit_hard();
    hard.block_min = 1;
    hard.block_max = 512;
    hard.rate = 48000;
    EXPECT_EQ(anira::v3compat::to_host_config(hard, model),
              anira::HostConfig(512.F, 48000.F, true, anira::HostConfig::k_first_streamable, true));
    const ContractHandle contract(hard);
    EXPECT_EQ(anira::v3compat::to_host_config(contract, model),
              anira::HostConfig(512.F, 48000.F, true, anira::HostConfig::k_first_streamable, true));
    EXPECT_EQ(anira::v3compat::to_host_config(model, 1024.F / 2048.F, 48000.F / 2048.F),
              anira::HostConfig(0.5F, 48000.F / 2048.F, false));

    const Thrown thrown =
        thrown_by([&] { anira::v3compat::to_host_config(explicit_hard(), model); });
    ASSERT_TRUE(thrown.m_thrown);
    EXPECT_EQ(thrown.m_status, ANIRA_ERROR_CONFIG);
    EXPECT_NE(thrown.m_what.find("geometry missing"), std::string::npos) << thrown.m_what;
}

TEST(AbiBridge, EnabledEnginesAndTheV2Default) {
    const std::vector<anira::Engine> engines = anira::v3compat::enabled_engines();
    for (const anira::Engine engine : engines) { EXPECT_NE(engine, ANIRA_ENGINE_NONE); }
    EXPECT_GE(anira::v3compat::v2_default_instances(), 1U);
}
