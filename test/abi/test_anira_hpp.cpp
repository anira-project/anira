// The C++20 face of the configuration ABI (anira/anira.hpp): the RAII handles, the
// builders and the aggregates, each a thin wrapper over one C entry of anira/abi/config.h.
// What lands in the C handles is read through src/capi/handles.h, as test_Handles does.

#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "capi/ext_registry.h"
#include "capi/handles.h"
#include "fixtures.h"

namespace {

using anira::ContextConfig;
using anira::ContractHandle;
using anira::JobOptionsHandle;
using anira::ModelConfig;
using anira::TensorSpec;

/// What a body threw: the status and the text of the anira::Error, or m_thrown == false.
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

/// A streamed float32 spec with [batch 1, channel 1, time <time>] axes and a fixed window.
TensorSpec streamed(std::string_view name, int64_t time = 512) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, time);
    spec.window(time, time, 0);
    return spec;
}

/// The two-entry, three-tensor model the builder tests share.
ModelConfig build_model() {
    ModelConfig model;
    const uint32_t torch = model.add_model_path(ANIRA_ENGINE_LIBTORCH, "model.pt");
    model.model_ext(torch, anira::ext::Entry{"decode"});
    const uint32_t tflite = model.add_model_path(ANIRA_ENGINE_TFLITE, "model.tflite");
    model.tensor_name(tflite, "audio_in", "args_0");
    model.tensor_layout(tflite, "audio_in", std::array{0u, 2u, 1u});
    model.input(streamed("audio_in"));
    model.input(TensorSpec("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC).axis(0, ANIRA_AXIS_ANY, 1));
    model.output(streamed("audio_out").latency(8));
    model.default_engine(ANIRA_ENGINE_LIBTORCH)
        .state(ANIRA_MODEL_STATEFUL)
        .max_instances(2)
        .anchor("audio_out");
    return model;
}

/// Moves a handle twice, by construction and by assignment over a live handle: the native
/// pointer travels, the sources are emptied.
template <class Handle>
void expect_move_semantics(Handle first, Handle second) {
    static_assert(!std::is_copy_constructible_v<Handle>);
    static_assert(!std::is_copy_assignable_v<Handle>);
    static_assert(std::is_nothrow_move_constructible_v<Handle>);
    static_assert(std::is_nothrow_move_assignable_v<Handle>);
    const auto* native = first.native();
    ASSERT_NE(native, nullptr);
    ASSERT_NE(second.native(), nullptr);
    Handle moved(std::move(first));
    EXPECT_EQ(moved.native(), native) << "the destination owns the handle";
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move) under test
    EXPECT_EQ(first.native(), nullptr) << "the source is empty";
    second = std::move(moved);  // the handle second held is destroyed here
    EXPECT_EQ(second.native(), native) << "move-assignment adopts the handle";
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move) under test
    EXPECT_EQ(moved.native(), nullptr);
}

/// A directory under <temp>/anira-hpp-test/<name>, removed on destruction. One per test, so
/// the tests stay independent under ctest -j.
struct ScratchDir {
    explicit ScratchDir(const char* name)
        : m_dir(std::filesystem::temp_directory_path() / "anira-hpp-test" / name) {
        std::filesystem::create_directories(m_dir);
    }
    ~ScratchDir() {
        std::error_code ignored;
        std::filesystem::remove_all(m_dir, ignored);
    }
    ScratchDir(const ScratchDir&) = delete;
    ScratchDir& operator=(const ScratchDir&) = delete;

    std::filesystem::path write(const char* name, std::string_view text) const {
        const std::filesystem::path file = m_dir / name;
        std::ofstream out(file, std::ios::binary);
        out << text;
        return file;
    }

    std::filesystem::path m_dir;
};

}  // namespace

// ---- RAII and move ---------------------------------------------------------------------------

TEST(AbiCxx, HandlesAreMoveOnlyAndMoveTheNativePointer) {
    expect_move_semantics(TensorSpec("a", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED),
                          TensorSpec("b", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC));
    expect_move_semantics(ModelConfig(), ModelConfig());
    expect_move_semantics(ContextConfig(), ContextConfig());
    expect_move_semantics(ContractHandle(anira::Hard{}), ContractHandle(anira::Async{}));
    expect_move_semantics(JobOptionsHandle(), JobOptionsHandle());
}

TEST(AbiCxx, MoveCarriesTheUpgradedFlagAndTheHandleContents) {
    ModelConfig source = ModelConfig::from_json(anira_test::k_simple_gain_v2);
    ASSERT_TRUE(source.upgraded());
    const ModelConfig moved(std::move(source));
    EXPECT_TRUE(moved.upgraded());
    EXPECT_EQ(moved.model_count(), 5u);
    ContextConfig target;
    target = ContextConfig::from_json(anira_test::k_rave_v2);
    EXPECT_TRUE(target.upgraded());
    EXPECT_EQ(target.native()->m_num_threads, 2u);
    ContractHandle contract = ContractHandle::from_json(anira_test::k_simple_gain_v2);
    EXPECT_TRUE(contract.upgraded());
    contract = ContractHandle(anira::Async{});
    EXPECT_FALSE(contract.upgraded()) << "assignment takes the source's flag";
    EXPECT_EQ(contract.kind(), ANIRA_CONTRACT_ASYNC);
}

TEST(AbiCxx, ReleaseHandsTheContractOutAndEmptiesTheHandle) {
    ContractHandle handle(anira::Hard{.block_min = 64, .block_max = 64, .rate = 48000.0});
    anira_contract* raw = handle.release();
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(handle.native(), nullptr);
    EXPECT_EQ(anira_contract_get_kind(raw), ANIRA_CONTRACT_HARD);
    const ContractHandle adopted(raw);
    EXPECT_EQ(adopted.native(), raw) << "the adopting constructor takes ownership back";
}

// ---- error mapping ---------------------------------------------------------------------------

TEST(AbiCxx, SetterFailuresThrowErrorNamingTheEntry) {
    TensorSpec spec("x", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    const Thrown axis = thrown_by([&] { spec.axis(ANIRA_MAX_RANK, ANIRA_AXIS_TIME, 1); });
    EXPECT_TRUE(axis.m_thrown);
    EXPECT_EQ(axis.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(axis.m_what.find("anira_tensor_spec_set_axis"), std::string::npos) << axis.m_what;
    EXPECT_NE(axis.m_what.find(anira_status_string(ANIRA_ERROR_INVALID_ARGUMENT)),
              std::string::npos)
        << axis.m_what;
    EXPECT_EQ(spec.native()->m_ndim, 0u) << "a rejected call leaves the spec as it was";

    const Thrown ratio = thrown_by([&] { spec.time_ratio(1, 0); });
    EXPECT_TRUE(ratio.m_thrown);
    EXPECT_EQ(ratio.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(ratio.m_what.find("anira_tensor_spec_set_time_ratio"), std::string::npos)
        << ratio.m_what;
    EXPECT_THROW(spec.latency(-1), std::runtime_error) << "an Error is a runtime_error";
    EXPECT_NO_THROW(spec.time_ratio(0, 0)) << "(0, 0) = derive";
}

TEST(AbiCxx, LoaderFailuresCarryTheParsersMessage) {
    const Thrown malformed = thrown_by([] { ModelConfig::from_json("{not json"); });
    EXPECT_TRUE(malformed.m_thrown);
    EXPECT_EQ(malformed.m_status, ANIRA_ERROR_JSON);
    EXPECT_NE(malformed.m_what.find("malformed"), std::string::npos) << malformed.m_what;

    const Thrown key_path = thrown_by(
        [] { ModelConfig::from_json(R"({"models": [{"engine": "foo", "path": "x"}]})"); });
    EXPECT_EQ(key_path.m_status, ANIRA_ERROR_JSON);
    EXPECT_NE(key_path.m_what.find("models[0].engine"), std::string::npos) << key_path.m_what;

    const Thrown context =
        thrown_by([] { ContextConfig::from_json(R"({"wait_strategy": "nap"})"); });
    EXPECT_EQ(context.m_status, ANIRA_ERROR_JSON);
    EXPECT_NE(context.m_what.find("wait_strategy"), std::string::npos) << context.m_what;

    const Thrown contract =
        thrown_by([] { ContractHandle::from_json(R"({"hard": {}, "async": {}})"); });
    EXPECT_EQ(contract.m_status, ANIRA_ERROR_JSON);
    EXPECT_NE(contract.m_what.find("exactly one root"), std::string::npos) << contract.m_what;
}

TEST(AbiCxx, ModelPathDistinguishesABytesEntryFromABadIndex) {
    ModelConfig model;
    const std::array<std::byte, 4> blob{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    const uint32_t bytes_index = model.add_model_bytes(ANIRA_ENGINE_LIBTORCH, blob);
    const uint32_t path_index = model.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, "model.onnx");
    EXPECT_EQ(bytes_index, 0u);
    EXPECT_EQ(path_index, 1u);
    EXPECT_EQ(model.model_path(path_index), "model.onnx");
    EXPECT_EQ(model.model_bytes(bytes_index).size(), blob.size());
    EXPECT_NE(model.model_bytes(bytes_index).data(), blob.data()) << "COPY holds its own bytes";

    const Thrown bytes = thrown_by([&] { static_cast<void>(model.model_path(bytes_index)); });
    EXPECT_TRUE(bytes.m_thrown);
    EXPECT_EQ(bytes.m_status, ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(bytes.m_what.find("anira_model_config_model_path"), std::string::npos)
        << bytes.m_what;
    const Thrown range = thrown_by([&] { static_cast<void>(model.model_path(999)); });
    EXPECT_TRUE(range.m_thrown);
    EXPECT_EQ(range.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    const Thrown path_bytes = thrown_by([&] { static_cast<void>(model.model_bytes(path_index)); });
    EXPECT_EQ(path_bytes.m_status, ANIRA_ERROR_INVALID_STATE) << "the mirror image";
    EXPECT_NE(path_bytes.m_what.find("anira_model_config_model_bytes"), std::string::npos)
        << path_bytes.m_what;
}

// ---- builders and JSON -----------------------------------------------------------------------

TEST(AbiCxx, BuildersLandInTheHandle) {
    const ModelConfig model = build_model();
    const anira_model_config& cfg = *model.native();
    ASSERT_EQ(cfg.m_models.size(), 2u);
    EXPECT_EQ(cfg.m_models[0].m_engine, ANIRA_ENGINE_LIBTORCH);
    const auto* entry = cfg.m_models[0].m_ext.payload<anira::capi::EntryPayload>("entry");
    ASSERT_NE(entry, nullptr) << "model_ext deep-copied the entry record";
    EXPECT_EQ(entry->m_name, "decode");
    EXPECT_EQ(cfg.m_models[1].m_engine, ANIRA_ENGINE_TFLITE);
    ASSERT_EQ(cfg.m_models[1].m_tensors.count("audio_in"), 1u);
    EXPECT_EQ(cfg.m_models[1].m_tensors.at("audio_in").m_name, "args_0");
    EXPECT_EQ(cfg.m_models[1].m_tensors.at("audio_in").m_layout, (std::vector<uint32_t>{0, 2, 1}));
    ASSERT_EQ(cfg.m_inputs.size(), 2u);
    EXPECT_EQ(cfg.m_inputs[0].m_name, "audio_in");
    EXPECT_EQ(cfg.m_inputs[0].m_ndim, 3u);
    EXPECT_EQ(cfg.m_inputs[0].m_axes[2].m_tag, ANIRA_AXIS_TIME);
    EXPECT_EQ(cfg.m_inputs[0].m_axes[2].m_extent, 512);
    EXPECT_EQ(cfg.m_inputs[0].m_window_min, 512);
    EXPECT_EQ(cfg.m_inputs[0].m_window_max, 512);
    EXPECT_EQ(cfg.m_inputs[1].m_role, ANIRA_ROLE_STATIC);
    ASSERT_EQ(cfg.m_outputs.size(), 1u);
    EXPECT_EQ(cfg.m_outputs[0].m_latency, 8);
    EXPECT_EQ(cfg.m_default_engine, ANIRA_ENGINE_LIBTORCH);
    EXPECT_EQ(cfg.m_state, ANIRA_MODEL_STATEFUL);
    EXPECT_EQ(cfg.m_max_instances, 2u);
    EXPECT_EQ(cfg.m_anchor, "audio_out");
    EXPECT_EQ(model.model_count(), 2u);
    EXPECT_EQ(model.model_engine(0), ANIRA_ENGINE_LIBTORCH);
    EXPECT_TRUE(model.model_engine_id(0).empty()) << "a built-in engine has no id";
    EXPECT_EQ(model.model_path(1), "model.tflite");
}

TEST(AbiCxx, BuildersRoundTripThroughToJsonByteStably) {
    const ModelConfig model = build_model();
    const std::string text = model.to_json();
    constexpr std::array<std::string_view, 11> k_fragments = {
        R"("entry": {)",
        R"("name": "decode")",
        R"("tensors": {)",
        R"("audio_in": {)",
        R"("name": "args_0")",
        R"("layout": [)",
        R"("anchor": "audio_out")",
        R"("max_instances": 2)",
        R"("state": "stateful")",
        R"("default_engine": "libtorch")",
        R"("latency": 8)",
    };
    for (const std::string_view fragment : k_fragments) {
        EXPECT_NE(text.find(fragment), std::string::npos) << fragment << " missing in:\n" << text;
    }
    EXPECT_EQ(text.find("\"tensors\""), text.rfind("\"tensors\""))
        << "only the tflite entry carries a tensor record";
    const ModelConfig again = ModelConfig::from_json(text);
    EXPECT_FALSE(again.upgraded());
    EXPECT_EQ(again.to_json(), text) << "a written file reads back to the same text";
    EXPECT_EQ(again.model_count(), 2u);
    EXPECT_EQ(again.model_path(0), "model.pt") << "no base_dir: paths stay as written";
    EXPECT_EQ(again.native()->m_models[1].m_tensors.at("audio_in").m_layout,
              (std::vector<uint32_t>{0, 2, 1}));
}

TEST(AbiCxx, ToJsonGrowsPastTheErrorMessageCapacity) {
    ModelConfig model;
    model.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, "m.onnx");
    constexpr int k_inputs = 24;
    for (int i = 0; i < k_inputs; ++i) { model.input(streamed("input_" + std::to_string(i))); }
    const std::string text = model.to_json();
    EXPECT_GT(text.size(), 4096u) << "well past ANIRA_ERROR_MESSAGE_CAPACITY (512)";
    EXPECT_EQ(text.find('\0'), std::string::npos) << "the NUL is not part of the text";
    EXPECT_EQ(text.back(), '}');
    const ModelConfig again = ModelConfig::from_json(text);
    EXPECT_EQ(again.native()->m_inputs.size(), static_cast<size_t>(k_inputs));
    EXPECT_EQ(again.to_json(), text);
}

// ---- the 2.x upgrade -------------------------------------------------------------------------

TEST(AbiCxx, TheVersionTwoUpgradeHoldsBackTheLegacyContractOnce) {
    ModelConfig gain = ModelConfig::from_json(anira_test::k_simple_gain_v2);
    EXPECT_TRUE(gain.upgraded());
    EXPECT_EQ(gain.model_count(), 5u);
    EXPECT_EQ(gain.model_engine(0), ANIRA_ENGINE_LIBTORCH);
    EXPECT_EQ(gain.model_engine(4), ANIRA_ENGINE_EXECUTORCH);
    std::optional<ContractHandle> legacy = gain.take_legacy_contract();
    ASSERT_TRUE(legacy.has_value());
    // The handle, or an empty one; has_value() was asserted above.
    const ContractHandle contract = std::move(legacy).value_or(ContractHandle{nullptr});
    ASSERT_NE(contract.native(), nullptr);
    EXPECT_EQ(contract.kind(), ANIRA_CONTRACT_HARD);
    EXPECT_TRUE(contract.upgraded()) << "the product of a 2.x document, like from_json of one";
    EXPECT_TRUE(contract.native()->m_legacy);
    const anira::capi::HardContract* hard = contract.native()->hard();
    ASSERT_NE(hard, nullptr);
    EXPECT_EQ(hard->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_DOUBLE_EQ(hard->m_budget_ms, 5.0);
    EXPECT_EQ(hard->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(hard->m_warmup_iterations, 1u);
    EXPECT_FALSE(gain.take_legacy_contract().has_value()) << "a second take yields nothing";
}

TEST(AbiCxx, AVersionThreeDocumentIsNotUpgradedAndCarriesNoLegacyContract) {
    ModelConfig model = ModelConfig::from_json(anira_test::k_model_v3, "/base");
    EXPECT_FALSE(model.upgraded());
    EXPECT_EQ(model.model_count(), 3u);
    EXPECT_EQ(model.model_path(0), "/base/model.onnx") << "relative paths resolve against base_dir";
    EXPECT_EQ(model.model_path(2), "/abs/model.mlpackage") << "absolute paths stay";
    EXPECT_EQ(model.model_engine_id(2), "de.tu-berlin.coreml");
    EXPECT_FALSE(model.take_legacy_contract().has_value());
}

// ---- contracts -------------------------------------------------------------------------------

TEST(AbiCxx, ContractHandleMintsAHardAggregate) {
    const anira::Hard hard{
        .block_min = 256,
        .block_max = 512,
        .rate = 44100.0,
        .budget = ANIRA_BUDGET_EXPLICIT,
        .budget_value = std::chrono::milliseconds{42} + std::chrono::microseconds{660},
        .warmup = ANIRA_WARMUP_FIXED,
        .warmup_iterations = 3,
        .on_miss = ANIRA_MISS_ZEROS,
        .wait_ratio = 0.25,
        .edge_cost = ANIRA_EDGE_COST_STRICT,
    };
    ContractHandle handle(hard);
    EXPECT_EQ(handle.kind(), ANIRA_CONTRACT_HARD);
    EXPECT_FALSE(handle.upgraded());
    const anira::capi::HardContract* fields = handle.native()->hard();
    ASSERT_NE(fields, nullptr);
    EXPECT_EQ(fields->m_block_min, 256u);
    EXPECT_EQ(fields->m_block_max, 512u);
    EXPECT_DOUBLE_EQ(fields->m_rate, 44100.0);
    EXPECT_EQ(fields->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_DOUBLE_EQ(fields->m_budget_ms, 42.66) << "42 ms 660 us as milliseconds";
    EXPECT_EQ(fields->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(fields->m_warmup_iterations, 3u);
    EXPECT_EQ(fields->m_on_miss, ANIRA_MISS_ZEROS);
    EXPECT_DOUBLE_EQ(fields->m_wait_ratio, 0.25);
    EXPECT_EQ(handle.native()->m_edge_cost, ANIRA_EDGE_COST_STRICT);

    handle.hard_geometry(1, 512, 48000.0);
    const anira::capi::HardContract* patched = handle.native()->hard();
    ASSERT_NE(patched, nullptr);
    EXPECT_EQ(patched->m_block_min, 1u);
    EXPECT_EQ(patched->m_block_max, 512u);
    EXPECT_DOUBLE_EQ(patched->m_rate, 48000.0);
    EXPECT_DOUBLE_EQ(patched->m_budget_ms, 42.66) << "the geometry patch leaves the rest";

    const ContractHandle defaults{anira::Hard{}};
    const anira::capi::HardContract* zero = defaults.native()->hard();
    ASSERT_NE(zero, nullptr);
    EXPECT_EQ(zero->m_budget, ANIRA_BUDGET_MEASURED);
    EXPECT_EQ(zero->m_warmup, ANIRA_WARMUP_UNTIL_STABLE);
    EXPECT_EQ(zero->m_on_miss, ANIRA_MISS_BYPASS);
    EXPECT_DOUBLE_EQ(zero->m_wait_ratio, 0.0);
    EXPECT_EQ(defaults.native()->m_edge_cost, ANIRA_EDGE_COST_PERMISSIVE);
}

TEST(AbiCxx, ContractHandleMintsAnAsyncAggregate) {
    const anira::Async with_deadline{
        .deadline = std::chrono::milliseconds{5},
        .on_late = ANIRA_LATE_DROP,
        .priority = ANIRA_PRIORITY_INTERACTIVE,
        .lanes = 2,
        .max_in_flight = 4,
        .delivery = ANIRA_DELIVERY_IMMEDIATE,
        .edge_cost = ANIRA_EDGE_COST_STRICT,
    };
    const ContractHandle handle(with_deadline);
    EXPECT_EQ(handle.kind(), ANIRA_CONTRACT_ASYNC);
    const anira::capi::AsyncContract* fields = handle.native()->asynchronous();
    ASSERT_NE(fields, nullptr);
    EXPECT_DOUBLE_EQ(fields->m_deadline_ms, 5.0);
    EXPECT_EQ(fields->m_on_late, ANIRA_LATE_DROP);
    EXPECT_EQ(fields->m_priority, ANIRA_PRIORITY_INTERACTIVE);
    EXPECT_EQ(fields->m_lanes, 2u);
    EXPECT_EQ(fields->m_max_in_flight, 4u);
    EXPECT_EQ(fields->m_delivery, ANIRA_DELIVERY_IMMEDIATE);
    EXPECT_EQ(handle.native()->m_edge_cost, ANIRA_EDGE_COST_STRICT);

    const ContractHandle offline{anira::Async{}};
    EXPECT_EQ(offline.kind(), ANIRA_CONTRACT_ASYNC);
    const anira::capi::AsyncContract* none = offline.native()->asynchronous();
    ASSERT_NE(none, nullptr);
    EXPECT_DOUBLE_EQ(none->m_deadline_ms, -1.0) << "no deadline: the offline posture";
    EXPECT_EQ(none->m_on_late, ANIRA_LATE_FINISH);
    EXPECT_EQ(none->m_priority, ANIRA_PRIORITY_AUTO);
    EXPECT_EQ(none->m_delivery, ANIRA_DELIVERY_POLLED);

    const ContractHandle variant{
        anira::Contract{anira::Async{.deadline = std::chrono::microseconds{33300}}}};
    EXPECT_EQ(variant.kind(), ANIRA_CONTRACT_ASYNC);
    ASSERT_NE(variant.native()->asynchronous(), nullptr);
    EXPECT_DOUBLE_EQ(variant.native()->asynchronous()->m_deadline_ms, 33.3);

    const Thrown wrong =
        thrown_by([&] { ContractHandle(anira::Async{}).hard_geometry(1, 1, 1.0); });
    EXPECT_TRUE(wrong.m_thrown);
    EXPECT_EQ(wrong.m_status, ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_NE(wrong.m_what.find("anira_contract_hard_set_geometry"), std::string::npos)
        << wrong.m_what;
}

TEST(AbiCxx, ContractHandleFromJsonReportsAnUpgrade) {
    const ContractHandle hard = ContractHandle::from_json(anira_test::k_contract_hard_v3);
    EXPECT_EQ(hard.kind(), ANIRA_CONTRACT_HARD);
    EXPECT_FALSE(hard.upgraded());
    ASSERT_NE(hard.native()->hard(), nullptr);
    EXPECT_EQ(hard.native()->hard()->m_block_max, 512u);
    EXPECT_FALSE(hard.native()->m_legacy);

    const ContractHandle asynchronous = ContractHandle::from_json(anira_test::k_contract_async_v3);
    EXPECT_EQ(asynchronous.kind(), ANIRA_CONTRACT_ASYNC);
    EXPECT_FALSE(asynchronous.upgraded());
    ASSERT_NE(asynchronous.native()->asynchronous(), nullptr);
    EXPECT_DOUBLE_EQ(asynchronous.native()->asynchronous()->m_deadline_ms, 33.3);
    EXPECT_EQ(asynchronous.native()->m_edge_cost, ANIRA_EDGE_COST_STRICT);

    const ContractHandle legacy = ContractHandle::from_json(anira_test::k_simple_gain_v2);
    EXPECT_EQ(legacy.kind(), ANIRA_CONTRACT_HARD);
    EXPECT_TRUE(legacy.upgraded()) << "a 2.x document yields the legacy Hard directly";
    EXPECT_TRUE(legacy.native()->m_legacy);
    ASSERT_NE(legacy.native()->hard(), nullptr);
    EXPECT_DOUBLE_EQ(legacy.native()->hard()->m_budget_ms, 5.0);
}

// ---- context config --------------------------------------------------------------------------

TEST(AbiCxx, ContextConfigSettersLandInTheHandle) {
    ContextConfig context;
    const anira_context_config& defaults = *context.native();
    EXPECT_EQ(defaults.m_num_threads, ANIRA_THREADS_AUTO);
    EXPECT_EQ(defaults.m_wait, ANIRA_WAIT_SPIN_BACKOFF);
    EXPECT_EQ(defaults.m_log_level, ANIRA_LOG_WARNING);
    EXPECT_FALSE(context.upgraded());

    context.threads(2, ANIRA_WAIT_BLOCKING)
        .log_level(ANIRA_LOG_ERROR)
        .log_drain(ANIRA_LOG_DRAIN_MANUAL, 25)
        .log_queue_capacity(1024);
    const anira_context_config& fields = *context.native();
    EXPECT_EQ(fields.m_num_threads, 2u);
    EXPECT_EQ(fields.m_wait, ANIRA_WAIT_BLOCKING);
    EXPECT_EQ(fields.m_log_level, ANIRA_LOG_ERROR);
    EXPECT_EQ(fields.m_log_drain, ANIRA_LOG_DRAIN_MANUAL);
    EXPECT_EQ(fields.m_drain_interval_ms, 25u);
    EXPECT_EQ(fields.m_queue_capacity, 1024u);
    const std::string text = context.to_json();
    EXPECT_NE(text.find("\"num_threads\": 2"), std::string::npos) << text;
    EXPECT_NE(text.find("\"drain\": \"manual\""), std::string::npos) << text;
}

TEST(AbiCxx, ContextConfigFromJsonRoundTripsByteStably) {
    const ContextConfig context = ContextConfig::from_json(anira_test::k_context_v3);
    EXPECT_FALSE(context.upgraded());
    EXPECT_EQ(context.native()->m_num_threads, 0u) << "0 = bring your own threads";
    EXPECT_EQ(context.native()->m_queue_capacity, 512u);
    EXPECT_TRUE(context.native()->m_cuda.has_value());
    EXPECT_TRUE(context.native()->m_vulkan.has_value());
    EXPECT_FALSE(context.native()->m_d3d12.has_value());
    const std::string once = context.to_json();
    EXPECT_EQ(ContextConfig::from_json(once).to_json(), once);

    const ContextConfig legacy = ContextConfig::from_json(anira_test::k_rave_v2);
    EXPECT_TRUE(legacy.upgraded()) << "a 2.x context_config upgrades";
    EXPECT_EQ(legacy.native()->m_log_level, ANIRA_LOG_ERROR);
}

// ---- job options -----------------------------------------------------------------------------

TEST(AbiCxx, JobOptionsHandleLandsTheAggregate) {
    const JobOptionsHandle defaults;
    EXPECT_TRUE(defaults.native()->m_head_trim.empty());
    EXPECT_TRUE(defaults.native()->m_tail_flush);
    EXPECT_EQ(defaults.native()->m_below_min, ANIRA_PAD_REJECT);

    const JobOptionsHandle options(
        anira::JobOptions{.head_trim = {1, 2}, .tail_flush = false, .below_min = ANIRA_PAD_ZEROS});
    const anira_job_options& fields = *options.native();
    EXPECT_EQ(fields.m_head_trim, (std::vector<int64_t>{1, 2}));
    EXPECT_FALSE(fields.m_tail_flush);
    EXPECT_EQ(fields.m_below_min, ANIRA_PAD_ZEROS);
    EXPECT_TRUE(fields.m_borrowed_ext.empty());

    const Thrown below = thrown_by([] { JobOptionsHandle(anira::JobOptions{.head_trim = {-2}}); });
    EXPECT_TRUE(below.m_thrown) << "a trim below -1 is refused";
    EXPECT_EQ(below.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(below.m_what.find("anira_job_options_set_head_trim"), std::string::npos)
        << below.m_what;
}

TEST(AbiCxx, JobOptionsHandleKeepsTheExtensionRecordAlive) {
    JobOptionsHandle options;
    options.ext(anira::ext::Entry{"x"});
    const anira_job_options& fields = *options.native();
    ASSERT_EQ(fields.m_borrowed_ext.size(), 1u);
    // The C entry borrows the record; the handle owns it, so it is still readable here.
    const anira_ext_header* borrowed = fields.m_borrowed_ext[0];
    ASSERT_NE(borrowed, nullptr);
    EXPECT_EQ(std::string_view(borrowed->kind), "entry");
    EXPECT_EQ(borrowed->version, 1u);
    EXPECT_EQ(borrowed->struct_size, sizeof(anira_ext_entry));
    options.ext(anira::ext::Entry{"y"});
    EXPECT_EQ(fields.m_borrowed_ext.size(), 1u) << "a second set of the kind replaces the slot";
    EXPECT_NE(fields.m_borrowed_ext[0], borrowed) << "with the new record";

    JobOptionsHandle moved(std::move(options));
    EXPECT_EQ(moved.native()->m_borrowed_ext.size(), 1u) << "the kept records move along";
    EXPECT_EQ(std::string_view(moved.native()->m_borrowed_ext[0]->kind), "entry");
}

// ---- files -----------------------------------------------------------------------------------

TEST(AbiCxx, ModelConfigFromFileResolvesPathsAgainstTheFilesDirectory) {
    const ScratchDir scratch("model");
    const std::filesystem::path file =
        scratch.write("model.json",
                      R"({"models": [{"engine": "onnxruntime", "path": "sub/m.onnx"},
                       {"engine": "libtorch", "path": "/abs/m.pt"}],
            "inputs": [{"name": "audio_in", "axes": [["time", "dynamic"]],
                        "window": {"min": 64, "max": 64}}]})");
    const ModelConfig model = ModelConfig::from_file(file);
    EXPECT_FALSE(model.upgraded());
    ASSERT_EQ(model.model_count(), 2u);
    // Resolved paths are joined in generic form: forward slashes on every platform.
    EXPECT_EQ(model.model_path(0),
              (scratch.m_dir / "sub" / "m.onnx").lexically_normal().generic_string());
    EXPECT_EQ(model.model_path(1), "/abs/m.pt") << "a rooted path stays as written";
    EXPECT_EQ(model.native()->m_inputs[0].m_window_min, 64);
}

TEST(AbiCxx, ContextAndContractFromFileReadTheFixtures) {
    const ScratchDir scratch("context-contract");
    const ContextConfig context =
        ContextConfig::from_file(scratch.write("context.json", anira_test::k_context_v3));
    EXPECT_FALSE(context.upgraded());
    EXPECT_EQ(context.to_json(), ContextConfig::from_json(anira_test::k_context_v3).to_json());

    const ContractHandle hard =
        ContractHandle::from_file(scratch.write("hard.json", anira_test::k_contract_hard_v3));
    EXPECT_EQ(hard.kind(), ANIRA_CONTRACT_HARD);
    EXPECT_FALSE(hard.upgraded());
    ASSERT_NE(hard.native()->hard(), nullptr);
    EXPECT_DOUBLE_EQ(hard.native()->hard()->m_rate, 48000.0);

    const ContractHandle asynchronous =
        ContractHandle::from_file(scratch.write("async.json", anira_test::k_contract_async_v3));
    EXPECT_EQ(asynchronous.kind(), ANIRA_CONTRACT_ASYNC);

    const ContractHandle legacy =
        ContractHandle::from_file(scratch.write("v2.json", anira_test::k_rave_v2));
    EXPECT_TRUE(legacy.upgraded());
    ASSERT_NE(legacy.native()->hard(), nullptr);
    EXPECT_DOUBLE_EQ(legacy.native()->hard()->m_budget_ms, 42.66);
}

TEST(AbiCxx, FromFileOnAMissingFileThrowsNoSuchFile) {
    const std::filesystem::path missing =
        std::filesystem::temp_directory_path() / "anira-hpp-test" / "missing" / "nope.json";
    const Thrown model = thrown_by([&] { ModelConfig::from_file(missing); });
    EXPECT_TRUE(model.m_thrown);
    EXPECT_EQ(model.m_status, ANIRA_ERROR_NO_SUCH_FILE);
    EXPECT_NE(model.m_what.find("nope.json"), std::string::npos) << model.m_what;
    const Thrown context = thrown_by([&] { ContextConfig::from_file(missing); });
    EXPECT_TRUE(context.m_thrown);
    EXPECT_EQ(context.m_status, ANIRA_ERROR_NO_SUCH_FILE);
    EXPECT_NE(context.m_what.find("nope.json"), std::string::npos) << context.m_what;
    const Thrown contract = thrown_by([&] { ContractHandle::from_file(missing); });
    EXPECT_TRUE(contract.m_thrown);
    EXPECT_EQ(contract.m_status, ANIRA_ERROR_NO_SUCH_FILE);
    EXPECT_NE(contract.m_what.find("nope.json"), std::string::npos) << contract.m_what;
}

// ---- after the header review ------------------------------------------------------------------

TEST(AbiCxx, JobOptionsExtensionKeepsTheValueItPointsInto) {
    JobOptionsHandle options;
    options.ext(anira::ext::Entry{"decode"});  // a temporary: the handle copies the value
    const anira_job_options& fields = *options.native();
    ASSERT_EQ(fields.m_borrowed_ext.size(), 1u);
    // The borrowed record's name pointer must point into storage the handle owns.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) the record starts with its header
    const auto* entry = reinterpret_cast<const anira_ext_entry*>(fields.m_borrowed_ext[0]);
    EXPECT_STREQ(entry->name, "decode");
    const JobOptionsHandle moved(std::move(options));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* after = reinterpret_cast<const anira_ext_entry*>(moved.native()->m_borrowed_ext[0]);
    EXPECT_STREQ(after->name, "decode") << "the kept value moves with the handle";
}

TEST(AbiCxx, EmptyContractHandleThrowsOnKind) {
    ContractHandle source(anira::Hard{});
    const ContractHandle destination(std::move(source));
    EXPECT_FALSE(destination.empty());
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move) the point of the test
    EXPECT_TRUE(source.empty());
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    const Thrown kind = thrown_by([&] { static_cast<void>(source.kind()); });
    EXPECT_TRUE(kind.m_thrown);
    EXPECT_EQ(kind.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(destination.kind(), ANIRA_CONTRACT_HARD);
}

TEST(AbiCxx, LegacyContractReportsUpgraded) {
    ModelConfig model = ModelConfig::from_json(anira_test::k_simple_gain_v2);
    std::optional<ContractHandle> legacy = model.take_legacy_contract();
    ASSERT_TRUE(legacy.has_value());
    const ContractHandle contract = std::move(legacy).value_or(ContractHandle{nullptr});
    EXPECT_TRUE(contract.upgraded()) << "the product of a 2.x document";
    EXPECT_EQ(contract.kind(), ANIRA_CONTRACT_HARD);
}

TEST(AbiCxx, ContractSettersPatchALoadedContract) {
    ContractHandle hard = ContractHandle::from_json(anira_test::k_contract_hard_v3);
    hard.hard_budget(ANIRA_BUDGET_EXPLICIT, std::chrono::milliseconds(7))
        .hard_warmup(ANIRA_WARMUP_FIXED, 4)
        .hard_on_miss(ANIRA_MISS_ZEROS)
        .hard_wait_ratio(0.5)
        .edge_cost(ANIRA_EDGE_COST_STRICT);
    const anira::capi::HardContract* fields = hard.native()->hard();
    ASSERT_NE(fields, nullptr);
    EXPECT_EQ(fields->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_DOUBLE_EQ(fields->m_budget_ms, 7.0);
    EXPECT_EQ(fields->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(fields->m_warmup_iterations, 4u);
    EXPECT_EQ(fields->m_on_miss, ANIRA_MISS_ZEROS);
    EXPECT_DOUBLE_EQ(fields->m_wait_ratio, 0.5);
    EXPECT_EQ(hard.native()->m_edge_cost, ANIRA_EDGE_COST_STRICT);
    hard.hard_budget(ANIRA_BUDGET_MEASURED);
    EXPECT_EQ(fields->m_budget, ANIRA_BUDGET_MEASURED);

    ContractHandle async(anira::Async{});
    async.async_deadline(std::chrono::milliseconds(3))
        .async_policy(ANIRA_LATE_DROP, ANIRA_PRIORITY_BATCH, 2, 4, ANIRA_DELIVERY_IMMEDIATE);
    const anira::capi::AsyncContract* async_fields = async.native()->asynchronous();
    ASSERT_NE(async_fields, nullptr);
    EXPECT_DOUBLE_EQ(async_fields->m_deadline_ms, 3.0);
    EXPECT_EQ(async_fields->m_on_late, ANIRA_LATE_DROP);
    EXPECT_EQ(async_fields->m_priority, ANIRA_PRIORITY_BATCH);
    EXPECT_EQ(async_fields->m_lanes, 2u);
    EXPECT_EQ(async_fields->m_max_in_flight, 4u);
    EXPECT_EQ(async_fields->m_delivery, ANIRA_DELIVERY_IMMEDIATE);
    async.async_deadline(std::nullopt);
    EXPECT_DOUBLE_EQ(async_fields->m_deadline_ms, -1.0);
    const Thrown wrong = thrown_by([&] { async.hard_warmup(ANIRA_WARMUP_NONE); });
    EXPECT_TRUE(wrong.m_thrown);
    EXPECT_EQ(wrong.m_status, ANIRA_ERROR_WRONG_CONTRACT);
}

TEST(AbiCxx, HardRingDtypeLandsInTheHandle) {
    ContractHandle contract{anira::Hard{}};
    contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_I16);
    EXPECT_EQ(contract.native()->hard()->m_ring_dtypes.at("audio_in"), ANIRA_DTYPE_I16);
    const Thrown empty = thrown_by([&] { contract.hard_ring_dtype("", ANIRA_DTYPE_F32); });
    ASSERT_TRUE(empty.m_thrown);
    EXPECT_EQ(empty.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    ContractHandle async_contract{anira::Async{}};
    const Thrown wrong =
        thrown_by([&] { async_contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_F32); });
    ASSERT_TRUE(wrong.m_thrown);
    EXPECT_EQ(wrong.m_status, ANIRA_ERROR_WRONG_CONTRACT);
}

TEST(AbiCxx, TensorLayoutEmptySpanClears) {
    ModelConfig model;
    const uint32_t tflite = model.add_model_path(ANIRA_ENGINE_TFLITE, "m.tflite");
    model.tensor_layout(tflite, "audio_in", std::array<uint32_t, 3>{0u, 2u, 1u});
    ASSERT_EQ(model.native()->m_models[0].m_tensors.count("audio_in"), 1u);
    model.tensor_layout(tflite, "audio_in", {});
    EXPECT_EQ(model.native()->m_models[0].m_tensors.count("audio_in"), 0u)
        << "an empty span clears the layout, and a record without a name disappears";
}

TEST(AbiCxx, DeviceBlockClearsWithNull) {
    ContextConfig context;
    const anira_cuda_desc cuda = ANIRA_CUDA_DESC_INIT;
    context.cuda(cuda);
    EXPECT_TRUE(context.native()->m_cuda.has_value());
    context.cuda(nullptr);
    EXPECT_FALSE(context.native()->m_cuda.has_value());
}

TEST(AbiCxx, RegisteredExtKindsListsEntry) {
    const std::vector<std::string_view> kinds = anira::registered_ext_kinds();
    EXPECT_NE(std::ranges::find(kinds, "entry"), kinds.end());
}

TEST(AbiCxx, EmptyTextIsAJsonErrorNotANullPointer) {
    const Thrown empty = thrown_by([] { ModelConfig::from_json(std::string_view{}); });
    EXPECT_TRUE(empty.m_thrown);
    EXPECT_EQ(empty.m_status, ANIRA_ERROR_JSON) << empty.m_what;
}

TEST(AbiCxx, ReadingADirectoryIsNoSuchFile) {
    const Thrown directory =
        thrown_by([] { ContextConfig::from_file(std::filesystem::temp_directory_path()); });
    EXPECT_TRUE(directory.m_thrown);
    EXPECT_EQ(directory.m_status, ANIRA_ERROR_NO_SUCH_FILE);
}
