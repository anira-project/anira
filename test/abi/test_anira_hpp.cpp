// The C++20 face of the configuration ABI (anira/anira.hpp): the RAII handles, the
// builders and the aggregates, each a thin wrapper over one C entry of anira/abi/config.h;
// the custom stage (anira::Stage) and the custom engine (anira::Engine) as classes, run end to
// end through a C-created handler. What lands in the C handles is read through
// src/capi/handles.h, as test_Handles does.

#include <anira/abi/config.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/RingBuffer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "../support/log_record_collector.h"
#include "capi/ext_registry.h"
#include "capi/handles.h"
#include "dlpack_producer.h"
#include "fixtures.h"
#include "handler_support.h"

namespace {

using anira::ContextConfig;
using anira::ContractHandle;
using anira::JobOptionsHandle;
using anira::ModelConfig;
using anira::SyncToken;
using anira::Tensor;
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

// ext::ProviderOptions mints a record that stands on its own (its sets, arrays and strings
// in its own storage), which the context config copies: the JSON form carries the sets.
TEST(AbiCxx, ProviderOptionsExtensionMintsASelfContainedRecord) {
    anira::ext::ProviderOptions value;
    value.sets.push_back({.engine = ANIRA_ENGINE_ONNXRUNTIME,
                          .engine_id = "",
                          .provider = ANIRA_PROVIDER_CUDA,
                          .provider_id = "",
                          .options = {{"device_id", "0"}}});
    value.sets.push_back({.engine = ANIRA_ENGINE_NONE,
                          .engine_id = "org.example.engine",
                          .provider = ANIRA_PROVIDER_DEFAULT,
                          .provider_id = "fast",
                          .options = {}});
    const auto native = anira::detail::ExtTraits<anira::ext::ProviderOptions>::mint(value);
    ASSERT_EQ(native.num_sets, 2u);
    ASSERT_NE(native.sets, nullptr);
    EXPECT_EQ(native.sets[0].engine, static_cast<uint32_t>(ANIRA_ENGINE_ONNXRUNTIME));
    EXPECT_EQ(native.sets[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_CUDA));
    EXPECT_EQ(native.sets[0].engine_id, nullptr);
    ASSERT_EQ(native.sets[0].num_options, 1u);
    EXPECT_STREQ(native.sets[0].keys[0], "device_id");
    EXPECT_STREQ(native.sets[0].values[0], "0");
    EXPECT_STREQ(native.sets[1].engine_id, "org.example.engine");
    EXPECT_STREQ(native.sets[1].provider_id, "fast");
    EXPECT_EQ(native.sets[1].keys, nullptr);
    EXPECT_EQ(native.header.struct_size, sizeof(anira_ext_provider_options));
    EXPECT_STREQ(native.header.kind, "provider_options");

    anira::ContextConfig config;
    config.ext(value);
    const std::string json = config.to_json();
    EXPECT_NE(json.find("\"provider_options\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"onnxruntime:cuda\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"org.example.engine:fast\""), std::string::npos) << json;
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

namespace {
// A backup function of ANIRA_MISS_CALLBACK; declared ANIRA_NONBLOCKING, as clang asks of a
// function converted to anira_miss_fn. Never called here.
anira_status ANIRA_CALL cxx_decline_miss(anira_handler* /*handler*/,
                                         const anira_tensor* /*inputs*/,
                                         uint32_t /*num_inputs*/,
                                         const anira_tensor* /*outputs*/,
                                         uint32_t /*num_outputs*/,
                                         void* /*user_data*/) ANIRA_NONBLOCKING {
    return ANIRA_ERROR_NOT_SUPPORTED;
}
}  // namespace

TEST(AbiCxx, TheMissFunctionTravelsThroughTheAggregateAndTheSetter) {
    int user = 0;
    const anira::Hard hard{
        .block_min = 64,
        .block_max = 64,
        .rate = 48000.0,
        .on_miss = ANIRA_MISS_CALLBACK,
        .miss_fn = &cxx_decline_miss,
        .miss_user_data = &user,
    };
    const ContractHandle minted(hard);
    const anira::capi::HardContract* fields = minted.native()->hard();
    ASSERT_NE(fields, nullptr);
    EXPECT_EQ(fields->m_on_miss, ANIRA_MISS_CALLBACK);
    EXPECT_EQ(fields->m_miss_fn, &cxx_decline_miss);
    EXPECT_EQ(fields->m_miss_user_data, &user);

    const ContractHandle plain{anira::Hard{}};
    EXPECT_EQ(plain.native()->hard()->m_miss_fn, nullptr);

    ContractHandle loaded = ContractHandle::from_json(anira_test::k_contract_hard_v3);
    loaded.hard_on_miss(ANIRA_MISS_CALLBACK).hard_miss_fn(&cxx_decline_miss, &user);
    EXPECT_EQ(loaded.native()->hard()->m_miss_fn, &cxx_decline_miss);
    EXPECT_EQ(loaded.native()->hard()->m_miss_user_data, &user);
    loaded.hard_miss_fn(nullptr, nullptr);
    EXPECT_EQ(loaded.native()->hard()->m_miss_fn, nullptr);

    ContractHandle async_contract{anira::Async{}};
    EXPECT_THROW(async_contract.hard_miss_fn(&cxx_decline_miss, nullptr), anira::Error);
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

// The host-end domain is common to both kinds: it lands in the handle of either, by name; an
// empty name and a domain that is no anira_domain value are refused.
TEST(AbiCxx, HostDomainLandsInTheHandleOfEitherKind) {
    ContractHandle hard{anira::Hard{}};
    hard.host_domain("audio_in", ANIRA_DOMAIN_HOST).host_domain("state_in", ANIRA_DOMAIN_CUDA);
    EXPECT_EQ(hard.native()->m_host_domains.at("audio_in"), ANIRA_DOMAIN_HOST);
    EXPECT_EQ(hard.native()->m_host_domains.at("state_in"), ANIRA_DOMAIN_CUDA);
    ContractHandle async_contract{anira::Async{}};
    async_contract.host_domain("audio_in", ANIRA_DOMAIN_HOST_PINNED);
    EXPECT_EQ(async_contract.native()->m_host_domains.at("audio_in"), ANIRA_DOMAIN_HOST_PINNED);
    const Thrown empty = thrown_by([&] { hard.host_domain("", ANIRA_DOMAIN_HOST); });
    ASSERT_TRUE(empty.m_thrown);
    EXPECT_EQ(empty.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    const Thrown unknown = thrown_by([&] { hard.host_domain("audio_in", ANIRA_DOMAIN_FORCE32); });
    ASSERT_TRUE(unknown.m_thrown);
    EXPECT_EQ(unknown.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(hard.native()->m_host_domains.at("audio_in"), ANIRA_DOMAIN_HOST) << "untouched";
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

// ---- pipeline and plan report ----------------------------------------------------------------

TEST(AbiCxx, PipelineIsMoveOnlyAndHoldsOneInferenceStage) {
    expect_move_semantics(anira::Pipeline(), anira::Pipeline());

    anira::ModelConfig model = anira_test::gain_with_custom();
    anira::Pipeline pipe{anira::stage::Inference(model)};
    EXPECT_NE(pipe.native(), nullptr);
    const Thrown second = thrown_by([&] { pipe.inference(model); });
    EXPECT_TRUE(second.m_thrown);
    EXPECT_EQ(second.m_status, ANIRA_ERROR_CONFIG);
    EXPECT_NE(second.m_what.find("a second inference stage"), std::string::npos) << second.m_what;

    const Thrown variants = thrown_by([&] {
        const anira::Pipeline two{
            anira::stage::Inference({std::cref(model), std::cref(model)}, {})};
    });
    EXPECT_TRUE(variants.m_thrown);
    EXPECT_EQ(variants.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(variants.m_what.find("one variant per inference stage"), std::string::npos)
        << variants.m_what;

    // A candidate's provider is checked for its syntax here (a provider of the enum and a
    // provider_id at once); whether an engine serves it is the handler's question at create.
    const Thrown provider = thrown_by([&] {
        const anira::Pipeline gpu{
            anira::stage::Inference(model,
                                    {anira::BackendId{.struct_size = sizeof(anira::BackendId),
                                                      .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                                      .provider = ANIRA_PROVIDER_CUDA,
                                                      .engine_id = nullptr,
                                                      .provider_id = "com.example.npu"}})};
    });
    EXPECT_TRUE(provider.m_thrown);
    EXPECT_EQ(provider.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(provider.m_what.find("at once"), std::string::npos) << provider.m_what;

    EXPECT_EQ(anira::stage::Inference(model).variants().size(), 1u);
    EXPECT_TRUE(anira::stage::Inference(model).candidates().empty());
}

TEST(AbiCxx, PlanReportRoundTripsOverAPreparedHandler) {
    static_assert(std::is_copy_constructible_v<anira::PlanReport>);
    const anira_test::Context context;
    const anira::ModelConfig model = anira_test::gain_with_custom();
    // The custom row alone: one plan on every leg, the engine-less ones included.
    anira::Pipeline pipe{
        anira::stage::Inference(model,
                                {anira::BackendId{.struct_size = sizeof(anira::BackendId),
                                                  .engine = ANIRA_ENGINE_NONE,
                                                  .provider = ANIRA_PROVIDER_DEFAULT,
                                                  .engine_id = nullptr}})};
    anira_handler* h = nullptr;
    anira_error err{};
    ASSERT_EQ(anira_handler_create(context.m_context, pipe.native(), &h, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_prepare(h, anira_test::explicit_contract().native(), &err), ANIRA_OK)
        << err.message;

    const anira::PlanReport report(anira_handler_plan_report(h));
    EXPECT_EQ(report.num_plans(), 1u);
    const std::vector<anira_plan_info> plans = report.plans();
    ASSERT_EQ(plans.size(), 1u);
    EXPECT_EQ(plans[0].engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE));
    ASSERT_NE(plans[0].engine_id, nullptr);
    EXPECT_EQ(std::string_view(plans[0].engine_id), "anira.v2.custom");
    EXPECT_DOUBLE_EQ(plans[0].budget_ms, 5.0);
    EXPECT_EQ(plans[0].variant, 0u);
    EXPECT_EQ(plans[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));

    const std::vector<anira_plan_slot> inputs = report.slots(0, true);
    ASSERT_EQ(inputs.size(), 2u);
    EXPECT_EQ(inputs[0].slot, 0u);
    EXPECT_EQ(inputs[1].slot, 1u);
    for (const anira_plan_slot& slot : inputs) {
        EXPECT_EQ(slot.is_input, 1u);
        ASSERT_NE(slot.recipe, nullptr);
        EXPECT_EQ(std::string_view(slot.recipe), "host");
    }
    EXPECT_EQ(report.slots(0, false).size(), 2u);
    EXPECT_TRUE(report.extensions(0).empty());

    const Thrown range = thrown_by([&] { report.slots(report.num_plans(), true); });
    EXPECT_TRUE(range.m_thrown);
    EXPECT_EQ(range.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(range.m_what.find("anira_plan_report_slots"), std::string::npos) << range.m_what;

    EXPECT_EQ(anira::PlanReport(nullptr).num_plans(), 0u);
    const Thrown null_report = thrown_by([] { anira::PlanReport(nullptr).plans(); });
    EXPECT_TRUE(null_report.m_thrown);
    EXPECT_EQ(null_report.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(null_report.m_what.find("anira_plan_report_plans"), std::string::npos)
        << null_report.m_what;

    anira_handler_destroy(h);
}

// ---- stages ------------------------------------------------------------------------------------
//
// A phase function of a Stage runs inside an ANIRA_NONBLOCKING entry or on an inference thread:
// the test stages hold no gtest assertion there (a failing EXPECT allocates, which under
// RealtimeSanitizer aborts the process), allocate nothing and write atomics the test reads once
// the block's inference was collected. The streams run in lockstep, one block and then the wait.

namespace {

constexpr size_t k_hop = anira_test::k_block;

/// A mono stream through the engine-free custom row, one hop in and one hop out: BackendBase's
/// process is an exact pass-through, so the output stream is the input behind the latency.
ModelConfig stage_stream_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in", static_cast<int64_t>(k_hop)));
    model.output(streamed("out", static_cast<int64_t>(k_hop)));
    return model;
}

anira::BackendId custom_row() {
    return anira::BackendId{.struct_size = sizeof(anira::BackendId),
                            .engine = ANIRA_ENGINE_NONE,
                            .provider = ANIRA_PROVIDER_DEFAULT,
                            .engine_id = nullptr};
}

/// What prepare of a CountingStage does.
enum class PrepareMode : uint8_t {
    Ok,
    ThrowError,
    ThrowRuntimeError,
    ThrowInt,
    ThrowBadAlloc,
    ThrowBudget,
    ReturnNull
};

/// What one CountingPrepared saw and did, kept by the registration beyond the object's life
/// (anira deletes the Prepared at unprepare; the test reads the counts afterwards).
struct PreparedCounts {
    anira_handler* m_handler = nullptr;    ///< the record's handler
    uint32_t m_num_entries = 0;            ///< the record's count; every entry seen is below it
    uint32_t m_num_entries_by_getter = 0;  ///< anira_handler_num_entries during prepare
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    uint32_t m_plans = 0;
    std::vector<std::string> m_consumers;
    std::vector<std::string> m_input_names;
    std::atomic<int> m_pre{0};
    std::atomic<int> m_before{0};
    std::atomic<int> m_after{0};
    std::atomic<int> m_post{0};
    std::atomic<int> m_resets{0};
    std::atomic<int> m_broken{0};
    std::atomic<bool> m_deleted{false};
    /// One float per entry, this handler's own: the first sample pre_process saw there last.
    std::vector<float> m_scratch;
};

/// The Prepared of one handler: every phase counts itself and notes whether the context answered
/// as its phase promises, asking the legal questions of its phase only (a refused one would be
/// recorded, and the tests read anira_handler_rt_error as ANIRA_OK behind this stage);
/// pre_process and post_process then call the base class, anira's default body; post_process
/// scales the model output first. The scratch is its own, sized by the record's entry count.
class CountingPrepared final : public anira::Stage::Prepared {
public:
    CountingPrepared(std::shared_ptr<PreparedCounts> counts, float scale)
        : m_counts(std::move(counts)), m_scale(scale) {}
    ~CountingPrepared() override { m_counts->m_deleted.store(true); }
    CountingPrepared(const CountingPrepared&) = delete;
    CountingPrepared& operator=(const CountingPrepared&) = delete;
    CountingPrepared(CountingPrepared&&) = delete;
    CountingPrepared& operator=(CountingPrepared&&) = delete;

    anira_status pre_process(anira::StageContext& ctx) noexcept override {
        m_counts->m_pre.fetch_add(1);
        anira::Role in_role = ANIRA_ROLE_FORCE32;
        anira::Role out_role = ANIRA_ROLE_FORCE32;
        anira::RingView ring;
        anira::Tensor tensor{};
        const bool as_promised =
            ctx.phase() == ANIRA_PHASE_PRE_PROCESS && ctx.engine() == ANIRA_ENGINE_NONE &&
            ctx.provider() == ANIRA_PROVIDER_DEFAULT && ctx.variant() == 0 &&
            ctx.num_inputs() == 1 && ctx.num_outputs() == 1 &&
            ctx.ticket() == ANIRA_TICKET_INVALID && ctx.entry() < m_counts->m_num_entries &&
            ctx.input_role(0, in_role) == ANIRA_OK && in_role == ANIRA_ROLE_STREAMED &&
            ctx.output_role(0, out_role) == ANIRA_OK && out_role == ANIRA_ROLE_STREAMED &&
            ctx.input_ring(0, ring) == ANIRA_OK && ring && ring.dtype() == ANIRA_DTYPE_F32 &&
            ring.num_channels() == 1 && ring.available(0) >= k_hop &&
            ctx.input_tensor(0, tensor) == ANIRA_OK && tensor.num_elements() == k_hop &&
            ctx.native() != nullptr;
        if (!as_promised) { m_counts->m_broken.fetch_add(1); }
        const anira_status status = anira::Stage::Prepared::pre_process(ctx);
        // The chunk's first sample into this handler's scratch, at the chunk's entry.
        anira::Tensor filled{};
        if (status == ANIRA_OK && ctx.entry() < m_counts->m_scratch.size() &&
            ctx.input_tensor(0, filled) == ANIRA_OK && filled.data_f32() != nullptr) {
            m_counts->m_scratch[ctx.entry()] = filled.data_f32()[0];
        }
        return status;
    }
    anira_status before_inference(anira::StageContext& ctx) noexcept override {
        m_counts->m_before.fetch_add(1);
        anira::Tensor tensor{};
        const bool as_promised = ctx.phase() == ANIRA_PHASE_BEFORE_INFERENCE &&
                                 ctx.input_tensor(0, tensor) == ANIRA_OK &&
                                 tensor.data_f32() != nullptr;
        if (!as_promised) { m_counts->m_broken.fetch_add(1); }
        return ANIRA_OK;
    }
    anira_status after_inference(anira::StageContext& ctx) noexcept override {
        m_counts->m_after.fetch_add(1);
        anira::Tensor tensor{};
        const bool as_promised = ctx.phase() == ANIRA_PHASE_AFTER_INFERENCE &&
                                 ctx.output_tensor(0, tensor) == ANIRA_OK &&
                                 tensor.data_f32() != nullptr;
        if (!as_promised) { m_counts->m_broken.fetch_add(1); }
        return ANIRA_OK;
    }
    anira_status post_process(anira::StageContext& ctx) noexcept override {
        m_counts->m_post.fetch_add(1);
        anira::RingView ring;
        anira::Tensor tensor{};
        const bool as_promised =
            ctx.phase() == ANIRA_PHASE_POST_PROCESS && ctx.output_ring(0, ring) == ANIRA_OK &&
            ring && ctx.output_tensor(0, tensor) == ANIRA_OK && tensor.num_elements() == k_hop;
        if (!as_promised) {
            m_counts->m_broken.fetch_add(1);
            return ANIRA_ERROR_INTERNAL;
        }
        float* const samples = tensor.data_f32();
        for (size_t n = 0; n < k_hop; ++n) { samples[n] *= m_scale; }
        return anira::Stage::Prepared::post_process(ctx);
    }
    void reset(anira::StageContext& ctx) noexcept override {
        m_counts->m_resets.fetch_add(1);
        // The role answers in RESET; a ring or a tensor would be refused, and recorded.
        anira::Role role = ANIRA_ROLE_FORCE32;
        const bool as_promised =
            ctx.phase() == ANIRA_PHASE_RESET && ctx.input_role(0, role) == ANIRA_OK &&
            role == ANIRA_ROLE_STREAMED && ctx.entry() < m_counts->m_num_entries;
        if (!as_promised) { m_counts->m_broken.fetch_add(1); }
    }

private:
    std::shared_ptr<PreparedCounts> m_counts;
    float m_scale;
};

/// The registration: the mask, the promise, the kinds, what init and prepare do, and the
/// counts of every Prepared it made, in prepare order.
class CountingStage : public anira::Stage {
public:
    explicit CountingStage(uint32_t mask) : m_mask(mask) {}

    uint32_t phases() const noexcept override { return m_mask; }
    /// Both bits by default: every body here allocates nothing and blocks on nothing, and a
    /// Hard contract takes a filled pre_process or post_process only with the promise.
    uint32_t flags() const noexcept override { return m_flags; }

    void init(const anira::InitInfo& info) override {
        ++m_inited;
        m_init_log_level = info.log_level();
        m_init_num_threads = info.num_threads();
        m_init_context = info.context();
        if (m_init_throws) { throw anira::Error(ANIRA_ERROR_NOT_SUPPORTED, "the stage refuses"); }
    }

    std::unique_ptr<anira::Stage::Prepared> prepare(const anira::PrepareInfo& info) override {
        ++m_prepared;
        auto counts = std::make_shared<PreparedCounts>();
        counts->m_handler = info.handler();
        counts->m_num_entries = info.num_entries();
        // The handler counts as prepared while its stage prepares: the getter answers.
        counts->m_num_entries_by_getter = anira_handler_num_entries(info.handler());
        counts->m_num_inputs = static_cast<uint32_t>(info.inputs().size());
        counts->m_num_outputs = static_cast<uint32_t>(info.outputs().size());
        counts->m_plans = info.report().num_plans();
        for (const anira_plan_ext& row : info.report().extensions(0)) {
            counts->m_consumers.emplace_back(row.consumer);
        }
        for (const char* name : info.input_names()) { counts->m_input_names.emplace_back(name); }
        counts->m_scratch.assign(info.num_entries(), -1.0F);
        m_counts.push_back(counts);
        switch (m_mode) {
            case PrepareMode::ThrowError:
                throw anira::Error(ANIRA_ERROR_CONFIG, "the stage refuses this plan");
            case PrepareMode::ThrowRuntimeError:
                throw std::runtime_error("the stage ran out of luck");
            case PrepareMode::ThrowInt:
                throw 7;  // no std::exception: what the trampoline must survive too
            case PrepareMode::ThrowBadAlloc: throw std::bad_alloc();
            case PrepareMode::ThrowBudget: throw anira::Error(ANIRA_ERROR_BUDGET, "over budget");
            case PrepareMode::ReturnNull: return nullptr;
            case PrepareMode::Ok: break;
        }
        return std::make_unique<CountingPrepared>(counts, m_scale);
    }
    void release() noexcept override { m_released.fetch_add(1); }
    std::span<const char* const> consumed_kinds() const noexcept override {
        return m_consumes_entry ? std::span<const char* const>(k_kinds)
                                : std::span<const char* const>{};
    }

    /// The counts of the latest prepare.
    PreparedCounts& last() const { return *m_counts.back(); }
    std::shared_ptr<PreparedCounts> last_shared() const { return m_counts.back(); }

    static constexpr std::array<const char*, 1> k_kinds{"model:entry"};

    uint32_t m_mask;
    uint32_t m_flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST | ANIRA_STAGE_FLAG_REALTIME_HOOKS;
    float m_scale = 1.0F;
    bool m_consumes_entry = false;
    bool m_init_throws = false;
    PrepareMode m_mode = PrepareMode::Ok;
    std::atomic<int> m_released{0};
    int m_inited = 0;
    int m_prepared = 0;
    anira_log_level m_init_log_level = ANIRA_LOG_DEBUG;
    uint32_t m_init_num_threads = 0;
    const anira_context* m_init_context = nullptr;
    std::vector<std::shared_ptr<PreparedCounts>> m_counts;  ///< one per prepare, in order
};

/// A handler created through the C entry over a C++ pipeline, destroyed with the scope.
struct CHandler {
    CHandler(const anira_test::Context& context, const anira::Pipeline& pipe) {
        m_status = anira_handler_create(context.m_context, pipe.native(), &m_handler, &m_err);
    }
    ~CHandler() { anira_handler_destroy(m_handler); }
    CHandler(const CHandler&) = delete;
    CHandler& operator=(const CHandler&) = delete;
    CHandler(CHandler&&) = delete;
    CHandler& operator=(CHandler&&) = delete;

    anira_status prepare(const anira::ContractHandle& contract) {
        m_err = ANIRA_ERROR_INIT;
        return anira_handler_prepare(m_handler, contract.native(), &m_err);
    }
    void destroy() {
        anira_handler_destroy(m_handler);
        m_handler = nullptr;
    }

    anira_handler* m_handler = nullptr;
    anira_status m_status = ANIRA_OK;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// Drives `blocks` blocks of the ramp through a prepared mono float handler in lockstep and
/// appends both streams.
void drive(anira_handler* h, size_t blocks, std::vector<float>& in, std::vector<float>& out) {
    const std::array<int64_t, 2> extents{1, static_cast<int64_t>(k_hop)};
    for (size_t k = 0; k < blocks; ++k) {
        std::vector<float> block = anira_test::ramp(k + 1, k_hop);
        std::vector<float> result(k_hop, -1.0F);
        in.insert(in.end(), block.begin(), block.end());
        const Tensor source = Tensor::from_host(block.data(), ANIRA_DTYPE_F32, extents);
        const Tensor sink = Tensor::from_host(result.data(), ANIRA_DTYPE_F32, extents);
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_process(h, &source, 0, &sink, 0, &delivered), ANIRA_OK);
        ASSERT_EQ(delivered, k_hop);
        anira_test::wait_for_block(h, prev);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        out.insert(out.end(), result.begin(), result.end());
    }
}

/// The output stream is the input behind the latency, times `scale`.
void expect_scaled_passthrough(anira_handler* h,
                               const std::vector<float>& in,
                               const std::vector<float>& out,
                               float scale) {
    const size_t latency = anira_handler_get_latency(h, 0);
    ASSERT_EQ(in.size(), out.size());
    for (size_t n = 0; n < out.size(); ++n) {
        const float wanted = n < latency ? 0.0F : scale * in[n - latency];
        ASSERT_EQ(out[n], wanted) << "sample " << n;
    }
}

}  // namespace

// A Stage subclass end to end through a C-created handler: every phase of the mask runs once
// per chunk on the handler's Prepared with the context its phase promises, "call super" is
// anira's default body, prepare sees the record (the handler, the report, the entry count, the
// templates and the names), consumed_kinds joins the walk as the consumer "stage", the Prepared
// dies with the handler, and the shared_ptr the pipeline took is given back exactly once, when
// the last carrier dies.
TEST(AbiCxx, AStageSubclassRunsItsPhasesThroughACHandler) {
    const anira_test::Context context;
    ModelConfig model = stage_stream_model();
    model.model_ext(0, anira::ext::Entry{"forward"});  // consumed by the stage alone
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_all_phases);
    stage->m_consumes_entry = true;
    stage->m_scale = 0.5F;
    EXPECT_EQ(stage.use_count(), 1);

    std::optional<anira::Pipeline> pipe;
    pipe.emplace();
    pipe->inference(model, {custom_row()});
    {
        const anira::stage::Custom custom(stage);
        EXPECT_EQ(stage.use_count(), 2);
        pipe->add(custom);
        EXPECT_EQ(stage.use_count(), 3) << "the pipeline's carrier holds its own copy";
    }
    EXPECT_EQ(stage.use_count(), 2);

    CHandler handler(context, *pipe);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(stage.use_count(), 2) << "the handler shares the pipeline's carrier";
    // flags() reached the descriptor.
    ASSERT_NE(handler.m_handler->m_pipeline.m_stage, nullptr);
    EXPECT_EQ(handler.m_handler->m_pipeline.m_stage->desc().flags,
              ANIRA_STAGE_FLAG_REALTIME_PRE_POST | ANIRA_STAGE_FLAG_REALTIME_HOOKS);
    pipe.reset();  // the handler carries the stage alone now
    EXPECT_EQ(stage.use_count(), 2);
    EXPECT_EQ(stage->m_released.load(), 0);

    ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(stage->m_prepared, 1);
    ASSERT_EQ(stage->m_counts.size(), 1U);
    const std::shared_ptr<PreparedCounts> counts = stage->last_shared();
    EXPECT_EQ(counts->m_handler, handler.m_handler);
    EXPECT_EQ(counts->m_num_entries, anira_handler_num_entries(handler.m_handler));
    EXPECT_EQ(counts->m_num_entries_by_getter, counts->m_num_entries);
    EXPECT_GT(counts->m_num_entries, 0U);
    EXPECT_EQ(counts->m_num_inputs, 1U);
    EXPECT_EQ(counts->m_num_outputs, 1U);
    ASSERT_EQ(counts->m_input_names.size(), 1U);
    EXPECT_EQ(counts->m_input_names[0], "in");
    EXPECT_EQ(counts->m_plans, 1U);
    ASSERT_EQ(counts->m_consumers.size(), 1U);
    EXPECT_EQ(counts->m_consumers[0], "stage");
    EXPECT_EQ(handler.m_handler->m_stage_prepared != nullptr, true)
        << "the Prepared is the C pointer";

    std::vector<float> in;
    std::vector<float> out;
    drive(handler.m_handler, 3, in, out);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
    EXPECT_EQ(counts->m_pre.load(), 3);
    EXPECT_EQ(counts->m_before.load(), 3);
    EXPECT_EQ(counts->m_after.load(), 3);
    EXPECT_EQ(counts->m_post.load(), 3);
    EXPECT_EQ(counts->m_resets.load(), 1) << "the first chunk of the stream";
    EXPECT_EQ(counts->m_broken.load(), 0) << "a phase saw another context than its phase promises";
    expect_scaled_passthrough(handler.m_handler, in, out, 0.5F);
    EXPECT_FALSE(counts->m_deleted.load());

    handler.destroy();
    EXPECT_TRUE(counts->m_deleted.load()) << "the Prepared died with its handler";
    EXPECT_EQ(stage->m_released.load(), 1);
    EXPECT_EQ(stage.use_count(), 1) << "release deleted the carrier's copy";
}

// The mask decides, not the overrides: a stage of post_process alone leaves pre_process NULL
// in the descriptor, so anira's default pre_process keeps running (the stream arrives) and the
// stage's own pre_process, before_inference and after_inference are never called.
TEST(AbiCxx, APostOnlyStageLeavesTheDefaultPreProcessRunning) {
    const anira_test::Context context;
    const ModelConfig model = stage_stream_model();
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_post_process);
    stage->m_scale = 2.0F;
    const anira::Pipeline pipe{anira::stage::Inference(model, {custom_row()}),
                               anira::stage::Custom(stage)};
    CHandler handler(context, pipe);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK) << handler.m_err.message;

    std::vector<float> in;
    std::vector<float> out;
    drive(handler.m_handler, 3, in, out);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
    EXPECT_EQ(stage->last().m_pre.load(), 0);
    EXPECT_EQ(stage->last().m_before.load(), 0);
    EXPECT_EQ(stage->last().m_after.load(), 0);
    EXPECT_EQ(stage->last().m_post.load(), 3);
    EXPECT_EQ(stage->last().m_broken.load(), 0);
    expect_scaled_passthrough(handler.m_handler, in, out, 2.0F);

    // A stage of no phase at all is one of prepare and release: both defaults keep running.
    auto silent = std::make_shared<CountingStage>(0U);
    const anira::Pipeline quiet{anira::stage::Inference(model, {custom_row()}),
                                anira::stage::Custom(silent)};
    CHandler plain(context, quiet);
    ASSERT_EQ(plain.m_status, ANIRA_OK) << plain.m_err.message;
    ASSERT_EQ(plain.prepare(anira_test::explicit_contract()), ANIRA_OK) << plain.m_err.message;
    EXPECT_EQ(silent->m_prepared, 1);
    std::vector<float> plain_in;
    std::vector<float> plain_out;
    drive(plain.m_handler, 2, plain_in, plain_out);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(silent->last().m_pre.load() + silent->last().m_post.load(), 0);
    EXPECT_EQ(silent->last().m_resets.load(), 1) << "the reset slot is filled whatever the mask";
    expect_scaled_passthrough(plain.m_handler, plain_in, plain_out, 1.0F);
}

// Pipeline::add refuses what it cannot describe, and a refused add keeps no copy of the stage.
TEST(AbiCxx, AddRefusesANullStageAndAnUnknownPhaseBit) {
    anira::Pipeline pipe;
    const Thrown null_stage =
        thrown_by([&] { pipe.add(anira::stage::Custom(std::shared_ptr<anira::Stage>())); });
    EXPECT_TRUE(null_stage.m_thrown);
    EXPECT_EQ(null_stage.m_status, ANIRA_ERROR_INVALID_ARGUMENT);

    auto stage = std::make_shared<CountingStage>(anira::Stage::phase_bit(ANIRA_PHASE_INFERENCE));
    const Thrown bit = thrown_by([&] { pipe.add(anira::stage::Custom(stage)); });
    EXPECT_TRUE(bit.m_thrown);
    EXPECT_EQ(bit.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(bit.m_what.find("phases"), std::string::npos) << bit.m_what;
    EXPECT_EQ(stage.use_count(), 1);
    EXPECT_EQ(stage->m_released.load(), 0);

    // The C entry's own refusal: the copy made for the carrier dies with the throw, and
    // release is not called for an add that never happened.
    anira::Pipeline moved_from;
    const anira::Pipeline taken = std::move(moved_from);
    stage->m_mask = anira::Stage::k_pre_process;
    // A moved-from pipeline is a NULL anira_pipeline: the C entry refuses it.
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move) the point of the test
    const Thrown refused = thrown_by([&] { moved_from.add(anira::stage::Custom(stage)); });
    EXPECT_TRUE(refused.m_thrown);
    EXPECT_EQ(refused.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(refused.m_what.find("NULL pipeline"), std::string::npos) << refused.m_what;
    EXPECT_EQ(stage.use_count(), 1);
    EXPECT_EQ(stage->m_released.load(), 0);
    EXPECT_NE(taken.native(), nullptr);
}

// A throw never crosses the C boundary: prepare of the handler fails with the status of an
// anira::Error, with ANIRA_ERROR_INTERNAL for anything else, the message names the stage, and
// what() reaches the log. A null Prepared is ANIRA_ERROR_INTERNAL too: nothing could run the
// phases. A refused prepare deletes nothing that was not returned, and unprepares nothing.
TEST(AbiCxx, AThrowingStagePrepareFailsTheHandlersPrepare) {
    const anira_test::Context context;
    anira_test::RecordCollector collector;
    const ModelConfig model = stage_stream_model();
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_post_process);
    const anira::Pipeline pipe{anira::stage::Inference(model, {custom_row()}),
                               anira::stage::Custom(stage)};
    CHandler handler(context, pipe);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    const anira::ContractHandle contract = anira_test::explicit_contract();

    stage->m_mode = PrepareMode::ThrowError;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::string_view(handler.m_err.message).find("the stage refused prepare"),
              std::string_view::npos)
        << handler.m_err.message;
    EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";

    stage->m_mode = PrepareMode::ThrowRuntimeError;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_INTERNAL);
    stage->m_mode = PrepareMode::ThrowInt;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_INTERNAL);
    stage->m_mode = PrepareMode::ThrowBudget;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_BUDGET);
    stage->m_mode = PrepareMode::ReturnNull;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_INTERNAL);
    EXPECT_NE(std::string_view(handler.m_err.message).find("the stage refused prepare"),
              std::string_view::npos)
        << handler.m_err.message;
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.has("the stage's prepare threw: the stage refuses this plan", "native"));
    EXPECT_TRUE(collector.has("prepare threw: the stage ran out of luck", "native"));
    EXPECT_TRUE(collector.has("an exception that is no std::exception", "native"));
    EXPECT_TRUE(collector.has("the stage's prepare returned no Prepared", "native"));
#endif
    // Five refusals: nothing was returned, so nothing was deleted or unprepared.
    ASSERT_EQ(stage->m_counts.size(), 5U);
    for (const std::shared_ptr<PreparedCounts>& counts : stage->m_counts) {
        EXPECT_FALSE(counts->m_deleted.load());
    }

    stage->m_mode = PrepareMode::Ok;
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(stage->m_prepared, 6);
    std::vector<float> in;
    std::vector<float> out;
    drive(handler.m_handler, 2, in, out);
    ASSERT_FALSE(HasFatalFailure());
    expect_scaled_passthrough(handler.m_handler, in, out, 1.0F);
}

// Two pipelines that take one stage: a copy and a release per add.
TEST(AbiCxx, OneStageInTwoPipelinesIsReleasedOncePerAdd) {
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_after_inference);
    const anira::stage::Custom custom(stage);
    {
        anira::Pipeline first;
        anira::Pipeline second;
        first.add(custom);
        second.add(custom);
        EXPECT_EQ(stage.use_count(), 4);
        EXPECT_EQ(stage->m_released.load(), 0);
    }
    EXPECT_EQ(stage->m_released.load(), 2);
    EXPECT_EQ(stage.use_count(), 2);
}

// A pipeline holds one custom stage: the second add throws ANIRA_ERROR_INVALID_STATE, keeps no
// copy of the refused stage and never calls its release; the first stage is untouched.
TEST(AbiCxx, ASecondCustomStageIsRefused) {
    auto first = std::make_shared<CountingStage>(anira::Stage::k_post_process);
    auto second = std::make_shared<CountingStage>(anira::Stage::k_after_inference);
    anira::Pipeline pipe;
    pipe.add(anira::stage::Custom(first));
    EXPECT_EQ(first.use_count(), 2);
    const Thrown refused = thrown_by([&] { pipe.add(anira::stage::Custom(second)); });
    EXPECT_TRUE(refused.m_thrown);
    EXPECT_EQ(refused.m_status, ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(refused.m_what.find("already has a stage"), std::string::npos) << refused.m_what;
    EXPECT_EQ(second.use_count(), 1);
    EXPECT_EQ(second->m_released.load(), 0);
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(first->m_released.load(), 0);
}

// flags() is the stage's promise: without ANIRA_STAGE_FLAG_REALTIME_PRE_POST a stage that fills
// pre_process is refused at prepare under a Hard contract, by name; a bit the C header does
// not define is refused at add; with the promise the same stage runs.
TEST(AbiCxx, TheRealTimePromiseOfAStageIsCheckedAtPrepare) {
    const anira_test::Context context;
    const ModelConfig model = stage_stream_model();
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_pre_process);
    stage->m_flags = 0;
    {
        const anira::Pipeline pipe{anira::stage::Inference(model, {custom_row()}),
                                   anira::stage::Custom(stage)};
        CHandler handler(context, pipe);
        ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
        EXPECT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string_view(handler.m_err.message).find("the stage: pre_process is filled"),
                  std::string_view::npos)
            << handler.m_err.message;
        EXPECT_NE(
            std::string_view(handler.m_err.message).find("ANIRA_STAGE_FLAG_REALTIME_PRE_POST"),
            std::string_view::npos)
            << handler.m_err.message;
        EXPECT_TRUE(stage->m_counts.empty()) << "refused ahead of the stage's prepare";
    }
    stage->m_flags = 4U;
    {
        anira::Pipeline pipe;
        const Thrown bit = thrown_by([&] { pipe.add(anira::stage::Custom(stage)); });
        EXPECT_TRUE(bit.m_thrown);
        EXPECT_EQ(bit.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_NE(bit.m_what.find("flags"), std::string::npos) << bit.m_what;
        EXPECT_EQ(stage.use_count(), 1);
    }
    stage->m_flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST;  // read again by the next add
    {
        const anira::Pipeline pipe{anira::stage::Inference(model, {custom_row()}),
                                   anira::stage::Custom(stage)};
        CHandler handler(context, pipe);
        ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK)
            << handler.m_err.message;
        std::vector<float> in;
        std::vector<float> out;
        drive(handler.m_handler, 2, in, out);
        ASSERT_FALSE(HasFatalFailure());
        EXPECT_EQ(stage->last().m_pre.load(), 2);
        EXPECT_EQ(stage->last().m_broken.load(), 0) << "the entry is below the count prepare read";
        expect_scaled_passthrough(handler.m_handler, in, out, 1.0F);
    }
}

// One registration, two handlers from one pipeline: prepare returns a Prepared per handler, each
// with its own scratch (the chunks of one handler never land in the other's), the phases and the
// reset of a handler run on its own Prepared, reset is seen at every new stream, and every
// Prepared is deleted exactly once, at the re-prepare or the destroy of its handler; release
// fires after all of them.
TEST(AbiCxx, APreparedPerHandlerOwnsItsScratchAndDiesWithItsPrepare) {
    const anira_test::Context context;
    const ModelConfig model = stage_stream_model();
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_all_phases);
    std::optional<anira::Pipeline> pipe;
    pipe.emplace();
    pipe->inference(model, {custom_row()});
    pipe->add(anira::stage::Custom(stage));
    CHandler first(context, *pipe);
    CHandler second(context, *pipe);
    ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
    pipe.reset();
    const anira::ContractHandle contract = anira_test::explicit_contract();
    ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.prepare(contract), ANIRA_OK) << second.m_err.message;
    ASSERT_EQ(stage->m_counts.size(), 2U);
    const std::shared_ptr<PreparedCounts> of_first = stage->m_counts[0];
    const std::shared_ptr<PreparedCounts> of_second = stage->m_counts[1];
    EXPECT_EQ(of_first->m_handler, first.m_handler);
    EXPECT_EQ(of_second->m_handler, second.m_handler);
    EXPECT_NE(first.m_handler->m_stage_prepared, nullptr);
    EXPECT_NE(first.m_handler->m_stage_prepared, second.m_handler->m_stage_prepared);

    // Three chunks through the first handler, two through the second, on their own Prepared.
    std::vector<float> in_first;
    std::vector<float> out_first;
    std::vector<float> in_second;
    std::vector<float> out_second;
    drive(first.m_handler, 3, in_first, out_first);
    drive(second.m_handler, 2, in_second, out_second);
    ASSERT_FALSE(HasFatalFailure());
    expect_scaled_passthrough(first.m_handler, in_first, out_first, 1.0F);
    expect_scaled_passthrough(second.m_handler, in_second, out_second, 1.0F);
    EXPECT_EQ(of_first->m_pre.load(), 3);
    EXPECT_EQ(of_first->m_post.load(), 3);
    EXPECT_EQ(of_second->m_pre.load(), 2);
    EXPECT_EQ(of_second->m_post.load(), 2);
    EXPECT_EQ(of_first->m_broken.load() + of_second->m_broken.load(), 0);
    // Each scratch holds its own handler's chunks: the third block of the ramp went through the
    // first handler alone, so its first sample sits in the first scratch and in no other (one
    // shared scratch would hold it for both).
    const float third = anira_test::ramp(3, k_hop)[0];
    EXPECT_NE(std::ranges::find(of_first->m_scratch, third), of_first->m_scratch.end());
    EXPECT_EQ(std::ranges::find(of_second->m_scratch, third), of_second->m_scratch.end());
    // reset ran for the first chunk of each stream, and runs again after anira_handler_reset.
    EXPECT_EQ(of_first->m_resets.load(), 1);
    EXPECT_EQ(of_second->m_resets.load(), 1);
    anira_handler_reset(first.m_handler);
    std::vector<float> in_again;
    std::vector<float> out_again;
    drive(first.m_handler, 1, in_again, out_again);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(of_first->m_resets.load(), 2);
    EXPECT_EQ(of_second->m_resets.load(), 1);
    // A re-prepare of the first handler: its Prepared dies once and a new one takes over; the
    // second handler's lives on.
    ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(stage->m_counts.size(), 3U);
    EXPECT_TRUE(of_first->m_deleted.load());
    EXPECT_FALSE(of_second->m_deleted.load());
    EXPECT_FALSE(stage->m_counts[2]->m_deleted.load());
    EXPECT_EQ(first.m_handler->m_stage_prepared != nullptr, true);
    EXPECT_EQ(stage->m_prepared, 3);
    // The destroy of each handler deletes its Prepared; release fires after the last.
    first.destroy();
    EXPECT_TRUE(stage->m_counts[2]->m_deleted.load());
    EXPECT_FALSE(of_second->m_deleted.load());
    EXPECT_EQ(stage->m_released.load(), 0);
    second.destroy();
    EXPECT_TRUE(of_second->m_deleted.load());
    EXPECT_EQ(stage->m_released.load(), 1);
    EXPECT_EQ(stage.use_count(), 1);
}

namespace {

/// pre_process over an int16 input ring, written with the typed RingView: one hop of int16 out
/// of the ring, float32 into the model tensor. Nothing in anira converts. The scratch is the
/// Prepared's, one per handler.
class Int16InputPrepared final : public anira::Stage::Prepared {
public:
    anira_status pre_process(anira::StageContext& ctx) noexcept override {
        anira::Tensor tensor{};
        const anira_status exposed = ctx.input_tensor(0, tensor);
        if (exposed != ANIRA_OK) { return exposed; }
        anira::RingView ring;
        const anira_status has_ring = ctx.input_ring(0, ring);
        if (has_ring != ANIRA_OK) { return has_ring; }
        float* const samples = tensor.data_f32();
        if (samples == nullptr || ring.dtype() != ANIRA_DTYPE_I16) { return ANIRA_ERROR_CONFIG; }
        if (ring.pop_block(0, std::span<int16_t>(m_scratch)) != k_hop) {
            return ANIRA_ERROR_INTERNAL;
        }
        for (size_t n = 0; n < k_hop; ++n) {
            samples[n] = static_cast<float>(m_scratch[n]) / 32768.0F;
        }
        return ANIRA_OK;
    }

private:
    std::array<int16_t, k_hop> m_scratch{};
};

class Int16InputStage final : public anira::Stage {
public:
    uint32_t phases() const noexcept override { return k_pre_process; }
    uint32_t flags() const noexcept override { return ANIRA_STAGE_FLAG_REALTIME_PRE_POST; }
    std::unique_ptr<anira::Stage::Prepared> prepare(const anira::PrepareInfo& /*info*/) override {
        return std::make_unique<Int16InputPrepared>();
    }
};

}  // namespace

// The typed RingView calls on an int16 ring: the element type is the dtype the C entry is told,
// so int16_t moves and float is refused (0, nothing moved); a span too short for the windows
// of pop_windows is refused by the view; a view of no ring answers 0 to everything.
TEST(AbiCxx, RingViewIsTypedByItsElement) {
    anira::RingBuffer words;
    ASSERT_TRUE(words.initialize_with_positions(1, 16, ANIRA_DTYPE_I16));
    anira::RingView ring(&words);
    ASSERT_TRUE(ring);
    EXPECT_EQ(ring.native(), &words);
    EXPECT_EQ(ring.dtype(), ANIRA_DTYPE_I16);
    EXPECT_EQ(ring.num_channels(), 1U);

    const std::array<int16_t, 4> in{-3, 7, 32767, -32768};
    EXPECT_EQ(ring.push_block(0, std::span<const int16_t>(in)), 4U);
    EXPECT_EQ(ring.available(0), 4U);
    std::array<int16_t, 4> out{};
    EXPECT_EQ(ring.pop_block(0, std::span<int16_t>(out).first(2)), 2U);
    EXPECT_EQ(out[0], -3);
    EXPECT_EQ(out[1], 7);
    EXPECT_EQ(ring.available_past(0), 2U);
    EXPECT_EQ(ring.peek_past_block(0, std::span<int16_t>(out).first(2)), 2U);
    EXPECT_EQ(out[1], 7) << "the element popped last is the last of the history";

    // Two windows of [1 old, 1 new] behind an offset of 1: {7, 32767} and {32767, -32768}.
    std::array<int16_t, 5> windows{};
    EXPECT_EQ(ring.pop_windows(0, std::span<int16_t>(windows).first(4), 1, 1, 1, 2), 0U)
        << "a span too short for offset + num_batches * (num_new + num_old)";
    EXPECT_EQ(ring.available(0), 2U) << "a refused pop pops nothing";
    EXPECT_EQ(ring.pop_windows(0, std::span<int16_t>(windows), 1, 1, 1, 2), 4U);
    EXPECT_EQ(windows, (std::array<int16_t, 5>{0, 7, 32767, 32767, -32768}));

    const int16_t fill = -5;
    EXPECT_EQ(ring.push_fill(0, fill, 3), 3U);
    EXPECT_EQ(ring.discard(0, 1), 1U);
    EXPECT_EQ(ring.pop_block(0, std::span<int16_t>(out).first(2)), 2U);
    EXPECT_EQ(out[0], -5);

    // Nothing converts: a float call on the int16 ring moves nothing, in either direction.
    std::array<float, 4> floats{-1.0F, -1.0F, -1.0F, -1.0F};
    EXPECT_EQ(ring.push_fill(0, fill, 4), 4U);
    EXPECT_EQ(ring.pop_block(0, std::span<float>(floats)), 0U);
    EXPECT_EQ(ring.peek_past_block(0, std::span<float>(floats)), 0U);
    EXPECT_EQ(ring.push_block(0, std::span<const float>(floats)), 0U);
    EXPECT_EQ(ring.push_fill(0, 1.0F, 4), 0U);
    EXPECT_EQ(floats[0], -1.0F);
    EXPECT_EQ(ring.available(0), 4U);

    anira::RingView none;
    EXPECT_FALSE(none);
    EXPECT_EQ(none.native(), nullptr);
    EXPECT_EQ(none.dtype(), 0U);
    EXPECT_EQ(none.num_channels(), 0U);
    EXPECT_EQ(none.available(0) + none.available_past(0) + none.discard(0, 1), 0U);
    EXPECT_EQ(none.pop_block(0, std::span<int16_t>(out)), 0U);
    EXPECT_EQ(none.push_block(0, std::span<const int16_t>(in)), 0U);
}

// The same view inside a stage, end to end: an int16 ring declared on the contract needs a stage
// that fills pre_process, and the C++ stage's mask is what satisfies prepare.
TEST(AbiCxx, AnInt16RingThroughAConvertingStage) {
    const anira_test::Context context;
    const ModelConfig model = stage_stream_model();
    // ZEROS: BYPASS would copy the int16 input ring into the float32 output ring.
    anira::ContractHandle contract =
        anira_test::explicit_contract(anira_test::k_block, anira_test::k_rate, ANIRA_MISS_ZEROS);
    contract.hard_ring_dtype("in", ANIRA_DTYPE_I16);

    const anira::Pipeline pipe{anira::stage::Inference(model, {custom_row()}),
                               anira::stage::Custom(std::make_shared<Int16InputStage>())};
    CHandler handler(context, pipe);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    const size_t latency = anira_handler_get_latency(h, 0);

    std::vector<int16_t> in_stream;
    std::vector<float> out_stream;
    const std::array<int64_t, 2> extents{1, static_cast<int64_t>(k_hop)};
    for (size_t k = 0; k < 3; ++k) {
        std::array<int16_t, k_hop> in{};
        for (size_t n = 0; n < k_hop; ++n) {
            in.at(n) = static_cast<int16_t>(static_cast<int>((k * k_hop) + n) - 700);
        }
        in_stream.insert(in_stream.end(), in.begin(), in.end());
        std::array<float, k_hop> out{};
        const Tensor source = Tensor::from_host(in.data(), ANIRA_DTYPE_I16, extents);
        const Tensor sink = Tensor::from_host(out.data(), ANIRA_DTYPE_F32, extents);
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_process(h, &source, 0, &sink, 0, &delivered), ANIRA_OK);
        anira_test::wait_for_block(h, prev);
        ASSERT_FALSE(HasFatalFailure());
        out_stream.insert(out_stream.end(), out.begin(), out.end());
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    for (size_t n = 0; n < out_stream.size(); ++n) {
        const float wanted =
            n < latency ? 0.0F : static_cast<float>(in_stream.at(n - latency)) / 32768.0F;
        ASSERT_EQ(out_stream.at(n), wanted) << "sample " << n;
    }
}

// TensorSpec::state_source is the C setter: the pairing lands in the model's JSON on the input
// spec alone and survives the round trip byte for byte; the setter's refusals throw.
TEST(AbiCxx, StateSourceRoundTripsThroughToJson) {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in"));
    model.input(TensorSpec("state_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE)
                    .axis(0, ANIRA_AXIS_ANY, 2)
                    .state_source("state_out"));
    model.output(streamed("out"));
    model.output(
        TensorSpec("state_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE).axis(0, ANIRA_AXIS_ANY, 2));

    const std::string text = model.to_json();
    EXPECT_NE(text.find("\"state_source\": \"state_out\""), std::string::npos) << text;
    EXPECT_EQ(text.find("\"state_source\""), text.rfind("\"state_source\""))
        << "stated once, on the input";
    EXPECT_EQ(ModelConfig::from_json(text).to_json(), text);

    // A second call replaces the name.
    TensorSpec spec("state_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
    spec.state_source("first").state_source("second");
    ModelConfig renamed;
    renamed.input(spec);
    EXPECT_NE(renamed.to_json().find("\"state_source\": \"second\""), std::string::npos);

    const Thrown empty = thrown_by([&] { spec.state_source(""); });
    EXPECT_TRUE(empty.m_thrown);
    EXPECT_EQ(empty.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(empty.m_what.find("anira_tensor_spec_set_state_source"), std::string::npos)
        << empty.m_what;
    const Thrown role = thrown_by([] { streamed("in").state_source("state_out"); });
    EXPECT_TRUE(role.m_thrown);
    EXPECT_EQ(role.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
}

// ---- engines -----------------------------------------------------------------------------------
//
// process and reset of an Engine::Prepared run on an inference thread: the test engines hold no
// gtest assertion there, allocate nothing and write atomics the test reads once the block's
// inference was collected. The stream runs in lockstep, one block and then the wait.

namespace {

/// The id the C++ engine is registered under, and the path of its row, which names no file:
/// anira never opens a registered row's path.
constexpr const char* k_cxx_engine_id = "org.example.cxx";
constexpr const char* k_cxx_never_opened = "never-opened.bin";

/// The mono stream of the stage tests, on one model entry the registered C++ engine serves.
ModelConfig engine_stream_model() {
    ModelConfig model;
    model.add_model_path(k_cxx_engine_id, k_cxx_never_opened);
    model.input(streamed("in", static_cast<int64_t>(k_hop)));
    model.output(streamed("out", static_cast<int64_t>(k_hop)));
    return model;
}

/// What one loaded model of a CountingEngine read at load, kept by the registration beyond the
/// Loaded's life (anira deletes the Loaded at unload; the test reads afterwards).
struct EngineLoadCounts {
    uint32_t m_row = 0;
    uint32_t m_model_count = 0;
    anira::EngineKind m_engine = ANIRA_ENGINE_FORCE32;
    std::string m_engine_id;
    std::string m_path;
    std::size_t m_num_bytes = 0;
    uint32_t m_instances = 0;
    anira::Provider m_provider = ANIRA_PROVIDER_DEFAULT;
    std::string m_provider_id;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<Tensor> m_inputs;   ///< the templates
    std::vector<Tensor> m_outputs;  ///< the templates
    std::atomic<bool> m_unloaded{false};
};

/// What one prepared handle of a CountingEngine saw at prepare and did on the inference
/// threads, kept by the registration beyond the Prepared's life (anira deletes the Prepared at
/// unprepare; the test reads afterwards).
struct EnginePreparedCounts {
    // The record, read at prepare: the stage's.
    anira_handler* m_handler = nullptr;
    uint32_t m_num_entries = 0;
    bool m_exclusive = false;
    std::size_t m_num_inputs = 0;
    std::size_t m_num_outputs = 0;
    // What the inference threads saw.
    std::atomic<int> m_processed{0};
    std::atomic<int> m_resets{0};
    std::atomic<int> m_broken{0};  ///< a context unlike the records promise
    std::atomic<uint32_t> m_max_instance{0};
    std::atomic<bool> m_deleted{false};
    /// The context reset saw, cleared by the process that follows; whether that process ran on
    /// the same context.
    std::atomic<const anira_engine_ctx*> m_pending_reset{nullptr};
    std::atomic<int> m_process_after_reset_same_ctx{0};
    std::atomic<int> m_process_after_reset_other_ctx{0};
};

/// Whether `tensor` is `expected` in dtype, rank and extents.
bool same_shape(const anira_tensor& tensor, const anira_tensor& expected) {
    if (tensor.dtype != expected.dtype || tensor.ndim != expected.ndim) { return false; }
    for (uint32_t axis = 0; axis < expected.ndim && axis < ANIRA_MAX_RANK; ++axis) {
        if (tensor.shape[axis] != expected.shape[axis]) { return false; }
    }
    return true;
}

/// The Prepared of one handler: process multiplies the first input into the first output by
/// `gain`, checking every call against the records its load and its prepare kept (the loaded
/// pointer of the context is the Loaded, an exclusive handle's calls carry the flag and
/// instance 0, a shared handle's an instance below the count); reset counts itself and notes
/// its context, so that the process behind it can say whether it ran on the same one.
class GainPrepared final : public anira::Engine::Prepared {
public:
    GainPrepared(std::shared_ptr<EnginePreparedCounts> counts,
                 std::shared_ptr<EngineLoadCounts> load,
                 const anira::Engine::Loaded* loaded,
                 float gain)
        : m_counts(std::move(counts)), m_load(std::move(load)), m_loaded(loaded), m_gain(gain) {}
    ~GainPrepared() override { m_counts->m_deleted.store(true); }
    GainPrepared(const GainPrepared&) = delete;
    GainPrepared& operator=(const GainPrepared&) = delete;
    GainPrepared(GainPrepared&&) = delete;
    GainPrepared& operator=(GainPrepared&&) = delete;

    anira_status process(anira::EngineContext& ctx) noexcept override {
        m_counts->m_processed.fetch_add(1);
        if (const anira_engine_ctx* reset_ctx = m_counts->m_pending_reset.exchange(nullptr)) {
            if (reset_ctx == ctx.native()) {
                m_counts->m_process_after_reset_same_ctx.fetch_add(1);
            } else {
                m_counts->m_process_after_reset_other_ctx.fetch_add(1);
            }
        }
        const std::span<const Tensor> inputs = ctx.inputs();
        const std::span<Tensor> outputs = ctx.outputs();
        const bool exclusive = ctx.exclusive();
        const bool instance_as_promised =
            exclusive ? ctx.instance() == 0 && ctx.flags() == ANIRA_ENGINE_CALL_EXCLUSIVE
                      : ctx.instance() < m_load->m_instances && ctx.flags() == 0;
        bool as_promised = ctx.native() != nullptr && ctx.loaded() == m_loaded &&
                           exclusive == m_counts->m_exclusive && instance_as_promised &&
                           ctx.ticket() == ANIRA_TICKET_INVALID &&
                           inputs.size() == m_load->m_inputs.size() &&
                           outputs.size() == m_load->m_outputs.size();
        for (size_t i = 0; as_promised && i < inputs.size(); ++i) {
            as_promised = same_shape(inputs[i], m_load->m_inputs[i]);
        }
        for (size_t i = 0; as_promised && i < outputs.size(); ++i) {
            as_promised = same_shape(outputs[i], m_load->m_outputs[i]);
        }
        if (!as_promised) {
            m_counts->m_broken.fetch_add(1);
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        uint32_t seen = m_counts->m_max_instance.load();
        while (ctx.instance() > seen &&
               !m_counts->m_max_instance.compare_exchange_weak(seen, ctx.instance())) {}
        const float* const in = inputs[0].data_f32();
        float* const out = outputs[0].data_f32();
        if (in == nullptr || out == nullptr) {
            m_counts->m_broken.fetch_add(1);
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        const size_t count = std::min(inputs[0].num_elements(), outputs[0].num_elements());
        for (size_t n = 0; n < count; ++n) { out[n] = in[n] * m_gain; }
        return ANIRA_OK;
    }
    void reset(anira::EngineContext& ctx) noexcept override {
        m_counts->m_resets.fetch_add(1);
        if (ctx.inputs().size() != m_load->m_inputs.size() || !m_counts->m_exclusive ||
            !ctx.exclusive()) {
            m_counts->m_broken.fetch_add(1);
        }
        m_counts->m_pending_reset.store(ctx.native());
    }

private:
    std::shared_ptr<EnginePreparedCounts> m_counts;
    std::shared_ptr<EngineLoadCounts> m_load;
    const anira::Engine::Loaded* m_loaded;
    float m_gain;
};

/// Where a CountingEngine applies its PrepareMode: at load (the registration's) or at prepare
/// (the Loaded's).
enum class ThrowAt : uint8_t { Load, Prepare };

class CountingEngine;

/// The Loaded of one loaded model: the record it read, and the Prepared it makes per handler.
class CountingLoaded final : public anira::Engine::Loaded {
public:
    CountingLoaded(CountingEngine& engine, std::shared_ptr<EngineLoadCounts> counts)
        : m_engine(&engine), m_counts(std::move(counts)) {}
    ~CountingLoaded() override { m_counts->m_unloaded.store(true); }
    CountingLoaded(const CountingLoaded&) = delete;
    CountingLoaded& operator=(const CountingLoaded&) = delete;
    CountingLoaded(CountingLoaded&&) = delete;
    CountingLoaded& operator=(CountingLoaded&&) = delete;

    std::unique_ptr<anira::Engine::Prepared> prepare(const anira::PrepareInfo& info) override;

private:
    CountingEngine* m_engine;
    std::shared_ptr<EngineLoadCounts> m_counts;
};

/// The registration: the promise, the kinds, what init, load and prepare do, and the counts of
/// every Loaded and every Prepared it made, in order.
class CountingEngine : public anira::Engine {
public:
    uint32_t flags() const noexcept override { return m_flags; }
    std::span<const char* const> consumed_kinds() const noexcept override {
        return m_consumes_entry ? std::span<const char* const>(k_kinds)
                                : std::span<const char* const>{};
    }
    std::span<const char* const> providers() const noexcept override { return m_providers; }

    void init(const anira::InitInfo& info) override {
        ++m_inited;
        m_init_log_level = info.log_level();
        m_init_num_threads = info.num_threads();
        m_init_context = info.context();
        if (m_init_throws) { throw anira::Error(ANIRA_ERROR_NOT_SUPPORTED, "the engine refuses"); }
    }

    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& info) override {
        ++m_loaded;
        auto counts = std::make_shared<EngineLoadCounts>();
        counts->m_row = info.row();
        counts->m_model_count = info.model_count();
        counts->m_engine = info.model_engine(info.row());
        counts->m_engine_id = std::string(info.model_engine_id(info.row()));
        counts->m_path = std::string(info.model_path(info.row()));
        counts->m_num_bytes = info.model_bytes(info.row()).size();
        counts->m_instances = info.instances();
        counts->m_provider = info.provider();
        counts->m_provider_id = std::string(info.provider_id());
        for (const char* name : info.input_names()) { counts->m_input_names.emplace_back(name); }
        for (const char* name : info.output_names()) { counts->m_output_names.emplace_back(name); }
        counts->m_inputs.assign(info.inputs().begin(), info.inputs().end());
        counts->m_outputs.assign(info.outputs().begin(), info.outputs().end());
        m_loads.push_back(counts);
        if (m_throw_at == ThrowAt::Load && refuse()) { return nullptr; }
        return std::make_unique<CountingLoaded>(*this, counts);
    }
    void release() noexcept override { m_released.fetch_add(1); }

    /// Applies m_mode: throws, or says whether to return null.
    bool refuse() const {
        switch (m_mode) {
            case PrepareMode::ThrowError:
                throw anira::Error(ANIRA_ERROR_CONFIG, "the engine refuses this model");
            case PrepareMode::ThrowRuntimeError:
                throw std::runtime_error("the engine ran out of luck");
            case PrepareMode::ThrowInt:
                throw 7;  // no std::exception: what the trampoline must survive too
            case PrepareMode::ThrowBadAlloc: throw std::bad_alloc();
            case PrepareMode::ThrowBudget: throw anira::Error(ANIRA_ERROR_BUDGET, "over budget");
            case PrepareMode::ReturnNull: return true;
            case PrepareMode::Ok: break;
        }
        return false;
    }

    /// The counts of the latest load, and of the latest prepare.
    std::shared_ptr<EngineLoadCounts> last_load() const { return m_loads.back(); }
    std::shared_ptr<EnginePreparedCounts> last() const { return m_counts.back(); }

    static constexpr std::array<const char*, 1> k_kinds{"model:entry"};

    uint32_t m_flags = 0;
    float m_gain = 1.0F;
    bool m_consumes_entry = false;
    std::vector<const char*> m_providers;  ///< what providers() answers
    bool m_init_throws = false;
    PrepareMode m_mode = PrepareMode::Ok;
    ThrowAt m_throw_at = ThrowAt::Prepare;
    std::atomic<int> m_released{0};
    int m_inited = 0;
    int m_loaded = 0;
    int m_prepared = 0;
    anira_log_level m_init_log_level = ANIRA_LOG_DEBUG;
    uint32_t m_init_num_threads = 0;
    const anira_context* m_init_context = nullptr;
    std::vector<std::shared_ptr<EngineLoadCounts>> m_loads;       ///< one per load, in order
    std::vector<std::shared_ptr<EnginePreparedCounts>> m_counts;  ///< one per prepare, in order
};

std::unique_ptr<anira::Engine::Prepared> CountingLoaded::prepare(const anira::PrepareInfo& info) {
    ++m_engine->m_prepared;
    auto counts = std::make_shared<EnginePreparedCounts>();
    counts->m_handler = info.handler();
    counts->m_num_entries = info.num_entries();
    counts->m_exclusive = info.exclusive();
    counts->m_num_inputs = info.inputs().size();
    counts->m_num_outputs = info.outputs().size();
    m_engine->m_counts.push_back(counts);
    if (m_engine->m_throw_at == ThrowAt::Prepare && m_engine->refuse()) { return nullptr; }
    return std::make_unique<GainPrepared>(counts, m_counts, this, m_engine->m_gain);
}

}  // namespace

// An Engine subclass end to end through a C-created handler: register_engine hands the
// object's C engine a copy of the shared_ptr, a handler created from the pipeline carries it
// on, init sees the facts of the core, load sees its record (the row, the variant's facts
// through the getters, the templates, the names, one instance), prepare sees the stage's
// record (the handler, its entry count, not exclusive), every block goes through process with
// the context the records promise, the flags and the consumed kinds reach the plan report under
// the engine's id, the Prepared and the Loaded die with the handler, and the shared_ptr is
// given back exactly once, when the last reference to the C engine dies.
TEST(AbiCxx, AnEngineSubclassRunsThroughACHandler) {
    const anira_test::Context context;
    ModelConfig model = engine_stream_model();
    model.model_ext(0, anira::ext::Entry{"forward"});  // consumed by the engine alone
    auto engine = std::make_shared<CountingEngine>();
    engine->m_gain = 0.5F;
    engine->m_flags = ANIRA_ENGINE_FLAG_REALTIME_SAFE;
    engine->m_consumes_entry = true;
    EXPECT_EQ(engine.use_count(), 1);

    std::optional<anira::Pipeline> pipe;
    pipe.emplace();
    pipe->register_engine(k_cxx_engine_id, engine);
    EXPECT_EQ(engine.use_count(), 2) << "the object's C engine holds its own copy";
    pipe->inference(model);
    CHandler handler(context, *pipe);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.use_count(), 2) << "the handler shares the pipeline's C engine";
    pipe.reset();  // the handler carries the engine alone now
    EXPECT_EQ(engine.use_count(), 2);
    EXPECT_EQ(engine->m_released.load(), 0);

    ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine->m_inited, 1);
    EXPECT_EQ(engine->m_loaded, 1);
    EXPECT_EQ(engine->m_prepared, 1);
    EXPECT_EQ(engine->m_init_num_threads, 2U) << "the pool of the context";
    EXPECT_EQ(engine->m_init_context, context.m_context);
    ASSERT_EQ(engine->m_loads.size(), 1U);
    ASSERT_EQ(engine->m_counts.size(), 1U);
    const std::shared_ptr<EngineLoadCounts> loaded = engine->last_load();
    const std::shared_ptr<EnginePreparedCounts> counts = engine->last();
    EXPECT_EQ(loaded->m_row, 0U);
    EXPECT_EQ(loaded->m_model_count, 1U);
    EXPECT_EQ(loaded->m_engine, ANIRA_ENGINE_NONE);
    EXPECT_EQ(loaded->m_engine_id, k_cxx_engine_id);
    EXPECT_EQ(loaded->m_path, k_cxx_never_opened)
        << "the engine reads the row; anira never opens it";
    EXPECT_EQ(loaded->m_num_bytes, 0U) << "a path row has no bytes";
    EXPECT_EQ(loaded->m_instances, 1U);
    EXPECT_EQ(loaded->m_input_names, (std::vector<std::string>{"in"}));
    EXPECT_EQ(loaded->m_output_names, (std::vector<std::string>{"out"}));
    ASSERT_EQ(loaded->m_inputs.size(), 1U);
    ASSERT_EQ(loaded->m_outputs.size(), 1U);
    EXPECT_EQ(counts->m_handler, handler.m_handler);
    EXPECT_EQ(counts->m_num_entries, anira_handler_num_entries(handler.m_handler));
    EXPECT_FALSE(counts->m_exclusive) << "a stateless model's handler";
    EXPECT_EQ(counts->m_num_inputs, 1U);
    EXPECT_EQ(counts->m_num_outputs, 1U);
    for (const Tensor* tensor : {&loaded->m_inputs[0], &loaded->m_outputs[0]}) {
        EXPECT_EQ(tensor->dtype, static_cast<anira_dtype>(ANIRA_DTYPE_F32));
        EXPECT_EQ(tensor->ndim, 3U);
        EXPECT_EQ(tensor->num_elements(), k_hop);
        EXPECT_EQ(tensor->domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        EXPECT_EQ(tensor->data_f32(), nullptr) << "a template carries no memory";
    }
    // The plan: ANIRA_ENGINE_NONE with the id, the flags, every slot bound by the engine, the
    // consumed extension under the engine's id.
    const anira::PlanReport report(anira_handler_plan_report(handler.m_handler));
    ASSERT_EQ(report.num_plans(), 1U);
    const std::vector<anira_plan_info> plans = report.plans();
    ASSERT_EQ(plans.size(), 1U);
    EXPECT_EQ(plans[0].engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE));
    ASSERT_NE(plans[0].engine_id, nullptr);
    EXPECT_STREQ(plans[0].engine_id, k_cxx_engine_id);
    EXPECT_EQ(plans[0].engine_flags, static_cast<uint32_t>(ANIRA_ENGINE_FLAG_REALTIME_SAFE));
    for (const bool inputs : {true, false}) {
        const std::vector<anira_plan_slot> slots = report.slots(0, inputs);
        ASSERT_EQ(slots.size(), 1U);
        EXPECT_EQ(slots[0].binding, static_cast<uint32_t>(ANIRA_BINDING_ENGINE));
    }
    const std::vector<anira_plan_ext> exts = report.extensions(0);
    ASSERT_EQ(exts.size(), 1U);
    EXPECT_STREQ(exts[0].consumer, k_cxx_engine_id);

    std::vector<float> in;
    std::vector<float> out;
    drive(handler.m_handler, 3, in, out);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
    EXPECT_EQ(counts->m_processed.load(), 3);
    EXPECT_EQ(counts->m_resets.load(), 0) << "a shared handle is never reset";
    EXPECT_EQ(counts->m_broken.load(), 0) << "process saw another context than the records promise";
    EXPECT_EQ(counts->m_max_instance.load(), 0U);
    expect_scaled_passthrough(handler.m_handler, in, out, 0.5F);
    EXPECT_FALSE(counts->m_deleted.load());
    EXPECT_FALSE(loaded->m_unloaded.load());

    handler.destroy();
    EXPECT_TRUE(counts->m_deleted.load()) << "the Prepared died with its handler";
    EXPECT_TRUE(loaded->m_unloaded.load()) << "and the Loaded with the last handler on it";
    EXPECT_EQ(engine->m_released.load(), 1);
    EXPECT_EQ(engine.use_count(), 1) << "release deleted the carrier's copy";
}

// A throw never crosses the C boundary, at load or at prepare: prepare of the handler fails
// with the status of an anira::Error, with ANIRA_ERROR_OUT_OF_MEMORY for a std::bad_alloc and
// ANIRA_ERROR_INTERNAL for anything else, the message names the engine and the slot, and
// what() reaches the log. A null Loaded or Prepared is ANIRA_ERROR_INTERNAL too: nothing could
// run the inference. A refused load deletes nothing that was not returned and owes no unload;
// a refused prepare owes no unprepare, and the Loaded the failed prepare ran on goes back with
// the session the failure releases.
TEST(AbiCxx, AThrowingEngineLoadOrPrepareFailsTheHandlersPrepare) {
    const anira_test::Context context;
    anira_test::RecordCollector collector;
    const ModelConfig model = engine_stream_model();
    const anira::ContractHandle contract = anira_test::explicit_contract();
    for (const ThrowAt where : {ThrowAt::Load, ThrowAt::Prepare}) {
        const char* const slot = where == ThrowAt::Load ? "load" : "prepare";
        SCOPED_TRACE(slot);
        auto engine = std::make_shared<CountingEngine>();
        engine->m_throw_at = where;
        anira::Pipeline pipe;
        pipe.register_engine(k_cxx_engine_id, engine).inference(model);
        CHandler handler(context, pipe);
        ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
        const std::string refused = std::string("the engine 'org.example.cxx' refused ") + slot;

        engine->m_mode = PrepareMode::ThrowError;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::string_view(handler.m_err.message).find(refused), std::string_view::npos)
            << handler.m_err.message;
        EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";

        engine->m_mode = PrepareMode::ThrowRuntimeError;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_INTERNAL);
        engine->m_mode = PrepareMode::ThrowInt;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_INTERNAL);
        engine->m_mode = PrepareMode::ThrowBadAlloc;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_OUT_OF_MEMORY);
        engine->m_mode = PrepareMode::ThrowBudget;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_BUDGET);
        engine->m_mode = PrepareMode::ReturnNull;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_INTERNAL);
        EXPECT_NE(std::string_view(handler.m_err.message).find(refused), std::string_view::npos)
            << handler.m_err.message;
#ifdef ENABLE_LOGGING
        const std::string threw = std::string("the engine's ") + slot + " threw: ";
        EXPECT_TRUE(collector.has((threw + "the engine refuses this model").c_str(), "native"));
        EXPECT_TRUE(collector.has((threw + "the engine ran out of luck").c_str(), "native"));
        EXPECT_TRUE(collector.has("an exception that is no std::exception", "native"));
        if (where == ThrowAt::Load) {
            EXPECT_TRUE(collector.has("the engine's load returned no Loaded", "native"));
        } else {
            EXPECT_TRUE(collector.has("the engine's prepare returned no Prepared", "native"));
        }
#endif
        // Six refusals: nothing was returned at the refusing slot, so nothing was deleted there
        // or given back; six loads either way (a failed prepare releases its session, and with
        // it the Loaded the session had acquired, so the next prepare loads again).
        ASSERT_EQ(engine->m_loads.size(), 6U);
        if (where == ThrowAt::Load) {
            EXPECT_EQ(engine->m_prepared, 0) << "nothing prepared behind a refused load";
            for (const std::shared_ptr<EngineLoadCounts>& counts : engine->m_loads) {
                EXPECT_FALSE(counts->m_unloaded.load()) << "a refused load owes no unload";
            }
        } else {
            ASSERT_EQ(engine->m_counts.size(), 6U);
            for (const std::shared_ptr<EnginePreparedCounts>& counts : engine->m_counts) {
                EXPECT_FALSE(counts->m_deleted.load());
            }
            for (const std::shared_ptr<EngineLoadCounts>& counts : engine->m_loads) {
                EXPECT_TRUE(counts->m_unloaded.load()) << "given back with the failed session";
            }
        }

        engine->m_mode = PrepareMode::Ok;
        ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
        EXPECT_EQ(engine->m_loaded, 7);
        EXPECT_EQ(engine->m_prepared, where == ThrowAt::Load ? 1 : 7);
        std::vector<float> in;
        std::vector<float> out;
        drive(handler.m_handler, 2, in, out);
        ASSERT_FALSE(HasFatalFailure());
        expect_scaled_passthrough(handler.m_handler, in, out, 1.0F);
    }
}

// init runs once per registration (the stage's per Pipeline::add, the engine's per C engine),
// at the first prepare of a handler that reaches it, with the facts of the core in effect (the
// pool of the context, the context of that handler); a throw fails that prepare with its
// status, the message names the slot, nothing is loaded or prepared behind it, and the next
// prepare calls init again; release fires whether init ever ran or not.
TEST(AbiCxx, InitRunsOncePerRegistrationAndAThrowFailsThePrepare) {
    const anira_test::Context context(2);
    const anira::ContractHandle contract = anira_test::explicit_contract();
    const ModelConfig model = engine_stream_model();
    auto engine = std::make_shared<CountingEngine>();
    auto stage = std::make_shared<CountingStage>(anira::Stage::k_post_process);
    engine->m_init_throws = true;
    stage->m_init_throws = true;
    {
        const anira::Pipeline pipe{anira::stage::Inference(model).engine(k_cxx_engine_id, engine),
                                   anira::stage::Custom(stage)};
        CHandler first(context, pipe);
        CHandler second(context, pipe);
        ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
        EXPECT_EQ(engine->m_inited + stage->m_inited, 0) << "init belongs to the first prepare";

        // The engine's init comes first (the session's create loads its model), so the stage's
        // is not reached while the engine refuses.
        EXPECT_EQ(first.prepare(contract), ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_NE(
            std::string_view(first.m_err.message).find("the engine 'org.example.cxx' refused init"),
            std::string_view::npos)
            << first.m_err.message;
        EXPECT_EQ(engine->m_inited, 1);
        EXPECT_EQ(engine->m_loaded, 0) << "nothing loaded behind a refused init";
        EXPECT_EQ(stage->m_inited, 0) << "the session failed before the stage";
        engine->m_init_throws = false;
        EXPECT_EQ(first.prepare(contract), ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_NE(std::string_view(first.m_err.message).find("the stage refused init"),
                  std::string_view::npos)
            << first.m_err.message;
        EXPECT_EQ(engine->m_inited, 2) << "tried again";
        EXPECT_EQ(engine->m_loaded, 1);
        EXPECT_EQ(engine->m_prepared, 1);
        EXPECT_EQ(stage->m_inited, 1);
        EXPECT_EQ(stage->m_prepared, 0) << "the stage's prepare waits for its init";
        EXPECT_EQ(anira_handler_plan_report(first.m_handler), nullptr);
        stage->m_init_throws = false;
        ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
        EXPECT_EQ(engine->m_inited, 2) << "initialised: not again";
        EXPECT_EQ(stage->m_inited, 2) << "tried again";
        EXPECT_EQ(stage->m_prepared, 1);
        ASSERT_EQ(second.prepare(contract), ANIRA_OK) << second.m_err.message;
        ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
        EXPECT_EQ(engine->m_inited, 2) << "once per C engine";
        EXPECT_EQ(stage->m_inited, 2) << "once per add";
        // The facts: the pool of the context and the context of the handler whose prepare
        // reached the registrations.
        EXPECT_EQ(engine->m_init_num_threads, 2U);
        EXPECT_EQ(engine->m_init_context, context.m_context);
        EXPECT_EQ(engine->m_init_log_level, ANIRA_LOG_ERROR) << "the context's level";
        EXPECT_EQ(stage->m_init_num_threads, 2U);
        EXPECT_EQ(stage->m_init_context, context.m_context);
        EXPECT_EQ(stage->m_init_log_level, ANIRA_LOG_ERROR);
        std::vector<float> in;
        std::vector<float> out;
        drive(first.m_handler, 2, in, out);
        ASSERT_FALSE(HasFatalFailure());
        expect_scaled_passthrough(first.m_handler, in, out, 1.0F);
    }
    EXPECT_EQ(engine->m_released.load(), 1);
    EXPECT_EQ(stage->m_released.load(), 1);

    // Never prepared: released, never initialised.
    auto idle_engine = std::make_shared<CountingEngine>();
    auto idle_stage = std::make_shared<CountingStage>(anira::Stage::k_post_process);
    {
        const anira::Pipeline unused{
            anira::stage::Inference(model).engine(k_cxx_engine_id, idle_engine),
            anira::stage::Custom(idle_stage)};
    }
    EXPECT_EQ(idle_engine->m_inited + idle_stage->m_inited, 0);
    EXPECT_EQ(idle_engine->m_released.load(), 1) << "release owes nothing to init";
    EXPECT_EQ(idle_stage->m_released.load(), 1);
}

// The inference stage brings its engines along: stage::Inference::engine in an initializer
// list of the pipeline and through Pipeline::add registers each one before the stage is added,
// exactly as register_engine does, and the handler runs on it; a second pipeline registering the
// same object while the first lives reuses its C engine (no second copy); a second registration
// under one id is ANIRA_ERROR_INVALID_STATE and keeps no copy; release fires once, for the one C
// engine, with its last reference.
TEST(AbiCxx, InferenceRegistersItsEnginesWhenTheStageIsAdded) {
    const anira_test::Context context;
    const ModelConfig model = engine_stream_model();
    auto engine = std::make_shared<CountingEngine>();
    engine->m_gain = 2.0F;
    {
        const anira::Pipeline listed{
            anira::stage::Inference(model).engine(k_cxx_engine_id, engine)};
        EXPECT_EQ(engine.use_count(), 2) << "the object's C engine holds its copy";
        CHandler handler(context, listed);
        ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK)
            << handler.m_err.message;
        std::vector<float> in;
        std::vector<float> out;
        drive(handler.m_handler, 2, in, out);
        ASSERT_FALSE(HasFatalFailure());
        expect_scaled_passthrough(handler.m_handler, in, out, 2.0F);
        EXPECT_EQ(engine->last()->m_processed.load(), 2);

        // Through add, on a second pipeline: the stage lists what it brings, the pipeline reuses
        // the C engine the first one holds, and a second registration under the id on that
        // pipeline is refused by the C entry.
        anira::Pipeline added;
        anira::stage::Inference stage(model);
        stage.engine(k_cxx_engine_id, engine);
        ASSERT_EQ(stage.engines().size(), 1U);
        EXPECT_EQ(stage.engines()[0].first, k_cxx_engine_id);
        EXPECT_EQ(stage.engines()[0].second, engine);
        added.add(stage);
        EXPECT_EQ(engine.use_count(), 3) << "the stage's pair; the C engine's copy is the first's";
        const Thrown twice = thrown_by([&] { added.register_engine(k_cxx_engine_id, engine); });
        EXPECT_TRUE(twice.m_thrown);
        EXPECT_EQ(twice.m_status, ANIRA_ERROR_INVALID_STATE);
        EXPECT_NE(twice.m_what.find("already has an engine"), std::string::npos) << twice.m_what;
        EXPECT_EQ(engine.use_count(), 3) << "a refused registration keeps no copy";
        EXPECT_EQ(engine->m_released.load(), 0);
    }
    EXPECT_EQ(engine->m_released.load(), 1) << "once, for the one C engine";
    EXPECT_EQ(engine.use_count(), 1);
}

// register_engine refuses what it cannot describe, and a refused registration keeps no copy
// of the engine: a null engine before the C call, an id that is no reverse-URI name and a
// flags() bit the C header does not define by the C entries. A refused id comes after the C
// engine was created, so that C engine is dropped again and release answers it; a refused
// flag comes before it exists, and nothing is released.
TEST(AbiCxx, RegisterEngineRefusesANullEngineABadIdAndAnUnknownFlag) {
    anira::Pipeline pipe;
    const Thrown null_engine =
        thrown_by([&] { pipe.register_engine(k_cxx_engine_id, std::shared_ptr<anira::Engine>()); });
    EXPECT_TRUE(null_engine.m_thrown);
    EXPECT_EQ(null_engine.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(null_engine.m_what.find("null engine"), std::string::npos) << null_engine.m_what;

    auto engine = std::make_shared<CountingEngine>();
    const Thrown bad_id = thrown_by([&] { pipe.register_engine("noreverseuri", engine); });
    EXPECT_TRUE(bad_id.m_thrown);
    EXPECT_EQ(bad_id.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(bad_id.m_what.find("reverse-URI"), std::string::npos) << bad_id.m_what;
    EXPECT_EQ(engine.use_count(), 1);
    EXPECT_EQ(engine->m_released.load(), 1) << "the C engine the refused call created";

    engine->m_flags = 16U;  // the bit above the four the C header defines
    const Thrown bit = thrown_by([&] { pipe.register_engine(k_cxx_engine_id, engine); });
    EXPECT_TRUE(bit.m_thrown);
    EXPECT_EQ(bit.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(bit.m_what.find("flags"), std::string::npos) << bit.m_what;
    EXPECT_EQ(engine.use_count(), 1);
    EXPECT_EQ(engine->m_released.load(), 1) << "no C engine was created";

    engine->m_flags = ANIRA_ENGINE_FLAG_DYNAMIC_TIME;  // read again by the next registration
    pipe.register_engine(k_cxx_engine_id, engine);
    EXPECT_EQ(engine.use_count(), 2);
}

// providers() fills the descriptor's list: a candidate per listed provider is a plan per
// provider, each load's EngineLoadInfo says which one it is for (the enum's value, or DEFAULT
// with the custom name), the report's rows carry both, and a candidate naming a provider the
// object does not list is refused at create.
TEST(AbiCxx, ProvidersAreTheEnginesTwins) {
    const anira_test::Context context;
    auto engine = std::make_shared<CountingEngine>();
    engine->m_providers = {"coreml", "com.example.npu"};
    const anira::BackendId on_coreml{.struct_size = sizeof(anira::BackendId),
                                     .engine = ANIRA_ENGINE_NONE,
                                     .provider = ANIRA_PROVIDER_COREML,
                                     .engine_id = k_cxx_engine_id,
                                     .provider_id = nullptr};
    const anira::BackendId on_npu{.struct_size = sizeof(anira::BackendId),
                                  .engine = ANIRA_ENGINE_NONE,
                                  .provider = ANIRA_PROVIDER_DEFAULT,
                                  .engine_id = k_cxx_engine_id,
                                  .provider_id = "com.example.npu"};
    anira::Pipeline pipe;
    pipe.register_engine(k_cxx_engine_id, engine);
    pipe.inference(engine_stream_model(), {on_coreml, on_npu});
    CHandler handler(context, pipe);
    ASSERT_EQ(handler.m_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(anira_test::explicit_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine->m_inited, 1);
    EXPECT_EQ(engine->m_loaded, 2) << "a provider is part of the loaded model";
    ASSERT_EQ(engine->m_loads.size(), 2U);
    EXPECT_EQ(engine->m_loads[0]->m_provider, ANIRA_PROVIDER_COREML);
    EXPECT_EQ(engine->m_loads[0]->m_provider_id, "");
    EXPECT_EQ(engine->m_loads[1]->m_provider, ANIRA_PROVIDER_DEFAULT);
    EXPECT_EQ(engine->m_loads[1]->m_provider_id, "com.example.npu");
    const anira::PlanReport report(anira_handler_plan_report(handler.m_handler));
    const std::vector<anira_plan_info> plans = report.plans();
    ASSERT_EQ(plans.size(), 2U);
    EXPECT_EQ(plans[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
    EXPECT_EQ(plans[0].provider_id, nullptr);
    EXPECT_EQ(plans[1].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
    ASSERT_NE(plans[1].provider_id, nullptr);
    EXPECT_STREQ(plans[1].provider_id, "com.example.npu");
    handler.destroy();

    const anira::BackendId on_cuda{.struct_size = sizeof(anira::BackendId),
                                   .engine = ANIRA_ENGINE_NONE,
                                   .provider = ANIRA_PROVIDER_CUDA,
                                   .engine_id = k_cxx_engine_id,
                                   .provider_id = nullptr};
    anira::Pipeline other;
    other.register_engine(k_cxx_engine_id, engine);
    other.inference(engine_stream_model(), {on_cuda});
    const CHandler refused(context, other);
    EXPECT_EQ(refused.m_status, ANIRA_ERROR_NOT_SUPPORTED) << refused.m_err.message;
    EXPECT_NE(std::string(refused.m_err.message).find("'cuda'"), std::string::npos)
        << refused.m_err.message;
    EXPECT_EQ(engine->m_loaded, 2) << "a refused create loads nothing";
}

// One registration, one loaded model, one Prepared per handler: a session-exclusive model is
// loaded once with no shared instance, prepare returns an exclusive Prepared per handler
// (whose calls carry the flag and instance 0), process and reset of a handler run on its own
// Prepared, reset is seen at every new stream (after prepare, after anira_handler_reset, after
// a re-prepare) on the context of the process that follows, every Prepared is deleted exactly
// once, at the re-prepare or the destroy of its handler, and the Loaded with the last of
// them; release fires after all of them. Two handlers of a stateless model share one Loaded
// too, each on a Prepared of its own that claims the shared instance.
TEST(AbiCxx, APreparedPerHandlerDiesWithItsUnprepareAndResetSeesEveryNewStream) {
    const anira_test::Context context;
    const anira::ContractHandle contract = anira_test::explicit_contract();
    {
        ModelConfig model = engine_stream_model();
        model.state(ANIRA_MODEL_STATEFUL);
        auto engine = std::make_shared<CountingEngine>();
        std::optional<anira::Pipeline> pipe;
        pipe.emplace();
        pipe->register_engine(k_cxx_engine_id, engine).inference(model);
        CHandler first(context, *pipe);
        CHandler second(context, *pipe);
        ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
        pipe.reset();
        ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.prepare(contract), ANIRA_OK) << second.m_err.message;
        ASSERT_EQ(engine->m_loads.size(), 1U) << "a stateful model is loaded once too";
        ASSERT_EQ(engine->m_counts.size(), 2U) << "a Prepared per handler";
        const std::shared_ptr<EngineLoadCounts> loaded = engine->last_load();
        const std::shared_ptr<EnginePreparedCounts> of_first = engine->m_counts[0];
        const std::shared_ptr<EnginePreparedCounts> of_second = engine->m_counts[1];
        EXPECT_EQ(loaded->m_instances, 0U) << "no shared instance: each handler builds its own";
        EXPECT_TRUE(of_first->m_exclusive);
        EXPECT_TRUE(of_second->m_exclusive);

        std::vector<float> in_first;
        std::vector<float> out_first;
        std::vector<float> in_second;
        std::vector<float> out_second;
        drive(first.m_handler, 3, in_first, out_first);
        drive(second.m_handler, 2, in_second, out_second);
        ASSERT_FALSE(HasFatalFailure());
        expect_scaled_passthrough(first.m_handler, in_first, out_first, 1.0F);
        expect_scaled_passthrough(second.m_handler, in_second, out_second, 1.0F);
        EXPECT_EQ(of_first->m_processed.load(), 3);
        EXPECT_EQ(of_second->m_processed.load(), 2);
        EXPECT_EQ(of_first->m_broken.load() + of_second->m_broken.load(), 0);
        // reset ran for the first inference of each stream, on the context of the process that
        // followed, and runs again after anira_handler_reset.
        EXPECT_EQ(of_first->m_resets.load(), 1);
        EXPECT_EQ(of_second->m_resets.load(), 1);
        EXPECT_EQ(of_first->m_process_after_reset_same_ctx.load(), 1);
        EXPECT_EQ(of_first->m_process_after_reset_other_ctx.load(), 0);
        anira_handler_reset(first.m_handler);
        std::vector<float> in_again;
        std::vector<float> out_again;
        drive(first.m_handler, 1, in_again, out_again);
        ASSERT_FALSE(HasFatalFailure());
        EXPECT_EQ(of_first->m_resets.load(), 2);
        EXPECT_EQ(of_first->m_process_after_reset_same_ctx.load(), 2);
        EXPECT_EQ(of_second->m_resets.load(), 1);
        // A re-prepare of the first handler: its Prepared dies once and a new one takes over,
        // with a reset of its own at its first inference; the second handler's lives on, and
        // with it the Loaded.
        ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(engine->m_loads.size(), 1U) << "the second handler held the Loaded";
        ASSERT_EQ(engine->m_counts.size(), 3U);
        EXPECT_TRUE(of_first->m_deleted.load());
        EXPECT_FALSE(of_second->m_deleted.load());
        EXPECT_FALSE(engine->m_counts[2]->m_deleted.load());
        EXPECT_FALSE(loaded->m_unloaded.load());
        drive(first.m_handler, 1, in_again, out_again);
        ASSERT_FALSE(HasFatalFailure());
        EXPECT_EQ(engine->m_counts[2]->m_resets.load(), 1);
        EXPECT_EQ(engine->m_prepared, 3);
        EXPECT_EQ(engine->m_loaded, 1);
        // The destroy of each handler deletes its Prepared, the last one the Loaded too;
        // release fires after the last.
        first.destroy();
        EXPECT_TRUE(engine->m_counts[2]->m_deleted.load());
        EXPECT_FALSE(of_second->m_deleted.load());
        EXPECT_FALSE(loaded->m_unloaded.load());
        EXPECT_EQ(engine->m_released.load(), 0);
        second.destroy();
        EXPECT_TRUE(of_second->m_deleted.load());
        EXPECT_TRUE(loaded->m_unloaded.load());
        EXPECT_EQ(engine->m_released.load(), 1);
        EXPECT_EQ(engine.use_count(), 1);
    }
    {
        // Stateless: two handlers of one pipeline share one loaded model (the pool), each on a
        // Prepared of its own that claims the shared instance; the Loaded dies with the last
        // handler sharing it.
        const ModelConfig model = engine_stream_model();
        auto engine = std::make_shared<CountingEngine>();
        anira::Pipeline pipe;
        pipe.register_engine(k_cxx_engine_id, engine).inference(model);
        CHandler first(context, pipe);
        CHandler second(context, pipe);
        ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
        ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.prepare(contract), ANIRA_OK) << second.m_err.message;
        ASSERT_EQ(engine->m_loads.size(), 1U) << "one loaded model for two equal handlers";
        ASSERT_EQ(engine->m_counts.size(), 2U) << "a Prepared per handler";
        const std::shared_ptr<EngineLoadCounts> shared = engine->last_load();
        EXPECT_EQ(shared->m_instances, 1U);
        EXPECT_FALSE(engine->m_counts[0]->m_exclusive);
        EXPECT_FALSE(engine->m_counts[1]->m_exclusive);
        std::vector<float> in;
        std::vector<float> out;
        drive(first.m_handler, 2, in, out);
        drive(second.m_handler, 2, in, out);
        ASSERT_FALSE(HasFatalFailure());
        EXPECT_EQ(engine->m_counts[0]->m_processed.load(), 2);
        EXPECT_EQ(engine->m_counts[1]->m_processed.load(), 2);
        EXPECT_EQ(engine->m_counts[0]->m_broken.load() + engine->m_counts[1]->m_broken.load(), 0);
        EXPECT_EQ(engine->m_counts[0]->m_resets.load() + engine->m_counts[1]->m_resets.load(), 0)
            << "a shared handle is never reset";
        first.destroy();
        EXPECT_TRUE(engine->m_counts[0]->m_deleted.load());
        EXPECT_FALSE(shared->m_unloaded.load()) << "the second handler still shares it";
        second.destroy();
        EXPECT_TRUE(engine->m_counts[1]->m_deleted.load());
        EXPECT_TRUE(shared->m_unloaded.load());
        EXPECT_EQ(engine->m_released.load(), 0) << "the pipeline still carries the engine";
    }
}

// The Engine object is the engine's identity: one object registered on two Pipelines (the
// plugin-instance idiom: one object per binary, one Pipeline per instance) gives one C engine,
// so two handlers of equal configurations share one loaded model; release fires once. After
// every Pipeline holding it is gone, a registration creates a fresh C engine, answered by a
// release of its own.
TEST(AbiCxx, OneEngineObjectOnTwoPipelinesSharesOneLoadedModel) {
    const anira_test::Context context;
    const anira::ContractHandle contract = anira_test::explicit_contract();
    const ModelConfig model = engine_stream_model();
    auto engine = std::make_shared<CountingEngine>();
    {
        std::optional<anira::Pipeline> first_pipe;
        std::optional<anira::Pipeline> second_pipe;
        first_pipe.emplace();
        second_pipe.emplace();
        first_pipe->register_engine(k_cxx_engine_id, engine).inference(model);
        second_pipe->register_engine(k_cxx_engine_id, engine).inference(model);
        EXPECT_EQ(engine.use_count(), 2) << "one C engine, one copy";
        CHandler first(context, *first_pipe);
        CHandler second(context, *second_pipe);
        ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
        first_pipe.reset();
        second_pipe.reset();
        ASSERT_EQ(first.prepare(contract), ANIRA_OK) << first.m_err.message;
        ASSERT_EQ(second.prepare(contract), ANIRA_OK) << second.m_err.message;
        EXPECT_EQ(engine->m_inited, 1) << "one C engine: one init";
        EXPECT_EQ(engine->m_loaded, 1) << "one loaded model across the two pipelines";
        EXPECT_EQ(engine->m_prepared, 2) << "a Prepared per handler";
        std::vector<float> in;
        std::vector<float> out;
        drive(first.m_handler, 2, in, out);
        drive(second.m_handler, 2, in, out);
        ASSERT_FALSE(HasFatalFailure());
        EXPECT_EQ(engine->m_counts[0]->m_processed.load() + engine->m_counts[1]->m_processed.load(),
                  4);
        first.destroy();
        EXPECT_FALSE(engine->last_load()->m_unloaded.load());
        second.destroy();
        EXPECT_TRUE(engine->last_load()->m_unloaded.load());
    }
    EXPECT_EQ(engine->m_released.load(), 1);
    EXPECT_EQ(engine.use_count(), 1);
    {
        anira::Pipeline again;
        again.register_engine(k_cxx_engine_id, engine);
        EXPECT_EQ(engine.use_count(), 2) << "a fresh C engine";
    }
    EXPECT_EQ(engine->m_released.load(), 2);
    EXPECT_EQ(engine.use_count(), 1);
}

// The enum alias is EngineKind: what a model config takes and answers for a built-in engine,
// and what the stage context reports; the name Engine is the class.
TEST(AbiCxx, EngineKindIsTheEnumAlias) {
    static_assert(std::is_same_v<anira::EngineKind, anira_engine>);
    ModelConfig model = engine_stream_model();
    const anira::EngineKind custom = model.model_engine(0);
    EXPECT_EQ(custom, ANIRA_ENGINE_NONE);
    const anira::EngineKind torch = ANIRA_ENGINE_LIBTORCH;
    const uint32_t index = model.add_model_path(torch, "model.pt");
    EXPECT_EQ(model.model_engine(index), torch);
    model.default_engine(torch);
    EXPECT_EQ(model.model_engine_id(index), std::string_view{});
}
// ---- runtime tensors -------------------------------------------------------------------------

TEST(AbiCxx, TensorIsTheCStructWithFactoriesOnIt) {
    std::array<float, 6> data{};
    const std::array<int64_t, 2> shape{2, 3};
    const Tensor tensor = Tensor::from_host(data.data(), ANIRA_DTYPE_F32, shape);
    const anira_tensor* const record = &tensor;  // a Tensor* is an anira_tensor*
    EXPECT_EQ(static_cast<const void*>(record), static_cast<const void*>(&tensor));
    EXPECT_EQ(record->domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
    EXPECT_EQ(record->dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(record->ndim, 2U);
    EXPECT_EQ(record->shape[1], 3);
    EXPECT_EQ(record->release, nullptr);
    EXPECT_EQ(tensor.data_f32(), data.data());
    EXPECT_EQ(tensor.data(ANIRA_DTYPE_F32), data.data());
    EXPECT_EQ(tensor.data(ANIRA_DTYPE_I16), nullptr);
    EXPECT_EQ(tensor.num_elements(), 6U);
    EXPECT_EQ(tensor.extent(1), 3U);
    EXPECT_EQ(tensor.extent(2), 0U);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 6U) << "and the C entries take it as it is";

    const Tensor scalar = Tensor::from_host(data.data(), ANIRA_DTYPE_F32, {});
    EXPECT_EQ(scalar.ndim, 0U);
    EXPECT_EQ(scalar.num_elements(), 1U);

    // More extents than ANIRA_MAX_RANK: refused like the C entry (the all-zero record), never
    // truncated into a legal rank.
    const std::array<int64_t, 9> nine{1, 1, 1, 1, 1, 1, 1, 1, 1};
    const Tensor refused = Tensor::from_host(data.data(), ANIRA_DTYPE_F32, nine);
    EXPECT_EQ(refused.dtype, 0U);
    EXPECT_EQ(refused.ndim, 0U);
    EXPECT_EQ(refused.data_f32(), nullptr);
}

TEST(AbiCxx, TensorFactoriesNameTheirDomainAndFence) {
    std::array<float, 6> data{};
    const std::array<int64_t, 2> shape{2, 3};
    int event = 0;
    const auto domain_of = [](const Tensor& tensor) { return tensor.domain; };
    const auto domain = [](anira_domain value) { return static_cast<uint32_t>(value); };
    const auto kind = [](anira_sync_kind value) { return static_cast<uint32_t>(value); };
    EXPECT_EQ(domain_of(Tensor::from_pinned(data.data(), ANIRA_DTYPE_F32, shape)),
              domain(ANIRA_DOMAIN_HOST_PINNED));
    const Tensor cuda = Tensor::from_cuda(data.data(), 1, &event, ANIRA_DTYPE_F32, shape);
    EXPECT_EQ(cuda.domain, domain(ANIRA_DOMAIN_CUDA));
    EXPECT_EQ(cuda.handle.cuda.device, 1);
    EXPECT_EQ(cuda.acquire.kind, kind(ANIRA_SYNC_CUDA_EVENT));
    EXPECT_EQ(cuda.data_f32(), nullptr) << "a device tensor has no host pointer";
    EXPECT_EQ(cuda.num_elements(), 6U);
    const Tensor gl = Tensor::from_gl_buffer(7, 0x90D2, nullptr, ANIRA_DTYPE_F32, shape);
    EXPECT_EQ(gl.domain, domain(ANIRA_DOMAIN_GL_BUFFER));
    EXPECT_EQ(gl.handle.gl.id, 7U);
    EXPECT_EQ(gl.acquire.kind, kind(ANIRA_SYNC_NONE));
    const Tensor vulkan = Tensor::from_vulkan(1, 2, 3, 4, 5, ANIRA_DTYPE_F32, shape);
    EXPECT_EQ(vulkan.domain, domain(ANIRA_DOMAIN_VULKAN_BUFFER));
    EXPECT_EQ(vulkan.handle.vk.offset, 3U);
    EXPECT_EQ(vulkan.acquire.kind, kind(ANIRA_SYNC_VK_TIMELINE));
    EXPECT_EQ(vulkan.acquire.u.vk.value, 5U);
    const Tensor opaque = Tensor::from_opaque_fd(5, 4096, ANIRA_DTYPE_F32, shape);
    EXPECT_EQ(opaque.domain, domain(ANIRA_DOMAIN_OPAQUE_FD));
    EXPECT_EQ(opaque.handle.opaque.size, 4096U);
    SyncToken fence{};
    fence.kind = kind(ANIRA_SYNC_QUEUE_ORDERED);
    const Tensor wgpu = Tensor::from_wgpu_buffer(&event, 64, &fence, ANIRA_DTYPE_F32, shape);
    EXPECT_EQ(wgpu.domain, domain(ANIRA_DOMAIN_WGPU_BUFFER));
    EXPECT_EQ(wgpu.handle.wgpu.offset, 64U);
    EXPECT_EQ(wgpu.acquire.kind, kind(ANIRA_SYNC_QUEUE_ORDERED));
    const Tensor dmabuf = Tensor::from_dmabuf(5, 4096, 512, -1, ANIRA_DTYPE_F32, shape);
    EXPECT_EQ(dmabuf.domain, domain(ANIRA_DOMAIN_DMABUF));
    EXPECT_EQ(dmabuf.handle.dmabuf.offset, 512U);
    EXPECT_EQ(dmabuf.acquire.kind, kind(ANIRA_SYNC_NONE));
}

TEST(AbiCxx, TensorFromHostPlanarIsTypedByItsElement) {
    std::array<float, 3> left{};
    std::array<float, 3> right{};
    const std::array<int64_t, 2> shape{2, 3};
    std::array<float*, 2> channels{left.data(), right.data()};
    const Tensor planar = Tensor::from_host_planar<float>(channels, shape);
    EXPECT_EQ(planar.dtype, ANIRA_DTYPE_F32) << "the element type names the dtype";
    EXPECT_EQ(planar.flags, static_cast<uint32_t>(ANIRA_TENSOR_PLANAR));
    EXPECT_EQ(planar.handle.planes.count, 2U);
    EXPECT_EQ(planar.plane<float>(0), left.data());
    EXPECT_EQ(planar.plane<float>(1), right.data());
    EXPECT_EQ(planar.plane<float>(2), nullptr);
    EXPECT_EQ(planar.plane<int16_t>(0), nullptr) << "another dtype";
    EXPECT_EQ(planar.data_f32(), nullptr) << "a planar tensor is never one block";
    EXPECT_EQ(planar.num_elements(), 6U);

    // Read-only planes: a const element type sets ANIRA_TENSOR_READ_ONLY.
    const std::array<const float*, 2> read{left.data(), right.data()};
    const Tensor readable = Tensor::from_host_planar<const float>(read, shape);
    EXPECT_EQ(
        readable.flags,
        static_cast<uint32_t>(ANIRA_TENSOR_PLANAR) | static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY));
    EXPECT_EQ(readable.plane<const float>(1), right.data());

    // One pointer per plane: a span of another size than shape[0] is refused like the C entry.
    const std::array<int64_t, 2> three{3, 3};
    const Tensor refused = Tensor::from_host_planar<float>(channels, three);
    EXPECT_EQ(refused.dtype, 0U);
    EXPECT_EQ(refused.flags, 0U);
    EXPECT_EQ(refused.num_elements(), 0U);
}

TEST(AbiCxx, TensorFromDlpackThrowsErrorWithTheStatus) {
    anira_test::DlpackProducer producer;
    producer.m_managed.dl_tensor.device.device_type = anira_test::kDLCUDA;
    const Thrown device =
        thrown_by([&] { static_cast<void>(Tensor::from_dlpack(&producer.m_managed)); });
    EXPECT_TRUE(device.m_thrown);
    EXPECT_EQ(device.m_status, ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(device.m_what.find("dlpack: device type 2"), std::string::npos) << device.m_what;
    EXPECT_EQ(producer.m_deleted, 0) << "the caller keeps a managed tensor anira refused";

    const Thrown null = thrown_by([] { static_cast<void>(Tensor::from_dlpack(nullptr)); });
    EXPECT_TRUE(null.m_thrown);
    EXPECT_EQ(null.m_status, ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(null.m_what.find("dlpack: NULL managed tensor"), std::string::npos) << null.m_what;

    producer.m_managed.dl_tensor.device.device_type = anira_test::kDLCPU;
    Tensor tensor = Tensor::from_dlpack(&producer.m_managed);
    EXPECT_EQ(tensor.data_f32(), producer.m_data.data());
    EXPECT_EQ(tensor.num_elements(), 6U);
    EXPECT_EQ(tensor.strides[0], 3);
    ASSERT_NE(tensor.release, nullptr);
    EXPECT_EQ(producer.m_deleted, 0);
    tensor.release(&tensor);  // the holder of the record releases it: once
    EXPECT_EQ(producer.m_deleted, 1);
    EXPECT_EQ(tensor.release, nullptr);
}

TEST(AbiCxx, SyncTokenResetsAndDups) {
    int event = 0;
    SyncToken token{};
    EXPECT_EQ(token.kind, static_cast<uint32_t>(ANIRA_SYNC_NONE)) << "value-initialised: zero";
    token.kind = static_cast<uint32_t>(ANIRA_SYNC_CUDA_EVENT);
    token.u.cuda_event = &event;
    const SyncToken copy = token.dup();  // a non-owning kind: a plain copy
    EXPECT_EQ(copy.kind, static_cast<uint32_t>(ANIRA_SYNC_CUDA_EVENT));
    EXPECT_EQ(copy.u.cuda_event, &event);
    const anira_sync_token* const record = &token;  // a SyncToken* is an anira_sync_token*
    EXPECT_EQ(record->u.cuda_event, &event);
    token.reset();
    EXPECT_EQ(token.kind, static_cast<uint32_t>(ANIRA_SYNC_NONE));
    EXPECT_EQ(token.u.cuda_event, nullptr);

    // An owning kind whose fd cannot be duplicated: dup() throws with the entry's status. The
    // token is left alone afterwards (a reset would close a descriptor it never owned).
    SyncToken broken{};
    broken.kind = static_cast<uint32_t>(ANIRA_SYNC_SYNC_FILE_FD);
    broken.u.fd = std::numeric_limits<int32_t>::max();
    const Thrown refused = thrown_by([&] { static_cast<void>(broken.dup()); });
    EXPECT_TRUE(refused.m_thrown);
    EXPECT_EQ(refused.m_status, ANIRA_ERROR_INVALID_ARGUMENT) << "not an open descriptor";
    EXPECT_NE(refused.m_what.find("anira_sync_token_dup"), std::string::npos) << refused.m_what;
}
