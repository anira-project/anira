#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "capi/ext_registry.h"
#include "capi/handles.h"
#include "fixtures.h"

namespace {

struct Loaded {
    anira_model_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    anira_status m_status = ANIRA_OK;
    Loaded(const char* text, const char* base_dir = nullptr) {
        m_status =
            anira_model_config_from_json(text, std::strlen(text), base_dir, &m_config, &m_err);
    }
    ~Loaded() { anira_model_config_destroy(m_config); }
    Loaded(const Loaded&) = delete;
    Loaded& operator=(const Loaded&) = delete;
};

std::string model_text(const anira_model_config* config) {
    size_t len = 0;
    EXPECT_EQ(anira_model_config_to_json(config, nullptr, 0, &len), ANIRA_ERROR_BUFFER_TOO_SMALL);
    std::vector<char> buf(len + 1);
    EXPECT_EQ(anira_model_config_to_json(config, buf.data(), buf.size(), &len), ANIRA_OK);
    return {buf.data(), len};
}

std::string machine_text(const anira_machine_config* config) {
    size_t len = 0;
    EXPECT_EQ(anira_machine_config_to_json(config, nullptr, 0, &len), ANIRA_ERROR_BUFFER_TOO_SMALL);
    std::vector<char> buf(len + 1);
    EXPECT_EQ(anira_machine_config_to_json(config, buf.data(), buf.size(), &len), ANIRA_OK);
    return {buf.data(), len};
}

anira_status load_fails(const char* text, const char* expected_fragment) {
    anira_model_config* config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_status status =
        anira_model_config_from_json(text, std::strlen(text), nullptr, &config, &err);
    EXPECT_TRUE(ANIRA_FAILED(status)) << text;
    EXPECT_EQ(config, nullptr) << "out-parameters are written only on success";
    EXPECT_NE(std::strstr(err.message, expected_fragment), nullptr) << err.message;
    anira_model_config_destroy(config);
    return status;
}

}  // namespace

// ---- model file ------------------------------------------------------------------------------

TEST(AbiJsonModel, LoadsTheDocumentExample) {
    const Loaded m(anira_test::k_model_v3, "/base");
    ASSERT_EQ(m.m_status, ANIRA_OK) << m.m_err.message;
    const anira_model_config& cfg = *m.m_config;
    ASSERT_EQ(cfg.m_models.size(), 3u);
    EXPECT_EQ(cfg.m_models[0].m_engine, ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_EQ(cfg.m_models[0].m_path, "/base/model.onnx")
        << "relative paths resolve against base_dir";
    EXPECT_EQ(cfg.m_models[0].m_tensor_names.at("audio_in"), "input_0");
    EXPECT_EQ(cfg.m_models[1].m_engine, ANIRA_ENGINE_LIBTORCH);
    const auto* entry = cfg.m_models[1].m_ext.payload<anira::capi::EntryPayload>("entry");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->m_name, "forward_streaming");
    EXPECT_TRUE(cfg.m_models[2].is_custom());
    EXPECT_EQ(cfg.m_models[2].m_engine_id, "de.tu-berlin.coreml");
    EXPECT_EQ(cfg.m_models[2].m_path, "/abs/model.mlpackage") << "absolute paths stay";
    EXPECT_EQ(cfg.m_default_engine, ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_EQ(cfg.m_state, ANIRA_MODEL_STATELESS);
    EXPECT_EQ(cfg.m_max_instances, 4u);
    EXPECT_EQ(cfg.m_anchor_index, 0u);
    EXPECT_FALSE(cfg.m_anchor_is_input);
    ASSERT_EQ(cfg.m_inputs.size(), 2u);
    const anira_tensor_spec& in = cfg.m_inputs[0];
    EXPECT_EQ(in.m_name, "audio_in");
    EXPECT_EQ(in.m_dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(in.m_role, ANIRA_ROLE_STREAMED);
    EXPECT_EQ(in.m_ndim, 3u);
    EXPECT_EQ(in.m_axes[1].m_tag, ANIRA_AXIS_CHANNEL);
    EXPECT_EQ(in.m_axes[1].m_extent, 2);
    EXPECT_EQ(in.m_axes[2].m_tag, ANIRA_AXIS_TIME);
    EXPECT_EQ(in.m_axes[2].m_extent, ANIRA_DYNAMIC);
    EXPECT_EQ(in.m_window_min, 2048);
    EXPECT_EQ(in.m_window_max, 8192);
    EXPECT_EQ(in.m_context, 1024);
    EXPECT_EQ(cfg.m_inputs[1].m_role, ANIRA_ROLE_STATIC);
    EXPECT_EQ(cfg.m_inputs[1].m_dtype, ANIRA_DTYPE_F32) << "dtype defaults to float32";
    ASSERT_EQ(cfg.m_outputs.size(), 1u);
    EXPECT_EQ(cfg.m_outputs[0].m_window_max, ANIRA_UNBOUNDED);
    EXPECT_EQ(cfg.m_outputs[0].m_latency, 512);
    EXPECT_EQ(cfg.m_outputs[0].m_ratio_num, 1);
    EXPECT_EQ(cfg.m_outputs[0].m_ratio_den, 2);
    EXPECT_FALSE(cfg.m_upgraded);
    anira_contract* legacy = nullptr;
    EXPECT_EQ(anira_model_config_take_legacy_contract(m.m_config, &legacy), ANIRA_OK);
    EXPECT_EQ(legacy, nullptr) << "a v3 document carries no legacy contract";
}

TEST(AbiJsonModel, RoundTripIsByteStable) {
    const Loaded first(anira_test::k_model_v3);
    ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
    const std::string once = model_text(first.m_config);
    const Loaded second(once.c_str());
    ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
    EXPECT_EQ(model_text(second.m_config), once);
    EXPECT_NE(once.find("\"entry\": {"), std::string::npos) << "extensions are written back";
    EXPECT_NE(once.find("\"anchor\": {"), std::string::npos);
    EXPECT_NE(once.find("\"max\": \"unbounded\""), std::string::npos);
}

TEST(AbiJsonModel, BufferProtocol) {
    const Loaded m(anira_test::k_model_v3);
    ASSERT_EQ(m.m_status, ANIRA_OK);
    size_t len = 0;
    EXPECT_EQ(anira_model_config_to_json(m.m_config, nullptr, 0, &len),
              ANIRA_ERROR_BUFFER_TOO_SMALL);
    EXPECT_GT(len, 0u);
    std::vector<char> small(len);  // one byte short of the NUL
    size_t reported = 0;
    EXPECT_EQ(anira_model_config_to_json(m.m_config, small.data(), small.size(), &reported),
              ANIRA_ERROR_BUFFER_TOO_SMALL);
    EXPECT_EQ(reported, len) << "out_len is always written";
    EXPECT_EQ(anira_model_config_to_json(m.m_config, small.data(), small.size(), nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiJsonModel, RejectionsNameTheKeyPath) {
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "foo", "path": "x"}]})", "models[0].engine"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "onnxruntime"}]})", "models[0].path"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"models": [{"path": "x"}]})", "models[0].engine"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"inputs": [{"name": "a", "role": "weird"}]})", "inputs[0].role"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(
        load_fails(R"({"inputs": [{"name": "a", "axes": [["time", 0]]}]})", "inputs[0].axes[0][1]"),
        ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"inputs": [{"name": "a", "axes": [["sideways", 1]]}]})",
                         "inputs[0].axes[0][0]"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"inputs": [{"name": "a", "latency": 3}]})", "inputs[0].latency"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"inputs": [{"role": "streamed"}]})", "inputs[0].name"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"anchor": {"output": "nope"}})", "anchor.output"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"max_instances": 0})", "max_instances"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"state": 1})", "state"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails("{not json", "malformed"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails("[1, 2]", "not a JSON object"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"mystery": 3})", "mystery"), ANIRA_ERROR_JSON)
        << "a non-object unknown key cannot be an extension";
    EXPECT_EQ(
        load_fails(
            R"({"models": [{"engine": "libtorch", "path": "x", "entry": {"version": 2, "name": "f"}}]})",
            "version 2"),
        ANIRA_ERROR_EXTENSION_VERSION);
}

TEST(AbiJsonModel, UnknownObjectKeysAreStoredAsExtensions) {
    const Loaded m(
        R"({"models": [{"engine": "onnxruntime", "path": "x", "ort_session": {"graph_capture": true}}], "mystery": {"a": 1}})");
    ASSERT_EQ(m.m_status, ANIRA_OK) << m.m_err.message;
    const anira::capi::ExtSlot* slot = m.m_config->m_ext.find("mystery");
    ASSERT_NE(slot, nullptr);
    EXPECT_FALSE(slot->known());
    const anira::capi::ExtSlot* row = m.m_config->m_models[0].m_ext.find("ort_session");
    ASSERT_NE(row, nullptr);
    EXPECT_FALSE(row->known());
    const std::string text = model_text(m.m_config);
    EXPECT_NE(text.find("\"graph_capture\": true"), std::string::npos)
        << "unknown text is written back verbatim";
}

TEST(AbiJsonModel, FromFileUsesTheFilesDirectoryAsBaseDir) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "anira-abi-json-test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / "model.json";
    {
        std::ofstream out(file);
        out << R"({"models": [{"engine": "onnxruntime", "path": "sub/m.onnx"}]})";
    }
    anira_model_config* config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_model_config_from_json_file(file.string().c_str(), &config, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(config->m_models[0].m_path, (dir / "sub" / "m.onnx").lexically_normal().string());
    anira_model_config_destroy(config);
    config = nullptr;
    EXPECT_EQ(
        anira_model_config_from_json_file((dir / "missing.json").string().c_str(), &config, &err),
        ANIRA_ERROR_NO_SUCH_FILE);
    EXPECT_EQ(config, nullptr);
    EXPECT_NE(std::strstr(err.message, "missing.json"), nullptr);
    std::filesystem::remove_all(dir);
}

// ---- machine file ----------------------------------------------------------------------------

TEST(AbiJsonMachine, LoadsTheDocumentExampleAndRoundTrips) {
    anira_machine_config* mc = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_machine_config_from_json(anira_test::k_machine_v3,
                                             std::strlen(anira_test::k_machine_v3),
                                             &mc,
                                             &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(mc->m_num_threads, 0u) << "0 = bring your own threads";
    EXPECT_EQ(mc->m_wait, ANIRA_WAIT_SPIN_BACKOFF);
    EXPECT_EQ(mc->m_log_level, ANIRA_LOG_WARNING);
    EXPECT_EQ(mc->m_log_drain, ANIRA_LOG_DRAIN_THREAD);
    EXPECT_EQ(mc->m_queue_capacity, 512u);
    ASSERT_TRUE(mc->m_cuda.has_value());
    const anira_cuda_desc cuda = mc->m_cuda.value_or(anira_cuda_desc{});
    EXPECT_EQ(cuda.device, 1);
    EXPECT_EQ(cuda.pinned_pool_limit, 67108864u);
    EXPECT_EQ(cuda.ownership, static_cast<uint32_t>(ANIRA_OWNERSHIP_OWNED))
        << "JSON blocks are owned";
    ASSERT_TRUE(mc->m_vulkan.has_value());
    EXPECT_EQ(mc->m_vulkan_device, 2);
    EXPECT_EQ(mc->m_vulkan.value_or(anira_vulkan_desc{}).queue_family, 3u);
    EXPECT_TRUE(mc->m_metal.has_value());
    ASSERT_TRUE(mc->m_gl.has_value());
    EXPECT_EQ(mc->m_gl.value_or(anira_gl_desc{}).threads,
              static_cast<uint32_t>(ANIRA_GL_CALLER_THREAD));
    EXPECT_FALSE(mc->m_d3d12.has_value());
    EXPECT_FALSE(mc->m_upgraded);
    const std::string once = machine_text(mc);
    anira_machine_config* again = nullptr;
    ASSERT_EQ(anira_machine_config_from_json(once.c_str(), once.size(), &again, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(machine_text(again), once);
    anira_machine_config_destroy(again);
    anira_machine_config_destroy(mc);
}

TEST(AbiJsonMachine, Rejections) {
    const char* npu = R"({"npu": {"plugins": "/opt"}})";
    const std::vector<std::pair<const char*, const char*>> cases = {
        {R"({"log_level": "warning"})", "version 2"},
        {R"({"log": {"colour": "blue"}})", "log.colour"},
        {R"({"cuda": {"foo": 1}})", "cuda.foo"},
        {R"({"wait_strategy": "nap"})", "wait_strategy"},
        {R"({"num_threads": -1})", "num_threads"},
        {R"({"gl": {"threads": "many"}})", "gl.threads"},
    };
    for (const auto& [text, fragment] : cases) {
        anira_machine_config* mc = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_machine_config_from_json(text, std::strlen(text), &mc, &err),
                  ANIRA_ERROR_JSON)
            << text;
        EXPECT_EQ(mc, nullptr);
        EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
    }
    anira_machine_config* mc = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_machine_config_from_json(npu, std::strlen(npu), &mc, &err), ANIRA_OK)
        << err.message;
    EXPECT_NE(mc->m_ext.find("npu"), nullptr)
        << "unknown keys are extensions that fail prepare by name";
    anira_machine_config_destroy(mc);
}

// ---- contract file ---------------------------------------------------------------------------

TEST(AbiJsonContract, HardAndAsyncFiles) {
    anira_contract* hard = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_contract_from_json(anira_test::k_contract_hard_v3,
                                       std::strlen(anira_test::k_contract_hard_v3),
                                       &hard,
                                       &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(anira_contract_get_kind(hard), ANIRA_CONTRACT_HARD);
    EXPECT_EQ(hard->hard()->m_block_max, 512u);
    EXPECT_EQ(hard->hard()->m_rate, 48000.0);
    EXPECT_EQ(hard->hard()->m_budget, ANIRA_BUDGET_MEASURED);
    EXPECT_EQ(hard->hard()->m_warmup, ANIRA_WARMUP_UNTIL_STABLE);
    EXPECT_EQ(hard->m_edge_cost, ANIRA_EDGE_COST_PERMISSIVE);
    anira_contract_destroy(hard);

    anira_contract* async_contract = nullptr;
    ASSERT_EQ(anira_contract_from_json(anira_test::k_contract_async_v3,
                                       std::strlen(anira_test::k_contract_async_v3),
                                       &async_contract,
                                       &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(anira_contract_get_kind(async_contract), ANIRA_CONTRACT_ASYNC);
    EXPECT_DOUBLE_EQ(async_contract->asynchronous()->m_deadline_ms, 33.3);
    EXPECT_EQ(async_contract->asynchronous()->m_on_late, ANIRA_LATE_DROP);
    EXPECT_EQ(async_contract->m_edge_cost, ANIRA_EDGE_COST_STRICT);
    anira_contract_destroy(async_contract);

    const char* explicit_text =
        R"({"hard": {"budget": {"ms": 1.8}, "warmup": {"fixed": 200}, "on_miss": "zeros"}})";
    ASSERT_EQ(anira_contract_from_json(explicit_text, std::strlen(explicit_text), &hard, &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(hard->hard()->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_DOUBLE_EQ(hard->hard()->m_budget_ms, 1.8);
    EXPECT_EQ(hard->hard()->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(hard->hard()->m_warmup_iterations, 200u);
    EXPECT_EQ(hard->hard()->m_on_miss, ANIRA_MISS_ZEROS);
    EXPECT_EQ(hard->hard()->m_block_max, 0u) << "geometry keys are optional; the host patches them";
    anira_contract_destroy(hard);
}

TEST(AbiJsonContract, Rejections) {
    const std::vector<std::pair<const char*, const char*>> cases = {
        {R"({"hard": {}, "async": {}})", "exactly one root"},
        {R"({"edge_cost": "strict"})", "exactly one root"},
        {R"({"hard": {"budget": "guess"}})", "hard.budget"},
        {R"({"hard": {"warmup": {"fixed": -1}}})", "hard.warmup.fixed"},
        {R"({"hard": {"block_min": 9, "block_max": 1}})", "block_min exceeds"},
        {R"({"async": {"lanes": "two"}})", "async.lanes"},
        {R"({"async": {"deadline_ms": 1, "colour": 2}})", "async.colour"},
    };
    for (const auto& [text, fragment] : cases) {
        anira_contract* contract = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_contract_from_json(text, std::strlen(text), &contract, &err),
                  ANIRA_ERROR_JSON)
            << text;
        EXPECT_EQ(contract, nullptr);
        EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
    }
}
