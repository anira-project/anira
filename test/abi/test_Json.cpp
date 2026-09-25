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
#include "pair_read.h"

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

std::string context_text(const anira_context_config* config) {
    size_t len = 0;
    EXPECT_EQ(anira_context_config_to_json(config, nullptr, 0, &len), ANIRA_ERROR_BUFFER_TOO_SMALL);
    std::vector<char> buf(len + 1);
    EXPECT_EQ(anira_context_config_to_json(config, buf.data(), buf.size(), &len), ANIRA_OK);
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
    EXPECT_EQ(cfg.m_models[0].m_tensors.at("audio_in").m_name, "input_0")
        << "the string form of a tensor record";
    EXPECT_EQ(cfg.m_models[1].m_tensors.at("audio_in").m_name, "x") << "the object form";
    EXPECT_EQ(cfg.m_models[1].m_tensors.at("gain").m_layout,
              (std::vector<uint32_t>{0, ANIRA_AXIS_INSERT}))
        << "a layout with an inserted unit axis";
    EXPECT_TRUE(cfg.m_models[1].m_tensors.at("gain").m_name.empty());
    EXPECT_TRUE(cfg.m_models[2].m_tensors.empty()) << "no record: positional, the spec's order";
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
    EXPECT_EQ(cfg.m_anchor, "mask_out");
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
    EXPECT_EQ(in.m_overlap, 1024);
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
    EXPECT_NE(once.find("\"anchor\": \"mask_out\""), std::string::npos);
    EXPECT_NE(once.find("\"audio_in\": \"input_0\""), std::string::npos)
        << "a name-only record is written as a string";
    EXPECT_NE(once.find("\"gain\": {\n          \"layout\": [\n            0,\n            "
                        "\"insert\"\n          ]\n        }"),
              std::string::npos)
        << "a layout record is written as an object:\n"
        << once;
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

// Declared state: "role": "state" on both halves of a pair, "state_source" on the input half
// only. The document's order is the model's: the state first on the input side, last on the
// output side.
TEST(AbiJsonModel, AStatePairRoundTrips) {
    constexpr const char* k_text = R"({
        "models": [{"engine": "anira.v2.custom", "path": "custom-processor"}],
        "inputs": [
            {"name": "h_in", "dtype": "float32", "role": "state",
             "axes": [["batch", 1], ["feature", 64]], "state_source": "h_out"},
            {"name": "audio_in", "dtype": "float32", "role": "streamed",
             "axes": [["batch", 1], ["time", 512]], "window": {"min": 512, "max": 512}}
        ],
        "outputs": [
            {"name": "audio_out", "dtype": "float32", "role": "streamed",
             "axes": [["batch", 1], ["time", 512]], "window": {"min": 512, "max": 512}},
            {"name": "h_out", "dtype": "float32", "role": "state",
             "axes": [["batch", 1], ["feature", 64]]}
        ]
    })";
    const Loaded first(k_text);
    ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(first.m_config->m_inputs.size(), 2U);
    EXPECT_EQ(first.m_config->m_inputs[0].m_role, ANIRA_ROLE_STATE);
    EXPECT_EQ(first.m_config->m_inputs[0].m_state_source, "h_out");
    EXPECT_EQ(first.m_config->m_outputs[1].m_role, ANIRA_ROLE_STATE);
    EXPECT_TRUE(first.m_config->m_outputs[1].m_state_source.empty());

    const std::string once = model_text(first.m_config);
    EXPECT_NE(once.find("\"role\": \"state\""), std::string::npos) << once;
    EXPECT_NE(once.find("\"state_source\": \"h_out\""), std::string::npos) << once;
    EXPECT_EQ(once.find("\"state_source\""), once.rfind("\"state_source\""))
        << "written once, on the input half:\n"
        << once;
    const Loaded second(once.c_str());
    ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
    EXPECT_EQ(second.m_config->m_inputs[0].m_state_source, "h_out");
    EXPECT_EQ(model_text(second.m_config), once);

    // A source set from C is the same member.
    anira_tensor_spec* spec = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_tensor_spec_create("h_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE, &spec, &err),
              ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_set_state_source(spec, "h_out"), ANIRA_OK);
    EXPECT_EQ(spec->m_state_source, "h_out");
    anira_tensor_spec_destroy(spec);
}

TEST(AbiJsonModel, StateSourceRejectionsNameTheKeyPath) {
    // The pairing is stated on the input: the key on an output spec is refused where it stands.
    EXPECT_EQ(load_fails(R"({"outputs": [{"name": "h_out", "role": "state",
                                          "state_source": "h_out"}]})",
                         "outputs[0].state_source"),
              ANIRA_ERROR_JSON);
    // The name is a string.
    EXPECT_EQ(load_fails(R"({"inputs": [{"name": "h_in", "role": "state", "state_source": 3}]})",
                         "inputs[0].state_source"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(
        load_fails(R"({"inputs": [{"name": "h_in", "role": "state", "state_source": ["h_out"]}]})",
                   "inputs[0].state_source"),
        ANIRA_ERROR_JSON);
}

// The "provider" key beside "engine" pins the entry: the enum's spelling to its value, any
// other word to a custom provider's name, "default" to a neutral entry, as no key does; to_json
// writes the key back, so the pin survives a round trip and two entries of one engine may
// differ in it alone. The pair is two keys: a ':' in the engine word is refused naming the key.
TEST(AbiJsonModel, AProviderKeyPinsTheEntry) {
    const Loaded loaded(R"({"models": [
        {"engine": "executorch", "provider": "coreml", "path": "net.coreml.pte"},
        {"engine": "executorch", "provider": "com.example.npu", "path": "net.npu.pte"},
        {"engine": "com.example.engine", "provider": "fast", "path": "net.bin"},
        {"engine": "onnxruntime", "provider": "default", "path": "net.onnx"},
        {"engine": "libtorch", "path": "net.pt"}]})");
    ASSERT_EQ(loaded.m_status, ANIRA_OK) << loaded.m_err.message;
    const anira_model_config* config = loaded.m_config;
    EXPECT_EQ(anira_test::model_engine(config, 0),
              (anira_test::EngineRead{ANIRA_OK, ANIRA_ENGINE_EXECUTORCH, ""}));
    EXPECT_EQ(anira_test::model_provider(config, 0),
              (anira_test::ProviderRead{ANIRA_OK, ANIRA_PROVIDER_COREML, ""}));
    EXPECT_EQ(anira_test::model_provider(config, 1),
              (anira_test::ProviderRead{ANIRA_OK, ANIRA_PROVIDER_CUSTOM, "com.example.npu"}));
    EXPECT_EQ(anira_test::model_engine(config, 2),
              (anira_test::EngineRead{ANIRA_OK, ANIRA_ENGINE_CUSTOM, "com.example.engine"}));
    EXPECT_EQ(anira_test::model_provider(config, 2),
              (anira_test::ProviderRead{ANIRA_OK, ANIRA_PROVIDER_CUSTOM, "fast"}));
    EXPECT_EQ(anira_test::model_provider(config, 3),
              (anira_test::ProviderRead{ANIRA_OK, ANIRA_PROVIDER_DEFAULT, ""}))
        << "\"default\" spells a neutral entry";
    EXPECT_EQ(anira_test::model_provider(config, 4),
              (anira_test::ProviderRead{ANIRA_OK, ANIRA_PROVIDER_DEFAULT, ""}));
    const std::string text = model_text(loaded.m_config);
    EXPECT_NE(text.find("\"engine\": \"executorch\""), std::string::npos) << text;
    EXPECT_NE(text.find("\"provider\": \"coreml\""), std::string::npos) << text;
    EXPECT_NE(text.find("\"provider\": \"com.example.npu\""), std::string::npos) << text;
    EXPECT_NE(text.find("\"provider\": \"fast\""), std::string::npos) << text;
    EXPECT_EQ(text.find("\"provider\": \"default\""), std::string::npos)
        << "a neutral entry carries no provider key:\n"
        << text;
    const Loaded again(text.c_str());
    ASSERT_EQ(again.m_status, ANIRA_OK) << again.m_err.message;
    EXPECT_EQ(model_text(again.m_config), text);
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "executorch:coreml", "path": "x"}]})",
                         "models[0].engine"),
              ANIRA_ERROR_JSON)
        << "a suffix on the engine word: the pair is two keys";
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "executorch", "provider": "", "path": "x"}]})",
                         "models[0].provider"),
              ANIRA_ERROR_JSON)
        << "an empty provider";
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "foo", "provider": "coreml", "path": "x"}]})",
                         "models[0].engine"),
              ANIRA_ERROR_JSON)
        << "the engine word is checked as before";
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
    EXPECT_EQ(load_fails(R"({"anchor": "nope"})", "anchor"), ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"anchor": {"output": "x"}})", "anchor"), ANIRA_ERROR_JSON)
        << "the anchor is a bare canonical name";
    EXPECT_EQ(
        load_fails(R"({"inputs": [{"name": "a"}], "outputs": [{"name": "a"}]})", "outputs[0].name"),
        ANIRA_ERROR_JSON)
        << "canonical names are unique across both sides";
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "tflite", "path": "m",
                             "tensor_names": {"a": "b"}}]})",
                         "models[0].tensor_names"),
              ANIRA_ERROR_JSON)
        << "the pre-release key is refused by name, never read as an extension";
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "tflite", "path": "m",
                             "tensors": {"a": {"layout": [0, 0]}}}]})",
                         "models[0].tensors.a.layout"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "tflite", "path": "m",
                             "tensors": {"a": {"layout": [0, "flip"]}}}]})",
                         "models[0].tensors.a.layout[1]"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "tflite", "path": "m",
                             "tensors": {"a": {"nam": "x"}}}]})",
                         "models[0].tensors.a.nam"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "tflite", "path": "m", "tensors": {"a": {}}}]})",
                         "models[0].tensors.a"),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(load_fails(R"({"models": [{"engine": "tflite", "path": "m", "tensors": {"a": ""}}]})",
                         "models[0].tensors.a"),
              ANIRA_ERROR_JSON);
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
    // Resolved paths are joined in generic form: forward slashes on every platform.
    EXPECT_EQ(config->m_models[0].m_path,
              (dir / "sub" / "m.onnx").lexically_normal().generic_string());
    anira_model_config_destroy(config);
    config = nullptr;
    EXPECT_EQ(
        anira_model_config_from_json_file((dir / "missing.json").string().c_str(), &config, &err),
        ANIRA_ERROR_NO_SUCH_FILE);
    EXPECT_EQ(config, nullptr);
    EXPECT_NE(std::strstr(err.message, "missing.json"), nullptr);
    std::filesystem::remove_all(dir);
}

// ---- context file ----------------------------------------------------------------------------

TEST(AbiJsonContext, LoadsTheDocumentExampleAndRoundTrips) {
    anira_context_config* context_config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_context_config_from_json(anira_test::k_context_v3,
                                             std::strlen(anira_test::k_context_v3),
                                             &context_config,
                                             &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(context_config->m_num_threads, 0u) << "0 = bring your own threads";
    EXPECT_EQ(context_config->m_wait, ANIRA_WAIT_SPIN_BACKOFF);
    EXPECT_EQ(context_config->m_log_level, ANIRA_LOG_WARNING);
    EXPECT_EQ(context_config->m_log_drain, ANIRA_LOG_DRAIN_THREAD);
    EXPECT_EQ(context_config->m_queue_capacity, 512u);
    ASSERT_TRUE(context_config->m_cuda.has_value());
    const anira_cuda_desc cuda = context_config->m_cuda.value_or(anira_cuda_desc{});
    EXPECT_EQ(cuda.device, 1);
    EXPECT_EQ(cuda.pinned_pool_limit, 67108864u);
    EXPECT_EQ(cuda.ownership, static_cast<uint32_t>(ANIRA_OWNERSHIP_OWNED))
        << "JSON blocks are owned";
    ASSERT_TRUE(context_config->m_vulkan.has_value());
    EXPECT_EQ(context_config->m_vulkan.value_or(anira_vulkan_desc{}).device_index, 2);
    EXPECT_EQ(context_config->m_vulkan.value_or(anira_vulkan_desc{}).queue_family, 3u);
    EXPECT_TRUE(context_config->m_metal.has_value());
    ASSERT_TRUE(context_config->m_gl.has_value());
    EXPECT_EQ(context_config->m_gl.value_or(anira_gl_desc{}).threads,
              static_cast<uint32_t>(ANIRA_GL_CALLER_THREAD));
    EXPECT_FALSE(context_config->m_d3d12.has_value());
    EXPECT_FALSE(context_config->m_upgraded);
    const std::string once = context_text(context_config);
    anira_context_config* again = nullptr;
    ASSERT_EQ(anira_context_config_from_json(once.c_str(), once.size(), &again, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(context_text(again), once);
    anira_context_config_destroy(again);
    anira_context_config_destroy(context_config);
}

TEST(AbiJsonContext, AVulkanDeviceIndexSetFromCSurvivesToJson) {
    anira_context_config* context_config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_context_config_create(&context_config, &err), ANIRA_OK) << err.message;
    anira_vulkan_desc vulkan = ANIRA_VULKAN_DESC_INIT;
    vulkan.queue_family = 3;
    vulkan.device_index = 2;
    ASSERT_EQ(anira_context_config_set_vulkan(context_config, &vulkan), ANIRA_OK);
    const std::string text = context_text(context_config);
    EXPECT_NE(text.find("\"device\""), std::string::npos) << "the JSON key stays vulkan.device";
    anira_context_config* again = nullptr;
    ASSERT_EQ(anira_context_config_from_json(text.c_str(), text.size(), &again, &err), ANIRA_OK)
        << err.message;
    const anira_vulkan_desc loaded = again->m_vulkan.value_or(anira_vulkan_desc{});
    EXPECT_EQ(loaded.device_index, 2) << "the slot the C setter filled, through to_json and back";
    EXPECT_EQ(loaded.queue_family, 3u);
    EXPECT_EQ(context_text(again), text);
    anira_context_config_destroy(again);
    anira_context_config_destroy(context_config);
}

TEST(AbiJsonContext, Rejections) {
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
        anira_context_config* context_config = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_context_config_from_json(text, std::strlen(text), &context_config, &err),
                  ANIRA_ERROR_JSON)
            << text;
        EXPECT_EQ(context_config, nullptr);
        EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
    }
    anira_context_config* context_config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_context_config_from_json(npu, std::strlen(npu), &context_config, &err),
              ANIRA_OK)
        << err.message;
    EXPECT_NE(context_config->m_ext.find("npu"), nullptr)
        << "unknown keys are extensions that fail prepare by name";
    anira_context_config_destroy(context_config);
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

// "callback" names the policy; a file cannot carry the function, so a parsed contract has
// none and the host sets the pair afterwards (prepare refuses the policy without one).
TEST(AbiJsonContract, OnMissCallbackNamesThePolicyOnly) {
    anira_contract* hard = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    const char* text =
        R"({"hard": {"budget": {"ms": 1.8}, "warmup": {"fixed": 2}, "on_miss": "callback"}})";
    ASSERT_EQ(anira_contract_from_json(text, std::strlen(text), &hard, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(hard->hard()->m_on_miss, ANIRA_MISS_CALLBACK);
    EXPECT_EQ(hard->hard()->m_miss_fn, nullptr);
    EXPECT_EQ(hard->hard()->m_miss_user_data, nullptr);
    anira_contract_destroy(hard);

    const char* unknown = R"({"hard": {"on_miss": "call_back"}})";
    EXPECT_EQ(anira_contract_from_json(unknown, std::strlen(unknown), &hard, &err),
              ANIRA_ERROR_JSON);
}

TEST(AbiJsonContract, RingDtypesAreReadByTensorName) {
    static constexpr const char* k_text = R"({ "hard": {
        "block_min": 512, "block_max": 512, "rate": 48000, "budget": {"ms": 5},
        "ring_dtypes": {"audio_in": "int16", "audio_out": "float32"}
    } })";
    anira_contract* hard = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_contract_from_json(k_text, std::strlen(k_text), &hard, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(hard->hard()->m_ring_dtypes.size(), 2u);
    EXPECT_EQ(hard->hard()->m_ring_dtypes.at("audio_in"), ANIRA_DTYPE_I16);
    EXPECT_EQ(hard->hard()->m_ring_dtypes.at("audio_out"), ANIRA_DTYPE_F32);
    anira_contract_destroy(hard);

    static constexpr const char* k_bad = R"({ "hard": { "ring_dtypes": {"audio_in": "f32"} } })";
    hard = nullptr;
    EXPECT_EQ(anira_contract_from_json(k_bad, std::strlen(k_bad), &hard, &err), ANIRA_ERROR_JSON);
    EXPECT_NE(std::string(err.message).find("hard.ring_dtypes.audio_in"), std::string::npos)
        << err.message;
    EXPECT_EQ(hard, nullptr);
}

// The declared stream latency per output (anira_contract_hard_set_latency): non-negative
// integers by canonical name, at most INT32_MAX like the setter; whether a name is a Streamed
// output is prepare's question.
TEST(AbiJsonContract, LatenciesAreReadByTensorName) {
    static constexpr const char* k_text = R"({ "hard": {
        "block_min": 512, "block_max": 512, "rate": 48000, "budget": {"ms": 5},
        "latencies": {"audio_out": 1024, "aux_out": 0}
    } })";
    anira_contract* hard = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_contract_from_json(k_text, std::strlen(k_text), &hard, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(hard->hard()->m_latencies.size(), 2u);
    EXPECT_EQ(hard->hard()->m_latencies.at("audio_out"), 1024u);
    EXPECT_EQ(hard->hard()->m_latencies.at("aux_out"), 0u);
    anira_contract_destroy(hard);

    for (const char* bad : {R"({ "hard": { "latencies": {"audio_out": -1} } })",
                            R"({ "hard": { "latencies": {"audio_out": 1024.5} } })",
                            R"({ "hard": { "latencies": {"audio_out": "1024"} } })",
                            R"({ "hard": { "latencies": {"audio_out": 2147483648} } })"}) {
        hard = nullptr;
        err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_contract_from_json(bad, std::strlen(bad), &hard, &err), ANIRA_ERROR_JSON)
            << bad;
        EXPECT_NE(std::string(err.message).find("hard.latencies.audio_out"), std::string::npos)
            << err.message;
        EXPECT_EQ(hard, nullptr);
    }
    static constexpr const char* k_not_object = R"({ "hard": { "latencies": [1024] } })";
    EXPECT_EQ(anira_contract_from_json(k_not_object, std::strlen(k_not_object), &hard, &err),
              ANIRA_ERROR_JSON);
    EXPECT_NE(std::string(err.message).find("hard.latencies"), std::string::npos) << err.message;
}

// The declared host-end domain per tensor is a top-level key, common to both kinds like
// edge_cost, the words the lower-case suffixes of anira_domain.
TEST(AbiJsonContract, HostDomainsAreReadByTensorName) {
    static constexpr const char* k_hard = R"({ "hard": {
        "block_min": 512, "block_max": 512, "rate": 48000, "budget": {"ms": 5}
    }, "host_domains": {"audio_in": "host", "gain": "host_pinned"} })";
    anira_contract* contract = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_contract_from_json(k_hard, std::strlen(k_hard), &contract, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(contract->m_host_domains.size(), 2u);
    EXPECT_EQ(contract->m_host_domains.at("audio_in"), ANIRA_DOMAIN_HOST);
    EXPECT_EQ(contract->m_host_domains.at("gain"), ANIRA_DOMAIN_HOST_PINNED);
    anira_contract_destroy(contract);

    static constexpr const char* k_async = R"({ "async": {}, "host_domains": {"x": "cuda"} })";
    contract = nullptr;
    ASSERT_EQ(anira_contract_from_json(k_async, std::strlen(k_async), &contract, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(contract->m_host_domains.size(), 1u);
    EXPECT_EQ(contract->m_host_domains.at("x"), ANIRA_DOMAIN_CUDA);
    anira_contract_destroy(contract);

    static constexpr const char* k_bad = R"({ "hard": {}, "host_domains": {"audio_in": "gpu"} })";
    contract = nullptr;
    EXPECT_EQ(anira_contract_from_json(k_bad, std::strlen(k_bad), &contract, &err),
              ANIRA_ERROR_JSON);
    EXPECT_NE(std::string(err.message).find("host_domains.audio_in"), std::string::npos)
        << err.message;
    EXPECT_EQ(contract, nullptr);

    static constexpr const char* k_not_object = R"({ "hard": {}, "host_domains": "host" })";
    EXPECT_EQ(anira_contract_from_json(k_not_object, std::strlen(k_not_object), &contract, &err),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(contract, nullptr);
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
