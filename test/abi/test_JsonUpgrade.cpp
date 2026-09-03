#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../support/log_record_collector.h"
#include "capi/ext_registry.h"
#include "capi/handles.h"
#include "fixtures.h"

namespace {

struct Upgraded {
    anira_model_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    anira_status m_status = ANIRA_OK;
    Upgraded(const char* text, const char* base_dir = nullptr) {
        m_status =
            anira_model_config_from_json(text, std::strlen(text), base_dir, &m_config, &m_err);
    }
    ~Upgraded() { anira_model_config_destroy(m_config); }
    Upgraded(const Upgraded&) = delete;
    Upgraded& operator=(const Upgraded&) = delete;
};

std::string model_text(const anira_model_config* config) {
    size_t len = 0;
    anira_model_config_to_json(config, nullptr, 0, &len);
    std::vector<char> buf(len + 1);
    EXPECT_EQ(anira_model_config_to_json(config, buf.data(), buf.size(), &len), ANIRA_OK);
    return {buf.data(), len};
}

void expect_axes(const anira_tensor_spec& spec,
                 const std::vector<anira_axis_tag>& tags,
                 const std::vector<int64_t>& extents) {
    ASSERT_EQ(spec.m_ndim, tags.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        EXPECT_EQ(spec.m_axes[i].m_tag, tags[i]) << spec.m_name << " axis " << i;
        EXPECT_EQ(spec.m_axes[i].m_extent, extents[i]) << spec.m_name << " axis " << i;
    }
}

}  // namespace

// First in this translation unit, and the first version 2 load of the process when the
// test_abi binary runs its suites in registration order: the one-per-process warning.
#if defined(ENABLE_LOGGING) && !defined(NDEBUG)
TEST(AbiJsonUpgrade, WarnsOncePerProcess) {
    const anira::LogLevel previous = anira::get_log_level();
    anira::set_log_level(anira::LogLevel::Warning);
    anira_test::RecordCollector collector;
    {
        const Upgraded first(anira_test::k_simple_gain_v2);
        ASSERT_EQ(first.m_status, ANIRA_SUCCESS_UPGRADED) << first.m_err.message;
        const Upgraded second(anira_test::k_rave_v2);
        ASSERT_EQ(second.m_status, ANIRA_SUCCESS_UPGRADED) << second.m_err.message;
    }
    EXPECT_TRUE(collector.has("version 2 configuration", "native"));
    int count = 0;
    {
        const std::scoped_lock<std::mutex> lock(collector.m_mutex);
        for (const auto& record : collector.m_records) {
            if (record.m_message.find("version 2 configuration") != std::string::npos) {
                count += 1;
                EXPECT_EQ(record.m_group, "anira.config");
            }
        }
    }
    EXPECT_EQ(count, 1) << "one warning per process";
    anira::set_log_level(previous);
}
#endif

TEST(AbiJsonUpgrade, SimpleGainUpgrades) {
    const Upgraded m(anira_test::k_simple_gain_v2, "/models");
    ASSERT_EQ(m.m_status, ANIRA_SUCCESS_UPGRADED) << m.m_err.message;
    EXPECT_FALSE(ANIRA_FAILED(m.m_status));
    const anira_model_config& cfg = *m.m_config;
    EXPECT_TRUE(cfg.m_upgraded);
    ASSERT_EQ(cfg.m_models.size(), 5u)
        << "every row survives on every build; presence is prepare's";
    EXPECT_EQ(cfg.m_models[0].m_engine, ANIRA_ENGINE_LIBTORCH);
    EXPECT_EQ(cfg.m_models[1].m_engine, ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_EQ(cfg.m_models[2].m_engine, ANIRA_ENGINE_TFLITE);
    EXPECT_EQ(cfg.m_models[3].m_engine, ANIRA_ENGINE_LITERT);
    EXPECT_EQ(cfg.m_models[4].m_engine, ANIRA_ENGINE_EXECUTORCH);
    EXPECT_EQ(cfg.m_models[1].m_path, "/models/models/simple_gain_network_mono.onnx");
    ASSERT_EQ(cfg.m_inputs.size(), 2u);
    ASSERT_EQ(cfg.m_outputs.size(), 2u);
    EXPECT_EQ(cfg.m_inputs[0].m_name, "input_0");
    EXPECT_EQ(cfg.m_inputs[0].m_role, ANIRA_ROLE_STREAMED);
    expect_axes(cfg.m_inputs[0],
                {ANIRA_AXIS_ANY, ANIRA_AXIS_CHANNEL, ANIRA_AXIS_TIME},
                {1, 1, 512});
    EXPECT_EQ(cfg.m_inputs[0].m_window_min, 512);
    EXPECT_EQ(cfg.m_inputs[0].m_window_max, 512);
    EXPECT_EQ(cfg.m_inputs[0].m_context, 0);
    EXPECT_EQ(cfg.m_inputs[1].m_role, ANIRA_ROLE_STATIC) << "size 0 = static";
    expect_axes(cfg.m_inputs[1], {ANIRA_AXIS_ANY}, {1});
    EXPECT_EQ(cfg.m_outputs[0].m_role, ANIRA_ROLE_STREAMED);
    EXPECT_EQ(cfg.m_outputs[0].m_latency, 0);
    EXPECT_EQ(cfg.m_outputs[1].m_role, ANIRA_ROLE_STATIC);
    EXPECT_EQ(cfg.m_state, ANIRA_MODEL_STATELESS);
    EXPECT_EQ(cfg.m_max_instances, anira::InferenceConfig::Defaults::m_num_parallel_processors)
        << "absent in the file: the 2.x constructor's default";
    EXPECT_TRUE(cfg.m_anchor.empty()) << "the default: the first streamed tensor";

    anira_contract* legacy = nullptr;
    ASSERT_EQ(anira_model_config_take_legacy_contract(m.m_config, &legacy), ANIRA_OK);
    ASSERT_NE(legacy, nullptr);
    EXPECT_EQ(anira_contract_get_kind(legacy), ANIRA_CONTRACT_HARD);
    EXPECT_TRUE(legacy->m_legacy);
    EXPECT_EQ(legacy->hard()->m_budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_DOUBLE_EQ(legacy->hard()->m_budget_ms, 5.0);
    EXPECT_EQ(legacy->hard()->m_warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(legacy->hard()->m_warmup_iterations, 1u);
    EXPECT_DOUBLE_EQ(legacy->hard()->m_wait_ratio, 0.0);
    EXPECT_EQ(legacy->hard()->m_block_max, 0u) << "no geometry: the host patches it";
    anira_contract_destroy(legacy);
    legacy = nullptr;
    EXPECT_EQ(anira_model_config_take_legacy_contract(m.m_config, &legacy), ANIRA_OK);
    EXPECT_EQ(legacy, nullptr) << "a second take yields NULL";

    const std::string v3 = model_text(m.m_config);
    EXPECT_NE(v3.find("\"engine\": \"onnxruntime\""), std::string::npos)
        << "written back in v3 spelling";
    EXPECT_EQ(v3.find("max_inference_time"), std::string::npos)
        << "contract keys do not belong to the model file";
}

TEST(AbiJsonUpgrade, RaveShapedDocument) {
    const Upgraded m(anira_test::k_rave_v2);
    ASSERT_EQ(m.m_status, ANIRA_SUCCESS_UPGRADED) << m.m_err.message;
    const anira_model_config& cfg = *m.m_config;
    ASSERT_EQ(cfg.m_models.size(), 1u);
    const auto* entry = cfg.m_models[0].m_ext.payload<anira::capi::EntryPayload>("entry");
    ASSERT_NE(entry, nullptr) << "model_function becomes the entry extension";
    EXPECT_EQ(entry->m_name, "decode");
    EXPECT_EQ(cfg.m_state, ANIRA_MODEL_STATEFUL);
    EXPECT_EQ(cfg.m_max_instances, 3u);
    ASSERT_EQ(cfg.m_inputs.size(), 1u);
    expect_axes(cfg.m_inputs[0], {ANIRA_AXIS_ANY, ANIRA_AXIS_CHANNEL, ANIRA_AXIS_TIME}, {1, 4, 1});
    EXPECT_EQ(cfg.m_inputs[0].m_window_min, 1);
    EXPECT_EQ(cfg.m_inputs[0].m_context, 0);
    ASSERT_EQ(cfg.m_outputs.size(), 1u);
    expect_axes(cfg.m_outputs[0],
                {ANIRA_AXIS_ANY, ANIRA_AXIS_CHANNEL, ANIRA_AXIS_TIME},
                {1, 1, 2048});
    EXPECT_EQ(cfg.m_outputs[0].m_window_min, 2048);
    EXPECT_EQ(cfg.m_outputs[0].m_latency, 2048);

    // The same document through the other two loaders.
    anira_machine_config* mc = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_machine_config_from_json(anira_test::k_rave_v2,
                                             std::strlen(anira_test::k_rave_v2),
                                             &mc,
                                             &err),
              ANIRA_SUCCESS_UPGRADED)
        << err.message;
    EXPECT_EQ(mc->m_num_threads, 2u);
    EXPECT_EQ(mc->m_wait, ANIRA_WAIT_BLOCKING);
    EXPECT_EQ(mc->m_log_level, ANIRA_LOG_ERROR) << "the bare log_level key upgrades";
    EXPECT_TRUE(mc->m_upgraded);
    anira_machine_config_destroy(mc);
    anira_contract* contract = nullptr;
    ASSERT_EQ(anira_contract_from_json(anira_test::k_rave_v2,
                                       std::strlen(anira_test::k_rave_v2),
                                       &contract,
                                       &err),
              ANIRA_SUCCESS_UPGRADED)
        << err.message;
    EXPECT_EQ(anira_contract_get_kind(contract), ANIRA_CONTRACT_HARD);
    EXPECT_DOUBLE_EQ(contract->hard()->m_budget_ms, 42.66);
    EXPECT_EQ(contract->hard()->m_warmup_iterations, 5u);
    EXPECT_DOUBLE_EQ(contract->hard()->m_wait_ratio, 0.5);
    anira_contract_destroy(contract);
}

TEST(AbiJsonUpgrade, HybridShapedDocumentUsesThePerChannelWindow) {
    const Upgraded m(anira_test::k_hybrid_v2);
    ASSERT_EQ(m.m_status, ANIRA_SUCCESS_UPGRADED) << m.m_err.message;
    const anira_tensor_spec& in = m.m_config->m_inputs[0];
    expect_axes(in, {ANIRA_AXIS_ANY, ANIRA_AXIS_CHANNEL, ANIRA_AXIS_TIME}, {256, 1, 150});
    EXPECT_EQ(in.m_window_min, 38400) << "elements / channels";
    EXPECT_EQ(in.m_context, 38400 - 256);
    const anira_tensor_spec& out = m.m_config->m_outputs[0];
    // The axis carrying the per-channel element count (256) is Time; the other unit axis
    // carries the channel count 1.
    expect_axes(out, {ANIRA_AXIS_TIME, ANIRA_AXIS_CHANNEL}, {256, 1});
    EXPECT_EQ(out.m_window_min, 256);
    EXPECT_EQ(out.m_context, 0);
}

TEST(AbiJsonUpgrade, TensorShapeSpellings) {
    const char* nested =
        R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "CUSTOM"}],
        "tensor_shape": [{"input_shape": [[1, 1, 512]], "output_shape": [[1, 1, 512]]}], "max_inference_time": 1.0}})";
    const char* flat =
        R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "CUSTOM"}],
        "tensor_shape": [{"input_shape": [1, 1, 512], "output_shape": [1, 1, 512]}], "max_inference_time": 1.0}})";
    const Upgraded a(nested);
    const Upgraded b(flat);
    ASSERT_EQ(a.m_status, ANIRA_SUCCESS_UPGRADED) << a.m_err.message;
    ASSERT_EQ(b.m_status, ANIRA_SUCCESS_UPGRADED) << b.m_err.message;
    EXPECT_EQ(model_text(a.m_config), model_text(b.m_config))
        << "the flat shorthand equals the nested form";
    EXPECT_TRUE(a.m_config->m_models[0].is_custom());
    EXPECT_EQ(a.m_config->m_models[0].m_engine_id, "anira.v2.custom");
    EXPECT_EQ(a.m_config->m_inputs[0].m_window_min, 512)
        << "no processing_spec: the whole tensor per inference";
    EXPECT_EQ(a.m_config->m_inputs[0].m_context, 0);

    const char* universal =
        R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"}],
        "tensor_shape": [{"input_shape": [1, 1, 512], "output_shape": [1, 1, 512], "inference_backend": "UNIVERSAL"},
                         {"input_shape": [1, 1, 512], "output_shape": [1, 1, 512], "inference_backend": "ONNX"}], "max_inference_time": 1.0}})";
    const Upgraded c(universal);
    EXPECT_EQ(c.m_status, ANIRA_SUCCESS_UPGRADED) << c.m_err.message;

    // A per-backend entry that permutes unit axes becomes a layout on that backend's rows.
    const char* differing =
        R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"},
                                                {"model_path": "m.tflite", "inference_backend": "TFLITE"}],
        "tensor_shape": [{"input_shape": [1, 1, 512], "output_shape": [1, 1, 512]},
                         {"input_shape": [1, 512, 1], "output_shape": [1, 1, 512], "inference_backend": "TFLITE"}], "max_inference_time": 1.0}})";
    const Upgraded d(differing);
    ASSERT_EQ(d.m_status, ANIRA_SUCCESS_UPGRADED) << d.m_err.message;
    EXPECT_TRUE(d.m_config->m_models[0].m_tensors.empty()) << "the ONNX row is canonical";
    ASSERT_EQ(d.m_config->m_models[1].m_tensors.count("input_0"), 1u);
    EXPECT_EQ(d.m_config->m_models[1].m_tensors.at("input_0").m_layout,
              (std::vector<uint32_t>{0, 2, 1}));
    EXPECT_EQ(d.m_config->m_models[1].m_tensors.count("output_0"), 0u) << "equal dims: no record";
    EXPECT_NE(model_text(d.m_config)
                  .find("\"input_0\": {\n          \"layout\": [\n            0,\n            2,\n "
                        "           1\n          ]\n        }"),
              std::string::npos)
        << model_text(d.m_config);

    // A different rank by unit axes is a layout too (the squeezed batch axis, the rank-3 scalar).
    const char* rank =
        R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"},
                                                {"model_path": "m.tflite", "inference_backend": "TFLITE"}],
        "tensor_shape": [{"input_shape": [[1, 1, 512], [1]], "output_shape": [[1, 1, 512]]},
                         {"input_shape": [[1, 512], [1, 1, 1]], "output_shape": [[1, 1, 512]], "inference_backend": "TFLITE"}],
        "processing_spec": {"preprocess_input_size": [512, 0]}, "max_inference_time": 1.0}})";
    const Upgraded r(rank);
    ASSERT_EQ(r.m_status, ANIRA_SUCCESS_UPGRADED) << r.m_err.message;
    EXPECT_EQ(r.m_config->m_models[1].m_tensors.at("input_0").m_layout,
              (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(r.m_config->m_models[1].m_tensors.at("input_1").m_layout,
              (std::vector<uint32_t>{0, ANIRA_AXIS_INSERT, ANIRA_AXIS_INSERT}));

    // Moving an axis of another extent is not a layout the upgrade can write.
    const char* transposed =
        R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"}],
        "tensor_shape": [{"input_shape": [1, 2, 512], "output_shape": [1, 1, 512]},
                         {"input_shape": [1, 512, 2], "output_shape": [1, 1, 512], "inference_backend": "TFLITE"}], "max_inference_time": 1.0}})";
    const Upgraded t(transposed);
    EXPECT_EQ(t.m_status, ANIRA_ERROR_JSON);
    EXPECT_NE(std::strstr(t.m_err.message, "tensor_shape[1].input_shape[0]"), nullptr)
        << t.m_err.message;

    // The canonical entry is the universal one wherever it sits: HybridNNConfig.h lists the
    // TFLite and LiteRT rows first.
    const char* hybrid_order =
        R"({"inference_config": {"model_data": [{"model_path": "m.tflite", "inference_backend": "TFLITE"},
                                                {"model_path": "m.tflite", "inference_backend": "LITERT"},
                                                {"model_path": "m.pt", "inference_backend": "LIBTORCH"}],
        "tensor_shape": [{"input_shape": [256, 150, 1], "output_shape": [256, 1], "inference_backend": "TFLITE"},
                         {"input_shape": [256, 150, 1], "output_shape": [256, 1], "inference_backend": "LITERT"},
                         {"input_shape": [256, 1, 150], "output_shape": [256, 1]}],
        "processing_spec": {"preprocess_input_size": [256], "postprocess_output_size": [256]}, "max_inference_time": 1.0}})";
    const Upgraded h(hybrid_order);
    ASSERT_EQ(h.m_status, ANIRA_SUCCESS_UPGRADED) << h.m_err.message;
    expect_axes(h.m_config->m_inputs[0],
                {ANIRA_AXIS_ANY, ANIRA_AXIS_CHANNEL, ANIRA_AXIS_TIME},
                {256, 1, 150});
    EXPECT_EQ(h.m_config->m_models[0].m_tensors.at("input_0").m_layout,
              (std::vector<uint32_t>{0, 2, 1}));
    EXPECT_EQ(h.m_config->m_models[1].m_tensors.at("input_0").m_layout,
              (std::vector<uint32_t>{0, 2, 1}));
    EXPECT_TRUE(h.m_config->m_models[2].m_tensors.empty());
}

TEST(AbiJsonUpgrade, TimeFirstShapesKeepTheirTimeAxis) {
    // StatefulRNNConfig.h: {2048, 1, 1} for LibTorch, {1, 2048, 1} for TFLite. The axis carrying
    // the per-channel element count is Time, wherever it sits; the TFLite row gets a layout.
    const char* rnn =
        R"({"inference_config": {"model_data": [{"model_path": "m.pt", "inference_backend": "LIBTORCH"},
                                                {"model_path": "m.tflite", "inference_backend": "TFLITE"}],
        "tensor_shape": [{"input_shape": [2048, 1, 1], "output_shape": [2048, 1, 1], "inference_backend": "LIBTORCH"},
                         {"input_shape": [1, 2048, 1], "output_shape": [1, 2048, 1], "inference_backend": "TFLITE"}],
        "max_inference_time": 42.66, "warm_up": 2, "session_exclusive_processor": true}})";
    const Upgraded m(rnn);
    ASSERT_EQ(m.m_status, ANIRA_SUCCESS_UPGRADED) << m.m_err.message;
    expect_axes(m.m_config->m_inputs[0],
                {ANIRA_AXIS_TIME, ANIRA_AXIS_ANY, ANIRA_AXIS_CHANNEL},
                {2048, 1, 1});
    EXPECT_EQ(m.m_config->m_inputs[0].m_window_min, 2048);
    EXPECT_EQ(m.m_config->m_inputs[0].m_context, 0);
    EXPECT_EQ(m.m_config->m_models[1].m_tensors.at("input_0").m_layout,
              (std::vector<uint32_t>{1, 0, 2}));
    EXPECT_EQ(m.m_config->m_models[1].m_tensors.at("output_0").m_layout,
              (std::vector<uint32_t>{1, 0, 2}));
    EXPECT_EQ(m.m_config->m_state, ANIRA_MODEL_STATEFUL);
}

TEST(AbiJsonUpgrade, RejectionsNameTheKeyPath) {
    const std::vector<std::pair<const char*, const char*>> cases = {
        {R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "PYTORCH"}], "tensor_shape": [{"input_shape": [1], "output_shape": [1]}]}})",
         "inference_config.model_data[0].inference_backend"},
        {R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"}]}})",
         "inference_config.tensor_shape"},
        {R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"}], "tensor_shape": [{"input_shape": [1, 1, 512], "output_shape": [1]}], "processing_spec": {"preprocess_input_size": [1024]}}})",
         "processing size exceeds"},
        {R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"}], "tensor_shape": [{"input_shape": [1, 3, 512], "output_shape": [1]}], "processing_spec": {"preprocess_input_channels": [2]}}})",
         "channel count 2"},
        {R"({"inference_config": {"model_data": [{"model_path": "m", "inference_backend": "ONNX"}], "tensor_shape": [{"input_shape": [1], "output_shape": [1]}], "max_inference_time": -1}})",
         "max_inference_time"},
    };
    for (const auto& [text, fragment] : cases) {
        const Upgraded m(text);
        EXPECT_EQ(m.m_status, ANIRA_ERROR_JSON) << text;
        EXPECT_NE(std::strstr(m.m_err.message, fragment), nullptr) << m.m_err.message;
    }
}
