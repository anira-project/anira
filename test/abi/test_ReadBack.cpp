// The read-back of anira/abi/config.h: every getter against the setter or the loader that
// stored its answer, through the public entries only. The value getters answer a no-value
// constant for NULL; the status getters write their out-parameters on success only, after
// the checks in the order NULL handle, NULL out-parameter, contract kind, index.
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "fixtures.h"

namespace {

struct Spec {
    Spec(const char* name,
         anira_dtype dtype = ANIRA_DTYPE_F32,
         anira_role role = ANIRA_ROLE_STREAMED) {
        EXPECT_EQ(anira_tensor_spec_create(name, dtype, role, &m_spec, &m_err), ANIRA_OK)
            << m_err.message;
    }
    ~Spec() { anira_tensor_spec_destroy(m_spec); }
    Spec(const Spec&) = delete;
    Spec& operator=(const Spec&) = delete;
    anira_tensor_spec* m_spec = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

struct Model {
    Model() { EXPECT_EQ(anira_model_config_create(&m_config, &m_err), ANIRA_OK); }
    explicit Model(const char* text) {
        m_status =
            anira_model_config_from_json(text, std::strlen(text), nullptr, &m_config, &m_err);
    }
    ~Model() { anira_model_config_destroy(m_config); }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    anira_model_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    anira_status m_status = ANIRA_OK;
};

struct Contract {
    // A Hard contract with a geometry, an Async one, or one loaded from JSON text.
    Contract(uint32_t block_min, uint32_t block_max, double rate) {
        EXPECT_EQ(anira_contract_create_hard(block_min, block_max, rate, &m_contract, &m_err),
                  ANIRA_OK);
    }
    Contract() { EXPECT_EQ(anira_contract_create_async(&m_contract, &m_err), ANIRA_OK); }
    explicit Contract(const char* text) {
        m_status = anira_contract_from_json(text, std::strlen(text), &m_contract, &m_err);
    }
    ~Contract() { anira_contract_destroy(m_contract); }
    Contract(const Contract&) = delete;
    Contract& operator=(const Contract&) = delete;
    anira_contract* m_contract = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    anira_status m_status = ANIRA_OK;
};

struct ContextConfig {
    ContextConfig() { EXPECT_EQ(anira_context_config_create(&m_config, &m_err), ANIRA_OK); }
    explicit ContextConfig(const char* text) {
        m_status = anira_context_config_from_json(text, std::strlen(text), &m_config, &m_err);
    }
    ~ContextConfig() { anira_context_config_destroy(m_config); }
    ContextConfig(const ContextConfig&) = delete;
    ContextConfig& operator=(const ContextConfig&) = delete;
    anira_context_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    anira_status m_status = ANIRA_OK;
};

// The layout of one tensor of one entry through the count protocol: the length first, then
// the axes into a buffer of exactly that length.
std::vector<uint32_t> layout_of(const anira_model_config* config,
                                uint32_t model_index,
                                const char* canonical) {
    uint32_t count = 0;
    EXPECT_EQ(anira_model_config_tensor_layout(config, model_index, canonical, &count, nullptr),
              ANIRA_OK);
    std::vector<uint32_t> axes(count);
    if (count > 0) {
        EXPECT_EQ(
            anira_model_config_tensor_layout(config, model_index, canonical, &count, axes.data()),
            ANIRA_OK);
    }
    return axes;
}

// A backup function of ANIRA_MISS_CALLBACK; never called here.
anira_status ANIRA_CALL decline_miss(anira_handler* /*handler*/,
                                     const anira_tensor* /*inputs*/,
                                     uint32_t /*num_inputs*/,
                                     const anira_tensor* /*outputs*/,
                                     uint32_t /*num_outputs*/,
                                     void* /*user_data*/) ANIRA_NONBLOCKING {
    return ANIRA_ERROR_NOT_SUPPORTED;
}

void on_record(const anira_log_record* /*record*/, void* /*user_data*/) {}

}  // namespace

// ---- the tensor spec ------------------------------------------------------------------------

TEST(AbiReadBack, SpecAnswersWhatCreateStored) {
    const Spec spec("gain", ANIRA_DTYPE_I16, ANIRA_ROLE_STATIC);
    EXPECT_STREQ(anira_tensor_spec_name(spec.m_spec), "gain");
    EXPECT_EQ(anira_tensor_spec_dtype(spec.m_spec), ANIRA_DTYPE_I16);
    EXPECT_EQ(anira_tensor_spec_role(spec.m_spec), ANIRA_ROLE_STATIC);
    EXPECT_EQ(anira_tensor_spec_ndim(spec.m_spec), 0u) << "no axis set yet";
    EXPECT_EQ(anira_tensor_spec_latency(spec.m_spec), 0);
    EXPECT_EQ(anira_tensor_spec_state_source(spec.m_spec), nullptr);
    int64_t window_min = -1;
    int64_t window_max = -1;
    int64_t overlap = -1;
    EXPECT_EQ(anira_tensor_spec_window(spec.m_spec, &window_min, &window_max, &overlap), ANIRA_OK);
    EXPECT_EQ(window_min, 0) << "0, 0, 0 until set_window ran, on every role";
    EXPECT_EQ(window_max, 0);
    EXPECT_EQ(overlap, 0);
    int64_t num = -1;
    int64_t den = -1;
    EXPECT_EQ(anira_tensor_spec_time_ratio(spec.m_spec, &num, &den), ANIRA_OK);
    EXPECT_EQ(num, 0) << "(0, 0) = derive";
    EXPECT_EQ(den, 0);
}

TEST(AbiReadBack, SpecAxesIncludingAHoleAndDynamic) {
    const Spec spec("audio_in");
    ASSERT_EQ(anira_tensor_spec_set_axis(spec.m_spec, 0, ANIRA_AXIS_BATCH, 1), ANIRA_OK);
    ASSERT_EQ(anira_tensor_spec_set_axis(spec.m_spec, 2, ANIRA_AXIS_TIME, ANIRA_DYNAMIC), ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_ndim(spec.m_spec), 3u) << "one more than the highest axis set";
    anira_axis_tag tag = ANIRA_AXIS_CHANNEL;
    int64_t extent = -7;
    EXPECT_EQ(anira_tensor_spec_axis(spec.m_spec, 0, &tag, &extent), ANIRA_OK);
    EXPECT_EQ(tag, ANIRA_AXIS_BATCH);
    EXPECT_EQ(extent, 1);
    EXPECT_EQ(anira_tensor_spec_axis(spec.m_spec, 1, &tag, &extent), ANIRA_OK);
    EXPECT_EQ(tag, ANIRA_AXIS_ANY) << "a hole reads the default of an unset axis";
    EXPECT_EQ(extent, 0);
    EXPECT_EQ(anira_tensor_spec_axis(spec.m_spec, 2, &tag, &extent), ANIRA_OK);
    EXPECT_EQ(tag, ANIRA_AXIS_TIME);
    EXPECT_EQ(extent, ANIRA_DYNAMIC);
    // Out of range and NULL out-parameters: nothing written.
    tag = ANIRA_AXIS_CHANNEL;
    extent = -7;
    EXPECT_EQ(anira_tensor_spec_axis(spec.m_spec, 3, &tag, &extent), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_axis(spec.m_spec, 0, nullptr, &extent),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_axis(spec.m_spec, 0, &tag, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_axis(nullptr, 0, &tag, &extent), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(tag, ANIRA_AXIS_CHANNEL);
    EXPECT_EQ(extent, -7);
}

TEST(AbiReadBack, SpecWindowRatioLatencyAndStateSource) {
    const Spec spec("audio_out");
    ASSERT_EQ(anira_tensor_spec_set_window(spec.m_spec, 2048, ANIRA_UNBOUNDED, 1024), ANIRA_OK);
    ASSERT_EQ(anira_tensor_spec_set_time_ratio(spec.m_spec, 1, 2), ANIRA_OK);
    ASSERT_EQ(anira_tensor_spec_set_latency(spec.m_spec, 512), ANIRA_OK);
    int64_t window_min = 0;
    int64_t window_max = 0;
    int64_t overlap = 0;
    EXPECT_EQ(anira_tensor_spec_window(spec.m_spec, &window_min, &window_max, &overlap), ANIRA_OK);
    EXPECT_EQ(window_min, 2048);
    EXPECT_EQ(window_max, ANIRA_UNBOUNDED);
    EXPECT_EQ(overlap, 1024);
    int64_t num = 0;
    int64_t den = 0;
    EXPECT_EQ(anira_tensor_spec_time_ratio(spec.m_spec, &num, &den), ANIRA_OK);
    EXPECT_EQ(num, 1);
    EXPECT_EQ(den, 2);
    EXPECT_EQ(anira_tensor_spec_latency(spec.m_spec), 512);

    const Spec state("state_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
    EXPECT_EQ(anira_tensor_spec_state_source(state.m_spec), nullptr) << "unset";
    ASSERT_EQ(anira_tensor_spec_set_state_source(state.m_spec, "state_out"), ANIRA_OK);
    EXPECT_STREQ(anira_tensor_spec_state_source(state.m_spec), "state_out");

    // The NULL answers of both shapes.
    EXPECT_EQ(anira_tensor_spec_window(spec.m_spec, nullptr, &window_max, &overlap),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_window(nullptr, &window_min, &window_max, &overlap),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_time_ratio(spec.m_spec, &num, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_time_ratio(nullptr, &num, &den), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(window_min, 2048) << "untouched on failure";
    EXPECT_EQ(anira_tensor_spec_name(nullptr), nullptr);
    EXPECT_EQ(anira_tensor_spec_dtype(nullptr), 0u);
    EXPECT_EQ(anira_tensor_spec_role(nullptr), ANIRA_ROLE_FORCE32);
    EXPECT_EQ(anira_tensor_spec_ndim(nullptr), 0u);
    EXPECT_EQ(anira_tensor_spec_latency(nullptr), 0);
    EXPECT_EQ(anira_tensor_spec_state_source(nullptr), nullptr);
}

// ---- the model config ------------------------------------------------------------------------

// The pointers are the config's own copies, valid until the next add_input or add_output moves
// them: every read below is taken right after the add it follows (reading a dead pointer is
// undefined behaviour, so the rule is stated here, not tested).
TEST(AbiReadBack, ConfigSpecsAreTheConfigsCopies) {
    const Model model;
    const Spec in("audio_in");
    const Spec gain("gain", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    const Spec out("audio_out");
    ASSERT_EQ(anira_tensor_spec_set_latency(out.m_spec, 256), ANIRA_OK);
    EXPECT_EQ(anira_model_config_num_inputs(model.m_config), 0u);
    ASSERT_EQ(anira_model_config_add_input(model.m_config, in.m_spec), ANIRA_OK);
    ASSERT_EQ(anira_model_config_add_input(model.m_config, gain.m_spec), ANIRA_OK);
    ASSERT_EQ(anira_model_config_add_output(model.m_config, out.m_spec), ANIRA_OK);
    EXPECT_EQ(anira_model_config_num_inputs(model.m_config), 2u);
    EXPECT_EQ(anira_model_config_num_outputs(model.m_config), 1u);
    const anira_tensor_spec* first = anira_model_config_input(model.m_config, 0);
    ASSERT_NE(first, nullptr);
    EXPECT_NE(first, in.m_spec) << "the config's own copy, not the caller's spec";
    EXPECT_STREQ(anira_tensor_spec_name(first), "audio_in");
    EXPECT_EQ(anira_tensor_spec_role(anira_model_config_input(model.m_config, 1)),
              ANIRA_ROLE_STATIC);
    const anira_tensor_spec* output = anira_model_config_output(model.m_config, 0);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(anira_tensor_spec_latency(output), 256);
    // A later change of the caller's spec does not reach the copy.
    ASSERT_EQ(anira_tensor_spec_set_latency(out.m_spec, 1024), ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_latency(anira_model_config_output(model.m_config, 0)), 256);

    EXPECT_EQ(anira_model_config_input(model.m_config, 2), nullptr) << "out of range";
    EXPECT_EQ(anira_model_config_output(model.m_config, 1), nullptr);
    EXPECT_EQ(anira_model_config_input(nullptr, 0), nullptr);
    EXPECT_EQ(anira_model_config_output(nullptr, 0), nullptr);
    EXPECT_EQ(anira_model_config_num_inputs(nullptr), 0u);
    EXPECT_EQ(anira_model_config_num_outputs(nullptr), 0u);
}

TEST(AbiReadBack, ConfigScalars) {
    const Model model;
    // The defaults a fresh config answers.
    EXPECT_EQ(anira_model_config_default_engine(model.m_config), ANIRA_ENGINE_NONE);
    EXPECT_EQ(anira_model_config_default_engine_id(model.m_config), nullptr);
    EXPECT_EQ(anira_model_config_state(model.m_config), ANIRA_MODEL_STATELESS);
    EXPECT_EQ(anira_model_config_max_instances(model.m_config), 1u);
    EXPECT_EQ(anira_model_config_anchor(model.m_config), nullptr);

    ASSERT_EQ(anira_model_config_set_default_engine(model.m_config, ANIRA_ENGINE_LIBTORCH),
              ANIRA_OK);
    EXPECT_EQ(anira_model_config_default_engine(model.m_config), ANIRA_ENGINE_LIBTORCH);
    EXPECT_EQ(anira_model_config_default_engine_id(model.m_config), nullptr);
    ASSERT_EQ(anira_model_config_set_default_engine_custom(model.m_config, "org.example.engine"),
              ANIRA_OK);
    EXPECT_EQ(anira_model_config_default_engine(model.m_config), ANIRA_ENGINE_NONE)
        << "a custom default";
    EXPECT_STREQ(anira_model_config_default_engine_id(model.m_config), "org.example.engine");
    ASSERT_EQ(anira_model_config_set_default_engine(model.m_config, ANIRA_ENGINE_NONE), ANIRA_OK);
    EXPECT_EQ(anira_model_config_default_engine_id(model.m_config), nullptr) << "back to plan 0";

    ASSERT_EQ(anira_model_config_set_state(model.m_config, ANIRA_MODEL_STATEFUL), ANIRA_OK);
    EXPECT_EQ(anira_model_config_state(model.m_config), ANIRA_MODEL_STATEFUL);
    ASSERT_EQ(anira_model_config_set_max_instances(model.m_config, 4), ANIRA_OK);
    EXPECT_EQ(anira_model_config_max_instances(model.m_config), 4u);

    ASSERT_EQ(anira_model_config_set_anchor(model.m_config, "audio_out"), ANIRA_OK);
    EXPECT_STREQ(anira_model_config_anchor(model.m_config), "audio_out")
        << "as written, unresolved";
    ASSERT_EQ(anira_model_config_set_anchor(model.m_config, nullptr), ANIRA_OK);
    EXPECT_EQ(anira_model_config_anchor(model.m_config), nullptr) << "NULL restores the default";
    ASSERT_EQ(anira_model_config_set_anchor(model.m_config, "audio_out"), ANIRA_OK);
    ASSERT_EQ(anira_model_config_set_anchor(model.m_config, ""), ANIRA_OK);
    EXPECT_EQ(anira_model_config_anchor(model.m_config), nullptr) << "so does the empty name";

    EXPECT_EQ(anira_model_config_default_engine(nullptr), ANIRA_ENGINE_NONE);
    EXPECT_EQ(anira_model_config_default_engine_id(nullptr), nullptr);
    EXPECT_EQ(anira_model_config_state(nullptr), ANIRA_MODEL_STATE_FORCE32);
    EXPECT_EQ(anira_model_config_max_instances(nullptr), 0u);
    EXPECT_EQ(anira_model_config_anchor(nullptr), nullptr);
}

TEST(AbiReadBack, ConfigTensorRecords) {
    Model model;
    uint32_t index = 99;
    ASSERT_EQ(anira_model_config_add_model_path(model.m_config,
                                                ANIRA_ENGINE_ONNXRUNTIME,
                                                "model.onnx",
                                                &index,
                                                &model.m_err),
              ANIRA_OK);
    ASSERT_EQ(index, 0u);
    EXPECT_EQ(anira_model_config_tensor_name(model.m_config, 0, "audio_in"), nullptr)
        << "positional without a record";
    ASSERT_EQ(anira_model_config_set_tensor_name(model.m_config, 0, "audio_in", "input_0"),
              ANIRA_OK);
    EXPECT_STREQ(anira_model_config_tensor_name(model.m_config, 0, "audio_in"), "input_0");

    // The count protocol: no record is count 0 and OK, the NULL array asks for the length, a
    // short array is ANIRA_INCOMPLETE with its capacity written and the length in count.
    EXPECT_TRUE(layout_of(model.m_config, 0, "audio_in").empty()) << "a name record alone";
    EXPECT_TRUE(layout_of(model.m_config, 0, "gain").empty()) << "no record";
    const std::array<uint32_t, 3> layout = {0, 2, 1};
    ASSERT_EQ(anira_model_config_set_tensor_layout(model.m_config, 0, "audio_in", layout.data(), 3),
              ANIRA_OK);
    EXPECT_EQ(layout_of(model.m_config, 0, "audio_in"), (std::vector<uint32_t>{0, 2, 1}));
    std::array<uint32_t, 3> shorter = {7, 7, 7};
    uint32_t count = 2;
    EXPECT_EQ(
        anira_model_config_tensor_layout(model.m_config, 0, "audio_in", &count, shorter.data()),
        ANIRA_INCOMPLETE);
    EXPECT_EQ(count, 3u) << "the length";
    EXPECT_EQ(shorter[0], 0u);
    EXPECT_EQ(shorter[1], 2u);
    EXPECT_EQ(shorter[2], 7u) << "nothing past the capacity";
    const std::array<uint32_t, 2> inserted = {0, ANIRA_AXIS_INSERT};
    ASSERT_EQ(anira_model_config_set_tensor_layout(model.m_config, 0, "gain", inserted.data(), 2),
              ANIRA_OK);
    EXPECT_EQ(layout_of(model.m_config, 0, "gain"), (std::vector<uint32_t>{0, ANIRA_AXIS_INSERT}));
    EXPECT_EQ(anira_model_config_tensor_name(model.m_config, 0, "gain"), nullptr)
        << "a layout record alone binds positionally";
    // The clear: the layout goes, the name stays.
    ASSERT_EQ(anira_model_config_set_tensor_layout(model.m_config, 0, "audio_in", nullptr, 0),
              ANIRA_OK);
    EXPECT_TRUE(layout_of(model.m_config, 0, "audio_in").empty());
    EXPECT_STREQ(anira_model_config_tensor_name(model.m_config, 0, "audio_in"), "input_0");

    count = 5;
    EXPECT_EQ(anira_model_config_tensor_layout(model.m_config, 1, "gain", &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "index out of range";
    EXPECT_EQ(anira_model_config_tensor_layout(model.m_config, 0, "", &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_tensor_layout(model.m_config, 0, nullptr, &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_tensor_layout(model.m_config, 0, "gain", nullptr, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_tensor_layout(nullptr, 0, "gain", &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(count, 5u) << "untouched on failure";
    EXPECT_EQ(anira_model_config_tensor_name(model.m_config, 1, "audio_in"), nullptr);
    EXPECT_EQ(anira_model_config_tensor_name(model.m_config, 0, nullptr), nullptr);
    EXPECT_EQ(anira_model_config_tensor_name(nullptr, 0, "audio_in"), nullptr);
}

TEST(AbiReadBack, ConfigModelExtensions) {
    Model model;
    ASSERT_EQ(anira_model_config_add_model_path(model.m_config,
                                                ANIRA_ENGINE_LIBTORCH,
                                                "model.pt",
                                                nullptr,
                                                &model.m_err),
              ANIRA_OK);
    ASSERT_EQ(anira_model_config_add_model_path(model.m_config,
                                                ANIRA_ENGINE_EXECUTORCH,
                                                "model.pte",
                                                nullptr,
                                                &model.m_err),
              ANIRA_OK);
    EXPECT_EQ(anira_model_config_model_ext(model.m_config, 0, "entry"), nullptr) << "absent";

    // Set as a struct: the name is copied, the record read within its struct_size.
    std::string name = "forward_streaming";
    anira_ext_entry entry = ANIRA_EXT_ENTRY_INIT;
    entry.name = name.c_str();
    ASSERT_EQ(anira_model_config_set_model_ext(model.m_config, 0, &entry.header, &model.m_err),
              ANIRA_OK);
    name.assign("changed");
    const anira_ext_header* header = anira_model_config_model_ext(model.m_config, 0, "entry");
    ASSERT_NE(header, nullptr);
    EXPECT_STREQ(header->kind, "entry");
    EXPECT_EQ(header->version, 1u);
    ASSERT_GE(header->struct_size, sizeof(anira_ext_entry));
    EXPECT_NE(header, &entry.header) << "the config's own record";
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the header is the first member
    EXPECT_STREQ(reinterpret_cast<const anira_ext_entry*>(header)->name, "forward_streaming");

    // Set as JSON: the same typed record.
    static constexpr const char* k_entry = R"({"name": "decode"})";
    ASSERT_EQ(anira_model_config_set_model_ext_json(model.m_config,
                                                    1,
                                                    "entry",
                                                    k_entry,
                                                    std::strlen(k_entry),
                                                    &model.m_err),
              ANIRA_OK);
    header = anira_model_config_model_ext(model.m_config, 1, "entry");
    ASSERT_NE(header, nullptr);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the header is the first member
    EXPECT_STREQ(reinterpret_cast<const anira_ext_entry*>(header)->name, "decode");

    // An unknown kind is stored for prepare to name, and answers NULL.
    static constexpr const char* k_unknown = R"({"flag": true})";
    ASSERT_EQ(anira_model_config_set_model_ext_json(model.m_config,
                                                    0,
                                                    "com.example.unknown",
                                                    k_unknown,
                                                    std::strlen(k_unknown),
                                                    &model.m_err),
              ANIRA_OK);
    EXPECT_EQ(anira_model_config_model_ext(model.m_config, 0, "com.example.unknown"), nullptr);
    EXPECT_NE(anira_model_config_model_ext(model.m_config, 0, "entry"), nullptr)
        << "one slot per kind: the entry is still there";
    EXPECT_EQ(anira_model_config_model_ext(model.m_config, 2, "entry"), nullptr);
    EXPECT_EQ(anira_model_config_model_ext(model.m_config, 0, nullptr), nullptr);
    EXPECT_EQ(anira_model_config_model_ext(nullptr, 0, "entry"), nullptr);
}

TEST(AbiReadBack, ALoadedModelFileAnswersTheSame) {
    const Model model(anira_test::k_model_v3);
    ASSERT_EQ(model.m_status, ANIRA_OK) << model.m_err.message;
    const anira_model_config* config = model.m_config;
    EXPECT_EQ(anira_model_config_default_engine(config), ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_EQ(anira_model_config_state(config), ANIRA_MODEL_STATELESS);
    EXPECT_EQ(anira_model_config_max_instances(config), 4u);
    EXPECT_STREQ(anira_model_config_anchor(config), "mask_out");
    // The entries' tensor records: the string form, the object form, the inserted unit axis.
    EXPECT_STREQ(anira_model_config_tensor_name(config, 0, "audio_in"), "input_0");
    EXPECT_STREQ(anira_model_config_tensor_name(config, 1, "audio_in"), "x");
    EXPECT_EQ(layout_of(config, 1, "gain"), (std::vector<uint32_t>{0, ANIRA_AXIS_INSERT}));
    EXPECT_EQ(anira_model_config_tensor_name(config, 1, "gain"), nullptr);
    const anira_ext_header* header = anira_model_config_model_ext(config, 1, "entry");
    ASSERT_NE(header, nullptr);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the header is the first member
    EXPECT_STREQ(reinterpret_cast<const anira_ext_entry*>(header)->name, "forward_streaming");
    EXPECT_EQ(anira_model_config_model_ext(config, 0, "entry"), nullptr);

    ASSERT_EQ(anira_model_config_num_inputs(config), 2u);
    ASSERT_EQ(anira_model_config_num_outputs(config), 1u);
    const anira_tensor_spec* audio_in = anira_model_config_input(config, 0);
    EXPECT_STREQ(anira_tensor_spec_name(audio_in), "audio_in");
    EXPECT_EQ(anira_tensor_spec_dtype(audio_in), ANIRA_DTYPE_F32);
    EXPECT_EQ(anira_tensor_spec_role(audio_in), ANIRA_ROLE_STREAMED);
    ASSERT_EQ(anira_tensor_spec_ndim(audio_in), 3u);
    anira_axis_tag tag = ANIRA_AXIS_ANY;
    int64_t extent = 0;
    EXPECT_EQ(anira_tensor_spec_axis(audio_in, 1, &tag, &extent), ANIRA_OK);
    EXPECT_EQ(tag, ANIRA_AXIS_CHANNEL);
    EXPECT_EQ(extent, 2);
    EXPECT_EQ(anira_tensor_spec_axis(audio_in, 2, &tag, &extent), ANIRA_OK);
    EXPECT_EQ(tag, ANIRA_AXIS_TIME);
    EXPECT_EQ(extent, ANIRA_DYNAMIC) << "the dynamic Time extent";
    int64_t window_min = 0;
    int64_t window_max = 0;
    int64_t overlap = 0;
    EXPECT_EQ(anira_tensor_spec_window(audio_in, &window_min, &window_max, &overlap), ANIRA_OK);
    EXPECT_EQ(window_min, 2048);
    EXPECT_EQ(window_max, 8192);
    EXPECT_EQ(overlap, 1024);
    EXPECT_EQ(anira_tensor_spec_role(anira_model_config_input(config, 1)), ANIRA_ROLE_STATIC);

    const anira_tensor_spec* mask_out = anira_model_config_output(config, 0);
    EXPECT_STREQ(anira_tensor_spec_name(mask_out), "mask_out");
    EXPECT_EQ(anira_tensor_spec_latency(mask_out), 512);
    EXPECT_EQ(anira_tensor_spec_window(mask_out, &window_min, &window_max, &overlap), ANIRA_OK);
    EXPECT_EQ(window_max, ANIRA_UNBOUNDED);
    int64_t num = 0;
    int64_t den = 0;
    EXPECT_EQ(anira_tensor_spec_time_ratio(mask_out, &num, &den), ANIRA_OK);
    EXPECT_EQ(num, 1);
    EXPECT_EQ(den, 2);
}

// ---- the contract ----------------------------------------------------------------------------

TEST(AbiReadBack, HardContractAnswersEverySetter) {
    const Contract hard(64, 512, 48000.0);
    uint32_t block_min = 0;
    uint32_t block_max = 0;
    double rate = 0.0;
    EXPECT_EQ(anira_contract_hard_geometry(hard.m_contract, &block_min, &block_max, &rate),
              ANIRA_OK);
    EXPECT_EQ(block_min, 64u);
    EXPECT_EQ(block_max, 512u);
    EXPECT_EQ(rate, 48000.0);
    // The defaults of a fresh Hard contract.
    anira_budget_kind budget = ANIRA_BUDGET_EXPLICIT;
    double budget_ms = -1.0;
    EXPECT_EQ(anira_contract_hard_budget(hard.m_contract, &budget, &budget_ms), ANIRA_OK);
    EXPECT_EQ(budget, ANIRA_BUDGET_MEASURED);
    EXPECT_EQ(budget_ms, 0.0);
    anira_warmup_mode warmup = ANIRA_WARMUP_NONE;
    uint32_t iterations = 99;
    EXPECT_EQ(anira_contract_hard_warmup(hard.m_contract, &warmup, &iterations), ANIRA_OK);
    EXPECT_EQ(warmup, ANIRA_WARMUP_UNTIL_STABLE);
    EXPECT_EQ(iterations, 0u);
    anira_miss_policy policy = ANIRA_MISS_ZEROS;
    EXPECT_EQ(anira_contract_hard_on_miss(hard.m_contract, &policy), ANIRA_OK);
    EXPECT_EQ(policy, ANIRA_MISS_BYPASS);
    int user = 0;
    anira_miss_fn fn = &decline_miss;
    void* user_data = &user;
    EXPECT_EQ(anira_contract_hard_miss_fn(hard.m_contract, &fn, &user_data), ANIRA_OK);
    EXPECT_EQ(fn, nullptr);
    EXPECT_EQ(user_data, nullptr);
    double ratio = -1.0;
    EXPECT_EQ(anira_contract_hard_wait_ratio(hard.m_contract, &ratio), ANIRA_OK);
    EXPECT_EQ(ratio, 0.0);
    EXPECT_EQ(anira_contract_hard_num_ring_dtypes(hard.m_contract), 0u);
    EXPECT_EQ(anira_contract_edge_cost(hard.m_contract), ANIRA_EDGE_COST_PERMISSIVE);

    // Every setter, then its getter.
    ASSERT_EQ(anira_contract_hard_set_geometry(hard.m_contract, 128, 1024, 44100.0), ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_budget(hard.m_contract, ANIRA_BUDGET_EXPLICIT, 42.66),
              ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_warmup(hard.m_contract, ANIRA_WARMUP_FIXED, 5), ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_on_miss(hard.m_contract, ANIRA_MISS_CALLBACK), ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_miss_fn(hard.m_contract, &decline_miss, &user), ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_wait_ratio(hard.m_contract, 0.5), ANIRA_OK);
    ASSERT_EQ(anira_contract_set_edge_cost(hard.m_contract, ANIRA_EDGE_COST_STRICT), ANIRA_OK);
    EXPECT_EQ(anira_contract_hard_geometry(hard.m_contract, &block_min, &block_max, &rate),
              ANIRA_OK);
    EXPECT_EQ(block_min, 128u);
    EXPECT_EQ(block_max, 1024u);
    EXPECT_EQ(rate, 44100.0);
    EXPECT_EQ(anira_contract_hard_budget(hard.m_contract, &budget, &budget_ms), ANIRA_OK);
    EXPECT_EQ(budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(budget_ms, 42.66);
    EXPECT_EQ(anira_contract_hard_warmup(hard.m_contract, &warmup, &iterations), ANIRA_OK);
    EXPECT_EQ(warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(iterations, 5u);
    EXPECT_EQ(anira_contract_hard_on_miss(hard.m_contract, &policy), ANIRA_OK);
    EXPECT_EQ(policy, ANIRA_MISS_CALLBACK);
    EXPECT_EQ(anira_contract_hard_miss_fn(hard.m_contract, &fn, &user_data), ANIRA_OK);
    EXPECT_EQ(fn, &decline_miss);
    EXPECT_EQ(user_data, &user);
    EXPECT_EQ(anira_contract_hard_wait_ratio(hard.m_contract, &ratio), ANIRA_OK);
    EXPECT_EQ(ratio, 0.5);
    EXPECT_EQ(anira_contract_edge_cost(hard.m_contract), ANIRA_EDGE_COST_STRICT);
    // A budget kind back to MEASURED reads 0 ms; a warmup that is not FIXED reads 0 iterations.
    ASSERT_EQ(anira_contract_hard_set_budget(hard.m_contract, ANIRA_BUDGET_MEASURED, 0.0),
              ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_warmup(hard.m_contract, ANIRA_WARMUP_UNTIL_STABLE, 7),
              ANIRA_OK);
    EXPECT_EQ(anira_contract_hard_budget(hard.m_contract, &budget, &budget_ms), ANIRA_OK);
    EXPECT_EQ(budget_ms, 0.0);
    EXPECT_EQ(anira_contract_hard_warmup(hard.m_contract, &warmup, &iterations), ANIRA_OK);
    EXPECT_EQ(iterations, 0u);
}

TEST(AbiReadBack, RingDtypesAreEnumeratedInBytewiseOrder) {
    const Contract hard(512, 512, 48000.0);
    // Set out of order; "Z" sorts before "a" bytewise.
    ASSERT_EQ(anira_contract_hard_set_ring_dtype(hard.m_contract, "mask_out", ANIRA_DTYPE_F32),
              ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_ring_dtype(hard.m_contract, "audio_in", ANIRA_DTYPE_I16),
              ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_set_ring_dtype(hard.m_contract, "Zeta", ANIRA_DTYPE_I32),
              ANIRA_OK);
    ASSERT_EQ(anira_contract_hard_num_ring_dtypes(hard.m_contract), 3u);
    const std::array<const char*, 3> names = {"Zeta", "audio_in", "mask_out"};
    const std::array<anira_dtype, 3> dtypes = {ANIRA_DTYPE_I32, ANIRA_DTYPE_I16, ANIRA_DTYPE_F32};
    for (uint32_t i = 0; i < 3; ++i) {
        const char* canonical = nullptr;
        anira_dtype dtype = 0;
        EXPECT_EQ(anira_contract_hard_ring_dtype(hard.m_contract, i, &canonical, &dtype), ANIRA_OK);
        EXPECT_STREQ(canonical, names.at(i)) << i;
        EXPECT_EQ(dtype, dtypes.at(i)) << i;
    }
    // An overwrite keeps the count; set to F32 it is still listed (what a lookup could not tell).
    ASSERT_EQ(anira_contract_hard_set_ring_dtype(hard.m_contract, "audio_in", ANIRA_DTYPE_F32),
              ANIRA_OK);
    EXPECT_EQ(anira_contract_hard_num_ring_dtypes(hard.m_contract), 3u);
    const char* canonical = "untouched";
    anira_dtype dtype = ANIRA_DTYPE_I8;
    EXPECT_EQ(anira_contract_hard_ring_dtype(hard.m_contract, 1, &canonical, &dtype), ANIRA_OK);
    EXPECT_EQ(dtype, ANIRA_DTYPE_F32);
    canonical = "untouched";
    dtype = ANIRA_DTYPE_I8;
    EXPECT_EQ(anira_contract_hard_ring_dtype(hard.m_contract, 3, &canonical, &dtype),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_ring_dtype(hard.m_contract, 0, nullptr, &dtype),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_ring_dtype(hard.m_contract, 0, &canonical, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_ring_dtype(nullptr, 0, &canonical, &dtype),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_STREQ(canonical, "untouched");
    EXPECT_EQ(dtype, ANIRA_DTYPE_I8);
    EXPECT_EQ(anira_contract_hard_num_ring_dtypes(nullptr), 0u);
}

TEST(AbiReadBack, TheHardFamilyIsWrongContractOnAsync) {
    const Contract async_contract;
    uint32_t u = 7;
    double d = -1.0;
    anira_budget_kind budget = ANIRA_BUDGET_EXPLICIT;
    anira_warmup_mode warmup = ANIRA_WARMUP_NONE;
    anira_miss_policy policy = ANIRA_MISS_ZEROS;
    anira_miss_fn fn = &decline_miss;
    void* user_data = &u;
    const char* canonical = "untouched";
    anira_dtype dtype = ANIRA_DTYPE_I8;
    const anira_contract* c = async_contract.m_contract;
    EXPECT_EQ(anira_contract_hard_geometry(c, &u, &u, &d), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_budget(c, &budget, &d), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_warmup(c, &warmup, &u), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_on_miss(c, &policy), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_miss_fn(c, &fn, &user_data), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_wait_ratio(c, &d), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_ring_dtype(c, 0, &canonical, &dtype), ANIRA_ERROR_WRONG_CONTRACT)
        << "the kind before the index";
    EXPECT_EQ(anira_contract_hard_num_ring_dtypes(c), 0u);
    // Nothing written on any of them.
    EXPECT_EQ(u, 7u);
    EXPECT_EQ(d, -1.0);
    EXPECT_EQ(budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(warmup, ANIRA_WARMUP_NONE);
    EXPECT_EQ(policy, ANIRA_MISS_ZEROS);
    EXPECT_EQ(fn, &decline_miss);
    EXPECT_EQ(user_data, &u);
    EXPECT_STREQ(canonical, "untouched");
    EXPECT_EQ(dtype, ANIRA_DTYPE_I8);
    // The edge cost is common to both kinds.
    EXPECT_EQ(anira_contract_edge_cost(c), ANIRA_EDGE_COST_PERMISSIVE);
    ASSERT_EQ(anira_contract_set_edge_cost(async_contract.m_contract, ANIRA_EDGE_COST_STRICT),
              ANIRA_OK);
    EXPECT_EQ(anira_contract_edge_cost(c), ANIRA_EDGE_COST_STRICT);
}

TEST(AbiReadBack, TheNullAnswersOfTheContract) {
    const Contract hard(1, 1, 1.0);
    uint32_t u = 7;
    double d = -1.0;
    anira_budget_kind budget = ANIRA_BUDGET_EXPLICIT;
    anira_warmup_mode warmup = ANIRA_WARMUP_NONE;
    anira_miss_policy policy = ANIRA_MISS_ZEROS;
    anira_miss_fn fn = &decline_miss;
    void* user_data = &u;
    // A NULL handle first, then a NULL out-parameter: INVALID_ARGUMENT before the kind.
    EXPECT_EQ(anira_contract_hard_geometry(nullptr, &u, &u, &d), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_geometry(hard.m_contract, nullptr, &u, &d),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_geometry(hard.m_contract, &u, nullptr, &d),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_geometry(hard.m_contract, &u, &u, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_budget(nullptr, &budget, &d), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_budget(hard.m_contract, nullptr, &d),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_budget(hard.m_contract, &budget, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_warmup(nullptr, &warmup, &u), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_warmup(hard.m_contract, nullptr, &u),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_warmup(hard.m_contract, &warmup, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_on_miss(nullptr, &policy), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_on_miss(hard.m_contract, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_miss_fn(nullptr, &fn, &user_data), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_miss_fn(hard.m_contract, nullptr, &user_data),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_miss_fn(hard.m_contract, &fn, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_wait_ratio(nullptr, &d), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_wait_ratio(hard.m_contract, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    const Contract async_contract;
    EXPECT_EQ(anira_contract_hard_wait_ratio(async_contract.m_contract, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "the NULL out-parameter before the kind";
    EXPECT_EQ(u, 7u);
    EXPECT_EQ(d, -1.0);
    EXPECT_EQ(budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(warmup, ANIRA_WARMUP_NONE);
    EXPECT_EQ(policy, ANIRA_MISS_ZEROS);
    EXPECT_EQ(fn, &decline_miss);
    EXPECT_EQ(anira_contract_edge_cost(nullptr), ANIRA_EDGE_COST_PERMISSIVE);
}

TEST(AbiReadBack, ALoadedContractFileAnswersTheSame) {
    const Contract hard(anira_test::k_contract_hard_v3);
    ASSERT_EQ(hard.m_status, ANIRA_OK) << hard.m_err.message;
    uint32_t block_min = 0;
    uint32_t block_max = 0;
    double rate = 0.0;
    EXPECT_EQ(anira_contract_hard_geometry(hard.m_contract, &block_min, &block_max, &rate),
              ANIRA_OK);
    EXPECT_EQ(block_min, 512u);
    EXPECT_EQ(block_max, 512u);
    EXPECT_EQ(rate, 48000.0);
    anira_budget_kind budget = ANIRA_BUDGET_EXPLICIT;
    double budget_ms = -1.0;
    EXPECT_EQ(anira_contract_hard_budget(hard.m_contract, &budget, &budget_ms), ANIRA_OK);
    EXPECT_EQ(budget, ANIRA_BUDGET_MEASURED);
    anira_miss_policy policy = ANIRA_MISS_ZEROS;
    EXPECT_EQ(anira_contract_hard_on_miss(hard.m_contract, &policy), ANIRA_OK);
    EXPECT_EQ(policy, ANIRA_MISS_BYPASS);
    EXPECT_EQ(anira_contract_hard_num_ring_dtypes(hard.m_contract), 0u);

    static constexpr const char* k_typed = R"({ "hard": {
        "block_min": 256, "block_max": 1024, "rate": 44100,
        "budget": {"ms": 2.5}, "warmup": {"fixed": 3}, "on_miss": "zeros", "wait_ratio": 0.25,
        "ring_dtypes": {"audio_in": "int16"}
    }, "edge_cost": "strict" })";
    const Contract typed(k_typed);
    ASSERT_EQ(typed.m_status, ANIRA_OK) << typed.m_err.message;
    EXPECT_EQ(anira_contract_hard_budget(typed.m_contract, &budget, &budget_ms), ANIRA_OK);
    EXPECT_EQ(budget, ANIRA_BUDGET_EXPLICIT);
    EXPECT_EQ(budget_ms, 2.5);
    anira_warmup_mode warmup = ANIRA_WARMUP_NONE;
    uint32_t iterations = 0;
    EXPECT_EQ(anira_contract_hard_warmup(typed.m_contract, &warmup, &iterations), ANIRA_OK);
    EXPECT_EQ(warmup, ANIRA_WARMUP_FIXED);
    EXPECT_EQ(iterations, 3u);
    EXPECT_EQ(anira_contract_hard_on_miss(typed.m_contract, &policy), ANIRA_OK);
    EXPECT_EQ(policy, ANIRA_MISS_ZEROS);
    double ratio = 0.0;
    EXPECT_EQ(anira_contract_hard_wait_ratio(typed.m_contract, &ratio), ANIRA_OK);
    EXPECT_EQ(ratio, 0.25);
    ASSERT_EQ(anira_contract_hard_num_ring_dtypes(typed.m_contract), 1u);
    const char* canonical = nullptr;
    anira_dtype dtype = 0;
    EXPECT_EQ(anira_contract_hard_ring_dtype(typed.m_contract, 0, &canonical, &dtype), ANIRA_OK);
    EXPECT_STREQ(canonical, "audio_in");
    EXPECT_EQ(dtype, ANIRA_DTYPE_I16);
    EXPECT_EQ(anira_contract_edge_cost(typed.m_contract), ANIRA_EDGE_COST_STRICT);

    const Contract async_contract(anira_test::k_contract_async_v3);
    ASSERT_EQ(async_contract.m_status, ANIRA_OK) << async_contract.m_err.message;
    EXPECT_EQ(anira_contract_hard_on_miss(async_contract.m_contract, &policy),
              ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_edge_cost(async_contract.m_contract), ANIRA_EDGE_COST_STRICT);
}

// ---- the context config ----------------------------------------------------------------------

TEST(AbiReadBack, ContextConfigThreadsAndLog) {
    const ContextConfig config;
    uint32_t num_threads = 0;
    anira_wait_strategy wait = ANIRA_WAIT_BLOCKING;
    EXPECT_EQ(anira_context_config_threads(config.m_config, &num_threads, &wait), ANIRA_OK);
    EXPECT_EQ(num_threads, ANIRA_THREADS_AUTO) << "unset";
    EXPECT_EQ(wait, ANIRA_WAIT_SPIN_BACKOFF);
    ASSERT_EQ(anira_context_config_set_threads(config.m_config, 3, ANIRA_WAIT_BLOCKING), ANIRA_OK);
    EXPECT_EQ(anira_context_config_threads(config.m_config, &num_threads, &wait), ANIRA_OK);
    EXPECT_EQ(num_threads, 3u);
    EXPECT_EQ(wait, ANIRA_WAIT_BLOCKING);

    // The defaults read back as the record's initializer spells them.
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    desc.level = ANIRA_LOG_DEBUG;
    EXPECT_EQ(anira_context_config_log(config.m_config, &desc), ANIRA_OK);
    const anira_log_desc defaults = ANIRA_LOG_DESC_INIT;
    EXPECT_EQ(desc.struct_size, sizeof(anira_log_desc));
    EXPECT_EQ(desc.abi_version, ANIRA_ABI_VERSION);
    EXPECT_EQ(desc.callback, nullptr);
    EXPECT_EQ(desc.user_data, nullptr);
    EXPECT_EQ(desc.level, defaults.level);
    EXPECT_EQ(desc.drain, defaults.drain);
    EXPECT_EQ(desc.queue_capacity, defaults.queue_capacity);
    EXPECT_EQ(desc.drain_interval_ms, defaults.drain_interval_ms);
    EXPECT_EQ(desc.flags, 0u);

    // Every field set_log stores, back through the getter.
    int user = 0;
    anira_log_desc set = ANIRA_LOG_DESC_INIT;
    set.callback = on_record;
    set.user_data = &user;
    set.level = ANIRA_LOG_ERROR;
    set.drain = ANIRA_LOG_DRAIN_MANUAL;
    set.queue_capacity = 10;  // clamped to 64
    set.drain_interval_ms = 25;
    set.flags = ANIRA_LOG_FLAG_TRACE_FAILURES;
    ASSERT_EQ(anira_context_config_set_log(config.m_config, &set), ANIRA_OK);
    anira_log_desc got = ANIRA_LOG_DESC_INIT;
    EXPECT_EQ(anira_context_config_log(config.m_config, &got), ANIRA_OK);
    EXPECT_EQ(got.callback, &on_record);
    EXPECT_EQ(got.user_data, &user);
    EXPECT_EQ(got.level, static_cast<uint32_t>(ANIRA_LOG_ERROR));
    EXPECT_EQ(got.drain, static_cast<uint32_t>(ANIRA_LOG_DRAIN_MANUAL));
    EXPECT_EQ(got.queue_capacity, 64u) << "clamped as the setter clamps it";
    EXPECT_EQ(got.drain_interval_ms, 25u);
    EXPECT_EQ(got.flags, static_cast<uint32_t>(ANIRA_LOG_FLAG_TRACE_FAILURES));

    // A shorter record is written within its struct_size, which it keeps.
    anira_log_desc shorter = ANIRA_LOG_DESC_INIT;
    shorter.struct_size = static_cast<uint32_t>(offsetof(anira_log_desc, level));
    shorter.level = ANIRA_LOG_INFO;
    EXPECT_EQ(anira_context_config_log(config.m_config, &shorter), ANIRA_OK);
    EXPECT_EQ(shorter.struct_size, offsetof(anira_log_desc, level));
    EXPECT_EQ(shorter.callback, &on_record);
    EXPECT_EQ(shorter.level, static_cast<uint32_t>(ANIRA_LOG_INFO)) << "past struct_size";
    // Below the three leading slots, and the NULL answers.
    anira_log_desc tiny = ANIRA_LOG_DESC_INIT;
    tiny.struct_size = static_cast<uint32_t>(sizeof(uint32_t));
    EXPECT_EQ(anira_context_config_log(config.m_config, &tiny), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_context_config_log(config.m_config, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_context_config_log(nullptr, &got), ANIRA_ERROR_INVALID_ARGUMENT);
    num_threads = 11;
    EXPECT_EQ(anira_context_config_threads(config.m_config, nullptr, &wait),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_context_config_threads(config.m_config, &num_threads, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_context_config_threads(nullptr, &num_threads, &wait),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(num_threads, 11u);
}

TEST(AbiReadBack, ALoadedContextFileAnswersTheSame) {
    const ContextConfig config(anira_test::k_context_v3);
    ASSERT_EQ(config.m_status, ANIRA_OK) << config.m_err.message;
    uint32_t num_threads = 99;
    anira_wait_strategy wait = ANIRA_WAIT_BLOCKING;
    EXPECT_EQ(anira_context_config_threads(config.m_config, &num_threads, &wait), ANIRA_OK);
    EXPECT_EQ(num_threads, 0u) << "0 = bring your own threads, as the file says";
    EXPECT_EQ(wait, ANIRA_WAIT_SPIN_BACKOFF);
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    EXPECT_EQ(anira_context_config_log(config.m_config, &desc), ANIRA_OK);
    EXPECT_EQ(desc.level, static_cast<uint32_t>(ANIRA_LOG_WARNING));
    EXPECT_EQ(desc.drain, static_cast<uint32_t>(ANIRA_LOG_DRAIN_THREAD));
    EXPECT_EQ(desc.queue_capacity, 512u);
    EXPECT_EQ(desc.drain_interval_ms, 10u);

    // The version 2 spelling the loader shim reads: the context_config root.
    const ContextConfig upgraded(anira_test::k_rave_v2);
    ASSERT_EQ(upgraded.m_status, ANIRA_SUCCESS_UPGRADED) << upgraded.m_err.message;
    EXPECT_EQ(anira_context_config_threads(upgraded.m_config, &num_threads, &wait), ANIRA_OK);
    EXPECT_EQ(num_threads, 2u);
    EXPECT_EQ(wait, ANIRA_WAIT_BLOCKING);
    EXPECT_EQ(anira_context_config_log(upgraded.m_config, &desc), ANIRA_OK);
    EXPECT_EQ(desc.level, static_cast<uint32_t>(ANIRA_LOG_ERROR));
}
