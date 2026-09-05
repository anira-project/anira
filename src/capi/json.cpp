/*
 * The JSON loaders and writers of section 8: the three v3 files (model, context,
 * contract), the version 2 auto-upgrade of section 8.4, and to_json in v3 spelling. All
 * of it over nlohmann::json, which never reaches a header. Loaders are dumb: strings to
 * enums, numbers, construct; semantic validation is prepare's.
 */
#include <anira/InferenceConfig.h>
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>  // IWYU pragma: keep - declares the nlohmann::json type name
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "ext_registry.h"
#include "handles.h"
#include "layout.h"

using anira::capi::StatusError;
using anira::capi::translate_exception;

namespace {

// Insertion-ordered so that to_json writes a fixed key order.
using Json = nlohmann::ordered_json;

[[noreturn]] void fail_json(const std::string& path, const std::string& what) {
    throw StatusError(ANIRA_ERROR_JSON, path + ": " + what);
}

std::string child(const std::string& path, const char* key) {
    return path.empty() ? std::string(key) : path + "." + key;
}

std::string index(const std::string& path, size_t i) {
    return path + "[" + std::to_string(i) + "]";
}

Json parse_text(const char* utf8, size_t len) {
    if (utf8 == nullptr) { throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT, "JSON: NULL text"); }
    try {
        Json document = Json::parse(std::string_view(utf8, len));
        if (!document.is_object()) { fail_json("", "the document is not a JSON object"); }
        return document;
    } catch (const Json::parse_error& e) {
        fail_json("", std::string("malformed JSON: ") + e.what());
    }
}

// ---- typed accessors --------------------------------------------------------------------

std::string require_string(const Json& node, const std::string& path) {
    if (!node.is_string()) { fail_json(path, "a string is required, got " + node.dump()); }
    return node.get<std::string>();
}

uint32_t require_u32(const Json& node, const std::string& path) {
    if (!node.is_number_unsigned() || node.get<uint64_t>() > UINT32_MAX) {
        fail_json(path, "an unsigned integer is required, got " + node.dump());
    }
    return node.get<uint32_t>();
}

int64_t require_i64(const Json& node, const std::string& path) {
    if (!node.is_number_integer()) {
        fail_json(path, "an integer is required, got " + node.dump());
    }
    return node.get<int64_t>();
}

double require_number(const Json& node, const std::string& path) {
    if (!node.is_number()) { fail_json(path, "a number is required, got " + node.dump()); }
    return node.get<double>();
}

bool require_bool(const Json& node, const std::string& path) {
    if (!node.is_boolean()) { fail_json(path, "a boolean is required, got " + node.dump()); }
    return node.get<bool>();
}

void require_object(const Json& node, const std::string& path) {
    if (!node.is_object()) { fail_json(path, "an object is required, got " + node.dump()); }
}

void require_array(const Json& node, const std::string& path) {
    if (!node.is_array()) { fail_json(path, "an array is required, got " + node.dump()); }
}

// A closed vocabulary: a string outside it is ANIRA_ERROR_JSON with the value named.
template <class Enum, size_t N>
Enum vocabulary(const Json& node,
                const std::string& path,
                const std::array<std::pair<const char*, Enum>, N>& words) {
    const std::string word = require_string(node, path);
    for (const auto& [name, value] : words) {
        if (word == name) { return value; }
    }
    std::string expected;
    for (const auto& [name, value] : words) {
        static_cast<void>(value);
        expected += expected.empty() ? "" : ", ";
        expected += std::string(R"(")") + name + R"(")";
    }
    fail_json(path, R"(")" + word + R"(" is not one of )" + expected);
}

template <class Enum, size_t N>
const char* word_of(Enum value, const std::array<std::pair<const char*, Enum>, N>& words) {
    for (const auto& [name, candidate] : words) {
        if (candidate == value) { return name; }
    }
    return words[0].first;
}

// ---- vocabularies ------------------------------------------------------------------------

const std::array<std::pair<const char*, anira_engine>, 5> k_engines{{
    {"onnxruntime", ANIRA_ENGINE_ONNXRUNTIME},
    {"libtorch", ANIRA_ENGINE_LIBTORCH},
    {"tflite", ANIRA_ENGINE_TFLITE},
    {"litert", ANIRA_ENGINE_LITERT},
    {"executorch", ANIRA_ENGINE_EXECUTORCH},
}};
const std::array<std::pair<const char*, anira_engine>, 5> k_engines_v2{{
    {"ONNX", ANIRA_ENGINE_ONNXRUNTIME},
    {"LIBTORCH", ANIRA_ENGINE_LIBTORCH},
    {"TFLITE", ANIRA_ENGINE_TFLITE},
    {"LITERT", ANIRA_ENGINE_LITERT},
    {"EXECUTORCH", ANIRA_ENGINE_EXECUTORCH},
}};
constexpr const char* k_v2_custom_engine = "anira.v2.custom";
const std::array<std::pair<const char*, anira_dtype>, 10> k_dtypes{{
    {"float32", ANIRA_DTYPE_F32},
    {"float64", ANIRA_DTYPE_F64},
    {"float16", ANIRA_DTYPE_F16},
    {"bfloat16", ANIRA_DTYPE_BF16},
    {"int8", ANIRA_DTYPE_I8},
    {"uint8", ANIRA_DTYPE_U8},
    {"int16", ANIRA_DTYPE_I16},
    {"int32", ANIRA_DTYPE_I32},
    {"int64", ANIRA_DTYPE_I64},
    {"bool", ANIRA_DTYPE_BOOL8},
}};
const std::array<std::pair<const char*, anira_role>, 3> k_roles{{
    {"streamed", ANIRA_ROLE_STREAMED},
    {"buffer", ANIRA_ROLE_BUFFER},
    {"static", ANIRA_ROLE_STATIC},
}};
const std::array<std::pair<const char*, anira_axis_tag>, 7> k_axis_tags{{
    {"batch", ANIRA_AXIS_BATCH},
    {"channel", ANIRA_AXIS_CHANNEL},
    {"time", ANIRA_AXIS_TIME},
    {"height", ANIRA_AXIS_HEIGHT},
    {"width", ANIRA_AXIS_WIDTH},
    {"feature", ANIRA_AXIS_FEATURE},
    {"any", ANIRA_AXIS_ANY},
}};
const std::array<std::pair<const char*, anira_model_state>, 2> k_states{{
    {"stateless", ANIRA_MODEL_STATELESS},
    {"stateful", ANIRA_MODEL_STATEFUL},
}};
const std::array<std::pair<const char*, anira_wait_strategy>, 2> k_waits{{
    {"spin_backoff", ANIRA_WAIT_SPIN_BACKOFF},
    {"blocking", ANIRA_WAIT_BLOCKING},
}};
const std::array<std::pair<const char*, anira_log_level>, 4> k_levels{{
    {"debug", ANIRA_LOG_DEBUG},
    {"info", ANIRA_LOG_INFO},
    {"warning", ANIRA_LOG_WARNING},
    {"error", ANIRA_LOG_ERROR},
}};
const std::array<std::pair<const char*, anira_log_drain>, 2> k_drains{{
    {"thread", ANIRA_LOG_DRAIN_THREAD},
    {"manual", ANIRA_LOG_DRAIN_MANUAL},
}};
const std::array<std::pair<const char*, anira_miss_policy>, 3> k_miss{{
    {"bypass", ANIRA_MISS_BYPASS},
    {"hold_last", ANIRA_MISS_HOLD_LAST},
    {"zeros", ANIRA_MISS_ZEROS},
}};
const std::array<std::pair<const char*, anira_late_policy>, 2> k_late{{
    {"finish", ANIRA_LATE_FINISH},
    {"drop", ANIRA_LATE_DROP},
}};
const std::array<std::pair<const char*, anira_priority>, 3> k_priorities{{
    {"auto", ANIRA_PRIORITY_AUTO},
    {"interactive", ANIRA_PRIORITY_INTERACTIVE},
    {"batch", ANIRA_PRIORITY_BATCH},
}};
const std::array<std::pair<const char*, anira_delivery>, 2> k_deliveries{{
    {"polled", ANIRA_DELIVERY_POLLED},
    {"immediate", ANIRA_DELIVERY_IMMEDIATE},
}};
const std::array<std::pair<const char*, anira_edge_cost>, 2> k_edge_costs{{
    {"permissive", ANIRA_EDGE_COST_PERMISSIVE},
    {"strict", ANIRA_EDGE_COST_STRICT},
}};
const std::array<std::pair<const char*, anira_gl_threads>, 2> k_gl_threads{{
    {"caller_thread", ANIRA_GL_CALLER_THREAD},
    {"shared_context", ANIRA_GL_SHARED_CONTEXT},
}};
const std::array<std::pair<const char*, anira_exec_policy>, 2> k_exec{{
    {"worker", ANIRA_EXEC_WORKER},
    {"user_driven", ANIRA_EXEC_USER_DRIVEN},
}};

// ---- extensions in JSON -----------------------------------------------------------------

// A key the loader does not own is an extension: the object goes through the registry
// (parsed for a known kind, kept verbatim for an unknown one). A non-object value can be
// neither, and is named.
void set_ext_from_json(anira::capi::ExtBag& bag,
                       const std::string& kind,
                       const Json& value,
                       const std::string& path) {
    if (!value.is_object()) {
        fail_json(child(path, kind.c_str()), "an extension is a JSON object, got " + value.dump());
    }
    const std::string text = value.dump();
    anira_error err = ANIRA_ERROR_INIT;
    const anira_status status = bag.set_json(kind.c_str(), text, &err);
    if (ANIRA_FAILED(status)) {
        throw StatusError(static_cast<anira_status>(err.status),
                          child(path, kind.c_str()) + ": " + err.message);
    }
}

Json ext_to_json(const anira::capi::ExtSlot& slot) {
    Json object = Json::parse(slot.to_json(), nullptr, false);
    if (object.is_discarded() || !object.is_object()) { object = Json::object(); }
    if (slot.version() != 1) {
        Json with_version = Json::object();
        with_version["version"] = slot.version();
        for (auto& [key, value] : object.items()) { with_version[key] = value; }
        return with_version;
    }
    return object;
}

void write_exts(Json& object, const anira::capi::ExtBag& bag) {
    for (const anira::capi::ExtSlot& slot : bag.slots()) {
        object[slot.kind()] = ext_to_json(slot);
    }
}

std::string resolve_path(const std::string& path, const char* base_dir) {
    if (base_dir == nullptr || base_dir[0] == '\0' || path.empty()) { return path; }
    const std::filesystem::path p(path);
    // A rooted path stays as written. On Windows "/abs/x" has no drive letter and is not
    // absolute there, yet it is not relative to base_dir either.
    if (p.is_absolute() || p.has_root_directory() || p.has_root_name()) { return path; }
    // Forward slashes on every platform: they are what the JSON file carries, what to_json
    // writes back, and what the file APIs of every supported platform accept.
    return (std::filesystem::path(base_dir) / p).lexically_normal().generic_string();
}

// ---- v3 model file (8.1) ------------------------------------------------------------------

void load_spec_v3(const Json& node,
                  const std::string& path,
                  anira_tensor_spec& spec,
                  bool is_output) {
    require_object(node, path);
    const auto name = node.find("name");
    if (name == node.end()) { fail_json(child(path, "name"), "required"); }
    spec.m_name = require_string(*name, child(path, "name"));
    if (spec.m_name.empty()) { fail_json(child(path, "name"), "must not be empty"); }
    spec.m_dtype = ANIRA_DTYPE_F32;
    spec.m_role = ANIRA_ROLE_STREAMED;
    for (const auto& [key, value] : node.items()) {
        const std::string key_path = child(path, key.c_str());
        if (key == "name") {
            continue;
        } else if (key == "dtype") {
            spec.m_dtype = vocabulary(value, key_path, k_dtypes);
        } else if (key == "role") {
            spec.m_role = vocabulary(value, key_path, k_roles);
        } else if (key == "axes") {
            require_array(value, key_path);
            if (value.size() > ANIRA_MAX_RANK) {
                fail_json(key_path, "at most " + std::to_string(ANIRA_MAX_RANK) + " axes");
            }
            for (size_t i = 0; i < value.size(); ++i) {
                const std::string axis_path = index(key_path, i);
                const Json& axis = value[i];
                if (!axis.is_array() || axis.size() != 2) {
                    fail_json(axis_path, "an axis is [tag, extent], got " + axis.dump());
                }
                const anira_axis_tag tag = vocabulary(axis[0], index(axis_path, 0), k_axis_tags);
                int64_t extent = 0;
                if (axis[1].is_string()) {
                    if (axis[1].get<std::string>() != "dynamic") {
                        fail_json(index(axis_path, 1),
                                  R"(an extent is a positive integer or "dynamic")");
                    }
                    extent = ANIRA_DYNAMIC;
                } else {
                    extent = require_i64(axis[1], index(axis_path, 1));
                    if (extent <= 0) {
                        fail_json(index(axis_path, 1),
                                  R"(an extent is a positive integer or "dynamic")");
                    }
                }
                spec.m_axes[i] =
                    anira::capi::Axis{.m_tag = tag, .m_extent = extent, .m_written = true};
            }
            spec.m_ndim = static_cast<uint32_t>(value.size());
        } else if (key == "window") {
            require_object(value, key_path);
            for (const auto& [wkey, wvalue] : value.items()) {
                const std::string wpath = child(key_path, wkey.c_str());
                if (wkey == "min") {
                    spec.m_window_min = require_i64(wvalue, wpath);
                    if (spec.m_window_min < 0) { fail_json(wpath, "must not be negative"); }
                } else if (wkey == "max") {
                    if (wvalue.is_string()) {
                        if (wvalue.get<std::string>() != "unbounded") {
                            fail_json(wpath, R"(an integer or "unbounded" is required)");
                        }
                        spec.m_window_max = ANIRA_UNBOUNDED;
                    } else {
                        spec.m_window_max = require_i64(wvalue, wpath);
                        if (spec.m_window_max < 0) { fail_json(wpath, "must not be negative"); }
                    }
                } else {
                    fail_json(wpath, "unknown window key");
                }
            }
        } else if (key == "overlap") {
            spec.m_overlap = require_i64(value, key_path);
            if (spec.m_overlap < 0) { fail_json(key_path, "must not be negative"); }
        } else if (key == "latency") {
            if (!is_output) { fail_json(key_path, "latency is an output key"); }
            spec.m_latency = require_i64(value, key_path);
            if (spec.m_latency < 0) { fail_json(key_path, "must not be negative"); }
        } else if (key == "time_ratio") {
            require_array(value, key_path);
            if (value.size() != 2) { fail_json(key_path, "time_ratio is [num, den]"); }
            spec.m_ratio_num = require_i64(value[0], index(key_path, 0));
            spec.m_ratio_den = require_i64(value[1], index(key_path, 1));
            if (spec.m_ratio_num < 0 || spec.m_ratio_den < 0 ||
                (spec.m_ratio_den == 0 && spec.m_ratio_num != 0)) {
                fail_json(key_path, "time_ratio is [num, den] with den > 0, or [0, 0]");
            }
        } else {
            set_ext_from_json(spec.m_ext, key, value, path);
        }
    }
}

void set_engine_from_json(const Json& node,
                          const std::string& path,
                          anira_engine& engine,
                          std::string& engine_id) {
    const std::string word = require_string(node, path);
    for (const auto& [name, value] : k_engines) {
        if (word == name) {
            engine = value;
            engine_id.clear();
            return;
        }
    }
    if (word.find('.') == std::string::npos) {
        fail_json(path,
                  R"(")" + word +
                      R"(" is neither a built-in engine (onnxruntime, libtorch, tflite, litert, )"
                      "executorch) nor a reverse-URI custom engine name");
    }
    engine = ANIRA_ENGINE_NONE;
    engine_id = word;
}

// models[].tensors.<canonical>.layout: spec axis indices and "insert" (ANIRA_AXIS_INSERT).
std::vector<uint32_t> parse_layout(const Json& node, const std::string& path) {
    require_array(node, path);
    std::vector<uint32_t> axes;
    axes.reserve(node.size());
    for (size_t k = 0; k < node.size(); ++k) {
        const std::string item_path = index(path, k);
        if (node[k].is_string()) {
            if (node[k].get<std::string>() != "insert") {
                fail_json(item_path, R"(a spec axis index or "insert", got )" + node[k].dump());
            }
            axes.push_back(ANIRA_AXIS_INSERT);
        } else {
            axes.push_back(require_u32(node[k], item_path));
        }
    }
    std::string why;
    if (!anira::capi::valid_layout_shape(axes, &why)) { fail_json(path, why); }
    return axes;
}

Json layout_to_json(const std::vector<uint32_t>& axes) {
    Json array = Json::array();
    for (const uint32_t axis : axes) {
        if (axis == ANIRA_AXIS_INSERT) {
            array.push_back("insert");
        } else {
            array.push_back(axis);
        }
    }
    return array;
}

// models[].tensors.<canonical>: the export's name as a string, or {"name", "layout"}.
void load_tensor_record(const Json& node,
                        const std::string& path,
                        anira::capi::TensorBinding& binding) {
    if (node.is_string()) {
        binding.m_name = node.get<std::string>();
        if (binding.m_name.empty()) { fail_json(path, "must not be empty"); }
        return;
    }
    require_object(node, path);
    for (const auto& [key, value] : node.items()) {
        const std::string key_path = child(path, key.c_str());
        if (key == "name") {
            binding.m_name = require_string(value, key_path);
            if (binding.m_name.empty()) { fail_json(key_path, "must not be empty"); }
        } else if (key == "layout") {
            binding.m_layout = parse_layout(value, key_path);
        } else {
            fail_json(key_path, R"(unknown key; a tensor record has "name" and "layout")");
        }
    }
    if (binding.m_name.empty() && binding.m_layout.empty()) {
        fail_json(path, "an empty tensor record");
    }
}

Json tensor_record_to_json(const anira::capi::TensorBinding& binding) {
    if (binding.m_layout.empty()) { return binding.m_name; }
    Json object = Json::object();
    if (!binding.m_name.empty()) { object["name"] = binding.m_name; }
    object["layout"] = layout_to_json(binding.m_layout);
    return object;
}

void write_tensor_records(Json& object, const anira::capi::ModelEntry& entry) {
    if (entry.m_tensors.empty()) { return; }
    Json records = Json::object();
    for (const auto& [canonical, binding] : entry.m_tensors) {
        records[canonical] = tensor_record_to_json(binding);
    }
    object["tensors"] = records;
}

// Canonical names are unique across both sides; the anchor and the tensor records refer to a
// tensor by bare name.
void check_spec_names(const anira_model_config& cfg) {
    std::vector<std::string> seen;
    const auto check = [&seen](const std::vector<anira_tensor_spec>& specs, const char* side) {
        for (size_t i = 0; i < specs.size(); ++i) {
            for (const std::string& name : seen) {
                if (name == specs[i].m_name) {
                    fail_json(child(index(side, i), "name"),
                              "\"" + name + "\" is already the name of another tensor");
                }
            }
            seen.push_back(specs[i].m_name);
        }
    };
    check(cfg.m_inputs, "inputs");
    check(cfg.m_outputs, "outputs");
    if (!cfg.m_anchor.empty()) {
        bool found = false;
        for (const std::string& name : seen) { found = found || name == cfg.m_anchor; }
        if (!found) {
            fail_json("anchor", "\"" + cfg.m_anchor + "\" names no input or output tensor");
        }
    }
}

void load_model_v3(const Json& root, const char* base_dir, anira_model_config& cfg) {
    for (const auto& [key, value] : root.items()) {
        const std::string path = key;
        if (key == "models") {
            require_array(value, path);
            for (size_t i = 0; i < value.size(); ++i) {
                const std::string entry_path = index(path, i);
                require_object(value[i], entry_path);
                anira::capi::ModelEntry entry;
                bool has_engine = false;
                for (const auto& [ekey, evalue] : value[i].items()) {
                    const std::string ekey_path = child(entry_path, ekey.c_str());
                    if (ekey == "engine") {
                        set_engine_from_json(evalue, ekey_path, entry.m_engine, entry.m_engine_id);
                        has_engine = true;
                    } else if (ekey == "path") {
                        entry.m_path = resolve_path(require_string(evalue, ekey_path), base_dir);
                        if (entry.m_path.empty()) { fail_json(ekey_path, "must not be empty"); }
                    } else if (ekey == "tensors") {
                        require_object(evalue, ekey_path);
                        for (const auto& [canonical, record] : evalue.items()) {
                            if (canonical.empty()) { fail_json(ekey_path, "an empty tensor name"); }
                            load_tensor_record(record,
                                               child(ekey_path, canonical.c_str()),
                                               entry.m_tensors[canonical]);
                        }
                    } else if (ekey == "tensor_names") {
                        fail_json(ekey_path,
                                  R"(renamed: "tensors": { <canonical>: <export name> | )"
                                  R"({ "name": ..., "layout": [...] } })");
                    } else {
                        set_ext_from_json(entry.m_ext, ekey, evalue, entry_path);
                    }
                }
                if (!has_engine) { fail_json(child(entry_path, "engine"), "required"); }
                if (entry.m_path.empty()) { fail_json(child(entry_path, "path"), "required"); }
                cfg.m_models.push_back(std::move(entry));
            }
        } else if (key == "default_engine") {
            set_engine_from_json(value, path, cfg.m_default_engine, cfg.m_default_engine_id);
        } else if (key == "state") {
            cfg.m_state = vocabulary(value, path, k_states);
        } else if (key == "max_instances") {
            cfg.m_max_instances = require_u32(value, path);
            if (cfg.m_max_instances == 0) { fail_json(path, "must be at least 1"); }
        } else if (key == "anchor") {
            cfg.m_anchor = require_string(value, path);
            if (cfg.m_anchor.empty()) { fail_json(path, "must not be empty"); }
        } else if (key == "inputs" || key == "outputs") {
            require_array(value, path);
            std::vector<anira_tensor_spec>& specs = key == "inputs" ? cfg.m_inputs : cfg.m_outputs;
            for (size_t i = 0; i < value.size(); ++i) {
                anira_tensor_spec spec;
                load_spec_v3(value[i], index(path, i), spec, key == "outputs");
                specs.push_back(std::move(spec));
            }
        } else {
            set_ext_from_json(cfg.m_ext, key, value, "");
        }
    }
    check_spec_names(cfg);
}

// ---- v3 context file (8.2) ----------------------------------------------------------------

void load_log_block(const Json& node,
                    const std::string& path,
                    anira_context_config& context_config) {
    require_object(node, path);
    for (const auto& [key, value] : node.items()) {
        const std::string key_path = child(path, key.c_str());
        if (key == "level") {
            context_config.m_log_level = vocabulary(value, key_path, k_levels);
        } else if (key == "drain") {
            context_config.m_log_drain = vocabulary(value, key_path, k_drains);
        } else if (key == "queue_capacity") {
            const uint32_t capacity = require_u32(value, key_path);
            context_config.m_queue_capacity = capacity < 64      ? 64
                                              : capacity > 65536 ? 65536
                                                                 : capacity;
        } else if (key == "drain_interval_ms") {
            const uint32_t interval = require_u32(value, key_path);
            context_config.m_drain_interval_ms = interval == 0 ? 10 : interval;
        } else {
            fail_json(key_path, "unknown log key");
        }
    }
}

template <class Desc, class Fill>
void load_device_block(const Json& node,
                       const std::string& path,
                       std::optional<Desc>& slot,
                       const Desc& defaults,
                       Fill&& fill) {
    require_object(node, path);
    Desc desc = defaults;
    for (const auto& [key, value] : node.items()) {
        if (!fill(key, value, child(path, key.c_str()), desc)) {
            fail_json(child(path, key.c_str()), "unknown device key");
        }
    }
    slot = desc;
}

void load_context_v3(const Json& root, anira_context_config& context_config) {
    for (const auto& [key, value] : root.items()) {
        const std::string path = key;
        if (key == "num_threads") {
            context_config.m_num_threads = require_u32(value, path);
        } else if (key == "wait_strategy") {
            context_config.m_wait = vocabulary(value, path, k_waits);
        } else if (key == "log") {
            load_log_block(value, path, context_config);
        } else if (key == "log_level") {
            fail_json(path, "the version 2 spelling; use log.level in a version 3 file");
        } else if (key == "cuda") {
            static const anira_cuda_desc k_defaults = ANIRA_CUDA_DESC_INIT;
            load_device_block(
                value,
                path,
                context_config.m_cuda,
                k_defaults,
                [](const std::string& k, const Json& v, const std::string& p, anira_cuda_desc& d) {
                    if (k == "device") {
                        d.device = static_cast<int32_t>(require_i64(v, p));
                        return true;
                    }
                    if (k == "pinned_pool_limit") {
                        d.pinned_pool_limit = static_cast<uint64_t>(require_i64(v, p));
                        return true;
                    }
                    return false;
                });
        } else if (key == "vulkan") {
            static const anira_vulkan_desc k_defaults = ANIRA_VULKAN_DESC_INIT;
            int32_t device = 0;
            load_device_block(value,
                              path,
                              context_config.m_vulkan,
                              k_defaults,
                              [&device](const std::string& k,
                                        const Json& v,
                                        const std::string& p,
                                        anira_vulkan_desc& d) {
                                  if (k == "device") {
                                      device = static_cast<int32_t>(require_i64(v, p));
                                      return true;
                                  }
                                  if (k == "queue_family") {
                                      d.queue_family = require_u32(v, p);
                                      return true;
                                  }
                                  if (k == "queue_index") {
                                      d.queue_index = require_u32(v, p);
                                      return true;
                                  }
                                  return false;
                              });
            context_config.m_vulkan_device = device;
        } else if (key == "metal") {
            static const anira_metal_desc k_defaults = ANIRA_METAL_DESC_INIT;
            load_device_block(
                value,
                path,
                context_config.m_metal,
                k_defaults,
                [](const std::string&, const Json&, const std::string&, anira_metal_desc&) {
                    return false;
                });
        } else if (key == "gl") {
            static const anira_gl_desc k_defaults = ANIRA_GL_DESC_INIT;
            load_device_block(
                value,
                path,
                context_config.m_gl,
                k_defaults,
                [](const std::string& k, const Json& v, const std::string& p, anira_gl_desc& d) {
                    if (k == "threads") {
                        d.threads = static_cast<uint32_t>(vocabulary(v, p, k_gl_threads));
                        return true;
                    }
                    return false;
                });
        } else if (key == "d3d12") {
            static const anira_d3d12_desc k_defaults = ANIRA_D3D12_DESC_INIT;
            load_device_block(
                value,
                path,
                context_config.m_d3d12,
                k_defaults,
                [](const std::string&, const Json&, const std::string&, anira_d3d12_desc&) {
                    return false;
                });
        } else if (key == "webgpu") {
            static const anira_webgpu_desc k_defaults = ANIRA_WEBGPU_DESC_INIT;
            load_device_block(value,
                              path,
                              context_config.m_webgpu,
                              k_defaults,
                              [](const std::string& k,
                                 const Json& v,
                                 const std::string& p,
                                 anira_webgpu_desc& d) {
                                  if (k == "exec") {
                                      d.exec = static_cast<uint32_t>(vocabulary(v, p, k_exec));
                                      return true;
                                  }
                                  return false;
                              });
        } else {
            set_ext_from_json(context_config.m_ext, key, value, "");
        }
    }
}

// ---- v3 contract file (8.3) ---------------------------------------------------------------

void load_hard_v3(const Json& node, const std::string& path, anira::capi::HardContract& hard) {
    require_object(node, path);
    for (const auto& [key, value] : node.items()) {
        const std::string key_path = child(path, key.c_str());
        if (key == "block_min") {
            hard.m_block_min = require_u32(value, key_path);
        } else if (key == "block_max") {
            hard.m_block_max = require_u32(value, key_path);
        } else if (key == "rate") {
            hard.m_rate = require_number(value, key_path);
            if (hard.m_rate < 0.0) { fail_json(key_path, "must not be negative"); }
        } else if (key == "budget") {
            if (value.is_string()) {
                if (value.get<std::string>() != "measured") {
                    fail_json(key_path, R"(budget is "measured" or {"ms": x})");
                }
                hard.m_budget = ANIRA_BUDGET_MEASURED;
                hard.m_budget_ms = 0.0;
            } else if (value.is_object() && value.size() == 1 && value.contains("ms")) {
                hard.m_budget = ANIRA_BUDGET_EXPLICIT;
                hard.m_budget_ms = require_number(value["ms"], child(key_path, "ms"));
                if (!(hard.m_budget_ms > 0.0)) {
                    fail_json(child(key_path, "ms"), "must be positive");
                }
            } else {
                fail_json(key_path, R"(budget is "measured" or {"ms": x})");
            }
        } else if (key == "warmup") {
            if (value.is_string()) {
                const std::string word = value.get<std::string>();
                if (word == "until_stable") {
                    hard.m_warmup = ANIRA_WARMUP_UNTIL_STABLE;
                } else if (word == "none") {
                    hard.m_warmup = ANIRA_WARMUP_NONE;
                } else {
                    fail_json(key_path, R"(warmup is "until_stable", "none" or {"fixed": n})");
                }
                hard.m_warmup_iterations = 0;
            } else if (value.is_object() && value.size() == 1 && value.contains("fixed")) {
                hard.m_warmup = ANIRA_WARMUP_FIXED;
                hard.m_warmup_iterations = require_u32(value["fixed"], child(key_path, "fixed"));
            } else {
                fail_json(key_path, R"(warmup is "until_stable", "none" or {"fixed": n})");
            }
        } else if (key == "on_miss") {
            hard.m_on_miss = vocabulary(value, key_path, k_miss);
        } else if (key == "wait_ratio") {
            hard.m_wait_ratio = require_number(value, key_path);
            if (hard.m_wait_ratio < 0.0) { fail_json(key_path, "must not be negative"); }
        } else if (key == "ring_dtypes") {
            require_object(value, key_path);
            for (const auto& [tensor, word] : value.items()) {
                if (tensor.empty()) { fail_json(key_path, "a tensor name must not be empty"); }
                hard.m_ring_dtypes[tensor] =
                    vocabulary(word, child(key_path, tensor.c_str()), k_dtypes);
            }
        } else {
            fail_json(key_path, "unknown hard contract key");
        }
    }
    if (hard.m_block_min > hard.m_block_max) { fail_json(path, "block_min exceeds block_max"); }
}

void load_async_v3(const Json& node,
                   const std::string& path,
                   anira::capi::AsyncContract& async_part) {
    require_object(node, path);
    for (const auto& [key, value] : node.items()) {
        const std::string key_path = child(path, key.c_str());
        if (key == "deadline_ms") {
            async_part.m_deadline_ms = require_number(value, key_path);
        } else if (key == "on_late") {
            async_part.m_on_late = vocabulary(value, key_path, k_late);
        } else if (key == "priority") {
            async_part.m_priority = vocabulary(value, key_path, k_priorities);
        } else if (key == "lanes") {
            async_part.m_lanes = require_u32(value, key_path);
        } else if (key == "max_in_flight") {
            async_part.m_max_in_flight = require_u32(value, key_path);
        } else if (key == "delivery") {
            async_part.m_delivery = vocabulary(value, key_path, k_deliveries);
        } else {
            fail_json(key_path, "unknown async contract key");
        }
    }
}

void load_contract_v3(const Json& root, anira_contract& contract) {
    const bool has_hard = root.contains("hard");
    const bool has_async = root.contains("async");
    if (has_hard == has_async) {
        fail_json("", R"(a contract file has exactly one root, "hard" or "async")");
    }
    for (const auto& [key, value] : root.items()) {
        if (key == "hard") {
            anira::capi::HardContract hard;
            load_hard_v3(value, key, hard);
            contract.m_kind = hard;
        } else if (key == "async") {
            anira::capi::AsyncContract async_part;
            load_async_v3(value, key, async_part);
            contract.m_kind = async_part;
        } else if (key == "edge_cost") {
            contract.m_edge_cost = vocabulary(value, key, k_edge_costs);
        } else {
            set_ext_from_json(contract.m_ext, key, value, "");
        }
    }
}

// ---- version 2 documents (8.4) ------------------------------------------------------------

bool is_v2(const Json& root) {
    return root.contains("inference_config") || root.contains("context_config");
}

// A version 2 document carries its two roots and nothing of 3.x beside them: a 3.x root key
// next to inference_config or context_config is a mixed document, refused by name rather
// than read as 2.x with the 3.x keys ignored.
void refuse_mixed_roots(const Json& root, std::initializer_list<const char*> v3_keys) {
    for (const char* key : v3_keys) {
        if (root.contains(key)) {
            fail_json(key,
                      "a version 2 document (an inference_config or context_config root) cannot "
                      "also carry this 3.x root key; write one document or the other");
        }
    }
}

void warn_upgraded_once(const char* what) {
    static std::atomic_flag warned = ATOMIC_FLAG_INIT;
    if (!warned.test_and_set()) {
        ANIRA_LOG_WARNING(anira::log_group::k_config,
                          "version 2 configuration (%s) upgraded to version 3: max_inference_time, "
                          "warm_up and blocking_ratio moved to the contract; take it with "
                          "anira_model_config_take_legacy_contract",
                          what);
    }
}

// The universal shape entry of tensor_shape[] (or the first per-backend entry when every
// entry agrees): input and output shapes as lists of dims.
using ShapeList = std::vector<std::vector<int64_t>>;

ShapeList parse_v2_shape(const Json& node, const std::string& path) {
    require_array(node, path);
    if (node.empty()) { fail_json(path, "must not be empty"); }
    ShapeList shapes;
    if (node.front().is_array()) {
        for (size_t i = 0; i < node.size(); ++i) {
            const std::string item_path = index(path, i);
            require_array(node[i], item_path);
            std::vector<int64_t> dims;
            dims.reserve(node[i].size());
            for (size_t j = 0; j < node[i].size(); ++j) {
                dims.push_back(require_i64(node[i][j], index(item_path, j)));
            }
            shapes.push_back(dims);
        }
    } else {
        std::vector<int64_t> dims;
        dims.reserve(node.size());
        for (size_t j = 0; j < node.size(); ++j) {
            dims.push_back(require_i64(node[j], index(path, j)));
        }
        shapes.push_back(dims);
    }
    return shapes;
}

std::vector<int64_t> parse_v2_list(const Json& node, const std::string& path) {
    require_array(node, path);
    std::vector<int64_t> out;
    out.reserve(node.size());
    for (size_t i = 0; i < node.size(); ++i) {
        const int64_t value = require_i64(node[i], index(path, i));
        if (value < 0) { fail_json(index(path, i), "must not be negative"); }
        out.push_back(value);
    }
    return out;
}

// One v2 tensor (a shape, channels and a size) becomes one v3 spec: trailing axis time,
// the last other axis whose extent equals the channel count channel, every other axis any;
// window = the per-channel element count (v2's default processing size), context = window
// minus the v2 size; size 0 = static (decision 7 of the M1 plan).
void upgrade_spec(const std::vector<int64_t>& dims,
                  int64_t channels,
                  int64_t size,
                  bool size_given,
                  int64_t latency,
                  const std::string& name,
                  const std::string& path,
                  anira_tensor_spec& spec) {
    if (dims.empty() || dims.size() > ANIRA_MAX_RANK) {
        fail_json(path, "a shape has 1 to " + std::to_string(ANIRA_MAX_RANK) + " dims");
    }
    int64_t elements = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] <= 0) {
            fail_json(index(path, i), "a dim must be positive, got " + std::to_string(dims[i]));
        }
        elements *= dims[i];
    }
    spec.m_name = name;
    spec.m_dtype = ANIRA_DTYPE_F32;
    spec.m_ndim = static_cast<uint32_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        spec.m_axes[i] =
            anira::capi::Axis{.m_tag = ANIRA_AXIS_ANY, .m_extent = dims[i], .m_written = true};
    }
    if (size_given && size == 0) {
        spec.m_role = ANIRA_ROLE_STATIC;
        return;
    }
    spec.m_role = ANIRA_ROLE_STREAMED;
    if (channels <= 0) { fail_json(path, "a channel count must be positive"); }
    if (elements % channels != 0) {
        fail_json(path, "the shape's element count is not a multiple of its channel count");
    }
    const int64_t per_channel = elements / channels;
    const int64_t hop = size_given ? size : per_channel;
    if (hop > per_channel) {
        fail_json(path, "the processing size exceeds the tensor's per-channel element count");
    }
    // Time: the last axis whose extent is the per-channel element count (the window), which is
    // how a time-first export ({2048, 1, 1}) is told apart from a channels-first one; else the
    // trailing axis. Channel: the last other axis carrying the channel count.
    size_t time_axis = dims.size() - 1;
    for (size_t i = dims.size(); i-- > 0;) {
        if (dims[i] == per_channel) {
            time_axis = i;
            break;
        }
    }
    spec.m_axes[time_axis].m_tag = ANIRA_AXIS_TIME;
    bool channel_found = false;
    for (size_t i = dims.size(); i-- > 0;) {
        if (i != time_axis && dims[i] == channels) {
            spec.m_axes[i].m_tag = ANIRA_AXIS_CHANNEL;
            channel_found = true;
            break;
        }
    }
    if (!channel_found && channels > 1) {
        fail_json(path, "no axis carries the channel count " + std::to_string(channels));
    }
    spec.m_window_min = per_channel;
    spec.m_window_max = per_channel;
    spec.m_overlap = per_channel - hop;
    spec.m_latency = latency;
}

void upgrade_model_v2(const Json& root, const char* base_dir, anira_model_config& cfg) {
    const auto inference = root.find("inference_config");
    if (inference == root.end()) {
        fail_json("inference_config", "required in a version 2 model document");
    }
    require_object(*inference, "inference_config");
    const std::string base = "inference_config";
    auto legacy = std::make_unique<anira_contract>();
    anira::capi::HardContract hard;
    bool has_contract_key = false;

    // tensor_shape[] entries, decided after the loop: the universal entry (else the first) is
    // the canonical spec list, every other entry a per-engine layout over it (section 8.4).
    struct V2ShapeEntry {
        std::string m_path;
        bool m_universal = false;
        anira_engine m_engine = ANIRA_ENGINE_NONE;
        std::string m_engine_id;
        ShapeList m_inputs;
        ShapeList m_outputs;
    };
    std::vector<V2ShapeEntry> shape_entries;
    ShapeList inputs;
    ShapeList outputs;
    std::vector<int64_t> in_channels;
    std::vector<int64_t> out_channels;
    std::vector<int64_t> in_sizes;
    std::vector<int64_t> out_sizes;
    std::vector<int64_t> latencies;
    bool sizes_given = false;
    bool warm_up_given = false;
    bool instances_given = false;

    for (const auto& [key, value] : inference->items()) {
        const std::string path = child(base, key.c_str());
        if (key == "model_data") {
            require_array(value, path);
            for (size_t i = 0; i < value.size(); ++i) {
                const std::string entry_path = index(path, i);
                require_object(value[i], entry_path);
                anira::capi::ModelEntry entry;
                for (const auto& [ekey, evalue] : value[i].items()) {
                    const std::string ekey_path = child(entry_path, ekey.c_str());
                    if (ekey == "inference_backend") {
                        const std::string word = require_string(evalue, ekey_path);
                        if (word == "CUSTOM") {
                            entry.m_engine_id = k_v2_custom_engine;
                        } else {
                            entry.m_engine = vocabulary(evalue, ekey_path, k_engines_v2);
                        }
                    } else if (ekey == "model_path") {
                        entry.m_path = resolve_path(require_string(evalue, ekey_path), base_dir);
                    } else if (ekey == "model_function") {
                        anira_ext_entry ext = ANIRA_EXT_ENTRY_INIT;
                        const std::string function = require_string(evalue, ekey_path);
                        ext.name = function.c_str();
                        anira_error err = ANIRA_ERROR_INIT;
                        if (ANIRA_FAILED(entry.m_ext.set(&ext.header, &err))) {
                            throw StatusError(static_cast<anira_status>(err.status), err.message);
                        }
                    } else {
                        set_ext_from_json(entry.m_ext, ekey, evalue, entry_path);
                    }
                }
                if (entry.m_engine == ANIRA_ENGINE_NONE && entry.m_engine_id.empty()) {
                    fail_json(child(entry_path, "inference_backend"), "required");
                }
                if (entry.m_path.empty()) {
                    fail_json(child(entry_path, "model_path"), "required");
                }
                cfg.m_models.push_back(std::move(entry));
            }
        } else if (key == "tensor_shape") {
            require_array(value, path);
            if (value.empty()) { fail_json(path, "must not be empty"); }
            shape_entries.reserve(value.size());
            for (size_t i = 0; i < value.size(); ++i) {
                V2ShapeEntry shape_entry;
                shape_entry.m_path = index(path, i);
                require_object(value[i], shape_entry.m_path);
                const auto in = value[i].find("input_shape");
                const auto out = value[i].find("output_shape");
                if (in == value[i].end() || out == value[i].end()) {
                    fail_json(shape_entry.m_path, "input_shape and output_shape are required");
                }
                shape_entry.m_inputs =
                    parse_v2_shape(*in, child(shape_entry.m_path, "input_shape"));
                shape_entry.m_outputs =
                    parse_v2_shape(*out, child(shape_entry.m_path, "output_shape"));
                const auto backend = value[i].find("inference_backend");
                const std::string backend_path = child(shape_entry.m_path, "inference_backend");
                if (backend == value[i].end() ||
                    require_string(*backend, backend_path) == "UNIVERSAL") {
                    shape_entry.m_universal = true;
                } else if (require_string(*backend, backend_path) == "CUSTOM") {
                    shape_entry.m_engine_id = k_v2_custom_engine;
                } else {
                    shape_entry.m_engine = vocabulary(*backend, backend_path, k_engines_v2);
                }
                shape_entries.push_back(std::move(shape_entry));
            }
        } else if (key == "processing_spec") {
            require_object(value, path);
            for (const auto& [pkey, pvalue] : value.items()) {
                const std::string pkey_path = child(path, pkey.c_str());
                if (pkey == "preprocess_input_channels") {
                    in_channels = parse_v2_list(pvalue, pkey_path);
                } else if (pkey == "postprocess_output_channels") {
                    out_channels = parse_v2_list(pvalue, pkey_path);
                } else if (pkey == "preprocess_input_size") {
                    in_sizes = parse_v2_list(pvalue, pkey_path);
                    sizes_given = true;
                } else if (pkey == "postprocess_output_size") {
                    out_sizes = parse_v2_list(pvalue, pkey_path);
                    sizes_given = true;
                } else if (pkey == "internal_model_latency") {
                    latencies = parse_v2_list(pvalue, pkey_path);
                } else {
                    fail_json(pkey_path, "unknown processing_spec key");
                }
            }
        } else if (key == "max_inference_time") {
            hard.m_budget = ANIRA_BUDGET_EXPLICIT;
            hard.m_budget_ms = require_number(value, path);
            if (!(hard.m_budget_ms > 0.0)) { fail_json(path, "must be positive"); }
            has_contract_key = true;
        } else if (key == "warm_up") {
            hard.m_warmup = ANIRA_WARMUP_FIXED;
            hard.m_warmup_iterations = require_u32(value, path);
            warm_up_given = true;
            has_contract_key = true;
        } else if (key == "blocking_ratio") {
            hard.m_wait_ratio = require_number(value, path);
            if (hard.m_wait_ratio < 0.0) { fail_json(path, "must not be negative"); }
            has_contract_key = true;
        } else if (key == "num_parallel_processors") {
            cfg.m_max_instances = require_u32(value, path);
            if (cfg.m_max_instances == 0) { fail_json(path, "must be at least 1"); }
            instances_given = true;
        } else if (key == "session_exclusive_processor") {
            cfg.m_state = require_bool(value, path) ? ANIRA_MODEL_STATEFUL : ANIRA_MODEL_STATELESS;
        } else {
            set_ext_from_json(cfg.m_ext, key, value, base);
        }
    }
    if (shape_entries.empty()) { fail_json(child(base, "tensor_shape"), "required"); }
    size_t canonical = 0;
    bool universal_seen = false;
    for (size_t i = 0; i < shape_entries.size(); ++i) {
        if (!shape_entries[i].m_universal) { continue; }
        if (universal_seen) {
            fail_json(shape_entries[i].m_path, "a second universal tensor_shape entry");
        }
        universal_seen = true;
        canonical = i;
    }
    inputs = shape_entries[canonical].m_inputs;
    outputs = shape_entries[canonical].m_outputs;
    // A per-engine entry that differs from the canonical one is a layout on that engine's
    // rows: the same bytes, another axis order (section 5); anything else cannot be upgraded.
    struct PendingLayout {
        anira_engine m_engine;
        std::string m_engine_id;
        std::string m_tensor;
        std::vector<uint32_t> m_axes;
    };
    std::vector<PendingLayout> pending_layouts;
    for (size_t i = 0; i < shape_entries.size(); ++i) {
        if (i == canonical) { continue; }
        const V2ShapeEntry& shape_entry = shape_entries[i];
        if (shape_entry.m_universal) { continue; }  // unreachable: the second one failed above
        const auto derive = [&](const ShapeList& entry_shapes,
                                const ShapeList& canonical_shapes,
                                const char* side_key,
                                const char* prefix) {
            if (entry_shapes.size() != canonical_shapes.size()) {
                fail_json(child(shape_entry.m_path, side_key),
                          "lists " + std::to_string(entry_shapes.size()) + " tensors where " +
                              shape_entries[canonical].m_path + " lists " +
                              std::to_string(canonical_shapes.size()));
            }
            for (size_t t = 0; t < entry_shapes.size(); ++t) {
                if (entry_shapes[t] == canonical_shapes[t]) { continue; }
                const std::optional<std::vector<uint32_t>> axes =
                    anira::capi::stable_fill_layout(canonical_shapes[t], entry_shapes[t]);
                if (!axes.has_value()) {
                    fail_json(index(child(shape_entry.m_path, side_key), t),
                              "is not an axis layout of " + shape_entries[canonical].m_path +
                                  "'s shape: the extents other than 1 differ (version 3 permutes "
                                  "axes per engine and cannot reshape; write the version 3 "
                                  "document by hand)");
                }
                pending_layouts.push_back(PendingLayout{.m_engine = shape_entry.m_engine,
                                                        .m_engine_id = shape_entry.m_engine_id,
                                                        .m_tensor = prefix + std::to_string(t),
                                                        .m_axes = *axes});
            }
        };
        derive(shape_entry.m_inputs, inputs, "input_shape", "input_");
        derive(shape_entry.m_outputs, outputs, "output_shape", "output_");
    }
    if (!in_channels.empty() && in_channels.size() != inputs.size()) {
        fail_json(child(base, "processing_spec.preprocess_input_channels"),
                  "one entry per input tensor");
    }
    if (!out_channels.empty() && out_channels.size() != outputs.size()) {
        fail_json(child(base, "processing_spec.postprocess_output_channels"),
                  "one entry per output tensor");
    }
    if (!in_sizes.empty() && in_sizes.size() != inputs.size()) {
        fail_json(child(base, "processing_spec.preprocess_input_size"),
                  "one entry per input tensor");
    }
    if (!out_sizes.empty() && out_sizes.size() != outputs.size()) {
        fail_json(child(base, "processing_spec.postprocess_output_size"),
                  "one entry per output tensor");
    }
    if (!latencies.empty() && latencies.size() != outputs.size()) {
        fail_json(child(base, "processing_spec.internal_model_latency"),
                  "one entry per output tensor");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        anira_tensor_spec spec;
        upgrade_spec(inputs[i],
                     in_channels.empty() ? 1 : in_channels[i],
                     in_sizes.empty() ? 0 : in_sizes[i],
                     !in_sizes.empty(),
                     0,
                     "input_" + std::to_string(i),
                     index(child(base, "tensor_shape.input_shape"), i),
                     spec);
        cfg.m_inputs.push_back(std::move(spec));
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        anira_tensor_spec spec;
        upgrade_spec(outputs[i],
                     out_channels.empty() ? 1 : out_channels[i],
                     out_sizes.empty() ? 0 : out_sizes[i],
                     !out_sizes.empty(),
                     latencies.empty() ? 0 : latencies[i],
                     "output_" + std::to_string(i),
                     index(child(base, "tensor_shape.output_shape"), i),
                     spec);
        cfg.m_outputs.push_back(std::move(spec));
    }
    for (const PendingLayout& layout : pending_layouts) {
        for (anira::capi::ModelEntry& row : cfg.m_models) {
            const bool same_engine = layout.m_engine_id.empty()
                                         ? (row.m_engine == layout.m_engine && !row.is_custom())
                                         : row.m_engine_id == layout.m_engine_id;
            if (same_engine) { row.m_tensors[layout.m_tensor].m_layout = layout.m_axes; }
        }
    }
    static_cast<void>(sizes_given);
    static_cast<void>(has_contract_key);
    // The keys a 2.x file may leave out took the InferenceConfig constructor's defaults: a
    // fixed warm-up count (0) and half the hardware threads as parallel processors. The
    // upgrade writes them, so the bridge to the 2.x runtime yields the same configuration.
    if (!warm_up_given) {
        hard.m_warmup = ANIRA_WARMUP_FIXED;
        hard.m_warmup_iterations = anira::InferenceConfig::Defaults::k_warm_up;
    }
    if (!instances_given) {
        cfg.m_max_instances = anira::InferenceConfig::Defaults::m_num_parallel_processors;
    }
    legacy->m_kind = hard;
    legacy->m_legacy = true;
    cfg.m_legacy_contract = std::move(legacy);
    cfg.m_upgraded = true;
}

void upgrade_context_v2(const Json& root, anira_context_config& context_config) {
    const auto context = root.find("context_config");
    if (context == root.end()) {
        context_config.m_upgraded = true;  // a document with only inference_config: the defaults
        return;
    }
    require_object(*context, "context_config");
    for (const auto& [key, value] : context->items()) {
        const std::string path = child("context_config", key.c_str());
        if (key == "num_threads") {
            context_config.m_num_threads = require_u32(value, path);
        } else if (key == "wait_strategy") {
            context_config.m_wait = vocabulary(value, path, k_waits);
        } else if (key == "log_level") {
            context_config.m_log_level = vocabulary(value, path, k_levels);
        } else if (key == "log") {
            load_log_block(value, path, context_config);
        } else {
            set_ext_from_json(context_config.m_ext, key, value, "context_config");
        }
    }
    context_config.m_upgraded = true;
}

// ---- to_json ---------------------------------------------------------------------------------

Json spec_to_json(const anira_tensor_spec& spec, bool is_output) {
    Json object = Json::object();
    object["name"] = spec.m_name;
    object["dtype"] = word_of(spec.m_dtype, k_dtypes);
    object["role"] = word_of(spec.m_role, k_roles);
    Json axes = Json::array();
    for (uint32_t i = 0; i < spec.m_ndim; ++i) {
        Json axis = Json::array();
        axis.push_back(word_of(spec.m_axes[i].m_tag, k_axis_tags));
        if (spec.m_axes[i].m_extent == ANIRA_DYNAMIC) {
            axis.push_back("dynamic");
        } else {
            axis.push_back(spec.m_axes[i].m_extent);
        }
        axes.push_back(axis);
    }
    object["axes"] = axes;
    if (spec.m_role == ANIRA_ROLE_STREAMED) {
        Json window = Json::object();
        window["min"] = spec.m_window_min;
        if (spec.m_window_max == ANIRA_UNBOUNDED) {
            window["max"] = "unbounded";
        } else {
            window["max"] = spec.m_window_max;
        }
        object["window"] = window;
        object["overlap"] = spec.m_overlap;
    }
    if (is_output && spec.m_latency != 0) { object["latency"] = spec.m_latency; }
    if (spec.m_ratio_num != 0 || spec.m_ratio_den != 0) {
        object["time_ratio"] = Json::array({spec.m_ratio_num, spec.m_ratio_den});
    }
    write_exts(object, spec.m_ext);
    return object;
}

Json model_to_json(const anira_model_config& cfg) {
    Json root = Json::object();
    Json models = Json::array();
    for (const anira::capi::ModelEntry& entry : cfg.m_models) {
        Json object = Json::object();
        object["engine"] =
            entry.is_custom() ? entry.m_engine_id.c_str() : word_of(entry.m_engine, k_engines);
        if (!entry.m_path.empty()) { object["path"] = entry.m_path; }
        write_tensor_records(object, entry);
        write_exts(object, entry.m_ext);
        models.push_back(object);
    }
    root["models"] = models;
    if (!cfg.m_default_engine_id.empty()) {
        root["default_engine"] = cfg.m_default_engine_id;
    } else if (cfg.m_default_engine != ANIRA_ENGINE_NONE) {
        root["default_engine"] = word_of(cfg.m_default_engine, k_engines);
    }
    root["state"] = word_of(cfg.m_state, k_states);
    root["max_instances"] = cfg.m_max_instances;
    if (!cfg.m_anchor.empty()) { root["anchor"] = cfg.m_anchor; }
    Json inputs = Json::array();
    for (const anira_tensor_spec& spec : cfg.m_inputs) {
        inputs.push_back(spec_to_json(spec, false));
    }
    root["inputs"] = inputs;
    Json outputs = Json::array();
    for (const anira_tensor_spec& spec : cfg.m_outputs) {
        outputs.push_back(spec_to_json(spec, true));
    }
    root["outputs"] = outputs;
    write_exts(root, cfg.m_ext);
    return root;
}

Json context_to_json(const anira_context_config& context_config) {
    Json root = Json::object();
    if (context_config.m_num_threads != ANIRA_THREADS_AUTO) {
        root["num_threads"] = context_config.m_num_threads;
    }
    root["wait_strategy"] = word_of(context_config.m_wait, k_waits);
    Json log = Json::object();
    log["level"] = word_of(context_config.m_log_level, k_levels);
    log["drain"] = word_of(context_config.m_log_drain, k_drains);
    log["queue_capacity"] = context_config.m_queue_capacity;
    log["drain_interval_ms"] = context_config.m_drain_interval_ms;
    root["log"] = log;
    if (context_config.m_cuda) {
        Json cuda = Json::object();
        cuda["device"] = context_config.m_cuda->device;
        cuda["pinned_pool_limit"] = context_config.m_cuda->pinned_pool_limit;
        root["cuda"] = cuda;
    }
    if (context_config.m_vulkan) {
        Json vulkan = Json::object();
        vulkan["device"] = context_config.m_vulkan_device;
        vulkan["queue_family"] = context_config.m_vulkan->queue_family;
        vulkan["queue_index"] = context_config.m_vulkan->queue_index;
        root["vulkan"] = vulkan;
    }
    if (context_config.m_metal) { root["metal"] = Json::object(); }
    if (context_config.m_gl) {
        Json gl = Json::object();
        gl["threads"] =
            word_of(static_cast<anira_gl_threads>(context_config.m_gl->threads), k_gl_threads);
        root["gl"] = gl;
    }
    if (context_config.m_d3d12) { root["d3d12"] = Json::object(); }
    if (context_config.m_webgpu) {
        Json webgpu = Json::object();
        webgpu["exec"] =
            word_of(static_cast<anira_exec_policy>(context_config.m_webgpu->exec), k_exec);
        root["webgpu"] = webgpu;
    }
    write_exts(root, context_config.m_ext);
    return root;
}

anira_status write_text(const std::string& text, char* buf, size_t cap, size_t* out_len) {
    if (out_len == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    *out_len = text.size();
    if (buf == nullptr || cap < text.size() + 1) { return ANIRA_ERROR_BUFFER_TOO_SMALL; }
    std::memcpy(buf, text.data(), text.size());
    buf[text.size()] = '\0';
    return ANIRA_OK;
}

}  // namespace

// ==== entry points ==============================================================================

anira_status ANIRA_CALL anira_model_config_from_json(const char* utf8,
                                                     size_t len,
                                                     const char* base_dir,
                                                     anira_model_config** out,
                                                     anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "model config: NULL out");
    ANIRA_CAPI_REQUIRE(utf8 != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL JSON text");
    const Json root = parse_text(utf8, len);
    auto cfg = std::make_unique<anira_model_config>();
    anira_status status = ANIRA_OK;
    if (is_v2(root)) {
        refuse_mixed_roots(
            root,
            {"models", "inputs", "outputs", "default_engine", "state", "max_instances", "anchor"});
        upgrade_model_v2(root, base_dir, *cfg);
        warn_upgraded_once("model document");
        status = ANIRA_SUCCESS_UPGRADED;
    } else {
        load_model_v3(root, base_dir, *cfg);
    }
    *out = cfg.release();
    return status;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_from_json_file(const char* utf8_path,
                                                          anira_model_config** out,
                                                          anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "model config: NULL out");
    ANIRA_CAPI_REQUIRE(utf8_path != nullptr && utf8_path[0] != '\0',
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL or empty path");
    const std::ifstream file(std::filesystem::path(utf8_path), std::ios::binary);
    if (!file) {
        anira::capi::fail(err, ANIRA_ERROR_NO_SUCH_FILE, __func__, "cannot open '%s'", utf8_path);
        return ANIRA_ERROR_NO_SUCH_FILE;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    const std::string base_dir = std::filesystem::path(utf8_path).parent_path().string();
    return anira_model_config_from_json(text.data(),
                                        text.size(),
                                        base_dir.empty() ? nullptr : base_dir.c_str(),
                                        out,
                                        err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_to_json(const anira_model_config* config,
                                                   char* buf,
                                                   size_t cap,
                                                   size_t* out_len) ANIRA_NOEXCEPT try {
    if (config == nullptr || out_len == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    return write_text(model_to_json(*config).dump(2), buf, cap, out_len);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_take_legacy_contract(anira_model_config* config,
                                                                anira_contract** out) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    *out = config->m_legacy_contract.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_context_config_from_json(const char* utf8,
                                                       size_t len,
                                                       anira_context_config** out,
                                                       anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "context config: NULL out");
    ANIRA_CAPI_REQUIRE(utf8 != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "context config: NULL JSON text");
    const Json root = parse_text(utf8, len);
    auto context_config = std::make_unique<anira_context_config>();
    anira_status status = ANIRA_OK;
    if (is_v2(root)) {
        refuse_mixed_roots(root,
                           {"num_threads",
                            "wait_strategy",
                            "log",
                            "cuda",
                            "gl",
                            "vulkan",
                            "metal",
                            "d3d12",
                            "webgpu"});
        upgrade_context_v2(root, *context_config);
        warn_upgraded_once("context document");
        status = ANIRA_SUCCESS_UPGRADED;
    } else {
        load_context_v3(root, *context_config);
    }
    *out = context_config.release();
    return status;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_context_config_to_json(const anira_context_config* config,
                                                     char* buf,
                                                     size_t cap,
                                                     size_t* out_len) ANIRA_NOEXCEPT try {
    if (config == nullptr || out_len == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    return write_text(context_to_json(*config).dump(2), buf, cap, out_len);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_from_json(const char* utf8,
                                                 size_t len,
                                                 anira_contract** out,
                                                 anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "contract: NULL out");
    ANIRA_CAPI_REQUIRE(utf8 != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "contract: NULL JSON text");
    const Json root = parse_text(utf8, len);
    if (is_v2(root)) {
        refuse_mixed_roots(root, {"hard", "async", "edge_cost"});
        anira_model_config cfg;
        upgrade_model_v2(root, nullptr, cfg);
        warn_upgraded_once("contract document");
        *out = cfg.m_legacy_contract.release();
        return ANIRA_SUCCESS_UPGRADED;
    }
    auto contract = std::make_unique<anira_contract>();
    load_contract_v3(root, *contract);
    *out = contract.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }
