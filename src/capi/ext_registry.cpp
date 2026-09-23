#include "ext_registry.h"

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>  // IWYU pragma: keep - declares the nlohmann::json type name
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "handles.h"
#include "translate.h"
#include "words.h"

namespace anira::capi {

namespace {

// ---- the "entry" row (version 1) -------------------------------------------------------

void entry_fix_header(EntryPayload& payload) {
    payload.m_hdr.header.struct_size = sizeof(anira_ext_entry);
    payload.m_hdr.header.version = 1;
    payload.m_hdr.header.kind = "entry";
    payload.m_hdr.name = payload.m_name.c_str();
}

void* entry_clone(const anira_ext_header* header) {
    anira_ext_entry entry = ANIRA_EXT_ENTRY_INIT;
    const size_t readable = header->struct_size < sizeof(anira_ext_entry) ? header->struct_size
                                                                          : sizeof(anira_ext_entry);
    std::memcpy(&entry, header, readable);
    auto* payload = new EntryPayload();
    payload->m_name = entry.name != nullptr ? entry.name : "";
    entry_fix_header(*payload);
    return payload;
}

void entry_destroy(void* payload) {
    delete static_cast<EntryPayload*>(payload);
}

void* entry_from_json(std::string_view utf8, std::string& error) {
    const nlohmann::json object = nlohmann::json::parse(utf8, nullptr, false);
    if (object.is_discarded() || !object.is_object()) {
        error = "entry: not a JSON object";
        return nullptr;
    }
    const auto name = object.find("name");
    if (name == object.end() || !name->is_string()) {
        error = "entry.name: a string is required";
        return nullptr;
    }
    auto* payload = new EntryPayload();
    payload->m_name = name->get<std::string>();
    entry_fix_header(*payload);
    return payload;
}

std::string entry_to_json(const void* payload) {
    const auto* entry = static_cast<const EntryPayload*>(payload);
    return nlohmann::json{{"name", entry->m_name}}.dump();
}

// ---- the "provider_options" row (version 1) ----------------------------------------------

constexpr const char* k_provider_options = "provider_options";
// The fixed head of anira_provider_option_set: struct_size, engine, provider, num_options.
constexpr uint32_t k_option_set_head = 4 * sizeof(uint32_t);

void provider_options_fix_header(ProviderOptionsPayload& payload) {
    payload.fix_header();
}

void* provider_options_clone(const anira_ext_header* header) {
    anira_ext_provider_options record = ANIRA_EXT_PROVIDER_OPTIONS_INIT;
    const size_t readable = header->struct_size < sizeof(anira_ext_provider_options)
                                ? header->struct_size
                                : sizeof(anira_ext_provider_options);
    std::memcpy(&record, header, readable);
    auto* payload = new ProviderOptionsPayload();
    if (record.sets != nullptr) {
        for (uint32_t i = 0; i < record.num_sets; ++i) {
            anira_provider_option_set set = ANIRA_PROVIDER_OPTION_SET_INIT;
            const anira_provider_option_set& given = record.sets[i];
            if (given.struct_size < k_option_set_head) { continue; }
            const size_t bytes = given.struct_size < sizeof(anira_provider_option_set)
                                     ? given.struct_size
                                     : sizeof(anira_provider_option_set);
            std::memcpy(&set, &given, bytes);
            ProviderOptionSet copy;
            copy.m_engine = static_cast<anira_engine>(set.engine);
            copy.m_engine_id = set.engine_id != nullptr ? set.engine_id : "";
            copy.m_provider = static_cast<anira_provider>(set.provider);
            copy.m_provider_id = set.provider_id != nullptr ? set.provider_id : "";
            if (set.keys != nullptr && set.values != nullptr) {
                for (uint32_t k = 0; k < set.num_options; ++k) {
                    if (set.keys[k] == nullptr || set.values[k] == nullptr) { continue; }
                    copy.m_options.emplace_back(set.keys[k], set.values[k]);
                }
            }
            payload->m_sets.push_back(std::move(copy));
        }
    }
    provider_options_fix_header(*payload);
    return payload;
}

void provider_options_destroy(void* payload) {
    delete static_cast<ProviderOptionsPayload*>(payload);
}

// The backend word of a set: a model entry's "engine" word with its provider suffix.
bool parse_backend_word(std::string_view word, ProviderOptionSet& set, std::string& error) {
    const size_t colon = word.find(':');
    const std::string_view engine = word.substr(0, colon);
    if (engine.empty()) {
        error = "provider_options.sets[].backend: names no engine";
        return false;
    }
    if (const std::optional<anira_engine> known = engine_of_word(engine)) {
        set.m_engine = *known;
        set.m_engine_id.clear();
    } else if (engine.find('.') != std::string_view::npos) {
        set.m_engine = ANIRA_ENGINE_NONE;
        set.m_engine_id = std::string(engine);
    } else {
        error = "provider_options.sets[].backend: '" + std::string(engine) +
                "' is neither a built-in engine's word nor a custom engine's reverse-URI id";
        return false;
    }
    set.m_provider = ANIRA_PROVIDER_DEFAULT;
    set.m_provider_id.clear();
    if (colon == std::string_view::npos) { return true; }
    const std::string_view suffix = word.substr(colon + 1);
    if (suffix.empty()) {
        error = "provider_options.sets[].backend: names no provider after the ':'";
        return false;
    }
    if (const std::optional<anira_provider> known = provider_of_word(suffix)) {
        set.m_provider = *known;
    } else {
        set.m_provider_id = std::string(suffix);
    }
    return true;
}

std::string backend_word(const ProviderOptionSet& set) {
    std::string word = set.m_engine_id.empty() ? engine_word(set.m_engine) : set.m_engine_id;
    if (set.m_provider != ANIRA_PROVIDER_DEFAULT) {
        word += ":";
        word += provider_word(set.m_provider);
    } else if (!set.m_provider_id.empty()) {
        word += ":" + set.m_provider_id;
    }
    return word;
}

void* provider_options_from_json(std::string_view utf8, std::string& error) {
    const nlohmann::json object = nlohmann::json::parse(utf8, nullptr, false);
    if (object.is_discarded() || !object.is_object()) {
        error = "provider_options: not a JSON object";
        return nullptr;
    }
    const auto sets = object.find("sets");
    if (sets == object.end() || !sets->is_array()) {
        error = "provider_options.sets: an array is required";
        return nullptr;
    }
    auto payload = std::make_unique<ProviderOptionsPayload>();
    size_t index = 0;
    for (const nlohmann::json& node : *sets) {
        const std::string at = "provider_options.sets[" + std::to_string(index++) + "]";
        if (!node.is_object()) {
            error = at + ": not a JSON object";
            return nullptr;
        }
        const auto backend = node.find("backend");
        if (backend == node.end() || !backend->is_string()) {
            error = at + ".backend: a string is required";
            return nullptr;
        }
        ProviderOptionSet set;
        if (!parse_backend_word(backend->get<std::string>(), set, error)) {
            error.replace(0, std::strlen("provider_options.sets[]"), at);
            return nullptr;
        }
        const auto options = node.find("options");
        if (options != node.end()) {
            if (!options->is_object()) {
                error = at + ".options: a JSON object of strings is required";
                return nullptr;
            }
            for (const auto& [key, value] : options->items()) {
                if (!value.is_string()) {
                    error = at;
                    error += ".options.";
                    error += key;
                    error += ": a string is required";
                    return nullptr;
                }
                set.m_options.emplace_back(key, value.get<std::string>());
            }
        }
        payload->m_sets.push_back(std::move(set));
    }
    provider_options_fix_header(*payload);
    return payload.release();
}

std::string provider_options_to_json(const void* payload) {
    const auto* options = static_cast<const ProviderOptionsPayload*>(payload);
    nlohmann::json sets = nlohmann::json::array();
    for (const ProviderOptionSet& set : options->m_sets) {
        nlohmann::json pairs = nlohmann::json::object();
        for (const auto& [key, value] : set.m_options) { pairs[key] = value; }
        sets.push_back(nlohmann::json{{"backend", backend_word(set)}, {"options", pairs}});
    }
    return nlohmann::json{{"sets", sets}}.dump();
}

const char* host_name(std::string_view host) {
    return host == "tensor_spec" ? "tensor" : host == "model" ? "model" : host.data();
}

}  // namespace

bool ProviderOptionSet::names(anira_engine engine,
                              std::string_view engine_id,
                              anira_provider provider,
                              std::string_view provider_id) const noexcept {
    const bool same_engine =
        m_engine_id.empty() ? (engine_id.empty() && m_engine == engine) : m_engine_id == engine_id;
    return same_engine && m_provider == provider && m_provider_id == provider_id;
}

void ProviderOptionsPayload::fix_header() {
    m_records.clear();
    m_keys.clear();
    m_values.clear();
    m_records.reserve(m_sets.size());
    m_keys.reserve(m_sets.size());
    m_values.reserve(m_sets.size());
    for (const ProviderOptionSet& set : m_sets) {
        std::vector<const char*> keys;
        std::vector<const char*> values;
        keys.reserve(set.m_options.size());
        values.reserve(set.m_options.size());
        for (const auto& [key, value] : set.m_options) {
            keys.push_back(key.c_str());
            values.push_back(value.c_str());
        }
        m_keys.push_back(std::move(keys));
        m_values.push_back(std::move(values));
        anira_provider_option_set record = ANIRA_PROVIDER_OPTION_SET_INIT;
        record.engine = static_cast<uint32_t>(set.m_engine);
        record.provider = static_cast<uint32_t>(set.m_provider);
        record.num_options = static_cast<uint32_t>(set.m_options.size());
        record.engine_id = set.m_engine_id.empty() ? nullptr : set.m_engine_id.c_str();
        record.provider_id = set.m_provider_id.empty() ? nullptr : set.m_provider_id.c_str();
        record.keys = m_keys.back().empty() ? nullptr : m_keys.back().data();
        record.values = m_values.back().empty() ? nullptr : m_values.back().data();
        m_records.push_back(record);
    }
    m_hdr.header.struct_size = sizeof(anira_ext_provider_options);
    m_hdr.header.version = 1;
    m_hdr.header.kind = k_provider_options;
    m_hdr.sets = m_records.empty() ? nullptr : m_records.data();
    m_hdr.num_sets = static_cast<uint32_t>(m_records.size());
    m_hdr.reserved = 0;
}

const ProviderOptionSet* ProviderOptionsPayload::find(anira_engine engine,
                                                      std::string_view engine_id,
                                                      anira_provider provider,
                                                      std::string_view provider_id) const noexcept {
    for (const ProviderOptionSet& set : m_sets) {
        if (set.names(engine, engine_id, provider, provider_id)) { return &set; }
    }
    return nullptr;
}

const std::vector<ExtRow>& ext_rows() {
    static const std::vector<ExtRow> k_rows = {
        {.m_kind = "entry",
         .m_version = 1,
         .m_struct_size = sizeof(anira_ext_entry),
         .m_clone = entry_clone,
         .m_destroy = entry_destroy,
         .m_from_json = entry_from_json,
         .m_to_json = entry_to_json},
        {.m_kind = k_provider_options,
         .m_version = 1,
         .m_struct_size = sizeof(anira_ext_provider_options),
         .m_clone = provider_options_clone,
         .m_destroy = provider_options_destroy,
         .m_from_json = provider_options_from_json,
         .m_to_json = provider_options_to_json},
    };
    return k_rows;
}

const std::vector<ExtConsumer>& ext_consumers() {
    static const std::vector<ExtConsumer> k_consumers = {
#ifdef USE_ONNXRUNTIME
        {.m_name = "OnnxRuntimeAdapter",
         .m_engine = ANIRA_ENGINE_ONNXRUNTIME,
         .m_consumed = {"context:provider_options"}},
#endif
#ifdef USE_LIBTORCH
        {.m_name = "LibTorchAdapter",
         .m_engine = ANIRA_ENGINE_LIBTORCH,
         .m_consumed = {"model:entry"}},
#endif
#ifdef USE_EXECUTORCH
        {.m_name = "ExecuTorchAdapter",
         .m_engine = ANIRA_ENGINE_EXECUTORCH,
         .m_consumed = {"model:entry"}},
#endif
    };
    return k_consumers;
}

const std::vector<const char*>& ext_kinds() {
    static const std::vector<const char*> k_kinds = [] {
        std::vector<const char*> kinds;
        for (const ExtRow& row : ext_rows()) {
            bool seen = false;
            for (const char* kind : kinds) {
                if (std::strcmp(kind, row.m_kind) == 0) { seen = true; }
            }
            if (!seen) { kinds.push_back(row.m_kind); }
        }
        return kinds;
    }();
    return k_kinds;
}

namespace {

const ExtRow* find_row(std::string_view kind, uint32_t version, bool& kind_known) {
    kind_known = false;
    for (const ExtRow& row : ext_rows()) {
        if (kind == row.m_kind) {
            kind_known = true;
            if (row.m_version == version) { return &row; }
        }
    }
    return nullptr;
}

}  // namespace

// ---- ExtSlot -----------------------------------------------------------------------------

ExtSlot::ExtSlot(const ExtSlot& other)
    : m_kind(other.m_kind)
    , m_version(other.m_version)
    , m_row(other.m_row)
    , m_raw_json(other.m_raw_json)
    , m_raw_header(other.m_raw_header) {
    if (other.m_row != nullptr && other.m_payload != nullptr) {
        m_payload = other.m_row->m_clone(static_cast<const anira_ext_header*>(other.m_payload));
    }
}

ExtSlot& ExtSlot::operator=(const ExtSlot& other) {
    if (this != &other) {
        ExtSlot copy(other);
        *this = std::move(copy);
    }
    return *this;
}

ExtSlot::ExtSlot(ExtSlot&& other) noexcept
    : m_kind(std::move(other.m_kind))
    , m_version(other.m_version)
    , m_row(other.m_row)
    , m_payload(other.m_payload)
    , m_raw_json(std::move(other.m_raw_json))
    , m_raw_header(std::move(other.m_raw_header)) {
    other.m_payload = nullptr;
    other.m_row = nullptr;
}

ExtSlot& ExtSlot::operator=(ExtSlot&& other) noexcept {
    if (this != &other) {
        reset();
        m_kind = std::move(other.m_kind);
        m_version = other.m_version;
        m_row = other.m_row;
        m_payload = other.m_payload;
        m_raw_json = std::move(other.m_raw_json);
        m_raw_header = std::move(other.m_raw_header);
        other.m_payload = nullptr;
        other.m_row = nullptr;
    }
    return *this;
}

ExtSlot::~ExtSlot() {
    reset();
}

void ExtSlot::reset() noexcept {
    if (m_row != nullptr && m_payload != nullptr) { m_row->m_destroy(m_payload); }
    m_payload = nullptr;
    m_row = nullptr;
    m_raw_json.clear();
    m_raw_header.clear();
}

std::string ExtSlot::to_json() const {
    if (m_row != nullptr && m_payload != nullptr) { return m_row->m_to_json(m_payload); }
    return m_raw_json;
}

// ---- ExtBag ------------------------------------------------------------------------------

ExtSlot& ExtBag::slot_for(std::string_view kind) {
    for (ExtSlot& slot : m_slots) {
        if (slot.m_kind == kind) {
            slot.reset();
            return slot;
        }
    }
    m_slots.emplace_back();
    m_slots.back().m_kind = std::string(kind);
    return m_slots.back();
}

const ExtSlot* ExtBag::find(std::string_view kind) const noexcept {
    for (const ExtSlot& slot : m_slots) {
        if (slot.m_kind == kind) { return &slot; }
    }
    return nullptr;
}

anira_status ExtBag::set(const anira_ext_header* header, anira_error* err) {
    ANIRA_CAPI_REQUIRE(header != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "extension: NULL header");
    ANIRA_CAPI_REQUIRE(header->struct_size >= sizeof(anira_ext_header),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "extension: struct_size %u is smaller than the header (%u)",
                       static_cast<unsigned>(header->struct_size),
                       static_cast<unsigned>(sizeof(anira_ext_header)));
    ANIRA_CAPI_REQUIRE(header->kind != nullptr && header->kind[0] != '\0',
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "extension: NULL or empty kind");
    bool kind_known = false;
    const ExtRow* row = find_row(header->kind, header->version, kind_known);
    if (kind_known && row == nullptr) {
        fail(err,
             ANIRA_ERROR_EXTENSION_VERSION,
             nullptr,
             "extension '%s': version %u is not registered in this build",
             header->kind,
             static_cast<unsigned>(header->version));
        return ANIRA_ERROR_EXTENSION_VERSION;
    }
    ExtSlot& slot = slot_for(header->kind);
    slot.m_version = header->version;
    if (row != nullptr) {
        slot.m_row = row;
        slot.m_payload = row->m_clone(header);
    } else {
        // Unknown kind: keep the header bytes so the walk can name it; nothing behind the
        // pointers it may hold is interpretable here.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) the header is a byte image
        // here
        const auto* bytes = reinterpret_cast<const unsigned char*>(header);
        slot.m_raw_header.assign(bytes, bytes + header->struct_size);
    }
    return ANIRA_OK;
}

anira_status ExtBag::set_json(const char* kind, std::string_view utf8, anira_error* err) {
    ANIRA_CAPI_REQUIRE(kind != nullptr && kind[0] != '\0',
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "extension: NULL or empty kind");
    ANIRA_CAPI_REQUIRE(utf8.data() != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "extension '%s': NULL JSON text",
                       kind);
    const nlohmann::json object = nlohmann::json::parse(utf8, nullptr, false);
    if (object.is_discarded() || !object.is_object()) {
        fail(err, ANIRA_ERROR_JSON, nullptr, "extension '%s': not a JSON object", kind);
        return ANIRA_ERROR_JSON;
    }
    uint32_t version = 1;
    if (const auto it = object.find("version"); it != object.end()) {
        if (!it->is_number_unsigned()) {
            fail(err,
                 ANIRA_ERROR_JSON,
                 nullptr,
                 "extension '%s': \"version\" must be a positive integer",
                 kind);
            return ANIRA_ERROR_JSON;
        }
        version = it->get<uint32_t>();
    }
    bool kind_known = false;
    const ExtRow* row = find_row(kind, version, kind_known);
    if (kind_known && row == nullptr) {
        fail(err,
             ANIRA_ERROR_EXTENSION_VERSION,
             nullptr,
             "extension '%s': version %u is not registered in this build",
             kind,
             static_cast<unsigned>(version));
        return ANIRA_ERROR_EXTENSION_VERSION;
    }
    if (row != nullptr && row->m_from_json == nullptr) {
        fail(err,
             ANIRA_ERROR_JSON,
             nullptr,
             "extension '%s': a code-only kind has no JSON form",
             kind);
        return ANIRA_ERROR_JSON;
    }
    void* payload = nullptr;
    if (row != nullptr) {
        nlohmann::json body = object;
        body.erase("version");
        std::string error;
        payload = row->m_from_json(body.dump(), error);
        if (payload == nullptr) {
            fail(err, ANIRA_ERROR_JSON, nullptr, "extension '%s': %s", kind, error.c_str());
            return ANIRA_ERROR_JSON;
        }
    }
    ExtSlot& slot = slot_for(kind);
    slot.m_version = version;
    if (row != nullptr) {
        slot.m_row = row;
        slot.m_payload = payload;
    } else {
        slot.m_raw_json = std::string(utf8);
    }
    return ANIRA_OK;
}

// ---- the consumed-or-fail walk -----------------------------------------------------------

namespace {

// The translator's candidate rule (translate.h engine_is_candidate) for a consumer: NULL =
// every engine; an entry with an engine_id keeps the custom engine of that name; a built-in
// engine keeps its rows; {ANIRA_ENGINE_NONE, DEFAULT, NULL} keeps every custom engine. The
// provider is not read for a consumer: an engine reads its extensions on every provider.
bool candidate(anira_engine engine,
               const std::string& engine_id,
               const anira_backend_id* candidates,
               uint32_t num_candidates) {
    return engine_is_candidate(engine, engine_id, candidates, num_candidates);
}

const char* consumer_of(std::string_view host,
                        std::string_view kind,
                        anira_engine entry_engine,
                        const anira_backend_id* candidates,
                        uint32_t num_candidates) {
    const std::string wanted = std::string(host) + ":" + std::string(kind);
    for (const ExtConsumer& consumer : ext_consumers()) {
        if (consumer.m_engine != ANIRA_ENGINE_NONE) {
            if (!candidate(consumer.m_engine, "", candidates, num_candidates)) { continue; }
            // An adapter reads the entries of its own engine only.
            if (host == "model" && entry_engine != consumer.m_engine) { continue; }
        }
        for (const std::string& consumed : consumer.m_consumed) {
            if (consumed == wanted) { return consumer.m_name; }
        }
    }
    return nullptr;
}

// The consumers of a pipeline that read "<host>:<kind>", in the vector's order. Its stage
// (neither engine key) reads a slot wherever it sits: no candidate filter. A registered
// engine (m_engine_id) reads only while it is a candidate, and a "model" host only when the
// entry is its own (entry_engine_id): the rule of the build's adapters, keyed by the id.
std::vector<const char*> pipeline_consumers_of(std::string_view host,
                                               std::string_view kind,
                                               const std::string& entry_engine_id,
                                               const anira_backend_id* candidates,
                                               uint32_t num_candidates,
                                               const std::vector<ExtConsumer>* pipeline) {
    std::vector<const char*> names;
    if (pipeline == nullptr) { return names; }
    const std::string wanted = std::string(host) + ":" + std::string(kind);
    for (const ExtConsumer& consumer : *pipeline) {
        if (!consumer.m_engine_id.empty()) {
            if (!candidate(ANIRA_ENGINE_NONE, consumer.m_engine_id, candidates, num_candidates)) {
                continue;
            }
            if (host == "model" && entry_engine_id != consumer.m_engine_id) { continue; }
        }
        if (std::ranges::find(consumer.m_consumed, wanted) != consumer.m_consumed.end()) {
            names.push_back(consumer.m_name);
        }
    }
    return names;
}

// One host's bag: every slot known and consumed, else the failure names it. With rows, each
// consumed slot is recorded as one plan row per consumer: anira's own adapter, then the
// pipeline's consumers.
anira_status check_bag(const ExtBag& bag,
                       std::string_view host,
                       const std::string& where,
                       anira_engine entry_engine,
                       const std::string& entry_engine_id,
                       const anira_backend_id* candidates,
                       uint32_t num_candidates,
                       anira_error* err,
                       std::vector<ExtPlanRow>* rows,
                       const std::vector<ExtConsumer>* pipeline) {
    for (const ExtSlot& slot : bag.slots()) {
        if (!slot.known()) {
            fail(err,
                 ANIRA_ERROR_EXTENSION_UNKNOWN,
                 nullptr,
                 "extension '%s' on %s %s is not known to this build",
                 slot.kind().c_str(),
                 host_name(host),
                 where.c_str());
            return ANIRA_ERROR_EXTENSION_UNKNOWN;
        }
        const char* consumer =
            consumer_of(host, slot.kind(), entry_engine, candidates, num_candidates);
        const std::vector<const char*> stage_consumers = pipeline_consumers_of(host,
                                                                               slot.kind(),
                                                                               entry_engine_id,
                                                                               candidates,
                                                                               num_candidates,
                                                                               pipeline);
        if (consumer == nullptr && stage_consumers.empty()) {
            fail(err,
                 ANIRA_ERROR_EXTENSION_UNCONSUMED,
                 nullptr,
                 "extension '%s' on %s %s is not consumed by any stage in this build",
                 slot.kind().c_str(),
                 host_name(host),
                 where.c_str());
            return ANIRA_ERROR_EXTENSION_UNCONSUMED;
        }
        if (rows != nullptr) {
            std::string at = host_name(host);
            if (!where.empty()) { at += " " + where; }
            if (consumer != nullptr) {
                rows->push_back(
                    ExtPlanRow{.m_host = at, .m_kind = slot.kind(), .m_consumer = consumer});
            }
            for (const char* stage : stage_consumers) {
                rows->push_back(
                    ExtPlanRow{.m_host = at, .m_kind = slot.kind(), .m_consumer = stage});
            }
        }
    }
    return ANIRA_OK;
}

// The walk both public entries share: the specs, the candidate entries, the model config,
// then the context config and the contract when given.
anira_status walk_bags(const anira_model_config& model,
                       const anira_context_config* config,
                       const anira_contract* contract,
                       const anira_backend_id* candidates,
                       uint32_t num_candidates,
                       anira_error* err,
                       std::vector<ExtPlanRow>* rows,
                       const std::vector<ExtConsumer>* pipeline) {
    // A host that is no model entry has no engine: nothing is filtered by one there.
    const std::string no_engine_id;
    for (const anira_tensor_spec& spec : model.m_inputs) {
        const anira_status status = check_bag(spec.m_ext,
                                              "tensor_spec",
                                              "'" + spec.m_name + "'",
                                              ANIRA_ENGINE_NONE,
                                              no_engine_id,
                                              candidates,
                                              num_candidates,
                                              err,
                                              rows,
                                              pipeline);
        if (ANIRA_FAILED(status)) { return status; }
    }
    for (const anira_tensor_spec& spec : model.m_outputs) {
        const anira_status status = check_bag(spec.m_ext,
                                              "tensor_spec",
                                              "'" + spec.m_name + "'",
                                              ANIRA_ENGINE_NONE,
                                              no_engine_id,
                                              candidates,
                                              num_candidates,
                                              err,
                                              rows,
                                              pipeline);
        if (ANIRA_FAILED(status)) { return status; }
    }
    for (size_t i = 0; i < model.m_models.size(); ++i) {
        const ModelEntry& entry = model.m_models[i];
        if (!row_is_candidate(entry, candidates, num_candidates)) {
            continue;  // filtered out: no candidate runs the entry (its engine, or its pin)
        }
        const anira_status status = check_bag(entry.m_ext,
                                              "model",
                                              std::to_string(i),
                                              entry.m_engine,
                                              entry.m_engine_id,
                                              candidates,
                                              num_candidates,
                                              err,
                                              rows,
                                              pipeline);
        if (ANIRA_FAILED(status)) { return status; }
    }
    const anira_status status = check_bag(model.m_ext,
                                          "model_config",
                                          "",
                                          ANIRA_ENGINE_NONE,
                                          no_engine_id,
                                          candidates,
                                          num_candidates,
                                          err,
                                          rows,
                                          pipeline);
    if (ANIRA_FAILED(status)) { return status; }
    if (config != nullptr) {
        const anira_status context_status = check_bag(config->m_ext,
                                                      "context",
                                                      "",
                                                      ANIRA_ENGINE_NONE,
                                                      no_engine_id,
                                                      candidates,
                                                      num_candidates,
                                                      err,
                                                      rows,
                                                      pipeline);
        if (ANIRA_FAILED(context_status)) { return context_status; }
    }
    if (contract != nullptr) {
        const anira_status contract_status = check_bag(contract->m_ext,
                                                       "contract",
                                                       "",
                                                       ANIRA_ENGINE_NONE,
                                                       no_engine_id,
                                                       candidates,
                                                       num_candidates,
                                                       err,
                                                       rows,
                                                       pipeline);
        if (ANIRA_FAILED(contract_status)) { return contract_status; }
    }
    return ANIRA_OK;
}

}  // namespace

anira_status ext_check_consumed(const anira_model_config& model,
                                const anira_context_config* config,
                                const anira_contract* contract,
                                const anira_backend_id* candidates,
                                uint32_t num_candidates,
                                anira_error* err,
                                const std::vector<ExtConsumer>* pipeline) {
    return walk_bags(model, config, contract, candidates, num_candidates, err, nullptr, pipeline);
}

std::vector<ExtPlanRow> ext_consumed_rows(const anira_model_config& model,
                                          const anira_contract* contract,
                                          const anira_backend_id* candidates,
                                          uint32_t num_candidates,
                                          const std::vector<ExtConsumer>* pipeline) {
    std::vector<ExtPlanRow> rows;
    anira_error ignored = ANIRA_ERROR_INIT;
    static_cast<void>(
        walk_bags(model, nullptr, contract, candidates, num_candidates, &ignored, &rows, pipeline));
    return rows;
}

// ---- BytesCarrier (handles.h) ------------------------------------------------------------

BytesCarrier::BytesCarrier(const void* bytes,
                           size_t size,
                           anira_bytes_ownership ownership,
                           anira_bytes_release_fn release,
                           void* ctx)
    : m_size(size), m_ownership(ownership), m_release(release), m_ctx(ctx) {
    if (ownership == ANIRA_BYTES_COPY) {
        const auto* src = static_cast<const unsigned char*>(bytes);
        m_copy.assign(src, src + size);
        m_bytes = m_copy.data();
        m_release = nullptr;  // a copy owes the caller nothing
        m_ctx = nullptr;
    } else {
        m_bytes = bytes;
    }
}

BytesCarrier::~BytesCarrier() {
    if (m_release != nullptr) { m_release(m_bytes, m_ctx); }
}

}  // namespace anira::capi
