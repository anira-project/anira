#include "ext_registry.h"

#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>  // IWYU pragma: keep - declares the nlohmann::json type name
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "handles.h"

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

const char* host_name(std::string_view host) {
    return host == "tensor_spec" ? "tensor" : host == "model" ? "model" : host.data();
}

}  // namespace

const std::vector<ExtRow>& ext_rows() {
    static const std::vector<ExtRow> k_rows = {
        {.m_kind = "entry",
         .m_version = 1,
         .m_struct_size = sizeof(anira_ext_entry),
         .m_clone = entry_clone,
         .m_destroy = entry_destroy,
         .m_from_json = entry_from_json,
         .m_to_json = entry_to_json},
    };
    return k_rows;
}

const std::vector<ExtConsumer>& ext_consumers() {
    static const std::vector<ExtConsumer> k_consumers = {
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

bool candidate(anira_engine engine, const anira_engine* candidates, uint32_t num_candidates) {
    if (candidates == nullptr) { return true; }
    for (uint32_t i = 0; i < num_candidates; ++i) {
        if (candidates[i] == engine) { return true; }
    }
    return false;
}

const char* consumer_of(std::string_view host,
                        std::string_view kind,
                        anira_engine entry_engine,
                        const anira_engine* candidates,
                        uint32_t num_candidates) {
    const std::string wanted = std::string(host) + ":" + std::string(kind);
    for (const ExtConsumer& consumer : ext_consumers()) {
        if (consumer.m_engine != ANIRA_ENGINE_NONE) {
            if (!candidate(consumer.m_engine, candidates, num_candidates)) { continue; }
            // An adapter reads the entries of its own engine only.
            if (host == "model" && entry_engine != consumer.m_engine) { continue; }
        }
        for (const std::string& consumed : consumer.m_consumed) {
            if (consumed == wanted) { return consumer.m_name; }
        }
    }
    return nullptr;
}

anira_status check_bag(const ExtBag& bag,
                       std::string_view host,
                       const std::string& where,
                       anira_engine entry_engine,
                       const anira_engine* candidates,
                       uint32_t num_candidates,
                       anira_error* err) {
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
        if (consumer_of(host, slot.kind(), entry_engine, candidates, num_candidates) == nullptr) {
            fail(err,
                 ANIRA_ERROR_EXTENSION_UNCONSUMED,
                 nullptr,
                 "extension '%s' on %s %s is not consumed by any stage in this build",
                 slot.kind().c_str(),
                 host_name(host),
                 where.c_str());
            return ANIRA_ERROR_EXTENSION_UNCONSUMED;
        }
    }
    return ANIRA_OK;
}

}  // namespace

anira_status ext_check_consumed(const anira_model_config& model,
                                const anira_context_config* context,
                                const anira_contract* contract,
                                const anira_engine* candidates,
                                uint32_t num_candidates,
                                anira_error* err) {
    for (const anira_tensor_spec& spec : model.m_inputs) {
        const anira_status status = check_bag(spec.m_ext,
                                              "tensor_spec",
                                              "'" + spec.m_name + "'",
                                              ANIRA_ENGINE_NONE,
                                              candidates,
                                              num_candidates,
                                              err);
        if (ANIRA_FAILED(status)) { return status; }
    }
    for (const anira_tensor_spec& spec : model.m_outputs) {
        const anira_status status = check_bag(spec.m_ext,
                                              "tensor_spec",
                                              "'" + spec.m_name + "'",
                                              ANIRA_ENGINE_NONE,
                                              candidates,
                                              num_candidates,
                                              err);
        if (ANIRA_FAILED(status)) { return status; }
    }
    for (size_t i = 0; i < model.m_models.size(); ++i) {
        const ModelEntry& entry = model.m_models[i];
        if (!candidate(entry.m_engine, candidates, num_candidates)) { continue; }  // filtered out
        const anira_status status = check_bag(entry.m_ext,
                                              "model",
                                              std::to_string(i),
                                              entry.m_engine,
                                              candidates,
                                              num_candidates,
                                              err);
        if (ANIRA_FAILED(status)) { return status; }
    }
    const anira_status status = check_bag(model.m_ext,
                                          "model_config",
                                          "",
                                          ANIRA_ENGINE_NONE,
                                          candidates,
                                          num_candidates,
                                          err);
    if (ANIRA_FAILED(status)) { return status; }
    if (context != nullptr) {
        const anira_status context_status = check_bag(context->m_ext,
                                                      "context",
                                                      "",
                                                      ANIRA_ENGINE_NONE,
                                                      candidates,
                                                      num_candidates,
                                                      err);
        if (ANIRA_FAILED(context_status)) { return context_status; }
    }
    if (contract != nullptr) {
        const anira_status contract_status = check_bag(contract->m_ext,
                                                       "contract",
                                                       "",
                                                       ANIRA_ENGINE_NONE,
                                                       candidates,
                                                       num_candidates,
                                                       err);
        if (ANIRA_FAILED(contract_status)) { return contract_status; }
    }
    return ANIRA_OK;
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
