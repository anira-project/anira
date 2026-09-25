#ifndef ANIRA_CAPI_EXT_REGISTRY_H
#define ANIRA_CAPI_EXT_REGISTRY_H

/*
 * The extension registry of section 1b: one row per (kind, version), the slots a config
 * handle carries (ExtBag), and the consumed-or-fail walk. Private to src/capi and the
 * tests. JSON crosses this header as text only; nlohmann stays inside the .cpp.
 */

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct anira_model_config;
struct anira_context_config;
struct anira_contract;

namespace anira::capi {

/// One registered (kind, version): how to deep-copy, destroy and (de)serialize a payload.
/// from_json == nullptr marks a code-only kind.
struct ExtRow {
    const char* m_kind;
    uint32_t m_version;
    uint32_t m_struct_size;
    /// Deep copy from a header the caller owns (only min(header->struct_size, m_struct_size)
    /// bytes are read); the returned payload's first member is a valid anira_ext_header whose
    /// kind and every pointer are owned by the payload.
    void* (*m_clone)(const anira_ext_header* header);
    void (*m_destroy)(void* payload);
    /// Parses the extension object (without its "version" member); a failure text in error.
    void* (*m_from_json)(std::string_view utf8, std::string& error);
    /// The extension object as JSON text, without the "version" member.
    std::string (*m_to_json)(const void* payload);
    /// Refuses a malformed C record before it is cloned (ANIRA_ERROR_INVALID_ARGUMENT naming
    /// the fault); NULL for a kind whose clone reads any record.
    anira_status (*m_check)(const anira_ext_header* header, anira_error* err) = nullptr;
};

/// A stage or an engine and what it reads, as "<host>:<kind>" entries (hosts: tensor_spec,
/// model, model_config, context, contract, job). The build's table (ext_consumers) is anira's
/// own adapters, one per built-in engine, keyed by m_engine; a pipeline's vector holds its
/// stage (neither key: it reads a slot wherever it sits, whatever the candidates) and one
/// consumer per registered engine that declares kinds, keyed by m_engine_id (the id it was
/// registered under, which the plan report's consumer column reads). An engine consumer of
/// either kind reads its "model:" kinds from the entries of its own engine alone and its other
/// kinds from any host, and only while it is a candidate.
struct ExtConsumer {
    const char* m_name;
    anira_engine m_engine;    ///< a built-in engine's adapter; CUSTOM for a registered engine,
                              ///< NONE for a stage
    std::string m_engine_id;  ///< a registered engine's id; empty otherwise
    std::vector<std::string> m_consumed;
};

/// The rows of this build, in registration order (function-local static).
ANIRA_API const std::vector<ExtRow>& ext_rows();
/// The consumers of this build.
ANIRA_API const std::vector<ExtConsumer>& ext_consumers();
/// The distinct registered kinds, in first-registration order.
ANIRA_API const std::vector<const char*>& ext_kinds();

/// One slot of a bag: a known kind holds a payload owned through its row; an unknown kind
/// holds what was handed in (the header bytes, or the JSON text) and fails prepare by name.
class ANIRA_API ExtSlot {
public:
    ExtSlot() = default;
    ExtSlot(const ExtSlot& other);
    ExtSlot& operator=(const ExtSlot& other);
    ExtSlot(ExtSlot&& other) noexcept;
    ExtSlot& operator=(ExtSlot&& other) noexcept;
    ~ExtSlot();

    const std::string& kind() const noexcept { return m_kind; }
    uint32_t version() const noexcept { return m_version; }
    bool known() const noexcept { return m_row != nullptr; }
    const ExtRow* row() const noexcept { return m_row; }
    const void* payload() const noexcept { return m_payload; }
    template <class T>
    const T* payload_as() const noexcept {
        return static_cast<const T*>(m_payload);
    }
    const std::string& raw_json() const noexcept { return m_raw_json; }
    const std::vector<unsigned char>& raw_header() const noexcept { return m_raw_header; }
    /// The JSON text of a known slot through its row, the stored text of an unknown one.
    std::string to_json() const;

private:
    friend class ExtBag;
    void reset() noexcept;

    std::string m_kind;
    uint32_t m_version = 1;
    const ExtRow* m_row = nullptr;
    void* m_payload = nullptr;
    std::string m_raw_json;
    std::vector<unsigned char> m_raw_header;
};

/// The extension slots of one host: one per kind, a second set of a kind replaces the first.
class ANIRA_API ExtBag {
public:
    /// NULL or short header -> INVALID_ARGUMENT; known kind at an unregistered version ->
    /// EXTENSION_VERSION; unknown kind -> stored (the header bytes) for the walk to name.
    anira_status set(const anira_ext_header* header, anira_error* err);
    /// The JSON twin: {"version": N, ...} (version defaults to 1); a known kind is parsed
    /// through its row, an unknown kind keeps the text.
    anira_status set_json(const char* kind, std::string_view utf8, anira_error* err);

    const ExtSlot* find(std::string_view kind) const noexcept;
    template <class T>
    const T* payload(std::string_view kind) const noexcept {
        const ExtSlot* slot = find(kind);
        return slot != nullptr && slot->known() ? slot->payload_as<T>() : nullptr;
    }
    const std::vector<ExtSlot>& slots() const noexcept { return m_slots; }
    bool empty() const noexcept { return m_slots.empty(); }

private:
    ExtSlot& slot_for(std::string_view kind);
    std::vector<ExtSlot> m_slots;
};

/// The payload of the "entry" kind (version 1): the header's name points into m_name.
struct EntryPayload {
    anira_ext_entry m_hdr;
    std::string m_name;
};

/// One set of the "provider_options" kind: a backend as the two-axis id and the options its
/// runtime takes for the provider, as string pairs in the runtime's own vocabulary.
struct ProviderOptionSet {
    anira_engine m_engine = ANIRA_ENGINE_NONE;
    std::string m_engine_id;
    anira_provider m_provider = ANIRA_PROVIDER_DEFAULT;
    std::string m_provider_id;
    std::vector<std::pair<std::string, std::string>> m_options;

    /// Whether the set is the backend's: the engine (a built-in one by value, a custom one by
    /// id) on the provider (a provider of the enum by value, a custom one by name).
    bool names(anira_engine engine,
               std::string_view engine_id,
               anira_provider provider,
               std::string_view provider_id) const noexcept;
};

/// The payload of the "provider_options" kind (version 1): the header's sets point into
/// m_records, whose strings and arrays point into m_sets, m_keys and m_values (rebuilt by
/// fix_header after every change of m_sets).
struct ANIRA_API ProviderOptionsPayload {
    anira_ext_provider_options m_hdr;
    std::vector<ProviderOptionSet> m_sets;
    std::vector<anira_provider_option_set> m_records;
    std::vector<std::vector<const char*>> m_keys;
    std::vector<std::vector<const char*>> m_values;

    /// Rebuilds the C view over m_sets.
    void fix_header();
    /// The set of a backend, or NULL when the payload has none for it.
    const ProviderOptionSet* find(anira_engine engine,
                                  std::string_view engine_id,
                                  anira_provider provider,
                                  std::string_view provider_id) const noexcept;
};

/// The consumed-or-fail walk of section 1b over a model config (its specs, its entries, the
/// config itself) and, when given, a context config and a contract: every slot must be a
/// known kind that a consumer in the candidate set reads from that host. candidates == NULL
/// means every consumer of this build; a built-in engine names its adapter,
/// ANIRA_ENGINE_CUSTOM with an engine_id a custom engine's rows (the provider is not read); an
/// engine adapter consumes only the entries of its own engine. `pipeline` are the consumers a
/// pipeline declares (NULL for none): its stage (anira_stage_desc::consumed_kinds), which reads a
/// slot wherever it sits, whatever the candidates, and its registered engines
/// (anira_engine_desc::consumed_kinds, one consumer per engine keyed by its id), each of which
/// reads its "model:" kinds from its own entries alone and its other kinds from any host, while it
/// is a candidate. On failure err carries the offending name.
ANIRA_API anira_status ext_check_consumed(const anira_model_config& model,
                                          const anira_context_config* config,
                                          const anira_contract* contract,
                                          const anira_backend_id* candidates,
                                          uint32_t num_candidates,
                                          anira_error* err,
                                          const std::vector<ExtConsumer>* pipeline = nullptr);

/// One row of anira_plan_ext: a consumed slot and its consumer.
struct ExtPlanRow {
    std::string m_host;      ///< the host and where on it ("tensor 'in'", "model 0", "contract")
    std::string m_kind;      ///< the extension kind
    std::string m_consumer;  ///< the consumer's registered name
};

/// The extensions one plan consumes, in walk order (specs, the candidate entries, the model
/// config, the contract): one row per slot and consumer, anira's own adapter first, then every
/// consumer of `pipeline` that reads the slot's kind (the stage; a registered engine among the
/// candidates, on its own entries for a "model:" kind), in the vector's order. Never fails:
/// ext_check_consumed ran first.
ANIRA_API std::vector<ExtPlanRow> ext_consumed_rows(
    const anira_model_config& model,
    const anira_contract* contract,
    const anira_backend_id* candidates,
    uint32_t num_candidates,
    const std::vector<ExtConsumer>* pipeline = nullptr);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_EXT_REGISTRY_H
