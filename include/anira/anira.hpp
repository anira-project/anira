/**
 * @file anira.hpp
 * @brief The C++20 face of the anira 3 configuration ABI: RAII handles and builders over the
 * C entries of anira/abi/config.h, one C call per method, every failure an anira::Error.
 *
 * Header-only and not ABI-stable: it is compiled into the user's binary, so the promise it
 * carries is the C ABI's. Nothing here is exported from libanira, nothing here touches the
 * 2.x C++ classes (it includes no anira/system, anira/utils or third-party header), and it
 * can be included beside <anira/anira.h>.
 *
 * Scope at this pre-release: the configuration half only (tensor specs, model, machine and
 * contract configuration, job options). The Machine, the InferenceHandler and the _wait twins
 * arrive with the 3.x runtime.
 *
 * Deviations from the architecture document, section 6 (stated here and on the docs page):
 * anira::JsonConfigLoader is not declared (the 2.x class of that name is still in every
 * example; use ModelConfig::from_file, MachineConfig::from_file, ContractHandle::from_file);
 * ModelConfig::take_legacy_contract returns std::optional<ContractHandle>, since a handle
 * cannot be read back into a Hard aggregate; MachineConfig::log_sink takes the raw
 * (anira_log_fn, void*) pair; ModelConfig::anchor takes the tensor's canonical name;
 * ContractHandle, JobOptionsHandle and the upgraded() queries are additions; set_model_bytes
 * chains (the document's returns void); add_model_bytes and set_model_bytes carry the
 * (release, ctx) pair of the C entry and have custom-engine twins; JobOptions has no
 * on_complete yet (it arrives with Ticket); a contract file loads into a ContractHandle,
 * patched with hard_geometry and the other setters, not into an anira::Contract aggregate.
 *
 * Requirements: C++20 with exceptions; std::filesystem, which on Apple platforms means a
 * deployment target of macOS 10.15 / iOS 13 or later.
 *
 * Every entry is [main-thread] and may allocate; the ABI is checked once per process on the
 * first handle created (anira_check_abi against ANIRA_ABI_VERSION).
 */
#ifndef ANIRA_HPP
#define ANIRA_HPP

#if !defined(__cplusplus) || \
    (__cplusplus < 202002L && !(defined(_MSVC_LANG) && _MSVC_LANG >= 202002L))
#error "anira.hpp requires C++20; the C headers under anira/abi/ are C11"
#endif

#if !defined(__cpp_exceptions) && !defined(_CPPUNWIND)
#error \
    "anira.hpp throws anira::Error; build with exceptions until ANIRA_CXX_NO_EXCEPTIONS lands (v3.0.0-alpha.2)"
#endif

#if defined(ANIRA_CXX_NO_EXCEPTIONS) || defined(ANIRA_CXX_MANUAL_INIT) || \
    defined(ANIRA_NO_PROTOTYPES)
#error \
    "ANIRA_CXX_NO_EXCEPTIONS, ANIRA_CXX_MANUAL_INIT and ANIRA_NO_PROTOTYPES land with the handler half (v3.0.0-alpha.2)"
#endif

#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>  // IWYU pragma: keep - the umbrella of the C headers
#include <anira/abi/log.h>
#include <anira/abi/machine.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <ratio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace anira {

// ---- aliases -------------------------------------------------------------------------------

using DType = anira_dtype;
using Engine = anira_engine;
using Provider = anira_provider;
using Domain = anira_domain;
using SyncKind = anira_sync_kind;
using Role = anira_role;
using AxisTag = anira_axis_tag;
using BackendId = anira_backend_id;

namespace detail {

/// ANIRA_FAILED / ANIRA_SUCCEEDED without the C macros (no C-style cast in the wrapper).
constexpr bool failed(anira_status status) noexcept {
    return static_cast<int32_t>(status) < 0;
}
constexpr bool succeeded(anira_status status) noexcept {
    return !failed(status);
}
/// ANIRA_ABI_VERSION, spelled with static_casts.
inline constexpr uint32_t k_abi_version =
    (static_cast<uint32_t>(ANIRA_ABI_MAJOR) << 16U) | static_cast<uint32_t>(ANIRA_ABI_MINOR);
/// A string_view as C text: an empty view has a null data(), the C entries want a pointer.
inline const char* text_of(std::string_view text) noexcept {
    return text.data() != nullptr ? text.data() : "";
}

}  // namespace detail

// ---- errors --------------------------------------------------------------------------------

// NOLINTBEGIN(readability-identifier-naming) public fields spell the document's names

/**
 * @brief A failed C entry: the status and the message anira wrote into its anira_error, or
 * the entry's name when the entry carries no error record.
 */
struct Error : std::runtime_error {
    anira_status status;

    explicit Error(const anira_error& err)
        : std::runtime_error(err.message[0] != '\0' ? std::string(err.message)
                                                    : std::string(anira_status_string(
                                                          static_cast<anira_status>(err.status))))
        , status(static_cast<anira_status>(err.status)) {}

    Error(anira_status status_value, std::string_view entry_name)
        : std::runtime_error(std::string(entry_name) + ": " + anira_status_string(status_value))
        , status(status_value) {}

    /// The status a C entry returned and the message it wrote (or the status's text).
    Error(anira_status status_value, const anira_error& err)
        : std::runtime_error(err.message[0] != '\0'
                                 ? std::string(err.message)
                                 : std::string(anira_status_string(status_value)))
        , status(status_value) {}
};

/**
 * @brief The value-or-error return the exception-free mode will use
 * (ANIRA_CXX_NO_EXCEPTIONS, v3.0.0-alpha.2); declared now so that signatures can name it.
 */
template <class T>
struct Result {
    T value{};
    anira_error error{};  // status ANIRA_OK, empty message: what ANIRA_ERROR_INIT spells

    bool ok() const noexcept { return detail::succeeded(static_cast<anira_status>(error.status)); }
};

// ---- extensions (section 1b) ---------------------------------------------------------------

namespace ext {

/// The one extension kind of 3.0: the entry point of a LibTorch or ExecuTorch file.
struct Entry {
    std::string name;
};

}  // namespace ext

// NOLINTEND(readability-identifier-naming)

namespace detail {

/**
 * @brief Maps an extension value type onto its C record. A specialisation provides
 * `using Native = anira_ext_<kind>;` and `static Native mint(const Ext&)`, where the record's
 * first member is its anira_ext_header. The minted record must outlive the C call it feeds;
 * set_ext copies, so a temporary suffices there.
 */
template <class Ext>
struct ExtTraits;

template <>
struct ExtTraits<ext::Entry> {
    using Native = anira_ext_entry;

    static Native mint(const ext::Entry& entry) {
        Native native = ANIRA_EXT_ENTRY_INIT;
        native.name = entry.name.c_str();
        return native;
    }
};

/// Runs anira_check_abi once per process, on the first handle created.
inline void abi_check_once() {
    static const anira_status k_status = anira_check_abi(k_abi_version);
    if (failed(k_status)) { throw Error(k_status, "anira_check_abi"); }
}

/// Throws for a failed status of an entry that carries an anira_error.
inline void check(anira_status status, const anira_error& err) {
    if (failed(status)) { throw Error(status, err); }
}

/// Throws for a failed status of an entry without an anira_error.
inline void check(anira_status status, const char* entry) {
    if (failed(status)) { throw Error(status, entry); }
}

/// The path as UTF-8, which is what every C entry taking a path expects.
inline std::string utf8(const std::filesystem::path& path) {
    const auto text = path.u8string();  // std::u8string, or std::string under -fno-char8_t
    return {text.begin(), text.end()};
}

/// Reads a whole file as text; ANIRA_ERROR_NO_SUCH_FILE when it is not a readable file.
inline std::string read_text(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        throw Error(ANIRA_ERROR_NO_SUCH_FILE, utf8(path));
    }
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) { throw Error(ANIRA_ERROR_NO_SUCH_FILE, utf8(path)); }
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    } catch (const std::ios_base::failure& failure) {
        throw Error(ANIRA_ERROR_NO_SUCH_FILE, utf8(path) + ": " + failure.what());
    }
}

/// The two-call protocol of the to_json writers: size with a NULL buffer, then fill.
template <class Writer>
std::string write_json(const char* entry, Writer&& writer) {
    std::size_t len = 0;
    const anira_status sized = writer(nullptr, 0, &len);
    if (failed(sized) && sized != ANIRA_ERROR_BUFFER_TOO_SMALL) { throw Error(sized, entry); }
    std::string text(len + 1, '\0');
    check(writer(text.data(), text.size(), &len), entry);
    text.resize(len);
    return text;
}

/// An extension value kept beside its C record, so that pointers the record holds into
/// the value (ext::Entry::name) stay valid as long as the record does.
template <class Ext>
struct KeptExt {
    Ext m_value;
    typename ExtTraits<Ext>::Native m_native;

    explicit KeptExt(Ext kept)
        : m_value(std::move(kept)), m_native(ExtTraits<Ext>::mint(m_value)) {}
    KeptExt(const KeptExt&) = delete;
    KeptExt& operator=(const KeptExt&) = delete;
    KeptExt(KeptExt&&) = delete;
    KeptExt& operator=(KeptExt&&) = delete;
    ~KeptExt() = default;
};

}  // namespace detail

// ---- tensor spec (section 2) ---------------------------------------------------------------

/**
 * @brief One input or output of the model: your canonical name, the data type, the role, the
 * tagged axes in the model's memory order, and, for a streamed tensor, the window and context
 * it is consumed with. Move-only; copied into a ModelConfig by input()/output().
 */
class TensorSpec {
public:
    /// = anira_tensor_spec_create. @throws Error when the C entry refuses the arguments.
    TensorSpec(std::string_view name, DType dtype, Role role) {
        detail::abi_check_once();
        anira_error err{};
        detail::check(
            anira_tensor_spec_create(std::string(name).c_str(), dtype, role, &m_spec, &err),
            err);
    }
    ~TensorSpec() { anira_tensor_spec_destroy(m_spec); }
    TensorSpec(const TensorSpec&) = delete;
    TensorSpec& operator=(const TensorSpec&) = delete;
    TensorSpec(TensorSpec&& other) noexcept : m_spec(std::exchange(other.m_spec, nullptr)) {}
    TensorSpec& operator=(TensorSpec&& other) noexcept {
        if (this != &other) {
            anira_tensor_spec_destroy(m_spec);
            m_spec = std::exchange(other.m_spec, nullptr);
        }
        return *this;
    }

    /// Axis i (model memory order) with its tag and extent (ANIRA_DYNAMIC on a streamed
    /// Time axis). @throws Error{ANIRA_ERROR_INVALID_ARGUMENT}
    TensorSpec& axis(uint32_t i, AxisTag tag, int64_t extent) {
        detail::check(anira_tensor_spec_set_axis(m_spec, i, tag, extent),
                      "anira_tensor_spec_set_axis");
        return *this;
    }
    /// Streamed only: elements along the Time axis per inference and the context kept.
    TensorSpec& window(int64_t window_min, int64_t window_max, int64_t context) {
        detail::check(anira_tensor_spec_set_window(m_spec, window_min, window_max, context),
                      "anira_tensor_spec_set_window");
        return *this;
    }
    /// Time advance against the anchor: num elements per den anchor elements.
    TensorSpec& time_ratio(int64_t num, int64_t den) {
        detail::check(anira_tensor_spec_set_time_ratio(m_spec, num, den),
                      "anira_tensor_spec_set_time_ratio");
        return *this;
    }
    /// Outputs only: the model's internal delay along the Time axis.
    TensorSpec& latency(int64_t latency_elements) {
        detail::check(anira_tensor_spec_set_latency(m_spec, latency_elements),
                      "anira_tensor_spec_set_latency");
        return *this;
    }
    /// An extension record (section 1b); copied.
    template <class Ext>
    TensorSpec& ext(const Ext& value) {
        const auto native = detail::ExtTraits<Ext>::mint(value);
        anira_error err{};
        detail::check(anira_tensor_spec_set_ext(m_spec, &native.header, &err), err);
        return *this;
    }
    /// The JSON twin of ext(): a kind and its JSON text.
    TensorSpec& ext_json(std::string_view kind, std::string_view utf8) {
        anira_error err{};
        detail::check(anira_tensor_spec_set_ext_json(m_spec,
                                                     std::string(kind).c_str(),
                                                     detail::text_of(utf8),
                                                     utf8.size(),
                                                     &err),
                      err);
        return *this;
    }

    const anira_tensor_spec* native() const noexcept { return m_spec; }
    anira_tensor_spec* native() noexcept { return m_spec; }

private:
    anira_tensor_spec* m_spec = nullptr;
};

// ---- contracts (section 3) -----------------------------------------------------------------

// NOLINTBEGIN(readability-identifier-naming) the aggregates spell the document's field names

/// The real-time stream: host geometry, budget, warmup, miss policy, wait ratio.
struct Hard {
    uint32_t block_min = 0;
    uint32_t block_max = 0;
    double rate = 0;
    anira_budget_kind budget = ANIRA_BUDGET_MEASURED;
    std::chrono::nanoseconds budget_value{};  ///< Explicit only
    anira_warmup_mode warmup = ANIRA_WARMUP_UNTIL_STABLE;
    uint32_t warmup_iterations = 0;  ///< Fixed only
    anira_miss_policy on_miss = ANIRA_MISS_BYPASS;
    double wait_ratio = 0;  ///< v2 blocking_ratio
    anira_edge_cost edge_cost = ANIRA_EDGE_COST_PERMISSIVE;
};

/// Jobs without a real-time deadline: the offline posture.
struct Async {
    std::optional<std::chrono::nanoseconds> deadline;  ///< absent = no deadline
    anira_late_policy on_late = ANIRA_LATE_FINISH;
    anira_priority priority = ANIRA_PRIORITY_AUTO;
    uint32_t lanes = 0;
    uint32_t max_in_flight = 0;
    anira_delivery delivery = ANIRA_DELIVERY_POLLED;
    anira_edge_cost edge_cost = ANIRA_EDGE_COST_PERMISSIVE;
};

using Contract = std::variant<Hard, Async>;

/// The frame-invariant half of an Async job's options (section 6).
struct JobOptions {
    std::vector<int64_t> head_trim;
    bool tail_flush = true;
    anira_pad_policy below_min = ANIRA_PAD_REJECT;
};

// NOLINTEND(readability-identifier-naming)

namespace detail {

template <class Duration>
double milliseconds_of(const Duration& value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

/// A Hard aggregate as a handle; the handle is destroyed before a throw.
inline anira_contract* mint(const Hard& hard) {
    anira_error err{};
    anira_contract* contract = nullptr;
    check(anira_contract_create_hard(hard.block_min, hard.block_max, hard.rate, &contract, &err),
          err);
    try {
        check(anira_contract_hard_set_budget(contract,
                                             hard.budget,
                                             milliseconds_of(hard.budget_value)),
              "anira_contract_hard_set_budget");
        check(anira_contract_hard_set_warmup(contract, hard.warmup, hard.warmup_iterations),
              "anira_contract_hard_set_warmup");
        check(anira_contract_hard_set_on_miss(contract, hard.on_miss),
              "anira_contract_hard_set_on_miss");
        check(anira_contract_hard_set_wait_ratio(contract, hard.wait_ratio),
              "anira_contract_hard_set_wait_ratio");
        check(anira_contract_set_edge_cost(contract, hard.edge_cost),
              "anira_contract_set_edge_cost");
    } catch (...) {
        anira_contract_destroy(contract);
        throw;
    }
    return contract;
}

/// An Async aggregate as a handle.
inline anira_contract* mint(const Async& async) {
    anira_error err{};
    anira_contract* contract = nullptr;
    check(anira_contract_create_async(&contract, &err), err);
    try {
        check(anira_contract_async_set_deadline(
                  contract,
                  async.deadline.has_value() ? milliseconds_of(*async.deadline) : -1.0),
              "anira_contract_async_set_deadline");
        check(anira_contract_async_set_policy(contract,
                                              async.on_late,
                                              async.priority,
                                              async.lanes,
                                              async.max_in_flight,
                                              async.delivery),
              "anira_contract_async_set_policy");
        check(anira_contract_set_edge_cost(contract, async.edge_cost),
              "anira_contract_set_edge_cost");
    } catch (...) {
        anira_contract_destroy(contract);
        throw;
    }
    return contract;
}

inline anira_contract* mint(const Contract& contract) {
    return std::visit([](const auto& value) { return mint(value); }, contract);
}

}  // namespace detail

/**
 * @brief An anira_contract with its lifetime: minted from a Hard or Async aggregate, loaded
 * from a contract file, or adopted from a C entry (take_legacy_contract). Move-only.
 */
class ContractHandle {
public:
    /// @throws Error when the C entry refuses the call; the status says why.
    explicit ContractHandle(const Hard& hard)
        : m_contract((detail::abi_check_once(), detail::mint(hard))) {}
    explicit ContractHandle(const Async& async)
        : m_contract((detail::abi_check_once(), detail::mint(async))) {}
    explicit ContractHandle(const Contract& contract)
        : m_contract((detail::abi_check_once(), detail::mint(contract))) {}
    /// Takes ownership of a handle a C entry handed out (never NULL: an empty handle is what
    /// a moved-from object is, and every query on it throws).
    explicit ContractHandle(anira_contract* adopt) noexcept : m_contract(adopt) {}

    /// A contract file (section 8.3); a 2.x document yields its Hard contract and upgraded().
    static ContractHandle from_json(std::string_view utf8) {
        detail::abi_check_once();
        anira_error err{};
        anira_contract* contract = nullptr;
        const anira_status status =
            anira_contract_from_json(detail::text_of(utf8), utf8.size(), &contract, &err);
        detail::check(status, err);
        ContractHandle handle(contract);
        handle.m_upgraded = status == ANIRA_SUCCESS_UPGRADED;
        return handle;
    }
    static ContractHandle from_file(const std::filesystem::path& path) {
        return from_json(detail::read_text(path));
    }

    ~ContractHandle() { anira_contract_destroy(m_contract); }
    ContractHandle(const ContractHandle&) = delete;
    ContractHandle& operator=(const ContractHandle&) = delete;
    ContractHandle(ContractHandle&& other) noexcept
        : m_contract(std::exchange(other.m_contract, nullptr)), m_upgraded(other.m_upgraded) {}
    ContractHandle& operator=(ContractHandle&& other) noexcept {
        if (this != &other) {
            anira_contract_destroy(m_contract);
            m_contract = std::exchange(other.m_contract, nullptr);
            m_upgraded = other.m_upgraded;
        }
        return *this;
    }

    /// Whether this object holds a contract (false after a move).
    bool empty() const noexcept { return m_contract == nullptr; }
    /// Hard or Async. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} on an empty handle.
    anira_contract_kind kind() const {
        if (m_contract == nullptr) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT, "anira_contract_get_kind");
        }
        return anira_contract_get_kind(m_contract);
    }

    // -- the setters, for a contract that exists only as a handle (loaded, legacy) --

    /// Patches a Hard contract's stream geometry, e.g. of one loaded from a file.
    ContractHandle& hard_geometry(uint32_t block_min, uint32_t block_max, double rate) {
        detail::check(anira_contract_hard_set_geometry(m_contract, block_min, block_max, rate),
                      "anira_contract_hard_set_geometry");
        return *this;
    }
    /// The ring dtype of one tensor under this Hard contract, by canonical name: the element
    /// type of the host's samples, which the ring holds as is (the pre- and post-processor
    /// convert to the spec's dtype); F32 for every tensor never set. Data only in this
    /// pre-release: the bridge to the 2.x runtime accepts F32 alone.
    ContractHandle& hard_ring_dtype(std::string_view canonical, DType dtype) {
        const std::string name(canonical);
        detail::check(anira_contract_hard_set_ring_dtype(m_contract, name.c_str(), dtype),
                      "anira_contract_hard_set_ring_dtype");
        return *this;
    }
    /// The per-inference budget: MEASURED, or EXPLICIT with the value.
    template <class Rep, class Period>
    ContractHandle& hard_budget(anira_budget_kind kind_value,
                                std::chrono::duration<Rep, Period> value = {}) {
        detail::check(
            anira_contract_hard_set_budget(m_contract, kind_value, detail::milliseconds_of(value)),
            "anira_contract_hard_set_budget");
        return *this;
    }
    ContractHandle& hard_budget(anira_budget_kind kind_value) {
        return hard_budget(kind_value, std::chrono::nanoseconds{});
    }
    ContractHandle& hard_warmup(anira_warmup_mode mode, uint32_t iterations = 0) {
        detail::check(anira_contract_hard_set_warmup(m_contract, mode, iterations),
                      "anira_contract_hard_set_warmup");
        return *this;
    }
    ContractHandle& hard_on_miss(anira_miss_policy policy) {
        detail::check(anira_contract_hard_set_on_miss(m_contract, policy),
                      "anira_contract_hard_set_on_miss");
        return *this;
    }
    ContractHandle& hard_wait_ratio(double ratio) {
        detail::check(anira_contract_hard_set_wait_ratio(m_contract, ratio),
                      "anira_contract_hard_set_wait_ratio");
        return *this;
    }
    /// The per-job deadline of an Async contract; nullopt = none.
    ContractHandle& async_deadline(std::optional<std::chrono::nanoseconds> deadline) {
        detail::check(
            anira_contract_async_set_deadline(m_contract,
                                              deadline ? detail::milliseconds_of(*deadline) : -1.0),
            "anira_contract_async_set_deadline");
        return *this;
    }
    ContractHandle& async_policy(anira_late_policy on_late,
                                 anira_priority priority,
                                 uint32_t lanes,
                                 uint32_t max_in_flight,
                                 anira_delivery delivery) {
        detail::check(anira_contract_async_set_policy(m_contract,
                                                      on_late,
                                                      priority,
                                                      lanes,
                                                      max_in_flight,
                                                      delivery),
                      "anira_contract_async_set_policy");
        return *this;
    }
    ContractHandle& edge_cost(anira_edge_cost cost) {
        detail::check(anira_contract_set_edge_cost(m_contract, cost),
                      "anira_contract_set_edge_cost");
        return *this;
    }
    template <class Ext>
    ContractHandle& ext(const Ext& value) {
        const auto native = detail::ExtTraits<Ext>::mint(value);
        anira_error err{};
        detail::check(anira_contract_set_ext(m_contract, &native.header, &err), err);
        return *this;
    }
    ContractHandle& ext_json(std::string_view kind, std::string_view utf8) {
        anira_error err{};
        detail::check(anira_contract_set_ext_json(m_contract,
                                                  std::string(kind).c_str(),
                                                  detail::text_of(utf8),
                                                  utf8.size(),
                                                  &err),
                      err);
        return *this;
    }
    /// Whether this contract came out of a 2.x document (from_json, or a
    /// ModelConfig::take_legacy_contract).
    bool upgraded() const noexcept { return m_upgraded; }

    const anira_contract* native() const noexcept { return m_contract; }
    anira_contract* native() noexcept { return m_contract; }
    /// Hands the handle out (to a C entry that takes ownership); this object becomes empty.
    anira_contract* release() noexcept { return std::exchange(m_contract, nullptr); }

private:
    friend class ModelConfig;
    ContractHandle(anira_contract* adopt, bool upgraded) noexcept
        : m_contract(adopt), m_upgraded(upgraded) {}

    anira_contract* m_contract = nullptr;
    bool m_upgraded = false;
};

// ---- model config (section 5) --------------------------------------------------------------

/**
 * @brief The model: one entry per engine (a file or bytes) with what that export calls each
 * tensor and how it lays out its axes, the input and output specs, the default engine, the
 * state, the instance ceiling and the anchor. Move-only.
 */
class ModelConfig {
public:
    /// @throws Error when the C entry refuses the call; the status says why.
    ModelConfig() {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_model_config_create(&m_config, &err), err);
    }
    /// A model file (section 8.1); relative paths resolve against base_dir. A 2.x document is
    /// upgraded: upgraded() is true and take_legacy_contract() carries its Hard contract.
    static ModelConfig from_json(std::string_view utf8, std::string_view base_dir = {}) {
        detail::abi_check_once();
        anira_error err{};
        anira_model_config* config = nullptr;
        const std::string base(base_dir);
        const anira_status status =
            anira_model_config_from_json(detail::text_of(utf8),
                                         utf8.size(),
                                         base.empty() ? nullptr : base.c_str(),
                                         &config,
                                         &err);
        detail::check(status, err);
        return {config, status == ANIRA_SUCCESS_UPGRADED};
    }
    /// Reads a model file; its directory is the base_dir.
    static ModelConfig from_file(const std::filesystem::path& path) {
        detail::abi_check_once();
        anira_error err{};
        anira_model_config* config = nullptr;
        const anira_status status =
            anira_model_config_from_json_file(detail::utf8(path).c_str(), &config, &err);
        detail::check(status, err);
        return {config, status == ANIRA_SUCCESS_UPGRADED};
    }

    ~ModelConfig() { anira_model_config_destroy(m_config); }
    ModelConfig(const ModelConfig&) = delete;
    ModelConfig& operator=(const ModelConfig&) = delete;
    ModelConfig(ModelConfig&& other) noexcept
        : m_config(std::exchange(other.m_config, nullptr)), m_upgraded(other.m_upgraded) {}
    ModelConfig& operator=(ModelConfig&& other) noexcept {
        if (this != &other) {
            anira_model_config_destroy(m_config);
            m_config = std::exchange(other.m_config, nullptr);
            m_upgraded = other.m_upgraded;
        }
        return *this;
    }

    // -- model entries --

    /// A file for a built-in engine; returns the entry index.
    uint32_t add_model_path(Engine engine, const std::filesystem::path& path) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_path(m_config,
                                                        engine,
                                                        detail::utf8(path).c_str(),
                                                        &index,
                                                        &err),
                      err);
        return index;
    }
    /// A file for a custom engine registered under a reverse-URI name.
    uint32_t add_model_path(std::string_view engine_id, const std::filesystem::path& path) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_path_custom(m_config,
                                                               std::string(engine_id).c_str(),
                                                               detail::utf8(path).c_str(),
                                                               &index,
                                                               &err),
                      err);
        return index;
    }
    /// Bytes for a built-in engine: copied, or borrowed until the last carrier of the bytes
    /// dies (this config, a set_model_bytes replacement, later the handler that copied the
    /// config), when release(bytes, ctx), if given, is called exactly once.
    uint32_t add_model_bytes(Engine engine,
                             std::span<const std::byte> bytes,
                             anira_bytes_ownership ownership = ANIRA_BYTES_COPY,
                             anira_bytes_release_fn release = nullptr,
                             void* ctx = nullptr) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_bytes(m_config,
                                                         engine,
                                                         bytes.data(),
                                                         bytes.size(),
                                                         ownership,
                                                         release,
                                                         ctx,
                                                         &index,
                                                         &err),
                      err);
        return index;
    }
    uint32_t add_model_bytes(std::string_view engine_id,
                             std::span<const std::byte> bytes,
                             anira_bytes_ownership ownership = ANIRA_BYTES_COPY,
                             anira_bytes_release_fn release = nullptr,
                             void* ctx = nullptr) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_bytes_custom(m_config,
                                                                std::string(engine_id).c_str(),
                                                                bytes.data(),
                                                                bytes.size(),
                                                                ownership,
                                                                release,
                                                                ctx,
                                                                &index,
                                                                &err),
                      err);
        return index;
    }
    /// Replaces an entry's source with bytes, e.g. to patch a path a JSON file named.
    ModelConfig& set_model_bytes(uint32_t index,
                                 std::span<const std::byte> bytes,
                                 anira_bytes_ownership ownership = ANIRA_BYTES_COPY,
                                 anira_bytes_release_fn release = nullptr,
                                 void* ctx = nullptr) {
        anira_error err{};
        detail::check(anira_model_config_set_model_bytes(m_config,
                                                         index,
                                                         bytes.data(),
                                                         bytes.size(),
                                                         ownership,
                                                         release,
                                                         ctx,
                                                         &err),
                      err);
        return *this;
    }
    uint32_t model_count() const noexcept { return anira_model_config_model_count(m_config); }
    Engine model_engine(uint32_t index) const noexcept {
        return anira_model_config_model_engine(m_config, index);
    }
    /// The custom engine's name; empty for a built-in engine. The view is owned by the config
    /// and valid until the config is mutated, moved or destroyed.
    std::string_view model_engine_id(uint32_t index) const noexcept {
        const char* id = anira_model_config_model_engine_id(m_config, index);
        return id != nullptr ? std::string_view(id) : std::string_view();
    }
    /// The entry's path, owned by the config and valid until it is mutated, moved or
    /// destroyed. @throws Error{ANIRA_ERROR_INVALID_STATE} on a bytes entry.
    std::string_view model_path(uint32_t index) const {
        const char* path = anira_model_config_model_path(m_config, index);
        if (path == nullptr) {
            throw Error(
                index < model_count() ? ANIRA_ERROR_INVALID_STATE : ANIRA_ERROR_INVALID_ARGUMENT,
                "anira_model_config_model_path");
        }
        return path;
    }
    /// The entry's bytes; the span is invalidated by set_model_bytes on that entry and by the
    /// config's destruction. @throws Error on a path entry.
    std::span<const std::byte> model_bytes(uint32_t index) const {
        const void* bytes = nullptr;
        std::size_t size = 0;
        detail::check(anira_model_config_model_bytes(m_config, index, &bytes, &size),
                      "anira_model_config_model_bytes");
        return {static_cast<const std::byte*>(bytes), size};
    }
    /// What this entry's export calls the tensor you named canonical (binds it by name).
    ModelConfig& tensor_name(uint32_t index,
                             std::string_view canonical,
                             std::string_view engine_name) {
        detail::check(anira_model_config_set_tensor_name(m_config,
                                                         index,
                                                         std::string(canonical).c_str(),
                                                         std::string(engine_name).c_str()),
                      "anira_model_config_set_tensor_name");
        return *this;
    }
    /// The order in which this entry's export holds the tensor's axes: the spec axis at each
    /// file position, ANIRA_AXIS_INSERT for a unit axis the spec lacks. An empty span clears.
    ModelConfig& tensor_layout(uint32_t index,
                               std::string_view canonical,
                               std::span<const uint32_t> axes) {
        detail::check(anira_model_config_set_tensor_layout(m_config,
                                                           index,
                                                           std::string(canonical).c_str(),
                                                           axes.empty() ? nullptr : axes.data(),
                                                           static_cast<uint32_t>(axes.size())),
                      "anira_model_config_set_tensor_layout");
        return *this;
    }
    /// An extension on one entry, e.g. ext::Entry{"decode"}.
    template <class Ext>
    ModelConfig& model_ext(uint32_t index, const Ext& value) {
        const auto native = detail::ExtTraits<Ext>::mint(value);
        anira_error err{};
        detail::check(anira_model_config_set_model_ext(m_config, index, &native.header, &err), err);
        return *this;
    }
    ModelConfig& model_ext_json(uint32_t index, std::string_view kind, std::string_view utf8) {
        anira_error err{};
        detail::check(anira_model_config_set_model_ext_json(m_config,
                                                            index,
                                                            std::string(kind).c_str(),
                                                            detail::text_of(utf8),
                                                            utf8.size(),
                                                            &err),
                      err);
        return *this;
    }

    // -- tensors, selection, state, anchor --

    /// Appends an input spec (copied; the spec may be destroyed afterwards).
    ModelConfig& input(const TensorSpec& spec) {
        detail::check(anira_model_config_add_input(m_config, spec.native()),
                      "anira_model_config_add_input");
        return *this;
    }
    ModelConfig& output(const TensorSpec& spec) {
        detail::check(anira_model_config_add_output(m_config, spec.native()),
                      "anira_model_config_add_output");
        return *this;
    }
    ModelConfig& default_engine(Engine engine) {
        detail::check(anira_model_config_set_default_engine(m_config, engine),
                      "anira_model_config_set_default_engine");
        return *this;
    }
    ModelConfig& default_engine(std::string_view engine_id) {
        detail::check(
            anira_model_config_set_default_engine_custom(m_config, std::string(engine_id).c_str()),
            "anira_model_config_set_default_engine_custom");
        return *this;
    }
    ModelConfig& state(anira_model_state value) {
        detail::check(anira_model_config_set_state(m_config, value),
                      "anira_model_config_set_state");
        return *this;
    }
    ModelConfig& max_instances(uint32_t value) {
        detail::check(anira_model_config_set_max_instances(m_config, value),
                      "anira_model_config_set_max_instances");
        return *this;
    }
    /// The streamed tensor that is the model's clock, by canonical name; empty = default.
    ModelConfig& anchor(std::string_view canonical) {
        const std::string name(canonical);
        detail::check(
            anira_model_config_set_anchor(m_config, name.empty() ? nullptr : name.c_str()),
            "anira_model_config_set_anchor");
        return *this;
    }
    template <class Ext>
    ModelConfig& ext(const Ext& value) {
        const auto native = detail::ExtTraits<Ext>::mint(value);
        anira_error err{};
        detail::check(anira_model_config_set_ext(m_config, &native.header, &err), err);
        return *this;
    }
    ModelConfig& ext_json(std::string_view kind, std::string_view utf8) {
        anira_error err{};
        detail::check(anira_model_config_set_ext_json(m_config,
                                                      std::string(kind).c_str(),
                                                      detail::text_of(utf8),
                                                      utf8.size(),
                                                      &err),
                      err);
        return *this;
    }

    // -- JSON and the 2.x upgrade --

    /// The model file in 3.x spelling, fixed key order.
    std::string to_json() const {
        return detail::write_json("anira_model_config_to_json",
                                  [this](char* buf, std::size_t cap, std::size_t* len) {
                                      return anira_model_config_to_json(m_config, buf, cap, len);
                                  });
    }
    /// Whether from_json/from_file read a 2.x document.
    bool upgraded() const noexcept { return m_upgraded; }
    /// The Hard contract a 2.x upgrade held back (max_inference_time, warm_up,
    /// blocking_ratio); once, and only after an upgrade.
    std::optional<ContractHandle> take_legacy_contract() {
        anira_contract* contract = nullptr;
        detail::check(anira_model_config_take_legacy_contract(m_config, &contract),
                      "anira_model_config_take_legacy_contract");
        if (contract == nullptr) { return std::nullopt; }
        return ContractHandle(contract, true);  // the product of a 2.x document
    }

    const anira_model_config* native() const noexcept { return m_config; }
    anira_model_config* native() noexcept { return m_config; }

private:
    ModelConfig(anira_model_config* config, bool upgraded) noexcept
        : m_config(config), m_upgraded(upgraded) {}

    anira_model_config* m_config = nullptr;
    bool m_upgraded = false;
};

// ---- machine config (section 4) ------------------------------------------------------------

/**
 * @brief The process: the inference thread pool, logging, the devices anira may use.
 * Move-only.
 */
class MachineConfig {
public:
    /// @throws Error when the C entry refuses the call; the status says why.
    MachineConfig() {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_machine_config_create(&m_config, &err), err);
    }
    /// A machine file (section 8.2); a 2.x document's context_config is upgraded.
    static MachineConfig from_json(std::string_view utf8) {
        detail::abi_check_once();
        anira_error err{};
        anira_machine_config* config = nullptr;
        const anira_status status =
            anira_machine_config_from_json(detail::text_of(utf8), utf8.size(), &config, &err);
        detail::check(status, err);
        return {config, status == ANIRA_SUCCESS_UPGRADED};
    }
    static MachineConfig from_file(const std::filesystem::path& path) {
        return from_json(detail::read_text(path));
    }

    ~MachineConfig() { anira_machine_config_destroy(m_config); }
    MachineConfig(const MachineConfig&) = delete;
    MachineConfig& operator=(const MachineConfig&) = delete;
    MachineConfig(MachineConfig&& other) noexcept
        : m_config(std::exchange(other.m_config, nullptr)), m_upgraded(other.m_upgraded) {}
    MachineConfig& operator=(MachineConfig&& other) noexcept {
        if (this != &other) {
            anira_machine_config_destroy(m_config);
            m_config = std::exchange(other.m_config, nullptr);
            m_upgraded = other.m_upgraded;
        }
        return *this;
    }

    /// The thread pool: ANIRA_THREADS_AUTO sizes it, 0 means the host brings its threads.
    MachineConfig& threads(uint32_t num_threads,
                           anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF) {
        detail::check(anira_machine_config_set_threads(m_config, num_threads, wait),
                      "anira_machine_config_set_threads");
        return *this;
    }
    MachineConfig& log_level(anira_log_level level) {
        detail::check(anira_machine_config_set_log_level(m_config, level),
                      "anira_machine_config_set_log_level");
        return *this;
    }
    MachineConfig& log_drain(anira_log_drain drain, uint32_t interval_ms = 10) {
        detail::check(anira_machine_config_set_log_drain(m_config, drain, interval_ms),
                      "anira_machine_config_set_log_drain");
        return *this;
    }
    MachineConfig& log_queue_capacity(uint32_t capacity) {
        detail::check(anira_machine_config_set_log_queue_capacity(m_config, capacity),
                      "anira_machine_config_set_log_queue_capacity");
        return *this;
    }
    MachineConfig& log_flags(uint32_t flags) {
        detail::check(anira_machine_config_set_log_flags(m_config, flags),
                      "anira_machine_config_set_log_flags");
        return *this;
    }
    /// The sink callback and its user data (the raw pair at this pre-release).
    MachineConfig& log_sink(anira_log_fn callback, void* user_data = nullptr) {
        detail::check(anira_machine_config_set_log_sink(m_config, callback, user_data),
                      "anira_machine_config_set_log_sink");
        return *this;
    }
    /// The one-shot descriptor equal to the five scalar log setters.
    MachineConfig& log(const anira_log_desc& desc) {
        detail::check(anira_machine_config_set_log(m_config, &desc),
                      "anira_machine_config_set_log");
        return *this;
    }
    MachineConfig& cuda(const anira_cuda_desc& desc) { return cuda(&desc); }
    /// The pointer form: NULL clears the block.
    MachineConfig& cuda(const anira_cuda_desc* desc) {
        detail::check(anira_machine_config_set_cuda(m_config, desc),
                      "anira_machine_config_set_cuda");
        return *this;
    }
    MachineConfig& gl(const anira_gl_desc& desc) { return gl(&desc); }
    /// The pointer form: NULL clears the block.
    MachineConfig& gl(const anira_gl_desc* desc) {
        detail::check(anira_machine_config_set_gl(m_config, desc), "anira_machine_config_set_gl");
        return *this;
    }
    MachineConfig& vulkan(const anira_vulkan_desc& desc) { return vulkan(&desc); }
    /// The pointer form: NULL clears the block.
    MachineConfig& vulkan(const anira_vulkan_desc* desc) {
        detail::check(anira_machine_config_set_vulkan(m_config, desc),
                      "anira_machine_config_set_vulkan");
        return *this;
    }
    MachineConfig& metal(const anira_metal_desc& desc) { return metal(&desc); }
    /// The pointer form: NULL clears the block.
    MachineConfig& metal(const anira_metal_desc* desc) {
        detail::check(anira_machine_config_set_metal(m_config, desc),
                      "anira_machine_config_set_metal");
        return *this;
    }
    MachineConfig& d3d12(const anira_d3d12_desc& desc) { return d3d12(&desc); }
    /// The pointer form: NULL clears the block.
    MachineConfig& d3d12(const anira_d3d12_desc* desc) {
        detail::check(anira_machine_config_set_d3d12(m_config, desc),
                      "anira_machine_config_set_d3d12");
        return *this;
    }
    MachineConfig& webgpu(const anira_webgpu_desc& desc) { return webgpu(&desc); }
    /// The pointer form: NULL clears the block.
    MachineConfig& webgpu(const anira_webgpu_desc* desc) {
        detail::check(anira_machine_config_set_webgpu(m_config, desc),
                      "anira_machine_config_set_webgpu");
        return *this;
    }
    template <class Ext>
    MachineConfig& ext(const Ext& value) {
        const auto native = detail::ExtTraits<Ext>::mint(value);
        anira_error err{};
        detail::check(anira_machine_config_set_ext(m_config, &native.header, &err), err);
        return *this;
    }
    MachineConfig& ext_json(std::string_view kind, std::string_view utf8) {
        anira_error err{};
        detail::check(anira_machine_config_set_ext_json(m_config,
                                                        std::string(kind).c_str(),
                                                        detail::text_of(utf8),
                                                        utf8.size(),
                                                        &err),
                      err);
        return *this;
    }
    std::string to_json() const {
        return detail::write_json("anira_machine_config_to_json",
                                  [this](char* buf, std::size_t cap, std::size_t* len) {
                                      return anira_machine_config_to_json(m_config, buf, cap, len);
                                  });
    }
    bool upgraded() const noexcept { return m_upgraded; }

    const anira_machine_config* native() const noexcept { return m_config; }
    anira_machine_config* native() noexcept { return m_config; }

private:
    MachineConfig(anira_machine_config* config, bool upgraded) noexcept
        : m_config(config), m_upgraded(upgraded) {}

    anira_machine_config* m_config = nullptr;
    bool m_upgraded = false;
};

// ---- machine (section 4) -------------------------------------------------------------------

namespace detail {

/**
 * @brief Runs the two-call enumeration protocol of section 6a: the count, then the rows.
 * `call(count, out)` is the C entry with everything but those two bound; a stride-explicit
 * entry binds sizeof(T).
 */
template <class T, class Call>
std::vector<T> enumerate(Call&& call, const char* entry) {
    uint32_t count = 0;
    check(call(&count, static_cast<T*>(nullptr)), entry);
    std::vector<T> rows(count);
    if (count == 0) { return rows; }
    check(call(&count, rows.data()), entry);
    rows.resize(std::min<std::size_t>(count, rows.size()));
    return rows;
}

}  // namespace detail

/**
 * @brief A view over a machine's probed capabilities (anira_capabilities): what backends are
 * usable here, which memory domains a tensor may live in, the extension kinds this build
 * understands, and the edge registry. Valid while the Machine is; refreshed in place by
 * Machine::probe.
 */
class Capabilities {
public:
    explicit Capabilities(const anira_capabilities* capabilities) noexcept
        : m_capabilities(capabilities) {}

    /// The backends compiled in and usable here.
    std::vector<BackendId> backends() const {
        return detail::enumerate<BackendId>(
            [this](uint32_t* count, BackendId* out) {
                return anira_capabilities_backends(m_capabilities, sizeof(BackendId), count, out);
            },
            "anira_capabilities_backends");
    }
    std::vector<Domain> domains() const {
        return detail::enumerate<Domain>(
            [this](uint32_t* count, Domain* out) {
                return anira_capabilities_domains(m_capabilities, count, out);
            },
            "anira_capabilities_domains");
    }
    std::vector<std::string> ext_kinds() const {
        const std::vector<const char*> kinds = detail::enumerate<const char*>(
            [this](uint32_t* count, const char** out) {
                return anira_capabilities_ext_kinds(m_capabilities, count, out);
            },
            "anira_capabilities_ext_kinds");
        return {kinds.begin(), kinds.end()};
    }
    /// Every row of the edge registry, available or not.
    std::vector<anira_edge_info> edges() const {
        return detail::enumerate<anira_edge_info>(
            [this](uint32_t* count, anira_edge_info* out) {
                return anira_capabilities_edges(
                    m_capabilities, sizeof(anira_edge_info), count, out);
            },
            "anira_capabilities_edges");
    }
    /// One row, by domain and backend.
    /// @throws Error with ANIRA_ERROR_EDGE_UNREACHABLE when the registry has no such row.
    anira_edge_info edge(Domain from, const BackendId& to) const {
        anira_edge_info row = ANIRA_EDGE_INFO_INIT;
        detail::check(anira_capabilities_edge(m_capabilities, from, &to, &row),
                      "anira_capabilities_edge");
        return row;
    }

    const anira_capabilities* native() const noexcept { return m_capabilities; }

private:
    const anira_capabilities* m_capabilities;
};

/**
 * @brief An anira_machine with its lifetime: a refcounted handle over this copy's core,
 * created from a MachineConfig (section 4). Two machines in one copy are two views of one
 * core with two log sinks.
 */
class Machine {
public:
    /// @throws Error when the C entry refuses the config (a device block, an unconsumed
    /// machine extension).
    explicit Machine(const MachineConfig& config) {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_machine_create(config.native(), &m_machine, &err), err);
    }
    ~Machine() { anira_machine_destroy(m_machine); }
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;
    Machine(Machine&& other) noexcept : m_machine(std::exchange(other.m_machine, nullptr)) {}
    Machine& operator=(Machine&& other) noexcept {
        if (this != &other) {
            anira_machine_destroy(m_machine);
            m_machine = std::exchange(other.m_machine, nullptr);
        }
        return *this;
    }

    Capabilities capabilities() const { return Capabilities(anira_machine_capabilities(m_machine)); }
    /// Re-runs the probe; `force` re-runs every rung even where a cached answer exists.
    void probe(bool force = false) {
        anira_error err{};
        detail::check(anira_machine_probe(m_machine, force ? 1U : 0U, &err), err);
    }
    /// Delivers the queued real-time records to the sinks (ANIRA_LOG_DRAIN_MANUAL).
    std::size_t drain_log() { return anira_machine_drain_log(m_machine); }
    /// The size of the inference thread pool serving this machine.
    uint32_t num_inference_threads() const { return anira_machine_num_inference_threads(m_machine); }
    /// The size of a tensor's byte image on this machine.
    uint64_t byte_image_bytes(uint64_t num_elements, DType dtype) const {
        return anira_machine_byte_image_bytes(m_machine, num_elements, dtype);
    }

    const anira_machine* native() const noexcept { return m_machine; }
    anira_machine* native() noexcept { return m_machine; }

private:
    anira_machine* m_machine = nullptr;
};

/// What this build compiled in, without a machine (anira_enabled_backends).
inline std::vector<BackendId> enabled_backends() {
    return detail::enumerate<BackendId>(
        [](uint32_t* count, BackendId* out) {
            return anira_enabled_backends(sizeof(BackendId), count, out);
        },
        "anira_enabled_backends");
}

/// The steady clock of anira_now_ms / anira_now_ns, for deadlines and submit timestamps.
inline double now_ms() noexcept {
    return anira_now_ms();
}
inline uint64_t now_ns() noexcept {
    return anira_now_ns();
}

/// anira_shutdown: effective only when no Machine and no handler exist in this copy.
inline anira_status shutdown() noexcept {
    return anira_shutdown();
}
/// anira_release_core_if_idle: true when the core was freed.
inline bool release_core_if_idle() noexcept {
    return anira_release_core_if_idle() != 0U;
}
inline bool has_core() noexcept {
    return anira_has_core() != 0U;
}

// ---- job options (section 6) ---------------------------------------------------------------

/**
 * @brief An anira_job_options with its lifetime, minted from a JobOptions aggregate.
 * Extension values set through ext() are copied and kept alive, with their C records, by the
 * handle (the C entry borrows them until submit).
 */
class JobOptionsHandle {
public:
    /// @throws Error when the C entry refuses the call; the status says why.
    explicit JobOptionsHandle(const JobOptions& options = {}) {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_job_options_create(&m_options, &err), err);
        try {
            if (!options.head_trim.empty()) {
                detail::check(
                    anira_job_options_set_head_trim(m_options,
                                                    static_cast<uint32_t>(options.head_trim.size()),
                                                    options.head_trim.data()),
                    "anira_job_options_set_head_trim");
            }
            detail::check(anira_job_options_set_tail_flush(m_options, options.tail_flush ? 1u : 0u),
                          "anira_job_options_set_tail_flush");
            detail::check(anira_job_options_set_below_min(m_options, options.below_min),
                          "anira_job_options_set_below_min");
        } catch (...) {
            anira_job_options_destroy(m_options);
            throw;
        }
    }
    ~JobOptionsHandle() { anira_job_options_destroy(m_options); }
    JobOptionsHandle(const JobOptionsHandle&) = delete;
    JobOptionsHandle& operator=(const JobOptionsHandle&) = delete;
    JobOptionsHandle(JobOptionsHandle&& other) noexcept
        : m_options(std::exchange(other.m_options, nullptr)), m_kept(std::move(other.m_kept)) {}
    JobOptionsHandle& operator=(JobOptionsHandle&& other) noexcept {
        if (this != &other) {
            anira_job_options_destroy(m_options);
            m_options = std::exchange(other.m_options, nullptr);
            m_kept = std::move(other.m_kept);
        }
        return *this;
    }

    /// A per-job extension. The C entry borrows the record until submit, so the handle keeps
    /// the value and its record alive (a copy of the value: a temporary argument is fine).
    template <class Ext>
    JobOptionsHandle& ext(const Ext& value) {
        auto kept = std::make_shared<detail::KeptExt<Ext>>(value);
        m_kept.reserve(m_kept.size() + 1);  // so that the push below cannot throw
        detail::check(anira_job_options_set_ext(m_options, &kept->m_native.header),
                      "anira_job_options_set_ext");
        m_kept.push_back(std::move(kept));
        return *this;
    }
    JobOptionsHandle& ext_json(std::string_view kind, std::string_view utf8) {
        detail::check(anira_job_options_set_ext_json(m_options,
                                                     std::string(kind).c_str(),
                                                     detail::text_of(utf8),
                                                     utf8.size()),
                      "anira_job_options_set_ext_json");
        return *this;
    }

    const anira_job_options* native() const noexcept { return m_options; }
    anira_job_options* native() noexcept { return m_options; }

private:
    anira_job_options* m_options = nullptr;
    std::vector<std::shared_ptr<void>> m_kept;
};

/// The extension kinds this build understands (anira_registered_ext_kinds); the names are
/// static storage.
inline std::vector<std::string_view> registered_ext_kinds() {
    uint32_t count = 0;
    detail::check(anira_registered_ext_kinds(&count, nullptr), "anira_registered_ext_kinds");
    std::vector<const char*> names(count, nullptr);
    if (count != 0) {
        detail::check(anira_registered_ext_kinds(&count, names.data()),
                      "anira_registered_ext_kinds");
    }
    std::vector<std::string_view> kinds;
    kinds.reserve(names.size());
    for (const char* name : names) {
        if (name != nullptr) { kinds.emplace_back(name); }
    }
    return kinds;
}

}  // namespace anira

#endif  // ANIRA_HPP
