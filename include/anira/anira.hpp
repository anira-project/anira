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
 * Scope at this pre-release: the configuration half (tensor specs, model, context and
 * contract configuration, job options) with their read-back (SpecView, the ModelConfig
 * getters, ContractHandle::hard()), the Context over the core, the pipeline half of section 6
 * (Pipeline, stage::Inference, stage::Custom and PlanReport), the stages and the custom
 * engines of section 7 (Stage, Stage::Prepared, StageContext and RingView over
 * anira/abi/stage.h; Engine, Engine::Loaded, Engine::Prepared, EngineLoadInfo and
 * EngineContext over anira/abi/engine.h, registered through Pipeline::register_engine or
 * stage::Inference::engine; InitInfo and PrepareInfo, the records both share, over
 * anira/abi/lifecycle.h) and the runtime tensor of section 1: Tensor and SyncToken, the C
 * structs of anira/abi/tensor.h with factories on them. The InferenceHandler class and the
 * _wait twins arrive with the runtime cut-over; until then a handler is driven through the C
 * entries of anira/abi/handler.h.
 *
 * Deviations from the architecture document, section 6 (stated here and on the docs page):
 * anira::JsonConfigLoader is not declared (the 2.x class of that name is still in every
 * example; use ModelConfig::from_file, ContextConfig::from_file, ContractHandle::from_file);
 * ModelConfig::take_legacy_contract returns std::optional<ContractHandle>: a handle without
 * geometry, patched with hard_geometry, which hard() reads back into a Hard;
 * ContextConfig::log_sink takes the raw (anira_log_fn, void*) pair; ModelConfig::anchor takes
 * the tensor's canonical name;
 * ContractHandle, JobOptionsHandle and the upgraded() queries are additions; set_model_bytes
 * chains (the document's returns void); add_model_bytes and set_model_bytes carry the
 * (release, ctx) pair of the C entry and have custom-engine twins; JobOptions has no
 * on_complete yet (it arrives with Ticket); a contract file loads into a ContractHandle,
 * patched with hard_geometry and the other setters, which hard() reads back into a Hard;
 * Tensor has data(DType) beside data_f32, and from_host_planar / plane<T> for planar host
 * memory (typed by the element, which the document does not have); from_metal, from_iosurface,
 * from_ahardwarebuffer and from_d3d12 are not declared and there is no anira_all.hpp (the draft
 * factories have no C++ spelling while unmeasured: call anira_tensor_init_metal(&tensor, ...)
 * on a Tensor); Stage is the registration and Stage::Prepared what its prepare returns per
 * handler, the C lifecycle's prepared pointer as a class on which the phases and reset run and
 * which anira deletes at unprepare; Stage::init takes an InitInfo and Stage::prepare a
 * PrepareInfo, the C records as views, whose handler is the anira_handler* (no handler class
 * yet); a Stage states the phases
 * it fills in phases(), because the C descriptor tells a filled phase from a NULL one, and its
 * real-time promise in flags() (anira_stage_desc::flags); StageContext has
 * input_tensor / output_tensor filling a Tensor, the C accessors, in place of model_input /
 * model_output references, every accessor returning the C status and filling an out-parameter
 * (asking for what a slot does not have in a phase is the stage's bug, and is recorded), and
 * entry() for the chunk's entry; the RingView calls take a std::span; stage::Custom takes no
 * domains and has no single-callable form (a stage works at the host end of every slot and never
 * crosses a domain), and a Pipeline holds at most one of them; the variant of Pipeline is named
 * Pipeline::AnyStage; a custom engine is anira::Engine, registered
 * under its id and shaped like Stage with one level more, with Engine::Loaded what its load
 * returns per loaded model (EngineLoadInfo the record), Engine::Prepared what the Loaded's
 * prepare returns per handler (PrepareInfo the record, the stage's) and EngineContext the
 * per-call context, and the enum alias of anira_engine is EngineKind, so that the class may
 * carry the name.
 *
 * Requirements: C++20 with exceptions; std::filesystem, which on Apple platforms means a
 * deployment target of macOS 10.15 / iOS 13 or later.
 *
 * Every entry is [main-thread] and may allocate, except the Tensor factories and accessors,
 * which are the [thread-safe] [callback-safe] real-time field fills and reads of
 * anira/abi/tensor.h (noexcept; only Tensor::from_dlpack is [main-thread] and throws),
 * SyncToken::reset / dup, which are [thread-safe, !audio-thread], and everything a phase
 * callback of a Stage reaches: RingView and StageContext are [callback-safe] and nonblocking
 * (noexcept, no allocation; a status or a count is returned, never thrown), and EngineContext,
 * what an Engine's process reaches on an inference thread, is noexcept field reads. The ABI is
 * checked once per process on the first handle created (anira_check_abi against
 * ANIRA_ABI_VERSION); a Tensor, a RingView, a StageContext and an EngineContext are no handles
 * and check nothing.
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
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>  // IWYU pragma: keep - the umbrella of the C headers
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/abi/version.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <ios>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ratio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace anira {

// ---- aliases -------------------------------------------------------------------------------

using DType = anira_dtype;
/// The engines of anira_engine (ANIRA_ENGINE_CUSTOM with an id for a registered one, the pair
/// rule); the class of a custom engine is anira::Engine.
using EngineKind = anira_engine;
using Provider = anira_provider;
using Domain = anira_domain;
using SyncKind = anira_sync_kind;
using Role = anira_role;
using AxisTag = anira_axis_tag;
using BackendId = anira_backend_id;

// NOLINTBEGIN(readability-identifier-naming) the aggregates spell the pair's two halves
/// An engine as the pair names it (the pair rule of anira_engine): the value, and the custom
/// engine's id beside ANIRA_ENGINE_CUSTOM, empty for every other value. What every engine getter
/// returns; the view is owned by what it was read from (the config, the record of a call).
struct EngineRef {
    EngineKind kind = ANIRA_ENGINE_NONE;
    std::string_view id;
};

/// A provider as the pair names it (the pair rule of anira_provider): the value, and the custom
/// provider's name beside ANIRA_PROVIDER_CUSTOM, empty for every other value. What every
/// provider getter returns; the view is owned by what it was read from.
struct ProviderRef {
    Provider kind = ANIRA_PROVIDER_DEFAULT;
    std::string_view id;
};
// NOLINTEND(readability-identifier-naming)

namespace detail {

/// ANIRA_FAILED / ANIRA_SUCCEEDED without the C macros (no C-style cast in the wrapper).
constexpr bool failed(anira_status status) noexcept {
    return static_cast<int32_t>(status) < 0;
}
constexpr bool succeeded(anira_status status) noexcept {
    return !failed(status);
}

/// A pair read through a status getter with two out-parameters: the value and its id, the
/// default EngineRef / ProviderRef when the getter refuses.
template <class Read>
EngineRef engine_ref(Read read) noexcept {
    anira_engine engine = ANIRA_ENGINE_NONE;
    const char* id = nullptr;
    if (failed(read(&engine, &id))) { return EngineRef{}; }
    return EngineRef{.kind = engine,
                     .id = id != nullptr ? std::string_view(id) : std::string_view()};
}
template <class Read>
ProviderRef provider_ref(Read read) noexcept {
    anira_provider provider = ANIRA_PROVIDER_DEFAULT;
    const char* id = nullptr;
    if (failed(read(&provider, &id))) { return ProviderRef{}; }
    return ProviderRef{.kind = provider,
                       .id = id != nullptr ? std::string_view(id) : std::string_view()};
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

/// The "entry" kind: the entry point of a LibTorch or ExecuTorch file.
struct Entry {
    std::string name;
};

/// The "provider_options" kind, on the context config: the options an engine's runtime takes
/// for a provider, one set per backend, as string pairs in the runtime's own vocabulary
/// (ONNX Runtime's execution provider option keys; an Engine's own words). Each set is checked
/// against its backend: the engine must consume the kind (the ONNX Runtime adapter, which reads
/// the set at load into the execution provider's options; an Engine whose consumed_kinds()
/// lists "context:provider_options", which reads it through EngineLoadInfo::option_keys()),
/// else the set is refused as unconsumed naming the backend, and the provider must be one the
/// engine serves here, else ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create; a set for a
/// backend no plan runs on is not an error. The options are part of the loaded model.
struct ProviderOptions {
    struct Set {
        EngineKind engine = ANIRA_ENGINE_NONE;  ///< ANIRA_ENGINE_CUSTOM with engine_id for a
                                                ///< custom engine
        std::string engine_id;
        Provider provider = ANIRA_PROVIDER_DEFAULT;  ///< ANIRA_PROVIDER_CUSTOM with provider_id
        std::string provider_id;
        std::vector<std::pair<std::string, std::string>> options;
    };
    std::vector<Set> sets;
};

}  // namespace ext

// NOLINTEND(readability-identifier-naming)

namespace detail {

/**
 * @brief Maps an extension value type onto its C record. A specialisation provides
 * `using Native = anira_ext_<kind>;` and `static Native mint(const Ext&)`, where the record's
 * first member is its anira_ext_header. The minted record must outlive the C call it feeds;
 * set_ext copies, so a temporary suffices there. A kind that is read back
 * (ModelConfig::model_ext<Ext>(index)) also provides `static constexpr const char* k_kind`,
 * the kind's id, and `static Ext read(const anira_ext_header*)`, the value copied out of the typed
 * record a getter hands out (read within its struct_size; no pointer into it is kept).
 */
template <class Ext>
struct ExtTraits;

template <>
struct ExtTraits<ext::Entry> {
    using Native = anira_ext_entry;
    static constexpr const char* k_kind = "entry";

    static Native mint(const ext::Entry& entry) {
        Native native = ANIRA_EXT_ENTRY_INIT;
        native.name = entry.name.c_str();
        return native;
    }
    static ext::Entry read(const anira_ext_header* header) {
        Native native = ANIRA_EXT_ENTRY_INIT;
        std::memcpy(&native, header, std::min<std::size_t>(header->struct_size, sizeof(Native)));
        return ext::Entry{.name = native.name != nullptr ? native.name : ""};
    }
};

/// The C record of ext::ProviderOptions with the storage its pointers reach into: the sets,
/// their key and value arrays and copies of every string, so that the minted record stands on
/// its own for as long as it lives (the set call copies it; a job options handle keeps it).
struct ProviderOptionsNative : anira_ext_provider_options {
    std::vector<anira_provider_option_set> m_sets;
    std::vector<std::vector<const char*>> m_keys;
    std::vector<std::vector<const char*>> m_values;
    std::vector<std::string> m_strings;  ///< every string, at a stable address (reserved once)
};

template <>
struct ExtTraits<ext::ProviderOptions> {
    using Native = ProviderOptionsNative;

    static Native mint(const ext::ProviderOptions& value) {
        Native native;
        static_cast<anira_ext_provider_options&>(native) = ANIRA_EXT_PROVIDER_OPTIONS_INIT;
        std::size_t strings = 0;
        for (const ext::ProviderOptions::Set& set : value.sets) {
            strings += 2 + 2 * set.options.size();
        }
        native.m_strings.reserve(strings);  // never grown afterwards: the pointers stay put
        const auto keep = [&native](const std::string& text) -> const char* {
            if (text.empty()) { return nullptr; }
            native.m_strings.push_back(text);
            return native.m_strings.back().c_str();
        };
        native.m_sets.reserve(value.sets.size());
        native.m_keys.reserve(value.sets.size());
        native.m_values.reserve(value.sets.size());
        for (const ext::ProviderOptions::Set& set : value.sets) {
            std::vector<const char*> keys;
            std::vector<const char*> values;
            for (const auto& [key, option] : set.options) {
                keys.push_back(keep(key));
                values.push_back(keep(option));
            }
            native.m_keys.push_back(std::move(keys));
            native.m_values.push_back(std::move(values));
            anira_provider_option_set record = ANIRA_PROVIDER_OPTION_SET_INIT;
            record.engine = static_cast<uint32_t>(set.engine);
            record.provider = static_cast<uint32_t>(set.provider);
            record.num_options = static_cast<uint32_t>(set.options.size());
            record.engine_id = keep(set.engine_id);
            record.provider_id = keep(set.provider_id);
            record.keys = native.m_keys.back().empty() ? nullptr : native.m_keys.back().data();
            record.values =
                native.m_values.back().empty() ? nullptr : native.m_values.back().data();
            native.m_sets.push_back(record);
        }
        native.sets = native.m_sets.empty() ? nullptr : native.m_sets.data();
        native.num_sets = static_cast<uint32_t>(native.m_sets.size());
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

// ---- runtime tensors (section 1) -----------------------------------------------------------

namespace detail {

/// The rank the C factories take for a shape; a span longer than ANIRA_MAX_RANK maps onto a
/// rank they refuse, so no size is ever truncated into a legal one.
constexpr uint32_t rank_of(std::span<const int64_t> shape) noexcept {
    constexpr std::size_t k_max_rank = ANIRA_MAX_RANK;
    return static_cast<uint32_t>(shape.size() > k_max_rank ? k_max_rank + 1 : shape.size());
}

/// The plane count the C factory takes; a span too long for uint32_t maps onto a count no
/// shape[0] of a real tensor equals, so the factory refuses it.
constexpr uint32_t plane_count_of(std::size_t planes) noexcept {
    constexpr std::size_t k_max_count = 0xffffffffU;
    return static_cast<uint32_t>(planes > k_max_count ? k_max_count : planes);
}

/// The anira_dtype of a C++ element type, one lane: what the typed spellings
/// (Tensor::from_host_planar, Tensor::plane) derive the dtype from. A closed list: the
/// fixed-width integers, float, double and bool; const is ignored.
template <class T>
constexpr DType dtype_of() noexcept {
    using Element = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<Element, float>) {
        return ANIRA_DTYPE_F32;
    } else if constexpr (std::is_same_v<Element, double>) {
        return ANIRA_DTYPE_F64;
    } else if constexpr (std::is_same_v<Element, bool>) {
        static_assert(sizeof(bool) == 1, "ANIRA_DTYPE_BOOL8 is one byte");
        return ANIRA_DTYPE_BOOL8;
    } else if constexpr (std::is_same_v<Element, int8_t> || std::is_same_v<Element, int16_t> ||
                         std::is_same_v<Element, int32_t> || std::is_same_v<Element, int64_t>) {
        return ANIRA_MAKE_DTYPE(ANIRA_DTYPE_INT, sizeof(Element) * 8, 1);
    } else if constexpr (std::is_same_v<Element, uint8_t> || std::is_same_v<Element, uint16_t> ||
                         std::is_same_v<Element, uint32_t> || std::is_same_v<Element, uint64_t>) {
        return ANIRA_MAKE_DTYPE(ANIRA_DTYPE_UINT, sizeof(Element) * 8, 1);
    } else {
        static_assert(sizeof(Element) == 0,
                      "no anira_dtype for this element type: use the fixed-width integers, "
                      "float, double or bool, or the untyped C entries");
        return 0;
    }
}

}  // namespace detail

/**
 * @brief anira_sync_token with its two calls on it: the same 24-byte C struct, no member added,
 * so a SyncToken* is an anira_sync_token*.
 *
 * Not an RAII type: the token travels inside a trivially copyable anira_tensor and every
 * hand-off is a transfer, so reset() is explicit. Both calls are [thread-safe, !audio-thread]:
 * close and dup are system calls.
 */
struct SyncToken : anira_sync_token {
    /// Closes the fd of an owning kind and zeroes the token (anira_sync_token_reset).
    void reset() noexcept { anira_sync_token_reset(this); }

    /// A duplicate that outlives its source (anira_sync_token_dup): dup() for the two owning fd
    /// kinds, a plain copy otherwise. Throws anira::Error when the fd cannot be duplicated.
    SyncToken dup() const {
        SyncToken out{};
        detail::check(anira_sync_token_dup(this, &out), "anira_sync_token_dup");
        return out;
    }
};

/**
 * @brief anira_tensor with names and factories on it: the C struct itself (216 bytes, trivially
 * copyable, no member added), so a Tensor* is an anira_tensor*.
 *
 * The from_* factories are the anira_tensor_init_* field fills over a std::span shape:
 * [thread-safe] [callback-safe], real-time, noexcept. They refuse like the C entries: dtype 0,
 * more than ANIRA_MAX_RANK extents or a negative extent give the all-zero record (dtype 0).
 * Only from_dlpack can fail with a message and throws. The factories of the draft platform
 * arms (anira/abi/draft/tensor_platform.h) have no C++ spelling while they are unmeasured:
 * call anira_tensor_init_metal(&tensor, ...) and its siblings on a Tensor.
 */
struct Tensor : anira_tensor {
    /// anira_tensor_init_host: pageable host memory, borrowed.
    static Tensor from_host(void* data,
                            DType element_dtype,
                            std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_host(&tensor,
                               data,
                               element_dtype,
                               detail::rank_of(extents),
                               extents.data());
        return tensor;
    }

    /// anira_tensor_init_pinned: page-locked host memory, borrowed.
    static Tensor from_pinned(void* data,
                              DType element_dtype,
                              std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_pinned(&tensor,
                                 data,
                                 element_dtype,
                                 detail::rank_of(extents),
                                 extents.data());
        return tensor;
    }

    /// anira_tensor_init_host_planar: planar host memory, one pointer per plane of axis 0 (the
    /// channel pointers of an audio host), borrowed like the memory. The element type names the
    /// dtype and is spelled at the call, Tensor::from_host_planar<float>(channels, shape); the
    /// span's size must equal shape[0], else the all-zero record comes back, as in C. A const
    /// element type sets ANIRA_TENSOR_READ_ONLY. Only an entry that says so accepts a planar
    /// tensor.
    template <class T>
    static Tensor from_host_planar(std::span<T* const> planes,
                                   std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        // The one cast of the typed spelling: T* const* to the const void* the C entry takes.
        anira_tensor_init_host_planar(&tensor,
                                      static_cast<const void*>(planes.data()),
                                      detail::plane_count_of(planes.size()),
                                      detail::dtype_of<T>(),
                                      detail::rank_of(extents),
                                      extents.data());
        if constexpr (std::is_const_v<T>) {
            if (tensor.dtype != 0) {
                tensor.flags |= static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY);
            }
        }
        return tensor;
    }

    /// anira_tensor_init_cuda: a device pointer; event is a cudaEvent_t or nullptr (visible).
    static Tensor from_cuda(void* ptr,
                            int32_t device,
                            void* event,
                            DType element_dtype,
                            std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_cuda(&tensor,
                               ptr,
                               device,
                               event,
                               element_dtype,
                               detail::rank_of(extents),
                               extents.data());
        return tensor;
    }

    /// anira_tensor_init_gl_buffer: a GL buffer object; glsync is a GLsync or nullptr (visible).
    static Tensor from_gl_buffer(uint32_t id,
                                 uint32_t target,
                                 void* glsync,
                                 DType element_dtype,
                                 std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_gl_buffer(&tensor,
                                    id,
                                    target,
                                    glsync,
                                    element_dtype,
                                    detail::rank_of(extents),
                                    extents.data());
        return tensor;
    }

    /// anira_tensor_init_vulkan: VkBuffer, VkDeviceMemory and offset at their wire width; a
    /// timeline semaphore of 0 means visible.
    static Tensor from_vulkan(uint64_t buffer,
                              uint64_t memory,
                              uint64_t offset,
                              uint64_t timeline,
                              uint64_t value,
                              DType element_dtype,
                              std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_vulkan(&tensor,
                                 buffer,
                                 memory,
                                 offset,
                                 timeline,
                                 value,
                                 element_dtype,
                                 detail::rank_of(extents),
                                 extents.data());
        return tensor;
    }

    /// anira_tensor_init_opaque_fd: exported opaque memory; no fence.
    static Tensor from_opaque_fd(int32_t fd,
                                 uint64_t size,
                                 DType element_dtype,
                                 std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_opaque_fd(&tensor,
                                    fd,
                                    size,
                                    element_dtype,
                                    detail::rank_of(extents),
                                    extents.data());
        return tensor;
    }

    /// anira_tensor_init_wgpu_buffer: the fence is copied into acquire and that copy is the
    /// hand-off (the caller does not reset its source); nullptr means visible.
    static Tensor from_wgpu_buffer(void* buffer,
                                   uint64_t offset,
                                   const SyncToken* fence,
                                   DType element_dtype,
                                   std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_wgpu_buffer(&tensor,
                                      buffer,
                                      offset,
                                      fence,
                                      element_dtype,
                                      detail::rank_of(extents),
                                      extents.data());
        return tensor;
    }

    /// anira_tensor_init_dmabuf: exported buffer memory; a sync_fd of 0 or more is owned by the
    /// tensor's acquire token from here on, a negative one means visible.
    static Tensor from_dmabuf(int32_t fd,
                              uint64_t size,
                              uint64_t offset,
                              int32_t sync_fd,
                              DType element_dtype,
                              std::span<const int64_t> extents) noexcept {
        Tensor tensor{};
        anira_tensor_init_dmabuf(&tensor,
                                 fd,
                                 size,
                                 offset,
                                 sync_fd,
                                 element_dtype,
                                 detail::rank_of(extents),
                                 extents.data());
        return tensor;
    }

    /// anira_tensor_init_dlpack over a DLManagedTensorVersioned* of DLPack major 1 ([main-thread]).
    /// On success the managed tensor belongs to the record and release calls its deleter once;
    /// on failure this throws anira::Error and the caller keeps the managed tensor.
    static Tensor from_dlpack(void* dl_managed_tensor_versioned) {
        Tensor tensor{};
        anira_error err{};
        detail::check(anira_tensor_init_dlpack(&tensor, dl_managed_tensor_versioned, &err), err);
        return tensor;
    }

    /// anira_tensor_data_f32: the first element of a one-block host float32 tensor, nullptr
    /// otherwise (a device tensor, another dtype, a planar tensor).
    float* data_f32() const noexcept { return anira_tensor_data_f32(this); }

    /// anira_tensor_data: the first element of a host tensor whose dtype equals the argument,
    /// nullptr otherwise; never converts.
    void* data(DType element_dtype) const noexcept {
        return anira_tensor_data(this, element_dtype);
    }

    /// anira_tensor_plane: the first element of one plane of a planar host tensor whose dtype is
    /// T's, nullptr otherwise (a one-block tensor, a plane out of range, another dtype).
    template <class T>
    T* plane(uint32_t index) const noexcept {
        return static_cast<T*>(anira_tensor_plane(this, index, detail::dtype_of<T>()));
    }

    /// anira_tensor_num_elements: the product of the extents, 1 at rank 0, 0 for the all-zero
    /// record of a refused factory.
    std::size_t num_elements() const noexcept { return anira_tensor_num_elements(this); }

    /// anira_tensor_extent: one extent, 0 for an axis at or above ndim.
    std::size_t extent(uint32_t axis) const noexcept { return anira_tensor_extent(this, axis); }
};

static_assert(sizeof(SyncToken) == sizeof(anira_sync_token) &&
              std::is_trivially_copyable_v<SyncToken> && std::is_standard_layout_v<SyncToken>);
static_assert(sizeof(Tensor) == sizeof(anira_tensor) && std::is_trivially_copyable_v<Tensor> &&
              std::is_standard_layout_v<Tensor>);

// ---- tensor spec (section 2) ---------------------------------------------------------------

/**
 * @brief A read-only view of a tensor spec over the anira_tensor_spec getters: what
 * ModelConfig::input_spec / output_spec hand out (the config's own copy, valid until the config
 * is mutated, moved or destroyed) and what TensorSpec::view() answers over a spec being built.
 * Every read is noexcept; an axis at or beyond ndim() reads {ANIRA_AXIS_ANY, 0}, an extent no
 * set axis has (the rule of Tensor::extent). Trivially copyable.
 */
class SpecView {
public:
    // NOLINTBEGIN(readability-identifier-naming) the aggregates spell the setters' parameters
    struct Axis {
        AxisTag tag = ANIRA_AXIS_ANY;
        int64_t extent = 0;  ///< ANIRA_DYNAMIC where it was set so
    };
    struct Window {
        int64_t min = 0;
        int64_t max = 0;  ///< ANIRA_UNBOUNDED where it was set so
        int64_t overlap = 0;
    };
    struct TimeRatio {
        int64_t num = 0;  ///< (0, 0) = derive
        int64_t den = 0;
    };
    // NOLINTEND(readability-identifier-naming)

    explicit SpecView(const anira_tensor_spec* spec) noexcept : m_spec(spec) {}

    /// The canonical name, owned by the spec.
    std::string_view name() const noexcept {
        const char* text = anira_tensor_spec_name(m_spec);
        return text != nullptr ? std::string_view(text) : std::string_view();
    }
    DType dtype() const noexcept { return anira_tensor_spec_dtype(m_spec); }
    Role role() const noexcept { return anira_tensor_spec_role(m_spec); }
    /// One more than the highest axis set; a hole below it reads {ANIRA_AXIS_ANY, 0}.
    uint32_t ndim() const noexcept { return anira_tensor_spec_ndim(m_spec); }
    Axis axis(uint32_t i) const noexcept {
        Axis out;
        if (anira_tensor_spec_axis(m_spec, i, &out.tag, &out.extent) != ANIRA_OK) { return {}; }
        return out;
    }
    /// 0, 0, 0 until TensorSpec::window ran.
    Window window() const noexcept {
        Window out;
        if (anira_tensor_spec_window(m_spec, &out.min, &out.max, &out.overlap) != ANIRA_OK) {
            return {};
        }
        return out;
    }
    TimeRatio time_ratio() const noexcept {
        TimeRatio out;
        if (anira_tensor_spec_time_ratio(m_spec, &out.num, &out.den) != ANIRA_OK) { return {}; }
        return out;
    }
    /// The model's internal latency along the Time axis.
    int64_t latency() const noexcept { return anira_tensor_spec_latency(m_spec); }
    /// A State input's source, the canonical name of its State output; empty without one.
    std::string_view state_source() const noexcept {
        const char* text = anira_tensor_spec_state_source(m_spec);
        return text != nullptr ? std::string_view(text) : std::string_view();
    }
    const anira_tensor_spec* native() const noexcept { return m_spec; }

private:
    const anira_tensor_spec* m_spec = nullptr;
};

/**
 * @brief One input or output of the model: your canonical name, the data type, the role, the
 * tagged axes in the model's memory order, and, for a streamed tensor, the window and overlap
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
    /// Streamed only: elements along the Time axis per inference and the overlap kept.
    TensorSpec& window(int64_t window_min, int64_t window_max, int64_t overlap) {
        detail::check(anira_tensor_spec_set_window(m_spec, window_min, window_max, overlap),
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
    /// State inputs only (role ANIRA_ROLE_STATE): the canonical name of the state output this
    /// input is fed from on the next inference (anira_tensor_spec_set_state_source). Both
    /// halves of the pair carry the role, the pairing is stated here, once; the name is
    /// resolved when the model is validated. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for
    /// an empty name or a spec of another role.
    TensorSpec& state_source(std::string_view output_canonical) {
        detail::check(
            anira_tensor_spec_set_state_source(m_spec, std::string(output_canonical).c_str()),
            "anira_tensor_spec_set_state_source");
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

    /// What the spec holds so far, through the one read path of SpecView; valid while this
    /// object lives.
    SpecView view() const noexcept { return SpecView(m_spec); }

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
    /// ANIRA_MISS_CALLBACK only: the backup function that fills a missed block, and the pointer
    /// it is handed. A raw function pointer, called on the driver thread; it and what
    /// miss_user_data points at must outlive the prepared handler. Under clang the function
    /// must be declared ANIRA_NONBLOCKING.
    anira_miss_fn miss_fn = nullptr;
    void* miss_user_data = nullptr;
    double wait_ratio = 0;  ///< v2 blocking_ratio
    /// The ring dtype per Streamed tensor by canonical name (anira_contract_hard_set_ring_dtype;
    /// ANIRA_DTYPE_F32 for every tensor not named): the element type of the host's samples.
    std::map<std::string, DType> ring_dtypes{};
    /// The declared stream latency per Streamed output by canonical name, in samples of that
    /// output (anira_contract_hard_set_latency): it replaces the computed figure and primes the
    /// receive ring; every output not named keeps the computed one.
    std::map<std::string, uint32_t> latencies{};
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
        check(anira_contract_hard_set_miss_fn(contract, hard.miss_fn, hard.miss_user_data),
              "anira_contract_hard_set_miss_fn");
        check(anira_contract_hard_set_wait_ratio(contract, hard.wait_ratio),
              "anira_contract_hard_set_wait_ratio");
        for (const auto& [name, dtype] : hard.ring_dtypes) {
            check(anira_contract_hard_set_ring_dtype(contract, name.c_str(), dtype),
                  "anira_contract_hard_set_ring_dtype");
        }
        for (const auto& [name, samples] : hard.latencies) {
            check(anira_contract_hard_set_latency(contract, name.c_str(), samples),
                  "anira_contract_hard_set_latency");
        }
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
    /// type the tensor forms of the Hard entries carry across the ABI and the ring holds as is;
    /// F32 for every tensor never set. Nothing
    /// converts: a name that is not a Streamed tensor is ANIRA_ERROR_CONFIG at
    /// anira_handler_prepare, and so is a ring dtype that differs from the spec's dtype unless a
    /// stage of the pipeline fills the phase that moves that ring (anira_pipeline_add_stage).
    ContractHandle& hard_ring_dtype(std::string_view canonical, DType dtype) {
        const std::string name(canonical);
        detail::check(anira_contract_hard_set_ring_dtype(m_contract, name.c_str(), dtype),
                      "anira_contract_hard_set_ring_dtype");
        return *this;
    }
    /// The declared stream latency of one output under this Hard contract, by canonical name,
    /// in samples of that output (anira_contract_hard_set_latency): what the handler reports
    /// for the slot and primes its receive ring with, replacing the computed figure. A name
    /// that is no Streamed output, or a figure below the model's internal latency, is
    /// ANIRA_ERROR_CONFIG at anira_handler_prepare. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT}
    /// for an empty name or a figure above INT32_MAX.
    ContractHandle& hard_latency(std::string_view canonical, uint32_t samples) {
        const std::string name(canonical);
        detail::check(anira_contract_hard_set_latency(m_contract, name.c_str(), samples),
                      "anira_contract_hard_set_latency");
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
    /// The backup function of ANIRA_MISS_CALLBACK (NULL clears it); see Hard::miss_fn.
    ContractHandle& hard_miss_fn(anira_miss_fn fn, void* user_data) {
        detail::check(anira_contract_hard_set_miss_fn(m_contract, fn, user_data),
                      "anira_contract_hard_set_miss_fn");
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
    /// The host-end domain of one tensor, by canonical name, on either contract kind
    /// (anira_contract_set_host_domain): the domain anira allocates the ring, the model
    /// tensor, the Static store and the state buffers of the slot in, and the domain every
    /// stage phase works in; ANIRA_DOMAIN_HOST for every tensor never set. Any tensor of
    /// either side, State and Static included. A name that is no tensor is ANIRA_ERROR_CONFIG
    /// at anira_handler_prepare, and in this pre-release any domain but ANIRA_DOMAIN_HOST is
    /// ANIRA_ERROR_NOT_SUPPORTED there. In a contract file: the top-level "host_domains".
    ContractHandle& host_domain(std::string_view canonical, Domain domain) {
        const std::string name(canonical);
        detail::check(anira_contract_set_host_domain(m_contract, name.c_str(), domain),
                      "anira_contract_set_host_domain");
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

    /// The Hard aggregate this handle holds, a loaded or legacy one alike: every field the
    /// read-back of anira/abi/config.h answers, ring_dtypes and latencies from their
    /// enumerations, and budget_value rounded to the nanosecond (the handle stores double
    /// milliseconds; std::chrono::round, so 42.66 ms reads 42660000 ns). Minting the result
    /// gives an equal contract. @throws Error{ANIRA_ERROR_WRONG_CONTRACT} on an Async handle,
    /// Error{ANIRA_ERROR_INVALID_ARGUMENT} on an empty one.
    Hard hard() const {
        if (m_contract == nullptr) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT, "anira_contract_hard_geometry");
        }
        Hard out;
        detail::check(
            anira_contract_hard_geometry(m_contract, &out.block_min, &out.block_max, &out.rate),
            "anira_contract_hard_geometry");
        double budget_ms = 0.0;
        detail::check(anira_contract_hard_budget(m_contract, &out.budget, &budget_ms),
                      "anira_contract_hard_budget");
        out.budget_value = std::chrono::round<std::chrono::nanoseconds>(
            std::chrono::duration<double, std::milli>(budget_ms));
        detail::check(anira_contract_hard_warmup(m_contract, &out.warmup, &out.warmup_iterations),
                      "anira_contract_hard_warmup");
        detail::check(anira_contract_hard_on_miss(m_contract, &out.on_miss),
                      "anira_contract_hard_on_miss");
        detail::check(anira_contract_hard_miss_fn(m_contract, &out.miss_fn, &out.miss_user_data),
                      "anira_contract_hard_miss_fn");
        detail::check(anira_contract_hard_wait_ratio(m_contract, &out.wait_ratio),
                      "anira_contract_hard_wait_ratio");
        const uint32_t num_ring_dtypes = anira_contract_hard_num_ring_dtypes(m_contract);
        for (uint32_t i = 0; i < num_ring_dtypes; ++i) {
            const char* name = nullptr;
            DType dtype = 0;
            detail::check(anira_contract_hard_ring_dtype(m_contract, i, &name, &dtype),
                          "anira_contract_hard_ring_dtype");
            out.ring_dtypes.emplace(name, dtype);
        }
        const uint32_t num_latencies = anira_contract_hard_num_latencies(m_contract);
        for (uint32_t i = 0; i < num_latencies; ++i) {
            const char* name = nullptr;
            uint32_t samples = 0;
            detail::check(anira_contract_hard_latency(m_contract, i, &name, &samples),
                          "anira_contract_hard_latency");
            out.latencies.emplace(name, samples);
        }
        out.edge_cost = anira_contract_edge_cost(m_contract);
        return out;
    }

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
 * @brief The model: one entry per export (an engine, and a provider when pinned; a file or
 * bytes) with what that export calls each tensor and how it lays out its axes, the input and
 * output specs, the default engine, the state, the instance ceiling and the anchor. Move-only.
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
    uint32_t add_model_path(EngineKind engine, const std::filesystem::path& path) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_path(m_config,
                                                        engine,
                                                        nullptr,
                                                        detail::utf8(path).c_str(),
                                                        &index,
                                                        &err),
                      err);
        return index;
    }
    /// A file for a custom engine registered under a reverse-URI name (ANIRA_ENGINE_CUSTOM with
    /// the id).
    uint32_t add_model_path(std::string_view engine_id, const std::filesystem::path& path) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_path(m_config,
                                                        ANIRA_ENGINE_CUSTOM,
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
    uint32_t add_model_bytes(EngineKind engine,
                             std::span<const std::byte> bytes,
                             anira_bytes_ownership ownership = ANIRA_BYTES_COPY,
                             anira_bytes_release_fn release = nullptr,
                             void* ctx = nullptr) {
        anira_error err{};
        uint32_t index = 0;
        detail::check(anira_model_config_add_model_bytes(m_config,
                                                         engine,
                                                         nullptr,
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
        detail::check(anira_model_config_add_model_bytes(m_config,
                                                         ANIRA_ENGINE_CUSTOM,
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
    /// The entry's engine as the pair (anira_model_config_model_engine): ANIRA_ENGINE_CUSTOM with
    /// the id for a custom one; {ANIRA_ENGINE_NONE, empty} for an index out of range. The id is
    /// owned by the config and valid until it is mutated, moved or destroyed.
    EngineRef model_engine(uint32_t index) const noexcept {
        return detail::engine_ref([&](anira_engine* engine, const char** id) {
            return anira_model_config_model_engine(m_config, index, engine, id);
        });
    }
    /// The provider the entry is pinned to as the pair (anira_model_config_model_provider):
    /// ANIRA_PROVIDER_CUSTOM with the name for a custom pin, ANIRA_PROVIDER_DEFAULT for an entry
    /// without a pin and for an index out of range. The name is owned by the config.
    ProviderRef model_provider(uint32_t index) const noexcept {
        return detail::provider_ref([&](anira_provider* provider, const char** id) {
            return anira_model_config_model_provider(m_config, index, provider, id);
        });
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
    /// Pins the entry to the provider its file is built for (an ExecuTorch export lowered for
    /// a provider (an ExecuTorch delegate), an ONNX Runtime .ort compiled for an execution
    /// provider): a provider of the enum here, a custom name through the string overload.
    /// Only a candidate naming that provider runs a pinned entry; an entry without a pin runs
    /// on any provider of its engine, the candidate deciding. Two entries of one engine may
    /// coexist when their pins differ. ANIRA_PROVIDER_DEFAULT unpins.
    ModelConfig& model_provider(uint32_t index, Provider provider) {
        anira_error err{};
        detail::check(
            anira_model_config_set_model_provider(m_config, index, provider, nullptr, &err),
            err);
        return *this;
    }
    /// Pins the entry to a custom provider, by its name in the engine's own vocabulary
    /// (ANIRA_PROVIDER_CUSTOM with the name).
    ModelConfig& model_provider(uint32_t index, std::string_view provider_id) {
        anira_error err{};
        detail::check(anira_model_config_set_model_provider(m_config,
                                                            index,
                                                            ANIRA_PROVIDER_CUSTOM,
                                                            std::string(provider_id).c_str(),
                                                            &err),
                      err);
        return *this;
    }
    /// What this entry's export calls the tensor you named canonical (binds it by name).
    ModelConfig& tensor_name(uint32_t index,
                             std::string_view canonical,
                             std::string_view export_name) {
        detail::check(anira_model_config_set_tensor_name(m_config,
                                                         index,
                                                         std::string(canonical).c_str(),
                                                         std::string(export_name).c_str()),
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

    // -- the read-back --

    /// The number of input specs, State specs included.
    uint32_t input_count() const noexcept { return anira_model_config_num_inputs(m_config); }
    uint32_t output_count() const noexcept { return anira_model_config_num_outputs(m_config); }
    /// The config's own copy of input spec i, valid until the config is mutated, moved or
    /// destroyed. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} out of range.
    SpecView input_spec(uint32_t i) const {
        const anira_tensor_spec* spec = anira_model_config_input(m_config, i);
        if (spec == nullptr) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT, "anira_model_config_input");
        }
        return SpecView(spec);
    }
    SpecView output_spec(uint32_t i) const {
        const anira_tensor_spec* spec = anira_model_config_output(m_config, i);
        if (spec == nullptr) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT, "anira_model_config_output");
        }
        return SpecView(spec);
    }
    /// The default engine as the pair (anira_model_config_default_engine): ANIRA_ENGINE_NONE
    /// for plan 0, ANIRA_ENGINE_CUSTOM with the id for a custom one.
    EngineRef default_engine() const noexcept {
        return detail::engine_ref([&](anira_engine* engine, const char** id) {
            return anira_model_config_default_engine(m_config, engine, id);
        });
    }
    /// The default provider as the pair (anira_model_config_default_provider):
    /// ANIRA_PROVIDER_DEFAULT for none, ANIRA_PROVIDER_CUSTOM with the name for a custom one.
    ProviderRef default_provider() const noexcept {
        return detail::provider_ref([&](anira_provider* provider, const char** id) {
            return anira_model_config_default_provider(m_config, provider, id);
        });
    }
    /// As set; a model with a declared State pair runs as ANIRA_MODEL_STATEFUL anyway.
    anira_model_state state() const noexcept { return anira_model_config_state(m_config); }
    uint32_t max_instances() const noexcept { return anira_model_config_max_instances(m_config); }
    /// The anchor's canonical name as set; empty for the default.
    std::string_view anchor() const noexcept {
        const char* name = anira_model_config_anchor(m_config);
        return name != nullptr ? std::string_view(name) : std::string_view();
    }
    /// What entry `index` calls the tensor; empty when it binds the tensor positionally.
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an index out of range.
    std::string_view tensor_name(uint32_t index, std::string_view canonical) const {
        check_entry(index, "anira_model_config_tensor_name");
        const char* name =
            anira_model_config_tensor_name(m_config, index, std::string(canonical).c_str());
        return name != nullptr ? std::string_view(name) : std::string_view();
    }
    /// The axis order entry `index` holds the tensor in; empty for the spec's order.
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an index out of range or an empty name.
    std::vector<uint32_t> tensor_layout(uint32_t index, std::string_view canonical) const {
        const std::string name(canonical);
        uint32_t count = 0;
        detail::check(
            anira_model_config_tensor_layout(m_config, index, name.c_str(), &count, nullptr),
            "anira_model_config_tensor_layout");
        std::vector<uint32_t> axes(count);
        if (count > 0) {
            detail::check(anira_model_config_tensor_layout(m_config,
                                                           index,
                                                           name.c_str(),
                                                           &count,
                                                           axes.data()),
                          "anira_model_config_tensor_layout");
        }
        return axes;
    }
    /// The extension of kind Ext on entry `index`, copied out (ext::Entry: the entry point);
    /// nullopt when the entry carries none. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an
    /// index out of range.
    template <class Ext>
    std::optional<Ext> model_ext(uint32_t index) const {
        check_entry(index, "anira_model_config_model_ext");
        const anira_ext_header* header =
            anira_model_config_model_ext(m_config, index, detail::ExtTraits<Ext>::k_kind);
        if (header == nullptr) { return std::nullopt; }
        return detail::ExtTraits<Ext>::read(header);
    }

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
    /// The engine the handler starts on (anira_model_config_set_default_engine): a built-in
    /// engine, or ANIRA_ENGINE_NONE for plan 0.
    ModelConfig& default_engine(EngineKind engine) {
        anira_error err{};
        detail::check(anira_model_config_set_default_engine(m_config, engine, nullptr, &err), err);
        return *this;
    }
    /// A custom engine the handler starts on, by its id (ANIRA_ENGINE_CUSTOM with the id).
    ModelConfig& default_engine(std::string_view engine_id) {
        anira_error err{};
        detail::check(anira_model_config_set_default_engine(m_config,
                                                            ANIRA_ENGINE_CUSTOM,
                                                            std::string(engine_id).c_str(),
                                                            &err),
                      err);
        return *this;
    }
    /// The provider the handler starts on beside the default engine: the first plan of the
    /// default engine (of any engine without one) on this provider. The default engine's rule
    /// holds: ANIRA_ERROR_CONFIG at create when no entry could run on it (each pinned to another
    /// provider); a plan table without such a plan starts as without it, with one Warning at
    /// prepare (anira_model_config_set_default_provider). A provider of the enum here, a
    /// custom name through the string overload; ANIRA_PROVIDER_DEFAULT sets none.
    ModelConfig& default_provider(Provider provider) {
        detail::check(anira_model_config_set_default_provider(m_config, provider, nullptr),
                      "anira_model_config_set_default_provider");
        return *this;
    }
    /// A custom provider the handler starts on, by its name in the engine's own vocabulary
    /// (ANIRA_PROVIDER_CUSTOM with the name).
    ModelConfig& default_provider(std::string_view provider_id) {
        detail::check(anira_model_config_set_default_provider(m_config,
                                                              ANIRA_PROVIDER_CUSTOM,
                                                              std::string(provider_id).c_str()),
                      "anira_model_config_set_default_provider");
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

    void check_entry(uint32_t index, const char* entry) const {
        if (index >= model_count()) { throw Error(ANIRA_ERROR_INVALID_ARGUMENT, entry); }
    }

    anira_model_config* m_config = nullptr;
    bool m_upgraded = false;
};

// ---- context config (section 4) ------------------------------------------------------------

/**
 * @brief The process: the inference thread pool, logging, the devices anira may use.
 * Move-only.
 */
class ContextConfig {
public:
    /// @throws Error when the C entry refuses the call; the status says why.
    ContextConfig() {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_context_config_create(&m_config, &err), err);
    }
    /// A context file (section 8.2); a 2.x document's context_config root is upgraded.
    static ContextConfig from_json(std::string_view utf8) {
        detail::abi_check_once();
        anira_error err{};
        anira_context_config* config = nullptr;
        const anira_status status =
            anira_context_config_from_json(detail::text_of(utf8), utf8.size(), &config, &err);
        detail::check(status, err);
        return {config, status == ANIRA_SUCCESS_UPGRADED};
    }
    static ContextConfig from_file(const std::filesystem::path& path) {
        return from_json(detail::read_text(path));
    }

    ~ContextConfig() { anira_context_config_destroy(m_config); }
    ContextConfig(const ContextConfig&) = delete;
    ContextConfig& operator=(const ContextConfig&) = delete;
    ContextConfig(ContextConfig&& other) noexcept
        : m_config(std::exchange(other.m_config, nullptr)), m_upgraded(other.m_upgraded) {}
    ContextConfig& operator=(ContextConfig&& other) noexcept {
        if (this != &other) {
            anira_context_config_destroy(m_config);
            m_config = std::exchange(other.m_config, nullptr);
            m_upgraded = other.m_upgraded;
        }
        return *this;
    }

    /// The thread pool: ANIRA_THREADS_AUTO sizes it, 0 means the host brings its threads.
    ContextConfig& threads(uint32_t num_threads,
                           anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF) {
        detail::check(anira_context_config_set_threads(m_config, num_threads, wait),
                      "anira_context_config_set_threads");
        return *this;
    }
    ContextConfig& log_level(anira_log_level level) {
        detail::check(anira_context_config_set_log_level(m_config, level),
                      "anira_context_config_set_log_level");
        return *this;
    }
    ContextConfig& log_drain(anira_log_drain drain, uint32_t interval_ms = 10) {
        detail::check(anira_context_config_set_log_drain(m_config, drain, interval_ms),
                      "anira_context_config_set_log_drain");
        return *this;
    }
    ContextConfig& log_queue_capacity(uint32_t capacity) {
        detail::check(anira_context_config_set_log_queue_capacity(m_config, capacity),
                      "anira_context_config_set_log_queue_capacity");
        return *this;
    }
    ContextConfig& log_flags(uint32_t flags) {
        detail::check(anira_context_config_set_log_flags(m_config, flags),
                      "anira_context_config_set_log_flags");
        return *this;
    }
    /// The sink callback and its user data (the raw pair at this pre-release).
    ContextConfig& log_sink(anira_log_fn callback, void* user_data = nullptr) {
        detail::check(anira_context_config_set_log_sink(m_config, callback, user_data),
                      "anira_context_config_set_log_sink");
        return *this;
    }
    /// The one-shot descriptor equal to the five scalar log setters.
    ContextConfig& log(const anira_log_desc& desc) {
        detail::check(anira_context_config_set_log(m_config, &desc),
                      "anira_context_config_set_log");
        return *this;
    }
    ContextConfig& cuda(const anira_cuda_desc& desc) { return cuda(&desc); }
    /// The pointer form: NULL clears the block.
    ContextConfig& cuda(const anira_cuda_desc* desc) {
        detail::check(anira_context_config_set_cuda(m_config, desc),
                      "anira_context_config_set_cuda");
        return *this;
    }
    ContextConfig& gl(const anira_gl_desc& desc) { return gl(&desc); }
    /// The pointer form: NULL clears the block.
    ContextConfig& gl(const anira_gl_desc* desc) {
        detail::check(anira_context_config_set_gl(m_config, desc), "anira_context_config_set_gl");
        return *this;
    }
    ContextConfig& vulkan(const anira_vulkan_desc& desc) { return vulkan(&desc); }
    /// The pointer form: NULL clears the block.
    ContextConfig& vulkan(const anira_vulkan_desc* desc) {
        detail::check(anira_context_config_set_vulkan(m_config, desc),
                      "anira_context_config_set_vulkan");
        return *this;
    }
    ContextConfig& metal(const anira_metal_desc& desc) { return metal(&desc); }
    /// The pointer form: NULL clears the block.
    ContextConfig& metal(const anira_metal_desc* desc) {
        detail::check(anira_context_config_set_metal(m_config, desc),
                      "anira_context_config_set_metal");
        return *this;
    }
    ContextConfig& d3d12(const anira_d3d12_desc& desc) { return d3d12(&desc); }
    /// The pointer form: NULL clears the block.
    ContextConfig& d3d12(const anira_d3d12_desc* desc) {
        detail::check(anira_context_config_set_d3d12(m_config, desc),
                      "anira_context_config_set_d3d12");
        return *this;
    }
    ContextConfig& webgpu(const anira_webgpu_desc& desc) { return webgpu(&desc); }
    /// The pointer form: NULL clears the block.
    ContextConfig& webgpu(const anira_webgpu_desc* desc) {
        detail::check(anira_context_config_set_webgpu(m_config, desc),
                      "anira_context_config_set_webgpu");
        return *this;
    }
    template <class Ext>
    ContextConfig& ext(const Ext& value) {
        const auto native = detail::ExtTraits<Ext>::mint(value);
        anira_error err{};
        detail::check(anira_context_config_set_ext(m_config, &native.header, &err), err);
        return *this;
    }
    ContextConfig& ext_json(std::string_view kind, std::string_view utf8) {
        anira_error err{};
        detail::check(anira_context_config_set_ext_json(m_config,
                                                        std::string(kind).c_str(),
                                                        detail::text_of(utf8),
                                                        utf8.size(),
                                                        &err),
                      err);
        return *this;
    }
    std::string to_json() const {
        return detail::write_json("anira_context_config_to_json",
                                  [this](char* buf, std::size_t cap, std::size_t* len) {
                                      return anira_context_config_to_json(m_config, buf, cap, len);
                                  });
    }
    bool upgraded() const noexcept { return m_upgraded; }

    const anira_context_config* native() const noexcept { return m_config; }
    anira_context_config* native() noexcept { return m_config; }

private:
    ContextConfig(anira_context_config* config, bool upgraded) noexcept
        : m_config(config), m_upgraded(upgraded) {}

    anira_context_config* m_config = nullptr;
    bool m_upgraded = false;
};

// ---- context (section 4) -------------------------------------------------------------------

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
 * @brief A view over a context's probed capabilities (anira_capabilities): what backends are
 * usable here, which memory domains a tensor may live in, the extension kinds this build
 * understands, and the edge registry. Valid while the Context is; refreshed in place by
 * Context::probe.
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
                return anira_capabilities_edges(m_capabilities,
                                                sizeof(anira_edge_info),
                                                count,
                                                out);
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
 * @brief An anira_context with its lifetime: a refcounted handle over this copy's core,
 * created from a ContextConfig (section 4). Two contexts in one copy are two views of one
 * core with two log sinks.
 */
class Context {
public:
    /// @throws Error when the C entry refuses the config (a device block, an unconsumed
    /// context extension).
    explicit Context(const ContextConfig& config) {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_context_create(config.native(), &m_context, &err), err);
    }
    ~Context() { anira_context_destroy(m_context); }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&& other) noexcept : m_context(std::exchange(other.m_context, nullptr)) {}
    Context& operator=(Context&& other) noexcept {
        if (this != &other) {
            anira_context_destroy(m_context);
            m_context = std::exchange(other.m_context, nullptr);
        }
        return *this;
    }

    Capabilities capabilities() const {
        return Capabilities(anira_context_capabilities(m_context));
    }
    /// Re-runs the probe; `force` re-runs every rung even where a cached answer exists.
    void probe(bool force = false) {
        anira_error err{};
        detail::check(anira_context_probe(m_context, force ? 1U : 0U, &err), err);
    }
    /// The size of a tensor's byte image under the edges this context probed.
    uint64_t byte_image_bytes(uint64_t num_elements, DType dtype) const {
        return anira_context_byte_image_bytes(m_context, num_elements, dtype);
    }

    const anira_context* native() const noexcept { return m_context; }
    anira_context* native() noexcept { return m_context; }

private:
    anira_context* m_context = nullptr;
};

/// What this build compiled in, without a context: one row per engine on the default provider
/// (anira_enabled_engines).
inline std::vector<BackendId> enabled_engines() {
    return detail::enumerate<BackendId>(
        [](uint32_t* count, BackendId* out) {
            return anira_enabled_engines(sizeof(BackendId), count, out);
        },
        "anira_enabled_engines");
}

/// The steady clock of anira_now_ms / anira_now_ns, for deadlines and submit timestamps.
inline double now_ms() noexcept {
    return anira_now_ms();
}
inline uint64_t now_ns() noexcept {
    return anira_now_ns();
}

/// anira_drain_log: delivers the queued real-time records of the core to the sinks, the
/// host's pump under ANIRA_LOG_DRAIN_MANUAL.
inline std::size_t drain_log() noexcept {
    return anira_drain_log();
}
/// anira_num_inference_threads: the size of the core's inference thread pool.
inline uint32_t num_inference_threads() noexcept {
    return anira_num_inference_threads();
}

/// anira_shutdown: effective only when no Context and no handler exist in this copy.
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

// ---- plan report (section 6) ---------------------------------------------------------------

/**
 * @brief A view of a handler's anira_plan_report (valid until its next prepare or destroy):
 * the rows copied into vectors through the stride-explicit enumerators. Plan 0 is the only
 * plan of a single-candidate pipeline; the budget is the row field anira_plan_info::budget_ms.
 */
class PlanReport {
public:
    /// Wraps a handler's report; a null report has no plans and its enumerators throw
    /// ANIRA_ERROR_INVALID_ARGUMENT.
    explicit PlanReport(const anira_plan_report* report) noexcept : m_report(report) {}

    /// The number of plans; a plan is a dense index below it.
    uint32_t num_plans() const noexcept { return anira_plan_report_num_plans(m_report); }
    /// One anira_plan_info per plan, in plan order.
    std::vector<anira_plan_info> plans() const {
        return detail::enumerate<anira_plan_info>(
            [this](uint32_t* count, anira_plan_info* out) {
                return anira_plan_report_plans(m_report, sizeof(anira_plan_info), count, out);
            },
            "anira_plan_report_plans");
    }
    /// The input (inputs == true) or output slots of one plan, in tensor order.
    std::vector<anira_plan_slot> slots(uint32_t plan, bool inputs) const {
        return detail::enumerate<anira_plan_slot>(
            [this, plan, inputs](uint32_t* count, anira_plan_slot* out) {
                return anira_plan_report_slots(m_report,
                                               plan,
                                               inputs ? 1U : 0U,
                                               sizeof(anira_plan_slot),
                                               count,
                                               out);
            },
            "anira_plan_report_slots");
    }
    /// The extensions one plan consumes.
    std::vector<anira_plan_ext> extensions(uint32_t plan) const {
        return detail::enumerate<anira_plan_ext>(
            [this, plan](uint32_t* count, anira_plan_ext* out) {
                return anira_plan_report_exts(m_report, plan, sizeof(anira_plan_ext), count, out);
            },
            "anira_plan_report_exts");
    }

    const anira_plan_report* native() const noexcept { return m_report; }

private:
    const anira_plan_report* m_report;
};

// ---- stages (section 7) --------------------------------------------------------------------

/**
 * @brief A non-owning view of an anira_ring, the ring of a Streamed tensor as a phase callback
 * of a Stage gets it from StageContext::input_ring / output_ring: the anira_ring accessors of
 * anira/abi/stage.h, one C call per method, the element type T mapped onto its anira_dtype at
 * compile time (float, double, bool and the fixed-width integers; anything else is a compiler
 * error).
 *
 * Nothing converts: T's dtype must be dtype(). A call with another one moves nothing, returns 0
 * and records ANIRA_ERROR_CONFIG in the handler's anira_handler_rt_error with one latched
 * record, as the C entry does. A view of no ring (what a refused input_ring / output_ring leaves)
 * is false, and every call on it returns 0. The view is valid until the callback returns.
 *
 * [driver-thread] [callback-safe], nonblocking: every method is noexcept, allocates nothing and
 * returns the count the C entry returns.
 */
class RingView {
public:
    /// A view of no ring.
    RingView() noexcept = default;
    /// Wraps a ring of the stage context; NULL is a view of no ring.
    explicit RingView(anira_ring* ring) noexcept : m_ring(ring) {}

    /// Whether the slot has a ring in this phase.
    explicit operator bool() const noexcept { return m_ring != nullptr; }

    /// anira_ring_dtype: the element type the ring stores, the ring dtype of the Hard contract
    /// (ANIRA_DTYPE_F32 when none was declared); 0 for a view of no ring.
    DType dtype() const noexcept { return anira_ring_dtype(m_ring); }
    /// anira_ring_num_channels: the extent of the tensor's Channel axis, 1 without one.
    uint32_t num_channels() const noexcept { return anira_ring_num_channels(m_ring); }
    /// anira_ring_available: the unread elements of one channel.
    std::size_t available(uint32_t channel) const noexcept {
        return anira_ring_available(m_ring, channel);
    }
    /// anira_ring_available_past: the consumed elements of one channel the ring still holds as
    /// history, what peek_past_block can address.
    std::size_t available_past(uint32_t channel) const noexcept {
        return anira_ring_available_past(m_ring, channel);
    }

    /// anira_ring_pop_block: pops out.size() elements of one channel, oldest first; elements
    /// beyond the available ones are zero. @return out.size(), or 0 with nothing popped.
    template <class T>
    std::size_t pop_block(uint32_t channel, std::span<T> out) noexcept {
        static_assert(!std::is_const_v<T>, "pop_block writes its span");
        return anira_ring_pop_block(m_ring, channel, out.data(), detail::dtype_of<T>(), out.size());
    }
    /// anira_ring_peek_past_block: copies the out.size() most recently consumed elements of one
    /// channel without popping, oldest first (the receptive field of a window longer than its
    /// hop). @return out.size(), or 0 with nothing written.
    template <class T>
    std::size_t peek_past_block(uint32_t channel, std::span<T> out) const noexcept {
        static_assert(!std::is_const_v<T>, "peek_past_block writes its span");
        return anira_ring_peek_past_block(m_ring,
                                          channel,
                                          out.data(),
                                          detail::dtype_of<T>(),
                                          out.size());
    }
    /// anira_ring_push_block: pushes in.size() elements into one channel. @return in.size(), or
    /// 0 with nothing pushed.
    template <class T>
    std::size_t push_block(uint32_t channel, std::span<T> in) noexcept {
        return anira_ring_push_block(m_ring, channel, in.data(), detail::dtype_of<T>(), in.size());
    }
    /// anira_ring_push_fill: pushes count copies of one element into one channel. @return
    /// count, or 0 with nothing pushed.
    template <class T>
    std::size_t push_fill(uint32_t channel, const T& value, std::size_t count) noexcept {
        return anira_ring_push_fill(m_ring, channel, &value, detail::dtype_of<T>(), count);
    }
    /// anira_ring_discard: drops up to count unread elements of one channel (they become
    /// history); no element crosses the call, so it takes no element type. @return the
    /// elements dropped.
    std::size_t discard(uint32_t channel, std::size_t count) noexcept {
        return anira_ring_discard(m_ring, channel, count);
    }
    /// anira_ring_pop_windows: pops num_batches overlapping windows of one channel, window b
    /// at out[offset + b * (num_new + num_old)], each num_old elements of history followed by
    /// num_new freshly popped ones. The span is the whole destination: one too short for
    /// offset + num_batches * (num_new + num_old) elements is refused here (0, nothing popped,
    /// nothing recorded). @return the elements written, or 0.
    template <class T>
    std::size_t pop_windows(uint32_t channel,
                            std::span<T> out,
                            std::size_t num_new,
                            std::size_t num_old,
                            std::size_t offset,
                            uint32_t num_batches) noexcept {
        static_assert(!std::is_const_v<T>, "pop_windows writes its span");
        const std::size_t needed =
            offset + static_cast<std::size_t>(num_batches) * (num_new + num_old);
        if (out.size() < needed) { return 0; }
        return anira_ring_pop_windows(m_ring,
                                      channel,
                                      out.data(),
                                      detail::dtype_of<T>(),
                                      num_new,
                                      num_old,
                                      offset,
                                      num_batches);
    }

    anira_ring* native() const noexcept { return m_ring; }

private:
    anira_ring* m_ring = nullptr;
};

/**
 * @brief What a phase callback of a Stage sees: a view of the anira_stage_ctx anira fills on
 * its own stack for the call, with the six context accessors of anira/abi/stage.h on it, one C
 * call per method. The tensors and the rings are not in the record: a stage asks per slot, the
 * tensor's position in the model configuration's input or output list (the number every entry
 * of anira_handler uses). Valid until the callback returns, and never kept beyond it.
 *
 * What a phase exposes: pre_process the input rings and the model's input tensors (a State
 * input has no host end there), before_inference the input tensors, after_inference the output
 * tensors, post_process the output tensors (State outputs excepted) and the output rings. Every
 * accessor returns a status and fills an out-parameter, reset on any status but ANIRA_OK. A
 * slot always has a role; asking for a ring or a tensor the slot does not have in this phase is
 * a bug in the stage, not an answer: a stage knows what a slot is, from its model config or by
 * asking the role first, so the accessor answers ANIRA_ERROR_INVALID_STATE and records it in
 * anira_handler_rt_error with one latched record per kind naming the entry, the slot and the
 * phase, as it records a slot out of range (ANIRA_ERROR_INVALID_ARGUMENT).
 *
 * [driver-thread | inference-thread] [callback-safe], nonblocking: every method is noexcept and
 * allocates nothing, one C call per method.
 */
class StageContext {
public:
    /// Wraps the record a phase callback received; never NULL (anira hands out none).
    explicit StageContext(const anira_stage_ctx* ctx) noexcept : m_ctx(ctx) {}

    /// The phase this call runs in.
    anira_phase phase() const noexcept { return static_cast<anira_phase>(m_ctx->phase); }
    /// anira_stage_engine: the engine of the plan the chunk was submitted under, as the pair
    /// (ANIRA_ENGINE_CUSTOM with the id for a custom one). Valid for the duration of the call.
    EngineRef engine() const noexcept {
        return detail::engine_ref([this](anira_engine* engine, const char** id) {
            return anira_stage_engine(m_ctx, engine, id);
        });
    }
    /// anira_stage_provider: the provider of that plan, as the pair (ANIRA_PROVIDER_CUSTOM
    /// with the name for a custom one).
    ProviderRef provider() const noexcept {
        return detail::provider_ref([this](anira_provider* provider, const char** id) {
            return anira_stage_provider(m_ctx, provider, id);
        });
    }
    /// The variant of that plan; 0 in this pre-release.
    uint32_t variant() const noexcept { return m_ctx->variant; }
    /// The tensors of the model's input list, State tensors included: the input slots.
    uint32_t num_inputs() const noexcept { return m_ctx->num_inputs; }
    /// The tensors of the model's output list, State tensors included: the output slots.
    uint32_t num_outputs() const noexcept { return m_ctx->num_outputs; }
    /// The ticket of the submitting job under an Async contract; ANIRA_TICKET_INVALID under a
    /// Hard one.
    anira_ticket ticket() const noexcept { return m_ctx->ticket; }
    /// The entry this chunk occupies, 0 .. anira_handler_num_entries() - 1: the same value in
    /// all four phases of the chunk, reused by a later chunk once this one has completed. The
    /// index of a per-chunk scratch a stage sized in its prepare (one slot per entry), so that
    /// what pre_process pops or computes is found again by before_inference, without an
    /// allocation: one entry holds one chunk at a time, however many are in flight.
    uint32_t entry() const noexcept { return m_ctx->entry; }

    /// anira_stage_input_role: fills out with what the tensor of an input slot is, the same
    /// answer in every phase; the first question about a slot the stage does not know from its
    /// model config. @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a slot at or beyond
    /// num_inputs(), recorded, with out ANIRA_ROLE_FORCE32, which is no role. A slot always has
    /// a role: never ANIRA_ERROR_INVALID_STATE.
    anira_status input_role(uint32_t slot, Role& out) const noexcept {
        return anira_stage_input_role(m_ctx, slot, &out);
    }
    /// anira_stage_output_role: the output twin, against num_outputs().
    anira_status output_role(uint32_t slot, Role& out) const noexcept {
        return anira_stage_output_role(m_ctx, slot, &out);
    }
    /// anira_stage_input_ring: fills out with a view of the ring of a Streamed input in
    /// pre_process, the phase that moves the input rings. @return ANIRA_OK;
    /// ANIRA_ERROR_INVALID_STATE for a tensor of another role and in every other phase, which is
    /// the stage's bug and is recorded, ANIRA_ERROR_INVALID_ARGUMENT for a slot out of range;
    /// out is then a view of no ring.
    anira_status input_ring(uint32_t slot, RingView& out) const noexcept {
        anira_ring* ring = nullptr;
        const anira_status status = anira_stage_input_ring(m_ctx, slot, &ring);
        out = RingView(ring);
        return status;
    }
    /// anira_stage_output_ring: fills out with a view of the ring of a Streamed output in
    /// post_process, the phase that moves the output rings. @return as input_ring.
    anira_status output_ring(uint32_t slot, RingView& out) const noexcept {
        anira_ring* ring = nullptr;
        const anira_status status = anira_stage_output_ring(m_ctx, slot, &ring);
        out = RingView(ring);
        return status;
    }
    /// anira_stage_input_tensor: fills out with the model end of an input slot, the tensor the
    /// engine binds (before_inference: every slot; pre_process: every slot with a host end,
    /// which a State slot has not). Built by this call over the memory the tensor has right
    /// now: ask again in every callback, keep neither the descriptor nor its data pointer.
    /// @return ANIRA_OK; ANIRA_ERROR_INVALID_STATE in a phase that exposes the outputs and for
    /// a State slot in pre_process, which is the stage's bug and is recorded,
    /// ANIRA_ERROR_INVALID_ARGUMENT for a slot out of range; out is then all-zero.
    anira_status input_tensor(uint32_t slot, Tensor& out) const noexcept {
        return anira_stage_input_tensor(m_ctx, slot, &out);
    }
    /// anira_stage_output_tensor: fills out with the model end of an output slot, the tensor
    /// the engine wrote (after_inference: every slot; post_process: every slot with a host end,
    /// which a State slot has not). @return as input_tensor, against the phases that expose
    /// the outputs.
    anira_status output_tensor(uint32_t slot, Tensor& out) const noexcept {
        return anira_stage_output_tensor(m_ctx, slot, &out);
    }

    const anira_stage_ctx* native() const noexcept { return m_ctx; }

private:
    const anira_stage_ctx* m_ctx;
};

/**
 * @brief The record a Stage's or an Engine's init receives, anira_init_info of
 * anira/abi/lifecycle.h as views: the facts of the core in effect when the registration is
 * first used, and the context it is used under. Valid for the duration of init.
 */
class InitInfo {
public:
    explicit InitInfo(const anira_init_info* info) noexcept : m_info(info) {}

    /// The level in effect in this copy of anira (the core reconciles one per process).
    anira_log_level log_level() const noexcept {
        return static_cast<anira_log_level>(m_info->log_level);
    }
    /// The size of the inference-thread pool this copy of anira runs; 0 when the context
    /// brought its own threads or on a target without threads.
    uint32_t num_threads() const noexcept { return m_info->num_threads; }
    /// The context of the handler whose prepare reached the registration first: its
    /// capabilities are legal to query; valid until init returns, never kept.
    const anira_context* context() const noexcept { return m_info->context; }

    const anira_init_info* native() const noexcept { return m_info; }

private:
    const anira_init_info* m_info;
};

/**
 * @brief The record a Stage's and an Engine::Loaded's prepare receives, anira_prepare_info of
 * anira/abi/lifecycle.h as views. Valid for the duration of prepare: the spans die with the
 * call, the handler and the report stay valid while the handler stays prepared.
 */
class PrepareInfo {
public:
    explicit PrepareInfo(const anira_prepare_info* info) noexcept : m_info(info) {}

    /// The handler being prepared: its getters and the two Static entries are legal, never
    /// prepare, destroy or a Hard entry.
    anira_handler* handler() const noexcept { return m_info->handler; }
    /// This prepare's plan report.
    PlanReport report() const noexcept { return PlanReport(m_info->report); }
    /// anira_handler_num_entries of this prepare: the chunks that can be in flight at once, one
    /// scratch slot per entry (StageContext::entry).
    uint32_t num_entries() const noexcept { return m_info->num_entries; }
    /// One template per slot of the model's input list, State tensors included: the model end
    /// of the slot in the spec's dtype and shape at the pinned window, the slot's host domain,
    /// all-zero strides, no memory (never dereferenced). What StageContext::input_tensor will
    /// fill, minus the data. A Tensor is an anira_tensor with names on it (one size, standard
    /// layout), so the C array reads as one of Tensor.
    std::span<const Tensor> inputs() const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) a Tensor is the record
        return {static_cast<const Tensor*>(m_info->inputs), m_info->num_inputs};
    }
    /// The output twin.
    std::span<const Tensor> outputs() const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) a Tensor is the record
        return {static_cast<const Tensor*>(m_info->outputs), m_info->num_outputs};
    }
    /// The canonical names of the input list, in slot order.
    std::span<const char* const> input_names() const noexcept {
        return {m_info->input_names, m_info->num_inputs};
    }
    /// The canonical names of the output list, in slot order.
    std::span<const char* const> output_names() const noexcept {
        return {m_info->output_names, m_info->num_outputs};
    }
    /// The flags of this prepare: ANIRA_PREPARE_EXCLUSIVE when the handler's inferences run one
    /// at a time and in order (a model declared stateful or with a declared State pair), with a
    /// reset at every stream start.
    uint32_t flags() const noexcept { return m_info->flags; }
    /// Whether ANIRA_PREPARE_EXCLUSIVE is set: an Engine::Loaded builds the session's own
    /// executor in its Prepared then, since the session's calls claim no shared instance.
    bool exclusive() const noexcept { return (m_info->flags & ANIRA_PREPARE_EXCLUSIVE) != 0U; }

    const anira_prepare_info* native() const noexcept { return m_info; }

private:
    const anira_prepare_info* m_info;
};

/**
 * @brief A stage of the pipeline as a class: the registration. Subclass it, hand it to a Pipeline
 * as stage::Custom(std::make_shared<YourStage>(...)). It is anira_stage_desc with virtual
 * functions in place of the function pointers, and the semantics are those of anira/abi/stage.h,
 * whose lifecycle it follows: the registration is one object shared by the pipeline and every
 * handler created from it (phases(), flags() and consumed_kinds() are read once, when the stage
 * is added); init() is called once per registration, by the first anira_handler_prepare of a
 * handler of the pipeline, with the facts of the core in effect (InitInfo); prepare() is called
 * once per anira_handler_prepare of each handler and returns that handler's Stage::Prepared, the
 * C lifecycle's prepared pointer as a class: the four phases and reset run on it, so what a
 * stage keeps per handler (its per-entry scratch, sized by PrepareInfo::num_entries) is a member
 * of the Prepared and never of the registration, which two handlers of one pipeline share.
 * anira owns the Prepared from prepare on and deletes it at the C unprepare: at the next
 * anira_handler_prepare of that handler, once the old session is released, and at
 * anira_handler_destroy, after the last phase call; release() runs once, when the last carrier
 * dies, after every Prepared was deleted, whether or not init ever ran.
 *
 * A Pipeline holds at most one stage, and the stage owns every phase it fills for every slot.
 * The phases of one chunk: pre_process takes the host end of every slot (the ring of a
 * Streamed tensor) to the model tensor, the chunking and every conversion, on the thread where
 * the host end is produced (the thread that drives the Hard entries); then, on an inference
 * thread, anira feeds the State inputs, before_inference runs, anira crosses the edge into the
 * backend's domain, the engine runs, anira crosses back, after_inference runs, anira captures
 * the State outputs; then post_process takes the model tensors back to the host end. A stage
 * never crosses a domain: every phase works in the host domain of the slot. reset runs for the
 * first chunk of a new stream (after prepare, after anira_handler_reset), before that chunk's
 * pre_process and on its thread, with the chunk's context in ANIRA_PHASE_RESET, where the
 * accessors answer the role only: what the Prepared keeps between chunks (a resampler, an overlap
 * buffer, a filter's history) starts over there.
 *
 * A stage states the phases it fills in phases(), which a subclass must define. The C
 * descriptor tells a filled phase from a NULL one: a NULL pre_process or post_process slot
 * means anira's default body runs (the window pop or push of every Streamed slot), a filled
 * slot means this stage owns the phase and composes what it does not handle itself by calling
 * the default through the base class ("call super"): Prepared::pre_process(ctx) is
 * anira_stage_default_pre_process, Prepared::post_process(ctx) is
 * anira_stage_default_post_process. A virtual function is never NULL, so the mask says which
 * of the four this stage takes part in: only those reach the descriptor, the others stay NULL
 * and are never called; the reset slot is always filled, with a base body that does nothing.
 *
 * flags() is the stage's real-time promise (anira_stage_desc::flags):
 * ANIRA_STAGE_FLAG_REALTIME_PRE_POST says pre_process, post_process and reset allocate nothing,
 * lock nothing and block on nothing, ANIRA_STAGE_FLAG_REALTIME_HOOKS says the same of
 * before_inference and after_inference; the default promises nothing. anira_handler_prepare
 * checks the promise against the placement: under a Hard contract a filled pre_process or
 * post_process runs on the driving thread and requires ANIRA_STAGE_FLAG_REALTIME_PRE_POST, else
 * prepare fails with ANIRA_ERROR_CONFIG naming the flag. The phase functions and reset are
 * noexcept (an override that throws does not compile without it, and terminates with it) and
 * carry no real-time attribute of their own; everything StageContext, RingView and Tensor offer
 * is nonblocking, so a body that keeps its promise is composed of those calls. A status other
 * than ANIRA_OK fails the chunk: the status goes into anira_handler_rt_error with one latched
 * record naming the phase, and the chunk delivers zeros at its stream position.
 *
 * Why the ownership spelling: the Prepared is polymorphic and lives on the heap; prepare
 * returns a std::unique_ptr because anira is its one owner from then on (the trampoline
 * releases it into the C out_prepared on success; a refused prepare destroys nothing that was
 * not returned); the registration is a std::shared_ptr because it IS shared, by the pipeline
 * and every handler created from it.
 *
 * A stage has no name: a Pipeline holds one, so a name would identify nothing; every record
 * about it says "the stage", and the consumer column of its anira_plan_ext rows reads "stage".
 */
class Stage {
public:
    /// The bit of one phase in phases().
    static constexpr uint32_t phase_bit(anira_phase phase) noexcept {
        return uint32_t{1} << static_cast<uint32_t>(phase);
    }
    static constexpr uint32_t k_pre_process = uint32_t{1} << ANIRA_PHASE_PRE_PROCESS;
    static constexpr uint32_t k_post_process = uint32_t{1} << ANIRA_PHASE_POST_PROCESS;
    static constexpr uint32_t k_before_inference = uint32_t{1} << ANIRA_PHASE_BEFORE_INFERENCE;
    static constexpr uint32_t k_after_inference = uint32_t{1} << ANIRA_PHASE_AFTER_INFERENCE;
    /// Every bit phases() may carry.
    static constexpr uint32_t k_all_phases =
        k_pre_process | k_post_process | k_before_inference | k_after_inference;

    /**
     * @brief One handler's prepared stage: what Stage::prepare returns, the C prepared pointer
     * as a class. The four phases and reset run on it, for that handler's chunks alone, and what
     * the stage keeps per handler lives here. Created by prepare on the main thread (it may
     * allocate there), deleted by anira at the C unprepare (the next anira_handler_prepare of
     * the handler, once the old session is released, or anira_handler_destroy), after the last
     * phase call; never copied, never moved.
     */
    class Prepared {
    public:
        Prepared() = default;
        virtual ~Prepared() = default;
        Prepared(const Prepared&) = delete;
        Prepared& operator=(const Prepared&) = delete;
        Prepared(Prepared&&) = delete;
        Prepared& operator=(Prepared&&) = delete;

        /// ANIRA_PHASE_PRE_PROCESS, on the thread where the host end is produced (the driver
        /// thread under a Hard contract): takes the host end of every slot to its model tensor.
        /// The base class is the default fill (anira_stage_default_pre_process).
        virtual anira_status pre_process(StageContext& ctx) noexcept {
            return anira_stage_default_pre_process(ctx.native());
        }
        /// ANIRA_PHASE_POST_PROCESS, on the thread where the host end is consumed: takes the
        /// model tensor of every output slot back to its host end. The base class is the default
        /// push (anira_stage_default_post_process).
        virtual anira_status post_process(StageContext& ctx) noexcept {
            return anira_stage_default_post_process(ctx.native());
        }
        /// ANIRA_PHASE_BEFORE_INFERENCE, on an inference thread behind the State feed and ahead
        /// of the edge into the backend's domain. The base class does nothing.
        virtual anira_status before_inference(StageContext& /*ctx*/) noexcept { return ANIRA_OK; }
        /// ANIRA_PHASE_AFTER_INFERENCE, on an inference thread behind the edge out of the
        /// backend's domain and ahead of the State capture. The base class does nothing.
        virtual anira_status after_inference(StageContext& /*ctx*/) noexcept { return ANIRA_OK; }
        /// ANIRA_PHASE_RESET: the first chunk of a new stream (after prepare, after
        /// anira_handler_reset), before that chunk's pre_process, on its thread, never
        /// mid-stream; the context answers the role only (a ring or a tensor asked here is
        /// refused and recorded). What this object keeps between chunks starts over. The base
        /// class does nothing.
        virtual void reset(StageContext& /*ctx*/) noexcept {}
    };

    Stage() = default;
    virtual ~Stage() = default;
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;
    Stage(Stage&&) = delete;
    Stage& operator=(Stage&&) = delete;

    /// The phases this stage fills: an OR of k_pre_process, k_post_process, k_before_inference
    /// and k_after_inference. Read once, when the stage is added to a Pipeline; 0 is a stage of
    /// prepare, reset, unprepare and release alone.
    virtual uint32_t phases() const noexcept = 0;
    /// The stage's real-time promise, written into anira_stage_desc::flags by Pipeline::add: an
    /// OR of ANIRA_STAGE_FLAG_REALTIME_PRE_POST and ANIRA_STAGE_FLAG_REALTIME_HOOKS; 0 (the
    /// default) promises nothing. Under a Hard contract a stage that fills pre_process or
    /// post_process must promise ANIRA_STAGE_FLAG_REALTIME_PRE_POST, else anira_handler_prepare
    /// refuses it with ANIRA_ERROR_CONFIG. Read once, when the stage is added to a Pipeline.
    virtual uint32_t flags() const noexcept { return 0; }
    /// The extensions the stage reads at prepare, as "<host>:<kind>" strings (hosts:
    /// tensor_spec, model, model_config, context, contract); they join the consumed-or-fail
    /// walk. Read once, when the stage is added; the strings are copied.
    virtual std::span<const char* const> consumed_kinds() const noexcept { return {}; }

    /// Called once per Pipeline::add of this stage, by the first anira_handler_prepare of a
    /// handler of that pipeline, on its caller's thread ([main-thread]; it may allocate),
    /// before that handler's prepare, with the facts of the core in effect (the level, the
    /// thread count, the handler's context, whose capabilities are legal to query). What the
    /// stage keeps for its whole registration (a lookup table, a thread of its own) is built
    /// here, as a member of the object. It may throw, as prepare may: the status fails that
    /// anira_handler_prepare, the registration counts as uninitialised, and the next prepare
    /// calls init again. The base class does nothing.
    virtual void init(const InitInfo& /*info*/) {}
    /// Called by anira_handler_prepare on its caller's thread after the plan report is built,
    /// once per prepare of each handler ([main-thread]; it may allocate): returns the Prepared
    /// that runs the phases for that handler, sized by the record (its num_entries for a
    /// per-chunk scratch, its templates for the tensors), which anira owns from then on. Of
    /// anira it may call the [callback-safe] entries, the handler's getters and the two Static
    /// entries, never prepare, destroy or a Hard entry. It may throw: an anira::Error fails the
    /// prepare with its status, std::bad_alloc with ANIRA_ERROR_OUT_OF_MEMORY, anything else
    /// with ANIRA_ERROR_INTERNAL, and what() goes to the log (group "anira.hpp"), since the C
    /// callback has no error record; a null return fails it with ANIRA_ERROR_INTERNAL, since
    /// nothing could run the phases. A refused prepare leaves the handler unprepared and
    /// deletes nothing that was not returned.
    virtual std::unique_ptr<Prepared> prepare(const PrepareInfo& info) = 0;
    /// Called once per Pipeline::add of this stage, when the last pipeline or handler that
    /// carries it dies, on the thread of that destroy ([main-thread]), after every Prepared of
    /// it was deleted, whether or not init ever ran; no callback of that chain runs afterwards.
    /// The object itself lives as long as a shared_ptr to it does.
    virtual void release() noexcept {}
};

namespace detail {

/**
 * @brief The C callbacks of a stage::Custom. user_data is a heap-allocated
 * std::shared_ptr<Stage>, so the control block keeps the registration alive for as long as a
 * carrier of the descriptor exists; release deletes it, exactly once. prepared is the
 * Stage::Prepared of one handler, released from prepare's unique_ptr into the C out_prepared and
 * deleted at unprepare, exactly once per successful prepare.
 */
struct StageTrampolines {
    static Stage& stage_of(void* user_data) noexcept {
        return **static_cast<std::shared_ptr<Stage>*>(user_data);
    }
    static Stage::Prepared& prepared_of(void* prepared) noexcept {
        return *static_cast<Stage::Prepared*>(prepared);
    }

    // The four phase trampolines and the reset carry no real-time attribute, like the C
    // typedefs: the promise is the stage's flags(), not a property of the type. They run on the
    // handler's Prepared; the registration in user_data is not read on a per-chunk path.
    static anira_status ANIRA_CALL pre_process(const anira_stage_ctx* ctx,
                                               void* prepared,
                                               void* /*user_data*/) noexcept {
        StageContext context(ctx);
        return prepared_of(prepared).pre_process(context);
    }
    static anira_status ANIRA_CALL post_process(const anira_stage_ctx* ctx,
                                                void* prepared,
                                                void* /*user_data*/) noexcept {
        StageContext context(ctx);
        return prepared_of(prepared).post_process(context);
    }
    static anira_status ANIRA_CALL before_inference(const anira_stage_ctx* ctx,
                                                    void* prepared,
                                                    void* /*user_data*/) noexcept {
        StageContext context(ctx);
        return prepared_of(prepared).before_inference(context);
    }
    static anira_status ANIRA_CALL after_inference(const anira_stage_ctx* ctx,
                                                   void* prepared,
                                                   void* /*user_data*/) noexcept {
        StageContext context(ctx);
        return prepared_of(prepared).after_inference(context);
    }
    static void ANIRA_CALL reset(const anira_stage_ctx* ctx,
                                 void* prepared,
                                 void* /*user_data*/) noexcept {
        StageContext context(ctx);
        prepared_of(prepared).reset(context);
    }

    /// A throw never crosses the C boundary: it becomes the status, and its text a log line.
    static anira_status ANIRA_CALL init(const anira_init_info* info, void* user_data) noexcept {
        try {
            stage_of(user_data).init(InitInfo(info));
            return ANIRA_OK;
        } catch (const Error& error) {
            log_throw("init", error.what());
            return failed(error.status) ? error.status : ANIRA_ERROR_INTERNAL;
        } catch (const std::bad_alloc& error) {
            log_throw("init", error.what());
            return ANIRA_ERROR_OUT_OF_MEMORY;
        } catch (const std::exception& error) {
            log_throw("init", error.what());
            return ANIRA_ERROR_INTERNAL;
        } catch (...) {
            log_throw("init", "an exception that is no std::exception");
            return ANIRA_ERROR_INTERNAL;
        }
    }

    /// A throw never crosses the C boundary: it becomes the status, and its text a log line. On
    /// success the Prepared leaves the unique_ptr for the C out_prepared, anira's from then on;
    /// a throw destroys nothing that was not returned, and a null return is ANIRA_ERROR_INTERNAL.
    static anira_status ANIRA_CALL prepare(const anira_prepare_info* info,
                                           void* user_data,
                                           void** out_prepared) noexcept {
        Stage& stage = stage_of(user_data);
        try {
            std::unique_ptr<Stage::Prepared> prepared = stage.prepare(PrepareInfo(info));
            if (prepared == nullptr) {
                anira_log(ANIRA_LOG_ERROR,
                          "anira.hpp",
                          "the stage's prepare returned no Prepared: nothing can run the phases");
                return ANIRA_ERROR_INTERNAL;
            }
            *out_prepared = prepared.release();
            return ANIRA_OK;
        } catch (const Error& error) {
            log_throw("prepare", error.what());
            return failed(error.status) ? error.status : ANIRA_ERROR_INTERNAL;
        } catch (const std::bad_alloc& error) {
            log_throw("prepare", error.what());
            return ANIRA_ERROR_OUT_OF_MEMORY;
        } catch (const std::exception& error) {
            log_throw("prepare", error.what());
            return ANIRA_ERROR_INTERNAL;
        } catch (...) {
            log_throw("prepare", "an exception that is no std::exception");
            return ANIRA_ERROR_INTERNAL;
        }
    }

    /// The Prepared of one handler, back from the C lifecycle: deleted here, once.
    static void ANIRA_CALL unprepare(void* prepared, void* /*user_data*/) noexcept {
        std::unique_ptr<Stage::Prepared> holder(static_cast<Stage::Prepared*>(prepared));
        holder.reset();
    }

    static void ANIRA_CALL release(void* user_data) noexcept {
        const std::unique_ptr<std::shared_ptr<Stage>> holder(
            static_cast<std::shared_ptr<Stage>*>(user_data));
        (*holder)->release();
    }

private:
    /// Formats into a fixed buffer: nothing here may throw inside the handler of a throw.
    static void log_throw(const char* slot, const char* what) noexcept {
        std::array<char, ANIRA_ERROR_MESSAGE_CAPACITY> text{};
        std::snprintf(text.data(), text.size(), "the stage's %s threw: %s", slot, what);
        anira_log(ANIRA_LOG_ERROR, "anira.hpp", text.data());
    }
};

}  // namespace detail

// ---- engines (section 7) -------------------------------------------------------------------

/**
 * @brief The record an Engine's load receives, anira_engine_load_info of anira/abi/engine.h as
 * views. Valid for the duration of load: the spans and the model die with the call, so an
 * engine copies what it keeps (the templates, the names, what it loads from the row's path or
 * bytes).
 */
class EngineLoadInfo {
public:
    explicit EngineLoadInfo(const anira_engine_load_info* info) noexcept : m_info(info) {}

    /// This engine's entry in the variant's model list: the row whose path or bytes the engine
    /// loads (model_path(row()), model_bytes(row())).
    uint32_t row() const noexcept { return m_info->row; }
    /// The variant, anira's own copy of the model configuration: read through the config
    /// getters of anira/abi/config.h during the call, never kept.
    const anira_model_config* model() const noexcept { return m_info->model; }
    /// anira_model_config_model_count: the entries of the variant.
    uint32_t model_count() const noexcept { return anira_model_config_model_count(m_info->model); }
    /// anira_model_config_model_engine: an entry's engine as the pair, ANIRA_ENGINE_CUSTOM with
    /// the id for a custom entry (the row's is the id this engine was registered under);
    /// {ANIRA_ENGINE_NONE, empty} for an index out of range. Valid for the duration of the call.
    EngineRef model_engine(uint32_t index) const noexcept {
        return detail::engine_ref([&](anira_engine* engine, const char** id) {
            return anira_model_config_model_engine(m_info->model, index, engine, id);
        });
    }
    /// anira_model_config_model_path: an entry's file, which anira never opens for a registered
    /// engine's row; empty for a bytes entry or an index out of range. Valid for the duration of
    /// the call.
    std::string_view model_path(uint32_t index) const noexcept {
        const char* path = anira_model_config_model_path(m_info->model, index);
        return path != nullptr ? std::string_view(path) : std::string_view();
    }
    /// anira_model_config_model_bytes: an entry's bytes, valid for the duration of the call;
    /// empty for a path entry or an index out of range.
    std::span<const std::byte> model_bytes(uint32_t index) const noexcept {
        const void* bytes = nullptr;
        std::size_t size = 0;
        if (anira_model_config_model_bytes(m_info->model, index, &bytes, &size) != ANIRA_OK ||
            bytes == nullptr) {
            return {};
        }
        return {static_cast<const std::byte*>(bytes), size};
    }
    /// One template per slot of the model's input list, State tensors included: the ENGINE side
    /// of the slot, the spec's dtype and the extents in the engine's order (the entry's layout
    /// applied) at the pinned window, all-zero strides, ANIRA_DOMAIN_HOST, no memory (never
    /// dereferenced). What EngineContext::inputs will carry, minus the data. A Tensor is an
    /// anira_tensor with names on it (one size, standard layout), so the C array reads as one
    /// of Tensor.
    std::span<const Tensor> inputs() const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) a Tensor is the record
        return {static_cast<const Tensor*>(m_info->inputs), m_info->num_inputs};
    }
    /// The output twin.
    std::span<const Tensor> outputs() const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) a Tensor is the record
        return {static_cast<const Tensor*>(m_info->outputs), m_info->num_outputs};
    }
    /// The name each input slot binds to, in slot order: the entry's tensors record where it
    /// names the slot, else the canonical name. An engine whose side has names binds by these,
    /// and refuses load for a name its side does not have.
    std::span<const char* const> input_names() const noexcept {
        return {m_info->input_names, m_info->num_inputs};
    }
    /// The output twin.
    std::span<const char* const> output_names() const noexcept {
        return {m_info->output_names, m_info->num_outputs};
    }
    /// The shared call slots of this loaded model: the process calls that may run at once on
    /// them, each on its own instance below this count (EngineContext::instance), the model's
    /// max_instances for a stateless model; 0 for a model declared stateful or with a declared
    /// State pair, whose handlers are exclusive (PrepareInfo::exclusive at their prepare) and
    /// run on what their Prepared builds, never on a shared slot.
    uint32_t instances() const noexcept { return m_info->instances; }
    /// The provider this load is for, as the pair: a provider of the enum, or
    /// ANIRA_PROVIDER_CUSTOM with the custom provider's name in the engine's own vocabulary (an
    /// entry of providers()). A provider is part of the loaded model: two providers of one model
    /// are two loads, each its own Loaded. What a load cannot serve it refuses with
    /// ANIRA_ERROR_NOT_SUPPORTED (the handler checked the plan's provider against providers()
    /// at create; a device missing at run time is the load's to report). Valid for the
    /// duration of the call.
    ProviderRef provider() const noexcept {
        return ProviderRef{.kind = static_cast<Provider>(m_info->provider),
                           .id = m_info->provider_id != nullptr
                                     ? std::string_view(m_info->provider_id)
                                     : std::string_view()};
    }
    /// The provider options of this backend: the set of the context config's
    /// "provider_options" extension for the engine on this provider (ext::ProviderOptions), as
    /// two parallel spans of the keys and the values in the engine's own vocabulary. Handed to
    /// an Engine whose consumed_kinds() lists "context:provider_options" and empty otherwise
    /// (a set for an Engine that does not list it is refused at anira_handler_create, naming
    /// the backend). The options are part of the loaded model: two contexts with different
    /// options for one backend load twice. Valid for the duration of the call.
    std::span<const char* const> option_keys() const noexcept {
        return {m_info->option_keys, m_info->num_options};
    }
    /// The values, one per key.
    std::span<const char* const> option_values() const noexcept {
        return {m_info->option_values, m_info->num_options};
    }

    const anira_engine_load_info* native() const noexcept { return m_info; }

private:
    const anira_engine_load_info* m_info;
};

/**
 * @brief What a process or reset call of an Engine::Prepared sees: a view of the
 * anira_engine_ctx anira fills on its own stack for the call, the engine's twin of
 * StageContext. The tensors are in the record, one descriptor per slot of either side in slot
 * order (the tensor's position in the model configuration's input or output list, State
 * tensors included), over the memory the engine reads and writes. Valid until the callback
 * returns, and never kept beyond it. The engine rule of anira/abi/engine.h binds here: read
 * every extent and every memory handle from the tensors of THIS call, never from a value kept
 * at prepare, and never assume a tensor is anira's own buffer (the halves of a declared State
 * pair alternate between two buffers, unless the engine's flags() carry
 * ANIRA_ENGINE_FLAG_STATE_ALIAS, where one stable buffer is both halves of every call and the
 * engine updates it in place; a later pre-release hands a caller's Buffer tensor over in
 * place).
 *
 * [inference-thread]: every method is noexcept, allocates nothing and reads one field of the
 * record.
 */
class EngineContext {
public:
    /// Wraps the record a process or reset call received; never NULL (anira hands out none).
    explicit EngineContext(const anira_engine_ctx* ctx) noexcept : m_ctx(ctx) {}

    /// The shared slot of the loaded model this call runs on, 0 .. EngineLoadInfo::instances
    /// - 1; never two calls at once on one instance. 0 for an exclusive call, which claimed no
    /// slot.
    uint32_t instance() const noexcept { return m_ctx->instance; }
    /// The chunk's position in the handler's inference queue (StageContext::entry of the same
    /// chunk).
    uint32_t entry() const noexcept { return m_ctx->entry; }
    /// ANIRA_TICKET_INVALID under a Hard contract; the job's ticket under an Async one.
    anira_ticket ticket() const noexcept { return m_ctx->ticket; }
    /// Per-call flags: ANIRA_ENGINE_CALL_EXCLUSIVE for a call of an exclusive handler.
    uint32_t flags() const noexcept { return m_ctx->flags; }
    /// Whether ANIRA_ENGINE_CALL_EXCLUSIVE is set: the call is an exclusive handler's, runs on
    /// what its Prepared built and claimed no shared slot.
    bool exclusive() const noexcept { return (m_ctx->flags & ANIRA_ENGINE_CALL_EXCLUSIVE) != 0U; }
    /// What this loaded model's load handed back (the C loaded pointer): for an anira::Engine
    /// the Engine::Loaded the call runs on, which its Prepared holds already.
    void* loaded() const noexcept { return m_ctx->loaded; }
    /// The model's input list, State tensors included: the descriptors over the memory the
    /// engine reads, the spec's dtype and the engine-side extents of the load record's
    /// templates, with the data.
    std::span<const Tensor> inputs() const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) a Tensor is the record
        return {static_cast<const Tensor*>(m_ctx->inputs), m_ctx->num_inputs};
    }
    /// The output twin, over the memory the engine writes.
    std::span<Tensor> outputs() const noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) a Tensor is the record
        return {static_cast<Tensor*>(m_ctx->outputs), m_ctx->num_outputs};
    }

    const anira_engine_ctx* native() const noexcept { return m_ctx; }

private:
    const anira_engine_ctx* m_ctx;
};

namespace detail {
struct EngineHandle;
struct EngineTrampolines;
}  // namespace detail

/**
 * @brief A custom engine as a class: the engine object. Subclass it, construct it with its
 * reverse-URI id (Engine(std::string id)), hand it to a Pipeline (Pipeline::register_engine, or
 * stage::Inference::engine, which Pipeline::add registers before it adds the stage), and name
 * the id on a model entry
 * (ModelConfig::add_model_path(id, path)): that entry is then a plan like a built-in engine's,
 * ANIRA_ENGINE_CUSTOM with the id wherever the engine-provider pair travels. It is
 * anira_engine_desc with virtual functions in place of the function pointers, and the
 * semantics are those of anira/abi/engine.h, whose lifecycle it follows, the stage's lifecycle
 * with one level more and the same words (Stage, Stage::Prepared). The object is the engine's
 * identity and its id the engine's name, as an anira_custom_engine and its id are in C: the
 * first Pipeline::register_engine of an object creates its C engine under the id (flags(),
 * consumed_kinds() and providers() are read then, once), and every later registration of the
 * same object, on another Pipeline, adds that same C engine for as long as anything holds it
 * (a Pipeline, a handler created from one, a loaded model in the pool: the object keeps a
 * detached handle of it, anira_custom_engine_detach). The levels, from
 * the outermost in: init() is called once per C engine, by the first anira_handler_prepare
 * that reaches it, with the facts of the core in effect (InitInfo); load() is called once per
 * loaded model anira pools and returns that model's Engine::Loaded, the C lifecycle's loaded
 * pointer as a class: the weights live there, with the shared executors of its instances;
 * Loaded::prepare() is called once per handler that runs the model and returns that handler's
 * Engine::Prepared, the C prepared pointer as a class: process and reset run on it, so what an
 * exclusive handler keeps per stream (its own executor, since its calls claim no shared
 * instance) is a member of the Prepared, and what every handler shares a member of the Loaded;
 * nothing an inference needs is a member of the Engine. Two handlers share one Loaded and its
 * instances when they run the same Engine object on an equal model configuration resolved to
 * equal tensors, whichever Pipelines they were created from (two Engine objects never share),
 * a stateful model included: it is loaded once, with no shared instance, and every handler of
 * it is exclusive (PrepareInfo::exclusive). So instances of one plugin that share a model load
 * it once when they register one Engine object, kept by the plugin for the whole binary. anira
 * owns the Prepared from prepare on and deletes it at the C unprepare (the handler's next
 * prepare or its destroy, on the thread of that entry, after its in-flight inferences have
 * drained), the Loaded from load on and deletes it at the C unload (when the last handler
 * holding it is re-prepared or destroyed, after every Prepared over it); release() runs once
 * per C engine, when the last reference to it dies, after every Loaded was deleted, whether or
 * not init ever ran.
 *
 * anira never opens a registered row's path or reads its bytes: load does, through the record
 * (EngineLoadInfo::model_path(row()), model_bytes(row()), the config getters on the variant),
 * and binds the slots by the names the record carries where its side has names, so every slot
 * of the plan reports ANIRA_BINDING_ENGINE. process runs on an inference thread, in
 * ANIRA_PHASE_INFERENCE, over an EngineContext, on the shared instance the context names or,
 * for an exclusive call (EngineContext::exclusive), on what the Prepared built; it may block
 * and must not allocate per call. reset runs for an exclusive handler at the first inference
 * of a new stream (after prepare, after anira_handler_reset), with that inference's context,
 * right before its process; a handler whose model is stateless is never reset. A status other
 * than ANIRA_OK from process fails the chunk: it lands in anira_handler_rt_error with one
 * latched record, anira zeroes the outputs, the chunk delivers zeros at its stream position, a
 * State pair keeps its last good value (an aliasing engine writes in place: what a failed call
 * left in the buffer is its own).
 *
 * flags() is the engine's promise (anira_engine_desc::flags): an OR of
 * ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, ANIRA_ENGINE_FLAG_REALTIME_SAFE,
 * ANIRA_ENGINE_FLAG_DYNAMIC_TIME and ANIRA_ENGINE_FLAG_STATE_ALIAS, declared and reported in
 * anira_plan_info::engine_flags in this
 * pre-release; the default promises nothing. process and reset are noexcept (an override that
 * throws does not compile without it, and terminates with it) and carry no real-time attribute
 * of their own, like the C typedefs: whether process is real-time is the flag, and everything
 * EngineContext and Tensor offer is nonblocking, so a body that keeps the promise is composed of
 * those calls and the engine's own real-time code.
 *
 * Why the ownership spelling: the Loaded and the Prepared are polymorphic and live on the heap;
 * load and prepare return a std::unique_ptr because anira is their one owner from then on (the
 * trampoline releases it into the C out pointer on success; a refused call destroys nothing
 * that was not returned); the Engine is a std::shared_ptr because it IS shared, by every
 * pipeline it is registered on and every handler created from them: its C engine holds a copy
 * until release.
 */
class Engine {
public:
    /**
     * @brief One handler's prepared handle over a loaded model: what Engine::Loaded::prepare
     * returns, the C prepared pointer as a class. process and reset run on it: a shared call on
     * the instance the context names (the Loaded's), an exclusive call on what this object
     * built at prepare (its own executor). Created by prepare on the main thread (it may
     * allocate there), deleted by anira at the C unprepare (the handler's next prepare or its
     * destroy), after the last process and before the Loaded; never copied, never moved.
     */
    class Prepared {
    public:
        Prepared() = default;
        virtual ~Prepared() = default;
        Prepared(const Prepared&) = delete;
        Prepared& operator=(const Prepared&) = delete;
        Prepared(Prepared&&) = delete;
        Prepared& operator=(Prepared&&) = delete;

        /// ANIRA_PHASE_INFERENCE, on an inference thread: one inference over the context's
        /// tensors, on ctx.instance() of the Loaded, or on this object's own executor when
        /// ctx.exclusive(). May block, must not allocate per call; every extent and every
        /// memory handle from the context's tensors, never from load or prepare. Any status but
        /// ANIRA_OK fails the chunk (zeros at its stream position, the status latched in
        /// anira_handler_rt_error).
        virtual anira_status process(EngineContext& ctx) noexcept = 0;
        /// This plan's first inference of a new stream of an exclusive handler (after prepare,
        /// after anira_handler_reset; every plan is reset at its own first inference, a plan
        /// switch alone is no boundary), with that inference's context, right before its
        /// process; never for a handler whose model is stateless: the state this object keeps
        /// per stream starts over. The base class does nothing.
        virtual void reset(EngineContext& /*ctx*/) noexcept {}
    };

    /**
     * @brief One loaded model of the engine: what Engine::load returns, the C loaded pointer as
     * a class. The weights and the executors of the shared instances (EngineLoadInfo::instances
     * of them, never two calls at once on one) live here; prepare hands every handler that runs
     * the model its Prepared. Created by load on the main thread under the core's lifecycle
     * lock (it may allocate and read files there), deleted by anira at the C unload (when the
     * last handler holding the model is re-prepared or destroyed), after every Prepared over it;
     * never copied, never moved.
     */
    class Loaded {
    public:
        Loaded() = default;
        virtual ~Loaded() = default;
        Loaded(const Loaded&) = delete;
        Loaded& operator=(const Loaded&) = delete;
        Loaded(Loaded&&) = delete;
        Loaded& operator=(Loaded&&) = delete;

        /// Called by anira_handler_prepare on its caller's thread, once per handler that runs
        /// this model ([main-thread]; it may allocate), after the plan report is built, with
        /// the record the stage's prepare receives: returns the Prepared that runs the
        /// handler's inferences, which anira owns from then on. Under PrepareInfo::exclusive
        /// the handler's calls claim no shared instance, so the Prepared builds the executor
        /// they run on (and what it keeps per stream); a stateless handler's Prepared routes
        /// its calls to the shared instances. Of anira it may call the [callback-safe] entries,
        /// the handler's getters and the two Static entries, never prepare, destroy or a Hard
        /// entry. The prepares of one Loaded never overlap, nor a prepare and the destruction
        /// of a Prepared of it (anira serialises them per loaded model; those of different
        /// loaded models may run at once). It may throw, as Engine::load may; a null return is
        /// ANIRA_ERROR_INTERNAL.
        virtual std::unique_ptr<Prepared> prepare(const PrepareInfo& info) = 0;
    };

    /// The engine object under its id: the reverse-URI name model entries name it by
    /// (ModelConfig::add_model_path(id, path)), which the plan report and every message about
    /// the engine carry, fixed for the object's life as anira_custom_engine_create fixes a C
    /// engine's. It is checked when the object's C engine is created, at its first
    /// registration: an id without a '.' or with the prefix "anira." is
    /// ANIRA_ERROR_INVALID_ARGUMENT there. The engines of one Pipeline have distinct ids; two
    /// objects may carry one id on two Pipelines (two instances of a plugin, each with its own
    /// engine) and never share a loaded model.
    explicit Engine(std::string id) : m_id(std::move(id)) {}
    virtual ~Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    /// The id the object was constructed with.
    const std::string& id() const noexcept { return m_id; }

    /// The engine's promises, written into anira_engine_desc::flags at registration: an OR of
    /// ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, ANIRA_ENGINE_FLAG_REALTIME_SAFE,
    /// ANIRA_ENGINE_FLAG_DYNAMIC_TIME and ANIRA_ENGINE_FLAG_STATE_ALIAS; 0 (the default)
    /// promises nothing. A bit anira/abi/enums.h does not define is refused at registration
    /// with ANIRA_ERROR_INVALID_ARGUMENT. Read once, when the engine is registered.
    virtual uint32_t flags() const noexcept { return 0; }
    /// The extensions the engine reads at load, as "<host>:<kind>" strings (hosts:
    /// tensor_spec, model, model_config, context, contract); they join the consumed-or-fail
    /// walk for the entries of this engine, keyed by its id. Read once, when the engine is
    /// registered; the strings are copied.
    virtual std::span<const char* const> consumed_kinds() const noexcept { return {}; }
    /// The providers the engine serves beyond ANIRA_PROVIDER_DEFAULT, written into
    /// anira_engine_desc::providers at registration: the JSON spellings of anira_provider
    /// ("cuda", "webgpu", "directml", "coreml", "xnnpack", "vulkan") name a provider of the
    /// enum, any other string a custom provider in the engine's own vocabulary (the name a
    /// candidate's provider_id and a model entry's pin spell). Empty (the default) serves the
    /// default provider alone; a candidate naming a provider the list lacks is
    /// ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create. Read once, when the engine is
    /// registered; the strings are copied. A provider is part of the loaded model: two
    /// providers of one model are two loads (EngineLoadInfo::provider, provider_id).
    virtual std::span<const char* const> providers() const noexcept { return {}; }
    /// Which of providers() are usable here, now, as a bitmask over the list (bit i set:
    /// providers()[i]; the default provider is always served and has no bit): the engine's
    /// own GetAvailableProviders, a device present, a library loaded. Called before init and
    /// any number of times, on the main thread ([main-thread]; it may log, must not call an
    /// entry that takes the core's lifecycle lock): at anira_handler_create, where every
    /// candidate's provider of the engine is checked against the answer (a declared provider
    /// the answer clears is ANIRA_ERROR_NOT_SUPPORTED, "declares provider 'x' but its query
    /// reports it unavailable here"), and at Pipeline::capabilities, which reports the
    /// engine's rows beside the context's. The base answers every bit: every declared
    /// provider is usable. It may throw, as init may: the status fails the calling entry.
    virtual std::uint64_t query(const InitInfo& /*info*/) const { return ~std::uint64_t{0}; }

    /// Called once per C engine of this object, by the first anira_handler_prepare that reaches
    /// it (a row of the engine survives validation), under the core's lifecycle lock
    /// ([main-thread]; it may allocate and log, and may query the context's capabilities, but
    /// must not call an entry that takes that lock), before the engine's first load, with the
    /// facts of the core in effect: the level, the thread count, the handler's context. What
    /// the engine keeps for its whole life (a device context, a thread pool, a weights cache)
    /// is built here, as a member of the object. It may throw, as load may: the status fails
    /// that anira_handler_prepare, the C engine counts as uninitialised, and the next prepare
    /// calls init again. The base class does nothing.
    virtual void init(const InitInfo& /*info*/) {}
    /// Called by anira_handler_prepare on its caller's thread, once per loaded model anira
    /// pools ([main-thread], under the core's lifecycle lock, so it may allocate, read files
    /// and log, but must not call an entry that takes that lock: context create and destroy,
    /// handler create, destroy and prepare, anira_inference_thread_create, anira_shutdown,
    /// anira_release_core_if_idle): loads the row's model out of the record, binds the slots
    /// by the record's names and sizes the shared instances the record counts (none for a
    /// model whose handlers are exclusive), and returns the Loaded, which anira owns from then
    /// on. It may throw: an anira::Error fails the prepare with its status, std::bad_alloc with
    /// ANIRA_ERROR_OUT_OF_MEMORY, anything else with ANIRA_ERROR_INTERNAL, and what() goes to
    /// the log (group "anira.hpp"), since the C callback has no error record; a null return
    /// fails it with ANIRA_ERROR_INTERNAL, since nothing could run the inference. The handler's
    /// prepare then fails with the status, its message naming the engine, and unload is not
    /// called.
    virtual std::unique_ptr<Loaded> load(const EngineLoadInfo& info) = 0;
    /// Called once per C engine of this object, when the last reference to it dies (every
    /// Pipeline that registered it, every handler created from them, every loaded model), on
    /// the thread of that destroy ([main-thread]), after every Loaded of it was deleted,
    /// whether or not init ever ran; no callback of that C engine runs afterwards. A
    /// registration after that creates a new C engine, answered by a release of its own. The
    /// object itself lives as long as a shared_ptr to it does.
    virtual void release() noexcept {}

private:
    friend struct detail::EngineTrampolines;
    std::string m_id;
    /// The handle of this object's C engine, detached (anira_custom_engine_detach) once the
    /// C engine was added to its first Pipeline: it names the C engine for every later
    /// registration for as long as anything holds it (a Pipeline, a handler created from one,
    /// a loaded model in the pool) and keeps it alive no longer, since the C engine owns a
    /// shared_ptr to this object (its user_data) and a strong reference back would keep both
    /// alive forever. Reset by the C engine's release, so a registration after that creates a
    /// new C engine. Guarded by m_handle_mutex: two plugin instances may register one object
    /// from two threads, and the release runs on the thread of the last destroy.
    std::mutex m_handle_mutex;
    std::shared_ptr<detail::EngineHandle> m_handle;
};

namespace detail {

/**
 * @brief The handle of the C engine of an Engine object (anira_custom_engine): a reference
 * until Pipeline::register_engine detached it, a name afterwards; freed at destruction. Held
 * by the object (Engine::m_handle) until the C engine's release, and by a registration for
 * the duration of its addition.
 */
struct EngineHandle {
    EngineHandle() = default;
    ~EngineHandle() { anira_custom_engine_destroy(m_engine); }
    EngineHandle(const EngineHandle&) = delete;
    EngineHandle& operator=(const EngineHandle&) = delete;
    EngineHandle(EngineHandle&&) = delete;
    EngineHandle& operator=(EngineHandle&&) = delete;

    anira_custom_engine* m_engine = nullptr;
};

/**
 * @brief The C callbacks of an Engine registered through Pipeline::register_engine, and the C
 * engine of an Engine object (handle_of). user_data is a heap-allocated std::shared_ptr<Engine>,
 * so the control block keeps the object alive for as long as the C engine exists; release
 * deletes it, exactly once per C engine. loaded is the Engine::Loaded of one loaded model,
 * released from load's unique_ptr into the C out_loaded and deleted at unload; prepared the
 * Engine::Prepared of one handler on it, released from prepare's unique_ptr into the C
 * out_prepared and deleted at unprepare, exactly once per successful call each.
 */
struct EngineTrampolines {
    /// The C engine of `engine`: the one its handle names while anything holds it, else a new
    /// one (anira_custom_engine_create) under the object's id, whose descriptor is read from the
    /// object now: flags(), consumed_kinds(), providers(), the nine slots. A new C engine's
    /// handle keeps it alive until Pipeline::register_engine added it and detached the handle
    /// (a refused addition detaches too, so the C engine goes again and release answers it).
    /// @throws Error when the C entry refuses the id (no reverse-URI name, or anira's own) or
    /// the descriptor (a flags() bit anira/abi/enums.h does not define); a refused create keeps
    /// no copy of the object and never calls its release.
    static std::shared_ptr<EngineHandle> handle_of(const std::shared_ptr<Engine>& engine) {
        const std::scoped_lock<std::mutex> lock(engine->m_handle_mutex);
        if (engine->m_handle != nullptr) { return engine->m_handle; }
        const std::span<const char* const> kinds = engine->consumed_kinds();
        const std::span<const char* const> providers = engine->providers();
        anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
        desc.consumed_kinds = kinds.empty() ? nullptr : kinds.data();
        desc.num_consumed_kinds = static_cast<uint32_t>(kinds.size());
        desc.providers = providers.empty() ? nullptr : providers.data();
        desc.num_providers = static_cast<uint32_t>(providers.size());
        desc.flags = engine->flags();  // the C entry refuses a bit it does not define
        // The shared lifecycle, every slot filled: process and reset on the handler's Prepared,
        // prepare returning it and unprepare deleting it, load returning the Loaded and unload
        // deleting it, init and release for the C engine.
        desc.process = &process;
        desc.reset = &reset;
        desc.prepare = &prepare;
        desc.unprepare = &unprepare;
        desc.load = &load;
        desc.unload = &unload;
        desc.init = &init;
        desc.release = &release;
        desc.query = &query;
        // The C engine owns the copy from a successful create on and deletes it in release; a
        // refused create never calls release, so the copy dies here with the throw. The handle
        // exists first, so that nothing between the create and its owner can leak the engine.
        auto holder = std::make_unique<std::shared_ptr<Engine>>(engine);
        desc.user_data = holder.get();
        auto handle = std::make_shared<EngineHandle>();
        anira_error err{};
        check(anira_custom_engine_create(engine->id().c_str(), &desc, &handle->m_engine, &err),
              err);
        [[maybe_unused]] const std::shared_ptr<Engine>* const carried = holder.release();
        engine->m_handle = handle;
        return handle;
    }

    /// Whether `handle` still names `engine`'s C engine: false once that C engine was released
    /// (its last holder gone on another thread between a registration's lookup and its
    /// addition), which the registration answers with a fresh C engine.
    static bool still_named(Engine& engine, const std::shared_ptr<EngineHandle>& handle) {
        const std::scoped_lock<std::mutex> lock(engine.m_handle_mutex);
        return engine.m_handle == handle;
    }

    static Engine& engine_of(void* user_data) noexcept {
        return **static_cast<std::shared_ptr<Engine>*>(user_data);
    }
    static Engine::Loaded& loaded_of(void* loaded) noexcept {
        return *static_cast<Engine::Loaded*>(loaded);
    }
    static Engine::Prepared& prepared_of(void* prepared) noexcept {
        return *static_cast<Engine::Prepared*>(prepared);
    }

    // process and reset carry no real-time attribute, like the C typedefs: the promise is the
    // engine's flags(), not a property of the type. They run on the handler's Prepared; the
    // registration in user_data and the Loaded in the context are not read on a per-inference
    // path.
    static anira_status ANIRA_CALL process(const anira_engine_ctx* ctx,
                                           void* prepared,
                                           void* /*user_data*/) noexcept {
        EngineContext context(ctx);
        return prepared_of(prepared).process(context);
    }
    static void ANIRA_CALL reset(const anira_engine_ctx* ctx,
                                 void* prepared,
                                 void* /*user_data*/) noexcept {
        EngineContext context(ctx);
        prepared_of(prepared).reset(context);
    }

    /// A throw never crosses the C boundary: it becomes the status, and its text a log line.
    static anira_status ANIRA_CALL init(const anira_init_info* info, void* user_data) noexcept {
        return guarded("init", [&] {
            engine_of(user_data).init(InitInfo(info));
            return ANIRA_OK;
        });
    }
    /// The query: the object's query() as the bitmask, a throw the status.
    static anira_status ANIRA_CALL query(const anira_init_info* info,
                                         void* user_data,
                                         std::uint64_t* out_available) noexcept {
        return guarded("query", [&] {
            *out_available = engine_of(user_data).query(InitInfo(info));
            return ANIRA_OK;
        });
    }

    /// On success the Loaded leaves the unique_ptr for the C out_loaded, anira's from then on;
    /// a throw destroys nothing that was not returned, and a null return is ANIRA_ERROR_INTERNAL.
    static anira_status ANIRA_CALL load(const anira_engine_load_info* info,
                                        void* user_data,
                                        void** out_loaded) noexcept {
        return guarded("load", [&] {
            std::unique_ptr<Engine::Loaded> loaded =
                engine_of(user_data).load(EngineLoadInfo(info));
            if (loaded == nullptr) {
                anira_log(ANIRA_LOG_ERROR,
                          "anira.hpp",
                          "the engine's load returned no Loaded: nothing can run the inference");
                return ANIRA_ERROR_INTERNAL;
            }
            *out_loaded = loaded.release();
            return ANIRA_OK;
        });
    }

    /// The Loaded of one loaded model, back from the C lifecycle: deleted here, once.
    static void ANIRA_CALL unload(void* loaded, void* /*user_data*/) noexcept {
        std::unique_ptr<Engine::Loaded> holder(static_cast<Engine::Loaded*>(loaded));
        holder.reset();
    }

    /// On success the Prepared leaves the unique_ptr for the C out_prepared, anira's from then
    /// on; a throw destroys nothing that was not returned, and a null return is
    /// ANIRA_ERROR_INTERNAL, as is a call without a Loaded (an Engine always loads one).
    static anira_status ANIRA_CALL prepare(const anira_prepare_info* info,
                                           void* loaded,
                                           void* /*user_data*/,
                                           void** out_prepared) noexcept {
        return guarded("prepare", [&] {
            if (loaded == nullptr) {
                anira_log(ANIRA_LOG_ERROR,
                          "anira.hpp",
                          "the engine's prepare was called without a Loaded");
                return ANIRA_ERROR_INTERNAL;
            }
            std::unique_ptr<Engine::Prepared> prepared =
                loaded_of(loaded).prepare(PrepareInfo(info));
            if (prepared == nullptr) {
                anira_log(ANIRA_LOG_ERROR,
                          "anira.hpp",
                          "the engine's prepare returned no Prepared: nothing can run the "
                          "inference");
                return ANIRA_ERROR_INTERNAL;
            }
            *out_prepared = prepared.release();
            return ANIRA_OK;
        });
    }

    /// The Prepared of one handler, back from the C lifecycle: deleted here, once.
    static void ANIRA_CALL unprepare(void* prepared, void* /*user_data*/) noexcept {
        std::unique_ptr<Engine::Prepared> holder(static_cast<Engine::Prepared*>(prepared));
        holder.reset();
    }

    static void ANIRA_CALL release(void* user_data) noexcept {
        const std::unique_ptr<std::shared_ptr<Engine>> holder(
            static_cast<std::shared_ptr<Engine>*>(user_data));
        Engine& engine = **holder;
        {
            // The handle names this C engine no more: a later registration creates a new one.
            const std::scoped_lock<std::mutex> lock(engine.m_handle_mutex);
            engine.m_handle.reset();
        }
        engine.release();
    }

private:
    /// Runs `body` with the C boundary's rule: a throw becomes the status (an anira::Error its
    /// own, std::bad_alloc ANIRA_ERROR_OUT_OF_MEMORY, anything else ANIRA_ERROR_INTERNAL) and
    /// its text a log line naming the slot.
    template <class Body>
    static anira_status guarded(const char* slot, Body&& body) noexcept {
        try {
            return body();
        } catch (const Error& error) {
            log_throw(slot, error.what());
            return failed(error.status) ? error.status : ANIRA_ERROR_INTERNAL;
        } catch (const std::bad_alloc& error) {
            log_throw(slot, error.what());
            return ANIRA_ERROR_OUT_OF_MEMORY;
        } catch (const std::exception& error) {
            log_throw(slot, error.what());
            return ANIRA_ERROR_INTERNAL;
        } catch (...) {
            log_throw(slot, "an exception that is no std::exception");
            return ANIRA_ERROR_INTERNAL;
        }
    }

    /// Formats into a fixed buffer: nothing here may throw inside the handler of a throw.
    static void log_throw(const char* slot, const char* what) noexcept {
        std::array<char, ANIRA_ERROR_MESSAGE_CAPACITY> text{};
        std::snprintf(text.data(), text.size(), "the engine's %s threw: %s", slot, what);
        anira_log(ANIRA_LOG_ERROR, "anira.hpp", text.data());
    }
};

}  // namespace detail
// ---- pipeline (section 6) ------------------------------------------------------------------

namespace stage {

/**
 * @brief The inference stage of a Pipeline: the model configuration(s) it may run, the
 * candidate backends (empty = the default set: every engine this build carries on
 * ANIRA_PROVIDER_DEFAULT, every custom engine an entry names (ANIRA_ENGINE_CUSTOM with its
 * id), and every provider a model
 * entry of the variant is pinned to, on that entry's engine, so that a pinned entry runs on its
 * pin and a neutral one on the default provider) and the custom engines it brings along
 * (engine(impl): Pipeline::add registers each one on the pipeline through
 * Pipeline::register_engine before it adds the stage). Holds pointers into the ModelConfigs, which
 * must outlive the Pipeline's construction; the pipeline copies them
 * (anira_pipeline_add_inference). One variant in this pre-release.
 */
class Inference {
public:
    /// One model with its candidate backends (empty = the default set).
    Inference(const ModelConfig& model, std::initializer_list<BackendId> candidates = {})
        : m_variants{model.native()}, m_candidates(candidates) {}
    /// Several variants of one stage; the C entry refuses more than one with
    /// ANIRA_ERROR_NOT_SUPPORTED until plan sets land.
    Inference(std::initializer_list<std::reference_wrapper<const ModelConfig>> variants,
              std::initializer_list<BackendId> candidates)
        : m_candidates(candidates) {
        m_variants.reserve(variants.size());
        for (const ModelConfig& variant : variants) { m_variants.push_back(variant.native()); }
    }

    /// A custom engine of this stage, registered on the pipeline under its id (Engine::id) by
    /// Pipeline::add before the stage is added (Pipeline::register_engine, one call per
    /// engine, in the order given), so that a model entry of the stage's configuration may
    /// name the id. A null implementation and an id the pipeline already has are refused
    /// there, by Pipeline::register_engine's rules.
    Inference& engine(std::shared_ptr<Engine> implementation) {
        m_engines.push_back(std::move(implementation));
        return *this;
    }

    /// The variants' native configs, in the order given.
    std::span<const anira_model_config* const> variants() const noexcept { return m_variants; }
    /// The candidate backends, in the order given; empty for the default set.
    std::span<const BackendId> candidates() const noexcept { return m_candidates; }
    /// The engines of engine(), in the order given.
    std::span<const std::shared_ptr<Engine>> engines() const noexcept { return m_engines; }

private:
    std::vector<const anira_model_config*> m_variants;
    std::vector<BackendId> m_candidates;
    std::vector<std::shared_ptr<Engine>> m_engines;
};

/**
 * @brief A Stage subclass as the one custom stage of a Pipeline: the pre- and post-processing
 * around the inference stage. Holds the shared_ptr; Pipeline::add copies it into the
 * pipeline's carrier (anira_pipeline_add_stage), which the pipeline and every handler created
 * from it share, so the object lives at least until the last of them is destroyed, whatever
 * the caller does with its own pointer. A pipeline holds at most one (a second add throws
 * ANIRA_ERROR_INVALID_STATE); the position relative to the inference stage means nothing.
 */
class Custom {
public:
    explicit Custom(std::shared_ptr<Stage> implementation) noexcept
        : m_stage(std::move(implementation)) {}

    /// The stage; Pipeline::add refuses a null one with ANIRA_ERROR_INVALID_ARGUMENT.
    const std::shared_ptr<Stage>& stage() const noexcept { return m_stage; }

private:
    std::shared_ptr<Stage> m_stage;
};

}  // namespace stage

/**
 * @brief The backends and edges a handler of one Pipeline sees on one Context: the context's
 * probed rows (Capabilities) and then one row per custom engine registered on the pipeline
 * and provider it serves here, the default provider first, then every declared provider its
 * Engine::query reports usable (anira_pipeline_capabilities_backends /
 * anira_pipeline_capabilities_edge). Every call runs the engines' queries; the built-in rows
 * are the context's last probe's. The strings of the rows point into the pipeline's engines
 * and the context's store: valid while the Pipeline lives and until the context's next probe.
 */
class PipelineCapabilities {
public:
    PipelineCapabilities(const anira_pipeline* pipeline, const anira_context* context) noexcept
        : m_pipeline(pipeline), m_context(context) {}

    /// The context's backends, then the custom engines' rows (ANIRA_ENGINE_CUSTOM with the
    /// engine's id). @throws Error with a query's status when an engine's query() throws.
    std::vector<BackendId> backends() const {
        return detail::enumerate<BackendId>(
            [this](uint32_t* count, BackendId* out) {
                return anira_pipeline_capabilities_backends(m_pipeline,
                                                            m_context,
                                                            sizeof(BackendId),
                                                            count,
                                                            out);
            },
            "anira_pipeline_capabilities_backends");
    }
    /// One row, by domain and backend: the context's registry for a built-in engine, the
    /// pipeline's for a custom one (engine_id set).
    /// @throws Error with ANIRA_ERROR_EDGE_UNREACHABLE when neither registry has such a row.
    anira_edge_info edge(Domain from, const BackendId& to) const {
        anira_edge_info row = ANIRA_EDGE_INFO_INIT;
        detail::check(anira_pipeline_capabilities_edge(m_pipeline, m_context, from, &to, &row),
                      "anira_pipeline_capabilities_edge");
        return row;
    }

private:
    const anira_pipeline* m_pipeline;
    const anira_context* m_context;
};

/**
 * @brief An anira_pipeline with its lifetime: the stages a handler runs, exactly one
 * stage::Inference and at most one stage::Custom around it, and the custom engines registered
 * on it. A custom engine is no stage: it is part of the inference stage, one more
 * implementation its candidates resolve to, registered through register_engine or on the
 * stage (stage::Inference::engine, which add registers before it adds the stage) over
 * anira_pipeline_add_engine, and named by a model entry of the configuration by its id
 * (Engine::id; ModelConfig::add_model_path(id, path)). The pipeline holds the C engine of
 * every Engine object registered on it, so another Pipeline registering the same object reuses
 * it, and handlers of both share loaded models. Move-only; copied by the handler that takes
 * it, engines included: a registration after a handler's create does not reach that handler.
 */
class Pipeline {
public:
    /// One stage of any kind. (Not named Stage: that is the base class of a custom stage.)
    using AnyStage = std::variant<stage::Inference, stage::Custom>;

    /// @throws Error when a C entry refuses the call; the status says why.
    Pipeline() {
        detail::abi_check_once();
        anira_error err{};
        detail::check(anira_pipeline_create(&m_pipeline, &err), err);
    }
    /// Creates the pipeline and adds every stage in order.
    /// @throws Error as add does.
    Pipeline(std::initializer_list<AnyStage> stages) : Pipeline() {
        for (const AnyStage& any_stage : stages) { add(any_stage); }
    }
    ~Pipeline() { anira_pipeline_destroy(m_pipeline); }
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&& other) noexcept : m_pipeline(std::exchange(other.m_pipeline, nullptr)) {}
    Pipeline& operator=(Pipeline&& other) noexcept {
        if (this != &other) {
            anira_pipeline_destroy(m_pipeline);
            m_pipeline = std::exchange(other.m_pipeline, nullptr);
        }
        return *this;
    }

    /// Adds the inference stage (anira_pipeline_add_inference).
    Pipeline& inference(const ModelConfig& model,
                        std::initializer_list<BackendId> candidates = {}) {
        return add(stage::Inference(model, candidates));
    }
    /// Adds one stage of any kind. A stage::Custom is read once, here: its phases(), flags()
    /// and consumed_kinds() fill an anira_stage_desc (anira_pipeline_add_stage), and
    /// only the phases of the mask reach it; its prepare runs per handler at
    /// anira_handler_prepare and the Prepared it returns runs the phases and the reset of
    /// that handler until the C unprepare deletes it; Stage::release answers this add once.
    /// @throws Error when the C entry refuses it: a second inference stage is
    /// ANIRA_ERROR_CONFIG, a second custom stage ANIRA_ERROR_INVALID_STATE, more than one
    /// variant ANIRA_ERROR_NOT_SUPPORTED; a null custom stage, a phases() bit that names none
    /// of the four phases, a flags() bit anira/abi/stage.h does not define, or a candidate
    /// whose provider is no value of anira_provider, carries a provider of the enum and a
    /// provider_id at once, or an empty provider_id, is ANIRA_ERROR_INVALID_ARGUMENT. Whether
    /// an engine serves a candidate's provider is anira_handler_create's question.
    Pipeline& add(const AnyStage& any_stage) {
        std::visit([this](const auto& value) { add_stage(value); }, any_stage);
        return *this;
    }
    /// Registers a custom engine on the pipeline under its id, Engine::id
    /// (anira_pipeline_add_engine). The object's C engine is created at its first registration
    /// (anira_custom_engine_create: the id, and flags(), consumed_kinds() and providers() fill
    /// an anira_engine_desc whose slots are the class's virtuals, read once, then) and reused by
    /// every later one while a Pipeline holding it lives, on another Pipeline, so handlers of
    /// all of them share loaded models of equal configurations. The C engine holds a copy of
    /// the shared_ptr, so the object lives at least until the last pipeline and handler that
    /// carry it are destroyed, whatever the caller does with its own pointer; Engine::release
    /// answers the C engine once. Legal before or after the inference stage; a model entry that
    /// names the id is a plan of the handler, an engine no entry names is not a plan and not
    /// an error, and an entry whose id no engine of the pipeline has is
    /// ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create.
    /// @throws Error ANIRA_ERROR_INVALID_ARGUMENT for a null engine (before the C call), an
    /// id without a '.' or with the prefix "anira.", or a flags() bit anira/abi/enums.h does
    /// not define (the C engine's create refuses them, and nothing is created);
    /// ANIRA_ERROR_INVALID_STATE when this pipeline already has an engine with the id, this
    /// object or another. A refused call keeps no reference the call created: a C engine the
    /// call created is dropped again, and Engine::release answers it.
    Pipeline& register_engine(const std::shared_ptr<Engine>& implementation) {
        if (implementation == nullptr) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT,
                        "anira::Pipeline::register_engine: a null engine");
        }
        std::shared_ptr<detail::EngineHandle> handle =
            detail::EngineTrampolines::handle_of(implementation);
        anira_error err{};
        anira_status status = anira_pipeline_add_engine(m_pipeline, handle->m_engine, &err);
        if (status == ANIRA_ERROR_INVALID_STATE &&
            !detail::EngineTrampolines::still_named(*implementation, handle)) {
            // The C engine the handle named was released between the lookup and the addition
            // (its last holder gone on another thread): a fresh one.
            handle = detail::EngineTrampolines::handle_of(implementation);
            status = anira_pipeline_add_engine(m_pipeline, handle->m_engine, &err);
        }
        // Added or refused, the handle keeps the C engine alive no longer: the pipeline holds
        // it now, or nothing does and Engine::release answers.
        anira_custom_engine_detach(handle->m_engine);
        detail::check(status, err);
        return *this;
    }

    /// The backends and edges a handler of this pipeline sees on a context: the context's
    /// rows and the custom engines' (their Engine::query runs on every call).
    PipelineCapabilities capabilities(const Context& context) const noexcept {
        return {m_pipeline, context.native()};
    }
    /// The same over a C context handle.
    PipelineCapabilities capabilities(const anira_context* context) const noexcept {
        return {m_pipeline, context};
    }

    const anira_pipeline* native() const noexcept { return m_pipeline; }
    anira_pipeline* native() noexcept { return m_pipeline; }

private:
    void add_stage(const stage::Inference& stage) {
        // The stage's engines first, so that a model entry naming one is served once the
        // stage is in; a refused registration leaves the stage out.
        for (const std::shared_ptr<Engine>& engine : stage.engines()) { register_engine(engine); }
        anira_error err{};
        const std::span<const anira_model_config* const> variants = stage.variants();
        const std::span<const BackendId> candidates = stage.candidates();
        detail::check(anira_pipeline_add_inference(m_pipeline,
                                                   variants.data(),
                                                   static_cast<uint32_t>(variants.size()),
                                                   candidates.empty() ? nullptr : candidates.data(),
                                                   static_cast<uint32_t>(candidates.size()),
                                                   &err),
                      err);
    }

    void add_stage(const stage::Custom& custom) {
        const std::shared_ptr<Stage>& implementation = custom.stage();
        if (implementation == nullptr) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT, "anira::stage::Custom: a null stage");
        }
        const uint32_t phases = implementation->phases();
        if ((phases & ~Stage::k_all_phases) != 0U) {
            throw Error(ANIRA_ERROR_INVALID_ARGUMENT,
                        "anira::Stage::phases: a bit that names none of the four phases");
        }
        const std::span<const char* const> kinds = implementation->consumed_kinds();
        anira_stage_desc desc = ANIRA_STAGE_DESC_INIT;
        desc.consumed_kinds = kinds.empty() ? nullptr : kinds.data();
        desc.num_consumed_kinds = static_cast<uint32_t>(kinds.size());
        desc.flags = implementation->flags();  // the C entry refuses a bit it does not define
        // A NULL phase slot means "not taking part": only the phases of the mask are filled, so
        // that a stage of other phases leaves anira's default pre_process / post_process running.
        if ((phases & Stage::k_pre_process) != 0U) {
            desc.pre_process = &detail::StageTrampolines::pre_process;
        }
        if ((phases & Stage::k_post_process) != 0U) {
            desc.post_process = &detail::StageTrampolines::post_process;
        }
        if ((phases & Stage::k_before_inference) != 0U) {
            desc.before_inference = &detail::StageTrampolines::before_inference;
        }
        if ((phases & Stage::k_after_inference) != 0U) {
            desc.after_inference = &detail::StageTrampolines::after_inference;
        }
        // The shared lifecycle: reset on the handler's Prepared (always filled, the base body
        // does nothing), prepare returning it, unprepare deleting it, init and release for the
        // registration.
        desc.reset = &detail::StageTrampolines::reset;
        desc.prepare = &detail::StageTrampolines::prepare;
        desc.unprepare = &detail::StageTrampolines::unprepare;
        desc.init = &detail::StageTrampolines::init;
        desc.release = &detail::StageTrampolines::release;
        // The carrier owns the copy from a successful add on and deletes it in release; a
        // refused add never calls release, so the copy dies here with the throw.
        auto holder = std::make_unique<std::shared_ptr<Stage>>(implementation);
        desc.user_data = holder.get();
        anira_error err{};
        detail::check(anira_pipeline_add_stage(m_pipeline, &desc, &err), err);
        [[maybe_unused]] const std::shared_ptr<Stage>* const carried = holder.release();
    }

    anira_pipeline* m_pipeline = nullptr;
};

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
