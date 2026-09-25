/**
 * @file compat/v2.hpp
 * @brief The 2.x source-compatible face of anira over the 3.x C entries: namespace anira::v2,
 * header-only C++20 over <anira/anira.hpp> and the anira/abi headers, nothing of the 2.x tree.
 *
 * A 2.x program compiles against this header by including it in place of <anira/anira.h> and
 * spelling the 2.x names in anira::v2 (a `using namespace anira::v2;` does it); it names the
 * classes of the last 2.x release, v2.3.0: anira::v2::ContextConfig and anira::v2::Context, the
 * loader's get_context_config(). Every class runs over the C handles, so what it does is what
 * anira 3 does; where that differs from 2.x the difference is stated on the member and on the
 * compat page of the documentation (docs/sphinx/migration.rst). Every failure is an
 * anira::Error, which derives from std::runtime_error (2.x threw std::invalid_argument or
 * asserted); a catch of std::exception still sees it.
 *
 * Scope at this pre-release: the configuration half (InferenceBackend and its conversions,
 * ModelData, TensorShape, ProcessingSpec, InferenceConfig, JsonConfigLoader, ContextConfig,
 * HostConfig and the log enums) and the processor half of the runtime (RingBuffer and BufferF as
 * views, PrePostProcessor, LegacyProcessorStage, which runs a 2.x processor as the stage of a
 * 3.x pipeline, PassthroughEngine, the 2.x base backend as an anira::Engine, and the static
 * Context) and the InferenceHandler over the C handler of anira/abi/handler.h.
 *
 * Deprecated from the start: every entity of namespace anira::v2 is removed one minor release
 * after 3.0.0; there is no [[deprecated]] attribute until then.
 *
 * Every entry is [main-thread] and may allocate, as anira.hpp's are, except what a phase of the
 * processor reaches: RingBuffer and BufferF, the processor's atomics (set_input, get_output and
 * their twins, any thread) and its five helpers are noexcept, allocate nothing and carry
 * ANIRA_NONBLOCKING; the processor's virtuals carry no attribute (a host's override decides),
 * and the stage's phases are noexcept and unattributed like every anira.hpp trampoline.
 */
#ifndef ANIRA_COMPAT_V2_HPP
#define ANIRA_COMPAT_V2_HPP

#include <anira/abi/config.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <istream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace anira::v2 {

// ---- backends ------------------------------------------------------------------------------

/// The id a 2.x custom backend runs under: the engine a CUSTOM row of a 2.x configuration names
/// (ANIRA_ENGINE_CUSTOM with this id), and the one id with the prefix "anira." a custom engine
/// may be created with. A 2.x custom backend is an anira::Engine constructed with it.
inline constexpr const char* k_custom_engine_id = "anira.v2.custom";

// NOLINTBEGIN(readability-identifier-naming) the enumerators spell the 2.x names
/**
 * @brief The 2.x backend enum under its 2.x names, each value the anira_engine value of its
 * engine, in the 3.x order: ONNX is ANIRA_ENGINE_ONNXRUNTIME, EXECUTORCH, LITERT, LIBTORCH and
 * TFLITE their engines', CUSTOM is ANIRA_ENGINE_CUSTOM. Every enumerator exists in every build,
 * whatever engines it carries. Source compatibility is in the names: the 2.x values shifted
 * with the engines of the build, so code that stored a backend as an integer or indexed an
 * array by it is not source-compatible. Unscoped, as in 2.x: TFLITE and
 * InferenceBackend::TFLITE both spell.
 */
enum InferenceBackend : uint32_t {
    ONNX = ANIRA_ENGINE_ONNXRUNTIME,
    EXECUTORCH = ANIRA_ENGINE_EXECUTORCH,
    LITERT = ANIRA_ENGINE_LITERT,
    LIBTORCH = ANIRA_ENGINE_LIBTORCH,
    TFLITE = ANIRA_ENGINE_TFLITE,
    CUSTOM = ANIRA_ENGINE_CUSTOM
};
// NOLINTEND(readability-identifier-naming)

static_assert(ONNX == static_cast<uint32_t>(ANIRA_ENGINE_ONNXRUNTIME));
static_assert(EXECUTORCH == static_cast<uint32_t>(ANIRA_ENGINE_EXECUTORCH));
static_assert(LITERT == static_cast<uint32_t>(ANIRA_ENGINE_LITERT));
static_assert(LIBTORCH == static_cast<uint32_t>(ANIRA_ENGINE_LIBTORCH));
static_assert(TFLITE == static_cast<uint32_t>(ANIRA_ENGINE_TFLITE));
static_assert(CUSTOM == static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM));

namespace detail {

/// Whether an engine value is one of the five built-in engines the enum names.
constexpr bool is_built_in(uint32_t value) noexcept {
    return value == ONNX || value == EXECUTORCH || value == LITERT || value == LIBTORCH ||
           value == TFLITE;
}

}  // namespace detail

/// The backend as the engine pair of anira 3: the same value as anira_engine, and for CUSTOM
/// the id k_custom_engine_id. No provider: every plan the shim builds is on the engine's CPU
/// path (ANIRA_PROVIDER_CPU). A value outside the enum is {ANIRA_ENGINE_NONE}.
constexpr anira::EngineRef to_engine(InferenceBackend backend) noexcept {
    if (backend == CUSTOM) {
        return anira::EngineRef{.kind = ANIRA_ENGINE_CUSTOM, .id = k_custom_engine_id};
    }
    if (!detail::is_built_in(backend)) { return anira::EngineRef{}; }
    return anira::EngineRef{.kind = static_cast<anira_engine>(backend), .id = {}};
}

/// The engine pair as a 2.x backend: a built-in engine's value as it is, ANIRA_ENGINE_CUSTOM as
/// CUSTOM whatever its id (to a 2.x processor every custom engine is CUSTOM), and anything else
/// (ANIRA_ENGINE_NONE, which no plan carries) as CUSTOM.
constexpr InferenceBackend to_backend(anira::EngineRef engine) noexcept {
    const auto value = static_cast<uint32_t>(engine.kind);
    return detail::is_built_in(value) ? static_cast<InferenceBackend>(value) : CUSTOM;
}

/// Whether an entry's or a plan's engine pair is this backend's: the engine equal, and for
/// CUSTOM the id equal to k_custom_engine_id.
constexpr bool names(anira::EngineRef engine, InferenceBackend backend) noexcept {
    const anira::EngineRef wanted = to_engine(backend);
    if (wanted.kind == ANIRA_ENGINE_NONE || engine.kind != wanted.kind) { return false; }
    return engine.kind != ANIRA_ENGINE_CUSTOM || engine.id == wanted.id;
}

/// Whether this build carries the backend's engine (anira::enabled_engines); CUSTOM always.
/// Allocates: main thread only.
inline bool is_available(InferenceBackend backend) {
    if (backend == CUSTOM) { return true; }
    const anira::EngineRef engine = to_engine(backend);
    if (engine.kind == ANIRA_ENGINE_NONE) { return false; }
    for (const anira::BackendId& row : anira::enabled_engines()) {
        if (row.engine == static_cast<uint32_t>(engine.kind)) { return true; }
    }
    return false;
}

namespace detail {

/// Throws an anira::Error with the status and this message (no entry name prefixed).
[[noreturn]] inline void fail(anira_status status, const std::string& message) {
    anira_error err = ANIRA_ERROR_INIT;
    err.status = status;
    std::snprintf(err.message, sizeof(err.message), "%s", message.c_str());
    throw anira::Error(status, err);
}

/// One record through anira_log under the shim's group.
inline void warn(const std::string& message) noexcept {
    anira_log(ANIRA_LOG_WARNING, "anira.compat", message.c_str());
}

/// The backends in the enum's order, and the index of each into a per-backend table.
inline constexpr std::size_t k_num_backends = 6;
inline constexpr std::array<InferenceBackend, k_num_backends> k_backends{ONNX,
                                                                         EXECUTORCH,
                                                                         LITERT,
                                                                         LIBTORCH,
                                                                         TFLITE,
                                                                         CUSTOM};
constexpr bool is_backend(InferenceBackend backend) noexcept {
    return backend == CUSTOM || is_built_in(backend);
}
/// The table index of a backend of the enum (is_backend).
constexpr std::size_t backend_index(InferenceBackend backend) noexcept {
    for (std::size_t i = 0; i < k_num_backends; ++i) {
        if (k_backends[i] == backend) { return i; }
    }
    return 0;
}

/// The word the version-2 document spells a backend with.
constexpr const char* backend_word(InferenceBackend backend) noexcept {
    switch (backend) {
        case LIBTORCH: return "LIBTORCH";
        case ONNX: return "ONNX";
        case TFLITE: return "TFLITE";
        case LITERT: return "LITERT";
        case EXECUTORCH: return "EXECUTORCH";
        case CUSTOM: return "CUSTOM";
    }
    return nullptr;
}

/// A JSON string literal: quotes, backslashes and control characters escaped.
inline void append_json_string(std::string& out, std::string_view text) {
    out.push_back('"');
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20U) {
                    std::array<char, 8> escaped{};
                    std::snprintf(escaped.data(),
                                  escaped.size(),
                                  "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += escaped.data();
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

/// A float as a JSON number that reads back as the same float: nine significant digits in the
/// classic locale (a host's LC_NUMERIC never turns the point into a comma).
inline std::string json_number(float value) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text.precision(std::numeric_limits<float>::max_digits10);
    text << value;
    return text.str();
}

}  // namespace detail

// ---- the context -----------------------------------------------------------------------------

/// How an idle inference thread waits for work.
enum class WaitStrategy { SpinBackoff, Blocking };

inline const char* to_string(WaitStrategy wait_strategy) noexcept {
    return wait_strategy == WaitStrategy::Blocking ? "blocking" : "spin_backoff";
}

/// The level of the records anira logs.
enum class LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3 };

inline const char* to_string(LogLevel log_level) noexcept {
    switch (log_level) {
        case LogLevel::Debug: return "debug";
        case LogLevel::Info: return "info";
        case LogLevel::Warning: return "warning";
        case LogLevel::Error: return "error";
    }
    return "unknown";
}

/// Info in a debug build, Error under NDEBUG: evaluated in the TU that includes this header,
/// as the 2.x inline was.
constexpr LogLevel default_log_level() noexcept {
#ifdef NDEBUG
    return LogLevel::Error;
#else
    return LogLevel::Info;
#endif
}

/// Who drains the real-time log queue: anira's drain thread, or the host (anira_drain_log).
enum class LogDrain { Thread = 0, Manual = 1 };

inline const char* to_string(LogDrain log_drain) noexcept {
    switch (log_drain) {
        case LogDrain::Thread: return "thread";
        case LogDrain::Manual: return "manual";
    }
    return "unknown";
}

/// The drain thread natively, the host under Emscripten.
constexpr LogDrain default_log_drain() noexcept {
#ifdef __EMSCRIPTEN__
    return LogDrain::Manual;
#else
    return LogDrain::Thread;
#endif
}

/// The logging part of a ContextConfig.
struct LogConfig {
    LogLevel m_level = default_log_level();
    LogDrain m_drain = default_log_drain();
    size_t m_queue_capacity = 512;      ///< real-time records; anira clamps it to [64, 65536]
    uint32_t m_drain_interval_ms = 10;  ///< the drain thread's period

    bool operator==(const LogConfig& other) const noexcept {
        return m_level == other.m_level && m_drain == other.m_drain &&
               m_queue_capacity == other.m_queue_capacity &&
               m_drain_interval_ms == other.m_drain_interval_ms;
    }
    bool operator!=(const LogConfig& other) const noexcept { return !(*this == other); }
};

/**
 * @brief The 2.x ContextConfig (v2.3.0's name; the pre-release that preceded anira 3 called it
 * CoreConfig): a plain value with the 2.x fields, minted into an anira::ContextConfig when a
 * handler is built (to_context_config). The default thread count is k_threads_auto, the
 * sentinel anira resolves (half the hardware threads, at least 1; 0 on WebAssembly) where 2.x
 * stored the resolved count: a host that read m_num_threads back to learn the count reads the
 * sentinel.
 */
struct ContextConfig {
    static constexpr unsigned k_threads_auto = ANIRA_THREADS_AUTO;

    ContextConfig(unsigned num_threads = k_threads_auto,
                  WaitStrategy wait_strategy = WaitStrategy::SpinBackoff,
                  LogLevel log_level = default_log_level())
        : m_num_threads(num_threads), m_wait_strategy(wait_strategy) {
        m_log.m_level = log_level;
    }

    /// The values a loaded context config holds (anira_context_config_threads and
    /// anira_context_config_log), e.g. one ContextConfig::from_json read from a 2.x document. A
    /// field the document left out reads anira's default: ANIRA_THREADS_AUTO threads, and the
    /// Warning level where 2.x took default_log_level().
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} on an empty handle.
    explicit ContextConfig(const anira::ContextConfig& loaded) {
        uint32_t num_threads = 0;
        anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF;
        anira::detail::check(anira_context_config_threads(loaded.native(), &num_threads, &wait),
                             "anira_context_config_threads");
        anira_log_desc log = ANIRA_LOG_DESC_INIT;
        anira::detail::check(anira_context_config_log(loaded.native(), &log),
                             "anira_context_config_log");
        m_num_threads = num_threads;
        m_wait_strategy =
            wait == ANIRA_WAIT_BLOCKING ? WaitStrategy::Blocking : WaitStrategy::SpinBackoff;
        switch (log.level) {
            case ANIRA_LOG_DEBUG: m_log.m_level = LogLevel::Debug; break;
            case ANIRA_LOG_INFO: m_log.m_level = LogLevel::Info; break;
            case ANIRA_LOG_ERROR: m_log.m_level = LogLevel::Error; break;
            default: m_log.m_level = LogLevel::Warning; break;
        }
        m_log.m_drain = log.drain == ANIRA_LOG_DRAIN_MANUAL ? LogDrain::Manual : LogDrain::Thread;
        m_log.m_queue_capacity = log.queue_capacity;
        m_log.m_drain_interval_ms = log.drain_interval_ms;
    }

    unsigned m_num_threads = k_threads_auto;
    WaitStrategy m_wait_strategy = WaitStrategy::SpinBackoff;
    LogConfig m_log;

    /// The anira::ContextConfig of these values: the threads with the wait strategy, the log
    /// level, the drain with its interval and the queue capacity; no sink, no flags, no device.
    anira::ContextConfig to_context_config() const {
        anira::ContextConfig config;
        config.threads(m_num_threads,
                       m_wait_strategy == WaitStrategy::Blocking ? ANIRA_WAIT_BLOCKING
                                                                 : ANIRA_WAIT_SPIN_BACKOFF);
        anira_log_level level = ANIRA_LOG_WARNING;
        switch (m_log.m_level) {
            case LogLevel::Debug: level = ANIRA_LOG_DEBUG; break;
            case LogLevel::Info: level = ANIRA_LOG_INFO; break;
            case LogLevel::Warning: level = ANIRA_LOG_WARNING; break;
            case LogLevel::Error: level = ANIRA_LOG_ERROR; break;
        }
        config.log_level(level);
        config.log_drain(
            m_log.m_drain == LogDrain::Manual ? ANIRA_LOG_DRAIN_MANUAL : ANIRA_LOG_DRAIN_THREAD,
            m_log.m_drain_interval_ms);
        config.log_queue_capacity(static_cast<uint32_t>(
            std::min<size_t>(m_log.m_queue_capacity, std::numeric_limits<uint32_t>::max())));
        return config;
    }
};

/**
 * @brief The 2.x HostConfig: the host's block size and sample rate in samples of the reference
 * stream, whether smaller blocks come, and which tensor is the reference stream
 * (k_first_streamable: the first Streamed input, else the first Streamed output). The handler's
 * prepare refuses a fractional m_buffer_size (ANIRA_ERROR_CONFIG): a host that scaled its block
 * to another stream anchors on that stream instead (m_tensor_index, m_tensor_is_input) and
 * passes its own block. The session's arithmetic of 2.x (resolve_reference,
 * get_reference_size, get_relative_buffer_size, get_relative_sample_rate) is not declared.
 */
struct HostConfig {
    static constexpr size_t k_first_streamable = static_cast<size_t>(-1);

    HostConfig() = default;
    HostConfig(float host_buffer_size,
               float host_sample_rate,
               bool allow_smaller_buffers = false,
               size_t tensor_index = k_first_streamable,
               bool tensor_is_input = true)
        : m_buffer_size(host_buffer_size)
        , m_sample_rate(host_sample_rate)
        , m_allow_smaller_buffers(allow_smaller_buffers)
        , m_tensor_index(tensor_index)
        , m_tensor_is_input(tensor_is_input) {}

    float m_buffer_size = 0;                     ///< the host's largest block, a whole number
    float m_sample_rate = 0;                     ///< Hz
    bool m_allow_smaller_buffers = false;        ///< blocks below m_buffer_size may come
    size_t m_tensor_index = k_first_streamable;  ///< the reference tensor's index
    bool m_tensor_is_input = true;               ///< the side m_tensor_index counts on

    /// Equal within 1e-6 on the two floats, as in 2.x.
    bool operator==(const HostConfig& other) const noexcept {
        return std::abs(m_buffer_size - other.m_buffer_size) < 1e-6F &&
               std::abs(m_sample_rate - other.m_sample_rate) < 1e-6F &&
               m_allow_smaller_buffers == other.m_allow_smaller_buffers &&
               m_tensor_index == other.m_tensor_index &&
               m_tensor_is_input == other.m_tensor_is_input;
    }
    bool operator!=(const HostConfig& other) const noexcept { return !(*this == other); }
};

// ---- the configuration -----------------------------------------------------------------------

using TensorShapeList = std::vector<std::vector<int64_t>>;

/**
 * @brief One model entry of a 2.x configuration: a path (m_is_binary false; the characters are
 * owned by the object) or model bytes (m_is_binary true; borrowed, the caller's buffer outlives
 * every configuration and handler built from it, as in 2.x), the backend, and the entry point
 * of a LibTorch or ExecuTorch file.
 */
struct ModelData {
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for NULL data or a size of 0 (2.x asserted).
    ModelData(void* data,
              size_t size,
              InferenceBackend backend,
              std::string model_function = "",
              bool is_binary = true)
        : m_data(data)
        , m_size(size)
        , m_backend(backend)
        , m_model_function(std::move(model_function))
        , m_is_binary(is_binary) {
        if (data == nullptr || size == 0) {
            detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                         "anira::v2::ModelData: the data pointer is null or the size is 0");
        }
        if (!m_is_binary) {
            m_path.assign(static_cast<const char*>(data), size);
            m_data = m_path.data();
        }
    }
    ModelData(const std::string& model_path,
              InferenceBackend backend,
              const std::string& model_function = "",
              bool is_binary = false)
        : ModelData(const_cast<char*>(model_path.data()),
                    model_path.size(),
                    backend,
                    model_function,
                    is_binary) {}

    ModelData(const ModelData& other)
        : m_data(other.m_data)
        , m_size(other.m_size)
        , m_backend(other.m_backend)
        , m_model_function(other.m_model_function)
        , m_is_binary(other.m_is_binary)
        , m_path(other.m_path) {
        repoint();
    }
    ModelData(ModelData&& other) noexcept
        : m_data(other.m_data)
        , m_size(other.m_size)
        , m_backend(other.m_backend)
        , m_model_function(std::move(other.m_model_function))
        , m_is_binary(other.m_is_binary)
        , m_path(std::move(other.m_path)) {
        repoint();
    }
    ModelData& operator=(const ModelData& other) {
        if (this != &other) {
            m_data = other.m_data;
            m_size = other.m_size;
            m_backend = other.m_backend;
            m_model_function = other.m_model_function;
            m_is_binary = other.m_is_binary;
            m_path = other.m_path;
            repoint();
        }
        return *this;
    }
    ModelData& operator=(ModelData&& other) noexcept {
        if (this != &other) {
            m_data = other.m_data;
            m_size = other.m_size;
            m_backend = other.m_backend;
            m_model_function = std::move(other.m_model_function);
            m_is_binary = other.m_is_binary;
            m_path = std::move(other.m_path);
            repoint();
        }
        return *this;
    }
    ~ModelData() = default;

    void* m_data;                  ///< the bytes, or the path's characters (not NUL-terminated)
    size_t m_size;                 ///< the byte count, or the path's length
    InferenceBackend m_backend;    ///< the backend that runs the entry
    std::string m_model_function;  ///< the entry point (LibTorch, ExecuTorch); empty = default
    bool m_is_binary;              ///< bytes (true) or a path (false)

    /// The 2.x rule: size, backend and kind equal, then the bytes by pointer or the path by
    /// content; the entry point is not compared.
    bool operator==(const ModelData& other) const noexcept {
        if (m_size != other.m_size || m_backend != other.m_backend ||
            m_is_binary != other.m_is_binary) {
            return false;
        }
        if (m_is_binary) { return m_data == other.m_data; }
        return std::string_view(static_cast<const char*>(m_data), m_size) ==
               std::string_view(static_cast<const char*>(other.m_data), other.m_size);
    }
    bool operator!=(const ModelData& other) const noexcept { return !(*this == other); }

private:
    void repoint() noexcept {
        if (!m_is_binary) { m_data = m_path.data(); }
    }

    std::string m_path;  ///< a path entry's characters, which m_data points at
};

/**
 * @brief The tensor shapes of a 2.x configuration: universal (every backend) or of one backend,
 * whose row must hold the universal shapes with their axes permuted or unit axes inserted
 * (anira 3 keeps one shape per tensor and an axis layout per engine; a row that reshapes
 * is refused with ANIRA_ERROR_JSON when the configuration is built).
 */
struct TensorShape {
    TensorShapeList m_tensor_input_shape;
    TensorShapeList m_tensor_output_shape;
    InferenceBackend m_backend = CUSTOM;  ///< a backend row's backend; CUSTOM on a universal row
    bool m_universal = false;

    TensorShape() = delete;
    /// A universal row. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an empty list.
    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape)
        : m_tensor_input_shape(std::move(input_shape))
        , m_tensor_output_shape(std::move(output_shape))
        , m_universal(true) {
        check();
    }
    /// A row of one backend. @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an empty list.
    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape, InferenceBackend backend)
        : m_tensor_input_shape(std::move(input_shape))
        , m_tensor_output_shape(std::move(output_shape))
        , m_backend(backend) {
        check();
    }

    bool is_universal() const noexcept { return m_universal; }

    /// The 2.x rule: two universal rows by their shapes, two backend rows by their shapes and
    /// backend, never one of each.
    bool operator==(const TensorShape& other) const noexcept {
        if (m_universal != other.m_universal) { return false; }
        return m_tensor_input_shape == other.m_tensor_input_shape &&
               m_tensor_output_shape == other.m_tensor_output_shape &&
               (m_universal || m_backend == other.m_backend);
    }
    bool operator!=(const TensorShape& other) const noexcept { return !(*this == other); }

private:
    void check() const {
        if (m_tensor_input_shape.empty() || m_tensor_output_shape.empty()) {
            detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                         "anira::v2::TensorShape: at least one input and one output shape are "
                         "required");
        }
    }
};

/**
 * @brief The 2.x processing specification: per input the channels and the samples per
 * inference (0 = a non-streamable tensor), per output likewise and the model's internal
 * latency; the element counts are computed. An empty vector takes the 2.x default (1 channel,
 * the per-channel element count, latency 0).
 */
struct ProcessingSpec {
    std::vector<size_t> m_preprocess_input_channels;
    std::vector<size_t> m_postprocess_output_channels;
    std::vector<size_t> m_preprocess_input_size;
    std::vector<size_t> m_postprocess_output_size;
    std::vector<size_t> m_internal_model_latency;
    std::vector<size_t> m_tensor_input_size;   ///< computed: the elements of each input
    std::vector<size_t> m_tensor_output_size;  ///< computed: the elements of each output

    ProcessingSpec() = default;
    ProcessingSpec(std::vector<size_t> preprocess_input_channels,
                   std::vector<size_t> preprocess_output_channels,
                   std::vector<size_t> preprocess_input_size,
                   std::vector<size_t> postprocess_output_size,
                   std::vector<size_t> internal_model_latency)
        : m_preprocess_input_channels(std::move(preprocess_input_channels))
        , m_postprocess_output_channels(std::move(preprocess_output_channels))
        , m_preprocess_input_size(std::move(preprocess_input_size))
        , m_postprocess_output_size(std::move(postprocess_output_size))
        , m_internal_model_latency(std::move(internal_model_latency)) {}
    ProcessingSpec(std::vector<size_t> preprocess_input_channels,
                   std::vector<size_t> preprocess_output_channels)
        : ProcessingSpec(std::move(preprocess_input_channels),
                         std::move(preprocess_output_channels),
                         {},
                         {},
                         {}) {}
    ProcessingSpec(std::vector<size_t> preprocess_input_channels,
                   std::vector<size_t> preprocess_output_channels,
                   std::vector<size_t> preprocess_input_size,
                   std::vector<size_t> postprocess_output_size)
        : ProcessingSpec(std::move(preprocess_input_channels),
                         std::move(preprocess_output_channels),
                         std::move(preprocess_input_size),
                         std::move(postprocess_output_size),
                         {}) {}

    /// The five given vectors; the computed sizes are not compared (2.x).
    bool operator==(const ProcessingSpec& other) const noexcept {
        return m_preprocess_input_channels == other.m_preprocess_input_channels &&
               m_postprocess_output_channels == other.m_postprocess_output_channels &&
               m_preprocess_input_size == other.m_preprocess_input_size &&
               m_postprocess_output_size == other.m_postprocess_output_size &&
               m_internal_model_latency == other.m_internal_model_latency;
    }
    bool operator!=(const ProcessingSpec& other) const noexcept { return !(*this == other); }
};

class JsonConfigLoader;

/**
 * @brief The 2.x InferenceConfig as a value over an anira::ModelConfig: the constructors write
 * their arguments as a version-2 document and load it through ModelConfig::from_json, the one
 * upgrade path (a bytes entry borrowed, its path the placeholder "anira:bytes"), and take the
 * legacy Hard contract; the 2.x public fields and getters are read back from the handle.
 *
 * What the constructors normalise as 2.x did: max_inference_time must be positive
 * (ANIRA_ERROR_CONFIG, first); a session-exclusive configuration runs one processor; a
 * processor count below 1 is 1 (with a Warning); a non-empty ProcessingSpec vector with the
 * wrong entry count is dropped and defaulted (2.x did so silently, the shim with a Warning); a
 * non-streamable tensor with more than one channel is refused (ANIRA_ERROR_CONFIG); a
 * model_function on a backend other than LIBTORCH or EXECUTORCH is dropped (with the 2.x
 * message as a Warning); a backend row of a backend without a model entry is dropped. What
 * anira 3 refuses where 2.x ran: a backend row that is no axis permutation of the universal one
 * (ANIRA_ERROR_JSON), a streamed tensor with channels > 1 and no axis of that extent
 * (ANIRA_ERROR_JSON). A backend this build lacks stays an entry, which a handler skips.
 *
 * Copyable (the copy loads its document again, the bytes borrowed again) and movable;
 * operator== compares the configurations anira 3 holds (the model file, the three legacy
 * scalars, the borrowed bytes by pointer and size), not the spelling.
 */
class InferenceConfig {
public:
    struct Defaults {
        static constexpr unsigned k_warm_up = 0;
        static constexpr bool k_session_exclusive_processor = false;
        static constexpr float k_blocking_ratio = 0.F;
        /// Half the hardware threads, at least 1. A function where 2.x had a static member: a
        /// header has no out-of-line definition, and an inline variable would be bound
        /// STB_GNU_UNIQUE by GCC, which pins a plugin in memory.
        static unsigned num_parallel_processors() noexcept {
            const unsigned hardware = std::thread::hardware_concurrency();
            return hardware / 2 > 0 ? hardware / 2 : 1;
        }
    };

    /// An empty configuration: every getter answers empty, model_config() throws
    /// ANIRA_ERROR_INVALID_STATE.
    InferenceConfig() = default;
    /// @throws Error with the status of the normalisation or of ModelConfig::from_json.
    InferenceConfig(std::vector<ModelData> model_data,
                    std::vector<TensorShape> tensor_shape,
                    ProcessingSpec processing_spec,
                    float max_inference_time,
                    unsigned warm_up = Defaults::k_warm_up,
                    bool session_exclusive_processor = Defaults::k_session_exclusive_processor,
                    float blocking_ratio = Defaults::k_blocking_ratio,
                    unsigned num_parallel_processors = Defaults::num_parallel_processors()) {
        m_document = document_of(std::move(model_data),
                                 std::move(tensor_shape),
                                 std::move(processing_spec),
                                 max_inference_time,
                                 warm_up,
                                 session_exclusive_processor,
                                 blocking_ratio,
                                 num_parallel_processors,
                                 m_borrowed);
        load();
    }
    InferenceConfig(std::vector<ModelData> model_data,
                    std::vector<TensorShape> tensor_shape,
                    float max_inference_time,
                    unsigned warm_up = Defaults::k_warm_up,
                    bool session_exclusive_processor = Defaults::k_session_exclusive_processor,
                    float blocking_ratio = Defaults::k_blocking_ratio,
                    unsigned num_parallel_processors = Defaults::num_parallel_processors())
        : InferenceConfig(std::move(model_data),
                          std::move(tensor_shape),
                          ProcessingSpec(),
                          max_inference_time,
                          warm_up,
                          session_exclusive_processor,
                          blocking_ratio,
                          num_parallel_processors) {}

    /// Loads the document again: an equal configuration over its own handles.
    InferenceConfig(const InferenceConfig& other)
        : m_document(other.m_document), m_base_dir(other.m_base_dir), m_borrowed(other.m_borrowed) {
        if (!m_document.empty()) { load(); }
    }
    InferenceConfig& operator=(const InferenceConfig& other) {
        if (this != &other) {
            InferenceConfig copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    InferenceConfig(InferenceConfig&&) noexcept = default;
    InferenceConfig& operator=(InferenceConfig&&) noexcept = default;
    ~InferenceConfig() = default;

    // -- the model entries --

    /// The first entry of the backend: its path, or its bytes as a string (2.x); "" without one.
    std::string get_model_path(InferenceBackend backend) const {
        const ModelData* entry = get_model_data(backend);
        if (entry == nullptr) { return ""; }
        return {static_cast<const char*>(entry->m_data), entry->m_size};
    }
    /// The first entry of the backend (a pointer into m_model_data); nullptr without one.
    const ModelData* get_model_data(InferenceBackend backend) const {
        for (const ModelData& entry : m_model_data) {
            if (entry.m_backend == backend) { return &entry; }
        }
        return nullptr;
    }
    std::string get_model_function(InferenceBackend backend) const {
        const ModelData* entry = get_model_data(backend);
        return entry != nullptr ? entry->m_model_function : std::string();
    }
    bool is_model_binary(InferenceBackend backend) const {
        const ModelData* entry = get_model_data(backend);
        return entry != nullptr && entry->m_is_binary;
    }

    // -- the tensor shapes --

    /// The universal shapes: anira 3's one shape per tensor.
    const TensorShapeList& get_tensor_input_shape() const noexcept { return m_input_shape; }
    const TensorShapeList& get_tensor_output_shape() const noexcept { return m_output_shape; }
    /// The shapes the backend's first entry holds: the universal ones with that entry's axis
    /// layout applied (a channels-last TFLite row reads channels last), the universal ones for
    /// a backend without an entry or a layout.
    const TensorShapeList& get_tensor_input_shape(InferenceBackend backend) const noexcept {
        return detail::is_backend(backend) ? m_input_shapes[detail::backend_index(backend)]
                                           : m_input_shape;
    }
    const TensorShapeList& get_tensor_output_shape(InferenceBackend backend) const noexcept {
        return detail::is_backend(backend) ? m_output_shapes[detail::backend_index(backend)]
                                           : m_output_shape;
    }

    // -- the processing specification --

    const std::vector<size_t>& get_tensor_input_size() const noexcept {
        return m_processing_spec.m_tensor_input_size;
    }
    const std::vector<size_t>& get_tensor_output_size() const noexcept {
        return m_processing_spec.m_tensor_output_size;
    }
    const std::vector<size_t>& get_preprocess_input_channels() const noexcept {
        return m_processing_spec.m_preprocess_input_channels;
    }
    const std::vector<size_t>& get_postprocess_output_channels() const noexcept {
        return m_processing_spec.m_postprocess_output_channels;
    }
    const std::vector<size_t>& get_preprocess_input_size() const noexcept {
        return m_processing_spec.m_preprocess_input_size;
    }
    const std::vector<size_t>& get_postprocess_output_size() const noexcept {
        return m_processing_spec.m_postprocess_output_size;
    }
    const std::vector<size_t>& get_internal_model_latency() const noexcept {
        return m_processing_spec.m_internal_model_latency;
    }

    // -- the 2.x public fields, read back from the handles --

    std::vector<ModelData> m_model_data;      ///< one per model entry, in entry order
    std::vector<TensorShape> m_tensor_shape;  ///< the universal row, the rows of the entries
                                              ///< with a layout, a universal clone per other
                                              ///< backend of an entry
    ProcessingSpec m_processing_spec;
    float m_max_inference_time = 0;              ///< ms, the legacy contract's explicit budget
    unsigned m_warm_up = 0;                      ///< the legacy contract's fixed warm-up count
    bool m_session_exclusive_processor = false;  ///< the model's state is STATEFUL
    float m_blocking_ratio = 0;                  ///< the legacy contract's wait ratio
    unsigned m_num_parallel_processors = 0;      ///< the model's max_instances; 1 when exclusive

    /// The configurations anira 3 holds: the model files (ModelConfig::to_json), the three
    /// legacy scalars, the borrowed bytes by pointer and size.
    bool operator==(const InferenceConfig& other) const {
        if (m_model.has_value() != other.m_model.has_value()) { return false; }
        if (!m_model.has_value()) { return true; }
        if (m_borrowed.size() != other.m_borrowed.size()) { return false; }
        for (size_t i = 0; i < m_borrowed.size(); ++i) {
            if (m_borrowed[i].first != other.m_borrowed[i].first ||
                m_borrowed[i].second.data() != other.m_borrowed[i].second.data() ||
                m_borrowed[i].second.size() != other.m_borrowed[i].second.size()) {
                return false;
            }
        }
        return std::abs(m_max_inference_time - other.m_max_inference_time) < 1e-6F &&
               m_warm_up == other.m_warm_up &&
               std::abs(m_blocking_ratio - other.m_blocking_ratio) < 1e-6F &&
               m_model->to_json() == other.m_model->to_json();
    }
    bool operator!=(const InferenceConfig& other) const { return !(*this == other); }

    // -- what the runtime reads --

    /// The model configuration, bytes entries borrowed (the caller's buffer outlives this
    /// object and every handler built from it, as in 2.x).
    /// @throws Error{ANIRA_ERROR_INVALID_STATE} on an empty configuration.
    const anira::ModelConfig& model_config() const {
        if (!m_model.has_value()) {
            detail::fail(ANIRA_ERROR_INVALID_STATE,
                         "anira::v2::InferenceConfig::model_config: an empty configuration");
        }
        return *m_model;
    }
    /// A fresh ModelConfig of the same document, bytes entries borrowed again: a handler's
    /// private copy, whose anchor it may set without touching this configuration.
    /// @throws Error{ANIRA_ERROR_INVALID_STATE} on an empty configuration.
    anira::ModelConfig model_config_copy() const {
        if (m_document.empty()) {
            detail::fail(ANIRA_ERROR_INVALID_STATE,
                         "anira::v2::InferenceConfig::model_config_copy: an empty configuration");
        }
        anira::ModelConfig copy = anira::ModelConfig::from_json(m_document, m_base_dir);
        for (const auto& [row, bytes] : m_borrowed) {
            copy.set_model_bytes(row, bytes, ANIRA_BYTES_BORROW);
        }
        return copy;
    }
    /// The legacy Hard contract read back: the explicit budget (max_inference_time), the fixed
    /// warm-up, the wait ratio (blocking_ratio), ANIRA_MISS_ZEROS; no geometry, which the
    /// handler's prepare sets.
    /// @throws Error{ANIRA_ERROR_INVALID_STATE} on an empty configuration.
    anira::Hard hard() const {
        if (!m_legacy.has_value()) {
            detail::fail(ANIRA_ERROR_INVALID_STATE,
                         "anira::v2::InferenceConfig::hard: an empty configuration");
        }
        return m_legacy->hard();
    }

private:
    friend class JsonConfigLoader;

    /// A configuration of a version-2 document's text; relative paths resolve against
    /// base_dir. @throws Error{ANIRA_ERROR_CONFIG} for a document that is not version 2.
    static InferenceConfig from_document(std::string document, std::string base_dir) {
        InferenceConfig config;
        config.m_document = std::move(document);
        config.m_base_dir = std::move(base_dir);
        config.load();
        return config;
    }

    /// The document of the constructors' arguments, normalised as 2.x did (see the class
    /// comment); the bytes entries land in `borrowed` by entry index.
    static std::string document_of(
        std::vector<ModelData> model_data,
        std::vector<TensorShape> tensor_shape,
        ProcessingSpec spec,
        float max_inference_time,
        unsigned warm_up,
        bool session_exclusive_processor,
        float blocking_ratio,
        unsigned num_parallel_processors,
        std::vector<std::pair<uint32_t, std::span<const std::byte>>>& borrowed) {
        if (!(max_inference_time > 0.F) || !std::isfinite(max_inference_time)) {
            detail::fail(ANIRA_ERROR_CONFIG,
                         "max_inference_time must be greater than 0, got " +
                             detail::json_number(max_inference_time));
        }
        if (!(blocking_ratio >= 0.F) || !std::isfinite(blocking_ratio)) {
            detail::fail(ANIRA_ERROR_CONFIG,
                         "blocking_ratio must be a finite number of at least 0, got " +
                             detail::json_number(blocking_ratio));
        }
        if (tensor_shape.empty()) {
            detail::fail(ANIRA_ERROR_CONFIG, "at least one tensor shape must be provided");
        }
        if (session_exclusive_processor) { num_parallel_processors = 1; }
        if (num_parallel_processors < 1) {
            num_parallel_processors = 1;
            detail::warn("Number of parallel processors must be at least 1. Setting to 1.");
        }

        // The canonical row: the universal one, else the first (what the upgrade takes).
        size_t canonical = 0;
        for (size_t i = 0; i < tensor_shape.size(); ++i) {
            if (tensor_shape[i].m_universal) {
                canonical = i;
                break;
            }
        }
        const size_t num_inputs = tensor_shape[canonical].m_tensor_input_shape.size();
        const size_t num_outputs = tensor_shape[canonical].m_tensor_output_shape.size();
        const auto per_tensor =
            [](std::vector<size_t>& values, size_t count, const char* name, const char* side) {
                if (!values.empty() && values.size() != count) {
                    detail::warn(std::string(name) + " has " + std::to_string(values.size()) +
                                 " entries; the model has " + std::to_string(count) + " " + side +
                                 " tensors: the default is used");
                    values.clear();
                }
            };
        per_tensor(spec.m_preprocess_input_channels,
                   num_inputs,
                   "preprocess_input_channels",
                   "input");
        per_tensor(spec.m_postprocess_output_channels,
                   num_outputs,
                   "postprocess_output_channels",
                   "output");
        per_tensor(spec.m_preprocess_input_size, num_inputs, "preprocess_input_size", "input");
        per_tensor(spec.m_postprocess_output_size,
                   num_outputs,
                   "postprocess_output_size",
                   "output");
        per_tensor(spec.m_internal_model_latency, num_outputs, "internal_model_latency", "output");
        const auto non_streamable = [](const std::vector<size_t>& sizes,
                                       const std::vector<size_t>& channels,
                                       const char* side,
                                       const char* key) {
            for (size_t i = 0; i < sizes.size(); ++i) {
                const size_t count = channels.empty() ? 1 : channels[i];
                if (sizes[i] == 0 && count != 1) {
                    detail::fail(ANIRA_ERROR_CONFIG,
                                 std::string(side) + " tensor " + std::to_string(i) +
                                     " is non-streamable (" + key + " 0) but has " +
                                     std::to_string(count) +
                                     " channels; a non-streamable tensor has exactly 1");
                }
            }
        };
        non_streamable(spec.m_preprocess_input_size,
                       spec.m_preprocess_input_channels,
                       "input",
                       "preprocess_input_size");
        non_streamable(spec.m_postprocess_output_size,
                       spec.m_postprocess_output_channels,
                       "output",
                       "postprocess_output_size");

        std::string doc = R"({"inference_config":{"model_data":[)";
        for (size_t i = 0; i < model_data.size(); ++i) {
            ModelData& entry = model_data[i];
            const char* word = detail::backend_word(entry.m_backend);
            if (word == nullptr) {
                detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                             "model_data[" + std::to_string(i) +
                                 "]: the backend is no value of anira::v2::InferenceBackend");
            }
            if (!entry.m_model_function.empty() && entry.m_backend != LIBTORCH &&
                entry.m_backend != EXECUTORCH) {
                detail::warn(
                    "Model function is only applicable to the LIBTORCH and EXECUTORCH backends.");
                entry.m_model_function.clear();
            }
            if (i > 0) { doc += ','; }
            doc += R"({"model_path":)";
            if (entry.m_is_binary) {
                detail::append_json_string(doc, "anira:bytes");
                borrowed.emplace_back(
                    static_cast<uint32_t>(i),
                    std::span<const std::byte>(static_cast<const std::byte*>(entry.m_data),
                                               entry.m_size));
            } else {
                detail::append_json_string(
                    doc,
                    std::string_view(static_cast<const char*>(entry.m_data), entry.m_size));
            }
            doc += R"(,"inference_backend":)";
            detail::append_json_string(doc, word);
            if (!entry.m_model_function.empty()) {
                doc += R"(,"model_function":)";
                detail::append_json_string(doc, entry.m_model_function);
            }
            doc += '}';
        }
        doc += R"(],"tensor_shape":[)";
        const auto has_entry = [&model_data](InferenceBackend backend) {
            for (const ModelData& entry : model_data) {
                if (entry.m_backend == backend) { return true; }
            }
            return false;
        };
        const auto append_shapes = [&doc](const TensorShapeList& shapes) {
            doc += '[';
            for (size_t t = 0; t < shapes.size(); ++t) {
                if (t > 0) { doc += ','; }
                doc += '[';
                for (size_t d = 0; d < shapes[t].size(); ++d) {
                    if (d > 0) { doc += ','; }
                    doc += std::to_string(shapes[t][d]);
                }
                doc += ']';
            }
            doc += ']';
        };
        bool first_row = true;
        for (size_t i = 0; i < tensor_shape.size(); ++i) {
            const TensorShape& row = tensor_shape[i];
            if (i != canonical && !row.m_universal && !has_entry(row.m_backend)) {
                continue;  // a backend row no entry reads
            }
            if (!row.m_universal && detail::backend_word(row.m_backend) == nullptr) {
                detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                             "tensor_shape[" + std::to_string(i) +
                                 "]: the backend is no value of anira::v2::InferenceBackend");
            }
            if (!first_row) { doc += ','; }
            first_row = false;
            doc += R"({"input_shape":)";
            append_shapes(row.m_tensor_input_shape);
            doc += R"(,"output_shape":)";
            append_shapes(row.m_tensor_output_shape);
            if (!row.m_universal) {
                doc += R"(,"inference_backend":)";
                detail::append_json_string(doc, detail::backend_word(row.m_backend));
            }
            doc += '}';
        }
        doc += R"(],"processing_spec":{)";
        bool first_key = true;
        const auto append_list = [&doc, &first_key](const char* key,
                                                    const std::vector<size_t>& values) {
            if (values.empty()) { return; }
            if (!first_key) { doc += ','; }
            first_key = false;
            detail::append_json_string(doc, key);
            doc += ":[";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i > 0) { doc += ','; }
                doc += std::to_string(values[i]);
            }
            doc += ']';
        };
        append_list("preprocess_input_channels", spec.m_preprocess_input_channels);
        append_list("postprocess_output_channels", spec.m_postprocess_output_channels);
        append_list("preprocess_input_size", spec.m_preprocess_input_size);
        append_list("postprocess_output_size", spec.m_postprocess_output_size);
        append_list("internal_model_latency", spec.m_internal_model_latency);
        doc += R"(},"max_inference_time":)";
        doc += detail::json_number(max_inference_time);
        doc += R"(,"warm_up":)";
        doc += std::to_string(warm_up);
        doc += R"(,"blocking_ratio":)";
        doc += detail::json_number(blocking_ratio);
        doc += R"(,"session_exclusive_processor":)";
        doc += session_exclusive_processor ? "true" : "false";
        doc += R"(,"num_parallel_processors":)";
        doc += std::to_string(num_parallel_processors);
        doc += "}}";
        return doc;
    }

    /// The handles of the document, the model config (bytes entries borrowed) and its legacy
    /// Hard contract, and the 2.x fields read back from them. @throws Error;
    /// ANIRA_ERROR_CONFIG for a document that is not version 2.
    void load() {
        anira::ModelConfig model = anira::ModelConfig::from_json(m_document, m_base_dir);
        for (const auto& [row, bytes] : m_borrowed) {
            model.set_model_bytes(row, bytes, ANIRA_BYTES_BORROW);
        }
        std::optional<anira::ContractHandle> legacy = model.take_legacy_contract();
        if (!legacy.has_value()) {
            detail::fail(ANIRA_ERROR_CONFIG,
                         "not a version 2 document (no inference_config root): a 3.x model file "
                         "loads through anira::ModelConfig::from_file");
        }
        read_back(model, *legacy);
        m_model.emplace(std::move(model));
        m_legacy.emplace(std::move(*legacy));
    }

    /// The canonical dims of a spec: the extents, a dynamic Time extent read as the window.
    static std::vector<int64_t> dims_of(const anira::SpecView& spec) {
        std::vector<int64_t> dims(spec.ndim());
        for (uint32_t k = 0; k < spec.ndim(); ++k) {
            const int64_t extent = spec.axis(k).extent;
            dims[k] = extent > 0 ? extent : spec.window().min;
        }
        return dims;
    }

    /// The dims an entry holds a tensor in: the canonical dims permuted by its layout, a unit
    /// axis where the layout inserts one.
    static std::vector<int64_t> laid_out(const std::vector<int64_t>& dims,
                                         const std::vector<uint32_t>& layout) {
        if (layout.empty()) { return dims; }
        std::vector<int64_t> out(layout.size());
        for (size_t k = 0; k < layout.size(); ++k) {
            out[k] =
                layout[k] == ANIRA_AXIS_INSERT || layout[k] >= dims.size() ? 1 : dims[layout[k]];
        }
        return out;
    }

    /// The 2.x fields and the per-backend shapes, from the handles.
    void read_back(const anira::ModelConfig& model, const anira::ContractHandle& contract) {
        const uint32_t num_entries = model.model_count();

        m_model_data.clear();
        m_model_data.reserve(num_entries);
        for (uint32_t e = 0; e < num_entries; ++e) {
            const InferenceBackend backend = to_backend(model.model_engine(e));
            const std::optional<anira::ext::Entry> function = model.model_ext<anira::ext::Entry>(e);
            const std::string name = function.has_value() ? function->name : std::string();
            const char* path = anira_model_config_model_path(model.native(), e);
            if (path != nullptr) {
                m_model_data.emplace_back(std::string(path), backend, name, false);
            } else {
                const std::span<const std::byte> bytes = model.model_bytes(e);
                m_model_data.emplace_back(const_cast<std::byte*>(bytes.data()),
                                          bytes.size(),
                                          backend,
                                          name,
                                          true);
            }
        }

        ProcessingSpec spec;
        std::vector<std::vector<int64_t>> inputs;
        std::vector<std::vector<int64_t>> outputs;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        const auto channels_of = [](const anira::SpecView& view) -> size_t {
            if (view.role() != ANIRA_ROLE_STREAMED) { return 1; }
            for (uint32_t k = 0; k < view.ndim(); ++k) {
                const anira::SpecView::Axis axis = view.axis(k);
                if (axis.tag == ANIRA_AXIS_CHANNEL && axis.extent > 0) {
                    return static_cast<size_t>(axis.extent);
                }
            }
            return 1;
        };
        const auto size_of = [](const anira::SpecView& view) -> size_t {
            if (view.role() != ANIRA_ROLE_STREAMED) { return 0; }
            const anira::SpecView::Window window = view.window();
            return static_cast<size_t>(window.min - window.overlap);
        };
        const auto elements_of = [](const std::vector<int64_t>& dims) -> size_t {
            size_t count = 1;
            for (const int64_t extent : dims) { count *= static_cast<size_t>(extent); }
            return count;
        };
        for (uint32_t i = 0; i < model.input_count(); ++i) {
            const anira::SpecView view = model.input_spec(i);
            inputs.push_back(dims_of(view));
            input_names.emplace_back(view.name());
            spec.m_tensor_input_size.push_back(elements_of(inputs.back()));
            spec.m_preprocess_input_channels.push_back(channels_of(view));
            spec.m_preprocess_input_size.push_back(size_of(view));
        }
        for (uint32_t i = 0; i < model.output_count(); ++i) {
            const anira::SpecView view = model.output_spec(i);
            outputs.push_back(dims_of(view));
            output_names.emplace_back(view.name());
            spec.m_tensor_output_size.push_back(elements_of(outputs.back()));
            spec.m_postprocess_output_channels.push_back(channels_of(view));
            spec.m_postprocess_output_size.push_back(size_of(view));
            spec.m_internal_model_latency.push_back(
                static_cast<size_t>(std::max<int64_t>(view.latency(), 0)));
        }
        m_processing_spec = std::move(spec);
        m_input_shape = inputs;
        m_output_shape = outputs;

        // The shapes an entry holds: the canonical ones through the entry's layouts.
        const auto entry_shapes = [&](uint32_t e, bool& has_layout) {
            TensorShapeList in;
            TensorShapeList out;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const std::vector<uint32_t> layout = model.tensor_layout(e, input_names[i]);
                has_layout = has_layout || !layout.empty();
                in.push_back(laid_out(inputs[i], layout));
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
                const std::vector<uint32_t> layout = model.tensor_layout(e, output_names[i]);
                has_layout = has_layout || !layout.empty();
                out.push_back(laid_out(outputs[i], layout));
            }
            return std::pair<TensorShapeList, TensorShapeList>{std::move(in), std::move(out)};
        };
        m_tensor_shape.clear();
        m_tensor_shape.emplace_back(m_input_shape, m_output_shape);
        std::vector<InferenceBackend> clones;
        for (size_t b = 0; b < detail::k_num_backends; ++b) {
            const InferenceBackend backend = detail::k_backends[b];
            m_input_shapes[b] = m_input_shape;
            m_output_shapes[b] = m_output_shape;
            for (uint32_t e = 0; e < num_entries; ++e) {
                if (!names(model.model_engine(e), backend)) { continue; }
                bool has_layout = false;
                auto [in, out] = entry_shapes(e, has_layout);
                if (has_layout) {
                    m_input_shapes[b] = std::move(in);
                    m_output_shapes[b] = std::move(out);
                }
                break;
            }
        }
        // The rows in entry order: a backend row per entry with a layout, then a universal
        // clone per other backend of an entry (the 2.x set).
        std::array<bool, detail::k_num_backends> seen{};
        for (uint32_t e = 0; e < num_entries; ++e) {
            const InferenceBackend backend = m_model_data[e].m_backend;
            if (seen[detail::backend_index(backend)]) { continue; }
            seen[detail::backend_index(backend)] = true;
            if (m_input_shapes[detail::backend_index(backend)] != m_input_shape ||
                m_output_shapes[detail::backend_index(backend)] != m_output_shape) {
                m_tensor_shape.emplace_back(m_input_shapes[detail::backend_index(backend)],
                                            m_output_shapes[detail::backend_index(backend)],
                                            backend);
            } else {
                clones.push_back(backend);
            }
        }
        for (const InferenceBackend backend : clones) {
            TensorShape clone(m_input_shape, m_output_shape);
            clone.m_backend = backend;
            m_tensor_shape.push_back(std::move(clone));
        }

        const anira::Hard legacy = contract.hard();
        // The budget as the handle stores it (double milliseconds), not rounded to the
        // nanosecond: the float the constructor was given reads back as the same float.
        anira_budget_kind budget = ANIRA_BUDGET_MEASURED;
        double budget_ms = 0.0;
        anira::detail::check(anira_contract_hard_budget(contract.native(), &budget, &budget_ms),
                             "anira_contract_hard_budget");
        m_max_inference_time = static_cast<float>(budget_ms);
        m_warm_up = legacy.warmup_iterations;
        m_blocking_ratio = static_cast<float>(legacy.wait_ratio);
        m_session_exclusive_processor = model.state() == ANIRA_MODEL_STATEFUL;
        // 2.x ran a session-exclusive configuration on one processor whatever the count said;
        // a document may leave the count out, which the upgrade fills with the default.
        m_num_parallel_processors = m_session_exclusive_processor ? 1 : model.max_instances();
    }

    std::optional<anira::ModelConfig> m_model;
    std::optional<anira::ContractHandle> m_legacy;
    std::string m_document;
    std::string m_base_dir;
    std::vector<std::pair<uint32_t, std::span<const std::byte>>> m_borrowed;
    TensorShapeList m_input_shape;
    TensorShapeList m_output_shape;
    std::array<TensorShapeList, detail::k_num_backends> m_input_shapes;
    std::array<TensorShapeList, detail::k_num_backends> m_output_shapes;
};

/**
 * @brief The 2.x JsonConfigLoader over the 3.x loaders: reads a version-2 document once and
 * builds both objects, the ContextConfig (anira::ContextConfig::from_json, read back) and the
 * InferenceConfig (ModelConfig::from_json, the one upgrade path). No leniency where 2.x logged
 * and returned nullptr: a file that does not open is ANIRA_ERROR_NO_SUCH_FILE, malformed text,
 * a wrong type or a word outside a vocabulary ANIRA_ERROR_JSON with the key path, a document
 * that is not version 2 ANIRA_ERROR_CONFIG, all from the constructor. A relative model_path
 * resolves against the document's directory (the 3.x rule; 2.x left it to the working
 * directory), and a backend this build lacks is kept as an entry a handler skips (2.x dropped
 * it).
 */
class JsonConfigLoader {
public:
    /// Relative model paths resolve against the file's directory.
    explicit JsonConfigLoader(const std::string& file_path) {
        const std::filesystem::path path(file_path);
        initialize(anira::detail::read_text(path), anira::detail::utf8(path.parent_path()));
    }
    /// Model paths as written.
    explicit JsonConfigLoader(std::istream& stream) {
        initialize(
            std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()),
            std::string());
    }

    /// The document's context_config (the defaults without one); once: the second call returns
    /// nullptr, as in 2.x.
    std::unique_ptr<ContextConfig> get_context_config() { return std::move(m_context_config); }
    /// The document's inference_config; once: the second call returns nullptr, as in 2.x.
    /// @throws Error{ANIRA_ERROR_CONFIG} for a document that carries only a context_config.
    std::unique_ptr<InferenceConfig> get_inference_config() {
        if (m_no_inference_config) {
            detail::fail(ANIRA_ERROR_CONFIG,
                         "anira::v2::JsonConfigLoader: the document has no inference_config");
        }
        return std::move(m_inference_config);
    }

private:
    void initialize(const std::string& text, std::string base_dir) {
        // What the model load refused, rethrown below where it decides the outcome.
        anira_status model_status = ANIRA_OK;
        std::string model_message;
        try {
            m_inference_config = std::make_unique<InferenceConfig>(
                InferenceConfig::from_document(text, std::move(base_dir)));
        } catch (const anira::Error& error) {
            model_status = error.status;
            model_message = error.what();
        }
        const bool model_failed = model_status != ANIRA_OK;
        std::optional<anira::ContextConfig> context;
        try {
            context.emplace(anira::ContextConfig::from_json(text));
        } catch (const anira::Error&) {
            // Malformed text, or a 3.x model file (whose keys no context file has): the model
            // load said which.
            if (model_failed) { detail::fail(model_status, model_message); }
            throw;
        }
        if (!context->upgraded()) {
            if (model_failed && model_status != ANIRA_ERROR_CONFIG) {
                detail::fail(ANIRA_ERROR_CONFIG,
                             "anira::v2::JsonConfigLoader: not a version 2 document (no "
                             "inference_config or context_config root): a 3.x context file "
                             "loads through anira::ContextConfig::from_file");
            }
            if (model_failed) { detail::fail(model_status, model_message); }
        }
        if (model_failed) {
            // A version 2 document whose inference_config did not load. Without the key at all
            // (it cannot appear unquoted in the text) the document is a context_config alone,
            // which get_inference_config() answers; otherwise the model load's error stands.
            if (text.find("\"inference_config\"") != std::string::npos) {
                detail::fail(model_status, model_message);
            }
            m_no_inference_config = true;
        }
        m_context_config = std::make_unique<ContextConfig>(*context);
    }

    std::unique_ptr<ContextConfig> m_context_config;
    std::unique_ptr<InferenceConfig> m_inference_config;
    bool m_no_inference_config = false;
};

// ---- the runtime -------------------------------------------------------------------------------

/**
 * @brief The 2.x RingBuffer as a view of the anira_ring of one Streamed slot, the float face over
 * the ring accessors of anira/abi/stage.h (anira_ring_*): what a PrePostProcessor's pre_process
 * pops and its post_process pushes. A view of no ring (the slot of a Static or State tensor)
 * moves nothing and answers 0. Valid for the duration of the phase call that handed it out.
 * Absent from the 2.x class: get_future_sample and get_past_sample (no accessor reads one
 * element; peek_past_block reads the history as a block), get_num_samples (the capacity is
 * anira's), swap_data, initialize_with_positions and clear_with_positions (anira owns the ring).
 *
 * [driver-thread] [callback-safe], nonblocking: every method is noexcept and allocates nothing.
 */
class RingBuffer {
public:
    /// A view of no ring.
    RingBuffer() noexcept = default;
    explicit RingBuffer(anira_ring* ring) noexcept : m_ring(ring) {}

    /// Whether the view has a ring.
    explicit operator bool() const noexcept ANIRA_NONBLOCKING { return m_ring != nullptr; }

    /// The extent of the tensor's Channel axis, 1 without one; 0 for a view of no ring.
    size_t get_num_channels() const noexcept ANIRA_NONBLOCKING {
        return anira_ring_num_channels(m_ring);
    }
    /// The unread samples of one channel.
    size_t get_available_samples(size_t channel) const noexcept ANIRA_NONBLOCKING {
        return anira_ring_available(m_ring, static_cast<uint32_t>(channel));
    }
    /// The consumed samples of one channel the ring still holds as history.
    size_t get_available_past_samples(size_t channel) const noexcept ANIRA_NONBLOCKING {
        return anira_ring_available_past(m_ring, static_cast<uint32_t>(channel));
    }

    void push_sample(size_t channel, float sample) noexcept ANIRA_NONBLOCKING {
        anira_ring_push_block(m_ring, static_cast<uint32_t>(channel), &sample, ANIRA_DTYPE_F32, 1);
    }
    /// The oldest unread sample of one channel; 0 when none is available.
    float pop_sample(size_t channel) noexcept ANIRA_NONBLOCKING {
        float sample = 0.F;
        anira_ring_pop_block(m_ring, static_cast<uint32_t>(channel), &sample, ANIRA_DTYPE_F32, 1);
        return sample;
    }
    void push_block(size_t channel,
                    const float* data,
                    size_t num_samples) noexcept ANIRA_NONBLOCKING {
        anira_ring_push_block(m_ring,
                              static_cast<uint32_t>(channel),
                              data,
                              ANIRA_DTYPE_F32,
                              num_samples);
    }
    /// Pops num_samples samples of one channel, oldest first; those beyond the available ones
    /// are zero.
    void pop_block(size_t channel, float* data, size_t num_samples) noexcept ANIRA_NONBLOCKING {
        anira_ring_pop_block(m_ring,
                             static_cast<uint32_t>(channel),
                             data,
                             ANIRA_DTYPE_F32,
                             num_samples);
    }
    /// Copies the num_samples most recently consumed samples of one channel, oldest first,
    /// without popping; history the ring never held reads as zero.
    void peek_past_block(size_t channel,
                         float* data,
                         size_t num_samples) const noexcept ANIRA_NONBLOCKING {
        anira_ring_peek_past_block(m_ring,
                                   static_cast<uint32_t>(channel),
                                   data,
                                   ANIRA_DTYPE_F32,
                                   num_samples);
    }
    void push_fill(size_t channel, float value, size_t num_samples) noexcept ANIRA_NONBLOCKING {
        anira_ring_push_fill(m_ring,
                             static_cast<uint32_t>(channel),
                             &value,
                             ANIRA_DTYPE_F32,
                             num_samples);
    }
    /// Drops up to num_samples unread samples of one channel. @return the samples dropped.
    size_t discard(size_t channel, size_t num_samples) noexcept ANIRA_NONBLOCKING {
        return anira_ring_discard(m_ring, static_cast<uint32_t>(channel), num_samples);
    }
    /// num_batches overlapping windows of one channel, window b at data + offset + b * (num_new
    /// + num_old): num_old samples of history, then num_new popped ones. @return the elements
    /// written, 0 when refused (a dtype that is not the ring's among them).
    size_t pop_windows(size_t channel,
                       void* data,
                       anira_dtype dtype,
                       size_t num_new,
                       size_t num_old,
                       size_t offset,
                       size_t num_batches) noexcept ANIRA_NONBLOCKING {
        return anira_ring_pop_windows(m_ring,
                                      static_cast<uint32_t>(channel),
                                      data,
                                      dtype,
                                      num_new,
                                      num_old,
                                      offset,
                                      static_cast<uint32_t>(num_batches));
    }

    /// The element type of the ring (the Hard contract's ring dtype, F32 unless declared); 0
    /// for a view of no ring.
    anira_dtype dtype() const noexcept ANIRA_NONBLOCKING { return anira_ring_dtype(m_ring); }
    anira_ring* native() const noexcept ANIRA_NONBLOCKING { return m_ring; }

private:
    anira_ring* m_ring = nullptr;
};

/**
 * @brief The 2.x BufferF as a view of the model end of one slot: one channel of
 * get_num_samples() floats, the tensor's elements packed (every channel of a multichannel tensor
 * lies in that one channel, channel c at [c * n, (c + 1) * n), as 2.x laid it out). Re-pointed
 * for every phase call; valid for the duration of the call. Channel 0 is the only channel: a
 * pointer to another is nullptr and a sample of another reads 0. Absent from the 2.x class: the
 * owning BufferF(channels, samples) (a host buffer is the host's own container now), swap_data,
 * get_sample_rate.
 *
 * [driver-thread | inference-thread] [callback-safe], nonblocking.
 */
class BufferF {
public:
    /// A buffer of nothing: no elements.
    BufferF() noexcept = default;
    BufferF(float* data, size_t num_samples) noexcept : m_data(data), m_num_samples(num_samples) {}

    size_t get_num_channels() const noexcept ANIRA_NONBLOCKING { return 1; }
    size_t get_num_samples() const noexcept ANIRA_NONBLOCKING { return m_num_samples; }

    const float* get_read_pointer(size_t channel) const noexcept ANIRA_NONBLOCKING {
        return channel == 0 ? m_data : nullptr;
    }
    const float* get_read_pointer(size_t channel,
                                  size_t sample_index) const noexcept ANIRA_NONBLOCKING {
        return channel == 0 && m_data != nullptr ? m_data + sample_index : nullptr;
    }
    float* get_write_pointer(size_t channel) noexcept ANIRA_NONBLOCKING {
        return channel == 0 ? m_data : nullptr;
    }
    float* get_write_pointer(size_t channel, size_t sample_index) noexcept ANIRA_NONBLOCKING {
        return channel == 0 && m_data != nullptr ? m_data + sample_index : nullptr;
    }
    /// One sample; 0 outside the buffer.
    float get_sample(size_t channel, size_t sample_index) const noexcept ANIRA_NONBLOCKING {
        return channel == 0 && sample_index < m_num_samples ? m_data[sample_index] : 0.F;
    }
    /// One sample; nothing outside the buffer.
    void set_sample(size_t channel, size_t sample_index, float value) noexcept ANIRA_NONBLOCKING {
        if (channel == 0 && sample_index < m_num_samples) { m_data[sample_index] = value; }
    }
    /// Zeroes every sample.
    void clear() noexcept ANIRA_NONBLOCKING {
        for (size_t i = 0; i < m_num_samples; ++i) { m_data[i] = 0.F; }
    }
    float* data() noexcept ANIRA_NONBLOCKING { return m_data; }
    const float* data() const noexcept ANIRA_NONBLOCKING { return m_data; }
    /// The array of the one channel pointer, valid while this view is.
    const float* const* get_array_of_read_pointers() const noexcept ANIRA_NONBLOCKING {
        return &m_data;
    }
    float* const* get_array_of_write_pointers() noexcept ANIRA_NONBLOCKING { return &m_data; }

private:
    float* m_data = nullptr;
    size_t m_num_samples = 0;
};

/**
 * @brief The 2.x PrePostProcessor over the views: the four virtuals with their 2.x signatures and
 * default bodies, the values of the non-streamable tensors as per-element atomics any thread may
 * set and read, and the five 2.x helpers with their 2.x rules (a plain pop lands ring channel c
 * at [c * n, (c + 1) * n) of the buffer; the window, offset and batched pops write every ring
 * channel to the same position, mono windows). It runs inside a LegacyProcessorStage: the
 * phases of the stage call the virtuals, pre_process and post_process on the thread that drives
 * the handler, before_inference and after_inference on an inference thread, with the backend of
 * the plan the chunk runs on. A virtual that throws fails its chunk (the chunk delivers zeros and
 * the status lands in anira_handler_rt_error: an anira::Error's status, else
 * ANIRA_ERROR_ENGINE); in 2.x the throw reached the host. A Static input a custom pre_process
 * leaves alone reads zeros, where 2.x left what the buffer last held.
 */
class PrePostProcessor {
public:
    PrePostProcessor() = delete;
    /// Sizes the atomics from the configuration: one per element of every non-streamable
    /// tensor, zero until set (inputs) or until an inference is collected (outputs). The
    /// configuration outlives the processor.
    explicit PrePostProcessor(InferenceConfig& inference_config)
        : m_inference_config(inference_config)
        , m_inputs(values_of(inference_config.get_preprocess_input_size(),
                             inference_config.get_tensor_input_size()))
        , m_outputs(values_of(inference_config.get_postprocess_output_size(),
                              inference_config.get_tensor_output_size())) {}
    virtual ~PrePostProcessor() = default;
    PrePostProcessor(const PrePostProcessor&) = delete;
    PrePostProcessor& operator=(const PrePostProcessor&) = delete;
    PrePostProcessor(PrePostProcessor&&) = delete;
    PrePostProcessor& operator=(PrePostProcessor&&) = delete;

    /// The model input of every slot from its host end: a streamable tensor pops its hop from
    /// the ring (with the window's history ahead of it when the tensor holds more than one hop
    /// per channel), a non-streamable one takes its atomics.
    virtual void pre_process(std::vector<RingBuffer>& input,
                             std::vector<BufferF>& output,
                             [[maybe_unused]] InferenceBackend current_inference_backend) {
        const std::vector<size_t>& sizes = m_inference_config.get_preprocess_input_size();
        const std::vector<size_t>& elements = m_inference_config.get_tensor_input_size();
        const std::vector<size_t>& channels = m_inference_config.get_preprocess_input_channels();
        for (size_t tensor_index = 0; tensor_index < sizes.size(); ++tensor_index) {
            if (sizes[tensor_index] > 0) {
                const size_t num_new_samples = sizes[tensor_index];
                const size_t tensor_size = elements[tensor_index] / channels[tensor_index];
                if (tensor_size > num_new_samples) {
                    pop_samples_from_buffer(input[tensor_index],
                                            output[tensor_index],
                                            num_new_samples,
                                            tensor_size - num_new_samples);
                } else {
                    pop_samples_from_buffer(input[tensor_index],
                                            output[tensor_index],
                                            num_new_samples);
                }
            } else {
                for (size_t sample = 0; sample < elements[tensor_index]; ++sample) {
                    output[tensor_index].set_sample(0, sample, get_input(tensor_index, sample));
                }
            }
        }
    }
    /// The host end of every output slot from the model output: a streamable tensor pushes its
    /// hop into the ring, a non-streamable one sets its atomics.
    virtual void post_process(std::vector<BufferF>& input,
                              std::vector<RingBuffer>& output,
                              [[maybe_unused]] InferenceBackend current_inference_backend) {
        const std::vector<size_t>& sizes = m_inference_config.get_postprocess_output_size();
        const std::vector<size_t>& elements = m_inference_config.get_tensor_output_size();
        for (size_t tensor_index = 0; tensor_index < sizes.size(); ++tensor_index) {
            if (sizes[tensor_index] > 0) {
                push_samples_to_buffer(input[tensor_index],
                                       output[tensor_index],
                                       sizes[tensor_index]);
            } else {
                for (size_t sample = 0; sample < elements[tensor_index]; ++sample) {
                    set_output(input[tensor_index].get_sample(0, sample), tensor_index, sample);
                }
            }
        }
    }
    /// On an inference thread, before the model runs, over the model inputs. Does nothing.
    virtual void before_inference([[maybe_unused]] std::vector<BufferF>& input,
                                  [[maybe_unused]] InferenceBackend current_inference_backend) {}
    /// On an inference thread, after the model ran, over the model outputs. Does nothing.
    virtual void after_inference([[maybe_unused]] std::vector<BufferF>& output,
                                 [[maybe_unused]] InferenceBackend current_inference_backend) {}

    /// Element j of non-streamable input i, any thread; captured when an inference is
    /// submitted. An index out of range, or a streamable tensor's, is ignored (2.x asserted).
    void set_input(const float& input, size_t i, size_t j) noexcept ANIRA_NONBLOCKING {
        if (i < m_inputs.size() && j < m_inputs[i].size()) {
            m_inputs[i][j].store(input, std::memory_order_relaxed);
        }
    }
    void set_output(const float& output, size_t i, size_t j) noexcept ANIRA_NONBLOCKING {
        if (i < m_outputs.size() && j < m_outputs[i].size()) {
            m_outputs[i][j].store(output, std::memory_order_relaxed);
        }
    }
    /// Element j of non-streamable input i; 0 out of range.
    float get_input(size_t i, size_t j) const noexcept ANIRA_NONBLOCKING {
        return i < m_inputs.size() && j < m_inputs[i].size()
                   ? m_inputs[i][j].load(std::memory_order_relaxed)
                   : 0.F;
    }
    /// Element j of non-streamable output i as the last collected inference left it, any
    /// thread; 0 out of range.
    float get_output(size_t i, size_t j) const noexcept ANIRA_NONBLOCKING {
        return i < m_outputs.size() && j < m_outputs[i].size()
                   ? m_outputs[i][j].load(std::memory_order_relaxed)
                   : 0.F;
    }

    /// num_samples of every ring channel c into [c * num_samples, (c + 1) * num_samples).
    void pop_samples_from_buffer(RingBuffer& input,
                                 BufferF& output,
                                 size_t num_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t i = 0; i < input.get_num_channels(); ++i) {
            input.pop_block(i, output.get_write_pointer(0, i * num_samples), num_samples);
        }
    }
    /// A window of num_old samples of history and num_new popped ones at the buffer's start.
    void pop_samples_from_buffer(RingBuffer& input,
                                 BufferF& output,
                                 size_t num_new_samples,
                                 size_t num_old_samples) noexcept ANIRA_NONBLOCKING {
        pop_samples_from_buffer(input, output, num_new_samples, num_old_samples, 0);
    }
    /// The window at an offset; every ring channel writes the same position (mono windows).
    void pop_samples_from_buffer(RingBuffer& input,
                                 BufferF& output,
                                 size_t num_new_samples,
                                 size_t num_old_samples,
                                 size_t offset) noexcept ANIRA_NONBLOCKING {
        for (size_t i = 0; i < input.get_num_channels(); ++i) {
            float* window = output.get_write_pointer(0, offset);
            input.peek_past_block(i, window, num_old_samples);
            input.pop_block(i, window + num_old_samples, num_new_samples);
        }
    }
    /// num_batches windows (RingBuffer::pop_windows); every ring channel writes the same windows.
    void pop_samples_from_buffer(RingBuffer& input,
                                 BufferF& output,
                                 size_t num_new_samples,
                                 size_t num_old_samples,
                                 size_t offset,
                                 size_t num_batches) noexcept ANIRA_NONBLOCKING {
        for (size_t i = 0; i < input.get_num_channels(); ++i) {
            input.pop_windows(i,
                              output.get_write_pointer(0, 0),
                              ANIRA_DTYPE_F32,
                              num_new_samples,
                              num_old_samples,
                              offset,
                              num_batches);
        }
    }
    /// num_samples of every ring channel c from [c * num_samples, (c + 1) * num_samples).
    void push_samples_to_buffer(const BufferF& input,
                                RingBuffer& output,
                                size_t num_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t i = 0; i < output.get_num_channels(); ++i) {
            output.push_block(i, input.get_read_pointer(0, i * num_samples), num_samples);
        }
    }

    /// The configuration the processor was built with: the slot roles its stage reads.
    const InferenceConfig& config() const noexcept { return m_inference_config; }

protected:
    InferenceConfig& m_inference_config;

private:
    /// One vector of atomics per slot: the elements of a non-streamable tensor, none for a
    /// streamable one.
    static std::vector<std::vector<std::atomic<float>>> values_of(
        const std::vector<size_t>& sizes,
        const std::vector<size_t>& elements) {
        std::vector<std::vector<std::atomic<float>>> values;
        values.reserve(sizes.size());
        for (size_t i = 0; i < sizes.size(); ++i) {
            values.emplace_back(sizes[i] == 0 && i < elements.size() ? elements[i] : 0);
        }
        return values;
    }

    std::vector<std::vector<std::atomic<float>>> m_inputs;
    std::vector<std::vector<std::atomic<float>>> m_outputs;
};

/**
 * @brief The anira::Stage that runs one 2.x PrePostProcessor: all four phases and
 * ANIRA_STAGE_FLAG_REALTIME_PRE_POST (the 2.x processor ran pre_process and post_process on the
 * audio thread already). What a 2.x host that moves to a 3.x Pipeline keeps its processor with:
 * `pipeline.add(anira::stage::Custom(std::make_shared<anira::v2::LegacyProcessorStage>(pp)))`.
 * Per phase call it points the processor's view vectors at the chunk's rings and model tensors
 * (per entry, so concurrent chunks never share them) and calls the virtual with the backend of
 * the chunk's plan (to_backend of StageContext::engine()). The processor and its configuration
 * outlive every handler that runs the stage.
 */
class LegacyProcessorStage final : public anira::Stage {
public:
    explicit LegacyProcessorStage(PrePostProcessor& processor) noexcept : m_processor(processor) {}

    uint32_t phases() const noexcept override { return k_all_phases; }
    uint32_t flags() const noexcept override { return ANIRA_STAGE_FLAG_REALTIME_PRE_POST; }
    /// The view vectors per entry, sized for the handler. @throws Error{ANIRA_ERROR_CONFIG}
    /// when the processor's configuration has another tensor count than the handler's model.
    std::unique_ptr<anira::Stage::Prepared> prepare(const anira::PrepareInfo& info) override;

    class Prepared;

private:
    PrePostProcessor& m_processor;
};

/// One handler's views of the stage: per entry the four vectors the 2.x virtuals take, one
/// element per slot of the side (State and Static positions included), re-pointed per call.
class LegacyProcessorStage::Prepared final : public anira::Stage::Prepared {
public:
    Prepared(PrePostProcessor& processor,
             std::vector<anira_role> input_roles,
             std::vector<anira_role> output_roles,
             uint32_t num_entries)
        : m_processor(processor)
        , m_input_roles(std::move(input_roles))
        , m_output_roles(std::move(output_roles))
        , m_in_rings(num_entries, std::vector<RingBuffer>(m_input_roles.size()))
        , m_in_buffers(num_entries, std::vector<BufferF>(m_input_roles.size()))
        , m_out_buffers(num_entries, std::vector<BufferF>(m_output_roles.size()))
        , m_out_rings(num_entries, std::vector<RingBuffer>(m_output_roles.size())) {}

    anira_status pre_process(anira::StageContext& ctx) noexcept override {
        const uint32_t e = ctx.entry();
        if (e >= m_in_rings.size()) { return ANIRA_ERROR_INTERNAL; }
        for (uint32_t slot = 0; slot < m_input_roles.size(); ++slot) {
            anira::RingView ring;
            anira::Tensor tensor{};
            if (m_input_roles[slot] == ANIRA_ROLE_STREAMED) { ctx.input_ring(slot, ring); }
            if (m_input_roles[slot] != ANIRA_ROLE_STATE) { ctx.input_tensor(slot, tensor); }
            m_in_rings[e][slot] = RingBuffer(ring.native());
            m_in_buffers[e][slot] = BufferF(tensor.data_f32(), tensor.num_elements());
        }
        return run(ANIRA_PHASE_PRE_PROCESS, [&] {
            m_processor.pre_process(m_in_rings[e], m_in_buffers[e], to_backend(ctx.engine()));
        });
    }
    anira_status post_process(anira::StageContext& ctx) noexcept override {
        const uint32_t e = ctx.entry();
        if (e >= m_out_rings.size()) { return ANIRA_ERROR_INTERNAL; }
        for (uint32_t slot = 0; slot < m_output_roles.size(); ++slot) {
            anira::RingView ring;
            anira::Tensor tensor{};
            if (m_output_roles[slot] == ANIRA_ROLE_STREAMED) { ctx.output_ring(slot, ring); }
            if (m_output_roles[slot] != ANIRA_ROLE_STATE) { ctx.output_tensor(slot, tensor); }
            m_out_rings[e][slot] = RingBuffer(ring.native());
            m_out_buffers[e][slot] = BufferF(tensor.data_f32(), tensor.num_elements());
        }
        return run(ANIRA_PHASE_POST_PROCESS, [&] {
            m_processor.post_process(m_out_buffers[e], m_out_rings[e], to_backend(ctx.engine()));
        });
    }
    anira_status before_inference(anira::StageContext& ctx) noexcept override {
        const uint32_t e = ctx.entry();
        if (e >= m_in_buffers.size()) { return ANIRA_ERROR_INTERNAL; }
        for (uint32_t slot = 0; slot < m_input_roles.size(); ++slot) {
            anira::Tensor tensor{};
            ctx.input_tensor(slot, tensor);
            m_in_buffers[e][slot] = BufferF(tensor.data_f32(), tensor.num_elements());
        }
        return run(ANIRA_PHASE_BEFORE_INFERENCE, [&] {
            m_processor.before_inference(m_in_buffers[e], to_backend(ctx.engine()));
        });
    }
    anira_status after_inference(anira::StageContext& ctx) noexcept override {
        const uint32_t e = ctx.entry();
        if (e >= m_out_buffers.size()) { return ANIRA_ERROR_INTERNAL; }
        for (uint32_t slot = 0; slot < m_output_roles.size(); ++slot) {
            anira::Tensor tensor{};
            ctx.output_tensor(slot, tensor);
            m_out_buffers[e][slot] = BufferF(tensor.data_f32(), tensor.num_elements());
        }
        return run(ANIRA_PHASE_AFTER_INFERENCE, [&] {
            m_processor.after_inference(m_out_buffers[e], to_backend(ctx.engine()));
        });
    }
    // reset: the base's, which does nothing (2.x had no reset hook; the rings are anira's).

private:
    /// A throw never crosses the C boundary: an anira::Error fails the chunk with its status,
    /// anything else with ANIRA_ERROR_ENGINE and one real-time record naming the phase.
    template <class Call>
    static anira_status run(anira_phase phase, Call&& call) noexcept {
        try {
            std::forward<Call>(call)();
            return ANIRA_OK;
        } catch (const anira::Error& error) {
            return anira::detail::failed(error.status) ? error.status : ANIRA_ERROR_ENGINE;
        } catch (...) {
            anira_log_rt(ANIRA_LOG_ERROR,
                         "anira.compat",
                         "a 2.x PrePostProcessor threw out of a phase; the chunk delivers zeros",
                         static_cast<int32_t>(phase),
                         0);
            return ANIRA_ERROR_ENGINE;
        }
    }

    PrePostProcessor& m_processor;
    std::vector<anira_role> m_input_roles;
    std::vector<anira_role> m_output_roles;
    std::vector<std::vector<RingBuffer>> m_in_rings;
    std::vector<std::vector<BufferF>> m_in_buffers;
    std::vector<std::vector<BufferF>> m_out_buffers;
    std::vector<std::vector<RingBuffer>> m_out_rings;
};

inline std::unique_ptr<anira::Stage::Prepared> LegacyProcessorStage::prepare(
    const anira::PrepareInfo& info) {
    const anira::ModelConfig& model = m_processor.config().model_config();
    if (model.input_count() != info.inputs().size() ||
        model.output_count() != info.outputs().size()) {
        detail::fail(ANIRA_ERROR_CONFIG,
                     "anira::v2::LegacyProcessorStage: the PrePostProcessor's configuration has " +
                         std::to_string(model.input_count()) + " inputs and " +
                         std::to_string(model.output_count()) + " outputs, the handler's model " +
                         std::to_string(info.inputs().size()) + " and " +
                         std::to_string(info.outputs().size()));
    }
    std::vector<anira_role> input_roles;
    std::vector<anira_role> output_roles;
    input_roles.reserve(model.input_count());
    output_roles.reserve(model.output_count());
    for (uint32_t i = 0; i < model.input_count(); ++i) {
        input_roles.push_back(model.input_spec(i).role());
    }
    for (uint32_t i = 0; i < model.output_count(); ++i) {
        output_roles.push_back(model.output_spec(i).role());
    }
    return std::make_unique<Prepared>(m_processor,
                                      std::move(input_roles),
                                      std::move(output_roles),
                                      info.num_entries());
}

/**
 * @brief The 2.x BackendBase::process as an anira::Engine under k_custom_engine_id: output i a
 * copy of input i when both have the same element count and dtype, zeros otherwise, every output
 * beyond the input count zeros. What a CUSTOM row runs when no engine was passed, and what a
 * host that passed a plain BackendBase passes now. No providers list, so it serves
 * ANIRA_PROVIDER_CPU alone, and a neutral CUSTOM row under the default candidate set is one plan
 * on the CPU path. flags(): ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL.
 * Three trivial levels: load keeps nothing (the row's path is never opened), the Loaded hands
 * every handler, exclusive or not, a Prepared that keeps nothing, so a shared and an exclusive
 * call run the same body and a reset has nothing to clear. Its code is the host's (the header is
 * compiled into it): a handler must be destroyed before the module that holds the code unloads.
 */
class PassthroughEngine final : public anira::Engine {
public:
    PassthroughEngine() : anira::Engine(k_custom_engine_id) {}

    uint32_t flags() const noexcept override {
        return ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL;
    }
    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& /*info*/) override {
        return std::make_unique<Loaded>();
    }

private:
    class Prepared final : public anira::Engine::Prepared {
    public:
        anira_status process(anira::EngineContext& ctx) noexcept override {
            const std::span<const anira::Tensor> inputs = ctx.inputs();
            const std::span<anira::Tensor> outputs = ctx.outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                const anira::Tensor& out = outputs[i];
                void* target = out.data(out.dtype);
                if (target == nullptr) { continue; }
                const size_t bytes = out.num_elements() * ANIRA_DTYPE_BITS(out.dtype) *
                                     ANIRA_DTYPE_LANES(out.dtype) / 8;
                const void* source = nullptr;
                if (i < inputs.size() && inputs[i].dtype == out.dtype &&
                    inputs[i].num_elements() == out.num_elements()) {
                    source = inputs[i].data(inputs[i].dtype);
                }
                if (source != nullptr) {
                    std::memmove(target, source, bytes);
                } else {
                    std::memset(target, 0, bytes);
                }
            }
            return ANIRA_OK;
        }
    };
    class Loaded final : public anira::Engine::Loaded {
    public:
        std::unique_ptr<anira::Engine::Prepared> prepare(
            const anira::PrepareInfo& /*info*/) override {
            return std::make_unique<Prepared>();
        }
    };
};

/**
 * @brief The 2.x static face of the core (v2.3.0's anira::Context; the pre-release that preceded
 * anira 3 called it anira::Core), over the core entries. The rest of the 2.x class (the
 * inference threads of the host, the session list) is not declared: a host drives its own
 * threads through anira_inference_thread_create on a context.
 */
struct Context {
    /// anira_shutdown: ANIRA_ERROR_INVALID_STATE while a context or a handler lives, which
    /// through this header means while an InferenceHandler lives (2.x stopped the pool
    /// regardless).
    static anira_status shutdown() noexcept { return anira_shutdown(); }
    /// anira_release_core_if_idle: true when the core was freed.
    static bool release_core_if_idle() noexcept { return anira_release_core_if_idle() != 0U; }
    static bool has_core() noexcept { return anira_has_core() != 0U; }
    /// The size of the core's inference thread pool: 0 until the first handler is prepared
    /// (the core builds the pool then); the host's own threads are not counted.
    static unsigned get_num_inference_threads() noexcept { return anira_num_inference_threads(); }
    /// anira_drain_log: delivers the queued real-time records to the sinks.
    static size_t drain_log() noexcept { return anira_drain_log(); }
};

/**
 * @brief The 2.x InferenceHandler over the C handler of anira/abi/handler.h: a context (from the
 * ContextConfig), a pipeline built from a private copy of the configuration's model config, the
 * custom engine and a LegacyProcessorStage over the processor, and one anira_handler created from
 * them. The processor and the configuration are the caller's and outlive the handler, as in 2.x.
 *
 * What differs from 2.x: nothing loads at construction (a model that does not load fails
 * prepare); a prepare with the settings of the last one only resets the stream, one with another
 * geometry loads the models again, one with another reference tensor rebuilds the C handler; the
 * single forms carry their one slot per side and nothing else (2.x resent what the last call
 * left in the other slots); push_data never waits, pop_data with a deadline polls on a handler
 * without a blocking ratio, and set_non_realtime is never refused (a wait without an inference
 * loop fails the call instead, which rt_error() reports); set_inference_backend of a backend
 * without a plan keeps the selection and logs; get_num_inference_threads is the pool size, 0 until
 * the first prepare in the process; every failure is an anira::Error.
 *
 * Real time: reset, get_latency, get_available_samples, set_inference_backend,
 * get_inference_backend and rt_error never wait and carry ANIRA_NONBLOCKING; the process, push and
 * pop forms are noexcept but carry no attribute, since a blocking ratio or set_non_realtime makes
 * them wait for the inference (the 2.x declarations claimed the attribute and waited anyway).
 * Without either, on the thread that drives the handler, they allocate nothing and wait for
 * nothing.
 */
class InferenceHandler {
public:
    InferenceHandler() = delete;
    InferenceHandler(const InferenceHandler&) = delete;
    InferenceHandler& operator=(const InferenceHandler&) = delete;
    InferenceHandler(InferenceHandler&&) = delete;
    InferenceHandler& operator=(InferenceHandler&&) = delete;

    /// A CUSTOM row runs a PassthroughEngine (the 2.x base backend). Loads no model.
    /// @throws Error with the status of the context, the pipeline or anira_handler_create
    /// (ANIRA_ERROR_CONFIG for a configuration whose every entry names an engine this build
    /// lacks).
    InferenceHandler(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     const ContextConfig& context_config = ContextConfig())
        : InferenceHandler(pp_processor, inference_config, nullptr, context_config) {}
    /// The CUSTOM row runs the caller's engine, which carries k_custom_engine_id (the id the
    /// CUSTOM row names; a 2.x custom backend is an anira::Engine constructed with it) and
    /// outlives the handler, as 2.x required of a BackendBase. Its providers() decide where the
    /// plan runs: the CPU path without a list, else its first listed provider.
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an engine of another id,
    /// Error{ANIRA_ERROR_CONFIG} for a configuration without a CUSTOM row, and as the other
    /// constructor.
    InferenceHandler(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     anira::Engine& custom_engine,
                     const ContextConfig& context_config = ContextConfig())
        : InferenceHandler(pp_processor, inference_config, &custom_engine, context_config) {}
    /// Destroys the C handler (in-flight inferences drain first), then the stage and the context.
    ~InferenceHandler() { anira_handler_destroy(m_handler); }

    /// The backend the next block runs on; before prepare the request is kept and applied at
    /// prepare. A backend without a plan keeps the selection and logs once through the real-time
    /// queue (group anira.compat); rt_error() is untouched. Any thread.
    void set_inference_backend(InferenceBackend inference_backend) noexcept ANIRA_NONBLOCKING {
        m_requested.store(static_cast<uint32_t>(inference_backend), std::memory_order_relaxed);
        if (m_prepared.load(std::memory_order_acquire)) { select(inference_backend); }
    }
    /// The backend of the selected plan; before prepare the request, else the 2.x initial rule
    /// (CUSTOM with a caller's engine, else the first entry this build runs, else CUSTOM).
    InferenceBackend get_inference_backend() const noexcept ANIRA_NONBLOCKING {
        if (m_prepared.load(std::memory_order_acquire)) {
            const uint32_t plan = anira_handler_get_plan(m_handler);
            if (plan < m_plan_backend.size()) { return m_plan_backend[plan]; }
        }
        const uint32_t requested = m_requested.load(std::memory_order_relaxed);
        return requested != k_no_request ? static_cast<InferenceBackend>(requested)
                                         : m_initial_backend;
    }

    /// Prepares for the host's stream: the configuration's legacy contract with the host
    /// geometry (m_buffer_size whole, block_min 1 when smaller blocks come), the reference
    /// tensor as the model's anchor. The settings of the last successful prepare only reset the
    /// stream (the models stay loaded); another geometry prepares again and loads the models
    /// again; another reference tensor rebuilds the C handler. The selected backend is applied
    /// again. @throws Error: ANIRA_ERROR_CONFIG for a fractional or non-positive block size or
    /// rate, ANIRA_ERROR_INVALID_ARGUMENT for a reference tensor out of range or not streamable,
    /// else the status of anira_handler_prepare (a model that does not load, a rule of the
    /// configuration); the handler is then unprepared.
    void prepare(HostConfig new_audio_config) { prepare_with(new_audio_config, {}); }
    /// prepare() with the stream latency of one output declared (the 2.x custom latency): what
    /// get_latency reports for it and what its receive ring is primed with, in place of the
    /// computed figure; raised to the model's internal latency with a Warning when below it.
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for an index out of range or an output that
    /// is not streamable, and as prepare().
    void prepare(HostConfig new_audio_config, unsigned custom_latency, size_t tensor_index = 0) {
        const uint32_t count = m_model.output_count();
        if (tensor_index >= count) {
            detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                         "anira::v2::InferenceHandler::prepare: custom latency tensor index " +
                             std::to_string(tensor_index) + " is out of range (the model has " +
                             std::to_string(count) + " output tensors)");
        }
        std::map<std::string, uint32_t> latencies;
        declare_latency(latencies, static_cast<uint32_t>(tensor_index), custom_latency);
        prepare_with(new_audio_config, latencies);
    }
    /// prepare() with a stream latency per output; a non-streamable output carries 0.
    /// @throws Error{ANIRA_ERROR_INVALID_ARGUMENT} for a vector of another size or a figure on a
    /// non-streamable output, and as prepare().
    void prepare(HostConfig new_audio_config, std::vector<unsigned> custom_latency) {
        const uint32_t count = m_model.output_count();
        if (custom_latency.size() != count) {
            detail::fail(
                ANIRA_ERROR_INVALID_ARGUMENT,
                "anira::v2::InferenceHandler::prepare: " + std::to_string(custom_latency.size()) +
                    " custom latencies for a model with " + std::to_string(count) +
                    " output tensors");
        }
        std::map<std::string, uint32_t> latencies;
        for (uint32_t i = 0; i < count; ++i) {
            if (m_model.output_spec(i).role() != ANIRA_ROLE_STREAMED) {
                if (custom_latency[i] != 0) {
                    detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                                 "anira::v2::InferenceHandler::prepare: output tensor " +
                                     std::to_string(i) +
                                     " is not streamable; its custom latency must be 0");
                }
                continue;
            }
            declare_latency(latencies, i, custom_latency[i]);
        }
        prepare_with(new_audio_config, latencies);
    }

    /// In place: the block of slot tensor_index is pushed and replaced by the output.
    size_t process(float* const* data, size_t num_samples, size_t tensor_index = 0) noexcept {
        return process(data, num_samples, data, num_samples, tensor_index);
    }
    /// One block of input slot tensor_index in, one of output slot tensor_index out; no other
    /// slot is carried. A non-streamable input slot takes min(num_input_samples, its size)
    /// values into the processor; a non-streamable output slot is read back likewise.
    size_t process(const float* const* input_data,
                   size_t num_input_samples,
                   float* const* output_data,
                   size_t num_output_samples,
                   size_t tensor_index = 0) noexcept {
        carry_one(tensor_index, input_data, num_input_samples, output_data, num_output_samples);
        process(m_single_in.data(),
                m_single_in_counts.data(),
                m_single_out.data(),
                m_single_out_counts.data());
        return tensor_index < m_single_out_counts.size() ? m_single_out_counts[tensor_index] : 0;
    }
    /// Every slot at once: data[slot][channel][sample] with one count per slot (0 leaves the
    /// slot out). The delivered counts are written into num_output_samples, which is returned;
    /// num_input_samples is not written.
    size_t* process(const float* const* const* input_data,
                    size_t* num_input_samples,
                    float* const* const* output_data,
                    size_t* num_output_samples) noexcept {
        take_inputs(input_data, num_input_samples);
        present_outputs(output_data, num_output_samples);
        anira_status status = ANIRA_OK;
        bool waited = false;
        if (m_non_realtime.load(std::memory_order_relaxed)) {
            waited = true;
            status = anira_handler_process_multi_wait(m_handler,
                                                      m_inputs.data(),
                                                      num_slots(m_inputs),
                                                      m_outputs.data(),
                                                      num_slots(m_outputs),
                                                      m_delivered.data(),
                                                      ANIRA_WAIT_FOREVER);
        } else if (m_blocking) {
            waited = true;
            status = anira_handler_process_multi_wait(m_handler,
                                                      m_inputs.data(),
                                                      num_slots(m_inputs),
                                                      m_outputs.data(),
                                                      num_slots(m_outputs),
                                                      m_delivered.data(),
                                                      ANIRA_WAIT_CONTRACT);
        } else {
            status = anira_handler_process_multi(m_handler,
                                                 m_inputs.data(),
                                                 num_slots(m_inputs),
                                                 m_outputs.data(),
                                                 num_slots(m_outputs),
                                                 m_delivered.data());
        }
        give_outputs(output_data, num_output_samples, status, waited);
        return num_output_samples;
    }

    /// One block of input slot tensor_index; no other slot is carried. Never waits.
    void push_data(const float* const* input_data,
                   size_t num_input_samples,
                   size_t tensor_index = 0) noexcept {
        carry_one(tensor_index, input_data, num_input_samples, nullptr, 0);
        push_data(m_single_in.data(), m_single_in_counts.data());
    }
    /// Every input slot at once. Never waits (2.x waited under set_non_realtime).
    void push_data(const float* const* const* input_data, size_t* num_input_samples) noexcept {
        take_inputs(input_data, num_input_samples);
        anira_handler_push_data_multi(m_handler, m_inputs.data(), num_slots(m_inputs));
    }

    /// One block of output slot tensor_index; no other slot is carried.
    size_t pop_data(float* const* output_data,
                    size_t num_output_samples,
                    size_t tensor_index = 0) noexcept {
        carry_one(tensor_index, nullptr, 0, output_data, num_output_samples);
        pop_data(m_single_out.data(), m_single_out_counts.data());
        return tensor_index < m_single_out_counts.size() ? m_single_out_counts[tensor_index] : 0;
    }
    /// pop_data() that waits for the block's inference until wait_until (polling on a handler
    /// without a blocking ratio, where 2.x neither waited nor collected); under
    /// set_non_realtime without limit, as in 2.x.
    size_t pop_data(float* const* output_data,
                    size_t num_output_samples,
                    std::chrono::steady_clock::time_point wait_until,
                    size_t tensor_index = 0) noexcept {
        carry_one(tensor_index, nullptr, 0, output_data, num_output_samples);
        pop_data(m_single_out.data(), m_single_out_counts.data(), wait_until);
        return tensor_index < m_single_out_counts.size() ? m_single_out_counts[tensor_index] : 0;
    }
    /// Every output slot at once; the counts are written into num_output_samples.
    size_t* pop_data(float* const* const* output_data, size_t* num_output_samples) noexcept {
        present_outputs(output_data, num_output_samples);
        anira_status status = ANIRA_OK;
        const bool waited = m_non_realtime.load(std::memory_order_relaxed);
        if (waited) {
            status = anira_handler_pop_data_multi_wait(m_handler,
                                                       m_outputs.data(),
                                                       num_slots(m_outputs),
                                                       m_delivered.data(),
                                                       ANIRA_WAIT_FOREVER);
        } else {
            status = anira_handler_pop_data_multi(m_handler,
                                                  m_outputs.data(),
                                                  num_slots(m_outputs),
                                                  m_delivered.data());
        }
        give_outputs(output_data, num_output_samples, status, waited);
        return num_output_samples;
    }
    /// The deadline form of the multi pop.
    size_t* pop_data(float* const* const* output_data,
                     size_t* num_output_samples,
                     std::chrono::steady_clock::time_point wait_until) noexcept {
        present_outputs(output_data, num_output_samples);
        double timeout_ms = ANIRA_WAIT_FOREVER;
        if (!m_non_realtime.load(std::memory_order_relaxed)) {
            const auto remaining = wait_until - std::chrono::steady_clock::now();
            timeout_ms =
                std::max(0.0, std::chrono::duration<double, std::milli>(remaining).count());
        }
        const anira_status status = anira_handler_pop_data_multi_wait(m_handler,
                                                                      m_outputs.data(),
                                                                      num_slots(m_outputs),
                                                                      m_delivered.data(),
                                                                      timeout_ms);
        give_outputs(output_data, num_output_samples, status, true);
        return num_output_samples;
    }

    /// The latency of output slot tensor_index in samples: the declared figure of a custom
    /// latency, else the computed one; 0 out of range and for a non-streamable output.
    unsigned get_latency(size_t tensor_index = 0) const noexcept ANIRA_NONBLOCKING {
        return anira_handler_get_latency(m_handler, static_cast<uint32_t>(tensor_index));
    }
    /// Every output's latency, index-aligned. Allocates: main thread.
    std::vector<unsigned> get_latency_vector() const {
        uint32_t count = m_model.output_count();
        std::vector<uint32_t> figures(count);
        if (count > 0) { anira_handler_get_latencies(m_handler, &count, figures.data()); }
        return {figures.begin(), figures.end()};
    }
    /// Collects the completed inferences and reports the samples waiting in one channel of the
    /// output ring of tensor_index; 0 out of range.
    size_t get_available_samples(size_t tensor_index,
                                 size_t channel = 0) const noexcept ANIRA_NONBLOCKING {
        size_t available = 0;
        anira_handler_get_available_samples(m_handler,
                                            static_cast<uint32_t>(tensor_index),
                                            static_cast<uint32_t>(channel),
                                            &available);
        return available;
    }
    /// Whether the process and pop forms wait without limit for their inference (an offline
    /// render). Never refused: without an inference loop each such call fails instead
    /// (ANIRA_ERROR_INVALID_STATE in rt_error(), the counts of the nonblocking stem returned).
    void set_non_realtime(bool is_non_realtime) noexcept ANIRA_NONBLOCKING {
        m_non_realtime.store(is_non_realtime, std::memory_order_relaxed);
    }
    /// anira_drain_log: delivers the queued real-time records to the sinks.
    size_t drain_log() noexcept { return anira_drain_log(); }
    /// The size of the core's inference thread pool: 0 until the first handler is prepared.
    static unsigned get_num_inference_threads() noexcept { return anira_num_inference_threads(); }
    /// Restarts the stream: the rings, the declared state and rt_error() (anira_handler_reset).
    void reset() noexcept ANIRA_NONBLOCKING { anira_handler_reset(m_handler); }

    /// The C handler, for a host that mixes in a 3.x entry; valid until the next prepare that
    /// changes the reference tensor, or the destruction.
    anira_handler* native() const noexcept ANIRA_NONBLOCKING { return m_handler; }
    /// The last real-time failure (anira_handler_rt_error).
    anira_status rt_error() const noexcept ANIRA_NONBLOCKING {
        return anira_handler_rt_error(m_handler);
    }

private:
    static constexpr uint32_t k_no_request = 0xFFFFFFFFU;

    InferenceHandler(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     anira::Engine* custom_engine,
                     const ContextConfig& context_config)
        : m_context(context_config.to_context_config())
        , m_config(inference_config)
        , m_pp(pp_processor)
        , m_model(inference_config.model_config_copy())
        , m_stage(std::make_shared<LegacyProcessorStage>(pp_processor))
        , m_blocking(inference_config.m_blocking_ratio > 0.F) {
        bool has_custom_row = false;
        for (uint32_t e = 0; e < m_model.model_count(); ++e) {
            has_custom_row = has_custom_row || names(m_model.model_engine(e), CUSTOM);
        }
        if (custom_engine != nullptr) {
            if (custom_engine->id() != k_custom_engine_id) {
                detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                             "anira::v2::InferenceHandler: the custom engine's id is '" +
                                 custom_engine->id() + "', the CUSTOM row names '" +
                                 k_custom_engine_id +
                                 "': construct the engine with anira::v2::k_custom_engine_id");
            }
            if (!has_custom_row) {
                detail::fail(ANIRA_ERROR_CONFIG,
                             "anira::v2::InferenceHandler: a custom engine for a configuration "
                             "without a CUSTOM row; add ModelData(..., CUSTOM)");
            }
            // The caller keeps the engine alive past the handler, as 2.x required of its
            // BackendBase: the pointer the pipeline holds deletes nothing.
            m_engine = std::shared_ptr<anira::Engine>(custom_engine, [](anira::Engine*) {});
        } else if (has_custom_row) {
            m_engine = std::make_shared<PassthroughEngine>();
        }
        m_caller_engine = custom_engine != nullptr;
        m_initial_backend = initial_backend(m_caller_engine);
        build_face(m_config.get_preprocess_input_channels(),
                   m_config.get_tensor_input_size(),
                   true,
                   m_inputs,
                   m_input_planes,
                   m_input_roles,
                   m_input_sizes);
        build_face(m_config.get_postprocess_output_channels(),
                   m_config.get_tensor_output_size(),
                   false,
                   m_outputs,
                   m_output_planes,
                   m_output_roles,
                   m_output_sizes);
        m_delivered.assign(m_outputs.size(), 0);
        m_single_in.assign(m_inputs.size(), nullptr);
        m_single_in_counts.assign(m_inputs.size(), 0);
        m_single_out.assign(m_outputs.size(), nullptr);
        m_single_out_counts.assign(m_outputs.size(), 0);
        m_handler = build_handler();
    }

    /// The 2.x initial rule over the configuration: CUSTOM with a caller's engine, else the
    /// first entry this build runs, else CUSTOM. Allocates (is_available): the constructor.
    InferenceBackend initial_backend(bool caller_engine) const {
        if (caller_engine) { return CUSTOM; }
        for (uint32_t e = 0; e < m_model.model_count(); ++e) {
            const InferenceBackend backend = to_backend(m_model.model_engine(e));
            if (backend != CUSTOM && is_available(backend)) { return backend; }
        }
        return CUSTOM;
    }

    /// One tensor per slot: a planar float32 {channels, 0} over the handler's own plane array
    /// for a Streamed slot, the empty {1, 0} for a Static or State slot, which never changes.
    void build_face(const std::vector<size_t>& channels,
                    const std::vector<size_t>& sizes,
                    bool inputs,
                    std::vector<anira_tensor>& tensors,
                    std::vector<std::vector<void*>>& planes,
                    std::vector<anira_role>& roles,
                    std::vector<size_t>& static_sizes) const {
        const uint32_t count = inputs ? m_model.input_count() : m_model.output_count();
        tensors.assign(count, anira_tensor{});
        planes.clear();
        planes.reserve(count);
        roles.clear();
        static_sizes.clear();
        for (uint32_t slot = 0; slot < count; ++slot) {
            const anira_role role =
                inputs ? m_model.input_spec(slot).role() : m_model.output_spec(slot).role();
            roles.push_back(role);
            static_sizes.push_back(role == ANIRA_ROLE_STATIC && slot < sizes.size() ? sizes[slot]
                                                                                    : 0);
            const size_t num_channels =
                role == ANIRA_ROLE_STREAMED && slot < channels.size() ? channels[slot] : 1;
            planes.emplace_back(num_channels, nullptr);
            const std::array<int64_t, 2> shape{static_cast<int64_t>(num_channels), 0};
            anira_tensor_init_host_planar(&tensors[slot],
                                          static_cast<const void*>(planes.back().data()),
                                          static_cast<uint32_t>(num_channels),
                                          ANIRA_DTYPE_F32,
                                          2,
                                          shape.data());
            if (inputs) { tensors[slot].flags |= static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY); }
        }
    }

    /// The C handler of the private model config: the pipeline (the engine, the inference stage
    /// under the default candidate set, the stage) is dropped once the handler holds its copy.
    anira_handler* build_handler() {
        anira::Pipeline pipeline;
        if (m_engine != nullptr) { pipeline.register_engine(m_engine); }
        pipeline.inference(m_model);
        pipeline.add(anira::stage::Custom(m_stage));
        anira_error err = ANIRA_ERROR_INIT;
        anira_handler* handler = nullptr;
        anira::detail::check(
            anira_handler_create(m_context.native(), pipeline.native(), &handler, &err),
            err);
        return handler;
    }

    /// The canonical name of the reference tensor; empty for the default (the first Streamed
    /// input, else the first Streamed output: the 2.x rule).
    std::string anchor_of(const HostConfig& host) const {
        if (host.m_tensor_index == HostConfig::k_first_streamable) { return {}; }
        const std::string side = host.m_tensor_is_input ? "input" : "output";
        const uint32_t count =
            host.m_tensor_is_input ? m_model.input_count() : m_model.output_count();
        if (host.m_tensor_index >= count) {
            detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                         "HostConfig: reference tensor " + side + "[" +
                             std::to_string(host.m_tensor_index) +
                             "] is out of range (the model has " + std::to_string(count) + " " +
                             side + " tensors).");
        }
        const auto index = static_cast<uint32_t>(host.m_tensor_index);
        const anira::SpecView spec =
            host.m_tensor_is_input ? m_model.input_spec(index) : m_model.output_spec(index);
        if (spec.role() != ANIRA_ROLE_STREAMED) {
            detail::fail(
                ANIRA_ERROR_INVALID_ARGUMENT,
                "HostConfig: reference tensor " + side + "[" + std::to_string(host.m_tensor_index) +
                    "] is not streamable (its " +
                    (host.m_tensor_is_input ? "preprocess_input_size" : "postprocess_output_size") +
                    " is 0).");
        }
        return std::string(spec.name());
    }

    /// The declared latency of one Streamed output, raised to the model's internal latency with
    /// the 2.x Warning.
    void declare_latency(std::map<std::string, uint32_t>& latencies,
                         uint32_t index,
                         unsigned requested) const {
        const anira::SpecView spec = m_model.output_spec(index);
        if (spec.role() != ANIRA_ROLE_STREAMED) {
            detail::fail(ANIRA_ERROR_INVALID_ARGUMENT,
                         "anira::v2::InferenceHandler::prepare: output tensor " +
                             std::to_string(index) +
                             " is not streamable; it has no stream latency to declare");
        }
        const auto floor = static_cast<unsigned>(std::max<int64_t>(spec.latency(), 0));
        if (requested < floor) {
            detail::warn("custom latency " + std::to_string(requested) + " for tensor " +
                         std::to_string(index) + " is below the internal model latency " +
                         std::to_string(floor) + "; clamping");
        }
        latencies[std::string(spec.name())] = std::max(requested, floor);
    }

    void prepare_with(const HostConfig& host, const std::map<std::string, uint32_t>& latencies) {
        const float block = host.m_buffer_size;
        if (!(block > 0.F) || std::floor(block) != block || !std::isfinite(block)) {
            detail::fail(ANIRA_ERROR_CONFIG,
                         "HostConfig: m_buffer_size " + detail::json_number(block) +
                             " is not a whole number of samples above 0; a host whose block is "
                             "measured on another stream anchors on that stream instead "
                             "(m_tensor_index, m_tensor_is_input) and passes its own block");
        }
        if (!(host.m_sample_rate > 0.F) || !std::isfinite(host.m_sample_rate)) {
            detail::fail(ANIRA_ERROR_CONFIG,
                         "HostConfig: m_sample_rate " + detail::json_number(host.m_sample_rate) +
                             " is not above 0");
        }
        const std::string wanted = anchor_of(host);
        if (m_prepared.load(std::memory_order_acquire) && wanted == m_anchor &&
            m_last_host.has_value() && *m_last_host == host && latencies == m_last_latencies) {
            anira_handler_reset(m_handler);
            return;
        }
        if (wanted != m_anchor) {
            // A new handler first, while the old one still holds the engine's C engine (one
            // C engine, one init), then the old one goes.
            m_model.anchor(wanted);
            anira_handler* rebuilt = nullptr;
            try {
                rebuilt = build_handler();
            } catch (...) {
                m_model.anchor(m_anchor);
                throw;
            }
            m_prepared.store(false, std::memory_order_release);
            anira_handler_destroy(m_handler);
            m_handler = rebuilt;
            m_anchor = wanted;
        }
        anira::Hard hard = m_config.hard();
        const auto frames = static_cast<uint32_t>(block);
        hard.block_min = host.m_allow_smaller_buffers ? 1 : frames;
        hard.block_max = frames;
        hard.rate = host.m_sample_rate;
        hard.latencies = latencies;
        const anira::ContractHandle contract(hard);
        m_prepared.store(false, std::memory_order_release);
        m_last_host.reset();
        anira_error err = ANIRA_ERROR_INIT;
        anira::detail::check(anira_handler_prepare(m_handler, contract.native(), &err), err);

        const anira_plan_report* report = anira_handler_plan_report(m_handler);
        uint32_t count = anira_plan_report_num_plans(report);
        std::vector<anira_plan_info> rows(count);
        if (count > 0) {
            anira::detail::check(
                anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
                "anira_plan_report_plans");
        }
        m_plan_backend.clear();
        for (uint32_t i = 0; i < count; ++i) {
            m_plan_backend.push_back(to_backend(anira::EngineRef{
                .kind = static_cast<anira_engine>(rows[i].engine),
                .id = rows[i].engine_id != nullptr ? std::string_view(rows[i].engine_id)
                                                   : std::string_view()}));
        }
        const uint32_t requested = m_requested.load(std::memory_order_relaxed);
        if (requested != k_no_request) {
            select(static_cast<InferenceBackend>(requested));
        } else {
            anira_handler_set_plan(m_handler, initial_plan());
        }
        m_last_host = host;
        m_last_latencies = latencies;
        m_prepared.store(true, std::memory_order_release);
    }

    /// The 2.x initial selection over the plan table: the custom plan for a caller's engine, else
    /// the first built-in plan, else the custom plan, else plan 0.
    uint32_t initial_plan() const noexcept {
        std::optional<uint32_t> custom;
        std::optional<uint32_t> built_in;
        for (uint32_t i = 0; i < m_plan_backend.size(); ++i) {
            if (m_plan_backend[i] == CUSTOM) {
                if (!custom.has_value()) { custom = i; }
            } else if (!built_in.has_value()) {
                built_in = i;
            }
        }
        if (custom.has_value() && m_caller_engine) { return *custom; }
        if (built_in.has_value()) { return *built_in; }
        return custom.value_or(0);
    }

    /// The first plan of the backend; none keeps the selection and logs once.
    void select(InferenceBackend backend) noexcept ANIRA_NONBLOCKING {
        for (uint32_t i = 0; i < m_plan_backend.size(); ++i) {
            if (m_plan_backend[i] == backend) {
                anira_handler_set_plan(m_handler, i);
                return;
            }
        }
        anira_log_rt(ANIRA_LOG_ERROR,
                     "anira.compat",
                     "set_inference_backend: no plan runs on this backend; the selection is "
                     "unchanged",
                     static_cast<int32_t>(to_engine(backend).kind),
                     0);
    }

    static uint32_t num_slots(const std::vector<anira_tensor>& tensors) noexcept ANIRA_NONBLOCKING {
        return static_cast<uint32_t>(tensors.size());
    }

    /// shape[1] and, for a count above 0, the caller's channel pointers; a count of 0 leaves the
    /// slot out and its pointers unread.
    static void present(anira_tensor& tensor,
                        std::vector<void*>& planes,
                        const float* const* channels,
                        size_t count) noexcept ANIRA_NONBLOCKING {
        tensor.shape[1] = static_cast<int64_t>(count);
        for (size_t channel = 0; channel < planes.size(); ++channel) {
            // A plane is a void*: nothing writes through an input's (read-only), and an
            // output's channels came in without the const.
            planes[channel] = count > 0 ? const_cast<float*>(channels[channel]) : nullptr;
        }
    }

    /// The single forms: one slot per side carried, every other slot empty.
    void carry_one(size_t slot,
                   const float* const* input_data,
                   size_t num_input_samples,
                   float* const* output_data,
                   size_t num_output_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t i = 0; i < m_single_in.size(); ++i) {
            m_single_in[i] = i == slot ? input_data : nullptr;
            m_single_in_counts[i] = i == slot ? num_input_samples : 0;
        }
        for (size_t i = 0; i < m_single_out.size(); ++i) {
            m_single_out[i] = i == slot ? output_data : nullptr;
            m_single_out_counts[i] = i == slot ? num_output_samples : 0;
        }
    }

    /// The input half of a call: a Streamed slot presented, a Static slot's values into the
    /// processor (at most its size), a State slot never carried.
    void take_inputs(const float* const* const* input_data,
                     const size_t* num_input_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
            const size_t count = num_input_samples[slot];
            if (m_input_roles[slot] == ANIRA_ROLE_STREAMED) {
                present(m_inputs[slot],
                        m_input_planes[slot],
                        count > 0 ? input_data[slot] : nullptr,
                        count);
            } else if (m_input_roles[slot] == ANIRA_ROLE_STATIC && count > 0) {
                const float* values = input_data[slot][0];
                const size_t num_values = std::min(count, m_input_sizes[slot]);
                for (size_t j = 0; j < num_values; ++j) { m_pp.set_input(values[j], slot, j); }
            }
        }
    }

    /// The output half, before the call: the Streamed requests presented.
    void present_outputs(float* const* const* output_data,
                         const size_t* num_output_samples) noexcept ANIRA_NONBLOCKING {
        for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
            if (m_output_roles[slot] != ANIRA_ROLE_STREAMED) { continue; }
            const size_t count = num_output_samples[slot];
            present(m_outputs[slot],
                    m_output_planes[slot],
                    count > 0 ? output_data[slot] : nullptr,
                    count);
        }
    }

    /// The output half, after the call: a Streamed slot's delivered count; a requested Static
    /// slot's values from the processor (at most its size, the count clamped) on a delivered
    /// block, the request zeroed and 0 on a missed one, untouched and 0 on a failure.
    void give_outputs(float* const* const* output_data,
                      size_t* num_output_samples,
                      anira_status status,
                      bool waited) noexcept ANIRA_NONBLOCKING {
        // A wait without an inference loop ran the nonblocking stem: its block stands.
        const bool delivered =
            status == ANIRA_OK || (waited && status == ANIRA_ERROR_INVALID_STATE);
        for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
            if (m_output_roles[slot] == ANIRA_ROLE_STREAMED) {
                num_output_samples[slot] = m_delivered[slot];
                continue;
            }
            const size_t request = num_output_samples[slot];
            num_output_samples[slot] = 0;
            if (m_output_roles[slot] != ANIRA_ROLE_STATIC || request == 0) { continue; }
            float* values = output_data[slot][0];
            if (delivered) {
                const size_t num_values = std::min(request, m_output_sizes[slot]);
                for (size_t j = 0; j < num_values; ++j) { values[j] = m_pp.get_output(slot, j); }
                num_output_samples[slot] = num_values;
            } else if (status == ANIRA_MISSED) {
                for (size_t j = 0; j < request; ++j) { values[j] = 0.F; }
            }
        }
    }

    anira::Context m_context;
    InferenceConfig& m_config;
    PrePostProcessor& m_pp;
    anira::ModelConfig m_model;  ///< the private copy: the anchor is set here, never on m_config
    std::shared_ptr<anira::Engine> m_engine;
    std::shared_ptr<LegacyProcessorStage> m_stage;
    anira_handler* m_handler = nullptr;
    std::string m_anchor;  ///< the anchor the current C handler was created with; "" = default
    std::optional<HostConfig> m_last_host;
    std::map<std::string, uint32_t> m_last_latencies;
    std::atomic<bool> m_prepared{false};
    std::atomic<bool> m_non_realtime{false};
    std::atomic<uint32_t> m_requested{k_no_request};
    bool m_blocking = false;       ///< the configuration's blocking_ratio is above 0
    bool m_caller_engine = false;  ///< the CUSTOM row runs the caller's engine
    InferenceBackend m_initial_backend = CUSTOM;
    std::vector<InferenceBackend> m_plan_backend;  ///< per plan of the last report
    // The float face, sized at construction: one tensor per slot over the planes.
    std::vector<anira_tensor> m_inputs;
    std::vector<anira_tensor> m_outputs;
    std::vector<std::vector<void*>> m_input_planes;
    std::vector<std::vector<void*>> m_output_planes;
    std::vector<anira_role> m_input_roles;
    std::vector<anira_role> m_output_roles;
    std::vector<size_t> m_input_sizes;   ///< the element count of a Static input, else 0
    std::vector<size_t> m_output_sizes;  ///< the element count of a Static output, else 0
    std::vector<size_t> m_delivered;
    std::vector<const float* const*> m_single_in;
    std::vector<size_t> m_single_in_counts;
    std::vector<float* const*> m_single_out;
    std::vector<size_t> m_single_out_counts;
};

}  // namespace anira::v2

#endif  // ANIRA_COMPAT_V2_HPP
