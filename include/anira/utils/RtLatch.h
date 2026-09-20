#ifndef ANIRA_RTLATCH_H
#define ANIRA_RTLATCH_H

#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/system/Exports.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

/**
 * @file RtLatch.h
 * @brief The real-time latch behind anira_handler_rt_error and the once-per-prepare records
 * of the scheduler's operational sites.
 *
 * A real-time refusal or failure carries no anira_error: the entry records it into a latch
 * (last-wins into the rt_error word, one bit per kind) and logs it through the real-time
 * queue on the kind's first occurrence since the latch was last re-armed; later occurrences
 * are counted, and the drain's summary reports a persisting condition. Every member is a
 * relaxed atomic: the record through the queue is what carries the information, the word is
 * diagnostic. Nothing here allocates, locks or throws.
 *
 * Two kinds of latch exist: one per handler (the session points at it from construction,
 * so a failed inference on the inference thread lands in the handler's word), and one per
 * operational site of the scheduler (RtSite), process-wide, re-armed by every prepare.
 */

namespace anira {

/// The status kinds anira_handler_rt_error latches: one bit each. ANIRA_ERROR_CAPACITY is
/// back-pressure and never latches.
enum class RtKind : uint32_t {
    WrongContract = 1U << 0,
    NotPrepared = 1U << 1,
    Config = 1U << 2,
    InvalidState = 1U << 3,
    InvalidArgument = 1U << 4,
    Engine = 1U << 5,
};

/// The kind bit of a status; 0 for a status that never latches.
constexpr uint32_t rt_kind_bit(anira_status status) noexcept {
    switch (status) {
        case ANIRA_ERROR_WRONG_CONTRACT: return static_cast<uint32_t>(RtKind::WrongContract);
        case ANIRA_ERROR_NOT_PREPARED: return static_cast<uint32_t>(RtKind::NotPrepared);
        case ANIRA_ERROR_CONFIG: return static_cast<uint32_t>(RtKind::Config);
        case ANIRA_ERROR_INVALID_STATE: return static_cast<uint32_t>(RtKind::InvalidState);
        case ANIRA_ERROR_INVALID_ARGUMENT: return static_cast<uint32_t>(RtKind::InvalidArgument);
        case ANIRA_ERROR_ENGINE: return static_cast<uint32_t>(RtKind::Engine);
        default: return 0;
    }
}

/**
 * @brief A per-handler (or per-site) real-time latch: the last failure, a bit per kind, the
 * suppressed count and the count the drain's summary last reported.
 *
 * Constant-initialized, relaxed everywhere: the record it gates is what carries the
 * information, the word is diagnostic. Re-armed by anira_handler_prepare and
 * anira_handler_reset (a handler's latch) or by every prepare of every session (a site's).
 */
struct RtLatch {
    /// The last recorded status (anira_status as int32_t); ANIRA_OK after a re-arm.
    std::atomic<int32_t> m_rt_error{ANIRA_OK};
    /// One bit per RtKind already logged since the last re-arm (a site latch uses bit 0).
    std::atomic<uint32_t> m_latched{0};
    /// Occurrences not logged since the last re-arm.
    std::atomic<uint32_t> m_suppressed{0};
    /// The summary's last reading of m_suppressed; zeroed by rearm (two writers, relaxed).
    std::atomic<uint32_t> m_reported{0};

    /**
     * @brief Records a failure (last-wins into rt_error).
     * @return True on this kind's first occurrence since the last re-arm: log it. False when
     * it was suppressed and counted, or for a kind that never latches.
     */
    bool record(anira_status status) noexcept ANIRA_NONBLOCKING {
        const uint32_t bit = rt_kind_bit(status);
        if (bit == 0) { return false; }
        m_rt_error.store(static_cast<int32_t>(status), std::memory_order_relaxed);
        if ((m_latched.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
            m_suppressed.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    /**
     * @brief The per-site form: one kind.
     * @return True on the first occurrence since the last re-arm; later ones are counted.
     */
    bool first() noexcept ANIRA_NONBLOCKING {
        if (m_latched.exchange(1, std::memory_order_relaxed) != 0) {
            m_suppressed.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    /**
     * @brief Re-arms every kind and clears rt_error.
     * @return The count suppressed since the last re-arm (the final count, whatever the
     * summary reported in between).
     */
    uint32_t rearm() noexcept ANIRA_NONBLOCKING {
        m_latched.store(0, std::memory_order_relaxed);
        m_rt_error.store(ANIRA_OK, std::memory_order_relaxed);
        m_reported.store(0, std::memory_order_relaxed);
        return m_suppressed.exchange(0, std::memory_order_relaxed);
    }

    /// The last recorded status; ANIRA_OK when nothing was recorded since the last re-arm.
    anira_status rt_error() const noexcept ANIRA_NONBLOCKING {
        return static_cast<anira_status>(m_rt_error.load(std::memory_order_relaxed));
    }
};

/// The operational real-time sites of the scheduler, one process-wide latch each.
enum class RtSite : uint8_t {
    NoFreeInferenceQueue,      // S1  Core: no free inference queue in the session
    EnqueueDropped,            // S2  Core: the global job queue refused the inference
    RequeueDropped,            // S3  Core: the requeue of the inference data failed
    OutputNotConsumed,         // S4  InferenceManager: a receive buffer is full
    WaitWithoutSemaphore,      // S5  InferenceManager: a wait on a session without semaphores
    CatchUpMissingSamples,     // S6  InferenceManager: missing samples caught up
    MissingSamples,            // S7  InferenceManager: a block missed
    PendingDispatchDropped,    // S8  SessionElement: the pending stateful dispatch queue is full
    NextDispatchDropped,       // S9  InferenceThread: the next dispatch could not be enqueued
    NoLibTorchModel,           // S10 InferenceThread: no LibTorch model, default processor
    NoOnnxRuntimeModel,        // S11 InferenceThread: no ONNX Runtime model
    NoTFLiteModel,             // S12 InferenceThread: no TFLite model
    NoLiteRtModel,             // S13 InferenceThread: no LiteRT model
    NoExecuTorchModel,         // S14 InferenceThread: no ExecuTorch model
    InferenceThreadBodyThrew,  // the catch-all of the inference thread's loop body
    Count
};

/// One short phrase per site, for the summary and the tests (index = the enum value).
inline constexpr std::array<const char*, static_cast<size_t>(RtSite::Count)> k_rt_site_names = {
    "no free inference queue",
    "inference dropped (core queue full)",
    "inference dropped (requeue failed)",
    "output stream not consumed",
    "wait without a semaphore",
    "catch-up of missing samples",
    "missing samples",
    "stateful dispatch dropped (pending queue full)",
    "next dispatch dropped (core queue full)",
    "libtorch model not provided",
    "onnxruntime model not provided",
    "tflite model not provided",
    "litert model not provided",
    "executorch model not provided",
    "inference thread body threw",
};

}  // namespace anira

namespace anira::detail {

/**
 * @brief The process-wide latch of one operational site: a constant-initialized array, no
 * registration, no guard. Re-armed by every prepare of every session.
 */
ANIRA_API RtLatch& rt_site(RtSite site) noexcept;

/**
 * @brief The interval at which the drain's summary reports a persisting real-time condition,
 * in milliseconds; 10000 by default. Exported so the tests can lower it.
 */
ANIRA_API void set_rt_summary_interval_ms(uint32_t ms) noexcept;

/// The summary interval in effect (set_rt_summary_interval_ms).
ANIRA_API uint32_t rt_summary_interval_ms() noexcept;

}  // namespace anira::detail

#endif  // ANIRA_RTLATCH_H
