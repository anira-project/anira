#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#include <concurrentqueue.h>
#ifndef __EMSCRIPTEN__
#include <blockingconcurrentqueue.h>
#endif

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"
#include "../abi/enums.h"
#include "../abi/status.h"
#include "../abi/tensor.h"
#include "../utils/Buffer.h"
#include "../utils/HostConfig.h"
#include "../utils/InferenceBackend.h"
#include "../utils/RealtimeSanitizer.h"
#include "../utils/RingBuffer.h"
#include "../utils/RtLatch.h"
#include "../utils/Semaphore.h"

namespace anira {

namespace backend {
/// The engine room's interface (src/backends/Adapter.h): what a plan of the session runs on.
/// Held through a std::shared_ptr, which needs no complete type here; no engine header is
/// included by any public header.
class Adapter;
}  // namespace backend

/**
 * @brief Session management class for individual inference instances
 *
 * The SessionElement class represents a single inference session, managing all
 * resources and state required for neural network inference processing. Each session
 * is independent and can have different configurations, backends, and processing
 * parameters while sharing the global inference thread pool and core.
 *
 * Key responsibilities:
 * - Managing input/output ring buffers for continuous audio streaming
 * - Coordinating with backend processors (LibTorch, ONNX, TensorFlow Lite)
 * - Latency compensation (the latency itself is computed by LatencyCalculator)
 * - Managing thread-safe data structures for multi-threaded processing
 * - Session lifecycle management and resource cleanup
 *
 * The session uses ring buffers for efficient audio streaming and maintains
 * multiple thread-safe structures to enable concurrent processing without
 * blocking the audio thread. Latency, the number of those structures and the ring
 * sizes are computed in closed form by LatencyCalculator from the model
 * characteristics and the host audio configuration.
 *
 * @note Each session has a unique ID and maintains its own processing state
 *       while participating in the global inference scheduling system.
 */
/**
 * @brief The element type of every send and receive ring of a session, per slot.
 *
 * What the driver pushes into an input ring and pops out of an output ring: the ring dtype the
 * host declared for the slot on the Hard contract (`anira_contract_hard_set_ring_dtype`). A slot
 * beyond the end of a vector, or an empty vector, is float32, the 2.x default and what the 2.x
 * InferenceManager::prepare() passes for every slot.
 */
struct RingDtypes {
    std::vector<anira_dtype> m_inputs;   ///< Per input tensor: the send ring's element type
    std::vector<anira_dtype> m_outputs;  ///< Per output tensor: the receive ring's element type
};

/**
 * @brief A caller's latency per output tensor, replacing the computed one.
 *
 * One entry per output tensor, in samples of that output's stream: a value of 0 or more
 * replaces the latency LatencyCalculator computed for that output (clamped up to the model's
 * internal latency, which the receive ring must still cover), a negative value keeps the
 * computed one. An empty vector, or one whose length is not the number of output tensors,
 * keeps every computed latency. A non-streamable output has no stream latency whatever its
 * entry says.
 */
struct CustomLatencies {
    std::vector<long> m_outputs;  ///< Per output tensor: the latency in samples, or negative
};

class ANIRA_API SessionElement {
public:
    /**
     * @brief Constructor that initializes a session with specified components
     *
     * Creates a new session element with a unique ID and associates it with
     * the provided preprocessing/postprocessing pipeline and inference configuration.
     * The session is not fully initialized until prepare() is called.
     *
     * @param new_session_id Unique identifier for this session
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration containing model settings
     * @param producer_token Producer token bound to the global inference queue, moved into the
     * session (see m_producer_token)
     * @param rt_latch The real-time latch this session records its failures into (see m_rt):
     * a 3.x handler's own, so anira_handler_rt_error reads one word; nullptr (a 2.x session)
     * gives the session its own latch (m_rt_own)
     */
    SessionElement(int new_session_id,
                   PrePostProcessor& pp_processor,
                   InferenceConfig& inference_config,
                   moodycamel::ProducerToken&& producer_token,
                   anira::RtLatch* rt_latch = nullptr);

    /**
     * @brief Wait-free clear of the session's audio-thread-owned state.
     *
     * Resets the audio-thread-owned state (send/receive ring buffers, timestamp
     * bookkeeping, latency re-seed) and any ThreadSafeStruct that is currently
     * free, but never touches a struct that a worker still holds in flight
     * (m_free == false). It therefore does NOT require Core::drain_inference_queue()
     * as a precondition and never blocks the caller.
     *
     * Correctness is provided by the session generation (see m_generation): the
     * caller (Core::reset_session) bumps the generation first, which makes every
     * already-dispatched inference "stale" — its result is ignored by
     * Core::new_data_request() and its struct is reclaimed by
     * Core::reclaim_stale_structs() (run from new_data_submitted()) once the
     * worker publishes completion. Valid for all session types; the stateful
     * dispatch chain is reconciled separately (see discard_pending_dispatches()
     * and the generation filter in try_acquire_next_dispatch()).
     */
    void clear();

    /**
     * @brief Prepares the session for processing with specified audio configuration
     *
     * Initializes all buffers, calculates latencies, and configures the session
     * for processing with the provided host audio configuration. This method
     * must be called before the session can process audio data.
     *
     * The host configuration's reference stream (HostConfig::resolve_reference) is
     * resolved here, once, and stored in m_reference / m_input_driven for the
     * real-time path.
     *
     * @param spec Host configuration containing sample rate, buffer size, and audio settings
     * @param custom_latencies A caller's latency per output tensor (see CustomLatencies; empty
     * keeps every computed latency); entries for non-streamable outputs are ignored, their
     * latency is always 0
     * @param ring_dtypes The element type of every send and receive ring, per slot (see
     * RingDtypes; float32 for every slot the vectors do not name); the ring sizes are element
     * counts, so they do not depend on it
     * @throws std::invalid_argument if the host config's reference stream cannot be resolved
     *         (explicit reference out of range or not streamable, or no streamable tensor at all),
     *         or if a ring dtype is not one of the scalar dtypes the rings store
     */
    void prepare(const HostConfig& spec,
                 const CustomLatencies& custom_latencies = {},
                 const RingDtypes& ring_dtypes = {});

    /**
     * @brief Whether every streamable receive ring can take one more inference result
     *
     * True if each streamable output's ring buffer has at least postprocess_output_size
     * free samples (trivially true when no output is streamable). Used by the push-side
     * collection in Core::collect_completed() so that a completed result is only
     * post-processed when it fits, and unread output is never overwritten.
     *
     * @return True if a completed inference can be post-processed without overflowing a ring
     */
    bool receive_rings_have_room();

    std::vector<RingBuffer> m_send_buffer;  ///< Ring buffers for input data streaming to inference
    std::vector<RingBuffer> m_receive_buffer;  ///< Ring buffers for output data streaming from
                                               ///< inference

    /**
     * @brief Thread-safe data structure for concurrent inference processing
     *
     * This nested structure provides thread-safe coordination between the audio
     * thread and inference threads. Each structure can hold one inference request
     * and includes synchronization primitives to ensure safe concurrent access.
     *
     * The structure uses atomic operations and semaphores to coordinate:
     * - Availability checking (m_free)
     * - Completion notification (m_done_semaphore, m_done_atomic)
     * - Data integrity during concurrent access
     * - Timestamping for latency tracking
     */
    struct ThreadSafeStruct {
        /**
         * @brief Constructor that initializes thread-safe structure with tensor dimensions
         *
         * Creates buffers for input and output tensors with the specified sizes
         * and initializes synchronization primitives.
         *
         * @param tensor_input_size Vector of input tensor sizes
         * @param tensor_output_size Vector of output tensor sizes
         */
        ThreadSafeStruct(const std::vector<size_t>& tensor_input_size,
                         const std::vector<size_t>& tensor_output_size);

        /**
         * @brief Builds the descriptor of every tensor of the chunk over its buffer
         *
         * Control thread, once per prepare, after the buffers exist: one anira_tensor per
         * slot of either side, over the buffer's memory, with the spec's rank and extents
         * (the universal TensorShape rows) and the storage's dtype, float32, all-zero
         * strides. What the engine call hands its adapter (anira_engine_ctx::inputs /
         * outputs) and what a stage context copies for a slot.
         *
         * @param input_shapes The spec's extents per input tensor
         * @param output_shapes The spec's extents per output tensor
         */
        void bind_tensors(const TensorShapeList& input_shapes,
                          const TensorShapeList& output_shapes);

        /**
         * @brief Re-points every descriptor at its buffer's memory of the moment
         *
         * A field fill on the driving thread at every claim of the struct
         * (Core::pre_process): a 2.x processor of this line may swap a buffer's memory with
         * its own during an inference (TFLiteProcessor, until its adapter binds the
         * descriptors), so a struct holds another block each time it comes round; the
         * descriptors follow. Goes with the swap.
         */
        void rebind_tensors() noexcept;

        std::atomic<bool> m_free{true};  ///< Atomic flag indicating if this structure is available
                                         ///< for use
        anira::Semaphore m_done_semaphore{0};    ///< Semaphore for blocking wait on inference
                                                 ///< completion
        std::atomic<bool> m_done_atomic{false};  ///< Atomic flag for non-blocking completion
                                                 ///< checking

        unsigned long m_time_stamp;  ///< Timestamp for latency tracking and debugging. Written
                                     ///< only on the session's driving (audio) thread — keep it
                                     ///< that way; it is a plain field.
        // Session generation this struct was dispatched under (stamped in
        // Core::pre_process). A wait-free reset bumps SessionElement::m_generation;
        // a dispatch whose stamp no longer matches is "stale" — its result is discarded
        // and the struct reclaimed. Written on the audio thread at dispatch, read on the
        // worker thread and audio thread; a dispatch's stamp is stable for its lifetime,
        // so a plain value (not atomic) is sufficient. 64-bit so the counter cannot
        // wrap within any realistic session lifetime (unsigned long is 32-bit on
        // LLP64/Windows).
        uint64_t m_dispatch_generation{0};
        // Dispatch-gate token this struct was dispatched under (stamped in
        // try_acquire_next_dispatch, stateful sessions only). The holder passes it
        // back to release_dispatch(), whose epoch check makes a laggard release
        // from before a force_reset_dispatch_chain() fail silently instead of
        // stomping a newer era's in-flight dispatch. Plain field: stable for the
        // dispatch's lifetime, synced by the gate/queue handoffs.
        uint64_t m_dispatch_epoch{0};
        // The plan this chunk runs on: the dense index into SessionElement::m_plans,
        // stamped in Core::pre_process from SessionElement::m_current_plan with one load.
        // The hooks, the engine call and post_process all resolve this stamp and never read
        // the session's atomic again, so a plan switch that lands while the chunk is queued
        // or in flight cannot split it across two plans (or run two engines, or none, for
        // it). The index and not the backend: two plans may run on one backend (two variants
        // of a model, two providers of an engine), and only the index tells them apart.
        // Plain field: written on the driving thread at dispatch, stable for the dispatch's
        // lifetime, synced by the queue handoffs like the two stamps above.
        uint32_t m_plan{0};
        // What a stage left on the chunk (a 3.x handler's StageProcessor; a 2.x
        // processor never writes it): ANIRA_OK, or the status of the phase callback that
        // failed. Core::pre_process resets it, then completes the chunk as zeros without
        // enqueueing it when pre_process failed; InferenceThread::do_inference reads it around
        // the hooks and skips the engine call, or zeroes the outputs, on a failure. Plain
        // field: written by the thread that holds the struct, synced by the queue handoffs
        // like the stamps above.
        anira_status m_stage_status{ANIRA_OK};
        // Whether the chunk completed as zeros: dropped (SessionElement::complete_with_zeros:
        // a full queue, a failed pre_process, a stale or unclaimable task) or failed on the
        // inference thread (a throw, a failed before_inference or after_inference), its output
        // buffers cleared. Core::pre_process resets it. A 3.x stage processor reads it in
        // post_process: such a chunk does not overwrite the handler's Static output store.
        // Plain field, synced by the queue handoffs like the stamps above (written before the
        // done signal, read after it).
        bool m_completed_as_zeros{false};
        // The struct's position in SessionElement::m_inference_queue: the chunk's entry
        // (anira_engine_ctx::entry, anira_stage_ctx::entry), set by prepare with the pool.
        uint32_t m_entry{0};
        std::vector<BufferF> m_tensor_input_data;   ///< Input tensor data buffers
        std::vector<BufferF> m_tensor_output_data;  ///< Output tensor data buffers
        // One descriptor per slot of either side over the buffers above (bind_tensors,
        // rebind_tensors): the engine call's anira_engine_ctx names these arrays, a stage
        // context copies one for a slot. Written on the driving thread at the claim, read on
        // the inference thread; synced by the queue handoffs like the fields above.
        std::vector<anira_tensor> m_input_tensors;
        std::vector<anira_tensor> m_output_tensors;
    };

    std::vector<std::shared_ptr<ThreadSafeStruct>> m_inference_queue;  ///< Pool of thread-safe
                                                                       ///< structures for
                                                                       ///< concurrent processing

    /**
     * @brief One plan of the table: the adapter the plan runs on and what a chunk stamped
     * with the plan reports.
     */
    struct PlanSlot {
        /// The prepared model the plan runs on (never null in a registered session): shared
        /// with the core's pool and with every other session on the same model, or this
        /// session's own (a session-exclusive model, a 2.x custom backend, the roundtrip).
        std::shared_ptr<backend::Adapter> m_adapter;
        /// The engine and provider a stage context reports for the chunk: a built-in engine,
        /// or ANIRA_ENGINE_NONE with m_engine_id for a custom one.
        anira_engine m_engine = ANIRA_ENGINE_NONE;
        anira_provider m_provider = ANIRA_PROVIDER_DEFAULT;
        std::string m_engine_id;
        /// The 2.x backend the plan runs on: what set_backend selects by and get_backend
        /// answers (InferenceManager); the value the 2.x pre- and post-processing virtuals
        /// receive for the chunk.
        InferenceBackend m_legacy_backend = InferenceBackend::CUSTOM;
        /// A 2.x row for a backend of the build without a model: the roundtrip runs for the
        /// plan, and the inference thread logs RtSite::NoModelForBackend once per prepare.
        bool m_missing_model = false;
    };

    /**
     * @brief The plan table: dense plan index -> the plan.
     *
     * Built by Core::create_session from the requests it is given (backend::PlanRequest): the
     * 2.x constructors ask for one row per configured model, in m_model_data order, then every
     * other backend of this build, so that every backend a 2.x caller can name has a row; a
     * 3.x handler asks for exactly its plans. Several rows may name one backend. Never empty
     * in a registered session, and never changed afterwards, so the driving thread and the
     * inference threads read it without synchronization.
     */
    std::vector<PlanSlot> m_plans;

    /**
     * @brief The selected plan: the one the next submitted chunk is stamped with.
     *
     * The whole runtime selection is this one atomic (select_plan stores it, a chunk takes
     * it with one load in Core::pre_process), so concurrent callers have no pair to tear.
     * Always below m_plans.size(). Initialized by Core::create_session to the first
     * configured model whose engine has an adapter (the CUSTOM row when a custom backend
     * was provided or nothing matches).
     */
    std::atomic<uint32_t> m_current_plan{0};

    /// Selects a plan: one relaxed store. False, and nothing stored, for an index out of range.
    bool select_plan(uint32_t plan) noexcept;

    /// The 2.x backend a plan runs on. CUSTOM for an index out of range, which a stamp never is.
    InferenceBackend plan_backend(uint32_t plan) const noexcept;

    /// The first plan that runs on a 2.x backend (the 2.x selection by backend), if any does.
    std::optional<uint32_t> plan_of_backend(InferenceBackend backend) const noexcept;

    /// No chunk carries this dispatch stamp: what m_engine_generation holds after prepare, so
    /// that the first inference of the session is the first of a new stream.
    static constexpr uint64_t k_no_generation = UINT64_MAX;

    /**
     * @brief The dispatch stamp the engine last ran under: the engine's reset boundary.
     *
     * A session-exclusive session only, whose inferences run one at a time and in order
     * (the dispatch gate): InferenceThread::inference compares the chunk's dispatch stamp
     * with it, and a difference is the first inference of a new stream (after prepare, after
     * a reset), so the plan's adapter resets its engine's own state right before that
     * inference's process and the stamp is adopted. Read and written on the inference thread
     * alone, between the gate's acquire and release; set to k_no_generation by prepare, at
     * quiescence. A shared (pooled) adapter is never reset: only an exclusive session gets
     * here.
     */
    uint64_t m_engine_generation = k_no_generation;

    unsigned long m_current_queue = 0;         ///< Current position in the inference queue
    std::vector<unsigned long> m_time_stamps;  ///< Vector of timestamps for performance monitoring
    size_t m_pending_pull_samples = 0;  ///< Generator sessions only (!m_input_driven): samples of
                                        ///< the reference output the driving thread has demanded
                                        ///< that no submitted inference covers yet. Plain field,
                                        ///< written and read only on the session's driving thread
                                        ///< (like m_time_stamps).

    const int m_session_id;  ///< Unique identifier for this session (immutable)

    std::atomic<bool> m_initialized{false};   ///< Atomic flag indicating if the session is fully
                                              ///< initialized
    std::atomic<int> m_active_inferences{0};  ///< Atomic counter of currently active inference
                                              ///< operations

    // Monotonic generation counter, bumped by Core::reset_session() (audio
    // thread) and Core::prepare_session() (control thread, quiescent). Every
    // inference dispatched under an earlier generation is "stale": its output is
    // discarded and its struct reclaimed, without the caller ever waiting for the
    // worker. Workers read it to decide whether to skip a stale dispatch. seq_cst
    // on the write pairs with the worker's register-before-read of
    // m_active_inferences (store-buffering), matching the existing m_initialized
    // handshake. 64-bit: wrap-around (ABA) is unreachable even at per-block reset
    // rates.
    std::atomic<uint64_t> m_generation{0};

    // This session's explicit producer token for the global inference queue.
    // Pinning one token per session keeps enqueue allocation-free on the audio
    // thread (no implicit-producer registration) and gives the token RAII
    // lifetime: the underlying producer slot is recycled when the session is
    // destroyed. A ProducerToken must never be used by two threads at once —
    // this holds here because all enqueues of a session are serialized: the
    // non-stateful path enqueues only from the session's single driving thread,
    // and the stateful path enqueues only while holding the
    // m_stateful_dispatch_gate (whose acquire/release ordering also makes
    // the token's state visible when ownership migrates between threads).
    moodycamel::ProducerToken m_producer_token;  ///< Per-session producer token for the global
                                                 ///< inference queue

    // --- Stateful in-order dispatch ---
    // For stateful models, only ONE of this session's tasks may be in the global
    // inference queue (and therefore running) at a time. Prepared tasks wait here
    // in submission order and are released one at a time as each completes, which
    // guarantees in-order, mutually-exclusive execution without spinning. Other
    // sessions are unaffected and keep using the shared thread pool in parallel.
    //
    // Gate word layout: bit 0 = busy (a task of this session is queued or
    // running), bits 63..1 = dispatch epoch. The epoch gives the gate an owner
    // identity: force_reset_dispatch_chain() (quiescent contexts only) opens a
    // new epoch after waiting out any transient holder, and release_dispatch()
    // is an epoch-checked CAS — so any release still carrying a pre-reset token
    // is inert, whatever the interleaving. A safety invariant, not a
    // specific-scenario patch: it holds for every release path (worker tail,
    // stale skip, dispatch failure, drain).
    static constexpr uint64_t k_dispatch_busy = 1;
    // The wait-free reset()/dispatch guarantees assume these 64-bit atomics are
    // implemented lock-free; a target where they are not would silently reintroduce
    // locks into [[clang::nonblocking]] paths — fail the build loudly instead.
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
                  "anira's wait-free reset/dispatch requires lock-free 64-bit atomics");
    std::atomic<uint64_t> m_stateful_dispatch_gate{0};  ///< {epoch, busy} dispatch gate; busy while
                                                        ///< a stateful task of this session is
                                                        ///< queued or running
    // Pre-sized in the constructor to m_num_parallel_processors — a pending
    // entry is always a distinct ThreadSafeStruct, so the depth is bounded —
    // and fed exclusively through m_dispatch_producer_token, so enqueues never
    // allocate: neither on the audio thread (real-time safety) nor, on
    // WebAssembly, from a non-main WASM instance (which must not touch the
    // shared allocator).
    moodycamel::ConcurrentQueue<std::shared_ptr<ThreadSafeStruct>>
        m_dispatch_pending;  ///< Prepared-but-not-yet-dispatched stateful
                             ///< tasks, in submission order
    // Same single-producer discipline as m_producer_token above: only the
    // session's driving thread enqueues pending dispatches.
    moodycamel::ProducerToken m_dispatch_producer_token;  ///< Explicit producer for
                                                          ///< m_dispatch_pending

    /** @brief Queue a prepared stateful task awaiting dispatch (called in submission order on the
     * session's driving thread; allocation-free). If the pending queue rejects the task — which
     * the capacity bound makes unreachable — it is completed with zeroed output instead. */
    void enqueue_pending_dispatch(std::shared_ptr<ThreadSafeStruct> thread_safe_struct);
    /** @brief Claim the next stateful task to dispatch, or nullptr if one is already in flight or
     * none are pending. Pending entries whose dispatch generation is stale (a wait-free reset
     * bumped m_generation after they were prepared) are returned straight to the free pool and
     * skipped: they were never handed to a worker, so the gate-holder owns them exclusively. The
     * returned struct carries the gate token in m_dispatch_epoch, which the holder must pass back
     * to release_dispatch(). */
    std::shared_ptr<ThreadSafeStruct> try_acquire_next_dispatch();
    /** @brief Mark the in-flight stateful task finished, allowing the next to be dispatched.
     * Epoch-checked: a token from before a force_reset_dispatch_chain() fails silently. */
    void release_dispatch(uint64_t token);
    /** @brief Wait-free reset kick for the stateful dispatch chain (driving thread only, called
     * right after the generation bump): if no task is in flight, acquires the gate, returns every
     * pending entry to the free pool (all are stale — same driving thread, so nothing fresh can
     * have been prepared since the bump) and releases. Never enqueues, so the reset path stays
     * free of queue/semaphore/logging syscalls. If a task is in flight, does nothing: the worker
     * filters the stale prefix at its next task boundary. */
    void discard_pending_dispatches();
    /** @brief Quiescent-only (a drain has run; no task of this session is queued or running,
     * though a laggard worker may still transiently hold the gate while filtering stale
     * entries): waits out any transient gate holder, opens a new dispatch epoch (so any
     * release carrying a pre-reset token is inert, whatever the interleaving), then flushes
     * the pending queue. Blocking (sleeps in 50us steps while the gate is held) — called
     * only from prepare(), never from a real-time path. */
    void force_reset_dispatch_chain() ANIRA_BLOCKING;
    /** @brief Complete a task without running inference: zero its output tensors and signal
     * completion. Used when the global queue rejects a task, so the dropped inference still
     * yields (silent) output at its correct stream position and the struct is freed normally. */
    void complete_with_zeros(const std::shared_ptr<ThreadSafeStruct>& thread_safe_struct);

    PrePostProcessor& m_pp_processor;  ///< Reference to the preprocessing/postprocessing pipeline
    InferenceConfig& m_inference_config;  ///< Reference to the inference configuration

    RtLatch m_rt_own;  ///< The session's own real-time latch: what m_rt points at when the
                       ///< constructor received no latch (a 2.x session); nothing reads its
                       ///< word, but a failed inference records ENGINE into it all the same
    RtLatch* m_rt;     ///< The real-time latch the session's failures are recorded into (a
                       ///< failed inference records ANIRA_ERROR_ENGINE on the inference
                       ///< thread): the 3.x handler's own (anira_handler_rt_error reads it),
                       ///< else &m_rt_own. Set by the constructor and published by the
                       ///< lifecycle lock that registers the session; never null
    // What every ring of this session points at (anira_ring::owner), set by prepare(): m_rt,
    // and the name of the stage a 3.x handler's chain is running on the driving thread.
    RingOwner m_ring_owner;

    // Written by InferenceManager::set_non_realtime() -- typically from a control/UI
    // thread, e.g. a host toggling offline bounce/render mode -- and read on the
    // audio thread inside Core::new_data_request(). A plain bool would be a
    // data race between those two threads, so this needs real synchronization.
    std::atomic<bool> m_is_non_real_time{false};  ///< True forces new_data_request() to
                                                  ///< block until each pending inference
                                                  ///< completes, ignoring blocking_ratio
                                                  ///< and any deadline, trading real-time
                                                  ///< safety for complete, deterministic
                                                  ///< output (see Core::new_data_request).

    std::vector<unsigned int> m_latency;  ///< Calculated latency values for each output tensor in
                                          ///< samples, index-aligned with the output tensor list;
                                          ///< 0 for non-streamable outputs
    ReferenceStream m_reference;  ///< Reference stream resolved once in prepare(); read on the
                                  ///< real-time path, never re-resolved there
    bool m_input_driven = true;   ///< True if any input tensor is streamable: inference is
                                  ///< triggered by arriving input samples. False for a generator
                                  ///< (no streamable input), whose inference is triggered by
                                  ///< output demand (see m_pending_pull_samples).
    size_t m_num_structs = 0;     ///< Number of allocated thread-safe structures: twice the
                               ///< steady-state bound of LatencyCalculator::get_num_structs(), so
                               ///< that the inferences a wait-free reset() strands until their
                               ///< workers finish never starve the fresh schedule (for testing
                               ///< access)
    std::vector<size_t> m_send_buffer_size;  ///< Calculated send buffer sizes (for testing access)
    std::vector<size_t> m_receive_buffer_size;  ///< Calculated receive buffer sizes (for testing
                                                ///< access)

private:
    /**
     * @brief Whether any input tensor is streamable
     *
     * @return True if at least one input has a non-zero preprocess_input_size
     */
    bool has_streamable_input() const;

    HostConfig m_host_config;  ///< Stored host configuration for this session

#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    ThreadSafeStruct* __doxygen_force_0;  ///< Placeholder for Doxygen documentation
    RingBuffer* __doxygen_force_1;        ///< Placeholder for Doxygen documentation
#endif
};

/**
 * @brief Data structure for passing inference requests between threads
 *
 * The InferenceData struct encapsulates all information needed to perform
 * an inference operation, including the session context and the specific
 * thread-safe data structure containing the input/output buffers.
 *
 * This structure is designed for efficient passing through lock-free
 * concurrent queues and enables decoupled processing between audio
 * threads and inference threads.
 *
 * @note This struct uses shared pointers to ensure safe memory management
 *       across multiple threads and to avoid data copying overhead.
 */
struct InferenceData {
    std::shared_ptr<SessionElement> m_session;  ///< Shared pointer to the session that owns this
                                                ///< inference request
    std::shared_ptr<SessionElement::ThreadSafeStruct> m_thread_safe_struct;  ///< Shared pointer to
                                                                             ///< the thread-safe
                                                                             ///< data structure
                                                                             ///< containing buffers
                                                                             ///< and
                                                                             ///< synchronization
};

/**
 * @brief Queue type used for passing InferenceData to the inference threads
 *
 * On native builds this is a moodycamel::BlockingConcurrentQueue so that
 * inference threads can optionally block on the queue's semaphore instead of
 * polling (see WaitStrategy). The blocking queue is a strict superset of the
 * plain ConcurrentQueue API, and as long as no consumer ever blocks, its
 * enqueue never makes a syscall — so CoreConfigs using
 * WaitStrategy::SpinBackoff keep the exact lock-free behavior of the plain
 * queue, at the cost of one extra atomic operation per enqueue/dequeue.
 *
 * On WebAssembly builds the plain ConcurrentQueue is used: inference loops
 * are driven cooperatively by JS Workers across WASM instances that share
 * memory via postMessage, so there is no pthreads runtime to block on, and a
 * blocked worker could not service its JS event loop anyway.
 */
#ifdef __EMSCRIPTEN__
using InferenceQueue = moodycamel::ConcurrentQueue<InferenceData>;
#else
using InferenceQueue = moodycamel::BlockingConcurrentQueue<InferenceData>;
#endif

}  // namespace anira

#endif  // ANIRA_SESSIONELEMENT_H