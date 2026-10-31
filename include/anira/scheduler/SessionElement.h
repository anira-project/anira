#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#include <concurrentqueue.h>
#ifndef __EMSCRIPTEN__
#include <blockingconcurrentqueue.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../abi/enums.h"
#include "../abi/status.h"
#include "../abi/tensor.h"
#include "../system/Exports.h"
#include "../utils/RealtimeSanitizer.h"
#include "../utils/RingBuffer.h"
#include "../utils/RtLatch.h"
#include "../utils/Semaphore.h"
#include "LatencyCalculator.h"

namespace anira {

namespace engine {
/// The engine room's interface (src/engines/Adapter.h): the loaded model a plan of the session
/// runs on and the session's prepared handle over it. Held through smart pointers, which need
/// no complete type here (the destructor of the session is defined where the types are
/// complete); no engine header is included by any public header.
class Loaded;
class Prepared;
}  // namespace engine

/// The jobs that wait for a free shared instance of one loaded model (src/scheduler/
/// InstanceWaitList.h, private to the scheduler): held through a smart pointer like the two
/// engine-room types above.
class InstanceWaitList;

namespace capi {
/// The processor of a C-created handler's session (src/capi/processor.h, private): what the
/// session runs every phase of a chunk on, and what holds the handler's two port vectors, the
/// rings of the session's Streamed slots among them. Held by reference; the session's .cpp
/// files include the private header.
class ProcessorRunner;
/// What the validator derived for the handler's model config, and the Hard half of the
/// contract the handler was prepared under (src/capi/validate.h and src/capi/handles.h,
/// private): the session reads both, with the model config, by reference, for as long as it
/// lives; the handler holds all three.
struct Derived;
struct DerivedSpec;
struct HardContract;
}  // namespace capi

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
 * sizes are computed in closed form by LatencyCalculator from what the validator derived
 * for the model config and from the Hard contract.
 *
 * The session's configuration is the C handler's, read by reference and never copied: the
 * model config of its variant, what the validator derived for it under the contract
 * (capi::Derived: every tensor's extents, hop and channel count, the anchor, the
 * exclusivity) and the contract's Hard half (capi::HardContract: the geometry, the budget, the
 * wait ratio). The handler installs the three before it builds the session and keeps them
 * until the session is released.
 *
 * @note Each session has a unique ID and maintains its own processing state
 *       while participating in the global inference scheduling system.
 */
/**
 * @brief The element type of every send and receive ring of a session, per slot.
 *
 * What the driver pushes into an input ring and pops out of an output ring: the ring dtype the
 * host declared for the slot on the Hard contract (`anira_contract_hard_set_ring_dtype`). A slot
 * beyond the end of a vector, or an empty vector, is float32.
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
     * Creates a new session element with a unique ID and associates it with the C handler's
     * processor runner and configuration. The session is not fully initialized until prepare()
     * is called.
     *
     * @param new_session_id Unique identifier for this session
     * @param processor The handler's processor runner: every phase of a chunk runs on it, and
     * its port vectors hold the session's rings (it outlives the session)
     * @param model The handler's model config (its variant): the model's own latency of every
     * output (it outlives the session)
     * @param derived What the validator derived for the model config under the contract (it
     * outlives the session)
     * @param hard The Hard half of the contract the handler is prepared under (it outlives the
     * session)
     * @param num_instances The instance count, resolved by Core::create_session (1 for an
     * exclusive session, else the model config's max_instances, ANIRA_INSTANCES_AUTO the
     * pool's size, clamped to the pool)
     * @param producer_token Producer token bound to the global inference queue, moved into the
     * session (see m_producer_token)
     * @param rt_latch The real-time latch this session records its failures into (see m_rt):
     * a 3.x handler's own, so anira_handler_rt_error reads one word; nullptr (a 2.x session)
     * gives the session its own latch (m_rt_own)
     */
    SessionElement(int new_session_id,
                   capi::ProcessorRunner& processor,
                   const anira_model_config& model,
                   const capi::Derived& derived,
                   const capi::HardContract& hard,
                   uint32_t num_instances,
                   moodycamel::ProducerToken&& producer_token,
                   anira::RtLatch* rt_latch = nullptr);

    /**
     * @brief Wait-free clear of the session's audio-thread-owned state.
     *
     * Resets the audio-thread-owned state (the rings of the Streamed slots, timestamp
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
     * @brief Prepares the session for the Hard contract it reads
     *
     * Calculates the latencies, sizes the ring of every Streamed slot in its stream port and
     * builds the chunks, for the contract's geometry, budget and wait ratio. This method must
     * be called before the session can process audio data. The host block in hops is stored
     * in m_block_hops and whether an input streams in m_input_driven, for the real-time path.
     *
     * @param custom_latencies A caller's latency per output tensor (see CustomLatencies; empty
     * keeps every computed latency); entries for non-streamable outputs are ignored, their
     * latency is always 0
     * @param ring_dtypes The element type of every send and receive ring, per slot (see
     * RingDtypes; float32 for every slot the vectors do not name); the ring sizes are element
     * counts, so they do not depend on it
     * @throws std::invalid_argument if the contract's geometry cannot size the session (see
     *         LatencyCalculator), or if a ring dtype is not one of the scalar dtypes the rings
     *         store
     */
    void prepare(const CustomLatencies& custom_latencies = {}, const RingDtypes& ring_dtypes = {});

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

    /**
     * @brief The ring of a Streamed input or output slot
     *
     * The ring lives in the slot's stream port, in the handler's port vector the processor
     * runner holds (one container, one index space); the session's prepare sizes it in place.
     * Only a Streamed slot has one: the caller checks that the slot streams (its hop is above 0)
     * before it asks.
     *
     * @param slot The slot, the tensor's position in its side of the model config
     */
    RingBuffer& send_ring(size_t slot) const noexcept;
    RingBuffer& receive_ring(size_t slot) const noexcept;

    /**
     * @name The facts of the configuration the core reads
     *
     * Read from the handler's model config, what the validator derived for it and the Hard
     * contract (m_model, m_derived, m_hard), one source each: nothing is copied into the
     * session. A slot is a tensor's position in its side of the model config.
     * @{
     */
    /// The number of input and output tensors of the model config.
    size_t num_inputs() const noexcept;
    size_t num_outputs() const noexcept;
    /// A slot's hop: the samples per channel one inference consumes or produces on a
    /// Streamed slot (its window minus its context); 0 for a slot without a stream (Static,
    /// State), which has no ring.
    size_t input_hop(size_t slot) const noexcept;
    size_t output_hop(size_t slot) const noexcept;
    /// A slot's channel count: the Channel axis extent of a Streamed slot, 1 without one and
    /// for every other slot.
    size_t input_channels(size_t slot) const noexcept;
    size_t output_channels(size_t slot) const noexcept;
    /// The reference stream: the model config's anchor as the validator resolved it (the named
    /// Streamed tensor, else the first Streamed input, else the first Streamed output), the
    /// stream the contract's geometry counts in.
    bool reference_is_input() const noexcept;
    size_t reference_slot() const noexcept;
    /// Whether every chunk's completion is signalled on its semaphore: the contract's wait
    /// ratio is above 0, so the driving thread waits for results there; without a wait it
    /// polls the chunk's atomic flag.
    bool uses_semaphore() const noexcept;
    /// Whether the session runs one inference at a time and in order (capi::Derived::
    /// m_exclusive: a stateful model or a declared State pair): the dispatch gate.
    bool exclusive() const noexcept;
    /** @} */

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
         * @brief The chunk's storage and its descriptors, built once by the session's prepare
         *
         * One allocation holds the buffer of every slot of either side but a State slot, each
         * buffer starting on a 64-byte boundary, in the storage's dtype (float32 in this
         * pre-release) with the spec's extents, zeroed. Every slot gets one descriptor over its
         * buffer, with the spec's rank and extents and all-zero strides: what the engine call
         * hands its adapter (anira_engine_ctx::inputs / outputs) and what a processor context
         * copies for a slot. A State slot has no buffer: its descriptor is the processor
         * runner's to bind over the pair's port ahead of every inference.
         *
         * @param inputs What the validator derived per input tensor: the spec's extents
         * (capi::DerivedSpec::m_dims)
         * @param outputs The same per output tensor
         * @param input_state Per input tensor: whether it is a State slot
         * @param output_state Per output tensor: whether it is a State slot
         */
        ThreadSafeStruct(const std::vector<capi::DerivedSpec>& inputs,
                         const std::vector<capi::DerivedSpec>& outputs,
                         const std::vector<bool>& input_state,
                         const std::vector<bool>& output_state);
        ~ThreadSafeStruct() = default;
        // The descriptors name the block: one struct, one place.
        ThreadSafeStruct(const ThreadSafeStruct&) = delete;
        ThreadSafeStruct& operator=(const ThreadSafeStruct&) = delete;
        ThreadSafeStruct(ThreadSafeStruct&&) = delete;
        ThreadSafeStruct& operator=(ThreadSafeStruct&&) = delete;

        /// Zeroes the buffers of the chunk's inputs, State slots excepted (they have none).
        void zero_inputs() noexcept;
        /// Zeroes the buffers of the chunk's outputs: what a chunk that completes as zeros
        /// delivers. A State slot has no buffer; its pair keeps its own value.
        void zero_outputs() noexcept;

        /// The alignment of every slot's buffer, in bytes.
        static constexpr size_t k_alignment = 64;

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
        // Whether the chunk completed as zeros: dropped (SessionElement::complete_with_zeros:
        // a full queue, a failed pre_process, a stale or unclaimable task) or failed on the
        // inference thread (a throw, a failed before_inference or after_inference), its output
        // buffers zeroed. Core::pre_process resets it. The processor runner reads it in
        // post_process: such a chunk does not overwrite the handler's Static output store.
        // Plain field, synced by the queue handoffs like the stamps above (written before the
        // done signal, read after it).
        bool m_completed_as_zeros{false};
        // The struct's position in SessionElement::m_inference_queue: the chunk's entry
        // (anira_engine_ctx::entry, anira_processor_ctx::entry), set by prepare with the pool.
        uint32_t m_entry{0};
        // One descriptor per slot of either side over the chunk's storage (built by the
        // constructor; a State slot's bound by the processor runner per inference): the engine
        // call's anira_engine_ctx names these arrays, a processor context copies one for a
        // slot. Written by the thread that holds the struct, synced by the queue handoffs like
        // the fields above.
        std::vector<anira_tensor> m_input_tensors;
        std::vector<anira_tensor> m_output_tensors;
        // The chunk's links while its job waits for a free instance of its plan's loaded model
        // (an inference thread dequeued it while every instance was busy and put it on the
        // model's InstanceWaitList, PlanSlot::m_waiting): the session the job belongs to, kept
        // alive while it waits, and the next waiting struct. Written and read by the list alone,
        // under its lock; empty and null otherwise.
        std::shared_ptr<SessionElement> m_waiting_session;
        ThreadSafeStruct* m_waiting_next = nullptr;

    private:
        /// The one allocation of the chunk's buffers, over-allocated by k_alignment so that the
        /// first buffer starts on a boundary whatever the allocator handed out.
        std::vector<std::byte> m_block;
        /// Per slot of either side, its buffer in m_block; empty for a State slot.
        std::vector<std::span<std::byte>> m_input_bytes;
        std::vector<std::span<std::byte>> m_output_bytes;
    };

    std::vector<std::shared_ptr<ThreadSafeStruct>> m_inference_queue;  ///< Pool of thread-safe
                                                                       ///< structures for
                                                                       ///< concurrent processing

    /// No chunk carries this dispatch stamp: what a plan's m_engine_generation holds after
    /// prepare, so that the plan's first inference of the session is its first of a new stream.
    static constexpr uint64_t k_no_generation = UINT64_MAX;

    /**
     * @brief One plan of the table: the adapter the plan runs on and what a chunk stamped
     * with the plan reports.
     */
    struct PlanSlot {
        /// The loaded model the plan runs on (never null in a registered session): shared with
        /// the core's pool and with every other session on the same model, exclusive sessions
        /// included, or this session's own (a 2.x custom backend, the roundtrip).
        std::shared_ptr<engine::Loaded> m_loaded;
        /// This session's handle over the loaded model, what the inference thread runs: made
        /// by Core::create_session for a built-in engine's and a 2.x plan, by the C handler's
        /// prepare (with the shared prepare record) for a registered engine's plan, and given
        /// back by Core::release_session after the session's in-flight inferences drained,
        /// before the loaded model is released. Declared after m_loaded, so it dies first.
        std::unique_ptr<engine::Prepared> m_prepared;
        /// The jobs that wait for a free shared instance of m_loaded: the pool entry's list,
        /// shared like the loaded model with every session on it (Core::acquire_loaded_locked);
        /// null for a plan whose loaded model is this session's own (a 2.x custom backend, the
        /// roundtrip), which claims no instance. An inference thread that dequeues a job of the
        /// plan while every instance is busy puts it here instead of waiting with it.
        std::shared_ptr<InstanceWaitList> m_waiting;
        /// A registered engine's plan: its Prepared is the C handler's to make, with the
        /// record of its prepare; null until then.
        bool m_registered = false;

        // Defined in SessionElement.cpp, where the two engine-room types are complete: a
        // unique_ptr over a forward-declared type needs its deleter there, not here.
        PlanSlot();
        ~PlanSlot();
        PlanSlot(PlanSlot&&) noexcept;
        PlanSlot& operator=(PlanSlot&&) noexcept;
        PlanSlot(const PlanSlot&) = delete;
        PlanSlot& operator=(const PlanSlot&) = delete;
        /// The backend a processor context reports for the chunk: a built-in engine, or
        /// ANIRA_ENGINE_CUSTOM with m_engine_id for a custom one, on a provider of the enum, or
        /// ANIRA_PROVIDER_CUSTOM with m_provider_id for a custom one.
        anira_engine m_engine = ANIRA_ENGINE_NONE;
        anira_provider m_provider = ANIRA_PROVIDER_CPU;
        std::string m_engine_id;
        std::string m_provider_id;
        /// The plan's engine keeps its own aliasing of a declared State pair
        /// (ANIRA_ENGINE_FLAG_STATE_ALIAS in its flags): the processor runner binds one stable
        /// buffer as both halves of every pair for a chunk of this plan and never flips it.
        bool m_state_alias = false;
        /**
         * @brief The dispatch stamp this plan's engine last ran under: the plan's reset boundary
         *
         * A session-exclusive session only, whose inferences run one at a time and in order
         * (the dispatch gate): InferenceThread::inference compares the chunk's dispatch stamp
         * with it, and a difference is this plan's first inference of a new stream (after
         * prepare, after a reset), so the plan's adapter resets its engine's own state right
         * before that inference's process and the stamp is adopted. Per plan, since every plan
         * runs its own executor: after a reset each plan is reset at its own first inference
         * of the new stream, whichever plan ran first, and a plan switch alone is no boundary.
         * Read and written on the inference thread alone, between the gate's acquire and
         * release; set to k_no_generation by prepare, at quiescence. A shared (pooled) adapter
         * is never reset: only an exclusive session gets here.
         */
        uint64_t m_engine_generation = k_no_generation;
    };

    /**
     * @brief The plan table: dense plan index -> the plan.
     *
     * Built by Core::create_session from the requests the C handler gives it
     * (engine::PlanRequest): exactly its plans. Never empty in a registered session, and never
     * changed afterwards, so the driving thread and the inference threads read it without
     * synchronization.
     */
    std::vector<PlanSlot> m_plans;

    /**
     * @brief The selected plan: the one the next submitted chunk is stamped with.
     *
     * The whole runtime selection is this one atomic (select_plan stores it, a chunk takes
     * it with one load in Core::pre_process), so concurrent callers have no pair to tear.
     * Always below m_plans.size(). 0 when Core::create_session returns; the C handler
     * selects its initial plan before the first prepare.
     */
    std::atomic<uint32_t> m_current_plan{0};

    /// Selects a plan: one relaxed store. False, and nothing stored, for an index out of range.
    bool select_plan(uint32_t plan) noexcept;

    unsigned long m_current_queue = 0;         ///< Current position in the inference queue
    std::vector<unsigned long> m_time_stamps;  ///< Vector of timestamps for performance monitoring
    size_t m_pending_pull_samples = 0;  ///< Generator sessions only (!m_input_driven): samples of
                                        ///< the reference output the driving thread has demanded
                                        ///< that no submitted inference covers yet. Plain field,
                                        ///< written and read only on the session's driving thread
                                        ///< (like m_time_stamps).

    const int m_session_id;  ///< Unique identifier for this session (immutable)

    std::atomic<bool> m_initialized{false};  ///< Atomic flag indicating if the session is fully
                                             ///< initialized
    /// The session's jobs outside the free pool, counted from before their enqueue into the
    /// global queue (the audio thread's dispatch, an exclusive chain's continuation) until the
    /// worker finished the job or a drain completed it as zeros: what
    /// Core::drain_inference_queue waits down to zero, so that no job is invisible to it,
    /// however late its worker is. On a heap object of its own, which a worker holds beside the
    /// session: its decrement is the last thing the worker does for the job, after it dropped
    /// its reference to the session, so a session dies on the control thread that released it,
    /// never on an inference thread.
    const std::shared_ptr<std::atomic<int>> m_active_inferences =
        std::make_shared<std::atomic<int>>(0);

    // Monotonic generation counter, bumped by Core::reset_session() (audio
    // thread) and Core::prepare_session() (control thread, quiescent). Every
    // inference dispatched under an earlier generation is "stale": its output is
    // discarded and its struct reclaimed, without the caller ever waiting for the
    // worker. Workers read it to decide whether to skip a stale dispatch. seq_cst
    // on the write pairs with the worker's seq_cst read of it, as m_initialized's
    // does. 64-bit: wrap-around (ABA) is unreachable even at per-block reset rates.
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
    // Pre-sized in the constructor to m_num_instances — a pending
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

    capi::ProcessorRunner& m_processor;  ///< The handler's processor runner: the four phases of
                                         ///< every chunk, and the port vectors of the rings
    const anira_model_config& m_model;   ///< The handler's model config (its variant)
    const capi::Derived& m_derived;      ///< What the validator derived for it under the contract
    const capi::HardContract& m_hard;    ///< The Hard half of the contract of the prepare
    /// The instance count (Core::create_session: 1 for an exclusive session, else the model
    /// config's, ANIRA_INSTANCES_AUTO resolved and clamped to the pool): at most this many
    /// inferences of the session run at once (LatencyCalculator's n), and the pending dispatch
    /// queue of an exclusive session is sized by it.
    const uint32_t m_num_instances;

    RtLatch m_rt_own;  ///< The session's own real-time latch: what m_rt points at when the
                       ///< constructor received no latch (a 2.x session); nothing reads its
                       ///< word, but a failed inference records ENGINE into it all the same
    RtLatch* m_rt;     ///< The real-time latch the session's failures are recorded into (a
                       ///< failed inference records ANIRA_ERROR_ENGINE on the inference
                       ///< thread): the 3.x handler's own (anira_handler_rt_error reads it),
                       ///< else &m_rt_own. Set by the constructor and published by the
                       ///< lifecycle lock that registers the session; never null
    // What every ring of this session points at (anira_ring::owner), set by prepare(): m_rt,
    // and the name of the processor a 3.x handler's chain is running on the driving thread.
    RingOwner m_ring_owner;

    // Written by InferenceManager::set_non_realtime() -- typically from a control/UI
    // thread, e.g. a host toggling offline bounce/render mode -- and read on the
    // audio thread inside Core::new_data_request(). A plain bool would be a
    // data race between those two threads, so this needs real synchronization.
    std::atomic<bool> m_is_non_real_time{false};  ///< True forces new_data_request() to
                                                  ///< block until each pending inference
                                                  ///< completes, ignoring the wait ratio
                                                  ///< and any deadline, trading real-time
                                                  ///< safety for complete, deterministic
                                                  ///< output (see Core::new_data_request).

    std::vector<unsigned int> m_latency;  ///< Calculated latency values for each output tensor in
                                          ///< samples, index-aligned with the output tensor list;
                                          ///< 0 for non-streamable outputs
    LatencyCalculator::Rational m_block_hops;  ///< The host block in hops, rho = block_max / R
                                               ///< (LatencyCalculator::get_block_hops), set by
                                               ///< prepare(): what a host block may carry on a
                                               ///< Streamed slot (LatencyCalculator::
                                               ///< block_samples)
    bool m_input_driven = true;  ///< True if any input tensor is streamable: inference is
                                 ///< triggered by arriving input samples. False for a generator
                                 ///< (no streamable input), whose inference is triggered by
                                 ///< output demand (see m_pending_pull_samples).
    size_t m_num_structs = 0;    ///< Number of allocated thread-safe structures: twice the
                                 ///< steady-state bound of LatencyCalculator::get_num_structs(), so
                                 ///< that the inferences a wait-free reset() strands until their
                                 ///< workers finish never starve the fresh schedule (for testing
                                 ///< access)
    std::vector<size_t> m_send_buffer_size;     ///< Calculated send ring sizes (for testing access)
    std::vector<size_t> m_receive_buffer_size;  ///< Calculated receive ring sizes (for testing
                                                ///< access)

private:
    /**
     * @brief Whether any input tensor is streamable
     *
     * @return True if at least one input has a non-zero preprocess_input_size
     */
    bool has_streamable_input() const;

    /// The model's own latency of an output, in samples of its stream (the spec's latency).
    size_t output_model_latency(size_t slot) const noexcept;

#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    ThreadSafeStruct* __doxygen_force_0;  ///< Placeholder for Doxygen documentation
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
 * polling (see anira_wait_strategy). The blocking queue is a strict superset of the
 * plain ConcurrentQueue API, and as long as no consumer ever blocks, its
 * enqueue never makes a syscall — so context configs on
 * ANIRA_WAIT_SPIN_BACKOFF keep the exact lock-free behavior of the plain
 * queue, at the cost of one extra atomic operation per enqueue/dequeue.
 *
 * On WebAssembly builds the plain ConcurrentQueue is used: the pool's threads
 * are Wasm Workers on the module's shared memory, and an idle Worker parks on
 * the core's wake word (memory.atomic.wait32, notified by every enqueue)
 * instead of a semaphore, under either wait strategy.
 */
#ifdef __EMSCRIPTEN__
using InferenceQueue = moodycamel::ConcurrentQueue<InferenceData>;
#else
using InferenceQueue = moodycamel::BlockingConcurrentQueue<InferenceData>;
#endif

}  // namespace anira

#endif  // ANIRA_SESSIONELEMENT_H