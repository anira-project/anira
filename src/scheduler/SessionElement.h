#ifndef ANIRA_SESSIONELEMENT_H
#define ANIRA_SESSIONELEMENT_H

#include <concurrentqueue.h>
#ifndef __EMSCRIPTEN__
#include <blockingconcurrentqueue.h>
#endif

#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <tanh/core/threading/RCU.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "../utils/RcuReaders.h"
#include "../utils/RealtimeSanitizer.h"
#include "../utils/RingBuffer.h"
#include "../utils/RtLatch.h"
#include "../utils/Semaphore.h"
#include "ChunkStorage.h"
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
 * exclusivity) and the contract's Hard half (capi::HardContract: the host blocks, the budget, the
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

/**
 * @brief What a session is sized by over the variants of its handler's inference stage.
 *
 * The variants of a stage have one form (capi::check_same_form), which the session reads from
 * the handler's derivation, and differ in what the stream cannot see. Two of those figures size
 * the session, so the handler resolves them over every variant and hands them over as values:
 * the session holds no model config.
 */
struct VariantBounds {
    /// Per output tensor: the model's own latency in samples of its stream (the spec's
    /// latency), the largest any variant declares. The reported latency carries it, and the
    /// receive ring is primed without it, so a variant with less plays that many samples
    /// early. One entry per tensor of the output list; 0 for an output without a stream.
    std::vector<size_t> m_internal_latency;
    /// The shared instances of a stateless model: the fewest any variant allows (its
    /// max_instances), ANIRA_INSTANCES_AUTO when every variant leaves the count to the pool.
    /// What Core::create_session resolves the session's instance count from; each plan's
    /// loaded model keeps its own variant's count.
    uint32_t m_max_instances = ANIRA_INSTANCES_AUTO;
};

/**
 * @brief What fills the hops a session's submission drops: the InferenceManager, under the
 * contract's miss policy
 *
 * drop() runs once per dropped hop, at the drop, while every Streamed send ring holds the hop at
 * its read position: it consumes the hop and keeps its fill. place() runs once per dropped hop at
 * its ordered placement (SessionElement::place_pending_fills), in the order of the drops: it
 * pushes the oldest fill not yet placed into every Streamed receive ring, each of which has room
 * for a hop. Driving thread only; real-time safe.
 */
class DroppedHops {
public:
    DroppedHops() = default;
    virtual ~DroppedHops() = default;
    DroppedHops(const DroppedHops&) = delete;
    DroppedHops& operator=(const DroppedHops&) = delete;
    DroppedHops(DroppedHops&&) = delete;
    DroppedHops& operator=(DroppedHops&&) = delete;

    virtual void drop() ANIRA_NONBLOCKING = 0;
    virtual void place() ANIRA_NONBLOCKING = 0;
};

class SessionElement {
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
     * @param internal_latency The model's own latency of every output, the largest over the
     * handler's variants (VariantBounds::m_internal_latency), copied
     * @param derived The form of the stream: what the validator derived for the handler's
     * variants under the contract, the same for every one (it outlives the session)
     * @param hard The Hard half of the contract the handler is prepared under (it outlives the
     * session)
     * @param num_instances The instance count, resolved by Core::create_session (1 for an
     * exclusive session, else the fewest max_instances of the handler's variants,
     * ANIRA_INSTANCES_AUTO the pool's size, clamped to the pool)
     * @param producer_token Producer token bound to the global inference queue, moved into the
     * session (see m_producer_token)
     * @param rt_latch The real-time latch this session records its failures into (see m_rt):
     * a 3.x handler's own, so anira_handler_rt_error reads one word; nullptr (a 2.x session)
     * gives the session its own latch (m_rt_own)
     * @param selection The atomic that holds the session's selected plan (see m_current_plan):
     * a 3.x handler's own, so anira_handler_get_plan reads one word of the handler; nullptr
     * (a 2.x session) gives the session its own (m_current_plan_own). Set to 0 here.
     */
    SessionElement(int new_session_id,
                   capi::ProcessorRunner& processor,
                   std::vector<size_t> internal_latency,
                   const capi::Derived& derived,
                   const capi::HardContract& hard,
                   uint32_t num_instances,
                   moodycamel::ProducerToken&& producer_token,
                   anira::RtLatch* rt_latch = nullptr,
                   std::atomic<uint32_t>* selection = nullptr);

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
     * builds the chunks, for the contract's host blocks, budget and wait ratio. This method must
     * be called before the session can process audio data. The host block in hops is stored
     * in m_block_hops and whether an input streams in m_input_driven, for the real-time path.
     *
     * @param custom_latencies A caller's latency per output tensor (see CustomLatencies; empty
     * keeps every computed latency); entries for non-streamable outputs are ignored, their
     * latency is always 0
     * @param ring_dtypes The element type of every send and receive ring, per slot (see
     * RingDtypes; float32 for every slot the vectors do not name); the ring sizes are element
     * counts, so they do not depend on it
     * @throws std::invalid_argument if the contract's host blocks cannot size the session (see
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

    /// Drops one hop: consumes it from every Streamed send ring and keeps its fill
    /// (m_dropped_hops; without one, the fill is silence). Driving thread only.
    void drop_hop();

    /**
     * @brief Owes the receive rings one hop of fill per dropped hop, in stream order
     *
     * A hop the submission dropped (no slot was free: every slot holds a chunk not yet
     * collected) produces one hop of fill on every streamable output at its own stream
     * position, behind every chunk claimed before it: counted on the youngest chunk not yet
     * collected (ThreadSafeStruct::m_fills_after), which Core::post_process moves to
     * m_pending_fills behind its own output. With no such chunk (after a wait-free reset the
     * slots in flight are stale), the fill is due at once. Driving thread only.
     *
     * @param hops The hops dropped
     */
    void owe_fills(size_t hops);

    /**
     * @brief Places the fills owed ahead of every chunk not yet collected
     *
     * One hop of fill on every streamable output per owed hop (DroppedHops::place, or zeros),
     * while every receive ring has room for one; the collection calls it before it places the
     * next chunk. Driving thread only.
     *
     * @return True when nothing is owed any more, false when a ring is full
     */
    bool place_pending_fills();

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
     * Read from what the validator derived for the handler's variants (the form they share)
     * and the Hard contract (m_derived, m_hard), one source each, by reference. A slot is a
     * tensor's position in its side of the model config.
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
    /// stream the contract's host blocks count in.
    bool reference_is_input() const noexcept;
    size_t reference_slot() const noexcept;
    /// Whether every chunk's completion is signalled on its semaphore: the contract's wait
    /// ratio is above 0, so the driving thread waits for results there; without a wait it
    /// polls the chunk's atomic flag.
    bool uses_semaphore() const noexcept;
    /// Whether the session runs one inference at a time and in order (capi::Derived::
    /// m_exclusive: a stateful model or a declared State pair): the dispatch gate.
    bool exclusive() const noexcept;
    /// The most jobs of the session the inference queue can hold at once, what prepare reserves
    /// for it there (Core::reserve_inference_capacity): one for an exclusive session (its
    /// dispatch gate lets one job be queued or running), else the struct pool (a queued job
    /// holds a struct that is not free until its chunk is collected). Valid after prepare.
    size_t queue_capacity() const noexcept;
    /** @} */

    /// One plan of the table (defined below): a chunk points at the slot of its plan.
    struct PlanSlot;

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
         * One block of `storage` holds the buffer of every slot of either side but a State
         * slot, each buffer starting on an `alignment` boundary, in the spec's dtype with the
         * spec's extents, zeroed. Every slot gets one descriptor over its buffer, with the
         * spec's rank, extents and dtype, the storage's domain and all-zero strides: what the
         * engine call
         * hands its adapter (anira_engine_ctx::inputs / outputs) and what a processor context
         * copies for a slot. A State slot has no buffer: its descriptor is the processor
         * runner's to bind over the pair's port ahead of every inference.
         *
         * @param inputs What the validator derived per input tensor: the spec's extents and
         * dtype (capi::DerivedSpec::m_dims, m_dtype)
         * @param outputs The same per output tensor
         * @param input_state Per input tensor: whether it is a State slot
         * @param output_state Per output tensor: whether it is a State slot
         * @param storage Where the block lives (host_chunk_storage() in this pre-release)
         * @param alignment The boundary every slot's buffer starts on, in bytes, a power of two
         * (the largest any of the session's plans' engines needs, buffer_alignment)
         */
        ThreadSafeStruct(const std::vector<capi::DerivedSpec>& inputs,
                         const std::vector<capi::DerivedSpec>& outputs,
                         const std::vector<bool>& input_state,
                         const std::vector<bool>& output_state,
                         ChunkStorage& storage,
                         size_t alignment);
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

        /// The boundary every slot's buffer of this chunk starts on, in bytes.
        size_t alignment() const noexcept { return m_block.alignment(); }

        /// Takes the chunk's plan slot off the chunk (m_slot is NULL afterwards): the caller
        /// holds the chunk, uses the slot for the rest of the chunk's worker side and gives the
        /// chunk's count back (PlanSlot::release_chunk) when it is done with the slot. NULL
        /// when the slot was taken before, or never stamped.
        PlanSlot* take_slot() noexcept;
        /// Ends the chunk's count on its plan slot, for a chunk that no inference thread gets
        /// (dropped before or at an enqueue, discarded from a queue, completed by a drain): the
        /// slot is taken off the chunk and the count given back at once. Nothing for a chunk
        /// whose slot was taken before. The caller holds the chunk.
        void leave_slot() noexcept;

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
        // The state epoch this chunk was formed under (stamped in Core::pre_process beside
        // the dispatch generation, from SessionElement::m_state_epoch): the driving thread
        // opens a new one when it forms the first chunk on another variant than the chunk
        // before it under ANIRA_VARIANT_STATE_FRESH. The declared state's binding and the
        // plan's engine reset compare it as they compare the generation: a chunk of a new
        // epoch starts its plan's state over ahead of its inference. Unlike the generation
        // it makes no chunk stale: what is in flight across a variant switch is delivered.
        // Plain field, synced by the queue handoffs like the stamps above.
        uint64_t m_state_epoch{0};
        // The plan this chunk runs on: the dense index into SessionElement::m_plans,
        // stamped in Core::pre_process from SessionElement::m_current_plan with one load
        // (SessionElement::stamp_plan, which stamps the plan's slot and its copies below with
        // it). The hooks, the engine call and post_process all use what was stamped and never
        // read the session's atomic or its table again, so a plan switch that lands while the
        // chunk is queued or in flight cannot split it across two plans (or run two engines, or
        // none, for it). The index and not the backend: two plans may run on one backend (two
        // variants of a model, two providers of an engine), and only the index tells them apart.
        // Plain field: written on the driving thread at dispatch, stable for the dispatch's
        // lifetime, synced by the queue handoffs like the two stamps above.
        uint32_t m_plan{0};
        // The slot of that plan, for the worker side alone: the chunk counts on it from the
        // stamp (PlanSlot::active_chunks) until whoever finishes the chunk's worker side took
        // the pointer off the chunk (take_slot, leave_slot) and gave the count back. NULL from
        // then on, and for a chunk never stamped: the driving thread's steps behind the
        // inference (post_process, the collection, the miss path) have the copies below and
        // never a slot, which nothing keeps alive for them. Plain field: written by the thread
        // that holds the chunk (the stamp; the taker, ahead of the chunk's done signal or of
        // its return to the free pool), synced by the queue handoffs like the stamps above.
        PlanSlot* m_slot = nullptr;
        // What the chunk's plan reports, copied off its slot at the stamp: the variant and the
        // backend every phase's processor context names (PlanSlot::m_variant, m_engine,
        // m_provider and the two ids, the session's own strings, see SessionElement::intern_id)
        // and whether the plan's engine keeps its own aliasing of a declared State pair
        // (PlanSlot::m_state_alias). The processor runner reads these and nothing of the slot,
        // so the chunk's phases on the driving thread need no slot; the state epoch of the
        // stream compares the variant too (SessionElement::stamp_state_epoch). Variant 0, no
        // engine and no provider for an index the table did not have. Plain fields, written
        // with m_plan.
        uint32_t m_variant = 0;
        anira_engine m_engine = ANIRA_ENGINE_NONE;
        anira_provider m_provider = ANIRA_PROVIDER_NONE;
        const char* m_engine_id = nullptr;
        const char* m_provider_id = nullptr;
        bool m_state_alias = false;
        // Whether the chunk completed as zeros: dropped (SessionElement::complete_with_zeros:
        // a full queue, a failed pre_process, a stale or unclaimable task) or failed on the
        // inference thread (a throw, a failed before_inference or after_inference), its output
        // buffers zeroed. Core::pre_process resets it. The processor runner reads it in
        // post_process: such a chunk does not overwrite the handler's Static output store.
        // Plain field, synced by the queue handoffs like the stamps above (written before the
        // done signal, read after it).
        bool m_completed_as_zeros{false};
        // The hops dropped while this chunk was the youngest one not yet collected: no slot
        // was free for them (SessionElement::owe_fills). Their fill follows this chunk's
        // output into the receive rings, ahead of every later chunk. Driving thread only (set
        // by the submission, moved out by Core::post_process); Core::pre_process resets it.
        size_t m_fills_after{0};
        // How long the inference thread spent on this chunk's job, in nanoseconds of
        // steady_clock: from its before_inference to its after_inference (the declared state's
        // feed and capture, the processor's hooks and the engine call; the wait for an instance
        // lies before it). Written by the inference thread before the done signal, read after it
        // (the warm-up pass of a prepare reads it); 0 until the chunk's first job ran.
        int64_t m_inference_ns{0};
        // The struct's position in SessionElement::m_struct_pool: the chunk's entry
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
        /// The one block of the chunk's buffers, from its storage, on the chunk's alignment.
        ChunkBlock m_block;
        /// Per slot of either side, its buffer in m_block; empty for a State slot.
        std::vector<std::span<std::byte>> m_input_bytes;
        std::vector<std::span<std::byte>> m_output_bytes;
    };

    /// The session's chunks, built by prepare: m_num_structs of them (twice the steady-state
    /// bound, see prepare), each a chunk's storage and descriptors, free or held by one chunk
    /// from its submission to its collection. An index into it is the chunk's entry
    /// (ThreadSafeStruct::m_entry). What the inference queue carries is a pointer to one of
    /// them, so a session never has more jobs in the queue than this pool has structs.
    std::vector<std::shared_ptr<ThreadSafeStruct>> m_struct_pool;

    /// No chunk carries this dispatch stamp: what a plan's m_engine_generation holds after
    /// prepare, so that the plan's first inference of the session is its first of a new stream.
    static constexpr uint64_t k_no_generation = UINT64_MAX;

    /**
     * @brief One plan of the table: the adapter the plan runs on and what a chunk stamped
     * with the plan reports.
     *
     * An object of its own, at one address for its life: the table holds it by pointer, and so
     * does every chunk stamped with the plan (ThreadSafeStruct::m_slot) until its worker side is
     * finished. Neither copied nor moved.
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
        PlanSlot(const PlanSlot&) = delete;
        PlanSlot& operator=(const PlanSlot&) = delete;
        PlanSlot(PlanSlot&&) = delete;
        PlanSlot& operator=(PlanSlot&&) = delete;
        /// The variant the plan runs: its index in the inference stage's list
        /// (anira_plan_info.variant), what a processor context reports for the chunk
        /// (anira_processor_ctx.variant; a chunk copies it at its stamp).
        uint32_t m_variant = 0;
        /// The backend a processor context reports for the chunk: a built-in engine, or
        /// ANIRA_ENGINE_CUSTOM with m_engine_id for a custom one, on a provider of the enum, or
        /// ANIRA_PROVIDER_CUSTOM with m_provider_id for a custom one. The two ids are the
        /// session's own strings (SessionElement::intern_id), NULL for a built-in engine and a
        /// provider of the enum: a chunk copies the pointers at its stamp, and they stay valid
        /// for the session's life whatever happens to the slot.
        anira_engine m_engine = ANIRA_ENGINE_NONE;
        anira_provider m_provider = ANIRA_PROVIDER_CPU;
        const char* m_engine_id = nullptr;
        const char* m_provider_id = nullptr;
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
        /// The state epoch this plan's engine last ran under, beside m_engine_generation and
        /// compared with it (ThreadSafeStruct::m_state_epoch): the plan's first inference of a
        /// new state epoch, behind a switch to another variant under
        /// ANIRA_VARIANT_STATE_FRESH, resets its engine too. Read and written like the
        /// generation; 0 after prepare.
        uint64_t m_engine_state_epoch = 0;

        /**
         * @brief Who still needs the slot: the chunks stamped with the plan whose worker side
         * is not finished
         *
         * A chunk counts from its stamp (SessionElement::stamp_plan) until nothing on the
         * worker side touches it or the slot any more: an inference thread finished its job
         * (run, failed, skipped as stale or given up; after the thread took the next waiting job
         * of the slot's model or gave its instance back), or the chunk was dropped, discarded
         * or completed as zeros without one (ThreadSafeStruct::leave_slot). While it waits in an
         * exclusive session's pending queue, in the inference queue or on the model's wait
         * list, it counts. An inference thread that put a chunk on the wait list counts once
         * more until it is done with the slot's model and list. The count does not wait for the
         * driving thread: a chunk whose inference finished and that nobody collected does not
         * count. Zero therefore means that no inference thread reads the slot and none will
         * unless a new chunk is stamped with it.
         */
        uint32_t active_chunks() const noexcept {
            return m_active_chunks.load(std::memory_order::seq_cst);
        }
        /// Counts one in: the stamp, and an inference thread's own hold around a wait-list
        /// push. Sequentially consistent, like the session's count of its jobs.
        void retain_chunk() noexcept { m_active_chunks.fetch_add(1, std::memory_order::seq_cst); }
        /// Counts one out: the caller is done with the slot and does not touch it again.
        void release_chunk() noexcept { m_active_chunks.fetch_sub(1, std::memory_order::release); }

    private:
        static_assert(std::atomic<uint32_t>::is_always_lock_free,
                      "the stamp of a chunk on the driving thread requires a lock-free count");
        std::atomic<uint32_t> m_active_chunks{0};
    };

    /**
     * @brief The plans of the session, the control side: dense plan index -> the plan's slot.
     *
     * The list that OWNS the slots. Built by Core::create_session from the requests the C
     * handler gives it (engine::PlanRequest): exactly its plans, each an object of its own.
     * An index is a plan's for the session's life: a plan that is taken out leaves its entry
     * in place, empty (Core::take_plan_slot), and a plan that is added gets the next index
     * (Core::add_plan_slot). Never empty in a registered session.
     *
     * Read and changed on control threads only, never on a real-time path: by the control
     * thread of the session's handler (prepare, the warm-up's stamp, the report), and by the
     * core under its lifecycle lock, where the release of ANOTHER session walks the lists of
     * all registered sessions (which loaded model is still held). So the list changes under
     * the lifecycle lock alone (Core::create_session, add_plan_slot, take_plan_slot), and the
     * handler's control thread, the only one that changes it, reads it without the lock. The
     * driving thread never reads it: it stamps from the published table (m_plan_table).
     */
    std::vector<std::shared_ptr<PlanSlot>> m_plans;

    /**
     * @brief The plan table as the driving thread reads it: one published version
     *
     * Dense plan index -> the plan's slot, NULL at an index whose plan was taken out. The
     * pointers do not own: the slots are m_plans' (or, for a slot just taken out, its
     * remover's). A version is immutable once published, and it is destroyed later than it is
     * retired (by a later publication), so its destructor must not, and does not, look at a
     * slot.
     */
    struct PlanTable {
        std::vector<PlanSlot*> m_slots;
    };

    /// A read lease on the published plan table: the version of the claim, for the lease's
    /// life (RcuReaders::Lease).
    using PlanLease = RcuReaders<PlanTable>::Lease;

    /**
     * @brief Claims a read lease on the published plan table
     *
     * What every entry of the driving thread holds for the whole call (the six Hard entries,
     * their _wait twins, the ring state, reset) and stamps its chunks from: the first free
     * reader slot by compare-and-swap, the others in turn, so the driving thread need not be
     * one thread, and two calls that overlap each read through a slot of their own. No
     * allocation, no lock.
     *
     * @return The lease, or nothing when every slot is busy (ANIRA_ERROR_CAPACITY; with one
     * driving thread it does not happen)
     */
    std::optional<PlanLease> claim_plans() const noexcept ANIRA_NONBLOCKING {
        return m_plan_readers.claim();
    }

    /**
     * @brief Publishes m_plans as the table the driving thread stamps from
     *
     * One raw pointer per entry of the list, NULL for an empty one. Control thread (the
     * handler's, which alone changes the list); allocates and takes the table's writer lock.
     * A lease claimed from now on sees this version; one that is out keeps its own. When it
     * throws (std::bad_alloc) nothing was published. stage_plans() and commit_plans() in one
     * call.
     */
    void publish_plans();

    /// A plan table that is built and not yet published: what stage_plans() makes and
    /// commit_plans() publishes (thl::RCU::Staged). Dropped without a commit, it frees its
    /// table and the session is as it was.
    using StagedPlanTable = thl::RCU<PlanTable>::Staged;

    /**
     * @brief The first half of publish_plans(), for a list the caller is about to change
     *
     * Builds the table m_plans will be published as once the caller has appended `added` to
     * the list, in this order (Core::add_plan_slot), and emptied the entries `removed`
     * (Core::take_plan_slot), and makes the room the publication needs. Everything of the
     * publication that can fail happens here: when it throws (std::bad_alloc) nothing was
     * published and nothing changed. For a control entry that changes the list and what it
     * publishes beside the playing stream, and must have everything that can fail behind it
     * before the first change is visible (anira_handler_add_variant,
     * anira_handler_remove_variant).
     *
     * Control thread (the handler's). Between this call and commit_plans() of its result
     * nothing else publishes the table (thl::RCU::commit's rule: no publish_plans(), no
     * second commit); the handler's control lock keeps that.
     *
     * @param added The slots the caller appends before it commits, in the order it appends them
     * @param removed The indices of the list whose entries the caller empties before it commits
     * @return The table, to be given to commit_plans()
     */
    StagedPlanTable stage_plans(std::span<const std::shared_ptr<PlanSlot>> added = {},
                                std::span<const uint32_t> removed = {});

    /**
     * @brief The second half of publish_plans(): publishes a table stage_plans() built
     *
     * The list has been changed as stage_plans() was told (asserted): the table is m_plans,
     * entry for entry. Allocates nothing and cannot fail. Control thread; takes the table's
     * writer lock. A lease claimed from now on sees this version.
     */
    void commit_plans(StagedPlanTable&& table) noexcept;

    /**
     * @brief Waits until no read lease claimed before the last publish_plans() is still out
     *
     * When it returns, every entry of the driving thread that could stamp a chunk from an
     * older version has returned: a slot the newest version no longer names is never stamped
     * again. It frees nothing. The slot is its remover's to free, once nothing counts on it
     * any more (PlanSlot::active_chunks, Core::wait_until_unused). Control thread; blocks for
     * as long as such an entry runs (a _wait twin for its timeout).
     */
    void synchronize_plans() ANIRA_BLOCKING;

    /**
     * @brief The selected plan: the one the next submitted chunk is stamped with.
     *
     * The whole runtime selection is this one atomic (select_plan stores it, a chunk takes
     * it with one load in Core::pre_process), so concurrent callers have no pair to tear.
     * 0 when Core::create_session returns; the C handler selects its initial plan before the
     * first prepare. The atomic itself is the C handler's (anira_handler::m_selected_plan,
     * which outlives the session as its latch does), so the handler's getter reads the
     * selection without reaching into the session; a 2.x session refers to its own,
     * m_current_plan_own. A value may name an index the table a chunk is stamped from does not
     * have, or one whose plan was taken out: stamp_selected_plan says what the chunk runs then.
     */
    std::atomic<uint32_t> m_current_plan_own{0};
    std::atomic<uint32_t>& m_current_plan;

    /// Selects a plan: one relaxed store. False, and nothing stored, for an index the
    /// published table does not have (its count is an atomic the publication stores: the
    /// check reads no container, from whatever thread it runs).
    bool select_plan(uint32_t plan) noexcept;

    /// Selects `plan` if `expected` is still the selected plan: one compare-and-swap of the
    /// selection. False, nothing stored and `expected` the plan selected now, when another
    /// caller moved it meanwhile (or spuriously: the caller loops), so a selection computed
    /// from the selected plan (one axis of it kept, the other moved) is never stored over a
    /// newer one. `plan` is in range (the caller resolved it in the table).
    bool select_plan_if(uint32_t& expected, uint32_t plan) noexcept ANIRA_NONBLOCKING;

    /// No chunk has been formed since prepare: what m_last_variant holds then, so that the
    /// stream's first chunk is no variant switch.
    static constexpr uint32_t k_no_variant = UINT32_MAX;

    /**
     * @brief The state epoch of the stream: the variant switches that start the state over.
     *
     * Core::pre_process stamps every chunk with it (ThreadSafeStruct::m_state_epoch) and
     * opens a new one (stamp_state_epoch) when the chunk's plan runs another variant than the
     * chunk formed before it, under ANIRA_VARIANT_STATE_FRESH (the Hard contract's
     * m_variant_state); under ANIRA_VARIANT_STATE_KEEP, and for a plan switch inside one
     * variant, it stays. The state binding of the processor runner and the engine reset of
     * the inference thread compare the chunk's stamp with the one they last ran under.
     * Driving thread only, like m_last_variant; both start over at prepare.
     */
    uint64_t m_state_epoch = 0;
    uint32_t m_last_variant = k_no_variant;  ///< The variant of the chunk formed last

    /// The state epoch of a chunk formed on a plan of `variant` (the stamped chunk's copy,
    /// ThreadSafeStruct::m_variant), which becomes the chunk formed last: the current epoch, or
    /// a new one when the variant differs from the last chunk's under
    /// ANIRA_VARIANT_STATE_FRESH. Driving thread only; no allocation, no lock.
    uint64_t stamp_state_epoch(uint32_t variant) noexcept ANIRA_NONBLOCKING;

    /**
     * @brief Stamps a chunk of the stream with the selected plan, out of a published table
     *
     * Where a chunk of the stream gets its plan (Core::pre_process, on the driving thread),
     * and the one read of the plan table on a real-time path: `table` is the version of the
     * read lease the running entry holds, one for the whole call, so every chunk the call
     * forms is stamped from the same table. The chunk gets the index, the plan's slot, on which
     * it counts from here (PlanSlot::active_chunks), and the slot's variant, backend and
     * aliasing as copies (ThreadSafeStruct::m_variant and the fields beside it).
     *
     * The selection is loaded once. When `table` has no plan under it, the chunk runs the plan
     * of the chunk formed before it (m_applied_plan; the first plan the table has when that one
     * is gone too):
     * - an index beyond the table (a plan selected in a table published after the lease was
     *   claimed) changes nothing else: the selection stands, and the next call stamps it;
     * - an index whose plan was taken out moves the selection back to the plan the chunk runs
     *   (a compare-and-swap, so a selection made meanwhile stands) and records
     *   ANIRA_ERROR_NOT_FOUND in the session's latch, logged once.
     * A table without any plan, which a registered session never has, leaves the chunk
     * without a slot: its inference fails (ANIRA_ERROR_INTERNAL).
     *
     * Wait-free, no allocation, no lock. The caller holds the chunk (it took it from the free
     * pool).
     */
    void stamp_selected_plan(ThreadSafeStruct& chunk,
                             const PlanTable& table) noexcept ANIRA_NONBLOCKING;

    /**
     * @brief Stamps a chunk with one plan of the session's own list (m_plans)
     *
     * The stamp of a chunk that is no part of the stream: a warm-up inference of plan `plan`,
     * on the control thread of the session's handler (Core::warm_up_inference), which reads
     * the list and not the published table. The selection and m_applied_plan are not touched.
     * An index the list does not have, or an empty entry, leaves the chunk without a slot.
     */
    void stamp_plan(ThreadSafeStruct& chunk, uint32_t plan) noexcept;

    /// The plan of the chunk of the stream formed last: what a chunk runs when the published
    /// table has no plan under the selection (stamp_selected_plan). The selected plan at
    /// prepare. Driving thread only, like the state epoch above.
    uint32_t m_applied_plan = 0;

    /// The boundary every chunk's buffers start on, in bytes: what prepare built the struct
    /// pool with, the largest any plan of m_plans asked for then
    /// (engine::Loaded::buffer_alignment). A plan added behind the prepare whose engine asks
    /// for more cannot bind the chunks in place. 0 before the first prepare.
    size_t m_chunk_alignment = 0;

    /**
     * @brief The session's own copy of an id string a plan names, for the session's life
     *
     * A custom engine's id or a custom provider's name of a plan request, kept once per
     * distinct string in a store the session owns and never shrinks: the pointer a plan slot
     * holds and a chunk copies (PlanSlot::m_engine_id, ThreadSafeStruct::m_engine_id), valid
     * until the session dies whatever happens to the slot. Allocates: the table's builder
     * calls it, on a control thread, never a real-time path.
     *
     * @param id The string; empty for a built-in engine or a provider of the enum
     * @return The stored string, or NULL for an empty one
     */
    const char* intern_id(const std::string& id);

    /**
     * @brief intern_id() in two steps: id strings that are not the session's yet
     *
     * For a control entry that must have everything that can fail behind it before it changes
     * the session (anira_handler_add_variant's commit). stage() answers with the pointer the
     * session holds the string under once adopt_ids() ran: the session's own copy when it has
     * the string already, else the copy this object makes, in a node that adopt_ids() moves
     * into the session's store, where it stays at its address. stage() allocates and may
     * throw; adopt_ids() does neither. Dropped without adopt_ids(), the object frees its
     * strings and the session is as it was (a pointer stage() gave for a new string is then
     * dangling). Control thread; nothing else interns in the session between the two calls
     * (the handler's control lock).
     */
    class StagedIds {
    public:
        /**
         * @param session The session the string is for
         * @param id The string; NULL or empty for a built-in engine or a provider of the enum
         * @return The string as the session holds it after adopt_ids(), or NULL for none
         */
        const char* stage(const SessionElement& session, const char* id);

    private:
        friend class SessionElement;
        std::set<std::string> m_ids;  ///< the strings the session lacks: node-based, as its store
    };

    /// The second half: the staged strings join the session's store, node by node. Allocates
    /// nothing and cannot fail.
    void adopt_ids(StagedIds&& ids) noexcept;

    unsigned long m_current_queue = 0;         ///< The time stamp of the next submitted chunk
    std::vector<unsigned long> m_time_stamps;  ///< Vector of timestamps for performance monitoring
    size_t m_pending_pull_samples = 0;  ///< Generator sessions only (!m_input_driven): samples of
                                        ///< the reference output the driving thread has demanded
                                        ///< that no submitted inference covers yet. Plain field,
                                        ///< written and read only on the session's driving thread
                                        ///< (like m_time_stamps).
    size_t m_pending_fills = 0;  ///< Hops of fill the receive rings owe behind the last chunk
                                 ///< collected and ahead of every later one (owe_fills,
                                 ///< place_pending_fills). Driving thread only; a reset clears
                                 ///< it.
    DroppedHops* m_dropped_hops = nullptr;  ///< What fills a dropped hop: the manager of the
                                            ///< session, set by its prepare and cleared before it
                                            ///< releases the session; nullptr: silence

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
    // lifetime: the underlying producer slot, its blocks with it, is recycled by the
    // next session's token when the session is destroyed. Its blocks are reserved at
    // prepare (Core::reserve_inference_capacity, queue_capacity() of them), so the
    // try_enqueue through it neither allocates nor runs out of room. A ProducerToken must
    // never be used by two threads at once —
    // this holds here because all enqueues of a session are serialized: the
    // non-stateful path enqueues only from the session's single driving thread,
    // and the stateful path enqueues only while holding the
    // m_stateful_dispatch_gate (whose acquire/release ordering also makes
    // the token's state visible when ownership migrates between threads); the
    // reservation runs on the control thread while nothing of the session is queued.
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
    // Built with no blocks and reserved at prepare for the struct pool
    // (reserve_pending_dispatch) — a pending entry is always a distinct
    // ThreadSafeStruct, so the depth is at most m_num_structs — and fed exclusively
    // through m_dispatch_producer_token, so enqueues never allocate: neither on the
    // audio thread (real-time safety) nor, on WebAssembly, from a non-main WASM
    // instance (which must not touch the shared allocator).
    moodycamel::ConcurrentQueue<std::shared_ptr<ThreadSafeStruct>>
        m_dispatch_pending;  ///< Prepared-but-not-yet-dispatched stateful
                             ///< tasks, in submission order
    // Same single-producer discipline as m_producer_token above: only the
    // session's driving thread enqueues pending dispatches.
    moodycamel::ProducerToken m_dispatch_producer_token;  ///< Explicit producer for
                                                          ///< m_dispatch_pending

    /** @brief Queue a prepared stateful task awaiting dispatch (called in submission order on the
     * session's driving thread; allocation-free). If the pending queue rejects the task — which
     * the reservation for the struct pool makes unreachable — it is completed with zeroed output
     * instead. */
    void enqueue_pending_dispatch(std::shared_ptr<ThreadSafeStruct> thread_safe_struct);
    /** @brief prepare's step for an exclusive session, after force_reset_dispatch_chain(): gives
     * m_dispatch_producer_token the blocks for every struct of the pool at once (one allocating
     * enqueue_bulk of placeholders, QueueCapacity.h), then takes the placeholders back. Holds
     * the gate meanwhile: the gate holder is the pending queue's only consumer, so no laggard
     * worker pops a placeholder (an empty struct pointer) or reuses a block before the ring holds
     * them all. Blocking (waits out a transient gate holder in 50us steps) and allocating —
     * control thread only.
     * @throws std::bad_alloc when the blocks cannot be allocated */
    void reserve_pending_dispatch() ANIRA_BLOCKING;
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
     * yields (silent) output at its correct stream position and the struct is freed normally.
     * The chunk stops counting on its plan slot ahead of the signal
     * (ThreadSafeStruct::leave_slot), unless an inference thread's job took the slot off it
     * before (a session momentarily uninitialized, the last resort): that job gives the count
     * back at its end. */
    void complete_with_zeros(const std::shared_ptr<ThreadSafeStruct>& thread_safe_struct);

    capi::ProcessorRunner& m_processor;  ///< The handler's processor runner: the four phases of
                                         ///< every chunk, and the port vectors of the rings
    /// The model's own latency of every output, in samples of its stream: the largest over the
    /// handler's variants (VariantBounds::m_internal_latency), one entry per output tensor.
    const std::vector<size_t> m_internal_latency;
    const capi::Derived& m_derived;    ///< The form of the stream: what the validator derived
                                       ///< for the handler's variants under the contract
    const capi::HardContract& m_hard;  ///< The Hard half of the contract of the prepare
    /// The instance count (Core::create_session: 1 for an exclusive session, else the fewest
    /// any variant allows, ANIRA_INSTANCES_AUTO resolved and clamped to the pool): the count
    /// the session is sized for (LatencyCalculator's n), and the pending dispatch queue of an
    /// exclusive session is sized by it.
    const uint32_t m_num_instances;
    /// The inference time the next prepare() sizes the session for, in ms (LatencyCalculator's
    /// T): the contract's explicit budget from the construction on; a 3.x handler sets it before
    /// each prepare (one clock step for its warm-up pass under a MEASURED budget, then the
    /// slowest plan's figure). The contract keeps what the user set. Control thread only, while
    /// no chunk of the session exists.
    double m_budget_ms;

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
    LatencyCalculator::Rational m_block_hops;  ///< The host block in hops, rho = host_block_max / R
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
    /// The id strings of intern_id(): node-based, so a stored string never moves.
    std::set<std::string> m_interned_ids;

    /// The published plan table: what publish_plans() replaces and a read lease holds a
    /// version of. Empty until Core::create_session publishes the list it built.
    thl::RCU<PlanTable> m_plan_table;
    /// The reader slots of m_plan_table, added once, here, on the control thread that makes
    /// the session; claim_plans() hands out the leases. Declared behind the table it reads.
    RcuReaders<PlanTable> m_plan_readers{m_plan_table};
    /// The entries of the table published last (its indices, the empty ones included): what
    /// select_plan checks an index against, from any thread. Stored by publish_plans().
    std::atomic<uint32_t> m_num_published_plans{0};

    /// The stamp itself: the index and, for a slot, the count and the copies of what the slot
    /// reports; no slot, variant 0 and no backend for NULL. The caller holds the chunk.
    static void stamp_slot(ThreadSafeStruct& chunk,
                           uint32_t plan,
                           PlanSlot* slot) noexcept ANIRA_NONBLOCKING;

    /**
     * @brief Whether any input tensor is streamable
     *
     * @return True if at least one input has a non-zero preprocess_input_size
     */
    bool has_streamable_input() const;

    /// The model's own latency of an output, in samples of its stream (the spec's latency, the
    /// largest over the variants: m_internal_latency).
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