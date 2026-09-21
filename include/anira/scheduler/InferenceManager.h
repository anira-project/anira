#ifndef ANIRA_INFERENCEMANAGER_H
#define ANIRA_INFERENCEMANAGER_H

#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>

#include <chrono>
#include <cstddef>
#include <vector>

#include "../CoreConfig.h"
#include "../InferenceConfig.h"
#include "../PrePostProcessor.h"
#include "../utils/HostConfig.h"
#include "../utils/InferenceBackend.h"
#include "Core.h"
#include "InferenceThread.h"

/// The context config the 3.x constructor takes (its body is src/capi/handles.h).
struct anira_context_config;

namespace anira {

/// The real-time latch the 3.x constructor hands the session (include/anira/utils/RtLatch.h).
struct RtLatch;

/**
 * @brief Central manager class for coordinating neural network inference operations
 *
 * The InferenceManager class serves as the primary coordinator for neural network
 * inference in real-time audio processing applications. It manages the complete
 * inference pipeline including input preprocessing, backend execution scheduling,
 * output postprocessing, and session management with multiple inference threads.
 *
 * Key responsibilities:
 * - Managing inference sessions and thread coordination
 * - Handling input/output data flow and buffering
 * - Coordinating with PrePostProcessor for data transformation
 * - Managing latency compensation and sample counting
 * - Providing thread-safe access to inference operations
 * - Supporting both real-time and non-real-time processing modes
 *
 * The manager supports multiple processing patterns:
 * - Synchronous processing with immediate input/output
 * - Asynchronous push/pop processing for decoupled operation
 * - Multi-tensor processing for complex model architectures
 * - Custom latency handling for different model types
 *
 * Host blocks reach the rings through one copy path, written over anira_tensor (see "The
 * tensor stems" below): one tensor per slot, of any ring element type, planar, contiguous or
 * interleaved. The manager takes host tensors and nothing else. The float face of the library,
 * anira::InferenceHandler, presents its channel pointers as planar float32 tensors before it
 * calls it; a direct caller that holds channel pointers, a host of the C ABI included, does
 * the same with anira_tensor_init_host_planar().
 *
 * @note This class coordinates between multiple components and should be used
 *       as the primary interface for inference operations rather than directly
 *       accessing lower-level components.
 *
 * @see InferenceThread, PrePostProcessor, Core, HostConfig, InferenceConfig
 */
class ANIRA_API InferenceManager {
public:
    /**
     * @brief Default constructor is deleted to prevent uninitialized instances
     */
    InferenceManager() = delete;

    /**
     * @brief The 2.x constructor: initializes the inference manager with all required
     * components
     *
     * Creates an inference manager with the specified preprocessing/postprocessing pipeline,
     * inference configuration, and optional custom backend. Maps the CoreConfig onto a
     * context config (anira::capi::make_context_config) and delegates to the 3.x
     * constructor, so the core reads one configuration type. Leaves with the 2.x classes.
     *
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration containing model settings
     * @param custom_processor Pointer to a custom backend processor (can be nullptr for default
     * backends)
     * @param core_config Configuration for the inference core and thread management
     */
    InferenceManager(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     BackendBase* custom_processor,
                     const CoreConfig& core_config);

    /**
     * @brief The 3.x constructor: the context's config passed through unchanged
     *
     * Creates the session through Core::create_session() with the context config as the
     * handler's context holds it (the core reads its six scalars and reconciles them
     * against the configuration in effect) and the handler's real-time latch.
     *
     * @param pp_processor Reference to the preprocessing/postprocessing pipeline
     * @param inference_config Reference to the inference configuration containing model settings
     * @param custom_processor Pointer to a custom backend processor (can be nullptr for default
     * backends)
     * @param context_config The context's configuration (anira_context_config_*)
     * @param rt_latch The handler's latch the session records its failures into; nullptr
     * gives the session its own
     */
    InferenceManager(PrePostProcessor& pp_processor,
                     InferenceConfig& inference_config,
                     BackendBase* custom_processor,
                     const anira_context_config& context_config,
                     anira::RtLatch* rt_latch = nullptr);

    /**
     * @brief Destructor that properly cleans up inference resources
     *
     * Ensures proper shutdown of inference threads, cleanup of sessions,
     * and release of all managed resources.
     */
    ~InferenceManager();

    /**
     * @brief Prepares the inference manager for processing with new audio configuration
     *
     * Initializes the inference pipeline with the specified host configuration and
     * optional custom latency settings. This method must be called before processing
     * begins or when audio settings change.
     *
     * @param config Host configuration containing sample rate, buffer size, and audio settings
     * @param custom_latency Optional vector of custom latency values for each tensor (empty for
     * automatic calculation)
     */
    void prepare(HostConfig config, std::vector<long> custom_latency = {});

    /**
     * @brief The prepare of the 3.x path: like prepare(HostConfig, std::vector<long>), with
     * the caller's latencies and the element type of every send and receive ring named per
     * slot.
     *
     * @param config Host configuration containing sample rate, buffer size, and audio settings
     * @param custom_latencies A caller's latency per output tensor (see CustomLatencies)
     * @param ring_dtypes The ring dtype of every slot (see RingDtypes); the two-argument
     * overload passes float32 for every slot
     */
    void prepare(HostConfig config,
                 const CustomLatencies& custom_latencies,
                 const RingDtypes& ring_dtypes);

    /**
     * @name The tensor stems
     *
     * The host<->ring copy path, and the only way into it: what an entry that takes host
     * tensors calls, and what the float face (anira::InferenceHandler) calls with the planar
     * float32 tensors it presents. A call hands over one anira_tensor per slot, in slot order and
     * covering every slot of that side: `inputs` has one tensor per input tensor of the
     * InferenceConfig, `outputs` one per output tensor.
     *
     * The stems move the streamed slots. A non-streamable slot (no stream size in the
     * InferenceConfig: a Static tensor of a 3.x model) has no ring and is not the copy path's:
     * its tensor is never read, whatever it holds, its delivered count is always 0, and a miss
     * leaves its memory alone. Its values travel beside the stems: through the Static store of a
     * 3.x handler (whole tensors in the spec's shape, src/capi/static_store.h), and for the 2.x
     * face through anira::NonStreamableRouter (src/scheduler/NonStreamableRouter.h), which keeps
     * the value count, the clamp and the miss rules of the 2.x `float***` blocks.
     *
     * The tensor of a streamed slot is a host block of the logical shape `[channels, samples]`,
     * over memory in one of three descriptions, for any channel count:
     * - planar (ANIRA_TENSOR_PLANAR): `handle.planes.ptrs[c]` is channel `c`; `byte_offset` and
     *   `strides[1]` apply inside each plane, `strides[0]` is ignored;
     * - one block read by `strides` (in elements): channel `c` starts `c * strides[0]`
     *   elements after `handle.host.ptr + byte_offset` and its samples are `strides[1]`
     *   elements apart. Interleaved is `{1, channels}`, contiguous is `{samples, 1}`;
     * - one block with all-zero strides: packed row-major, that is `{samples, 1}`.
     *
     * Every description is copied straight between the host memory and the ring, one strided
     * ring call per channel, with no intermediate copy; a unit stride is the ring's plain
     * block copy.
     *
     * The request of a slot is `shape[1]`: the samples to push, or to pop. A slot whose
     * `shape[1]` is 0 is not touched and its memory arm is not read, so an empty tensor may carry
     * NULL memory: that is how a call leaves a slot out. The descriptors are const: anira never
     * writes a tensor, an output's included; it writes the memory the output names. In place is the
     * same tensor on both sides (or two tensors that describe the same memory the same way).
     *
     * Trust. The descriptors are validated by the caller (the library's own check is
     * anira::tensor_run::check_host_tensor, src/scheduler/TensorRun.h). The manager takes on
     * trust the rank (2), the domain (host memory), `shape[0]` (the slot's channel count), the
     * pointers, `byte_offset` and the strides (1 or more on the sample axis, or all zero), and
     * does not check them again: a malformed tensor is undefined behaviour here. The one thing
     * it checks is the dtype, because the float faces reach it without a validator: the
     * tensor's dtype must be the slot's (the ring dtype, see RingDtypes), nothing
     * converts. A mismatched input is not pushed; a mismatched output is zero-filled at the
     * tensor's own element size, is not popped and reports 0; either is logged once through
     * the real-time queue (RtSite::TensorDtypeMismatch).
     *
     * The returned array is the manager's own, one delivered count per output slot, valid
     * until the next call: the request of a delivered block, 0 for a non-streamable slot and
     * for every slot of a missed block (last_block_missed()). Nothing bounds a
     * request by the host block size: a push larger than the send ring keeps its tail, a
     * request the receive ring cannot serve is a miss.
     *
     * On a miss, ZEROS and HOLD_LAST work for every description. BYPASS copies the anchored
     * input's channels to every streamed output whose dtype is the input's, whatever the two
     * descriptions, and leaves the samples where they are when both name the same run (the
     * in-place call); input and output memory that overlap under two different descriptions
     * (an interleaved input over planar views of the same bytes) are undefined on a BYPASS
     * miss. The delivered path is safe for any description: the input is read completely
     * before an output is written.
     *
     * None of them allocates. process_nowait(), push_data() and pop_data() never wait and
     * are real-time safe; process_wait(), pop_data_wait() and the deadline form of pop_data()
     * wait, and process() waits when the InferenceConfig has a blocking ratio.
     */
    ///@{

    /**
     * @brief The wait-free process call: push, submit, collect what completed, pop
     *
     * Pushes the input, registers the output demand, submits the block and collects the
     * results that have completed (Core::new_data_request() without a deadline) before
     * popping; a block that has not completed is delivered through the miss policy
     * (set_miss_policy()) and counted as missing. The 3.x path's process form; process() is
     * this when InferenceConfig::m_blocking_ratio is 0.
     *
     * @param inputs One tensor per input slot (see "The tensor stems")
     * @param outputs One tensor per output slot; the same tensor as the input for in place
     * @return The delivered count per output slot (0 on a miss)
     *
     * @note Real-time safe: allocates nothing, never waits.
     */
    const size_t* process_nowait(const anira_tensor* inputs, const anira_tensor* outputs);

    /**
     * @brief The process call that waits up to a budget for the block's result
     *
     * Like process_nowait(), but waits for the submitted block's result on the session's
     * completion primitive (Core::new_data_request() with a deadline): the semaphore when
     * InferenceConfig::m_blocking_ratio > 0, else the atomic flag polled every
     * millisecond. The deadline is read from the clock after the submit, exactly where the
     * 2.x process() read it. A budget of std::chrono::steady_clock::duration::max() waits
     * without limit, re-checking that an inference thread runs (Core::WaitOutcome::NoThread
     * otherwise).
     *
     * @param inputs One tensor per input slot (see "The tensor stems")
     * @param outputs One tensor per output slot; the same tensor as the input for in place
     * @param budget How long to wait at most (duration::max() = without limit)
     * @param outcome How the wait ended
     * @return The delivered count per output slot (0 on a miss)
     *
     * @note Not real-time safe for as long as it waits.
     */
    const size_t* process_wait(const anira_tensor* inputs,
                               const anira_tensor* outputs,
                               std::chrono::steady_clock::duration budget,
                               Core::WaitOutcome& outcome);

    /**
     * @brief Pushes input to the inference pipeline for asynchronous processing
     *
     * Queues input for processing without waiting for results: the input side of the
     * decoupled push/pop pattern. Finished inferences are collected here as well, as long as
     * the receive buffers have room for them (Core::collect_completed()), so push-only usage
     * never exhausts the inference structs; a host that never pops a streamed output is
     * warned. On a generator session (no streamable input) there is nothing to push: inference
     * is driven by the output demand of the process and pop forms.
     *
     * @param inputs One tensor per input slot (see "The tensor stems")
     *
     * @note Real-time safe: allocates nothing, never waits.
     */
    void push_data(const anira_tensor* inputs);

    /**
     * @brief Pops processed output from the inference pipeline (non-blocking)
     *
     * The output side of the decoupled push/pop pattern: collects what has completed and
     * pops, without waiting. On a generator session (no streamable input) this is the pull
     * that drives inference: the requested number of samples on the reference output is added
     * to the demand and one inference is submitted per postprocess_output_size demanded
     * samples, before results are collected.
     *
     * @param outputs One tensor per output slot (see "The tensor stems")
     * @return The delivered count per output slot (0 on a miss)
     *
     * @note Real-time safe: allocates nothing, never waits.
     */
    const size_t* pop_data(const anira_tensor* outputs);

    /**
     * @brief pop_data() that waits up to a budget for the next result
     *
     * Registers the output demand (a generator is submitted here), then waits as
     * process_wait() does and pops. There is no input in a pop, so a BYPASS miss delivers
     * zeros.
     *
     * @param outputs One tensor per output slot (see "The tensor stems")
     * @param budget How long to wait at most (duration::max() = without limit)
     * @param outcome How the wait ended
     * @return The delivered count per output slot (0 on a miss)
     *
     * @note Not real-time safe for as long as it waits.
     */
    const size_t* pop_data_wait(const anira_tensor* outputs,
                                std::chrono::steady_clock::duration budget,
                                Core::WaitOutcome& outcome);

    /**
     * @brief The 2.x process call over host tensors: the dispatcher on
     * InferenceConfig::m_blocking_ratio
     *
     * With a blocking ratio above 0 this is process_wait() with the budget of
     * contract_wait_budget() (the block's own duration times the ratio, waited for on the
     * semaphore) and the outcome dropped; without one it is process_nowait(). What
     * anira::InferenceHandler::process() calls; the tensors pass through as they are.
     *
     * @param inputs One tensor per input slot (see "The tensor stems")
     * @param outputs One tensor per output slot; the same tensor as the input for in place
     * @return The delivered count per output slot (0 on a miss)
     *
     * @note Real-time safe without a blocking ratio; with one it waits up to the budget.
     */
    const size_t* process(const anira_tensor* inputs, const anira_tensor* outputs);

    /**
     * @brief The 2.x deadline pop over host tensors
     *
     * Registers the output demand (a generator is submitted here), then waits until
     * `wait_until` for the next result on the session's semaphore and pops. A session without
     * a blocking ratio has no semaphore to wait on: nothing is waited for and nothing is
     * collected, the call pops what the receive rings hold, and the site
     * RtSite::WaitWithoutSemaphore records it once. What
     * anira::InferenceHandler::pop_data() with a time point calls; pop_data_wait() is the 3.x
     * form, which waits on either completion primitive and reports how the wait ended.
     *
     * @param outputs One tensor per output slot (see "The tensor stems")
     * @param wait_until Time point until which the call waits for the result
     * @return The delivered count per output slot (0 on a miss)
     *
     * @note Not real-time safe for as long as it waits.
     */
    const size_t* pop_data(const anira_tensor* outputs,
                           std::chrono::steady_clock::time_point wait_until);

    /**
     * @brief The 2.x blocking_ratio wait of one process call
     *
     * InferenceConfig::m_blocking_ratio times the reference block's duration: `shape[1]` of the
     * anchored tensor over the host sample rate, in the arithmetic of the 2.x process(), so
     * the deadline the 2.x path waits for is unchanged. What a waiting entry computes before
     * it calls process_wait(). The process forms only: a pop has no input block to measure.
     *
     * @param inputs One tensor per input slot
     * @param outputs One tensor per output slot
     * @return The wait budget
     */
    std::chrono::steady_clock::duration contract_wait_budget(
        const anira_tensor* inputs,
        const anira_tensor* outputs) const noexcept;

    ///@}

    /**
     * @brief Replaces the session's plan table (SessionElement::m_plan_backends) and selects
     * plan 0: one backend per plan, in dense-index order.
     *
     * What a 3.x handler calls with exactly its plans, after construction and before
     * prepare(): only while no chunk of the session exists. A 2.x session keeps the table
     * Core::create_session built.
     *
     * @throws std::invalid_argument for an empty table
     * @throws std::logic_error after prepare(): the table is read without synchronization
     * and is replaced before prepare only; a refused call changes nothing
     */
    void set_plan_backends(std::vector<InferenceBackend> backends);

    /**
     * @brief Selects the plan the next submitted chunk runs on
     *
     * The whole runtime selection: one relaxed store of one atomic (the dense plan index),
     * wait-free and callable from any thread; nothing is reinitialized, every plan was
     * loaded when the session was created. The index and not the backend is what is stored,
     * so two plans on one backend stay distinct. The switch applies from the next submitted
     * chunk on: a chunk is stamped with the index at submission (Core::pre_process) and its
     * pre_process, before_inference, engine call, after_inference and post_process all run
     * under that stamp, so a chunk that is queued or in flight when the switch lands
     * finishes on the plan it started on.
     *
     * @param plan A dense index into the plan table
     * @return False, and the selection unchanged, for an index out of range
     */
    bool set_plan(uint32_t plan) noexcept;

    /// The plan selected last: one relaxed load of the value set_plan() stores.
    uint32_t get_plan() const noexcept;

    /**
     * @brief Sets the inference backend to use for neural network processing
     *
     * The 2.x selection by backend, over the plan table: set_plan() of the first plan that
     * runs on the backend, with everything set_plan() says. A backend no plan of the session
     * runs on leaves the selection unchanged and is logged once through the real-time queue
     * (RtSite::BackendWithoutPlan); a 2.x session has a plan for every backend of the build.
     *
     * @param new_inference_backend The backend type to use (ONNX, LibTorch, TensorFlow Lite, or
     * Custom)
     */
    void set_backend(InferenceBackend new_inference_backend);

    /**
     * @brief Gets the currently active inference backend
     *
     * @return The backend the selected plan runs on
     */
    InferenceBackend get_backend() const;

    /**
     * @brief Gets the processing latency for all tensors
     *
     * Returns the latency introduced by the inference processing in samples for each output
     * tensor. This includes buffering delays, preprocessing/postprocessing latency, and
     * model-specific processing latency. Non-streamable outputs carry no stream latency and
     * report 0.
     *
     * @return Vector containing latency values in samples for each output tensor index
     */
    std::vector<unsigned int> get_latency() const;

    /**
     * @brief The session's latency vector by reference
     *
     * Index-aligned with the output list, 0 for a non-streamable output; valid from
     * prepare() on. No copy: the ANIRA_NONBLOCKING accessors read it.
     *
     * @return The latency in samples per output tensor index
     */
    const std::vector<unsigned int>& latencies() const noexcept { return m_session->m_latency; }

    /**
     * @brief The session behind this manager
     *
     * For the stage chain of a 3.x handler, which binds to the session's structs and rings
     * after prepare() (the pre/post virtuals receive a struct's buffers, never the struct).
     * The session lives as long as the manager.
     *
     * @return The session
     */
    SessionElement& session() noexcept { return *m_session; }

    /**
     * @brief What a missed block delivers (anira_miss_policy)
     *
     * ANIRA_MISS_ZEROS is the 2.x behaviour and the default; ANIRA_MISS_HOLD_LAST repeats
     * the last delivered block of each output (the buffers are allocated at prepare());
     * ANIRA_MISS_BYPASS passes the anchored input's block through to the streamed outputs
     * of a process call, zeros elsewhere; ANIRA_MISS_CALLBACK calls the hook of
     * set_miss_hook() once for the whole block and zero-fills when it declines. Under every
     * policy a missed block returns 0 and counts as missing. Set before prepare(); never on
     * the driver thread.
     *
     * @param policy The miss policy
     */
    void set_miss_policy(anira_miss_policy policy) noexcept { m_on_miss = policy; }

    /// The policy set_miss_policy() stored (ANIRA_MISS_ZEROS by default). What a face that
    /// fills part of a missed block itself reads: the 2.x face's non-streamable outputs.
    anira_miss_policy miss_policy() const noexcept { return m_on_miss; }

    /**
     * @brief The function ANIRA_MISS_CALLBACK calls to fill a missed block
     *
     * Called once per missed block on the thread that drives the call, in the starvation path
     * of process_output(): after the catch-up and the all-or-nothing gate, before anything is
     * filled. It receives the arrays the stem was handed, as they are, so nothing is rebuilt
     * on a miss: the caller's tensors (under a float face of the library the planar float32
     * tensors that face presented, in which a slot the call did not carry is an empty tensor
     * without memory), and the manager's empty input tensors in a pop. The outputs' shape[1]
     * are the requests; the input block of a process form is already pushed and still intact
     * in host memory. It must be real-time safe and must not call back into this manager.
     */
    using MissHook = bool (*)(void* ctx,
                              const anira_tensor* inputs,
                              uint32_t num_inputs,
                              const anira_tensor* outputs,
                              uint32_t num_outputs) noexcept;

    /**
     * @brief Sets the hook of ANIRA_MISS_CALLBACK
     *
     * `true` from the hook means the outputs are filled; `false`, or no hook, zero-fills every
     * requested output. The accounting of a missed block (the missing samples, the record,
     * the returned counts of 0, last_block_missed()) is the same either way. Read under
     * ANIRA_MISS_CALLBACK only. Set before prepare(); never on the driver thread.
     *
     * @param hook The hook, or nullptr
     * @param ctx Handed to the hook as it is
     */
    void set_miss_hook(MissHook hook, void* ctx) noexcept {
        m_miss_hook = hook;
        m_miss_hook_ctx = ctx;
    }

    /**
     * @brief Whether the last process or pop form delivered a missed block
     *
     * True when the last process_output() took the starvation path: the miss policy filled
     * every requested output and their counts came back 0. The 3.x Hard entries turn it into
     * ANIRA_MISSED. Driver thread only; false after prepare() and reset().
     */
    bool last_block_missed() const noexcept { return m_last_missed; }

    /**
     * @brief Gets the number of samples received for a specific tensor and channel (for unit
     * testing)
     *
     * This method is primarily used for unit testing and debugging purposes to
     * monitor the data flow through the inference pipeline.
     *
     * @param tensor_index Index of the tensor to query
     * @param channel Channel index to query
     * @return Number of samples received for the specified tensor and channel
     */
    size_t get_available_samples(size_t tensor_index, size_t channel) const;

    /**
     * @brief Gets the current session ID
     *
     * Returns the unique identifier for the current inference session.
     * This can be useful for debugging and session tracking purposes.
     *
     * @return The current session ID
     */
    int get_session_id() const;

    /**
     * @brief Configures the session for non-real-time (offline) operation
     *
     * When enabled, Core::new_data_request() blocks the calling thread until
     * every pending inference for this session completes, instead of returning
     * early (blocking_ratio == 0) or giving up at a deadline (blocking_ratio > 0).
     * This means process()/pop_data() always yield complete output -- never a
     * dropped/zero-filled chunk -- at the cost of an unbounded wait, so it is
     * intended for offline rendering (e.g. bounce-to-disk), not the live audio
     * thread.
     *
     * @param is_non_realtime True to block for complete output (non-real-time
     * mode), false to restore the bounded/non-blocking real-time behavior
     *
     * @warning Not real-time safe while enabled. Requires at least one
     * inference thread to exist (Core::has_inference_threads()) — without
     * one the blocking waits could never complete, so the call is refused with
     * a warning. On WebAssembly that means spinning up at least one inference
     * worker (AniraWeb.spinUpInferenceWorker()) before enabling this mode; the
     * waits there are busy-waits (spins), so run offline processing in a
     * Worker or under an OfflineAudioContext rather than on the main thread.
     * The check runs once at enable time: stopping all inference threads while
     * non-real-time mode is active re-creates the hang.
     */
    void set_non_realtime(bool is_non_realtime) const;

    /**
     * @brief Drains the core's real-time log queue into the log sinks
     * (see InferenceHandler::drain_log()). Not real-time safe.
     */
    size_t drain_log() const;

    /**
     * @brief Wait-free reset of the inference session to its initial state.
     *
     * Clears the session's buffers and re-anchors the inference grid without ever
     * blocking on in-flight inferences (see Core::reset_session): every
     * already-dispatched inference is invalidated via the session generation, its
     * result discarded and its structure reclaimed lazily. Safe to call from the
     * audio thread, for all session types. Also resets the missing-samples
     * bookkeeping.
     *
     * @note Does NOT wait for in-flight inferences: a worker thread may still be
     *       executing a (discarded) inference — including user code in a custom
     *       backend or the before_inference()/after_inference() hooks — after
     *       this returns. Call prepare() if you need that quiescence.
     */
    void reset();

private:
    /**
     * @brief What the copy path knows about one slot, read once at prepare()
     */
    struct SlotCopy {
        size_t m_channels = 1;  ///< Runs of the slot's host block: the channel count
        size_t m_element_size = sizeof(float);  ///< Bytes of one element of m_dtype
        anira_dtype m_dtype = ANIRA_DTYPE_F32;  ///< The ring dtype
        bool m_streamed = false;  ///< Whether the slot has a ring; the stems move no other
    };

    /**
     * @brief Host -> ring: pushes every carried input (the one implementation under every
     * process and push form)
     *
     * A streamed slot is one strided ring push per channel; a non-streamable slot is never
     * carried (load_counts()). An input whose count is 0 is not touched and its tensor's memory arm
     * is not read (the single-tensor forms hand the other slots over without memory). An input
     * whose dtype is not the slot's is not pushed, its count is set to 0 and the site
     * RtSite::TensorDtypeMismatch records it.
     *
     * @param inputs One tensor per input slot; trusted but for the dtype
     * @param num_samples The manager's input counts, loaded from shape[1]
     */
    void process_input(const anira_tensor* inputs, size_t* num_samples);

    /**
     * @brief Ring -> host: pops every requested output (the one implementation under every
     * process and pop form)
     *
     * Pops every requested output when the whole block has completed; otherwise delivers
     * the block through the miss policy (zeros, the held block, or the anchored input's
     * block), counts the request as missing and returns 0 for every output. An output
     * whose count is 0 is not touched on either path (the single-tensor forms hand the
     * other slots over without memory). An output whose dtype is not the slot's is zero-filled at
     * the tensor's element size before anything else moves, and is then a count of 0.
     *
     * @param outputs One tensor per output slot; trusted but for the dtype
     * @param num_samples The manager's output counts, loaded from shape[1]
     * @param bypass_inputs The inputs of a process call (the BYPASS source); the manager's
     * empty inputs in a pop
     * @param bypass_num_input Their sample counts (all 0 in a pop)
     * @return num_samples: the delivered counts
     */
    size_t* process_output(const anira_tensor* outputs,
                           size_t* num_samples,
                           const anira_tensor* bypass_inputs,
                           const size_t* bypass_num_input);

    /**
     * @brief Delivers the held block of one streamed output (ANIRA_MISS_HOLD_LAST)
     *
     * Copies what the last delivered block left in the hold storage, zero-filled beyond it
     * (nothing held since prepare() or reset(): zeros).
     *
     * @param output The output's tensor
     * @param num_samples The requested samples
     * @param tensor_index The streamed output to fill
     */
    void hold_output(const anira_tensor& output, size_t num_samples, size_t tensor_index);

    /**
     * @brief Passes the anchored input's block through to one streamed output
     * (ANIRA_MISS_BYPASS)
     *
     * Channel by channel up to the input's channel count, min(input, output) samples then
     * zeros; output channels beyond the input's are zeros. A channel whose input run and
     * output run are the same memory read the same way (an in-place call) is left where it
     * is.
     *
     * @param output The output's tensor
     * @param num_samples The requested samples
     * @param tensor_index The streamed output to fill
     * @param input The anchored input's tensor, of the output's dtype
     * @param num_input The samples the call pushed of it
     */
    void bypass_output(const anira_tensor& output,
                       size_t num_samples,
                       size_t tensor_index,
                       const anira_tensor& input,
                       size_t num_input);

    /**
     * @brief Zeros the requested samples of one output (the 2.x clear rule for one tensor)
     *
     * @param output The output's tensor
     * @param num_samples The requested samples
     * @param tensor_index The output to clear
     */
    void clear_output(const anira_tensor& output, size_t num_samples, size_t tensor_index);

    /**
     * @brief Whether a carried tensor has its slot's dtype; a mismatch is recorded
     * (RtSite::TensorDtypeMismatch)
     */
    bool has_slot_dtype(const anira_tensor& tensor,
                        const SlotCopy& slot,
                        const char* side,
                        size_t tensor_index) const;

    /// Loads the manager's count array of one side: shape[1] of the tensor of every streamed
    /// slot, 0 for a non-streamable slot, whose tensor is not read.
    static void load_counts(const anira_tensor* tensors,
                            const std::vector<SlotCopy>& slots,
                            std::vector<size_t>& counts) noexcept;

    /// The arithmetic of contract_wait_budget() for a reference block of `reference_samples`.
    std::chrono::steady_clock::duration wait_budget_of(size_t reference_samples) const noexcept;

    /**
     * @brief Registers the host's output demand on a generator session
     *
     * For a session without streamable input the requested sample count of the
     * reference output is what drives inference (see Core::new_data_submitted()).
     * No-op for input-driven sessions.
     *
     * @param num_output_samples Array of requested output sample counts for each tensor
     */
    void request_output(const size_t* num_output_samples);

    /**
     * @brief Collects finished inferences without waiting
     *
     * Uses the completion signal the session actually runs on: the semaphore
     * (try_acquire) for blocking_ratio > 0, the atomic flag otherwise.
     */
    void collect_nonblocking();

private:
    InferenceConfig& m_inference_config;  ///< Reference to the inference configuration containing
                                          ///< model settings
    std::shared_ptr<SessionElement> m_session;  ///< Shared pointer to the current inference session
    HostConfig m_host_config;                   ///< Current host audio configuration

    std::vector<size_t> m_missing_samples;  ///< Track missing samples for latency compensation and
                                            ///< buffering

    anira_miss_policy m_on_miss = ANIRA_MISS_ZEROS;  ///< What a missed block delivers
    MissHook m_miss_hook = nullptr;                  ///< ANIRA_MISS_CALLBACK: fills a missed block
    void* m_miss_hook_ctx = nullptr;                 ///< The hook's first argument
    std::vector<std::vector<unsigned char>> m_hold;  ///< HOLD_LAST: per streamed output, the
                                                     ///< bytes of channels x capacity elements
                                                     ///< of the ring dtype, channel after channel
    std::vector<size_t> m_hold_capacity;  ///< Per output: the largest block a call may request
    std::vector<size_t> m_hold_len;       ///< Per output: samples held (0 = nothing delivered
                                          ///< since prepare() or reset())
    bool m_last_missed = false;           ///< The last process_output() took the starvation path
                                          ///< (last_block_missed()); driver thread only

    // The copy state of the tensor stems: sized and filled by prepare(), never resized on the
    // driver thread.
    std::vector<SlotCopy> m_input_slots;       ///< Per input slot
    std::vector<SlotCopy> m_output_slots;      ///< Per output slot
    std::vector<size_t> m_input_counts;        ///< Per input slot: the request of the running call
    std::vector<size_t> m_output_counts;       ///< Per output slot: the request, then the delivered
                                               ///< count; what the tensor stems return
    std::vector<anira_tensor> m_empty_inputs;  ///< One empty tensor per input slot: the inputs
                                               ///< of a pop

#if DOXYGEN
    // Since Doxygen does not find classes structures nested in std::shared_ptr
    Core* __doxygen_force_0;  ///< Placeholder for Doxygen to find Core class documentation
    SessionElement* __doxygen_force_1;  ///< Placeholder for Doxygen to find SessionElement class
                                        ///< documentation
#endif
};

}  // namespace anira

#endif  // ANIRA_INFERENCEMANAGER_H