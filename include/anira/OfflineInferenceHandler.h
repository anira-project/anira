#ifndef ANIRA_OFFLINEINFERENCEHANDLER_H
#define ANIRA_OFFLINEINFERENCEHANDLER_H

#include <concurrentqueue.h>
#ifndef __EMSCRIPTEN__
#include <blockingconcurrentqueue.h>
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <vector>
#ifndef __EMSCRIPTEN__
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#endif

#include "ContextConfig.h"
#include "InferenceConfig.h"
#include "PrePostProcessor.h"
#include "anira/system/AniraWinExports.h"
#include "backends/BackendBase.h"
#include "scheduler/InferenceManager.h"  // IWYU pragma: keep - Lane holds a unique_ptr<InferenceManager>
#include "utils/InferenceBackend.h"

namespace anira {

/// Monotonically increasing job identifier returned by OfflineInferenceHandler::submit().
using OfflineJobId = uint64_t;

/// Returned by OfflineInferenceHandler::submit() when a job is rejected at validation time.
inline constexpr OfflineJobId k_invalid_offline_job_id = 0;

/**
 * @brief How completed offline jobs are delivered to the application
 *
 * - Immediate: the lane worker thread invokes the callback right when the job finishes.
 * - Polled: completed jobs are queued; callbacks fire on the thread that calls
 *   OfflineInferenceHandler::poll() / poll_wait(). Useful to receive callbacks on a GUI/main
 *   loop thread.
 *
 * @note On WebAssembly delivery is always polled (the platform wrapper drains the completion
 *       queue); Immediate is coerced to Polled with a warning.
 */
enum class OfflineDeliveryMode { Immediate, Polled };

/**
 * @brief Result of a completed offline job, passed to the completion callback
 */
struct ANIRA_API OfflineJobResult {
    OfflineJobId m_job_id = k_invalid_offline_job_id;  ///< The id returned by submit()
    bool m_success = false;  ///< False for jobs dropped at shutdown or rejected input
    std::vector<size_t> m_num_output_samples_written;  ///< Per output tensor: samples per channel
                                                       ///< written to the output buffer (0 for
                                                       ///< non-streamable tensors)
    std::vector<std::vector<float>> m_non_streamable_outputs;  ///< Final values of non-streamable
                                                               ///< output tensors (empty vectors
                                                               ///< for streamable tensors)
};

/**
 * @brief Per-job options for OfflineInferenceHandler::submit() / process_job()
 */
struct ANIRA_API OfflineJobOptions {
    /**
     * @brief Per-job values for non-streamable input tensors (preprocess_input_size == 0)
     *
     * Outer index = input tensor index; an empty inner vector leaves the tensor's current value
     * untouched, otherwise the inner size must equal the tensor's total size. Values are applied
     * via PrePostProcessor::set_input() before the first chunk, stay constant for the whole job
     * and are snapshot-and-restored afterwards. Since the values live in the shared
     * PrePostProcessor, a job carrying them runs EXCLUSIVELY (all lanes drained first). For
     * values that are constant across jobs, set them once directly on the PrePostProcessor
     * instead to keep full lane parallelism.
     */
    std::vector<std::vector<float>> m_non_streamable_inputs;

    /**
     * @brief Head-trim per streamable output tensor
     *
     * -1 (default) trims the model-internal latency (== get_latency()) so the output is
     * input-aligned; >= 0 trims exactly that many samples (0 = raw untrimmed output, starting
     * with the model's warm-up samples for models with internal latency). Empty = default for
     * all tensors; otherwise the size must equal the number of output tensors.
     */
    std::vector<long> m_head_trim;

    /**
     * @brief Push zero chunks after the input so latency-delayed samples emerge
     *
     * Disable for raw-mode output; the written sample count is then clamped to what the real
     * input chunks produced.
     */
    bool m_tail_flush = true;
};

/**
 * @brief Asynchronous offline (non-real-time) inference handler with a job-based API
 *
 * The OfflineInferenceHandler processes whole, arbitrary-length buffers ("jobs") through the
 * same ring-buffer / PrePostProcessor / inference-thread-pool pipeline that the real-time
 * InferenceHandler uses. A job is submitted together with a completion callback; the handler
 * chunks the buffer internally, runs the chunks through the shared inference pool (online and
 * offline tasks are executed by the same threads), reassembles the time-aligned output into the
 * caller's buffer, and invokes the callback when the whole job is done.
 *
 * Internally the handler manages a number of parallel **lanes** (num_parallel_jobs). Each lane
 * owns its own session (ring buffers, inference-queue structs) and its own pump worker thread.
 * Jobs are dispatched FIFO to the first free lane, so multiple jobs run truly in parallel while
 * chunks within each job are additionally parallelized across the inference pool.
 *
 * Key semantics:
 * - Jobs are independent signals: the session stream state is cleared automatically after every
 *   job, so each job starts with pristine zero left-context.
 * - Output is input-aligned by default: the model-internal latency is trimmed from the head
 *   and the input is zero-flushed at the tail, so output sample 0 corresponds to input
 *   sample 0. The lanes are prepared without any real-time latency bookkeeping (no prefill,
 *   struct pool sized exactly to the in-flight bound, max_inference_time ignored).
 * - Completion callbacks fire asynchronously from the lane's worker thread
 *   (OfflineDeliveryMode::Immediate, default) or from whichever thread drains poll()
 *   (OfflineDeliveryMode::Polled). With multiple lanes, callbacks may fire concurrently and out
 *   of submission order.
 * - Input/output buffers are BORROWED: they must stay valid and untouched from submit() until
 *   the job's callback returns.
 *
 * @note This class is not real-time safe and is intended for offline rendering, file processing
 *       and batch inference. It can coexist with real-time InferenceHandler instances in the
 *       same process; both share the inference thread pool and backend model instances.
 */
class ANIRA_API OfflineInferenceHandler {
public:
    using JobId = OfflineJobId;                ///< See OfflineJobId
    using JobResult = OfflineJobResult;        ///< See OfflineJobResult
    using JobOptions = OfflineJobOptions;      ///< See OfflineJobOptions
    using DeliveryMode = OfflineDeliveryMode;  ///< See OfflineDeliveryMode
    static constexpr JobId k_invalid_job_id = k_invalid_offline_job_id;  ///< Invalid job id

    /// Completion callback type. Must not call submit()/wait()/poll() or destroy the handler.
    using JobCallback = std::function<void(const JobResult&)>;

    OfflineInferenceHandler() = delete;
    OfflineInferenceHandler(const OfflineInferenceHandler&) = delete;
    OfflineInferenceHandler& operator=(const OfflineInferenceHandler&) = delete;
    OfflineInferenceHandler(OfflineInferenceHandler&&) = delete;
    OfflineInferenceHandler& operator=(OfflineInferenceHandler&&) = delete;

    /**
     * @brief Constructs an offline handler with the default backend processors
     *
     * @param pp_processor Pre/post processor shared by all lanes. With num_parallel_jobs > 1 a
     * custom implementation must be re-entrant (pre_process/post_process must not write member
     * state).
     * @param inference_config Inference configuration (shared with the lanes' sessions)
     * @param context_config Optional context configuration (thread pool, wait strategy, ...)
     * @param num_parallel_jobs Number of parallel lanes. 0 = auto: 1 for session-exclusive
     * (stateful) configs, otherwise max(1, pool_threads / num_parallel_processors)
     * @param delivery_mode How completion callbacks are delivered (see OfflineDeliveryMode)
     */
    OfflineInferenceHandler(PrePostProcessor& pp_processor,
                            InferenceConfig& inference_config,
                            ContextConfig context_config = ContextConfig(),
                            unsigned int num_parallel_jobs = 0,
                            DeliveryMode delivery_mode = DeliveryMode::Immediate);

    /**
     * @brief Constructs an offline handler with a custom backend processor
     *
     * @param pp_processor Pre/post processor shared by all lanes (see other constructor)
     * @param inference_config Inference configuration (shared with the lanes' sessions)
     * @param custom_processor Custom backend processor used for InferenceBackend::CUSTOM
     * @param context_config Optional context configuration
     * @param num_parallel_jobs Number of parallel lanes (0 = auto)
     * @param delivery_mode How completion callbacks are delivered
     */
    OfflineInferenceHandler(PrePostProcessor& pp_processor,
                            InferenceConfig& inference_config,
                            BackendBase& custom_processor,
                            ContextConfig context_config = ContextConfig(),
                            unsigned int num_parallel_jobs = 0,
                            DeliveryMode delivery_mode = DeliveryMode::Immediate);

    /**
     * @brief Destructor: finishes the currently running jobs, fails all queued jobs
     *
     * Lane workers complete their in-flight jobs (the borrowed buffers stay owned by the jobs
     * until their callbacks ran), remaining queued jobs receive a callback with
     * m_success == false, then all worker threads are joined.
     */
    ~OfflineInferenceHandler();

    /**
     * @brief Prepares all lanes for processing; must be called before submitting jobs
     *
     * Resolves the lane count, creates the lanes' sessions with an internally chosen host
     * buffer size (num_parallel_processors * preprocess_input_size, which makes the buffer
     * adaptation zero and the pump's in-flight bound provably safe), enables non-real-time
     * mode and starts the lane worker threads.
     *
     * Takes no arguments by design: offline results are provably independent of the sample
     * rate (it only scales the struct-pool headroom and the latency bookkeeping, both
     * self-consistent for any rate), and independent of the HostConfig reference-tensor
     * choice (the internal chunk size makes the ratio anchor cancel out) — so the sessions
     * are prepared with an internal nominal rate and the first streamable input tensor as
     * the anchor.
     *
     * @note Requires at least one inference thread (see
     * InferenceHandler::get_num_inference_threads()); otherwise the blocking offline waits
     * could never complete and prepare() fails with an error log.
     */
    void prepare();

    /// Resolved number of parallel lanes (0 before prepare()).
    unsigned int get_num_parallel_jobs() const;

    /**
     * @brief Synchronously processes one job on the given lane (the pump core)
     *
     * Blocks the calling thread until the whole job is processed and returns the result by
     * value — no queue, no callback, no worker threads involved. Arguments are validated and
     * copied through the same internal path as submit() (build_job()), and the job runs
     * through the identical pump engine (run_job_on_lane()); submit() is exactly this engine
     * wrapped in the FIFO job queue, lane dispatch, and asynchronous completion delivery.
     *
     * Use cases: the WebAssembly wrapper's offline pump workers drive their lanes through
     * this method (no std::thread exists on WASM), and native callers that already own a
     * thread can use it for plain blocking semantics. Native applications normally use
     * submit() instead.
     *
     * @warning A lane has a single driver: this method must not race the handler's own
     * worker threads (or another process_job() call) for the same lane. On the web this
     * holds by construction (each pump worker is bound to one lane); natively call it only
     * on a lane the handler's workers are not serving.
     *
     * @param lane_index Lane to run the job on (< get_num_parallel_jobs())
     * @param input_data Input data as data[tensor][channel][sample]; nullptr entries and
     * num_input_samples == 0 for non-streamable tensors
     * @param num_input_samples Per input tensor: samples per channel to consume
     * @param output_data Output buffers as data[tensor][channel][sample]
     * @param output_capacity Per output tensor: samples per channel the buffer can hold
     * @param options Per-job options (non-streamable values, head trim, tail flush). Note
     * that the exclusive-job scheduling rule for per-job non-streamable values applies to
     * submit() only — synchronous callers must serialize such jobs themselves.
     * @return The job result; m_success is false if the arguments were invalid (m_job_id is
     * k_invalid_job_id; submit() assigns ids)
     */
    JobResult process_job(size_t lane_index,
                          const float* const* const* input_data,
                          const size_t* num_input_samples,
                          float* const* const* output_data,
                          const size_t* output_capacity,
                          const JobOptions& options = {});

#ifndef __EMSCRIPTEN__
    /**
     * @brief Submits a job (multi-tensor form); returns immediately
     *
     * Pointer conventions match InferenceHandler: data[tensor][channel][sample]. The tensor
     * and channel POINTER ARRAYS are copied at submit time; the SAMPLE BUFFERS they point to
     * are borrowed and must stay valid and untouched until the job's callback returns. The
     * per-tensor size arrays and the options are copied as well.
     *
     * @param input_data Input data per tensor (nullptr for non-streamable tensors)
     * @param num_input_samples Per input tensor: samples per channel (0 for non-streamable)
     * @param output_data Output buffers per tensor (nullptr for non-streamable tensors)
     * @param output_capacity Per output tensor: samples per channel available (0 for
     * non-streamable)
     * @param callback Invoked exactly once when the job completed (see OfflineDeliveryMode)
     * @param options Per-job options
     * @return The job id, or k_invalid_job_id if validation failed (no callback fires then)
     *
     * @warning Not real-time safe: allocates, locks a mutex, and notifies a condition
     * variable. Never call from an audio callback — if real-time code needs to trigger a
     * job, defer the submit() to a message/UI thread (e.g. via a lock-free flag or your
     * framework's async-message facility).
     */
    JobId submit(const float* const* const* input_data,
                 const size_t* num_input_samples,
                 float* const* const* output_data,
                 const size_t* output_capacity,
                 JobCallback callback,
                 JobOptions options = {});

    /**
     * @brief Submits a job (convenience form for one streamable input and output tensor)
     *
     * Usable when the config has exactly one streamable input and one streamable output tensor
     * (any number of non-streamable tensors alongside, served via JobOptions / JobResult).
     *
     * @param input_data Input audio as data[channel][sample]
     * @param num_input_samples Number of input samples per channel
     * @param output_data Output buffer as data[channel][sample]
     * @param output_capacity Number of samples per channel the output buffer can hold
     * @param callback Invoked exactly once when the job completed
     * @param options Per-job options
     * @return The job id, or k_invalid_job_id if validation failed
     *
     * @warning Not real-time safe (see the multi-tensor overload).
     */
    JobId submit(const float* const* input_data,
                 size_t num_input_samples,
                 float* const* output_data,
                 size_t output_capacity,
                 JobCallback callback,
                 JobOptions options = {});

    /**
     * @brief Blocks until the given job has been processed
     *
     * Returns immediately for unknown or already finished ids. In OfflineDeliveryMode::Polled
     * the callback may still be undelivered (it fires on the next poll()).
     *
     * @param job_id Id returned by submit()
     */
    void wait(JobId job_id);

    /// Blocks until all submitted jobs have been processed.
    void wait_all();

    /**
     * @brief Delivers completed jobs on the calling thread (OfflineDeliveryMode::Polled)
     *
     * Drains the completion queue and invokes each job's callback on the calling thread.
     * In OfflineDeliveryMode::Immediate this is a no-op.
     *
     * @return Number of callbacks delivered
     */
    size_t poll();

    /**
     * @brief Like poll(), but blocks up to timeout for at least one completed job
     *
     * @param timeout Maximum time to wait for a completion
     * @return Number of callbacks delivered
     */
    size_t poll_wait(std::chrono::milliseconds timeout);
#endif  // __EMSCRIPTEN__

    /**
     * @brief Dequeues one completed job result from the completion queue
     *
     * Platform-wrapper access (used by the WebAssembly bindings to drain completions on the
     * main thread). Native applications use poll() instead, which also invokes the callbacks.
     *
     * @param result Filled with the dequeued result on success
     * @return True if a result was dequeued
     */
    bool try_dequeue_completed(JobResult& result);

    /**
     * @brief Enqueues a completed job result into the completion queue
     *
     * Platform-wrapper access (used by the WebAssembly bindings: the offline pump worker
     * enqueues the result of process_job() so the main thread can drain it).
     *
     * @param result The result to enqueue
     */
    void enqueue_completed(JobResult&& result);

    /// Sets the inference backend for all lanes.
    void set_inference_backend(InferenceBackend inference_backend);

    /// Currently active inference backend.
    InferenceBackend get_inference_backend() const;

    /**
     * @brief Model-internal latency in samples for one output tensor
     *
     * Offline sessions carry no real-time latency bookkeeping, so this is exactly the
     * config's internal_model_latency. It is also the default head-trim, i.e. the number of
     * leading samples discarded so the output is input-aligned.
     *
     * @param tensor_index Output tensor index
     * @return Latency in samples (0 before prepare())
     */
    unsigned int get_latency(size_t tensor_index = 0) const;

    /// Latency of the offline session in samples for each output tensor (empty before prepare()).
    std::vector<unsigned int> get_latency_vector() const;

    /**
     * @brief Expected input-aligned output length per output tensor for a given input length
     *
     * Computed from the per-tensor size ratios: ceil(num_input_samples * out_size / in_size of
     * the reference tensor). 0 for non-streamable tensors.
     *
     * @param num_input_samples Input length (samples per channel) of the reference tensor
     * @return Expected output sample counts per output tensor
     */
    std::vector<size_t> get_expected_output_samples(size_t num_input_samples) const;

private:
    struct Lane;  ///< One session + pump state (definition in the source file)

    struct Job {
        JobId m_id = k_invalid_offline_job_id;
        // The tensor/channel pointer arrays are copied at submit time; only the sample
        // buffers they point to are borrowed from the caller.
        std::vector<std::vector<const float*>> m_input_channels;  ///< [tensor][channel]
        std::vector<std::vector<float*>> m_output_channels;       ///< [tensor][channel]
        std::vector<size_t> m_num_input_samples;
        std::vector<size_t> m_output_capacity;
        JobCallback m_callback;
        JobOptions m_options;
        bool m_exclusive = false;  ///< Carries non-streamable inputs -> runs alone
    };

    struct CompletedJob {
        JobResult m_result;
        JobCallback m_callback;
    };

    void create_lanes_and_prepare();  // anchor tensor: m_ref_tensor_index
    /**
     * @brief Validates raw job arguments and populates a Job (shared by submit() and
     * process_job())
     *
     * Checks head_trim / non-streamable option consistency, copies both pointer-array levels
     * and the per-tensor size arrays, and derives the exclusive flag from per-job
     * non-streamable values.
     *
     * @return true on success, false if the arguments are invalid (logged)
     */
    bool build_job(Job& job,
                   const float* const* const* input_data,
                   const size_t* num_input_samples,
                   float* const* const* output_data,
                   const size_t* output_capacity,
                   JobOptions options);

    void copy_channel_pointers(Job& job,
                               const float* const* const* input_data,
                               float* const* const* output_data);
    JobResult run_job_on_lane(Lane& lane, const Job& job);
    void deliver(CompletedJob&& completed);

    PrePostProcessor& m_pp_processor;
    InferenceConfig& m_inference_config;
    BackendBase* m_custom_processor = nullptr;
    ContextConfig m_context_config;
    unsigned int m_requested_parallel_jobs;
    DeliveryMode m_delivery_mode;

    std::vector<std::unique_ptr<Lane>> m_lanes;
    std::atomic<InferenceBackend> m_backend{InferenceBackend::CUSTOM};
    bool m_prepared = false;

    // Cached geometry (identical across lanes, computed in prepare())
    size_t m_ref_tensor_index = 0;
    unsigned int m_in_flight_bound = 1;  ///< P: max chunks in flight per lane
    std::vector<size_t> m_in_sizes;      ///< preprocess_input_size per input tensor
    std::vector<size_t> m_out_sizes;     ///< postprocess_output_size per output tensor
    std::vector<size_t> m_in_channels;
    std::vector<size_t> m_out_channels;
    std::vector<unsigned int> m_latency;

#ifdef __EMSCRIPTEN__
    moodycamel::ConcurrentQueue<CompletedJob> m_completed;
#else
    moodycamel::BlockingConcurrentQueue<CompletedJob> m_completed;

    void worker_loop(size_t lane_index);
    void fail_remaining_jobs();

    std::deque<Job> m_job_queue;
    std::mutex m_mutex;
    std::condition_variable m_queue_cv;       ///< Signals new jobs / shutdown to lane workers
    std::condition_variable m_completion_cv;  ///< Signals processed jobs to wait()/wait_all()
    std::unordered_set<JobId> m_unfinished;   ///< Queued + running job ids
    std::atomic<JobId> m_next_job_id{1};
    unsigned int m_active_jobs = 0;   ///< Jobs currently running on lanes (guarded by m_mutex)
    bool m_exclusive_active = false;  ///< An exclusive job is running (guarded by m_mutex)
    std::atomic<bool> m_should_exit{false};
    std::vector<std::thread> m_workers;
#endif

#if DOXYGEN
    // Since Doxygen does not find classes nested in std::vector / std::deque
    Lane* __doxygen_force_0;          ///< Placeholder for Doxygen documentation
    Job* __doxygen_force_1;           ///< Placeholder for Doxygen documentation
    CompletedJob* __doxygen_force_2;  ///< Placeholder for Doxygen documentation
#endif
};

}  // namespace anira

#endif  // ANIRA_OFFLINEINFERENCEHANDLER_H
