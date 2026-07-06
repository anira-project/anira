#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/OfflineInferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Context.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#ifndef __EMSCRIPTEN__
#include <deque>
#include <mutex>
#endif

namespace anira {

namespace {
size_t ceil_div(size_t numerator, size_t denominator) {
    return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}
}  // namespace

/**
 * @brief One parallel lane: an own session plus the pump's per-lane working memory
 *
 * The manager owns the lane's SessionElement (ring buffers, structs, producer token). The
 * staging buffers hold zero-padded partial/flush input chunks, the scratch buffers receive
 * trimmed/surplus output samples, and the pointer arrays are the persistent argument
 * structures for the multi-tensor push_data()/pop_data() calls.
 */
struct OfflineInferenceHandler::Lane {
    std::unique_ptr<InferenceManager> m_manager;

    std::vector<std::vector<float>> m_staging_data;        ///< [in tensor] channels * chunk size
    std::vector<std::vector<const float*>> m_in_channels;  ///< [in tensor][channel] chunk pointers
    std::vector<std::vector<float>> m_scratch_data;        ///< [out tensor] channels * capacity
    std::vector<size_t> m_scratch_capacity;                ///< [out tensor] samples per channel
    std::vector<std::vector<float*>> m_out_channels;       ///< [out tensor][channel] pop pointers

    std::vector<const float* const*> m_push_ptrs;  ///< [in tensor] -> channel pointer array
    std::vector<size_t> m_push_counts;             ///< [in tensor] samples per channel
    std::vector<float* const*> m_pop_ptrs;         ///< [out tensor] -> channel pointer array
    std::vector<size_t> m_pop_counts;              ///< [out tensor] samples per channel
};

OfflineInferenceHandler::OfflineInferenceHandler(PrePostProcessor& pp_processor,
                                                 InferenceConfig& inference_config,
                                                 ContextConfig context_config,
                                                 unsigned int num_parallel_jobs,
                                                 DeliveryMode delivery_mode)
    : m_pp_processor(pp_processor)
    , m_inference_config(inference_config)
    , m_context_config(std::move(context_config))
    , m_requested_parallel_jobs(num_parallel_jobs)
    , m_delivery_mode(delivery_mode) {
    // Apply the requested log level immediately (InferenceHandler gets this via its session
    // creation at construction; the offline lanes are only created in prepare()). The Context
    // re-applies/merges it at lane creation; most verbose wins, matching
    // Context::get_instance().
    set_log_level(std::min(get_log_level(), m_context_config.m_log_level));
#ifdef __EMSCRIPTEN__
    if (m_delivery_mode == DeliveryMode::Immediate) {
        LOG_WARNING << "[WARNING] DeliveryMode::Immediate is not available on WebAssembly - "
                       "completion is always delivered via the completion queue (polled). "
                       "Using DeliveryMode::Polled."
                    << '\n';
        m_delivery_mode = DeliveryMode::Polled;
    }
#endif
    if (m_inference_config.m_blocking_ratio > 0.f) {
        LOG_WARNING << "[WARNING] OfflineInferenceHandler is designed for configurations with "
                       "blocking_ratio == 0 (the atomic completion path). A blocking_ratio > 0 "
                       "works but adds semaphore bookkeeping that has no benefit offline."
                    << '\n';
    }
}

OfflineInferenceHandler::OfflineInferenceHandler(PrePostProcessor& pp_processor,
                                                 InferenceConfig& inference_config,
                                                 BackendBase& custom_processor,
                                                 ContextConfig context_config,
                                                 unsigned int num_parallel_jobs,
                                                 DeliveryMode delivery_mode)
    : OfflineInferenceHandler(pp_processor,
                              inference_config,
                              context_config,
                              num_parallel_jobs,
                              delivery_mode) {
    m_custom_processor = &custom_processor;
}

OfflineInferenceHandler::~OfflineInferenceHandler() {
#ifndef __EMSCRIPTEN__
    {
        std::scoped_lock const lock(m_mutex);
        m_should_exit.store(true, std::memory_order::release);
    }
    m_queue_cv.notify_all();
    for (auto& worker : m_workers) {
        if (worker.joinable()) { worker.join(); }
    }
    fail_remaining_jobs();
#endif
}

void OfflineInferenceHandler::prepare() {
    // The HostConfig needs SOME streamable reference tensor, but the choice is unobservable:
    // with the internal chunk size P * in_ref, every tensor's relative buffer size resolves
    // to P * its own size regardless of the anchor. Use the first streamable input.
    auto tensor_index = static_cast<size_t>(-1);
    const auto& in_sizes = m_inference_config.get_preprocess_input_size();
    for (size_t t = 0; t < in_sizes.size(); ++t) {
        if (in_sizes[t] > 0) {
            tensor_index = t;
            break;
        }
    }
#ifndef __EMSCRIPTEN__
    if (m_prepared) {
        // Stop and join the lane workers before rebuilding the lanes, then fail
        // whatever was still queued (jobs must not silently survive a re-prepare).
        {
            std::scoped_lock const lock(m_mutex);
            m_should_exit.store(true, std::memory_order::release);
        }
        m_queue_cv.notify_all();
        for (auto& worker : m_workers) {
            if (worker.joinable()) { worker.join(); }
        }
        m_workers.clear();
        fail_remaining_jobs();
        m_should_exit.store(false, std::memory_order::release);
    }
#endif
    m_prepared = false;
    m_lanes.clear();

    if (tensor_index >= m_inference_config.get_tensor_input_shape().size()) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler::prepare: the InferenceConfig has no "
                     "streamable input tensor (all preprocess_input_size entries are 0) — "
                     "there is no sample stream to process offline."
                  << '\n';
        return;
    }
    m_ref_tensor_index = tensor_index;

    create_lanes_and_prepare();
    if (m_lanes.empty()) { return; }

    m_prepared = true;

#ifndef __EMSCRIPTEN__
    for (size_t i = 0; i < m_lanes.size(); ++i) {
        m_workers.emplace_back([this, i] { worker_loop(i); });
    }
#endif
}

void OfflineInferenceHandler::create_lanes_and_prepare() {
    // The lanes are prepared with EXACTLY what offline processing needs, bypassing the
    // real-time latency/headroom heuristics entirely:
    //  - custom_latency = internal_model_latency: no latency prefill zeros are pushed,
    //    get_latency() reports only the model-internal latency, and the default head-trim
    //    trims exactly that. The RT inference-caused latency does not exist offline (the
    //    pump blocks until every chunk completed).
    //  - num_structs_override = P (the pump's in-flight bound): each push_data() claims one
    //    struct and every batch (<= P chunks) is fully drained before the next one starts,
    //    so exactly P structs suffice. This also decouples the allocation from
    //    max_inference_time, which is a real-time scheduling hint and is IGNORED offline.
    // With both overrides the sample rate influences nothing; a nominal value merely keeps
    // the HostConfig well-formed.
    constexpr float k_nominal_sample_rate = 48000.f;
    float const sample_rate = k_nominal_sample_rate;

    if (m_inference_config.m_max_inference_time > 0.f) {
        LOG_INFO << "[INFO] OfflineInferenceHandler: InferenceConfig::m_max_inference_time is a "
                    "real-time scheduling hint and is ignored for offline processing."
                 << '\n';
    }
    size_t const num_in = m_inference_config.get_tensor_input_shape().size();
    size_t const num_out = m_inference_config.get_tensor_output_shape().size();

    // The first lane creates (or joins) the Context, which also clamps
    // m_num_parallel_processors to the pool size - read P only afterwards.
    auto first_manager = std::make_unique<InferenceManager>(m_pp_processor,
                                                            m_inference_config,
                                                            m_custom_processor,
                                                            m_context_config);

    m_in_flight_bound = m_inference_config.m_session_exclusive_processor
                            ? 1
                            : std::max(1u, m_inference_config.m_num_parallel_processors);

    size_t const in_ref = m_inference_config.get_preprocess_input_size()[m_ref_tensor_index];
    HostConfig const host_config(static_cast<float>(m_in_flight_bound * in_ref),
                                 sample_rate,
                                 false,
                                 m_ref_tensor_index);

    std::vector<long> custom_latency(num_out);
    const auto& internal_latency = m_inference_config.get_internal_model_latency();
    for (size_t i = 0; i < num_out; ++i) {
        custom_latency[i] = static_cast<long>(internal_latency[i]);
    }
    first_manager->prepare(host_config, custom_latency, m_in_flight_bound);

    unsigned int const num_threads = Context::get_num_inference_threads();
    if (num_threads == 0) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler::prepare: no inference threads are "
                     "available, offline processing could never complete. Configure "
                     "ContextConfig::m_num_threads > 0 (web: spin up an inference worker) first."
                  << '\n';
        return;
    }
    first_manager->set_non_realtime(true);
    first_manager->set_backend(m_backend.load(std::memory_order_relaxed));

    unsigned int num_lanes = m_requested_parallel_jobs;
    if (num_lanes == 0) {
        num_lanes = m_inference_config.m_session_exclusive_processor
                        ? 1
                        : std::max(1u, num_threads / m_in_flight_bound);
    }
    if (m_inference_config.m_session_exclusive_processor && num_lanes > 1) {
        LOG_WARNING << "[WARNING] OfflineInferenceHandler: " << num_lanes
                    << " lanes requested for a session-exclusive (stateful) configuration. Each "
                       "lane instantiates its own backend processor and keeps independent model "
                       "state; job-to-lane assignment is nondeterministic."
                    << '\n';
    }

    // Cache the geometry (identical across lanes)
    m_in_sizes = m_inference_config.get_preprocess_input_size();
    m_out_sizes = m_inference_config.get_postprocess_output_size();
    m_in_channels = m_inference_config.get_preprocess_input_channels();
    m_out_channels = m_inference_config.get_postprocess_output_channels();
    m_latency = first_manager->get_latency();
    // The pump math below relies on this invariant (no prefill exists): the lanes were
    // prepared with custom_latency == internal_model_latency, so the session latency IS the
    // model-internal latency and the receive buffers start empty.
    for (size_t i = 0; i < num_out; ++i) {
        assert(m_latency[i] == m_inference_config.get_internal_model_latency()[i]);
    }

    for (unsigned int lane_index = 0; lane_index < num_lanes; ++lane_index) {
        auto lane = std::make_unique<Lane>();
        if (lane_index == 0) {
            lane->m_manager = std::move(first_manager);
        } else {
            lane->m_manager = std::make_unique<InferenceManager>(m_pp_processor,
                                                                 m_inference_config,
                                                                 m_custom_processor,
                                                                 m_context_config);
            lane->m_manager->prepare(host_config, custom_latency, m_in_flight_bound);
            lane->m_manager->set_non_realtime(true);
            lane->m_manager->set_backend(m_backend.load(std::memory_order_relaxed));
        }

        // The pump keeps at most m_in_flight_bound chunks in flight; the session
        // allocates at least twice that, so the drop path is unreachable.
        assert(m_in_flight_bound <= lane->m_manager->get_num_structs());

        lane->m_staging_data.resize(num_in);
        lane->m_in_channels.resize(num_in);
        lane->m_push_ptrs.resize(num_in);
        lane->m_push_counts.resize(num_in, 0);
        for (size_t t = 0; t < num_in; ++t) {
            if (m_in_sizes[t] > 0) {
                lane->m_staging_data[t].resize(m_in_channels[t] * m_in_sizes[t], 0.f);
                lane->m_in_channels[t].resize(m_in_channels[t], nullptr);
                lane->m_push_ptrs[t] = lane->m_in_channels[t].data();
            }
        }
        lane->m_scratch_data.resize(num_out);
        lane->m_scratch_capacity.resize(num_out, 0);
        lane->m_out_channels.resize(num_out);
        lane->m_pop_ptrs.resize(num_out);
        lane->m_pop_counts.resize(num_out, 0);
        for (size_t i = 0; i < num_out; ++i) {
            if (m_out_sizes[i] > 0) {
                lane->m_scratch_capacity[i] =
                    m_latency[i] + (m_in_flight_bound * m_out_sizes[i]) + m_out_sizes[i];
                lane->m_scratch_data[i].resize(m_out_channels[i] * lane->m_scratch_capacity[i]);
                lane->m_out_channels[i].resize(m_out_channels[i], nullptr);
                lane->m_pop_ptrs[i] = lane->m_out_channels[i].data();
            }
        }
        m_lanes.emplace_back(std::move(lane));
    }
}

unsigned int OfflineInferenceHandler::get_num_parallel_jobs() const {
    return static_cast<unsigned int>(m_lanes.size());
}

OfflineInferenceHandler::JobResult OfflineInferenceHandler::process_job(
    size_t lane_index,
    const float* const* const* input_data,
    const size_t* num_input_samples,
    float* const* const* output_data,
    const size_t* output_capacity,
    const JobOptions& options) {
    JobResult result;
    if (!m_prepared || lane_index >= m_lanes.size()) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler::process_job: handler not prepared or "
                     "lane index out of range."
                  << '\n';
        return result;
    }
    Job job;
    if (!build_job(job, input_data, num_input_samples, output_data, output_capacity, options)) {
        return result;
    }
    result = run_job_on_lane(*m_lanes[lane_index], job);
    result.m_success = true;
    return result;
}

bool OfflineInferenceHandler::build_job(Job& job,
                                        const float* const* const* input_data,
                                        const size_t* num_input_samples,
                                        float* const* const* output_data,
                                        const size_t* output_capacity,
                                        JobOptions options) {
    size_t const num_in = m_inference_config.get_tensor_input_shape().size();
    size_t const num_out = m_inference_config.get_tensor_output_shape().size();

    if (!options.m_head_trim.empty() && options.m_head_trim.size() != num_out) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler: head_trim must be empty or have one "
                     "entry per output tensor."
                  << '\n';
        return false;
    }
    if (!options.m_non_streamable_inputs.empty() &&
        options.m_non_streamable_inputs.size() != num_in) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler: non_streamable_inputs must be empty or "
                     "have one entry per input tensor."
                  << '\n';
        return false;
    }
    bool exclusive = false;
    for (size_t t = 0; t < options.m_non_streamable_inputs.size(); ++t) {
        if (options.m_non_streamable_inputs[t].empty()) { continue; }
        if (m_in_sizes[t] > 0) {
            LOG_ERROR << "[ERROR] OfflineInferenceHandler: non-streamable values given for "
                         "streamable input tensor "
                      << t << "." << '\n';
            return false;
        }
        exclusive = true;
    }

    copy_channel_pointers(job, input_data, output_data);
    job.m_num_input_samples.assign(num_input_samples, num_input_samples + num_in);
    job.m_output_capacity.assign(output_capacity, output_capacity + num_out);
    job.m_options = std::move(options);
    job.m_exclusive = exclusive;
    return true;
}

void OfflineInferenceHandler::copy_channel_pointers(Job& job,
                                                    const float* const* const* input_data,
                                                    float* const* const* output_data) {
    size_t const num_in = m_in_sizes.size();
    size_t const num_out = m_out_sizes.size();
    job.m_input_channels.resize(num_in);
    job.m_output_channels.resize(num_out);
    for (size_t t = 0; t < num_in; ++t) {
        if (m_in_sizes[t] > 0 && input_data[t] != nullptr) {
            job.m_input_channels[t].assign(input_data[t], input_data[t] + m_in_channels[t]);
        }
    }
    for (size_t i = 0; i < num_out; ++i) {
        if (m_out_sizes[i] > 0 && output_data[i] != nullptr) {
            job.m_output_channels[i].assign(output_data[i], output_data[i] + m_out_channels[i]);
        }
    }
}

OfflineInferenceHandler::JobResult OfflineInferenceHandler::run_job_on_lane(Lane& lane,
                                                                            const Job& job) {
    size_t const num_in = m_in_sizes.size();
    size_t const num_out = m_out_sizes.size();
    size_t const in_ref = m_in_sizes[m_ref_tensor_index];
    size_t const reference_length = job.m_num_input_samples[m_ref_tensor_index];
    InferenceManager& manager = *lane.m_manager;

    JobResult result;
    result.m_job_id = job.m_id;
    result.m_num_output_samples_written.assign(num_out, 0);
    result.m_non_streamable_outputs.resize(num_out);

    // Apply per-job non-streamable input values (snapshot-and-restore: a job override is
    // temporary, the handler-wide baseline set directly on the PrePostProcessor stays
    // authoritative for later jobs).
    std::vector<std::vector<float>> snapshot(num_in);
    for (size_t t = 0; t < num_in && t < job.m_options.m_non_streamable_inputs.size(); ++t) {
        const auto& values = job.m_options.m_non_streamable_inputs[t];
        if (values.empty()) { continue; }
        size_t const tensor_size = m_inference_config.get_tensor_input_size()[t];
        snapshot[t].resize(tensor_size);
        for (size_t j = 0; j < tensor_size; ++j) {
            snapshot[t][j] = m_pp_processor.get_input(t, j);
            if (j < values.size()) { m_pp_processor.set_input(values[j], t, j); }
        }
    }

    // Per-job pump math (see the offline-processing documentation): n_real real chunks cover
    // the input (last one zero-padded); trim_i leading samples are discarded so the output is
    // input-aligned; with tail flush enough zero chunks follow so the last latency-delayed
    // payload samples emerge.
    size_t const n_real = ceil_div(reference_length, in_ref);
    std::vector<size_t> trim(num_out, 0);
    std::vector<size_t> written(num_out, 0);
    size_t n_total = n_real;
    for (size_t i = 0; i < num_out; ++i) {
        if (m_out_sizes[i] == 0) { continue; }
        long const custom_trim =
            i < job.m_options.m_head_trim.size() ? job.m_options.m_head_trim[i] : -1;
        trim[i] = custom_trim < 0 ? m_latency[i] : static_cast<size_t>(custom_trim);
        size_t const expected = ceil_div(reference_length * m_out_sizes[i], in_ref);
        written[i] = std::min(expected, job.m_output_capacity[i]);
        if (written[i] < expected) {
            LOG_WARNING << "[WARNING] OfflineInferenceHandler: output capacity for tensor " << i
                        << " (" << job.m_output_capacity[i] << ") is smaller than the expected "
                        << expected << " samples; output will be truncated." << '\n';
        }
        if (job.m_options.m_tail_flush) {
            n_total = std::max(n_total, ceil_div(trim[i] + written[i], m_out_sizes[i]));
        } else {
            size_t const produced = n_real * m_out_sizes[i];
            written[i] = std::min(written[i], produced > trim[i] ? produced - trim[i] : 0);
        }
    }

    std::vector<size_t> popped(num_out, 0);
    size_t submitted = 0;
    while (submitted < n_total) {
        size_t const batch = std::min<size_t>(m_in_flight_bound, n_total - submitted);

        // ---- fill: one chunk per push_data call (claims exactly one struct each) ----
        for (size_t b = 0; b < batch; ++b) {
            size_t const chunk = submitted;
            for (size_t t = 0; t < num_in; ++t) {
                if (m_in_sizes[t] == 0) {
                    lane.m_push_counts[t] = 0;
                    continue;
                }
                size_t const start = chunk * m_in_sizes[t];
                size_t const available = job.m_num_input_samples[t];
                if (start + m_in_sizes[t] <= available) {
                    // Full in-range chunk: point directly at the user's memory (zero-copy)
                    for (size_t ch = 0; ch < m_in_channels[t]; ++ch) {
                        lane.m_in_channels[t][ch] = job.m_input_channels[t][ch] + start;
                    }
                } else {
                    // Partial or flush chunk: zero-pad through the staging buffer
                    for (size_t ch = 0; ch < m_in_channels[t]; ++ch) {
                        float* staging = lane.m_staging_data[t].data() + (ch * m_in_sizes[t]);
                        size_t const in_range =
                            start < available ? std::min(m_in_sizes[t], available - start) : 0;
                        for (size_t s = 0; s < in_range; ++s) {
                            staging[s] = job.m_input_channels[t][ch][start + s];
                        }
                        std::fill(staging + in_range, staging + m_in_sizes[t], 0.f);
                        lane.m_in_channels[t][ch] = staging;
                    }
                }
                lane.m_push_counts[t] = m_in_sizes[t];
            }
            manager.push_data(lane.m_push_ptrs.data(), lane.m_push_counts.data());
            ++submitted;
        }

        // ---- drain: the first pop_data call blocks (non-real-time mode) until every
        // in-flight inference completed, then the requested counts are deterministic ----
        for (int phase = 0; phase < 3; ++phase) {
            bool any = phase == 0;  // phase 0 always runs: it triggers the blocking drain
            for (size_t i = 0; i < num_out; ++i) {
                lane.m_pop_counts[i] = 0;
                if (m_out_sizes[i] == 0) { continue; }
                size_t const available = (submitted * m_out_sizes[i]) - popped[i];
                size_t take = 0;
                if (phase == 0) {  // trim -> scratch
                    take = trim[i] > popped[i] ? std::min(available, trim[i] - popped[i]) : 0;
                    for (size_t ch = 0; ch < m_out_channels[i]; ++ch) {
                        lane.m_out_channels[i][ch] =
                            lane.m_scratch_data[i].data() + (ch * lane.m_scratch_capacity[i]);
                    }
                } else if (phase == 1) {  // payload -> user output at the aligned offset
                    size_t const payload_end = trim[i] + written[i];
                    take =
                        payload_end > popped[i] ? std::min(available, payload_end - popped[i]) : 0;
                    if (take > 0) {
                        for (size_t ch = 0; ch < m_out_channels[i]; ++ch) {
                            lane.m_out_channels[i][ch] =
                                job.m_output_channels[i][ch] + (popped[i] - trim[i]);
                        }
                    }
                } else {  // surplus -> scratch (discarded)
                    take = available;
                    for (size_t ch = 0; ch < m_out_channels[i]; ++ch) {
                        lane.m_out_channels[i][ch] =
                            lane.m_scratch_data[i].data() + (ch * lane.m_scratch_capacity[i]);
                    }
                }
                lane.m_pop_counts[i] = take;
                any = any || take > 0;
            }
            if (!any) { continue; }
            manager.pop_data(lane.m_pop_ptrs.data(), lane.m_pop_counts.data());
            for (size_t i = 0; i < num_out; ++i) {
                popped[i] += lane.m_pop_counts[i];
                if (phase == 1) { result.m_num_output_samples_written[i] += lane.m_pop_counts[i]; }
            }
        }
    }

    // Non-streamable outputs: capture the values of the last processed chunk
    for (size_t i = 0; i < num_out; ++i) {
        if (m_out_sizes[i] > 0) { continue; }
        size_t const tensor_size = m_inference_config.get_tensor_output_size()[i];
        result.m_non_streamable_outputs[i].resize(tensor_size);
        for (size_t j = 0; j < tensor_size; ++j) {
            result.m_non_streamable_outputs[i][j] = m_pp_processor.get_output(i, j);
        }
    }

    // Jobs are independent signals: clear the stream state (ring buffers incl. past-sample
    // history, structs) so the next job starts with zero left-context.
    manager.reset();

    // Restore the handler-wide non-streamable baseline
    for (size_t t = 0; t < num_in; ++t) {
        for (size_t j = 0; j < snapshot[t].size(); ++j) {
            m_pp_processor.set_input(snapshot[t][j], t, j);
        }
    }

    return result;
}

void OfflineInferenceHandler::enqueue_completed(JobResult&& result) {
    m_completed.enqueue(CompletedJob{.m_result = std::move(result), .m_callback = JobCallback{}});
}

bool OfflineInferenceHandler::try_dequeue_completed(JobResult& result) {
    CompletedJob completed;
    if (!m_completed.try_dequeue(completed)) { return false; }
    result = std::move(completed.m_result);
    return true;
}

void OfflineInferenceHandler::deliver(CompletedJob&& completed) {
    if (m_delivery_mode == DeliveryMode::Immediate) {
        if (completed.m_callback) { completed.m_callback(completed.m_result); }
    } else {
        m_completed.enqueue(std::move(completed));
    }
}

void OfflineInferenceHandler::set_inference_backend(InferenceBackend inference_backend) {
    m_backend.store(inference_backend, std::memory_order_relaxed);
    for (auto& lane : m_lanes) { lane->m_manager->set_backend(inference_backend); }
}

InferenceBackend OfflineInferenceHandler::get_inference_backend() const {
    return m_backend.load(std::memory_order_relaxed);
}

unsigned int OfflineInferenceHandler::get_latency(size_t tensor_index) const {
    return tensor_index < m_latency.size() ? m_latency[tensor_index] : 0;
}

std::vector<unsigned int> OfflineInferenceHandler::get_latency_vector() const {
    return m_latency;
}

std::vector<size_t> OfflineInferenceHandler::get_expected_output_samples(
    size_t num_input_samples) const {
    std::vector<size_t> expected(m_out_sizes.size(), 0);
    if (m_in_sizes.empty()) { return expected; }
    size_t const in_ref = m_in_sizes[m_ref_tensor_index];
    for (size_t i = 0; i < m_out_sizes.size(); ++i) {
        if (m_out_sizes[i] > 0) {
            expected[i] = ceil_div(num_input_samples * m_out_sizes[i], in_ref);
        }
    }
    return expected;
}

#ifndef __EMSCRIPTEN__

OfflineInferenceHandler::JobId OfflineInferenceHandler::submit(
    const float* const* const* input_data,
    const size_t* num_input_samples,
    float* const* const* output_data,
    const size_t* output_capacity,
    JobCallback callback,
    JobOptions options) {
    if (!m_prepared) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler::submit called before prepare()." << '\n';
        return k_invalid_job_id;
    }
    Job job;
    if (!build_job(job,
                   input_data,
                   num_input_samples,
                   output_data,
                   output_capacity,
                   std::move(options))) {
        return k_invalid_job_id;
    }
    job.m_id = m_next_job_id.fetch_add(1);
    job.m_callback = std::move(callback);

    JobId const id = job.m_id;
    {
        std::scoped_lock const lock(m_mutex);
        m_unfinished.insert(id);
        m_job_queue.push_back(std::move(job));
    }
    m_queue_cv.notify_all();
    return id;
}

OfflineInferenceHandler::JobId OfflineInferenceHandler::submit(const float* const* input_data,
                                                               size_t num_input_samples,
                                                               float* const* output_data,
                                                               size_t output_capacity,
                                                               JobCallback callback,
                                                               JobOptions options) {
    size_t const num_in = m_inference_config.get_tensor_input_shape().size();
    size_t const num_out = m_inference_config.get_tensor_output_shape().size();

    size_t streamable_in = 0;
    size_t streamable_out = 0;
    size_t in_index = 0;
    size_t out_index = 0;
    for (size_t t = 0; t < num_in; ++t) {
        if (m_inference_config.get_preprocess_input_size()[t] > 0) {
            ++streamable_in;
            in_index = t;
        }
    }
    for (size_t i = 0; i < num_out; ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            ++streamable_out;
            out_index = i;
        }
    }
    if (streamable_in != 1 || streamable_out != 1) {
        LOG_ERROR << "[ERROR] OfflineInferenceHandler::submit (single-tensor overload) requires "
                     "exactly one streamable input and output tensor; use the multi-tensor "
                     "overload."
                  << '\n';
        return k_invalid_job_id;
    }

    std::vector<const float* const*> input_ptrs(num_in, nullptr);
    std::vector<float* const*> output_ptrs(num_out, nullptr);
    std::vector<size_t> input_counts(num_in, 0);
    std::vector<size_t> output_counts(num_out, 0);
    input_ptrs[in_index] = input_data;
    input_counts[in_index] = num_input_samples;
    output_ptrs[out_index] = output_data;
    output_counts[out_index] = output_capacity;
    return submit(input_ptrs.data(),
                  input_counts.data(),
                  output_ptrs.data(),
                  output_counts.data(),
                  std::move(callback),
                  std::move(options));
}

void OfflineInferenceHandler::wait(JobId job_id) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_completion_cv.wait(lock, [&] { return m_unfinished.count(job_id) == 0; });
}

void OfflineInferenceHandler::wait_all() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_completion_cv.wait(lock, [&] { return m_unfinished.empty(); });
}

size_t OfflineInferenceHandler::poll() {
    size_t delivered = 0;
    CompletedJob completed;
    while (m_completed.try_dequeue(completed)) {
        if (completed.m_callback) { completed.m_callback(completed.m_result); }
        ++delivered;
    }
    return delivered;
}

size_t OfflineInferenceHandler::poll_wait(std::chrono::milliseconds timeout) {
    CompletedJob completed;
    if (!m_completed.wait_dequeue_timed(completed, timeout)) { return 0; }
    if (completed.m_callback) { completed.m_callback(completed.m_result); }
    return 1 + poll();
}

void OfflineInferenceHandler::worker_loop(size_t lane_index) {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_queue_cv.wait(lock, [&] {
                if (m_should_exit.load(std::memory_order::acquire)) { return true; }
                if (m_job_queue.empty() || m_exclusive_active) { return false; }
                // An exclusive job (per-job non-streamable values live in the shared
                // PrePostProcessor) may only start once every lane is idle.
                return !m_job_queue.front().m_exclusive || m_active_jobs == 0;
            });
            if (m_should_exit.load(std::memory_order::acquire)) { return; }
            job = std::move(m_job_queue.front());
            m_job_queue.pop_front();
            ++m_active_jobs;
            if (job.m_exclusive) { m_exclusive_active = true; }
        }

        JobResult result = run_job_on_lane(*m_lanes[lane_index], job);
        result.m_success = true;

        deliver(
            CompletedJob{.m_result = std::move(result), .m_callback = std::move(job.m_callback)});

        {
            std::scoped_lock const lock(m_mutex);
            --m_active_jobs;
            if (job.m_exclusive) { m_exclusive_active = false; }
            m_unfinished.erase(job.m_id);
        }
        m_completion_cv.notify_all();
        m_queue_cv.notify_all();
    }
}

void OfflineInferenceHandler::fail_remaining_jobs() {
    std::deque<Job> remaining;
    {
        std::scoped_lock const lock(m_mutex);
        remaining.swap(m_job_queue);
    }
    for (auto& job : remaining) {
        JobResult result;
        result.m_job_id = job.m_id;
        result.m_success = false;
        if (job.m_callback) { job.m_callback(result); }
        {
            std::scoped_lock const lock(m_mutex);
            m_unfinished.erase(job.m_id);
        }
    }
    m_completion_cv.notify_all();
}

#endif  // __EMSCRIPTEN__

}  // namespace anira
