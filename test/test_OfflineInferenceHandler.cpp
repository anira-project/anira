#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/OfflineInferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

namespace {

constexpr size_t k_chunk_size = 64;
constexpr size_t k_overlap_context = 32;

/**
 * Identity model: one streamable mono tensor in and out (64 samples), the backend copies the
 * input tensor to the output tensor. Offline output must equal the input sample-exactly after
 * the default head-trim.
 */
InferenceConfig make_identity_config() {
    return InferenceConfig({},
                           {{{{1, 1, (int)k_chunk_size}}, {{1, 1, (int)k_chunk_size}}}},
                           ProcessingSpec({1}, {1}, {k_chunk_size}, {k_chunk_size}),
                           5.f,
                           0,
                           false,
                           0.f,
                           2);
}

class IdentityBackend : public BackendBase {
public:
    using BackendBase::BackendBase;
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        const float* read_ptr = input[0].get_read_pointer(0);
        float* write_ptr = output[0].get_write_pointer(0);
        for (size_t i = 0; i < m_inference_config.get_tensor_output_size()[0]; ++i) {
            write_ptr[i] = read_ptr[i];
        }
    }
};

/**
 * Overlap-save model (WindowedCodec stand-in): the input window is [32 context | 64 new]
 * samples, the backend emits the OLDEST 64 samples of the window, so the model output lags the
 * input by 32 samples = internal_model_latency. With the default head-trim and tail flush the
 * offline output must still equal the input sample-exactly.
 */
InferenceConfig make_overlap_config() {
    return InferenceConfig(
        {},
        {{{{1, 1, (int)(k_chunk_size + k_overlap_context)}}, {{1, 1, (int)k_chunk_size}}}},
        ProcessingSpec({1}, {1}, {k_chunk_size}, {k_chunk_size}, {k_overlap_context}),
        5.f,
        0,
        false,
        0.f,
        2);
}

class OverlapPrePostProcessor : public PrePostProcessor {
public:
    using PrePostProcessor::PrePostProcessor;
    void pre_process(std::vector<RingBuffer>& input,
                     std::vector<BufferF>& output,
                     [[maybe_unused]] InferenceBackend current_inference_backend) override {
        pop_samples_from_buffer(input[0], output[0], k_chunk_size, k_overlap_context);
    }
};

class OverlapBackend : public BackendBase {
public:
    using BackendBase::BackendBase;
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        const float* read_ptr = input[0].get_read_pointer(0);
        float* write_ptr = output[0].get_write_pointer(0);
        for (size_t i = 0; i < k_chunk_size; ++i) {
            write_ptr[i] = read_ptr[i];  // oldest part of the window -> 32 samples internal lag
        }
    }
};

/**
 * Gain model with non-streamable tensors: input tensors = [64-sample audio, scalar gain],
 * output tensors = [64-sample audio, scalar]. The backend scales the audio by the gain value
 * and echoes the gain to the non-streamable output.
 */
InferenceConfig make_gain_config() {
    return InferenceConfig({},
                           {{{{1, 1, (int)k_chunk_size}, {1}}, {{1, 1, (int)k_chunk_size}, {1}}}},
                           ProcessingSpec({1, 1}, {1, 1}, {k_chunk_size, 0}, {k_chunk_size, 0}),
                           5.f,
                           0,
                           false,
                           0.f,
                           2);
}

class GainBackend : public BackendBase {
public:
    using BackendBase::BackendBase;
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<SessionElement> session) override {
        float const gain = input[1].get_read_pointer(0)[0];
        const float* read_ptr = input[0].get_read_pointer(0);
        float* write_ptr = output[0].get_write_pointer(0);
        for (size_t i = 0; i < k_chunk_size; ++i) { write_ptr[i] = read_ptr[i] * gain; }
        output[1].get_write_pointer(0)[0] = gain;
    }
};

std::vector<float> make_ramp(size_t length) {
    std::vector<float> ramp(length);
    for (size_t i = 0; i < length; ++i) { ramp[i] = 0.001f * static_cast<float>(i + 1); }
    return ramp;
}

/// Runs one single-tensor job synchronously and returns the result.
OfflineJobResult run_single_job(OfflineInferenceHandler& handler,
                                const std::vector<float>& input,
                                std::vector<float>& output,
                                OfflineJobOptions options = {}) {
    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};
    OfflineJobResult result;
    OfflineJobId const id = handler.submit(
        input_channels,
        input.size(),
        output_channels,
        output.size(),
        [&result](const OfflineJobResult& r) { result = r; },
        std::move(options));
    EXPECT_NE(id, k_invalid_offline_job_id);
    handler.wait(id);
    return result;
}

void expect_matches_input(const std::vector<float>& input,
                          const std::vector<float>& output,
                          size_t num_samples) {
    for (size_t i = 0; i < num_samples; ++i) {
        ASSERT_FLOAT_EQ(output[i], input[i]) << "at sample " << i;
    }
}

}  // namespace

TEST(OfflineInferenceHandlerTest, ExactMultipleLength) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();
    ASSERT_EQ(handler.get_num_parallel_jobs(), 1u);

    std::vector<float> const input = make_ramp(8 * k_chunk_size);
    std::vector<float> output(input.size(), -1.f);
    OfflineJobResult const result = run_single_job(handler, input, output);

    ASSERT_TRUE(result.m_success);
    ASSERT_EQ(result.m_num_output_samples_written[0], input.size());
    expect_matches_input(input, output, input.size());
}

TEST(OfflineInferenceHandlerTest, NonMultipleLength) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();

    for (size_t const length : {5 * k_chunk_size + 17, static_cast<size_t>(13)}) {
        std::vector<float> const input = make_ramp(length);
        std::vector<float> output(length, -1.f);
        OfflineJobResult const result = run_single_job(handler, input, output);

        ASSERT_TRUE(result.m_success);
        ASSERT_EQ(result.m_num_output_samples_written[0],
                  handler.get_expected_output_samples(length)[0]);
        ASSERT_EQ(result.m_num_output_samples_written[0], length);
        expect_matches_input(input, output, length);
    }
}

TEST(OfflineInferenceHandlerTest, OverlapInternalLatencyAlignment) {
    InferenceConfig config = make_overlap_config();
    OverlapPrePostProcessor pp_processor(config);
    OverlapBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();
    ASSERT_GE(handler.get_latency(0), k_overlap_context);

    size_t const length = 4 * k_chunk_size + 11;
    std::vector<float> const input = make_ramp(length);
    std::vector<float> output(length, -1.f);
    OfflineJobResult const result = run_single_job(handler, input, output);

    ASSERT_TRUE(result.m_success);
    ASSERT_EQ(result.m_num_output_samples_written[0], length);
    expect_matches_input(input, output, length);
}

TEST(OfflineInferenceHandlerTest, JobIsolationAutoClear) {
    InferenceConfig config = make_overlap_config();
    OverlapPrePostProcessor pp_processor(config);
    OverlapBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();

    // The overlap context of job 2's first window must be zeros, not job 1's tail: with
    // automatic per-job clearing, two identical jobs must produce identical output.
    size_t const length = 3 * k_chunk_size;
    std::vector<float> const input = make_ramp(length);
    std::vector<float> output_first(length, -1.f);
    std::vector<float> output_second(length, -2.f);
    run_single_job(handler, input, output_first);
    run_single_job(handler, input, output_second);
    expect_matches_input(output_first, output_second, length);
    expect_matches_input(input, output_second, length);
}

TEST(OfflineInferenceHandlerTest, RawModeHeadTrimZeroNoFlush) {
    // The overlap model has internal latency, so raw mode (no trim, no flush) differs from
    // the aligned default: the first k_overlap_context samples are the model's warm-up.
    InferenceConfig config = make_overlap_config();
    OverlapPrePostProcessor pp_processor(config);
    OverlapBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();

    size_t const length = 4 * k_chunk_size;
    unsigned int const latency = handler.get_latency(0);
    ASSERT_GT(latency, 0u);

    std::vector<float> const input = make_ramp(length);
    std::vector<float> output(length + latency, -1.f);
    OfflineJobOptions options;
    options.m_head_trim = {0};
    options.m_tail_flush = false;

    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};
    OfflineJobResult result;
    OfflineJobId const id = handler.submit(
        input_channels,
        input.size(),
        output_channels,
        output.size(),
        [&result](const OfflineJobResult& r) { result = r; },
        options);
    handler.wait(id);

    // Raw output: no prefill zeros exist offline (latency == internal model latency); the
    // stream starts with the model's warm-up (zero left-context shifted through), then the
    // shifted input; written clamped to what the real chunks produced (no tail flush).
    ASSERT_EQ(latency, k_overlap_context);
    ASSERT_EQ(result.m_num_output_samples_written[0], length);  // n_real*out - trim(0)
    for (size_t i = 0; i < latency; ++i) { ASSERT_FLOAT_EQ(output[i], 0.f) << "warm-up " << i; }
    for (size_t i = 0; i < length - latency; ++i) {
        ASSERT_FLOAT_EQ(output[latency + i], input[i]) << "payload " << i;
    }
}

TEST(OfflineInferenceHandlerTest, NonStreamableTensorsExclusiveJobs) {
    InferenceConfig config = make_gain_config();
    PrePostProcessor pp_processor(config);
    GainBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 2);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();

    size_t const length = 4 * k_chunk_size;
    std::vector<float> const input = make_ramp(length);

    struct GainJob {
        float m_gain;
        std::vector<float> m_output;
        OfflineJobResult m_result;
    };
    std::vector<GainJob> jobs;
    jobs.push_back({2.f, std::vector<float>(length, -1.f), {}});
    jobs.push_back({-1.f, std::vector<float>(length, -1.f), {}});

    const float* input_channels[] = {input.data()};
    std::vector<const float* const*> input_ptrs = {input_channels, nullptr};
    std::vector<size_t> input_counts = {length, 0};

    for (auto& job : jobs) {
        float* output_channels[] = {job.m_output.data()};
        std::vector<float* const*> output_ptrs = {output_channels, nullptr};
        std::vector<size_t> output_counts = {length, 0};
        OfflineJobOptions options;
        options.m_non_streamable_inputs = {{}, {job.m_gain}};
        OfflineJobId const id = handler.submit(
            input_ptrs.data(),
            input_counts.data(),
            output_ptrs.data(),
            output_counts.data(),
            [&job](const OfflineJobResult& r) { job.m_result = r; },
            options);
        ASSERT_NE(id, k_invalid_offline_job_id);
    }
    handler.wait_all();

    for (const auto& job : jobs) {
        ASSERT_TRUE(job.m_result.m_success);
        ASSERT_EQ(job.m_result.m_num_output_samples_written[0], length);
        ASSERT_EQ(job.m_result.m_non_streamable_outputs[1].size(), 1u);
        ASSERT_FLOAT_EQ(job.m_result.m_non_streamable_outputs[1][0], job.m_gain);
        for (size_t i = 0; i < length; ++i) {
            ASSERT_FLOAT_EQ(job.m_output[i], input[i] * job.m_gain) << "at sample " << i;
        }
    }
}

TEST(OfflineInferenceHandlerTest, MultiJobFifoCallbackThreadWaitAll) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();

    constexpr size_t k_num_jobs = 4;
    size_t const length = 2 * k_chunk_size;
    std::vector<float> const input = make_ramp(length);
    std::vector<std::vector<float>> outputs(k_num_jobs, std::vector<float>(length, -1.f));

    std::mutex mutex;
    std::vector<OfflineJobId> completion_order;
    std::vector<std::thread::id> callback_threads;

    for (auto& output : outputs) {
        const float* input_channels[] = {input.data()};
        float* output_channels[] = {output.data()};
        handler.submit(input_channels,
                       length,
                       output_channels,
                       length,
                       [&](const OfflineJobResult& r) {
                           std::lock_guard<std::mutex> const lock(mutex);
                           completion_order.push_back(r.m_job_id);
                           callback_threads.push_back(std::this_thread::get_id());
                       });
    }
    handler.wait_all();

    ASSERT_EQ(completion_order.size(), k_num_jobs);
    // One lane -> strict FIFO completion
    for (size_t i = 1; i < completion_order.size(); ++i) {
        ASSERT_GT(completion_order[i], completion_order[i - 1]);
    }
    // Immediate delivery: callbacks fire on the lane worker thread, never the main thread
    for (const auto& thread_id : callback_threads) {
        ASSERT_NE(thread_id, std::this_thread::get_id());
    }
    for (const auto& output : outputs) { expect_matches_input(input, output, length); }
}

TEST(OfflineInferenceHandlerTest, ParallelJobsOutOfOrderCompletion) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 2);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();
    ASSERT_EQ(handler.get_num_parallel_jobs(), 2u);

    size_t const long_length = 4096 * k_chunk_size;
    size_t const short_length = k_chunk_size;
    std::vector<float> const long_input = make_ramp(long_length);
    std::vector<float> const short_input = make_ramp(short_length);
    std::vector<float> long_output(long_length, -1.f);
    std::vector<float> short_output(short_length, -1.f);

    std::mutex mutex;
    std::vector<OfflineJobId> completion_order;
    auto const record = [&](const OfflineJobResult& r) {
        std::lock_guard<std::mutex> const lock(mutex);
        completion_order.push_back(r.m_job_id);
    };

    const float* long_channels[] = {long_input.data()};
    float* long_out_channels[] = {long_output.data()};
    OfflineJobId const long_id =
        handler.submit(long_channels, long_length, long_out_channels, long_length, record);
    const float* short_channels[] = {short_input.data()};
    float* short_out_channels[] = {short_output.data()};
    OfflineJobId const short_id =
        handler.submit(short_channels, short_length, short_out_channels, short_length, record);
    handler.wait_all();

    ASSERT_EQ(completion_order.size(), 2u);
    // The short job overtakes the long one on the second lane
    ASSERT_EQ(completion_order.front(), short_id);
    ASSERT_EQ(completion_order.back(), long_id);
    expect_matches_input(long_input, long_output, long_length);
    expect_matches_input(short_input, short_output, short_length);
}

TEST(OfflineInferenceHandlerTest, PolledDelivery) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor,
                                    config,
                                    backend,
                                    ContextConfig(),
                                    1,
                                    OfflineDeliveryMode::Polled);
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();

    size_t const length = 2 * k_chunk_size;
    std::vector<float> const input = make_ramp(length);
    std::vector<float> output(length, -1.f);

    std::thread::id callback_thread;
    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};
    OfflineJobId const id = handler.submit(
        input_channels,
        length,
        output_channels,
        length,
        [&](const OfflineJobResult&) { callback_thread = std::this_thread::get_id(); });
    handler.wait(id);

    // Processed but not delivered yet - delivery happens on this thread via poll_wait/poll
    size_t delivered = handler.poll_wait(std::chrono::milliseconds(1000));
    delivered += handler.poll();
    ASSERT_EQ(delivered, 1u);
    ASSERT_EQ(callback_thread, std::this_thread::get_id());
    expect_matches_input(input, output, length);
}

TEST(OfflineInferenceHandlerTest, StatefulConfigDefaultsToOneLane) {
    InferenceConfig config({},
                           {{{{1, 1, (int)k_chunk_size}}, {{1, 1, (int)k_chunk_size}}}},
                           ProcessingSpec({1}, {1}, {k_chunk_size}, {k_chunk_size}),
                           5.f,
                           0,
                           true,  // session_exclusive_processor
                           0.f,
                           2);
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend);  // auto lanes
    handler.set_inference_backend(InferenceBackend::CUSTOM);
    handler.prepare();
    ASSERT_EQ(handler.get_num_parallel_jobs(), 1u);

    size_t const length = 3 * k_chunk_size;
    std::vector<float> const input = make_ramp(length);
    std::vector<float> output(length, -1.f);
    OfflineJobResult const result = run_single_job(handler, input, output);
    ASSERT_TRUE(result.m_success);
    expect_matches_input(input, output, length);
}

namespace {
class NotifyingIdentityBackend : public IdentityBackend {
public:
    NotifyingIdentityBackend(InferenceConfig& config, std::atomic<bool>& started)
        : IdentityBackend(config), m_started(started) {}
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 std::shared_ptr<SessionElement> session) override {
        m_started.store(true, std::memory_order::release);
        IdentityBackend::process(input, output, std::move(session));
    }

private:
    std::atomic<bool>& m_started;
};
}  // namespace

TEST(OfflineInferenceHandlerTest, DestructorFailsQueuedJobs) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    std::atomic<bool> first_job_started{false};
    NotifyingIdentityBackend backend(config, first_job_started);

    std::mutex mutex;
    std::vector<bool> results;
    // Declared before the handler: the borrowed buffers must outlive the handler, whose
    // destructor still finishes the running job.
    size_t const long_length = 2048 * k_chunk_size;
    std::vector<float> const input = make_ramp(long_length);
    std::vector<std::vector<float>> outputs(3, std::vector<float>(long_length, 0.f));
    {
        OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
        handler.set_inference_backend(InferenceBackend::CUSTOM);
        handler.prepare();

        for (auto& output : outputs) {
            const float* input_channels[] = {input.data()};
            float* output_channels[] = {output.data()};
            handler.submit(input_channels,
                           long_length,
                           output_channels,
                           long_length,
                           [&](const OfflineJobResult& r) {
                               std::lock_guard<std::mutex> const lock(mutex);
                               results.push_back(r.m_success);
                           });
        }
        // Make the destructor race deterministic: job 1 is definitely running
        while (!first_job_started.load(std::memory_order::acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Destructor: the running job finishes, the queued jobs fail
    }
    ASSERT_EQ(results.size(), 3u);
    ASSERT_TRUE(results.front());  // the job that was already running succeeded
    ASSERT_FALSE(results[1]);      // queued jobs were dropped with m_success == false
    ASSERT_FALSE(results[2]);
}

TEST(OfflineInferenceHandlerTest, SubmitValidation) {
    InferenceConfig config = make_identity_config();
    PrePostProcessor pp_processor(config);
    IdentityBackend backend(config);
    OfflineInferenceHandler handler(pp_processor, config, backend, ContextConfig(), 1);
    handler.set_inference_backend(InferenceBackend::CUSTOM);

    std::vector<float> const input = make_ramp(k_chunk_size);
    std::vector<float> output(k_chunk_size, 0.f);
    const float* input_channels[] = {input.data()};
    float* output_channels[] = {output.data()};

    // Not prepared yet
    ASSERT_EQ(handler.submit(input_channels,
                             input.size(),
                             output_channels,
                             output.size(),
                             [](const OfflineJobResult&) {}),
              k_invalid_offline_job_id);

    handler.prepare();

    // head_trim size mismatch
    OfflineJobOptions bad_trim;
    bad_trim.m_head_trim = {0, 0};
    ASSERT_EQ(handler.submit(
                  input_channels,
                  input.size(),
                  output_channels,
                  output.size(),
                  [](const OfflineJobResult&) {},
                  bad_trim),
              k_invalid_offline_job_id);

    // non-streamable values on a streamable tensor
    OfflineJobOptions bad_values;
    bad_values.m_non_streamable_inputs = {{1.f}};
    ASSERT_EQ(handler.submit(
                  input_channels,
                  input.size(),
                  output_channels,
                  output.size(),
                  [](const OfflineJobResult&) {},
                  bad_values),
              k_invalid_offline_job_id);
}
