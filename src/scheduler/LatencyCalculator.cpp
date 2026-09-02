#include <anira/InferenceConfig.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace anira {

namespace {

// Absolute tolerance for the ceilings of times measured in host blocks or hops. The
// inputs are float milliseconds and sample rates, so anything closer to an integer than
// this is that integer; it also keeps a value that is exactly one host block from being
// rounded up to two by the last bit of a division.
constexpr double k_ceil_epsilon = 1e-6;

// Hard stop for the batch search of a configuration at the very edge of feasibility
// (kappa just below n), where the bound closes slowly. Far above any real config.
constexpr int64_t k_max_batches = int64_t{1} << 16;

int64_t ceil_div(int64_t numerator, int64_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

int64_t ceil_with_epsilon(double value) {
    return static_cast<int64_t>(std::ceil(value - k_ceil_epsilon));
}

}  // namespace

int64_t LatencyCalculator::greatest_common_divisor(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t const t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int64_t LatencyCalculator::least_common_multiple(int64_t a, int64_t b) {
    if (a == 0 || b == 0) { return 0; }
    return a / greatest_common_divisor(a, b) * b;
}

int64_t LatencyCalculator::buffer_adaptation(int64_t host_block_size, int64_t stream_size) {
    return stream_size - greatest_common_divisor(host_block_size, stream_size);
}

int64_t LatencyCalculator::buffer_adaptation_flexible(int64_t stream_size) {
    return std::max<int64_t>(stream_size - 1, 0);
}

LatencyCalculator::Rational LatencyCalculator::rationalize(double value, int64_t max_denominator) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return Rational{.m_numerator = 0, .m_denominator = 1};
    }
    // Continued-fraction convergents h/k; every convergent is in lowest terms.
    double const tolerance = value * 1e-6;
    int64_t h_prev = 1;
    int64_t k_prev = 0;
    auto h = static_cast<int64_t>(std::floor(value));
    int64_t k = 1;
    double fraction = value - std::floor(value);
    while (std::abs(value - static_cast<double>(h) / static_cast<double>(k)) > tolerance) {
        if (fraction < 1e-12) { break; }
        double const inverse = 1.0 / fraction;
        auto const a = static_cast<int64_t>(std::floor(inverse));
        fraction = inverse - std::floor(inverse);
        int64_t const h_next = a * h + h_prev;
        int64_t const k_next = a * k + k_prev;
        if (k_next > max_denominator) { break; }
        h_prev = h;
        k_prev = k;
        h = h_next;
        k = k_next;
    }
    return Rational{.m_numerator = h, .m_denominator = k};
}

std::vector<unsigned int> LatencyCalculator::sync_latencies(
    const std::vector<unsigned int>& latencies,
    const std::vector<size_t>& output_sizes) {
    std::vector<unsigned int> result;
    result.reserve(latencies.size());
    if (latencies.size() > 1) {
        double ratio = 0.0;
        for (size_t i = 0; i < latencies.size(); ++i) {
            if (output_sizes[i] > 0) {
                ratio = std::max(
                    ratio,
                    static_cast<double>(latencies[i]) / static_cast<double>(output_sizes[i]));
            }
        }
        auto const hops = static_cast<unsigned int>(std::ceil(ratio - k_ceil_epsilon));
        for (size_t const output_size : output_sizes) {
            result.push_back(output_size > 0 ? hops * static_cast<unsigned int>(output_size) : 0U);
        }
    } else if (!latencies.empty()) {
        result.push_back(output_sizes[0] > 0 ? latencies[0] : 0U);
    }
    return result;
}

int64_t LatencyCalculator::batch_delay(double host_blocks, double beta) {
    double const remaining = host_blocks - beta;
    if (remaining <= k_ceil_epsilon) { return 0; }
    return ceil_with_epsilon(remaining);
}

int64_t LatencyCalculator::burst_batches(const Timing& timing) {
    // One host block triggers at most ceil(rho) inferences, processed n at a time.
    return std::max<int64_t>(ceil_div(ceil_div(timing.m_p, timing.m_q), timing.m_n), 1);
}

int64_t LatencyCalculator::inference_latency_numerator(const Timing& timing) {
    // tau: inference time in host blocks. p * (m+1) * tau == (m+1) * kappa * q.
    double const tau =
        timing.m_kappa * static_cast<double>(timing.m_q) / static_cast<double>(timing.m_p);
    int64_t best = std::numeric_limits<int64_t>::min();
    for (int64_t m = 0; m < k_max_batches; ++m) {
        int64_t const d = batch_delay(static_cast<double>(m + 1) * tau, timing.m_beta);
        best = std::max(best, timing.m_p * d - m * timing.m_n * timing.m_q);
        if (timing.m_batch_cap > 0) {
            if (m + 1 >= timing.m_batch_cap) { break; }
            continue;
        }
        // Every later batch m' >= m + 1 satisfies rho d_m' - m' n <= rho max(0, (m'+1) tau -
        // beta + 1) - m' n, a bound that decreases in m' while kappa < n.
        auto const next = static_cast<double>(m + 1);
        double const bound = static_cast<double>(timing.m_p) *
                                 std::max(0.0, (next + 1.0) * tau - timing.m_beta + 1.0) -
                             next * static_cast<double>(timing.m_n * timing.m_q);
        if (bound <= static_cast<double>(best)) { break; }
    }
    return best;
}

int64_t LatencyCalculator::num_structs(const Timing& timing) {
    double const tau =
        timing.m_kappa * static_cast<double>(timing.m_q) / static_cast<double>(timing.m_p);
    int64_t best = std::numeric_limits<int64_t>::min();
    for (int64_t m = 0; m < k_max_batches; ++m) {
        int64_t const d = batch_delay(static_cast<double>(m + 1) * tau, timing.m_beta);
        best = std::max(best, ceil_div((d + 1) * timing.m_p, timing.m_q) - m * timing.m_n);
        if (timing.m_batch_cap > 0) {
            if (m + 1 >= timing.m_batch_cap) { break; }
            continue;
        }
        // (d_m' + 1) rho <= rho max(1, (m'+1) tau - beta + 2) for every later batch.
        auto const next = static_cast<double>(m + 1);
        double const rho = static_cast<double>(timing.m_p) / static_cast<double>(timing.m_q);
        double const bound =
            std::ceil(rho * std::max(1.0, (next + 1.0) * tau - timing.m_beta + 2.0) -
                      k_ceil_epsilon) -
            next * static_cast<double>(timing.m_n);
        if (bound <= static_cast<double>(best)) { break; }
    }
    return best;
}

void LatencyCalculator::smaller_buffers_worst_case(int64_t max_block,
                                                   int64_t grid_hop,
                                                   double kappa,
                                                   double beta,
                                                   int64_t n,
                                                   bool feasible,
                                                   int64_t& latency_numerator,
                                                   int64_t& structs) {
    // Block j of the grid is rho' = j / H hops and takes (m+1) kappa H / j host blocks per
    // batch. Values are kept as numerators over H (latency) and as counts (structs).
    auto const grid = static_cast<double>(grid_hop);
    auto const max_block_d = static_cast<double>(max_block);
    int64_t const batch_cap =
        feasible ? 0 : std::max<int64_t>(ceil_div(ceil_div(max_block, grid_hop), n), 1);

    latency_numerator = std::numeric_limits<int64_t>::min();
    structs = std::numeric_limits<int64_t>::min();

    auto evaluate = [&](int64_t m, int64_t j) -> int64_t {
        double const blocks = static_cast<double>(m + 1) * kappa * grid / static_cast<double>(j);
        int64_t const d = batch_delay(blocks, beta);
        latency_numerator = std::max(latency_numerator, j * d - m * n * grid_hop);
        structs = std::max(structs, ceil_div((d + 1) * j, grid_hop) - m * n);
        return d;
    };

    for (int64_t m = 0; m < k_max_batches; ++m) {
        double const work = static_cast<double>(m + 1) * kappa * grid;  // (m+1) kappa H
        int64_t j = max_block;
        while (j >= 1) {
            // The top of the current piece, plus its neighbour so that a last-bit rounding
            // error in the breakpoint below can never skip the true maximum.
            int64_t const c = evaluate(m, j);
            if (j > 1) { evaluate(m, j - 1); }
            // The next piece has ceiling >= c + 1; its last grid point is the largest j
            // with work / j - beta > c, i.e. j < work / (c + beta).
            int64_t next = j - 1;
            double const edge = static_cast<double>(c) + beta;
            if (edge > 0.0) { next = std::min(next, ceil_with_epsilon(work / edge) - 1); }
            if (next < 1) { break; }
            // Bound for every remaining piece: j d < work d / (d - 1 + beta), which is
            // non-increasing in d for beta <= 1 and below work otherwise.
            double const ratio =
                beta <= 1.0 ? (static_cast<double>(c) + 1.0) / (static_cast<double>(c) + beta)
                            : 1.0;
            double const latency_bound = work * ratio - static_cast<double>(m * n * grid_hop);
            double const structs_bound =
                std::ceil((work * ratio + static_cast<double>(next)) / grid - k_ceil_epsilon) -
                static_cast<double>(m * n);
            if (latency_bound <= static_cast<double>(latency_numerator) &&
                structs_bound <= static_cast<double>(structs)) {
                break;
            }
            j = next;
        }
        if (batch_cap > 0) {
            if (m + 1 >= batch_cap) { break; }
            continue;
        }
        // Bound over every block for the later batches: j d <= work' + j (1 - beta).
        auto const next_m = static_cast<double>(m + 1);
        double const next_work = (next_m + 1.0) * kappa * grid;
        double const spread = max_block_d * std::max(1.0 - beta, 0.0);
        double const latency_bound =
            next_work + spread - next_m * static_cast<double>(n * grid_hop);
        double const structs_bound =
            std::ceil((next_work + spread + max_block_d) / grid - k_ceil_epsilon) -
            next_m * static_cast<double>(n);
        if (latency_bound <= static_cast<double>(latency_numerator) &&
            structs_bound <= static_cast<double>(structs)) {
            break;
        }
    }
}

LatencyCalculator::LatencyCalculator(const InferenceConfig& inference_config,
                                     const HostConfig& host_config)
    : m_input_sizes(inference_config.get_preprocess_input_size())
    , m_output_sizes(inference_config.get_postprocess_output_size())
    , m_allow_smaller_buffers(host_config.m_allow_smaller_buffers) {
    // Throws std::invalid_argument for an unresolvable reference stream.
    auto const reference_size =
        static_cast<double>(host_config.get_reference_size(inference_config));
    if (!(host_config.m_buffer_size > 0.f) || !(host_config.m_sample_rate > 0.f)) {
        throw std::invalid_argument(
            "HostConfig: buffer size and sample rate must be positive to calculate a latency.");
    }

    // History a receptive-field model peeks at: the input tensor beyond one hop.
    const std::vector<size_t>& tensor_input_size = inference_config.get_tensor_input_size();
    const std::vector<size_t>& input_channels = inference_config.get_preprocess_input_channels();
    m_input_history.reserve(m_input_sizes.size());
    for (size_t i = 0; i < m_input_sizes.size(); ++i) {
        size_t const per_channel = tensor_input_size[i] / std::max<size_t>(input_channels[i], 1);
        m_input_history.push_back(per_channel > m_input_sizes[i] ? per_channel - m_input_sizes[i]
                                                                 : 0);
    }

    for (size_t const size : m_input_sizes) {
        m_max_hop = std::max(m_max_hop, static_cast<int64_t>(size));
    }
    for (size_t const size : m_output_sizes) {
        m_max_hop = std::max(m_max_hop, static_cast<int64_t>(size));
    }

    m_rho = rationalize(static_cast<double>(host_config.m_buffer_size) / reference_size);
    if (m_rho.m_numerator <= 0) {
        throw std::invalid_argument("HostConfig: the buffer size is too small to represent.");
    }
    m_kappa = static_cast<double>(inference_config.m_max_inference_time) *
              static_cast<double>(host_config.m_sample_rate) / (1000.0 * reference_size);
    m_beta = std::max(static_cast<double>(inference_config.m_blocking_ratio), 0.0);
    m_n = std::max<int64_t>(inference_config.m_num_parallel_processors, 1);
    m_feasible = m_kappa < static_cast<double>(m_n);
    if (!m_feasible) {
        ANIRA_LOG_WARNING(log_group::k_scheduler,
                          "The inference pool cannot keep up: every hop brings %.3f hop periods "
                          "of inference work (max_inference_time %.3f ms) for %lld parallel "
                          "processors. Latency and inference slots are sized for one host block "
                          "on an idle pool; the stream will underrun.",
                          m_kappa,
                          static_cast<double>(inference_config.m_max_inference_time),
                          static_cast<long long>(m_n));
    }

    Timing const timing{.m_p = m_rho.m_numerator,
                        .m_q = m_rho.m_denominator,
                        .m_kappa = m_kappa,
                        .m_beta = m_beta,
                        .m_n = m_n,
                        .m_batch_cap = 0};
    Timing capped = timing;
    if (!m_feasible) { capped.m_batch_cap = burst_batches(timing); }

    m_latency_numerator = (m_rho.m_denominator - 1) + inference_latency_numerator(capped);
    int64_t const fixed_structs = num_structs(capped);
    m_num_structs = static_cast<size_t>(std::max<int64_t>(fixed_structs, 1));

    if (m_allow_smaller_buffers) {
        // Largest block on the grid of the finest stream, floor(rho H).
        int64_t const max_block = m_rho.m_numerator * m_max_hop / m_rho.m_denominator;
        if (max_block >= 1) {
            int64_t grid_latency = 0;
            int64_t grid_structs = 0;
            smaller_buffers_worst_case(max_block,
                                       m_max_hop,
                                       m_kappa,
                                       m_beta,
                                       m_n,
                                       m_feasible,
                                       grid_latency,
                                       grid_structs);
            m_smaller_numerator = (m_max_hop - 1) + grid_latency;
            m_num_structs = static_cast<size_t>(
                std::max<int64_t>(static_cast<int64_t>(m_num_structs), grid_structs));
        } else {
            m_allow_smaller_buffers = false;  // no smaller whole-sample block exists
        }
    }

    // Send rings: one host block plus the largest leftover the adaptation can leave.
    m_send_buffer_sizes.reserve(m_input_sizes.size());
    for (size_t i = 0; i < m_input_sizes.size(); ++i) {
        auto const hop = static_cast<int64_t>(m_input_sizes[i]);
        if (hop == 0) {
            m_send_buffer_sizes.push_back(0);
            continue;
        }
        int64_t const block_numerator = m_rho.m_numerator * hop;
        bool const integer_block = block_numerator % m_rho.m_denominator == 0;
        int64_t const block = ceil_div(block_numerator, m_rho.m_denominator);
        int64_t const leftover = (integer_block && !m_allow_smaller_buffers)
                                     ? buffer_adaptation(block, hop)
                                     : buffer_adaptation_flexible(hop);
        m_send_buffer_sizes.push_back(
            static_cast<size_t>(block + leftover + static_cast<int64_t>(m_input_history[i])));
    }
}

double LatencyCalculator::get_latency_hops() const {
    return static_cast<double>(m_latency_numerator) / static_cast<double>(m_rho.m_denominator);
}

double LatencyCalculator::get_latency_hops_smaller_buffers() const {
    double const fixed = get_latency_hops();
    if (!m_allow_smaller_buffers) { return fixed; }
    return std::max(fixed,
                    static_cast<double>(m_smaller_numerator) / static_cast<double>(m_max_hop));
}

std::vector<float> LatencyCalculator::get_output_latencies() const {
    double const hops = get_latency_hops_smaller_buffers();
    std::vector<float> result;
    result.reserve(m_output_sizes.size());
    for (size_t const output_size : m_output_sizes) {
        result.push_back(
            output_size > 0 ? static_cast<float>(static_cast<double>(output_size) * hops) : 0.f);
    }
    return result;
}

std::vector<unsigned int> LatencyCalculator::get_synced_output_latencies() const {
    // floor(P_i Lambda) in exact integer arithmetic, for the fixed block and for the grid,
    // each synchronized across the outputs; the larger of the two per output.
    auto per_output = [&](int64_t numerator, int64_t denominator) {
        std::vector<unsigned int> latencies;
        latencies.reserve(m_output_sizes.size());
        for (size_t const output_size : m_output_sizes) {
            latencies.push_back(output_size > 0
                                    ? static_cast<unsigned int>(static_cast<int64_t>(output_size) *
                                                                numerator / denominator)
                                    : 0U);
        }
        return sync_latencies(latencies, m_output_sizes);
    };
    std::vector<unsigned int> result = per_output(m_latency_numerator, m_rho.m_denominator);
    if (m_allow_smaller_buffers) {
        std::vector<unsigned int> const grid = per_output(m_smaller_numerator, m_max_hop);
        for (size_t i = 0; i < result.size(); ++i) { result[i] = std::max(result[i], grid[i]); }
    }
    return result;
}

size_t LatencyCalculator::get_num_structs() const {
    return m_num_structs;
}

std::vector<size_t> LatencyCalculator::get_send_buffer_sizes() const {
    return m_send_buffer_sizes;
}

bool LatencyCalculator::is_feasible() const {
    return m_feasible;
}

LatencyCalculator::Rational LatencyCalculator::get_block_hops() const {
    return m_rho;
}

double LatencyCalculator::get_inference_hop_periods() const {
    return m_kappa;
}

}  // namespace anira
