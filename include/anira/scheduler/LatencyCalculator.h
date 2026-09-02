#ifndef ANIRA_LATENCYCALCULATOR_H
#define ANIRA_LATENCYCALCULATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../InferenceConfig.h"
#include "../utils/HostConfig.h"

namespace anira {

/**
 * @brief Latency, inference-slot count and ring-buffer sizing of a session, in closed form
 *
 * Everything a SessionElement needs to size itself for a host configuration is a
 * function of five numbers, all derived from the InferenceConfig and the HostConfig:
 *
 * - @f$\rho = B / R@f$, the host block measured in hops: @f$B@f$ is the host buffer size
 *   in samples of the reference stream and @f$R@f$ that stream's samples per inference
 *   (HostConfig::get_reference_size). @f$\rho@f$ is stored as a reduced fraction
 *   @f$p / q@f$ (see rationalize()).
 * - @f$\kappa = T \cdot f_s / (1000 \cdot R)@f$, the maximum inference time @f$T@f$ (ms)
 *   measured in hop periods (@f$f_s@f$ is the host sample rate of the reference stream).
 * - @f$\beta@f$, the blocking ratio: the driving thread waits @f$\beta@f$ host block
 *   periods for results inside every callback.
 * - @f$n@f$, the number of parallel processors of the session.
 * - @f$H@f$, the largest hop of any streamable tensor: the stream in which the host's
 *   block size is finest, and the unit of the `allow_smaller_buffers` block grid.
 *
 * The model is the scheduler's actual worst case: every inference takes exactly
 * @f$T@f$, at most @f$n@f$ run at once in submission order, an inference is submitted by
 * the host callback that completes its hop, results are collected once per callback
 * (after the blocking wait), and the host pushes and pops a sample only once a whole
 * one has accumulated in its block (the documented fractional-block convention, so that
 * per-stream block sizes may be fractional).
 *
 * @par Buffer adaptation (Rath & Geier, LAC 2026)
 * Repackaging a stream from constant host blocks of @f$b@f$ samples into blocks of
 * @f$P@f$ samples needs a delay of exactly @f$\Delta = P - \gcd(b, P)@f$ samples
 * (their corollary 3.8, replacing the PortAudio-style LCM loop). In hop units this is
 * @f$1 - 1/q@f$ for every stream at once, because @f$\gcd(\rho P, P) = P / q@f$
 * whenever @f$\rho P@f$ is an integer. When the host block may vary
 * (`allow_smaller_buffers`) or is fractional on a stream, the worst case is
 * @f$P - 1@f$ samples (their section 5).
 *
 * @par Inference queue
 * Let @f$\tau = \kappa / \rho@f$ be the inference time in host blocks and
 * @f$d_m = \max(0, \lceil (m + 1)\tau - \beta \rceil)@f$ the number of callbacks after
 * which the @f$(m+1)@f$-th batch of @f$n@f$ inferences submitted together is collected.
 * The receive ring never runs dry iff its zero priming @f$L@f$ (in hops) satisfies
 * @f$L \ge (k+1)\rho - C(k)@f$ for every callback @f$k@f$, where @f$C(k)@f$ counts the
 * inferences collected by callback @f$k@f$. With FIFO departures
 * @f$F_j = \max(a_j, F_{j-n}) + \tau@f$ and submissions @f$a_j = \lceil (j+1)/\rho \rceil - 1@f$
 * this gives @f$C(k) = \min_m [\lfloor \rho (k + 1 - d_m) \rfloor + m n]@f$, hence
 * @f[
 *   \Lambda = \max_k [(k+1)\rho - C(k)] = \frac{q-1}{q} + \max_{m \ge 0} [\rho\, d_m - m n],
 * @f]
 * the first term being the buffer adaptation above. The maximum over @f$m@f$ exists iff
 * @f$\kappa < n@f$ (the pool keeps up with the stream); only
 * @f$m < \rho / (n - \kappa)@f$ can contribute, so the search is finite. The latency of
 * output tensor @f$i@f$ in its own samples is @f$P_i \Lambda@f$ (see
 * get_output_latencies()), rounded down because the host pops whole samples only.
 *
 * @par Inference slots
 * By the same recursion the number of inferences submitted but not yet collected right
 * after the submissions of a callback is at most
 * @f[
 *   S = \max_{m \ge 0} \left[ \lceil (d_m + 1)\rho \rceil - m n \right],
 * @f]
 * the steady-state slot count (get_num_structs()); the session allocates twice as many
 * ThreadSafeStructs so that a wait-free reset, which strands the in-flight inferences
 * in their slots until the workers finish, never starves the fresh schedule.
 *
 * @par allow_smaller_buffers
 * The host may then use any block of @f$j@f$ samples of the finest stream,
 * @f$1 \le j \le \lfloor \rho H \rfloor@f$, i.e. @f$\rho' = j / H@f$ hops, with
 * @f$\tau' = \kappa / \rho'@f$. The worst case over that grid uses the flexible-host
 * adaptation @f$(H - 1)/H@f$ and the maximum of @f$\rho' d_m(\rho') - m n@f$ and of the
 * slot count over @f$j@f$. On every interval where @f$\lceil (m+1)\tau' - \beta \rceil@f$
 * is a constant @f$c@f$ both are increasing in @f$j@f$, so the maximum is attained at the
 * last grid point of the interval, @f$j_c = \lceil (m+1)\kappa H / (c - 1 + \beta) \rceil - 1@f$,
 * and the remaining intervals are bounded by @f$(m+1)\kappa\, c / (c - 1 + \beta)@f$.
 * This replaces the former countdown over every block size.
 *
 * All of this is computed once, in the constructor; the getters are trivial. The class
 * is public so that the formulas can be unit-tested against a brute-force simulation.
 */
class ANIRA_API LatencyCalculator {
public:
    /**
     * @brief A non-negative rational number
     */
    struct Rational {
        int64_t m_numerator = 0;    ///< Numerator
        int64_t m_denominator = 1;  ///< Denominator, always > 0
    };

    /**
     * @brief Computes every quantity for one host configuration
     *
     * @param inference_config Model configuration (hops, max inference time, blocking ratio,
     * parallel processors)
     * @param host_config Host configuration (buffer size and sample rate in reference-stream
     * samples, allow_smaller_buffers, reference selection)
     * @throws std::invalid_argument if the host config's reference stream cannot be resolved
     */
    LatencyCalculator(const InferenceConfig& inference_config, const HostConfig& host_config);

    /**
     * @brief The worst-case latency in hops, @f$\Lambda@f$ (fixed host block)
     *
     * @return Latency in hops, as a floating-point number
     */
    double get_latency_hops() const;

    /**
     * @brief The worst-case latency in hops over the smaller-block grid
     *
     * Equal to get_latency_hops() unless HostConfig::m_allow_smaller_buffers is set.
     *
     * @return Latency in hops, as a floating-point number
     */
    double get_latency_hops_smaller_buffers() const;

    /**
     * @brief Per-output float latency in samples of that output
     *
     * @f$P_i \Lambda@f$ for every streamable output tensor @f$i@f$, 0 for a non-streamable
     * one; the smaller-block grid is included when the host config allows smaller buffers.
     * Index-aligned with the output tensor list.
     *
     * @return Latency values in samples, one per output tensor
     */
    std::vector<float> get_output_latencies() const;

    /**
     * @brief Per-output integer latency, synchronized across outputs
     *
     * @f$\lfloor P_i \Lambda \rfloor@f$ per streamable output; with more than one output tensor
     * the latencies are then raised to a common whole number of hops
     * (see sync_latencies()). Non-streamable outputs report 0. This is what
     * SessionElement primes its receive rings with, before the internal model latency
     * and any custom latency are applied.
     *
     * @return Latency values in samples, one per output tensor
     */
    std::vector<unsigned int> get_synced_output_latencies() const;

    /**
     * @brief The steady-state number of inference slots, @f$S@f$
     *
     * The maximum number of inferences submitted but not yet collected at any callback.
     * SessionElement allocates twice this many ThreadSafeStructs: a wait-free reset
     * leaves the in-flight inferences in their slots until the workers finish, while
     * the fresh schedule needs @f$S@f$ slots of its own.
     *
     * @return Maximum number of inferences in flight at any callback, @f$S@f$
     */
    size_t get_num_structs() const;

    /**
     * @brief Send ring sizes per input tensor
     *
     * One host block plus the largest leftover the adaptation can leave in the ring
     * (@f$P - \gcd(\lceil b \rceil, P)@f$ for a constant integer block, @f$P - 1@f$ for a
     * fractional or variable one) plus the history a receptive-field model peeks at.
     * 0 for a non-streamable input.
     *
     * @return Ring sizes in samples, one per input tensor
     */
    std::vector<size_t> get_send_buffer_sizes() const;

    /**
     * @brief Whether the inference pool can keep up with the stream
     *
     * False when @f$\kappa \ge n@f$: every hop of audio brings @f$\kappa@f$ hop periods
     * of inference work for @f$n@f$ processors, so the queue grows without bound and no
     * finite latency covers it. The other getters then describe one host block processed
     * by an idle pool, the best that can be said.
     *
     * @return True if the configuration is feasible
     */
    bool is_feasible() const;

    /**
     * @brief The host block in hops, @f$\rho = B/R@f$ as a reduced fraction
     */
    Rational get_block_hops() const;

    /**
     * @brief The inference time in hop periods, @f$\kappa@f$
     */
    double get_inference_hop_periods() const;

    /**
     * @brief Greatest common divisor
     *
     * @param a First non-negative integer
     * @param b Second non-negative integer
     * @return gcd(a, b), with gcd(0, b) = b
     */
    static int64_t greatest_common_divisor(int64_t a, int64_t b);

    /**
     * @brief Least common multiple, overflow-safe
     *
     * Divides before multiplying, @f$a / \gcd(a, b) \cdot b@f$, so the intermediate never
     * exceeds the result; the former `a * b / gcd` overflowed `int` once the product
     * passed @f$2^{31}@f$.
     *
     * @param a First non-negative integer
     * @param b Second non-negative integer
     * @return lcm(a, b), 0 if either is 0
     */
    static int64_t least_common_multiple(int64_t a, int64_t b);

    /**
     * @brief Minimum delay for repackaging constant host blocks into stream blocks
     *
     * Rath & Geier, corollary 3.8: @f$\Delta = P - \gcd(b, P)@f$. Replaces the
     * PortAudio-style loop over all multiples of @f$b@f$ below @f$\mathrm{lcm}(b, P)@f$.
     *
     * @param host_block_size Host block size in samples of the stream, @f$b > 0@f$
     * @param stream_size Samples per inference of the stream, @f$P > 0@f$
     * @return The delay in samples
     */
    static int64_t buffer_adaptation(int64_t host_block_size, int64_t stream_size);

    /**
     * @brief Minimum delay when the host block size varies or is fractional
     *
     * Rath & Geier, section 5: with flexible host blocks the last overlapping host block
     * may start one sample before the stream block ends, so @f$\Delta = P - 1@f$.
     *
     * @param stream_size Samples per inference of the stream, @f$P > 0@f$
     * @return The delay in samples
     */
    static int64_t buffer_adaptation_flexible(int64_t stream_size);

    /**
     * @brief Best rational approximation of a non-negative value
     *
     * Continued-fraction convergents, stopping at the first within a relative tolerance
     * of 1e-6 (the precision of the float host buffer size) or at the denominator bound.
     *
     * @param value The value to approximate, >= 0
     * @param max_denominator Largest denominator to consider
     * @return The approximation as a reduced fraction
     */
    static Rational rationalize(double value, int64_t max_denominator = 1 << 20);

    /**
     * @brief Synchronizes integer latencies across several output tensors
     *
     * With one output the value is kept. With several, every streamable output is raised
     * to the same whole number of hops, @f$\lceil \max_i L_i / P_i \rceil \cdot P_i@f$;
     * non-streamable outputs stay at 0. Same rule as before this class existed.
     *
     * @param latencies Integer latencies in samples, one per output tensor
     * @param output_sizes Postprocess output sizes (hops), one per output tensor
     * @return Synchronized latencies in samples, one per output tensor
     */
    static std::vector<unsigned int> sync_latencies(const std::vector<unsigned int>& latencies,
                                                    const std::vector<size_t>& output_sizes);

private:
    /**
     * @brief The timing of one constant host block
     */
    struct Timing {
        int64_t m_p = 1;          ///< Host block in hops, numerator
        int64_t m_q = 1;          ///< Host block in hops, denominator
        double m_kappa = 0.0;     ///< Inference time in hop periods
        double m_beta = 0.0;      ///< Blocking ratio
        int64_t m_n = 1;          ///< Parallel processors
        int64_t m_batch_cap = 0;  ///< Batches to consider; 0 = until the bound closes
    };

    /**
     * @brief Callbacks after submission at which a batch is collected, @f$d_m@f$
     *
     * @param host_blocks Inference time of the batch in host blocks, @f$(m+1)\tau@f$
     * @param beta Blocking ratio
     * @return @f$\max(0, \lceil host\_blocks - \beta \rceil)@f$
     */
    static int64_t batch_delay(double host_blocks, double beta);

    /**
     * @brief Batches of one host block's inferences, the horizon for an infeasible config
     */
    static int64_t burst_batches(const Timing& timing);

    /**
     * @brief @f$\max_m [\rho d_m - m n]@f$ as a numerator over @f$q@f$
     */
    static int64_t inference_latency_numerator(const Timing& timing);

    /**
     * @brief @f$\max_m [\lceil (d_m + 1) \rho \rceil - m n]@f$
     */
    static int64_t num_structs(const Timing& timing);

    /**
     * @brief Worst case over the smaller-block grid @f$j / H@f$, @f$1 \le j \le J@f$
     *
     * @param max_block Largest block on the grid, @f$J@f$
     * @param grid_hop The grid unit, @f$H@f$
     * @param kappa Inference time in hop periods
     * @param beta Blocking ratio
     * @param n Parallel processors
     * @param feasible Whether @f$\kappa < n@f$
     * @param latency_numerator Out: @f$\max_{m, j} [j\, d_m - m n H]@f$ (over @f$H@f$)
     * @param structs Out: the maximum slot count over the grid
     */
    static void smaller_buffers_worst_case(int64_t max_block,
                                           int64_t grid_hop,
                                           double kappa,
                                           double beta,
                                           int64_t n,
                                           bool feasible,
                                           int64_t& latency_numerator,
                                           int64_t& structs);

    std::vector<size_t> m_input_sizes;     ///< Preprocess input sizes (hops), 0 = non-streamable
    std::vector<size_t> m_output_sizes;    ///< Postprocess output sizes (hops), 0 = non-streamable
    std::vector<size_t> m_input_history;   ///< Past samples a receptive-field input peeks at
    Rational m_rho;                        ///< Host block in hops
    double m_kappa = 0.0;                  ///< Inference time in hop periods
    double m_beta = 0.0;                   ///< Blocking ratio
    int64_t m_n = 1;                       ///< Parallel processors
    int64_t m_max_hop = 1;                 ///< Largest streamable hop, @f$H@f$
    bool m_feasible = true;                ///< @f$\kappa < n@f$
    bool m_allow_smaller_buffers = false;  ///< Whether the smaller-block grid applies
    int64_t m_latency_numerator = 0;       ///< Fixed block: @f$\Lambda = numerator / q@f$
    int64_t m_smaller_numerator = 0;       ///< Grid: @f$\Lambda' = numerator / H@f$
    size_t m_num_structs = 0;              ///< Slots, max over fixed block and grid
    std::vector<size_t> m_send_buffer_sizes;  ///< Send ring sizes per input tensor
};

}  // namespace anira

#endif  // ANIRA_LATENCYCALCULATOR_H
