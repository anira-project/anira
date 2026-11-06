#ifndef ANIRA_SCHEDULER_WARMUP_H
#define ANIRA_SCHEDULER_WARMUP_H
/*
 * The warm-up pass's estimator: when one plan's pass stops and the figure it gives. A prepare
 * whose contract warms up (anira_warmup_mode, anira_contract_hard_set_warmup) runs every plan
 * through the session on the inference threads (Core::warm_up_plan); each inference's time is
 * the job's on its thread (SessionElement::ThreadSafeStruct::m_inference_ns), never below one
 * step of the clock that took it. The warm-up pass also measures the inference time; a measured
 * budget is the maximum of its last four inferences; under an explicit budget the timings are
 * only reported.
 *
 * - UNTIL_STABLE stops at the first inference i >= 4 whose last four times all lie within
 *   max(20 % of their median, one clock step) of that median (STABLE), or at i = 32 without that
 *   (CAPPED).
 * - FIXED n runs n inferences (FIXED); the figure of n < 4 is the maximum of all n, the cold
 *   start included.
 * - The figure is the maximum of the last min(4, i) times, unrounded, with no margin; their
 *   median is reported beside it.
 *
 * Why the window and not the pass's maximum: the first inferences carry the cold start
 * (allocations, a runtime's profiling, caches), which a maximum over the pass would turn into
 * latency. Why the clock-step floor: a model of a few microseconds on a clock of 5 us (a browser's
 * coarsened timer) gives times like 5, 10, 10, 15 us, which a pure 20 % rule never accepts.
 */
#include <anira/abi/enums.h>

#include <array>
#include <chrono>
#include <cstdint>

namespace anira {

/// What one plan's warm-up pass measured.
struct WarmupFigure {
    uint32_t m_iterations = 0;  ///< the inferences the pass ran
    /// How it ended: ANIRA_WARMUP_OUTCOME_FIXED, _STABLE or _CAPPED (a pass never ends NONE or
    /// REUSED, which are the plan row's words for no pass at all).
    anira_warmup_outcome m_outcome = ANIRA_WARMUP_OUTCOME_FIXED;
    double m_measured_ms = 0.0;  ///< the maximum of the last min(4, m_iterations) times, in ms
    double m_median_ms = 0.0;    ///< their median, in ms
    double m_fastest_ms = 0.0;   ///< their minimum, in ms (what a CAPPED pass's record spans)
};

/// A model's figure: what a warm-up pass measured for one key of the core's figure index
/// (FigureKey), kept while the core lives.
struct ModelFigure {
    WarmupFigure m_figure;                                 ///< the pass's figure and outcome
    anira_warmup_mode m_mode = ANIRA_WARMUP_UNTIL_STABLE;  ///< the warm-up that measured it
    std::chrono::steady_clock::time_point m_when;          ///< when (for the log)
};

/**
 * @brief One plan's warm-up pass: when it stops, and the figure it gives
 *
 * Fed one inference time at a time, in the order the pass ran them; the contract's warm-up
 * decides the stop (the file comment). Plain arithmetic: no clock is read here.
 */
class WarmupEstimator {
public:
    static constexpr uint32_t k_window = 4;  ///< the times the criterion and the figure look at
    static constexpr uint32_t k_cap = 32;    ///< UNTIL_STABLE's last inference
    static constexpr double k_spread = 0.2;  ///< the share of the median the window may stray

    /**
     * @brief The estimator of one pass
     *
     * @param mode The contract's warm-up: ANIRA_WARMUP_UNTIL_STABLE or ANIRA_WARMUP_FIXED (a pass
     * runs for no other)
     * @param iterations FIXED's count (not read for UNTIL_STABLE); 0 is a pass that is done
     * before its first inference
     * @param clock_step_ns The clock's step (clock_step_ns()): the floor of every time and of
     * the criterion's spread
     */
    WarmupEstimator(anira_warmup_mode mode, uint32_t iterations, int64_t clock_step_ns) noexcept;

    /**
     * @brief Adds the time of the pass's next inference
     *
     * @param inference_ns The inference's time on its thread, in ns (raised to the clock step)
     * @return true when the pass stops with it; nothing may be added after
     */
    bool add(int64_t inference_ns) noexcept;

    /// Whether the pass stopped (also true before the first inference of a FIXED 0 pass).
    bool done() const noexcept { return m_done; }

    /// The pass's figure so far: final once done() is true; all zeros before the first time.
    WarmupFigure figure() const noexcept;

private:
    /// The last min(k_window, m_count) times, oldest first, in `window`; returns their count.
    uint32_t window(std::array<int64_t, k_window>& window) const noexcept;

    anira_warmup_mode m_mode;
    uint32_t m_iterations;
    int64_t m_clock_step_ns;
    std::array<int64_t, k_window> m_last{};  ///< a ring of the last k_window times
    uint32_t m_count = 0;
    bool m_done;
    anira_warmup_outcome m_outcome;
};

/**
 * @brief The step of std::chrono::steady_clock, in ns, as this process sees it
 *
 * The smallest positive difference of successive readings over several steps, measured once per
 * process at the first call (thread-safe): the clock's resolution together with the cost of one
 * reading, natively tens of nanoseconds; in a browser its coarsening (microseconds). Never below
 * 1 ns.
 */
int64_t clock_step_ns();

}  // namespace anira

#endif  // ANIRA_SCHEDULER_WARMUP_H
