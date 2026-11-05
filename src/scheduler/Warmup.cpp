#include "Warmup.h"

#include <anira/abi/enums.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace anira {

namespace {

constexpr double k_ns_per_ms = 1e6;
/// The steps clock_step_ns() takes the smallest of.
constexpr int k_clock_steps = 8;

/// The median of the first `count` entries of a window (sorted in place).
double median_of(std::array<int64_t, WarmupEstimator::k_window>& times, uint32_t count) noexcept {
    std::sort(times.begin(), times.begin() + count);
    if (count % 2 == 1) { return static_cast<double>(times[count / 2]); }
    return (static_cast<double>(times[(count / 2) - 1]) + static_cast<double>(times[count / 2])) /
           2.0;
}

int64_t measure_clock_step() noexcept {
    using clock = std::chrono::steady_clock;
    int64_t smallest = 0;
    clock::time_point last = clock::now();
    for (int steps = 0; steps < k_clock_steps;) {
        const clock::time_point now = clock::now();
        const int64_t step =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count();
        if (step <= 0) { continue; }
        smallest = smallest == 0 ? step : std::min(smallest, step);
        last = now;
        ++steps;
    }
    return std::max<int64_t>(smallest, 1);
}

}  // namespace

WarmupEstimator::WarmupEstimator(anira_warmup_mode mode,
                                 uint32_t iterations,
                                 int64_t clock_step_ns) noexcept
    : m_mode(mode)
    , m_iterations(iterations)
    , m_clock_step_ns(std::max<int64_t>(clock_step_ns, 1))
    , m_done(mode == ANIRA_WARMUP_FIXED && iterations == 0)
    , m_outcome(mode == ANIRA_WARMUP_FIXED ? WarmupOutcome::Fixed : WarmupOutcome::Stable) {}

uint32_t WarmupEstimator::window(std::array<int64_t, k_window>& window) const noexcept {
    const uint32_t count = std::min(m_count, k_window);
    for (uint32_t i = 0; i < count; ++i) { window[i] = m_last[(m_count - count + i) % k_window]; }
    return count;
}

bool WarmupEstimator::add(int64_t inference_ns) noexcept {
    if (m_done) { return true; }
    m_last[m_count % k_window] = std::max(inference_ns, m_clock_step_ns);
    ++m_count;
    if (m_mode == ANIRA_WARMUP_FIXED) {
        m_done = m_count >= m_iterations;
        return m_done;
    }
    if (m_count >= k_window) {
        std::array<int64_t, k_window> times{};
        const uint32_t count = window(times);
        const double median = median_of(times, count);
        const double spread = std::max(k_spread * median, static_cast<double>(m_clock_step_ns));
        const bool stable = std::all_of(times.begin(), times.begin() + count, [&](int64_t time) {
            return std::abs(static_cast<double>(time) - median) <= spread;
        });
        if (stable) {
            m_outcome = WarmupOutcome::Stable;
            m_done = true;
        } else if (m_count >= k_cap) {
            m_outcome = WarmupOutcome::Capped;
            m_done = true;
        }
    }
    return m_done;
}

WarmupFigure WarmupEstimator::figure() const noexcept {
    WarmupFigure figure;
    figure.m_iterations = m_count;
    figure.m_outcome = m_outcome;
    std::array<int64_t, k_window> times{};
    const uint32_t count = window(times);
    if (count == 0) { return figure; }
    figure.m_measured_ms =
        static_cast<double>(*std::max_element(times.begin(), times.begin() + count)) / k_ns_per_ms;
    figure.m_median_ms = median_of(times, count) / k_ns_per_ms;
    return figure;
}

int64_t clock_step_ns() {
    static const int64_t k_step = measure_clock_step();
    return k_step;
}

}  // namespace anira
