// The warm-up pass's estimator (src/scheduler/Warmup.h): when UNTIL_STABLE and FIXED n stop, the
// figure (the maximum of the last four times) and its median, the clock-step floor of every time
// and of the criterion's spread, and the clock step itself. Plain arithmetic over scripted times:
// nothing here depends on the machine.
#include <anira/abi/enums.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "scheduler/Warmup.h"

using anira::WarmupEstimator;
using anira::WarmupFigure;
using anira::WarmupOutcome;

namespace {

constexpr int64_t k_ms = 1000000;  ///< ns per ms
constexpr int64_t k_us = 1000;     ///< ns per us
constexpr int64_t k_fine_step = 10;

/// Feeds `times` (ns) until the estimator stops or the script ends; the times it took.
uint32_t feed(WarmupEstimator& estimator, const std::vector<int64_t>& times) {
    uint32_t taken = 0;
    for (const int64_t time : times) {
        if (estimator.done()) { break; }
        estimator.add(time);
        ++taken;
    }
    return taken;
}

/// `count` times repeating `pattern`.
std::vector<int64_t> repeat(const std::vector<int64_t>& pattern, size_t count) {
    std::vector<int64_t> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) { out.push_back(pattern[i % pattern.size()]); }
    return out;
}

}  // namespace

// Four equal times are stable at once: the earliest stop is the fourth inference.
TEST(Warmup, UntilStableStopsAtTheFourthOfFourEqualTimes) {
    WarmupEstimator estimator(ANIRA_WARMUP_UNTIL_STABLE, 0, k_fine_step);
    EXPECT_EQ(feed(estimator, repeat({2 * k_ms}, 10)), 4U);
    ASSERT_TRUE(estimator.done());
    const WarmupFigure figure = estimator.figure();
    EXPECT_EQ(figure.m_iterations, 4U);
    EXPECT_EQ(figure.m_outcome, WarmupOutcome::Stable);
    EXPECT_DOUBLE_EQ(figure.m_measured_ms, 2.0);
    EXPECT_DOUBLE_EQ(figure.m_median_ms, 2.0);
}

// The cold start leaves the window: a first inference of 10 ms and 1 ms after it stop at the
// fifth, and the figure is the window's 1 ms, not the first time.
TEST(Warmup, UntilStableLeavesTheColdStartOut) {
    WarmupEstimator estimator(ANIRA_WARMUP_UNTIL_STABLE, 0, k_fine_step);
    std::vector<int64_t> times{10 * k_ms};
    for (int i = 0; i < 31; ++i) { times.push_back(k_ms); }
    EXPECT_EQ(feed(estimator, times), 5U);
    const WarmupFigure figure = estimator.figure();
    EXPECT_EQ(figure.m_outcome, WarmupOutcome::Stable);
    EXPECT_DOUBLE_EQ(figure.m_measured_ms, 1.0);
}

// The window may stray 20 % of its median: 1.0, 1.15, 0.85, 1.0 ms agree (median 1.0), and the
// figure is their maximum, unrounded, without a margin; 1.0, 1.3, 1.0, 1.0 do not.
TEST(Warmup, TheSpreadIsTwentyPercentOfTheMedian) {
    WarmupEstimator inside(ANIRA_WARMUP_UNTIL_STABLE, 0, k_fine_step);
    EXPECT_EQ(feed(inside, {k_ms, 1150 * k_us, 850 * k_us, k_ms}), 4U);
    EXPECT_EQ(inside.figure().m_outcome, WarmupOutcome::Stable);
    EXPECT_DOUBLE_EQ(inside.figure().m_measured_ms, 1.15);
    EXPECT_DOUBLE_EQ(inside.figure().m_median_ms, 1.0);

    WarmupEstimator outside(ANIRA_WARMUP_UNTIL_STABLE, 0, k_fine_step);
    feed(outside, {k_ms, 1300 * k_us, k_ms, k_ms});
    EXPECT_FALSE(outside.done());
}

// Times that never agree stop at the cap, CAPPED, the figure the last four's maximum.
TEST(Warmup, AnUnsettledPassStopsAtTheCap) {
    WarmupEstimator estimator(ANIRA_WARMUP_UNTIL_STABLE, 0, k_fine_step);
    EXPECT_EQ(feed(estimator, repeat({k_ms, 5 * k_ms}, 100)), WarmupEstimator::k_cap);
    const WarmupFigure figure = estimator.figure();
    EXPECT_EQ(figure.m_iterations, WarmupEstimator::k_cap);
    EXPECT_EQ(figure.m_outcome, WarmupOutcome::Capped);
    EXPECT_DOUBLE_EQ(figure.m_measured_ms, 5.0);
    EXPECT_DOUBLE_EQ(figure.m_median_ms, 3.0);
}

// A model of microseconds on a clock of 5 us: 5, 10, 10, 15 us never lie within 20 % of their
// median, and agree within one clock step. The floor turns a CAPPED pass STABLE.
TEST(Warmup, TheClockStepFloorsTheSpread) {
    const std::vector<int64_t> coarse = repeat({5 * k_us, 10 * k_us, 10 * k_us, 15 * k_us}, 100);
    WarmupEstimator fine_clock(ANIRA_WARMUP_UNTIL_STABLE, 0, k_fine_step);
    EXPECT_EQ(feed(fine_clock, coarse), WarmupEstimator::k_cap);
    EXPECT_EQ(fine_clock.figure().m_outcome, WarmupOutcome::Capped);

    WarmupEstimator coarse_clock(ANIRA_WARMUP_UNTIL_STABLE, 0, 5 * k_us);
    EXPECT_EQ(feed(coarse_clock, coarse), 4U);
    EXPECT_EQ(coarse_clock.figure().m_outcome, WarmupOutcome::Stable);
    EXPECT_DOUBLE_EQ(coarse_clock.figure().m_measured_ms, 0.015);
}

// No time is below the clock step: a reading of 0 (an inference inside one step) counts as one.
TEST(Warmup, EveryTimeIsAtLeastOneClockStep) {
    WarmupEstimator estimator(ANIRA_WARMUP_FIXED, 2, 5 * k_us);
    feed(estimator, {0, 0});
    EXPECT_DOUBLE_EQ(estimator.figure().m_measured_ms, 0.005);
    EXPECT_DOUBLE_EQ(estimator.figure().m_median_ms, 0.005);
}

// FIXED n runs n, whatever the times; below four the figure is the maximum of all n, the cold
// start included, and the median that of an odd window is its middle time.
TEST(Warmup, FixedRunsItsCountAndTakesTheLastFour) {
    WarmupEstimator three(ANIRA_WARMUP_FIXED, 3, k_fine_step);
    EXPECT_EQ(feed(three, {10 * k_ms, 3 * k_ms, 2 * k_ms, k_ms}), 3U);
    EXPECT_EQ(three.figure().m_outcome, WarmupOutcome::Fixed);
    EXPECT_EQ(three.figure().m_iterations, 3U);
    EXPECT_DOUBLE_EQ(three.figure().m_measured_ms, 10.0);
    EXPECT_DOUBLE_EQ(three.figure().m_median_ms, 3.0);

    WarmupEstimator six(ANIRA_WARMUP_FIXED, 6, k_fine_step);
    EXPECT_EQ(feed(six, {10 * k_ms, k_ms, k_ms, k_ms, k_ms, 2 * k_ms, k_ms}), 6U);
    EXPECT_EQ(six.figure().m_outcome, WarmupOutcome::Fixed);
    EXPECT_DOUBLE_EQ(six.figure().m_measured_ms, 2.0) << "the last four, not the first time";
    EXPECT_DOUBLE_EQ(six.figure().m_median_ms, 1.0);
}

// FIXED 0 is a pass that is over before it starts: no time, no figure.
TEST(Warmup, FixedZeroIsDoneBeforeItsFirstInference) {
    const WarmupEstimator estimator(ANIRA_WARMUP_FIXED, 0, k_fine_step);
    EXPECT_TRUE(estimator.done());
    EXPECT_EQ(estimator.figure().m_iterations, 0U);
    EXPECT_DOUBLE_EQ(estimator.figure().m_measured_ms, 0.0);
}

// The clock step is positive and measured once: every call answers the same.
TEST(Warmup, TheClockStepIsMeasuredOnce) {
    const int64_t step = anira::clock_step_ns();
    EXPECT_GE(step, 1);
    EXPECT_EQ(anira::clock_step_ns(), step);
}
