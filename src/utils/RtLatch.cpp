// The process-wide site latches of include/anira/utils/RtLatch.h and the summary interval.
#include <anira/utils/RtLatch.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace anira::detail {

namespace {

// Constant-initialized (every RtLatch member is an atomic with a constexpr initializer):
// no registration, no guard, readable from the first real-time site that fires.
std::array<RtLatch, static_cast<size_t>(RtSite::Count)> s_sites{};

// The drain's summary interval: a runtime constant the tests lower.
std::atomic<uint32_t> s_summary_interval_ms{10000};

}  // namespace

RtLatch& rt_site(RtSite site) noexcept {
    return s_sites[static_cast<size_t>(site)];
}

void set_rt_summary_interval_ms(uint32_t ms) noexcept {
    s_summary_interval_ms.store(ms, std::memory_order_relaxed);
}

uint32_t rt_summary_interval_ms() noexcept {
    return s_summary_interval_ms.load(std::memory_order_relaxed);
}

}  // namespace anira::detail
