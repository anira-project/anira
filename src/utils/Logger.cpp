#include <anira/utils/Logger.h>
#include <tanh/core/Logger.h>

#include <atomic>

namespace anira::detail {

std::atomic<thl::Logger::rt::Queue*>& rt_log_queue_slot() noexcept {
    // Constant-initialised: valid before any static constructor and after every
    // static destructor, so a real-time producer never finds a torn-down object.
    static std::atomic<thl::Logger::rt::Queue*> slot{nullptr};
    return slot;
}

}  // namespace anira::detail
