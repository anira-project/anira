#ifndef ANIRA_TEST_ABI_MEMORY_SUPPORT_H
#define ANIRA_TEST_ABI_MEMORY_SUPPORT_H
/*
 * A fake machine for the unified-memory probe (src/platform/memory_probe.h): the answer of each
 * device API, installed in place of the real probes for the scope of one test (set_api_probe),
 * with a count of the probes run. A test makes it before the context whose rows or plans it
 * reads, so a device provider's class does not depend on the machine the suite runs on (a
 * "coreml" row is unified on a Mac, unknown elsewhere).
 */
#include <anira/abi/enums.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "platform/memory_probe.h"

namespace anira_test {

/// One device of a fake API.
inline anira::platform::ProbedDevice fake_device(std::string name,
                                                 anira_memory_kind kind,
                                                 std::string evidence = "fake") {
    return anira::platform::ProbedDevice{.m_name = std::move(name),
                                         .m_kind = kind,
                                         .m_evidence = std::move(evidence)};
}

/// An API of the fake machine that could not be asked.
inline anira::platform::ApiAnswer fake_failure(std::string why) {
    anira::platform::ApiAnswer answer;
    answer.m_failure = std::move(why);
    return answer;
}

/// An API of the fake machine with these devices, in the API's order.
inline anira::platform::ApiAnswer fake_devices(std::vector<anira::platform::ProbedDevice> devices) {
    anira::platform::ApiAnswer answer;
    answer.m_devices = std::move(devices);
    return answer;
}

/// The fake machine of one test: every API fails ("not on the fake machine") until the test
/// gives it an answer. One at a time per process.
class FakeMachine {
public:
    FakeMachine() {
        for (anira::platform::ApiAnswer& answer : m_answers) {
            answer = fake_failure("not on the fake machine");
        }
        s_current = this;
        anira::platform::set_api_probe(&FakeMachine::probe);
    }
    ~FakeMachine() {
        anira::platform::set_api_probe(nullptr);
        s_current = nullptr;
    }
    FakeMachine(const FakeMachine&) = delete;
    FakeMachine& operator=(const FakeMachine&) = delete;
    FakeMachine(FakeMachine&&) = delete;
    FakeMachine& operator=(FakeMachine&&) = delete;

    /// What `api` answers from now on.
    FakeMachine& set(anira::platform::DeviceApi api, anira::platform::ApiAnswer answer) {
        m_answers.at(static_cast<size_t>(api)) = std::move(answer);
        return *this;
    }
    /// How often `api` was probed.
    int probes(anira::platform::DeviceApi api) const {
        return m_probes.at(static_cast<size_t>(api)).load();
    }
    /// How often any API was probed.
    int probes() const {
        int total = 0;
        for (const std::atomic<int>& count : m_probes) { total += count.load(); }
        return total;
    }

private:
    static anira::platform::ApiAnswer probe(anira::platform::DeviceApi api) {
        FakeMachine* machine = s_current;
        if (machine == nullptr) { return fake_failure("no fake machine"); }
        machine->m_probes.at(static_cast<size_t>(api)).fetch_add(1);
        return machine->m_answers.at(static_cast<size_t>(api));
    }

    inline static FakeMachine* s_current = nullptr;
    std::array<anira::platform::ApiAnswer, anira::platform::k_device_apis.size()> m_answers;
    std::array<std::atomic<int>, anira::platform::k_device_apis.size()> m_probes{};
};

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_MEMORY_SUPPORT_H
