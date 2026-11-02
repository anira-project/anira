// The unified-memory probe (src/platform/memory_probe.h) against the machine it runs on: every
// API answers within the probe's timeout, either a failure that says why or its devices, each
// named, with the API's answer and a kind that answer settles. Which kinds a machine has is
// its own: the CI runners assert this consistency alone. A machine whose devices are known
// names them in ANIRA_TEST_MEMORY_KINDS, "<api>=<kind>" pairs split by ';' over the API names
// and the kind words of memory_probe.h ("CUDA=dedicated;OpenCL=dedicated;Vulkan=unified" on a
// box with one discrete NVIDIA card and Mesa's lavapipe), and every device of each API named
// must then have that kind. The Vulkan rule's cross-check is checked over synthetic devices.
#include <anira/abi/enums.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "platform/memory_probe.h"

namespace {

using anira::platform::ApiAnswer;
using anira::platform::DeviceApi;
using anira::platform::ProbedDevice;

// The kinds ANIRA_TEST_MEMORY_KINDS names, as (API name, kind word) pairs.
std::vector<std::pair<std::string, std::string>> expected_kinds() {
    std::vector<std::pair<std::string, std::string>> pairs;
    const char* text = std::getenv("ANIRA_TEST_MEMORY_KINDS");  // NOLINT(concurrency-mt-unsafe)
    if (text == nullptr) { return pairs; }
    std::string_view rest(text);
    while (!rest.empty()) {
        const size_t end = rest.find(';');
        const std::string_view pair = rest.substr(0, end);
        rest = end == std::string_view::npos ? std::string_view() : rest.substr(end + 1);
        const size_t equals = pair.find('=');
        if (equals == std::string_view::npos) { continue; }
        pairs.emplace_back(std::string(pair.substr(0, equals)),
                           std::string(pair.substr(equals + 1)));
    }
    return pairs;
}

}  // namespace

TEST(MemoryProbe, EveryApiAnswersConsistently) {
    for (const DeviceApi api : anira::platform::k_device_apis) {
        const char* name = anira::platform::device_api_name(api);
        SCOPED_TRACE(name);
        const auto start = std::chrono::steady_clock::now();
        const ApiAnswer answer = anira::platform::probe_api(api);
        const auto took = std::chrono::steady_clock::now() - start;
        EXPECT_LT(took, anira::platform::k_probe_timeout + std::chrono::seconds(1))
            << "the wait is bounded";
        if (!answer.m_failure.empty()) {
            EXPECT_TRUE(answer.m_devices.empty()) << "a failure lists no device";
            std::cout << name << ": not answered: " << answer.m_failure << "\n";
            continue;
        }
        if (answer.m_devices.empty()) { std::cout << name << ": no device\n"; }
        for (size_t i = 0; i < answer.m_devices.size(); ++i) {
            const ProbedDevice& device = answer.m_devices[i];
            std::cout << name << " device " << i << " '" << device.m_name
                      << "': " << anira::platform::memory_kind_word(device.m_kind) << " ("
                      << device.m_evidence << ")\n";
            EXPECT_FALSE(device.m_evidence.empty()) << "the API's answer is kept";
            EXPECT_TRUE(device.m_kind == ANIRA_MEMORY_UNKNOWN ||
                        device.m_kind == ANIRA_MEMORY_UNIFIED ||
                        device.m_kind == ANIRA_MEMORY_DEDICATED);
        }
    }
    // A probe that cannot run here says so, in its words.
#if !defined(__APPLE__)
    EXPECT_EQ(anira::platform::probe_api(DeviceApi::Metal).m_failure,
              "Metal exists on Apple platforms alone");
#endif
#if !defined(_WIN32)
    EXPECT_EQ(anira::platform::probe_api(DeviceApi::D3d12).m_failure,
              "D3D12 exists on Windows alone");
#endif
}

TEST(MemoryProbe, TheMachinesKnownDevicesHaveTheirKinds) {
    const std::vector<std::pair<std::string, std::string>> expected = expected_kinds();
    if (expected.empty()) { GTEST_SKIP() << "ANIRA_TEST_MEMORY_KINDS names no device"; }
    for (const auto& [api_name, kind_word] : expected) {
        SCOPED_TRACE(api_name);
        bool known = false;
        for (const DeviceApi api : anira::platform::k_device_apis) {
            if (api_name != anira::platform::device_api_name(api)) { continue; }
            known = true;
            const ApiAnswer answer = anira::platform::probe_api(api);
            ASSERT_TRUE(answer.m_failure.empty()) << answer.m_failure;
            ASSERT_FALSE(answer.m_devices.empty());
            for (const ProbedDevice& device : answer.m_devices) {
                EXPECT_EQ(anira::platform::memory_kind_word(device.m_kind), kind_word)
                    << device.m_name << ": " << device.m_evidence;
            }
        }
        EXPECT_TRUE(known) << "an API name of memory_probe.h";
    }
}

// The Vulkan rule over synthetic devices: the type, cross-checked with the memory types and
// heaps (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 1, HOST_VISIBLE_BIT 2; VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
// 1).
TEST(MemoryProbe, TheVulkanTypeIsCrossCheckedWithTheHeaps) {
    constexpr int32_t k_integrated = 1;
    constexpr int32_t k_discrete = 2;
    constexpr int32_t k_virtual = 3;
    constexpr int32_t k_cpu = 4;
    const std::vector<uint32_t> mappable{1U, 1U | 2U};  // device-local, then also host-visible
    const std::vector<uint32_t> unmappable{1U, 2U};     // never both
    const std::vector<uint32_t> split_heaps{1U, 0U};    // VRAM and a host heap
    const std::vector<uint32_t> local_heaps{1U};

    const ProbedDevice lavapipe =
        anira::platform::vulkan_device("llvmpipe", k_cpu, mappable, local_heaps);
    EXPECT_EQ(lavapipe.m_kind, ANIRA_MEMORY_UNIFIED);
    EXPECT_EQ(lavapipe.m_name, "llvmpipe");
    EXPECT_EQ(lavapipe.m_evidence, "type CPU; memory type 1 is device-local and host-visible");
    EXPECT_EQ(anira::platform::vulkan_device("igpu", k_integrated, mappable, local_heaps).m_kind,
              ANIRA_MEMORY_UNIFIED);
    const ProbedDevice contradicted =
        anira::platform::vulkan_device("igpu", k_integrated, unmappable, local_heaps);
    EXPECT_EQ(contradicted.m_kind, ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(contradicted.m_evidence,
              "type integrated GPU, but no memory type is both device-local and host-visible");

    const ProbedDevice discrete =
        anira::platform::vulkan_device("dgpu", k_discrete, mappable, split_heaps);
    EXPECT_EQ(discrete.m_kind, ANIRA_MEMORY_DEDICATED);
    EXPECT_EQ(discrete.m_evidence, "type discrete GPU; heap 1 is not device-local (the host's)");
    EXPECT_EQ(anira::platform::vulkan_device("dgpu", k_discrete, mappable, local_heaps).m_kind,
              ANIRA_MEMORY_UNKNOWN)
        << "a discrete GPU without a host heap is not what its type says";

    EXPECT_EQ(anira::platform::vulkan_device("vgpu", k_virtual, mappable, local_heaps).m_kind,
              ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(anira::platform::vulkan_device("other", 0, mappable, local_heaps).m_evidence,
              "type other (0)");
}

// A test probe replaces the real ones, on the probe's own thread; clearing it restores them.
TEST(MemoryProbe, AnInstalledProbeAnswersInPlaceOfTheApis) {
    anira::platform::set_api_probe([](DeviceApi api) {
        ApiAnswer answer;
        answer.m_devices.push_back(ProbedDevice{.m_name = anira::platform::device_api_name(api),
                                                .m_kind = ANIRA_MEMORY_UNIFIED,
                                                .m_evidence = "fake"});
        return answer;
    });
    const ApiAnswer faked = anira::platform::probe_api(DeviceApi::D3d12);
    anira::platform::set_api_probe(nullptr);
    ASSERT_EQ(faked.m_devices.size(), 1U);
    EXPECT_EQ(faked.m_devices[0].m_name, "D3D12");
    EXPECT_EQ(faked.m_devices[0].m_evidence, "fake");
#if !defined(_WIN32)
    EXPECT_FALSE(anira::platform::probe_api(DeviceApi::D3d12).m_failure.empty())
        << "the real probe again";
#endif
}

// A probe that does not answer within the timeout: the caller gets the failure at the timeout,
// not when the probe returns, and the probe finishes on its own afterwards (its thread holds
// the shared state, not the caller's).
TEST(MemoryProbe, AProbeThatHangsAnswersTheTimeout) {
    static std::atomic<bool> s_finished{false};
    s_finished = false;
    anira::platform::set_api_probe([](DeviceApi /*api*/) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        s_finished = true;
        return ApiAnswer{};
    });
    const auto start = std::chrono::steady_clock::now();
    const ApiAnswer answer =
        anira::platform::probe_api(DeviceApi::Vulkan, std::chrono::milliseconds(20));
    const auto took = std::chrono::steady_clock::now() - start;
    EXPECT_EQ(answer.m_failure, "the Vulkan probe did not answer within 20 ms");
    EXPECT_TRUE(answer.m_devices.empty());
    EXPECT_LT(took, std::chrono::milliseconds(250)) << "answered at the timeout";
    // The abandoned probe ends by itself (bounded wait), and only then goes the fake.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!s_finished && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(s_finished.load());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // its thread's last steps
    anira::platform::set_api_probe(nullptr);
}
