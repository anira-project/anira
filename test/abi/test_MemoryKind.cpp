// A context's memory kinds (src/capi/memory.h) and the provider rule (engine::provider_rule) on
// fake machines: which device each provider maps to, how a device's answer becomes the memory
// kind (an index out of range, every adapter of a self-choosing runtime agreeing or not, an
// API that could not be asked), what is asked when, and the class, mover and reason the rule
// gives each kind. The rows and plans built from them are AbiEngine's (test_Engine.cpp).
#include <anira/abi/enums.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "capi/memory.h"
#include "engines/Adapter.h"
#include "memory_support.h"
#include "platform/memory_probe.h"

namespace {

using anira::engine::DeviceMemory;
using anira::platform::DeviceApi;
using anira_test::fake_device;
using anira_test::fake_devices;
using Options = std::vector<std::pair<std::string, std::string>>;

// A machine with every API answering: a dedicated and a unified CUDA device, one dedicated HIP
// device, a unified and a dedicated DXGI adapter, a unified Metal device, Vulkan devices of
// both kinds (lavapipe beside a discrete card), two unified OpenCL devices.
void fill(anira_test::FakeMachine& machine) {
    machine.set(DeviceApi::Cuda,
                fake_devices({fake_device("A", ANIRA_MEMORY_DEDICATED, "integrated 0"),
                              fake_device("B", ANIRA_MEMORY_UNIFIED, "integrated 1")}));
    machine.set(DeviceApi::Hip, fake_devices({fake_device("H", ANIRA_MEMORY_DEDICATED)}));
    machine.set(DeviceApi::D3d12,
                fake_devices({fake_device("D0", ANIRA_MEMORY_UNIFIED, "UMA 1"),
                              fake_device("D1", ANIRA_MEMORY_DEDICATED, "UMA 0")}));
    machine.set(DeviceApi::Metal, fake_devices({fake_device("M", ANIRA_MEMORY_UNIFIED)}));
    machine.set(DeviceApi::Vulkan,
                fake_devices({fake_device("llvmpipe", ANIRA_MEMORY_UNIFIED, "type CPU"),
                              fake_device("dGPU", ANIRA_MEMORY_DEDICATED, "type discrete")}));
    machine.set(DeviceApi::OpenCl,
                fake_devices({fake_device("C0", ANIRA_MEMORY_UNIFIED),
                              fake_device("C1", ANIRA_MEMORY_UNIFIED)}));
}

}  // namespace

// The device of each provider, and its kind: a numbered device by ONNX Runtime's device_id (0
// without it, and for any other engine, whose options are its own vocabulary), out of range
// unknown; Core ML and ExecuTorch's MPS the Metal default device; a self-choosing runtime
// (WebGPU through Dawn, LiteRT's GPU accelerator, Vulkan) every device of its platform's API,
// unknown when they disagree, naming each; an NPU and a custom provider unknown, no API asked; a
// provider over host memory unified, no API asked.
TEST(AbiMemoryKind, EachProviderMapsToItsDevice) {
    anira_test::FakeMachine machine;
    fill(machine);
    anira::capi::MemoryKinds kinds;
    const auto of = [&kinds](anira_engine engine,
                             anira_provider provider,
                             std::string_view provider_id,
                             const Options& options = {}) {
        return kinds.memory_of(engine, provider, provider_id, options);
    };

    for (const auto& [provider, id] : std::vector<std::pair<anira_provider, std::string_view>>{
             {ANIRA_PROVIDER_CPU, ""},
             {ANIRA_PROVIDER_XNNPACK, ""},
             {ANIRA_PROVIDER_CUSTOM, "DnnlExecutionProvider"}}) {
        const DeviceMemory host = of(ANIRA_ENGINE_ONNXRUNTIME, provider, id);
        EXPECT_EQ(host.m_kind, ANIRA_MEMORY_UNIFIED) << id;
        EXPECT_TRUE(host.m_why.empty());
    }
    EXPECT_EQ(machine.probes(), 0) << "a provider over host memory asks no API";

    // CUDA by device_id.
    DeviceMemory memory = of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA, "");
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_DEDICATED);
    EXPECT_EQ(memory.m_why, "CUDA device 0 'A': integrated 0");
    memory = of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA, "", {{"device_id", "1"}});
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_UNIFIED);
    EXPECT_EQ(memory.m_why, "CUDA device 1 'B': integrated 1");
    memory = of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA, "", {{"device_id", "7"}});
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(memory.m_why, "CUDA has no device 7 (2 listed)");
    EXPECT_EQ(of(ANIRA_ENGINE_CUSTOM, ANIRA_PROVIDER_CUDA, "", {{"device_id", "1"}}).m_kind,
              ANIRA_MEMORY_DEDICATED)
        << "a custom engine's options are its own: device 0";
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME,
                 ANIRA_PROVIDER_CUSTOM,
                 "TensorrtExecutionProvider",
                 {{"device_id", "1"}})
                  .m_kind,
              ANIRA_MEMORY_UNIFIED);
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUSTOM, "NvTensorRTRTXExecutionProvider")
                  .m_kind,
              ANIRA_MEMORY_DEDICATED);
    EXPECT_EQ(machine.probes(DeviceApi::Cuda), 1) << "asked once, kept";

    // HIP.
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUSTOM, "ROCMExecutionProvider").m_why,
              "HIP device 0 'H': fake");
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME,
                 ANIRA_PROVIDER_CUSTOM,
                 "MIGraphXExecutionProvider",
                 {{"device_id", "1"}})
                  .m_why,
              "HIP has no device 1 (1 listed)");

    // DirectML's DXGI adapter.
    memory = of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_DIRECTML, "", {{"device_id", "1"}});
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_DEDICATED);
    EXPECT_EQ(memory.m_why, "D3D12 device 1 'D1': UMA 0");
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_DIRECTML, "").m_kind,
              ANIRA_MEMORY_UNIFIED);

    // Metal's default device.
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_COREML, "").m_why,
              "Metal device 0 'M': fake");
    EXPECT_EQ(of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_COREML, "").m_kind, ANIRA_MEMORY_UNIFIED);
    EXPECT_EQ(of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_CUSTOM, "MPSBackend").m_kind,
              ANIRA_MEMORY_UNIFIED);

    // Every Vulkan device: they disagree.
    const std::string mixed =
        "the runtime picks its device among Vulkan devices of different kinds, 'llvmpipe': type "
        "CPU (unified); 'dGPU': type discrete (dedicated)";
    memory = of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_VULKAN, "");
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(memory.m_why, mixed);

    // The self-choosing runtimes, over their platform's API.
    const DeviceMemory webgpu = of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU, "");
    const DeviceMemory litert = of(ANIRA_ENGINE_LITERT, ANIRA_PROVIDER_CUSTOM, "gpu");
#if defined(__APPLE__)
    EXPECT_EQ(webgpu.m_why, "Metal device 0 'M': fake");
    EXPECT_EQ(litert.m_kind, ANIRA_MEMORY_UNIFIED);
#elif defined(_WIN32)
    EXPECT_EQ(webgpu.m_kind, ANIRA_MEMORY_UNKNOWN);
    EXPECT_NE(webgpu.m_why.find("among D3D12 devices of different kinds"), std::string::npos);
    EXPECT_EQ(litert.m_why, webgpu.m_why);
#elif defined(__ANDROID__)
    EXPECT_EQ(webgpu.m_why, mixed);
    EXPECT_EQ(litert.m_kind, ANIRA_MEMORY_UNIFIED);
    EXPECT_EQ(litert.m_why, "every OpenCL device: 'C0': fake (unified); 'C1': fake (unified)");
#else
    EXPECT_EQ(webgpu.m_why, mixed);
    EXPECT_EQ(litert.m_why, mixed);
#endif

    // No API answers for an NPU or a custom provider.
    const int asked = machine.probes();
    memory = of(ANIRA_ENGINE_LITERT, ANIRA_PROVIDER_CUSTOM, "npu");
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(memory.m_why, "no probe knows the memory of provider 'npu'");
    EXPECT_EQ(of(ANIRA_ENGINE_CUSTOM, ANIRA_PROVIDER_CUSTOM, "com.example.npu").m_why,
              "no probe knows the memory of provider 'com.example.npu'");
    EXPECT_EQ(of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUSTOM, "QNNExecutionProvider").m_kind,
              ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(machine.probes(), asked) << "nothing asked for them";
}

// Every device of an API that agree give their kind; a single device its own; none, or an API
// that could not be asked, unknown with why. forget() asks again; the kept strings stay.
TEST(AbiMemoryKind, AnAnswerBecomesTheKind) {
    anira_test::FakeMachine machine;
    machine.set(DeviceApi::Vulkan,
                fake_devices({fake_device("a", ANIRA_MEMORY_DEDICATED),
                              fake_device("b", ANIRA_MEMORY_DEDICATED)}));
    anira::capi::MemoryKinds kinds;
    DeviceMemory memory = kinds.memory_of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_VULKAN, "", {});
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_DEDICATED);
    EXPECT_EQ(memory.m_why, "every Vulkan device: 'a': fake (dedicated); 'b': fake (dedicated)");

    machine.set(DeviceApi::Vulkan,
                fake_devices({fake_device("a", ANIRA_MEMORY_UNKNOWN, "type virtual GPU"),
                              fake_device("b", ANIRA_MEMORY_UNKNOWN, "type virtual GPU")}));
    EXPECT_EQ(kinds.memory_of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_VULKAN, "", {}).m_kind,
              ANIRA_MEMORY_DEDICATED)
        << "kept: the context asked before";
    kinds.forget();
    memory = kinds.memory_of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_VULKAN, "", {});
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_UNKNOWN) << "agreeing on unknown stays unknown";
    EXPECT_EQ(machine.probes(DeviceApi::Vulkan), 2);

    machine.set(DeviceApi::Vulkan, fake_devices({}));
    kinds.forget();
    EXPECT_EQ(kinds.memory_of(ANIRA_ENGINE_EXECUTORCH, ANIRA_PROVIDER_VULKAN, "", {}).m_why,
              "Vulkan lists no device");

    machine.set(DeviceApi::Cuda, anira_test::fake_failure("the CUDA driver is not loadable (x)"));
    memory = kinds.memory_of(ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA, "", {});
    EXPECT_EQ(memory.m_kind, ANIRA_MEMORY_UNKNOWN);
    EXPECT_EQ(memory.m_why, "the CUDA driver is not loadable (x)");

    const char* kept = kinds.keep("a reason");
    kinds.forget();
    EXPECT_EQ(kinds.keep("a reason"), kept) << "an equal text is kept once, and stays";
    EXPECT_STREQ(kept, "a reason");
}

// The provider rule per memory kind: zero-copy over host memory; on a device, moved by the
// engine, a unified transfer within unified memory, a dedicated transfer across the bus to
// dedicated memory, and ANIRA_EDGE_UNKNOWN where the kind is unknown (an NPU's here), the reason
// naming the kind and what was seen.
TEST(AbiMemoryKind, TheProviderRuleFollowsTheMemoryKind) {
    using anira::engine::provider_rule;
    const anira::engine::ProviderRule host =
        provider_rule(ANIRA_PROVIDER_CPU, "", DeviceMemory{.m_kind = ANIRA_MEMORY_UNIFIED});
    EXPECT_EQ(host.m_class, ANIRA_EDGE_ZERO_COPY);
    EXPECT_EQ(host.m_moved_by, static_cast<uint32_t>(ANIRA_MOVER_NONE));
    EXPECT_TRUE(host.m_reason.empty());

    const anira::engine::ProviderRule unified =
        provider_rule(ANIRA_PROVIDER_CUDA,
                      "",
                      DeviceMemory{.m_kind = ANIRA_MEMORY_UNIFIED, .m_why = "u"});
    EXPECT_EQ(unified.m_class, ANIRA_EDGE_UNIFIED_TRANSFER);
    EXPECT_EQ(unified.m_moved_by, static_cast<uint32_t>(ANIRA_MOVER_ENGINE));
    EXPECT_EQ(unified.m_reason,
              "the provider's device computes in the host's memory (u): its runtime hands the "
              "slot over to the device within that memory, no bus, and the results back, per "
              "call");

    const anira::engine::ProviderRule dedicated =
        provider_rule(ANIRA_PROVIDER_CUDA,
                      "",
                      DeviceMemory{.m_kind = ANIRA_MEMORY_DEDICATED, .m_why = "d"});
    EXPECT_EQ(dedicated.m_class, ANIRA_EDGE_DEDICATED_TRANSFER);
    EXPECT_EQ(dedicated.m_moved_by, static_cast<uint32_t>(ANIRA_MOVER_ENGINE));
    EXPECT_EQ(dedicated.m_reason,
              "the provider's device computes in memory of its own (d): its runtime moves the "
              "slot across the bus and the results back, per call");

    const anira::engine::ProviderRule unknown =
        provider_rule(ANIRA_PROVIDER_CUSTOM,
                      "npu",
                      DeviceMemory{.m_kind = ANIRA_MEMORY_UNKNOWN, .m_why = "?"});
    EXPECT_EQ(unknown.m_class, ANIRA_EDGE_UNKNOWN);
    EXPECT_EQ(unknown.m_moved_by, static_cast<uint32_t>(ANIRA_MOVER_ENGINE));
    EXPECT_EQ(unknown.m_reason, "memory kind not probed: ?");
}
