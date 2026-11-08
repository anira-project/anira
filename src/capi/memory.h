#ifndef ANIRA_CAPI_MEMORY_H
#define ANIRA_CAPI_MEMORY_H
/*
 * The memory kinds of a context: which physical memory the device of each device provider
 * computes in (anira_memory_kind), as the unified-memory probe (src/platform/memory_probe.h)
 * answers it. Asked lazily, the first time a capability row or a plan needs a device provider's
 * kind (never at the context's creation, never for a provider that runs over host memory),
 * once per device API, and kept for the context's life; anira_context_probe forgets the
 * answers, so the next row that needs one asks again. Control threads only, never the audio
 * thread (the probe loads libraries and waits). Private to src/capi.
 *
 * The provider -> device mapping (device_of in memory.cpp): ONNX Runtime's CUDA and TensorRT
 * providers, and any engine's ANIRA_PROVIDER_CUDA, are the CUDA device of the record's
 * "device_id" option (ONNX Runtime's; device 0 without it or for another engine); ROCm and
 * MIGraphX the HIP device of device_id; DirectML the DXGI adapter of device_id (D3D12); Core ML
 * the Metal system default device; the runtimes that choose their adapter themselves (WebGPU,
 * through Dawn: Metal on Apple, D3D12 on Windows, Vulkan elsewhere; LiteRT's "gpu"
 * accelerator: the same, OpenCL on Android; Vulkan, ExecuTorch's VulkanBackend) every device
 * of the backing API, one kind when all agree, else unknown naming both; an NPU and any other
 * custom provider unknown. On the web every device provider is unknown: a browser exposes no
 * memory kind.
 */
#include <anira/abi/enums.h>

#include <array>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../engines/Adapter.h"
#include "../platform/memory_probe.h"

namespace anira::capi {

/// The memory of a provider that runs over host memory (engine::host_memory_provider): the
/// host's own, UNIFIED, nothing probed.
anira::engine::DeviceMemory host_memory();

/// One context's memory kinds: the probe's answer per device API, asked on first need and
/// kept, and the strings the rows built from them point at.
class MemoryKinds {
public:
    /// The memory the device of `provider` computes in, for a backend of `engine` whose record
    /// carries `options` (the context's provider option set for the backend): UNIFIED with
    /// nothing asked for a provider that runs over host memory, else the mapped device's kind
    /// with what its API answered (the API probed now if this context has not asked it yet),
    /// UNKNOWN with why for a provider no API answers for. Thread-safe; it may wait for a probe
    /// (memory_probe.h's bound) under the lock.
    anira::engine::DeviceMemory memory_of(
        anira_engine engine,
        anira_provider provider,
        std::string_view provider_id,
        const std::vector<std::pair<std::string, std::string>>& options);

    /// Forgets every answer (anira_context_probe): the next row that needs one asks again. The
    /// kept strings stay, so a row read before keeps its reason.
    void forget();

    /// A copy of `text` the rows point at, kept for the context's life (an equal text is kept
    /// once).
    const char* keep(std::string text);

private:
    /// The probe's answer for `api`, asked now when this context has none (under m_mutex).
    const anira::platform::ApiAnswer& answer(anira::platform::DeviceApi api);

    std::mutex m_mutex;
    std::array<std::optional<anira::platform::ApiAnswer>, anira::platform::k_device_apis.size()>
        m_answers;
    std::set<std::string, std::less<>> m_texts;
};

}  // namespace anira::capi

#endif  // ANIRA_CAPI_MEMORY_H
