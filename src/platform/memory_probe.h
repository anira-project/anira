#ifndef ANIRA_PLATFORM_MEMORY_PROBE_H
#define ANIRA_PLATFORM_MEMORY_PROBE_H
/*
 * The unified-memory probe: which memory each device of a device API computes in, against the
 * host's (anira_memory_kind), asked of the API itself at run time and never inferred from a
 * platform's name. Every API is loaded dynamically (dlopen, LoadLibrary), so anira links none
 * of them, and a library once loaded stays loaded for the process: unloading a driver after
 * its initialisation (cuInit) is not safe in general, and a probe that timed out may still be
 * inside it. Each API is asked on a thread of its own, waited for a bounded time
 * (k_probe_timeout): a driver that hangs costs the caller that long once and answers a
 * failure. Control threads only (the context's capability entries, anira_handler_prepare),
 * never the audio thread: it loads libraries, creates a thread and allocates. Private to src/.
 */
#include <anira/abi/enums.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace anira::platform {

/// The device APIs anira can ask for a device's memory kind.
enum class DeviceApi : uint8_t {
    Metal,   ///< MTLDevice.hasUnifiedMemory of the system default device (Apple builds only)
    Vulkan,  ///< the physical device's type, cross-checked with its memory types and heaps
    Cuda,    ///< the CUDA driver's CU_DEVICE_ATTRIBUTE_INTEGRATED
    Hip,     ///< HIP's hipDeviceAttributeIntegrated (HIP 5.0 and later)
    OpenCl,  ///< CL_DEVICE_HOST_UNIFIED_MEMORY of every GPU device of every platform
    D3d12,   ///< D3D12_FEATURE_DATA_ARCHITECTURE.UMA of every DXGI adapter (Windows only)
};

/// Every API, in the order the tests and the messages list them.
inline constexpr std::array<DeviceApi, 6> k_device_apis{DeviceApi::Metal,
                                                        DeviceApi::Vulkan,
                                                        DeviceApi::Cuda,
                                                        DeviceApi::Hip,
                                                        DeviceApi::OpenCl,
                                                        DeviceApi::D3d12};

/// The API's name in a message: "Metal", "Vulkan", "CUDA", "HIP", "OpenCL", "D3D12".
const char* device_api_name(DeviceApi api) noexcept;

/// The word of a memory kind in a message and a log line: "unknown", "unified", "dedicated".
const char* memory_kind_word(anira_memory_kind kind) noexcept;

/// One device as its API describes it: its name, the kind its answer gives (UNKNOWN where the
/// answer does not settle it, with why) and the answer itself ("CU_DEVICE_ATTRIBUTE_INTEGRATED
/// 0", "type CPU; memory type 0 is device-local and host-visible").
struct ProbedDevice {
    std::string m_name;
    anira_memory_kind m_kind = ANIRA_MEMORY_UNKNOWN;
    std::string m_evidence;
};

/// What one API answered: its devices in the API's own order (the order its device indices
/// count, an ORT device_id's), or, when it could not be asked, why (m_failure non-empty, no
/// devices: the library is missing, an entry failed, the probe timed out, the API does not
/// exist on this platform). No failure and no device: the API answered that it has none.
struct ApiAnswer {
    std::vector<ProbedDevice> m_devices;
    std::string m_failure;
};

/// The Vulkan rule, one physical device: its type (VkPhysicalDeviceType) cross-checked with
/// the property flags of its memory types and the flags of its heaps. An integrated GPU or a
/// CPU device (Mesa's lavapipe) is unified when one of its memory types is both device-local
/// and host-visible (the host maps the device's memory, which is the host's); a discrete GPU
/// is dedicated when one of its heaps is not device-local (a heap of host memory beside its
/// own). Any other type, and a type its heaps contradict, stays UNKNOWN with what was seen.
ProbedDevice vulkan_device(std::string name,
                           int32_t device_type,
                           std::span<const uint32_t> type_flags,
                           std::span<const uint32_t> heap_flags);

/// How long one API's probe is waited for.
inline constexpr std::chrono::milliseconds k_probe_timeout{5000};

/// Asks one API: its own probe, or the installed test probe when there is one
/// (set_api_probe), on a thread of its own, waited for up to `timeout`. A probe that does not
/// answer in time is left to finish on its own (it holds nothing of the caller's) and answers
/// the failure "the <API> probe did not answer within <ms> ms". Whatever goes wrong is the
/// answer's failure. On the web, which has no thread to spare, the probe runs on the caller's
/// thread, and every real API fails: a browser exposes no memory kind.
ApiAnswer probe_api(DeviceApi api, std::chrono::milliseconds timeout = k_probe_timeout);

/// A probe the tests install in place of the real ones (a fake machine: mixed adapters, a
/// missing API, unified against dedicated devices, a probe that hangs); nullptr restores the
/// real probes. Process wide; a test sets it before the context whose rows it reads and clears
/// it after (and after a probe that timed out has returned).
using ApiProbe = ApiAnswer (*)(DeviceApi api);
void set_api_probe(ApiProbe probe) noexcept;

}  // namespace anira::platform

#endif  // ANIRA_PLATFORM_MEMORY_PROBE_H
