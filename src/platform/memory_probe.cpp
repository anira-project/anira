// The unified-memory probe (memory_probe.h): each device API asked for the memory its devices
// compute in, through the API's own entries, loaded at run time. The few types and constants
// each probe passes are declared here, at the layouts and values of the APIs' published
// headers (vulkan_core.h, cuda.h, hip_runtime_api.h, cl.h); D3D12 and Metal come from the
// platform's own SDK (Windows' headers, the Objective-C runtime's entries).
#include "memory_probe.h"

#include <anira/abi/enums.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// The SDK's D3D12 and DXGI interfaces, over windows.h's declarations.
#include <d3d12.h>
#include <dxgi.h>
#elif !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#endif

namespace anira::platform {

namespace {

// The calling convention of the APIs' entries: __stdcall on 32-bit Windows (VKAPI_CALL,
// CUDAAPI, CL_API_CALL), the platform's default everywhere else.
#if defined(_WIN32) && !defined(_WIN64)
#define ANIRA_PROBE_CALL __stdcall
#else
#define ANIRA_PROBE_CALL
#endif

// VkPhysicalDeviceType (vulkan_core.h); vulkan_device reads them on every platform.
constexpr int32_t k_vk_device_type_integrated_gpu = 1;
constexpr int32_t k_vk_device_type_discrete_gpu = 2;
constexpr int32_t k_vk_device_type_virtual_gpu = 3;
constexpr int32_t k_vk_device_type_cpu = 4;

std::atomic<ApiProbe>& installed_probe() noexcept {
    static std::atomic<ApiProbe> k_probe{nullptr};
    return k_probe;
}

ApiAnswer failed(std::string why) {
    ApiAnswer answer;
    answer.m_failure = std::move(why);
    return answer;
}

#if !defined(__EMSCRIPTEN__)

// ---- loading --------------------------------------------------------------------------------

// The first of `names` the loader finds, never unloaded (memory_probe.h); nullptr with what the
// loader said of the last name tried in `error`.
void* open_library(std::initializer_list<const char*> names, std::string& error) {
    error = "no library name on this platform";
    for (const char* name : names) {
#if defined(_WIN32)
        HMODULE module = LoadLibraryExA(name, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (module != nullptr) { return reinterpret_cast<void*>(module); }
        error = std::string(name) + ": LoadLibrary error " + std::to_string(GetLastError());
#else
        void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) { return handle; }
        const char* text = dlerror();
        error = text != nullptr ? std::string(text) : std::string(name) + ": not found";
#endif
    }
    return nullptr;
}

// An entry of a loaded library as the function type F; nullptr when the library lacks it.
template <class F>
F entry(void* library, const char* name) noexcept {
#if defined(_WIN32)
    FARPROC address = GetProcAddress(reinterpret_cast<HMODULE>(library), name);
    return reinterpret_cast<F>(reinterpret_cast<void*>(address));
#else
    return reinterpret_cast<F>(dlsym(library, name));
#endif
}

// ---- CUDA -----------------------------------------------------------------------------------

// cuda.h: CUresult and CUdevice are int; CU_DEVICE_ATTRIBUTE_INTEGRATED, "device is integrated
// with host memory"; CUDA_ERROR_NO_DEVICE.
constexpr int k_cu_device_attribute_integrated = 18;
constexpr int k_cuda_error_no_device = 100;
using CuInit = int(ANIRA_PROBE_CALL*)(unsigned int);
using CuDeviceGetCount = int(ANIRA_PROBE_CALL*)(int*);
using CuDeviceGet = int(ANIRA_PROBE_CALL*)(int*, int);
using CuDeviceGetName = int(ANIRA_PROBE_CALL*)(char*, int, int);
using CuDeviceGetAttribute = int(ANIRA_PROBE_CALL*)(int*, int, int);

ApiAnswer ask_cuda() {
#if defined(_WIN32)
    const std::initializer_list<const char*> names{"nvcuda.dll"};
#elif defined(__APPLE__)
    const std::initializer_list<const char*> names{};
#else
    const std::initializer_list<const char*> names{"libcuda.so.1", "libcuda.so"};
#endif
    std::string error;
    void* library = open_library(names, error);
    if (library == nullptr) { return failed("the CUDA driver is not loadable (" + error + ")"); }
    const auto init = entry<CuInit>(library, "cuInit");
    const auto get_count = entry<CuDeviceGetCount>(library, "cuDeviceGetCount");
    const auto get_device = entry<CuDeviceGet>(library, "cuDeviceGet");
    const auto get_name = entry<CuDeviceGetName>(library, "cuDeviceGetName");
    const auto get_attribute = entry<CuDeviceGetAttribute>(library, "cuDeviceGetAttribute");
    if (init == nullptr || get_count == nullptr || get_device == nullptr || get_name == nullptr ||
        get_attribute == nullptr) {
        return failed("the CUDA driver lacks an entry of the driver API");
    }
    const int status = init(0);
    if (status == k_cuda_error_no_device) { return {}; }
    if (status != 0) { return failed("cuInit returned " + std::to_string(status)); }
    int count = 0;
    if (const int counted = get_count(&count); counted != 0) {
        return failed("cuDeviceGetCount returned " + std::to_string(counted));
    }
    ApiAnswer answer;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        ProbedDevice probed;
        int device = 0;
        std::array<char, 256> name{};
        int integrated = 0;
        if (const int got = get_device(&device, ordinal); got != 0) {
            probed.m_evidence = "cuDeviceGet returned " + std::to_string(got);
        } else if (const int asked =
                       get_attribute(&integrated, k_cu_device_attribute_integrated, device);
                   asked != 0) {
            probed.m_evidence =
                "cuDeviceGetAttribute(INTEGRATED) returned " + std::to_string(asked);
        } else {
            probed.m_kind = integrated != 0 ? ANIRA_MEMORY_UNIFIED : ANIRA_MEMORY_DEDICATED;
            probed.m_evidence = "CU_DEVICE_ATTRIBUTE_INTEGRATED " + std::to_string(integrated);
        }
        if (get_name(name.data(), static_cast<int>(name.size()) - 1, device) == 0) {
            probed.m_name = name.data();
        }
        answer.m_devices.push_back(std::move(probed));
    }
    return answer;
}

// ---- HIP ------------------------------------------------------------------------------------

// hip_runtime_api.h since HIP 5.0 (ROCm 5.0 reordered hipDeviceAttribute_t to follow CUDA's
// names): hipDeviceAttributeIntegrated, 16 from 5.0 through 7.x; hipErrorNoDevice. The probe
// reads the attribute of a runtime of 5.0 or later only (hipRuntimeGetVersion's
// major * 10000000 + minor * 100000 + patch).
constexpr int k_hip_device_attribute_integrated = 16;
constexpr int k_hip_error_no_device = 100;
constexpr int k_hip_version_5_0 = 50000000;
using HipRuntimeGetVersion = int (*)(int*);
using HipGetDeviceCount = int (*)(int*);
using HipDeviceGetName = int (*)(char*, int, int);
using HipDeviceGetAttribute = int (*)(int*, int, int);

ApiAnswer ask_hip() {
#if defined(_WIN32)
    const std::initializer_list<const char*> names{"amdhip64_7.dll",
                                                   "amdhip64_6.dll",
                                                   "amdhip64.dll"};
#elif defined(__APPLE__) || defined(__ANDROID__)
    const std::initializer_list<const char*> names{};
#else
    const std::initializer_list<const char*> names{"libamdhip64.so",
                                                   "libamdhip64.so.7",
                                                   "libamdhip64.so.6",
                                                   "libamdhip64.so.5"};
#endif
    std::string error;
    void* library = open_library(names, error);
    if (library == nullptr) { return failed("the HIP runtime is not loadable (" + error + ")"); }
    const auto get_version = entry<HipRuntimeGetVersion>(library, "hipRuntimeGetVersion");
    const auto get_count = entry<HipGetDeviceCount>(library, "hipGetDeviceCount");
    const auto get_name = entry<HipDeviceGetName>(library, "hipDeviceGetName");
    const auto get_attribute = entry<HipDeviceGetAttribute>(library, "hipDeviceGetAttribute");
    if (get_version == nullptr || get_count == nullptr || get_name == nullptr ||
        get_attribute == nullptr) {
        return failed("the HIP runtime lacks an entry the probe calls");
    }
    int version = 0;
    if (const int asked = get_version(&version); asked != 0) {
        return failed("hipRuntimeGetVersion returned " + std::to_string(asked));
    }
    if (version < k_hip_version_5_0) {
        return failed("HIP " + std::to_string(version) +
                      " predates 5.0, whose attribute numbering the probe reads");
    }
    int count = 0;
    const int counted = get_count(&count);
    if (counted == k_hip_error_no_device) { return {}; }
    if (counted != 0) { return failed("hipGetDeviceCount returned " + std::to_string(counted)); }
    ApiAnswer answer;
    for (int device = 0; device < count; ++device) {
        ProbedDevice probed;
        std::array<char, 256> name{};
        int integrated = 0;
        if (const int asked = get_attribute(&integrated, k_hip_device_attribute_integrated, device);
            asked != 0) {
            probed.m_evidence =
                "hipDeviceGetAttribute(Integrated) returned " + std::to_string(asked);
        } else {
            probed.m_kind = integrated != 0 ? ANIRA_MEMORY_UNIFIED : ANIRA_MEMORY_DEDICATED;
            probed.m_evidence = "hipDeviceAttributeIntegrated " + std::to_string(integrated);
        }
        if (get_name(name.data(), static_cast<int>(name.size()) - 1, device) == 0) {
            probed.m_name = name.data();
        }
        answer.m_devices.push_back(std::move(probed));
    }
    return answer;
}

// ---- OpenCL ---------------------------------------------------------------------------------

// cl.h: cl_int, cl_uint and cl_bool are 32-bit, cl_device_type a 64-bit bit field; the
// handles are pointers.
constexpr uint64_t k_cl_device_type_gpu = 1U << 2U;
constexpr uint32_t k_cl_device_name = 0x102B;
constexpr uint32_t k_cl_device_host_unified_memory = 0x1035;  // deprecated in 2.0, still answered
constexpr int32_t k_cl_device_not_found = -1;
constexpr int32_t k_cl_platform_not_found_khr = -1001;  // the ICD loader without a platform
using ClGetPlatformIds = int32_t(ANIRA_PROBE_CALL*)(uint32_t, void**, uint32_t*);
using ClGetDeviceIds = int32_t(ANIRA_PROBE_CALL*)(void*, uint64_t, uint32_t, void**, uint32_t*);
using ClGetDeviceInfo = int32_t(ANIRA_PROBE_CALL*)(void*, uint32_t, size_t, void*, size_t*);

ApiAnswer ask_opencl() {
#if defined(_WIN32)
    const std::initializer_list<const char*> names{"OpenCL.dll"};
#elif defined(__APPLE__)
    const std::initializer_list<const char*> names{
        "/System/Library/Frameworks/OpenCL.framework/OpenCL"};
#elif defined(__ANDROID__)
    const std::initializer_list<const char*> names{"libOpenCL.so"};
#else
    const std::initializer_list<const char*> names{"libOpenCL.so.1", "libOpenCL.so"};
#endif
    std::string error;
    void* library = open_library(names, error);
    if (library == nullptr) { return failed("OpenCL is not loadable (" + error + ")"); }
    const auto get_platforms = entry<ClGetPlatformIds>(library, "clGetPlatformIDs");
    const auto get_devices = entry<ClGetDeviceIds>(library, "clGetDeviceIDs");
    const auto get_info = entry<ClGetDeviceInfo>(library, "clGetDeviceInfo");
    if (get_platforms == nullptr || get_devices == nullptr || get_info == nullptr) {
        return failed("the OpenCL library lacks an entry the probe calls");
    }
    uint32_t num_platforms = 0;
    const int32_t listed = get_platforms(0, nullptr, &num_platforms);
    if (listed == k_cl_platform_not_found_khr || (listed == 0 && num_platforms == 0)) { return {}; }
    if (listed != 0) { return failed("clGetPlatformIDs returned " + std::to_string(listed)); }
    std::vector<void*> platforms(num_platforms);
    if (const int32_t got = get_platforms(num_platforms, platforms.data(), nullptr); got != 0) {
        return failed("clGetPlatformIDs returned " + std::to_string(got));
    }
    ApiAnswer answer;
    for (void* platform : platforms) {
        uint32_t num_devices = 0;
        const int32_t counted =
            get_devices(platform, k_cl_device_type_gpu, 0, nullptr, &num_devices);
        if (counted == k_cl_device_not_found || num_devices == 0) { continue; }
        if (counted != 0) { return failed("clGetDeviceIDs returned " + std::to_string(counted)); }
        std::vector<void*> devices(num_devices);
        if (const int32_t got =
                get_devices(platform, k_cl_device_type_gpu, num_devices, devices.data(), nullptr);
            got != 0) {
            return failed("clGetDeviceIDs returned " + std::to_string(got));
        }
        for (void* device : devices) {
            ProbedDevice probed;
            std::array<char, 256> name{};
            if (get_info(device, k_cl_device_name, name.size() - 1, name.data(), nullptr) == 0) {
                probed.m_name = name.data();
            }
            uint32_t unified = 0;
            if (const int32_t asked = get_info(device,
                                               k_cl_device_host_unified_memory,
                                               sizeof(unified),
                                               &unified,
                                               nullptr);
                asked != 0) {
                probed.m_evidence =
                    "CL_DEVICE_HOST_UNIFIED_MEMORY returned " + std::to_string(asked);
            } else {
                probed.m_kind = unified != 0 ? ANIRA_MEMORY_UNIFIED : ANIRA_MEMORY_DEDICATED;
                probed.m_evidence = "CL_DEVICE_HOST_UNIFIED_MEMORY " + std::to_string(unified);
            }
            answer.m_devices.push_back(std::move(probed));
        }
    }
    return answer;
}

// ---- Vulkan ---------------------------------------------------------------------------------

// vulkan_core.h at the layouts the API fixes: the two create-info structs, the memory
// properties, the start of the device properties (the rest is read into spare storage).
constexpr int32_t k_vk_structure_type_application_info = 0;
constexpr int32_t k_vk_structure_type_instance_create_info = 1;
constexpr uint32_t k_vk_api_version_1_0 = 1U << 22U;
constexpr int32_t k_vk_error_incompatible_driver = -9;
constexpr uint32_t k_vk_instance_create_enumerate_portability = 0x1;
constexpr const char* k_vk_portability_enumeration = "VK_KHR_portability_enumeration";

struct VulkanApplicationInfo {
    int32_t m_type = k_vk_structure_type_application_info;
    const void* m_next = nullptr;
    const char* m_application_name = nullptr;
    uint32_t m_application_version = 0;
    const char* m_engine_name = nullptr;
    uint32_t m_engine_version = 0;
    uint32_t m_api_version = k_vk_api_version_1_0;
};

struct VulkanInstanceCreateInfo {
    int32_t m_type = k_vk_structure_type_instance_create_info;
    const void* m_next = nullptr;
    uint32_t m_flags = 0;
    const VulkanApplicationInfo* m_application = nullptr;
    uint32_t m_layer_count = 0;
    const char* const* m_layers = nullptr;
    uint32_t m_extension_count = 0;
    const char* const* m_extensions = nullptr;
};

struct VulkanMemoryType {
    uint32_t m_property_flags;
    uint32_t m_heap_index;
};
struct VulkanMemoryHeap {
    uint64_t m_size;
    uint32_t m_flags;
};
struct VulkanMemoryProperties {
    uint32_t m_type_count;
    std::array<VulkanMemoryType, 32> m_types;
    uint32_t m_heap_count;
    std::array<VulkanMemoryHeap, 16> m_heaps;
};

// VkPhysicalDeviceProperties: apiVersion, driverVersion, vendorID, deviceID, deviceType, then
// deviceName[256], the UUID, the limits and the sparse properties (824 bytes on LP64); the
// buffer holds more than all of it.
constexpr size_t k_vk_device_type_offset = 16;
constexpr size_t k_vk_device_name_offset = 20;
constexpr size_t k_vk_device_name_size = 256;
struct VulkanDeviceProperties {
    alignas(8) std::array<unsigned char, 4096> m_bytes{};
};

using VkVoidFunction = void(ANIRA_PROBE_CALL*)();
using VkGetProcAddr = VkVoidFunction(ANIRA_PROBE_CALL*)(void*, const char*);
using VkCreateInstance = int32_t(ANIRA_PROBE_CALL*)(const VulkanInstanceCreateInfo*,
                                                    const void*,
                                                    void**);
using VkDestroyInstance = void(ANIRA_PROBE_CALL*)(void*, const void*);
using VkEnumeratePhysicalDevices = int32_t(ANIRA_PROBE_CALL*)(void*, uint32_t*, void**);
using VkGetPhysicalDeviceProperties = void(ANIRA_PROBE_CALL*)(void*, VulkanDeviceProperties*);
using VkGetPhysicalDeviceMemoryProperties = void(ANIRA_PROBE_CALL*)(void*, VulkanMemoryProperties*);

template <class F>
F vulkan_entry(VkGetProcAddr get, void* instance, const char* name) noexcept {
    return reinterpret_cast<F>(get(instance, name));
}

ApiAnswer ask_vulkan() {
#if defined(_WIN32)
    const std::initializer_list<const char*> names{"vulkan-1.dll"};
#elif defined(__APPLE__)
    const std::initializer_list<const char*> names{"libvulkan.1.dylib", "libMoltenVK.dylib"};
#elif defined(__ANDROID__)
    const std::initializer_list<const char*> names{"libvulkan.so"};
#else
    const std::initializer_list<const char*> names{"libvulkan.so.1", "libvulkan.so"};
#endif
    std::string error;
    void* library = open_library(names, error);
    if (library == nullptr) { return failed("Vulkan is not loadable (" + error + ")"); }
    const auto get = entry<VkGetProcAddr>(library, "vkGetInstanceProcAddr");
    if (get == nullptr) { return failed("the Vulkan loader lacks vkGetInstanceProcAddr"); }
    const auto create = vulkan_entry<VkCreateInstance>(get, nullptr, "vkCreateInstance");
    if (create == nullptr) { return failed("the Vulkan loader lacks vkCreateInstance"); }
    VulkanApplicationInfo application;
    application.m_application_name = "anira memory probe";
    application.m_engine_name = "anira";
    VulkanInstanceCreateInfo info;
    info.m_application = &application;
    void* instance = nullptr;
    int32_t created = create(&info, nullptr, &instance);
    if (created == k_vk_error_incompatible_driver) {
        // A portability driver (MoltenVK) is listed only to an instance that asks for it.
        info.m_flags = k_vk_instance_create_enumerate_portability;
        info.m_extension_count = 1;
        info.m_extensions = &k_vk_portability_enumeration;
        created = create(&info, nullptr, &instance);
    }
    if (created != 0 || instance == nullptr) {
        return failed("vkCreateInstance returned " + std::to_string(created));
    }
    const auto destroy = vulkan_entry<VkDestroyInstance>(get, instance, "vkDestroyInstance");
    if (destroy == nullptr) { return failed("the Vulkan instance lacks vkDestroyInstance"); }
    // The instance goes with this scope, whichever return leaves it.
    struct Owned {
        Owned(void* instance, VkDestroyInstance destroy) noexcept
            : m_instance(instance), m_destroy(destroy) {}
        Owned(const Owned&) = delete;
        Owned& operator=(const Owned&) = delete;
        Owned(Owned&&) = delete;
        Owned& operator=(Owned&&) = delete;
        ~Owned() { m_destroy(m_instance, nullptr); }

        void* m_instance;
        VkDestroyInstance m_destroy;
    };
    const Owned owned(instance, destroy);
    const auto enumerate =
        vulkan_entry<VkEnumeratePhysicalDevices>(get, instance, "vkEnumeratePhysicalDevices");
    const auto properties =
        vulkan_entry<VkGetPhysicalDeviceProperties>(get, instance, "vkGetPhysicalDeviceProperties");
    const auto memory =
        vulkan_entry<VkGetPhysicalDeviceMemoryProperties>(get,
                                                          instance,
                                                          "vkGetPhysicalDeviceMemoryProperties");
    if (enumerate == nullptr || properties == nullptr || memory == nullptr) {
        return failed("the Vulkan instance lacks an entry the probe calls");
    }
    uint32_t count = 0;
    if (const int32_t counted = enumerate(instance, &count, nullptr); counted != 0) {
        return failed("vkEnumeratePhysicalDevices returned " + std::to_string(counted));
    }
    std::vector<void*> devices(count);
    if (const int32_t got = enumerate(instance, &count, devices.data()); got < 0) {
        return failed("vkEnumeratePhysicalDevices returned " + std::to_string(got));
    }
    devices.resize(count);
    ApiAnswer answer;
    for (void* device : devices) {
        const auto described = std::make_unique<VulkanDeviceProperties>();
        properties(device, described.get());
        VulkanMemoryProperties heaps{};
        memory(device, &heaps);
        int32_t type = 0;
        std::memcpy(&type, described->m_bytes.data() + k_vk_device_type_offset, sizeof(type));
        std::array<char, k_vk_device_name_size + 1> name{};
        std::memcpy(name.data(),
                    described->m_bytes.data() + k_vk_device_name_offset,
                    k_vk_device_name_size);
        std::vector<uint32_t> type_flags;
        for (uint32_t i = 0; i < heaps.m_type_count && i < heaps.m_types.size(); ++i) {
            type_flags.push_back(heaps.m_types.at(i).m_property_flags);
        }
        std::vector<uint32_t> heap_flags;
        for (uint32_t i = 0; i < heaps.m_heap_count && i < heaps.m_heaps.size(); ++i) {
            heap_flags.push_back(heaps.m_heaps.at(i).m_flags);
        }
        answer.m_devices.push_back(vulkan_device(name.data(), type, type_flags, heap_flags));
    }
    return answer;
}

#endif  // !__EMSCRIPTEN__

// ---- D3D12 ----------------------------------------------------------------------------------

#if defined(_WIN32)

std::string utf8_of(const WCHAR* text) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) { return {}; }
    std::string converted(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, converted.data(), size, nullptr, nullptr);
    converted.resize(static_cast<size_t>(size) - 1);
    return converted;
}

using CreateDxgiFactory1 = HRESULT(WINAPI*)(REFIID, void**);

ApiAnswer ask_d3d12() {
    std::string error;
    void* dxgi = open_library({"dxgi.dll"}, error);
    if (dxgi == nullptr) { return failed("DXGI is not loadable (" + error + ")"); }
    void* d3d12 = open_library({"d3d12.dll"}, error);
    if (d3d12 == nullptr) { return failed("D3D12 is not loadable (" + error + ")"); }
    const auto create_factory = entry<CreateDxgiFactory1>(dxgi, "CreateDXGIFactory1");
    const auto create_device = entry<PFN_D3D12_CREATE_DEVICE>(d3d12, "D3D12CreateDevice");
    if (create_factory == nullptr || create_device == nullptr) {
        return failed("DXGI or D3D12 lacks an entry the probe calls");
    }
    IDXGIFactory1* factory = nullptr;
    if (const HRESULT made =
            create_factory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
        FAILED(made) || factory == nullptr) {
        return failed("CreateDXGIFactory1 returned " +
                      std::to_string(static_cast<unsigned long>(made)));
    }
    ApiAnswer answer;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        ProbedDevice probed;
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc))) { probed.m_name = utf8_of(desc.Description); }
        ID3D12Device* device = nullptr;
        const HRESULT made = create_device(adapter,
                                           D3D_FEATURE_LEVEL_11_0,
                                           __uuidof(ID3D12Device),
                                           reinterpret_cast<void**>(&device));
        if (FAILED(made) || device == nullptr) {
            probed.m_evidence =
                "D3D12CreateDevice returned " + std::to_string(static_cast<unsigned long>(made));
        } else {
            D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
            architecture.NodeIndex = 0;
            if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE,
                                                      &architecture,
                                                      sizeof(architecture)))) {
                probed.m_kind =
                    architecture.UMA != FALSE ? ANIRA_MEMORY_UNIFIED : ANIRA_MEMORY_DEDICATED;
                probed.m_evidence = std::string("D3D12_FEATURE_DATA_ARCHITECTURE.UMA ") +
                                    (architecture.UMA != FALSE ? "1" : "0");
            } else {
                probed.m_evidence = "CheckFeatureSupport(ARCHITECTURE) failed";
            }
            device->Release();
        }
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            probed.m_evidence += " (a software adapter)";
        }
        adapter->Release();
        adapter = nullptr;
        answer.m_devices.push_back(std::move(probed));
    }
    factory->Release();
    return answer;
}

#endif  // _WIN32

// ---- Metal ----------------------------------------------------------------------------------

#if defined(__APPLE__)

// The Objective-C runtime's entries, through which the system default MTLDevice is asked
// (objc_msgSend called at the method's own signature, BOOL read as one byte on both ABIs).
using MtlCreateSystemDefaultDevice = void* (*)();
using SelRegisterName = void* (*)(const char*);
using AutoreleasePoolPush = void* (*)();
using AutoreleasePoolPop = void (*)(void*);
using SendBool = unsigned char (*)(void*, void*);
using SendBoolSelector = unsigned char (*)(void*, void*, void*);
using SendObject = void* (*)(void*, void*);
using SendText = const char* (*)(void*, void*);
using SendVoid = void (*)(void*, void*);

ApiAnswer ask_metal() {
    std::string error;
    void* metal = open_library({"/System/Library/Frameworks/Metal.framework/Metal"}, error);
    if (metal == nullptr) { return failed("Metal is not loadable (" + error + ")"); }
    void* objc = open_library({"/usr/lib/libobjc.A.dylib"}, error);
    if (objc == nullptr) {
        return failed("the Objective-C runtime is not loadable (" + error + ")");
    }
    const auto create = entry<MtlCreateSystemDefaultDevice>(metal, "MTLCreateSystemDefaultDevice");
    const auto selector = entry<SelRegisterName>(objc, "sel_registerName");
    const auto push = entry<AutoreleasePoolPush>(objc, "objc_autoreleasePoolPush");
    const auto pop = entry<AutoreleasePoolPop>(objc, "objc_autoreleasePoolPop");
    void* const send = dlsym(objc, "objc_msgSend");
    if (create == nullptr || selector == nullptr || push == nullptr || pop == nullptr ||
        send == nullptr) {
        return failed("Metal or the Objective-C runtime lacks an entry the probe calls");
    }
    void* const pool = push();
    ApiAnswer answer;
    void* const device = create();  // retained: released below
    if (device != nullptr) {
        ProbedDevice probed;
        void* const has_unified = selector("hasUnifiedMemory");
        const auto responds = reinterpret_cast<SendBoolSelector>(send);
        if (responds(device, selector("respondsToSelector:"), has_unified) != 0) {
            const bool unified = reinterpret_cast<SendBool>(send)(device, has_unified) != 0;
            probed.m_kind = unified ? ANIRA_MEMORY_UNIFIED : ANIRA_MEMORY_DEDICATED;
            probed.m_evidence = std::string("hasUnifiedMemory ") + (unified ? "YES" : "NO");
        } else {
            probed.m_evidence = "the device does not answer hasUnifiedMemory (before macOS 10.15)";
        }
        if (void* const text = reinterpret_cast<SendObject>(send)(device, selector("name"))) {
            if (const char* const name =
                    reinterpret_cast<SendText>(send)(text, selector("UTF8String"))) {
                probed.m_name = name;
            }
        }
        reinterpret_cast<SendVoid>(send)(device, selector("release"));
        answer.m_devices.push_back(std::move(probed));
    }
    pop(pool);
    return answer;
}

#endif  // __APPLE__

// One API's own probe, on the calling thread; whatever it throws is its failure.
ApiAnswer ask(DeviceApi api) noexcept {
    try {
        switch (api) {
#if defined(__EMSCRIPTEN__)
            case DeviceApi::Metal:
            case DeviceApi::Vulkan:
            case DeviceApi::Cuda:
            case DeviceApi::Hip:
            case DeviceApi::OpenCl:
            case DeviceApi::D3d12: return failed("the web exposes no memory kind");
#else
            case DeviceApi::Metal:
#if defined(__APPLE__)
                return ask_metal();
#else
                return failed("Metal exists on Apple platforms alone");
#endif
            case DeviceApi::Vulkan: return ask_vulkan();
            case DeviceApi::Cuda: return ask_cuda();
            case DeviceApi::Hip: return ask_hip();
            case DeviceApi::OpenCl: return ask_opencl();
            case DeviceApi::D3d12:
#if defined(_WIN32)
                return ask_d3d12();
#else
                return failed("D3D12 exists on Windows alone");
#endif
#endif
        }
        return failed("no probe for this API");
    } catch (const std::exception& error) {
        return failed(std::string("the probe failed: ") + error.what());
    } catch (...) { return failed("the probe failed"); }
}

#if !defined(__EMSCRIPTEN__)
// One probe in flight: what its thread hands back, kept alive by the waiter and the thread,
// whichever lets go last.
struct Pending {
    std::mutex m_mutex;
    std::condition_variable m_answered;
    bool m_done = false;
    ApiAnswer m_answer;
};
#endif

}  // namespace

const char* device_api_name(DeviceApi api) noexcept {
    switch (api) {
        case DeviceApi::Metal: return "Metal";
        case DeviceApi::Vulkan: return "Vulkan";
        case DeviceApi::Cuda: return "CUDA";
        case DeviceApi::Hip: return "HIP";
        case DeviceApi::OpenCl: return "OpenCL";
        case DeviceApi::D3d12: return "D3D12";
    }
    return "unknown";
}

const char* memory_kind_word(anira_memory_kind kind) noexcept {
    switch (kind) {
        case ANIRA_MEMORY_UNIFIED: return "unified";
        case ANIRA_MEMORY_DEDICATED: return "dedicated";
        default: return "unknown";
    }
}

ProbedDevice vulkan_device(std::string name,
                           int32_t device_type,
                           std::span<const uint32_t> type_flags,
                           std::span<const uint32_t> heap_flags) {
    // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT and VK_MEMORY_HEAP_DEVICE_LOCAL_BIT share the bit.
    constexpr uint32_t k_device_local = 0x1;
    constexpr uint32_t k_host_visible = 0x2;  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    ProbedDevice probed;
    probed.m_name = std::move(name);
    switch (device_type) {
        case k_vk_device_type_integrated_gpu:
        case k_vk_device_type_cpu: {
            const std::string type =
                device_type == k_vk_device_type_cpu ? "type CPU" : "type integrated GPU";
            for (size_t i = 0; i < type_flags.size(); ++i) {
                if ((type_flags[i] & (k_device_local | k_host_visible)) ==
                    (k_device_local | k_host_visible)) {
                    probed.m_kind = ANIRA_MEMORY_UNIFIED;
                    probed.m_evidence = type + "; memory type " + std::to_string(i) +
                                        " is device-local and host-visible";
                    return probed;
                }
            }
            probed.m_evidence = type + ", but no memory type is both device-local and host-visible";
            return probed;
        }
        case k_vk_device_type_discrete_gpu:
            for (size_t i = 0; i < heap_flags.size(); ++i) {
                if ((heap_flags[i] & k_device_local) == 0) {
                    probed.m_kind = ANIRA_MEMORY_DEDICATED;
                    probed.m_evidence = "type discrete GPU; heap " + std::to_string(i) +
                                        " is not device-local (the host's)";
                    return probed;
                }
            }
            probed.m_evidence = "type discrete GPU, but every heap is device-local";
            return probed;
        case k_vk_device_type_virtual_gpu: probed.m_evidence = "type virtual GPU"; return probed;
        default:
            probed.m_evidence = "type other (" + std::to_string(device_type) + ")";
            return probed;
    }
}

ApiAnswer probe_api(DeviceApi api, std::chrono::milliseconds timeout) {
    // The installed probe runs where the real one would, under the same wait.
    const ApiProbe installed = installed_probe().load(std::memory_order_acquire);
#if defined(__EMSCRIPTEN__)
    static_cast<void>(timeout);
    return installed != nullptr ? installed(api) : ask(api);
#else
    try {
        auto pending = std::make_shared<Pending>();
        std::thread thread([pending, api, installed] {
            ApiAnswer answer = installed != nullptr ? installed(api) : ask(api);
            const std::scoped_lock<std::mutex> lock(pending->m_mutex);
            pending->m_answer = std::move(answer);
            pending->m_done = true;
            pending->m_answered.notify_all();
        });
        std::unique_lock<std::mutex> lock(pending->m_mutex);
        if (!pending->m_answered.wait_for(lock, timeout, [&pending] { return pending->m_done; })) {
            lock.unlock();
            thread.detach();  // it finishes on its own; the shared state outlives it
            return failed(std::string("the ") + device_api_name(api) +
                          " probe did not answer within " + std::to_string(timeout.count()) +
                          " ms");
        }
        ApiAnswer answer = std::move(pending->m_answer);
        lock.unlock();
        thread.join();
        return answer;
    } catch (const std::exception& error) {
        return failed(std::string("the ") + device_api_name(api) +
                      " probe could not run: " + error.what());
    }
#endif
}

void set_api_probe(ApiProbe probe) noexcept {
    installed_probe().store(probe, std::memory_order_release);
}

}  // namespace anira::platform
