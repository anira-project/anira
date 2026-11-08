// A context's memory kinds (memory.h): the provider -> device mapping and the answers of the
// unified-memory probe, asked on first need and kept.
#include "memory.h"

#include <anira/abi/enums.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "../engines/Adapter.h"
#include "../platform/memory_probe.h"
#include "../utils/Logger.h"
#include "words.h"

namespace anira::capi {

namespace {

using anira::platform::ApiAnswer;
using anira::platform::DeviceApi;
using anira::platform::ProbedDevice;

/// The device of a provider: one device of an API by its index, or, for a runtime that
/// chooses its adapter itself, every device of the API (no index).
struct DeviceQuery {
    DeviceApi m_api = DeviceApi::Vulkan;
    std::optional<uint32_t> m_index;
};

/// The record's "device_id" option, the device index ONNX Runtime's device providers take; 0
/// without one (or with one that is no index, which the runtime refuses at load itself).
uint32_t device_id_of(const std::vector<std::pair<std::string, std::string>>& options) {
    for (const auto& [key, value] : options) {
        if (key != "device_id") { continue; }
        uint32_t index = 0;
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), index);
        if (error == std::errc() && end == value.data() + value.size()) { return index; }
    }
    return 0;
}

/// The API a self-choosing GPU runtime runs over on this platform: Dawn's (ONNX Runtime's
/// WebGPU provider, LiteRT's GPU accelerator off Android) is Metal on Apple, D3D12 on Windows
/// and Vulkan elsewhere; LiteRT's GPU accelerator on Android is OpenCL.
DeviceApi dawn_api() noexcept {
#if defined(__APPLE__)
    return DeviceApi::Metal;
#elif defined(_WIN32)
    return DeviceApi::D3d12;
#else
    return DeviceApi::Vulkan;
#endif
}

DeviceQuery every_device_of(DeviceApi api) noexcept {
    // Metal is asked for its system default device alone, the one these runtimes take.
    if (api == DeviceApi::Metal) { return DeviceQuery{.m_api = api, .m_index = 0}; }
    return DeviceQuery{.m_api = api, .m_index = std::nullopt};
}

/// The device a device provider computes on (memory.h's mapping); nothing for a provider no
/// API answers for, with why in `none`.
std::optional<DeviceQuery> device_of(
    anira_engine engine,
    anira_provider provider,
    std::string_view provider_id,
    const std::vector<std::pair<std::string, std::string>>& options,
    std::string& none) {
#if defined(__EMSCRIPTEN__)
    static_cast<void>(engine);
    static_cast<void>(provider);
    static_cast<void>(provider_id);
    static_cast<void>(options);
    none = "the web exposes no memory kind (WebGPU names none)";
    return std::nullopt;
#else
    const bool onnxruntime = engine == ANIRA_ENGINE_ONNXRUNTIME;
    const uint32_t device_id = onnxruntime ? device_id_of(options) : 0U;
    if (provider_id.empty()) {
        switch (provider) {
            case ANIRA_PROVIDER_CUDA:
                return DeviceQuery{.m_api = DeviceApi::Cuda, .m_index = device_id};
            case ANIRA_PROVIDER_DIRECTML:
                return DeviceQuery{.m_api = DeviceApi::D3d12, .m_index = device_id};
            case ANIRA_PROVIDER_COREML: return every_device_of(DeviceApi::Metal);
            case ANIRA_PROVIDER_WEBGPU: return every_device_of(dawn_api());
            case ANIRA_PROVIDER_VULKAN: return every_device_of(DeviceApi::Vulkan);
            default: break;
        }
    } else if (onnxruntime) {
        if (provider_id == "TensorrtExecutionProvider" ||
            provider_id == "NvTensorRTRTXExecutionProvider") {
            return DeviceQuery{.m_api = DeviceApi::Cuda, .m_index = device_id};
        }
        if (provider_id == "ROCMExecutionProvider" || provider_id == "MIGraphXExecutionProvider") {
            return DeviceQuery{.m_api = DeviceApi::Hip, .m_index = device_id};
        }
    } else if (engine == ANIRA_ENGINE_LITERT && provider_id == "gpu") {
#if defined(__ANDROID__)
        return every_device_of(DeviceApi::OpenCl);
#else
        return every_device_of(dawn_api());
#endif
    } else if (engine == ANIRA_ENGINE_EXECUTORCH && provider_id == "MPSBackend") {
        return every_device_of(DeviceApi::Metal);
    }
    none = "no probe knows the memory of provider '" + provider_label(provider, provider_id) + "'";
    return std::nullopt;
#endif
}

/// "'name' (evidence)", how a device is named in a reason.
std::string device_text(const ProbedDevice& device) {
    return "'" + device.m_name + "': " + device.m_evidence;
}

/// The device memory one query reads off its API's answer.
anira::engine::DeviceMemory resolve(const DeviceQuery& query, const ApiAnswer& answer) {
    const std::string api = anira::platform::device_api_name(query.m_api);
    if (!answer.m_failure.empty()) {
        return anira::engine::DeviceMemory{.m_kind = ANIRA_MEMORY_UNKNOWN,
                                           .m_why = answer.m_failure};
    }
    if (query.m_index.has_value()) {
        const uint32_t index = *query.m_index;
        if (index >= answer.m_devices.size()) {
            return anira::engine::DeviceMemory{
                .m_kind = ANIRA_MEMORY_UNKNOWN,
                .m_why = api + " has no device " + std::to_string(index) + " (" +
                         std::to_string(answer.m_devices.size()) + " listed)"};
        }
        const ProbedDevice& device = answer.m_devices[index];
        return anira::engine::DeviceMemory{
            .m_kind = device.m_kind,
            .m_why = api + " device " + std::to_string(index) + " " + device_text(device)};
    }
    if (answer.m_devices.empty()) {
        return anira::engine::DeviceMemory{.m_kind = ANIRA_MEMORY_UNKNOWN,
                                           .m_why = api + " lists no device"};
    }
    // Every device: one kind where all agree, else the runtime's pick is not known.
    const anira_memory_kind kind = answer.m_devices.front().m_kind;
    bool agree = true;
    std::string listed;
    for (const ProbedDevice& device : answer.m_devices) {
        agree = agree && device.m_kind == kind;
        listed += listed.empty() ? "" : "; ";
        listed += device_text(device);
        listed += std::string(" (") + anira::platform::memory_kind_word(device.m_kind) + ")";
    }
    if (answer.m_devices.size() == 1) {
        return anira::engine::DeviceMemory{
            .m_kind = kind,
            .m_why = api + " device 0 " + device_text(answer.m_devices.front())};
    }
    if (agree && kind != ANIRA_MEMORY_UNKNOWN) {
        return anira::engine::DeviceMemory{.m_kind = kind,
                                           .m_why = "every " + api + " device: " + listed};
    }
    return anira::engine::DeviceMemory{.m_kind = ANIRA_MEMORY_UNKNOWN,
                                       .m_why = "the runtime picks its device among " + api +
                                                " devices of different kinds, " + listed};
}

}  // namespace

anira::engine::DeviceMemory host_memory() {
    return anira::engine::DeviceMemory{.m_kind = ANIRA_MEMORY_UNIFIED, .m_why = ""};
}

anira::engine::DeviceMemory MemoryKinds::memory_of(
    anira_engine engine,
    anira_provider provider,
    std::string_view provider_id,
    const std::vector<std::pair<std::string, std::string>>& options) {
    if (anira::engine::host_memory_provider(provider, provider_id)) { return host_memory(); }
    std::string none;
    const std::optional<DeviceQuery> query =
        device_of(engine, provider, provider_id, options, none);
    if (!query.has_value()) {
        return anira::engine::DeviceMemory{.m_kind = ANIRA_MEMORY_UNKNOWN,
                                           .m_why = std::move(none)};
    }
    const std::scoped_lock<std::mutex> lock(m_mutex);
    return resolve(*query, answer(query->m_api));
}

void MemoryKinds::forget() {
    const std::scoped_lock<std::mutex> lock(m_mutex);
    for (std::optional<ApiAnswer>& kept : m_answers) { kept.reset(); }
}

const char* MemoryKinds::keep(std::string text) {
    const std::scoped_lock<std::mutex> lock(m_mutex);
    return m_texts.insert(std::move(text)).first->c_str();
}

const ApiAnswer& MemoryKinds::answer(DeviceApi api) {
    std::optional<ApiAnswer>& kept = m_answers.at(static_cast<size_t>(api));
    if (kept.has_value()) { return *kept; }
    kept = anira::platform::probe_api(api);
    const char* name = anira::platform::device_api_name(api);
    if (!kept->m_failure.empty()) {
        ANIRA_LOG_DEBUG(anira::log_group::k_system,
                        "memory kinds: %s not answered: %s",
                        name,
                        kept->m_failure.c_str());
    }
    for (size_t index = 0; index < kept->m_devices.size(); ++index) {
        const ProbedDevice& device = kept->m_devices[index];
        ANIRA_LOG_DEBUG(anira::log_group::k_system,
                        "memory kinds: %s device %zu '%s': %s (%s)",
                        name,
                        index,
                        device.m_name.c_str(),
                        anira::platform::memory_kind_word(device.m_kind),
                        device.m_evidence.c_str());
    }
    return *kept;
}

}  // namespace anira::capi
