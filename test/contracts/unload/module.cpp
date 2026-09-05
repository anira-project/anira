// The plugin-shaped test module: anira embedded in a loadable library, driven through
// a C API. See CMakeLists.txt in this directory for what the test proves.

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <exception>
#include <stdexcept>
#include <vector>

#include "module_api.h"

namespace {

using namespace anira;

constexpr int k_block_size = 512;
constexpr double k_sample_rate = 48000.;

InferenceConfig make_inference_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, k_block_size}}, {{1, 1, k_block_size}})},
        1.f,
        0,
        false,
        0.f,
        2);
}

CoreConfig make_core_config() {
    return {2, WaitStrategy::SpinBackoff, LogLevel::Warning};
}

struct Instance {
    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler{m_pp_processor, m_inference_config, make_core_config()};
    BufferF m_buffer{1, k_block_size};
};

struct ThrowingProcessor : public BackendBase {
    using BackendBase::BackendBase;
    void prepare() override { throw std::runtime_error("test backend: cannot load model"); }
};

}  // namespace

extern "C" {

void* unloadtest_create(void) {
    try {
        return new Instance();
    } catch (const std::exception&) { return nullptr; }
}

void unloadtest_prepare(void* instance) {
    static_cast<Instance*>(instance)->m_handler.prepare(HostConfig(k_block_size, k_sample_rate));
}

void unloadtest_process(void* instance, int num_blocks) {
    auto* i = static_cast<Instance*>(instance);
    for (int block = 0; block < num_blocks; ++block) {
        i->m_handler.process(i->m_buffer.get_array_of_write_pointers(), k_block_size);
    }
}

void unloadtest_destroy(void* instance) {
    delete static_cast<Instance*>(instance);
}

int unloadtest_create_throwing(void) {
    InferenceConfig inference_config = make_inference_config();
    PrePostProcessor pp_processor(inference_config);
    ThrowingProcessor throwing_processor(inference_config);
    try {
        const InferenceHandler handler(pp_processor,
                                       inference_config,
                                       throwing_processor,
                                       make_core_config());
    } catch (const std::exception&) { return 1; }
    return 0;
}

unsigned int unloadtest_num_inference_threads(void) {
    return Core::get_num_inference_threads();
}

int unloadtest_has_inference_threads(void) {
    return Core::has_inference_threads() ? 1 : 0;
}

int unloadtest_num_sessions(void) {
    return Core::get_num_sessions();
}

int unloadtest_has_core(void) {
    return Core::has_core() ? 1 : 0;
}

void unloadtest_shutdown(void) {
    Core::shutdown();
}

void unloadtest_leak_thread(void) {
    // SpinBackoff (the configuration's default): the thread wakes every <= 100 us, so
    // it runs into the unmapped code within a millisecond of the unload.
    InferenceThread* leaked = Core::make_inference_thread().release();
    leaked->start();
}
}
