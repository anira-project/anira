#ifndef ANIRA_CLAP_PLUGIN_EXAMPLE_H
#define ANIRA_CLAP_PLUGIN_EXAMPLE_H

#include <iostream>

#include <clap/helpers/plugin.hh>
#include <clap/helpers/plugin-proxy.hh>

#include <atomic>
#include <array>
#include <unordered_map>
#include <memory>

#include <anira/anira.h>

#include "../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/desktop/models/hybrid-nn/HybridNNBypassProcessor.h"

#include "utils/DryWetMixer.h"

namespace anira::clap_plugin_example
{

struct AniraClapPluginExample : public clap::helpers::Plugin<clap::helpers::MisbehaviourHandler::Terminate,
                                       clap::helpers::CheckingLevel::Maximal>
{
    AniraClapPluginExample(const clap_host *host);
    ~AniraClapPluginExample();

    static clap_plugin_descriptor m_desc;

    bool init() noexcept override;

    bool activate(double sampleRate, uint32_t minFrameCount,
                  uint32_t maxFrameCount) noexcept override;

    enum ParamIds : uint32_t
    {
        pmDryWet = 14256,
        pmBackend = 14257
    };
    static constexpr int m_number_params = 2;

    bool implementsParams() const noexcept override { return true; }
    bool isValidParamId(clap_id paramId) const noexcept override;
    uint32_t paramsCount() const noexcept override;
    bool paramsInfo(uint32_t paramIndex, clap_param_info *info) const noexcept override;
    bool paramsValue(clap_id paramId, double *value) noexcept override;

    bool paramsValueToText(clap_id paramId, double value, char *display,
                           uint32_t size) noexcept override;

protected:
    bool paramsTextToValue(clap_id paramId, const char *display, double *value) noexcept override;
    bool implementsThreadPool() const noexcept override;
    void threadPoolExec(uint32_t taskIndex) noexcept override;

public:
    bool implementsAudioPorts() const noexcept override { return true; }
    uint32_t audioPortsCount(bool isInput) const noexcept override;
    bool audioPortsInfo(uint32_t index, bool isInput,
                        clap_audio_port_info *info) const noexcept override;

    clap_process_status process(const clap_process *process) noexcept override;
    void checkForEvents(const clap_process *process);
    void handleInboundEvent(const clap_event_header_t *evt);

    void paramsFlush(const clap_input_events *in, const clap_output_events *out) noexcept override;

    bool implementsLatency() const noexcept override;
    uint32_t latencyGet() const noexcept override;

  private:
    double m_param_dry_wet{100.0}, m_param_backend{3};
    std::unordered_map<clap_id, double *> m_param_to_value;
    const clap_host_thread_pool* m_clap_thread_pool{nullptr};
    uint32_t m_plugin_latency;

    ContextConfig m_anira_context;

    InferenceConfig m_inference_config = hybridnn_config;
    HybridNNPrePostProcessor m_pp_processor;
    HybridNNBypassProcessor m_bypass_processor;

    InferenceHandler m_inference_handler;

    utils::DryWetMixer m_dry_wet_mixer;

    enum Backend {
        OnnxRuntime,
        LibTorch,
        TensorFlowLite,
        Bypassed
    };
};

} // namespace anira::clap_plugin_example

#endif //ANIRA_CLAP_PLUGIN_EXAMPLE_H