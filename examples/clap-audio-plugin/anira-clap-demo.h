#ifndef ANIRA_CLAP_PLUGIN_EXAMPLE_H
#define ANIRA_CLAP_PLUGIN_EXAMPLE_H

#include <anira/anira.hpp>
#include <array>
#include <clap/helpers/plugin-proxy.hh>
#include <clap/helpers/plugin.hh>
#include <cstdint>
#include <unordered_map>

#include "../../extras/models/BypassEngine.h"
#include "../../extras/models/hybrid-nn/HybridNNProcessor.h"
#include "../../extras/models/model_files.h"
#include "utils/DryWetMixer.h"

namespace clap_plugin_example {

struct AniraClapPluginExample
    : public clap::helpers::Plugin<clap::helpers::MisbehaviourHandler::Terminate,
                                   clap::helpers::CheckingLevel::Maximal> {
    AniraClapPluginExample(const clap_host* host);
    ~AniraClapPluginExample();

    static clap_plugin_descriptor m_desc;

    bool activate(double sampleRate,
                  uint32_t minFrameCount,
                  uint32_t maxFrameCount) noexcept override;

    enum ParamIds : uint32_t { pmDryWet = 14256, pmBackend = 14257 };
    static constexpr int m_number_params = 2;

    bool implementsParams() const noexcept override { return true; }
    bool isValidParamId(clap_id paramId) const noexcept override;
    uint32_t paramsCount() const noexcept override;
    bool paramsInfo(uint32_t paramIndex, clap_param_info* info) const noexcept override;
    bool paramsValue(clap_id paramId, double* value) noexcept override;

    bool paramsValueToText(clap_id paramId,
                           double value,
                           char* display,
                           uint32_t size) noexcept override;

protected:
    bool paramsTextToValue(clap_id paramId, const char* display, double* value) noexcept override;

public:
    bool implementsAudioPorts() const noexcept override { return true; }
    uint32_t audioPortsCount(bool isInput) const noexcept override;
    bool audioPortsInfo(uint32_t index,
                        bool isInput,
                        clap_audio_port_info* info) const noexcept override;

    clap_process_status process(const clap_process* process) noexcept override;
    void checkForEvents(const clap_process* process);
    void handleInboundEvent(const clap_event_header_t* evt);

    void paramsFlush(const clap_input_events* in, const clap_output_events* out) noexcept override;

    bool implementsLatency() const noexcept override;
    uint32_t latencyGet() const noexcept override;

private:
    double m_param_dry_wet{100.0}, m_param_backend{3};
    std::unordered_map<clap_id, double*> m_param_to_value;
    uint32_t m_plugin_latency;

    // GuitarLSTM (the model file and the contract file of extras/models) with the entry of the
    // bypass engine, on one handler; the plan report of the last activate, where the Backend
    // choice is looked up.
    anira::Context m_context{anira::ContextConfig{}};
    anira::ModelConfig m_model_config;
    anira::ContractHandle m_contract;
    anira::InferenceHandler m_inference_handler;
    anira::PlanReport m_plan_report{nullptr};

    // The host's input as one planar tensor over its channel pointer.
    std::array<float*, 1> m_channels{};
    anira::Tensor m_audio{};

    utils::DryWetMixer m_dry_wet_mixer;
};

}  // namespace clap_plugin_example

#endif  // ANIRA_CLAP_PLUGIN_EXAMPLE_H
