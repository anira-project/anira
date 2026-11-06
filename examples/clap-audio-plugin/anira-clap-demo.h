#ifndef ANIRA_CLAP_PLUGIN_EXAMPLE_H
#define ANIRA_CLAP_PLUGIN_EXAMPLE_H

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <clap/helpers/plugin-proxy.hh>
#include <clap/helpers/plugin.hh>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "../../extras/models/BypassEngine.h"
#include "../../extras/models/hybrid-nn/HybridNNProcessor.h"
#include "../../extras/models/model_files.h"
#include "utils/DryWetMixer.h"

namespace clap_plugin_example {

/// A Backend choice: the name the parameter shows and the plan it stands for.
struct BackendChoice {
    const char* name;
    anira::PlanQuery query;
};

struct AniraClapPluginExample
    : public clap::helpers::Plugin<clap::helpers::MisbehaviourHandler::Terminate,
                                   clap::helpers::CheckingLevel::Maximal> {
    AniraClapPluginExample(const clap_host* host);
    ~AniraClapPluginExample();

    static clap_plugin_descriptor m_desc;

    bool activate(double sampleRate,
                  uint32_t minFrameCount,
                  uint32_t maxFrameCount) noexcept override;

    enum ParamIds : uint32_t { pmDryWet = 14256, pmBackend = 14257, pmTryNonetheless = 14258 };
    static constexpr int m_number_params = 3;

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

    // clap.render: the host says whether it renders offline (a bounce), which runs the Backend
    // choice whatever its verdict, since it does not run against the clock.
    bool implementsRender() const noexcept override { return true; }
    bool renderHasHardRealtimeRequirement() noexcept override { return false; }
    bool renderSetMode(clap_plugin_render_mode mode) noexcept override;

private:
    /// Selects the plan of the report that runs the Backend choice (m_param_backend). A choice
    /// without a plan in the report runs the Bypassed plan instead, and so does one whose plan is
    /// not real-time on this machine while "Try nonetheless" is off and the host renders in real
    /// time. Called by activate (the new report), by every change of the two parameters and by a
    /// change of the render mode.
    void apply_backend() noexcept;

    // "Try nonetheless" (on: accept a backend that is not real-time here) stays as the user set
    // it across activations, so the plugin asks again only while it is off. The constructor sets
    // the Backend to its default choice.
    double m_param_dry_wet{100.0}, m_param_backend{0}, m_param_try_nonetheless{0};
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
    // The Backend choices: the backends of this build that the model has an export for, then
    // Bypassed (the last). The default is the first: ONNX Runtime where the build has it.
    std::vector<BackendChoice> m_backends;
    size_t m_bypassed;
    static constexpr size_t k_default_backend = 0;
    // Per Backend choice, whether the last activate's report has a plan for it (an engine can
    // fail to run the model on this machine) and the real-time verdict (anira_rt_verdict) of that
    // plan, UNMEASURED without one. Five engines and Bypassed at most.
    static constexpr size_t k_max_backends = 6;
    std::array<bool, k_max_backends> m_has_plan{};
    std::array<uint32_t, k_max_backends> m_verdicts{};
    // The host's render mode: set on the main thread (renderSetMode), and the copy apply_backend
    // reads, taken by activate and by process when it changed.
    std::atomic<bool> m_offline{false};
    bool m_applied_offline{false};

    // The host's input as one planar tensor over its channel pointer.
    std::array<float*, 1> m_channels{};
    anira::Tensor m_audio{};

    utils::DryWetMixer m_dry_wet_mixer;
};

}  // namespace clap_plugin_example

#endif  // ANIRA_CLAP_PLUGIN_EXAMPLE_H
