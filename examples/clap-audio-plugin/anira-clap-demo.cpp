#include "anira-clap-demo.h"

#include <algorithm>
#include <array>
#include <clap/helpers/host-proxy.hh>
#include <clap/helpers/host-proxy.hxx>
#include <clap/helpers/plugin.hh>
#include <clap/helpers/plugin.hxx>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace clap_plugin_example {

namespace {

/// The Backend choices: their names and the plans they stand for.
constexpr std::array<const char*, 6> k_backend_names{"OnnxRuntime",
                                                     "LibTorch",
                                                     "TensorFlowLite",
                                                     "Bypassed",
                                                     "LiteRt",
                                                     "ExecuTorch"};
constexpr std::array<anira::PlanQuery, 6> k_backend_queries{{
    {.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}},
    {.backend = {.engine = ANIRA_ENGINE_LIBTORCH}},
    {.backend = {.engine = ANIRA_ENGINE_TFLITE}},
    {.backend = {.engine = ANIRA_ENGINE_CUSTOM, .engine_id = k_bypass_engine_id}},
    {.backend = {.engine = ANIRA_ENGINE_LITERT}},
    {.backend = {.engine = ANIRA_ENGINE_EXECUTORCH}},
}};
constexpr size_t k_bypassed = 3;

const char* outcome_word(uint32_t outcome) {
    switch (outcome) {
        case ANIRA_WARMUP_OUTCOME_FIXED: return "fixed";
        case ANIRA_WARMUP_OUTCOME_STABLE: return "stable";
        case ANIRA_WARMUP_OUTCOME_CAPPED: return "capped";
        case ANIRA_WARMUP_OUTCOME_REUSED:
            return "the model's figure, measured at an earlier prepare";
        default: return "no warm-up";
    }
}

const char* verdict_word(uint32_t verdict) {
    switch (verdict) {
        case ANIRA_RT_COMFORTABLE: return "comfortable";
        case ANIRA_RT_MARGINAL: return "marginal";
        case ANIRA_RT_NOT_REALTIME: return "NOT real-time";
        default: return "unmeasured";
    }
}

}  // namespace

AniraClapPluginExample::AniraClapPluginExample(const clap_host* host)
    : clap::helpers::Plugin<clap::helpers::MisbehaviourHandler::Terminate,
                            clap::helpers::CheckingLevel::Maximal>(&m_desc, host)
    , m_plugin_latency(0)
    , m_model_config([] {
        anira::ModelConfig config = anira::ModelConfig::from_file(k_hybridnn_model_json);
        config.add_model(k_bypass_engine_id);  // the entry of the bypass engine: no model file
        return config;
    }())
    , m_contract(anira::ContractHandle::from_file(k_hybridnn_contract_json))
    , m_inference_handler(m_context,
                          anira::Pipeline()
                              .add_engine(std::make_shared<BypassEngine>())
                              .add_inference(m_model_config)
                              .add_processor(std::make_shared<HybridNNProcessor>())) {
    m_param_to_value[pmDryWet] = &m_param_dry_wet;
    m_param_to_value[pmBackend] = &m_param_backend;
    m_param_to_value[pmTryNonetheless] = &m_param_try_nonetheless;
}

AniraClapPluginExample::~AniraClapPluginExample() = default;

const char* features[] = {CLAP_PLUGIN_FEATURE_AUDIO_EFFECT, nullptr};
clap_plugin_descriptor AniraClapPluginExample::m_desc = {
    CLAP_VERSION,
    "org.anira-project.anira-clap-plugin-example",
    "Anira Clap Plugin Example",
    "Anira Project",
    "https://github.com/anira-project/anira",
    "",
    "",
    "0.0.1",
    "A demo to show how to use CLAP with anira.",
    features};

bool AniraClapPluginExample::paramsInfo(uint32_t paramIndex, clap_param_info* info) const noexcept {
    if (paramIndex >= m_number_params) { return false; }

    info->flags = CLAP_PARAM_IS_AUTOMATABLE;

    switch (paramIndex) {
        case 0:
            info->id = pmDryWet;
            strncpy(info->name, "Mix", CLAP_NAME_SIZE);
            strncpy(info->module, "Demo", CLAP_NAME_SIZE);
            info->min_value = 0;
            info->max_value = 100;
            info->default_value = 100;
            break;
        case 1:
            info->id = pmBackend;
            strncpy(info->name, "Backend", CLAP_NAME_SIZE);
            strncpy(info->module, "Demo", CLAP_NAME_SIZE);
            info->min_value = 0;
            info->max_value = 5;
            info->default_value = 3;
            info->flags |= CLAP_PARAM_IS_STEPPED;
            break;
        case 2:
            info->id = pmTryNonetheless;
            strncpy(info->name, "Try nonetheless", CLAP_NAME_SIZE);
            strncpy(info->module, "Demo", CLAP_NAME_SIZE);
            info->min_value = 0;
            info->max_value = 1;
            info->default_value = 0;
            info->flags |= CLAP_PARAM_IS_STEPPED;
            break;
        default: return false;
    }
    return true;
}

bool AniraClapPluginExample::paramsValueToText(clap_id paramId,
                                               double value,
                                               char* display,
                                               uint32_t size) noexcept {
    auto pid = (ParamIds)paramId;
    std::string sValue{"ERROR"};

    auto n2s = [](auto n) {
        std::ostringstream oss;
        oss << std::round(n);
        return oss.str();
    };

    switch (pid) {
        case pmDryWet: sValue = n2s(value) + " %"; break;
        case pmBackend: sValue = k_backend_names.at(static_cast<size_t>(value)); break;
        case pmTryNonetheless: sValue = value >= 0.5 ? "On" : "Off"; break;
    }

    strncpy(display, sValue.c_str(), size);
    display[size - 1] = '\0';
    return true;
}

bool AniraClapPluginExample::paramsTextToValue(clap_id paramId,
                                               const char* display,
                                               double* value) noexcept {
    switch (paramId) {
        case pmDryWet: *value = std::clamp(std::atof(display), 0., 100.); return true;
        case pmBackend:
            for (size_t i = 0; i < k_backend_names.size(); ++i) {
                if (strcmp(display, k_backend_names[i]) == 0) {
                    *value = static_cast<double>(i);
                    return true;
                }
            }
            return false;
        case pmTryNonetheless:
            if (strcmp(display, "On") == 0 || strcmp(display, "Off") == 0) {
                *value = strcmp(display, "On") == 0 ? 1.0 : 0.0;
                return true;
            }
            return false;
        default: return false;
    }
}

bool AniraClapPluginExample::audioPortsInfo(uint32_t index,
                                            bool isInput,
                                            clap_audio_port_info* info) const noexcept {
    if (index > 0) { return false; }
    info->id = 0;
    snprintf(info->name, sizeof(info->name), "%s", "My Port Name");
    info->channel_count = 1;
    info->flags = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    return true;
}

bool AniraClapPluginExample::activate(double sampleRate,
                                      uint32_t minFrameCount,
                                      uint32_t maxFrameCount) noexcept {
    // The host blocks complete the contract: the host's real block range at sampleRate,
    // which CLAP's activate names on both ends (at least one sample). The latency anira reports
    // covers every block size of that range, and latencyGet() hands it to the host, which
    // compensates it. Each process call carries its own block (the tensor's extent is set per
    // call).
    try {
        m_contract.hard_host_blocks(std::max(1U, minFrameCount), maxFrameCount, sampleRate);
        m_plan_report = m_inference_handler.prepare(m_contract);
    } catch (const anira::Error& error) {
        std::cerr << "anira: " << error.what() << std::endl;
        return false;
    }
    // What prepare measured: per Backend choice, the row of the plan it runs (a choice whose
    // engine this build or the model lacks has none), and its real-time verdict for
    // apply_backend. The figures are an estimate at activate, measured on this machine under its
    // load of the moment.
    const std::vector<anira_plan_info> rows = m_plan_report.plans();
    for (size_t choice = 0; choice < k_backend_queries.size(); ++choice) {
        m_verdicts[choice] = ANIRA_RT_UNMEASURED;
        const std::optional<uint32_t> plan = m_plan_report.find(k_backend_queries[choice]);
        if (!plan.has_value()) { continue; }
        const anira_plan_info& row = rows[*plan];
        m_verdicts[choice] = row.rt_verdict;
        std::cout << "anira: " << k_backend_names[choice] << ": budget " << row.budget_ms
                  << " ms (measured " << row.measured_ms << " ms, median " << row.median_ms
                  << " ms, " << row.warmup_iterations << " warm-up inferences, "
                  << outcome_word(row.warmup_outcome) << "); real-time factor " << row.rt_factor
                  << " of " << row.rt_parallel << ": " << verdict_word(row.rt_verdict) << std::endl;
        if (row.rt_verdict == ANIRA_RT_NOT_REALTIME && m_param_try_nonetheless < 0.5) {
            std::cout << "anira: Backend '" << k_backend_names[choice] << "' runs at "
                      << row.rt_factor << "x real time on this machine (" << row.rt_parallel
                      << (row.rt_parallel == 1 ? " parallel inference" : " parallel inferences")
                      << "): it stays Bypassed until 'Try nonetheless' is on" << std::endl;
        }
    }
    // The handler starts on the model's default plan; the Backend choice the host shows (its
    // default "Bypassed", or what it set before this activate) is applied over the new report.
    m_applied_offline = m_offline.load();
    apply_backend();

    m_audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(m_channels),
        std::array<int64_t, 2>{1, static_cast<int64_t>(maxFrameCount)});

    m_plugin_latency = (uint32_t)m_inference_handler.get_latency();
    m_dry_wet_mixer.prepare(sampleRate, maxFrameCount, (size_t)m_plugin_latency);

    return true;
}

clap_process_status AniraClapPluginExample::process(const clap_process* process) noexcept {
    checkForEvents(process);
    // A change of the render mode since the last call (a bounce starts or ends).
    if (const bool offline = m_offline.load(); offline != m_applied_offline) {
        m_applied_offline = offline;
        apply_backend();
    }

    float** in = process->audio_inputs[0].data32;
    float** out = process->audio_outputs[0].data32;

    for (int channel = 0; channel < 1; ++channel) {
        for (int sample = 0; sample < process->frames_count; ++sample) {
            m_dry_wet_mixer.push_dry_sample(in[channel][sample]);
        }
    }

    m_channels[0] = in[0];
    m_audio.shape[1] = static_cast<int64_t>(process->frames_count);
    m_inference_handler.process(m_audio, 0, m_audio, 0);  // in place

    for (int channel = 0; channel < 1; ++channel) {
        for (int sample = 0; sample < process->frames_count; ++sample) {
            out[channel][sample] = m_dry_wet_mixer.mix_wet_sample(in[channel][sample]);
        }
    }

    return CLAP_PROCESS_SLEEP;
}

void AniraClapPluginExample::checkForEvents(const clap_process* process) {
    auto ev = process->in_events;
    auto sz = ev->size(ev);

    const clap_event_header_t* nextEvent{nullptr};
    uint32_t nextEventIndex{0};
    if (sz != 0) { nextEvent = ev->get(ev, nextEventIndex); }

    for (int i = 0; i < process->frames_count; ++i) {
        while (nextEvent && nextEvent->time == i) {
            handleInboundEvent(nextEvent);
            nextEventIndex++;
            if (nextEventIndex >= sz) {
                nextEvent = nullptr;
            } else {
                nextEvent = ev->get(ev, nextEventIndex);
            }
        }
    }
}

void AniraClapPluginExample::handleInboundEvent(const clap_event_header_t* evt) {
    if (evt->space_id != CLAP_CORE_EVENT_SPACE_ID) { return; }

    if (evt->type == CLAP_EVENT_PARAM_VALUE) {
        auto v = reinterpret_cast<const clap_event_param_value*>(evt);
        *m_param_to_value[v->param_id] = v->value;

        if (m_param_to_value[v->param_id] == &m_param_backend ||
            m_param_to_value[v->param_id] == &m_param_try_nonetheless) {
            apply_backend();
        } else if (m_param_to_value[v->param_id] == &m_param_dry_wet) {
            auto new_mix = static_cast<float>(m_param_dry_wet / 100.0);
            m_dry_wet_mixer.set_mix(new_mix);
        }
    }
}

void AniraClapPluginExample::paramsFlush(const clap_input_events* in,
                                         const clap_output_events* out) noexcept {
    auto sz = in->size(in);

    for (auto e = 0U; e < sz; ++e) {
        auto nextEvent = in->get(in, e);
        handleInboundEvent(nextEvent);
    }
}

void AniraClapPluginExample::apply_backend() noexcept {
    auto choice = static_cast<size_t>(m_param_backend);
    if (choice >= k_backend_queries.size()) { return; }
    // Not real-time here, not accepted, and played against the clock: the Bypassed plan (the
    // audio passes through, time-aligned, and the slow model costs no CPU meanwhile).
    if (m_verdicts[choice] == ANIRA_RT_NOT_REALTIME && m_param_try_nonetheless < 0.5 &&
        !m_applied_offline) {
        choice = k_bypassed;
    }
    if (const std::optional<uint32_t> plan = m_plan_report.find(k_backend_queries[choice])) {
        m_inference_handler.set_plan(*plan);
    }
}

bool AniraClapPluginExample::renderSetMode(clap_plugin_render_mode mode) noexcept {
    // Main thread: the next process call (or the next activate) applies it.
    m_offline.store(mode == CLAP_RENDER_OFFLINE);
    return true;
}

bool AniraClapPluginExample::isValidParamId(clap_id paramId) const noexcept {
    return m_param_to_value.find(paramId) != m_param_to_value.end();
}

bool AniraClapPluginExample::paramsValue(clap_id paramId, double* value) noexcept {
    *value = *m_param_to_value[paramId];
    return true;
}

uint32_t AniraClapPluginExample::paramsCount() const noexcept {
    return m_number_params;
}

uint32_t AniraClapPluginExample::audioPortsCount(bool isInput) const noexcept {
    return 1;
}

bool AniraClapPluginExample::implementsLatency() const noexcept {
    return true;
}

uint32_t AniraClapPluginExample::latencyGet() const noexcept {
    return m_plugin_latency;
}

}  // namespace clap_plugin_example
