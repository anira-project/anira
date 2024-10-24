#include "anira-clap-demo.h"
#include <iostream>
#include <cmath>
#include <cstring>

#include <clap/helpers/plugin.hh>
#include <clap/helpers/plugin.hxx>
#include <clap/helpers/host-proxy.hh>
#include <clap/helpers/host-proxy.hxx>
#include <iomanip>

namespace anira::clap_plugin_example
{

AniraClapPluginExample::AniraClapPluginExample(const clap_host *host)
    : clap::helpers::Plugin<clap::helpers::MisbehaviourHandler::Terminate,
                            clap::helpers::CheckingLevel::Maximal>(&m_desc, host)
{
    m_param_to_value[pmDryWet] = &m_param_dry_wet;
    m_param_to_value[pmBackend] = &m_param_backend;
}

AniraClapPluginExample::~AniraClapPluginExample() {

}

const char *features[] = {CLAP_PLUGIN_FEATURE_AUDIO_EFFECT, nullptr};
clap_plugin_descriptor AniraClapPluginExample::m_desc = {CLAP_VERSION,
                                            "org.anira-project.anira-clap-plugin-example",
                                            "Anira Clap Plugin Example",
                                            "Anira Project",
                                            "https://github.com/anira-project/anira",
                                            "",
                                            "",
                                            "1.0.0",
                                            "A demo to show how to use CLAP's host provided threads with anira.",
                                            features};

bool AniraClapPluginExample::paramsInfo(uint32_t paramIndex, clap_param_info *info) const noexcept
{
    if (paramIndex >= m_number_params)
        return false;

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
            info->max_value = 3;
            info->default_value = 3;
            info->flags |= CLAP_PARAM_IS_STEPPED;
            break;
        default:
            return false;
    }
    return true;
}

bool AniraClapPluginExample::paramsValueToText(clap_id paramId, double value, char *display,
                                    uint32_t size) noexcept
{
    auto pid = (ParamIds)paramId;
    std::string sValue{"ERROR"};

    auto n2s = [](auto n) {
        std::ostringstream oss;
        oss <<  std::round(n);
        return oss.str();
    };

    switch (pid) {
        case pmDryWet:
            sValue = n2s(value) + " %";
            break;
        case pmBackend:
        {
            auto newBackend = (Backend) static_cast<int>(value);
            switch (newBackend)
            {
            case OnnxRuntime:
                sValue = "OnnxRuntime";
                break;
            case LibTorch:
                sValue = "LibTorch";
                break;
            case TensorFlowLite:
                sValue = "TensorFlowLite";
                break;
            case Bypassed:
                sValue = "Bypassed";
                break;
            }
            break;
        }
    }

    strncpy(display, sValue.c_str(), size);
    display[size - 1] = '\0';
    return true;
}

bool AniraClapPluginExample::paramsTextToValue(clap_id paramId, const char *display, double *value) noexcept
{
    switch (paramId) {
        case pmDryWet:
            *value = std::clamp(std::atof(display), 0., 100.);
            return true;
        default:
            // TODO pmBackend
            return false;
    }
}

bool AniraClapPluginExample::audioPortsInfo(uint32_t index, bool isInput,
                                 clap_audio_port_info *info) const noexcept
{
    if (index > 0)
        return false;
    info->id = 0;
    snprintf(info->name, sizeof(info->name), "%s", "My Port Name");
    info->channel_count = 2;
    info->flags = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    return true;
}

bool AniraClapPluginExample::activate(double sampleRate, uint32_t minFrameCount,
                             uint32_t maxFrameCount) noexcept
{
    m_clap_thread_pool = (clap_host_thread_pool const*)_host.host()->get_extension(_host.host(), CLAP_EXT_THREAD_POOL);
    m_clap_thread_pool_available = (m_clap_thread_pool && m_clap_thread_pool->request_exec);

    return true;
}

clap_process_status AniraClapPluginExample::process(const clap_process *process) noexcept
{
    checkForEvents(process);

    float **in = process->audio_inputs[0].data32;
    float **out = process->audio_outputs[0].data32;
    uint32_t channelCount = std::min(process->audio_inputs[0].channel_count, process->audio_outputs[0].channel_count);

    for (int channel = 0; channel < channelCount; ++channel) {
        for (int sample = 0; sample < process->frames_count; ++sample) {
            out[channel][sample] = in[channel][sample];
        }
    }

    uint32_t num_tasks = 10;
    if (m_clap_thread_pool->request_exec(_host.host(), num_tasks)) {
        std::cout << "Tasks successfully submitted to the thread pool." << std::endl;
    } else {
        std::cout << "Tasks not successfully submitted to the thread pool." << std::endl;
    }

    return CLAP_PROCESS_SLEEP;
}

void AniraClapPluginExample::checkForEvents(const clap_process *process) {
    auto ev = process->in_events;
    auto sz = ev->size(ev);

    const clap_event_header_t *nextEvent{nullptr};
    uint32_t nextEventIndex{0};
    if (sz != 0) {
        nextEvent = ev->get(ev, nextEventIndex);
    }

    for (int i = 0; i < process->frames_count; ++i) {
        while (nextEvent && nextEvent->time == i) {
            handleInboundEvent(nextEvent);
            nextEventIndex++;
            if (nextEventIndex >= sz)
                nextEvent = nullptr;
            else
                nextEvent = ev->get(ev, nextEventIndex);
        }
    }
}

void AniraClapPluginExample::handleInboundEvent(const clap_event_header_t *evt)
{
    if (evt->space_id != CLAP_CORE_EVENT_SPACE_ID)
        return;

    if (evt->type == CLAP_EVENT_PARAM_VALUE) {
        auto v = reinterpret_cast<const clap_event_param_value *>(evt);
        *m_param_to_value[v->param_id] = v->value;
    }
}

void AniraClapPluginExample::paramsFlush(const clap_input_events *in, const clap_output_events *out) noexcept
{
    auto sz = in->size(in);

    for (auto e = 0U; e < sz; ++e) {
        auto nextEvent = in->get(in, e);
        handleInboundEvent(nextEvent);
    }
}

bool AniraClapPluginExample::isValidParamId(clap_id paramId) const noexcept
{
    return m_param_to_value.find(paramId) != m_param_to_value.end();
}

bool AniraClapPluginExample::paramsValue(clap_id paramId, double *value) noexcept
{
    *value = *m_param_to_value[paramId];
    return true;
}

bool AniraClapPluginExample::implementsThreadPool() const noexcept {
    return true;
}

void AniraClapPluginExample::threadPoolExec(uint32_t taskIndex) noexcept {
    std::cout << "threadPoolExec: " << taskIndex << std::endl;
}

uint32_t AniraClapPluginExample::paramsCount() const noexcept { return m_number_params; }

} // namespace anira::clap_plugin_example
