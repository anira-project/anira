#include "PluginProcessor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

#include "PluginEditor.h"

namespace {

/// Whether the model config has an entry (an export) for the engine.
bool has_entry(const anira::ModelConfig& config, anira::EngineKind engine) {
    for (uint32_t i = 0; i < config.model_count(); ++i) {
        if (config.model_engine(i).kind == engine) { return true; }
    }
    return false;
}

/// The "Backend Type" choices: in this order, each engine this build compiled in
/// (anira::enabled_engines) that every model of the plugin has an entry for, then BYPASS, the
/// bypass engine's plan.
std::vector<BackendChoice> backend_choices(
    std::initializer_list<const anira::ModelConfig*> configs) {
    const std::array<BackendChoice, 5> engines{{
        {"ONNX", {.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}}},
        {"LIBTORCH", {.backend = {.engine = ANIRA_ENGINE_LIBTORCH}}},
        {"LITERT", {.backend = {.engine = ANIRA_ENGINE_LITERT}}},
        {"TFLITE", {.backend = {.engine = ANIRA_ENGINE_TFLITE}}},
        {"EXECUTORCH", {.backend = {.engine = ANIRA_ENGINE_EXECUTORCH}}},
    }};
    const std::vector<anira::BackendId> built = anira::enabled_engines();
    std::vector<BackendChoice> choices;
    for (const BackendChoice& choice : engines) {
        const anira::EngineKind engine = choice.query.backend.engine;
        const bool in_build =
            std::any_of(built.begin(), built.end(), [engine](const anira::BackendId& id) {
                return id.engine == static_cast<uint32_t>(engine);
            });
        const bool in_models =
            std::all_of(configs.begin(), configs.end(), [engine](const anira::ModelConfig* c) {
                return has_entry(*c, engine);
            });
        if (in_build && in_models) { choices.push_back(choice); }
    }
    choices.push_back(
        {"BYPASS", {.backend = {.engine = ANIRA_ENGINE_CUSTOM, .engine_id = k_bypass_engine_id}}});
    return choices;
}

/// The default choice: ONNX Runtime where the build and the model have it, else the first.
int default_choice(const std::vector<BackendChoice>& choices) {
#if MODEL_TO_USE == 3
    // The stateful LSTM's ONNX export starts every inference from a zero state (the hidden state
    // is none of the graph's inputs or outputs); the LibTorch module keeps it across inferences.
    const juce::String preferred = "LIBTORCH";
#else
    const juce::String preferred = "ONNX";
#endif
    for (size_t i = 0; i < choices.size(); ++i) {
        if (choices[i].name == preferred) { return static_cast<int>(i); }
    }
    return 0;
}

juce::StringArray choice_names(const std::vector<BackendChoice>& choices) {
    juce::StringArray names;
    for (const BackendChoice& choice : choices) { names.add(choice.name); }
    return names;
}

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

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
    : AudioProcessor(BusesProperties()
#if MODEL_TO_USE == 5
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)
#else
                         .withInput("Input", juce::AudioChannelSet::mono(), true)
                         .withOutput("Output", juce::AudioChannelSet::mono(), true)
#endif
                         )
#if MODEL_TO_USE == 7
    , backends(backend_choices({&encoder.model_config, &decoder.model_config}))
#else
    , backends(backend_choices({&model.model_config}))
#endif
    , bypass_choice(backends.size() - 1)
    , default_backend(default_choice(backends))
    , backend(default_backend)
    , parameters(*this,
                 nullptr,
                 juce::Identifier(getName()),
                 PluginParameters::createParameterLayout(choice_names(backends), default_backend))
    , dry_wet_mixer(k_max_dry_latency) {
    jassert(backends.size() <= k_max_backends);
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    gain_tensor = anira::Tensor::from_host(&gain_value, ANIRA_DTYPE_F32, std::array<int64_t, 1>{1});
#endif
#if MODEL_TO_USE == 7
    for (size_t i = 0; i < latent_space.size(); ++i) { latent_space_ptrs[i] = &latent_space[i]; }
    latent = anira::Tensor::from_host_planar<float>(std::span<float* const>(latent_space_ptrs),
                                                    std::array<int64_t, 2>{4, 1});
#endif
    for (auto& parameterID : PluginParameters::getPluginParameterList()) {
        parameters.addParameterListener(parameterID, this);
    }
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {
    for (auto& parameterID : PluginParameters::getPluginParameterList()) {
        parameters.removeParameterListener(parameterID, this);
    }
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool AudioPluginAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const {
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms() {
    return 1;  // NB: some hosts don't cope very well if you tell them there are 0 programs,
               // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram() {
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram(int index) {
    juce::ignoreUnused(index);
}

const juce::String AudioPluginAudioProcessor::getProgramName(int index) {
    juce::ignoreUnused(index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName(int index, const juce::String& newName) {
    juce::ignoreUnused(index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    juce::dsp::ProcessSpec spec{sampleRate,
                                static_cast<juce::uint32>(samplesPerBlock),
                                static_cast<juce::uint32>(getTotalNumInputChannels())};

    dry_wet_mixer.prepare(spec);

    // The host blocks complete the contract: the host's real block range at sampleRate.
    // samplesPerBlock is the largest block JUCE promises, not a fixed one (a host may pass
    // smaller blocks, splitting at an automation point, say), and JUCE names no minimum, so the
    // range starts at one sample. The latency anira reports covers every block size of that
    // range, and setLatencySamples below hands it to the host, which compensates it. Each
    // processBlock carries its own block (the tensor's extent is set per call). The decoder of
    // variant 7 anchors on its audio output, so it takes the same host blocks.
    const auto host_block_max = static_cast<double>(samplesPerBlock);
#if MODEL_TO_USE != 7
    model.prepare(host_block_max, sampleRate);
#else
    encoder.prepare(host_block_max, sampleRate);
    decoder.prepare(host_block_max, sampleRate);
    // A new stream on both handlers: the count of input samples towards the next latent frame
    // starts over with it, so the frames are popped in step with the encoder's new stream.
    m_count_input_samples = 0;
#endif

    const auto num_channels = static_cast<size_t>(getTotalNumInputChannels());
    audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(channels.data(), num_channels),
        std::array<int64_t, 2>{static_cast<int64_t>(num_channels), samplesPerBlock});

#if MODEL_TO_USE != 7
    const int new_latency = static_cast<int>(model.handler.get_latency());
#else
    // The encoder's latency is in latent frames of 2048 samples.
    const int new_latency =
        static_cast<int>(encoder.handler.get_latency() * 2048 + decoder.handler.get_latency());
#endif

    setLatencySamples(new_latency);
    jassert(new_latency <= k_max_dry_latency);  // the dry path's delay covers it
    dry_wet_mixer.setWetLatency((float)new_latency);

    // What prepare measured: per backend choice, the row of the plan it runs, to the log, and
    // whether it has a plan and the plan's real-time verdict, for processBlock and the editor. A
    // choice can lack a plan although the build and the model have its engine: the engine could
    // not run the model on this machine. The figures are an estimate at prepare, measured on this
    // machine under its load of the moment.
#if MODEL_TO_USE != 7
    const std::array<const AniraModel*, 1> models{&model};
#else
    const std::array<const AniraModel*, 2> models{&encoder, &decoder};
#endif
    for (size_t choice = 0; choice < backends.size(); ++choice) {
        bool has_plan = true;
        uint32_t verdict = ANIRA_RT_UNMEASURED;
        float rt_factor = 0.0F;
        uint32_t rt_parallel = 1;
        for (const AniraModel* m : models) {
            const std::optional<uint32_t> plan = m->plan_report.find(backends[choice].query);
            if (!plan.has_value()) {
                has_plan = false;
                continue;
            }
            const anira_plan_info row = m->plan_report.plans()[*plan];
            juce::Logger::writeToLog(juce::String::formatted(
                "anira: %s: budget %.3g ms (measured %.3g ms, median %.3g ms, %u warm-up "
                "inferences, %s); real-time factor %.3g of %u (load %.3g): %s",
                backends[choice].name.toRawUTF8(),
                row.budget_ms,
                row.measured_ms,
                row.median_ms,
                row.warmup_iterations,
                outcome_word(row.warmup_outcome),
                row.rt_factor,
                row.rt_parallel,
                row.rt_factor / static_cast<double>(row.rt_parallel),
                verdict_word(row.rt_verdict)));
            // The worse of the handlers' verdicts (variant 7 runs two): the order of the enum
            // is the order of severity, UNMEASURED first.
            if (row.rt_verdict >= verdict) {
                verdict = row.rt_verdict;
                rt_factor = static_cast<float>(row.rt_factor);
                rt_parallel = row.rt_parallel;
            }
        }
        verdicts[choice].has_plan.store(has_plan);
        verdicts[choice].verdict.store(verdict);
        verdicts[choice].rt_factor.store(rt_factor);
        verdicts[choice].rt_parallel.store(rt_parallel);
        if (!has_plan) {
            juce::Logger::writeToLog("anira: " + backends[choice].name +
                                     ": no plan on this machine: it runs bypassed");
        } else if (verdict == ANIRA_RT_NOT_REALTIME &&
                   verdicts[choice].accepted.load() != ANIRA_RT_NOT_REALTIME) {
            juce::Logger::writeToLog(
                "anira: " + backends[choice].name +
                " is not real-time on this machine: it stays bypassed until 'Try nonetheless'");
        }
    }

    for (auto& parameterID : PluginParameters::getPluginParameterList()) {
        parameterChanged(parameterID, parameters.getParameterAsValue(parameterID).getValue());
    }
}

void AudioPluginAudioProcessor::releaseResources() {
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet()) { return false; }
#if MODEL_TO_USE == 5
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()) {
        return false;
    } else {
        return true;
    }
#else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()) {
        return false;
    } else {
        return true;
    }
#endif
}

void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& midiMessages) {
    juce::ignoreUnused(midiMessages);

    juce::ScopedNoDenormals noDenormals;

    dry_wet_mixer.pushDrySamples(buffer);

    for (int c = 0; c < buffer.getNumChannels(); ++c) {
        channels[static_cast<size_t>(c)] = buffer.getWritePointer(c);
    }
    audio.shape[1] = buffer.getNumSamples();

    // The backend choice: the plan of the report that runs it. The report is the last prepare's,
    // and prepareToPlay never runs beside this callback. A choice without a plan here runs the
    // bypass plan (the audio passes through, time-aligned), and so does one whose plan is not
    // real-time on this machine, until the user agrees ("Try nonetheless" in the editor); an
    // offline render (a bounce) runs that plan whatever the verdict, since it does not run against
    // the clock.
    const auto choice = static_cast<size_t>(backend.load());
    const bool runs = verdicts[choice].has_plan.load() && (isNonRealtime() || !held_back(choice));
    const anira::PlanQuery& query = backends[runs ? choice : bypass_choice].query;
#if MODEL_TO_USE != 7
    model.select(query);
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    gain_value = gain.load();
    model.handler.set_static_input(1, gain_tensor);
#endif
    model.handler.process(audio, 0, audio, 0);  // in place
#else
    encoder.select(query);
    decoder.select(query);
    // The encoder takes the audio and gives one latent frame per 2048 samples, which the
    // decoder turns back into audio.
    m_count_input_samples += buffer.getNumSamples();
    encoder.handler.push_data(audio, 0);
    while (m_count_input_samples >= 2048) {
        // A missed frame still takes its place in the stream: anira keeps the stream
        // time-aligned (the miss policy filled the frame, with zeros here, and the late frame
        // is dropped when it arrives), so it is counted and passed on like any other. Waiting
        // for it instead would leave the count ahead of the encoder's stream from then on:
        // every later pop too early, the decoder fed out of step, misses on every block.
        const anira_status popped = encoder.handler.pop_data(latent, 0);
        if (popped != ANIRA_OK && popped != ANIRA_MISSED) { break; }
        m_count_input_samples -= 2048;
        // Make some latent space modulation :) The offsets are the parameter listener's atomics:
        // reading the parameters' ValueTree here (getParameterAsValue) would allocate and
        // register a ValueTree listener on the audio thread, beside the message thread's use.
        for (size_t i = 0; i < latent_space.size(); ++i) {
            latent_space[i] += latent_offsets[i].load();
        }
        decoder.handler.push_data(latent, 0);
    }
    decoder.handler.pop_data(audio, 0);
#endif

    dry_wet_mixer.mixWetSamples(buffer);

    if (isNonRealtime()) { processesNonRealtime(buffer); }
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const {
    return true;  // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor() {
    return new PluginEditor(*this);
}

bool AudioPluginAudioProcessor::held_back(size_t choice) const noexcept {
    return verdicts[choice].verdict.load() == ANIRA_RT_NOT_REALTIME &&
           verdicts[choice].accepted.load() != ANIRA_RT_NOT_REALTIME;
}

bool AudioPluginAudioProcessor::waits_for_consent() const {
    return held_back(static_cast<size_t>(backend.load()));
}

void AudioPluginAudioProcessor::try_nonetheless() {
    // The user accepts the chosen backend's verdict: kept until a later prepare finds a worse one.
    const auto choice = static_cast<size_t>(backend.load());
    verdicts[choice].accepted.store(verdicts[choice].verdict.load());
}

juce::String AudioPluginAudioProcessor::verdict_line() const {
    const auto choice = static_cast<size_t>(backend.load());
    const BackendVerdict& v = verdicts[choice];
    const juce::String& name = backends[choice].name;
    if (!v.has_plan.load()) {
        return name + ": no plan on this machine: the plugin stays bypassed";
    }
    const float rt_factor = v.rt_factor.load();
    const uint32_t rt_parallel = v.rt_parallel.load();
    const juce::String parallel =
        juce::String(rt_parallel) +
        (rt_parallel == 1 ? " parallel inference" : " parallel inferences");
    switch (v.verdict.load()) {
        case ANIRA_RT_NOT_REALTIME:
            if (v.accepted.load() == ANIRA_RT_NOT_REALTIME) {
                return name + " runs at " + juce::String(rt_factor, 2) + "x real time on this " +
                       "machine (" + parallel + "): playing it anyway, it drops out";
            }
            return name + " runs at " + juce::String(rt_factor, 2) + "x real time on this " +
                   "machine (" + parallel + "): the plugin stays bypassed";
        case ANIRA_RT_MARGINAL: {
            const int headroom =
                juce::roundToInt(100.0F * (1.0F - rt_factor / static_cast<float>(rt_parallel)));
            return name + ": marginal on this machine (" + juce::String(headroom) +
                   " % headroom): may drop out under load";
        }
        default: return {};
    }
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused(destData);
}

void AudioPluginAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused(data, sizeInBytes);
}

void AudioPluginAudioProcessor::parameterChanged(const juce::String& parameterID, float newValue) {
    if (parameterID == PluginParameters::DRY_WET_ID.getParamID()) {
        dry_wet_mixer.setWetMixProportion(newValue);
    } else if (parameterID == PluginParameters::BACKEND_TYPE_ID.getParamID()) {
        backend.store(static_cast<int>(newValue));
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    } else if (parameterID == PluginParameters::GAIN_ID.getParamID()) {
        gain.store(newValue);
#endif
#if MODEL_TO_USE == 7
    } else if (parameterID == PluginParameters::LATENT_0_ID.getParamID()) {
        latent_offsets[0].store(newValue);
    } else if (parameterID == PluginParameters::LATENT_1_ID.getParamID()) {
        latent_offsets[1].store(newValue);
    } else if (parameterID == PluginParameters::LATENT_2_ID.getParamID()) {
        latent_offsets[2].store(newValue);
    } else if (parameterID == PluginParameters::LATENT_3_ID.getParamID()) {
        latent_offsets[3].store(newValue);
#endif
    }
}

void AudioPluginAudioProcessor::processesNonRealtime(const juce::AudioBuffer<float>& buffer) const {
    double durationInSeconds = static_cast<double>(buffer.getNumSamples()) / getSampleRate();
    auto durationInMilliseconds =
        std::chrono::duration<double, std::milli>(durationInSeconds * 1000);
    std::this_thread::sleep_for(durationInMilliseconds);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new AudioPluginAudioProcessor();
}
