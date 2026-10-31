#include "PluginProcessor.h"

#include <array>
#include <cstdint>
#include <span>

namespace {

/// The plan a "Backend Type" choice names (PluginParameters::backendTypes).
anira::PlanQuery backend_query(const juce::String& choice) {
    if (choice == "LIBTORCH") { return {.backend = {.engine = ANIRA_ENGINE_LIBTORCH}}; }
    if (choice == "ONNX") { return {.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}}; }
    if (choice == "TFLITE") { return {.backend = {.engine = ANIRA_ENGINE_TFLITE}}; }
    if (choice == "LITERT") { return {.backend = {.engine = ANIRA_ENGINE_LITERT}}; }
    if (choice == "EXECUTORCH") { return {.backend = {.engine = ANIRA_ENGINE_EXECUTORCH}}; }
    return {.backend = {.engine = ANIRA_ENGINE_CUSTOM, .engine_id = k_bypass_engine_id}};
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
    , parameters(*this,
                 nullptr,
                 juce::Identifier(getName()),
                 PluginParameters::createParameterLayout())
    , dry_wet_mixer(32768)  // 32768 samples of max latency compensation for the dry-wet mixer
{
    for (const juce::String& choice : PluginParameters::backendTypes) {
        backend_queries.push_back(backend_query(choice));
    }
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

    // The host geometry completes the contract: the host's real block range at sampleRate.
    // samplesPerBlock is the largest block JUCE promises, not a fixed one (a host may pass
    // smaller blocks, splitting at an automation point, say), and JUCE names no minimum, so the
    // range starts at one sample. The latency anira reports covers every block size of that
    // range, and setLatencySamples below hands it to the host, which compensates it. Each
    // processBlock carries its own block (the tensor's extent is set per call). The decoder of
    // variant 7 anchors on its audio output, so it takes the same geometry.
    const auto block_max = static_cast<double>(samplesPerBlock);
#if MODEL_TO_USE != 7
    model.prepare(block_max, sampleRate);
#else
    encoder.prepare(block_max, sampleRate);
    decoder.prepare(block_max, sampleRate);
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
    dry_wet_mixer.setWetLatency((float)new_latency);

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

    // The backend choice: the plan of the report that runs it (an engine this build or the
    // model lacks has none, and the choice changes nothing). The report is the last prepare's,
    // and prepareToPlay never runs beside this callback.
    const anira::PlanQuery& query = backend_queries[static_cast<size_t>(backend.load())];
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
        size_t received_samples = 0;
        encoder.handler.pop_data(latent, 0, &received_samples);
        if (received_samples == 0) { break; }
        m_count_input_samples -= 2048;
        // Make some latent space modulation :)
        latent_space[0] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_0_ID.getParamID()).getValue());
        latent_space[1] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_1_ID.getParamID()).getValue());
        latent_space[2] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_2_ID.getParamID()).getValue());
        latent_space[3] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_3_ID.getParamID()).getValue());
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
    return new juce::GenericAudioProcessorEditor(*this);
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
