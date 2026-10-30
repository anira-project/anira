#include "PluginProcessor.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

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
    // The context, the model configs, the contracts and the handlers are initialised in the
    // header, from the files.
    , dry_wet_mixer(32768)  // 32768 samples of max latency compensation for the dry-wet mixer
{
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    // The gain tensor: the model's Static input [1], over gain_value.
    const std::array<int64_t, 1> gain_extents{1};
    gain_tensor = anira::Tensor::from_host(&gain_value, ANIRA_DTYPE_F32, gain_extents);
#endif
#if MODEL_TO_USE == 7
    for (size_t i = 0; i < latent_space.size(); ++i) { latent_space_ptrs[i] = &latent_space[i]; }
    const std::array<int64_t, 2> latent_extents{4, 1};
    latent = anira::Tensor::from_host_planar<float>(std::span<float* const>(latent_space_ptrs),
                                                    latent_extents);
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

    // The host geometry completes the contract: a fixed block of samplesPerBlock samples at
    // sampleRate (a block_min of 1 would allow smaller blocks, at the price of more latency).
    // prepare returns the plan report, whose plans the "Backend Type" parameter picks from.
    const auto block = static_cast<double>(samplesPerBlock);
#if MODEL_TO_USE != 7
    contract.hard_geometry(block, block, sampleRate);
    backend_plans.assign(inference_handler.prepare(contract));
#else
    contract_encoder.hard_geometry(block, block, sampleRate);
    backend_plans_encoder.assign(inference_handler_encoder.prepare(contract_encoder));
    // The decoder anchors on its audio output (rave_funk_drum_decoder.model.json), so its
    // geometry is the same block and rate; its latent input follows at one frame per 2048
    // samples.
    contract_decoder.hard_geometry(block, block, sampleRate);
    backend_plans_decoder.assign(inference_handler_decoder.prepare(contract_decoder));
#endif

    // The audio tensor over the host's channels, built once per geometry: processBlock sets
    // the channel pointers and the block length of each call.
    const auto num_channels = static_cast<size_t>(getTotalNumInputChannels());
    const std::array<int64_t, 2> extents{static_cast<int64_t>(num_channels),
                                         static_cast<int64_t>(samplesPerBlock)};
    audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(channels.data(), num_channels),
        extents);

#if MODEL_TO_USE != 7
    int new_latency = (int)inference_handler.get_latency();  // The 0th tensor is the audio data
                                                             // tensor, so we only need the first
                                                             // element of the latency vector
#else
    // Encoder latency must be multiplied by 2048, because the encoder compresses the audio data by
    // a factor of 2048 in the time domain.
    int new_latency_encoder = (int)inference_handler_encoder.get_latency() * 2048;
    int new_latency_decoder = (int)inference_handler_decoder.get_latency();
    int new_latency = new_latency_encoder + new_latency_decoder;  // The total latency is the sum of
                                                                  // the latencies of both encoders
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

    // The audio tensor over this call's buffer: its channel pointers and its length.
    for (size_t c = 0; c < static_cast<size_t>(buffer.getNumChannels()) && c < channels.size();
         ++c) {
        channels[c] = buffer.getWritePointer(static_cast<int>(c));
    }
    audio.shape[1] = buffer.getNumSamples();

#if MODEL_TO_USE != 7
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    // The gain parameter into the model's Static input, from this (the driving) thread.
    gain_value = gain.load();
    inference_handler.set_static_input(1, gain_tensor);
#endif
    // In place: the same tensor is the input and the output block.
    inference_handler.process(audio, 0, audio, 0);
#else
    // For the RAVE model, we need to process the encoder and decoder separately.
    m_count_input_samples += buffer.getNumSamples();
    inference_handler_encoder.push_data(audio, 0);
    // Only pop data from the encoder when we have enough samples needed for one time step in the
    // latent space vector (2048 samples).
    while (m_count_input_samples >= 2048) {
        size_t received_samples = 0;
        inference_handler_encoder.pop_data(latent, 0, &received_samples);
        if (received_samples == 0) {
            std::cout << "No data received from encoder!" << std::endl;
            break;
        } else {
            m_count_input_samples -= 2048;
        }
        // Make some latent space modulation :)
        latent_space[0] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_0_ID.getParamID()).getValue());
        latent_space[1] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_1_ID.getParamID()).getValue());
        latent_space[2] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_2_ID.getParamID()).getValue());
        latent_space[3] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_3_ID.getParamID()).getValue());
        latent.shape[1] = static_cast<int64_t>(received_samples);
        inference_handler_decoder.push_data(latent, 0);
        latent.shape[1] = 1;
    }
    inference_handler_decoder.pop_data(audio, 0);
#endif

    dry_wet_mixer.mixWetSamples(buffer);

#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    // The simple-gain models also output their peak gain, a Static output (slot 1):
    // inference_handler.get_static_output(1, peak_tensor) reads it on this thread.
#endif
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
        // The choice names an engine; the handler runs the plan of the report that runs it
        // (an engine this build or the model lacks has none, and the setting changes nothing).
        const int choice = static_cast<int>(newValue);
#if MODEL_TO_USE != 7
        if (const std::optional<uint32_t> plan = backend_plans.plan_of(choice)) {
            inference_handler.set_plan(*plan);
        }
#else
        if (const std::optional<uint32_t> plan = backend_plans_encoder.plan_of(choice)) {
            inference_handler_encoder.set_plan(*plan);
        }
        if (const std::optional<uint32_t> plan = backend_plans_decoder.plan_of(choice)) {
            inference_handler_decoder.set_plan(*plan);
        }
#endif
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
