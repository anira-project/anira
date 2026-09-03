#include "PluginProcessor.h"

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
// model_config, contract and inference_config are initialised in the header, from the files.
// No ContextConfig is passed: the library default (half the hardware threads for inference)
// is what this plugin wants; section 3.1 of the usage guide shows how to change it.
#if MODEL_TO_USE != 7
    , pp_processor(inference_config)
#if MODEL_TO_USE <= 2
    , bypass_processor(inference_config)
    , inference_handler(pp_processor, inference_config, bypass_processor)
#else
    , inference_handler(pp_processor, inference_config)
#endif
#else
    , pp_processor_encoder(inference_config_encoder)
    , inference_handler_encoder(pp_processor_encoder, inference_config_encoder)
    , pp_processor_decoder(inference_config_decoder)
    , inference_handler_decoder(pp_processor_decoder, inference_config_decoder)
#endif
    , dry_wet_mixer(32768)  // 32768 samples of max latency compensation for the dry-wet mixer
{
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
    // sampleRate (a block_min of 1 would allow smaller blocks, at the price of more latency),
    // and the bridge turns the contract and the model config into the 2.x HostConfig.
    const auto block = static_cast<uint32_t>(samplesPerBlock);
#if MODEL_TO_USE != 7
    contract.hard_geometry(block, block, sampleRate);
    inference_handler.prepare(anira::v3compat::to_host_config(contract, model_config));
#else
    contract_encoder.hard_geometry(block, block, sampleRate);
    inference_handler_encoder.prepare(
        anira::v3compat::to_host_config(contract_encoder, model_config_encoder));
    // The decoder anchors on its audio output (rave_funk_drum_decoder.model.json), so its
    // geometry is the same block and rate; its latent input follows at one frame per 2048
    // samples.
    contract_decoder.hard_geometry(block, block, sampleRate);
    inference_handler_decoder.prepare(
        anira::v3compat::to_host_config(contract_decoder, model_config_decoder));
#endif

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

#if MODEL_TO_USE != 7
    inference_handler.process(buffer.getArrayOfWritePointers(), (size_t)buffer.getNumSamples());
#else
    // For the RAVE model, we need to process the encoder and decoder separately.
    float latent_space[4][1];
    float* latent_space_ptrs[4];
    for (int i = 0; i < 4; ++i) { latent_space_ptrs[i] = latent_space[i]; }
    m_count_input_samples += buffer.getNumSamples();
    inference_handler_encoder.push_data(buffer.getArrayOfWritePointers(),
                                        (size_t)buffer.getNumSamples());
    // Only pop data from the encoder when we have enough samples needed for one time step in the
    // latent space vector (2048 samples).
    while (m_count_input_samples >= 2048) {
        size_t received_samples = inference_handler_encoder.pop_data(latent_space_ptrs, 1);
        if (received_samples == 0) {
            std::cout << "No data received from encoder!" << std::endl;
            break;
        } else {
            m_count_input_samples -= 2048;
        }
        // Make some latent space modulation :)
        latent_space[0][0] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_0_ID.getParamID()).getValue());
        latent_space[1][0] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_1_ID.getParamID()).getValue());
        latent_space[2][0] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_2_ID.getParamID()).getValue());
        latent_space[3][0] += static_cast<float>(
            parameters.getParameterAsValue(PluginParameters::LATENT_3_ID.getParamID()).getValue());
        inference_handler_decoder.push_data(latent_space_ptrs, received_samples);
    }
    inference_handler_decoder.pop_data(buffer.getArrayOfWritePointers(),
                                       (size_t)buffer.getNumSamples());
#endif

    dry_wet_mixer.mixWetSamples(buffer);

#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    // For the simple-gain and simple-stereo-gain models, we can retrieve the peak gain value from
    // the post-processor. float peak_gain = pp_processor.get_output(1, 0); std::cout << "peak_gain:
    // " << peak_gain << std::endl;
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
        int paramInt = (int)newValue;
        auto paramString = PluginParameters::backendTypes[paramInt];
#if MODEL_TO_USE != 6 && MODEL_TO_USE != 7
#ifdef USE_TFLITE
        if (paramString == "TFLITE") {
            inference_handler.set_inference_backend(anira::InferenceBackend::TFLITE);
        }
#endif
#ifdef USE_LITERT
        if (paramString == "LITERT") {
            inference_handler.set_inference_backend(anira::InferenceBackend::LITERT);
        }
#endif
#ifdef USE_EXECUTORCH
        if (paramString == "EXECUTORCH") {
            inference_handler.set_inference_backend(anira::InferenceBackend::EXECUTORCH);
        }
#endif
#ifdef USE_ONNXRUNTIME
        if (paramString == "ONNX") {
            inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);
        }
#endif
#ifdef USE_LIBTORCH
        if (paramString == "LIBTORCH") {
            inference_handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);
        }
#endif
        if (paramString == "BYPASS") {
            inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
        }
#elif MODEL_TO_USE == 6
#ifdef USE_LIBTORCH
        if (paramString == "LIBTORCH") {
            inference_handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);
        }
#endif
        if (paramString == "BYPASS") {
            inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
        }
#else
#ifdef USE_LIBTORCH
        if (paramString == "LIBTORCH") {
            inference_handler_encoder.set_inference_backend(anira::InferenceBackend::LIBTORCH);
            inference_handler_decoder.set_inference_backend(anira::InferenceBackend::LIBTORCH);
        }
#endif
        if (paramString == "BYPASS") {
            inference_handler_encoder.set_inference_backend(anira::InferenceBackend::CUSTOM);
            inference_handler_decoder.set_inference_backend(anira::InferenceBackend::CUSTOM);
        }
#endif
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    } else if (parameterID == PluginParameters::GAIN_ID.getParamID()) {
        pp_processor.set_input(newValue, 1, 0);
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
