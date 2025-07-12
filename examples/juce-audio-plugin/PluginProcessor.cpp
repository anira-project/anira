#include "PluginProcessor.h"

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor() 
        : AudioProcessor (BusesProperties()
#if MODEL_TO_USE == 5
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
#else
                       .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                       .withOutput ("Output", juce::AudioChannelSet::mono(), true)
#endif
                       ),
        parameters (*this, nullptr, juce::Identifier (getName()), PluginParameters::createParameterLayout()),
        // Optional anira_context_config
        anira_context_config(
            std::thread::hardware_concurrency() / 2 > 0 ? std::thread::hardware_concurrency() / 2 : 1 // Total number of threads
        ),
        pp_processor(inference_config),
#if MODEL_TO_USE == 1 || MODEL_TO_USE == 2
        // The bypass_processor is not needed for inference, but for the round trip test to output audio when selecting the CUSTOM backend. It must be customized when default pp_processor is replaced by a custom one.
        bypass_processor(inference_config),
        inference_handler(pp_processor, inference_config, bypass_processor, anira_context_config),
#elif MODEL_TO_USE == 3 || MODEL_TO_USE == 4 || MODEL_TO_USE == 5
        inference_handler(pp_processor, inference_config),
#endif
        dry_wet_mixer(32768) // 32768 samples of max latency compensation for the dry-wet mixer
{
    for (auto & parameterID : PluginParameters::getPluginParameterList()) {
        parameters.addParameterListener(parameterID, this);
    }
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
    for (auto & parameterID : PluginParameters::getPluginParameterList()) {
        parameters.removeParameterListener(parameterID, this);
    }
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::dsp::ProcessSpec spec {sampleRate,
                                 static_cast<juce::uint32>(samplesPerBlock),
                                 static_cast<juce::uint32>(getTotalNumInputChannels())};

    anira::HostAudioConfig host_config {
        (size_t) samplesPerBlock,
        sampleRate
    };

    dry_wet_mixer.prepare(spec);

    inference_handler.prepare(host_config);

    auto new_latency = inference_handler.get_latency();
    setLatencySamples(new_latency);

    dry_wet_mixer.setWetLatency((float) new_latency);

    for (auto & parameterID : PluginParameters::getPluginParameterList()) {
        parameterChanged(parameterID, parameters.getParameterAsValue(parameterID).getValue());
    }
}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;
#if MODEL_TO_USE == 5
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    else
        return true;
#else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;
    else
        return true;
#endif
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);

    juce::ScopedNoDenormals noDenormals;

    dry_wet_mixer.pushDrySamples(buffer);

    inference_handler.process(buffer.getArrayOfWritePointers(), (size_t) buffer.getNumSamples());

    dry_wet_mixer.mixWetSamples(buffer);

#if MODEL_TO_USE == 4
    float peak_gain = pp_processor.get_output(1, 0);
    // std::cout << "peak_gain: " << peak_gain << std::endl;
#endif

    if (isNonRealtime()) {
        processesNonRealtime(buffer);
    }
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);
}

void AudioPluginAudioProcessor::parameterChanged(const juce::String &parameterID, float newValue) {
    if (parameterID == PluginParameters::DRY_WET_ID.getParamID()) {
        dry_wet_mixer.setWetMixProportion(newValue);
    } else if (parameterID == PluginParameters::BACKEND_TYPE_ID.getParamID()) {
        int paramInt = (int) newValue;
        auto paramString = PluginParameters::backendTypes[paramInt];
#ifdef USE_TFLITE
        if (paramString == "TFLITE") inference_handler.set_inference_backend(anira::TFLITE);
#endif
#ifdef USE_ONNXRUNTIME
        if (paramString == "ONNX") inference_handler.set_inference_backend(anira::ONNX);
#endif
#ifdef USE_LIBTORCH
        if (paramString == "LIBTORCH") inference_handler.set_inference_backend(anira::LIBTORCH);
#endif
        if (paramString == "BYPASS") inference_handler.set_inference_backend(anira::CUSTOM);
    } else if (parameterID == PluginParameters::GAIN_ID.getParamID()) {
        pp_processor.set_input(newValue, 1, 0);
    }
}

void AudioPluginAudioProcessor::processesNonRealtime(const juce::AudioBuffer<float>& buffer) const {
    double durationInSeconds = static_cast<double>(buffer.getNumSamples()) / getSampleRate();
    auto durationInMilliseconds = std::chrono::duration<double, std::milli>(durationInSeconds * 1000);
    std::this_thread::sleep_for(durationInMilliseconds);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}

anira::InferenceManager &AudioPluginAudioProcessor::get_inference_manager() {
    return inference_handler.get_inference_manager();
}
