#include "PluginProcessor.h"

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor() 
        : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
        parameters (*this, nullptr, juce::Identifier (getName()), PluginParameters::createParameterLayout()),
#if MODEL_TO_USE == 1
        inferenceHandler(prePostProcessor, cnnConfig)
#elif MODEL_TO_USE == 2
        inferenceHandler(prePostProcessor, hybridNNConfig)
#elif MODEL_TO_USE == 3
        inferenceHandler(prePostProcessor, statefulRNNConfig)
#endif
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
    juce::dsp::ProcessSpec monoSpec {sampleRate,
                                 static_cast<juce::uint32>(samplesPerBlock),
                                 static_cast<juce::uint32>(1)};

    anira::HostAudioConfig monoConfig {
        1,
        (size_t) samplesPerBlock,
        sampleRate
    };

    dryWetMixer.prepare(monoSpec);

    monoBuffer.setSize(1, samplesPerBlock);
    inferenceHandler.prepare(monoConfig);

    auto newLatency = inferenceHandler.getLatency();
    // dryWetMixer.setWetLatency(newLatency);
    setLatencySamples(newLatency);

}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);

    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    stereoToMono(monoBuffer, buffer);
    dryWetMixer.pushDrySamples(monoBuffer);

    auto inferenceBuffer = const_cast<float **>(monoBuffer.getArrayOfWritePointers());
    inferenceHandler.process(inferenceBuffer, (size_t) buffer.getNumSamples());

    dryWetMixer.mixWetSamples(monoBuffer);
    monoToStereo(buffer, monoBuffer);
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
        dryWetMixer.setWetMixProportion(newValue);
    } else if (parameterID == PluginParameters::BACKEND_TYPE_ID.getParamID()) {
        const auto paramInt = static_cast<int>(newValue);
        auto paramString = PluginParameters::backendTypes.getReference(paramInt);
#ifdef USE_TFLITE
        if (paramString == "TFLITE") inferenceHandler.setInferenceBackend(anira::TFLITE);
#endif
#ifdef USE_ONNXRUNTIME
        if (paramString == "ONNXRUNTIME") inferenceHandler.setInferenceBackend(anira::ONNX);
#endif
#ifdef USE_LIBTORCH
        if (paramString == "LIBTORCH") inferenceHandler.setInferenceBackend(anira::LIBTORCH);
#endif
    }
}
//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}

anira::InferenceManager &AudioPluginAudioProcessor::getInferenceManager() {
    return inferenceHandler.getInferenceManager();
}

void AudioPluginAudioProcessor::stereoToMono(juce::AudioBuffer<float> &targetMonoBlock,
                                             juce::AudioBuffer<float> &sourceBlock) {
    if (sourceBlock.getNumChannels() == 1) {
        targetMonoBlock.makeCopyOf(sourceBlock);
    } else {
        auto nSamples = sourceBlock.getNumSamples();

        auto monoWrite = targetMonoBlock.getWritePointer(0);
        auto lRead = sourceBlock.getReadPointer(0);
        auto rRead = sourceBlock.getReadPointer(1);

        juce::FloatVectorOperations::copy(monoWrite, lRead, nSamples);
        juce::FloatVectorOperations::add(monoWrite, rRead, nSamples);
        juce::FloatVectorOperations::multiply(monoWrite, 0.5f, nSamples);
    }
}

void AudioPluginAudioProcessor::monoToStereo(juce::AudioBuffer<float> &targetStereoBlock,
                                             juce::AudioBuffer<float> &sourceBlock) {
    if (targetStereoBlock.getNumChannels() == 1) {
        targetStereoBlock.makeCopyOf(sourceBlock);
    } else {
        auto nSamples = sourceBlock.getNumSamples();

        auto lWrite = targetStereoBlock.getWritePointer(0);
        auto rWrite = targetStereoBlock.getWritePointer(1);
        auto monoRead = sourceBlock.getReadPointer(0);

        juce::FloatVectorOperations::copy(lWrite, monoRead, nSamples);
        juce::FloatVectorOperations::copy(rWrite, monoRead, nSamples);
    }
}
