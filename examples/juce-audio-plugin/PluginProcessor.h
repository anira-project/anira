#pragma once

#include <JuceHeader.h>
#include <anira/anira.h>

#include "PluginParameters.h"
#include "../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../extras/models/stateful-rnn/StatefulRNNPrePostProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../extras/models/cnn/CNNConfig.h"
#include "../../extras/models/cnn/CNNPrePostProcessor.h"

//==============================================================================
class AudioPluginAudioProcessor  : public juce::AudioProcessor, private juce::AudioProcessorValueTreeState::Listener
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    anira::InferenceManager &getInferenceManager();
    juce::AudioProcessorValueTreeState& getValueTreeState() { return parameters; }

private:
    void parameterChanged (const juce::String& parameterID, float newValue) override;
    void stereoToMono(juce::AudioBuffer<float> &targetMonoBlock, juce::AudioBuffer<float> &sourceBlock);
    void monoToStereo(juce::AudioBuffer<float> &targetStereoBlock, juce::AudioBuffer<float> &sourceBlock);

private:
    juce::AudioProcessorValueTreeState parameters;
    juce::AudioBuffer<float> monoBuffer;

#if MODEL_TO_USE == 1
    CNNPrePostProcessor prePostProcessor;
    anira::InferenceConfig inferenceConfig = ccnConfig;
#elif MODEL_TO_USE == 2
    HybridNNPrePostProcessor prePostProcessor;
    anira::InferenceConfig inferenceConfig = hybridNNConfig;
#elif MODEL_TO_USE == 3
    StatefulRNNPrePostProcessor prePostProcessor;
    anira::InferenceConfig inferenceConfig = statefulRNNConfig;
#endif
    anira::InferenceHandler inferenceHandler;

    juce::dsp::DryWetMixer<float> dryWetMixer;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
