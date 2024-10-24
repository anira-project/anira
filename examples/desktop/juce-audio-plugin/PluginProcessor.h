#pragma once

#include <JuceHeader.h>

#include "PluginParameters.h"

#include <anira/anira.h>

#include "../../../extras/desktop/models/cnn/CNNConfig.h"
#include "../../../extras/desktop/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/desktop/models/cnn/advanced-configs/CNNNoneProcessor.h" // This one is only needed for the round trip test, when selecting the None backend
#include "../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/desktop/models/hybrid-nn/advanced-configs/HybridNNNoneProcessor.h" // Only needed for round trip test
#include "../../../extras/desktop/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../extras/desktop/models/stateful-rnn/StatefulRNNPrePostProcessor.h"

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

    juce::AudioProcessorValueTreeState& getValueTreeState() { return parameters; }
    anira::InferenceManager &get_inference_manager();

private:
    void parameterChanged (const juce::String& parameterID, float newValue) override;
    void stereoToMono(juce::AudioBuffer<float> &targetMonoBlock, juce::AudioBuffer<float> &sourceBlock);
    void monoToStereo(juce::AudioBuffer<float> &targetStereoBlock, juce::AudioBuffer<float> &sourceBlock);

private:
    juce::AudioProcessorValueTreeState parameters;
    juce::AudioBuffer<float> mono_buffer;

#if MODEL_TO_USE == 1
    anira::InferenceConfig inference_config = cnn_config;
    CNNPrePostProcessor pp_processor;
    CNNNoneProcessor none_processor; // This one is only needed for the round trip test, when selecting the None backend
#elif MODEL_TO_USE == 2
    anira::InferenceConfig inference_config = hybridnn_config;
    HybridNNPrePostProcessor pp_processor;
    HybridNNNoneProcessor none_processor; // This one is only needed for the round trip test, when selecting the None backend
#elif MODEL_TO_USE == 3
    anira::InferenceConfig inference_config = rnn_config;
    StatefulRNNPrePostProcessor pp_processor;
#endif
    anira::InferenceHandler inference_handler;

    juce::dsp::DryWetMixer<float> dry_wet_mixer;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
