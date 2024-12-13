#pragma once

#include <JuceHeader.h>

#include "PluginParameters.h"

#include <anira/anira.h>

#include "../../extras/models/cnn/CNNConfig.h"
#include "../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../extras/models/cnn/CNNBypassProcessor.h" // This one is only needed for the round trip test, when selecting the Custom backend
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNBypassProcessor.h" // Only needed for round trip test
#include "../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../extras/models/model-pool/SimpleGainConfig.h"
#include "../../extras/models/model-pool/SimpleStereoGainConfig.h"

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

    void processesNonRealtime(const juce::AudioBuffer<float>& buffer) const;

private:
    juce::AudioProcessorValueTreeState parameters;

    // Optional ContextConfig
    anira::ContextConfig anira_context_config;

#if MODEL_TO_USE == 1
    anira::InferenceConfig inference_config = cnn_config;
    CNNPrePostProcessor pp_processor;
    CNNBypassProcessor bypass_processor; // This one is only needed for the round trip test, when selecting the Custom backend
#elif MODEL_TO_USE == 2
    anira::InferenceConfig inference_config = hybridnn_config;
    HybridNNPrePostProcessor pp_processor;
    HybridNNBypassProcessor bypass_processor; // This one is only needed for the round trip test, when selecting the Custom backend
#elif MODEL_TO_USE == 3
    anira::InferenceConfig inference_config = rnn_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 4
    anira::InferenceConfig inference_config = gain_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 5
    anira::InferenceConfig inference_config = stereo_gain_config;
    anira::PrePostProcessor pp_processor;
#endif

    anira::InferenceHandler inference_handler;
    juce::dsp::DryWetMixer<float> dry_wet_mixer;

    std::atomic<bool> non_realtime = false;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
