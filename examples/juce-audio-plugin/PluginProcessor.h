#pragma once

// Include anira (and its LibTorch headers) before JuceHeader.h. JuceHeader.h
// does `using namespace juce;`, which leaks the type `juce::var` into the
// global namespace. LibTorch's custom_function.h uses an unqualified loop
// variable named `var`, and MSVC's two-phase template lookup then reports it as
// an ambiguous symbol (error C2872). Parsing LibTorch first avoids the clash.
#include <anira/anira.h>

#include <JuceHeader.h>

#include "PluginParameters.h"

#if MODEL_TO_USE == 0 || MODEL_TO_USE == 1
#if MODEL_TO_USE == 1
#include <BinaryData.h>
#endif
#include "../../extras/models/cnn/CNNConfig.h"
#include "../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../extras/models/cnn/CNNBypassProcessor.h"  // This one is only needed for the round trip test, when selecting the Custom backend
#elif MODEL_TO_USE == 2
#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNBypassProcessor.h"  // Only needed for round trip test
#elif MODEL_TO_USE == 3
#include "../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#elif MODEL_TO_USE == 4
#include "../../extras/models/model-pool/SimpleGainConfig.h"
#elif MODEL_TO_USE == 5
#include "../../extras/models/model-pool/SimpleStereoGainConfig.h"
#elif MODEL_TO_USE == 6
#include "../../extras/models/third-party/ircam-acids/RaveFunkDrumConfig.h"
#elif MODEL_TO_USE == 7
#include "../../extras/models/third-party/ircam-acids/RaveFunkDrumConfigEncoder.h"
#include "../../extras/models/third-party/ircam-acids/RaveFunkDrumConfigDecoder.h"
#endif

//==============================================================================
class AudioPluginAudioProcessor : public juce::AudioProcessor,
                                  private juce::AudioProcessorValueTreeState::Listener {
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
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
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getValueTreeState() { return parameters; }

private:
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    void processesNonRealtime(const juce::AudioBuffer<float>& buffer) const;

private:
    juce::AudioProcessorValueTreeState parameters;

#if MODEL_TO_USE == 8
    anira::JsonConfigLoader json_config_loader;
#endif

    // Optional ContextConfig
    anira::ContextConfig anira_context_config;

#if MODEL_TO_USE == 0
    anira::InferenceConfig inference_config = cnn_config;
    CNNPrePostProcessor pp_processor;
    CNNBypassProcessor bypass_processor;  // This one is only needed for the round trip test, when
                                          // selecting the Custom backend
#elif MODEL_TO_USE == 1
    std::vector<anira::ModelData> model_data = {
#ifdef USE_LIBTORCH
        {(void*)BinaryData::steerablenafxdynamic_pt,
         BinaryData::steerablenafxdynamic_ptSize,
         anira::InferenceBackend::LIBTORCH},
#endif
#ifdef USE_ONNXRUNTIME
        {(void*)BinaryData::steerablenafxlibtorchdynamic_onnx,
         BinaryData::steerablenafxlibtorchdynamic_onnxSize,
         anira::InferenceBackend::ONNX},
#endif
#ifdef USE_TFLITE
        {(void*)BinaryData::steerablenafxdynamic_tflite,
         BinaryData::steerablenafxdynamic_tfliteSize,
         anira::InferenceBackend::TFLITE},
#endif
    };
    anira::InferenceConfig inference_config = {model_data,
                                               tensor_shape_cnn_config,
                                               processing_spec_cnn_config,
                                               42.66f,
                                               2};
    CNNPrePostProcessor pp_processor;
    CNNBypassProcessor bypass_processor;  // This one is only needed for the round trip test, when
                                          // selecting the Custom backend
#elif MODEL_TO_USE == 2
    anira::InferenceConfig inference_config = hybridnn_config;
    HybridNNPrePostProcessor pp_processor;
    HybridNNBypassProcessor bypass_processor;  // This one is only needed for the round trip test,
                                               // when selecting the Custom backend
#elif MODEL_TO_USE == 3
    anira::InferenceConfig inference_config = rnn_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 4
    anira::InferenceConfig inference_config = gain_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 5
    anira::InferenceConfig inference_config = stereo_gain_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 6
    anira::InferenceConfig inference_config = rave_funk_drum_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 7
    anira::InferenceConfig inference_config_encoder = rave_funk_drum_encoder_config;
    anira::InferenceConfig inference_config_decoder = rave_funk_drum_decoder_config;
    anira::PrePostProcessor pp_processor_encoder;
    anira::PrePostProcessor pp_processor_decoder;
#elif MODEL_TO_USE == 8
    anira::InferenceConfig inference_config;
    anira::PrePostProcessor pp_processor;
#else
#error "MODEL_TO_USE must be defined to one of the available models."
#endif

#if MODEL_TO_USE != 7
    anira::InferenceHandler inference_handler;
#else
    anira::InferenceHandler inference_handler_encoder;
    anira::InferenceHandler inference_handler_decoder;
    int m_count_input_samples = 0;
#endif
    juce::dsp::DryWetMixer<float> dry_wet_mixer;

    std::atomic<bool> non_realtime = false;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
