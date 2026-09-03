#pragma once

// Include anira (and its LibTorch headers) before JuceHeader.h. JuceHeader.h
// does `using namespace juce;`, which leaks the type `juce::var` into the
// global namespace. LibTorch's custom_function.h uses an unqualified loop
// variable named `var`, and MSVC's two-phase template lookup then reports it as
// an ambiguous symbol (error C2872). Parsing LibTorch first avoids the clash.
// clang-format off
#include <anira/anira.h>
#include <anira/anira.hpp>
#include <anira/compat/v3_to_v2.h>
#include <JuceHeader.h>
// clang-format on

#include "../../extras/models/model_files.h"
#include "PluginParameters.h"

// The model this plugin runs is picked at configure time (MODEL_TO_USE, see CMakeLists.txt).
// Every variant is configured the same way: the model file and the contract file of a bundled
// model (extras/models, loaded with anira/anira.hpp) are bridged to the 2.x runtime classes
// this pre-release's InferenceHandler still takes (anira/compat/v3_to_v2.h). What differs per
// variant is the pair of files and, for the two models with their own pre/post processing,
// the processor types. Variant 1 runs the CNN of variant 0 with every file compiled into the
// plugin; variant 7 runs RAVE as an encoder and a decoder handler.
#if MODEL_TO_USE == 0 || MODEL_TO_USE == 1
#include "../../extras/models/cnn/CNNBypassProcessor.h"
#include "../../extras/models/cnn/CNNPrePostProcessor.h"
#elif MODEL_TO_USE == 2
#include "../../extras/models/hybrid-nn/HybridNNBypassProcessor.h"
#include "../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#endif
#if MODEL_TO_USE == 1
#include <BinaryData.h>

#include <cstddef>
#include <span>
#include <string_view>
#endif

namespace juce_plugin_example {

#if MODEL_TO_USE == 0 || MODEL_TO_USE == 1
inline constexpr const char* k_model_json = k_cnn_model_json;
inline constexpr const char* k_contract_json = k_cnn_contract_json;
using PrePostProcessor = CNNPrePostProcessor;
using BypassProcessor = CNNBypassProcessor;
#elif MODEL_TO_USE == 2
inline constexpr const char* k_model_json = k_hybridnn_model_json;
inline constexpr const char* k_contract_json = k_hybridnn_contract_json;
using PrePostProcessor = HybridNNPrePostProcessor;
using BypassProcessor = HybridNNBypassProcessor;
#elif MODEL_TO_USE == 3
inline constexpr const char* k_model_json = k_rnn_model_json;
inline constexpr const char* k_contract_json = k_rnn_contract_json;
using PrePostProcessor = anira::PrePostProcessor;
#elif MODEL_TO_USE == 4
inline constexpr const char* k_model_json = k_gain_model_json;
inline constexpr const char* k_contract_json = k_gain_contract_json;
using PrePostProcessor = anira::PrePostProcessor;
#elif MODEL_TO_USE == 5
inline constexpr const char* k_model_json = k_stereo_gain_model_json;
inline constexpr const char* k_contract_json = k_stereo_gain_contract_json;
using PrePostProcessor = anira::PrePostProcessor;
#elif MODEL_TO_USE == 6
inline constexpr const char* k_model_json = k_rave_funk_drum_model_json;
inline constexpr const char* k_contract_json = k_rave_funk_drum_contract_json;
using PrePostProcessor = anira::PrePostProcessor;
#elif MODEL_TO_USE == 7
// The encoder and the decoder are two entry points of one TorchScript file, run as two
// handlers; their files are named in the class below.
#else
#error "MODEL_TO_USE must be 0 to 7; see CMakeLists.txt"
#endif

#if MODEL_TO_USE == 1
/// Variant 1 reads nothing from disk: cnn.model.json, cnn.contract.json and the four exports
/// the model file names are compiled into the plugin (juce_add_binary_data, CMakeLists.txt).
/// The model config is loaded from the embedded JSON text, then each entry's source is replaced
/// with the embedded bytes of its engine (borrowed: BinaryData is static and outlives the
/// plugin). The description of the model stays in the file; only where the bytes come from
/// changes.
inline std::span<const std::byte> embedded(const char* data, int size) {
    return std::as_bytes(std::span{data, static_cast<size_t>(size)});
}

inline anira::ModelConfig embedded_cnn_model_config() {
    anira::ModelConfig config = anira::ModelConfig::from_json(
        std::string_view{BinaryData::cnn_model_json,
                         static_cast<size_t>(BinaryData::cnn_model_jsonSize)});
    for (uint32_t i = 0; i < config.model_count(); ++i) {
        switch (config.model_engine(i)) {
            case ANIRA_ENGINE_LIBTORCH:
                config.set_model_bytes(i,
                                       embedded(BinaryData::steerablenafxdynamic_pt,
                                                BinaryData::steerablenafxdynamic_ptSize),
                                       ANIRA_BYTES_BORROW);
                break;
            case ANIRA_ENGINE_ONNXRUNTIME:
                config.set_model_bytes(i,
                                       embedded(BinaryData::steerablenafxlibtorchdynamic_onnx,
                                                BinaryData::steerablenafxlibtorchdynamic_onnxSize),
                                       ANIRA_BYTES_BORROW);
                break;
            case ANIRA_ENGINE_TFLITE:
            case ANIRA_ENGINE_LITERT:
                config.set_model_bytes(i,
                                       embedded(BinaryData::steerablenafxdynamic_tflite,
                                                BinaryData::steerablenafxdynamic_tfliteSize),
                                       ANIRA_BYTES_BORROW);
                break;
            case ANIRA_ENGINE_EXECUTORCH:
                config.set_model_bytes(i,
                                       embedded(BinaryData::steerablenafxexecutorch_pte,
                                                BinaryData::steerablenafxexecutorch_pteSize),
                                       ANIRA_BYTES_BORROW);
                break;
            default: break;
        }
    }
    return config;
}

inline anira::ContractHandle embedded_cnn_contract() {
    return anira::ContractHandle::from_json(
        std::string_view{BinaryData::cnn_contract_json,
                         static_cast<size_t>(BinaryData::cnn_contract_jsonSize)});
}
#endif

}  // namespace juce_plugin_example

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

    // The configuration: the model file and the contract file, bridged to the 2.x
    // InferenceConfig. The model config outlives the InferenceConfig and the handler (an
    // embedded-bytes entry is borrowed), so it is declared first; members are destroyed in
    // reverse order. The processors and the handler are constructed in PluginProcessor.cpp.
#if MODEL_TO_USE != 7
#if MODEL_TO_USE == 1
    anira::ModelConfig model_config = juce_plugin_example::embedded_cnn_model_config();
    anira::ContractHandle contract = juce_plugin_example::embedded_cnn_contract();
#else
    anira::ModelConfig model_config =
        anira::ModelConfig::from_file(juce_plugin_example::k_model_json);
    anira::ContractHandle contract =
        anira::ContractHandle::from_file(juce_plugin_example::k_contract_json);
#endif
    anira::InferenceConfig inference_config =
        anira::v3compat::to_inference_config(model_config,
                                             contract,
                                             anira::v3compat::enabled_engines());
    juce_plugin_example::PrePostProcessor pp_processor;
#if MODEL_TO_USE <= 2
    // The round trip that outputs audio when the Custom backend is selected; it must match the
    // custom pre/post processor.
    juce_plugin_example::BypassProcessor bypass_processor;
#endif
    anira::InferenceHandler inference_handler;
#else
    // RAVE as two handlers: the encoder turns audio into latent frames, the decoder (anchored
    // on its audio output, one latent frame per 2048 samples) turns them back into audio.
    anira::ModelConfig model_config_encoder =
        anira::ModelConfig::from_file(k_rave_funk_drum_encoder_model_json);
    anira::ContractHandle contract_encoder =
        anira::ContractHandle::from_file(k_rave_funk_drum_encoder_contract_json);
    anira::InferenceConfig inference_config_encoder =
        anira::v3compat::to_inference_config(model_config_encoder,
                                             contract_encoder,
                                             anira::v3compat::enabled_engines());
    anira::PrePostProcessor pp_processor_encoder;
    anira::InferenceHandler inference_handler_encoder;

    anira::ModelConfig model_config_decoder =
        anira::ModelConfig::from_file(k_rave_funk_drum_decoder_model_json);
    anira::ContractHandle contract_decoder =
        anira::ContractHandle::from_file(k_rave_funk_drum_decoder_contract_json);
    anira::InferenceConfig inference_config_decoder =
        anira::v3compat::to_inference_config(model_config_decoder,
                                             contract_decoder,
                                             anira::v3compat::enabled_engines());
    anira::PrePostProcessor pp_processor_decoder;
    anira::InferenceHandler inference_handler_decoder;
    int m_count_input_samples = 0;
#endif
    juce::dsp::DryWetMixer<float> dry_wet_mixer;

    std::atomic<bool> non_realtime = false;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
