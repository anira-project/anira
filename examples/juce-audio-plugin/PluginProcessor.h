#pragma once

// Include anira before JuceHeader.h. JuceHeader.h does `using namespace juce;`, which leaks the
// type `juce::var` into the global namespace; the engine headers a host may include beside
// anira (LibTorch's custom_function.h uses an unqualified loop variable named `var`) then hit
// MSVC's two-phase lookup as an ambiguous symbol (error C2872). Parsing anira first avoids it.
// clang-format off
#include <anira/anira.hpp>
#include <JuceHeader.h>
// clang-format on

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include "../../extras/models/BypassEngine.h"
#include "../../extras/models/model_files.h"
#include "PluginParameters.h"

// The model this plugin runs is picked at configure time (MODEL_TO_USE, see CMakeLists.txt).
// Every variant is configured the same way: the model file and the contract file of a bundled
// model (extras/models), a Pipeline of its inference stage (the model's engines and the bypass
// engine, BypassEngine.h, behind the "BYPASS" setting) and, for the HybridNN model, the stage
// that builds its windows (HybridNNStage.h), one anira::InferenceHandler on a Context, prepared
// with the Hard contract of the host's block. The "Backend Type" parameter picks a plan of the
// PlanReport prepare returns. Variant 1 runs the CNN of variant 0 with every file compiled into
// the plugin; variant 7 runs RAVE as an encoder and a decoder handler.
#if MODEL_TO_USE == 2
#include "../../extras/models/hybrid-nn/HybridNNStage.h"
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
#elif MODEL_TO_USE == 2
inline constexpr const char* k_model_json = k_hybridnn_model_json;
inline constexpr const char* k_contract_json = k_hybridnn_contract_json;
#elif MODEL_TO_USE == 3
inline constexpr const char* k_model_json = k_rnn_model_json;
inline constexpr const char* k_contract_json = k_rnn_contract_json;
#elif MODEL_TO_USE == 4
inline constexpr const char* k_model_json = k_gain_model_json;
inline constexpr const char* k_contract_json = k_gain_contract_json;
#elif MODEL_TO_USE == 5
inline constexpr const char* k_model_json = k_stereo_gain_model_json;
inline constexpr const char* k_contract_json = k_stereo_gain_contract_json;
#elif MODEL_TO_USE == 6
inline constexpr const char* k_model_json = k_rave_funk_drum_model_json;
inline constexpr const char* k_contract_json = k_rave_funk_drum_contract_json;
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
        switch (config.model_engine(i).kind) {
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

/// The model config with the entry of the bypass engine, which needs no model file: the plan
/// behind the "BYPASS" setting.
inline anira::ModelConfig with_bypass(anira::ModelConfig config) {
    config.add_model(k_bypass_engine_id);
    return config;
}

/// The pipeline of a model config: its inference stage, which carries the bypass engine, and
/// the stage of the model when it has one of its own (the HybridNN windows; every other model
/// takes anira's default fill and push).
inline anira::Pipeline make_pipeline(const anira::ModelConfig& model) {
    anira::Pipeline pipeline;
    pipeline.add(anira::stage::Inference(model).engine(std::make_shared<BypassEngine>()));
#if MODEL_TO_USE == 2
    pipeline.add(anira::stage::Custom(std::make_shared<HybridNNStage>()));
#endif
    return pipeline;
}

/// The plan of a handler's report behind each "Backend Type" choice (PluginParameters::
/// backendTypes): assigned on the main thread after prepare, read by parameterChanged on any
/// thread.
class BackendPlans {
public:
    BackendPlans() {
        for (std::atomic<uint32_t>& plan : m_plans) { plan.store(k_none); }
    }

    void assign(const anira::PlanReport& report) {
        for (std::atomic<uint32_t>& plan : m_plans) { plan.store(k_none); }
        const std::vector<anira_plan_info> plans = report.plans();
        for (uint32_t i = 0; i < plans.size(); ++i) {
            const int choice = PluginParameters::backendTypes.indexOf(choice_of(plans[i]));
            if (choice < 0 || static_cast<size_t>(choice) >= m_plans.size()) { continue; }
            uint32_t expected = k_none;  // the first plan of an engine
            m_plans[static_cast<size_t>(choice)].compare_exchange_strong(expected, i);
        }
    }

    std::optional<uint32_t> plan_of(int choice) const noexcept {
        if (choice < 0 || static_cast<size_t>(choice) >= m_plans.size()) { return std::nullopt; }
        const uint32_t plan = m_plans[static_cast<size_t>(choice)].load();
        return plan == k_none ? std::nullopt : std::optional<uint32_t>(plan);
    }

private:
    static constexpr uint32_t k_none = UINT32_MAX;

    /// The parameter choice a plan's engine stands behind.
    static juce::String choice_of(const anira_plan_info& plan) {
        switch (plan.engine) {
            case ANIRA_ENGINE_LIBTORCH: return "LIBTORCH";
            case ANIRA_ENGINE_ONNXRUNTIME: return "ONNX";
            case ANIRA_ENGINE_TFLITE: return "TFLITE";
            case ANIRA_ENGINE_LITERT: return "LITERT";
            case ANIRA_ENGINE_EXECUTORCH: return "EXECUTORCH";
            case ANIRA_ENGINE_CUSTOM:
                return juce::String(plan.engine_id) == k_bypass_engine_id ? "BYPASS" : "";
            default: return "";
        }
    }

    std::array<std::atomic<uint32_t>, 8> m_plans;
};

}  // namespace juce_plugin_example

class AudioPluginAudioProcessor : public juce::AudioProcessor,
                                  private juce::AudioProcessorValueTreeState::Listener {
public:
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

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

    // The context of the handlers: the library default (half the hardware threads for
    // inference) is what this plugin wants; section 3.1 of the usage guide shows how to change
    // it. The model config and the contract outlive the handler (an embedded-bytes entry is
    // borrowed), so they are declared before it; members are destroyed in reverse order.
    anira::Context context{anira::ContextConfig{}};
#if MODEL_TO_USE != 7
#if MODEL_TO_USE == 1
    anira::ModelConfig model_config =
        juce_plugin_example::with_bypass(juce_plugin_example::embedded_cnn_model_config());
    anira::ContractHandle contract = juce_plugin_example::embedded_cnn_contract();
#else
    anira::ModelConfig model_config = juce_plugin_example::with_bypass(
        anira::ModelConfig::from_file(juce_plugin_example::k_model_json));
    anira::ContractHandle contract =
        anira::ContractHandle::from_file(juce_plugin_example::k_contract_json);
#endif
    anira::InferenceHandler inference_handler{context,
                                              juce_plugin_example::make_pipeline(model_config)};
    juce_plugin_example::BackendPlans backend_plans;
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    // The gain, a Static input (slot 1): the parameter's value, written into the handler from
    // the audio thread at the start of every block.
    std::atomic<float> gain{1.0F};
    float gain_value = 1.0F;
    anira::Tensor gain_tensor{};
#endif
#else
    // RAVE as two handlers: the encoder turns audio into latent frames, the decoder (anchored
    // on its audio output, one latent frame per 2048 samples) turns them back into audio.
    anira::ModelConfig model_config_encoder = juce_plugin_example::with_bypass(
        anira::ModelConfig::from_file(k_rave_funk_drum_encoder_model_json));
    anira::ContractHandle contract_encoder =
        anira::ContractHandle::from_file(k_rave_funk_drum_encoder_contract_json);
    anira::InferenceHandler inference_handler_encoder{
        context,
        juce_plugin_example::make_pipeline(model_config_encoder)};
    juce_plugin_example::BackendPlans backend_plans_encoder;

    anira::ModelConfig model_config_decoder = juce_plugin_example::with_bypass(
        anira::ModelConfig::from_file(k_rave_funk_drum_decoder_model_json));
    anira::ContractHandle contract_decoder =
        anira::ContractHandle::from_file(k_rave_funk_drum_decoder_contract_json);
    anira::InferenceHandler inference_handler_decoder{
        context,
        juce_plugin_example::make_pipeline(model_config_decoder)};
    juce_plugin_example::BackendPlans backend_plans_decoder;

    // One latent frame: four channels of one sample, as a planar tensor over them.
    std::array<float, 4> latent_space{};
    std::array<float*, 4> latent_space_ptrs{};
    anira::Tensor latent{};
    int m_count_input_samples = 0;
#endif
    // The host's audio as one planar tensor over its channel pointers, built in prepareToPlay;
    // processBlock points it at the buffer of each call and sets the block length (shape[1]).
    std::array<float*, 2> channels{};
    anira::Tensor audio{};

    juce::dsp::DryWetMixer<float> dry_wet_mixer;

    std::atomic<bool> non_realtime = false;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
