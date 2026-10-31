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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../extras/models/BypassEngine.h"
#include "../../extras/models/model_files.h"
#include "PluginParameters.h"

#if MODEL_TO_USE == 1
#include <BinaryData.h>

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#endif
#if MODEL_TO_USE == 2
#include "../../extras/models/hybrid-nn/HybridNNProcessor.h"
#endif

// The model this plugin runs is picked at configure time (MODEL_TO_USE, see CMakeLists.txt):
// the model file and the contract file of a bundled model (extras/models). Variant 1 runs the
// CNN of variant 0 from files compiled into the plugin; variant 7 runs RAVE as an encoder and a
// decoder handler.
#if MODEL_TO_USE == 0 || MODEL_TO_USE == 1
#define MODEL_FILE k_cnn_model_json
#define CONTRACT_FILE k_cnn_contract_json
#elif MODEL_TO_USE == 2
#define MODEL_FILE k_hybridnn_model_json
#define CONTRACT_FILE k_hybridnn_contract_json
#elif MODEL_TO_USE == 3
#define MODEL_FILE k_rnn_model_json
#define CONTRACT_FILE k_rnn_contract_json
#elif MODEL_TO_USE == 4
#define MODEL_FILE k_gain_model_json
#define CONTRACT_FILE k_gain_contract_json
#elif MODEL_TO_USE == 5
#define MODEL_FILE k_stereo_gain_model_json
#define CONTRACT_FILE k_stereo_gain_contract_json
#elif MODEL_TO_USE == 6
#define MODEL_FILE k_rave_funk_drum_model_json
#define CONTRACT_FILE k_rave_funk_drum_contract_json
#elif MODEL_TO_USE != 7
#error "MODEL_TO_USE must be 0 to 7; see CMakeLists.txt"
#endif

#if MODEL_TO_USE == 1
/// An embedded file (juce_add_binary_data, CMakeLists.txt) by its original file name.
inline std::span<const std::byte> embedded(std::string_view file_name) {
    for (int i = 0; i < BinaryData::namedResourceListSize; ++i) {
        const char* name = BinaryData::namedResourceList[i];
        if (file_name == BinaryData::getNamedResourceOriginalFilename(name)) {
            int size = 0;
            const char* data = BinaryData::getNamedResource(name, size);
            return std::as_bytes(std::span{data, static_cast<size_t>(size)});
        }
    }
    return {};
}

/// The model file and every export it names, from the plugin binary: each entry's source is
/// the embedded file of the same name, borrowed (BinaryData is static).
inline anira::ModelConfig embedded_model_config() {
    const std::span<const std::byte> json = embedded("cnn.model.json");
    anira::ModelConfig config =
        anira::ModelConfig::from_json({reinterpret_cast<const char*>(json.data()), json.size()});
    for (uint32_t i = 0; i < config.model_count(); ++i) {
        const std::string name = std::filesystem::path(config.model_path(i)).filename().string();
        config.set_model_bytes(i, embedded(name), ANIRA_BYTES_BORROW);
    }
    return config;
}

inline anira::ContractHandle embedded_contract() {
    const std::span<const std::byte> json = embedded("cnn.contract.json");
    return anira::ContractHandle::from_json(
        {reinterpret_cast<const char*>(json.data()), json.size()});
}
#endif

/// A model running on one handler: its model config (with the entry of the bypass engine, the
/// plan behind the "BYPASS" choice), its contract, the handler, and the plan report of the
/// last prepare, which processBlock looks the backend choice up in.
struct AniraModel {
    AniraModel(anira::Context& context, anira::ModelConfig config, anira::ContractHandle hard)
        : model_config([&] {
            config.add_model(k_bypass_engine_id);
            return std::move(config);
        }())
        , contract(std::move(hard))
        , handler(context,
                  anira::Pipeline()
                      .add_engine(std::make_shared<BypassEngine>())
                      .add_inference(model_config)
#if MODEL_TO_USE == 2
                      .add_processor(std::make_shared<HybridNNProcessor>())
#endif
          ) {
    }

    /// The host geometry completes the contract, blocks of 1 to block_max samples; prepare
    /// returns the plan report.
    void prepare(double block_max, double sample_rate) {
        contract.hard_geometry(1, block_max, sample_rate);
        plan_report = handler.prepare(contract);
    }
    /// The plan of the report that runs the query's backend; none: the selection stays.
    void select(const anira::PlanQuery& query) noexcept {
        if (const std::optional<uint32_t> plan = plan_report.find(query)) {
            handler.set_plan(*plan);
        }
    }

    anira::ModelConfig model_config;  // outlives the handler: an embedded entry is borrowed
    anira::ContractHandle contract;
    anira::InferenceHandler handler;
    anira::PlanReport plan_report{nullptr};
};

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

    // The library default (half the hardware threads for inference) is what this plugin
    // wants; section 3.1 of the usage guide shows how to change it.
    anira::Context context{anira::ContextConfig{}};
#if MODEL_TO_USE == 7
    AniraModel encoder{context,
                       anira::ModelConfig::from_file(k_rave_funk_drum_encoder_model_json),
                       anira::ContractHandle::from_file(k_rave_funk_drum_encoder_contract_json)};
    AniraModel decoder{context,
                       anira::ModelConfig::from_file(k_rave_funk_drum_decoder_model_json),
                       anira::ContractHandle::from_file(k_rave_funk_drum_decoder_contract_json)};
    // One latent frame: four channels of one sample, a planar tensor over them.
    std::array<float, 4> latent_space{};
    std::array<float*, 4> latent_space_ptrs{};
    anira::Tensor latent{};
    int m_count_input_samples = 0;
#elif MODEL_TO_USE == 1
    AniraModel model{context, embedded_model_config(), embedded_contract()};
#else
    AniraModel model{context,
                     anira::ModelConfig::from_file(MODEL_FILE),
                     anira::ContractHandle::from_file(CONTRACT_FILE)};
#endif
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    // The gain parameter, written into the model's Static input (slot 1) on the audio thread.
    std::atomic<float> gain{1.0F};
    float gain_value = 1.0F;
    anira::Tensor gain_tensor{};
#endif
    // The "Backend Type" choice: stored by the parameter listener, applied by processBlock.
    std::vector<anira::PlanQuery> backend_queries;
    std::atomic<int> backend{0};

    // The host's audio as one planar tensor over its channel pointers.
    std::array<float*, 2> channels{};
    anira::Tensor audio{};

    juce::dsp::DryWetMixer<float> dry_wet_mixer;

    std::atomic<bool> non_realtime = false;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
