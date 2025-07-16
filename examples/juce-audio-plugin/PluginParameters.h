#ifndef NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
#define NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H

#include "JuceHeader.h"

class PluginParameters {
    public:
    inline static const juce::ParameterID
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
            GAIN_ID = {"param_gain", 1},
#endif
#if MODEL_TO_USE == 7
            LATENT_0_ID = {"param_latent_0", 1},
            LATENT_1_ID = {"param_latent_1", 1},
            LATENT_2_ID = {"param_latent_2", 1},
            LATENT_3_ID = {"param_latent_3", 1},
#endif
            BACKEND_TYPE_ID = {"param_backend_type", 1},
            DRY_WET_ID = {"param_mix", 1}
    ;


    inline static const juce::String
#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
            GAIN_NAME = "Gain",
#endif
#if MODEL_TO_USE == 7
            LATENT_0_NAME = "Latent 0",
            LATENT_1_NAME = "Latent 1",
            LATENT_2_NAME = "Latent 2",
            LATENT_3_NAME = "Latent 3",
#endif
            BACKEND_TYPE_NAME = "Backend Type",
            DRY_WET_NAME = "Dry/Wet"
    ;

#if MODEL_TO_USE == 6 || MODEL_TO_USE == 7 || MODEL_TO_USE == 8
    inline static juce::StringArray backendTypes {"LIBTORCH", "BYPASS"};
#else
    inline static juce::StringArray backendTypes {"TFLITE", "LIBTORCH", "ONNX", "BYPASS"};
#endif
    inline static juce::String defaultBackend = "BYPASS";

    static juce::StringArray getPluginParameterList();
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    inline static juce::StringArray parameterList;

#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    inline static juce::NormalisableRange<float> gainRange {0.0f, 3.981072f, 0.00001f, 0.25f};
#endif
#if MODEL_TO_USE == 7
    inline static juce::NormalisableRange<float> latentRange {-1.0f, 1.0f, 0.00001f};
#endif
    inline static juce::NormalisableRange<float> dryWetRange {0.0f, 1.0f, 0.00001f};

#if MODEL_TO_USE == 7
    inline static juce::AudioParameterFloatAttributes latentAttributes = juce::AudioParameterFloatAttributes()
        .withStringFromValueFunction([](float x, auto) {
            return juce::String(x, 2);
        })
        .withLabel("Offset");
#endif

    inline static juce::AudioParameterFloatAttributes percentage_attributes = juce::AudioParameterFloatAttributes()
        .withStringFromValueFunction ([] (float x, auto) {
            return juce::String(x*100.f, 0) + " %";
        });

    inline static juce::AudioParameterFloatAttributes db_attributes = juce::AudioParameterFloatAttributes()
            .withStringFromValueFunction ([] (float x, auto) {
                auto db = juce::Decibels::gainToDecibels(x);
                return juce::String(db, 1) + " dB";
            });
};

#endif //NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
