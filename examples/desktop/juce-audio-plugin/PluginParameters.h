#ifndef NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
#define NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H

#include "JuceHeader.h"

class PluginParameters {
public:
    inline static const juce::ParameterID
            GAIN_ID = {"param_gain", 1},
            BACKEND_TYPE_ID = {"param_backend_type", 1},
            DRY_WET_ID = {"param_mix", 1}
    ;

    inline static const juce::String
            GAIN_NAME = "Gain",
            BACKEND_TYPE_NAME = "Backend Type",
            DRY_WET_NAME = "Dry/Wet"
    ;

    inline static juce::StringArray backendTypes {"TFLITE", "LIBTORCH", "ONNX", "BYPASS"};
    inline static juce::String defaultBackend = "BYPASS";

    static juce::StringArray getPluginParameterList();
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    inline static juce::StringArray parameterList;

    inline static juce::NormalisableRange<float> gainRange {0.0f, 3.981072f, 0.00001f, 0.25f};
    inline static juce::NormalisableRange<float> dryWetRange {0.0f, 1.0f, 0.01f};

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
