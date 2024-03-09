#ifndef NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
#define NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H

#include "JuceHeader.h"

class PluginParameters {
public:
    inline static const juce::ParameterID
            BACKEND_TYPE_ID = {"param_backend_type", 1},
            DRY_WET_ID = {"param_mix", 1}
    ;

    inline static const juce::String
            BACKEND_TYPE_NAME = "Backend Type",
            DRY_WET_NAME = "Dry/Wet"
    ;

    inline static juce::StringArray backendTypes {"TFLITE", "LIBTORCH", "ONNXRUNTIME", "NONE"};
    inline static juce::String defaultBackend {backendTypes[2]};

    static juce::StringArray getPluginParameterList();
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    inline static juce::StringArray parameterList;

    inline static juce::NormalisableRange<float> dryWetRange {0.0f, 1.0f, 0.01f};

};

#endif //NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
