#include "PluginParameters.h"

juce::AudioProcessorValueTreeState::ParameterLayout PluginParameters::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

#if MODEL_TO_USE == 4 || MODEL_TO_USE == 5
    params.push_back (std::make_unique<juce::AudioParameterFloat> (GAIN_ID,
                                                                   GAIN_NAME,
                                                                   gainRange,
                                                                   1.0f,
                                                                   db_attributes));
#endif

    params.push_back (std::make_unique<juce::AudioParameterChoice> (BACKEND_TYPE_ID,
                                                                    BACKEND_TYPE_NAME,
                                                                    backendTypes,
                                                                    backendTypes.indexOf(defaultBackend)));

    params.push_back( std::make_unique<juce::AudioParameterFloat>  (DRY_WET_ID,
                                                                    DRY_WET_NAME,
                                                                    dryWetRange,
                                                                    1.0f,
                                                                    percentage_attributes));

    if (parameterList.isEmpty()) {
        for (const auto & param : params) {
            parameterList.add(param->getParameterID());
        }
    }

    return { params.begin(), params.end() };
}

juce::StringArray PluginParameters::getPluginParameterList() {
    return parameterList;
}
