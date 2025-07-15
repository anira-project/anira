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
#if MODEL_TO_USE == 7
    params.push_back (std::make_unique<juce::AudioParameterFloat> (LATENT_0_ID,
                                                                    LATENT_0_NAME,
                                                                    latentRange,
                                                                    0.0f,
                                                                    latentAttributes));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (LATENT_1_ID,
                                                                    LATENT_1_NAME,
                                                                    latentRange,
                                                                    0.0f,
                                                                    latentAttributes));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (LATENT_2_ID,
                                                                    LATENT_2_NAME,
                                                                    latentRange,
                                                                    0.0f,
                                                                    latentAttributes));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (LATENT_3_ID,
                                                                    LATENT_3_NAME,
                                                                    latentRange,
                                                                    0.0f,
                                                                    latentAttributes));
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
