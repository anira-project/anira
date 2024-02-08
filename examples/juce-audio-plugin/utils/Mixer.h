#ifndef NN_INFERENCE_TEMPLATE_MIXER_H
#define NN_INFERENCE_TEMPLATE_MIXER_H

#include "JuceHeader.h"

class Mixer {
public:
    Mixer();
    ~Mixer() = default;

    void prepare(const juce::dsp::ProcessSpec& spec);

    void setWetLatency(int numberOfSamples);
    void setDryWetProportion(float dryWet);
    void setDryWetProportionPercentage (float dryWet);

    void setDrySamples(juce::AudioBuffer<float>& dryBuffer);
    void setWetSamples(juce::AudioBuffer<float>& wetBuffer);

    void setMixingCurve(juce::dsp::DryWetMixingRule rule);
private:
    juce::dsp::DryWetMixer<float> mixer {48000};
    float dryWetProportion = 1.0f;
};

#endif //NN_INFERENCE_TEMPLATE_MIXER_H
