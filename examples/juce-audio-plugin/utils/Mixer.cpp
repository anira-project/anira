#include "Mixer.h"

Mixer::Mixer() {
    mixer.setMixingRule(juce::dsp::DryWetMixingRule::linear);
    mixer.setWetMixProportion(dryWetProportion);
}

void Mixer::prepare(const juce::dsp::ProcessSpec &spec) {
    mixer.prepare(spec);
    mixer.reset();
}

void Mixer::setDryWetProportion(float dryWet) {
    dryWetProportion = dryWet;
    mixer.setWetMixProportion(dryWetProportion);
}

void Mixer::setDryWetProportionPercentage(float dryWet) {
    setDryWetProportion(dryWet/100);
}

void Mixer::setDrySamples(juce::AudioBuffer<float> &dryBuffer) {
    juce::dsp::AudioBlock<float> dryBlock(dryBuffer);
    mixer.pushDrySamples(dryBlock);
}

void Mixer::setWetSamples(juce::AudioBuffer<float> &wetBuffer) {
    juce::dsp::AudioBlock<float> wetBlock(wetBuffer);
    mixer.mixWetSamples(wetBlock);
}

void Mixer::setMixingCurve(juce::dsp::DryWetMixingRule rule) {
    mixer.setMixingRule(rule);
}

void Mixer::setWetLatency(int numberOfSamples) {
    mixer.setWetLatency((float) numberOfSamples);
}