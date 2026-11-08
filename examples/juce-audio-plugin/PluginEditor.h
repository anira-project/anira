#pragma once

#include "PluginProcessor.h"

/// The plugin's editor: the generic controls of its parameters, the line that says whether the
/// chosen backend keeps up on this machine (what prepare measured), and the "Try nonetheless"
/// button, shown while a backend that is not real-time here waits for the user's consent.
class PluginEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit PluginEditor(AudioPluginAudioProcessor& audio_processor);
    ~PluginEditor() override;

    void resized() override;

private:
    /// 4 Hz, on the message thread: the line and the button follow the processor's state.
    void timerCallback() override;

    AudioPluginAudioProcessor& m_processor;
    juce::GenericAudioProcessorEditor m_controls;
    juce::Label m_verdict;
    juce::TextButton m_try_nonetheless{"Try nonetheless"};

    static constexpr int k_verdict_height = 48;
    static constexpr int k_button_height = 28;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};
