#include "PluginEditor.h"

PluginEditor::PluginEditor(AudioPluginAudioProcessor& audio_processor)
    : AudioProcessorEditor(audio_processor)
    , m_processor(audio_processor)
    , m_controls(audio_processor) {
    addAndMakeVisible(m_controls);
    m_verdict.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(m_verdict);
    // The user agrees to run the chosen backend although prepare found it not real-time here.
    m_try_nonetheless.onClick = [this] { m_processor.try_nonetheless(); };
    addChildComponent(m_try_nonetheless);
    setSize(m_controls.getWidth(), m_controls.getHeight() + k_verdict_height + k_button_height);
    timerCallback();
    startTimerHz(4);
}

PluginEditor::~PluginEditor() {
    stopTimer();
}

void PluginEditor::resized() {
    juce::Rectangle<int> area = getLocalBounds();
    m_controls.setBounds(area.removeFromTop(m_controls.getHeight()));
    m_verdict.setBounds(area.removeFromTop(k_verdict_height).reduced(8, 0));
    m_try_nonetheless.setBounds(area.removeFromTop(k_button_height).reduced(8, 2));
}

void PluginEditor::timerCallback() {
    m_verdict.setText(m_processor.verdict_line(), juce::dontSendNotification);
    m_try_nonetheless.setVisible(m_processor.waits_for_consent());
}
