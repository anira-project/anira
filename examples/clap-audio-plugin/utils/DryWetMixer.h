#ifndef ANIRA_DRYWETMIXER_H
#define ANIRA_DRYWETMIXER_H

#include <cstddef>
#include <vector>

namespace clap_plugin_example::utils {

/// Mixes the wet signal with the dry one delayed by the wet path's latency. A process call pushes
/// its whole block of dry samples first (push_dry_sample), then mixes each wet sample with the
/// dry sample of `latency` samples earlier (mix_wet_sample).
class DryWetMixer {
public:
    /// Sizes the delay line for a latency of `latency_samples` and blocks of at most
    /// `buffer_size` samples, and fills it with silence.
    void prepare(double sample_rate, size_t buffer_size, size_t latency_samples);
    void push_dry_sample(float dry_sample);

    float mix_wet_sample(float wet_sample);

    void set_mix(float new_mix);

private:
    std::vector<float> m_delay_buffer;

    double m_sample_rate = 0.0;
    size_t m_buffer_size = 0;

    size_t m_latency_samples = 0;
    float m_mix = 1.0F;

    size_t m_write_index = 0;
    size_t m_read_index = 0;
};

}  // namespace clap_plugin_example::utils

#endif  // ANIRA_DRYWETMIXER_H
