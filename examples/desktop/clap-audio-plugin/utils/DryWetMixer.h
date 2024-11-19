#ifndef ANIRA_DRYWETMIXER_H
#define ANIRA_DRYWETMIXER_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace clap_plugin_example::utils {

class DryWetMixer {
public:
    DryWetMixer();

    void prepare(double sample_rate, size_t buffer_size, size_t latency_samples);
    void push_dry_sample(float dry_sample);

    float mix_wet_sample(float wet_sample);

    void set_mix(float new_mix);

private:
    std::vector<float> m_delay_buffer;

    double m_sample_rate;
    size_t m_buffer_size;

    size_t m_latency_samples;
    float m_mix;

    size_t m_write_index;
    size_t m_read_index;
};

}

#endif //ANIRA_DRYWETMIXER_H
