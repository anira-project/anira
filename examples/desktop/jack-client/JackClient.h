#pragma once

#include <jack/jack.h>
#include <getopt.h>
#include <anira/anira.h>
#include <semaphore>

#include "../../../extras/desktop/models/cnn/CNNConfig.h"
#include "../../../extras/desktop/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/desktop/models/cnn/advanced-configs/CNNNoneProcessor.h" // This one is only needed for the round trip test, when selecting the None backend
#include "../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/desktop/models/hybrid-nn/advanced-configs/HybridNNNoneProcessor.h" // Only needed for round trip test
#include "../../../extras/desktop/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../extras/desktop/models/stateful-rnn/StatefulRNNPrePostProcessor.h"

class JackClient {

public:
    JackClient() = delete;
    JackClient([[ maybe_unused ]] int argc, char *argv[]);
    ~JackClient();

    void prepare();
    void prepare(anira::HostAudioConfig host_audio_config);

    static int process (jack_nframes_t nframes, void *arg);
    static void shutdown (void *arg);
    static int buffer_size_callback(jack_nframes_t nframes, void *arg);
    static int sample_rate_callback(jack_nframes_t nframes, void *arg);
    
private:
    void parse_args(int argc, char* argv[]);

    jack_client_t* m_client;
    jack_status_t m_status;
    size_t m_n_input_channels;
    size_t m_n_output_channels; 
    jack_port_t **input_ports;
    jack_port_t **output_ports;
    jack_default_audio_sample_t **in;
    jack_default_audio_sample_t **out;

    size_t m_nframes;
    double m_samplerate;

    const char* m_client_name = nullptr;
    bool verbose = false;

#if MODEL_TO_USE == 1
    anira::InferenceConfig inferenceConfig = cnnConfig;
    CNNPrePostProcessor prePostProcessor;
    CNNNoneProcessor noneProcessor; // This one is only needed for the round trip test, when selecting the None backend
#elif MODEL_TO_USE == 2
    anira::InferenceConfig inferenceConfig = hybridNNConfig;
    HybridNNPrePostProcessor prePostProcessor;
    HybridNNNoneProcessor noneProcessor; // This one is only needed for the round trip test, when selecting the None backend
#elif MODEL_TO_USE == 3
    anira::InferenceConfig inferenceConfig = statefulRNNConfig;
    StatefulRNNPrePostProcessor prePostProcessor;
#endif


    anira::InferenceHandler inferenceHandler;

    // TODO select default inference backend
    anira::InferenceBackend m_inference_backend = anira::ONNX;
    std::binary_semaphore m_host_audio_semaphore;
};