#pragma once

#include <jack/jack.h>
#include <getopt.h>
#include <anira/anira.h>
#include <semaphore>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/cnn/CNNBypassProcessor.h" // This one is only needed for the round trip test, when selecting the Custom backend
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/models/hybrid-nn/HybridNNBypassProcessor.h" // Only needed for round trip test
#include "../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../extras/models/model-pool/SimpleGainConfig.h"
#include "../../../extras/models/model-pool/SimpleStereoGainConfig.h"

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
    anira::InferenceConfig inference_config = cnn_config;
    CNNPrePostProcessor pp_processor;
    CNNBypassProcessor bypass_processor; // This one is only needed for the round trip test, when selecting the Custom backend
#elif MODEL_TO_USE == 2
    anira::InferenceConfig inference_config = hybridnn_config;
    HybridNNPrePostProcessor pp_processor;
    HybridNNBypassProcessor bypass_processor; // This one is only needed for the round trip test, when selecting the Custom backend
#elif MODEL_TO_USE == 3
    anira::InferenceConfig inference_config = rnn_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 4
    anira::InferenceConfig inference_config = gain_config;
    anira::PrePostProcessor pp_processor;
#elif MODEL_TO_USE == 5
    anira::InferenceConfig inference_config = stereo_gain_config;
    anira::PrePostProcessor pp_processor;
#endif


    anira::InferenceHandler inference_handler;
    anira::ContextConfig anira_context_config;
    // TODO select default inference backend
    anira::InferenceBackend m_inference_backend = anira::ONNX;
    std::binary_semaphore m_host_audio_semaphore;
};