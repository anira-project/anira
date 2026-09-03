#ifndef ANIRA_EXTRAS_MODEL_FILES_H
#define ANIRA_EXTRAS_MODEL_FILES_H

// The configuration files of the bundled models: one model file (section 1.5 of the usage
// guide: the model entries with paths relative to the file, the tensor specs, the state) and
// one contract file (the budget and the warm-up; the host geometry is patched in at prepare)
// per model, next to the model directories they name. ANIRA_EXTRAS_MODELS_DIR is the root of
// that tree at run time (extras/CMakeLists.txt; the one compile definition of the tests and
// examples).
//
// Every example and test loads a model the same way:
//
//     anira::ModelConfig model = anira::ModelConfig::from_file(k_cnn_model_json);
//     anira::ContractHandle contract = anira::ContractHandle::from_file(k_cnn_contract_json);
//     anira::InferenceConfig config = anira::v3compat::to_inference_config(
//         model, contract, anira::v3compat::enabled_engines());
//
// (or anira_model_config_from_json_file / anira_contract_from_json from C). The files set no
// instance ceiling, so one processor per engine runs; a config that wants parallel instances
// says so with max_instances.

#define ANIRA_EXTRAS_MODEL_FILE(relative) ANIRA_EXTRAS_MODELS_DIR "/" relative

// The steerable-nafx CNN in three sizes (receptive fields of 13332, 1332 and 132 samples).
inline constexpr const char* k_cnn_model_json = ANIRA_EXTRAS_MODEL_FILE("cnn/cnn.model.json");
inline constexpr const char* k_cnn_contract_json = ANIRA_EXTRAS_MODEL_FILE("cnn/cnn.contract.json");
inline constexpr const char* k_medium_cnn_model_json =
    ANIRA_EXTRAS_MODEL_FILE("cnn/medium_cnn.model.json");
inline constexpr const char* k_medium_cnn_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("cnn/medium_cnn.contract.json");
inline constexpr const char* k_small_cnn_model_json =
    ANIRA_EXTRAS_MODEL_FILE("cnn/small_cnn.model.json");
inline constexpr const char* k_small_cnn_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("cnn/small_cnn.contract.json");

// GuitarLSTM, 256 windows of 150 samples per inference.
inline constexpr const char* k_hybridnn_model_json =
    ANIRA_EXTRAS_MODEL_FILE("hybrid-nn/hybridnn.model.json");
inline constexpr const char* k_hybridnn_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("hybrid-nn/hybridnn.contract.json");

// The stateful LSTM, 2048-sample chunks.
inline constexpr const char* k_rnn_model_json =
    ANIRA_EXTRAS_MODEL_FILE("stateful-rnn/rnn.model.json");
inline constexpr const char* k_rnn_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("stateful-rnn/rnn.contract.json");

// SimpleGainNetwork, mono and stereo: a 512-sample stream plus a static gain scalar.
inline constexpr const char* k_gain_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/gain.model.json");
inline constexpr const char* k_gain_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/gain.contract.json");
inline constexpr const char* k_stereo_gain_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/stereo_gain.model.json");
inline constexpr const char* k_stereo_gain_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/stereo_gain.contract.json");

// RAVE funk drum (IRCAM ACIDS; LibTorch only): the whole model, and its encoder and decoder
// halves as two entry points of the same TorchScript file. The decoder anchors on its audio
// output: the host's block and rate are audio samples, its input is latent frames.
inline constexpr const char* k_rave_funk_drum_model_json =
    ANIRA_EXTRAS_MODEL_FILE("third-party/ircam-acids/rave_funk_drum.model.json");
inline constexpr const char* k_rave_funk_drum_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("third-party/ircam-acids/rave_funk_drum.contract.json");
inline constexpr const char* k_rave_funk_drum_encoder_model_json =
    ANIRA_EXTRAS_MODEL_FILE("third-party/ircam-acids/rave_funk_drum_encoder.model.json");
inline constexpr const char* k_rave_funk_drum_encoder_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("third-party/ircam-acids/rave_funk_drum_encoder.contract.json");
inline constexpr const char* k_rave_funk_drum_decoder_model_json =
    ANIRA_EXTRAS_MODEL_FILE("third-party/ircam-acids/rave_funk_drum_decoder.model.json");
inline constexpr const char* k_rave_funk_drum_decoder_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("third-party/ircam-acids/rave_funk_drum_decoder.contract.json");

#undef ANIRA_EXTRAS_MODEL_FILE

#endif  // ANIRA_EXTRAS_MODEL_FILES_H
