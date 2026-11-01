#ifndef ANIRA_EXTRAS_MODEL_FILES_H
#define ANIRA_EXTRAS_MODEL_FILES_H

// The configuration files of the bundled models: one model file (section 1.5 of the usage
// guide: the model entries with paths relative to the file, the tensor specs, the state) and
// one contract file (the budget and the warm-up; the host blocks are patched in at prepare)
// per model, next to the model directories they name. ANIRA_EXTRAS_MODELS_DIR is the root of
// that tree at run time (extras/CMakeLists.txt; the one compile definition of the tests and
// examples).
//
// Every example and test loads a model the same way:
//
//     anira::ModelConfig model = anira::ModelConfig::from_file(k_cnn_model_json);
//     anira::ContractHandle contract = anira::ContractHandle::from_file(k_cnn_contract_json);
//     anira::Pipeline pipeline;
//     pipeline.add_inference(model);
//     anira::InferenceHandler handler(context, pipeline);
//     handler.prepare(contract.hard_host_blocks(block, block, host_rate));
//
// (or anira_model_config_from_json_file / anira_contract_from_json from C). The files set no
// instance ceiling, so the count is the inference pool's (ANIRA_INSTANCES_AUTO); a config that
// wants fewer parallel instances says so with max_instances.

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

// TypedGainNetwork (example-models): the gain fixture's tensor set at int16, int8 and float16,
// computed in float32 and cast at the boundary, so each runtime needs only a Cast for the
// boundary dtype and every output is exact for small integers and a gain of 2. Rows for every
// engine (the TFLite file's gain is [1, 1, 1] and its outputs keep the signature's keys, as
// SimpleGainNetwork's); the contract is gain.contract.json. The _unplanned files carry one
// ExecuTorch row each: the method at a static 512-sample window with its inputs and outputs
// left out of the memory plan, which a runtime binds in place.
inline constexpr const char* k_typed_gain_int16_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/typed_gain_int16.model.json");
inline constexpr const char* k_typed_gain_int8_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/typed_gain_int8.model.json");
inline constexpr const char* k_typed_gain_float16_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/typed_gain_float16.model.json");
inline constexpr const char* k_typed_gain_int16_unplanned_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/typed_gain_int16_unplanned.model.json");
inline constexpr const char* k_typed_gain_int8_unplanned_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/typed_gain_int8_unplanned.model.json");
inline constexpr const char* k_typed_gain_float16_unplanned_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/typed_gain_float16_unplanned.model.json");
// The ONNX dtype matrix: dtype_matrix_<word>.onnx for each of the ten dtypes anira names
// (data [1, 1, N] -> Cast(float32) -> x 2 -> Cast back -> processed_data), a path prefix the
// word and ".onnx" complete.
inline constexpr const char* k_dtype_matrix_onnx_prefix =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/example-models/TypedGainNetwork/models/dtype_matrix_");
// A fully integer TFLite model (y = 0.5 x on [1, 1, 64]) whose int8 input and output carry a
// scale and a zero point: quantised I/O, which the engines refuse at load.
inline constexpr const char* k_quantized_gain_int8_tflite = ANIRA_EXTRAS_MODEL_FILE(
    "model-pool/example-models/TypedGainNetwork/models/quantized_gain_network_int8.tflite");

// StatefulAccumulatorNetwork, stereo: a 64-sample stream with a declared State pair around it
// ([1, 2, 2]: the running sum and a block counter per channel; the state stands first in the
// inputs and last in the outputs, and anira feeds it back). Rows for every engine: the TFLite
// export's signature keeps the declared mapping under the keys args_0 / args_0_1 / output_0 /
// output_1, which the tflite row reaches by position (the signature runner lists them in key
// order) and the litert row by name (LiteRT lists the outputs in the file's order, [state_out,
// processed_data], so a tensors record names them).
inline constexpr const char* k_stateful_accumulator_model_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/stateful_accumulator.model.json");
inline constexpr const char* k_stateful_accumulator_contract_json =
    ANIRA_EXTRAS_MODEL_FILE("model-pool/stateful_accumulator.contract.json");

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
