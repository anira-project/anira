#ifndef ANIRA_STATEFULRNNCONFIG_H
#define ANIRA_STATEFULRNNCONFIG_H

// The stateful LSTM (extras/models/stateful-rnn) built in code, for the benchmark sweeps only:
// they vary the chunk with the host buffer, which the fixed window of rnn.model.json cannot. At
// 2048 the builder equals the file (a test keeps it so); everything else loads the file
// (model_files.h).
//
// One mono stream in and out in chunks of `chunk` samples, with hidden state carried across
// inferences, which is why the config is stateful (one processor, inferences in submission
// order). The PyTorch and ONNX exports hold the axes as [time, batch, channel]; the TensorFlow
// exports hold them as [batch, time, channel], which the layout on those entries says. The
// ExecuTorch export is fixed at 2048 (the stateful graph cannot be exported with a dynamic
// sequence axis): a benchmark that varies the chunk leaves that engine out of its candidates.

#include <anira/anira.hpp>
#include <array>
#include <cstdint>
#include <string>

/// The batch-first layout of a TensorFlow export of the LSTM: the file holds
/// [batch, time, channel] where the specs say [time, batch, channel].
inline anira::ModelConfig& rnn_batch_first(anira::ModelConfig& cfg, uint32_t entry) {
    cfg.tensor_layout(entry, "audio_in", std::array{1U, 0U, 2U});
    cfg.tensor_layout(entry, "audio_out", std::array{1U, 0U, 2U});
    return cfg;
}

inline anira::ModelConfig rnn_model_config(int64_t chunk = 2048) {
    const std::string dir = ANIRA_EXTRAS_MODELS_DIR "/stateful-rnn/stateful-lstm/models/model_0/";
    anira::ModelConfig cfg;
#ifdef USE_LIBTORCH
    cfg.add_model_path(ANIRA_ENGINE_LIBTORCH, dir + "stateful-lstm-dynamic.pt");
#endif
#ifdef USE_ONNXRUNTIME
    cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, dir + "stateful-lstm-libtorch.onnx");
#endif
#ifdef USE_TFLITE
    rnn_batch_first(cfg,
                    cfg.add_model_path(ANIRA_ENGINE_TFLITE, dir + "stateful-lstm-dynamic.tflite"));
#endif
#ifdef USE_LITERT
    rnn_batch_first(cfg,
                    cfg.add_model_path(ANIRA_ENGINE_LITERT, dir + "stateful-lstm-dynamic.tflite"));
#endif
#ifdef USE_EXECUTORCH
    cfg.add_model_path(ANIRA_ENGINE_EXECUTORCH, dir + "stateful-lstm-executorch.pte");
#endif
    cfg.input(anira::TensorSpec("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                  .axis(0, ANIRA_AXIS_TIME, chunk)
                  .axis(1, ANIRA_AXIS_BATCH, 1)
                  .axis(2, ANIRA_AXIS_CHANNEL, 1)
                  .window(chunk, chunk, 0));
    cfg.output(anira::TensorSpec("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                   .axis(0, ANIRA_AXIS_TIME, chunk)
                   .axis(1, ANIRA_AXIS_BATCH, 1)
                   .axis(2, ANIRA_AXIS_CHANNEL, 1)
                   .window(chunk, chunk, 0));
    cfg.state(ANIRA_MODEL_STATEFUL);
    return cfg;
}

#endif  // ANIRA_STATEFULRNNCONFIG_H
