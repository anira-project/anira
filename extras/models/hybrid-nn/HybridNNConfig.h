#ifndef ANIRA_HYBRIDNNCONFIG_H
#define ANIRA_HYBRIDNNCONFIG_H

// GuitarLSTM (extras/models/hybrid-nn) built in code, for the benchmark sweeps only: they vary
// the batch count with the host buffer, which the fixed window of hybridnn.model.json cannot,
// and therefore also name a different TensorFlow file per size (GuitarLSTM-<batches>.tflite).
// At 256 the builder equals the file (a test keeps it so); everything else loads the file
// (model_files.h).
//
// The model takes `batches` windows of 150 samples per inference, each window one sample
// further than the last, and returns one sample per window: a hop of `batches` samples with
// 149 samples of context ahead of it, which HybridNNPrePostProcessor builds the windows from.
// The PyTorch exports hold the input as [batch, channel, time]; the TensorFlow exports hold it
// as [batch, time, channel]. The Time axis of the spec is the 150-sample window of one batch,
// not the 405 ring elements one inference consumes: the ring extent and the axis are reconciled
// by the 3.x runtime's ring chunker; this pre-release's bridge carries the hop (window minus
// context) to the 2.x processing spec, which is all the 2.x runtime reads.

#include <anira/anira.hpp>
#include <array>
#include <cstdint>
#include <string>

inline constexpr int64_t k_hybridnn_window = 150;

inline anira::ModelConfig hybridnn_model_config(int64_t batches = 256) {
    const std::string pytorch_dir =
        ANIRA_EXTRAS_MODELS_DIR "/hybrid-nn/GuitarLSTM/pytorch-version/models/model_0/";
    // The TensorFlow exports are fixed per batch count; LiteRT loads the same flatbuffer as the
    // legacy TFLite backend.
    const std::string tensorflow_path =
        std::string(ANIRA_EXTRAS_MODELS_DIR
                    "/hybrid-nn/GuitarLSTM/tensorflow-version/models/model_0/GuitarLSTM-") +
        std::to_string(batches) + ".tflite";
    anira::ModelConfig cfg;
#ifdef USE_LIBTORCH
    cfg.add_model_path(ANIRA_ENGINE_LIBTORCH, pytorch_dir + "GuitarLSTM-dynamic.pt");
#endif
#ifdef USE_ONNXRUNTIME
    cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, pytorch_dir + "GuitarLSTM-libtorch-dynamic.onnx");
#endif
#ifdef USE_TFLITE
    cfg.tensor_layout(cfg.add_model_path(ANIRA_ENGINE_TFLITE, tensorflow_path),
                      "audio_in",
                      std::array{0U, 2U, 1U});
#endif
#ifdef USE_LITERT
    cfg.tensor_layout(cfg.add_model_path(ANIRA_ENGINE_LITERT, tensorflow_path),
                      "audio_in",
                      std::array{0U, 2U, 1U});
#endif
#ifdef USE_EXECUTORCH
    // Exported ahead-of-time from the same PyTorch weights as the LibTorch model.
    cfg.add_model_path(ANIRA_ENGINE_EXECUTORCH, pytorch_dir + "GuitarLSTM-executorch.pte");
#endif
    const int64_t context = k_hybridnn_window - 1;
    cfg.input(anira::TensorSpec("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                  .axis(0, ANIRA_AXIS_ANY, batches)
                  .axis(1, ANIRA_AXIS_CHANNEL, 1)
                  .axis(2, ANIRA_AXIS_TIME, k_hybridnn_window)
                  .window(batches + context, batches + context, context));
    cfg.output(anira::TensorSpec("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                   .axis(0, ANIRA_AXIS_ANY, batches)
                   .axis(1, ANIRA_AXIS_TIME, 1)
                   .window(batches, batches, 0));
    return cfg;
}

#endif  // ANIRA_HYBRIDNNCONFIG_H
