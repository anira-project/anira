#ifndef ANIRA_CNNCONFIG_H
#define ANIRA_CNNCONFIG_H

// The steerable-nafx CNN (extras/models/cnn) built in code, for the benchmark sweeps only:
// they vary the hop with the host buffer, which the fixed windows of cnn.model.json,
// medium_cnn.model.json and small_cnn.model.json cannot. At the default size the builder
// equals the file (a test keeps it so); everything else loads the files of model_files.h.
//
// One mono stream in, one mono stream out, a receptive field ahead of every hop (13332, 1332
// or 132 samples for the full, three-block and two-block model). The PyTorch exports hold the
// axes as [batch, channel, time]; the TensorFlow exports hold them as [batch, time, channel],
// which the layout on those entries says.

#include <anira/anira.hpp>
#include <array>
#include <cstdint>
#include <string>

/// The three sizes of the model.
enum class CnnSize { Full, Medium, Small };

inline constexpr int64_t cnn_receptive_field(CnnSize size) {
    switch (size) {
        case CnnSize::Medium: return 1332;
        case CnnSize::Small: return 132;
        case CnnSize::Full:
        default: return 13332;
    }
}

/// The model of `size` with every engine of this build: a mono input window of
/// `receptive_field + output_size` samples, of which `output_size` are new per inference, and
/// a mono output of `output_size` samples. At 2048 the configuration of the size's model file.
inline anira::ModelConfig cnn_model_config(int64_t output_size = 2048,
                                           CnnSize size = CnnSize::Full) {
    const std::string dir = ANIRA_EXTRAS_MODELS_DIR "/cnn/steerable-nafx/models/model_0/";
    const std::string stem = size == CnnSize::Full     ? "steerable-nafx"
                             : size == CnnSize::Medium ? "steerable-nafx-3_blocks"
                                                       : "steerable-nafx-2_blocks";
    anira::ModelConfig cfg;
    // The TensorFlow exports hold [batch, time, channel] where the specs say [batch, channel,
    // time].
    const auto channels_last = [&cfg](uint32_t entry) {
        cfg.tensor_layout(entry, "audio_in", std::array{0U, 2U, 1U});
        cfg.tensor_layout(entry, "audio_out", std::array{0U, 2U, 1U});
    };
#ifdef USE_LIBTORCH
    cfg.add_model_path(ANIRA_ENGINE_LIBTORCH, dir + stem + "-dynamic.pt");
#endif
#ifdef USE_ONNXRUNTIME
    cfg.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, dir + stem + "-libtorch-dynamic.onnx");
#endif
#ifdef USE_TFLITE
    channels_last(cfg.add_model_path(ANIRA_ENGINE_TFLITE, dir + stem + "-dynamic.tflite"));
#endif
#ifdef USE_LITERT
    channels_last(cfg.add_model_path(ANIRA_ENGINE_LITERT, dir + stem + "-dynamic.tflite"));
#endif
#ifdef USE_EXECUTORCH
    cfg.add_model_path(ANIRA_ENGINE_EXECUTORCH, dir + stem + "-executorch.pte");
#endif
    const int64_t receptive_field = cnn_receptive_field(size);
    const int64_t window = output_size + receptive_field;
    cfg.input(anira::TensorSpec("audio_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                  .axis(0, ANIRA_AXIS_BATCH, 1)
                  .axis(1, ANIRA_AXIS_CHANNEL, 1)
                  .axis(2, ANIRA_AXIS_TIME, window)
                  .window(window, window, receptive_field));
    cfg.output(anira::TensorSpec("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED)
                   .axis(0, ANIRA_AXIS_BATCH, 1)
                   .axis(1, ANIRA_AXIS_CHANNEL, 1)
                   .axis(2, ANIRA_AXIS_TIME, output_size)
                   .window(output_size, output_size, 0));
    return cfg;
}

#endif  // ANIRA_CNNCONFIG_H
