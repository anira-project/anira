#ifndef NN_INFERENCE_TEMPLATE_CONFIGS_H
#define NN_INFERENCE_TEMPLATE_CONFIGS_H

#include <anira/InferenceConfig.h>

#if MODEL_TO_USE == 1

#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 256
#endif

#define BATCH_SIZE 128
#define MODEL_INPUT_SIZE_BACKEND 150
#define MODEL_OUTPUT_SIZE_BACKEND 1

static anira::InferenceConfig config(
#ifdef USE_LIBTORCH
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/model_0-streaming.pt"),
        {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND},
        {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND},
#endif
#ifdef USE_ONNXRUNTIME
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/model_0-tflite-streaming.onnx"),
        {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1},
        {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND},
#endif
#ifdef USE_TFLITE
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/model_0-streaming.tflite"),
        {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1},
        {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND},
#endif
        BATCH_SIZE,
        1,
        MODEL_INPUT_SIZE_BACKEND,
        MODEL_OUTPUT_SIZE_BACKEND,
        MAX_INFERENCE_TIME,
        0,
        true,
        1
);

#elif MODEL_TO_USE == 2

#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 15380
#endif

#define BATCH_SIZE 1
#define MODEL_INPUT_SIZE_BACKEND 15380 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_OUTPUT_SIZE_BACKEND 2048

static InferenceConfig config(
#ifdef USE_LIBTORCH
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-2048.pt"),
        {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND},
        {BATCH_SIZE, 1, MODEL_OUTPUT_SIZE_BACKEND},
#endif
#ifdef USE_ONNXRUNTIME
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-2048.onnx"),
        {BATCH_SIZE, 1, MODEL_INPUT_SIZE_BACKEND},
        {BATCH_SIZE, 1, MODEL_OUTPUT_SIZE_BACKEND},
#endif
#ifdef USE_TFLITE
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-2048.tflite"),
        {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1},
        {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND, 1},
#endif
        BATCH_SIZE,
        2048,
        MODEL_INPUT_SIZE_BACKEND,
        MODEL_OUTPUT_SIZE_BACKEND,
        0,
        MAX_INFERENCE_TIME
);

#elif MODEL_TO_USE == 3

#if WIN32
#define MAX_INFERENCE_TIME 16384
#else
#define MAX_INFERENCE_TIME 15380
#endif

#define BATCH_SIZE 1
#define MODEL_INPUT_SIZE_BACKEND 2048 // Same as MODEL_INPUT_SIZE, but for streamable models
#define MODEL_OUTPUT_SIZE_BACKEND 2048

static InferenceConfig config(
#ifdef USE_LIBTORCH
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm.pt"),
        {MODEL_INPUT_SIZE_BACKEND, BATCH_SIZE, 1},
        {MODEL_OUTPUT_SIZE_BACKEND, BATCH_SIZE, 1},
#endif
#ifdef USE_ONNXRUNTIME
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-libtorch.onnx"),
        {MODEL_INPUT_SIZE_BACKEND, BATCH_SIZE, 1},
        {MODEL_OUTPUT_SIZE_BACKEND, BATCH_SIZE, 1},
#endif
#ifdef USE_TFLITE
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/stateful-lstm.tflite"),
        {BATCH_SIZE, MODEL_INPUT_SIZE_BACKEND, 1},
        {BATCH_SIZE, MODEL_OUTPUT_SIZE_BACKEND, 1},
#endif
        BATCH_SIZE,
        2048,
        MODEL_INPUT_SIZE_BACKEND,
        MODEL_OUTPUT_SIZE_BACKEND,
        MAX_INFERENCE_TIME,
        0,
        false,
        1
);
#endif

#endif //NN_INFERENCE_TEMPLATE_CONFIGS_H
