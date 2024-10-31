#ifndef ANIRA_SMALL_CNN_ADVANCED_CONFIGS_H
#define ANIRA_SMALL_CNN_ADVANCED_CONFIGS_H

#include <anira/anira.h>

#include "Small_CNNConfig_64.h"
#include "Small_CNNConfig_128.h"
#include "Small_CNNConfig_256.h"
#include "Small_CNNConfig_512.h"
#include "Small_CNNConfig_1024.h"
#include "Small_CNNConfig_2048.h"
#include "Small_CNNConfig_4096.h"
#include "Small_CNNConfig_8192.h"

#ifndef ADVANCED_CONFIGS
#define ADVANCED_CONFIGS

struct InferenceConfigBufferPair {
    anira::InferenceConfig config;
    int buffer_size;
};

typedef std::vector<InferenceConfigBufferPair> AdvancedInferenceConfigs;

#endif // ADVANCED_CONFIGS

static AdvancedInferenceConfigs small_cnn_advanced_configs = {
    {small_cnnConfig_64, 64},
    {small_cnnConfig_128, 128},
    {small_cnnConfig_256, 256},
    {small_cnnConfig_512, 512},
    {small_cnnConfig_1024, 1024},
    {small_cnnConfig_2048, 2048},
    {small_cnnConfig_4096, 4096},
    {small_cnnConfig_8192, 8192}
};

#endif // ANIRA_SMALL_CNN_ADVANCED_CONFIGS_H