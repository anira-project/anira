#ifndef ANIRA_MEDIUM_CNN_ADVANCED_CONFIGS_H
#define ANIRA_MEDIUM_CNN_ADVANCED_CONFIGS_H

#include <anira/anira.h>

#include "Medium_CNNConfig_64.h"
#include "Medium_CNNConfig_128.h"
#include "Medium_CNNConfig_256.h"
#include "Medium_CNNConfig_512.h"
#include "Medium_CNNConfig_1024.h"
#include "Medium_CNNConfig_2048.h"
#include "Medium_CNNConfig_4096.h"
#include "Medium_CNNConfig_8192.h"

#ifndef ADVANCED_CONFIGS
#define ADVANCED_CONFIGS

struct InferenceConfigBufferPair {
    anira::InferenceConfig config;
    int buffer_size;
};

typedef std::vector<InferenceConfigBufferPair> AdvancedInferenceConfigs;

#endif // ADVANCED_CONFIGS

static AdvancedInferenceConfigs medium_cnn_advanced_configs = {
    {medium_cnnConfig_64, 64},
    {medium_cnnConfig_128, 128},
    {medium_cnnConfig_256, 256},
    {medium_cnnConfig_512, 512},
    {medium_cnnConfig_1024, 1024},
    {medium_cnnConfig_2048, 2048},
    {medium_cnnConfig_4096, 4096},
    {medium_cnnConfig_8192, 8192}
};

#endif // ANIRA_MEDIUM_CNN_ADVANCED_CONFIGS_H