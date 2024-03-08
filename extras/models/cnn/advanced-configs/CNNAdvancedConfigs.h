#ifndef CNN_ADVANCED_CONFIGS_H
#define CNN_ADVANCED_CONFIGS_H

#include <anira/anira.h>

#include "CNNConfig_64.h"
#include "CNNConfig_128.h"
#include "CNNConfig_256.h"
#include "CNNConfig_512.h"
#include "CNNConfig_1024.h"
#include "CNNConfig_2048.h"
#include "CNNConfig_4096.h"
#include "CNNConfig_8192.h"

#ifndef ADVANCED_CONFIGS
#define ADVANCED_CONFIGS

struct InferenceConfigBufferPair {
    anira::InferenceConfig config;
    int bufferSize;
};

typedef std::vector<InferenceConfigBufferPair> AdvancedInferenceConfigs;

#endif // ADVANCED_CONFIGS

static AdvancedInferenceConfigs cnnAdvancedConfigs = {
    {cnnConfig_64, 64},
    {cnnConfig_128, 128},
    {cnnConfig_256, 256},
    {cnnConfig_512, 512},
    {cnnConfig_1024, 1024},
    {cnnConfig_2048, 2048},
    {cnnConfig_4096, 4096},
    {cnnConfig_8192, 8192}
};

#endif // CNN_ADVANCED_CONFIGS_H