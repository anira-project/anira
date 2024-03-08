#ifndef HYBRID_NN_ADVANCED_CONFIGS_H
#define HYBRID_NN_ADVANCED_CONFIGS_H

#include <anira/anira.h>

#include "HybridNNConfig_64.h"
#include "HybridNNConfig_128.h"
#include "HybridNNConfig_256.h"
#include "HybridNNConfig_512.h"
#include "HybridNNConfig_1024.h"
#include "HybridNNConfig_2048.h"
#include "HybridNNConfig_4096.h"
#include "HybridNNConfig_8192.h"

#ifndef ADVANCED_CONFIGS
#define ADVANCED_CONFIGS

struct InferenceConfigBufferPair {
    anira::InferenceConfig config;
    int bufferSize;
};

typedef std::vector<InferenceConfigBufferPair> AdvancedInferenceConfigs;

#endif // ADVANCED_CONFIGS

static AdvancedInferenceConfigs hybridNNAdvancedConfigs = {
    {hybridNNConfig_64, 64},
    {hybridNNConfig_128, 128},
    {hybridNNConfig_256, 256},
    {hybridNNConfig_512, 512},
    {hybridNNConfig_1024, 1024},
    {hybridNNConfig_2048, 2048},
    {hybridNNConfig_4096, 4096},
    {hybridNNConfig_8192, 8192}
};

#endif // HYBRID_NN_ADVANCED_CONFIGS_H