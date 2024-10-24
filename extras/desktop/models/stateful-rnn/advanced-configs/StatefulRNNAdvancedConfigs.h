#ifndef ANIRA_STATEFULRNN_ADVANCED_CONFIGS_H
#define ANIRA_STATEFULRNN_ADVANCED_CONFIGS_H

#include <anira/anira.h>

#include "StatefulRNNConfig_64.h"
#include "StatefulRNNConfig_128.h"
#include "StatefulRNNConfig_256.h"
#include "StatefulRNNConfig_512.h"
#include "StatefulRNNConfig_1024.h"
#include "StatefulRNNConfig_2048.h"
#include "StatefulRNNConfig_4096.h"
#include "StatefulRNNConfig_8192.h"

#ifndef ADVANCED_CONFIGS
#define ADVANCED_CONFIGS

struct InferenceConfigBufferPair {
    anira::InferenceConfig config;
    int buffer_size;
};

typedef std::vector<InferenceConfigBufferPair> AdvancedInferenceConfigs;

#endif // ADVANCED_CONFIGS

static AdvancedInferenceConfigs rnn_advanced_configs = {
    {statefulRNNConfig_64, 64},
    {statefulRNNConfig_128, 128},
    {statefulRNNConfig_256, 256},
    {statefulRNNConfig_512, 512},
    {statefulRNNConfig_1024, 1024},
    {statefulRNNConfig_2048, 2048},
    {statefulRNNConfig_4096, 4096},
    {statefulRNNConfig_8192, 8192}
};

#endif // ANIRA_STATEFULRNN_ADVANCED_CONFIGS_H