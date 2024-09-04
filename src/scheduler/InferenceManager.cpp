#include <anira/scheduler/InferenceManager.h>

namespace anira {

InferenceManager::InferenceManager(PrePostProcessor& ppP, InferenceConfig& config, BackendBase& noneProcessor) :
    inferenceThreadPool(InferenceThreadPool::getInstance(config)),
    session(inferenceThreadPool->createSession(ppP, config, noneProcessor)),
    inferenceConfig(config)
{
}

InferenceManager::~InferenceManager() {
    inferenceThreadPool->releaseSession(session, inferenceConfig);
}

void InferenceManager::setBackend(InferenceBackend newInferenceBackend) {
    session.currentBackend = newInferenceBackend;
}

InferenceBackend InferenceManager::getBackend() {
    return session.currentBackend;
}

void InferenceManager::prepare(HostAudioConfig newConfig) {
    spec = newConfig;

    inferenceThreadPool->prepare(session, spec);

    inferenceCounter = 0;

    initSamples = calculateLatency();
    for (size_t i = 0; i < spec.hostChannels; ++i) {
        for (size_t j = 0; j < initSamples; ++j) {
            session.receiveBuffer.pushSample(i, 0.f);
        }
    }
}

void InferenceManager::process(float ** inputBuffer, size_t inputSamples) {
    processInput(inputBuffer, inputSamples);

    inferenceThreadPool->newDataSubmitted(session);
    double timeInSec = static_cast<double>(inputSamples) / spec.hostSampleRate;
    inferenceThreadPool->newDataRequest(session, timeInSec);

    processOutput(inputBuffer, inputSamples);

    session.m_current_sample += inputSamples;
    if (session.m_current_sample >= UINT32_MAX) {
        session.m_current_sample = 0;
    }
}

void InferenceManager::processInput(float ** inputBuffer, size_t inputSamples) {
    for (size_t channel = 0; channel < spec.hostChannels; ++channel) {
        for (size_t sample = 0; sample < inputSamples; ++sample) {
            session.sendBuffer.pushSample(0, inputBuffer[channel][sample]);
        }
    }
}

void InferenceManager::processOutput(float ** inputBuffer, size_t inputSamples) {    
    while (inferenceCounter > 0) {
        if (session.receiveBuffer.getAvailableSamples(0) >= 2 * (size_t) inputSamples) {
            for (size_t channel = 0; channel < spec.hostChannels; ++channel) {
                for (size_t sample = 0; sample < inputSamples; ++sample) {
                    session.receiveBuffer.popSample(channel);
                }
            }
            inferenceCounter--;
#ifndef BELA
            std::cout << "[WARNING] Catch up samples!" << std::endl;
#else
            printf("[WARNING] Catch up samples!");
#endif
        }
        else {
            break;
        }
    }
    if (session.receiveBuffer.getAvailableSamples(0) >= (size_t) inputSamples) {
        for (size_t channel = 0; channel < spec.hostChannels; ++channel) {
            for (size_t sample = 0; sample < inputSamples; ++sample) {
                inputBuffer[channel][sample] = session.receiveBuffer.popSample(channel);
            }
        }
    }
    else {
        clearBuffer(inputBuffer, inputSamples);
        inferenceCounter++;
#ifndef BELA
            std::cout << "[WARNING] Missing samples!" << std::endl;
#else
            printf("[WARNING] Missing samples!\n");
#endif
    }
}

void InferenceManager::clearBuffer(float ** inputBuffer, size_t inputSamples) {
    for (size_t channel = 0; channel < spec.hostChannels; ++channel) {
        for (size_t sample = 0; sample < inputSamples; ++sample) {
            inputBuffer[channel][sample] = 0.f;
        }
    }
}

int InferenceManager::getLatency() const {
    return initSamples;
}

InferenceThreadPool& InferenceManager::getInferenceThreadPool() {
    return *inferenceThreadPool;
}

size_t InferenceManager::getNumReceivedSamples() {
    inferenceThreadPool->newDataRequest(session, 0); // TODO: Check if processOutput call is better here
    return session.receiveBuffer.getAvailableSamples(0);
}

int InferenceManager::getMissingBlocks() {
    return inferenceCounter.load();
}

int InferenceManager::getSessionID() const {
    return session.sessionID;
}

int InferenceManager::calculateLatency() {
    // First calculate some universal values
    int modelOutputSize = inferenceConfig.m_new_model_output_size;
    float hostBufferTime = (float) spec.hostBufferSize * 1000.f / (float) spec.hostSampleRate;
#ifndef USE_SEMAPHORE
    if (inferenceConfig.m_wait_in_process_block != 0.f) {
        std::cout << "[WARNING] Using a wait time in process block of 0 ms. Wait time is not supported without semaphores." << std::endl;
        inferenceConfig.m_wait_in_process_block = 0.f;
    }
#endif
    float waitTime = inferenceConfig.m_wait_in_process_block * hostBufferTime;

    // Then caclulate the different parts of the latency
    int bufferAdaptation = calculateBufferAdaptation(spec.hostBufferSize, modelOutputSize);

    int maxPossibleInferences = maxNumberOfInferences(spec.hostBufferSize, modelOutputSize);
    float totalInferenceTimeAfterWait = (maxPossibleInferences * inferenceConfig.m_max_inference_time) - waitTime;
    int numBuffersForMaxInferences = std::ceil(totalInferenceTimeAfterWait / hostBufferTime);
    int inferenceCausedLatency = numBuffersForMaxInferences * spec.hostBufferSize;

    int modelCausedLatency = inferenceConfig.m_model_latency;

    // Add it all together
    return bufferAdaptation + inferenceCausedLatency + modelCausedLatency;
}


int InferenceManager::calculateBufferAdaptation(int hostBufferSize, int modelOutputSize) {
    int res = 0;
    for (int i = hostBufferSize; i < leatCommonMultiple(hostBufferSize, modelOutputSize) ; i+=hostBufferSize) {
        res = std::max<int>(res, i%modelOutputSize);
    }
    return res;
}

int InferenceManager::maxNumberOfInferences(int hostBufferSize, int modelOutputSize) {
    float samplesInBuffer = hostBufferSize;
    int res = (int) (samplesInBuffer / (float) modelOutputSize);
    res = std::max<int>(res, 1);
    int numberOfInferences = 0;
    for (int i = hostBufferSize; i < leatCommonMultiple(hostBufferSize, modelOutputSize) ; i+=hostBufferSize) {
        numberOfInferences = (int) (samplesInBuffer / (float) modelOutputSize);
        res = std::max<int>(res, numberOfInferences);
        samplesInBuffer += hostBufferSize - numberOfInferences * modelOutputSize;
    }
    return res;
}

int InferenceManager::greatestCommonDivisor(int a, int b) {
    if (b == 0) {
        return a;
    } else {
        return greatestCommonDivisor(b, a % b);
    }
}

int InferenceManager::leatCommonMultiple(int a, int b) {
    return a * b / greatestCommonDivisor(a, b);
}

} // namespace anira