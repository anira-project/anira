#include <aria/scheduler/InferenceManager.h>

InferenceManager::InferenceManager(PrePostProcessor& ppP, InferenceConfig& config) :
    inferenceThreadPool(InferenceThreadPool::getInstance(config)),
    session(inferenceThreadPool->createSession(ppP, config)),
    inferenceConfig(config)
{
}

InferenceManager::~InferenceManager() {
    inferenceThreadPool->releaseSession(session);
}

void InferenceManager::setBackend(InferenceBackend newInferenceBackend) {
    session.currentBackend = newInferenceBackend;
}

InferenceBackend InferenceManager::getBackend() {
    return session.currentBackend;
}

void InferenceManager::prepare(HostAudioConfig newConfig) {
    spec = newConfig;

    session.sendBuffer.initializeWithPositions(1, (size_t) spec.hostSampleRate * 6); // TODO find appropriate size dynamically
    session.receiveBuffer.initializeWithPositions(1, (size_t) spec.hostSampleRate * 6); // TODO find appropriate size dynamically
    inferenceCounter = 0;

    init = true;
    bufferCount = 0;

    size_t result = spec.hostBufferSize % (inferenceConfig.m_batch_size * inferenceConfig.m_model_input_size);
    if (result == 0) {
        initSamples = inferenceConfig.m_max_inference_time + inferenceConfig.m_batch_size * inferenceConfig.m_model_latency;
    } else if (result > 0 && result < spec.hostBufferSize) {
        initSamples = inferenceConfig.m_max_inference_time + spec.hostBufferSize + inferenceConfig.m_batch_size * inferenceConfig.m_model_latency; //TODO not minimum possible
    } else {
        initSamples = inferenceConfig.m_max_inference_time + (inferenceConfig.m_batch_size * inferenceConfig.m_model_input_size) + inferenceConfig.m_batch_size * inferenceConfig.m_model_latency;
    }
}

void InferenceManager::process(float ** inputBuffer, size_t inputSamples) {
    processInput(inputBuffer, inputSamples);
    if (init) {
        bufferCount += inputSamples;
        clearBuffer(inputBuffer, inputSamples);
        if (bufferCount >= initSamples) init = false;
    } else {
        processOutput(inputBuffer, inputSamples);
    }
}

void InferenceManager::processInput(float ** inputBuffer, size_t inputSamples) {
    for (size_t channel = 0; channel < spec.hostChannels; ++channel) {
        for (size_t sample = 0; sample < inputSamples; ++sample) {
            session.sendBuffer.pushSample(0, inputBuffer[channel][sample]);
        }
    }

    inferenceThreadPool->newDataSubmitted(session);
}

void InferenceManager::processOutput(float ** inputBuffer, size_t inputSamples) {
    double timeInSec = static_cast<double>(inputSamples) / spec.hostSampleRate;
    inferenceThreadPool->newDataRequest(session, timeInSec);
    
    while (inferenceCounter > 0) {
        if (session.receiveBuffer.getAvailableSamples(0) >= 2 * (size_t) inputSamples) {
            for (size_t channel = 0; channel < spec.hostChannels; ++channel) {
                for (size_t sample = 0; sample < inputSamples; ++sample) {
                    session.receiveBuffer.popSample(channel);
                }
            }
            inferenceCounter--;
            std::cout << "##### catch up samples" << std::endl;
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
        std::cout << "##### missing samples" << std::endl;
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
    if ((int) initSamples % (int) spec.hostBufferSize == 0) return initSamples;
    else return ((int) ((float) initSamples / (float) spec.hostBufferSize) + 1) * (int) spec.hostBufferSize;
}

InferenceThreadPool& InferenceManager::getInferenceThreadPool() {
    return *inferenceThreadPool;
}

size_t InferenceManager::getNumReceivedSamples() {
    inferenceThreadPool->newDataRequest(session, 0); // TODO: Check if processOutput call is better here
    return session.receiveBuffer.getAvailableSamples(0);
}

bool InferenceManager::isInitializing() const {
    return init;
}

int InferenceManager::getMissingBlocks() {
    return inferenceCounter.load();
}

int InferenceManager::getSessionID() const {
    return session.sessionID;
}
