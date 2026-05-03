import { AniraAudioWorkletBase } from './worklet-base'

class InferenceWorklet extends AniraAudioWorkletBase {}

registerProcessor('inference-processor', InferenceWorklet)
