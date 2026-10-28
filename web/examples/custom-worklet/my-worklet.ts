// The audio worklet of the example: anira's worklet base with the stage classes it can
// construct. Bundled into one self-contained script (vite.config.ts); `registerProcessor`
// names the processor that configureAudioWorklet's `nodeName` asks for.
import { AniraAudioWorkletBase } from '@anira-project/anira/workers/worklet-base'
import { InputGainStage } from './gain-stage'

class GainWorklet extends AniraAudioWorkletBase {
  protected override stageClasses = { InputGainStage }
}

registerProcessor('gain-stage-processor', GainWorklet)
