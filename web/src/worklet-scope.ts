// Loaded into an audio context's AudioWorkletGlobalScope (createAudioNode, before the module's
// glue): what the scope lacks and anira's C code needs on the audio thread. The scope has no
// `performance` (https://github.com/WebAudio/web-audio-api/issues/2413); without it the glue's
// monotonic clock is missing, std::chrono::steady_clock fails there, and so does every bounded
// wait of the render callback (an offline render's anira_handler_process_wait). The polyfill is
// Date.now() made monotonic (clamped against steps of the wall clock). It imports nothing.
const scope = globalThis as { performance?: { now(): number; timeOrigin: number } }
if (scope.performance === undefined) {
  const timeOrigin = Date.now()
  let last = 0
  scope.performance = {
    now: () => (last = Math.max(last, Date.now() - timeOrigin)),
    timeOrigin,
  }
}

export {}
