// renderBurstFrames (src/AudioNode.ts): the burst of render quanta a context renders back to
// back, which a handler's Hard block_max must cover. Plain objects stand in for the contexts,
// with the values Chromium and Gecko report.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { renderBurstFrames } from '../src/AudioNode'

test('an OfflineAudioContext (no baseLatency) renders quantum by quantum', () => {
  assert.equal(renderBurstFrames({ sampleRate: 48000 }), 128)
  assert.equal(renderBurstFrames({ sampleRate: 44100, renderQuantumSize: 256 }), 256)
})

test("Chromium's baseLatency is its callback buffer: the burst, in whole quanta", () => {
  // Headless Chromium: 2048 frames at 48 kHz.
  assert.equal(renderBurstFrames({ sampleRate: 48000, baseLatency: 2048 / 48000 }), 2048)
  // A buffer of one quantum (max(frames per buffer, quantum)).
  assert.equal(renderBurstFrames({ sampleRate: 48000, baseLatency: 128 / 48000 }), 128)
  // 441 frames at 44.1 kHz (10 ms) round up to 4 quanta.
  assert.equal(
    renderBurstFrames({ sampleRate: 44100, baseLatency: 0.01, outputLatency: 0.05 }),
    512
  )
})

test("Gecko's baseLatency of 0: its outputLatency, else 2048 frames at 48 kHz", () => {
  assert.equal(
    renderBurstFrames({ sampleRate: 48000, baseLatency: 0, outputLatency: 0.02 }),
    1024
  )
  // Not running yet: neither latency is known.
  assert.equal(
    renderBurstFrames({ sampleRate: 48000, baseLatency: 0, outputLatency: 0 }),
    2048
  )
  assert.equal(renderBurstFrames({ sampleRate: 96000, baseLatency: 0 }), 4096)
})
