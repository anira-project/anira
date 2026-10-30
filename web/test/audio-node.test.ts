// The registration of an audio context with the module (src/AudioNode.ts), which every node's
// creation awaits: once per context however many nodes are created at once, and again after a
// failure.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { registeredContext } from '../src/AudioNode'

// A context whose addModule resolves on the next task (or rejects, while `failures` remain), and
// a module whose emscriptenRegisterAudioObject hands out consecutive handles; both count calls.
const fakes = (failures = 0) => {
  const calls = { addModule: 0, register: 0 }
  const audioContext = {
    audioWorklet: {
      addModule: async () => {
        calls.addModule++
        await new Promise((resolve) => setTimeout(resolve, 1))
        if (failures > 0) {
          failures--
          throw new Error('addModule failed')
        }
      },
    },
  } as unknown as BaseAudioContext
  const abi = {
    module: { emscriptenRegisterAudioObject: () => ++calls.register },
  } as unknown as Parameters<typeof registeredContext>[0]
  return { calls, audioContext, abi }
}

test('nodes created at once on one context share its one registration', async () => {
  for (const count of [2, 4]) {
    const { calls, audioContext, abi } = fakes()
    const handles = await Promise.all(
      Array.from({ length: count }, () => registeredContext(abi, audioContext))
    )
    assert.deepEqual(handles, Array(count).fill(1), `${count} at once`)
    assert.deepEqual(calls, { addModule: 1, register: 1 }, `${count} at once`)
    assert.equal(await registeredContext(abi, audioContext), 1, 'a later node')
    assert.deepEqual(calls, { addModule: 1, register: 1 }, 'a later node')
  }
})

test('each context is registered on its own', async () => {
  const one = fakes()
  const other = fakes()
  await Promise.all([
    registeredContext(one.abi, one.audioContext),
    registeredContext(other.abi, other.audioContext),
  ])
  assert.deepEqual(one.calls, { addModule: 1, register: 1 })
  assert.deepEqual(other.calls, { addModule: 1, register: 1 })
})

test('a failed registration fails the nodes awaiting it, and the next node tries again', async () => {
  const { calls, audioContext, abi } = fakes(1)
  const failed = await Promise.allSettled([
    registeredContext(abi, audioContext),
    registeredContext(abi, audioContext),
  ])
  assert.deepEqual(
    failed.map((r) => r.status),
    ['rejected', 'rejected']
  )
  assert.equal(await registeredContext(abi, audioContext), 1)
  assert.deepEqual(calls, { addModule: 2, register: 1 })
})
