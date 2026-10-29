// The worklet's copy rule (src/workers/delivery.ts): what a Hard process call's status and
// delivered count mean for the samples the node receives.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { anira_status } from '../src/abi/enums'
import { samplesToCopy } from '../src/workers/delivery'

const QUANTUM = 128

test('ANIRA_OK copies what was delivered', () => {
  assert.equal(samplesToCopy(anira_status.ANIRA_OK, QUANTUM, QUANTUM), QUANTUM)
  assert.equal(samplesToCopy(anira_status.ANIRA_OK, 64, QUANTUM), 64)
})

test('ANIRA_MISSED copies the whole quantum: the miss policy filled it, delivered is 0', () => {
  assert.equal(samplesToCopy(anira_status.ANIRA_MISSED, 0, QUANTUM), QUANTUM)
})

test('a failed status copies nothing', () => {
  assert.equal(
    samplesToCopy(anira_status.ANIRA_ERROR_INVALID_ARGUMENT, QUANTUM, QUANTUM),
    0
  )
  assert.equal(samplesToCopy(anira_status.ANIRA_ERROR_NOT_PREPARED, 0, QUANTUM), 0)
})

test('a delivered count is never more than the quantum and never negative', () => {
  assert.equal(samplesToCopy(anira_status.ANIRA_OK, QUANTUM * 2, QUANTUM), QUANTUM)
  // A size_t at or above 2^31 arrives from wasm32 as a negative number.
  assert.equal(samplesToCopy(anira_status.ANIRA_OK, -1, QUANTUM), QUANTUM)
})
