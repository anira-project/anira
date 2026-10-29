import { anira_status } from '../abi/enums'

/**
 * How many samples of the output block a Hard process call filled, which the worklet copies to
 * the node. `ANIRA_MISSED` is a success: the ABI reports `delivered = 0`, yet the miss policy
 * (BYPASS, HOLD_LAST, a CALLBACK) has filled the whole block, so the whole quantum is copied. A
 * failed status copies nothing (the quantum is silence); any other success copies what was
 * delivered, at most the quantum.
 */
export const samplesToCopy = (
  status: number,
  delivered: number,
  quantum: number
): number => {
  const s = status | 0
  if (s === anira_status.ANIRA_MISSED) return quantum
  if (s < 0) return 0
  return Math.min(delivered >>> 0, quantum)
}
