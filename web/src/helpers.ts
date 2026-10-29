import { anira_log_level } from './abi/enums'
import type { AniraBindings } from './abi/bindings'
import type { AniraLogRecord } from './runtime/hooks'

/** The library's release identity (`anira_version_string`), e.g. "3.0.0-alpha.2". */
export const getAniraVersion = (anira: { abi: AniraBindings }): string =>
  anira.abi.anira_version_string() ?? ''

/**
 * Delivers the queued real-time records (`anira_drain_log`) to the sinks: the C sinks of the
 * contexts and the JavaScript hook of `AniraWeb.setLogHook`. There is no drain thread on the
 * web; pump it from the main thread (`startLogPump`). Returns the number of records delivered.
 */
export const drainAniraLog = (anira: { abi: AniraBindings }): number =>
  anira.abi.anira_drain_log()

/** Pumps `drainAniraLog` on a main-thread interval; returns the function that stops it. */
export const startLogPump = (
  anira: { abi: AniraBindings },
  intervalMs = 20
): (() => void) => {
  const id = setInterval(() => drainAniraLog(anira), intervalMs)
  return () => clearInterval(id)
}

/** The console sink: one line per record, at the console level of its anira level. */
export const consoleLogSink = (record: AniraLogRecord): void => {
  const line = `[${record.group}] ${record.message}`
  switch (record.level) {
    case anira_log_level.ANIRA_LOG_ERROR:
      console.error(line)
      break
    case anira_log_level.ANIRA_LOG_WARNING:
      console.warn(line)
      break
    case anira_log_level.ANIRA_LOG_INFO:
      console.info(line)
      break
    default:
      console.debug(line)
  }
}
