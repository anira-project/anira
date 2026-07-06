import { setupOfflineWorker } from './offlineWorkerHandler'

// Default offline pump worker. Each instance drives exactly one lane of an
// OfflineInferenceHandler; spin up one worker per parallel lane (see
// OfflineInferenceHandler.prepare()). Inference itself is executed by the
// shared inference worker(s) — this worker only feeds chunks and collects
// results, blocking itself (never the main thread) while a job runs.

setupOfflineWorker()
