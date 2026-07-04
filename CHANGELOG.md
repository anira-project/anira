# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Batched overlapping-window extraction: a sixth `pop_samples_from_buffer(RingBuffer&, BufferF&, size_t num_new_samples, size_t num_old_samples, size_t offset, size_t num_batches)` overload that runs the offset extraction `num_batches` times in native code, advancing the output offset by the window size (`num_new_samples + num_old_samples`) each batch, producing a contiguously laid-out batched tensor. Exposed to WebAssembly as `prepostprocessor_pop_samples_from_buffer_batched` and mirrored by the TS `PrePostProcessor.popSamplesFromBuffer(ringBuffer, buffer, numNewSamples, numOldSamples, offset, numBatches)` overload. Motivated by batched/windowed models (e.g. HybridNN/GuitarLSTM): reimplementing the per-batch loop in JavaScript crosses the JS↔Wasm boundary once per batch element, which for large batches dominates the audio-thread budget in the AudioWorklet; the single native call removes that per-element overhead.
- RTSan real-time safety CI checks and testing (not done yet)
- `PrePostProcessor::before_inference()` / `after_inference()` hooks, called on the inference thread immediately before and after the backend runs (default implementations do nothing). Unlike `pre_process()`, which fills input tensors at submission time — stale for cross-inference state once multiple inferences are queued — `after_inference()` runs before the inference is marked done and before the next session-exclusive inference is dispatched, so state captured there is guaranteed visible to the next `before_inference()` call. Combined with `session_exclusive_processor = true`, this enables stateful models (e.g. recurrent hidden-state feedback) to splice state between consecutive inferences safely.
- Anira Web wiring for the `before_inference()` / `after_inference()` hooks. Both are exposed to WebAssembly (`jsprepostprocessor_wasm_before_inference` / `_after_inference`, `prepostprocessor_before_inference` / `_after_inference`) and mirrored on the TS `PrePostProcessor` / `JSPrePostProcessor` wrappers as `beforeInference()` / `afterInference()`. Because these hooks run on the inference thread — the inference *worker* in the WASM build, not the audio worklet where `preProcess` / `postProcess` run — a `JSPrePostProcessor` subclass overriding them must be registered on the worker: `AniraWeb.registerPrePostProcessor(ppProcessor, className)` (with matching `unregisterPrePostProcessor`) forwards the processor to every inference worker, and `setupInferenceWorker(processorClasses, prePostProcessorClasses)` gained a second argument for the subclass map. To keep pre/post-only users zero-cost, the C++ `JSPrePostProcessor::before_inference()` / `after_inference()` skip the JS boundary crossing unless the hooks have been armed (an opt-in flag flipped by `registerPrePostProcessor`, exposed as `JSPrePostProcessor.setInferenceHooks()` / `jsprepostprocessor_set_inference_hooks`). **Note:** `setupInferenceWorker`'s optional dependency-injection `createAnira` parameter moved from the second to the third argument to make room for the pre/post-processor class map.
- Configurable inference-thread wait strategy: `anira::WaitStrategy { SpinBackoff, Blocking }` in `ContextConfig` (second constructor argument, member `m_wait_strategy`, JSON key `context_config.wait_strategy` with values `"spin_backoff"` / `"blocking"`). With `Blocking`, idle inference threads block on the shared inference queue's semaphore (the queue is now `anira::InferenceQueue`, a `moodycamel::BlockingConcurrentQueue` on native builds) and are woken directly by the enqueue, instead of polling with the exponential-backoff spin loop. This eliminates idle CPU usage (~2 syscalls per 100 µs per idle thread with `SpinBackoff`) at the cost of one bounded, non-blocking semaphore signal on the submitting thread per submission; round-trip throughput is identical within measurement noise when inference time dominates. Only one strategy can be in effect per process — the first-created context's; a later `ContextConfig` requesting a different strategy is ignored and reported with a warning. On WebAssembly builds, where inference loops are driven cooperatively by JS Workers, `Blocking` is coerced to `SpinBackoff` with a warning (by both `JsonConfigLoader` and `Context::get_instance`), and the queue remains a plain `ConcurrentQueue`.
- Real-time factor and underrun reporting in the benchmark fixture: every `SingleIteration` line printed by `anira::benchmark::ProcessBlockFixture` now includes the iteration's RTF (measured runtime divided by the host buffer period) and an `[underrun]` marker when it exceeds `1.0`. `ProcessBlockFixture::repetition_step()` gained an optional `total_repetitions` parameter (default `0` keeps the old behavior); when passed, a `Summary/<benchmark>/<model>/<backend>/<buffer size>` line with `rtf_mean`, `rtf_max` and the underrun count over all iterations of all repetitions is printed after the final repetition of each benchmark instance. The bundled benchmarks pass their repetition count.
- The `advanced` and `cnn-size` benchmarks now also benchmark the `CUSTOM` backend (the roundtrip/bypass pipeline without inference). Since the custom backend needs no model file but the fixture and `update_processing_spec()` expect per-backend entries, the benchmark configs append a placeholder model entry and a universal (default) tensor shape for it.
- The benchmark CI workflow (`build_benchmark.yml`, pull requests only) now builds with `ANIRA_WITH_TESTS=ON` (the `desktop-benchmark-tests` preset combination) and runs the benchmark gtest suites via `ctest -R ^Benchmark`; the unit tests remain covered by `build_test.yml`. The shared test action gained an optional `CTEST_ARGS` input.
- Configurable log level: `anira::LogLevel { Debug, Info, Warning, Error }` in `ContextConfig` (third constructor argument, member `m_log_level`, JSON key `context_config.log_level` with values `"debug"` / `"info"` / `"warning"` / `"error"`). One level for the whole inference stack: it gates anira's own `LOG_DEBUG` / `LOG_INFO` / `LOG_WARNING` / `LOG_ERROR` output (new `LOG_DEBUG` and `LOG_WARNING` macros; warnings and errors go to `stderr`; the compile-time `ENABLE_LOGGING` switch still disables everything) and is forwarded to the backends when their processor instances are created — the ONNX Runtime environment severity (`Debug` → `VERBOSE`), the LiteRT environment min-logger severity (`kLiteRtEnvOptionTagMinLoggerSeverity`, which stops LiteRT's per-environment INFO spam; `Debug` → verbose) and the LibTorch/c10 `FLAGS_caffe2_log_level`. The TFLite backend is exempt: the prebuilt TFLite C library exports no runtime logging control. The level is process-global; when ContextConfigs disagree, the lowest (most verbose) requested level wins and the mismatch is reported with a warning. Defaults to `Info` in debug builds and `Error` in release builds (`NDEBUG`). The bundled benchmarks explicitly pass a `ContextConfig` with `LogLevel::Error` so backend logs do not pollute the benchmark output.
- Anira version mismatch reporting in `Context::get_instance` (previously an empty TODO branch): when a new session's `ContextConfig` carries a different `m_anira_version` than the existing context, a differing major version is reported as an error (likely API/ABI incompatibility), a differing minor/patch version as a warning.
- `Context::get_num_inference_threads()`: number of inference threads currently active process-wide, backed by the new `InferenceThread::get_num_active_threads()` counter (native: threads currently executing `run_loop()`; WebAssembly: externally driven threads between `start()` and `stop()`, i.e. the inference workers currently spun up — the counter lives in static, on the web shared, memory, so every WASM instance sees the same value). Also available as `InferenceHandler::get_num_inference_threads()`, and exported to JavaScript as `get_num_inference_threads` on the WASM module.

### Changed

- `ContextConfig`'s default `num_threads` is now platform-dependent via the new `anira::default_num_threads()` helper: half of the available CPU cores (minimum 1) on native builds as before, and `0` on WebAssembly. On WebAssembly a nonzero `num_threads` is additionally coerced to `0` with a warning (by both `Context::get_instance` and `JsonConfigLoader`, mirroring the `WaitStrategy::Blocking` coercion): the context cannot run inference threads there — `InferenceThread` owns no OS thread on wasm — so nonzero values only created inert pool objects that never execute and made the parallel-processor clamp in `Context::create_session` measure phantom capacity. Web inference threads are always supplied externally (`AniraWeb.spinUpInferenceWorker()`, backed by `Context::make_inference_thread()`, which `inference_thread_create_from_context()` now uses instead of constructing directly).

### Fixed

- The stateful in-order dispatch (`session_exclusive_processor = true`) no longer allocates on the audio thread: `SessionElement::enqueue_pending_dispatch()` used the allocating, token-less `moodycamel::ConcurrentQueue::enqueue()` — the first call creates the queue's implicit producer on the heap and later calls can allocate new blocks. On WebAssembly the allocation additionally ran in the audio-worklet WASM instance, where the shared, unsynchronized emmalloc heap must only ever be touched by the main instance (a worklet `malloc` racing a main-thread `malloc` can corrupt it). `m_dispatch_pending` is now pre-sized to `m_num_parallel_processors` — a pending entry is always a distinct `ThreadSafeStruct`, so the depth is bounded — and fed through a dedicated explicit `ProducerToken` created at session construction, making the enqueue a `try_enqueue` that never allocates. The (unreachable) queue-full case is handled like other queue-full drops: the task completes with zeroed output at its stream position.
- Web: the bundled audio worklet now installs a monotonic `performance.now()` polyfill (backed by `Date.now()`, clamped against backward clock steps) before the WASM module is instantiated. `AudioWorkletGlobalScope` does not expose a `performance` object, but Emscripten's `clock_time_get` shim calls `performance.now()`, so any wasm clock read on the audio thread — first hit by the timed completion wait armed when `InferenceConfig`'s `blocking_ratio > 0` — threw `ReferenceError: performance is not defined` (observed in Firefox) instead of processing. Note that the polyfill has millisecond granularity, so blocking-ratio deadlines on the web are only enforced coarsely; prefer larger buffer sizes or moderate ratios there.
- The advanced benchmark no longer instantiates the ONNX Runtime backend for the stateful RNN model: its ONNX export has a fixed input size of 2048, so the warm-up inferences of the eagerly created `OnnxRuntimeProcessor` failed with `Got invalid dimensions for input` warnings at every other buffer size. The ONNX model and tensor-shape entries are now removed from the adapted config (the ONNX benchmark case for this model was already excluded).

- Use-after-free when a pooled backend processor outlives the session that created it. `BackendBase::m_inference_config` (and each backend `Instance`'s aliasing reference) was bound to the originating session's host-owned `InferenceConfig`; releasing that session while a peer session kept the pooled processor alive freed the config and left the processor dereferencing freed memory on the next inference. The config is now an owned value on the processor, so its lifetime matches the processor. This supersedes the `session_exclusive_processor = true` workaround — sharing between equal-config sessions is preserved. Regression covered by `test/scheduler/test_ProcessorPooling.cpp`.
- Windows build failure in `Context::get_instance()`: `HighPriorityThread.h` included `<windows.h>` without `NOMINMAX`, so its `min`/`max` macros mangled the unrelated `std::min(...)` call used to resolve the effective log level, producing `C2589`/`C2059`/`C2737` errors. `NOMINMAX` and `WIN32_LEAN_AND_MEAN` are now defined before the include (only if not already defined, so a consumer's own definition wins).
- Hardened `SessionElement::m_is_non_real_time` (toggled via `InferenceHandler`/`InferenceManager::set_non_realtime()`): it is now `std::atomic<bool>` instead of a plain `bool` that was written from a control thread and read from the audio thread with no synchronization. `Context::new_data_request()`'s two overloads independently re-derived which primitive to block on from `blocking_ratio`; the `wait_until` overload only actually waited when `blocking_ratio > 0.f` and silently fell through without waiting otherwise, so a session with the default `blocking_ratio == 0.f` placed in non-real-time mode could read and free an in-flight `ThreadSafeStruct` while the inference thread was still writing to it. Both overloads now share one `Context::wait_for_completion()` helper that always waits on the same primitive `InferenceThread::do_inference()` actually signals. `set_non_realtime(true)` is now also refused with a warning when no inference threads are configured or active (`Context::has_inference_threads()`), since the resulting blocking waits could never be satisfied and would hang `process()`/`pop_data()` instead of blocking briefly. This replaces a wholesale refusal on WebAssembly: non-real-time mode now works there once at least one inference worker is spun up (`AniraWeb.spinUpInferenceWorker()`) — the waits execute as busy-spins, so run offline processing in a Worker or under an `OfflineAudioContext` rather than on the main thread.

## [v2.2.0] - 2026-06-23

### Added

- **Android support** (`arm64-v8a` + `x86_64`): the library, the prebuilt backends, and the full gtest suite running on a KVM-accelerated emulator, all wired into CI (`build_test_mobile.yml`). The glibc-only `pthread_*inheritsched` / `pthread_setattr_default_np` calls are gated behind `!__ANDROID__` (bionic lacks them) while keeping the portable `SCHED_FIFO` path.
- **iOS support** (device + simulator): the library and prebuilt backends shipped as an xcframework, with the full gtest suite running on the simulator in CI. Per-SDK xcframework slice selection; TFLite is consumed as a `TensorFlowLiteC.framework` via a generated shim so anira's `<tensorflow/lite/...>` includes resolve untouched.
- **LiteRT inference backend** (`anira::InferenceBackend::LITERT`): runs `.tflite` models through Google's native `LiteRt*` C API / CompiledModel runtime. Enabled by default (`ANIRA_WITH_LITERT=ON`) as the modern TensorFlow-Lite-family backend; wired into the examples and benchmarks.
- Data-driven backend downloads: prebuilt backends are fetched at configure time from the [`anira-project/backends`](https://github.com/anira-project/backends) release, pinned by `ANIRA_BACKENDS_VERSION`.
- Live backend integrity check: when GitHub is reachable, anira fetches each asset's published SHA256 and re-downloads any backend whose archive changed upstream or downloaded incompletely, instead of a committed hash lock. `ANIRA_BACKENDS_SKIP_REMOTE_CHECK=ON` skips the remote query for offline/reproducible builds.
- Bring-your-own-backend knobs: `ANIRA_<ENGINE>_ROOTDIR` (prebuilt tree), `ANIRA_<ENGINE>_URL` + `ANIRA_<ENGINE>_SHA256` (custom source), and per-engine `ANIRA_<ENGINE>_LINKAGE=shared|static`.
- clang-tidy conformance across the library, tests and benchmark sources, enforced in CI via the `tanh-lab/ci-actions/clang-tidy-check` action (`clang_tidy.yml`)
- `InstallConsumer` smoke test that builds against the installed package to catch packaging regressions.
- Release pipeline now publishes every tested arch×linkage artifact with backend-consistent naming: Android (static), iOS (xcframework), and Linux/macOS/Windows × `shared`/`static`, including macOS `universal`.

### Changed

- **Breaking:** the `InferenceConfig::Defaults` compile-time constants were renamed from the `m_` prefix to the `k_` prefix to match the constant-naming convention (`m_warm_up` → `k_warm_up`, `m_session_exclusive_processor` → `k_session_exclusive_processor`, `m_blocking_ratio` → `k_blocking_ratio`). The mutable `Defaults::m_num_parallel_processors` is unchanged.
- LiteRT is now the default TensorFlow-Lite-family backend. The legacy TensorFlow Lite backend (the older `TfLite*` C API, `ANIRA_WITH_TFLITE`) is the **same runtime** exposed through a different C API, so the two are now **mutually exclusive** — enable the legacy path with `-DANIRA_WITH_LITERT=OFF -DANIRA_WITH_TFLITE=ON`.
- `nlohmann_json` is now consumed as a release download instead of a git submodule.
- CMake options and their validation were consolidated inline into `CMakeLists.txt` (`AniraOptions.cmake` and the redundant linkage knob were dropped).
- `anira::calculate_min` / `anira::calculate_max` are now `inline` free functions instead of `const auto` lambdas (source-compatible: existing call sites and uses as a callable are unaffected)
- The internal logging helper `isLoggingEnabled()` was renamed to `is_logging_enabled()`
- Migrated the shared clang configs (`.clang-format`/`.clang-tidy`/`.clangd`) from the `tanh-lib` submodule symlinks to [`tanh-tooling`](https://github.com/tanh-lab/tanh-tooling) (pinned `v0.1.4`): committed as real files, kept in sync by the `clang_check.yml` drift check, and the now-unused `tanh-lib` submodule was removed (configs are byte-identical, so lint/format results are unchanged)
- Adopted the default Claude Code config: `.claude/settings.json` now enables the `tanh-tools` plugin from the tanh-tooling marketplace (its format/lint/type-check hooks supersede the previous bespoke `.claude/hooks`)
- CI now covers Windows-`arm64`, Linux-`aarch64` and macOS-`universal` legs (shared + static) in addition to the mobile workflow.

### Fixed

- Potential use-after-free in `Buffer::malloc_channels()` when channel-pointer allocation fails
- Installed `nlohmann_json` config so `find_package(anira)` works against the installed package
- Disabled backends now compile to empty translation units (guarded `.cpp` bodies), and `minimal-onnxruntime` is self-sufficient in static builds
- Value-initialize the `const InferenceConfig` in the install consumer
- clang-tidy violations and a CLAP static MSVC runtime mismatch; preserve the anira-before-`JuceHeader` include order on MSVC; qualified `mem*` calls with `std::` and added `<cstring>`
- Windows shared-DLL copy, the examples target list, and arm64 LibTorch in CI

## [v2.1.0] - 2026-06-14

### Added

- anira Web: the C++ library compiled to WebAssembly via Emscripten, published as the `@anira-project/anira` npm package
  - Emscripten/embind C++ wrappers exposing the core API (InferenceHandler, InferenceConfig, PrePostProcessor, ProcessingSpec, InferenceThread, Buffer, RingBuffer, HostConfig) to JavaScript
  - TypeScript API layer wrapping these bindings (`AniraWeb`, plus typed wrappers for InferenceHandler, InferenceConfig, ModelData, TensorShape, ProcessingSpec, BufferF, RingBuffer, HostConfig and the embind Vector types)
  - New ONNXRuntimeWebBackend (onnxruntime-web), plus `JSBackendBase`/`JSPrePostProcessor` hooks for implementing custom inference backends and pre/post processing in JavaScript/TypeScript
  - Web Audio API integration with an AudioWorklet base class and Web Worker–based off-thread inference
  - WASM build tooling (BuildWasm.cmake, DetectEmscripten.cmake, CMake presets), npm publish workflows, and dedicated Web API documentation (Sphinx + TypeDoc)
- `anira::Semaphore` wrapper for macOS 10.13 support
- macOS universal binary support: anira can now be built as a universal binary (arm64 + x86_64) when no pre-built backends are enabled (e.g. for a custom CoreML backend)
- tanh-lib submodule with clang-format and clang-tidy support
- Unregistering of pre/post processors and a prePostRegistry
- Validation that the maximum inference time must be greater than 0
- Function for freeing the stack pointer
- Sponsor information in the README

### Changed

- Enforce stateful model inference ordering via single-in-flight dispatch instead of spin-wait
- MSVC: support static linking via `ANIRA_STATIC_DEFINE`
- Refactored processPrePost
- Improved GitHub Actions workflows (node24, new workflow versions, no env vars)
- Added Prettier formatting

### Fixed

- Fixed a Windows build bug
- Fixed the documentation build step
- Fixed npm version drift (pinned onnxruntime-web to 1.19.2)

## [v2.0.3] - 2025-11-07

### Added

- JSON configuration loader (JsonConfigLoader) with nlohmann_json dependency, including unit tests
- Option to load LibTorch models as binary data
- `model_function` argument for `model_data` in the JsonConfigLoader
- JSON gain example config and JUCE plugin example with JSON inference config loading
- no_grad options for torch tensors and the inference stage (Fixes #45)

### Fixed

- JUCE plugin example failing to compile when MODEL_TO_USE is set to 6
- No-inference-engine CI build (JsonConfigLoader excluded from the no_inference_engine build)
- Missing preprocessor flag in RaveFunkDrumConfig.h

## [v2.0.2] - 2025-08-03

### Added

- New pop_data methods with wait_until
- Support for TFlite Binary Models

### Changed

- Improved latency calculation to take parallel processing into account
- All operating systems now use std::steady_clock for benchmarking
- Tests for Inference Manager and Session Element now use fixed number of threads 2, which is available on all gh runners

### Fixed

- Ringbuffer initialization now initializes the buffer with zero values
- Fixed the realtime sanitizer build option
- Fixed blocking operation in the InferenceHandler process method

## [v2.0.1] - 2025-07-31

### Changed

- Updated CI to build anira without inference engines to avoid missing preprocessor flags

### Fixed

- Ensure missing preprocessor flags are set for disabled backends
- Add virtual destructor to PrePostProcessor to avoid polymorphic cleanup issues

## [v2.0.0] - 2025-07-28

### Added

- New custom trained RAVE model in examples
- Defaults struct inside InferenceConfig
- Support for offline audio processing
- Option to disable std::cout and std::cerr output
- Possibility to load ONNX models as binary files
- InferenceHandler reset method with comprehensive tests
- Dynamic ring buffer allocation with overflow protection
- Test cases for latency calculation, dynamic ring buffer allocation, and inference struct calculation
- Custom latency preparation functionality
- Jack dependency for Linux JUCE applications
- Comprehensive Doxygen documentation with beautiful Shibuya theme
- Added ProcessingSpecs to the InferenceConfig class for better handling of input and output tensor specifications
- Added changelog documentation page

### Changed

- **Major update**: New shape handling and sizes management
- **Major update**: Support for non-audio input and non-audio output
- **Major update**: Support for multiple streamable and non-streamable tensors
- **Major update**: Input tensor sample rate must not be equal to output tensor sample rate anymore
- Refined latency calculation system:
  - Now supports smaller buffer sizes than host config (with allow_smaller_buffers flag)
  - Moved calculation logic to SessionElement
  - Better handling of models with internal latency
- Renamed HostAudioConfig to HostConfig
- Renamed AudioBuffer to Buffer
- Improved catchup and handling of missing samples
- Different backends can have different shapes while maintaining consistent processingSpecs
- Removed USE_CONTROLLED_BLOCKING preprocessor definition
- Removed external host thread possibility
- Complete documentation overhaul with new theme and structure

### Fixed

- Race condition in InferenceThread where derived class context was destroyed before base class destructor (PR #31)
- Project version compatibility when adding as subdirectory (PR #30)
- Internal latency management issues (PR #32)
- Build bugs and compiler warnings
- GitHub workflow issues
- Install script for nlohmann library
- CMakeLists configuration issues

## [v1.0.3] - 2025-01-24

### Fixed

- Fixed bug where version could not be detected when imported as a submodule

### Added

- Possibility to package as .deb package
- New checks and tests

## [v1.0.2] - 2024-12-06

### Added

- Full support for armv7l platform on Linux
- Benchmarks part of test suite when making pull requests
- Multiple improvements in CMake build chain

### Changed

- Bela examples now in separate repository

### Fixed

- Fixed Windows test suite

## [v1.0.1] - 2024-11-20

### Fixed

- Fixes #11: Issue where the concurrentqueue lib would not be found in the prebuilt binaries or installed lib

## [v1.0.0] - 2024-11-13

### Added

- **Major update with API changes** (see anira usage guide or examples for more information)
- Multichannel support
- Support for input and output of multiple tensors including threadsafe methods to retrieve and pass their state in the anira::PrePostProcessor
- New anira::Context that uses the same thread pool independent of the anira::InferenceConfig the anira::InferenceHandler has been initialized with
- CLAP plugin example
- Enhanced inference job submission

## [v0.1.3] - 2024-09-23

### Changed

- Updated libtorch to 2.4.1

### Fixed

- Fixes issue libomp not bundled with libtorch for macOS arm64
- x86_64 macOS stays with 2.2.2 since new version binaries are not built by pytorch

## [v0.1.2] - 2024-09-14

### Added

- New timestamps via counting inference buffers
- Enhanced thread synchronization and data sharing between threads
- Windows Ninja generator support
- Enhanced Windows dynamic libs
- New default values in InferenceConfig
- Updated documentation

### Changed

- Default threadsafe structs switched to atomic
- Port to new organization

### Fixed

- Solved debug build issues with Windows

## [v0.1.1] - 2024-08-28

### Added

- New Bela support and examples
- New thread synchronisation option with raw atomics

## [v0.1.0] - 2024-05-20

### Changed

- New anira::InferenceConfig layout

## [v0.0.8] - 2024-05-15

### Improved

- Improved latency calculation

## [v0.0.7] - 2024-04-27

### Changed

- Version 0.0.7 release

## [v0.0.6] - 2024-04-17

### Changed

- Version 0.0.6 release

## [v0.0.5] - 2024-04-11

### Changed

- Version 0.0.5 release

## [v0.0.4] - 2024-04-01

### Changed

- Version 0.0.4 release

## [v0.0.3] - 2024-03-30

### Changed

- Updated Windows CI workflow

## [v0.0.2] - 2024-03-27

### Changed

- Version 0.0.2 release

## [v0.0.1] - 2024-03-23

### Added

- Initial release (Version 0.0.1)
