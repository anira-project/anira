# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The default `PrePostProcessor` now handles receptive-field (sliding-window) models without a custom subclass: when an input tensor holds more samples than its `preprocess_input_size`, `pre_process` fills the head of the window with ring-buffer history and only the tail with fresh samples (the buffer allocation already reserved that history). Configure the hop in `preprocess_input_size` and the window in the tensor shape — e.g. steerable-nafx: shape `[1, 1, 15380]`, hop `2048`. Behavior is unchanged when hop == window. The window heuristic divides by the tensor's channel count (a multichannel tensor like a [1, 4, 1] latent frame holds channels x hop samples without being a window). Covered by the new `PrePostProcessorWindow` tests.
- `model_function` now also works with the ExecuTorch backend: a `.pte` exported with several named entry points (e.g. `encode`/`decode`) runs the configured method instead of the hardcoded `forward` (`Module::load_method`/`execute`), in both the C++ API and the JSON config loader. Covered by the new `ExecuTorchModelFunction` tests against the multi-function SimpleGainNetwork fixture from anira-project/example-models — one program, three methods (`forward`/`gain2`/`gain4`), told apart by their output gain.
- ExecuTorch backend (`ANIRA_WITH_EXECUTORCH`, **enabled by default** like the other first-class backends — note it requires CMake ≥ 3.24 on desktop; pass `-DANIRA_WITH_EXECUTORCH=OFF` to build with an older CMake): runs `.pte` programs exported ahead-of-time with `torch.export` — PyTorch's edge/mobile inference stack. New `ExecuTorchProcessor` (`InferenceBackend::EXECUTORCH`) with per-instance `executorch::extension::Module`s, CPU execution delegated to XNNPACK (pinned to a single thread, like the other backends), file- and binary-buffer model loading, and `EXECUTORCH` support in the JSON config loader. Prebuilt static libraries are downloaded from the anira-project/backends release for all desktop platforms plus Android/iOS; desktop wires through ExecuTorch's own CMake package (requires CMake ≥ 3.24), mobile links the single merged archive. Covered by the GuitarLSTM inference tests and a new `minimal-executorch` example. All bundled model configs carry `EXECUTORCH` entries backed by `.pte` exports: SimpleGainNetwork (header and JSON, models from anira-project/example-models), the steerable-nafx CNN variants and the stateful LSTM (exported with mutable state buffers at a fixed 2048-sample chunk). The advanced and cnn-size benchmarks include the `EXECUTORCH` backend (the stateful RNN is excluded there since its fixed-chunk export cannot follow the varying buffer size), and the JUCE and CLAP example plugins offer `EXECUTORCH` in their backend selectors. Desktop CI builds and tests the backend; mobile CI keeps it off until the merged mobile archives are validated on-device. In fully static builds ExecuTorch coexists with static LiteRT on all desktop platforms: the backends release v2.3.0 ships the desktop static LiteRT archives pre-isolated to their `LiteRt*` C API (vendored XNNPACK/cpuinfo/pthreadpool internals localized via partial link on Mach-O/ELF, renamed member-by-member on COFF — `scripts/isolate-static.sh` in anira-project/backends), so ExecuTorch's force-loaded XNNPACK is the only global copy — no symbol collision, no cross-binding. Bring-your-own static LiteRT archives (`ANIRA_LITERT_ROOTDIR`) must be isolated the same way. For the legacy TFLite backend and the mobile merged-lib paths, ExecuTorch remains auto-disabled in fully static builds as before (each engine bundles its own XNNPACK, whose symbols collide in one static image).

### Changed

- The active inference backend now defaults to the first model in the `InferenceConfig` whose backend is available in the build, instead of the `CUSTOM` roundtrip. Sessions used to start on `CUSTOM` until `set_inference_backend()` was called — forgetting that call silently passed audio through (bypass), one of the most common integration mistakes. A custom processor passed to the `InferenceHandler` constructor keeps `CUSTOM` active (running it is why it was passed), config entries for backends the build does not provide are skipped, and when nothing matches the previous `CUSTOM` default remains. `set_inference_backend()` continues to override as before. Migration: only setups that relied on the initial bypass *without* passing a custom processor need to call `set_inference_backend(anira::InferenceBackend::CUSTOM)` explicitly. Covered by the new `InferenceHandlerDefaultBackend` tests.
- Default backends release tag bumped from v2.1.1 to v2.3.0 (adds the ExecuTorch 1.3.1 archives and ships the desktop static LiteRT archives pre-isolated to the `LiteRt*` C API; other engine versions unchanged).
- **Breaking:** `InferenceHandler::reset()` (with the underlying `InferenceManager::reset()` / `Context::reset_session()`) is now wait-free and safe to call from the audio thread; the public `InferenceHandler::reset()` is annotated `ANIRA_REALTIME` (`[[clang::nonblocking]]`) like `process()`/`push_data()`/`pop_data()`. Instead of draining the inference queue (a `nanosleep`-based spin), it bumps a per-session generation counter (`SessionElement::m_generation`, stamped onto each dispatch) so every already-dispatched inference becomes "stale": its result is ignored by the `new_data_request()` generation guard and its structure is reclaimed lazily once the worker publishes completion. This works for stateless and stateful (`session_exclusive_processor`) sessions alike — for stateful sessions the pending-dispatch chain is reconciled by the gate-holder (the resetting thread when the dispatch gate is free, otherwise the worker at its next task boundary), and nothing is ever enqueued from the reset path. Audio output is unchanged (the old blocking `reset()` also discarded in-flight results — it merely waited first), but the former post-condition that no inference thread touches session state after the call is gone. Migration: where that quiescence was relied upon (e.g. before mutating state read by a custom `BackendBase` or the `before_inference()`/`after_inference()` hooks), call `prepare()` — which still drains — or synchronize within your own backend code. The WebAssembly/TS `reset()` wrapper inherits the new wait-free semantics unchanged (and is now safe to call from the audio worklet).
- The internal blocking drain (`Context::drain_inference_queue`) is now annotated `[[clang::blocking]]` under RTSan builds, so any future call reachable from a `[[clang::nonblocking]]` context is reported deterministically. It also drains to a fixpoint (a worker completing a session-exclusive task concurrently with a single-pass drain could previously slip a successor into the drain's window) and completes never-started tasks as silence at their stream positions instead of dropping them silently.
- The stateful dispatch gate (`SessionElement::m_stateful_dispatch_gate`) now carries an epoch alongside the busy bit: releases are epoch-checked, so a laggard inference thread that was preempted across a `prepare()` can no longer release — and thereby corrupt — the rebuilt session's in-flight dispatch. `prepare_session()` additionally bumps the session generation so such a laggard's task is skipped as stale instead of running the model on an orphaned structure.
- The RTSan CI job (`build_sanitizer.yml`) is un-parked: the moodycamel first-enqueue allocation it was parked on was fixed in v2.2.1 (verified clean), and the remaining known stream-logging-from-the-audio-path violations are suppressed via a scoped suppressions file (`.github/rtsan-suppressions.supp`) until logging is moved off the audio thread. All other real-time violations now fail CI, including the reset path (covered by new reset tests running under RTSan).

### Removed

- `InferenceHandler::reset_non_blocking()`, `InferenceManager::reset_non_blocking()` and `Context::reset_session_non_blocking()` (never released; only ever existed under [Unreleased]): superseded by the now wait-free `reset()`.

### Fixed

- `BackendBase::process` (the CUSTOM roundtrip and no-model fallback processor) no longer reads past the end of the output tensor vector when a model has more input than output tensors — e.g. a stateful model taking audio + state + prior and returning audio + state. It iterated the input count while indexing `output[]`, so the first buffer processed on the default processor with such a config corrupted memory (observed as a host crash in the wild the moment a 3-in/2-out model ran before `set_inference_backend()` was called — sessions start on `CUSTOM`). Pairwise-matching tensors now roundtrip as before; output tensors without a matching input are cleared, consistent with the existing behavior for channel/size mismatches. Covered by the new `BackendBase` tests.
- Backend runtime symbols are no longer exported from binaries embedding anira, which crashed hosts that ship their own copy of a backend runtime. Ableton Live 12 bundles an ONNX Runtime dylib; a plugin that exported ORT symbols (the prebuilt static archives carry default visibility, and the ORT C++ header emits weak globals like `Ort::Global<void>::api_` from anira's own TUs) had those weak-coalesced/interposed by the dynamic linker against Live's copy, so `Ort::GetApi()` resolved against a mismatched runtime, returned null for anira's `ORT_API_VERSION`, and the first `Ort::` call segfaulted the host at plugin instantiation. Three layers of fix: (1) `anira_target_link_static_backend` now links the prebuilt backend archives hidden — `-load_hidden` on Mach-O (ld64, Xcode ≥ 14), `--exclude-libs,<archive>` on ELF/Android; PE needs nothing since COFF only exports `dllexport` symbols. The options are `PUBLIC`, so they cover both a shared `libanira` and consumers linking the static anira. (2) anira compiles with `CXX_VISIBILITY_PRESET hidden`/`VISIBILITY_INLINES_HIDDEN` on all platforms, and `ANIRA_API` now expands to `__attribute__((visibility("default")))` on GCC/Clang — ELF/Mach-O follow the same public-API allowlist model that `dllexport` enforces on Windows, and the backend headers' weak globals stay module-private. (3) A runtime guard in `OnnxRuntimeProcessor` verifies `OrtGetApiBase()->GetApi(ORT_API_VERSION)` is non-null before the first ORT call and throws a descriptive `std::runtime_error` (propagating out of the `InferenceHandler` constructor) instead of letting a leaked-symbol configuration crash the host. Consumers embedding anira in a plugin are still advised to add an exported-symbols allowlist for their own dependencies — see the new "Host application ships its own backend runtime" section in the troubleshooting guide.
- `TensorShape::m_backend` is now default-initialized: the universal (backend-agnostic) constructor left it uninitialized, and `InferenceConfig`'s backend-matching lookups read it — an uninitialized-enum load flagged by UBSan.
- Session lifecycle calls are now thread-safe across sessions: `Context::get_instance()`, `create_session()`, `release_session()` and `prepare_session()` serialize their mutation of the shared lifecycle state (session registry, inference thread pool, backend processor pools, singleton pointer) with a static mutex, and the last-session pool teardown decision is made atomically (`fetch_sub` transition) so exactly one releaser tears the pool down. Previously, two `InferenceHandler` lifecycles overlapping on different threads — e.g. a host creating/destroying two plugin instances concurrently or in quick succession — raced the session vector, could both enter the pool teardown, and raced the singleton pointer, corrupting memory (observable as ThreadSanitizer races and intermittent crashes; covered by the new `ConcurrentLifecycleTest`). Realtime paths take no lock and remain wait-free.
- Stateful pending dispatches no longer survive `prepare()`: the pending-dispatch queue is flushed and the dispatch epoch advanced during the session rebuild, so a leftover entry can no longer reference an orphaned structure from before the rebuild.
- An inference dequeued while its session is momentarily uninitialized (during a `prepare()`/release drain) is now completed with zeroed output instead of skipped without a completion signal, which could strand its structure (and, for session-exclusive sessions, wedge the dispatch gate) until the next full reconfiguration.
- A session-exclusive task left awaiting dispatch — its dispatch raced a worker's task boundary so both sides bailed, or the global queue was momentarily full — is now re-kicked by any output poll (`new_data_request()`), not only by the next submission or a non-real-time wait; previously the chain could stall for a full host block (or indefinitely for callers that only poll).
- `Context::drain_inference_queue` no longer silently loses another session's inference when requeueing fails — the task is completed as silence at its stream position and its dispatch chain released.

## [v2.2.1] - 2026-07-04

### Added

- Batched overlapping-window extraction: a `pop_samples_from_buffer(...)` overload that extracts `num_batches` windows in one native call (also exposed to WebAssembly/TS), avoiding per-batch JS↔Wasm boundary crossings for batched/windowed models (e.g. HybridNN/GuitarLSTM).
- `PrePostProcessor::before_inference()` / `after_inference()` hooks, run on the inference thread right before/after the backend runs, letting stateful models (e.g. recurrent hidden-state feedback) splice state between consecutive inferences when combined with `session_exclusive_processor = true`. Wired up end-to-end in Anira Web (`registerPrePostProcessor`/`beforeInference()`/`afterInference()` on the TS side).
- Configurable inference-thread wait strategy: `anira::WaitStrategy { SpinBackoff, Blocking }` on `ContextConfig`. `Blocking` lets idle inference threads sleep on a semaphore instead of spin-polling, eliminating idle CPU usage at effectively no throughput cost; coerced to `SpinBackoff` with a warning on WebAssembly.
- Real-time factor and underrun reporting in the benchmark fixture (per-iteration RTF/`[underrun]` marker, plus an optional summary line with `rtf_mean`/`rtf_max`/underrun count).
- The `advanced` and `cnn-size` benchmarks now also cover the `CUSTOM` (no-inference) backend, and the benchmark CI workflow builds with tests enabled and runs the benchmark gtest suites.
- Configurable log level: `anira::LogLevel { Debug, Info, Warning, Error }` on `ContextConfig`, gating anira's own logging and forwarded to the ONNX Runtime, LiteRT and LibTorch backends (TFLite excepted — no runtime logging control in that prebuilt library). Defaults to `Info` in debug builds, `Error` in release builds.
- Anira version mismatch reporting in `Context::get_instance`: a differing major version between sessions is now reported as an error, a differing minor/patch as a warning.
- `Context::get_num_inference_threads()` / `InferenceHandler::get_num_inference_threads()`: number of inference threads currently active process-wide (also exported to WASM).

### Changed

- `ContextConfig`'s default `num_threads` is now platform-dependent: half the available CPU cores (as before) on native builds, `0` on WebAssembly, since inference threads there are always supplied externally via `AniraWeb.spinUpInferenceWorker()`.

### Fixed

- Stateful in-order dispatch (`session_exclusive_processor = true`) no longer allocates on the audio thread — `SessionElement`'s pending-dispatch queue is now pre-sized and fed through an explicit producer token.
- Web: the audio worklet now polyfills `performance.now()` before the WASM module loads, fixing a `ReferenceError` (observed in Firefox) whenever a blocking-ratio deadline was read from the audio thread.
- The advanced benchmark no longer instantiates ONNX Runtime for the stateful RNN model, whose fixed 2048-sample ONNX export failed warm-up at other buffer sizes.
- Use-after-free when a pooled backend processor outlived the session that created it — `InferenceConfig` is now owned per-processor instead of aliasing the originating session's copy.
- Windows build failure in `Context::get_instance()` caused by `<windows.h>`'s `min`/`max` macros mangling an unrelated `std::min(...)` call; fixed by defining `NOMINMAX`/`WIN32_LEAN_AND_MEAN` before the include.
- Hardened `SessionElement::m_is_non_real_time` to `std::atomic<bool>` and unified the two `new_data_request()` wait paths onto one `wait_for_completion()` helper — previously a default-`blocking_ratio` session in non-real-time mode could free an in-flight buffer while the inference thread was still writing to it. Non-real-time mode also now works on WebAssembly once an inference worker is running.

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
