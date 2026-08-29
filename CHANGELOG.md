# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **One-sided streaming**: models whose streamable tensors sit on one side only are now first-class. A *generator* (no streamable input, e.g. control parameters in / audio out) is driven by output demand — `process()`/`pop_data()` submit one inference per `postprocess_output_size` requested samples of the reference output and `push_data()` only stores parameter values — and an *analyser* (no streamable output) exposes its results as latest-completed values through `PrePostProcessor::get_output()`. The reference stream of a `HostConfig` may now be an input **or an output**: `m_tensor_index` defaults to the new `HostConfig::k_first_streamable` (first streamable input, else first streamable output) and the new `m_tensor_is_input` selects the direction of an explicit index; `HostConfig::resolve_reference()`/`get_reference_size()` expose the resolution, and naming a non-streamable or out-of-range tensor makes `prepare()` throw `std::invalid_argument` instead of hanging or falling back silently. Mirrored in the WebAssembly/TS wrapper (`tensorIsInput`, `HostConfig.firstStreamable()`). Covered by the new `OneSidedStreaming`, `HostConfigReference` and `InferenceManagerOneSided` tests.
- `build_install` CI workflow: installs anira into a fresh prefix (shared and static, Linux/macOS/Windows, ONNX Runtime on) and builds and runs `test/install` — an external project using `find_package(anira)` — against it, so a header, exported target or dependency (tanh core, backends) missing from the install tree fails CI instead of a downstream project.
- `Context` is now immortal and the inference thread pool exists exactly while sessions exist (#104): built when the first session registers, stopped and joined in the same critical section that unregisters the last one — a plugin host may unload the library the moment its last `InferenceHandler` is destroyed. `Context::shutdown()` backs this up for hosts that unload with a live instance (called automatically from a library-unload hook on ELF/Mach-O; on Windows call it from the plugin's module-exit entry point, e.g. CLAP `deinit`); `Context::release_core_if_idle()`/`has_core()` reclaim the core at unload. Covered by the new `ContextLifecycle` tests and a host-shaped `LibraryUnload` test (`test/unload`) that `dlopen`s a plugin-shaped module, unloads it and asserts it was really unmapped (skipped on macOS when dyld refuses to unload a module with statically linked backends).
- anira is compiled with `-fno-gnu-unique` under GCC, since glibc never unloads an object that defines an `STB_GNU_UNIQUE` symbol (`dlclose()` silently kept `libanira` mapped). `InferenceThread`'s active-thread counter and `InferenceConfig::Defaults::m_num_parallel_processors` moved out of line for the same reason.
- Dead-code stripping: anira's objects are compiled with one section per function and variable (`-ffunction-sections -fdata-sections`, `/Gy` on MSVC) and the shared `libanira` links with `--gc-sections` (ELF), `-dead_strip` (macOS) or `/OPT:REF` (MSVC), so everything the export allowlist does not reach and nothing references is dropped. A static anira has no link step; its consumer can pass the same linker flag. A static anira also localizes the bundled `libtanh_core.a` on its consumer's link (`--exclude-libs`, ELF), and `anira_exports` treats `thl::` like a backend-runtime symbol. The whole compile-side symbol policy (hidden visibility, `ANIRA_BUILDING`/`ANIRA_STATIC`, `-fno-gnu-unique`, sections) now lives in one CMake function, `anira_apply_symbol_policy()`, applied to `anira` and to the ExecuTorch object library alike.
- The CLAP example plugin builds with hidden visibility, `-fno-gnu-unique` under GCC, dead-code stripping and an export list that pins its export table to `clap_entry` (`cmake/clap.map` on ELF, `cmake/clap.exports` on macOS) — the plugin-side setup `docs/sphinx/troubleshooting.rst` recommends — instead of the compiler defaults.
- `anira_exports` CTest (`cmake/CheckExports.cmake`, registered with the unload tests): fails when the shared `libanira` exports any symbol outside namespace `anira`, when the plugin-shaped unload module exports a backend-runtime symbol (`Ort*`, `torch::`/`c10::`, `executorch::`, `xnn_*`, `TfLite*`/`LiteRt*`, …), or when either defines an `STB_GNU_UNIQUE` symbol — via `nm` on Linux/macOS and `dumpbin /exports` on Windows.
- The default `PrePostProcessor` handles receptive-field (sliding-window) models: when an input tensor holds more samples than its `preprocess_input_size`, the window is filled with ring-buffer history plus the fresh hop (e.g. steerable-nafx: shape `[1, 1, 15380]`, hop `2048`). Covered by the new `PrePostProcessorWindow` tests.
- `model_function` now also works with the ExecuTorch backend: a `.pte` with several named entry points runs the configured method instead of `forward`. Covered by the new `ExecuTorchModelFunction` tests.
- **ExecuTorch backend** (`ANIRA_WITH_EXECUTORCH`, enabled by default; needs CMake ≥ 3.24 on desktop): runs `.pte` programs exported with `torch.export` through the new `ExecuTorchProcessor` (`InferenceBackend::EXECUTORCH`, XNNPACK-delegated, file and buffer loading, JSON config support), with prebuilt static libraries from anira-project/backends for desktop and Android/iOS, `EXECUTORCH` entries in the bundled model configs, benchmarks and examples, and a `minimal-executorch` example. Mobile CI keeps it off until validated on-device; in fully static desktop builds it coexists with static LiteRT (pre-isolated archives in backends v2.3.0), while with the legacy TFLite backend and on mobile it stays auto-disabled in fully static builds.

### Changed

- **Breaking:** backend linkage now strictly follows `BUILD_SHARED_LIBS` — a shared anira links shared backends, a static anira links static backends — and the per-engine override `ANIRA_<ENGINE>_LINKAGE` is gone. These are the only two shapes in which every engine exists exactly once per process *and* a consumer can still reach it. An engine that does not ship the required linkage is disabled with a warning instead of being absorbed in the other linkage: LibTorch (shared-only) in static builds, as before, and now also ExecuTorch (static-only) in shared builds, which previously linked its static archives into `libanira`. iOS and Emscripten, where the backends ship static archives only, now refuse `BUILD_SHARED_LIBS=ON` (the `wasm-*` presets set it `OFF`) instead of silently forcing static backends. The rule is asserted at compile time by the new `BackendLinkage` test and, for the CI matrix, on the configure output. Migration: build static (`-DBUILD_SHARED_LIBS=OFF`) to use ExecuTorch; drop any `-DANIRA_<ENGINE>_LINKAGE=` argument.
- The default reference tensor of `HostConfig` is now the first streamable tensor instead of input 0. For every config whose input 0 is streamable this is identical; configs whose input 0 was non-streamable previously divided by zero in `prepare()`.
- `InferenceHandler::prepare(config, custom_latency, tensor_index)` throws `std::invalid_argument` for an out-of-range tensor index instead of reading out of bounds; the latency vector always reports 0 for non-streamable outputs, whatever custom latency was passed.
- `sizeof(anira::HostConfig)` changed (new `m_tensor_is_input` member): source-compatible, but not ABI-compatible with binaries built against older headers.
- The test suite reads its WAV fixtures through tanh-lib's new header-only `thl::core::read_wav` (Apache-2.0 core component) instead of the in-tree `test/WavReader.h` (#91). The old reader only understood 32-bit float mono files; the tanh decoder handles PCM 8/16/24/32-bit, float 32/64-bit, `WAVE_FORMAT_EXTENSIBLE` and multichannel files, and reports a reason on failure instead of printing to stdout. tanh-lib is bumped to v0.0.4 (the release that adds it).
- `Context::get_instance()` takes no arguments and returns a `Context&`; the `ContextConfig` now travels with the session as a fourth `Context::create_session()` parameter, applied by the first session and reconciled by later ones in the critical section that registers it. The old two-step API stays deprecated for one minor release; `InferenceHandler` users are unaffected. `Context::get_sessions()` returns a copy.
- On GCC/Linux, plugins embedding anira statically no longer share one context by accident through `STB_GNU_UNIQUE` binding; every binary has its own, as on macOS and Windows.
- The active backend now defaults to the first model in the `InferenceConfig` whose backend is available, instead of the silent `CUSTOM` bypass. Migration: setups that relied on the initial bypass without a custom processor must call `set_inference_backend(anira::InferenceBackend::CUSTOM)`.
- Default backends release tag bumped from v2.1.1 to v2.3.0 (ExecuTorch 1.3.1 archives, pre-isolated desktop static LiteRT archives).
- **Breaking:** `InferenceHandler::reset()` (with `InferenceManager::reset()` / `Context::reset_session()`) is now wait-free and safe on the audio thread: a per-session generation bump makes in-flight inferences stale instead of draining the queue. Output is unchanged, but the call no longer guarantees that no inference thread touches session state afterwards. Migration: where that was relied upon (e.g. before mutating state read by a custom `BackendBase` or the `before_inference()`/`after_inference()` hooks), call `prepare()` — which still drains — or synchronize in your backend.
- `Context::drain_inference_queue` is annotated `[[clang::blocking]]` under RTSan, drains to a fixpoint and completes never-started tasks as silence instead of dropping them.
- The stateful dispatch gate carries an epoch, so an inference thread preempted across a `prepare()` can no longer release — and corrupt — the rebuilt session's in-flight dispatch.
- The RTSan CI job (`build_sanitizer.yml`) is un-parked; the remaining logging-from-the-audio-path violations are suppressed via `.github/rtsan-suppressions.supp`, everything else fails CI.
- The export-decoration header is now `anira/system/AniraExports.h` — it decorates every platform, not only Windows as when it was named; `AniraWinExports.h` remains as a deprecated forwarding header. Its macros are platform-uniform and renamed: `ANIRA_BUILDING` (defined only while anira itself is compiled) replaces `ANIRA_EXPORTS`, `ANIRA_STATIC` replaces `ANIRA_STATIC_DEFINE`; the old spelling is still honoured. Only relevant when compiling anira with a build system other than its own CMake.
- `anira::Buffer<T>`, `anira::RingBuffer` and `anira::MemoryBlock<T>` are now aliases over [tanh-lib](https://github.com/tanh-lab/tanh-lib) v0.0.3's Apache-2.0 `thl::core` containers instead of in-tree implementations (tanh-lib is fetched at configure time, core component only — no AGPL code enters the build). New: `anira::RingBufferT<T>` exposes the now-templated ring buffer for non-float element types (e.g. integer token streams). **Breaking** (ring buffer): `anira::RingBuffer` no longer derives from `Buffer<float>`, so the inherited pointer/sample accessors (`get_read_pointer`, `get_sample`, `data`, `swap_data`, …) are gone — use `push`/`pop`/`get_future_sample`/`get_past_sample`; `get_available_past_samples()` now returns the number of consumed samples retained as history (0 right after `initialize_with_positions`/`clear_with_positions`) instead of the free slots behind the read position; `get_future_sample`/`get_past_sample` no longer range-check the offset against the available/consumed counts (they wrap modulo the capacity and return the slot's value, 0 after a clear) instead of logging and returning 0; `Buffer::resize()` zeroes the buffer instead of preserving the old bytes. Behavioral notes: the containers never log — a full-channel push still overwrites the oldest sample and popping an empty channel still yields 0, allocation failure now throws `std::bad_alloc` instead of leaving a buffer that claims memory it does not hold, and `swap_data()` with mismatched dimensions asserts in debug builds / is a no-op in release — and `Buffer` carries an additional optional `sample_rate` member (ABI change). The ring buffer wraps its indices without integer division (per-sample pushes on armv7l no longer pay a software divide). `tanh::Core` links no platform library (no libsystemd on Linux, plain sinks on WASM), so anira's system dependencies are unchanged. With `ANIRA_WITH_INSTALL=ON` the tanh core component is installed into the same prefix and `aniraConfig.cmake` resolves it via `find_dependency(tanh COMPONENTS Core)`; a consumer's `find_package(anira)` needs no extra setup (on Windows, `tanh_core.dll` is installed to `bin/` next to anira's `lib/`, so both directories belong on `PATH` at runtime).

### Removed

- `Context::release_instance()` and `Context::release_thread_pool()`: the context is never destroyed and the pool's lifetime follows the session registry; `Context::shutdown()` is the explicit teardown.
- `InferenceHandler::reset_non_blocking()`, `InferenceManager::reset_non_blocking()` and `Context::reset_session_non_blocking()` (never released): superseded by the now wait-free `reset()`.

### Fixed

- **One-sided streaming, redone** (redo of the reverted #101, see #110): `SessionElement::prepare()` no longer hangs for generator-style configs (the smaller-buffer countdown divided by the non-streamable reference input's size 0, giving an `inf` ratio) and no longer crashes for analyser-style configs (`sync_latencies()` indexed an empty latency vector); both latency passes now build vectors index-aligned with the output tensor list by construction (this also fixes #98: with `allow_smaller_buffers` and non-streamable output tensors the adjusted latency vector was read out of bounds -- a crash when every output is non-streamable, garbage latencies for a mixed output list), `calculate_num_structs()`/`max_num_inferences()` count in reference-stream samples over the driving side, and `Context::new_data_submitted()` no longer drains the whole structure pool on every call for sessions without streamable inputs.
- Push-only pipelines no longer stall after `m_num_structs` chunks (#99): `push_data()` now collects completed inferences before submitting, as long as the receive buffers have room for them. A host that produces a streamed output it never pops keeps its unread samples intact (results wait in their structures instead of overwriting the ring) and gets an "Output stream not consumed" warning; `get_available_samples()` now also collects correctly for sessions with `blocking_ratio > 0`.
- The blocking-mode deadline in `process()` is derived from the resolved reference stream instead of `num_input_samples[m_tensor_index]`, which read the parameter count of a non-streamable tensor.
- Sample counts for non-streamable tensors passed to `process()`/`push_data()`/`pop_data()` are clamped to the tensor size instead of reading or writing past the value storage, and non-streamable values read 0 before they are first set or produced (the storage was uninitialized memory).
- The TypeScript `HostConfig` constructor no longer silently creates an empty (zero buffer size) config when the optional arguments are omitted.
- `Context::create_session()` no longer leaks the session count when a backend processor's constructor or a custom processor's `prepare()` throws (#106), which kept the inference threads running for the rest of the process: registration is now the last step and attached processors are released on failure. Covered by the new `CreateSessionFailure` tests.
- `BackendBase::process` (the CUSTOM roundtrip) no longer reads past the end of the output tensor vector when a model has more input than output tensors, which corrupted memory and crashed hosts. Covered by the new `BackendBase` tests.
- Backend runtime symbols are no longer exported from binaries embedding anira, which crashed hosts that ship their own copy of a runtime (Ableton Live 12 bundles ONNX Runtime): the prebuilt archives are linked hidden, anira uses hidden visibility with `ANIRA_API` as the export allowlist, and `OnnxRuntimeProcessor` verifies the resolved ORT API version and throws instead of crashing. See the new "Host application ships its own backend runtime" troubleshooting section.
- The shared `libanira` now exports exactly the `anira::` API. `-fvisibility=hidden` alone still left ~3800 non-API symbols in its export table — `std::` instantiations (libstdc++ stamps namespace `std` default-visibility), LibTorch `C10_API` typeinfo, and the whole default-visibility ExecuTorch desktop runtime (`executorch::`, `xnn_*`, `pthreadpool_*`, vendored `c10::`, Eigen/BLAS), which a host shipping its own XNNPACK or LibTorch could interpose. An ELF version script (`cmake/anira.map`) / Mach-O export list (`cmake/anira.exports`) pins the export table to namespace `anira`, and the desktop ExecuTorch archives are linked with `--exclude-libs` on ELF (also propagated to consumers of a static anira). On macOS a plugin that links a *static* anira with ExecuTorch must restrict its own exports with `-exported_symbols_list` — ld64 has no hidden variant of `-force_load` — see the troubleshooting guide.
- A static anira no longer leaks its API out of the plugin embedding it. On ELF/Mach-O `ANIRA_API` expanded to `visibility("default")` even for a static build (`ANIRA_STATIC_DEFINE` was only ever defined for MSVC), so a plugin linking `libanira.a` with hidden visibility still exported the whole `anira::` surface (263 symbols on the unload test module) — under an `RTLD_GLOBAL` host two plugins embedding different anira versions would have interposed each other. `ANIRA_API` is now empty on every platform whenever anira is static (`ANIRA_STATIC`, propagated to consumers by the CMake package), and the `anira_exports` CTest asserts that a static build's module exports nothing of namespace `anira`.
- `TensorShape::m_backend` is now default-initialized (flagged by UBSan).
- Session lifecycle calls are thread-safe across sessions: `Context::create_session()`, `release_session()` and `prepare_session()` serialize their mutation of the shared state, so concurrently created/destroyed `InferenceHandler`s no longer corrupt memory. Covered by the new `ConcurrentLifecycleTest`.
- Stateful pending dispatches no longer survive `prepare()`.
- An inference dequeued while its session is momentarily uninitialized is completed with zeroed output instead of stranding its structure (or wedging the stateful dispatch gate).
- A session-exclusive task left awaiting dispatch after a raced task boundary or a momentarily full queue is re-kicked by any output poll, so the chain no longer stalls.
- `Context::drain_inference_queue` no longer silently loses another session's inference when requeueing fails.
- ExecuTorch desktop builds no longer fail on Linux distributions without Debian's multiarch layout

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
