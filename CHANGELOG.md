# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- RTSan real-time safety CI checks and testing (not done yet)
- clang-tidy conformance across the library, tests and benchmark sources, enforced in CI via the `tanh-lab/ci-actions/clang-tidy-check` action (`clang_tidy.yml`)

### Changed

- **Breaking:** the `InferenceConfig::Defaults` compile-time constants were renamed from the `m_` prefix to the `k_` prefix to match the constant-naming convention (`m_warm_up` → `k_warm_up`, `m_session_exclusive_processor` → `k_session_exclusive_processor`, `m_blocking_ratio` → `k_blocking_ratio`). The mutable `Defaults::m_num_parallel_processors` is unchanged.
- `anira::calculate_min` / `anira::calculate_max` are now `inline` free functions instead of `const auto` lambdas (source-compatible: existing call sites and uses as a callable are unaffected)
- The internal logging helper `isLoggingEnabled()` was renamed to `is_logging_enabled()`
- Migrated the shared clang configs (`.clang-format`/`.clang-tidy`/`.clangd`) from the `tanh-lib` submodule symlinks to [`tanh-tooling`](https://github.com/tanh-lab/tanh-tooling) (pinned `v0.1.4`): committed as real files, kept in sync by the `clang_check.yml` drift check, and the now-unused `tanh-lib` submodule was removed (configs are byte-identical, so lint/format results are unchanged)
- Adopted the default Claude Code config: `.claude/settings.json` now enables the `tanh-tools` plugin from the tanh-tooling marketplace (its format/lint/type-check hooks supersede the previous bespoke `.claude/hooks`)

### Fixed

- Potential use-after-free in `Buffer::malloc_channels()` when channel-pointer allocation fails

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
