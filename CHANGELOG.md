# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **One-sided streaming** (#98, #99, #110): a *generator* (no streamable input) is driven by output demand — `process()`/`pop_data()` submit one inference per `postprocess_output_size` samples — and an *analyser* (no streamable output) exposes latest-completed values through `PrePostProcessor::get_output()`. The `HostConfig` reference stream may be an input or an output: `m_tensor_index` defaults to the new `HostConfig::k_first_streamable`, the new `m_tensor_is_input` selects the direction, `resolve_reference()`/`get_reference_size()` expose the resolution, and a non-streamable or out-of-range reference makes `prepare()` throw `std::invalid_argument`. Mirrored in the WebAssembly/TS wrapper (`tensorIsInput`, `HostConfig.firstStreamable()`). Tests: `OneSidedStreaming`, `HostConfigReference`, `InferenceManagerOneSided`.
- **ExecuTorch backend** (`ANIRA_WITH_EXECUTORCH`, on by default; static-only; CMake ≥ 3.24 on desktop): `.pte` programs through the new `ExecuTorchProcessor` (`InferenceBackend::EXECUTORCH`, XNNPACK CPU delegate, file and buffer loading, JSON configs, `model_function` selects among several named entry points), prebuilt libraries for desktop and Android/iOS from anira-project/backends, `EXECUTORCH` entries in the bundled model configs, benchmarks and a `minimal-executorch` example. Tests: `ExecuTorchModelFunction`.
- Receptive-field (sliding-window) models in the default `PrePostProcessor`: an input tensor larger than its `preprocess_input_size` is filled with ring-buffer history plus the fresh hop. Tests: `PrePostProcessorWindow`.
- `Context` is immortal and owns the inference thread pool exactly while sessions exist (#104) — built with the first registration, stopped and joined with the last. `Context::shutdown()` covers hosts that unload with a live instance (automatic through a library-unload hook on ELF/Mach-O; on Windows call it from the module-exit entry point); `release_core_if_idle()`/`has_core()` reclaim the core. Tests: `ContextLifecycle` and the host-shaped `LibraryUnload` suite (`test/unload`), which loads, drives and unloads a plugin-shaped module and asserts it was unmapped.
- Symbol policy through the shared tanh-tooling CMake modules (`cmake/tanh/`, pinned like the clang configs): `tanh_apply_symbol_policy(anira EXPORT_PREFIX ANIRA)` — hidden visibility on every platform with `ANIRA_API` as the export allowlist, `-fno-gnu-unique` under GCC, one section per function/variable with `--gc-sections`/`-dead_strip`/`/OPT:REF` on the shared library, `/wd4251` — and `tanh_set_export_allowlist(anira NAMESPACE anira)`, which generates the version script / `-exported_symbols_list` pinning `libanira`'s table to namespace `anira`. A static anira still localizes the bundled `libtanh_core.a` on its consumer's link (defense-in-depth; tanh-lib now applies the same policy itself). The `anira_exports` CTest (`tanh_add_export_check`) fails on any non-`anira::` export of the shared library, any engine symbol exported from the plugin-shaped module, or an `STB_GNU_UNIQUE` symbol (on PE, which cannot interpose, the dllexports LibTorch's headers and the LiteRT/TFLite archives force are tolerated). The CLAP example and the unload module show the plugin side in two lines each (`SYMBOL clap_entry`, `SYMBOL unloadtest_*`). Platform decisions in CMake key on `TANH_OPERATING_SYSTEM` / `TANH_BINARY_FORMAT` (`cmake/tanh/platform.cmake`) instead of `APPLE`/`UNIX`/`WIN32`/`EMSDK_VERSION`; git versioning, RTSan, googletest/benchmark, CPack and install RPATHs come from the same modules.
- `build_install` CI workflow: installs anira into a fresh prefix (shared and static, Linux/macOS/Windows) and builds and runs `test/install` against it — a consumer of anira alone and one that calls ONNX Runtime through `anira::onnxruntime` (as an executable and as a hidden plugin-shaped module whose export table is scanned), plus a negative `try_compile` proving the engine header is unreachable through `anira::anira`. The `InstallConsumer` CTest runs the same locally.
- The Windows `build_test` legs run the suite: `ctest` is invoked with `-C Release` — without a configuration the Visual Studio generator listed no tests and the legs passed on "No tests were found". `test/install` copies a consumer's runtime DLLs next to it (`$<TARGET_RUNTIME_DLLS>`): Windows ships its own `onnxruntime.dll` in System32, which the loader prefers over `PATH`.
- CI preset catalog in `CMakePresets.json` (`ci-tests-{shared,static,gcc}`, `windows-{msvc,clang}-tests-*`, `macos-universal-tests-*`, `desktop-tests-rtsan`, `android-{emulator-tests,arm64-build}`, `ios-{simulator-tests,device-build}`): the configurations the CI workflows migrate onto (`docs/ci-overhaul.md`, step 2), reproducible locally via `cmake --preset <name>`. Backend sets ride on top through configure args; the test presets run ctest 4-way parallel with per-test timeouts. Developer presets are unchanged.
- **ABI (unstable):** `anira/abi/build_info.h`, generated at configure time by `cmake/build-info.cmake` from `git describe` and installed next to the headers: the semver triple (`ANIRA_VERSION_MAJOR/MINOR/PATCH`), `ANIRA_VERSION_STRING` (the full describe string), `ANIRA_MAKE_VERSION` (Vulkan packing) and `ANIRA_ABI_MAJOR`/`ANIRA_ABI_MINOR`, derived from the tag (`0.N` for the N-th `v3.0.0-*` pre-release, `X.Y` from `vX.Y.Z`, the next minor when the checkout is past a tag, `0.0` without a reachable v3 tag). The configure prints `ABI version: ...`.
- **ABI (unstable):** the C ABI registry `abi/anira.yml` and its generator `tools/abi/gen.py` (Python 3, PyYAML), which validate the header conventions of the architecture document (thread tags, the 64-bit rule on `ANIRA_NONBLOCKING` declarations, no struct by value, `struct_size`-first Tier-2 records, explicit enum values with a `_FORCE32` terminator) and emit the committed C headers `anira/abi/{export,status,version,enums,log}.h` (every value pinned, `anira_engine` and `anira_provider` as two axes), the TypeScript mirror `web/src/abi/enums.ts`, the symbol lists `abi/symbols-0.txt` / `symbols-draft.txt`, the wasm export list and the status-text table under `src/capi/generated/`, the layout test `test/abi/generated/test_layout.c` with its table `abi/layout-0.txt`, and one Sphinx page per enum. The headers are never edited by hand (`python3 tools/abi/gen.py --repo . --write`, or the `anira_abi_regen` target); `--diff-against <ref>` classifies registry changes as appended or breaking. Implemented: `anira_abi_version`, `anira_check_abi`, `anira_version`, `anira_version_string`, `anira_status_string`, `anira_drain_log`, `anira_log_rt`, `anira_log` (`src/capi/`, behind the exception firewall of `capi_internal.h`). Gates online: `anira_abi_generate` (the committed files match the registry), `anira_abi_layout` (gate 3: every enum width and terminator and every Tier-1 `sizeof`/`offsetof` as `_Static_assert`s, the printed table diffed against `abi/layout-0.txt`), `anira_header_c11` / `anira_header_cxx17` (gate 4: each header alone and all together with no anira define under `-std=c11 -Wall -Wextra -Werror -pedantic`, `/std:c11 /W4 /WX`) and `anira_header_coexist` (the v2 umbrella and the C headers in one TU, both orders); the gtest binary `test_abi`. `anira/system/Exports.h` now includes `anira/abi/export.h` and defines nothing itself.

### Changed

- The 3.x line lives on the `v3` branch: the workflows' `pull_request` filters gate `v3` like `main` (same ten required contexts), the project version comes from the `v3*` tags only (`tanh_git_version(... MATCH "v3*")`, so a checkout without a reachable v3 tag configures as `0.0.0+g<hash>`), the release workflow marks hyphenated tags as pre-releases, and the release artifacts carry the tag's version (`anira-3.0.0-alpha.1-<leg>`) while the install tree keeps the digits-only `project()` version.
- tanh-tooling v0.2.7 (pre-release aware `tanh_git_version`: `TANH_VERSION_PRERELEASE`, `TANH_VERSION_DISTANCE`, `MATCH`) pinned together with the tanh-lib commit that carries it (tanh-lab/tanh-lib#34); the drift check is the `tooling-config` job of `lint.yml`.
- The export allowlist admits the C entry points of the 3.x line (`tanh_set_export_allowlist(anira NAMESPACE anira SYMBOL "anira_*")`, `anira_exports` `ALLOW_REGEX "^_?anira_[a-z0-9_]+$"`); the plugin-shaped unload module's entry points are renamed `anira_test_*` -> `unloadtest_*` so the static-leg leak scan cannot mistake them for library exports.
- WebAssembly: `_anira_drain_log` is defined by the library (`src/capi/log.cpp`) and exported by name from `cmake/build-wasm.cmake`; the wrapper's copy is gone. `web/src/helpers.ts` keeps calling it.
- CI: the PR tier covers every backend and both linkages — `Linux-x86_64-static` (ExecuTorch's first PR-time compile and test run) and `Linux-x86_64-tflite-shared` join the three shared legs; a one-line guard in the required `result` job keeps them there. No additional macOS load.
- CI: `ci-install-{shared,static}` presets replace the inline `-D` flag strings of the install workflow — the last CI legs without a preset. The Windows install legs move under Ninja+vcvars like every other Windows leg.
- CI: every workflow's legs live in checked-in `.github/*_matrix.json` files with an explicit `pr` flag; `contributing.rst` names the places the tiers are defined instead of mirroring them. All ci-actions references pinned at `v0.3.10`.
- Tests: `test/` mirrors `include/anira/` — a test lives in its unit's directory, `contracts/` holds the build/link/packaging checks, `support/` the shared infrastructure. Pull requests run the full suite: the old 51-test exclusion (every backend-inference test among them) is gone, so PR and queue differ only in which legs build, and a plain `ctest` reproduces CI anywhere.
- CI: the `multiarch` sccache rows are dropped — measured over three queue runs they hit 0-22% against 52-54% on the unflagged Linux legs. The `dualarch` Rosetta second pass stays.
- Security: `web/package-lock.json` resolves `protobufjs` 7.6.6 (#120, CVE-2026-48712) — a lockfile refresh; `onnxruntime-web`'s `^7.2.4` range already admits the fix.
- **Breaking:** backend linkage follows `BUILD_SHARED_LIBS` — a shared anira links shared backends, a static anira static ones — and `ANIRA_<ENGINE>_LINKAGE` is gone. An engine that does not ship the required linkage is disabled with a warning: LibTorch (shared-only) in static builds, ExecuTorch (static-only) in shared builds, which previously absorbed its archives into `libanira`. iOS and Emscripten refuse `BUILD_SHARED_LIBS=ON` (the `wasm-*` presets set it `OFF`). Asserted by the `BackendLinkage` test and on the CI configure output. Migration: build static for ExecuTorch; drop `-DANIRA_<ENGINE>_LINKAGE`.
- **Breaking:** no public header includes an engine header — every processor keeps its engine state behind a named pimpl (`struct Instance`, defined in the `.cpp`; the pattern is documented on `BackendBase`), enforced by the `anira_header_isolation` CTest. Migration: include engine headers yourself and link the engine target (next entry).
- **Breaking:** the engines are explicit imported targets that anira links `PRIVATE` — `anira::onnxruntime`, `anira::tflite`, `anira::litert`, `anira::libtorch`, `anira::executorch` — defined by `cmake/aniraBackendHelpers.cmake` identically in the build tree and in the installed package (`aniraBackendTargets.cmake`, generated). `target_link_libraries(x anira::anira)` gives anira's headers and `USE_*` definitions and nothing of any engine (a static anira still hands its archives to the consumer's link as `$<LINK_ONLY:anira::<engine>>`, linked on demand and hidden through `-load_hidden`/`--exclude-libs`); a consumer whose own code calls an engine links `anira::<engine>` and gets the very file anira uses — one copy per process. Installed engine headers live under `include/anira-backends/<engine>/`; `TORCH_CXX_FLAGS` is a `PUBLIC` compile option of `anira`. Migration: add `target_link_libraries(your_target PRIVATE anira::<engine>)` wherever your code includes an engine header or linked `onnxruntime`/`tensorflowlite_c`/`LiteRt`/`${TORCH_LIBRARIES}` by name.
- **Breaking** (Windows install layout): DLLs install to `bin/`, import libraries and archives to `lib/`, as GNUInstallDirs and vcpkg/Conan do — `anira.dll`, the engine DLLs and tanh-lib's `tanh_core.dll` in one directory next to any executable. Migration: `<prefix>/bin` alone on `PATH`; copy DLLs from `bin/`.
- **Breaking:** anira logs through tanh-lib's `thl::Logger` (#56): every record carries an `anira.<component>` group, the configured level is applied as the runtime level (and still forwarded to the backends), and the host owns the sinks — anira never calls `set_config`. The `LOG_*` stream macros are replaced by printf-style `ANIRA_LOG_*` and real-time-safe `ANIRA_LOG_RT_*` (`ANIRA_WITH_LOGGING=OFF` still compiles them out; Release builds compile in Error records only). Everything reachable from `process()`/`push_data()`/`pop_data()` and the inference threads logs into a context-owned lock-free queue, drained by a low-priority context-owned thread that exists exactly while sessions exist (`LogDrain::Thread`) or by the host through `InferenceHandler::drain_log()` (`LogDrain::Manual`; forced on WebAssembly, `drainAniraLog()`). The RTSan CI job runs without suppressions and builds tanh-lib with `TANH_WITH_RTSAN`. Migration: output goes to tanh-lib's default sinks (platform log on Apple/Android, stdout/stderr elsewhere); hosts wanting a console or a callback configure `thl::Logger`.
- **Breaking:** `ContextConfig::m_log_level` → `ContextConfig::m_log` (`anira::LogConfig`: `m_level`, `m_drain`, `m_queue_capacity` = 512, `m_drain_interval_ms` = 10); the three-argument constructor is unchanged; JSON `context_config.log` block (`log_level` still accepted). Migration: `config.m_log_level = x` → `config.m_log.m_level = x`.
- **Breaking:** `InferenceHandler::reset()` (with `InferenceManager::reset()`/`Context::reset_session()`) is wait-free and safe on the audio thread — a per-session generation makes in-flight inferences stale instead of draining. Migration: where "no inference thread touches the session afterwards" was relied upon, call `prepare()` (which still drains) or synchronize in your backend.
- **Breaking** (containers): `anira::Buffer<T>`, `anira::RingBuffer` (plus the new `RingBufferT<T>`) and `anira::MemoryBlock<T>` are aliases over tanh-lib's Apache-2.0 `thl::core` containers (fetched at configure time, core component only; installed alongside anira and resolved by `find_dependency(tanh COMPONENTS Core)`). `RingBuffer` no longer derives from `Buffer<float>` — use `push`/`pop`/`get_future_sample`/`get_past_sample`; `get_available_past_samples()` returns the retained history; `get_*_sample` wrap instead of range-checking; `Buffer::resize()` zeroes; allocation failure throws `std::bad_alloc`; `Buffer` gains an optional `sample_rate` member (ABI change). Every ring-buffer access on the real-time path uses the block API (`push_block`/`pop_block`/`push_fill`/`discard`/`peek_past_block`), leaving no per-sample call on the audio thread or the inference worker (#111).
- `anira::HighPriorityThread` is deprecated for one release: `InferenceThread` composes tanh-lib's `thl::core::Thread` (`ThreadPriority::RealTime`), which drops the process-wide `pthread_setattr_default_np` side effect on Linux.
- `Context::get_instance()` takes no arguments and returns a `Context&`; the `ContextConfig` travels with the session as a fourth `Context::create_session()` argument (old two-step API deprecated for one minor release); `Context::get_sessions()` returns a copy. Statically embedded plugins on GCC/Linux no longer share one context by accident through `STB_GNU_UNIQUE` binding.
- The active backend defaults to the first model whose backend is available instead of the silent `CUSTOM` bypass. Migration: call `set_inference_backend(InferenceBackend::CUSTOM)` where the bypass was relied upon.
- `HostConfig`'s default reference tensor is the first streamable tensor (input 0 divided by zero in `prepare()` when non-streamable); `InferenceHandler::prepare(config, custom_latency, tensor_index)` throws `std::invalid_argument` for an out-of-range index and reports 0 latency for non-streamable outputs; `sizeof(HostConfig)` changed (ABI).
- `Context::drain_inference_queue` is `[[clang::blocking]]` under RTSan, drains to a fixpoint and completes never-started tasks as silence; the stateful dispatch gate carries an epoch, so a thread preempted across `prepare()` cannot corrupt the rebuilt session's dispatch.
- The export-decoration header is `anira/system/Exports.h`, a stub over tanh-lib's `tanh/core/ExportMacros.h` (`AniraExports.h` and `AniraWinExports.h` forward, deprecated); `ANIRA_BUILDING` replaces `ANIRA_EXPORTS`, `ANIRA_STATIC` replaces `ANIRA_STATIC_DEFINE` (old spellings honoured).
- The test suite reads WAV fixtures through tanh-lib's header-only `thl::core::read_wav` instead of `test/WavReader.h` (#91). The build-tree CMake modules are named lowercase-hyphen (`backends`, `validate-options`, `check-exports`, `detect-emscripten`, `build-wasm`, `test/install/run-install-test`); the package-shipped files keep CMake's `anira<Thing>.cmake`.
- The test suite builds as per-component binaries — `test_utils` (including the root `test_WavReader.cpp`, which exercises `thl::core::read_wav`), `test_scheduler`, `test_backends`, and `test_handler` (the root-file integration suite) — instead of one `tests` binary; the `test_*` names are what the shared ci-actions mobile runners auto-discover, and the mobile CI launches each binary once. Same tests, same desktop CTest totals (mobile registers one CTest entry per binary instead of one in total); the standalone checks (`anira_header_isolation`, the unload suite, `anira_exports`, `InstallConsumer`) are unchanged.
- The example-model fixture repositories are fetched at pinned commits instead of tracking their `main` branches (`extras/fetch-models.cmake`; override with `-DANIRA_MODELS_<NAME>_REF=<sha>`; the RAVE model URL and its SHA-256 are pinned the same way, and the fetched trees no longer keep `.git` metadata). An existing `extras/models/` checkout is left untouched, as before.
- CI: `ctest` runs 4-way parallel (build-tree-driving tests carry a `RESOURCE_LOCK`); the macOS x86_64 legs run natively on `macos-15-intel` instead of under Rosetta on Apple Silicon; the model fixtures are seeded once per push into a cross-platform Actions cache that every build job restores instead of cloning ~1.7 GB per job.
- CI: `build_test` runs on the shared `tanh-lab/ci-actions@v0.2.1` in preset mode over `.github/build_test_matrix.json`, with a tier split — pull requests run a representative 10-leg subset that fits the free-runner concurrency caps, every other event (push, dispatch, `on_tag`'s reusable call) runs all 23 legs, now including a gcc and a clang-cl coverage leg. Windows builds MSVC-first under Ninja inside a vcvars environment (sccache caches cl.exe for the first time); a `build_test result` job aggregates the matrix as the single required status. The local `setup`/`build`/`test` composites remain only for the workflows migrating in the next step.
- CI: tier policy v2 — a pull request builds one leg per platform family (Linux x64, Windows x64, macOS universal, one iOS, one Android, web) and runs only the fast tests (the heavy model-inference binaries `test_handler`/`test_backends` are excluded by their CTest labels); the merge queue runs the full sweep of every workflow including lint and docs; pushes to main run only the docs deploy and a build-only cache-warming job (queue- and PR-scoped Actions caches are not restorable elsewhere, so warm compile caches and the model-fixture entry must be produced on main). Install, examples, benchmarks and the clang checks no longer run on pull requests at all — the queue is their gate, and every workflow reports a `<name> result` status the merge queue requires.
- CI: ASan+UBSan and TSan join RTSan in the sanitizer workflow, and the workflow itself becomes a caller of `tanh-lab/ci-actions`' reusable `build-sanitizer` workflow over `.github/sanitizer_matrix.json` — the plan/matrix/cache/options logic lives once upstream, shared with tanh-lib, which drops its own hand-maintained copy. The two new legs run the engine-free scope (the prebuilt backend runtimes are uninstrumented, so a sanitized build linking them would report on frames it cannot see into), which also keeps them small enough to finish inside the RTSan leg they run beside: measured against a non-sanitized baseline of the same suite, ASan+UBSan costs 1.3x the test time and TSan 2.8x, so the queue pays nothing for either. The presets are `RelWithDebInfo` with `-DNDEBUG` dropped from the build-type flags rather than `Debug` — `-O0` cost 2.4x the TSan test time for no extra signal, and dropping `-DNDEBUG` (which `CMAKE_<LANG>_FLAGS` cannot override, being appended last) is what keeps `assert()` live. `UBSAN_OPTIONS=halt_on_error=1` is set upstream for every caller: UBSan defaults to print-and-continue, so without it a diagnosed undefined behaviour scrolled past and the leg still reported success. A standalone LSan leg is not included — ASan enables `detect_leaks` on Linux, so it already covers that ground. The library-unload suite does not run on the sanitized legs: its premise is that the library's memory goes away underneath code that may still touch it (`LeakedThreadCrashesOnUnload` leaks a thread on purpose, then dlcloses the module), which is exactly what a sanitizer reports; the uninstrumented `build_test` legs keep covering it. Every test binary now releases the context core after its last test (`test/CoreReclaim.cpp`), so the immortal-by-design core is reclaimed rather than reported as a leak.
- CI: `on_tag`'s desktop release builds use the same presets and toolchains the test matrix validates — clang on Linux/macOS, MSVC cl.exe under Ninja+vcvars on Windows (following `windows-latest`) — and the last local composites (`setup`/`build`/`test`) are gone; the anira-owned install/codesign/release composite remains and gains a `BUILD_DIR` input.
- CI: the remaining desktop workflows (install, examples, benchmark, docs, sanitizer) run on the shared `tanh-lab/ci-actions@v0.2.2` — Windows examples/benchmark builds move under Ninja+vcvars with sccache (previously ~11 min cold MSBuild), and the sanitizer job runs the `desktop-tests-rtsan` preset. macOS consolidates onto the 5-concurrent cap (`docs/ci-overhaul.md` §3.6): the universal build_test legs run both slices on one runner (native arm64 + `arch -x86_64` under Rosetta), one sequential iOS job replaces three, one macOS install job runs both linkages, and examples/benchmark PR subsets plus a docs path filter keep a pull request under the Free plan's 20-concurrent budget.
- CI: compile caching works for the first time — without `ACTIONS_CACHE_SERVICE_V2=on`, sccache spoke the retired v1 cache API and every write failed read-only (a permanent 0% hit rate since the caches were introduced); the flag is set in every sccache workflow and in tanh-lab/ci-actions' `cmake-build`. clang-tidy analyses only a PR's changed files (full sweep on push, or when a header/CMake/tidy-config file changed) — a full sweep is ~6 min.
- CI: the desktop build+test pipeline and the cache warmer are `tanh-lab/ci-actions` *reusable workflows* — `build_test.yml` and `warm_caches.yml` shrink to anira's facts (triggers, the matrix JSON, the fast-test labels, the models cache), and the plan/tier/assert/Rosetta/result logic lives once upstream, shared with tanh-lib. The Android emulator run goes through the shared `cmake-test-android` action (device staging, marker-based exit codes, per-binary failure collection) driven by the `android-*` presets; `.github/scripts/android_emulator_test.sh` is deleted. The wasm leg of `build_web` builds through the shared `cmake-build-wasm` action, whose `EMSDK_VERSION` default is the tanh-lab-wide emsdk pin; the npm package build stays anira-local in the same job. The iOS job stays anira-local by design — it is the macOS-cap consolidation of three backend sets on one simulator boot.
- CI: a full-suite coverage leg on every push to main — the `ci-tests-coverage` preset builds instrumented with clang source-based coverage (accurate under optimization, so Release stays), `llvm-cov` exports lcov over `libanira.so` and the test binaries, and the result uploads to Codecov via tokenless OIDC (the peer standard among comparable C++ libraries). The coverage badge sits at the top of the README; the per-file drill-down is the baseline metric for the test-suite audit (`docs/ci-overhaul.md`, step 9). Not queue-gated — a coverage dip is information, never a merge blocker.
- CI (audit fixes, 2026-08-31): the npm publish moved from an ungated tag-push workflow into `on_tag` behind the same test gates as every release artifact, building through the shared `cmake-build-wasm` action (`publish_web.yml` deleted); the benchmark queue gate runs ctest with `--no-tests=error` so lost test discovery can no longer pass green; the Rosetta dual-arch step fails on an empty glob (ci-actions v0.3.3, pinned repo-wide); `ci-base` carries the planned `/Z7` invariant (`CMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded`); `on_tag` runs under scoped permissions with the secret-adjacent third-party actions pinned by commit SHA; jobs that configure anira all carry the authenticated backend-check token; and the stale pre-stub tier descriptions were swept from the queue-only workflows.
- Tests: `InferenceManagerTest.WithEmptyCustomLatency`/`WithPartialCustomLatency` run over a 3-param subset (one single-tensor, one multi-tensor, one non-streamable mix) instead of the full 23-param latency sweep — the empty/partial custom-latency contracts branch only on the output-tensor count and the streamable mix, never on the per-config constants, so the sweep re-ran byte-identical code 20+ times (and registered 17 skip-only spawns). Confirmed by the adversarially verified step-9(a) audit; `Simple`/`WithCustomLatency` keep asserting every param. 40 fewer test executions per leg, coverage unchanged.
- CI: coverage measures what it claims — a second static coverage leg adds the ExecuTorch and TFLite code paths to the Codecov report (llvm-cov only sees files compiled into the uploaded binaries), and `codecov.yml` excludes the wasm-only and benchmark sources explicitly instead of silently. Every `cmake-build` job reports its sccache hit rate into the step summary (ci-actions v0.3.4, pinned repo-wide); the macOS universal legs opt into sccache's multi-arch caching (`multiarch` matrix key), ending their by-design 0% hit rate; `on_tag` resolves build directories through the shared `preset-binary-dir` action instead of a hand-maintained matrix column.
- Build: ExecuTorch is consumed as ONE merged `libexecutorch.a` per platform (anira-project/backends v2.4.0; `executorch_registrations.lib` whole-archived on Windows) with the kernel/backend registrations pre-linked, and links like the other static engines — `find_package(executorch)`, the imported-target sanitizing, the per-target force-load options, the whole-archive microkernels special case and the installed `lib/cmake/ExecuTorch` re-resolve are gone, and the CMake ≥ 3.24 desktop floor with them. The per-engine CMake wiring now lives in one file per backend (`cmake/backends/{onnxruntime,tflite,litert,libtorch,executorch}.cmake`) over two shared layout/target macros in `cmake/backends.cmake`. Consumers are unaffected: `anira::executorch` and every other `anira::<engine>` target keep their contract.
- CI/build (dedup leftovers, 2026-09-01): Emscripten detection is tanh-tooling's `tanh_detect_emscripten()` (v0.2.4; anira keeps only its wasm compile flags); the install-tree consumer check runs through the shared `install-consume-check` action (ci-actions v0.3.7, pinned repo-wide) with the anira-only export-table scan as a follow-up step; `clang_format.yml` + `clang_check.yml` merge into one `lint.yml` whose result-job names keep the ruleset's required contexts unchanged. `cmake/msvc-support.cmake` stays deliberately — it is anira product logic (backend DLL staging), not shareable tooling.
- Tests: the parameter sweeps are pruned to the parameterizations that reach distinct library code paths (step-9a audit, tranche 2; every prune survived two adversarial verifiers). `InferenceTest` runs the 2048 (deepest chunk/ring pressure) and 256 (buffer == tensor-size boundary) host configs instead of five — 512/1024 walked an identical branch sequence with only the chunk count differing, and the 300-frame row registered 24 cases that skipped on every leg. `WithCustomLatency`/`ResetStatefulHammer` move to an `InferenceControlTest` fixture instantiated for the bypass and LibTorch backends only (the scheduler paths they assert have no backend branch); `StatefulOrdering` drops the 512 case and the non-asserting reference row; `InferenceManagerTest.WithCustomLatency` joins the 3-param subset. Measured on one machine, ONNX+LiteRT shared: 246 → 180 tests, 20.1 s → 15.8 s, all green; per full-tier CI leg the audit projects ~236 s → ~153 s.
- Tests: the parameter sweeps are pruned to the parameterizations that reach distinct library code paths (step-9a audit, tranche 2; every prune survived two adversarial verifiers). `InferenceTest` runs the 2048 (deepest chunk/ring pressure) and 256 (buffer == tensor-size boundary) host configs instead of five — 512/1024 walked an identical branch sequence with only the chunk count differing, and the 300-frame row registered 24 cases that skipped on every leg. `WithCustomLatency`/`ResetStatefulHammer` move to an `InferenceControlTest` fixture instantiated for the bypass and LibTorch backends only (the scheduler paths they assert have no backend branch); `StatefulOrdering` drops the 512 case and the non-asserting reference row; `InferenceManagerTest.WithCustomLatency` joins the 3-param subset. `RingBufferTest.Initialization` is dropped as upstream-covered — `anira::RingBuffer` aliases `thl::core::RingBuffer`, whose own suite asserts the same post-init state — and the two invalid-swap `Buffer` tests and the two single/repeated session-failure tests merge into one each. Measured on one machine, ONNX+LiteRT shared: 246 → 177 tests, 20.1 s → 13.1 s, all green; per full-tier CI leg the audit projects ~236 s → ~153 s.
- Pins: tanh-lib at `1c992c0` (main after PRs #17, #19 and #20; the next tag once cut), backends release v2.3.0 (ExecuTorch 1.3.1, pre-isolated static desktop LiteRT).

### Removed

- `Context::release_instance()` and `Context::release_thread_pool()` (`Context::shutdown()` is the explicit teardown); the never-released `reset_non_blocking()` variants of `InferenceHandler`, `InferenceManager` and `Context`.
- `ContextConfig::operator==` and `operator!=`, which were `private` with no `friend` and no caller — unreachable from anywhere. `Context` compares configurations field by field so it can name the field that differs (`apply_or_compare_config_locked`), which is what the diagnostics need.
- The unreachable interpolation branch of `anira::calculate_percentile`. `percentile_index` is a `size_t`, so `percentile_index == static_cast<size_t>(percentile_index)` is always true and the branch never ran. The function computes the nearest rank — the result is always one of the input values — and its documentation said "linear interpolation when necessary"; the documentation now describes what it does.

### Fixed

- The fallback paths a host reaches by misconfiguration or under load are under test (`FallbackPaths`): selecting a backend the session has no processor for falls back to the default round-trip processor instead of emitting silence, `set_non_realtime(true)` is refused when no inference thread could ever satisfy the resulting wait (arming it would guarantee a hang), and the deadline-bounded `pop_data()` takes the blocking branch when a blocking ratio is configured. The backend fallback is one block per backend in `InferenceThread`, so the test iterates the compiled-in backends rather than repeating itself.

- Codecov is configured as a metric and nothing else: no project or patch commit status (so it can never fail a pull request or block the merge queue), no bot comment, no inline diff annotations. This matches `coverage.yml`'s stance that "a coverage dip is information, never a merge blocker". `default.profraw`, a 216 KB clang coverage artifact a test binary drops when run from the repo root, is removed from the repository and `*.profraw`/`*.profdata` are gitignored. The coverage workflow runs in the merge queue and reports a `coverage result` status, and a change to `CMakePresets.json` now triggers it on a pull request as a change to the workflow itself already did. A coverage *dip* is still never a merge blocker — `codecov.yml` turns the project and patch statuses off, so no number Codecov computes can fail anything — but the pipeline breaking (the instrumented build, the suite, the `llvm-profdata` merge, the `llvm-cov` export) is a real breakage, and it used to surface only after merging: the workflow ran on push to `main` alone, so a change to the preset it builds first executed once it had already landed.

- Data race in `Context::release_session()`: whether the released session was the last one was re-read from `m_sessions` *after* the lifecycle lock was dropped, while a concurrent `release_session()` erased from that same vector under the lock — so two `InferenceHandler`s destructed in parallel raced on the registry, and the "last session" verdict driving the final log drain could be read from a vector mid-erase. The emptiness is now captured inside the locked block and carried out, which keeps the drain call outside the lock (a host's log callback may re-enter the context). Found by the new TSan CI leg.

- The backend-archive, release-metadata and RAVE-model downloads retry once after a transient failure (SSL connect errors against GitHub occasionally failed a CI configure).

- The JUCE example links under clang on Linux (JUCE's recommended LTO flags produced undefined vtable references with clang + GNU ld and are dropped — an example gains nothing from LTO) and stages its Windows runtime DLLs generator-agnostically (the helper-tool and per-format paths assumed the Visual Studio generator's per-config layout; Ninja has none).

- Backend runtime symbols are no longer exported from binaries embedding anira, which crashed hosts that ship their own runtime (Ableton Live 12 bundles ONNX Runtime): the engine archives are linked hidden, `libanira` exports exactly the `anira::` API (previously ~3800 `std::`/`c10::`/`executorch::`/`xnn_*` symbols leaked past `-fvisibility=hidden`), a static anira leaks nothing (`ANIRA_API` used to expand to `visibility("default")` on ELF/Mach-O even when static), and `OnnxRuntimeProcessor` throws a descriptive error instead of crashing when `OrtGetApiBase()` resolved to a foreign runtime. On macOS a plugin embedding a static anira with ExecuTorch must restrict its own exports (`-exported_symbols_list`; see the troubleshooting guide).
- `USE_ANIRA_WEB` was defined in every build (the generator expression tested the literal `EMSDK_VERSION`), so desktop `OnnxRuntimeProcessor` created its `Ort::Env` with ONNX Runtime's global thread pools; Emscripten builds only now.
- Installing on `lib64` hosts (Fedora) produced a package whose `find_package(anira)` failed with LibTorch or ExecuTorch, whose CMake packages hardwire `<prefix>/lib/`; anira installs patched copies.
- One-sided streaming (redo of the reverted #101, see #110): `SessionElement::prepare()` no longer hangs for generators or crashes for analysers, latency vectors are index-aligned with the outputs (#98), push-only pipelines no longer stall after `m_num_structs` chunks (#99; unread results wait in their structures and an "Output stream not consumed" warning is logged), the blocking deadline uses the resolved reference stream, non-streamable sample counts are clamped and read 0 before they are first set, and the TS `HostConfig` constructor no longer creates an empty config.
- Session lifecycle: `Context::create_session()` no longer leaks the session count when a processor constructor or `prepare()` throws (#106); `create_session()`/`release_session()`/`prepare_session()` are thread-safe across sessions (`ConcurrentLifecycle` tests); an inference dequeued while its session is uninitialized completes with zeros; a session-exclusive task stranded by a raced boundary or full queue is re-kicked by any output poll; `drain_inference_queue` no longer loses another session's inference; stateful pending dispatches do not survive `prepare()`.
- `BackendBase::process` (the CUSTOM roundtrip) no longer reads past the output vector when a model has more inputs than outputs (`BackendBase` tests); `TensorShape::m_backend` is default-initialized (UBSan); ExecuTorch desktop builds work on Linux distributions without Debian's multiarch layout.
- `JsonConfigLoader` reports and skips two malformed `tensor_shape` inputs it used to let through: a non-array `input_shape`/`output_shape` logged an error and then parsed on regardless, so nlohmann's `type_error` escaped the loader; and a shape that could not be parsed produced a `TensorShape` with no dimensions, which asserted in debug builds and made `InferenceConfig` throw `std::invalid_argument` in release ones. Both now drop the entry like every other malformed entry. Tests: `JsonConfigLoaderErrors`.
- `LibtorchProcessor` throws `std::runtime_error` when a model cannot be loaded, like the other backends, so `Context::create_session()` rolls the session back. It used to log the failure and carry on with an empty module, letting an engine-specific `c10::Error` escape from the next call. Tests: `LibTorchProcessor`.
- `anira::HighPriorityThread` links again in a Windows consumer of a shared anira. The class is header-only — every member is inline and no anira translation unit includes the header — but it carried a class-scope `ANIRA_API`, which expands to `__declspec(dllimport)` for a consumer and told MSVC every member lived in the DLL. Nothing exported them, so instantiating the class was five unresolved externals. The decoration is gone; the class compiles into the consumer, as a header-only class should. The class also gained a warning that a derived class must call `stop()` in its own destructor: `~HighPriorityThread()` calls it too, but a base destructor runs after the derived part is gone, so a worker still inside `run()` would touch members that no longer exist — undefined behaviour for any `run()` that reads or writes the derived object's own state. Tests: `HighPriorityThread`.

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
