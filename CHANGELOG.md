# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **ABI (unstable):** `anira/abi/machine.h` and `anira/abi/thread.h`, the runtime half of section 4 and the user-driven threads of section 6 (ABI 0.2; 116 promised names). `anira_machine_create` / `destroy` / `probe` / `capabilities` over this copy's core: the config is reconciled into the core the way sessions are (the first user's config takes effect whole; later machines and sessions reconcile per field: log level most verbose wins; wait strategy, drain mode, queue capacity and interval first win with a warning; the thread count only shrinks, never to zero; once the last user is gone the next config takes effect whole), the config's sink is registered for the machine's lifetime, its `ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK` and `ANIRA_LOG_FLAG_TRACE_FAILURES` are held while it lives (counted across machines), a device block is `ANIRA_ERROR_NOT_SUPPORTED` in this pre-release. The Host-only capability report: `anira_capabilities_backends` / `domains` / `ext_kinds` / `edges` / `edge` over the Tier-2 records `anira_backend_id` and `anira_edge_info` (stride-explicit enumeration: `out == NULL` asks for the count, a short buffer returns `ANIRA_INCOMPLETE`), one zero-copy host edge per compiled-in engine at `ANIRA_RUNG_STATIC`; `anira_enabled_backends` without a machine. `anira_machine_byte_image_bytes` (the dense encoding), `anira_machine_drain_log`, `anira_machine_num_inference_threads` / `anira_num_inference_threads` (the pool size: 0 before the first handler and for a machine that brings its own threads), the steady clock `anira_now_ms` / `anira_now_ns` (`[callback-safe]`, `ANIRA_NONBLOCKING`), and the shutdown family `anira_shutdown` / `anira_release_core_if_idle` / `anira_has_core`. `anira_inference_thread_create` / `run_loop` / `execute` / `start` / `stop` / `has_exited` / `should_exit` / `is_running` / `destroy` over `Context::make_inference_thread`; `has_exited` is a new atomic set on every loop exit. Tests: `test_Machine`, `test_Thread` in `test_abi`; gate 4 covers both headers.
- **Log sinks:** an `anira_log_fn` set on a machine config receives every record the platform sink would, as the `anira_log_record` projection of `anira/abi/log.h` (`level`, `flags` with `ANIRA_LOG_RECORD_REALTIME` / `ANIRA_LOG_RECORD_CONTRACT_VIOLATION`, `dropped_before`, `sequence`, the timestamps, `group`, `message`), filtered by that machine's level. Behind it one registry per copy of anira (`anira::detail::add_log_sink` / `remove_log_sink` in `anira/utils/Logger.h`, one `thl::Logger` callback installed while a sink exists): two machines in one copy are two sinks over one core, `anira_machine_destroy` waits for its sink's in-flight calls and is refused with one Error record from inside that sink. The test collector (`test/support/log_record_collector.h`) is such a sink.
- **Presence gates:** `anira_abi_link` (the generated `test/abi/generated/link_probe.c` takes the address of every promised and draft entry point and links `anira::anira` the way a consumer does, on every leg that builds the tests, static ones included), `anira_symbol_baseline` (`cmake/abi-symbols.cmake`: every promised and draft name is in the shared library's real export table; presence mode, the "nothing else" half follows the export cut), and the WebAssembly link on the generated `web/src/abi/exports_wasm.txt` (`-sEXPORTED_FUNCTIONS=@...`): from now on a registry entry without a definition on Emscripten fails `build_web`.
- **C++ (not ABI-stable):** `anira::Machine` (over `anira_machine_create`, with `capabilities()`, `probe()`, `drain_log()`, `num_inference_threads()`, `byte_image_bytes()`), `anira::Capabilities` (the enumerations as vectors, `edge()` throwing `ANIRA_ERROR_EDGE_UNREACHABLE`), `anira::BackendId`, `anira::enabled_backends()`, `anira::now_ms()` / `now_ns()`, `anira::shutdown()` / `release_core_if_idle()` / `has_core()` in `anira/anira.hpp`; gate `anira_header_cxx20` probes them.
- **ABI (unstable):** `ANIRA_DTYPE_F64`, the 64-bit float dtype (`ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 64, 1)`), beside the nine scalar dtypes the headers already pinned; `"float64"` in the JSON dtype vocabulary and `web/src/abi/enums.ts`.

### Changed

- **Breaking:** `anira_shutdown` (and the CLAP example's `clap_deinit`, which now calls it) is effective only when no machine and no handler exist in this copy of anira; otherwise nothing happens and it returns `ANIRA_ERROR_INVALID_STATE`, so one client of a shared `libanira` cannot silence another's sessions. `anira::Context::shutdown()`'s unconditional form stays internal for the library-unload hook. Migration: destroy every machine and handler before the module-exit entry point runs, which a plugin does anyway.
- **Breaking:** `anira::ContextConfig::m_anira_version` and `m_enabled_backends` are gone, with the cross-session version and enabled-backends compares of the context (inside one copy every session shares one header; across two copies nothing can see the other). What the build compiled in is `anira_enabled_backends`; what is usable here is `anira_capabilities_backends` on a probed machine. Migration: nothing to write; code that read the two fields uses the two entries.
- The core owns the real-time log drain thread: a low-priority `thl::core::Thread` named `anira-log` of anira's own runs `Queue::drain()` every drain interval while a session or a machine exists (tanh-lib's `DrainThread` is no longer constructed), and the last user's release, `anira_machine_destroy` or `anira_shutdown` stops it and flushes the queue through the sinks on the calling thread. `ANIRA_LOG_FLAG_TRACE_FAILURES` is admitted by `anira_machine_config_set_log_flags` and the `flags` of `anira_log_desc`.
- Every input and output ring of the pipeline is an instantiation of `RingBufferT<T>` with `T` the element type of the slot's ring dtype: `anira_ring` (`anira/utils/RingBuffer.h`, the type `anira::RingBuffer` now names) holds one instantiation per scalar dtype the ABI pins (float32, float64, float16, bfloat16, int8, uint8, bool8, int16, int32, int64, with `T` the dtype's C type and the three that have none stored as their bits; no two dtypes share a ring; float32 unless a slot says otherwise), chosen at prepare: `SessionElement::prepare` and `Context::prepare_session` take the per-slot `RingDtypes` and the caller's `CustomLatencies` (the former `std::vector<long>`, named), which the 2.x `InferenceManager::prepare` fills from its arguments (a new overload carries both for the 3.x prepare). The ring's block API (`push_block`, `pop_block`, `peek_past_block`, `push_fill`, `push_zeros`, `discard`, `pop_windows`, which now holds the batched sliding-window pop of `PrePostProcessor::pop_samples_from_buffer`) takes the caller's dtype, returns 0 and writes nothing on a disagreement, and never converts; the 2.x float API stays as the float32 face, so every 2.x pre/post processor compiles unchanged. Storage only: the typed Hard entries that reach the rings follow with the 3.x handler.

## [v3.0.0-alpha.1] - 2026-09-03

The first pre-release of the 3.x line: the configuration layer of the versioned C ABI (`anira/abi/*.h`, ABI 0.1) with its JSON loaders, the C++20 builders of `anira/anira.hpp`, and a transitional bridge to the unchanged 2.x runtime. Nothing in it is a binary promise before v3.0.0: while the ABI major is 0 a library accepts only the exact ABI version its headers were generated with, and every pre-release may still change any of it. The soname stays 3 until the 3.x handler lands and flips it to the ABI major. The npm package is still the 2.x TypeScript API, published under the `next` tag.

### Added

- **ABI (unstable):** `anira/abi/build_info.h`, generated at configure time by `cmake/build-info.cmake` from `git describe` and installed next to the headers: the semver triple (`ANIRA_VERSION_MAJOR/MINOR/PATCH`), `ANIRA_VERSION_STRING` (the full describe string), `ANIRA_MAKE_VERSION` (Vulkan packing) and `ANIRA_ABI_MAJOR`/`ANIRA_ABI_MINOR`, derived from the tag (`0.N` for the N-th `v3.0.0-*` pre-release, `X.Y` from `vX.Y.Z`, the next minor when the checkout is past a tag, `0.0` without a reachable v3 tag). The configure prints `ABI version: ...`.
- **ABI (unstable):** the C ABI registry `abi/anira.yml` and its generator `tools/abi/gen.py` (Python 3, PyYAML), which validate the header conventions of the architecture document (thread tags, the 64-bit rule on `ANIRA_NONBLOCKING` declarations, no struct by value, `struct_size`-first Tier-2 records, explicit enum values with a `_FORCE32` terminator) and emit the committed C headers `anira/abi/{export,status,version,enums,log}.h` (every value pinned, `anira_engine` and `anira_provider` as two axes), the TypeScript mirror `web/src/abi/enums.ts`, the symbol lists `abi/symbols-0.txt` / `symbols-draft.txt`, the wasm export list and the status-text table under `src/capi/generated/`, the layout test `test/abi/generated/test_layout.c` with its table `abi/layout-0.txt`, and one Sphinx page per enum. The headers are never edited by hand (`python3 tools/abi/gen.py --repo . --write`, or the `anira_abi_regen` target); `--diff-against <ref>` classifies registry changes as appended or breaking. Implemented: `anira_abi_version`, `anira_check_abi`, `anira_version`, `anira_version_string`, `anira_status_string`, `anira_drain_log`, `anira_log_rt`, `anira_log` (`src/capi/`, behind the exception firewall of `capi_internal.h`). Gates online: `anira_abi_generate` (the committed files match the registry), `anira_abi_layout` (gate 3: every enum width and terminator and every Tier-1 `sizeof`/`offsetof` as `_Static_assert`s, the printed table diffed against `abi/layout-0.txt`), `anira_header_c11` / `anira_header_cxx17` (gate 4: each header alone and all together with no anira define under `-std=c11 -Wall -Wextra -Werror -pedantic`, `/std:c11 /W4 /WX`) and `anira_header_coexist` (the v2 umbrella and the C headers in one TU, both orders); the gtest binary `test_abi`. `anira/system/Exports.h` now includes `anira/abi/export.h` and defines nothing itself.
- **ABI (unstable):** the configuration layer, `anira/abi/config.h`: the opaque handles `anira_tensor_spec` (name, dtype, role, tagged axes, window/context, time ratio, latency), `anira_model_config` (model entries by path or bytes with COPY/BORROW ownership and a release callback that fires once, custom engine ids, canonical -> engine tensor names, inputs/outputs, default engine, state, max instances, anchor), `anira_machine_config` (threads, the log scalars and `anira_machine_config_set_log`, the six device descriptors `anira_{cuda,gl,vulkan,metal,d3d12,webgpu}_desc` with their `_INIT`s, read within `struct_size`), `anira_contract` (Hard with geometry, budget, warmup, on-miss and wait ratio; Async with deadline and policy; `edge_cost`; kind gating with `ANIRA_ERROR_WRONG_CONTRACT`) and `anira_job_options` (head trim, tail flush, below-min policy, borrowed per-job extensions); every single-argument rejection at set time, cross-field legality at prepare. The extension registry of section 1b: `anira_ext_header` / `anira_ext_entry` (`ANIRA_EXT_ENTRY_INIT`), one `set_ext` / `set_ext_json` pair per handle (a known kind at a registered version is deep-copied, `ANIRA_ERROR_EXTENSION_VERSION` for an unregistered version, an unknown kind is stored and fails the walk by name), `anira_registered_ext_kinds`, and the consumed-or-fail walk (`anira::capi::ext_check_consumed`, used by the translator and prepare) with the `entry` row consumed by the LibTorch and ExecuTorch adapters. Tests: `test_Handles`, `test_ExtRegistry` in `test_abi`; gate 4 covers `config.h`. 79 promised names in `abi/symbols-0.txt`.
- **ABI (unstable):** the per-entry tensor record of section 5, `models[].tensors` in the model file: what one engine's export calls a tensor (`anira_model_config_set_tensor_name`; with a name the entry binds that tensor by name, without one positionally, which is what every 2.x configuration did) and the order in which it holds the tensor's axes (`anira_model_config_set_tensor_layout`: spec axis indices per file position, `ANIRA_AXIS_INSERT` for a unit axis the spec lacks; a layout that moves only unit axes is a view, one that moves another axis is a transpose and refused at prepare in this pre-release). The canonical name is the author's and is unique across inputs and outputs; the anchor is set by that name (`anira_model_config_set_anchor(cfg, canonical)`, `"anchor": "mask_out"` in JSON; `ANIRA_ANCHOR_FIRST_STREAMED` is gone). `src/capi/layout.{h,cpp}` holds the classification and the stable-fill derivation the upgrade uses. 87 promised names.
- **ABI (unstable):** the JSON loaders and writers of section 8 (`src/capi/json.cpp`): `anira_model_config_from_json` / `_from_json_file` (relative model paths resolve against `base_dir`, the file's directory, joined with forward slashes on every platform), `anira_machine_config_from_json`, `anira_contract_from_json` (`{"hard": ...}` or `{"async": ...}` with the dual `budget` / `warmup` encodings and a top-level `edge_cost`), `anira_model_config_to_json` / `anira_machine_config_to_json` in v3 spelling with a fixed key order (a v2 file read and written back is the migration tool), and the version 2 auto-upgrade of section 8.4: either root marks a v2 document, `model_data` rows become models (upper-case engine names accepted on this path only, `CUSTOM` becomes the custom engine `anira.v2.custom`, `model_function` becomes the `entry` extension), the universal `tensor_shape` entry and the `processing_spec` become specs (the axis carrying the per-channel element count is time, else the trailing axis; the channel-count axis is channel; window = the per-channel element count, context = window minus the v2 size, size 0 = static; a per-backend entry that permutes unit axes becomes a `layout` on that backend's rows, one that changes another extent is `ANIRA_ERROR_JSON`), `max_inference_time` / `warm_up` / `blocking_ratio` are held back as a legacy Hard contract for `anira_model_config_take_legacy_contract` (or handed out directly by `anira_contract_from_json`), one `ANIRA_LOG_WARNING` per process, and `ANIRA_SUCCESS_UPGRADED`. Every loader failure is `ANIRA_ERROR_JSON` with the key path and the offending value; unknown keys are stored as extensions and fail prepare by name; the `vulkan` block's `device` index (which the descriptor has no slot for) is kept on the handle. Tests: `test_Json`, `test_JsonUpgrade` in `test_abi`. 86 promised names.
- **C++ (not ABI-stable; requires C++20):** `include/anira/anira.hpp`, the header-only configuration half of the 3.x C++ face: the move-only RAII handles `anira::TensorSpec`, `anira::ModelConfig`, `anira::ContractHandle` (minted from the `anira::Hard` / `anira::Async` aggregates, `anira::Contract` their `std::variant`), `anira::MachineConfig` and `anira::JobOptionsHandle` (from an `anira::JobOptions` aggregate), the enum aliases `anira::DType` / `Engine` / `Provider` / `Domain` / `SyncKind` / `Role` / `AxisTag`, and `anira::Result<T>` declared for the exception-free mode of a later pre-release. Every method is exactly one C call of `anira/abi/config.h` (`ModelConfig::tensor_layout` is `anira_model_config_set_tensor_layout`) and returns the handle for chaining; a failed call throws `anira::Error` (`std::runtime_error` with the `anira_status` in `.status` and the `anira_error` message, or the entry's name, in `what()`); every handle exposes `native()` for the C entries the builders do not wrap. `anira::ext::Entry{name}` is the typed form of the `entry` extension (`ext()` / `model_ext()` on the handles, `ext_json()` their JSON twin; `detail::ExtTraits` maps further kinds). The loaders `ModelConfig::from_json(text, base_dir)` / `from_file(path)`, `MachineConfig::from_json` / `from_file`, `ContractHandle::from_json` / `from_file` and the writers `to_json()` (a `std::string`) wrap section 8; a 2.x document is reported by `upgraded()` and `ModelConfig::take_legacy_contract()` returns the held-back Hard contract as `std::optional<ContractHandle>`. The ABI is checked once per process (`anira_check_abi(ANIRA_ABI_VERSION)`) by the first handle created; the header includes no anira/system, anira/utils or third-party header and rejects `ANIRA_CXX_NO_EXCEPTIONS`, `ANIRA_CXX_MANUAL_INIT` and `ANIRA_NO_PROTOTYPES` until the handler half lands. Gates: `anira_header_cxx20` (the header alone under `-std=c++20 -Wall -Wextra -Werror -pedantic` with no anira define) and `anira_header_cxx20_with_v2` (the 2.x umbrella `<anira/anira.h>` and `anira.hpp` in one TU, both orders), and the gtest suite `test_anira_hpp`. Documented deviations from section 6 of the architecture document: no `anira::JsonConfigLoader` yet (the loaders above take its place), `take_legacy_contract()` returns a `ContractHandle` rather than a `Hard` aggregate, `MachineConfig::log_sink` takes the raw `(anira_log_fn, void*)` pair, and `ModelConfig::anchor` takes the tensor's canonical name. `docs/sphinx/usage.rst` section 1 now spells every example with these builders (the C entries are its section 1.6), and the `migration.rst` table carries the builder beside each C call.
- **ABI (unstable):** `anira_contract_hard_set_ring_dtype(contract, canonical, dtype)`, the ring dtype of one tensor under a Hard contract: the element type of the host's samples for that tensor, by canonical name, which its ring holds as is (the Hard entries copy without conversion; the pre- and post-processor convert between the ring dtype and the spec's dtype on the inference thread); per tensor, so an input and an output may differ; float32 for every tensor never set. With `"ring_dtypes": {"audio_in": "int16"}` in the hard block of the contract file and `anira::ContractHandle::hard_ring_dtype`. Data only in this pre-release: the bridge to the 2.x runtime accepts float32 alone; the typed Hard entries arrive with the 3.x runtime. The architecture document's `anira_contract_hard_set_stream_dtype` is this entry under its final name. 88 promised names.
- **Transitional:** the bridge from the 3.x configuration to the 2.x runtime, `anira/compat/v3_to_v2.h` (`namespace anira::v3compat`; v3.0.0-alpha.1 only, removed with the 3.x handler): `to_inference_config(model, contract, candidates)`, `to_context_config(machine)`, `to_host_config(contract, model)` and `to_host_config(model, buffer_size, sample_rate, allow_smaller)`, each over the C handles (a status and an `anira_error`, `noexcept`) and over the `anira.hpp` handles (the 2.x object, `anira::Error` on failure; the model config by lvalue only, since a bytes entry is borrowed), plus `enabled_engines()` (the candidate list that lets one config serve every build; `ANIRA_ENGINE_NONE` keeps the custom-engine entries). Behind them, `src/capi/translate.cpp`: the section-2 validator (every rule `ANIRA_ERROR_CONFIG` naming the tensor or the entry; what the 2.x runtime cannot do `ANIRA_ERROR_NOT_SUPPORTED`: an Async contract, a `MEASURED` budget, `UNTIL_STABLE` warmup, a miss policy other than `BYPASS`, a dtype other than float32, a transposing layout, a dynamic Time extent on a Buffer tensor, an engine not in the build, a custom engine other than `anira.v2.custom`) and the mapping (entries to `ModelData` with the `entry` extension as the model function; the specs to one universal `TensorShape` plus one backend-qualified `TensorShape` per entry with a layout; Channel extents, hops and output latencies to the `ProcessingSpec`; the contract scalars to `max_inference_time` / `warm_up` / `blocking_ratio`; state and `max_instances`; the geometry and the anchor to `HostConfig`; the machine scalars to `ContextConfig`; a flexible window pinned from the geometry). `docs/sphinx/migration.rst` gains "The bridge to the 2.x runtime"; the usage guide's runtime note shows the three calls. Tests: `test_Translate`, `test_Bridge` in `test_abi`; `anira_header_cxx20_with_v2` compiles the bridge header in both include orders; `test/support/inference_config_eq.h` holds the field-by-field comparison the JSON loader test used.
- **Errors and logging:** the strategy of `docs/anira-v3-error-and-log-strategy.md`, documented for users in the new `docs/sphinx/logging.rst` ("Errors and logging") and in a troubleshooting section on reading anira's log per platform. A failure a call can return is returned (status + `anira_error`, `anira::Error` in C++) and not logged; a failure with no caller (anira-owned threads, `destroy` and void entries, sinks) is logged once at Error; real-time refusals do both. The C firewall: every C entry is `noexcept` (`ANIRA_NOEXCEPT` in the generated headers; an escape is a deterministic `std::terminate` instead of undefined behaviour), every failure names its entry, `ANIRA_ERROR_INTERNAL` is logged exactly once (the non-fatal CHECK), a failure swallowed by a `destroy` entry is logged once instead of vanishing, `anira_log` and `anira_drain_log` never recurse into the logger, and **a failing main-thread entry drains the real-time log queue on the caller's thread before returning** so the records that preceded the failure reach the sink first. `ANIRA_LOG_FLAG_TRACE_FAILURES` (registry, machine config) turns every failed status into one Error record as well; `anira::capi::set_trace_failures` is the process-wide switch until the 3.x runtime applies the flag. New `anira.capi` log group. `src/utils/StatusError.h`: `anira::StatusError` (a status plus a message, deriving `std::runtime_error`) is the one exception anira's own control paths throw; the backends throw it with `ANIRA_ERROR_MODEL_LOAD` / `ANIRA_ERROR_ENGINE` and the message `"<engine>: <path|memory>: <engine text>"` instead of a bare `std::runtime_error` the firewall could only classify as `INTERNAL`; `src/utils/ModelFile.h` checks every model path before an engine sees it and returns `ANIRA_ERROR_NO_SUCH_FILE` with the absolute path and the engine name on every backend. Tests: `AbiFirewallLogging` in `test_abi`, `MissingModelFileIsNoSuchFileOnEveryBackend`, `UnloadableModelIsModelLoadWithTheEngineText`, `RejectionsCarryTheDetailAndLogNothing`.

### Changed

- The 3.x line lives on the `v3` branch: the workflows' `pull_request` filters gate `v3` like `main` (same ten required contexts), the project version comes from the `v3*` tags only (`tanh_git_version(... MATCH "v3*")`, so a checkout without a reachable v3 tag configures as `0.0.0+g<hash>`), the release workflow marks hyphenated tags as pre-releases, and the release artifacts carry the tag's version (`anira-3.0.0-alpha.1-<leg>`) while the install tree keeps the digits-only `project()` version.
- tanh-tooling v0.2.7 (pre-release aware `tanh_git_version`: `TANH_VERSION_PRERELEASE`, `TANH_VERSION_DISTANCE`, `MATCH`) pinned together with the tanh-lib commit that carries it (tanh-lab/tanh-lib#34); the drift check is the `tooling-config` job of `lint.yml`.
- tanh-tooling v0.2.8 (library-neutral export-selector comment) and tanh-lib v0.2.0 (self-contained `TANH_API` header, `tanh/core/ExportMacros.h` a deprecated forwarding shim) pinned together; anira's own export header no longer depends on tanh-lib.
- The export allowlist admits the C entry points of the 3.x line (`tanh_set_export_allowlist(anira NAMESPACE anira SYMBOL "anira_*")`, `anira_exports` `ALLOW_REGEX "^_?anira_[a-z0-9_]+$"`); the plugin-shaped unload module's entry points are renamed `anira_test_*` -> `unloadtest_*` so the static-leg leak scan cannot mistake them for library exports.
- WebAssembly: `_anira_drain_log` is defined by the library (`src/capi/log.cpp`) and exported by name from `cmake/build-wasm.cmake`; the wrapper's copy is gone. `web/src/helpers.ts` keeps calling it.
- Docs: the guides describe the 3.x configuration API only. `docs/sphinx/usage.rst` section 1 is rewritten on the handles of `anira/abi/config.h` (tensor specs, model configuration, contracts, machine configuration, JSON files); everything about the 2.x classes and the JSON auto-upgrade moved to the new `docs/sphinx/migration.rst` (the 2.x to 3.x mapping tables, the 2.x document, `take_legacy_contract`, the write-back tool, the 2.x `JsonConfigLoader`), which the guides point to from short notes. The runtime sections still take the 2.x classes in this pre-release and say so.
- anira sets tanh-lib's new `TANH_LOG_COMPILED_MAX_LEVEL` option to 4 for its private copy in every build type: the runtime level is the only filter, and Warning and Info records exist in Release builds (before, a Release anira compiled everything below Error out). tanh-lib re-pinned to v0.3.0 (tanh-lab/tanh-lib#37 and #38: the configurable platform identity, per-record flags and the drop count on the record, which the 3.x runtime's log projection uses, and the option). anira's private logger now files its records under anira's own identity: the Android logcat tag and the Apple `os_log` subsystem and category are `anira` (before, tanh-lib's `thl` / `thl` / `logger`), so `adb logcat -s anira` and `log stream --predicate 'subsystem == "anira"'` find them.
- The version 2 upgrade writes the 2.x constructor's defaults for the keys a file leaves out: a missing `warm_up` is `warmup {"fixed": 0}` and a missing `num_parallel_processors` is `max_instances` = half the hardware threads (before, the handle defaults `UNTIL_STABLE` and 1 applied, which the bridge refuses and which would have run one processor where 2.x ran several).
- `InferenceConfig` and the backends no longer log a failure they also throw: the 16 log-then-throw sites moved the detail the log line carried (the offending value, the backend, the dimension and its index, the entry counts) into the thrown message. `std::invalid_argument` stays the exception type of the 2.x configuration checks.
- A warm-up inference that fails now fails construction on every engine (`ANIRA_ERROR_ENGINE`); the ONNX Runtime adapter used to log and construct the session anyway. The TensorFlow Lite adapter checks the result of model creation, interpreter creation, tensor allocation and resizing, and of every invoke; before, a failure went unnoticed and the output stayed stale.

- **Breaking (bundled models and examples):** the 2.x fixture headers with their `anira::InferenceConfig` statics (`CNNConfig.h`'s `cnn_config`, `Medium_CNNConfig.h`, `Small_CNNConfig.h`, `HybridNNConfig.h`'s `hybridnn_config`, `StatefulRNNConfig.h`'s `rnn_config`, `SimpleGainConfig.h`, `SimpleStereoGainConfig.h`, `RaveFunkDrumConfig.h` and its encoder and decoder headers) are replaced by configuration files next to each model directory, `extras/models/**/*.model.json` and `*.contract.json` (the CNN in three sizes, GuitarLSTM, the stateful LSTM, SimpleGainNetwork mono and stereo, RAVE funk drum whole and as encoder and decoder), named in `extras/models/model_files.h`. Every example, benchmark and test loads a model the same way: `ModelConfig::from_file`, `ContractHandle::from_file`, `to_inference_config(model, contract, enabled_engines())`; the geometry goes through `hard_geometry` + `to_host_config`; a machine configuration only where it differs from the default. No file sets an instance ceiling, so one processor per engine runs (the 2.x statics ran half the hardware threads). `CNNConfig.h`, `HybridNNConfig.h` and `StatefulRNNConfig.h` keep one builder each (`cnn_model_config(hop, size)`, `hybridnn_model_config(batches)`, `rnn_model_config(chunk)`) for the benchmark sweeps, which vary the shapes with the host buffer; a test keeps each builder equal to its file. `ANIRA_EXTRAS_MODELS_DIR` is the one compile definition of the tests and examples; the per-model CMake variables (`GUITARLSTM_MODELS_PATH_*`, `STEERABLENAFX_MODELS_PATH_*`, `STATEFULLSTM_MODELS_PATH_*`, `SIMPLEGAIN_MODEL_PATH`, `RAVE_MODEL_DIR`, `*_JSON_CONFIG_PATH`) and the generated 2.x documents (`SimpleGainConfig.json`, `RaveFunkDrumConfig*.json`) are gone; the 2.x loader and upgrade tests build their documents in memory (`test/support/v2_documents.h`).
- The JUCE example runs variants 0 to 7 (`MODEL_TO_USE`, now a cache variable): variant 1 compiles the CNN's model file, contract file and four exports into the plugin (`ModelConfig::from_json` on the embedded text, `set_model_bytes` with `ANIRA_BYTES_BORROW` per entry by engine), variant 7's RAVE decoder anchors on its audio output (`"anchor": "audio_out"`) and prepares with the host's block and rate like every other model, and the 2.x-file variant 8 is gone (the migration page carries that example). The CLAP example, the three benchmarks and the minimal-inference examples follow the same three lines; the benchmarks set their log level through `to_context_config(MachineConfig{}.log_level(ANIRA_LOG_ERROR))`.
- Docs: `examples.rst` states the one way the examples load a model and describes the JUCE variants; `benchmarking.rst` shows the sweep builders; `migration.rst` lists what replaced the fixture headers and carries the 2.x-file example.

### Fixed

- The context no longer starts tanh-lib's own log drain thread: setting the logger's platform identity at core start went through `thl::Logger::set_config()`, whose default `m_rt_enabled` also starts tanh-lib's drain thread over its default real-time queue, which anira never uses and never stopped. Inside a plugin that thread outlived every session and kept the module mapped after the host's `FreeLibrary` (the `LibraryUnload` tests on the Windows static release legs). The core passes `m_rt_enabled = false`; `Logger.TheCoreNeverStartsTanhLibsOwnDrainThread` pins it.
- `test/abi`: the consumer-shaped compile gates now really compile without the test model-path defines (a directory-level clear; the target-level clear never applied to a parent `add_compile_definitions`), and the "2.x umbrella first" coexistence order is actually compiled (clang-format had regrouped the includes so both drivers compiled the same order).
- A failing inference now delivers zeros for that task on every engine, never the previous job's output (ONNX Runtime copied stale outputs after a caught exception; LiteRT and ExecuTorch left the buffer untouched; TFLite ignored the status).
- LibTorch: a `c10::Error` thrown by `forward` on the inference thread escaped the instance and left it marked busy forever (the pool loop then spun on the remaining instances). The inference now catches per task and the busy flag is released on every exit path.

### Removed

- `test/utils/test_WavReader.cpp`: the test of tanh-lib's `thl::core::read_wav` against the GuitarLSTM wav fixture; tanh-lib carries its own (`test/core/test_WavReader.cpp`), and anira's round-trip tests read the same files through it anyway.

## [v2.3.0] - 2026-09-02

anira v2.3.0 is the last release of the 2.x line. Development continues on the v3 branch (the versioned C ABI, `docs/anira-v3-architecture.md`); 2.x receives fixes only.

### Added

- One-sided streaming (#98, #99, #110): generators are driven by output demand, analysers read through `PrePostProcessor::get_output()`, and the `HostConfig` reference stream may be an input or an output (`m_tensor_is_input`, `HostConfig::k_first_streamable`). Mirrored in the WebAssembly/TS wrapper.
- ExecuTorch backend (`ANIRA_WITH_EXECUTORCH`, static-only): `.pte` programs, named entry points (`model_function`), prebuilt libraries for desktop, Android and iOS, a `minimal-executorch` example.
- Receptive-field (sliding-window) models in the default `PrePostProcessor`: an input tensor larger than its `preprocess_input_size` is filled with ring-buffer history plus the fresh hop.
- `Context` is immortal and owns the inference thread pool exactly while sessions exist (#104); `Context::shutdown()` for hosts that unload live, `release_core_if_idle()`/`has_core()` reclaim the core.
- `LatencyCalculator` (#66) replaces `SessionElement`'s latency arithmetic: closed forms instead of the LCM loop, `is_feasible()`, correct struct and ring sizing for fractional and smaller host blocks. Derivation in `docs/sphinx/latency.rst`.
- Symbol policy from the shared tanh-tooling CMake modules (`cmake/tanh/`): hidden visibility everywhere, `libanira` exports namespace `anira` only, plugins leak no backend runtime symbol; the `anira_exports` CTest enforces it.
- CI on the shared `tanh-lab/ci-actions` workflows: every backend and both linkages in the PR tier, install, sanitizer (ASan+UBSan, TSan, RTSan) and coverage legs, a preset for every leg in `CMakePresets.json`, and the npm publish gated behind the same tests as every other release artifact.

### Changed

- **Breaking:** the latency calculation moves out of `SessionElement` into `LatencyCalculator`; `SessionElement` loses its `calculate_*`, `max_num_inferences` and gcd/lcm helpers. Reported latencies are unchanged for whole-sample host blocks; slot counts follow `2·S`.
- **Breaking:** backend linkage follows `BUILD_SHARED_LIBS` and `ANIRA_<ENGINE>_LINKAGE` is gone: LibTorch (shared-only) is disabled in static builds, ExecuTorch (static-only) in shared builds.
- **Breaking:** no public header includes an engine header, and the engines are explicit imported targets that anira links `PRIVATE` (`anira::onnxruntime`, `anira::tflite`, `anira::litert`, `anira::libtorch`, `anira::executorch`). Migration: link `anira::<engine>` wherever your own code includes an engine header.
- **Breaking** (Windows install layout): DLLs install to `bin/`, import libraries and archives to `lib/`.
- **Breaking:** logging goes through tanh-lib's `thl::Logger` with a real-time queue (#56): `ANIRA_LOG_*`/`ANIRA_LOG_RT_*` replace the `LOG_*` stream macros, `ContextConfig::m_log_level` -> `ContextConfig::m_log` (`anira::LogConfig`), and the host owns the sinks.
- **Breaking:** `InferenceHandler::reset()` is wait-free and safe on the audio thread. Migration: call `prepare()` where a drain was relied upon.
- **Breaking** (containers): `anira::Buffer<T>`, `anira::RingBuffer` and `anira::MemoryBlock<T>` are aliases over tanh-lib's `thl::core` containers; `RingBuffer` no longer derives from `Buffer<float>`, and every real-time ring-buffer access uses the block API (#111).
- `Context::get_instance()` takes no arguments and returns a `Context&`; the `ContextConfig` travels with `Context::create_session()` (old two-step API deprecated for one minor release). `anira::HighPriorityThread` is deprecated for one release.
- The active backend defaults to the first model backend that is available instead of the silent `CUSTOM` bypass; `HostConfig`'s default reference tensor is the first streamable tensor.
- The export header is `anira/system/Exports.h`; `ANIRA_BUILDING`/`ANIRA_STATIC` replace `ANIRA_EXPORTS`/`ANIRA_STATIC_DEFINE` (old spellings honoured).
- ExecuTorch is consumed as one merged `libexecutorch.a` per platform, and the CMake >= 3.24 desktop floor is gone. The test suite builds as per-component binaries (`test_utils`, `test_scheduler`, `test_backends`, `test_handler`).
- Pins: tanh-lib `751b2b1`, anira-project/backends v2.4.0 (ExecuTorch 1.3.1, pre-isolated static desktop LiteRT).

### Removed

- `Context::release_instance()` and `Context::release_thread_pool()` (`Context::shutdown()` is the explicit teardown), the never-released `reset_non_blocking()` variants, the unreachable `ContextConfig::operator==`/`!=`, and the dead interpolation branch of `anira::calculate_percentile`.

### Fixed

- Backend runtime symbols are no longer exported from binaries embedding anira, which crashed hosts that ship their own runtime (Ableton Live 12 bundles ONNX Runtime).
- One-sided streaming: `SessionElement::prepare()` no longer hangs for generators or crashes for analysers, push-only pipelines no longer stall after `m_num_structs` chunks (#99), latency vectors are index-aligned with the outputs (#98).
- Session lifecycle: `Context::create_session()` no longer leaks the session count when a processor constructor or `prepare()` throws (#106); `create_session()`/`release_session()`/`prepare_session()` are thread-safe across sessions; a data race in `Context::release_session()` (found by the TSan leg).
- `LatencyCalculator::least_common_multiple` no longer overflows `int`, the `allow_smaller_buffers` sweep no longer stalls for large blocks, and the send ring of a fractional host block reserves `P - 1` samples.
- `JsonConfigLoader` drops malformed `tensor_shape` entries instead of letting `nlohmann::type_error` escape; `LibtorchProcessor` throws `std::runtime_error` on an unloadable model like the other backends; `BackendBase::process` no longer reads past the output vector.
- `USE_ANIRA_WEB` is defined in Emscripten builds only; installing on `lib64` hosts works with LibTorch and ExecuTorch; `anira::HighPriorityThread` links again in a Windows consumer of a shared anira; the JUCE example links under clang on Linux.
- Tests and CI: the macOS test-discovery race with CMake 4.4.0 (`DISCOVERY_MODE PRE_TEST`), the intermittently failing one-sided blocking-deadline tests, downloads retried after transient failures, and compile caching that works for the first time (`ACTIONS_CACHE_SERVICE_V2`).

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
