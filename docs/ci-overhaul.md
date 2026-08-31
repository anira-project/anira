# CI overhaul — implementation plan

**Status**: draft for review · **Owner**: Valentin (M0 items 3+4), upstream coordination with Fares · **Date**: 2026-08-31
**Scope**: M0 item 3 (CI on `tanh-lab/ci-actions`) and M0 item 4 (test reorganisation) of the anira v3 architecture roadmap, plus runner-efficiency work on top. Both items touch `main` only and merge into `v3` from there; the `v2.3.0` tag waits for neither.

> The roadmap text this plan implements is the current v3 architecture draft (section 11, M0 items 3 and 4). The copy at `docs/anira-v3-architecture.md` is an older revision — where they differ (e.g. the old draft's `test/fixtures/` directory, since dropped), the newer draft governs: WAV fixtures come from `thl::core::read_wav`, model paths stay compile definitions from `extras/CMakeLists.txt`.

> **Beyond-mandate items needing explicit sign-off** (marked ⚠ where they appear): PR-tier coverage reduction (§3.6 — in tension with the roadmap's "merges with the full test suite green" rule), the model-repo pinning (§3.5 — changes configure behaviour for every consumer, not just CIs; ships with the PR that carries this document), the on_tag toolchain switch for release artifacts (§3.2), and the nightly cron / merge queue (§3.6). Everything else is either the letter of items 3–4 or pure mechanics.
>
> **Status**: steps 0–4 are merged — #129 (steps 0+1, incl. the resolved 0c: sccache spoke the retired v1 cache API without `ACTIONS_CACHE_SERVICE_V2=on`, every write failed read-only), #130 (diff-based clang-tidy), #131 (step 2, the preset catalog), ci-actions tagged `v0.1.0`→`v0.2.1` with tanh-lib pinned (tanh-lib#23). Step 4 (#132): build_test on the shared actions in preset mode, the PR/full tier split, MSVC under Ninja+vcvars (incl. native Windows-arm64), the gcc and clang-cl coverage legs, and the `build_test result` aggregation job — PR tier and full 23-leg tier validated green. Steps 5–7 follow; all open questions are resolved (§6), incl. the macOS consolidation (§3.6).

---

## 1. Goals and constraints

**Goals**, in order:

1. **M0 item 3**: replace anira's local `setup`/`build`/`test` composites (`.github/actions/`) with the shared `tanh-lab/ci-actions` actions in preset mode; per-platform presets (sanitizers, Android, iOS, Windows-arm64, macOS-universal, shared/static × backend sets); one explicit gcc job; `GITHUB_TOKEN` at job level everywhere; mobile test actions gain inputs for anira's extra pushes *or* anira keeps its mobile workflow (decided: the former — one shared mobile runner, §3.7); the install/codesign/release action stays anira's; ci-actions pinned to a tag like tanh-tooling.
2. **M0 item 4**: the single `tests` binary becomes `test_utils`, `test_scheduler`, `test_backends`, `test_handler`. No behaviour change; equal test counts before and after are the acceptance.
3. **Runner efficiency**: anira is open source and runs on free GitHub-hosted runners — the constraint is not minutes (free for public repos) but **concurrency**, and the job is to use the free pool well.

**Constraints**:

- **Free-runner concurrency caps** (docs.github.com/actions/reference/limits): 20 concurrent jobs total on Free / 40 Pro / 60 Team — and **5 concurrent macOS jobs on every non-Enterprise plan**. Larger runners are *always* billed, even for public repos. The design below assumes the worst case (20/5); check the anira-project org plan to confirm (§6 Q1).
- **Windows ships MSVC-first.** MSVC (cl.exe) remains the primary Windows toolchain across the full matrix; clang on Windows is added as secondary coverage and must be **clang-cl**, because `cmake/msvc-support.cmake` (DLL staging for tests/examples) and the `/Gy`, `/OPT:REF` branches are gated on `if(MSVC)` — true for cl and clang-cl, false for GNU-driver clang, under which shared-build test binaries would find no DLLs.
- **tanh-tooling rules**: `.clang-*` and `cmake/tanh/` are installed from the pinned tanh-tooling release and never edited by hand; every CMake addition here lives in `anira/cmake/` or `CMakePresets.json`. anira and the fetched tanh-lib pin the same tanh-tooling tag.
- **v3 branching**: M1's first PR adds `v3`/`v3/**` to the `main`-only branch filters. Everything below must treat a push to `v3` exactly like a push to `main` (full tier).
- **Roadmap merge rule**: milestones land as PRs that merge "with the full test suite green" — the PR tier in §3.6 deliberately relaxes *which runs* gate a merge and therefore needs sign-off (and ideally the merge queue) rather than silent adoption.

---

## 2. Measured baseline (2026-08-31)

All numbers from the two most recent `main` push runs (commits merging PR #124 and PR #127, 2026-08-29) via the GitHub API, plus log deep-dives of three build_test jobs. (Exception: build_sanitizer's "previous" figure substitutes the PR #122 run — its PR #124 push run failed at parse time in 0 s, the dangling `env:` key later fixed in c89eee1.)

### 2.1 Wall clock per workflow (latest / previous main run)

| Workflow | Wall (latest) | Wall (previous) | Jobs |
|---|---|---|---|
| build_test | 20:44 | **40:12** | 21 |
| build_test_mobile | 9:58 | 14:53 | 7 |
| build_install | 7:43 | **21:09** | 6 |
| clang_tidy | 8:46 | 8:46 | 1 |
| build_sanitizer | 5:07 | 6:32 | 1 |
| build_web | 4:46 | 4:46 | 1 |
| build_examples (PR-only; May data) | 11:24 | 11:19 | 8 |
| build_benchmark (PR-only; May data) | 12:47 | 13:38 | 4 |

A push to main fires **9 workflows ≈ 40 jobs** (the six push-triggered rows above plus clang_format, clang_check and build_docs_and_deploy). A PR to main fires those *plus* build_examples and build_benchmark — ~50 jobs.

### 2.2 The five findings

1. **Queueing dominates bad runs.** The 40:12 build_test run had queue delays of 31:14 / 29:12 / 28:10 / 25:12 / 24:31 on macOS jobs (max job *duration* was 12:10) — **~28 of 40 minutes were pure queueing**. build_install once spent 20:12 queueing a 55-second job. Even the good run queued 4–10 min on 10 of 21 jobs. Cause: ~40 jobs vs a 20-total/5-macOS concurrency cap, with **13 macOS jobs on a push** (8 build_test + 3 iOS + 2 build_install) — and up to **19 on a PR** (plus 4 build_examples, 2 build_benchmark) — contending for 5 slots.
2. **The compile cache is entirely dead.** sccache hit rate on consecutive main pushes: **0/45 on Linux, 0/45 on macOS, and 0 compile requests on Windows**. Windows: `CMAKE_<LANG>_COMPILER_LAUNCHER` is ignored under the Visual Studio generator (the build composite passes no `-G` on Windows), so sccache never runs. Linux/macOS: sccache runs but never hits — root cause to be confirmed (§4 step 0), candidates: per-commit volatile input invalidating every key, or GHA cache-scope/eviction issues.
3. **ctest is serial everywhere.** 307 tests: Linux 145 s, macOS-arm64 196 s, **Windows 421 s**, macOS-x86_64-shared **9:30 under Rosetta** (macOS runners are Apple-Silicon M1, 3 vCPU; x86_64 test binaries run emulated). `ctest --test-dir build --output-on-failure` with no `-j`; no test preset sets `execution.jobs`.
4. **`extras/models` is a 1.7 GB unpinned configure-time clone.** Four repos cloned `--branch main --depth 1` by `execute_process` into the source tree on every job (skipped only if the dir exists — which it never does in CI), with no pin and no integrity check. Configure totals 55–72 s per job. Backend archives, by contrast, are a **non-issue: ≤10 s** download+extract per job (stamp-file logic in `cmake/backends.cmake` would make caching safe, but there is little to win).
5. **Windows long poles beyond build_test**: build_examples Windows build 10:55, build_benchmark Windows 8:30 build + 3:46 test — full cold MSBuild compiles with zero caching. clang_tidy's check step is a reproducible 6:15 (already `xargs -P`-parallel; bounded by TU count).

---

## 3. Target design

### 3.1 Preset catalog

anira's `CMakePresets.json` grows a CI-facing preset set beside the existing developer presets (`desktop-*`, `clang-tidy`, `wasm-*`, `docs`). Existing developer presets stay untouched. Naming follows tanh-lib (`android-*`, `ios-*`, `desktop-debug-{asan,…}`) with the extra axes anira needs.

**Compiler policy per platform** (the roadmap's intent: "the implicit gcc coverage of today's unpinned Linux legs becomes one explicit gcc job, because the desktop-* presets force clang"):

- Hidden `ci-base`: Ninja, Release, `ANIRA_WITH_TESTS=ON`, **no compiler pin** (Windows presets need cl.exe from the environment).
- Hidden `ci-clang-base` (inherits `ci-base`): pins `clang`/`clang++`. **All Linux and macOS CI legs inherit this** — i.e. Linux coverage moves from distro gcc to clang, matching the preset family and tanh-lib. This *is* a compiler change for the Linux legs; it is the roadmap's stated design, and the gcc coverage that would otherwise vanish becomes the one explicit `ci-tests-gcc` job.
- Windows presets inherit plain `ci-base` (cl.exe via vcvars, §3.2); `windows-clang-tests-shared` pins `clang-cl`.

Two structural rules learned from the ci-actions internals:

- **Every preset used with `cmake-test-android` / `cmake-test-ios-simulator` needs its own literal `binaryDir` key** — those actions parse `CMakePresets.json` with a Python snippet that does not resolve `inherits` (KeyError otherwise; upstream U6).
- **Keep the catalog small by letting backend sets ride on top of platform presets.** Upstream change U3 (§3.8) lets `cmake-build` append `CMAKE_BUILD_ARGS` in preset mode; then a leg is (preset × backend flags). If U3 is rejected, the backend-set presets below multiply (~15 → ~40) — the plan works either way (§6 Q3).

| Preset (configure) | Inherits / key settings | Used by |
|---|---|---|
| `ci-tests-shared` / `ci-tests-static` | `ci-clang-base`; `BUILD_SHARED_LIBS` ON/OFF, engines ON | Linux (x64+arm64), macOS-arm64, macOS-x64 (native Intel runner) |
| `ci-tests-noengines` | engines OFF, shared | Linux |
| `ci-tests-tflite-{shared,static}` | ONNX+TFLite, LiteRT/LibTorch OFF | Linux, macOS |
| `ci-tests-gcc` | `ci-base` + `gcc`/`g++` pins | **the one explicit gcc job** |
| `windows-msvc-tests-{shared,static}` (+ `-tflite-{shared,static}`) | `ci-base`, no compiler pin — cl.exe from vcvars | all primary Windows legs, x64 and arm64 |
| `windows-clang-tests-shared` | `ci-base` + `clang-cl` pins | secondary Windows coverage (§3.2) |
| `macos-universal-tests-{shared,static}` | `ci-clang-base` + `CMAKE_OSX_ARCHITECTURES=arm64;x86_64` | macOS universal legs |
| `macos-x64-tests-{shared,static}` | + `CMAKE_OSX_ARCHITECTURES=x86_64` | only as Rosetta fallback after `macos-15-intel` retires (~Aug 2027) |
| `desktop-tests-rtsan` (+ later `-asan`, `-tsan`, `-lsan`) | `ci-clang-base` + `ANIRA_WITH_RTSAN=ON` etc. | build_sanitizer; extended set nightly-only, **engines-OFF scope** (§4 step 6) |
| `android-emulator-tests` | NDK toolchain, ABI x86_64, `ANIRA_EXTRAS_MODELS_DIR=/data/local/tmp/anira/models` | Android emulator job |
| `android-arm64-build` | ABI arm64-v8a, build-only | Android build coverage |
| `ios-simulator-tests` | iphonesimulator sysroot, arm64, static, tests ON | iOS simulator job |
| `ios-device-build` | iphoneos sysroot, static, build-only | iOS device build coverage |

Notes:

- **Windows-arm64**: the roadmap names a `Windows-arm64` preset family (tanh-lib has `windows-arm64-debug` using the VS generator's `architecture` field). This plan deliberately reinterprets it: under Ninja+vcvars the architecture comes from the environment, so the *same* `windows-msvc-tests-*` presets serve x64 and arm64 and no arm64-named preset is needed. Trade-off: `cmake --preset` alone cannot produce a Windows-arm64 configure without a vcvars shell. If local arm64 reproducibility matters, add `windows-arm64-tests-{shared,static}` presets on the tanh-lib pattern as a follow-up — flagged for review, not silently dropped.
- **msan is excluded** (tanh-lib has `desktop-debug-msan`): MSan requires every linked object instrumented; anira's prebuilt LibTorch/ONNX/LiteRT binaries are not, and even the noengines configuration links prebuilt gtest against an uninstrumented libc++ unless a full instrumented-stdlib toolchain is built. Out of scope; revisit if a noengines+instrumented-libc++ job ever justifies its cost.
- Matching **build presets** 1:1, and **test presets** for every testing preset with `output.outputOnFailure: true`, `execution.timeout`, and `execution.jobs: 4` — one value everywhere (a fifth job on the 3-vCPU Apple-Silicon runners is harmless; forking macOS-specific test presets is not worth the catalog growth). The `-C Release` workaround dies with the VS generator (all presets single-config Ninja).

**Parallel-ctest safety** (applies from step 0 on, *not* part of the reorg PR): tests that drive builds on the shared tree get serialized **up front**, not after a flake — `anira_header_isolation` (runs `cmake --build` on the build dir) and `InstallConsumer` (installs the same tree) get `RUN_SERIAL`/`RESOURCE_LOCK(build-tree)` properties at registration. gtest-discovered cases that prove timing-sensitive (`ConcurrentLifecycle`, `UserManagedThread`, semaphore tests) are pinned via a `set_tests_properties(<discovered-name> PROPERTIES RUN_SERIAL ON)` file included after the discovery include (CTest reads post-discovery includes; per-case names exist only there). Inference tests spawning their own pools will oversubscribe 3–4 vCPUs — acceptable for correctness-focused CI, watched for a week.

### 3.2 Windows: MSVC first, clang-cl second

- **Primary (all Windows matrix legs): MSVC cl.exe under Ninja.** The switch from the default Visual Studio generator to Ninja + cl requires the MSVC environment in the job — a vcvars step before `cmake-build`. This is what makes `CMAKE_<LANG>_COMPILER_LAUNCHER=sccache` effective on Windows at all (launchers are ignored by the VS generator; sccache supports cl). `cmake/msvc-support.cmake` already handles the non-VS-generator DLL path explicitly (`msvc-support.cmake:15-19`); no CMake change needed. Expected effect: 3:19–4:14 cold Windows builds approach the Linux ~1:40 warm; build_examples' 10:55 Windows build drops substantially.
  - **x64**: `ilammy/msvc-dev-cmd@v1` with `arch: x64` — well-trodden.
  - **arm64 needs a spike (step 4)**: `msvc-dev-cmd` documents `x64/x86/amd64_arm64` but not a native arm64-host arch; the current setup composite deliberately skips ninja on Windows-arm64 ("native MSVC (default generator) build"). Verify `VsDevCmd.bat -arch=arm64 -host_arch=arm64` plus an arm64 ninja and sccache binary on `windows-11-arm`. **Fallback if the spike fails**: the two arm64 legs keep the VS generator without sccache (status quo for them) while every x64 Windows leg still gets the win.
  - **Cache-safety invariant**: cl compiles with `/Zi` are uncacheable by sccache. Release carries no `/Zi` today, but pin it: set `CMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded` (/Z7) in `ci-base` and record "no `/Zi`, no PCH" next to the sccache config, so a future RelWithDebInfo leg can't silently return the hit rate to 0.
- **Secondary: one `windows-clang-tests-shared` leg** (clang-cl; LLVM already installed by `setup-cpp-build-tools`'s Windows branch; still needs the vcvars step for headers/libs). clang-cl keeps `MSVC=TRUE`, so DLL staging and the symbol policy work; the only known delta is `/wd4251` (guarded `$<COMPILE_LANG_AND_ID:CXX,MSVC>` — clang-cl may emit C4251-style warning spam; acceptable for a coverage leg). Runs in the push tier, not the PR tier: MSVC first.
- ⚠ **Release artifacts**: on_tag's `build_release` currently builds with the VS generator; switching it to Ninja+cl changes the build environment of shipped binaries (same toolset, different generator/driver). That is beyond item 3's letter — treated as its own sign-off item (§6 Q2) with a binary-level comparison (exported symbols via the `anira_exports` machinery, `dumpbin /headers` toolchain stamps) added to the step-5b dry-run acceptance. `windows-latest` now defaults to VS 2026; pin `windows-2022` if toolchain stability for artifacts is preferred.

### 3.3 Workflow shape after migration

`build_test` becomes tanh-lib-`pr-checks`-shaped: matrix rows carry `{name, os, preset, args, tier}`; steps are checkout → `setup-cpp-build-tools` → (vcvars on Windows) → `cmake-build` (PRESET + args) → linkage-assertion → `cmake-test` (PRESET). All at `tanh-lab/ci-actions/...@vX.Y.Z` — pinned.

Details that must not get lost in translation:

- **`GITHUB_TOKEN` moves to job level** in every job that configures anira (the backends integrity check reads it; the shared `cmake-build` sets no step-level token the way the local composite did). `build_test_mobile` and on_tag's iOS/Android release jobs already do this; **on_tag's desktop `build_release` job does not** — it currently rides the build composite's step-level token and gains the job-level env in step 5b.
- **The backend-linkage assertion survives**: it greps `configure.log` for `disabling ANIRA_WITH_EXECUTORCH` / `disabling ANIRA_WITH_LIBTORCH`. The local build composite tees configure output to `configure.log`; upstream `cmake-build` does not → upstream change U2 (hard requirement for the migration).
- **`build_sanitizer` is a full migration, not a pin bump**: it uses all three local composites (`build_sanitizer.yml:52,54,62`) on top of its `setup-cpp-build-tools@main` step. It moves to `cmake-build`/`cmake-test` with `desktop-tests-rtsan` before the composites can be deleted; its redundant step-level `GITHUB_TOKEN` env goes job-level.
- The `-C Release` `CTEST_ARGS` and the `setup` composite's per-OS ninja installs disappear (presets are single-config; `setup-cpp-build-tools` installs ninja).
- `CMAKE_BUILD_PARALLEL_LEVEL` stops being hardcoded per call site (the shared action defaults to 4, matching the 4-vCPU runners).
- Workflows that keep anira-specific steps keep them as plain steps: JUCE apt deps + freetype symlinks (build_examples/build_benchmark), docs apt deps, the install/codesign/release composite (`on_tag` — explicitly stays anira's), emsdk (build_web — untouched by this plan; already preset-based and fast).
- `clang_format`/`clang_tidy` get their `@main` refs bumped to the tag — those two really are pin bumps.
- **Required status checks**: after re-shaping, branch protection must not list per-leg names (push-tier-only legs would hang PRs as permanently-pending). Add one aggregating `build_test result` job (`needs:` the matrix, `if: always()`, fails on any failed/cancelled dep) and make it the only required check from this workflow. Same pattern for mobile/install.

### 3.4 Test reorganisation (M0 item 4)

`test/CMakeLists.txt`'s single `tests` target becomes four, over the existing directory grouping:

| Binary | Sources |
|---|---|
| `test_utils` | `test/utils/*` + root `test_WavReader.cpp` (exercises `thl::core::read_wav`, not the handler) |
| `test_scheduler` | `test/scheduler/*` (7 files) |
| `test_backends` | `test/backends/*` (2 files) |
| `test_handler` | root `test_InferenceHandler`, `test_OneSidedStreaming`, `test_StatefulOrdering`, `test_BackendLinkage` + the model fixtures |

Unchanged: model-path compile definitions from `extras/CMakeLists.txt` (directory-scoped, reach all four binaries), WAV fixtures via header-only `thl::core::read_wav`, and the standalone test groups (`anira_header_isolation`, the gtest-discovered unload suite, the `anira_exports` check, `InstallConsumer`) — the pattern M1's `test/abi/` follows.

Mechanical consequences the PR must carry:

- The mobile/desktop registration branch applies **per binary**: one `add_test` per binary on Android/iOS, `gtest_discover_tests` per binary elsewhere.
- The MSVC DLL-copy custom command (`test/CMakeLists.txt:119-126`) applies per binary.
- `.github/scripts/android_emulator_test.sh` pushes and runs **four binaries** instead of `tests` (loop; per-binary `ANIRA_EXIT` check), and `build_test_mobile.yml`'s iOS step spawns four `.app`s — or both switch to `test_*` auto-discovery, which is exactly what `cmake-test-android` / `cmake-test-ios-simulator` do (`find -name "test_*"` / `test_*.app`). **The `test_*` prefix is load-bearing: the glob does not match `tests`.** This item is therefore a prerequisite for adopting the shared *mobile* actions (step 7) — not for the desktop migration (step 4), which runs ctest via presets and never globs binary names.
- **Strictly mechanical**: no parallelism change rides in this PR (ctest `-j` lands in step 0, on the *current* binary — gtest discovery already registers per-case CTest entries), so the equal-test-count acceptance measures the reorg alone and a new flake is attributable.
- Acceptance: per-leg CTest totals equal before/after (currently **307** on a desktop engines-on leg); mobile runs still green.
- `CHANGELOG.md` entry under `### Changed`; no public API is touched (no Doxygen/Sphinx work beyond a contributing-guide mention if it names the `tests` target).

### 3.5 Caching strategy

Budget: 10 GB per repo, 7-day eviction, branch-scoped restores (PR runs restore main's caches — seed on main pushes). Spend it where measurements say it pays:

| Cache | Size | Key | Verdict |
|---|---|---|---|
| `extras/models` | ~1.6 GB, one entry | hash of the four model-repo pins | **Yes, with the design below** |
| sccache (GHA backend) | budget ~4–6 GB across legs | managed by sccache | **Yes, after fixing the 0% hit rate** (step 0c) |
| backend archives (`modules/`) | 60–200 MB per (OS, arch, linkage) | stamp `.sha256` | **No** — measured ≤10 s per job; safe (stamp semantics) but not worth quota. Revisit only if the backends release grows. |
| gtest/benchmark FetchContent | shallow clones, seconds | — | No. Optionally pin by commit SHA for stability, not speed. |

**Models cache design** (the naive version has three failure modes — cross-OS, thundering herd, restore cost):

- ⚠ **Pin the clones first** (beyond-mandate: changes configure behaviour for every consumer). A shallow clone cannot check out an arbitrary SHA — the implemented mechanism (`extras/fetch-models.cmake`) stages per-repo `git init` + `fetch --depth 1 <url> <sha>` + `checkout FETCH_HEAD` into a `.fetching` dir and renames into place only after success (an interrupted fetch can never leave a half tree the exists-check accepts), strips the `.git` object stores (pinned snapshots, ~half the size), and pins the RAVE download's URL and SHA-256. `ANIRA_MODELS_<NAME>_REF` defaults are `-D`-overridable. Unpinned clones under a cache would freeze silently; pinning is also a reproducibility fix on its own.
- **Seed once, restore everywhere**: a single job on main pushes saves the cache (`actions/cache` with `enableCrossOsArchive: true` — without it a Linux-saved entry is invisible to Windows; `continue-on-error` and skipped on tag refs, since on_tag gates releases on the workflow's conclusion and a tag-scoped entry is restorable by nothing). Every configuring job uses `actions/cache/restore` only, falling back to the pinned fetch on miss. **Cache exactly the five gitignored fetch destinations, never `extras/models/` wholesale** — the tree also holds git-tracked fixture headers and `.json.in` templates a restore must not clobber.
- **Measure in flight**: the restore step shipped unconditionally on all OSes (deviation from the original per-OS gate); the PR's own runs provide the restore-vs-clone numbers per OS, and the step is trivially removable per-OS where it loses (Windows tar extraction is the candidate).
- **Eviction budget**: models (~0.8 GB with `.git` stripped) + sccache (~4–6 GB) + emsdk/npm (~1 GB) fit in 10 GB, but monitor `gh api .../actions/caches` after step 4; if churn appears, sccache gets priority (it saves more minutes).
- **Alternative if this underdelivers** (noted per Valentin, 2026-08-31): publish the fixture trees as release assets on `anira-project/example-models` and download them directly, like the backend archives — one hashed asset, no git at all, and the cache becomes optional. The pinned-SHA fetch is compatible with moving there later; adopt it if cache eviction or restore times prove annoying.

sccache diagnosis (step 0c): re-run the same commit twice via `workflow_dispatch` — hits on the second run mean scope/eviction; still 0% means a volatile input poisoning every key (a generated header or define that changes per commit — `__DATE__`, version stamps). Fix at the source; keep `SCCACHE_GHA_VERSION` stable.

### 3.6 Matrix tiers ⚠

The queueing problem is solved by running fewer jobs per event, not faster jobs. Full coverage never leaves — it moves to where nobody is waiting on it. **This is a deliberate policy change against the roadmap's "merges with the full test suite green" rule and needs sign-off**: either adopt the **merge queue** in the same step (free for org-owned public repos; the `merge_group` event joins the full tier, so merges are still full-suite-gated and PRs stay fast), or accept that a full-tier failure surfaces on main/v3 and by convention blocks the next merge. Recommendation: merge queue.

**Tiering must cover build_examples and build_benchmark too** — they are PR-only today (~12 jobs, 6 of them macOS), so tiering build_test alone would leave PR macOS load at ~10 against the cap of 5.

| Tier | Event | build_test | mobile | install | sanitizer | examples | benchmark |
|---|---|---|---|---|---|---|---|
| **PR** | `pull_request` | 10 legs: Linux-x64 {shared,static}, Linux-arm64 shared, Win-MSVC-x64 {shared,static}, macOS-arm64 {shared,static}, macOS-x64 shared (`macos-15-intel`), Linux tflite shared, noengines | iOS onnx-litert + Android onnx-litert-shared | Linux {shared,static} | RTSan | Linux + Win + macOS-arm64, shared only | Linux + macOS-arm64 |
| **Full** | push to main/v3, `merge_group`, `workflow_dispatch`, `workflow_call` (on_tag), `schedule` | all 21 + gcc + windows-clang | all 7 | all 6 | RTSan (+ extended set on `schedule` only) | all 8 | all 4 |

- **Explicit event→tier mapping, no inference gaps**: `pull_request` → PR tier; *every other event* → full. The `plan` job (emits the include-array via `fromJSON`) implements exactly that switch, and the reusable-workflow path gets an explicit `tier` input so `on_tag` pins `tier: full` rather than relying on event inference — this keeps step 5b's `workflow_dispatch` dry-run a genuinely full run.
- **PR-tier concurrency math, all workflows counted**: macOS = 3 (build_test) + 1 (iOS) + 1 (examples) = **5**; total PR jobs ≈ 10+2+2+1+3+2+4 small = **~24**. At the Free cap (20/5) that is one short wave of overflow; at Team (60/5) macOS alone binds. Expected PR feedback: **~8 min p50, ~12 min p90** warm.
- Push tier keeps 13+ macOS jobs against 5 slots → ~3 waves; **~18 min p50, ~25 min p90** — asynchronous, nobody blocks on it. If more headroom is wanted, `macos-universal-static` and `macos-arm64-tflite-static` add the least coverage per minute.
- **Concurrency groups get the event in the key** (`build_test-${{ github.event_name }}-${{ github.ref }}`): today's `-${{ github.ref }}` groups would let a push to main cancel the nightly full+sanitizer run (or vice versa) under `cancel-in-progress: true`.
- macOS-x86_64 legs move to **`macos-15-intel`** (native x64, 4 vCPU, free, available until ~Aug 2027) — kills the 9:30 Rosetta test step; universal legs stay on Apple Silicon. Note the Intel pool is smaller, so watch its queue behaviour; revisit before the image retires.
- ⚠ **Nightly** (proposed addition, same sign-off as tiering): weekly cron on main running the full tier plus the extended sanitizer presets. Scheduled workflows run on the default branch only and auto-disable after 60 days of inactivity — fine for anira.

**macOS consolidation (decided 2026-08-31)** — the 5-concurrent macOS cap is the scarcest resource, so fewer-but-longer jobs win; this supersedes the macOS rows above and the `macos-15-intel` bullet:

- **Universal-only engine legs**: the four arch-specific macOS engine legs are dropped; `macos-universal-tests-{shared,static}` run the suite twice on one runner — natively (arm64) and as `arch -x86_64 ctest --preset …` (the x86_64 slice under Rosetta). Both slices of the artifact mac users actually get are executed, and the Intel-runner escape hatch (image retires ~Aug 2027) is no longer needed in build_test.
- **One macOS tflite leg** (static — the iOS-relevant linkage) instead of two; Linux/Windows keep both linkages. Precondition: a universal `tensorflowlite_c` archive in the backends release (verify before folding).
- **One iOS job**: the three backend sets run sequentially in a single job (onnx-litert device+sim+tests, tflite sim+tests, no-backend build) — one simulator boot instead of three.
- **One macOS install job**: shared and static installs sequentially.
- Effect: push-tier macOS **13 jobs / ~3 waves → 5 jobs / one wave**; PR tier: **2 macOS jobs** (universal-shared dual-arch + the iOS job), and x86_64 execution returns to PRs via the Rosetta pass.
- Deliberate trade-offs: no native-Intel execution (Rosetta instead), the second tflite linkage only in full runs, per-arch failures point at a step rather than a job. on_tag's release builds still produce the separate x86_64/arm64/universal artifacts.

### 3.7 Mobile

**Decision (Valentin, 2026-08-31): one shared mobile runner.** The mobile test runners live in ci-actions (`cmake-test-android` / `cmake-test-ios-simulator`) and anira and tanh-lib use the same ones — anira does not keep a separate mobile implementation long-term; when the runner needs to grow, it grows upstream and both consumers update. What the shared actions still need for anira (U5/U6, the next tag after the first): extra push paths (backend `.so`s from `modules/`, `libc++_shared.so`, the model tree), a device staging dir matching `ANIRA_EXTRAS_MODELS_DIR`, `LD_LIBRARY_PATH` on the run command, per-binary failure collection (report every broken suite in one run, as anira's loops do today), and an `inherits`-aware preset parser. iOS needs no pushes (simulator shares the host FS) but keeps the `.app` convention. Until those land in a tagged ci-actions, `build_test_mobile` keeps anira's own scripting as a stopgap (updated for the four binaries in step 1); step 7 then replaces it and deletes `.github/scripts/android_emulator_test.sh`.

### 3.8 Changes to ci-actions (ours to make; tanh-lib moves in lockstep)

`tanh-lab/ci-actions` is maintained by this team and freely changeable at any time — the one rule is that `tanh-lab/tanh-lib` consumes every action `@main` (pr-checks.yml), so a breaking change there updates tanh-lib in the same motion. Nothing is tagged in ci-actions today, which M0 item 3 explicitly ends: with the first tag the repo also gains a **`CHANGELOG.md`** (Keep a Changelog, like anira's), and from then on every change carries an entry and consumers pin tags.

**What anira's migration needs before its pin** (step 3, ~0.5 day, no external gate):
- **U2 — keep `configure.log`**: `cmake-build` tees configure output in both modes (anira's linkage assertion depends on it). Backward-compatible.
- **Cut the first tag** (proposal: `v0.1.0`) with the CHANGELOG, and bump tanh-lib's `@main` uses to it — tanh-lib's first pin.

**Worth doing, on their own schedule** (each rides a later tag with its changelog entry):
- **U1 — ctest parallelism**: not needed by anira (test presets carry `execution.jobs`); an optional `CTEST_ARGS` input on `cmake-test` would let tanh-lib parallelise too.
- **U3 — extra configure args in preset mode**: `cmake-build` appends `CMAKE_BUILD_ARGS` after `--preset`. Keeps anira's catalog at ~15 instead of ~40 presets (§6 Q3 is the design choice).
- **U4 — Windows generator fix**: manual mode's Windows branch inherits the sccache-inert VS generator; tanh-lib's own `windows-*` presets have the same dead cache.
- **U5 — mobile inputs** and **U6 — `inherits`-aware preset parser**: required for step 7 (§3.7 — one shared mobile runner for anira and tanh-lib); they ride the next tag after the first.

**Future consolidation — reusable workflows** (raised 2026-08-31: "why does this YAML live in anira at all?"): the *step* layer is fully shared, but GitHub only runs workflows from a repo's own `.github/workflows/`, so each consumer still carries job scaffolding (triggers, matrices, wiring) plus its repo-specific facts (backend matrix, model cache, linkage assertion, JUCE deps, mobile staging). The next consolidation level is `workflow_call` **reusable workflows** in ci-actions — e.g. a shared build-test workflow taking the matrix JSON and pin as inputs, shrinking each consumer's file to triggers + a `uses:` line (the config-check.yml pattern tanh-tooling already uses). Deliberately deferred until the shapes stop moving (post step 7) and tanh-lib/anira converge enough that the shared workflow doesn't become a parameter jungle.

---

## 4. Execution plan — single steps

Dependency graph: **steps 0, 1, 2 are independent and parallelisable**; step 3 blocks 4–6; step 4 needs **2+3** (not 1); step 5 needs 4; step 6 needs 5; step 7 needs **1** + U5/U6 in a tagged ci-actions.

**Step 0 — quick wins on the current workflows** *(anira, Valentin; ~0.5 day + a soak week)*
- a. `CTEST_ARGS: "-C Release -j 4"` in build_test (and `-j 4` in build_sanitizer). Works today: gtest discovery registers per-case CTest entries. Up-front serialization of the build-invoking tests (`anira_header_isolation`, `InstallConsumer` → shared `RESOURCE_LOCK`); the post-discovery `set_tests_properties` include (§3.1) is the documented mechanism for pinning a timing-flaky discovered case and is added reactively on the first flake, not speculatively.
- b. macOS-x86_64 legs → `runs-on: macos-15-intel` (build_test, build_examples, build_benchmark; on_tag untouched until 5b).
- c. sccache diagnosis per §3.5; fix the identified cause. If Linux/macOS hit rates stay ~0 after diagnosis, disable sccache there rather than paying setup overhead for nothing — Windows gets its cache in step 4.
- d. ⚠ Model pinning + cache per §3.5 (pin mechanism spelled there; seed-once/restore-only; `enableCrossOsArchive`; measured restore-vs-clone gate). CHANGELOG + docs for `ANIRA_MODELS_<X>_REF`.
- Acceptance: Windows test step ≤ 2:30, macOS-x64 test ≤ 3:00, configure ≤ 25 s warm on cache-hit legs; no new flakes over the soak week.

**Step 1 — test reorganisation (M0 item 4)** *(anira, Valentin; independent; ~1 day)*
As §3.4, strictly mechanical. Acceptance: per-leg CTest totals equal (307 on desktop engines-on legs); all workflows green including mobile.

**Step 2 — CI preset catalog** *(anira, Valentin; independent; ~1 day)*
As §3.1 (presets carry no test lists — this does not wait for step 1). Pure `CMakePresets.json` addition — no workflow change, zero CI risk, immediately usable locally. Acceptance: each preset configures+builds+tests locally on at least its primary platform.

**Step 3 — ci-actions: U2, CHANGELOG, first tag** *(tanh-lab/ci-actions + tanh-lib, Valentin; blocks 4–6; ~0.5 day)*
ci-actions is ours to change directly (§3.8). Land U2 (backward-compatible), add `CHANGELOG.md` (first entry: U2 and the tagging policy), cut `v0.1.0`, and bump tanh-lib's pr-checks.yml `@main` uses to the tag in the same motion. The remaining items (U1/U3–U6) land whenever needed and ride later tags with their entries.

**Step 4 — migrate `build_test`** *(anira, Valentin; needs 2+3; ~1 day + the arm64 spike)*
§3.3 shape: shared actions `@v0.1.0`, presets, job-level `GITHUB_TOKEN`, vcvars on Windows (x64: `msvc-dev-cmd`; **arm64: spike per §3.2 with VS-generator fallback**), `/Z7` invariant in `ci-base`, new push-tier legs `ci-tests-gcc` + `windows-clang-tests-shared`, tier plan-job with the explicit event mapping and `tier` input, event-keyed concurrency groups, **required-checks rewrite to the single aggregating result job**. Delete nothing yet. Acceptance: one full-tier `workflow_dispatch` run green across all legs; linkage assertions firing; Windows x64 sccache nonzero hits on a re-run; a PR-tier run ≤ 10 min wall.

**Step 5a — migrate build_install, build_examples, build_benchmark, build_docs_and_deploy, build_sanitizer** *(anira, Valentin; needs 4; ~1 day)*
build_sanitizer is a full composite→shared-actions migration (§3.3), not a pin bump. clang_format/clang_tidy pin bumps ride along. Examples/benchmark gain their PR-tier subsets (§3.6). Acceptance: all green on full-tier dispatch; examples/benchmark Windows builds measurably down (Ninja+sccache).

**Step 5b — on_tag** *(anira, Valentin; needs 5a; ~1 day)* ⚠
`build_release` moves to `cmake-build` + job-level token; decide the VS pin and the Ninja-for-artifacts question (§3.2, §6 Q2). Delete `.github/actions/{setup,build,test}`; **keep `.github/actions/install`**. Acceptance: `workflow_dispatch` dry-run with `tier: full` produces artifacts identical in layout to v2.2.x **plus** a binary-level comparison (exports via the `anira_exports` machinery, `dumpbin`/`otool` toolchain stamps) against the previous release.

**Step 6 — tiering for install/mobile + nightly** *(anira, Valentin; needs 5a; ~0.5 day)* ⚠
PR/full tiers for build_install and mobile per §3.6; weekly cron with the extended sanitizer presets — **engines-OFF scope**: asan/tsan/lsan run the noengines configuration (anira's own scheduler/utils/handler code); prebuilt engines are uninstrumented and would drown the jobs in false positives. Engine-on sanitizer coverage is explicitly out of scope. Acceptance: synthetic PR shows ≤ 5 concurrent macOS jobs, ≤ 10 min wall; push to main ~18 min p50 / 25 min p90.

**Step 7 — mobile onto the shared actions** *(ci-actions + anira + tanh-lib; needs 1, and U5/U6 in a tagged ci-actions)*
The decided end state (§3.7): extend `cmake-test-android`/`cmake-test-ios-simulator` with the U5 inputs and the U6 parser fix, tag, point anira's `build_test_mobile` at them, and delete the local emulator/simulator scripting. tanh-lib moves to the same tag in the same motion.

**Step 8 — docs + changelog sweep** *(with each step, not after)*
Each implementing PR carries its CHANGELOG entry (CI/build changes are in-policy) and updates `docs/sphinx/` where it names workflows, the `tests` target, local presets, or the new `ANIRA_MODELS_<X>_REF` variables.

---

## 5. Success metrics

| Metric | Baseline (measured) | Target (p50 / p90) |
|---|---|---|
| PR feedback wall (all PR workflows) | 20–40 min | ≤ 8 / 12 min |
| Push-to-main wall (all workflows) | up to 40 min | ≤ 18 / 25 min |
| Windows build step (build_test leg) | 3:19–4:14 cold, uncached | ≤ 2:00 warm |
| Windows ctest (307 tests) | 7:02 | ≤ 2:30 |
| macOS-x64 ctest | 9:30 (Rosetta) | ≤ 3:00 (native Intel) |
| sccache hit rate (warm, same deps) | 0% | ≥ 70% |
| Configure step | 55–72 s | ≤ 25 s (models cached) |
| Coverage | 21 build_test legs | unchanged + gcc + clang-cl legs (push tier) |

---

## 6. Decisions (all resolved 2026-08-31)

1. **Org plan**: Free — 20 concurrent jobs, 5 macOS. The design already assumed this worst case; PR tiers are sized to ≤20 total jobs across all workflows.
2. **Release-artifact toolchain** (step 5b): Ninja + cl, following `windows-latest`/VS — the configuration CI tests is what ships; the step-5b binary comparison validates the first release.
3. **U3**: implemented (ci-actions v0.2.0+ appends `CMAKE_BUILD_ARGS` in preset mode); the catalog stays at ~14 presets.
4. **Tier policy**: adopted, gated by the **merge queue** — `merge_group` runs the full tier on the merged result before landing (step 6 wires the trigger and branch protection).
5. **Windows-arm64 presets**: the vcvars reinterpretation stands — native arm64 vcvars + Ninja worked first try across the full-tier run.
6. **Extended sanitizers**: nightly-only, noengines scope.
7. **macOS consolidation**: see §3.6 — universal-only dual-arch legs, one iOS job, one tflite leg, one install job.
