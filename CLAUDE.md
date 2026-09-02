# CLAUDE.md

## Documentation & changelog policy

Whenever a feature is added or removed, or the API changes, update **both**:

1. **The documentation**
   - Doxygen comments in the public headers (`include/anira/**`) are the API-reference source — keep them in sync with signature and behavior changes.
   - The user guides in `docs/sphinx/*.rst` (e.g. `usage.rst`, `benchmarking.rst`) — update any prose and code examples that show the affected API.
2. **`CHANGELOG.md`**
   - Add an entry under `## [Unreleased]` in the matching `### Added` / `### Changed` / `### Fixed` / `### Removed` section (Keep a Changelog format).
   - Prefix breaking changes with `**Breaking:**` and describe the migration.

This applies to library code, the benchmark fixture, the bundled examples/benchmarks, and notable CI/build-system changes.

## Shared tooling configs

`.clang-format`, `.clang-tidy`, `.clangd` and every file under `cmake/tanh/` (platform axes, symbol policy, export check, git version, sanitizers, test deps, Apple defaults, CPack, RPATH) are installed verbatim from a pinned [tanh-tooling](https://github.com/tanh-lab/tanh-tooling) release (`install.sh` families `clang cmake`), and the `tooling-config` CI job (`lint.yml`, merge queue only) fails on any drift from that pin or on a foreign file in `cmake/tanh/`. Never edit these files by hand; changes go to tanh-tooling. To update: `curl -fsSL https://raw.githubusercontent.com/tanh-lab/tanh-tooling/vX.Y.Z/install.sh | sh -s -- clang cmake`, commit the rewritten files, and bump the workflow version and `ref` in `.github/workflows/lint.yml` in the same commit. The fetched tanh-lib carries its own copy of the same modules: **anira and the pinned tanh-lib must pin the same tanh-tooling tag** (`modules-version.cmake` warns otherwise). In CMake, branch on `TANH_OPERATING_SYSTEM` / `TANH_BINARY_FORMAT`, never on `APPLE`/`UNIX`/`WIN32`/`EMSDK_VERSION`; the export header is `anira/system/Exports.h` (a stub over tanh-lib's `tanh/core/ExportMacros.h`). See `docs/sphinx/contributing.rst`, "Code Style".
