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

`.clang-format`, `.clang-tidy` and `.clangd` are installed verbatim from a pinned [tanh-tooling](https://github.com/tanh-lab/tanh-tooling) release (`clang/install.sh`), and the `clang_check` CI job fails on any drift from that pin. Never edit these files by hand; style changes go to tanh-tooling. To update: run `install.sh` with the new tag (`TANH_TOOLING_REF=vX.Y.Z`), commit the rewritten files, and bump the `ref` in `.github/workflows/clang_check.yml` in the same commit. See `docs/sphinx/contributing.rst`, "Code Style".
