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
