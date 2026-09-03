# The model fixtures come from pinned commits of the example-model repositories,
# fetched by fetch-models.cmake (shared with the CI cache-seed job — see the pins
# and the refetch/override story there). RAVE is a LibTorch-only fixture, so its
# (large) download is requested only when that backend is enabled; the
# RAVE-specific tests are #ifdef USE_LIBTORCH-gated too.
set(ANIRA_MODELS_FETCH_RAVE ${ANIRA_WITH_LIBTORCH})
include("${CMAKE_CURRENT_SOURCE_DIR}/fetch-models.cmake")

# Physical location of the model tree on the build host: where the repos are
# cloned (above) and the JSON configs are generated (below).
set(_anira_models_host "${CMAKE_CURRENT_SOURCE_DIR}/models")

# Root the model paths baked into the test/example compile definitions (and into
# the generated JSON configs) resolve to at *runtime*. Defaults to the host tree,
# but can be overridden to stage the models elsewhere — e.g. set it to an on-device
# path like /data/local/tmp/anira/models for an Android emulator run and push
# ${_anira_models_host} there. Kept separate from _anira_models_host so the build
# still clones/writes on the host while the baked paths point at the run target.
set(ANIRA_EXTRAS_MODELS_DIR "${_anira_models_host}" CACHE PATH
    "Root the test/example model paths resolve to at runtime (override to relocate models, e.g. onto an Android device)")

# The model tree is addressed from that one root: the configuration files of the bundled
# models (extras/models/**/*.model.json, *.contract.json; named in extras/models/model_files.h)
# carry their model paths relative to themselves, and the tests and examples spell any other
# path as ANIRA_EXTRAS_MODELS_DIR "/<relative>".
