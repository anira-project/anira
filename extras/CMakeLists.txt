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

# Using CACHE PATH to set the default path for the models, since PARENT_SCOPE is only propagated one level up. Cache variables are global.
set(GUITARLSTM_MODELS_PATH_TENSORFLOW "${ANIRA_EXTRAS_MODELS_DIR}/hybrid-nn/GuitarLSTM/tensorflow-version/models/" CACHE PATH "Path to the GuitarLSTM TensorFlow models")
set(GUITARLSTM_MODELS_PATH_PYTORCH "${ANIRA_EXTRAS_MODELS_DIR}/hybrid-nn/GuitarLSTM/pytorch-version/models/" CACHE PATH "Path to the GuitarLSTM PyTorch models")

set(STEERABLENAFX_MODELS_PATH_TENSORFLOW "${ANIRA_EXTRAS_MODELS_DIR}/cnn/steerable-nafx/models/" CACHE PATH "Path to the SteerableNAFX TensorFlow models")
set(STEERABLENAFX_MODELS_PATH_PYTORCH "${ANIRA_EXTRAS_MODELS_DIR}/cnn/steerable-nafx/models/" CACHE PATH "Path to the SteerableNAFX PyTorch models")

set(STATEFULLSTM_MODELS_PATH_TENSORFLOW "${ANIRA_EXTRAS_MODELS_DIR}/stateful-rnn/stateful-lstm/models/" CACHE PATH "Path to the StatefulLSTM TensorFlow models")
set(STATEFULLSTM_MODELS_PATH_PYTORCH "${ANIRA_EXTRAS_MODELS_DIR}/stateful-rnn/stateful-lstm/models/" CACHE PATH "Path to the StatefulLSTM PyTorch models")

set(SIMPLEGAIN_MODEL_PATH "${ANIRA_EXTRAS_MODELS_DIR}/model-pool/example-models/SimpleGainNetwork/models/" CACHE PATH "Path to the SimpleGainNetwork models")

# RAVE config generation (the model itself was fetched above, LibTorch-only).
if(ANIRA_WITH_LIBTORCH)
    set(RAVE_MODEL_DIR "${ANIRA_EXTRAS_MODELS_DIR}/third-party/ircam-acids/RAVE" CACHE PATH "Path to the RAVE model")
    set(RAVE_FUNK_DRUM_MODEL_PATH "${RAVE_MODEL_DIR}/rave_funk_drum.ts")

    configure_file(
            "${_anira_models_host}/third-party/ircam-acids/RaveFunkDrumConfig.json.in"
            "${_anira_models_host}/third-party/ircam-acids/RaveFunkDrumConfig.json"
    )

    configure_file(
            "${_anira_models_host}/third-party/ircam-acids/RaveFunkDrumConfigDecoder.json.in"
            "${_anira_models_host}/third-party/ircam-acids/RaveFunkDrumConfigDecoder.json"
    )

    configure_file(
            "${_anira_models_host}/third-party/ircam-acids/RaveFunkDrumConfigEncoder.json.in"
            "${_anira_models_host}/third-party/ircam-acids/RaveFunkDrumConfigEncoder.json"
    )

    set(RAVE_MODEL_FUNK_DRUM_JSON_CONFIG_PATH "${ANIRA_EXTRAS_MODELS_DIR}/third-party/ircam-acids/RaveFunkDrumConfig.json" CACHE PATH "Path to the RAVE json model config")
    set(RAVE_MODEL_FUNK_DRUM_DECODER_JSON_CONFIG_PATH "${ANIRA_EXTRAS_MODELS_DIR}/third-party/ircam-acids/RaveFunkDrumConfigDecoder.json" CACHE PATH "Path to the RAVE json model config")
    set(RAVE_MODEL_FUNK_DRUM_ENCODER_JSON_CONFIG_PATH "${ANIRA_EXTRAS_MODELS_DIR}/third-party/ircam-acids/RaveFunkDrumConfigEncoder.json" CACHE PATH "Path to the RAVE json model config")
endif()

# Generated on the host; @SIMPLEGAIN_MODEL_PATH@ inside resolves to the runtime
# root, so the baked model_path is correct after the tree is staged on the target.
configure_file(
        "${_anira_models_host}/model-pool/SimpleGainConfig.json.in"
        "${_anira_models_host}/model-pool/SimpleGainConfig.json"
)

set(SIMPLE_GAIN_JSON_CONFIG_PATH "${ANIRA_EXTRAS_MODELS_DIR}/model-pool/SimpleGainConfig.json" CACHE PATH "Path to the SimpleGain json model config")
