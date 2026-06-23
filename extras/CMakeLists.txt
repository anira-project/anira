set(MODEL_REPOSITORIES
        "https://github.com/faressc/GuitarLSTM.git hybrid-nn/GuitarLSTM"
        "https://github.com/faressc/steerable-nafx.git cnn/steerable-nafx"
        "https://github.com/vackva/stateful-lstm.git stateful-rnn/stateful-lstm"
        "https://github.com/anira-project/example-models.git model-pool/example-models"
)

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git not found")
endif()

foreach(repo IN LISTS MODEL_REPOSITORIES)
    string(REPLACE " " ";" SPLIT_REPO_DETAILS ${repo})
    list(GET SPLIT_REPO_DETAILS 0 REPO_URL)
    list(GET SPLIT_REPO_DETAILS 1 INSTALL_PATH)

    set(GIT_CLONE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/${INSTALL_PATH}")

    if(NOT EXISTS "${GIT_CLONE_DIR}")
        message(STATUS "Cloning ${REPO_URL} (main branch, latest state) into ${GIT_CLONE_DIR}")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --branch main --single-branch --depth 1 ${REPO_URL} ${GIT_CLONE_DIR}
            RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "Git clone of ${REPO_URL} failed with ${GIT_CLONE_RESULT}")
        endif()
    endif()
endforeach()

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

# RAVE is a TorchScript (.ts) model exercised only by the LibTorch backend, which
# has no mobile build — so the (large) download and its configs are skipped unless
# LibTorch is enabled. The RAVE-specific tests are #ifdef USE_LIBTORCH-gated too.
if(ANIRA_WITH_LIBTORCH)
    set(RAVE_FUNK_DRUM_MODEL_URL "https://github.com/anira-project/example-models/raw/refs/heads/third-party/third-party/ircam-acids/RAVE/rave_funk_drum.ts")
    set(RAVE_MODEL_DIR "${ANIRA_EXTRAS_MODELS_DIR}/third-party/ircam-acids/RAVE" CACHE PATH "Path to the RAVE model")
    set(RAVE_FUNK_DRUM_MODEL_PATH "${RAVE_MODEL_DIR}/rave_funk_drum.ts")

    file(MAKE_DIRECTORY "${_anira_models_host}/third-party/ircam-acids/RAVE")

    if(NOT EXISTS "${_anira_models_host}/third-party/ircam-acids/RAVE/rave_funk_drum.ts")
        message(STATUS "Downloading RAVE model from ${RAVE_FUNK_DRUM_MODEL_URL}")
        file(DOWNLOAD
        "${RAVE_FUNK_DRUM_MODEL_URL}"
        "${_anira_models_host}/third-party/ircam-acids/RAVE/rave_funk_drum.ts"
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
        LOG DOWNLOAD_LOG
        )
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
        if(NOT DOWNLOAD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to download RAVE model: ${DOWNLOAD_LOG}")
        endif()
    endif()

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
