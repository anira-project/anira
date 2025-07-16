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
        message(STATUS "Cloning ${REPO_URL} into ${GIT_CLONE_DIR}")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} clone ${REPO_URL} ${GIT_CLONE_DIR}
                RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "Git clone of ${REPO_URL} failed with ${GIT_CLONE_RESULT}")
        endif()
    endif()
endforeach()

# Using CACHE PATH to set the default path for the models, since PARENT_SCOPE is only propagated one level up. Cache variables are global.
set(GUITARLSTM_MODELS_PATH_TENSORFLOW "${CMAKE_CURRENT_LIST_DIR}/models/hybrid-nn/GuitarLSTM/tensorflow-version/models/" CACHE PATH "Path to the GuitarLSTM TensorFlow models")
set(GUITARLSTM_MODELS_PATH_PYTORCH "${CMAKE_CURRENT_LIST_DIR}/models/hybrid-nn/GuitarLSTM/pytorch-version/models/" CACHE PATH "Path to the GuitarLSTM PyTorch models")

set(STEERABLENAFX_MODELS_PATH_TENSORFLOW "${CMAKE_CURRENT_LIST_DIR}/models/cnn/steerable-nafx/models/" CACHE PATH "Path to the SteerableNAFX TensorFlow models")
set(STEERABLENAFX_MODELS_PATH_PYTORCH "${CMAKE_CURRENT_LIST_DIR}/models/cnn/steerable-nafx/models/" CACHE PATH "Path to the SteerableNAFX PyTorch models")

set(STATEFULLSTM_MODELS_PATH_TENSORFLOW "${CMAKE_CURRENT_LIST_DIR}/models/stateful-rnn/stateful-lstm/models/" CACHE PATH "Path to the StatefulLSTM TensorFlow models")
set(STATEFULLSTM_MODELS_PATH_PYTORCH "${CMAKE_CURRENT_LIST_DIR}/models/stateful-rnn/stateful-lstm/models/" CACHE PATH "Path to the StatefulLSTM PyTorch models")

set(SIMPLEGAIN_MODEL_PATH "${CMAKE_CURRENT_LIST_DIR}/models/model-pool/example-models/SimpleGainNetwork/models/" CACHE PATH "Path to the SimpleGainNetwork models")

set(RAVE_MODEL_URL "https://play.forum.ircam.fr/rave-vst-api/get_model/darbouka_onnx")
set(RAVE_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/models/third-party/ircam-acids/RAVE" CACHE PATH "Path to the RAVE model")
set(RAVE_MODEL_DARBOUKA_PATH "${RAVE_MODEL_DIR}/darbouka_onnx.ts")

file(MAKE_DIRECTORY "${RAVE_MODEL_DIR}")

if(NOT EXISTS "${RAVE_MODEL_DARBOUKA_PATH}")
    message(STATUS "Downloading RAVE model from ${RAVE_MODEL_URL} to ${RAVE_MODEL_DARBOUKA_PATH}")
    file(DOWNLOAD
    "${RAVE_MODEL_URL}"
    "${RAVE_MODEL_DARBOUKA_PATH}"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
    LOG DOWNLOAD_LOG
    )
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    if(NOT DOWNLOAD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to download RAVE model: ${DOWNLOAD_LOG}")
    endif()
endif()

set(DARBOUKA_MODEL_ABSOLUTE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/third-party/ircam-acids/RAVE/darbouka_onnx.ts")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/models/third-party/ircam-acids/RaveDarboukaConfig.json.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/models/third-party/ircam-acids/RaveDarboukaConfig.json"
)
set(RAVE_MODEL_DARBOUKA_JSON_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/third-party/ircam-acids/RaveDarboukaConfig.json" CACHE PATH "Path to the RAVE json model config")
