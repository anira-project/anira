set(MODEL_REPOSITORIES
        "https://github.com/faressc/GuitarLSTM.git stateless-rnn/GuitarLSTM"
        "https://github.com/faressc/steerable-nafx.git cnn/steerable-nafx"
        "https://github.com/vackva/stateful-lstm.git stateful-rnn/stateful-lstm"
)

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git not found")
endif()

foreach(repo IN LISTS MODEL_REPOSITORIES)
    string(REPLACE " " ";" SPLIT_REPO_DETAILS ${repo})
    list(GET SPLIT_REPO_DETAILS 0 REPO_URL)
    list(GET SPLIT_REPO_DETAILS 1 INSTALL_PATH)

    set(GIT_CLONE_DIR "${CMAKE_SOURCE_DIR}/extras/models/${INSTALL_PATH}")

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

set(GUITARLSTM_MODELS_PATH_TENSORFLOW "${CMAKE_SOURCE_DIR}/extras/models/stateless-rnn/GuitarLSTM/tensorflow-version/models/" PARENT_SCOPE)
set(GUITARLSTM_MODELS_PATH_PYTORCH "${CMAKE_SOURCE_DIR}/extras/models/stateless-rnn/GuitarLSTM/pytorch-version/models/" PARENT_SCOPE)

set(STEERABLENAFX_MODELS_PATH_TENSORFLOW "${CMAKE_SOURCE_DIR}/extras/models/cnn/steerable-nafx/models/" PARENT_SCOPE)
set(STEERABLENAFX_MODELS_PATH_PYTORCH "${CMAKE_SOURCE_DIR}/extras/models/cnn/steerable-nafx/models/" PARENT_SCOPE)

set(STATEFULLSTM_MODELS_PATH_TENSORFLOW "${CMAKE_SOURCE_DIR}/extras/models/stateful-rnn/stateful-lstm/models/" PARENT_SCOPE)
set(STATEFULLSTM_MODELS_PATH_PYTORCH "${CMAKE_SOURCE_DIR}/extras/models/stateful-rnn/stateful-lstm/models/" PARENT_SCOPE)