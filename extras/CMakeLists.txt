set(MODEL_REPOSITORIES
        "GuitarLSTM https://github.com/faressc/GuitarLSTM.git"
        "steerable-nafx https://github.com/faressc/steerable-nafx.git"
        "stateful-lstm https://github.com/vackva/stateful-lstm.git"
)

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git not found")
endif()

foreach(repo IN LISTS MODEL_REPOSITORIES)
    string(REPLACE " " ";" SPLIT_REPO_DETAILS ${repo})
    list(GET SPLIT_REPO_DETAILS 0 REPO_NAME)
    list(GET SPLIT_REPO_DETAILS 1 REPO_URL)

    set(GIT_CLONE_DIR "${CMAKE_SOURCE_DIR}/extras/models/${REPO_NAME}")

    if(NOT EXISTS "${GIT_CLONE_DIR}")
        message(STATUS "Cloning ${REPO_NAME} into ${GIT_CLONE_DIR}")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} clone ${REPO_URL} ${GIT_CLONE_DIR}
                RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "Git clone of ${REPO_NAME} failed with ${GIT_CLONE_RESULT}")
        endif()
    endif()
endforeach()