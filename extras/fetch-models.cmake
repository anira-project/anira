# Fetch the example-model fixture repositories into <extras>/models, each pinned
# to a fixed commit. Included by extras/CMakeLists.txt at configure time, and
# runnable standalone to pre-seed the CI cache:
#
#     cmake -P extras/fetch-models.cmake
#
# A destination directory that already exists is left untouched (this is what
# lets the CI cache — and a developer's existing checkout — short-circuit the
# fetch; delete a subdirectory to refetch it at the pin). Each fetch stages into
# <dest>.fetching and renames into place only after a completed checkout, so an
# interrupted or failed fetch can never leave a half tree the exists-check would
# accept; the .git metadata is dropped from the result (the trees are pinned
# snapshots, not working clones). Bump a pin by editing its default here or
# passing -DANIRA_MODELS_<NAME>_REF=<sha>; the CI cache key hashes this file,
# so editing it invalidates the cached tree.

if(NOT DEFINED ANIRA_MODELS_GUITARLSTM_REF)
    set(ANIRA_MODELS_GUITARLSTM_REF "08c74183cc16878194b827f8b27304466da145f0") # faressc/GuitarLSTM (hybrid-nn)
endif()
if(NOT DEFINED ANIRA_MODELS_STEERABLENAFX_REF)
    set(ANIRA_MODELS_STEERABLENAFX_REF "e49546773b5f7e6dbb299d6820c9c53bd4e6a2f3") # faressc/steerable-nafx (cnn)
endif()
if(NOT DEFINED ANIRA_MODELS_STATEFULLSTM_REF)
    set(ANIRA_MODELS_STATEFULLSTM_REF "3ce7bec9bebaf9e4ab2327971cae1be722ded78c") # vackva/stateful-lstm (stateful-rnn)
endif()
if(NOT DEFINED ANIRA_MODELS_EXAMPLE_MODELS_REF)
    set(ANIRA_MODELS_EXAMPLE_MODELS_REF "6162b0aa82acb9da0d8e753d55bb3b6f6b3c325b") # anira-project/example-models (model-pool)
endif()
if(NOT DEFINED ANIRA_MODELS_RAVE_REF)
    set(ANIRA_MODELS_RAVE_REF "4800f15ab86c3ba091ef1505001c1e234df69980") # anira-project/example-models, third-party branch (RAVE)
endif()
set(_anira_rave_sha256 "1d50ba64fb0bc2d43ec10dda8414b37379a4a9b48b82c228fce5af8524d06507") # rave_funk_drum.ts at ANIRA_MODELS_RAVE_REF

set(_anira_models_fetch_dir "${CMAKE_CURRENT_LIST_DIR}/models")

# <git-url> <subdir under models/> <pinned ref>
set(_anira_model_repos
    "https://github.com/faressc/GuitarLSTM.git hybrid-nn/GuitarLSTM ${ANIRA_MODELS_GUITARLSTM_REF}"
    "https://github.com/faressc/steerable-nafx.git cnn/steerable-nafx ${ANIRA_MODELS_STEERABLENAFX_REF}"
    "https://github.com/vackva/stateful-lstm.git stateful-rnn/stateful-lstm ${ANIRA_MODELS_STATEFULLSTM_REF}"
    "https://github.com/anira-project/example-models.git model-pool/example-models ${ANIRA_MODELS_EXAMPLE_MODELS_REF}"
)

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git not found")
endif()

foreach(_repo IN LISTS _anira_model_repos)
    string(REPLACE " " ";" _fields "${_repo}")
    list(GET _fields 0 _url)
    list(GET _fields 1 _subdir)
    list(GET _fields 2 _ref)
    set(_dest "${_anira_models_fetch_dir}/${_subdir}")

    if(NOT EXISTS "${_dest}")
        message(STATUS "Fetching ${_url} @ ${_ref} into ${_dest}")
        set(_staging "${_dest}.fetching")
        file(REMOVE_RECURSE "${_staging}")
        # A shallow clone cannot check out an arbitrary commit, so init + fetch
        # the pinned ref directly (GitHub serves reachable SHAs to fetch).
        execute_process(
            COMMAND ${GIT_EXECUTABLE} init -q "${_staging}"
            RESULT_VARIABLE _init_result)
        if(NOT _init_result EQUAL "0")
            message(FATAL_ERROR "git init of ${_staging} failed with ${_init_result}")
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${_staging}" fetch -q --depth 1 "${_url}" "${_ref}"
            RESULT_VARIABLE _fetch_result)
        if(NOT _fetch_result EQUAL "0")
            message(FATAL_ERROR "git fetch of ${_url} @ ${_ref} failed with ${_fetch_result}")
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${_staging}" -c advice.detachedHead=false checkout -q FETCH_HEAD
            RESULT_VARIABLE _checkout_result)
        if(NOT _checkout_result EQUAL "0")
            message(FATAL_ERROR "git checkout of ${_url} @ ${_ref} failed with ${_checkout_result}")
        endif()
        # Pinned snapshot, not a working clone — drop the object store (it would
        # double the CI cache entry and the on-device staging on Android).
        file(REMOVE_RECURSE "${_staging}/.git")
        get_filename_component(_dest_parent "${_dest}" DIRECTORY)
        file(MAKE_DIRECTORY "${_dest_parent}")
        file(RENAME "${_staging}" "${_dest}")
    endif()
endforeach()

# RAVE TorchScript model — a LibTorch-only fixture, so extras/CMakeLists.txt
# requests it only when that backend is enabled. The standalone (cmake -P)
# invocation defaults to ON so a seeded CI cache covers the LibTorch legs too.
if(NOT DEFINED ANIRA_MODELS_FETCH_RAVE)
    set(ANIRA_MODELS_FETCH_RAVE ON)
endif()

if(ANIRA_MODELS_FETCH_RAVE)
    set(_rave_dir "${_anira_models_fetch_dir}/third-party/ircam-acids/RAVE")
    set(_rave_url "https://github.com/anira-project/example-models/raw/${ANIRA_MODELS_RAVE_REF}/third-party/ircam-acids/RAVE/rave_funk_drum.ts")

    if(NOT EXISTS "${_rave_dir}/rave_funk_drum.ts")
        message(STATUS "Downloading RAVE model from ${_rave_url}")
        file(MAKE_DIRECTORY "${_rave_dir}")
        # Download to a partial name and rename on success, so an interrupted
        # download never leaves a truncated model the exists-check accepts.
        file(REMOVE "${_rave_dir}/rave_funk_drum.ts.part")
        # One retry: transient TLS/connect errors against GitHub happen.
        foreach(_attempt RANGE 1 2)
            file(DOWNLOAD
                "${_rave_url}"
                "${_rave_dir}/rave_funk_drum.ts.part"
                SHOW_PROGRESS
                EXPECTED_HASH SHA256=${_anira_rave_sha256}
                STATUS _rave_status
                LOG _rave_log
            )
            list(GET _rave_status 0 _rave_result)
            if(_rave_result EQUAL 0)
                break()
            endif()
            if(_attempt EQUAL 1)
                message(STATUS "RAVE download failed, retrying once")
                file(REMOVE "${_rave_dir}/rave_funk_drum.ts.part")
                execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 3)
            endif()
        endforeach()
        if(NOT _rave_result EQUAL 0)
            file(REMOVE "${_rave_dir}/rave_funk_drum.ts.part")
            message(FATAL_ERROR "Failed to download RAVE model: ${_rave_log}")
        endif()
        file(RENAME "${_rave_dir}/rave_funk_drum.ts.part" "${_rave_dir}/rave_funk_drum.ts")
    endif()
endif()
