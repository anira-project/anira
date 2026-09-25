# Fetch the example-model fixture repositories into <extras>/models, each pinned
# to a fixed commit. Included by extras/CMakeLists.txt at configure time, and
# runnable standalone to pre-seed the CI cache:
#
#     cmake -P extras/fetch-models.cmake
#
# Every fetched tree carries a stamp, <dest>/.anira-pin, naming the URL and the
# ref it was fetched at. A tree whose stamp matches its pin is left untouched
# (this is what lets the CI cache and a developer's existing checkout
# short-circuit the fetch); a tree without a stamp, or stamped with another pin,
# is fetched again and replaced, so a bumped pin reaches every checkout at its
# next configure instead of only the fresh ones. Each fetch stages into
# <dest>.fetching and swaps into place only after a completed checkout, so an
# interrupted or failed fetch can never leave a half tree behind; when the
# refetch of an existing tree fails (offline), the old tree stays and a warning
# says it is not at the pin. The .git metadata is dropped from the result (the
# trees are pinned snapshots, not working clones, so anything written into one
# is lost at the next pin bump). Bump a pin by editing its default here or
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
    set(ANIRA_MODELS_EXAMPLE_MODELS_REF "d1bc5cf87b496bc5c6eb7a1f95e30333959a7c40") # anira-project/example-models (model-pool)
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

# Fetch <url> at <ref> into <staging> (removed first) and stamp it. Sets
# <result_var> to "" on success, else to what failed.
function(_anira_fetch_pinned url ref staging result_var)
    file(REMOVE_RECURSE "${staging}")
    # A shallow clone cannot check out an arbitrary commit, so init + fetch
    # the pinned ref directly (GitHub serves reachable SHAs to fetch).
    execute_process(
        COMMAND ${GIT_EXECUTABLE} init -q "${staging}"
        RESULT_VARIABLE _init_result)
    if(NOT _init_result EQUAL "0")
        set(${result_var} "git init of ${staging} failed with ${_init_result}" PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${staging}" fetch -q --depth 1 "${url}" "${ref}"
        RESULT_VARIABLE _fetch_result)
    if(NOT _fetch_result EQUAL "0")
        file(REMOVE_RECURSE "${staging}")
        set(${result_var} "git fetch of ${url} @ ${ref} failed with ${_fetch_result}" PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${staging}" -c advice.detachedHead=false checkout -q FETCH_HEAD
        RESULT_VARIABLE _checkout_result)
    if(NOT _checkout_result EQUAL "0")
        file(REMOVE_RECURSE "${staging}")
        set(${result_var} "git checkout of ${url} @ ${ref} failed with ${_checkout_result}" PARENT_SCOPE)
        return()
    endif()
    # Pinned snapshot, not a working clone — drop the object store (it would
    # double the CI cache entry and the on-device staging on Android).
    file(REMOVE_RECURSE "${staging}/.git")
    file(WRITE "${staging}/.anira-pin" "${url} ${ref}\n")
    set(${result_var} "" PARENT_SCOPE)
endfunction()

foreach(_repo IN LISTS _anira_model_repos)
    string(REPLACE " " ";" _fields "${_repo}")
    list(GET _fields 0 _url)
    list(GET _fields 1 _subdir)
    list(GET _fields 2 _ref)
    set(_dest "${_anira_models_fetch_dir}/${_subdir}")

    set(_found_pin "")
    if(EXISTS "${_dest}/.anira-pin")
        file(STRINGS "${_dest}/.anira-pin" _found_pin LIMIT_COUNT 1)
    endif()
    if(_found_pin STREQUAL "${_url} ${_ref}")
        continue()
    endif()

    if(EXISTS "${_dest}")
        if(_found_pin STREQUAL "")
            message(STATUS "Refetching ${_dest}: it carries no pin stamp, so it may predate ${_ref}")
        else()
            message(STATUS "Refetching ${_dest}: stamped '${_found_pin}', pinned at ${_ref}")
        endif()
    else()
        message(STATUS "Fetching ${_url} @ ${_ref} into ${_dest}")
    endif()
    set(_staging "${_dest}.fetching")
    _anira_fetch_pinned("${_url}" "${_ref}" "${_staging}" _error)
    if(NOT _error STREQUAL "")
        if(EXISTS "${_dest}")
            message(WARNING "${_error}; keeping ${_dest}, which is not at the pin ${_ref} (tests that need the pinned fixtures may fail)")
            continue()
        endif()
        message(FATAL_ERROR "${_error}")
    endif()
    file(REMOVE_RECURSE "${_dest}")
    get_filename_component(_dest_parent "${_dest}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dest_parent}")
    file(RENAME "${_staging}" "${_dest}")
endforeach()

# RAVE TorchScript model — a LibTorch-only fixture, so extras/CMakeLists.txt
# requests it only when that engine is enabled. The standalone (cmake -P)
# invocation defaults to ON so a seeded CI cache covers the LibTorch legs too.
if(NOT DEFINED ANIRA_MODELS_FETCH_RAVE)
    set(ANIRA_MODELS_FETCH_RAVE ON)
endif()

if(ANIRA_MODELS_FETCH_RAVE)
    set(_rave_dir "${_anira_models_fetch_dir}/third-party/ircam-acids/RAVE")
    set(_rave_url "https://github.com/anira-project/example-models/raw/${ANIRA_MODELS_RAVE_REF}/third-party/ircam-acids/RAVE/rave_funk_drum.ts")

    # The pin is the file's hash, so an existing file is checked against it: a
    # file downloaded at an older pin is downloaded again.
    set(_rave_current FALSE)
    if(EXISTS "${_rave_dir}/rave_funk_drum.ts")
        file(SHA256 "${_rave_dir}/rave_funk_drum.ts" _rave_found_sha256)
        if(_rave_found_sha256 STREQUAL _anira_rave_sha256)
            set(_rave_current TRUE)
        else()
            message(STATUS "Downloading the RAVE model again: its SHA256 is not the pinned one")
        endif()
    endif()
    if(NOT _rave_current)
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
