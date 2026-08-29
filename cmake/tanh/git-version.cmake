# ==============================================================================
# tanh-tooling · cmake/git-version.cmake — project version from git tags.
#
#   tanh_git_version(<source-dir>)
#
# Outputs (plain variables, caller scope):
#   TANH_VERSION_SHORT  numeric part of the nearest tag "vX.Y.Z" -> "X.Y.Z"; "0.0.0" when
#                       no tag is reachable (project() and find_package() need digits)
#   TANH_VERSION_FULL   `git describe --tags --dirty` without the "v", e.g. "1.2.0-3-gabc123-dirty";
#                       without a tag "0.0.0+g<hash>[-dirty]"; "0.0.0" outside a git checkout
#
# Meant to run BEFORE project(): project(<name> VERSION ${TANH_VERSION_SHORT}).
# Inputs: a git executable on PATH. Requires CMake >= 3.18.
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

function(tanh_git_version source_dir)
    set(_short "0.0.0")
    set(_full "0.0.0")
    find_program(_tanh_git git)
    if(_tanh_git)
        execute_process(COMMAND "${_tanh_git}" describe --tags --abbrev=0
            WORKING_DIRECTORY "${source_dir}"
            OUTPUT_VARIABLE _tag OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_tag)
            execute_process(COMMAND "${_tanh_git}" describe --tags --dirty
                WORKING_DIRECTORY "${source_dir}"
                OUTPUT_VARIABLE _describe OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            string(REGEX REPLACE "^v" "" _short "${_tag}")
            string(REGEX REPLACE "^v" "" _full "${_describe}")
        else()
            # No tag reachable (or not a git checkout): fall back to the commit hash.
            execute_process(COMMAND "${_tanh_git}" describe --always --dirty
                WORKING_DIRECTORY "${source_dir}"
                OUTPUT_VARIABLE _hash OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_hash)
                set(_full "${_short}+g${_hash}")
            endif()
        endif()
    endif()
    set(TANH_VERSION_SHORT "${_short}" PARENT_SCOPE)
    set(TANH_VERSION_FULL "${_full}" PARENT_SCOPE)
endfunction()
