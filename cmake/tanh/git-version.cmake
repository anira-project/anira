# ==============================================================================
# tanh-tooling · cmake/git-version.cmake — project version from git tags.
#
#   tanh_git_version(<source-dir> [MATCH <glob>...])
#
# Outputs (plain variables, caller scope):
#   TANH_VERSION_SHORT       numeric part of the nearest tag: "vX.Y.Z" -> "X.Y.Z", and a
#                            pre-release tag "vX.Y.Z-alpha.1" -> "X.Y.Z" (project() and
#                            find_package() need digits); "0.0.0" when no tag is reachable
#   TANH_VERSION_FULL        `git describe --tags --dirty` without the "v", e.g.
#                            "1.2.0-3-gabc123-dirty" or "3.0.0-alpha.1-3-gabc123"; without a
#                            tag "0.0.0+g<hash>[-dirty]"; "0.0.0" outside a git checkout
#   TANH_VERSION_PRERELEASE  the pre-release identifier of the nearest tag, "alpha.1" for
#                            "v3.0.0-alpha.1"; empty for a release tag or without a tag
#   TANH_VERSION_DISTANCE    commits between the nearest tag and HEAD; "0" exactly on the
#                            tag, and "0" without a tag
#
# MATCH restricts the tags considered (`git describe --match`), one glob per argument:
# a long-lived branch that also reaches an older release line names itself only
# after its own tags with MATCH "v3*", instead of after whichever tag is fewer
# commits away.
#
# Meant to run BEFORE project(): project(<name> VERSION ${TANH_VERSION_SHORT}).
# Inputs: a git executable on PATH. Requires CMake >= 3.18.
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

function(tanh_git_version source_dir)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "MATCH")
    set(_match "")
    foreach(_glob IN LISTS arg_MATCH)
        list(APPEND _match --match "${_glob}")
    endforeach()

    set(_short "0.0.0")
    set(_full "0.0.0")
    set(_prerelease "")
    set(_distance "0")
    find_program(_tanh_git git)
    if(_tanh_git)
        execute_process(COMMAND "${_tanh_git}" describe --tags --abbrev=0 ${_match}
            WORKING_DIRECTORY "${source_dir}"
            OUTPUT_VARIABLE _tag OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_tag)
            execute_process(COMMAND "${_tanh_git}" describe --tags --dirty ${_match}
                WORKING_DIRECTORY "${source_dir}"
                OUTPUT_VARIABLE _describe OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            execute_process(COMMAND "${_tanh_git}" rev-list --count "${_tag}..HEAD"
                WORKING_DIRECTORY "${source_dir}"
                OUTPUT_VARIABLE _distance OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            string(REGEX REPLACE "^v" "" _short "${_tag}")
            string(REGEX REPLACE "^v" "" _full "${_describe}")
            # SemVer: everything after the first "-" of the tag is the pre-release
            # identifier; project() takes the numeric part only.
            if(_short MATCHES "^([0-9][0-9.]*)-(.+)$")
                set(_short "${CMAKE_MATCH_1}")
                set(_prerelease "${CMAKE_MATCH_2}")
            endif()
            if(NOT _distance MATCHES "^[0-9]+$")
                set(_distance "0")
            endif()
        else()
            # No tag reachable (or not a git checkout): fall back to the commit hash.
            # --exclude keeps an annotated tag that MATCH rejected from naming HEAD.
            execute_process(COMMAND "${_tanh_git}" describe --always --dirty --exclude "*"
                WORKING_DIRECTORY "${source_dir}"
                OUTPUT_VARIABLE _hash OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_hash)
                set(_full "${_short}+g${_hash}")
            endif()
        endif()
    endif()
    set(TANH_VERSION_SHORT "${_short}" PARENT_SCOPE)
    set(TANH_VERSION_FULL "${_full}" PARENT_SCOPE)
    set(TANH_VERSION_PRERELEASE "${_prerelease}" PARENT_SCOPE)
    set(TANH_VERSION_DISTANCE "${_distance}" PARENT_SCOPE)
endfunction()
