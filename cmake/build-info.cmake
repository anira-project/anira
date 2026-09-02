# ==============================================================================
# anira · cmake/build-info.cmake — release identity and the ABI pair from the git tag.
#
#   anira_compute_abi_version()          after project(): ANIRA_ABI_MAJOR / ANIRA_ABI_MINOR
#   anira_generate_build_info(<target>)  writes <anira/abi/build_info.h> into the build tree
#                                        (${CMAKE_CURRENT_BINARY_DIR}/generated/include), adds
#                                        that directory to <target>'s build interface and
#                                        exports it as ANIRA_GENERATED_INCLUDE_DIR
#
# The semver triple comes from tanh_git_version (cmake/tanh/git-version.cmake): a release
# tag vX.Y.Z or a pre-release tag vX.Y.Z-<id> names the build; TANH_VERSION_PRERELEASE and
# TANH_VERSION_DISTANCE say which kind and how far past it HEAD is. The ABI pair is derived
# from the same tag and never written by hand:
#
#   no reachable tag on the 3.x line (the 0.0.0 fallback)   -> 0.0
#   a tag below the 3.x line (major < 3)                    -> 0.0
#   vX.0.0-<pre> before the first vX.Y.Z release             -> 0.N, N = number of vX.0.0-*
#                                                            tags reachable from HEAD, one
#                                                            counter over alphas and betas
#                                                            (alpha.1..3, beta.1..2 -> 0.1..0.5)
#   vX.Y.Z (X >= 3), or a pre-release of a later minor       -> X.Y: the ABI minor equals the
#                                                            semver minor, a patch never
#                                                            touches the headers
#   any commit past the tag (distance > 0)                   -> the next minor of the above,
#                                                            so a development header never
#                                                            claims a released promise it
#                                                            exceeds
#
# The tags must be present in the checkout for the numbers to be right; a shallow CI
# checkout without tags builds a self-consistent 0.0.0 / ABI 0.0 library.
# ==============================================================================
include_guard(GLOBAL)
set(_anira_build_info_module_dir "${CMAKE_CURRENT_LIST_DIR}")

function(anira_compute_abi_version)
    set(_major 0)
    set(_minor 0)
    set(_origin "no tag on the 3.x line reachable")
    if(NOT TANH_VERSION_FULL MATCHES "^0\\.0\\.0(\\+g|$)")
        if(PROJECT_VERSION_MAJOR LESS 3)
            set(_origin "v${TANH_VERSION_SHORT} is below the 3.x line")
        else()
            # Every tag of this major reachable from HEAD, split into the pre-releases of
            # the X.0.0 line and the releases.
            set(_prereleases "")
            set(_releases "")
            find_program(_anira_build_info_git git)
            if(_anira_build_info_git)
                execute_process(
                    COMMAND "${_anira_build_info_git}" tag --merged HEAD --list "v${PROJECT_VERSION_MAJOR}.*"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE _tags OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                string(REPLACE "\n" ";" _tags "${_tags}")
                foreach(_tag IN LISTS _tags)
                    if(_tag MATCHES "^v${PROJECT_VERSION_MAJOR}\\.0\\.0-")
                        list(APPEND _prereleases "${_tag}")
                    elseif(_tag MATCHES "^v${PROJECT_VERSION_MAJOR}\\.[0-9]+\\.[0-9]+$")
                        list(APPEND _releases "${_tag}")
                    endif()
                endforeach()
            endif()
            if(TANH_VERSION_PRERELEASE AND NOT _releases)
                list(LENGTH _prereleases _minor)
                set(_origin "v${TANH_VERSION_SHORT}-${TANH_VERSION_PRERELEASE}, pre-release ${_minor} of the ${PROJECT_VERSION_MAJOR}.0.0 line")
            else()
                set(_major "${PROJECT_VERSION_MAJOR}")
                set(_minor "${PROJECT_VERSION_MINOR}")
                set(_origin "v${TANH_VERSION_SHORT}")
                if(TANH_VERSION_PRERELEASE)
                    string(APPEND _origin "-${TANH_VERSION_PRERELEASE}")
                endif()
            endif()
            if(TANH_VERSION_DISTANCE GREATER 0)
                math(EXPR _minor "${_minor} + 1")
                string(APPEND _origin ", ${TANH_VERSION_DISTANCE} commit(s) past it: next minor")
            endif()
        endif()
    endif()
    set(ANIRA_ABI_MAJOR "${_major}" PARENT_SCOPE)
    set(ANIRA_ABI_MINOR "${_minor}" PARENT_SCOPE)
    message(STATUS "ABI version: ${_major}.${_minor} (${_origin})")
endfunction()

function(anira_generate_build_info target)
    if(NOT DEFINED ANIRA_ABI_MAJOR OR NOT DEFINED ANIRA_ABI_MINOR)
        message(FATAL_ERROR "anira_generate_build_info: call anira_compute_abi_version() first")
    endif()
    # project() leaves an absent component empty; the header wants a number.
    foreach(_component MAJOR MINOR PATCH)
        set(_version_${_component} "${PROJECT_VERSION_${_component}}")
        if(NOT _version_${_component})
            set(_version_${_component} 0)
        endif()
    endforeach()
    set(_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
    configure_file("${_anira_build_info_module_dir}/build_info.h.in"
        "${_dir}/anira/abi/build_info.h" @ONLY)
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${_dir}>)
    set(ANIRA_GENERATED_INCLUDE_DIR "${_dir}" PARENT_SCOPE)
endfunction()
