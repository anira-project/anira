# ==============================================================================
# tanh-tooling · cmake/install-helpers.cmake — install-tree conventions.
#
#   tanh_set_install_rpath(<target>... [EXTRA_ELF_PATHS <dir>...] [EXTRA_APPLE_PATHS <dir>...])
#     CMake clears the RPATH at install time; installed shared libraries and executables
#     then need it re-set to find their siblings. $ORIGIN (ELF) / @loader_path (Mach-O)
#     resolve to the directory of the loading binary at run time; extra absolute paths
#     (e.g. a vendor's MKL) are appended per format. No-op on PE (the loader searches
#     the binary's directory anyway) and Wasm.
#
# Includes the platform module explicitly. Requires CMake >= 3.18; include after project().
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/platform.cmake")

# cmake_parse_arguments: an empty value after a keyword is a value, not an omission
# (CMP0174 NEW); function bodies record the policy state of their definition.
cmake_policy(PUSH)
if(POLICY CMP0174)
    cmake_policy(SET CMP0174 NEW)
endif()

function(tanh_set_install_rpath)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "" "EXTRA_ELF_PATHS;EXTRA_APPLE_PATHS")
    if(NOT arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tanh_set_install_rpath: give at least one target")
    endif()
    if(TANH_BINARY_FORMAT STREQUAL "ELF")
        set(_rpath "$ORIGIN" ${arg_EXTRA_ELF_PATHS})
    elseif(TANH_BINARY_FORMAT STREQUAL "Mach-O")
        set(_rpath "@loader_path" ${arg_EXTRA_APPLE_PATHS})
    else()
        return()
    endif()
    foreach(_t IN LISTS arg_UNPARSED_ARGUMENTS)
        if(NOT TARGET ${_t})
            message(FATAL_ERROR "tanh_set_install_rpath: '${_t}' is not a target")
        endif()
        set_target_properties(${_t} PROPERTIES INSTALL_RPATH "${_rpath}")
    endforeach()
endfunction()

cmake_policy(POP)
