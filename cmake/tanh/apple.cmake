# ==============================================================================
# tanh-tooling · cmake/apple.cmake — macOS/iOS build defaults, no-ops elsewhere.
#
#   tanh_apple_deployment_target([MACOS <ver>] [IOS <ver>])
#     Sets CMAKE_OSX_DEPLOYMENT_TARGET (cache — the variable is read at generate time
#     for every target, which is why it must live in the cache) only when it is empty,
#     never over a value somebody chose: as a sub-project that would silently raise
#     the embedding project's own minimum. Reports a lower value as STATUS, not as an
#     error.
#   tanh_apple_sysroot_from_xcrun()
#     macOS with a non-Apple clang: CMake sets no sysroot, so the compile database
#     lacks -isysroot and clang-tidy cannot find system headers. Fills CMAKE_OSX_SYSROOT
#     from `xcrun --show-sdk-path` when unset.
#   tanh_apple_default_architectures()
#     Defaults CMAKE_OSX_ARCHITECTURES when unset (iOS: arm64 — device and, on Apple
#     Silicon, simulator; macOS: the host arch) and mirrors a single-arch macOS
#     selection into CMAKE_SYSTEM_PROCESSOR so arch-keyed asset pickers see it; a
#     universal selection is left as-is.
#   tanh_ios_disable_code_signing(<target>...)
#     Library targets need no signature (only apps do); clears the Xcode attributes.
#   tanh_ios_test_bundle(<target> BUNDLE_ID_PREFIX <com.example> [DEVELOPMENT_TEAM <id>])
#     Turns a test executable into an .app bundle with a bundle id (hyphenated target
#     name) so simctl/devicectl can install and launch it; signs it automatically when a
#     team id is given.
#
# Inputs: APPLE, CMAKE_SYSTEM_NAME, CMAKE_OSX_*. Requires CMake >= 3.18; include after project().
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

# cmake_parse_arguments: an empty value after a keyword is a value, not an omission
# (CMP0174 NEW); function bodies record the policy state of their definition.
cmake_policy(PUSH)
if(POLICY CMP0174)
    cmake_policy(SET CMP0174 NEW)
endif()

function(tanh_apple_deployment_target)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "MACOS;IOS" "")
    if(NOT APPLE)
        return()
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(_what "iOS")
        set(_min "${arg_IOS}")
    else()
        set(_what "macOS")
        set(_min "${arg_MACOS}")
    endif()
    if(NOT _min)
        return()
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        # CMake's Darwin initialisation pre-creates the cache entry with an empty value,
        # so a plain set(CACHE) would never take effect; FORCE here only ever replaces
        # that empty value, never a version somebody chose.
        set(CMAKE_OSX_DEPLOYMENT_TARGET "${_min}" CACHE STRING "Minimum ${_what} deployment version" FORCE)
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS _min)
        message(STATUS "${PROJECT_NAME} is built and tested for ${_what} >= ${_min}; CMAKE_OSX_DEPLOYMENT_TARGET is ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    message(STATUS "The minimum ${_what} version is set to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endfunction()

function(tanh_apple_sysroot_from_xcrun)
    if(NOT APPLE OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_OSX_SYSROOT)
        return()
    endif()
    execute_process(COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE _sysroot OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_sysroot)
        # Same as the deployment target: the cache entry pre-exists empty, so FORCE is
        # needed to write it — and it only ever replaces an empty value.
        set(CMAKE_OSX_SYSROOT "${_sysroot}" CACHE PATH "macOS SDK path" FORCE)
    endif()
endfunction()

function(tanh_ios_disable_code_signing)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        return()
    endif()
    foreach(_t IN LISTS ARGN)
        set_target_properties(${_t} PROPERTIES
            XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
            XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
            XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
    endforeach()
endfunction()

function(tanh_ios_test_bundle target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "BUNDLE_ID_PREFIX;DEVELOPMENT_TEAM" "")
    if(NOT arg_BUNDLE_ID_PREFIX)
        message(FATAL_ERROR "tanh_ios_test_bundle(${target}): BUNDLE_ID_PREFIX is required")
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        return()
    endif()
    string(REPLACE "_" "-" _suffix "${target}")  # bundle ids allow hyphens, not underscores
    set_target_properties(${target} PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "${target}"
        MACOSX_BUNDLE_GUI_IDENTIFIER "${arg_BUNDLE_ID_PREFIX}.${_suffix}"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${arg_BUNDLE_ID_PREFIX}.${_suffix}")
    if(arg_DEVELOPMENT_TEAM)
        set_target_properties(${target} PROPERTIES
            XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
            XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
            XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${arg_DEVELOPMENT_TEAM}")
    endif()
endfunction()

# See the header. A macro: it writes CMAKE_OSX_ARCHITECTURES / CMAKE_SYSTEM_PROCESSOR
# in the caller's scope.
macro(tanh_apple_default_architectures)
    if(APPLE)
        if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
            if(NOT CMAKE_OSX_ARCHITECTURES)
                set(CMAKE_OSX_ARCHITECTURES "arm64")
            endif()
        else()
            if(NOT CMAKE_OSX_ARCHITECTURES)
                set(CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}")
            endif()
            if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
                set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
            endif()
        endif()
    endif()
endmacro()

cmake_policy(POP)
