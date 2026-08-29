# ==============================================================================
# tanh-tooling · cmake/platform.cmake — one spelling of "which platform is this".
#
# CMake's own spellings are ambiguous for what a build usually needs to decide:
# APPLE covers macOS and iOS, UNIX covers Linux, Android, Emscripten and macOS,
# and most linker decisions (version script vs -exported_symbols_list vs
# dllexport, --exclude-libs vs -load_hidden) follow the object format, not the OS.
# This module resolves both axes once, in plain variables.
#
# Inputs (all CMake-provided):
#   CMAKE_SYSTEM_NAME, CMAKE_OSX_SYSROOT, EMSCRIPTEN, CMAKE_CXX_COMPILER
#   TANH_IOS_PLATFORM — optional, may be pre-set by a toolchain file to DEVICE or
#     SIMULATOR when the SDK is chosen at build time (Xcode) and cannot be derived.
#
# Outputs (plain variables in the including scope):
#   TANH_OPERATING_SYSTEM   Linux | macOS | iOS | Android | Windows | Emscripten |
#                           <CMAKE_SYSTEM_NAME> for any other UNIX | Unknown
#   TANH_BINARY_FORMAT      ELF | Mach-O | PE | Wasm | Unknown
#   TANH_IOS_PLATFORM       DEVICE | SIMULATOR | "" (not iOS, or not derivable)
#   TANH_PLATFORM_COMPILE_DEFINITIONS
#                           THL_PLATFORM_<OS>=1 [THL_PLATFORM_IOS_DEVICE=1 |
#                           THL_PLATFORM_IOS_SIMULATOR=1] — the definitions
#                           tanh::Core carries as PUBLIC compile definitions, for a
#                           library that wants to expose the same ones.
#
# Requires CMake >= 3.18. Include after project() (or after a toolchain file has
# set CMAKE_SYSTEM_NAME); usable inside a package config, which always runs after
# the consumer's project(). Errors out otherwise instead of guessing.
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

if(NOT CMAKE_SYSTEM_NAME)
    message(FATAL_ERROR "tanh/platform.cmake: CMAKE_SYSTEM_NAME is empty — include this module after "
                        "project(), or after a toolchain file has set CMAKE_SYSTEM_NAME")
endif()

if(NOT DEFINED TANH_IOS_PLATFORM)
    set(TANH_IOS_PLATFORM "")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(TANH_OPERATING_SYSTEM "iOS")
    set(TANH_BINARY_FORMAT "Mach-O")
    if(NOT TANH_IOS_PLATFORM)
        if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
            set(TANH_IOS_PLATFORM "SIMULATOR")
        elseif(CMAKE_OSX_SYSROOT MATCHES "iPhoneOS|iphoneos")
            set(TANH_IOS_PLATFORM "DEVICE")
        endif()
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(TANH_OPERATING_SYSTEM "Android")
    set(TANH_BINARY_FORMAT "ELF")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(TANH_OPERATING_SYSTEM "macOS")
    set(TANH_BINARY_FORMAT "Mach-O")
elseif(EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR CMAKE_CXX_COMPILER MATCHES "em\\+\\+")
    # Emscripten reports UNIX=TRUE; it must be resolved before any UNIX fallback.
    set(TANH_OPERATING_SYSTEM "Emscripten")
    set(TANH_BINARY_FORMAT "Wasm")
elseif(CMAKE_SYSTEM_NAME MATCHES "^(Windows|WindowsStore|CYGWIN|MSYS)$")
    set(TANH_OPERATING_SYSTEM "Windows")
    set(TANH_BINARY_FORMAT "PE")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TANH_OPERATING_SYSTEM "Linux")
    set(TANH_BINARY_FORMAT "ELF")
elseif(UNIX)
    # FreeBSD, OpenBSD, ...: ELF, but not Linux — keep the real name.
    set(TANH_OPERATING_SYSTEM "${CMAKE_SYSTEM_NAME}")
    set(TANH_BINARY_FORMAT "ELF")
else()
    set(TANH_OPERATING_SYSTEM "Unknown")
    set(TANH_BINARY_FORMAT "Unknown")
endif()

set(TANH_PLATFORM_COMPILE_DEFINITIONS "")
if(NOT TANH_OPERATING_SYSTEM STREQUAL "Unknown")
    string(TOUPPER "${TANH_OPERATING_SYSTEM}" _tanh_os_upper)
    list(APPEND TANH_PLATFORM_COMPILE_DEFINITIONS "THL_PLATFORM_${_tanh_os_upper}=1")
    if(TANH_IOS_PLATFORM)
        list(APPEND TANH_PLATFORM_COMPILE_DEFINITIONS "THL_PLATFORM_IOS_${TANH_IOS_PLATFORM}=1")
    endif()
    unset(_tanh_os_upper)
endif()
