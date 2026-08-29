# ==============================================================================
# tanh-tooling · cmake/ios.toolchain.cmake — iOS toolchain file (Xcode generator).
#
#   cmake -B build -G Xcode -DCMAKE_TOOLCHAIN_FILE=cmake/tanh/ios.toolchain.cmake \
#         -DTANH_IOS_PLATFORM=SIMULATOR|DEVICE [-DTANH_IOS_TEST_RUNNER_DIR=<dir>]
#
# TANH_IOS_PLATFORM (default SIMULATOR) picks SDK, architecture and Xcode platform; it is
# persisted in the environment because CMake re-reads the toolchain during its compiler
# checks, and mirrored into the cache. cmake/tanh/platform.cmake reads it as the hint for
# TANH_IOS_PLATFORM in the project.
# TANH_IOS_TEST_RUNNER_DIR: a directory holding run-ios-sim.sh / run-ios-device.sh (repo
# owned) to be used as CMAKE_CROSSCOMPILING_EMULATOR, so ctest can launch .app bundles.
# Deployment target defaults to 15.0; pass -DCMAKE_OSX_DEPLOYMENT_TARGET to override.
# Symbol visibility is deliberately not set here — that is per target
# (cmake/tanh/symbol-policy.cmake).
# ==============================================================================
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_SYSTEM_VERSION 15.0)
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0 CACHE STRING "Minimum iOS deployment version")
endif()

if(NOT DEFINED TANH_IOS_PLATFORM)
    if(DEFINED ENV{TANH_IOS_PLATFORM})
        set(TANH_IOS_PLATFORM "$ENV{TANH_IOS_PLATFORM}")
    else()
        set(TANH_IOS_PLATFORM "SIMULATOR")
    endif()
endif()
set(ENV{TANH_IOS_PLATFORM} "${TANH_IOS_PLATFORM}")
set(TANH_IOS_PLATFORM "${TANH_IOS_PLATFORM}" CACHE STRING "iOS platform: DEVICE or SIMULATOR" FORCE)

if(TANH_IOS_PLATFORM STREQUAL "DEVICE")
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_OSX_SYSROOT iphoneos)
    set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphoneos")
elseif(TANH_IOS_PLATFORM STREQUAL "SIMULATOR")
    set(CMAKE_OSX_ARCHITECTURES "arm64")  # Apple Silicon hosts
    set(CMAKE_OSX_SYSROOT iphonesimulator)
    set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphonesimulator")
else()
    message(FATAL_ERROR "Unknown TANH_IOS_PLATFORM '${TANH_IOS_PLATFORM}': use DEVICE or SIMULATOR")
endif()

# Resolve the SDK name to its path (platform.cmake derives DEVICE/SIMULATOR from it too).
execute_process(COMMAND xcrun --sdk ${CMAKE_OSX_SYSROOT} --show-sdk-path
    OUTPUT_VARIABLE _tanh_sdk_path OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_OSX_SYSROOT "${_tanh_sdk_path}")

set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES YES)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

# Lock the architecture: ONLY_ACTIVE_ARCH only works inside the Xcode IDE (it needs a run
# destination); for CLI builds set ARCHS explicitly so Xcode does not build every default
# architecture.
set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
set(CMAKE_XCODE_ATTRIBUTE_ARCHS "${CMAKE_OSX_ARCHITECTURES}")

if(TANH_IOS_TEST_RUNNER_DIR)
    if(TANH_IOS_PLATFORM STREQUAL "SIMULATOR")
        set(CMAKE_CROSSCOMPILING_EMULATOR "${TANH_IOS_TEST_RUNNER_DIR}/run-ios-sim.sh"
            CACHE STRING "iOS simulator test runner" FORCE)
    else()
        set(CMAKE_CROSSCOMPILING_EMULATOR "${TANH_IOS_TEST_RUNNER_DIR}/run-ios-device.sh"
            CACHE STRING "iOS device test runner" FORCE)
    endif()
endif()

message(STATUS "iOS toolchain: ${TANH_IOS_PLATFORM}, ${CMAKE_OSX_ARCHITECTURES}, SDK ${CMAKE_OSX_SYSROOT}, "
               "deployment target ${CMAKE_OSX_DEPLOYMENT_TARGET}")
