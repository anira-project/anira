# ==============================================================================
# validate-options.cmake — cross-option validation guards, in one place.
#
# Included after project() AND after Emscripten detection (so EMSDK_VERSION is
# resolved) but BEFORE the backends are set up / the library target is created,
# so the auto-disable below takes effect. include() runs in the caller's scope,
# so plain set() here updates the options the rest of the build sees.
# ==============================================================================

# LiteRT and TFLite are the same TensorFlow Lite runtime exposed through two C APIs;
# their static libraries export the same TfLite* symbols and collide when linked
# together. LiteRT is the default; TFLite is legacy.
if(ANIRA_WITH_TFLITE AND ANIRA_WITH_LITERT)
    message(FATAL_ERROR
        "ANIRA_WITH_TFLITE and ANIRA_WITH_LITERT are the same TensorFlow Lite runtime exposed "
        "through two C APIs and cannot be enabled together (their static libraries export the "
        "same TfLite* symbols). To use the legacy TFLite backend, set "
        "-DANIRA_WITH_LITERT=OFF -DANIRA_WITH_TFLITE=ON.")
endif()

# ------------------------------------------------------------------------------
# Backend linkage follows the library type, with no per-engine override: a shared
# anira links shared backends, a static anira links static backends
# (ANIRA_BACKEND_LINKAGE). These are the only two shapes anira supports, because they
# are the only ones in which every engine exists exactly once per process and a
# consumer can still reach it: a shared anira that absorbed a static engine could
# neither hand that engine to a consumer (its symbols are hidden inside libanira) nor
# keep a consumer's own copy of it from becoming a second instance. An engine whose
# prebuilt archives do not ship the required linkage is disabled with a warning:
#   * LibTorch ships shared-only (and its bundled XNNPACK collides with static LiteRT).
#   * ExecuTorch ships static-only (a force-loaded runtime that aborts when its
#     kernels register twice in one process).
# The rule is checked once more at compile time by the BackendLinkage test.
# ------------------------------------------------------------------------------
if(BUILD_SHARED_LIBS)
    set(ANIRA_BACKEND_LINKAGE "shared")
else()
    set(ANIRA_BACKEND_LINKAGE "static")
endif()

# iOS ships a single static xcframework per engine and Emscripten links everything
# into one wasm module: neither has a shared shape, so demand the static one
# explicitly instead of silently building something else than what was asked for.
if(BUILD_SHARED_LIBS AND (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR DEFINED EMSDK_VERSION))
    message(FATAL_ERROR "anira is static-only on iOS and Emscripten (the backends ship static "
                        "archives only there): configure with -DBUILD_SHARED_LIBS=OFF.")
endif()

# The messages start with the "disabling ..." phrase on purpose: CMake wraps message
# text, and build_test.yml greps the configure output for exactly that phrase.
if(ANIRA_WITH_LIBTORCH AND NOT ANIRA_BACKEND_LINKAGE STREQUAL "shared")
    message(WARNING "disabling ANIRA_WITH_LIBTORCH: LibTorch is shared-only and cannot be linked "
                    "into a fully static anira build (BUILD_SHARED_LIBS=OFF). Build shared to use LibTorch.")
    set(ANIRA_WITH_LIBTORCH OFF)
endif()

if(ANIRA_WITH_EXECUTORCH AND NOT ANIRA_BACKEND_LINKAGE STREQUAL "static")
    message(WARNING "disabling ANIRA_WITH_EXECUTORCH: ExecuTorch is static-only and cannot be linked "
                    "into a shared anira build (BUILD_SHARED_LIBS=ON) — anira links its backends in "
                    "the linkage of the library itself, so that every engine exists exactly once per "
                    "process. Build static (-DBUILD_SHARED_LIBS=OFF) to use ExecuTorch.")
    set(ANIRA_WITH_EXECUTORCH OFF)
endif()

# Android / iOS: the anira backends release ships no LibTorch mobile build (LibTorch
# is desktop-only upstream; the PyTorch mobile path is the ExecuTorch backend).
# LibTorch defaults ON, so a mobile build must opt out of it explicitly.
if((CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "iOS") AND ANIRA_WITH_LIBTORCH)
    message(FATAL_ERROR "LibTorch has no Android/iOS build in the anira backends release. Disable it "
                        "(-DANIRA_WITH_LIBTORCH=OFF) and use the ONNX Runtime, LiteRT or ExecuTorch backend on mobile.")
endif()

# ExecuTorch, LiteRT and TFLite all bundle their own (different) copy of XNNPACK.
# In a fully static anira every backend's archives are linked into one image, where
# the duplicate xnn_* symbols hard-collide at link time (and would cross-bind the
# delegates if they didn't). Shared builds are unaffected: LiteRT/TFLite are then
# self-contained shared libraries.
#
# ExecuTorch, LiteRT and TFLite all bundle their own (different) copy of XNNPACK
# (plus cpuinfo/pthreadpool). In a fully static anira every backend's archives are
# linked into one image, where duplicate xnn_* symbols hard-collide at link time
# (and would cross-bind the delegates if they didn't). Shared builds are unaffected.
#
# The DESKTOP static LiteRT archives are pre-isolated at packaging since backends
# release v2.3.0: only the LiteRt* C API is external, the vendored internals are
# localized (Mach-O/ELF) or renamed (COFF) — see scripts/isolate-static.sh in
# anira-project/backends. Static LiteRT + ExecuTorch therefore just works on
# desktop. (Bring-your-own LiteRT archives via ANIRA_LITERT_ROOTDIR must be
# isolated the same way, or the collision comes back as duplicate-symbol link
# errors.) The legacy TFLite backend and the mobile merged-lib archives are NOT
# isolated, so those combinations keep the ExecuTorch auto-disable (mirroring
# the LibTorch static auto-disable) instead of failing the default build.
if(NOT BUILD_SHARED_LIBS AND ANIRA_WITH_EXECUTORCH AND (ANIRA_WITH_LITERT OR ANIRA_WITH_TFLITE))
    if(NOT ANIRA_WITH_LITERT OR EMSDK_VERSION
       OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
        message(WARNING "ExecuTorch and LiteRT/TFLite bundle conflicting copies of XNNPACK and cannot "
                        "be combined in a fully static anira build (BUILD_SHARED_LIBS=OFF) on this "
                        "platform; disabling ANIRA_WITH_EXECUTORCH. Disable LiteRT/TFLite or build "
                        "shared to use ExecuTorch.")
        set(ANIRA_WITH_EXECUTORCH OFF)
    endif()
endif()

# ExecuTorch's desktop archives are wired through the ExecuTorch CMake package,
# whose config files demand CMake 3.24. Fail early with a clear message (the
# package's own cmake_minimum_required error is cryptic).
if(ANIRA_WITH_EXECUTORCH AND CMAKE_VERSION VERSION_LESS "3.24"
   AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(FATAL_ERROR "The ExecuTorch backend requires CMake >= 3.24 on desktop platforms "
                        "(required by ExecuTorch's exported package config); found ${CMAKE_VERSION}.")
endif()

# iOS: the ONNX Runtime xcframework vendors its own copy of the TfLite C API symbols,
# and the TFLite backend is a single pre-linked framework binary whose symbols all
# load unconditionally — so enabling both on iOS collides (duplicate TfLite* symbols
# at link time). Each works on its own; pair ONNX with LiteRT (the default TF-family
# backend, same .tflite models) instead of the legacy TFLite backend.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" AND ANIRA_WITH_TFLITE AND ANIRA_WITH_ONNXRUNTIME)
    message(FATAL_ERROR "On iOS, ANIRA_WITH_TFLITE and ANIRA_WITH_ONNXRUNTIME cannot be combined: the "
                        "ONNX Runtime xcframework vendors the TfLite C API symbols, which collide with the "
                        "TFLite framework. Use LiteRT alongside ONNX (-DANIRA_WITH_TFLITE=OFF -DANIRA_WITH_LITERT=ON), "
                        "or build TFLite on its own.")
endif()

# WebAssembly (Emscripten): only the ONNX Runtime backend is supported and the
# component targets do not apply. EMSDK_VERSION is set by cmake/detect-emscripten.cmake.
if(DEFINED EMSDK_VERSION)
    if(ANIRA_WITH_EXAMPLES)
        message(FATAL_ERROR "WebAssembly support is not compatible with examples. Set -DANIRA_WITH_EXAMPLES=OFF.")
    elseif(ANIRA_WITH_TESTS)
        message(FATAL_ERROR "WebAssembly support is not compatible with tests. Set -DANIRA_WITH_TESTS=OFF.")
    elseif(ANIRA_WITH_INSTALL)
        message(FATAL_ERROR "WebAssembly support is not compatible with install targets. Set -DANIRA_WITH_INSTALL=OFF.")
    elseif(ANIRA_WITH_LIBTORCH)
        message(FATAL_ERROR "Only the ONNX Runtime backend is supported for WebAssembly. Set -DANIRA_WITH_LIBTORCH=OFF and enable ANIRA_WITH_ONNXRUNTIME.")
    elseif(ANIRA_WITH_TFLITE)
        message(FATAL_ERROR "Only the ONNX Runtime backend is supported for WebAssembly. Set -DANIRA_WITH_TFLITE=OFF and enable ANIRA_WITH_ONNXRUNTIME.")
    elseif(ANIRA_WITH_LITERT)
        message(FATAL_ERROR "Only the ONNX Runtime backend is supported for WebAssembly. Set -DANIRA_WITH_LITERT=OFF and enable ANIRA_WITH_ONNXRUNTIME.")
    elseif(ANIRA_WITH_EXECUTORCH)
        message(FATAL_ERROR "Only the ONNX Runtime backend is supported for WebAssembly. Set -DANIRA_WITH_EXECUTORCH=OFF and enable ANIRA_WITH_ONNXRUNTIME.")
    endif()
endif()
