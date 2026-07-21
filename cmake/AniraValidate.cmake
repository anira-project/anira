# ==============================================================================
# AniraValidate.cmake — cross-option validation guards, in one place.
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

# LibTorch ships shared-only upstream (and its bundled XNNPACK collides with static
# LiteRT), so it cannot be linked into a fully static anira. Auto-disable it there.
if(NOT BUILD_SHARED_LIBS AND ANIRA_WITH_LIBTORCH)
    message(WARNING "LibTorch is shared-only and cannot be linked into a fully static anira build "
                    "(BUILD_SHARED_LIBS=OFF); disabling ANIRA_WITH_LIBTORCH. Build shared to use LibTorch.")
    set(ANIRA_WITH_LIBTORCH OFF)
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
# On desktop platforms the LiteRT + ExecuTorch combination is resolved instead
# of refused (see anira_localize_static_archive in AniraBackends.cmake;
# ANIRA_LITERT_LOCALIZE tells the link site to use it):
#   - Apple/ELF: LiteRT's archive is partially linked into one object whose
#     only remaining global symbols are the LiteRt* C API, demoting its
#     vendored XNNPACK/cpuinfo/pthreadpool internals to local symbols so
#     ExecuTorch's copy is the only global one.
#   - Windows (COFF has no partial link): those internals are renamed with an
#     anira_litert_ prefix instead — definitions and references consistently,
#     via llvm-objcopy --redefine-syms — which needs llvm-nm/llvm-objcopy
#     (shipped with LLVM; also bundled with Visual Studio's Clang tools).
# Where the tooling is missing, and for the legacy TFLite backend and the
# mobile merged-lib paths, auto-disable ExecuTorch as before (mirroring the
# LibTorch static auto-disable) instead of failing the default build.
set(ANIRA_LITERT_LOCALIZE FALSE)
if(NOT BUILD_SHARED_LIBS AND ANIRA_WITH_EXECUTORCH AND (ANIRA_WITH_LITERT OR ANIRA_WITH_TFLITE))
    if(ANIRA_WITH_LITERT AND NOT EMSDK_VERSION
       AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        if(WIN32)
            # Visual Studio bundles the tools when the Clang component is
            # installed; a standalone LLVM install works as well.
            find_program(ANIRA_LLVM_NM llvm-nm
                HINTS "$ENV{ProgramFiles}/LLVM/bin"
                      "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/x64/bin"
                      "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/ARM64/bin"
                      "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/bin")
            find_program(ANIRA_LLVM_OBJCOPY llvm-objcopy
                HINTS "$ENV{ProgramFiles}/LLVM/bin"
                      "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/x64/bin"
                      "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/ARM64/bin"
                      "$ENV{VSINSTALLDIR}/VC/Tools/Llvm/bin")
            if(ANIRA_LLVM_NM AND ANIRA_LLVM_OBJCOPY)
                set(ANIRA_LITERT_LOCALIZE TRUE)
                message(STATUS "anira: static LiteRT + ExecuTorch — LiteRT's non-API symbols "
                               "will be renamed (anira_litert_ prefix) to avoid the XNNPACK "
                               "symbol clash (using ${ANIRA_LLVM_OBJCOPY}).")
            else()
                message(WARNING "ExecuTorch and LiteRT both bundle XNNPACK, whose symbols collide in a "
                                "fully static anira build (BUILD_SHARED_LIBS=OFF). Resolving this on "
                                "Windows needs llvm-nm and llvm-objcopy, which were not found — install "
                                "LLVM (or Visual Studio's 'C++ Clang tools') and reconfigure. Disabling "
                                "ANIRA_WITH_EXECUTORCH for now.")
                set(ANIRA_WITH_EXECUTORCH OFF)
            endif()
        else()
            set(ANIRA_LITERT_LOCALIZE TRUE)
            message(STATUS "anira: static LiteRT + ExecuTorch — LiteRT's archive will be "
                           "localized to its LiteRt* API to avoid the XNNPACK symbol clash.")
        endif()
    else()
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
# component targets do not apply. EMSDK_VERSION is set by cmake/DetectEmscripten.cmake.
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
