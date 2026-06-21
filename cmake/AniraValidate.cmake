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
    endif()
endif()
