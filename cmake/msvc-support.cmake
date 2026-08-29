# ==============================================================================
# Windows specific settings
# ==============================================================================

# (The static TFLite archive's TFL_COMPILE_LIBRARY definition rides on anira::tflite,
# see cmake/backends.cmake.)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "You need to specify CMAKE_BUILD_TYPE")
endif()

# anira.dll only exists for a shared build; a static anira (.lib) is baked into the
# consumer, so there is nothing to copy.
if(BUILD_SHARED_LIBS)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        set(ANIRA_DLL "${anira_BINARY_DIR}/${CMAKE_BUILD_TYPE}/anira.dll")
    else()
        set(ANIRA_DLL "${anira_BINARY_DIR}/anira.dll")
    endif()
    list(APPEND ANIRA_SHARED_LIBS_WIN ${ANIRA_DLL})
    # tanh-lib's core component is a DLL in a shared build. anira calls into it
    # (thl::Logger), so every executable linking anira needs it beside it — the
    # generator expression resolves to the right per-generator/config path.
    list(APPEND ANIRA_SHARED_LIBS_WIN "$<TARGET_FILE:tanh::Core>")
endif()

# Add all necessary DLLs to a list for later copying. Only shared backends ship a
# runtime DLL; statically-linked backends are baked into anira.dll, so skip them.
if(ANIRA_WITH_ONNXRUNTIME AND NOT ANIRA_ONNXRUNTIME_IS_STATIC)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_ONNX "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_ONNX})
endif()
if (ANIRA_WITH_TFLITE AND NOT ANIRA_TFLITE_IS_STATIC)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_TFLITE "${ANIRA_TFLITE_SHARED_LIB_PATH}/*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_TFLITE})
endif()
if (ANIRA_WITH_LITERT AND NOT ANIRA_LITERT_IS_STATIC)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_LITERT "${ANIRA_LITERT_SHARED_LIB_PATH}/*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_LITERT})
endif()
if (ANIRA_WITH_LIBTORCH)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_LIBTORCH "${ANIRA_LIBTORCH_SHARED_LIB_PATH}/*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_LIBTORCH})
endif(ANIRA_WITH_LIBTORCH)

# Google Test and Google Benchmark DLLs (only built as DLLs in a shared build; with
# BUILD_SHARED_LIBS=OFF gtest/benchmark are static and there is no .dll to copy). Target
# file generator expressions, not paths: the targets come from the shared
# tanh-tooling module (cmake/tanh/test-deps.cmake) and their build directories are its
# business.
if(ANIRA_WITH_TESTS AND BUILD_SHARED_LIBS)
    list(APPEND ANIRA_SHARED_LIBS_WIN "$<TARGET_FILE:gtest>" "$<TARGET_FILE:gtest_main>")
endif()
if(ANIRA_WITH_BENCHMARK AND BUILD_SHARED_LIBS)
    list(APPEND ANIRA_SHARED_LIBS_WIN "$<TARGET_FILE:benchmark::benchmark>")
    if(NOT ANIRA_WITH_TESTS)
        # gtest_main is linked into anira for the benchmark fixtures as well.
        list(APPEND ANIRA_SHARED_LIBS_WIN "$<TARGET_FILE:gtest>" "$<TARGET_FILE:gtest_main>")
    endif()
endif()

# Make a list of all necessary DLLs for the project
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(ANIRA_SHARED_LIBS_WIN ${ANIRA_SHARED_LIBS_WIN} PARENT_SCOPE)
endif()
