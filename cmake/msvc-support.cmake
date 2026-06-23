# ==============================================================================
# Windows specific settings
# ==============================================================================

# Define the export symbol for MSVC builds (shared library)
target_compile_definitions(${PROJECT_NAME} PRIVATE ANIRA_EXPORTS)

# When built statically, tell anira's export header (PUBLIC, so consumers see it
# too) to skip dllexport/dllimport decoration — a static lib has no import stubs.
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ANIRA_STATIC_DEFINE)
endif()

# The TFLite C API headers default to __declspec(dllimport) on Windows; linking the
# static TFLite lib then leaves __imp_TfLite* unresolved (no import stubs). Defining
# TFL_COMPILE_LIBRARY switches the decoration to a direct (static) reference. PUBLIC
# so a consumer including anira's TFLite processor header agrees. (ONNX uses a
# function-pointer table and LiteRT's static lib ships import stubs, so only the
# legacy TFLite backend needs this.)
if(ANIRA_WITH_TFLITE AND ANIRA_TFLITE_IS_STATIC)
    target_compile_definitions(${PROJECT_NAME} PUBLIC TFL_COMPILE_LIBRARY)
endif()

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

# Google Benchmark and Google Test DLLs (only built as DLLs in a shared build; with
# BUILD_SHARED_LIBS=OFF gtest/benchmark are static and there is no .dll to copy).
if ((ANIRA_WITH_TESTS OR ANIRA_WITH_BENCHMARK) AND BUILD_SHARED_LIBS)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/gtest.dll")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/gtest_main.dll")
    else()
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/gtest.dll")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/gtest_main.dll")
    endif()
endif()

if (ANIRA_WITH_BENCHMARK AND BUILD_SHARED_LIBS)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/_deps/benchmark-build/src/${CMAKE_BUILD_TYPE}/benchmark.dll")
    else()
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/_deps/benchmark-build/src/benchmark.dll")
    endif()
endif()

# Make a list of all necessary DLLs for the project
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(ANIRA_SHARED_LIBS_WIN ${ANIRA_SHARED_LIBS_WIN} PARENT_SCOPE)
endif()
