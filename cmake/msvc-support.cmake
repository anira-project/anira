# ==============================================================================
# Windows specific settings
# ==============================================================================

# Define the export symbol for MSVC builds (shared library)
target_compile_definitions(${PROJECT_NAME} PRIVATE ANIRA_EXPORTS)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "You need to specify CMAKE_BUILD_TYPE")
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(ANIRA_DLL "${anira_BINARY_DIR}/${CMAKE_BUILD_TYPE}/anira.dll")
else()
    set(ANIRA_DLL "${anira_BINARY_DIR}/anira.dll")
endif()

list(APPEND ANIRA_SHARED_LIBS_WIN ${ANIRA_DLL})

# Add all necessary DLLs to a list for later copying
# Backend DLLs
if(ANIRA_WITH_ONNXRUNTIME)
    message(STATUS "Test: ${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}")
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_ONNX "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_ONNX})
endif(ANIRA_WITH_ONNXRUNTIME)
if (ANIRA_WITH_TFLITE)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_TFLITE "${ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH}/*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_TFLITE})
endif(ANIRA_WITH_TFLITE)
if (ANIRA_WITH_LIBTORCH)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_LIBTORCH "${ANIRA_LIBTORCH_SHARED_LIB_PATH}*.dll")
    list(APPEND ANIRA_SHARED_LIBS_WIN ${INFERENCE_ENGINE_DLLS_LIBTORCH})
endif(ANIRA_WITH_LIBTORCH)

# Google Benchmark and Google Test DLLs
if (ANIRA_WITH_BENCHMARK)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/gtest.dll")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/gtest_main.dll")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/_deps/benchmark-build/src/${CMAKE_BUILD_TYPE}/benchmark.dll")
    else()
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/gtest.dll")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/bin/gtest_main.dll")
        list(APPEND ANIRA_SHARED_LIBS_WIN "${CMAKE_BINARY_DIR}/_deps/benchmark-build/src/benchmark.dll")
    endif()
endif(ANIRA_WITH_BENCHMARK)

# Make a list of all necessary DLLs for the project
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(ANIRA_SHARED_LIBS_WIN ${ANIRA_SHARED_LIBS_WIN} PARENT_SCOPE)
endif()
