# ==============================================================================
# Windows specific settings
# ==============================================================================

# Define the export symbol for MSVC builds (shared library)
target_compile_definitions(${PROJECT_NAME} PRIVATE ANIRA_EXPORTS)

# Anira DLL
file(GLOB_RECURSE ANIRA_DLL "${anira_BINARY_DIR}/*anira.dll")

# Make a list of all necessary DLLs for the project
set(NECESSARY_DLLS ${ANIRA_DLL} PARENT_SCOPE)

# get_directory_property(hasParent PARENT_DIRECTORY)
# if(hasParent)
#     set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/" PARENT_SCOPE)
# else ()
#     set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/")
# endif()

# Add all necessary DLLs to a list for later copying (only for MSVC)
# Backend DLLs
if(ANIRA_WITH_ONNXRUNTIME)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_ONNX "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/*.dll")
    list(APPEND NECESSARY_DLLS ${INFERENCE_ENGINE_DLLS_ONNX})
endif(ANIRA_WITH_ONNXRUNTIME)
if (ANIRA_WITH_TFLITE)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_TFLITE "${ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH}/*.dll")
    list(APPEND NECESSARY_DLLS ${INFERENCE_ENGINE_DLLS_TFLITE})
endif(ANIRA_WITH_TFLITE)
if (ANIRA_WITH_LIBTORCH)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_LIBTORCH "${ANIRA_LIBTORCH_SHARED_LIB_PATH}*.dll")
    list(APPEND NECESSARY_DLLS ${INFERENCE_ENGINE_DLLS_LIBTORCH})
endif(ANIRA_WITH_LIBTORCH)

# Google Benchmark and Google Test DLLs
if (ANIRA_WITH_BENCHMARK)
    file(GLOB_RECURSE GOOGLE_BENCHMARK_DLL "${benchmark_BINARY_DIR}/*.dll")
    file(GLOB_RECURSE GOOGLE_TEST_DLL "${CMAKE_BINARY_DIR}/bin/*.dll")
    list(APPEND NECESSARY_DLLS ${GOOGLE_BENCHMARK_DLL})
    list(APPEND NECESSARY_DLLS ${GOOGLE_TEST_DLL})
endif(ANIRA_WITH_BENCHMARK)