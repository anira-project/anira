cmake_minimum_required(VERSION 3.15)

# Sets the minimum macOS version
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum version of the target platform" FORCE) 
	if(CMAKE_OSX_DEPLOYMENT_TARGET)
		message("The minimum macOS version is set to " $CACHE{CMAKE_OSX_DEPLOYMENT_TARGET}.)
	endif()
endif ()

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME aria)

project (${PROJECT_NAME} VERSION 0.0.1)

option(BUILD_ARCHITECTURE "Architecture to build for")

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Shall the library be built as a shared library?
option(BUILD_SHARED_LIBS "Build the library as a shared library" ON)

# Define options for backends
option(WITH_LIBTORCH "Build with LibTorch backend" OFF)
option(WITH_ONNXRUNTIME "Build with ONNX Runtime backend" OFF)
option(WITH_TFLITE "Build with TensorFlow Lite backend" OFF)
option(WITH_ALL "Build with all backends" OFF)

option(BACKEND_ALL "Build with BACKEND_ALL" ON)

if(BACKEND_ALL)
    set(BACKEND_LIBTORCH ON)
    set(BACKEND_ONNXRUNTIME ON)
    set(BACKEND_TFLITE ON)
endif()

# ==============================================================================
# Download and install the selected inference engines
# ==============================================================================

set(BACKEND_SOURCES)

if(BACKEND_LIBTORCH)
    include(cmake/SetupLibTorch.cmake)
    list(APPEND BACKEND_SOURCES src/backends/LibTorchProcessor.cpp)
endif()

if(BACKEND_ONNXRUNTIME)
    include(cmake/SetupOnnxRuntime.cmake)
    list(APPEND BACKEND_SOURCES src/backends/OnnxRuntimeProcessor.cpp)
endif()

if(BACKEND_TFLITE)
    include(cmake/SetupTensorflowLite.cmake)
    list(APPEND BACKEND_SOURCES src/backends/TFLiteProcessor.cpp)
endif()

if(NOT BACKEND_LIBTORCH AND NOT BACKEND_ONNXRUNTIME AND NOT BACKEND_TFLITE)
    message(FATAL_ERROR "No backend selected. Please select at least one backend by setting one of the following options to ON: BACKEND_LIBTORCH, BACKEND_ONNXRUNTIME, or BACKEND_TFLITE. For example, add '-DBACKEND_LIBTORCH=ON' to your CMake command line")
endif()

# ==============================================================================
# Build the library
# ==============================================================================

# No STATIC or SHARED keyword is needed here, since the BUILD_SHARED_LIBS option is set
add_library(${PROJECT_NAME})

# target_sources(${PROJECT_NAME} PRIVATE ${BACKEND_SOURCES})

target_sources(${PROJECT_NAME}
    PRIVATE

        # Backend
        ${BACKEND_SOURCES}

        # Scheduler
        src/scheduler/InferenceManager.cpp
        src/scheduler/InferenceThread.cpp
        src/scheduler/InferenceThreadPool.cpp
        src/scheduler/SessionElement.cpp

        # Utils
        src/utils/RingBuffer.cpp

        # Interface
        src/InferenceHandler.cpp
        src/PrePostProcessor.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    # where top-level project will look for the library's public headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # where external projects will look for the library's public headers
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    # Backend-specific definitions
    $<$<BOOL:${BACKEND_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${BACKEND_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${BACKEND_TFLITE}>:USE_TFLITE>
)

# Link backend-specific libraries conditionally
if(BACKEND_LIBTORCH)
    target_link_libraries(${PROJECT_NAME} PRIVATE torch)
endif()

if(BACKEND_ONNXRUNTIME)
    target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime)
endif()

if(BACKEND_TFLITE)
    target_link_libraries(${PROJECT_NAME} PRIVATE tensorflowlite_c)
endif()

if (MSVC)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS "modules/*.dll")
    foreach(FORMAT IN LISTS FORMATS_TO_BUILD)
        string(CONCAT CONCATENATED_TARGET ${PROJECT_NAME}_${FORMAT})
        add_custom_command(TARGET ${CONCATENATED_TARGET}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${INFERENCE_ENGINE_DLLS}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/${FORMAT})
    endforeach()
endif (MSVC)

# ==============================================================================
# Install the library
# ==============================================================================

# # note that it is not CMAKE_INSTALL_PREFIX we are checking here
# if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#     message(
#         STATUS
#         "CMAKE_INSTALL_PREFIX is not set\n"
#         "Default value: ${CMAKE_INSTALL_PREFIX}\n"
#         "Will set it to ${CMAKE_SOURCE_DIR}/install"
#     )
#     set(CMAKE_INSTALL_PREFIX
#         "${CMAKE_SOURCE_DIR}/install"
#         CACHE PATH "Where the library will be installed to" FORCE
#     )
# else()
#     message(
#         STATUS
#         "CMAKE_INSTALL_PREFIX was already set\n"
#         "Current value: ${CMAKE_INSTALL_PREFIX}"
#     )
# endif()

# # for CMAKE_INSTALL_INCLUDEDIR definition
# include(GNUInstallDirs)

# # without it public headers won't get installed
# set(public_headers
#     include/
# )

# # the variant with PUBLIC_HEADER property unfortunately does not preserve the folder structure
# # set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")
# # so instead we iterate through public headers and install them "manually"
# foreach(header ${public_headers})
#     file(RELATIVE_PATH header_file_path "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}" "${header}")
#     get_filename_component(header_directory_path "${header_file_path}" DIRECTORY)
#     install(
#         FILES ${header}
#         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${header_directory_path}"
#     )
# endforeach()

# # ==============================================================================

# # Test setup

# # ==============================================================================


# # Export only public symbols
# # We set this option for all our libraries since the google benchmark lib are setup with hidden visibility
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
# set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# # enable ctest
# include(CTest)

# # TODO start and stop executable test
# # This module enables populating content at configure time via any method supported by the ExternalProject module. Whereas ExternalProject_Add() downloads at build time, the FetchContent module makes content available immediately, allowing the configure step to use the content in commands like add_subdirectory(), include() or file() operations.

# # Externally provided libraries
# FetchContent_Declare(googletest
#     GIT_REPOSITORY https://github.com/google/googletest.git
#     GIT_PROGRESS TRUE
#     GIT_SHALLOW TRUE
#     GIT_TAG v1.14.0)

# FetchContent_Declare(benchmark
#     GIT_REPOSITORY https://github.com/google/benchmark.git
#     GIT_PROGRESS TRUE
#     GIT_SHALLOW TRUE
#     GIT_TAG v1.8.3)

# # This command ensures that each of the named dependencies are made available to the project by the time it returns. If the dependency has already been populated the command does nothing. Otherwise, the command populates the dependency and then calls add_subdirectory() on the result.
# FetchContent_MakeAvailable(googletest)

# # For benchmark we want to set the BENCMARK_ENABLE_TESTING to OFF therefore we cannot use FetchContent_MakeAvailable()
# # Check if population has already been performed
# FetchContent_GetProperties(benchmark)
# if(NOT benchmark_POPULATED)
#     # Fetch the content using previously declared details
#     FetchContent_Populate(benchmark)

#     # Set custom variables, policies, etc.
#     set(BENCHMARK_ENABLE_TESTING OFF)

#     if (APPLE AND (BUILD_ARCHITECTURE STREQUAL "arm64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64"))
#     set(HAVE_STD_REGEX ON)
#     set(RUN_HAVE_STD_REGEX 1)
#     endif()

#     # Bring the populated content into the build
#     add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})

#     # Supress warnings by making include directories system directories
#     get_property(BENCHMARK_INCLUDE_DIRS TARGET benchmark PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
#     target_include_directories(benchmark SYSTEM INTERFACE ${BENCHMARK_INCLUDE_DIRS})
# endif()

# # get all test cpp and header files
# file(GLOB_RECURSE TEST_SOURCES_${TARGET_NAME} CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h)
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/test PREFIX "test_${TARGET_NAME}" FILES ${TEST_SOURCES_${TARGET_NAME}})

# # Setup the test executable
# add_executable(Test_${TARGET_NAME} ${TEST_SOURCES_${TARGET_NAME}})
# set_property(TARGET Test_${TARGET_NAME} PROPERTY CXX_STANDARD 20)

# # Add include directories for all folders in test
# file(GLOB_RECURSE TEST_DIRS_${TARGET_NAME} LIST_DIRECTORIES true ${CMAKE_CURRENT_LIST_DIR}/test)

# foreach (DIR ${TEST_DIRS}_${TARGET_NAME})
#     if (IS_DIRECTORY ${DIR})
#         target_include_directories(Test_${TARGET_NAME} PRIVATE ${DIR})
#     endif ()
# endforeach ()

# # Link the test executable against gtest_main and google benchmark
# target_link_libraries(Test_${TARGET_NAME} PRIVATE gtest_main benchmark)

# # We can't link again to the shared juce target without ODL violations (https://github.com/sudara/pamplejuce/issues/31, https://forum.juce.com/t/windows-linker-issue-on-develop/55524/2)
# # Therefore we steal the compile definitions and include directories from the main target and pass them to our test target
# # Since we linked the shared juce targets in PRIVATE mode, they are not linked to the test target again
# target_link_libraries(Test_${TARGET_NAME} PRIVATE ${TARGET_NAME})
# target_compile_definitions(Test_${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},COMPILE_DEFINITIONS>)
# # Use SYSTEM to suppress warnings
# target_include_directories(Test_${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>)

# # include Loads and runs CMake code from the file given. Loads and runs CMake code from the file given.
# include(GoogleTest)

# if (BUILD_ARCHITECTURE STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR OR BUILD_ARCHITECTURE STREQUAL OFF)
#     # gtest_discover_tests will register a CTest test for each gtest and run them all in parallel with the rest of the Test.
#     gtest_discover_tests(Test_${TARGET_NAME} DISCOVERY_TIMEOUT 90)
# endif ()

# if (MSVC)
#     set(TEST_TARGET "Test_${TARGET_NAME}")
#     add_custom_command(TARGET ${TEST_TARGET}
#             PRE_BUILD
#             COMMAND ${CMAKE_COMMAND} -E copy_if_different
#             ${INFERENCE_ENGINE_DLLS}
#             $<TARGET_FILE_DIR:${TEST_TARGET}>)
# endif (MSVC)

