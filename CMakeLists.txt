cmake_minimum_required(VERSION 3.15)

# ==============================================================================
# Define the options for the anira library
# ==============================================================================

# Shall the library be built as a shared library?
option(BUILD_SHARED_LIBS "Build the library as a shared library" ON)
option(ANIRA_WITH_BENCHMARK "Build the library with benchmarking capabilities" ON)
option(ANIRA_WITH_EXTRAS "Build with extras (clone example model repos)" OFF)
option(ANIRA_WITH_EXAMPLES "Add Example Targets (juce plugin, benchmarks, minimal inference examples)" OFF)
option(ANIRA_WITH_INSTALL "Add install targets" ON)

# Select the backends for the inference engine, multiple backends can be selected
option(ANIRA_WITH_LIBTORCH "Build with LibTorch backend" ON)
option(ANIRA_WITH_ONNXRUNTIME "Build with ONNX Runtime backend" ON)
option(ANIRA_WITH_TFLITE "Build with TensorFlow Lite backend" ON)

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION 0.0.1)

# Sets the minimum macOS version, c++20 is only available from macOS 11.0
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum version of the target platform" FORCE) 
	if(CMAKE_OSX_DEPLOYMENT_TARGET)
		message("The minimum macOS version is set to " $CACHE{CMAKE_OSX_DEPLOYMENT_TARGET}.)
	endif()
endif ()

# Sets the cpp language minimum, 20 because of new semaphore types
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==============================================================================
# Download and install the selected inference engines
# ==============================================================================

if(NOT ANIRA_WITH_LIBTORCH AND NOT ANIRA_WITH_ONNXRUNTIME AND NOT ANIRA_WITH_TFLITE)
    message(FATAL_ERROR "No backend selected. Please select at least one backend by setting one of the following options to ON: ANIRA_WITH_LIBTORCH, ANIRA_WITH_ONNXRUNTIME, or ANIRA_WITH_TFLITE. For example, add '-DANIRA_WITH_LIBTORCH=ON' to your CMake command line.")
endif()

set(BACKEND_SOURCES)
set(BACKEND_BUILD_HEADER_DIRS)
set(BACKEND_BUILD_LIBRARY_DIRS)

if(ANIRA_WITH_LIBTORCH)
    include(cmake/SetupLibTorch.cmake)
    list(APPEND BACKEND_SOURCES src/backends/LibTorchProcessor.cpp)
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    include(cmake/SetupOnnxRuntime.cmake)
    list(APPEND BACKEND_SOURCES src/backends/OnnxRuntimeProcessor.cpp)
endif()

if(ANIRA_WITH_TFLITE)
    include(cmake/SetupTensorflowLite.cmake)
    list(APPEND BACKEND_SOURCES src/backends/TFLiteProcessor.cpp)
endif()

# ==============================================================================
# Build the library
# ==============================================================================

# add the library as a shared or static library depending on the option BUILD_SHARED_LIBS
add_library(${PROJECT_NAME})

# enable position independent code because otherwise the static library cannot be linked into a shared library
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add an alias so that the project can be used with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE

        # Backend
        src/backends/BackendBase.cpp
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

# add the include directories for the backends to the build interface
foreach(HEADER_DIR ${BACKEND_BUILD_HEADER_DIRS})
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
        $<BUILD_INTERFACE:${HEADER_DIR}>
    )
endforeach()

# include the public headers of the anira library in the top-level project include directory
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# include the link directories for all the backends to the build interface
foreach(LIBRARY_DIR ${BACKEND_BUILD_LIBRARY_DIRS})
    target_link_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${LIBRARY_DIR}>
    )
endforeach()

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
    # Backend-specific definitions
    $<$<BOOL:${ANIRA_WITH_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${ANIRA_WITH_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${ANIRA_WITH_TFLITE}>:USE_TFLITE>
)

# Link backend-specific libraries conditionally
# We use the PUBLIC keyword here to make sure that we don't get the DSO missing from command line error
if(ANIRA_WITH_LIBTORCH)
    target_link_libraries(${PROJECT_NAME} PUBLIC torch)
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)
endif()

if(ANIRA_WITH_TFLITE)
    target_link_libraries(${PROJECT_NAME} PUBLIC tensorflowlite_c)
endif()

if(ANIRA_WITH_BENCHMARK)
    include(cmake/benchmark-src.cmake)
endif()

# ==============================================================================
# Add install targets for the library
# ==============================================================================

if(ANIRA_WITH_INSTALL)
    include(cmake/install.cmake)
endif()

# ==============================================================================
# Add support for MSVC
# ==============================================================================

if (MSVC)
    include(cmake/msvc-support.cmake)
endif()

# ==============================================================================
# Build example targets and add extras (clone example model repos)
# ==============================================================================

# First we import the extras folder since we need the compile definitions (model paths) for the examples
if(ANIRA_WITH_EXTRAS)
    add_subdirectory(extras)
endif ()

if(ANIRA_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()