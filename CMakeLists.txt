cmake_minimum_required(VERSION 3.15)

# ==============================================================================
# Define the options for the anira library
# ==============================================================================

# Shall the library be built as a shared library?
option(BUILD_SHARED_LIBS "Build the library as a shared library" ON)
option(ANIRA_WITH_BENCHMARK "Build the library with benchmarking capabilities" OFF)
option(ANIRA_WITH_EXAMPLES "Add Example Targets for desktop plattforms (juce plugin, benchmarks, minimal inference and model examples)" OFF)
option(ANIRA_WITH_BELA_EXAMPLE "Add Example Targets for bela plattform (includes a model)" OFF)
option(ANIRA_WITH_INSTALL "Add install targets" OFF)
option(ANIRA_WITH_TESTS "Add Build Tests" OFF)

# Select the backends for the inference engine, multiple backends can be selected
option(ANIRA_WITH_LIBTORCH "Build with LibTorch backend" ON)
option(ANIRA_WITH_ONNXRUNTIME "Build with ONNX Runtime backend" ON)
option(ANIRA_WITH_TFLITE "Build with TensorFlow Lite backend" ON)

# Shall semaphores be used for synchronization instead of atomic variables?
option(ANIRA_WITH_SEMAPHORE "Use semaphores for synchronization instead of atomic variables" OFF)

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION 0.1.3)

if(APPLE)
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    else()
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(CMAKE_OSX_ARCHITECTURES "arm64")
        elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(CMAKE_OSX_ARCHITECTURES "x86_64")
        else()
            message(FATAL_ERROR "CMAKE_OSX_ARCHITECTURES not set to arm64 or x86_64. Please select one of the architectures as universal binaries are not supported by all backends.")
        endif()
    endif()
endif()

# Print the processor architecture selected for build. Defaults to CMAKE_HOST_SYSTEM_PROCESSOR when no crosscompile tolchain is defined. CMAKE_HOST_SYSTEM_PROCESSOR is only defined after the project() call. For OSX this value can be overwritten by the CMAKE_OSX_ARCHITECTURES.
message(STATUS "Selected processor architecture: ${CMAKE_SYSTEM_PROCESSOR}")

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

message(STATUS "Selected backends:")

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

if(ANIRA_WITH_SEMAPHORE)
    message(STATUS "Using semaphores for thread synchronization.")
else()
    message(STATUS "Using atomic variables for thread synchronization.")
endif()

## ==============================================================================
# Fetch threadsafe queue
# ==============================================================================

include(FetchContent)

FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG v1.0.4
)

FetchContent_MakeAvailable(concurrentqueue)


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
        src/scheduler/AniraContext.cpp
        src/scheduler/SessionElement.cpp

        # Utils
        src/utils/AudioBuffer.cpp
        src/utils/RingBuffer.cpp

        # Interface
        src/InferenceHandler.cpp
        src/PrePostProcessor.cpp
        src/InferenceConfig.cpp

        # System
        src/system/HighPriorityThread.cpp
)

# add the include directories for the backends to the build interface, public because the anira headers include the backend headers
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
    # Semaphore definitions
    $<$<BOOL:${ANIRA_WITH_SEMAPHORE}>:USE_SEMAPHORE>
    # Version number
    -DANIRA_VERSION="${PROJECT_VERSION}"
)

target_link_libraries(${PROJECT_NAME} PUBLIC concurrentqueue)

if(ANIRA_WITH_LIBTORCH)
    # The find_package(Torch) adds the libraries libc10.so and libkineto.a as full paths to ${TORCH_LIBRARIES}. This is no problem when we add anira as a subdirectory to another project, but when we install the library, the torch libraries will be link targets of the anira library with full paths and hence not found on other systems. Therefore, we link those libs privately and only add the torch target publicly.
    # Also until cmake 3.26, there is a bug where the torch_cpu library is not found when linking publicly https://gitlab.kitware.com/cmake/cmake/-/issues/24163 and anira is added as a subdirectory to another project, see
    # But this is necessary for when we install the library since otherwise symbols are not found
    if (CMAKE_VERSION VERSION_LESS "3.26.0" AND NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR))
        target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
        set(TORCH_LIBRARIES_ALL_PRIVATE TRUE)
    else()
        foreach(TORCH_LIB ${TORCH_LIBRARIES})
            if(TORCH_LIB STREQUAL "torch" OR TORCH_LIB STREQUAL "torch_library")
                target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIB})
            else()
                target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIB})
            endif()
        endforeach()
    endif()
endif()

# The onnxruntime library requires PUBLIC linking because otherwise "_OrtGetApiBase" symbol is not found
if(ANIRA_WITH_ONNXRUNTIME)
    target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)
endif()

if(ANIRA_WITH_TFLITE)
    target_link_libraries(${PROJECT_NAME} PUBLIC tensorflowlite_c)
endif()

if(ANIRA_WITH_BENCHMARK OR ANIRA_WITH_TESTS)
    include(cmake/test-deps.cmake)
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
if(ANIRA_WITH_EXAMPLES OR ANIRA_WITH_BELA_EXAMPLE)
    add_subdirectory(extras)
    add_subdirectory(examples)
endif()

if (ANIRA_WITH_TESTS)
    add_subdirectory(test)
endif()