cmake_minimum_required(VERSION 3.15)

# Sets the minimum macOS version
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum version of the target platform" FORCE) 
	if(CMAKE_OSX_DEPLOYMENT_TARGET)
		message("The minimum macOS version is set to " $CACHE{CMAKE_OSX_DEPLOYMENT_TARGET}.)
	endif()
endif ()

# ==============================================================================
# Define the options for the anira library
# ==============================================================================

# Shall the library be built as a shared library?
option(BUILD_SHARED_LIBS "Build the library as a shared library" ON)

option(ANIRA_WITH_BENCHMARK "Build the library with benchmarking capabilities" ON)

option(ANIRA_WITH_EXTRAS "Build with extras (clone example model repos)" ON)

option(ANIRA_BUILD_EXAMPLES "Add Example Targets (juce plugin, benchmark options, minimal inference examples)" ON)

# Define options for backends
option(ANIRA_WITH_LIBTORCH "Build with LibTorch backend" OFF)
option(ANIRA_WITH_ONNXRUNTIME "Build with ONNX Runtime backend" OFF)
option(ANIRA_WITH_TFLITE "Build with TensorFlow Lite backend" OFF)

option(ANIRA_BACKEND_ALL "Build with BACKEND_ALL" ON)

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION 0.0.1)

option(BUILD_ARCHITECTURE "Architecture to build for")

# for CMAKE_INSTALL_INCLUDEDIR definition
include(GNUInstallDirs)

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(ANIRA_WITH_LIBTORCH)
    set(BACKEND_LIBTORCH ON)
else()
    set(BACKEND_LIBTORCH OFF)
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    set(BACKEND_ONNXRUNTIME ON)
else()
    set(BACKEND_ONNXRUNTIME OFF)
endif()

if(ANIRA_WITH_TFLITE)
    set(BACKEND_TFLITE ON)
else()
    set(BACKEND_TFLITE OFF)
endif()

if(ANIRA_BACKEND_ALL)
    set(BACKEND_LIBTORCH ON)
    set(BACKEND_ONNXRUNTIME ON)
    set(BACKEND_TFLITE ON)
endif()

# ==============================================================================
# Download and install the selected inference engines
# ==============================================================================

set(BACKEND_SOURCES)
set(BACKEND_BUILD_HEADER_DIR)
set(BACKEND_BUILD_LIBRARY_DIRS)

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
    message(STATUS "No backend selected. Please select at least one backend by setting one of the following options to ON: BACKEND_LIBTORCH, BACKEND_ONNXRUNTIME, or BACKEND_TFLITE. For example, add '-DBACKEND_LIBTORCH=ON' to your CMake command line")
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

if(ANIRA_WITH_BENCHMARK)
    target_sources(${PROJECT_NAME}

        PRIVATE
            # TODO: find out why we need to add the header files here, so that they can find the <benchmark/benchmark.h> and <gtest/gtest.h> files
            include/anira/benchmark/ProcessBlockFixture.h
            src/benchmark/ProcessBlockFixture.cpp
    )
endif()

# add the include directories for the backends to the build interface as well as the install interface
foreach(HEADER_DIR ${BACKEND_BUILD_HEADER_DIR})
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
        $<BUILD_INTERFACE:${HEADER_DIR}>
    )
    file(RELATIVE_PATH HEADER_PATH_RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/modules" "${HEADER_DIR}")
    get_filename_component(HEADER_DIRECTORY_PATH "${HEADER_PATH_RELATIVE}" DIRECTORY)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${HEADER_DIRECTORY_PATH}>
    )
endforeach()

# include the public headers of the anira library
target_include_directories(${PROJECT_NAME}
    PUBLIC
    # where top-level project will look for the library's public headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # where external projects will look for the library's public headers
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# include the link directories for all the backends to the build interface as well as the install interface
foreach(LIBRARY_DIR ${BACKEND_BUILD_LIBRARY_DIRS})
    target_link_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${LIBRARY_DIR}>
    )
    file(RELATIVE_PATH LIBRARY_PATH_RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/modules" "${LIBRARY_DIR}")
    get_filename_component(LIBRARY_DIRECTORY_PATH "${LIBRARY_PATH_RELATIVE}" DIRECTORY)
    target_link_directories(${PROJECT_NAME} PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}/${LIBRARY_DIRECTORY_PATH}>
    )
endforeach()

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
    # Backend-specific definitions
    $<$<BOOL:${BACKEND_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${BACKEND_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${BACKEND_TFLITE}>:USE_TFLITE>
)

if (MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ANIRA_EXPORTS)
endif()

# Link backend-specific libraries conditionally
# We use the PUBLIC keyword here to make sure that we don't get the DSO missing from command line error
if(BACKEND_LIBTORCH)
    target_link_libraries(${PROJECT_NAME} PUBLIC torch)
endif()

if(BACKEND_ONNXRUNTIME)
    target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)
endif()

if(BACKEND_TFLITE)
    target_link_libraries(${PROJECT_NAME} PUBLIC tensorflowlite_c)
endif()

if(ANIRA_WITH_BENCHMARK)
    include(FetchContent)

    # This disables the default behavior of adding all targets to the CTest dashboard.
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    # enable ctest
    include(CTest)

    # Externally provided libraries
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE
        GIT_TAG v1.14.0)

    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE
        GIT_TAG v1.8.3)

    # This command ensures that each of the named dependencies are made available to the project by the time it returns. If the dependency has already been populated the command does nothing. Otherwise, the command populates the dependency and then calls add_subdirectory() on the result.
    FetchContent_MakeAvailable(googletest)

    # For benchmark we want to set the BENCMARK_ENABLE_TESTING to OFF therefore we cannot use FetchContent_MakeAvailable()
    # Check if population has already been performed
    FetchContent_GetProperties(benchmark)
    if(NOT benchmark_POPULATED)
        # Fetch the content using previously declared details
        FetchContent_Populate(benchmark)

        # Set custom variables, policies, etc.
        set(BENCHMARK_ENABLE_TESTING OFF)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

        if (APPLE AND (BUILD_ARCHITECTURE STREQUAL "arm64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64"))
        set(HAVE_STD_REGEX ON)
        set(RUN_HAVE_STD_REGEX 1)
        endif()

        # Bring the populated content into the build
        add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})

        # Supress warnings by making include directories system directories
        get_property(BENCHMARK_INCLUDE_DIRS TARGET benchmark PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(benchmark SYSTEM INTERFACE ${BENCHMARK_INCLUDE_DIRS})
    endif()
    
    # enable position independent code because otherwise the library cannot be linked into a shared library
    set_target_properties(gtest PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(benchmark PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${PROJECT_NAME} PUBLIC gtest_main benchmark)

    # include Loads and runs CMake code from the file given. Loads and runs CMake code from the file given.
    include(GoogleTest)
endif()

# ==============================================================================
# Install the library
# ==============================================================================

# define the dircetory where the library will be installed CMAKE_INSTALL_PREFIX
# note that it is not CMAKE_INSTALL_PREFIX we are checking here
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX will be set to ${CMAKE_SOURCE_DIR}/install"
    )
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Where the library will be installed to" FORCE
    )
else()
    message(STATUS "CMAKE_INSTALL_PREFIX was already set to ${CMAKE_INSTALL_PREFIX}")
endif()

# at install the rpath is cleared by default so we have to set it again for the installed shared library to find the other libraries
# in this case we set the rpath to the directories where the other libraries are installed
# $ORIGIN is a special token that gets replaced by the directory of the library at runtime
# from that point we can navigate to the other libraries
# it is a little strange but the onnxruntime library and libtorch work without this setting in the JUCE example... but tensorflowlite does not
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "$ORIGIN/../${TENSORFLOWLITE_DIR_NAME};$ORIGIN/../${ONNXRUNTIME_DIR_NAME};$ORIGIN/../../modules/${LIBTORCH_DIR_NAME}/lib"
)

# the variant with PUBLIC_HEADER property unfortunately does not preserve the folder structure therefore we use the simple install directory command
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/anira
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install the target and create export-set
install(TARGETS ${PROJECT_NAME}
    EXPORT "aniraTargets"
    # these get default values from GNUInstallDirs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME} # lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME} # lib
)

# libtorch has cmake config files that we can use to install the library later with find_package and then just link to it
# if we would implement the install manually we would have to include the include and lib directories to our install interface
if(BACKEND_LIBTORCH)
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/modules/${LIBTORCH_DIR_NAME}"
    )
endif()

# the other ones don't have cmake config files so we have to install them manually
if(BACKEND_ONNXRUNTIME)
    install(DIRECTORY "${ONNXRUNTIME_ROOTDIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${ONNXRUNTIME_DIR_NAME}"
    )
    install(DIRECTORY "${ONNXRUNTIME_ROOTDIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/${ONNXRUNTIME_DIR_NAME}"
    )
endif()

if(BACKEND_TFLITE)
    install(DIRECTORY "${TENSORFLOWLITE_ROOTDIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${TENSORFLOWLITE_DIR_NAME}"
    )
    install(DIRECTORY "${TENSORFLOWLITE_ROOTDIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/${TENSORFLOWLITE_DIR_NAME}"
    )
endif()

# set a debug postfix for the library
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

# ==============================================================================
# Generate config files
# ==============================================================================

# generate and install export file in the folder cmake with the name of the project and namespace
# this generates files called aniraTargets.cmake, aniraTargets-debug.cmake, aniraTargets-release.cmake
install(EXPORT "aniraTargets"
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

include(CMakePackageConfigHelpers)

# create config file from the template file Config.cmake.in and specify the install destination
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

# install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# ==============================================================================
# Add all necessary DLLs to a list for later copying (only for MSVC)
# ==============================================================================

if (MSVC)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_ONNX "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/*.dll")
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_TFLITE "${ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH}/*.dll")
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_LIBTORCH "${ANIRA_LIBTORCH_SHARED_LIB_PATH}*.dll")

    # These shared libs are only available after build
    file(GLOB_RECURSE GOOGLE_BENCHMARK_DLL "${benchmark_BINARY_DIR}/*.dll")
    file(GLOB_RECURSE GOOGLE_TEST_DLL "${CMAKE_BINARY_DIR}/bin/*.dll")
    file(GLOB_RECURSE ANIRA_DLL "${anira_BINARY_DIR}/*anira.dll")

    set(NECESSARY_DLLS
            ${INFERENCE_ENGINE_DLLS_ONNX}
            ${INFERENCE_ENGINE_DLLS_TFLITE}
            ${INFERENCE_ENGINE_DLLS_LIBTORCH}
            ${GOOGLE_BENCHMARK_DLL}
            ${GOOGLE_TEST_DLL}
            ${ANIRA_DLL}
    )
endif (MSVC)

# ==============================================================================
# Build example targets and add extras (clone example model repos)
# ==============================================================================

# First we import the extras folder since we need the compile definitions (model paths) for the examples

if(ANIRA_WITH_EXTRAS)
    add_subdirectory(extras)
endif ()

if(ANIRA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()