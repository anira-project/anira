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

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION 0.0.1)

option(BUILD_ARCHITECTURE "Architecture to build for")

# for CMAKE_INSTALL_INCLUDEDIR definition
include(GNUInstallDirs)

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Shall the library be built as a shared library?
option(ANIRA_BUILD_DYNAMIC_LIBRARY "Build the library as a shared library" ON)

# Define options for backends
option(ANIRA_WITH_LIBTORCH "Build with LibTorch backend" OFF)
option(ANIRA_WITH_ONNXRUNTIME "Build with ONNX Runtime backend" OFF)
option(ANIRA_WITH_TFLITE "Build with TensorFlow Lite backend" OFF)
option(ANIRA_WITH_ALL "Build with all backends" OFF)

option(ANIRA_BACKEND_ALL "Build with BACKEND_ALL" ON)

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
    message(FATAL_ERROR "No backend selected. Please select at least one backend by setting one of the following options to ON: BACKEND_LIBTORCH, BACKEND_ONNXRUNTIME, or BACKEND_TFLITE. For example, add '-DBACKEND_LIBTORCH=ON' to your CMake command line")
endif()

# ==============================================================================
# Build the library
# ==============================================================================

# add the library as a shared or static library
if (ANIRA_BUILD_DYNAMIC_LIBRARY)
    add_library(${PROJECT_NAME} SHARED)
    message(STATUS "Building anira as a shared library")
else()
    add_library(${PROJECT_NAME} STATIC)
    message(STATUS "Building anira as static library")
endif()

# enable position independent code because otherwise the library cannot be linked into a shared library
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add an alias so that the project can be used with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

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
    message(STATUS "Install header dir: ${CMAKE_INSTALL_INCLUDEDIR}/${HEADER_DIRECTORY_PATH}")
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
    message(STATUS "Install library path: ${CMAKE_INSTALL_LIBDIR}/${LIBRARY_DIRECTORY_PATH}")
endforeach()

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
    # Backend-specific definitions
    $<$<BOOL:${BACKEND_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${BACKEND_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${BACKEND_TFLITE}>:USE_TFLITE>
)

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

# ==============================================================================
# Install the library
# ==============================================================================

# define the dircetory where the library will be installed CMAKE_INSTALL_PREFIX
# note that it is not CMAKE_INSTALL_PREFIX we are checking here
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX is not set\n"
        "Default value: ${CMAKE_INSTALL_PREFIX}\n"
        "Will set it to ${CMAKE_SOURCE_DIR}/install"
    )
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Where the library will be installed to" FORCE
    )
else()
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX was already set\n"
        "Current value: ${CMAKE_INSTALL_PREFIX}"
    )
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
