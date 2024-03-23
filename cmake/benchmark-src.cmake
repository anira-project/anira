# ==============================================================================
# Sources related to the benchmarking options
# ==============================================================================

target_sources(${PROJECT_NAME}

PRIVATE
    # TODO: find out why we need to add the header files here, so that they can find the <benchmark/benchmark.h> and <gtest/gtest.h> files
    include/anira/benchmark/ProcessBlockFixture.h
    src/benchmark/ProcessBlockFixture.cpp
)

# This disables the default behavior of adding all targets to the CTest dashboard.
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

include(FetchContent)

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

    if (APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64"))
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