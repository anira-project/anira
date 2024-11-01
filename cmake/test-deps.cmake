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


# This command ensures that each of the named dependencies are made available to the project by the time it returns. If the dependency has already been populated the command does nothing. Otherwise, the command populates the dependency and then calls add_subdirectory() on the result.
FetchContent_MakeAvailable(googletest)

# enable position independent code because otherwise the library cannot be linked into a shared library
set_target_properties(gtest PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME} PUBLIC gtest_main)

# include Loads and runs CMake code from the file given. Loads and runs CMake code from the file given.
include(GoogleTest)
