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

set (PROJECT_NAME steerable-nafx-benchmark)

project (${PROJECT_NAME} VERSION 0.0.1)

option(BUILD_ARCHITECTURE "Architecture to build for")

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_subdirectory(anira) # set this to the path of the anira library if its a submodule of your repository
# list(APPEND CMAKE_PREFIX_PATH "/path/to/anira") # Use this if you use the precompiled version of anira
# find_package(anira REQUIRED)

add_executable(${PROJECT_NAME})

add_compile_definitions(STEERABLENAFX_MODELS_PATH_TENSORFLOW="${CMAKE_SOURCE_DIR}/extras/models/steerable-nafx/models/")
add_compile_definitions(STEERABLENAFX_MODELS_PATH_PYTORCH="${CMAKE_SOURCE_DIR}/extras/models/steerable-nafx/models/")

message("CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})

target_sources(${PROJECT_NAME} PRIVATE
    defineBenchmark.cpp
    defineTest.cpp
    MyPrePostProcessor.h
    MyConfig.h
)

target_link_libraries(${PROJECT_NAME} anira)

# gtest_discover_tests will register a CTest test for each gtest and run them all in parallel with the rest of the Test.
gtest_discover_tests(${PROJECT_NAME} DISCOVERY_TIMEOUT 90)