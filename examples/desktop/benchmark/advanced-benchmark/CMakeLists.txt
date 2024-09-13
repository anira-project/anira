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

set (PROJECT_NAME advanced-benchmark)

project (${PROJECT_NAME} VERSION 0.0.1)

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(ANIRA_WITH_BENCHMARK ON)
# add_subdirectory(anira) # set this to the path of the anira library if its a submodule of your repository
# list(APPEND CMAKE_PREFIX_PATH "/path/to/anira") # Use this if you use the precompiled version of anira
# find_package(anira REQUIRED)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    defineAdvancedBenchmark.cpp
	defineTestAdvancedBenchmark.cpp
)

target_link_libraries(${PROJECT_NAME} anira::anira)

# gtest_discover_tests will register a CTest test for each gtest and run them all in parallel with the rest of the Test.
gtest_discover_tests(${PROJECT_NAME} DISCOVERY_TIMEOUT 90)

if (MSVC)
	foreach(DLL ${ANIRA_SHARED_LIBS_WIN})
		add_custom_command(TARGET ${PROJECT_NAME}
				PRE_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${DLL}
				$<TARGET_FILE_DIR:${PROJECT_NAME}>)
	endforeach()
endif (MSVC)
