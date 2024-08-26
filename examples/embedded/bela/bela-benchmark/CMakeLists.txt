cmake_minimum_required(VERSION 3.15)

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME bela-benchmark)

project (${PROJECT_NAME} VERSION 0.0.1)

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add_subdirectory(anira) # set this to the path of the anira library if its a submodule of your repository
# list(APPEND CMAKE_PREFIX_PATH "/path/to/anira") # Use this if you use the precompiled version of anira
# find_package(anira REQUIRED)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    defineBelaBenchmark.cpp
	main.cpp
)

target_link_libraries(${PROJECT_NAME} anira::anira)