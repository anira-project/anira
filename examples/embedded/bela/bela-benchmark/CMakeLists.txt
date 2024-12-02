cmake_minimum_required(VERSION 3.15)

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME bela-benchmark)

project (${PROJECT_NAME} VERSION 0.0.1)

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Make sure to install the anira library under /root/anira if using the precompiled version of anira or set ANIRA_WITH_INSTALL to ON when adding the anira library as a subdirectory
# Obviously, this benchmark requires the anira library to be build with the benchmark option enabled

################################################################
# Setting up anira
################################################################

# set(ANIRA_WITH_BENCHMARK ON)
# set(ANIRA_WITH_ONNXRUNTIME OFF) # Not yet supported
# set(ANIRA_WITH_TFLITE OFF) # Not yet supported
# # In case the libs from anira shall be installed
# # This only works when the anira library is a subdirectory of the project
# set(ANIRA_WITH_INSTALL ON)
# add_subdirectory(anira) # set this to the path of the anira library if its a submodule of your repository
# list(APPEND CMAKE_PREFIX_PATH "anira-0.1.0-bela") # Use this if you use the precompiled version of anira
# find_package(anira REQUIRED)

################################################################
# Adding the executable
################################################################

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    defineBelaBenchmark.cpp
	main.cpp
)

target_link_libraries(${PROJECT_NAME} anira::anira)

################################################################
# Installing the executable
################################################################

include(cmake/install.cmake)
