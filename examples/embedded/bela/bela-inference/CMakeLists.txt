cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "bela-inference")

project(${PROJECT_NAME} VERSION 0.0.1)

message(STATUS "Building project ${PROJECT_NAME}")

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Make sure to install the anira library under /root/anira if using the precompiled version of anira or set ANIRA_WITH_INSTALL to ON when adding the anira library as a subdirectory

################################################################
# Setup required by Bela
################################################################

add_compile_options(
    -march=armv7-a
    -mtune=cortex-a8
    -mfloat-abi=hard
    -mfpu=neon
    -Wno-psabi
)

add_compile_options(
    -O3
    -g 
    -fPIC
    -ftree-vectorize
    -ffast-math
)

add_compile_definitions(DXENOMAI_SKIN_posix)

################################################################

find_library(COBALT_LIB REQUIRED
    NAMES cobalt libcobalt
    HINTS "${CMAKE_SYSROOT}/usr/xenomai/lib"
)

find_library(NEON_LIB REQUIRED
    NAMES NE10 libNE10
    HINTS "${CMAKE_SYSROOT}/usr/lib"
)

find_library(MATHNEON_LIB REQUIRED
    NAMES mathneon libmathneon
    HINTS "${CMAKE_SYSROOT}/usr/lib"
)

################################################################
# Setting up anira
################################################################

# set(ANIRA_WITH_SEMAPHORE OFF)
# set(ANIRA_WITH_ONNXRUNTIME OFF) # Not yet supported
# set(ANIRA_WITH_TFLITE OFF) # Not yet supported
# # In case the libs from anira shall be installed
# # This only works when the anira library is a subdirectory of the project
# set(ANIRA_WITH_INSTALL ON)
# add_subdirectory(anira) # set this to the path of the anira library if its a submodule of your repository
# list(APPEND CMAKE_PREFIX_PATH "anira-0.1.0-bela") # Use this if you use the precompiled version of anira
# find_package(anira REQUIRED)

set(EXE_NAME ${PROJECT_NAME})

file(GLOB SRC_FILES *.cpp)

add_executable(${EXE_NAME} ${SRC_FILES})

################################################################
# Adding include directories from cross-compilation toolchain
################################################################

target_include_directories(
    ${EXE_NAME} PRIVATE ${BELA_ROOT} ${BELA_ROOT}/include 
)

################################################################
# Linking the required libraries
################################################################

target_link_libraries(
    ${EXE_NAME}
    PRIVATE
    ${BELA_ROOT}/lib/libbelafull.so
    anira::anira
    ${COBALT_LIB}
    ${NEON_LIB}
    ${MATHNEON_LIB}
    dl
    prussdrv
    asound
    atomic
    sndfile
    pthread
    rt
)

################################################################
# Installing the executable
################################################################

include(cmake/install.cmake)
