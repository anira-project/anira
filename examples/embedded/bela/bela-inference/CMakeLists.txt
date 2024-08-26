cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "bela-inference")

project(${PROJECT_NAME} VERSION 0.0.1)

message(STATUS "Building project ${PROJECT_NAME}")

# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

####################################

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

####################################

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

####################################

set(EXE_NAME ${PROJECT_NAME})

file(GLOB SRC_FILES *.cpp)

add_executable(${EXE_NAME} ${SRC_FILES})

target_include_directories(
    ${EXE_NAME} PRIVATE ${BELA_ROOT} ${BELA_ROOT}/include 
)

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