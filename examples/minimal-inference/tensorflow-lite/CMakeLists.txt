cmake_minimum_required(VERSION 3.16)
project(minimal-tflite C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-tflite
  minimal-tflite.cpp
)

target_link_libraries(minimal-tflite
  anira::anira
)

if (MSVC)
    file(GLOB_RECURSE TFLITE_DLLS "${ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH}/*.dll")
    add_custom_command(TARGET minimal-tflite
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TFLITE_DLLS}
            $<TARGET_FILE_DIR:minimal-tflite>)
endif (MSVC)