cmake_minimum_required(VERSION 3.16)
project(minimal-onnxruntime C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-onnxruntime
  minimal-onnxruntime.cpp
)

target_link_libraries(minimal-onnxruntime anira)

if (MSVC)
    foreach(DLL ${NECESSARY_DLLS})
        add_custom_command(TARGET minimal-onnxruntime
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-onnxruntime>)
    endforeach()
endif (MSVC)