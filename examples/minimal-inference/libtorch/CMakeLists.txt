cmake_minimum_required(VERSION 3.16)
project(minimal-libtorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-libtorch
  minimal-libtorch.cpp
)

target_link_libraries(minimal-libtorch anira)

if (MSVC)
    foreach(DLL ${NECESSARY_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME}
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-libtorch>)
    endforeach()
endif (MSVC)
