cmake_minimum_required(VERSION 3.16)
project(minimal-libtorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-libtorch
  minimal-libtorch.cpp
)

target_link_libraries(minimal-libtorch
  anira::anira
  "${TORCH_LIBRARIES}"
)

if (MSVC)
    foreach(DLL ${ANIRA_SHARED_LIBS_WIN})
        add_custom_command(TARGET ${PROJECT_NAME}
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-libtorch>)
    endforeach()
endif (MSVC)
