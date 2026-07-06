cmake_minimum_required(VERSION 3.16)
project(offline-inference C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(offline-inference
  offline-inference.cpp
)

target_link_libraries(offline-inference
  anira::anira
)

if (MSVC)
    foreach(DLL ${ANIRA_SHARED_LIBS_WIN})
        add_custom_command(TARGET offline-inference
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:offline-inference>)
    endforeach()
endif (MSVC)
