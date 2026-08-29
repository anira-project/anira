cmake_minimum_required(VERSION 3.16)
project(minimal-litert C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-litert
  minimal-litert.cpp
)

# This example calls LiteRT directly, so it links the engine target next to anira:
# anira::litert is the same file anira uses (one copy per process) and is what
# makes the engine headers reachable — anira::anira alone carries none.
target_link_libraries(minimal-litert
  anira::anira
  anira::litert
)

if (MSVC)
    foreach(DLL ${ANIRA_SHARED_LIBS_WIN})
        add_custom_command(TARGET minimal-litert
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-litert>)
    endforeach()
endif (MSVC)
