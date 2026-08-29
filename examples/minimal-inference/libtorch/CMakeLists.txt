cmake_minimum_required(VERSION 3.16)
project(minimal-libtorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-libtorch
  minimal-libtorch.cpp
)

# This example calls LibTorch directly, so it links the engine target next to
# anira: anira::libtorch wraps the same torch libraries anira uses (one copy per
# process) and is what makes the torch headers reachable — anira::anira alone
# carries none.
target_link_libraries(minimal-libtorch
  anira::anira
  anira::libtorch
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
