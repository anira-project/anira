cmake_minimum_required(VERSION 3.16)
project(minimal-onnxruntime C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-onnxruntime
  minimal-onnxruntime.cpp
)

# This example calls ONNX Runtime directly, so it links the engine target next to
# anira: anira::onnxruntime is the same file anira uses (one copy per process) and
# is what makes the engine headers reachable — anira::anira alone carries none.
target_link_libraries(minimal-onnxruntime
  anira::anira
  anira::onnxruntime
)

if (MSVC)
    foreach(DLL ${ANIRA_SHARED_LIBS_WIN})
        add_custom_command(TARGET minimal-onnxruntime
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-onnxruntime>)
    endforeach()
endif (MSVC)
