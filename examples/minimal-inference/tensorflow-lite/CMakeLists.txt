cmake_minimum_required(VERSION 3.16)
project(minimal-tflite C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-tflite
  minimal-tflite.cpp
)

# This example calls the TensorFlow Lite C API directly, so it links the engine
# target next to anira: anira::tflite is the same file anira uses (one copy per
# process) and is what makes the engine headers reachable — anira::anira alone
# carries none.
target_link_libraries(minimal-tflite
  anira::anira
  anira::tflite
)

if (MSVC)
    foreach(DLL ${ANIRA_SHARED_LIBS_WIN})
        add_custom_command(TARGET minimal-tflite
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-tflite>)
    endforeach()
endif (MSVC)
