# Consumers of an installed anira package. Built against a fresh install prefix by
# the build_install CI workflow and by the InstallConsumer CTest
# (run-install-test.cmake), so a header, exported target or dependency
# (tanh::Core, the backends) missing from the install tree fails here rather than
# in a downstream project.
cmake_minimum_required(VERSION 3.25)
project(anira_install_consumer CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anira REQUIRED)

# 1. A consumer of anira alone: anira's headers, anira::anira, nothing of any engine.
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE anira::anira)

if(ANIRA_WITH_ONNXRUNTIME)
    # 2. A consumer that calls an engine itself. It includes onnxruntime_cxx_api.h and
    #    links anira::onnxruntime next to anira::anira — the very ONNX Runtime anira
    #    uses, so the process holds one copy of it.
    add_executable(consumer_engine main_engine.cpp)
    target_link_libraries(consumer_engine PRIVATE anira::anira anira::onnxruntime)

    # 3. The same TU as a plugin-shaped module, compiled hidden as the troubleshooting
    #    guide recommends for TUs that include engine headers. run-install-test.cmake
    #    checks that its export table carries no engine symbol: the engine-calling
    #    consumer still exports nothing engine-shaped.
    add_library(consumer_engine_module MODULE main_engine.cpp)
    target_compile_definitions(consumer_engine_module PRIVATE CONSUMER_ENGINE_MODULE)
    target_link_libraries(consumer_engine_module PRIVATE anira::anira anira::onnxruntime)
    set_target_properties(consumer_engine_module PROPERTIES
        PREFIX ""
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

    # 4. Negative: the same TU must NOT compile with anira::anira alone. The engine
    #    headers are reachable only through anira::onnxruntime — that failure is the
    #    point of the per-engine header layout. Compile only (no link), and require
    #    the failure to be the missing engine header, not something else.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    try_compile(_engine_without_target "${CMAKE_BINARY_DIR}/try-engine-without-target"
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main_engine.cpp"
        LINK_LIBRARIES anira::anira
        CXX_STANDARD 20
        OUTPUT_VARIABLE _engine_without_target_output)
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    if(_engine_without_target)
        message(FATAL_ERROR "main_engine.cpp compiled with anira::anira alone: an engine header is "
                            "reachable through anira's own include directories, which must not be.")
    elseif(NOT _engine_without_target_output MATCHES "onnxruntime_cxx_api")
        message(FATAL_ERROR "main_engine.cpp failed to compile with anira::anira alone, but not "
                            "because of the missing engine header:\n${_engine_without_target_output}")
    endif()
    message(STATUS "anira install consumer: engine header unreachable without anira::onnxruntime — OK")
endif()
