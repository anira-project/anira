# Consumers of an installed anira package. Built against a fresh install prefix by
# the build_install CI workflow and by the InstallConsumer CTest
# (run-install-test.cmake), so a header, exported target or dependency
# (the backends) missing from the install tree fails here rather than
# in a downstream project.
cmake_minimum_required(VERSION 3.25)
project(anira_install_consumer LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anira REQUIRED)
get_target_property(_anira_type anira::anira TYPE)

# 0. Gate 5: a pure-C consumer (C11), the raw-C sequence of the architecture document's section 9
#    over a registered engine; its include probes are gate 4 on the installed side. A static anira
#    is C++ inside, so the C executable links through the C++ driver there. The function-effects
#    analysis as an error where the compiler has it (gate 6; upstream clang 20 or later).
add_executable(consumer_c main_c.c)
target_link_libraries(consumer_c PRIVATE anira::anira)
set_target_properties(consumer_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
if(_anira_type STREQUAL "STATIC_LIBRARY")
    set_target_properties(consumer_c PROPERTIES LINKER_LANGUAGE CXX)
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "Clang"
        AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 20
        AND NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(consumer_c PRIVATE -Werror=function-effects)
endif()

# 1. A consumer of anira alone: anira's headers, anira::anira, nothing of any engine.
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE anira::anira)

# 1b. A 2.x-shaped consumer through the header-only compat shim (anira/compat/v2.hpp over
#     anira.hpp and the C entries), anira::anira alone.
add_executable(consumer_compat main_compat.cpp)
target_link_libraries(consumer_compat PRIVATE anira::anira)

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

# 5. Gate 5: a host that links its own tanh-lib, shared, before anira::anira. anira's copy of
#    tanh-lib's core is compiled into libanira in the inline namespace thl::anira; the host's is
#    tanh-lib's default, thl::v<major>_<minor>. No symbol of the two shares a mangled name, so
#    they cannot bind to each other on any platform and in either shape of libanira: a shared one
#    (which exports anira_* alone, and names no thl symbol in its dynamic table: the nm half of the
#    proof in run-install-test.cmake) and a static one, whose objects are linked into the host's
#    image beside the host's tanh-lib (the namespace gate anira_abi_namespaces proves the names).
#    The host pins the commit anira pins (cmake/tanh-lib-pin.cmake, the one file both read) and
#    builds it with its own namespace. run-install-test.cmake points FetchContent at the anira
#    build tree's tanh-lib source, CI fetches it.
include(FetchContent)
set(TANH_BUILD_CORE ON)
set(TANH_BUILD_STATE OFF)
set(TANH_BUILD_DSP OFF)
set(TANH_BUILD_MODULATION OFF)
set(TANH_BUILD_AUDIO_IO OFF)
set(TANH_WITH_TESTS OFF)
set(TANH_WITH_EXAMPLES OFF)
set(TANH_WITH_DOCS OFF)
set(TANH_WITH_INSTALL OFF)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/tanh-lib-pin.cmake")
FetchContent_Declare(tanh-lib
    GIT_REPOSITORY ${ANIRA_TANH_LIB_REPOSITORY}
    GIT_TAG ${ANIRA_TANH_LIB_COMMIT})
set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(tanh-lib)
unset(BUILD_SHARED_LIBS)
add_executable(consumer_tanh_first main_tanh_first.cpp)
# tanh::Core first, anira::anira second: the order in which anira's references to thl would bind
# to the host's copy if the two shared a name, and the one the gate must survive.
target_link_libraries(consumer_tanh_first PRIVATE tanh::Core anira::anira)

# Windows: an application ships its DLLs next to its executable. The loader searches
# the application directory first and System32 before PATH — and Windows itself ships
# an onnxruntime.dll (Windows ML, ORT 1.17) in System32, so an executable that calls
# ONNX Runtime with only <prefix>/bin on PATH binds to the wrong runtime and dies on
# the first API call (the very "host ships its own backend runtime" failure anira's
# symbol policy guards plugins against). $<TARGET_RUNTIME_DLLS> lists the DLLs of the
# imported SHARED targets in the link closure — anira's DLL and, for the
# engine consumers, onnxruntime.dll through anira::onnxruntime.
if(WIN32)
    foreach(_consumer consumer consumer_c consumer_compat consumer_engine consumer_engine_module
                      consumer_tanh_first)
        if(TARGET ${_consumer})
            add_custom_command(TARGET ${_consumer} POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                    "-DDLLS=$<TARGET_RUNTIME_DLLS:${_consumer}>"
                    "-DDEST=$<TARGET_FILE_DIR:${_consumer}>"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/copy-runtime-dlls.cmake"
                VERBATIM)
        endif()
    endforeach()
endif()
