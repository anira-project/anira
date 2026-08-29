# ==============================================================================
# tanh-tooling · cmake/test-deps.cmake — googletest / google benchmark, fetched once.
#
#   tanh_fetch_googletest([VERSION <tag>] [INSTALL])       -> targets gtest, gtest_main, gmock, gmock_main;
#                                                            include(GoogleTest) done (gtest_discover_tests)
#   tanh_fetch_googlebenchmark([VERSION <tag>] [INSTALL])  -> targets benchmark, benchmark_main
#   tanh_copy_runtime_dlls(<target>)             Windows shared builds: POST_BUILD copy of the
#                                                   target's transitive DLLs next to it, so
#                                                   gtest_discover_tests can run it at build time
#
# Idempotent (a no-op when the targets exist already — a parent project may have fetched
# them first), SYSTEM include directories, warnings off, PIC on, dashboard targets
# suppressed (CTEST_TARGETS_ADDED). Install rules are off unless INSTALL is given — a
# library that links gtest/benchmark into an installed, exported target (anira with
# ANIRA_WITH_BENCHMARK) needs them in an export set of their own. The dependencies' own options are steered
# through cache variables, as FetchContent requires. Call enable_testing()/include(CTest)
# yourself. Requires CMake >= 3.25 (FetchContent SYSTEM); include after project().
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

# cmake_parse_arguments: an empty value after a keyword is a value, not an omission
# (CMP0174 NEW); function bodies record the policy state of their definition.
cmake_policy(PUSH)
if(POLICY CMP0174)
    cmake_policy(SET CMP0174 NEW)
endif()

function(_tanh_quiet_targets)
    foreach(_t IN LISTS ARGN)
        if(TARGET ${_t})
            set_target_properties(${_t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
            target_compile_options(${_t} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
        endif()
    endforeach()
endfunction()

function(tanh_fetch_googletest)
    cmake_parse_arguments(PARSE_ARGV 0 arg "INSTALL" "VERSION" "")
    if(NOT arg_VERSION)
        set(arg_VERSION v1.17.0)
    endif()
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    if(NOT TARGET gtest_main)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG ${arg_VERSION}
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            SYSTEM)
        set(INSTALL_GTEST ${arg_INSTALL} CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # MSVC: same CRT as the code under test
        FetchContent_MakeAvailable(googletest)
        _tanh_quiet_targets(gtest gtest_main gmock gmock_main)
    endif()
    include(GoogleTest)
endfunction()

function(tanh_fetch_googlebenchmark)
    cmake_parse_arguments(PARSE_ARGV 0 arg "INSTALL" "VERSION" "")
    if(NOT arg_VERSION)
        set(arg_VERSION v1.9.5)
    endif()
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    if(NOT TARGET benchmark)
        include(FetchContent)
        FetchContent_Declare(googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG ${arg_VERSION}
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            SYSTEM)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL ${arg_INSTALL} CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)  # newer compilers' warnings are not our build errors
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            # benchmark's std::regex probe fails to run under Xcode on Apple Silicon.
            set(HAVE_STD_REGEX ON CACHE BOOL "" FORCE)
            set(RUN_HAVE_STD_REGEX 1 CACHE INTERNAL "")
        endif()
        FetchContent_MakeAvailable(googlebenchmark)
        _tanh_quiet_targets(benchmark benchmark_main)
    endif()
endfunction()

function(tanh_copy_runtime_dlls target)
    if(WIN32 AND BUILD_SHARED_LIBS)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
    endif()
endfunction()

cmake_policy(POP)
