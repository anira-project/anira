# ==============================================================================
# tanh-tooling · cmake/sanitizers.cmake — one sanitizer, one target.
#
#   tanh_add_sanitizer(<target> <rtsan|asan|ubsan|tsan|msan|lsan> [DEFINE <name>] [IGNORELIST <file>])
#
# Adds -fsanitize=<kind> to the target's compile and link options PUBLIC — a sanitized
# library needs every executable linking it to link the runtime too — plus an optional
# PUBLIC compile definition (e.g. TANH_WITH_ASAN, so headers can adapt) and an
# -fsanitize-ignorelist file. rtsan (RealtimeSanitizer) needs Clang >= 20. With MSVC only
# asan (/fsanitize=address) is available. "usan" is accepted as an alias of ubsan.
# Requires CMake >= 3.18; include after project().
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

# cmake_parse_arguments: an empty value after a keyword is a value, not an omission
# (CMP0174 NEW); function bodies record the policy state of their definition.
cmake_policy(PUSH)
if(POLICY CMP0174)
    cmake_policy(SET CMP0174 NEW)
endif()

function(tanh_add_sanitizer target kind)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "DEFINE;IGNORELIST" "")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tanh_add_sanitizer: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "tanh_add_sanitizer: '${target}' is not a target")
    endif()
    if(kind STREQUAL "rtsan")
        set(_san realtime)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "tanh_add_sanitizer(${target} rtsan): RealtimeSanitizer requires Clang "
                                "(current compiler: ${CMAKE_CXX_COMPILER_ID})")
        endif()
    elseif(kind STREQUAL "asan")
        set(_san address)
    elseif(kind MATCHES "^(ubsan|usan)$")
        set(_san undefined)
    elseif(kind STREQUAL "tsan")
        set(_san thread)
    elseif(kind STREQUAL "msan")
        set(_san memory)
    elseif(kind STREQUAL "lsan")
        set(_san leak)
    else()
        message(FATAL_ERROR "tanh_add_sanitizer: unknown sanitizer '${kind}' (rtsan, asan, ubsan, tsan, msan, lsan)")
    endif()

    if(MSVC)
        if(NOT _san STREQUAL "address")
            message(FATAL_ERROR "tanh_add_sanitizer(${target} ${kind}): only asan is available with MSVC")
        endif()
        target_compile_options(${target} PUBLIC /fsanitize=address)
    else()
        target_compile_options(${target} PUBLIC -fsanitize=${_san})
        target_link_options(${target} PUBLIC -fsanitize=${_san})
        if(arg_IGNORELIST)
            if(NOT EXISTS "${arg_IGNORELIST}")
                message(FATAL_ERROR "tanh_add_sanitizer(${target} ${kind}): IGNORELIST '${arg_IGNORELIST}' does not exist")
            endif()
            target_compile_options(${target} PUBLIC "-fsanitize-ignorelist=${arg_IGNORELIST}")
        endif()
    endif()
    if(arg_DEFINE)
        target_compile_definitions(${target} PUBLIC ${arg_DEFINE})
    endif()
endfunction()

cmake_policy(POP)
