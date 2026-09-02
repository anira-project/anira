# test/abi — the ABI gates of docs/anira-v3-architecture.md (section 6a) that are online
# at M1, and the registry's regenerate-and-diff check. The gtest suite test_abi is
# registered from test/CMakeLists.txt, next to the other per-component binaries.
#
#   anira_abi_generate    tools/abi/gen.py --check: the committed generated files match
#                         abi/anira.yml (desktop, when Python 3 with PyYAML is found)
#   anira_abi_layout      gate 3: generated/test_layout.c pins every enum's width and
#                         terminator and every Tier-1 sizeof/offsetof with _Static_assert;
#                         its printed table is diffed against abi/layout-<major>.txt
#   anira_header_c11      gate 4: every M1 header, alone and together, as C11 under the
#                         strict flags with no anira define at all
#   anira_header_cxx17    gate 4: the same as C++17
#   anira_header_cxx20    gate 4: anira.hpp alone as C++20 under the strict flags, the
#                         wrapper's API exercised at compile time (nothing runs)
#   anira_header_coexist  the v2 C++ umbrella and the C headers in one TU, both orders
#   anira_header_cxx20_with_v2
#                         the v2 C++ umbrella and anira.hpp in one TU, both orders
#
# On desktop the compile gates are EXCLUDE_FROM_ALL objects driven by a CTest (a
# regression shows up as a failed test with the compiler's error); on Android/iOS they
# are members of ALL, since the device cannot drive the build tree.

if(TANH_OPERATING_SYSTEM STREQUAL "Emscripten")
    # Gates 3 and 4 run under node from M2 (the umbrella); nothing here builds on Wasm yet.
    return()
endif()

set(_abi_headers export status version enums log config)
set(_abi_include_dirs "${anira_SOURCE_DIR}/include" "${ANIRA_GENERATED_INCLUDE_DIR}")
if(TANH_OPERATING_SYSTEM MATCHES "^(Android|iOS)$")
    set(_abi_on_device TRUE)
else()
    set(_abi_on_device FALSE)
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC" OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # C4201 (nameless struct/union): the ANIRA_PTR slots are anonymous unions, standard C11.
    set(_abi_strict_flags /W4 /WX /wd4201)
    set(_abi_werror /WX)
else()
    set(_abi_strict_flags -Wall -Wextra -Werror -pedantic)
    set(_abi_werror -Werror)
endif()

# add_compile_definitions() in test/CMakeLists.txt (the model paths) reaches every target
# of this directory at generate time, whatever the target's own COMPILE_DEFINITIONS says;
# the gates promise a consumer's compile line, so the directory's definitions go here.
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "")

# Only rewrite a generated wrapper when its text changes, so a reconfigure does not
# rebuild the gates (CMake 3.15: no file(CONFIGURE)).
function(anira_abi_write_if_changed path content)
    if(EXISTS "${path}")
        file(READ "${path}" _existing)
        if(_existing STREQUAL content)
            return()
        endif()
    endif()
    file(WRITE "${path}" "${content}")
endfunction()

# A consumer-shaped target: anira's include directories only, no anira define at all,
# the strict warning set, and the standard the gate names (never the directory's C++20):
# C11 for C; C++17 for CXX, or the standard given as the third argument (anira.hpp is
# C++20).
function(anira_abi_consumer_target target lang)
    set_property(TARGET ${target} PROPERTY COMPILE_DEFINITIONS "")
    target_include_directories(${target} PRIVATE ${_abi_include_dirs})
    target_compile_options(${target} PRIVATE ${_abi_strict_flags})
    if(lang STREQUAL "C")
        set_target_properties(${target} PROPERTIES
            C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    else()
        set(_cxx_standard 17)
        if(ARGC GREATER 2)
            set(_cxx_standard "${ARGV2}")
        endif()
        set_target_properties(${target} PROPERTIES
            CXX_STANDARD ${_cxx_standard} CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    endif()
endfunction()

# A target holding the v2 C++ umbrella: the include and define set anira_header_isolation
# (test/CMakeLists.txt) uses for it, C++20 (the umbrella's floor), and only the default
# warnings, as errors, since the v2 headers are not held to -Wall -Wextra.
function(anira_abi_umbrella_target target)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_include_directories(${target} PRIVATE
        ${_abi_include_dirs}
        $<TARGET_PROPERTY:concurrentqueue,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:tanh::Core,INTERFACE_INCLUDE_DIRECTORIES>)
    set_property(TARGET ${target} PROPERTY COMPILE_DEFINITIONS "")
    target_compile_definitions(${target} PRIVATE
        ${ANIRA_COMPILE_DEFINITIONS}
        $<TARGET_PROPERTY:tanh::Core,INTERFACE_COMPILE_DEFINITIONS>
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:ANIRA_STATIC>)
    target_compile_options(${target} PRIVATE ${_abi_werror})
endfunction()

# A build-driving CTest: mutually exclusive with the others under ctest -j.
function(anira_abi_build_test target)
    add_test(NAME ${target}
        COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --config $<CONFIG> --target ${target})
    set_tests_properties(${target} PROPERTIES RESOURCE_LOCK anira_build_tree)
endfunction()

# ---- gate 4: per-file self-containment -------------------------------------------------
set(_gate4_dir "${CMAKE_CURRENT_BINARY_DIR}/gate4")
set(_gate4_c "")
set(_gate4_cxx "")
foreach(_h IN LISTS _abi_headers)
    anira_abi_write_if_changed("${_gate4_dir}/only_${_h}.c"
        "#include <anira/abi/${_h}.h>\nint anira_gate4_${_h}_c(void);\nint anira_gate4_${_h}_c(void) { return 0; }\n")
    anira_abi_write_if_changed("${_gate4_dir}/only_${_h}.cpp"
        "#include <anira/abi/${_h}.h>\nint anira_gate4_${_h}_cxx();\nint anira_gate4_${_h}_cxx() { return 0; }\n")
    list(APPEND _gate4_c "${_gate4_dir}/only_${_h}.c")
    list(APPEND _gate4_cxx "${_gate4_dir}/only_${_h}.cpp")
endforeach()
if(_abi_on_device)
    add_library(anira_header_c11 OBJECT header_c.c ${_gate4_c})
    add_library(anira_header_cxx17 OBJECT header_cxx17.cpp ${_gate4_cxx})
else()
    add_library(anira_header_c11 OBJECT EXCLUDE_FROM_ALL header_c.c ${_gate4_c})
    add_library(anira_header_cxx17 OBJECT EXCLUDE_FROM_ALL header_cxx17.cpp ${_gate4_cxx})
    anira_abi_build_test(anira_header_c11)
    anira_abi_build_test(anira_header_cxx17)
endif()
anira_abi_consumer_target(anira_header_c11 C)
anira_abi_consumer_target(anira_header_cxx17 CXX)

# ---- gate 4: anira.hpp alone as C++20 ------------------------------------------------------
# The header-only wrapper the way a consumer compiles it: anira's include directories, no
# anira define, the strict flags, C++20 (the wrapper's floor; the C headers stay at C++17
# above). header_cxx20.cpp exercises the API at compile time; nothing runs.
if(_abi_on_device)
    add_library(anira_header_cxx20 OBJECT header_cxx20.cpp)
else()
    add_library(anira_header_cxx20 OBJECT EXCLUDE_FROM_ALL header_cxx20.cpp)
    anira_abi_build_test(anira_header_cxx20)
endif()
anira_abi_consumer_target(anira_header_cxx20 CXX 20)

# ---- the umbrellas in one TU -------------------------------------------------------------
# header_coexist.inc includes <anira/anira.h> and the C headers in the order its two
# drivers select; header_cxx20_with_v2.inc does the same for <anira/anira.h> and
# <anira/anira.hpp>. A macro redefined between the umbrellas (C4005 / -Wmacro-redefined /
# GCC's "redefined") or a name both declare in namespace anira is what these gates are for.
if(NOT _abi_on_device)
    add_library(anira_header_coexist OBJECT EXCLUDE_FROM_ALL
        header_coexist_v2_first.cpp header_coexist_abi_first.cpp)
    anira_abi_umbrella_target(anira_header_coexist)
    anira_abi_build_test(anira_header_coexist)
    add_library(anira_header_cxx20_with_v2 OBJECT EXCLUDE_FROM_ALL
        header_cxx20_with_v2.cpp header_cxx20_with_v2_reverse.cpp)
    anira_abi_umbrella_target(anira_header_cxx20_with_v2)
    anira_abi_build_test(anira_header_cxx20_with_v2)
endif()

# ---- gate 3: the Tier-1 layout table -----------------------------------------------------
if(_abi_on_device)
    add_library(anira_abi_layout OBJECT generated/test_layout.c)
else()
    add_executable(anira_abi_layout generated/test_layout.c)
    set(_layout_file "${anira_SOURCE_DIR}/abi/layout-${ANIRA_ABI_MAJOR}.txt")
    add_test(NAME anira_abi_layout
        COMMAND ${CMAKE_COMMAND}
            -DLAYOUT_EXE=$<TARGET_FILE:anira_abi_layout>
            -DEXPECTED=${_layout_file}
            -DACTUAL=${CMAKE_CURRENT_BINARY_DIR}/layout-${ANIRA_ABI_MAJOR}.actual.txt
            -P ${anira_SOURCE_DIR}/cmake/abi-layout-check.cmake)
    add_custom_target(anira_abi_layout_regen
        COMMAND ${CMAKE_COMMAND}
            -DLAYOUT_EXE=$<TARGET_FILE:anira_abi_layout>
            -DWRITE_TO=${_layout_file}
            -P ${anira_SOURCE_DIR}/cmake/abi-layout-check.cmake
        DEPENDS anira_abi_layout
        COMMENT "Rewriting ${_layout_file} from the built layout printer")
endif()
anira_abi_consumer_target(anira_abi_layout C)

# ---- the registry check ------------------------------------------------------------------
if(NOT _abi_on_device)
    # The python3 on PATH (the one a developer installed PyYAML into), not FindPython3's
    # newest-interpreter pick; ANIRA_ABI_PYTHON overrides it.
    find_program(ANIRA_ABI_PYTHON NAMES python3 python DOC "Python 3 with PyYAML for tools/abi/gen.py")
    set(_abi_yaml_rc 1)
    if(ANIRA_ABI_PYTHON)
        execute_process(COMMAND "${ANIRA_ABI_PYTHON}" -c "import yaml"
            RESULT_VARIABLE _abi_yaml_rc OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(_abi_yaml_rc EQUAL 0)
        add_test(NAME anira_abi_generate
            COMMAND "${ANIRA_ABI_PYTHON}" "${anira_SOURCE_DIR}/tools/abi/gen.py"
                --repo "${anira_SOURCE_DIR}" --check)
        add_custom_target(anira_abi_regen
            COMMAND "${ANIRA_ABI_PYTHON}" "${anira_SOURCE_DIR}/tools/abi/gen.py"
                --repo "${anira_SOURCE_DIR}" --write
            COMMENT "Regenerating the C ABI files from abi/anira.yml")
    else()
        message(STATUS "anira_abi_generate: no python3 with PyYAML on PATH (set ANIRA_ABI_PYTHON); "
                       "the registry check still runs in CI (build_web)")
    endif()
endif()
