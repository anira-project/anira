# test/abi — the ABI gates of docs/anira-v3-architecture.md (section 6a) that are online
# at M1, and the registry's regenerate-and-diff check. The gtest suite test_abi is
# registered from test/CMakeLists.txt, next to the other per-component binaries.
#
#   anira_abi_generate    tools/abi/gen.py --check: the committed generated files match
#                         abi/anira.yml (desktop, when Python 3 with PyYAML is found)
#   anira_abi_layout      gate 3: generated/test_layout.c pins every enum's width and
#                         terminator and every Tier-1 sizeof/offsetof with _Static_assert;
#                         its printed table is diffed against abi/layout-<major>.txt
#   anira_header_c11      gate 4: every header of _abi_headers, alone and together, as C11
#                         under the strict flags with no anira define at all
#   anira_header_cxx17    gate 4: the same as C++17
#   anira_header_cxx20    gate 4: anira.hpp alone as C++20 under the strict flags, the
#                         wrapper's API exercised at compile time (nothing runs)
#   anira_header_coexist  the v2 C++ umbrella and the C headers in one TU, both orders
#   anira_header_cxx20_with_v2
#                         the v2 C++ umbrella and anira.hpp in one TU, both orders
#   anira_header_compat   anira/compat/v2.hpp alone as C++20 under the strict flags, the
#                         shim's API exercised at compile time (nothing runs)
#   anira_header_compat_with_v2
#                         the v2 C++ umbrella, anira.hpp and the shim in one TU, both orders
#   anira_abi_rt_contract gate 6 (seed): test_rt_contract.c, ANIRA_NONBLOCKING functions
#                         calling every [callback-safe] tensor entry and the nonblocking Hard
#                         entries over host tensors, under -Werror=function-effects on clang
#                         20 or later
#   anira_abi_link        presence: generated/link_probe.c takes the address of every
#                         promised and draft entry point and links anira::anira the way a
#                         consumer does; an entry without a body fails this link (on a
#                         static library the only presence check there is)
#   anira_symbol_baseline presence: every promised and draft entry point is in the shared
#                         library's real export table (cmake/abi-symbols.cmake)
#
# On desktop the compile gates are EXCLUDE_FROM_ALL objects driven by a CTest (a
# regression shows up as a failed test with the compiler's error); on Android/iOS they
# are members of ALL, since the device cannot drive the build tree.
#
# On WebAssembly (ANIRA_WITH_ABI_TESTS, the wasm-abi preset; no CTest there) the gates are
# members of ALL and run under node (CMAKE_CROSSCOMPILING_EMULATOR, which Emscripten's
# toolchain file sets), through the target anira_abi_wasm_gates that build_web runs:
#   anira_header_c11 / anira_header_cxx17
#                         gate 4 as above, anira/anira_em.h with its declarations compiled
#   anira_abi_layout      gate 3: the printer as a node executable, its table diffed against
#                         abi/layout-<major>.txt (the wasm32 column of the layout identity)
#   anira_em_smoke        the trampolines of anira/anira_em.h against a hook object installed
#                         by em_smoke.pre.js, one instance, the queue driven by
#                         anira_inference_thread_execute; it also holds the link's
#                         thread-local block to its 1024-byte budget
#   anira_em_smoke_thread a user-driven inference thread for real: em_smoke_thread.mjs runs
#                         a second instance in a worker_threads Worker, with its own stack
#                         and thread-local block, inside anira_inference_thread_run_loop
#                         (the park on the wake word, the notify, the has_exited stop)

set(_abi_headers export status version enums log tensor config context core thread lifecycle stage engine handler draft/tensor_platform)
# The platform headers of the registry (`platform:`), beside the umbrellas: empty on every
# other target, so the per-file wrappers include them everywhere.
set(_abi_platform_headers anira_em)
set(_abi_include_dirs "${anira_SOURCE_DIR}/include" "${ANIRA_GENERATED_INCLUDE_DIR}")
if(TANH_OPERATING_SYSTEM MATCHES "^(Android|iOS)$")
    set(_abi_on_device TRUE)
else()
    set(_abi_on_device FALSE)
endif()
if(TANH_BINARY_FORMAT STREQUAL "Wasm")
    set(_abi_on_wasm TRUE)
else()
    set(_abi_on_wasm FALSE)
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
    # draft/tensor_platform: the include path keeps its slash, the wrapper's symbol and file
    # name take the C identifier (draft_tensor_platform).
    string(MAKE_C_IDENTIFIER "${_h}" _id)
    anira_abi_write_if_changed("${_gate4_dir}/only_${_id}.c"
        "#include <anira/abi/${_h}.h>\nint anira_gate4_${_id}_c(void);\nint anira_gate4_${_id}_c(void) { return 0; }\n")
    anira_abi_write_if_changed("${_gate4_dir}/only_${_id}.cpp"
        "#include <anira/abi/${_h}.h>\nint anira_gate4_${_id}_cxx();\nint anira_gate4_${_id}_cxx() { return 0; }\n")
    list(APPEND _gate4_c "${_gate4_dir}/only_${_id}.c")
    list(APPEND _gate4_cxx "${_gate4_dir}/only_${_id}.cpp")
endforeach()
foreach(_h IN LISTS _abi_platform_headers)
    anira_abi_write_if_changed("${_gate4_dir}/only_${_h}.c"
        "#include <anira/${_h}.h>\nint anira_gate4_${_h}_c(void);\nint anira_gate4_${_h}_c(void) { return 0; }\n")
    anira_abi_write_if_changed("${_gate4_dir}/only_${_h}.cpp"
        "#include <anira/${_h}.h>\nint anira_gate4_${_h}_cxx();\nint anira_gate4_${_h}_cxx() { return 0; }\n")
    list(APPEND _gate4_c "${_gate4_dir}/only_${_h}.c")
    list(APPEND _gate4_cxx "${_gate4_dir}/only_${_h}.cpp")
endforeach()
if(_abi_on_device OR _abi_on_wasm)
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

# ---- the WebAssembly arm: gate 3 and the anira_em.h smoke under node -------------------------
if(_abi_on_wasm)
    if(NOT CMAKE_CROSSCOMPILING_EMULATOR)
        message(FATAL_ERROR "ANIRA_WITH_ABI_TESTS on WebAssembly runs its gates under node: "
                            "CMAKE_CROSSCOMPILING_EMULATOR (Emscripten's toolchain file sets it) "
                            "names no node executable")
    endif()
    # Gate 3: the printer alone, no library; the same table every native leg prints.
    add_executable(anira_abi_layout generated/test_layout.c)
    anira_abi_consumer_target(anira_abi_layout C)
    target_link_options(anira_abi_layout PRIVATE -sENVIRONMENT=node)
    # The smoke: C over the hook object em_smoke.pre.js installs, linked against the library
    # the way AniraWeb is (shared memory, the C ABI's export list, the runtime methods the hook
    # object reads the heap with).
    set(_smoke_exports "${CMAKE_CURRENT_BINARY_DIR}/em_smoke_exports.txt")
    file(READ "${anira_SOURCE_DIR}/web/src/abi/exports_wasm.txt" _wasm_exports)
    anira_abi_write_if_changed("${_smoke_exports}" "${_wasm_exports}_main\n")
    add_executable(anira_em_smoke em_smoke.c)
    anira_abi_consumer_target(anira_em_smoke C)
    target_link_libraries(anira_em_smoke PRIVATE anira::anira)
    # The link is AniraWeb's: -pthread, whose objects carry the atomics and bulk-memory
    # features (anira's through CMAKE_C_FLAGS / CMAKE_CXX_FLAGS).
    set_target_properties(anira_em_smoke PROPERTIES LINKER_LANGUAGE CXX)
    target_link_options(anira_em_smoke PRIVATE
        -pthread
        -sENVIRONMENT=node
        -sSHARED_MEMORY=1
        -sINITIAL_MEMORY=268435456
        -sALLOW_MEMORY_GROWTH=0
        -sMALLOC=emmalloc
        -sSTACK_SIZE=8388608
        -sEXIT_RUNTIME=1
        -sASSERTIONS=1
        -sNO_DISABLE_EXCEPTION_CATCHING
        "-sEXPORTED_FUNCTIONS=@${_smoke_exports}"
        "-sEXPORTED_RUNTIME_METHODS=[\"UTF8ToString\",\"HEAPU32\",\"HEAPF32\"]"
        "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/em_smoke.pre.js")
    set_property(TARGET anira_em_smoke APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/em_smoke.pre.js" "${_smoke_exports}")
    # The inference thread for real (decision 9.12): an ES module of the same library that
    # em_smoke_thread.mjs instantiates twice on one memory, once in a worker_threads Worker
    # that runs anira_inference_thread_run_loop (the park, the notify, the has_exited stop).
    add_executable(anira_em_smoke_thread em_smoke_thread.c)
    anira_abi_consumer_target(anira_em_smoke_thread C)
    target_link_libraries(anira_em_smoke_thread PRIVATE anira::anira)
    set_target_properties(anira_em_smoke_thread PROPERTIES LINKER_LANGUAGE CXX SUFFIX ".mjs")
    target_link_options(anira_em_smoke_thread PRIVATE
        --no-entry
        -pthread
        -sENVIRONMENT=node
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sIMPORTED_MEMORY=1
        -sSHARED_MEMORY=1
        -sINITIAL_MEMORY=268435456
        -sALLOW_MEMORY_GROWTH=0
        -sMALLOC=emmalloc
        -sSTACK_SIZE=8388608
        -sASSERTIONS=1
        -sNO_DISABLE_EXCEPTION_CATCHING
        "-sEXPORTED_FUNCTIONS=@${anira_SOURCE_DIR}/web/src/abi/exports_wasm.txt"
        "-sEXPORTED_RUNTIME_METHODS=[\"HEAPU32\",\"HEAPF32\",\"stackRestore\"]")
    set_property(TARGET anira_em_smoke_thread APPEND PROPERTY LINK_DEPENDS
        "${anira_SOURCE_DIR}/web/src/abi/exports_wasm.txt")
    # Issue #212: the core's mutexes exclude across two threads of one module (a pthread
    # beside the main thread, on the threaded libc AniraWeb's Wasm Workers run on).
    add_executable(anira_em_mutex em_mutex.c)
    anira_abi_consumer_target(anira_em_mutex C)
    target_link_libraries(anira_em_mutex PRIVATE anira::anira)
    set_target_properties(anira_em_mutex PROPERTIES LINKER_LANGUAGE CXX)
    target_link_options(anira_em_mutex PRIVATE
        -pthread
        -sPTHREAD_POOL_SIZE=1
        -sENVIRONMENT=node
        -sINITIAL_MEMORY=268435456
        -sALLOW_MEMORY_GROWTH=0
        -sMALLOC=emmalloc
        -sSTACK_SIZE=8388608
        -sEXIT_RUNTIME=1
        -sASSERTIONS=1
        -sNO_DISABLE_EXCEPTION_CATCHING)
    # Both gates under node, in one target build_web runs after the build.
    add_custom_target(anira_abi_wasm_gates
        COMMAND ${CMAKE_COMMAND}
            "-DLAYOUT_RUNNER=${CMAKE_CROSSCOMPILING_EMULATOR}"
            -DLAYOUT_EXE=$<TARGET_FILE:anira_abi_layout>
            -DEXPECTED=${anira_SOURCE_DIR}/abi/layout-${ANIRA_ABI_MAJOR}.txt
            -DACTUAL=${CMAKE_CURRENT_BINARY_DIR}/layout-${ANIRA_ABI_MAJOR}.actual.txt
            -P ${anira_SOURCE_DIR}/cmake/abi-layout-check.cmake
        COMMAND "${CMAKE_CROSSCOMPILING_EMULATOR}" $<TARGET_FILE:anira_em_smoke>
        COMMAND "${CMAKE_CROSSCOMPILING_EMULATOR}" "${CMAKE_CURRENT_SOURCE_DIR}/em_smoke_thread.mjs"
            $<TARGET_FILE:anira_em_smoke_thread>
        COMMAND "${CMAKE_CROSSCOMPILING_EMULATOR}" $<TARGET_FILE:anira_em_mutex>
        DEPENDS anira_abi_layout anira_em_smoke anira_em_smoke_thread anira_em_mutex
        COMMENT "Running the WebAssembly ABI gates under node: the layout table and the anira_em.h smoke"
        VERBATIM)
    return()
endif()

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
if(NOT _abi_strict_flags MATCHES "/W4")
    # MSVC's /W4 refuses a parameter that hides a member (C4458), inherited ones included:
    # Tensor and SyncToken derive from the C records. -Wshadow is the same check in gcc and
    # clang, so the wrapper cannot pass here and fail on the Windows leg. clang's
    # -Wshadow-field extends it to the static factories: JUCE's recommended warning flags
    # carry it, so a plugin that includes anira.hpp would otherwise see one warning per
    # factory parameter named after a field of anira_tensor.
    target_compile_options(anira_header_cxx20 PRIVATE -Wshadow)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(anira_header_cxx20 PRIVATE -Wshadow-field)
    endif()
endif()
# What a phase callback of an anira::Stage reaches (RingView, StageContext, the Tensor reads)
# must be nonblocking: header_cxx20.cpp calls all of it from an ANIRA_NONBLOCKING function, and
# clang's function-effects analysis turns a method that allocates, locks or calls a function
# that is not nonblocking into a compiler error. The compilers that have the analysis are those
# of gate 6 below (upstream clang 20 or later behind the GNU front end).
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20
        AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(anira_header_cxx20 PRIVATE -Werror=function-effects)
endif()

# ---- the compat gate: anira/compat/v2.hpp alone as C++20 --------------------------------------
# The 2.x source-compatible face the way a 2.x consumer compiles it: anira's include directories,
# no anira define, the strict flags and C++20, -Wshadow / -Wshadow-field as for anira.hpp above.
# header_compat.cpp exercises the shim at compile time; nothing runs. -Werror=function-effects
# under the same compilers as the cxx20 gate: a shim function that carries ANIRA_NONBLOCKING and
# allocates, locks or calls an unattributed function is a compiler error there (a function
# without the attribute is not analysed, so the flag is never wrong).
if(_abi_on_device)
    add_library(anira_header_compat OBJECT header_compat.cpp)
else()
    add_library(anira_header_compat OBJECT EXCLUDE_FROM_ALL header_compat.cpp)
    anira_abi_build_test(anira_header_compat)
endif()
anira_abi_consumer_target(anira_header_compat CXX 20)
if(NOT _abi_strict_flags MATCHES "/W4")
    target_compile_options(anira_header_compat PRIVATE -Wshadow)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(anira_header_compat PRIVATE -Wshadow-field)
    endif()
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20
        AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(anira_header_compat PRIVATE -Werror=function-effects)
endif()

# ---- gate 6 (seed): the real-time contract in a consumer's compiler -------------------------
# test_rt_contract.c holds ANIRA_NONBLOCKING functions calling every [callback-safe] entry of
# the tensor headers and the nonblocking Hard entries over host tensors. Consumer-shaped like the header gates (C11, the strict flags, no anira
# define), plus -Werror=function-effects where the compiler has the analysis: upstream clang
# 20 or later behind the GNU front end. gcc rejects the flag outright; an older clang (an NDK)
# and AppleClang, whose version numbers are its own, may not know the warning group, which
# the strict -Werror would turn fatal; clang-cl is left out until a leg proves it. There the
# file is plain C11. Nothing links it: a regression is a compiler error.
if(_abi_on_device)
    add_library(anira_abi_rt_contract OBJECT test_rt_contract.c)
else()
    add_library(anira_abi_rt_contract OBJECT EXCLUDE_FROM_ALL test_rt_contract.c)
    anira_abi_build_test(anira_abi_rt_contract)
endif()
anira_abi_consumer_target(anira_abi_rt_contract C)
if(CMAKE_C_COMPILER_ID STREQUAL "Clang"
        AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 20
        AND NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(anira_abi_rt_contract PRIVATE -Werror=function-effects)
endif()

# ---- the umbrellas in one TU -------------------------------------------------------------
# header_coexist.inc includes <anira/anira.h> and the C headers in the order its two
# drivers select; header_cxx20_with_v2.inc does the same for <anira/anira.h> and
# <anira/anira.hpp>, header_compat_with_v2.inc for <anira/anira.h> and the compat shim. A macro redefined between the umbrellas (C4005 / -Wmacro-redefined /
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
    add_library(anira_header_compat_with_v2 OBJECT EXCLUDE_FROM_ALL
        header_compat_with_v2.cpp header_compat_with_v2_reverse.cpp)
    anira_abi_umbrella_target(anira_header_compat_with_v2)
    anira_abi_build_test(anira_header_compat_with_v2)
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

# ---- the presence gates ------------------------------------------------------------------
# The link probe is a consumer-shaped executable: anira's include directories, no anira
# define of its own (the PUBLIC ones of anira::anira reach it like any consumer), the
# strict flags, and the link against anira::anira — shared or static. It runs on desktop
# (the table is printed); on Android/iOS it is built as a member of ALL, which is the check.
add_executable(anira_abi_link generated/link_probe.c)
target_link_libraries(anira_abi_link PRIVATE anira::anira)
anira_abi_consumer_target(anira_abi_link C)
if(ANIRA_WITH_UBSAN)
    # The probe is C, so the C driver links it, and that driver leaves out the C++ half of
    # the UBSan runtime. libanira's -fsanitize=vptr checks live there
    # (__ubsan_handle_dynamic_type_cache_miss, __ubsan_vptr_type_cache), and with a static
    # sanitizer runtime the executable has to supply them: the link fails on clang 20 /
    # Ubuntu. The source still compiles as C; only this instrumented build links it through
    # the C++ driver.
    set_target_properties(anira_abi_link PROPERTIES LINKER_LANGUAGE CXX)
endif()
if(NOT _abi_on_device)
    add_test(NAME anira_abi_link COMMAND anira_abi_link)
    if(MSVC AND ANIRA_SHARED_LIBS_WIN)
        # Windows: anira.dll and the runtime DLLs are searched next to the executable.
        add_custom_command(TARGET anira_abi_link PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ANIRA_SHARED_LIBS_WIN} $<TARGET_FILE_DIR:anira_abi_link>)
    endif()
    include("${anira_SOURCE_DIR}/cmake/abi-symbols.cmake")
    anira_add_symbol_baseline(NAME anira_symbol_baseline LIBRARY anira
        SYMBOLS "${anira_SOURCE_DIR}/abi/symbols-${ANIRA_ABI_MAJOR}.txt"
                "${anira_SOURCE_DIR}/abi/symbols-draft.txt")
endif()

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
