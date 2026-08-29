# Host-shaped library-unload test.
#
# A plugin host loads a plugin as a shared library and unloads it (dlclose /
# FreeLibrary) once the last instance is gone. Any anira thread still alive at that
# moment executes unmapped code and crashes the host. The in-process tests cannot
# observe this — libanira stays mapped for the whole test run — so this directory
# builds a tiny plugin-shaped MODULE embedding anira, and a test executable that does
# NOT link anira (nothing but the module keeps anira mapped). The executable loads the
# module, drives an InferenceHandler through the module's C entry points, unloads it,
# asserts that the module (and, in shared builds, libanira itself) really got
# unmapped, and then stays alive for a moment: a surviving thread would run into
# unmapped memory and kill the process. A death test that deliberately leaks a thread
# proves the harness can see exactly that.

include(${anira_SOURCE_DIR}/cmake/tanh/check-exports.cmake)

add_library(anira_unload_module MODULE module.cpp)
target_link_libraries(anira_unload_module PRIVATE anira::anira)
set_target_properties(anira_unload_module PROPERTIES
    PREFIX ""
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
# What a plugin should do (see docs/sphinx/troubleshooting.rst): hidden symbol
# visibility, no STB_GNU_UNIQUE symbols under GCC (glibc never unloads an object that
# defines one), dead-code stripping, and an export table pinned to its entry points —
# hidden visibility cannot hide what a header stamps default-visibility itself, and on
# macOS a static anira with ExecuTorch force-loads archives ld64 cannot hide otherwise.
# The anira_exports check below asserts the result.
tanh_apply_symbol_policy(anira_unload_module)
tanh_set_export_allowlist(anira_unload_module SYMBOL anira_test_*)

add_executable(anira_unload_test test_LibraryUnload.cpp)
set_target_properties(anira_unload_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
# Deliberately not linked against anira. gtest comes straight from the FetchContent
# targets (cmake/test-deps.cmake links gtest_main into anira only as a convenience for
# the in-process test binary).
target_link_libraries(anira_unload_test PRIVATE gtest gtest_main ${CMAKE_DL_LIBS})
add_dependencies(anira_unload_test anira_unload_module)

# The shared backend runtimes the module links transitively. The test executable pins
# them in the process before it loads the module (RTLD_NODELETE on POSIX, a leaked
# LoadLibrary handle on Windows), so that unloading the module never unloads them: the
# unmapped assertion is about anira's own binaries — libtorch's global destructors are
# not safe to run on dlclose, and TLS-bearing runtimes may not be unloadable at all.
# Always by full path: the shared engines are SHARED IMPORTED targets whose
# $<TARGET_FILE> is the very file anira links (on Windows the DLL; a full path is
# also what LOAD_WITH_ALTERED_SEARCH_PATH is defined for, and it resolves the DLL's
# own dependencies next to it), LibTorch by its lib/ directory.
set(_anira_unload_pinned "")
if(ANIRA_WITH_LIBTORCH AND ANIRA_LIBTORCH_SHARED_LIB_PATH)
    list(APPEND _anira_unload_pinned
        "${ANIRA_LIBTORCH_SHARED_LIB_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}torch${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
foreach(_anira_unload_engine onnxruntime litert tflite)
    string(TOUPPER "${_anira_unload_engine}" _anira_unload_ENGINE)
    if(ANIRA_WITH_${_anira_unload_ENGINE} AND NOT ANIRA_${_anira_unload_ENGINE}_IS_STATIC)
        list(APPEND _anira_unload_pinned "$<TARGET_FILE:anira::${_anira_unload_engine}>")
    endif()
endforeach()
unset(_anira_unload_engine)
unset(_anira_unload_ENGINE)
# '|'-joined: a ';'-list would be split into several compile definitions.
list(JOIN _anira_unload_pinned "|" _anira_unload_pinned_joined)

target_compile_definitions(anira_unload_test PRIVATE
    ANIRA_UNLOAD_MODULE_PATH="$<TARGET_FILE:anira_unload_module>"
    ANIRA_UNLOAD_PINNED_LIBS="${_anira_unload_pinned_joined}")
if(BUILD_SHARED_LIBS)
    if(TANH_BINARY_FORMAT STREQUAL "PE")
        # GetModuleHandleEx works on the file name.
        target_compile_definitions(anira_unload_test PRIVATE
            ANIRA_UNLOAD_LIB_PATH="$<TARGET_FILE_NAME:anira>")
    else()
        # The name the loader actually maps (the DT_NEEDED / install name), not the
        # unversioned symlink.
        target_compile_definitions(anira_unload_test PRIVATE
            ANIRA_UNLOAD_LIB_PATH="$<TARGET_SONAME_FILE:anira>")
    endif()
endif()

# The death test must run in the "threadsafe" style (a fresh re-executed process);
# passed on the command line rather than via GTEST_FLAG_SET, whose flag variable is
# not exported from gtest's DLL build on Windows. macOS crash reporting can make the
# re-execution slow, hence the timeout.
gtest_discover_tests(anira_unload_test
    EXTRA_ARGS --gtest_death_test_style=threadsafe
    DISCOVERY_TIMEOUT 90
    PROPERTIES TIMEOUT 120)

# Export-table check (cmake/tanh/check-exports.cmake): the shared libanira exports
# exactly the ANIRA_API surface (namespace anira, nothing else — no std::, backend or
# STB_GNU_UNIQUE symbol), and the plugin-shaped module exports no backend-runtime
# symbol and, with a static anira, nothing of namespace anira either. See the
# symbol-policy notes in the top-level CMakeLists.txt. On PE, which cannot interpose,
# the dllexports LibTorch's headers and the LiteRT/TFLite archives force are tolerated.
tanh_add_export_check(NAME anira_exports
    LIBRARY anira MODULE anira_unload_module
    NAMESPACES anira
    FORBID_NAMESPACES Ort onnxruntime torch c10 caffe2 executorch tflite litert flatbuffers Eigen absl thl
    FORBID_PREFIXES Ort OrtApi xnn_ pthreadpool_ cpuinfo_ kai_ TfLite LiteRt
    TOLERATE_REGEX_PE "@(torch|c10|caffe2)@@" "^(TfLite|LiteRt)")

# Windows: the module, anira.dll and every runtime DLL are searched relative to the
# application, so copy them next to the executable (the module is built into the same
# directory already). Same pattern as test/CMakeLists.txt.
if(MSVC AND ANIRA_SHARED_LIBS_WIN)
    add_custom_command(TARGET anira_unload_test
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ANIRA_SHARED_LIBS_WIN}
            $<TARGET_FILE_DIR:anira_unload_test>)
endif()
