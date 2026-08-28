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

add_library(anira_unload_module MODULE module.cpp)
target_link_libraries(anira_unload_module PRIVATE anira::anira)
set_target_properties(anira_unload_module PROPERTIES
    PREFIX ""
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    # What a plugin should do (see docs/sphinx/troubleshooting.rst): hidden symbol
    # visibility, and — under GCC — no STB_GNU_UNIQUE symbols, since glibc never
    # unloads an object that defines one.
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(anira_unload_module PRIVATE -fno-gnu-unique)
endif()

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
# On Windows every runtime DLL is copied next to the executable (below), so bare file
# names resolve through the application directory.
set(_anira_unload_pinned "")
if(ANIRA_WITH_LIBTORCH AND ANIRA_LIBTORCH_SHARED_LIB_PATH)
    if(WIN32)
        list(APPEND _anira_unload_pinned "torch.dll")
    else()
        list(APPEND _anira_unload_pinned
            "${ANIRA_LIBTORCH_SHARED_LIB_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}torch${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
endif()
if(ANIRA_WITH_ONNXRUNTIME AND NOT ANIRA_ONNXRUNTIME_IS_STATIC)
    if(WIN32)
        list(APPEND _anira_unload_pinned "${ANIRA_ONNXRUNTIME_LIB_BASENAME}.dll")
    else()
        list(APPEND _anira_unload_pinned
            "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${ANIRA_ONNXRUNTIME_LIB_BASENAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
endif()
if(ANIRA_WITH_LITERT AND NOT ANIRA_LITERT_IS_STATIC)
    if(WIN32)
        list(APPEND _anira_unload_pinned "${ANIRA_LITERT_LIB_BASENAME}.dll")
    else()
        list(APPEND _anira_unload_pinned
            "${ANIRA_LITERT_SHARED_LIB_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${ANIRA_LITERT_LIB_BASENAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
endif()
if(ANIRA_WITH_TFLITE AND NOT ANIRA_TFLITE_IS_STATIC)
    if(ANIRA_TFLITE_SHARED_LIB_PATH)
        set(_anira_unload_tflite_dir "${ANIRA_TFLITE_SHARED_LIB_PATH}")
    else()
        set(_anira_unload_tflite_dir "${ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH}")
    endif()
    if(WIN32)
        list(APPEND _anira_unload_pinned "${ANIRA_TFLITE_LIB_BASENAME}.dll")
    else()
        list(APPEND _anira_unload_pinned
            "${_anira_unload_tflite_dir}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${ANIRA_TFLITE_LIB_BASENAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
endif()
# '|'-joined: a ';'-list would be split into several compile definitions.
list(JOIN _anira_unload_pinned "|" _anira_unload_pinned_joined)

target_compile_definitions(anira_unload_test PRIVATE
    ANIRA_UNLOAD_MODULE_PATH="$<TARGET_FILE:anira_unload_module>"
    ANIRA_UNLOAD_PINNED_LIBS="${_anira_unload_pinned_joined}")
if(BUILD_SHARED_LIBS)
    if(WIN32)
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

# Export-table check (cmake/CheckExports.cmake): the shared libanira exports exactly
# the ANIRA_API surface (namespace anira, nothing else — no std::, backend or
# STB_GNU_UNIQUE symbol), and the plugin-shaped module exports no backend-runtime
# symbol. See the symbol-visibility notes in the top-level CMakeLists.txt.
if(WIN32)
    set(_anira_exports_format PE)
    # dumpbin lives next to link.exe; it is not on PATH outside a developer shell.
    get_filename_component(_anira_linker_dir "${CMAKE_LINKER}" DIRECTORY)
    find_program(ANIRA_DUMPBIN dumpbin HINTS "${_anira_linker_dir}")
    set(_anira_exports_tool "${ANIRA_DUMPBIN}")
    set(_anira_exports_tool_arg "-DDUMPBIN=${ANIRA_DUMPBIN}")
else()
    if(APPLE)
        set(_anira_exports_format MACHO)
    else()
        set(_anira_exports_format ELF)
    endif()
    if(CMAKE_NM)
        set(ANIRA_NM "${CMAKE_NM}")
    else()
        find_program(ANIRA_NM NAMES nm llvm-nm)
    endif()
    set(_anira_exports_tool "${ANIRA_NM}")
    set(_anira_exports_tool_arg "-DNM=${ANIRA_NM}")
endif()
if(_anira_exports_tool)
    if(BUILD_SHARED_LIBS)
        set(_anira_exports_library "$<TARGET_FILE:anira>")
    else()
        set(_anira_exports_library "")
    endif()
    add_test(NAME anira_exports
        COMMAND ${CMAKE_COMMAND}
            -DFORMAT=${_anira_exports_format}
            ${_anira_exports_tool_arg}
            "-DLIBRARY=${_anira_exports_library}"
            "-DMODULE=$<TARGET_FILE:anira_unload_module>"
            -P "${anira_SOURCE_DIR}/cmake/CheckExports.cmake")
else()
    message(WARNING "anira: no nm/dumpbin found — the anira_exports test is not registered")
endif()

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
