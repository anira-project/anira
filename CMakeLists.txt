cmake_minimum_required(VERSION 3.15)

# CMP0099 NEW (3.19): a static anira's PRIVATE engine link options (the
# --exclude-libs hiding) must reach the consumer's link.
cmake_policy(SET CMP0099 NEW)

# ==============================================================================
# Build options — every user-facing option lives here. Cross-option validation
# (mutual exclusivity, libtorch auto-disable, WASM guards) is in
# cmake/validate-options.cmake, included after project() / Emscripten detection.
# ==============================================================================

# --- Library type --------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build anira as a shared library" ON)

# --- Components ----------------------------------------------------------------
option(ANIRA_WITH_BENCHMARK "Build the benchmark examples (examples/benchmark) and their fixture" OFF)
option(ANIRA_WITH_EXAMPLES  "Add example targets (juce plugin, benchmarks, minimal inference and model examples)" OFF)
option(ANIRA_WITH_INSTALL   "Add install targets" OFF)
option(ANIRA_WITH_TESTS     "Add build tests" OFF)
option(ANIRA_WITH_ABI_TESTS "Add the ABI gates alone (test/abi); the one test option WebAssembly allows, run under node" OFF)
option(ANIRA_WITH_DOCS      "Build documentation" OFF)
option(ANIRA_WITH_LOGGING   "Enable logging printouts" ON)
option(ANIRA_WITH_RTSAN     "Enable RealtimeSanitizer (rtsan) checks (requires clang 20)" OFF)
option(ANIRA_WITH_ASAN      "Enable AddressSanitizer" OFF)
option(ANIRA_WITH_UBSAN     "Enable UndefinedBehaviorSanitizer" OFF)
option(ANIRA_WITH_TSAN      "Enable ThreadSanitizer" OFF)
option(ANIRA_WITH_LSAN      "Enable LeakSanitizer (standalone)" OFF)
option(ANIRA_BUILD_WASM     "Build WebAssembly module (requires Emscripten toolchain)" OFF)
option(ANIRA_LINK_COMPILE_COMMANDS "Link build/compile_commands.json, which clangd reads, to this tree's (top-level builds only)" OFF)

# ==============================================================================
# Inference engines — wiring per engine in cmake/engines/<engine>.cmake,
# prebuilts from anira-project/backends (cmake/engines.cmake). LiteRT and
# TFLite are the same runtime through different C APIs — mutually exclusive.
# ==============================================================================
option(ANIRA_WITH_LIBTORCH    "Build with the LibTorch engine" ON)
option(ANIRA_WITH_ONNXRUNTIME "Build with the ONNX Runtime engine" ON)
option(ANIRA_WITH_LITERT      "Build with the LiteRT engine (LiteRt* C API; runs .tflite via the CompiledModel runtime)" ON)
option(ANIRA_WITH_TFLITE      "Build with the legacy TensorFlow Lite engine (TfLite* C API); mutually exclusive with ANIRA_WITH_LITERT" OFF)
option(ANIRA_WITH_EXECUTORCH  "Build with the ExecuTorch engine (runs .pte models; static-only)" ON)

set(ANIRA_BACKENDS_VERSION "" CACHE STRING "anira-project/backends release tag to download (default: built-in)")
option(ANIRA_BACKENDS_SKIP_REMOTE_CHECK "Don't query GitHub for engine archive integrity at configure" OFF)
# Linkage follows BUILD_SHARED_LIBS (cmake/validate-options.cmake). Bring your own
# engine via -DANIRA_<ENGINE>_ROOTDIR / _URL [+ _SHA256].

# ==============================================================================
# Project version from git (cmake/tanh/git-version.cmake)
# ==============================================================================

include(cmake/tanh/git-version.cmake)
# The 3.x line names itself after its own tags only: this branch also reaches the 2.x
# tags, and `git describe` would otherwise pick whichever is fewer commits away.
# Without a reachable v3 tag the build is 0.0.0+g<hash> (ABI 0.0, cmake/build-info.cmake).
tanh_git_version(${CMAKE_CURRENT_SOURCE_DIR} MATCH "v3*")
set(PROJECT_VERSION_SHORT "${TANH_VERSION_SHORT}")
set(PROJECT_VERSION_FULL "${TANH_VERSION_FULL}")

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION ${PROJECT_VERSION_SHORT})

# ANIRA_ABI_MAJOR / ANIRA_ABI_MINOR from the tag (rules in cmake/build-info.cmake); the
# generated <anira/abi/build_info.h> is attached to the library target below.
include(cmake/build-info.cmake)
anira_compute_abi_version()

# Platform axes (cmake/tanh/platform.cmake): TANH_OPERATING_SYSTEM, TANH_BINARY_FORMAT
# (ELF | Mach-O | PE | Wasm), TANH_IOS_PLATFORM. Every OS/format decision below keys on
# these, never on APPLE/UNIX/WIN32. anira_tanh_core carries the matching THL_PLATFORM_*
# compile definitions for C++. The symbol-policy module comes with it: engines.cmake
# (below, before the library target) already needs tanh_hidden_archive_link_items to
# wire the static engine archives.
include(cmake/tanh/platform.cmake)
include(cmake/tanh/symbol-policy.cmake)


message(STATUS "Building ${PROJECT_NAME} for ${CMAKE_SYSTEM_NAME} on ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Project version: ${PROJECT_VERSION_SHORT} (${PROJECT_VERSION_FULL})")

# Minimum macOS version 11.0 (the first release with full C++20 <semaphore> support)
# unless the caller set one — anira::Semaphore falls back to a portable backend below
# 11.0 (e.g. -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13). Never FORCE'd (cmake/tanh/apple.cmake).
include(cmake/tanh/apple.cmake)
tanh_apple_deployment_target(MACOS 11.0)
# Arch defaults + CMAKE_SYSTEM_PROCESSOR mirror for the engine asset picker. The
# prebuilt ONNX/LiteRT iOS xcframeworks ship an arm64-only simulator slice; an
# Intel simulator build needs no engine (or TFLite's fat framework).
tanh_apple_default_architectures()

# clangd reads build/compile_commands.json (.clangd). The development presets turn
# ANIRA_LINK_COMPILE_COMMANDS on, so each configure of one of their trees links its
# database there and clangd follows the tree configured last (cmake/tanh/compile-commands.cmake).
if(ANIRA_LINK_COMPILE_COMMANDS)
    include(cmake/tanh/compile-commands.cmake)
    tanh_link_compile_commands()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Detect Emscripten — must be after project() so CMAKE_CXX_COMPILER is set.
# EMSDK_VERSION is needed by engines.cmake (WASM target selection) below.
if(ANIRA_BUILD_WASM)
    tanh_detect_emscripten()
    if(WASM)
        # -pthread: Emscripten's threaded libc and libc++ (real futex mutexes across the Wasm
        # Workers and the audio worklet that share the module's memory; issue #212). Every
        # object of the link carries it, the fetched dependencies' C and C++ included.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -msimd128 -sNO_DISABLE_EXCEPTION_CATCHING")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -msimd128")
    endif()
endif()

# Validate the option combination (mutual exclusivity, libtorch auto-disable, WASM
# guards). After Emscripten detection so EMSDK_VERSION is known; before the engines
# are set up so the libtorch auto-disable takes effect.
include(cmake/validate-options.cmake)

# ==============================================================================
# Download and install the selected inference engines
# ==============================================================================

set(BACKEND_SOURCES)

# Each anira_setup_backend() call defines the engine's imported target anira::<id>
# (see cmake/engines.cmake and cmake/aniraEngineHelpers.cmake): the file
# anira links, carrying the engine's include directories and definitions. anira
# links them PRIVATE below; a consumer whose own code calls an engine links the
# matching anira::<id> explicitly and gets the same file — one engine per process.
include(cmake/engines.cmake)

if(ANIRA_WITH_LIBTORCH)
    anira_setup_backend(libtorch)
    list(APPEND BACKEND_SOURCES src/engines/LibTorchAdapter.cpp)
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    anira_setup_backend(onnxruntime)
    list(APPEND BACKEND_SOURCES src/engines/OnnxRuntimeAdapter.cpp)
endif()

if(ANIRA_WITH_TFLITE)
    anira_setup_backend(tflite)
    list(APPEND BACKEND_SOURCES src/engines/TFLiteAdapter.cpp)
endif()

if(ANIRA_WITH_LITERT)
    anira_setup_backend(litert)
    list(APPEND BACKEND_SOURCES src/engines/LiteRtAdapter.cpp)
endif()

if(ANIRA_WITH_EXECUTORCH)
    anira_setup_backend(executorch)
    list(APPEND BACKEND_SOURCES src/engines/ExecuTorchAdapter.cpp)
endif()

## ==============================================================================
# Fetch threadsafe queue
# ==============================================================================

include(FetchContent)

# Private and header-only (docs/anira-v3-architecture.md, section 6b): fetched without running its
# own CMake, whose install rules would put it into anira's package, and wrapped in a build-only
# interface target that the objects link.
FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG v1.0.5
    SOURCE_SUBDIR anira-uses-the-headers-only
)

FetchContent_MakeAvailable(concurrentqueue)
add_library(anira_concurrentqueue INTERFACE)
target_include_directories(anira_concurrentqueue SYSTEM INTERFACE "${concurrentqueue_SOURCE_DIR}")
# The queue's namespace is anira's own in every TU that includes it (the objects and the white-box
# suites that link them): moodycamel's templates carry no ABI namespace, so in a static libanira
# their instantiations would merge by name with a host's own concurrentqueue. The preprocessor
# renames the token; nothing is patched on disk, and no public header names the queue (gate 4).
target_compile_definitions(anira_concurrentqueue INTERFACE moodycamel=anira_moodycamel)

# ==============================================================================
# Fetch tanh-lib (Apache-2.0 core component only)
# ==============================================================================

# Only the Apache-2.0 core component is built, never the AGPL ones. It is private
# (docs/anira-v3-architecture.md, section 6b): anira_tanh_core, a static archive compiled hidden
# (TANH_STATIC, an empty TANH_API) in the inline namespace thl::anira, its objects absorbed into
# libanira below, nothing of it installed and nothing of it in anira's package or export table.
# The namespace keeps it apart from a host's own tanh-lib (thl::v<major>_<minor> by default, or
# whatever the host chose) on every platform and in every link mode: no symbol of the two copies
# shares a mangled name, header-only code included.
#
# The repository is fetched under anira's own FetchContent name and only populated: anira never
# runs tanh-lib's top-level CMakeLists.txt, so it builds its pin whatever a parent project
# declared under "tanh-lib", and a parent's tanh_core and anira's anira_tanh_core coexist in one
# build. The core is defined by the function of the pinned tree itself
# (cmake/core-library.cmake, included right before the call, as the function asks).
include(cmake/tanh-lib-pin.cmake)
FetchContent_Declare(
    anira_tanh_lib
    GIT_REPOSITORY ${ANIRA_TANH_LIB_REPOSITORY}
    GIT_TAG ${ANIRA_TANH_LIB_COMMIT}
    SOURCE_SUBDIR anira-uses-the-core-only
)
FetchContent_MakeAvailable(anira_tanh_lib)
if(NOT EXISTS "${anira_tanh_lib_SOURCE_DIR}/cmake/core-library.cmake")
    message(FATAL_ERROR
        "anira: the tanh-lib at ${anira_tanh_lib_SOURCE_DIR} has no cmake/core-library.cmake "
        "(tanh_add_core_library and the ABI namespace); anira needs the commit of "
        "cmake/tanh-lib-pin.cmake or a later one (FETCHCONTENT_SOURCE_DIR_ANIRA_TANH_LIB points "
        "anira at another checkout)")
endif()
include(${anira_tanh_lib_SOURCE_DIR}/cmake/core-library.cmake)
# Every log level is compiled into anira's private logger in every build type, so that the
# runtime level of the context config is the only filter and a warning the call survives
# exists in a shipped build (docs/anira-v3-error-and-log-strategy.md, rule 4). tanh-lib's
# AUTO would compile everything below Error out of Release builds.
tanh_add_core_library(anira_tanh_core
    SOURCE_DIR ${anira_tanh_lib_SOURCE_DIR}
    ABI_NAMESPACE anira
    STATIC
    LOG_COMPILED_MAX_LEVEL 4)
# Instrumented as anira is, with tanh-lib's own defines (its sources test TANH_WITH_RTSAN; UBSan
# is USAN in tanh-lib's spelling) and its ASan ignorelist, as tanh-lib instruments its tanh_core.
if(ANIRA_WITH_RTSAN OR ANIRA_WITH_ASAN OR ANIRA_WITH_UBSAN OR ANIRA_WITH_TSAN OR ANIRA_WITH_LSAN)
    include(cmake/tanh/sanitizers.cmake)
    if(ANIRA_WITH_RTSAN)
        tanh_add_sanitizer(anira_tanh_core rtsan DEFINE TANH_WITH_RTSAN)
    endif()
    if(ANIRA_WITH_ASAN)
        tanh_add_sanitizer(anira_tanh_core asan DEFINE TANH_WITH_ASAN
            IGNORELIST "${anira_tanh_lib_SOURCE_DIR}/sanitizers/asan_ignorelist.txt")
    endif()
    if(ANIRA_WITH_UBSAN)
        tanh_add_sanitizer(anira_tanh_core ubsan DEFINE TANH_WITH_USAN)
    endif()
    if(ANIRA_WITH_TSAN)
        tanh_add_sanitizer(anira_tanh_core tsan DEFINE TANH_WITH_TSAN)
    endif()
    if(ANIRA_WITH_LSAN)
        tanh_add_sanitizer(anira_tanh_core lsan DEFINE TANH_WITH_LSAN)
    endif()
endif()

# Fetch the release tarball instead of cloning the repo — the full git
# history plus test suite is ~150MB while the release archive is a few MB, the
# URL_HASH verifies integrity, and the tarball is content-addressable so CMake
# reuses a cached copy across build folders.
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    SYSTEM
)
# Private and header-only, like the queue: as a subproject nlohmann_json installs nothing
# (JSON_Install defaults to OFF there), and only the objects link it.
FetchContent_MakeAvailable(nlohmann_json)

# ==============================================================================
# Build the library
# ==============================================================================

# ------------------------------------------------------------------------------
# Symbol policy (cmake/tanh/symbol-policy.cmake; the rules are written up in
# docs/sphinx/troubleshooting.rst, "Host application ships its own engine runtime"):
#
#  * Hidden symbol visibility on every platform; the C entry points are re-exported through
#    ANIRA_API (include/anira/abi/export.h) — the same allowlist model Windows enforces via
#    dllexport. No C++ class or function carries it. This keeps the engine runtimes private: without it, weak
#    C++ symbols from the engine headers (e.g. ONNX Runtime's Ort::Global<void>::api_)
#    are exported from any plugin that embeds anira and get weak-coalesced by the
#    dynamic linker against a host that ships its own copy of the runtime (Ableton
#    Live 12 bundles an ONNX Runtime dylib), making Ort::GetApi() resolve against a
#    mismatched runtime and crashing the host on the first Ort:: call. ANIRA_BUILDING
#    (PRIVATE) selects dllexport while anira is compiled; ANIRA_STATIC (PUBLIC, through
#    the CMake package) makes ANIRA_API empty for a static anira, whose objects become
#    part of the consumer and must neither be exported from it nor be dllimport'ed.
#  * -fno-gnu-unique under GCC (glibc never unloads an object defining a unique symbol),
#    one section per function/variable and link-time GC of the shared library.
#  * The shared libanira pins its export table to the anira_* entry points at link time with a
#    generated version script / -exported_symbols_list: -fvisibility=hidden cannot hide
#    what a header stamps default-visibility itself (libstdc++ wraps namespace std in
#    visibility("default"), LibTorch's C10_API classes emit default-visibility
#    vtables/typeinfo into anira's own objects, and the prebuilt ExecuTorch desktop
#    archives are compiled with default visibility and force-loaded). Without it
#    libanira exported ~4000 std::, executorch::, xnn_*, c10:: and BLAS symbols next to
#    the ~400 of the API. test/unload registers the anira_exports CTest that diffs the
#    real export table on every desktop platform.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# The objects and the library. anira_objects compiles every source of anira once, under the
# symbol policy and with every build-time requirement (anira's include directories, the
# engines, the USE_* switches, tanh-lib, the queue and the JSON library). libanira, shared or
# static, is linked from those objects alone. The white-box test binaries (test/CMakeLists.txt)
# link the objects themselves and reach the private classes; everything else (the black-box
# suites, the gates, the install consumers, the examples) links anira::anira as a consumer does.
# ------------------------------------------------------------------------------
add_library(anira_objects OBJECT)
tanh_apply_symbol_policy(anira_objects EXPORT_PREFIX ANIRA)

# anira_tanh_core's objects (core.cpp, Dispatcher, Logger, Thread, RCU) are part of libanira,
# shared or static, compiled hidden with an empty TANH_API in the inline namespace thl::anira
# (section 6b of docs/anira-v3-architecture.md). A shared libanira exports none of them; a static
# one is linked into the consumer's image, where hidden separates nothing, but no symbol of anira's
# copy shares a mangled name with a host's own tanh-lib (thl::v<major>_<minor>), so the two never
# bind to each other on any platform. On ELF the static libanira is moreover one prelinked object
# in which every hidden symbol but the anira_* entries is local (cmake/prelink-archive.cmake): it
# keeps anira's own C++ internals out of the consumer's link. Mach-O and PE archives keep the
# objects as they are, and so do a multi-config generator and a toolchain without readelf and
# objcopy. The path taken is printed, and the static Android legs of CI check that it is the
# prelinked one (build_test_mobile.yml).
set(_anira_plain_reason "")
get_property(_anira_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT BUILD_SHARED_LIBS)
    if(NOT TANH_BINARY_FORMAT STREQUAL "ELF")
        set(_anira_plain_reason "not ELF: ${TANH_BINARY_FORMAT}")
    elseif(_anira_multi_config)
        # One fixed OUTPUT of the prelink cannot hold the objects of several configurations.
        set(_anira_plain_reason "multi-config generator")
    elseif(NOT CMAKE_READELF)
        set(_anira_plain_reason "no readelf")
    elseif(NOT CMAKE_OBJCOPY)
        set(_anira_plain_reason "no objcopy")
    endif()
    if(_anira_plain_reason STREQUAL "")
        message(STATUS "anira: static libanira prelinked (ELF; only anira_* global)")
    else()
        message(STATUS "anira: static libanira is a plain archive (${_anira_plain_reason})")
    endif()
endif()
if(NOT BUILD_SHARED_LIBS AND _anira_plain_reason STREQUAL "")
    set(_anira_prelinked "${CMAKE_CURRENT_BINARY_DIR}/anira_prelinked.o")
    # The cross-compiling target of a clang driver (an NDK's); g++ takes no --target. Nothing of
    # CMAKE_CXX_FLAGS is forwarded: a sanitizer or coverage flag on the -r link would drag its
    # runtime into the object, so a flag that changes the relocatable output itself (-m32 on a
    # 64-bit host) is not supported here.
    set(_anira_cxx_target "")
    if(CMAKE_CXX_COMPILER_TARGET AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(_anira_cxx_target "-DCXX_TARGET=${CMAKE_CXX_COMPILER_TARGET}")
    endif()
    add_custom_command(OUTPUT "${_anira_prelinked}"
        COMMAND ${CMAKE_COMMAND} "-DCXX=${CMAKE_CXX_COMPILER}" ${_anira_cxx_target}
            "-DREADELF=${CMAKE_READELF}" "-DOBJCOPY=${CMAKE_OBJCOPY}" "-DOUTPUT=${_anira_prelinked}"
            "-DINPUTS=$<TARGET_OBJECTS:anira_objects>;$<TARGET_OBJECTS:anira_tanh_core>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prelink-archive.cmake"
        DEPENDS anira_objects anira_tanh_core $<TARGET_OBJECTS:anira_objects>
            $<TARGET_OBJECTS:anira_tanh_core>
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prelink-archive.cmake"
        COMMENT "Prelinking the static libanira (one object, only anira_* global)"
        VERBATIM)
    # An object alone names no language. The archive is C++ inside, and CMake links a static
    # library's consumers with the runtimes of its source languages (its link interface languages),
    # so an empty C++ source names it, as the objects did. Written only when its text changes: a
    # reconfigure must not re-archive libanira.a and relink every static consumer.
    set(_anira_link_language "${CMAKE_CURRENT_BINARY_DIR}/anira_link_language.cpp")
    set(_anira_link_language_text
        "// The link language of the prelinked static libanira (CMakeLists.txt): C++.\n")
    set(_anira_link_language_old "")
    if(EXISTS "${_anira_link_language}")
        file(READ "${_anira_link_language}" _anira_link_language_old)
    endif()
    if(NOT _anira_link_language_old STREQUAL _anira_link_language_text)
        file(WRITE "${_anira_link_language}" "${_anira_link_language_text}")
    endif()
    add_library(${PROJECT_NAME} STATIC "${_anira_prelinked}" "${_anira_link_language}")
    unset(_anira_prelinked)
    unset(_anira_cxx_target)
    unset(_anira_link_language)
    unset(_anira_link_language_text)
    unset(_anira_link_language_old)
else()
    add_library(${PROJECT_NAME} $<TARGET_OBJECTS:anira_objects> $<TARGET_OBJECTS:anira_tanh_core>)
endif()
unset(_anira_plain_reason)
unset(_anira_multi_config)
# The shared object is named after the ABI pair (cmake/build-info.cmake), not after the release:
# the soname is the ABI major (libanira.so.0, libanira.0.dylib and anira-0.dll through the 3.0.0
# pre-releases; 3 from v3.0.0 on) and the file version is the pair. On Mach-O the current and the
# compatibility version are the pair too, so a client linked against one ABI minor refuses an
# older library. On PE, where a DLL has no soname, the major is part of the DLL's name; a static
# archive keeps the plain name.
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${ANIRA_ABI_MAJOR}.${ANIRA_ABI_MINOR}
    SOVERSION ${ANIRA_ABI_MAJOR}
    MACHO_CURRENT_VERSION ${ANIRA_ABI_MAJOR}.${ANIRA_ABI_MINOR}
    MACHO_COMPATIBILITY_VERSION ${ANIRA_ABI_MAJOR}.${ANIRA_ABI_MINOR})
if(BUILD_SHARED_LIBS AND TANH_BINARY_FORMAT STREQUAL "PE")
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME}-${ANIRA_ABI_MAJOR})
endif()
tanh_apply_symbol_policy(${PROJECT_NAME} EXPORT_PREFIX ANIRA)
# The exported surface is the C ABI alone: the entry points anira_* (extern "C"). No C++ symbol
# of anira crosses the library boundary; anira.hpp and compat/v2.hpp are header-only over the C
# entries, and the white-box tests link the objects (anira_objects) instead.
tanh_set_export_allowlist(${PROJECT_NAME} SYMBOL "anira_*")
# add an alias so that the project can be used with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(anira_objects
    PRIVATE

        # Backend
        src/engines/Adapter.cpp
        src/engines/Adapters.cpp
        src/engines/DescriptorAdapter.cpp
        ${BACKEND_SOURCES}

        # Scheduler
        src/scheduler/InferenceManager.cpp
        src/scheduler/InferenceThread.cpp
        src/scheduler/Core.cpp
        src/scheduler/SessionElement.cpp
        src/scheduler/LatencyCalculator.cpp

        # Utils
        src/utils/Logger.cpp
        src/utils/LogSinks.cpp
        src/utils/RtLatch.cpp

        # Interface

        # System
)

# include the public headers of the anira library in the top-level project include directory
target_include_directories(anira_objects
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# <anira/abi/build_info.h>: the version triple and the ABI pair, generated into the
# build tree (never under include/, which the install rule copies verbatim).
anira_generate_build_info(anira_objects)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ANIRA_GENERATED_INCLUDE_DIR}>
)

# The C ABI (include/anira/abi/, generated from abi/anira.yml by tools/abi/gen.py) and
# its implementation: every entry body sits behind the exception firewall of
# src/capi/capi_internal.h; the generated link probe (test/abi/generated/link_probe.c,
# anira_abi_link) makes a promised name without a definition a link error.
target_sources(anira_objects PRIVATE
    src/capi/config.cpp
    src/capi/error.cpp
    src/capi/ext_registry.cpp
    src/capi/json.cpp
    src/capi/layout.cpp
    src/capi/log.cpp
    src/capi/context.cpp
    src/capi/core.cpp
    src/capi/thread.cpp
    src/capi/validate.cpp
    src/capi/providers.cpp
    src/capi/version.cpp
    src/capi/handler.cpp
    src/capi/processor.cpp
    src/capi/engine.cpp
    src/capi/tensor.cpp
)
# anira/anira_em.h, the Emscripten companion (registry `platform: emscripten`): the trampolines
# into the JavaScript hook object, the JS log sink, the inference Workers and the Wasm Audio
# Worklet node. The header is empty on every other binary format, so the TUs exist on Wasm
# alone.
if(TANH_BINARY_FORMAT STREQUAL "Wasm")
    target_sources(anira_objects PRIVATE
        src/platform/emscripten/em_audio.cpp
        src/platform/emscripten/em_console.cpp
        src/platform/emscripten/em_hooks.cpp
        src/platform/emscripten/em_runtime.cpp
        src/platform/emscripten/em_worker.cpp)
    # Two objects every consumer's module needs whether or not it calls into them, so every
    # link against anira references their anchors: the console sink (em_console.cpp), switched
    # on by its TU's static constructor, and the runtime that never exits on its own
    # (em_runtime.cpp, the noExitRuntime dependency).
    target_link_options(${PROJECT_NAME} INTERFACE
        "-Wl,--undefined=anira_em_console_anchor"
        "-Wl,--undefined=anira_em_runtime_anchor"
        # Out of memory is a status (ANIRA_ERROR_OUT_OF_MEMORY), not an abort: malloc returns
        # NULL and operator new throws std::bad_alloc, which every entry's firewall turns into
        # the status. Emscripten's default with a fixed memory (ALLOW_MEMORY_GROWTH=0) aborts
        # the module instead. A consumer that prefers the abort passes -sABORTING_MALLOC=1
        # after this (the last -s wins).
        "SHELL:-s ABORTING_MALLOC=0")
    # A Wasm executable that links the library (AniraWeb and the node gates of test/abi) peaks
    # at about 4.9 GB while it links (anira_em_pool, measured on 2026-10-29); four at once ran a
    # 16 GB CI runner out of memory. Those links join this Ninja pool, so at most two run at a time.
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS anira_wasm_link=2)
endif()

# The objects are compiled with them. They are build-time switches, never a consumer's: anira's
# install interface carries none of them (what a build compiled in is anira_enabled_engines).
# In this build tree anira passes them on to the bundled consumers, whose sources still select
# their engine configurations at compile time (the extras, the examples, test_InferenceHandler).
set(ANIRA_COMPILE_DEFINITIONS
    # Backend-specific definitions
    $<$<BOOL:${ANIRA_WITH_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${ANIRA_WITH_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<STREQUAL:${TANH_OPERATING_SYSTEM},Emscripten>:USE_ANIRA_WEB>
    $<$<BOOL:${ANIRA_WITH_TFLITE}>:USE_TFLITE>
    $<$<BOOL:${ANIRA_WITH_LITERT}>:USE_LITERT>
    $<$<BOOL:${ANIRA_WITH_EXECUTORCH}>:USE_EXECUTORCH>
    $<$<BOOL:${ANIRA_WITH_LOGGING}>:ENABLE_LOGGING>
    # Version number
    -DANIRA_VERSION="${PROJECT_VERSION_FULL}"
)

target_compile_definitions(anira_objects PUBLIC ${ANIRA_COMPILE_DEFINITIONS})
foreach(_anira_definition IN LISTS ANIRA_COMPILE_DEFINITIONS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${_anira_definition}>)
endforeach()
unset(_anira_definition)

# The internal chain behind the ANIRA_NONBLOCKING entries is annotated post-3.0; until
# then clang's function-effects analysis is silenced for the library alone (consumer-shaped
# targets carry -Werror=function-effects from the tensor PR on).
target_compile_options(anira_objects PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-function-effects>
)

if(ANIRA_WITH_RTSAN OR ANIRA_WITH_ASAN OR ANIRA_WITH_UBSAN OR ANIRA_WITH_TSAN OR ANIRA_WITH_LSAN)
    include(cmake/tanh/sanitizers.cmake)
endif()
# The objects are instrumented; the library carries the same flags on to its consumers' links.
foreach(_anira_target anira_objects ${PROJECT_NAME})
    if(ANIRA_WITH_RTSAN)
        tanh_add_sanitizer(${_anira_target} rtsan DEFINE ANIRA_WITH_RTSAN)
    endif()
    if(ANIRA_WITH_ASAN)
        tanh_add_sanitizer(${_anira_target} asan DEFINE ANIRA_WITH_ASAN)
    endif()
    if(ANIRA_WITH_UBSAN)
        tanh_add_sanitizer(${_anira_target} ubsan DEFINE ANIRA_WITH_UBSAN)
    endif()
    if(ANIRA_WITH_TSAN)
        tanh_add_sanitizer(${_anira_target} tsan DEFINE ANIRA_WITH_TSAN)
    endif()
    if(ANIRA_WITH_LSAN)
        tanh_add_sanitizer(${_anira_target} lsan DEFINE ANIRA_WITH_LSAN)
    endif()

    # Backends link PRIVATE through their anira::<engine> targets (details per
    # engine in cmake/engines/<engine>.cmake; how each is linked in
    # cmake/aniraEngineHelpers.cmake): the library for its own link and, when static, its
    # export; the objects so that a white-box binary linking them gets the same closure.
    anira_link_backends(${_anira_target})
    # The LibTorch engine's init looks OpenBLAS up with dlsym (src/engines/LibTorchAdapter.cpp,
    # pin_openblas_to_one_thread); glibc before 2.34 keeps dlsym in libdl.
    if(ANIRA_WITH_LIBTORCH AND TANH_BINARY_FORMAT STREQUAL "ELF")
        target_link_libraries(${_anira_target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endforeach()
unset(_anira_target)

# tanh-lib, the queue and the JSON library: the objects' build-time requirements (include
# directories, TANH_STATIC and the THL_* definitions), which reach the white-box binaries that
# link the objects and nothing else; libanira carries tanh's objects (above) and links none of
# the three. The one system library anira_tanh_core links, repeated for the absorbed objects.
target_link_libraries(anira_objects PUBLIC anira_concurrentqueue nlohmann_json::nlohmann_json anira_tanh_core)
if(TANH_OPERATING_SYSTEM STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} PRIVATE log)
endif()

# LibTorch's libstdc++ ABI switch (TORCH_CXX_FLAGS, -D_GLIBCXX_USE_CXX11_ABI) is a requirement of
# every translation unit that shares C++ types with LibTorch: the objects and the white-box
# binaries that link them. No C++ type crosses libanira's boundary, so its consumers do not get it.
if(ANIRA_WITH_LIBTORCH AND TORCH_CXX_FLAGS)
    separate_arguments(_anira_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(anira_objects PUBLIC ${_anira_torch_cxx_flags})
    unset(_anira_torch_cxx_flags)
endif()

if(ANIRA_WITH_BENCHMARK OR ANIRA_WITH_TESTS)
    include(cmake/test-deps.cmake)
endif()

# ==============================================================================
# Add install targets for the library
# ==============================================================================

if(ANIRA_WITH_INSTALL)
    include(cmake/install.cmake)
    include(cmake/package.cmake)   # CPack (cmake/tanh/package.cmake), after every install() rule
endif()

# ==============================================================================
# Add support for MSVC
# ==============================================================================

if (MSVC)
    include(cmake/msvc-support.cmake)
endif()

# ==============================================================================
# Build example targets and add extras (clone example model repos)
# ==============================================================================

# First we import the extras folder since we need the compile definitions (model paths) for the examples
if(ANIRA_WITH_EXAMPLES OR ANIRA_WITH_TESTS OR ANIRA_WITH_BENCHMARK)
    add_subdirectory(extras)
endif()

# The examples, and the benchmarks with their fixture (examples/benchmark), which
# ANIRA_WITH_BENCHMARK builds on its own as well.
if(ANIRA_WITH_EXAMPLES OR ANIRA_WITH_BENCHMARK)
    add_subdirectory(examples)
endif()

if (ANIRA_WITH_TESTS)
    add_subdirectory(test)
elseif(ANIRA_WITH_ABI_TESTS)
    # The gates of test/abi without the rest of the suite: on WebAssembly the layout printer
    # and the anira_em.h smoke run under node (test/abi/CMakeLists.txt, its Wasm arm).
    add_subdirectory(test/abi)
endif()

if (ANIRA_WITH_DOCS)
    add_subdirectory(docs)
endif()

# ==============================================================================
# Build WebAssembly module (optional, requires Emscripten)
# ==============================================================================

if(ANIRA_BUILD_WASM)
    include(cmake/build-wasm.cmake)
endif()
