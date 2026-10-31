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
# these, never on APPLE/UNIX/WIN32. tanh::Core carries the matching THL_PLATFORM_*
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

FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG v1.0.5
)

FetchContent_MakeAvailable(concurrentqueue)

# ==============================================================================
# Fetch tanh-lib (Apache-2.0 core component only)
# ==============================================================================

# Only the Apache-2.0 core component (tanh::Core) is fetched — never the AGPL
# ones. Installs re-resolve it via find_dependency(tanh COMPONENTS Core).
# Plain (non-cache) variables: they shadow tanh-lib's option() defaults under
# CMP0077 NEW without clobbering a parent that fetches tanh-lib itself.
set(TANH_BUILD_CORE ON)
set(TANH_BUILD_STATE OFF)
set(TANH_BUILD_DSP OFF)
set(TANH_BUILD_MODULATION OFF)
set(TANH_BUILD_AUDIO_IO OFF)
set(TANH_WITH_TESTS OFF)
set(TANH_WITH_EXAMPLES OFF)
set(TANH_WITH_DOCS OFF)
set(TANH_WITH_JOURNALD OFF)
set(TANH_WITH_RTSAN ${ANIRA_WITH_RTSAN})
# tanh-lib is instrumented in the same build (its options mirror anira's;
# UBSan is USAN in tanh-lib's spelling).
set(TANH_WITH_ASAN ${ANIRA_WITH_ASAN})
set(TANH_WITH_USAN ${ANIRA_WITH_UBSAN})
set(TANH_WITH_TSAN ${ANIRA_WITH_TSAN})
set(TANH_WITH_LSAN ${ANIRA_WITH_LSAN})
set(TANH_WITH_INSTALL ${ANIRA_WITH_INSTALL})
# Every log level is compiled into anira's private logger in every build type, so that the
# runtime level of the context config is the only filter and a warning the call survives
# exists in a shipped build (docs/anira-v3-error-and-log-strategy.md, rule 4). tanh-lib's
# own default would compile everything below Error out of Release builds.
set(TANH_LOG_COMPILED_MAX_LEVEL 4)

FetchContent_Declare(
    tanh-lib
    GIT_REPOSITORY https://github.com/tanh-lab/tanh-lib.git
    # anira and the fetched tanh-lib must pin the same tanh-tooling tag
    # (modules-version.cmake warns otherwise).
    # v0.4.0 plus tanh-tooling v0.3.0 (tanh-lab/tanh-lib#73; nothing in the core changed
    # since v0.4.0). Needed since v0.4.0: the strided RingBuffer calls (tanh-lab/tanh-lib#50:
    # push_block / pop_block / peek_past_block with a stride) that anira_ring forwards to.
    # Needed since v0.3.0: the logger carriers (tanh-lab/tanh-lib#37: platform identity,
    # per-record flags, the drop count on the record) and the TANH_LOG_COMPILED_MAX_LEVEL
    # option (#38). tanh-tooling v0.3.0, the same tooling tag this tree pins.
    GIT_TAG 7ae7c690d57806acc56dd4b5ffbba50551913966
)
FetchContent_MakeAvailable(tanh-lib)

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
# When installing anira, also install nlohmann_json's headers + CMake config so the
# installed aniraConfig.cmake can resolve find_package(nlohmann_json). As a subproject
# nlohmann_json defaults JSON_Install=OFF (unlike concurrentqueue, which installs by
# default), which previously left the installed package referencing a config that was
# never shipped.
if(ANIRA_WITH_INSTALL)
    set(JSON_Install ON)
endif()
FetchContent_MakeAvailable(nlohmann_json)

# ==============================================================================
# Build the library
# ==============================================================================

# ------------------------------------------------------------------------------
# Symbol policy (cmake/tanh/symbol-policy.cmake; the rules are written up in
# docs/sphinx/troubleshooting.rst, "Host application ships its own engine runtime"):
#
#  * Hidden symbol visibility on every platform; the public API is re-exported through
#    ANIRA_API (include/anira/system/Exports.h) — the same allowlist model Windows
#    enforces via dllexport. This keeps the engine runtimes private: without it, weak
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
#  * The shared libanira pins its export table to namespace anira at link time with a
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

add_library(${PROJECT_NAME} $<TARGET_OBJECTS:anira_objects>)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
tanh_apply_symbol_policy(${PROJECT_NAME} EXPORT_PREFIX ANIRA)
# The exported surface: the C++ namespace and, from the 3.x line on, the C entry
# points anira_* (extern "C", so the namespace pattern cannot cover them).
tanh_set_export_allowlist(${PROJECT_NAME} NAMESPACE anira SYMBOL "anira_*")
# tanh-lib's components are compiled hidden with an empty TANH_API when static, so the
# objects of libtanh_core.a are already hidden inside a plugin that embeds a static
# anira. Localizing the archive on the consumer's link on top of that is
# defense-in-depth against a tanh-lib built without that policy and against the
# default-visibility libstdc++ instantiations its objects carry; ELF only (Mach-O
# consumers own their export list), INTERFACE only (a shared libanira links
# libtanh_core.so, a runtime dependency, not something to hide).
if(NOT BUILD_SHARED_LIBS AND TANH_BINARY_FORMAT STREQUAL "ELF")
    target_link_options(${PROJECT_NAME} INTERFACE "LINKER:--exclude-libs,libtanh_core.a")
endif()
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
# src/capi/capi_internal.h; the generated proc table makes a promised name without a
# definition a link error.
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
# into the JavaScript hook object, the JS log sink and the per-instance stacks. The header is
# empty on every other binary format, so the TU exists on Wasm alone.
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

# The objects are compiled with them, and anira passes them on to its consumers; the header
# gates of test/abi take the same list.
set(ANIRA_COMPILE_DEFINITIONS
    # Backend-specific definitions
    $<$<BOOL:${ANIRA_WITH_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${ANIRA_WITH_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:${EMSDK_VERSION}>:USE_ANIRA_WEB>
    $<$<BOOL:${ANIRA_WITH_TFLITE}>:USE_TFLITE>
    $<$<BOOL:${ANIRA_WITH_LITERT}>:USE_LITERT>
    $<$<BOOL:${ANIRA_WITH_EXECUTORCH}>:USE_EXECUTORCH>
    $<$<BOOL:${ANIRA_WITH_LOGGING}>:ENABLE_LOGGING>
    # Version number
    -DANIRA_VERSION="${PROJECT_VERSION_FULL}"
)

foreach(_anira_target anira_objects ${PROJECT_NAME})
    target_compile_definitions(${_anira_target}
        PUBLIC
        ${ANIRA_COMPILE_DEFINITIONS}
    )
endforeach()

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

    target_link_libraries(${_anira_target} PUBLIC concurrentqueue nlohmann_json::nlohmann_json tanh::Core)

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
