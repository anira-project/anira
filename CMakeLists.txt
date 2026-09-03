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
option(ANIRA_WITH_BENCHMARK "Build the library with benchmarking capabilities" OFF)
option(ANIRA_WITH_EXAMPLES  "Add example targets (juce plugin, benchmarks, minimal inference and model examples)" OFF)
option(ANIRA_WITH_INSTALL   "Add install targets" OFF)
option(ANIRA_WITH_TESTS     "Add build tests" OFF)
option(ANIRA_WITH_DOCS      "Build documentation" OFF)
option(ANIRA_WITH_LOGGING   "Enable logging printouts" ON)
option(ANIRA_WITH_RTSAN     "Enable RealtimeSanitizer (rtsan) checks (requires clang 20)" OFF)
option(ANIRA_WITH_ASAN      "Enable AddressSanitizer" OFF)
option(ANIRA_WITH_UBSAN     "Enable UndefinedBehaviorSanitizer" OFF)
option(ANIRA_WITH_TSAN      "Enable ThreadSanitizer" OFF)
option(ANIRA_WITH_LSAN      "Enable LeakSanitizer (standalone)" OFF)
option(ANIRA_BUILD_WASM     "Build WebAssembly module (requires Emscripten toolchain)" OFF)

# ==============================================================================
# Inference backends — wiring per engine in cmake/backends/<engine>.cmake,
# prebuilts from anira-project/backends (cmake/backends.cmake). LiteRT and
# TFLite are the same runtime through different C APIs — mutually exclusive.
# ==============================================================================
option(ANIRA_WITH_LIBTORCH    "Build with the LibTorch backend" ON)
option(ANIRA_WITH_ONNXRUNTIME "Build with the ONNX Runtime backend" ON)
option(ANIRA_WITH_LITERT      "Build with the LiteRT backend (LiteRt* C API; runs .tflite via the CompiledModel runtime)" ON)
option(ANIRA_WITH_TFLITE      "Build with the legacy TensorFlow Lite backend (TfLite* C API); mutually exclusive with ANIRA_WITH_LITERT" OFF)
option(ANIRA_WITH_EXECUTORCH  "Build with the ExecuTorch backend (runs .pte models; static-only)" ON)

set(ANIRA_BACKENDS_VERSION "" CACHE STRING "anira-project/backends release tag to download (default: built-in)")
option(ANIRA_BACKENDS_SKIP_REMOTE_CHECK "Don't query GitHub for backend integrity at configure" OFF)
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
# compile definitions for C++. The symbol-policy module comes with it: backends.cmake
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
# Arch defaults + CMAKE_SYSTEM_PROCESSOR mirror for the backend asset picker. The
# prebuilt ONNX/LiteRT iOS xcframeworks ship an arm64-only simulator slice; an
# Intel simulator build needs no backend (or TFLite's fat framework).
tanh_apple_default_architectures()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Detect Emscripten — must be after project() so CMAKE_CXX_COMPILER is set.
# EMSDK_VERSION is needed by backends.cmake (WASM target selection) below.
if(ANIRA_BUILD_WASM)
    tanh_detect_emscripten()
    if(WASM)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -matomics -msimd128 -mbulk-memory -sNO_DISABLE_EXCEPTION_CATCHING")
    endif()
endif()

# Validate the option combination (mutual exclusivity, libtorch auto-disable, WASM
# guards). After Emscripten detection so EMSDK_VERSION is known; before the backends
# are set up so the libtorch auto-disable takes effect.
include(cmake/validate-options.cmake)

# ==============================================================================
# Download and install the selected inference engines
# ==============================================================================

set(BACKEND_SOURCES)

# Each anira_setup_backend() call defines the engine's imported target anira::<id>
# (see cmake/backends.cmake and cmake/aniraBackendHelpers.cmake): the file
# anira links, carrying the engine's include directories and definitions. anira
# links them PRIVATE below; a consumer whose own code calls an engine links the
# matching anira::<id> explicitly and gets the same file — one engine per process.
include(cmake/backends.cmake)

if(ANIRA_WITH_LIBTORCH)
    anira_setup_backend(libtorch)
    list(APPEND BACKEND_SOURCES src/backends/LibTorchProcessor.cpp)
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    anira_setup_backend(onnxruntime)
    list(APPEND BACKEND_SOURCES src/backends/OnnxRuntimeProcessor.cpp)
endif()

if(ANIRA_WITH_TFLITE)
    anira_setup_backend(tflite)
    list(APPEND BACKEND_SOURCES src/backends/TFLiteProcessor.cpp)
endif()

if(ANIRA_WITH_LITERT)
    anira_setup_backend(litert)
    list(APPEND BACKEND_SOURCES src/backends/LiteRtProcessor.cpp)
endif()

if(ANIRA_WITH_EXECUTORCH)
    anira_setup_backend(executorch)
    list(APPEND BACKEND_SOURCES src/backends/ExecuTorchProcessor.cpp)
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

FetchContent_Declare(
    tanh-lib
    GIT_REPOSITORY https://github.com/tanh-lab/tanh-lib.git
    # anira and the fetched tanh-lib must pin the same tanh-tooling tag
    # (modules-version.cmake warns otherwise).
    # v0.2.0: the upstream container coverage of tanh-lab/tanh-lib#31 (which the trimmed
    # test/utils container tests rely on), the self-contained TANH_API header (#35) and
    # tanh-tooling v0.2.8 (#36), the same tooling tag this tree pins.
    GIT_TAG 488ad6ab80aeaed2e7b9ebda70a92910b63d35f4
)
FetchContent_MakeAvailable(tanh-lib)

# One compiled log ceiling in every build type. tanh-lib compiles Warning, Info and Debug
# out of Release builds (THL_LOG_COMPILED_MAX_LEVEL=1, PUBLIC on tanh_core, 4 otherwise).
# anira's private copy keeps every level compiled in, so that the runtime level of the
# machine config is the only filter and a warning the call survives exists in a shipped
# build (docs/anira-v3-error-and-log-strategy.md, rule 4).
foreach(_prop COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
    get_target_property(_defs tanh_core ${_prop})
    if(_defs)
        list(FILTER _defs EXCLUDE REGEX "THL_LOG_COMPILED_MAX_LEVEL")
    else()
        set(_defs "")
    endif()
    set_target_properties(tanh_core PROPERTIES ${_prop} "${_defs};THL_LOG_COMPILED_MAX_LEVEL=4")
endforeach()
unset(_defs)
unset(_prop)

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
# docs/sphinx/troubleshooting.rst, "Host application ships its own backend runtime"):
#
#  * Hidden symbol visibility on every platform; the public API is re-exported through
#    ANIRA_API (include/anira/system/Exports.h) — the same allowlist model Windows
#    enforces via dllexport. This keeps the backend runtimes private: without it, weak
#    C++ symbols from the backend headers (e.g. ONNX Runtime's Ort::Global<void>::api_)
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
add_library(${PROJECT_NAME})
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

target_sources(${PROJECT_NAME}
    PRIVATE

        # Backend
        src/backends/BackendBase.cpp
        ${BACKEND_SOURCES}

        # Scheduler
        src/scheduler/InferenceManager.cpp
        src/scheduler/InferenceThread.cpp
        src/scheduler/Context.cpp
        src/scheduler/SessionElement.cpp
        src/scheduler/LatencyCalculator.cpp

        # Utils
        src/utils/JsonConfigLoader.cpp
        src/utils/Logger.cpp

        # Interface
        src/InferenceHandler.cpp
        src/PrePostProcessor.cpp
        src/InferenceConfig.cpp

        # System
)

# include the public headers of the anira library in the top-level project include directory
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# <anira/abi/build_info.h>: the version triple and the ABI pair, generated into the
# build tree (never under include/, which the install rule copies verbatim).
anira_generate_build_info(${PROJECT_NAME})

# The C ABI (include/anira/abi/, generated from abi/anira.yml by tools/abi/gen.py) and
# its implementation: every entry body sits behind the exception firewall of
# src/capi/capi_internal.h; the generated proc table makes a promised name without a
# definition a link error.
target_sources(${PROJECT_NAME} PRIVATE
    src/capi/config.cpp
    src/capi/error.cpp
    src/capi/ext_registry.cpp
    src/capi/json.cpp
    src/capi/layout.cpp
    src/capi/log.cpp
    src/capi/version.cpp
)

# Shared between the anira target and the isolated ExecuTorch object target below —
# the two must agree on every USE_* define (they shape the anira class layouts).
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

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
    ${ANIRA_COMPILE_DEFINITIONS}
)

if(ANIRA_WITH_RTSAN OR ANIRA_WITH_ASAN OR ANIRA_WITH_UBSAN OR ANIRA_WITH_TSAN OR ANIRA_WITH_LSAN)
    include(cmake/tanh/sanitizers.cmake)
endif()
if(ANIRA_WITH_RTSAN)
    tanh_add_sanitizer(${PROJECT_NAME} rtsan DEFINE ANIRA_WITH_RTSAN)
endif()
if(ANIRA_WITH_ASAN)
    tanh_add_sanitizer(${PROJECT_NAME} asan DEFINE ANIRA_WITH_ASAN)
endif()
if(ANIRA_WITH_UBSAN)
    tanh_add_sanitizer(${PROJECT_NAME} ubsan DEFINE ANIRA_WITH_UBSAN)
endif()
if(ANIRA_WITH_TSAN)
    tanh_add_sanitizer(${PROJECT_NAME} tsan DEFINE ANIRA_WITH_TSAN)
endif()
if(ANIRA_WITH_LSAN)
    tanh_add_sanitizer(${PROJECT_NAME} lsan DEFINE ANIRA_WITH_LSAN)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC concurrentqueue nlohmann_json::nlohmann_json tanh::Core)

# Backends link PRIVATE through their anira::<engine> targets (details per
# engine in cmake/backends/<engine>.cmake; how each is linked in
# cmake/aniraBackendHelpers.cmake).
anira_link_backends(${PROJECT_NAME})

if(ANIRA_WITH_BENCHMARK OR ANIRA_WITH_TESTS)
    include(cmake/test-deps.cmake)
endif()

if(ANIRA_WITH_BENCHMARK)
    include(cmake/benchmark-src.cmake)
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
if(ANIRA_WITH_EXAMPLES OR ANIRA_WITH_TESTS)
    add_subdirectory(extras)
endif()

if(ANIRA_WITH_EXAMPLES)    
    add_subdirectory(examples)
endif()

if (ANIRA_WITH_TESTS)
    add_subdirectory(test)
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
