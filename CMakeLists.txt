cmake_minimum_required(VERSION 3.15)

# CMP0099 (CMake 3.19): INTERFACE_LINK_OPTIONS (and _DIRECTORIES/_DEPENDS) of a
# static library's PRIVATE dependencies are transitive to whoever links the static
# library. A static anira links its engine archives PRIVATE through the
# anira::<engine> targets, whose INTERFACE_LINK_OPTIONS carry the --exclude-libs
# that keep the archives' symbols out of the consumer's export table; with the OLD
# behaviour those options stopped at anira and the plugin embedding it exported
# every LiteRt*/Ort*/executorch:: symbol. (The archives themselves travel as link
# dependencies either way.) The installed export records the same policy.
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
option(ANIRA_BUILD_WASM     "Build WebAssembly module (requires Emscripten toolchain)" OFF)

# --- Inference backends --------------------------------------------------------
# Multiple backends can be enabled. LiteRT (LiteRt* C API) is the default
# TensorFlow-family backend; TFLite (legacy TfLite* C API) is the SAME underlying
# runtime through the older API, so the two are mutually exclusive (their static
# libraries export the same TfLite* symbols). Enforced in cmake/validate-options.cmake.
option(ANIRA_WITH_LIBTORCH    "Build with the LibTorch backend" ON)
option(ANIRA_WITH_ONNXRUNTIME "Build with the ONNX Runtime backend" ON)
option(ANIRA_WITH_LITERT      "Build with the LiteRT backend (LiteRt* C API; runs .tflite via the CompiledModel runtime)" ON)
option(ANIRA_WITH_TFLITE      "Build with the legacy TensorFlow Lite backend (TfLite* C API); mutually exclusive with ANIRA_WITH_LITERT" OFF)
option(ANIRA_WITH_EXECUTORCH  "Build with the ExecuTorch backend (runs .pte models; static-only)" ON)

# --- Pre-built backend download ------------------------------------------------
# Backends are downloaded from the anira-project/backends release with this tag.
# Integrity is checked live: anira fetches each asset's published sha256 from GitHub
# at configure (when reachable) and re-downloads anything that changed upstream or
# downloaded incompletely — nothing is pinned in-repo. Empty -> built-in default tag.
set(ANIRA_BACKENDS_VERSION "" CACHE STRING "anira-project/backends release tag to download (default: built-in)")

# Skip the live integrity check (no GitHub query at configure). For fully offline /
# reproducible builds; a backend already in modules/ is then reused as-is.
option(ANIRA_BACKENDS_SKIP_REMOTE_CHECK "Don't query GitHub for backend integrity at configure" OFF)

# Backend linkage follows BUILD_SHARED_LIBS: a shared anira links shared backends, a
# static anira links static backends, and an engine that does not ship the required
# linkage (LibTorch in static builds, ExecuTorch in shared ones) is disabled with a
# warning — see cmake/validate-options.cmake for why these are the only two shapes.
# Bring your own engine, or use a custom source, via the per-engine variables
# (<ENGINE> = LIBTORCH | ONNXRUNTIME | TFLITE | LITERT | EXECUTORCH):
#   -DANIRA_<ENGINE>_ROOTDIR=/path/to/prebuilt   (a tree with include/ + lib/)
#   -DANIRA_<ENGINE>_URL=...  [-DANIRA_<ENGINE>_SHA256=...]

# ==============================================================================
# Project version from git (cmake/tanh/git-version.cmake)
# ==============================================================================

include(cmake/tanh/git-version.cmake)
tanh_git_version(${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_VERSION_SHORT "${TANH_VERSION_SHORT}")
set(PROJECT_VERSION_FULL "${TANH_VERSION_FULL}")

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION ${PROJECT_VERSION_SHORT})

# Platform axes (cmake/tanh/platform.cmake): TANH_OPERATING_SYSTEM, TANH_BINARY_FORMAT
# (ELF | Mach-O | PE | Wasm), TANH_IOS_PLATFORM. Every OS/format decision below keys on
# these, never on APPLE/UNIX/WIN32. tanh::Core carries the matching THL_PLATFORM_*
# compile definitions for C++. The symbol-policy module comes with it: backends.cmake
# (below, before the library target) already needs tanh_hidden_archive_link_items to
# wire the static engine archives.
include(cmake/tanh/platform.cmake)
include(cmake/tanh/symbol-policy.cmake)

if(APPLE)
    if(TANH_OPERATING_SYSTEM STREQUAL "iOS")
        # iOS targets a single architecture per xcframework slice; default to arm64
        # (device and, on Apple Silicon, simulator) when the caller didn't pick one.
        # Note: the prebuilt ONNX/LiteRT iOS xcframeworks ship an arm64-only simulator
        # slice, so an x86_64 (Intel) simulator build is only viable with no backend
        # (or the TFLite framework, which ships a fat simulator slice).
        if(NOT CMAKE_OSX_ARCHITECTURES)
            set(CMAKE_OSX_ARCHITECTURES "arm64")
        endif()
    else()
        # macOS: default to the host architecture when unset. A single-arch selection is
        # mirrored into CMAKE_SYSTEM_PROCESSOR for the backend asset picker; a universal
        # (arm64;x86_64) selection is left as-is and maps to the universal backend archives.
        if(NOT CMAKE_OSX_ARCHITECTURES)
            set(CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
        endif()
    endif()
endif()

# Print the processor architecture selected for build. Defaults to CMAKE_HOST_SYSTEM_PROCESSOR when no crosscompile tolchain is defined. CMAKE_HOST_SYSTEM_PROCESSOR is only defined after the project() call. For OSX this value can be overwritten by the CMAKE_OSX_ARCHITECTURES.

message(STATUS "Building ${PROJECT_NAME} for ${CMAKE_SYSTEM_NAME} on ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Project version: ${PROJECT_VERSION_SHORT} (${PROJECT_VERSION_FULL})")

# Minimum macOS version 11.0 (the first release with full C++20 <semaphore> support)
# unless the caller set one — anira::Semaphore falls back to a portable backend below
# 11.0 (e.g. -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13). Never FORCE'd (cmake/tanh/apple.cmake).
include(cmake/tanh/apple.cmake)
tanh_apple_deployment_target(MACOS 11.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Detect Emscripten — must be after project() so CMAKE_CXX_COMPILER is set.
# EMSDK_VERSION is needed by backends.cmake (WASM target selection) below.
if(ANIRA_BUILD_WASM)
    include(cmake/detect-emscripten.cmake)
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

# anira's Buffer / RingBuffer / MemoryBlock are aliases over thl::core (see
# include/anira/utils/*.h). Only the core component is built: tanh::Core is
# tanh-lib's dedicated Apache-2.0 target — the AGPL components (state, dsp,
# modulation, audio-io) are never fetched into the build. tanh::Core links no
# platform library (the Linux journald sink stays opt-in and off here), so
# anira's system-dependency footprint is unchanged on every target incl. WASM.
#
# When anira installs, tanh's core component installs into the same prefix
# with its own CMake package; aniraConfig.cmake re-resolves it through
# find_dependency(tanh COMPONENTS Core), so a consumer's find_package(anira)
# links the exported tanh::Core.
#
# Pinned to a tagged tanh-lib release. (Temporarily a commit on tanh-lib PR #17 —
# THL_LOG_* macros, thl::core::Thread, owned rt log queues; bump to the v0.1.0 tag once cut.)
#
# Plain (non-cache) variables on purpose: tanh-lib's cmake_minimum_required
# puts CMP0077 in NEW mode, so these shadow its option() defaults while anira
# is the project populating tanh-lib, and never leak into the cache. A parent
# that fetches tanh-lib itself (with more components) before adding anira keeps
# its own configuration — FORCE'd cache entries would have clobbered it.
set(TANH_BUILD_CORE ON)
set(TANH_BUILD_STATE OFF)
set(TANH_BUILD_DSP OFF)
set(TANH_BUILD_MODULATION OFF)
set(TANH_BUILD_AUDIO_IO OFF)
set(TANH_WITH_TESTS OFF)
set(TANH_WITH_EXAMPLES OFF)
set(TANH_WITH_DOCS OFF)
set(TANH_WITH_JOURNALD OFF)
# tanh-lib's own [[clang::nonblocking]] annotations (thl::Logger::rt, the buffers)
# are checked in the same RTSan build as anira's ANIRA_REALTIME entry points.
set(TANH_WITH_RTSAN ${ANIRA_WITH_RTSAN})
set(TANH_WITH_INSTALL ${ANIRA_WITH_INSTALL})

FetchContent_Declare(
    tanh-lib
    GIT_REPOSITORY https://github.com/tanh-lab/tanh-lib.git
    # tanh-lib main after PRs #17 (THL_LOG_*, thl::core::Thread, owned rt log
    # queues), #19 (RingBuffer push_fill/discard/peek_past_block) and #20 (RCU
    # thread-local registry, Windows DLL TLS); bump to the next tag once cut.
    # tanh-lib v0.1.0: symbol policy on every component, shared tanh-tooling modules
    # (pins tanh-tooling v0.1.5 — the same tag anira pins; see modules-version.cmake).
    GIT_TAG v0.1.0
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
tanh_set_export_allowlist(${PROJECT_NAME} NAMESPACE anira)
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

if(ANIRA_WITH_RTSAN)
    include(cmake/tanh/sanitizers.cmake)
    tanh_add_sanitizer(${PROJECT_NAME} rtsan DEFINE ANIRA_WITH_RTSAN)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC concurrentqueue nlohmann_json::nlohmann_json tanh::Core)

# ------------------------------------------------------------------------------
# Backends link PRIVATE, through their anira::<engine> targets. No public header
# includes an engine header (BackendBase.h), and a consumer's own use of an engine
# is its own explicit link of anira::<engine>. For a static anira the engine
# archives still reach the consumer's link line as $<LINK_ONLY:anira::<engine>> —
# link dependencies (archive, --exclude-libs, system libraries) without compile
# usage requirements — so the installed package defines the same targets before
# importing anira (cmake/install.cmake). How each engine is linked — on demand,
# hidden, never whole-archive — is documented in cmake/aniraBackendHelpers.cmake.
# ------------------------------------------------------------------------------
if(ANIRA_WITH_EXECUTORCH)
    # anira::executorch wraps ONE merged libexecutorch.a per platform (runtime +
    # extensions + kernels + XNNPACK delegate, registrations pre-linked into one
    # archive member; Windows whole-archives the separate registrations lib), so
    # it links on demand like the other static engines — see
    # cmake/backends/executorch.cmake. ExecuTorch is static-only, so anira is
    # static here (cmake/validate-options.cmake) and hands the runtime on to its
    # consumer as $<LINK_ONLY:anira::executorch>: the single place it is linked,
    # which matters — the pre-linked registrations abort at static-init when
    # they register twice in one process.
    #
    # The ExecuTorch headers vendor a copy of c10 that used to be kept out of every
    # TU but ExecuTorchProcessor.cpp, so that it could not shadow LibTorch's real
    # c10 (which broke the LibTorch backend at link time on MSVC). LibTorch is
    # shared-only and ExecuTorch static-only, so the two can no longer share an
    # image and the processor is a regular anira source.
    target_link_libraries(${PROJECT_NAME} PRIVATE anira::executorch)
endif()

if(ANIRA_WITH_LIBTORCH)
    # anira::libtorch wraps torch, torch_library and what find_package(Torch) lists
    # by absolute path (libc10, libkineto, ...). PRIVATE: LibTorch is shared-only,
    # so anira is shared here and none of it reaches a consumer or the installed
    # export. (The CMake < 3.26 subproject special case that once split
    # torch/torch_library PUBLIC from the rest is the same thing now.)
    target_link_libraries(${PROJECT_NAME} PRIVATE anira::libtorch)
    # TORCH_CXX_FLAGS is LibTorch's libstdc++ ABI switch (-D_GLIBCXX_USE_CXX11_ABI=...;
    # empty in the current prebuilds). Compiling anira against LibTorch means
    # compiling anira with it, and it then decides the std::string ABI of anira's
    # own public API — so it is a PUBLIC requirement of anira, in the build tree and
    # in the installed export, not only of anira::libtorch.
    if(TORCH_CXX_FLAGS)
        separate_arguments(_anira_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(${PROJECT_NAME} PUBLIC ${_anira_torch_cxx_flags})
        unset(_anira_torch_cxx_flags)
    endif()
endif()

foreach(_anira_engine onnxruntime tflite litert)
    string(TOUPPER "${_anira_engine}" _anira_ENGINE)
    if(ANIRA_WITH_${_anira_ENGINE})
        target_link_libraries(${PROJECT_NAME} PRIVATE anira::${_anira_engine})
    endif()
endforeach()
unset(_anira_engine)
unset(_anira_ENGINE)

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
