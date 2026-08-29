cmake_minimum_required(VERSION 3.15)

# ==============================================================================
# Build options — every user-facing option lives here. Cross-option validation
# (mutual exclusivity, libtorch auto-disable, WASM guards) is in
# cmake/AniraValidate.cmake, included after project() / Emscripten detection.
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
# libraries export the same TfLite* symbols). Enforced in cmake/AniraValidate.cmake.
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

# Backend linkage follows BUILD_SHARED_LIBS (shared anira -> shared backends, static
# anira -> static backends). Decouple a single engine, bring your own, or use a custom
# source via the per-engine variables (<ENGINE> = LIBTORCH | ONNXRUNTIME | TFLITE | LITERT | EXECUTORCH):
#   -DANIRA_<ENGINE>_LINKAGE=shared|static
#   -DANIRA_<ENGINE>_ROOTDIR=/path/to/prebuilt   (a tree with include/ + lib/)
#   -DANIRA_<ENGINE>_URL=...  [-DANIRA_<ENGINE>_SHA256=...]

# ==============================================================================
# Get project version from git
# ==============================================================================

execute_process(COMMAND git describe --dirty
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE PROJECT_VERSION_FULL
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE GIT_DESCRIBE_RESULT)

execute_process(COMMAND git describe --tags --abbrev=0
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE PROJECT_VERSION_SHORT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(GIT_DESCRIBE_RESULT EQUAL 0 AND PROJECT_VERSION_SHORT)
  # Retrieve the v from the short version string
  string(SUBSTRING ${PROJECT_VERSION_SHORT} 1 -1 PROJECT_VERSION_SHORT)
  string(SUBSTRING ${PROJECT_VERSION_FULL} 1 -1 PROJECT_VERSION_FULL)
else()
  set(PROJECT_VERSION_SHORT "0.0.0")
  set(PROJECT_VERSION_FULL "0.0.0-unknown")
  message(WARNING "git describe failed - no tags found. Using fallback version ${PROJECT_VERSION_SHORT}")
endif()

# ==============================================================================
# Setup the project
# ==============================================================================

set (PROJECT_NAME anira)

project (${PROJECT_NAME} VERSION ${PROJECT_VERSION_SHORT})

if(APPLE)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
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

if(APPLE)
    set(ANIRA_OPERATING_SYSTEM "macOS")
elseif(UNIX)
    set(ANIRA_OPERATING_SYSTEM "Linux")
elseif(WIN32)
    set(ANIRA_OPERATING_SYSTEM "Windows")
endif()

# Print the processor architecture selected for build. Defaults to CMAKE_HOST_SYSTEM_PROCESSOR when no crosscompile tolchain is defined. CMAKE_HOST_SYSTEM_PROCESSOR is only defined after the project() call. For OSX this value can be overwritten by the CMAKE_OSX_ARCHITECTURES.

message(STATUS "Building ${PROJECT_NAME} for ${CMAKE_SYSTEM_NAME} on ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Project version: ${PROJECT_VERSION_SHORT} (${PROJECT_VERSION_FULL})")

# Sets the minimum macOS version. Defaults to 11.0 (the first release with full
# C++20 <semaphore> support), but respects a user-provided value so the library
# can target older systems (e.g. -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13). On targets
# below 11.0, anira::Semaphore automatically falls back to a portable backend.
if (APPLE)
	if (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
		set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum version of the target platform" FORCE)
	endif()
	message(STATUS "The minimum macOS version is set to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Detect Emscripten — must be after project() so CMAKE_CXX_COMPILER is set.
# EMSDK_VERSION is needed by AniraBackends.cmake (WASM target selection) below.
if(ANIRA_BUILD_WASM)
    include(cmake/DetectEmscripten.cmake)
endif()

# Validate the option combination (mutual exclusivity, libtorch auto-disable, WASM
# guards). After Emscripten detection so EMSDK_VERSION is known; before the backends
# are set up so the libtorch auto-disable takes effect.
include(cmake/AniraValidate.cmake)

# ==============================================================================
# Backend download machinery + per-engine setup (cmake/backends/*.cmake)
# ==============================================================================

include(cmake/AniraBackends.cmake)

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
# Pinned to a tagged tanh-lib release.
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
set(TANH_WITH_INSTALL ${ANIRA_WITH_INSTALL})

FetchContent_Declare(
    tanh-lib
    GIT_REPOSITORY https://github.com/tanh-lab/tanh-lib.git
    GIT_TAG v0.0.3
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

# add the library as a shared or static library depending on the option BUILD_SHARED_LIBS
add_library(${PROJECT_NAME})

# enable position independent code because otherwise the static library cannot be linked into a shared library
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# Compile with hidden symbol visibility on every platform; the public API is
# re-exported through ANIRA_API (AniraWinExports.h) — the same allowlist model
# Windows enforces via dllexport. This keeps the backend runtimes private:
# without it, weak C++ symbols from the backend headers (e.g. ONNX Runtime's
# Ort::Global<void>::api_) are exported from any plugin that embeds anira and
# get weak-coalesced by the dynamic linker against a host application that
# ships its own copy of the runtime (Ableton Live 12 bundles an ONNX Runtime
# dylib), making Ort::GetApi() resolve against a mismatched runtime and
# crashing the host on the first Ort:: call.
set_target_properties(${PROJECT_NAME} PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
# GCC binds exported vague-linkage data (inline variables, template statics, static
# locals of inline functions — libstdc++ emits one from every make_shared TU) as
# STB_GNU_UNIQUE, and glibc never unloads a shared object that defines a unique
# symbol that was looked up: it is implicitly marked NODELETE. A plugin host's
# dlclose() would then silently keep libanira — or a plugin embedding it — mapped
# forever, and hide any thread-at-unload bug behind it. Plain weak binding is all
# anira needs (the unload test asserts that the library really gets unmapped).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-gnu-unique)
endif()
# add an alias so that the project can be used with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE

        # Backend
        src/backends/BackendBase.cpp

        # Scheduler
        src/scheduler/InferenceManager.cpp
        src/scheduler/InferenceThread.cpp
        src/scheduler/Context.cpp
        src/scheduler/SessionElement.cpp

        # Utils
        src/utils/JsonConfigLoader.cpp

        # Interface
        src/InferenceHandler.cpp
        src/PrePostProcessor.cpp
        src/InferenceConfig.cpp

        # System
        src/system/HighPriorityThread.cpp
)

# include the public headers of the anira library in the top-level project include directory
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Backend-specific USE_<ENGINE> definitions are added by anira_setup_<engine>() below.
target_compile_definitions(${PROJECT_NAME}
    PUBLIC
    $<$<BOOL:EMSDK_VERSION>:USE_ANIRA_WEB>
    $<$<BOOL:${ANIRA_WITH_LOGGING}>:ENABLE_LOGGING>
    -DANIRA_VERSION="${PROJECT_VERSION_FULL}"
)

if(ANIRA_WITH_RTSAN)
    include(cmake/real-time-sanitizers.cmake)
    anira_rtsan_configure(${PROJECT_NAME})
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC concurrentqueue nlohmann_json::nlohmann_json tanh::Core)

# ==============================================================================
# Inference backends — download, sources, defines and linking live in
# cmake/AniraBackends.cmake (fetch) + cmake/backends/<engine>.cmake (wiring).
# ==============================================================================

if(ANIRA_WITH_EXECUTORCH)
    anira_setup_executorch(${PROJECT_NAME})   # before libtorch: both bundle XNNPACK
endif()
if(ANIRA_WITH_LIBTORCH)
    anira_setup_libtorch(${PROJECT_NAME})
endif()
if(ANIRA_WITH_ONNXRUNTIME)
    anira_setup_onnxruntime(${PROJECT_NAME})
endif()
if(ANIRA_WITH_TFLITE)
    anira_setup_tflite(${PROJECT_NAME})
endif()
if(ANIRA_WITH_LITERT)
    anira_setup_litert(${PROJECT_NAME})
endif()

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
    include(cmake/package.cmake)
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
    include(cmake/BuildWasm.cmake)
endif()
