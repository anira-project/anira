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
option(ANIRA_WITH_EXECUTORCH  "Build with the ExecuTorch backend (runs .pte models; static-only, requires CMake >= 3.24 on desktop)" ON)

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
# Download and install the selected inference engines
# ==============================================================================

set(BACKEND_SOURCES)
set(BACKEND_BUILD_HEADER_DIRS)
set(BACKEND_BUILD_LIBRARY_DIRS)

include(cmake/AniraBackends.cmake)

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
    if(NOT TARGET executorch)
        # Mobile merged-lib path: LibTorch has no mobile build, so ExecuTorch's
        # vendored c10 headers cannot clash there and the processor compiles as a
        # regular anira source. On desktop it is compiled in an isolated object
        # target instead (see the ExecuTorch link block below).
        list(APPEND BACKEND_SOURCES src/backends/ExecuTorchProcessor.cpp)
    endif()
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
        src/utils/Buffer.cpp
        src/utils/RingBuffer.cpp
        src/utils/JsonConfigLoader.cpp

        # Interface
        src/InferenceHandler.cpp
        src/PrePostProcessor.cpp
        src/InferenceConfig.cpp

        # System
        src/system/HighPriorityThread.cpp
)

# add the include directories for the backends to the build interface, public because the anira headers include the backend headers
foreach(HEADER_DIR ${BACKEND_BUILD_HEADER_DIRS})
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
        $<BUILD_INTERFACE:${HEADER_DIR}>
    )
endforeach()

# include the public headers of the anira library in the top-level project include directory
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# include the link directories for all the backends to the build interface
foreach(LIBRARY_DIR ${BACKEND_BUILD_LIBRARY_DIRS})
    target_link_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${LIBRARY_DIR}>
    )
endforeach()

# Shared between the anira target and the isolated ExecuTorch object target below —
# the two must agree on every USE_* define (they shape the anira class layouts).
set(ANIRA_COMPILE_DEFINITIONS
    # Backend-specific definitions
    $<$<BOOL:${ANIRA_WITH_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${ANIRA_WITH_ONNXRUNTIME}>:USE_ONNXRUNTIME>
    $<$<BOOL:EMSDK_VERSION>:USE_ANIRA_WEB>
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
    include(cmake/real-time-sanitizers.cmake)
    anira_rtsan_configure(${PROJECT_NAME})
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC concurrentqueue nlohmann_json::nlohmann_json)

if(ANIRA_WITH_EXECUTORCH)
    if(TARGET executorch)
        # Desktop: the ExecuTorch CMake package was found by anira_setup_backend.
        # Link the runtime plus the extension (Module/TensorPtr/data loader),
        # kernel (optimized + quantized CPU ops) and XNNPACK-delegate targets —
        # the ops/backend targets carry the per-platform force-load options that
        # their static-initializer registration requires. The CoreML/MPS/MLX
        # delegates are deliberately not linked: anira pins its backends to
        # portable CPU execution, and their exported targets reference absolute
        # SDK paths from the machine that built the archives.
        #
        # Two ordering/scope constraints:
        #  * This block must come BEFORE the libtorch link block: libtorch_cpu
        #    exports its own (different) XNNPACK, and if the torch dylibs precede
        #    ExecuTorch's static archives on the link line, part of the delegate's
        #    xnn_* references bind to libtorch's copy and delegate init fails at
        #    runtime.
        #  * The force-load options propagate to whatever links these targets, and
        #    ExecuTorch aborts at static-init when its kernels register twice in
        #    one process. A shared anira must therefore link them PRIVATE (the
        #    registrations live inside libanira and consumers reach the runtime
        #    through it); only a static anira propagates them to the consumer,
        #    which is then the single place the runtime is linked.
        if(BUILD_SHARED_LIBS)
            set(ANIRA_EXECUTORCH_LINK_SCOPE PRIVATE)
        else()
            set(ANIRA_EXECUTORCH_LINK_SCOPE PUBLIC)
        endif()
        target_link_libraries(${PROJECT_NAME} ${ANIRA_EXECUTORCH_LINK_SCOPE}
            executorch executorch_extensions executorch_kernels xnnpack_backend)

        # ExecuTorchProcessor.cpp is the only TU that may see the ExecuTorch headers:
        # they vendor a copy of c10 that must never shadow LibTorch's real c10 in the
        # other backend TUs (it breaks the LibTorch backend at link time on MSVC).
        # anira_setup_backend stripped the compile usage requirements off the imported
        # ExecuTorch targets, so compile it in its own object target with explicit
        # include dirs and merge the objects into anira.
        add_library(anira-executorch-objs OBJECT src/backends/ExecuTorchProcessor.cpp)
        set_target_properties(anira-executorch-objs PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON)
        target_compile_definitions(anira-executorch-objs PRIVATE
            ${ANIRA_COMPILE_DEFINITIONS}
            ${ANIRA_EXECUTORCH_COMPILE_DEFINITIONS})
        target_include_directories(anira-executorch-objs PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_include_directories(anira-executorch-objs SYSTEM PRIVATE
            "${ANIRA_EXECUTORCH_ROOTDIR}/include"
            "${ANIRA_EXECUTORCH_ROOTDIR}/include/executorch/runtime/core/portable_type/c10")
        target_link_libraries(anira-executorch-objs PRIVATE
            concurrentqueue nlohmann_json::nlohmann_json)
        if(MSVC)
            # Match anira's export decoration (see cmake/msvc-support.cmake): the
            # objects become part of anira, so ANIRA_API must resolve identically.
            if(BUILD_SHARED_LIBS)
                target_compile_definitions(anira-executorch-objs PRIVATE ANIRA_EXPORTS)
            else()
                target_compile_definitions(anira-executorch-objs PRIVATE ANIRA_STATIC_DEFINE)
            endif()
        endif()
        target_sources(${PROJECT_NAME} PRIVATE $<TARGET_OBJECTS:anira-executorch-objs>)
        # XNNPACK's runtime-dispatch config tables reference its microkernel symbols
        # from data sections, which on-demand archive resolution does not satisfy
        # (Apple's linker fails with "does not have address" fixup errors on archive
        # members it never materialized). The microkernel archive is a leaf with no
        # dependencies or colliding symbols, so link it whole-archive.
        if(TARGET xnnpack-microkernels-prod)
            target_link_libraries(${PROJECT_NAME} ${ANIRA_EXECUTORCH_LINK_SCOPE}
                "$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack-microkernels-prod>")
        endif()
    else()
        # Mobile: a single merged libexecutorch.a (per ABI on Android, per
        # xcframework slice on iOS), linked like the other static backends.
        anira_target_link_static_backend(${PROJECT_NAME} "${ANIRA_EXECUTORCH_STATIC_LIB}")
    endif()
endif()

if(ANIRA_WITH_LIBTORCH)
    # The find_package(Torch) adds the libraries libc10.so and libkineto.a as full paths to ${TORCH_LIBRARIES}. This is no problem when we add anira as a subdirectory to another project, but when we install the library, the torch libraries will be link targets of the anira library with full paths and hence not found on other systems. Therefore, we link those libs privately and only add the torch target publicly.
    # Also until cmake 3.26, there is a bug where the torch_cpu library is not found when linking publicly https://gitlab.kitware.com/cmake/cmake/-/issues/24163 and anira is added as a subdirectory to another project, see
    # But this is necessary for when we install the library since otherwise symbols are not found
    # Another problem are that on armv7l with benchmarking enabled, some symbols are not found when linking the torch_cpu library privately
    if (CMAKE_VERSION VERSION_LESS "3.26.0" AND NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR))
        target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
        set(TORCH_LIBRARIES_ALL_PRIVATE TRUE)
        if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
            target_link_libraries(${PROJECT_NAME} PUBLIC torch_cpu)
        endif()
    else()
        foreach(TORCH_LIB ${TORCH_LIBRARIES})
            if(TORCH_LIB STREQUAL "torch" OR TORCH_LIB STREQUAL "torch_library")
                target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIB})
            else()
                target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIB})
            endif()
        endforeach()
    endif()
endif()

# Enforce the ExecuTorch-before-LibTorch ordering constraint documented above.
# Both bundle their own XNNPACK; if the torch dylibs precede ExecuTorch's static
# archives on the link line, part of the XNNPACK delegate's xnn_* references bind
# to libtorch's (different) copy and delegate init fails at runtime with a bare
# Internal error. LINK_LIBRARIES preserves the target_link_libraries call order,
# so a future reshuffle of the two blocks above is caught here at configure time
# instead of as a runtime failure.
if(ANIRA_WITH_EXECUTORCH AND ANIRA_WITH_LIBTORCH AND TARGET executorch)
    get_target_property(_anira_link_order ${PROJECT_NAME} LINK_LIBRARIES)
    list(FIND _anira_link_order "executorch" _anira_executorch_index)
    list(FIND _anira_link_order "torch" _anira_torch_index)
    if(_anira_executorch_index EQUAL -1
       OR (_anira_torch_index GREATER -1 AND _anira_executorch_index GREATER _anira_torch_index))
        message(FATAL_ERROR
            "anira: internal link-order violation — the ExecuTorch targets must be linked "
            "before the LibTorch libraries (both bundle XNNPACK; see the comment at the "
            "ExecuTorch link block). Restore the ExecuTorch block above the LibTorch block.")
    endif()
    unset(_anira_link_order)
    unset(_anira_executorch_index)
    unset(_anira_torch_index)
endif()

# Backends link PUBLIC: anira's public headers include the backend headers, and
# (onnxruntime) "_OrtGetApiBase" is otherwise dropped. Static archives are linked
# on-demand (NOT whole-archive — the onnxruntime/tflite archives carry duplicate
# protobuf/absl symbols that force-loading would clash; anira only needs the C API,
# which the linker resolves on demand) — see anira_target_link_static_backend in
# cmake/AniraBackends.cmake.
if(ANIRA_WITH_ONNXRUNTIME)
    if(ANIRA_ONNXRUNTIME_IS_STATIC)
        anira_target_link_static_backend(${PROJECT_NAME} "${ANIRA_ONNXRUNTIME_STATIC_LIB}")
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)
    endif()
endif()

if(ANIRA_WITH_TFLITE)
    if(ANIRA_TFLITE_IS_STATIC)
        anira_target_link_static_backend(${PROJECT_NAME} "${ANIRA_TFLITE_STATIC_LIB}")
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC tensorflowlite_c)
    endif()
endif()

if(ANIRA_WITH_LITERT)
    if(ANIRA_LITERT_IS_STATIC)
        # The desktop static archives are pre-isolated to the LiteRt* C API at
        # packaging (backends >= v2.3.0), so they coexist with ExecuTorch's
        # force-loaded XNNPACK — see the note in AniraValidate.cmake.
        anira_target_link_static_backend(${PROJECT_NAME} "${ANIRA_LITERT_STATIC_LIB}")
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC LiteRt)
    endif()
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
