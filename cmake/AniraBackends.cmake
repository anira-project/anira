# ==============================================================================
# AniraBackends.cmake — data-driven download + setup of pre-built inference engines
# ==============================================================================
#
# Single entry point for fetching the pre-built backend binaries anira links
# against. Replaces the per-engine SetupOnnxRuntime / SetupLibTorch /
# SetupTensorflowLite scripts.
#
# Binaries come from the anira-project/backends GitHub release whose tag is
# ANIRA_BACKENDS_VERSION. Every archive is named
#   <libname>-<version>-<OS>-<arch>[-<extra>]-<linkage>[-debug].zip
# and unpacks to a uniform tree (include/ + lib/, plus share/ + bin/ for
# libtorch). The SHA256 of every published archive lives in backends_lock.cmake;
# that file is also the source of truth for *which* archives a release ships.
#
# Per call:  anira_setup_backend(<id>)        id = libtorch|onnxruntime|tflite|litert
#
# Configurable cache variables (see CMakeLists.txt for the user-facing options):
#   ANIRA_BACKENDS_VERSION          release tag to download from (default v2.1.1)
#   ANIRA_BACKEND_LINKAGE           auto|shared|static (auto follows BUILD_SHARED_LIBS)
#   ANIRA_<ENGINE>_LINKAGE          per-engine linkage override
#   ANIRA_<ENGINE>_ROOTDIR          bring-your-own: use this prebuilt tree, skip download
#   ANIRA_<ENGINE>_URL              override the download URL (custom mirror/build)
#   ANIRA_<ENGINE>_SHA256           expected hash for ANIRA_<ENGINE>_URL
#   ANIRA_<ENGINE>_VERSION          override the engine version (else derived from the lock)
# where <ENGINE> is LIBTORCH | ONNXRUNTIME | TFLITE | LITERT.
# ==============================================================================

include_guard(GLOBAL)

# Repo-relative locations resolved once, robust to anira being a subproject.
get_filename_component(ANIRA_BACKENDS_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(ANIRA_BACKENDS_MODULES_DIR "${ANIRA_BACKENDS_CMAKE_DIR}/../modules" ABSOLUTE)

# Load the asset->sha256 lockfile (sets ANIRA_BACKEND_SHA256_<asset> + ANIRA_BACKENDS_LOCK_TAG).
include("${ANIRA_BACKENDS_CMAKE_DIR}/backends_lock.cmake")

if(NOT DEFINED ANIRA_BACKENDS_VERSION OR ANIRA_BACKENDS_VERSION STREQUAL "")
    set(ANIRA_BACKENDS_VERSION "${ANIRA_BACKENDS_LOCK_TAG}")
endif()
if(NOT ANIRA_BACKENDS_VERSION STREQUAL ANIRA_BACKENDS_LOCK_TAG)
    message(WARNING
        "ANIRA_BACKENDS_VERSION (${ANIRA_BACKENDS_VERSION}) differs from the lockfile tag "
        "(${ANIRA_BACKENDS_LOCK_TAG}). Hashes won't match — regenerate the lockfile with:\n"
        "  cmake -DTAG=${ANIRA_BACKENDS_VERSION} -P cmake/UpdateBackendsLock.cmake")
endif()

# The known OS tokens, used both to build and to parse asset names.
set(_ANIRA_OS_TOKENS "macOS|Linux|Windows|Android|iOS|WASM")

# ------------------------------------------------------------------------------
# _anira_backend_libname(<id> <out>) — archive/lib prefix for an engine id.
# ------------------------------------------------------------------------------
function(_anira_backend_libname id out)
    if(id STREQUAL "libtorch")
        set(${out} "libtorch" PARENT_SCOPE)
    elseif(id STREQUAL "onnxruntime")
        set(${out} "onnxruntime" PARENT_SCOPE)
    elseif(id STREQUAL "tflite")
        set(${out} "tensorflowlite_c" PARENT_SCOPE)
    elseif(id STREQUAL "litert")
        set(${out} "LiteRt" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unknown backend id '${id}' (expected libtorch|onnxruntime|tflite|litert)")
    endif()
endfunction()

# ------------------------------------------------------------------------------
# _anira_engine_version(<libname> <out>) — derive the engine version from the
# lockfile so the version always matches the pinned release (data-driven).
# ------------------------------------------------------------------------------
function(_anira_engine_version libname out)
    get_cmake_property(_vars VARIABLES)
    foreach(_v ${_vars})
        if(_v MATCHES "^ANIRA_BACKEND_SHA256_${libname}-(.+)-(${_ANIRA_OS_TOKENS})-")
            set(${out} "${CMAKE_MATCH_1}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out} "" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# _anira_list_engine_assets(<libname> <out>) — newline list of "<OS>-<arch>-<linkage>"
# suffixes this release ships for an engine, for helpful error messages.
# ------------------------------------------------------------------------------
function(_anira_list_engine_assets libname out)
    get_cmake_property(_vars VARIABLES)
    set(_suffixes "")
    foreach(_v ${_vars})
        if(_v MATCHES "^ANIRA_BACKEND_SHA256_${libname}-.+-((${_ANIRA_OS_TOKENS}).*)$")
            list(APPEND _suffixes "    ${CMAKE_MATCH_1}")
        endif()
    endforeach()
    list(SORT _suffixes)
    list(JOIN _suffixes "\n" _joined)
    set(${out} "${_joined}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# _anira_target_tokens(<out_os> <out_arch>) — map this build's platform/arch to
# the (OS, arch) tokens used in asset names. Sets arch to "" for WASM and the
# special "armv7l" sentinel for the legacy Bela path.
# ------------------------------------------------------------------------------
function(_anira_target_tokens out_os out_arch)
    if(EMSDK_VERSION)
        set(${out_os} "WASM" PARENT_SCOPE)
        set(${out_arch} "" PARENT_SCOPE)
        return()
    endif()

    if(APPLE)
        # Prefer the explicit OSX architecture selection; supports universal.
        if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
            set(_arch "universal")
        elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(_arch "arm64")
        elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(_arch "x86_64")
        else()
            set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        set(${out_os} "macOS" PARENT_SCOPE)
        set(${out_arch} "${_arch}" PARENT_SCOPE)
        return()
    endif()

    if(UNIX) # Linux
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(_arch "aarch64")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
            set(_arch "armv7l") # legacy sentinel — no backends asset
        else()
            set(_arch "x86_64")
        endif()
        set(${out_os} "Linux" PARENT_SCOPE)
        set(${out_arch} "${_arch}" PARENT_SCOPE)
        return()
    endif()

    if(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64")
            set(_arch "arm64")
        else()
            set(_arch "x86_64")
        endif()
        set(${out_os} "Windows" PARENT_SCOPE)
        set(${out_arch} "${_arch}" PARENT_SCOPE)
        return()
    endif()

    message(FATAL_ERROR "anira backends: unsupported platform")
endfunction()

# ------------------------------------------------------------------------------
# _anira_resolve_linkage(<id> <supported> <out>) — pick shared|static for an engine.
# ------------------------------------------------------------------------------
function(_anira_resolve_linkage id supported out)
    string(TOUPPER "${id}" _ID)

    if(DEFINED ANIRA_${_ID}_LINKAGE AND NOT ANIRA_${_ID}_LINKAGE STREQUAL "")
        set(_linkage "${ANIRA_${_ID}_LINKAGE}")
    elseif(ANIRA_BACKEND_LINKAGE STREQUAL "shared")
        set(_linkage "shared")
    elseif(ANIRA_BACKEND_LINKAGE STREQUAL "static")
        set(_linkage "static")
    else() # auto
        if(BUILD_SHARED_LIBS)
            set(_linkage "shared")
        else()
            set(_linkage "static")
        endif()
    endif()

    # WASM only ships static.
    if(EMSDK_VERSION)
        set(_linkage "static")
    endif()

    if(NOT _linkage IN_LIST supported)
        if(supported STREQUAL "shared")
            message(STATUS "anira: ${id} ships shared-only — forcing shared linkage (requested '${_linkage}').")
            set(_linkage "shared")
        else()
            message(FATAL_ERROR "anira: ${id} does not support '${_linkage}' linkage (supported: ${supported}).")
        endif()
    endif()

    set(${out} "${_linkage}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# _anira_download_extract(<url> <sha256> <dest> <archive> <flatten>) — fetch +
# unpack an archive into <dest> (skips if already present). <sha256> may be ""
# (legacy/override with no hash). <flatten> moves a single nested top dir up.
# ------------------------------------------------------------------------------
function(_anira_download_extract url sha256 dest archive flatten)
    if(EXISTS "${dest}/")
        message(STATUS "anira: backend found at ${dest}")
        return()
    endif()

    file(MAKE_DIRECTORY "${dest}")
    message(STATUS "anira: downloading ${url}")

    set(_zip "${CMAKE_BINARY_DIR}/import/${archive}")
    set(_hash_args "")
    if(NOT sha256 STREQUAL "")
        set(_hash_args EXPECTED_HASH "SHA256=${sha256}")
    endif()

    file(DOWNLOAD "${url}" "${_zip}" STATUS _st SHOW_PROGRESS ${_hash_args})
    list(GET _st 0 _code)
    list(GET _st 1 _msg)

    # file(DOWNLOAD) treats a 404's small HTML body as a success, so also size-check.
    set(_size 0)
    if(EXISTS "${_zip}")
        file(SIZE "${_zip}" _size)
    endif()
    if(NOT _code EQUAL 0 OR _size LESS 1024)
        file(REMOVE_RECURSE "${dest}")
        file(REMOVE "${_zip}")
        message(FATAL_ERROR "anira: failed to download backend archive.\n  URL: ${url}\n  Reason: ${_msg}")
    endif()

    file(ARCHIVE_EXTRACT INPUT "${_zip}" DESTINATION "${dest}")

    if(flatten)
        # Some legacy archives nest everything under a single top-level dir.
        string(REGEX REPLACE "\\.(zip|tgz|tar\\.gz)$" "" _stem "${archive}")
        if(EXISTS "${dest}/${_stem}/")
            file(COPY "${dest}/${_stem}/" DESTINATION "${dest}/")
            file(REMOVE_RECURSE "${dest}/${_stem}")
        endif()
    endif()
endfunction()

# ------------------------------------------------------------------------------
# _anira_setup_legacy_armv7l(<id>) — preserved pre-backends download paths for
# 32-bit ARM (Bela). backends does not publish armv7l; these upstream mirrors do.
# Sets ANIRA_<ID>_ROOTDIR + appends header/lib dirs in the calling scope.
# ------------------------------------------------------------------------------
macro(_anira_setup_legacy_armv7l id)
    set(_la_id "${id}") # macro arg -> real variable for if() comparisons
    string(TOUPPER "${_la_id}" _LID)
    if(_la_id STREQUAL "libtorch")
        set(_lver "2.5.1")
        set(_dir "${ANIRA_BACKENDS_MODULES_DIR}/libtorch-${_lver}-Linux-armv7l")
        _anira_download_extract(
            "https://github.com/pelinski/bela-torch/releases/download/v${_lver}/pytorch-v${_lver}.tar.gz"
            "" "${_dir}" "pytorch-v${_lver}.tar.gz" TRUE)
        list(APPEND CMAKE_PREFIX_PATH "${_dir}")
        find_package(Torch REQUIRED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        set(LIBTORCH_ROOTDIR "${_dir}")
        set(ANIRA_LIBTORCH_ROOTDIR "${_dir}")
        set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${_dir}")
        list(APPEND BACKEND_BUILD_HEADER_DIRS "${_dir}/include")
        list(APPEND BACKEND_BUILD_HEADER_DIRS "${_dir}/include/torch/csrc/api/include")
    elseif(_la_id STREQUAL "onnxruntime")
        set(_lver "1.19.2")
        set(_dir "${ANIRA_BACKENDS_MODULES_DIR}/onnxruntime-${_lver}-Linux-armv7l")
        _anira_download_extract(
            "https://github.com/faressc/onnxruntime-cpp-lib/releases/download/v${_lver}/onnxruntime-${_lver}-Linux-armv7l.tar.gz"
            "" "${_dir}" "onnxruntime-${_lver}-Linux-armv7l.tar.gz" TRUE)
        set(ANIRA_ONNXRUNTIME_ROOTDIR "${_dir}")
        set(ANIRA_ONNXRUNTIME_SHARED_LIB_PATH "${_dir}")
        set(ANIRA_ONNXRUNTIME_LIB_BASENAME "onnxruntime")
        set(ANIRA_ONNXRUNTIME_IS_STATIC FALSE)
        list(APPEND BACKEND_BUILD_HEADER_DIRS "${_dir}/include/onnxruntime")
        list(APPEND BACKEND_BUILD_LIBRARY_DIRS "${_dir}/lib")
    elseif(_la_id STREQUAL "tflite")
        set(_lver "2.17.0")
        set(_dir "${ANIRA_BACKENDS_MODULES_DIR}/tensorflowlite-${_lver}-Linux-armv7l")
        _anira_download_extract(
            "https://github.com/faressc/tflite-c-lib/releases/download/v${_lver}/tensorflowlite_c-${_lver}-Linux-armv7l.zip"
            "" "${_dir}" "tensorflowlite_c-${_lver}-Linux-armv7l.zip" TRUE)
        set(ANIRA_TFLITE_ROOTDIR "${_dir}")
        set(ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH "${_dir}")
        set(ANIRA_TFLITE_LIB_BASENAME "tensorflowlite_c")
        set(ANIRA_TFLITE_IS_STATIC FALSE)
        list(APPEND BACKEND_BUILD_HEADER_DIRS "${_dir}/include")
        list(APPEND BACKEND_BUILD_LIBRARY_DIRS "${_dir}/lib")
    else()
        message(FATAL_ERROR "anira: ${id} has no armv7l (Bela) build.")
    endif()
endmacro()

# ==============================================================================
# anira_setup_backend(<id>) — main entry point. A macro so find_package(Torch),
# CMAKE_CXX_FLAGS, and the BACKEND_BUILD_*_DIRS accumulators all act on the
# including (directory) scope.
# ==============================================================================
macro(anira_setup_backend id)
    # Bind the macro argument to a real variable: macro args are string
    # substitutions, not variables, so `if(id STREQUAL ...)` would test the
    # literal "id". Use _ab_id in all comparisons below.
    set(_ab_id "${id}")
    string(TOUPPER "${_ab_id}" _ab_ID)
    _anira_backend_libname("${_ab_id}" _ab_libname)

    # Supported linkages per engine (libtorch is shared-only).
    if(_ab_id STREQUAL "libtorch")
        set(_ab_supported "shared")
    else()
        set(_ab_supported "shared;static")
    endif()
    _anira_resolve_linkage("${_ab_id}" "${_ab_supported}" _ab_linkage)

    # ---- Bring-your-own: ANIRA_<ID>_ROOTDIR (or legacy LIBTORCH_ROOTDIR / TENSORFLOWLITE_ROOTDIR).
    set(_ab_byo "")
    if(DEFINED ANIRA_${_ab_ID}_ROOTDIR AND NOT ANIRA_${_ab_ID}_ROOTDIR STREQUAL "")
        set(_ab_byo "${ANIRA_${_ab_ID}_ROOTDIR}")
    elseif(_ab_id STREQUAL "libtorch" AND DEFINED LIBTORCH_ROOTDIR AND NOT LIBTORCH_ROOTDIR STREQUAL "" AND IS_DIRECTORY "${LIBTORCH_ROOTDIR}")
        set(_ab_byo "${LIBTORCH_ROOTDIR}")
    elseif(_ab_id STREQUAL "tflite" AND DEFINED TENSORFLOWLITE_ROOTDIR AND NOT TENSORFLOWLITE_ROOTDIR STREQUAL "" AND IS_DIRECTORY "${TENSORFLOWLITE_ROOTDIR}")
        set(_ab_byo "${TENSORFLOWLITE_ROOTDIR}")
    endif()

    _anira_target_tokens(_ab_os _ab_arch)

    if(NOT _ab_byo STREQUAL "")
        # Use the provided tree as-is.
        message(STATUS "anira: using bring-your-own ${_ab_id} at ${_ab_byo}")
        set(_ab_rootdir "${_ab_byo}")
    elseif(_ab_arch STREQUAL "armv7l")
        # Legacy Bela path (no backends asset, no lockfile).
        _anira_setup_legacy_armv7l("${_ab_id}")
        set(_ab_rootdir "${ANIRA_${_ab_ID}_ROOTDIR}")
    else()
        # ---- Normal path: resolve version + asset name, then download from backends.
        if(DEFINED ANIRA_${_ab_ID}_VERSION AND NOT ANIRA_${_ab_ID}_VERSION STREQUAL "")
            set(_ab_version "${ANIRA_${_ab_ID}_VERSION}")
        else()
            _anira_engine_version("${_ab_libname}" _ab_version)
        endif()
        if(_ab_version STREQUAL "")
            message(FATAL_ERROR "anira: no ${_ab_libname} assets in the lockfile (tag ${ANIRA_BACKENDS_LOCK_TAG}). Regenerate cmake/backends_lock.cmake.")
        endif()
        set(ANIRA_${_ab_ID}_VERSION "${_ab_version}") # expose the resolved version (e.g. for BuildWasm license bundling)

        # Windows static additionally ships a Debug variant.
        set(_ab_linktoken "${_ab_linkage}")
        if(_ab_linkage STREQUAL "static" AND WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(_ab_linktoken "static-debug")
        endif()

        # Asset name: <libname>-<version>-<OS>[-<arch>]-<linktoken>  (WASM has no arch token).
        if(_ab_os STREQUAL "WASM")
            set(_ab_asset "${_ab_libname}-${_ab_version}-WASM-${_ab_linktoken}")
        else()
            set(_ab_asset "${_ab_libname}-${_ab_version}-${_ab_os}-${_ab_arch}-${_ab_linktoken}")
        endif()

        # Resolve hash + URL (overrides win; else lockfile + backends release URL).
        if(DEFINED ANIRA_${_ab_ID}_SHA256 AND NOT ANIRA_${_ab_ID}_SHA256 STREQUAL "")
            set(_ab_sha "${ANIRA_${_ab_ID}_SHA256}")
        elseif(DEFINED ANIRA_BACKEND_SHA256_${_ab_asset})
            set(_ab_sha "${ANIRA_BACKEND_SHA256_${_ab_asset}}")
        else()
            set(_ab_sha "")
        endif()

        if(DEFINED ANIRA_${_ab_ID}_URL AND NOT ANIRA_${_ab_ID}_URL STREQUAL "")
            set(_ab_url "${ANIRA_${_ab_ID}_URL}")
        else()
            if(NOT DEFINED ANIRA_BACKEND_SHA256_${_ab_asset})
                _anira_list_engine_assets("${_ab_libname}" _ab_avail)
                message(FATAL_ERROR
                    "anira: ${id} archive '${_ab_asset}.zip' is not part of the ${ANIRA_BACKENDS_LOCK_TAG} "
                    "backends release.\nAvailable ${_ab_libname} archives for this release:\n${_ab_avail}\n"
                    "If this platform/linkage should exist, regenerate the lockfile; otherwise set "
                    "ANIRA_${_ab_ID}_ROOTDIR to your own prebuilt tree or ANIRA_${_ab_ID}_URL to a custom build.")
            endif()
            set(_ab_url "https://github.com/anira-project/backends/releases/download/${ANIRA_BACKENDS_VERSION}/${_ab_asset}.zip")
        endif()

        set(_ab_rootdir "${ANIRA_BACKENDS_MODULES_DIR}/${_ab_asset}")
        _anira_download_extract("${_ab_url}" "${_ab_sha}" "${_ab_rootdir}" "${_ab_asset}.zip" FALSE)
    endif()

    # ---- Wire the engine into the build (libtorch is special: it has CMake config files).
    if(_ab_id STREQUAL "libtorch")
        if(_ab_byo STREQUAL "" AND NOT _ab_arch STREQUAL "armv7l")
            list(APPEND CMAKE_PREFIX_PATH "${_ab_rootdir}")
            find_package(Torch REQUIRED)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
            list(APPEND BACKEND_BUILD_HEADER_DIRS "${_ab_rootdir}/include")
            list(APPEND BACKEND_BUILD_HEADER_DIRS "${_ab_rootdir}/include/torch/csrc/api/include")
        endif()
        # -w silences the (many) warnings from the prebuilt torch headers.
        if(TARGET torch)
            target_link_options(torch INTERFACE "-w")
        endif()
        if(TARGET torch_library)
            target_link_options(torch_library INTERFACE "-w")
        endif()
        set(LIBTORCH_ROOTDIR "${_ab_rootdir}")
        set(ANIRA_LIBTORCH_ROOTDIR "${_ab_rootdir}")
        set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${_ab_rootdir}")
        set(ANIRA_LIBTORCH_LINKAGE "shared")
    elseif(_ab_arch STREQUAL "armv7l")
        # legacy macro already populated header/lib dirs + ANIRA_<ID>_* vars
        set(ANIRA_${_ab_ID}_LINKAGE "${_ab_linkage}")
    else()
        # onnxruntime / tflite / litert: uniform include/ + lib/.
        if(_ab_linkage STREQUAL "static")
            set(ANIRA_${_ab_ID}_IS_STATIC TRUE)
        else()
            set(ANIRA_${_ab_ID}_IS_STATIC FALSE)
        endif()
        set(ANIRA_${_ab_ID}_ROOTDIR "${_ab_rootdir}")
        set(ANIRA_${_ab_ID}_LINKAGE "${_ab_linkage}")
        set(ANIRA_${_ab_ID}_LIB_BASENAME "${_ab_libname}")
        # legacy shared-lib-path var consumed by msvc-support / BuildWasm / examples
        set(ANIRA_${_ab_ID}_SHARED_LIB_PATH "${_ab_rootdir}")
        list(APPEND BACKEND_BUILD_HEADER_DIRS "${_ab_rootdir}/include")
        list(APPEND BACKEND_BUILD_LIBRARY_DIRS "${_ab_rootdir}/lib")

        # Full path to the static archive (for whole-archive linking in CMakeLists).
        if(_ab_linkage STREQUAL "static")
            if(WIN32)
                set(ANIRA_${_ab_ID}_STATIC_LIB "${_ab_rootdir}/lib/${_ab_libname}.lib")
            else()
                set(ANIRA_${_ab_ID}_STATIC_LIB "${_ab_rootdir}/lib/lib${_ab_libname}.a")
            endif()
        endif()
    endif()

    message(STATUS "anira: ${id} ready (${_ab_linkage}) at ${_ab_rootdir}")
endmacro()

# ------------------------------------------------------------------------------
# anira_target_link_static_backend(<target> <archive-path>) — link a static
# backend archive on-demand, plus the system libraries it depends on. PUBLIC so
# everything propagates to whatever links anira.
#
# NB: we deliberately do NOT whole-archive these. The onnxruntime/tflite static
# archives vendor overlapping copies of protobuf/absl/onnx and contain multiple
# members defining the same symbols (resolved on demand during a normal link);
# force-loading them produces thousands of duplicate-symbol errors. anira drives
# the engines through their C API, which the linker resolves on demand.
# ------------------------------------------------------------------------------
function(anira_target_link_static_backend target libpath)
    if(EMSDK_VERSION)
        target_link_libraries(${target} PUBLIC "${libpath}")
    elseif(MSVC)
        target_link_libraries(${target} PUBLIC "${libpath}")
    elseif(APPLE)
        # Static onnxruntime/tflite/litert pull in absl/CoreFoundation time-zone +
        # Apple logging code, so the system frameworks must be linked.
        target_link_libraries(${target} PUBLIC
            "${libpath}" "-framework Foundation" "-framework CoreFoundation")
    else() # Linux / other ELF
        find_package(Threads REQUIRED)
        target_link_libraries(${target} PUBLIC
            "${libpath}" Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
endfunction()
