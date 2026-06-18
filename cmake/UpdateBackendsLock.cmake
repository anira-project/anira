# ==============================================================================
# UpdateBackendsLock.cmake — regenerate cmake/backends_lock.cmake
# ==============================================================================
#
# Maintainer-only tool (NOT included during a normal build). It queries the
# anira-project/backends GitHub release for a given tag, reads the SHA256 digest
# GitHub stores for every asset, and writes them to cmake/backends_lock.cmake.
#
# The generated lockfile is what the build reads at configure time
# (cmake/AniraBackends.cmake) to (a) verify every downloaded archive and (b)
# know which archives a given release actually ships. Re-run this whenever the
# backends repo publishes or re-publishes a release (e.g. once the build runner
# adds web / new platforms):
#
#   cmake -DTAG=v2.1.1 -P cmake/UpdateBackendsLock.cmake
#
# Honors the GITHUB_TOKEN environment variable to avoid the 60 req/h
# unauthenticated rate limit. Requires CMake >= 3.19 for string(JSON) — this is
# a maintainer tool, so it does not constrain the build's own minimum version.
# ==============================================================================

cmake_minimum_required(VERSION 3.19)

if(NOT DEFINED TAG OR TAG STREQUAL "")
    set(TAG "v2.1.1")
    message(STATUS "TAG not provided, defaulting to ${TAG}")
endif()

if(NOT DEFINED REPO OR REPO STREQUAL "")
    set(REPO "anira-project/backends")
endif()

set(_lockfile "${CMAKE_CURRENT_LIST_DIR}/backends_lock.cmake")
set(_api_url "https://api.github.com/repos/${REPO}/releases/tags/${TAG}")
set(_json_path "${CMAKE_CURRENT_LIST_DIR}/.backends_release_${TAG}.json")

message(STATUS "Fetching release metadata: ${_api_url}")

set(_headers HTTPHEADER "Accept: application/vnd.github+json")
if(DEFINED ENV{GITHUB_TOKEN} AND NOT "$ENV{GITHUB_TOKEN}" STREQUAL "")
    list(APPEND _headers HTTPHEADER "Authorization: Bearer $ENV{GITHUB_TOKEN}")
    message(STATUS "Using GITHUB_TOKEN for authentication")
endif()

file(DOWNLOAD "${_api_url}" "${_json_path}" STATUS _dl_status ${_headers})
list(GET _dl_status 0 _dl_code)
list(GET _dl_status 1 _dl_msg)
if(NOT _dl_code EQUAL 0)
    file(REMOVE "${_json_path}")
    message(FATAL_ERROR "Failed to fetch release metadata from ${_api_url}\n  Reason: ${_dl_msg}")
endif()

file(READ "${_json_path}" _json)
file(REMOVE "${_json_path}")

# Guard against an error payload (e.g. {"message":"Not Found"}) parsing as valid JSON.
string(JSON _assets ERROR_VARIABLE _assets_err GET "${_json}" assets)
if(_assets_err OR _assets STREQUAL "")
    message(FATAL_ERROR "Release '${TAG}' has no assets array (is the tag correct?). Response began: ${_json}")
endif()

string(JSON _count LENGTH "${_json}" assets)
if(_count EQUAL 0)
    message(FATAL_ERROR "Release '${TAG}' lists zero assets.")
endif()

message(STATUS "Release '${TAG}' has ${_count} assets")

set(_lines "")
set(_n_locked 0)
math(EXPR _last "${_count} - 1")
foreach(_i RANGE ${_last})
    string(JSON _name GET "${_json}" assets ${_i} name)

    # Only lock the engine archives this build consumes (.zip). Skip checksum
    # sidecars, source tarballs GitHub auto-attaches, etc.
    if(NOT _name MATCHES "\\.zip$")
        continue()
    endif()
    string(REGEX REPLACE "\\.zip$" "" _asset "${_name}")

    string(JSON _digest ERROR_VARIABLE _digest_err GET "${_json}" assets ${_i} digest)
    if(_digest_err OR _digest STREQUAL "" OR _digest MATCHES "NOTFOUND")
        message(WARNING "Asset '${_name}' has no digest — skipping (re-run after GitHub finishes processing it).")
        continue()
    endif()
    string(REGEX REPLACE "^sha256:" "" _sha "${_digest}")

    list(APPEND _lines "set(ANIRA_BACKEND_SHA256_${_asset} \"${_sha}\")")
    math(EXPR _n_locked "${_n_locked} + 1")
endforeach()

list(SORT _lines)
list(JOIN _lines "\n" _body)

file(WRITE "${_lockfile}"
"# ==============================================================================
# backends_lock.cmake — AUTO-GENERATED, DO NOT EDIT BY HAND
# ==============================================================================
# Maps each anira-project/backends release asset to its SHA256 digest. Read at
# configure time by cmake/AniraBackends.cmake to verify downloads and to know
# which archives this release ships. Regenerate with:
#   cmake -DTAG=<tag> -P cmake/UpdateBackendsLock.cmake
# ==============================================================================

set(ANIRA_BACKENDS_LOCK_TAG \"${TAG}\")

${_body}
")

message(STATUS "Wrote ${_n_locked} asset hashes to ${_lockfile}")
