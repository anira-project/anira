# Installed-package smoke test, run as a CTest via `cmake -P`.
#
# Installs the just-built anira into a throwaway prefix, then configures, builds
# and runs the separate find_package(anira) consumers of test/install against it:
# `consumer` (anira alone) and, when ONNX Runtime is part of the package,
# `consumer_engine` (calls the engine itself through anira::onnxruntime) plus its
# plugin-shaped twin, whose export table must carry no engine symbol. The
# consumer project itself asserts that the engine header is unreachable without
# anira::onnxruntime. This is the only coverage for the install/export path
# (cmake --install + downstream find_package), which the in-tree build never
# exercises.
#
# Required -D arguments (set by the add_test() wiring in test/CMakeLists.txt):
#   ANIRA_BUILD_DIR  the anira build tree to install from
#   CONSUMER_SRC     path to the consumer project (test/install)
#   WORK_DIR         scratch dir for the prefix + consumer build
#   GENERATOR        CMake generator to reuse for the consumer
#   BUILD_TYPE       build configuration (Debug/Release/...)

foreach(_var ANIRA_BUILD_DIR CONSUMER_SRC WORK_DIR GENERATOR BUILD_TYPE)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "run_install_test: required argument ${_var} not set")
    endif()
endforeach()

set(_prefix "${WORK_DIR}/prefix")
set(_consumer_build "${WORK_DIR}/consumer-build")
file(REMOVE_RECURSE "${_prefix}" "${_consumer_build}")

# Run a command and fail the test (non-zero exit) if it does not succeed.
function(_step _desc)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "install-test step failed (${_desc}): exit code ${_rc}")
    endif()
endfunction()

message(STATUS "[install-test] installing anira -> ${_prefix}")
_step("install" ${CMAKE_COMMAND} --install "${ANIRA_BUILD_DIR}" --prefix "${_prefix}"
    --config "${BUILD_TYPE}")

message(STATUS "[install-test] configuring consumers (find_package(anira))")
_step("configure" ${CMAKE_COMMAND} -S "${CONSUMER_SRC}" -B "${_consumer_build}"
    -G "${GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${_prefix}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")

message(STATUS "[install-test] building consumers")
_step("build" ${CMAKE_COMMAND} --build "${_consumer_build}" --config "${BUILD_TYPE}")

# Locate a built file across single- and multi-config layouts ("" if absent).
function(_find_built name out)
    set(_found "")
    foreach(_cand
            "${_consumer_build}/${name}"
            "${_consumer_build}/${name}.exe"
            "${_consumer_build}/${BUILD_TYPE}/${name}.exe"
            "${_consumer_build}/${BUILD_TYPE}/${name}")
        if(EXISTS "${_cand}")
            set(_found "${_cand}")
            break()
        endif()
    endforeach()
    set(${out} "${_found}" PARENT_SCOPE)
endfunction()

# Make the installed anira + backend libraries resolvable at runtime (lib64 is
# GNUInstallDirs' libdir on Fedora-style hosts).
set(_libdir "${_prefix}/lib")
if(APPLE)
    set(ENV{DYLD_LIBRARY_PATH} "${_libdir}:$ENV{DYLD_LIBRARY_PATH}")
elseif(UNIX)
    set(ENV{LD_LIBRARY_PATH} "${_libdir}:${_prefix}/lib64:$ENV{LD_LIBRARY_PATH}")
elseif(WIN32)
    set(ENV{PATH} "${_libdir};$ENV{PATH}")
endif()

_find_built(consumer _exe)
if(_exe STREQUAL "")
    message(FATAL_ERROR "install-test: consumer executable not found under ${_consumer_build}")
endif()
message(STATUS "[install-test] running ${_exe}")
_step("run" "${_exe}")

# The engine-calling consumer exists only when the package carries ONNX Runtime.
_find_built(consumer_engine _engine_exe)
if(NOT _engine_exe STREQUAL "")
    message(STATUS "[install-test] running ${_engine_exe}")
    _step("run consumer_engine" "${_engine_exe}")

    # Its plugin-shaped twin must export nothing engine-shaped: the engine archive is
    # linked hidden and the TU is compiled hidden. Same forbid list as
    # cmake/check-exports.cmake for ONNX Runtime; nm-based, so Linux and macOS only.
    if(NOT WIN32)
        set(_module "${_consumer_build}/consumer_engine_module${CMAKE_SHARED_MODULE_SUFFIX}")
        if(NOT EXISTS "${_module}")
            set(_module "${_consumer_build}/consumer_engine_module.so")
        endif()
        find_program(_nm NAMES nm llvm-nm)
        if(_nm AND EXISTS "${_module}")
            if(APPLE)
                set(_nm_args -gU)
            else()
                set(_nm_args -D --defined-only)
            endif()
            execute_process(COMMAND "${_nm}" ${_nm_args} "${_module}"
                OUTPUT_VARIABLE _exports RESULT_VARIABLE _rc)
            if(NOT _rc EQUAL 0)
                message(FATAL_ERROR "install-test: nm failed on ${_module}")
            endif()
            string(REGEX MATCHALL "[^\n]*[^\n]*(_ZN3Ort|_ZN11onnxruntime|[ _]Ort[A-Z]|OrtGetApiBase)[^\n]*" _leaks "${_exports}")
            if(_leaks)
                list(JOIN _leaks "\n" _leaks)
                message(FATAL_ERROR "install-test: consumer_engine_module exports engine symbols:\n${_leaks}")
            endif()
            message(STATUS "[install-test] consumer_engine_module exports no engine symbol — OK")
        endif()
    endif()
endif()
message(STATUS "[install-test] OK")
