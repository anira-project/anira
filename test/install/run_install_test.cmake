# Installed-package smoke test, run as a CTest via `cmake -P`.
#
# Installs the just-built anira into a throwaway prefix, then configures, builds
# and runs a separate find_package(anira) consumer against it. This is the only
# coverage for the install/export path (cmake --install + downstream
# find_package), which the in-tree build never exercises.
#
# Required -D arguments (set by the add_test() wiring in test/CMakeLists.txt):
#   ANIRA_BUILD_DIR  the anira build tree to install from
#   CONSUMER_SRC     path to the consumer project (test/install/consumer)
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

message(STATUS "[install-test] configuring consumer (find_package(anira))")
_step("configure" ${CMAKE_COMMAND} -S "${CONSUMER_SRC}" -B "${_consumer_build}"
    -G "${GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${_prefix}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")

message(STATUS "[install-test] building consumer")
_step("build" ${CMAKE_COMMAND} --build "${_consumer_build}" --config "${BUILD_TYPE}")

# Locate the consumer executable across single- and multi-config layouts.
set(_exe "")
foreach(_cand
        "${_consumer_build}/consumer"
        "${_consumer_build}/consumer.exe"
        "${_consumer_build}/${BUILD_TYPE}/consumer.exe"
        "${_consumer_build}/${BUILD_TYPE}/consumer")
    if(EXISTS "${_cand}")
        set(_exe "${_cand}")
        break()
    endif()
endforeach()
if(_exe STREQUAL "")
    message(FATAL_ERROR "install-test: consumer executable not found under ${_consumer_build}")
endif()

# Make the installed anira + backend libraries resolvable at runtime.
set(_libdir "${_prefix}/lib")
if(APPLE)
    set(ENV{DYLD_LIBRARY_PATH} "${_libdir}:$ENV{DYLD_LIBRARY_PATH}")
elseif(UNIX)
    set(ENV{LD_LIBRARY_PATH} "${_libdir}:$ENV{LD_LIBRARY_PATH}")
elseif(WIN32)
    set(ENV{PATH} "${_libdir};$ENV{PATH}")
endif()

message(STATUS "[install-test] running ${_exe}")
_step("run" "${_exe}")
message(STATUS "[install-test] OK")
