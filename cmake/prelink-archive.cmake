# ==============================================================================
# cmake/prelink-archive.cmake — the one object of a static ELF libanira.
#
# A static libanira's objects (anira's and the absorbed tanh_core's) are linked into one
# relocatable object, and every symbol of it that is defined, STB_GLOBAL and STV_HIDDEN — anira's
# C++ internals and tanh-lib's, compiled under the hidden symbol policy — is made local, except
# the anira_* C entries. A consumer's link then binds nothing of its own to anira's copy of
# tanh-lib, and nothing of anira's to the consumer's: hidden alone does not do that inside one
# static link, where the consumer's objects and the archive's are linked together, a reference of
# the consumer's (a host that uses tanh-lib itself) binds to the archive's hidden definition like
# to any object's, and anira's own references to thl are default-visibility undefined ones that
# the first definition on the link line satisfies. In the prelinked object they are resolved
# inside it. Weak definitions (the COMDAT groups of inline and template instantiations) stay as
# they are: the final link keeps one group per signature, and a group whose symbols were made
# local would be discarded under its references. tanh-lib's header-only code is therefore merged
# with a host's instantiations of the same signature, and is separate only as long as both use
# one tanh-lib version.
#
#   cmake -DCXX=<compiler> [-DCXX_TARGET=<triple>] -DREADELF=<readelf> -DOBJCOPY=<objcopy>
#         -DOUTPUT=<object> -DINPUTS=<object>;<object>... -P cmake/prelink-archive.cmake
#
# ELF only (cmake/tanh: TANH_BINARY_FORMAT) and single-configuration generators; CXX_TARGET is a
# clang driver's --target. The caller archives OUTPUT as libanira.a.
# ==============================================================================
cmake_minimum_required(VERSION 3.18)
foreach(_var CXX READELF OBJCOPY OUTPUT INPUTS)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "prelink-archive: ${_var} is required")
    endif()
endforeach()

set(_relocatable "${OUTPUT}.relocatable.o")
set(_target_flag "")
if(CXX_TARGET)
    set(_target_flag "--target=${CXX_TARGET}")
endif()
# The compiler driver runs the toolchain's linker (ld, lld) in relocatable mode. No build-id: the
# consumer's own link gives the final image one, and a second note from the object would be a
# duplicate. None of the build's compile flags is passed (see CMakeLists.txt).
execute_process(
    COMMAND "${CXX}" ${_target_flag} -r -nostdlib -Wl,--build-id=none -o "${_relocatable}" ${INPUTS}
    RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "prelink-archive: the relocatable link failed:\n${_err}")
endif()

execute_process(COMMAND "${READELF}" -sW "${_relocatable}"
    OUTPUT_VARIABLE _symbols RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "prelink-archive: readelf failed:\n${_err}")
endif()
# "Num: Value Size Type Bind Vis Ndx Name": defined (Ndx a section index), GLOBAL, HIDDEN.
string(REGEX MATCHALL "[^\n]* GLOBAL +HIDDEN +[0-9]+ +[^\n ]+" _hidden "${_symbols}")
set(_localize "")
foreach(_line IN LISTS _hidden)
    string(REGEX REPLACE "^.* ([^ ]+)$" "\\1" _name "${_line}")
    if(NOT _name MATCHES "^anira_")
        string(APPEND _localize "${_name}\n")
    endif()
endforeach()
file(WRITE "${OUTPUT}.localize.txt" "${_localize}")

execute_process(COMMAND "${OBJCOPY}" "--localize-symbols=${OUTPUT}.localize.txt" "${_relocatable}" "${OUTPUT}"
    RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "prelink-archive: objcopy failed:\n${_err}")
endif()
file(REMOVE "${_relocatable}")
