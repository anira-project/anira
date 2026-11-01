# ==============================================================================
# cmake/abi-namespaces.cmake — the namespace gate anira_abi_namespaces.
#
# libanira carries two third-party codebases inside it, tanh-lib's core and concurrentqueue, and
# keeps them apart from a host's own copies by name: tanh-lib's in its inline ABI namespace
# thl::anira (anira_tanh_core, CMakeLists.txt), the queue's in anira_moodycamel (a compile
# definition on anira_concurrentqueue). This file registers a CTest that reads the symbols of the
# built library and fails on any thl symbol outside thl::anira and on any moodycamel symbol, the
# templates and inline functions included, so neither name can reach a consumer's link:
#
#   Itanium (ELF, Mach-O)   every "3thl" source name is followed by "5anira"; no "10moodycamel"
#   MSVC (PE)               every "@thl@@" scope is "@anira@thl@@"; no "@moodycamel@@"
#
# It reads every symbol, defined and undefined, the local ones included: a static library's
# archive members (the prelinked object on ELF), a shared library's full symbol table on ELF and
# Mach-O (nm, not the export table). A PE DLL has no symbol table of its internals, so there the
# objects the DLL is linked from are read instead. Nothing to read on Wasm: skipped.
#
#   include(cmake/abi-namespaces.cmake)
#   anira_add_namespace_check(NAME <test> LIBRARY <target> OBJECTS <target>...)
#
#   cmake -DFORMAT=ELF|MACHO|PE -DNM=<nm>|-DDUMPBIN=<dumpbin> -DFILES_FROM=<list file>
#         -P abi-namespaces.cmake
# runs the check (what the registered test does); the list file names one file per line (written
# by file(GENERATE) per configuration, since a target's objects are a generator-expression list).
# ==============================================================================
if(NOT CMAKE_SCRIPT_MODE_FILE)
    include_guard(GLOBAL)
    function(anira_add_namespace_check)
        cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME;LIBRARY" "OBJECTS")
        if(arg_UNPARSED_ARGUMENTS OR NOT arg_NAME OR NOT arg_LIBRARY)
            message(FATAL_ERROR "anira_add_namespace_check: NAME and LIBRARY are required")
        endif()
        if(TANH_BINARY_FORMAT STREQUAL "ELF")
            set(_format ELF)
        elseif(TANH_BINARY_FORMAT STREQUAL "Mach-O")
            set(_format MACHO)
        elseif(TANH_BINARY_FORMAT STREQUAL "PE")
            set(_format PE)
        else()
            message(STATUS "anira_add_namespace_check(${arg_NAME}): skipped on ${TANH_BINARY_FORMAT}")
            return()
        endif()
        get_target_property(_type ${arg_LIBRARY} TYPE)
        set(_files "$<TARGET_FILE:${arg_LIBRARY}>")
        if(_format STREQUAL "PE" AND _type STREQUAL "SHARED_LIBRARY")
            # The DLL's internals are not in any symbol table: read what it is linked from.
            set(_files "")
            foreach(_target IN LISTS arg_OBJECTS)
                get_target_property(_object_type ${_target} TYPE)
                if(_object_type STREQUAL "OBJECT_LIBRARY")
                    list(APPEND _files "$<TARGET_OBJECTS:${_target}>")
                else()
                    list(APPEND _files "$<TARGET_FILE:${_target}>")
                endif()
            endforeach()
        endif()
        if(_format STREQUAL "PE")
            # dumpbin lives next to link.exe; it is not on PATH outside a developer shell.
            get_filename_component(_linker_dir "${CMAKE_LINKER}" DIRECTORY)
            find_program(ANIRA_ABI_DUMPBIN dumpbin HINTS "${_linker_dir}")
            set(_tool "${ANIRA_ABI_DUMPBIN}")
            set(_tool_arg "-DDUMPBIN=${ANIRA_ABI_DUMPBIN}")
        else()
            if(CMAKE_NM)
                set(ANIRA_ABI_NM "${CMAKE_NM}")
            else()
                find_program(ANIRA_ABI_NM NAMES nm llvm-nm)
            endif()
            set(_tool "${ANIRA_ABI_NM}")
            set(_tool_arg "-DNM=${ANIRA_ABI_NM}")
        endif()
        if(NOT _tool)
            message(WARNING "anira_add_namespace_check(${arg_NAME}): no nm/dumpbin found — test not registered")
            return()
        endif()
        set(_list "${CMAKE_CURRENT_BINARY_DIR}/${arg_NAME}-$<CONFIG>.files")
        file(GENERATE OUTPUT "${_list}" CONTENT "$<JOIN:${_files},\n>\n")
        add_test(NAME ${arg_NAME}
            COMMAND ${CMAKE_COMMAND} -DFORMAT=${_format} ${_tool_arg} "-DFILES_FROM=${_list}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    endfunction()
    return()
endif()

# ------------------------------------------------------------------------------
# Script mode: the check.
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)
if(NOT FORMAT MATCHES "^(ELF|MACHO|PE)$")
    message(FATAL_ERROR "abi-namespaces: FORMAT must be ELF, MACHO or PE (got '${FORMAT}')")
endif()
if(NOT FILES_FROM OR NOT EXISTS "${FILES_FROM}")
    message(FATAL_ERROR "abi-namespaces: FILES_FROM '${FILES_FROM}' does not exist")
endif()
file(STRINGS "${FILES_FROM}" FILES)
if(NOT FILES)
    message(FATAL_ERROR "abi-namespaces: ${FILES_FROM} names no file")
endif()

# The names of one nm/dumpbin call over <files>..., appended to _names in the caller. Each name is
# cut out of the whole output by one regex pass, with no CMake loop over its lines: a PE build's
# symbol tables run to hundreds of thousands of lines, and a loop that appends line by line took
# most of the test's ten minutes on Windows.
function(_anira_read_symbols)
    if(FORMAT STREQUAL "PE")
        execute_process(COMMAND "${DUMPBIN}" /nologo /symbols ${ARGN}
            OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    else()
        execute_process(COMMAND "${NM}" ${ARGN}
            OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    endif()
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "abi-namespaces: reading the symbols of ${ARGN} failed: ${_err}")
    endif()
    string(REPLACE ";" "," _raw "${_raw}")
    if(FORMAT STREQUAL "PE")
        # "... | ?name@scope@@... (undecorated)": the decorated name after the line's first bar.
        string(REGEX MATCHALL "\n[^|\n]*\\| +[^ \n]+" _found "\n${_raw}")
        list(TRANSFORM _found REPLACE "^\n[^|]*\\| +" "")
    else()
        # "<address> <type> <name>", the whole line: every line gets a newline of its own on
        # both sides, so that one match can end at its line's end without taking the next's start.
        string(REPLACE "\n" "\n\n" _raw "\n${_raw}\n")
        string(REGEX MATCHALL "\n[0-9A-Fa-f]* +[A-Za-z?-] +[^ \n]+\n" _found "${_raw}")
        list(TRANSFORM _found REPLACE "^\n[0-9A-Fa-f]* +[A-Za-z?-] +([^ \n]+)\n$" "\\1")
    endif()
    list(REMOVE_DUPLICATES _found)
    list(APPEND _names ${_found})
    set(_names "${_names}" PARENT_SCOPE)
endfunction()

# The files in batches, one tool call each: a shared PE build reads every object of the DLL, and a
# process per object adds up. A batch stays far below Windows' 32767-character command line.
set(_names "")
set(_batch "")
set(_batch_length 0)
foreach(_file IN LISTS FILES)
    if(NOT EXISTS "${_file}")
        message(FATAL_ERROR "abi-namespaces: '${_file}' does not exist")
    endif()
    string(LENGTH "${_file}" _length)
    math(EXPR _batch_length "${_batch_length} + ${_length} + 3")
    if(_batch AND _batch_length GREATER 16000)
        _anira_read_symbols(${_batch})
        set(_batch "")
        math(EXPR _batch_length "${_length} + 3")
    endif()
    list(APPEND _batch "${_file}")
endforeach()
_anira_read_symbols(${_batch})
list(REMOVE_DUPLICATES _names)
list(LENGTH _names _count)
# Only a name that spells thl or moodycamel can count or fail below.
list(FILTER _names INCLUDE REGEX "thl|moodycamel")

set(_thl 0)
set(_queue 0)
set(_foreign "")
foreach(_name IN LISTS _names)
    if(FORMAT STREQUAL "PE")
        if(_name MATCHES "@anira@thl@@")
            math(EXPR _thl "${_thl} + 1")
        endif()
        if(_name MATCHES "@anira_moodycamel@@")
            math(EXPR _queue "${_queue} + 1")
        endif()
        string(REPLACE "@anira@thl@@" "" _rest "${_name}")
        string(REPLACE "@anira_moodycamel@@" "" _rest "${_rest}")
        if(_rest MATCHES "@thl@@" OR _rest MATCHES "@moodycamel@@")
            list(APPEND _foreign "${_name}")
        endif()
    else()
        string(REPLACE "3thl5anira" "" _rest "${_name}")
        string(REPLACE "16anira_moodycamel" "" _rest "${_rest}")
        if(_name MATCHES "3thl5anira")
            math(EXPR _thl "${_thl} + 1")
        endif()
        if(_name MATCHES "16anira_moodycamel")
            math(EXPR _queue "${_queue} + 1")
        endif()
        # A source name of length 3 spelling "thl" (not the tail of a longer length, "13thl…"),
        # or the queue's own namespace name.
        if(_rest MATCHES "(^|[^0-9])3thl" OR _rest MATCHES "(^|[^0-9])10moodycamel")
            list(APPEND _foreign "${_name}")
        endif()
    endif()
endforeach()

list(LENGTH _foreign _bad)
if(_bad GREATER 0)
    list(SUBLIST _foreign 0 20 _shown)
    string(REPLACE ";" "\n  " _shown "${_shown}")
    message(FATAL_ERROR
        "abi-namespaces: ${_bad} of ${_count} symbols name thl outside thl::anira or name "
        "moodycamel (anira's copies must be thl::anira and anira_moodycamel):\n  ${_shown}")
endif()
if(_thl EQUAL 0)
    message(FATAL_ERROR
        "abi-namespaces: none of ${_count} symbols lies in thl::anira; the check read nothing "
        "of tanh-lib (a stripped library, or the wrong file)")
endif()
message(STATUS "[abi-namespaces] ${_count} symbols: ${_thl} in thl::anira, ${_queue} in "
               "anira_moodycamel, none in another thl namespace or in moodycamel — OK")
