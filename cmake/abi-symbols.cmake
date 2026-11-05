# ==============================================================================
# cmake/abi-symbols.cmake — the presence gate anira_symbol_baseline.
#
# The registry (abi/anira.yml) promises the entry points of abi/symbols-<major>.txt and
# drafts those of abi/symbols-draft.txt; the shared libanira must export every one of
# them, and nothing else. This file registers a CTest that reads the library's real export
# table (nm on ELF and Mach-O, dumpbin on PE) and reports each promised or draft name that is
# missing and each exported name that is in neither list (a C++ symbol, an internal C
# function, a third-party one), apart from the symbols a linker defines on its own and, on PE,
# the exports the engine archives force (anira_exports tolerates the same ones).
#
#   include(cmake/abi-symbols.cmake)
#   anira_add_symbol_baseline(NAME <test> LIBRARY <target> SYMBOLS <file>...)
#
# Skipped with a STATUS message on a static library (no export table; the link probe
# anira_abi_link is the presence check there), on a binary format without one (Wasm),
# and where no nm/dumpbin can be found.
#
#   cmake -DFORMAT=ELF|MACHO|PE -DNM=<nm>|-DDUMPBIN=<dumpbin> -DLIBRARY=<path>
#         -DSYMBOLS=<file>;<file> -P abi-symbols.cmake
# runs the check (what the registered test does).
# ==============================================================================
if(NOT CMAKE_SCRIPT_MODE_FILE)
    include_guard(GLOBAL)
    function(anira_add_symbol_baseline)
        cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME;LIBRARY" "SYMBOLS")
        if(arg_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "anira_add_symbol_baseline: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
        endif()
        if(NOT arg_NAME OR NOT arg_LIBRARY OR NOT arg_SYMBOLS)
            message(FATAL_ERROR "anira_add_symbol_baseline: NAME, LIBRARY and SYMBOLS are required")
        endif()
        get_target_property(_type ${arg_LIBRARY} TYPE)
        if(NOT _type STREQUAL "SHARED_LIBRARY")
            message(STATUS "anira_add_symbol_baseline(${arg_NAME}): skipped, ${arg_LIBRARY} is ${_type} (anira_abi_link is the presence check)")
            return()
        endif()
        if(TANH_BINARY_FORMAT STREQUAL "ELF")
            set(_format ELF)
        elseif(TANH_BINARY_FORMAT STREQUAL "Mach-O")
            set(_format MACHO)
        elseif(TANH_BINARY_FORMAT STREQUAL "PE")
            set(_format PE)
        else()
            message(STATUS "anira_add_symbol_baseline(${arg_NAME}): skipped, no dynamic export table on ${TANH_BINARY_FORMAT}")
            return()
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
            message(WARNING "anira_add_symbol_baseline(${arg_NAME}): no nm/dumpbin found — test not registered")
            return()
        endif()
        add_test(NAME ${arg_NAME}
            COMMAND ${CMAKE_COMMAND}
                -DFORMAT=${_format}
                ${_tool_arg}
                "-DLIBRARY=$<TARGET_FILE:${arg_LIBRARY}>"
                "-DSYMBOLS=${arg_SYMBOLS}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    endfunction()
    return()
endif()

# ------------------------------------------------------------------------------
# Script mode: the check.
# ------------------------------------------------------------------------------
# A -P script starts with no policy settings: say what this file relies on (IN_LIST).
cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0057 NEW)
if(NOT FORMAT MATCHES "^(ELF|MACHO|PE)$")
    message(FATAL_ERROR "abi-symbols: FORMAT must be ELF, MACHO or PE (got '${FORMAT}')")
endif()
if(NOT LIBRARY OR NOT EXISTS "${LIBRARY}")
    message(FATAL_ERROR "abi-symbols: LIBRARY '${LIBRARY}' does not exist")
endif()
if(NOT SYMBOLS)
    message(FATAL_ERROR "abi-symbols: SYMBOLS is required")
endif()

# The export table, one name per entry; a leading underscore (Mach-O, 32-bit PE) is
# stripped so the names compare with the registry's spelling.
set(_exports "")
if(FORMAT STREQUAL "PE")
    execute_process(COMMAND "${DUMPBIN}" /nologo /exports "${LIBRARY}"
        OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "abi-symbols: dumpbin failed on ${LIBRARY}: ${_err}")
    endif()
    string(REPLACE "\n" ";" _lines "${_raw}")
    foreach(_line IN LISTS _lines)
        # "   ordinal hint RVA      name" — forwarded exports ("name = ...") keep
        # their first token.
        if(_line MATCHES "^ +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +([^ \r]+)")
            list(APPEND _exports "${CMAKE_MATCH_1}")
        endif()
    endforeach()
else()
    if(FORMAT STREQUAL "ELF")
        set(_args -D --defined-only)
    else()
        set(_args -g -U)
    endif()
    execute_process(COMMAND "${NM}" ${_args} "${LIBRARY}"
        OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "abi-symbols: nm failed on ${LIBRARY}: ${_err}")
    endif()
    string(REPLACE "\n" ";" _lines "${_raw}")
    foreach(_line IN LISTS _lines)
        # "address type name"; only the defined text symbols are entry points.
        if(_line MATCHES "^[0-9A-Fa-f]* +([A-Za-z?]) +([^ ]+)")
            list(APPEND _exports "${CMAKE_MATCH_2}")
        endif()
    endforeach()
endif()
set(_names "")
foreach(_export IN LISTS _exports)
    string(REGEX REPLACE "^_" "" _plain "${_export}")
    # An ELF version script suffixes nothing; a versioned symbol would read name@VERSION. On PE
    # an @ is part of a C++ name's MSVC decoration (??0Future@ivalue@c10@@QEAA@XZ), which the
    # tolerance below matches whole; Mach-O has no symbol versions.
    if(FORMAT STREQUAL "ELF")
        string(REGEX REPLACE "@.*$" "" _plain "${_plain}")
    endif()
    list(APPEND _names "${_plain}")
endforeach()
list(REMOVE_DUPLICATES _names)

set(_missing "")
set(_expected 0)
set(_wanted_all "")
foreach(_file IN LISTS SYMBOLS)
    if(NOT EXISTS "${_file}")
        message(FATAL_ERROR "abi-symbols: symbol file '${_file}' does not exist")
    endif()
    file(STRINGS "${_file}" _wanted)
    foreach(_name IN LISTS _wanted)
        string(STRIP "${_name}" _name)
        if(_name STREQUAL "" OR _name MATCHES "^#")
            continue()
        endif()
        math(EXPR _expected "${_expected} + 1")
        list(APPEND _wanted_all "${_name}")
        if(NOT "${_name}" IN_LIST _names)
            list(APPEND _missing "${_name}")
        endif()
    endforeach()
endforeach()

if(_missing)
    list(LENGTH _missing _count)
    string(REPLACE ";" "\n  " _report "${_missing}")
    message(FATAL_ERROR "abi-symbols: ${_count} of ${_expected} promised or draft entry points are not exported by ${LIBRARY}:\n  ${_report}")
endif()

# Nothing but the lists: every other exported name fails, except what the linker defines
# itself (the ELF section bounds, the Mach-O header and dyld's marker) and, on PE, the exports
# that the engine archives force with __declspec(dllexport) (LibTorch's c10/torch templates,
# the TFLite and LiteRT C APIs), which anira_exports tolerates as well.
set(_extra "")
foreach(_name IN LISTS _names)
    if(_name IN_LIST _wanted_all)
        continue()
    endif()
    if(_name MATCHES "^(_init|_fini|_edata|_end|__bss_start|_DYNAMIC|_GLOBAL_OFFSET_TABLE_|_mh_dylib_header|radr://[0-9]+)$")
        continue()
    endif()
    if(FORMAT STREQUAL "PE" AND _name MATCHES "(@(torch|c10|caffe2)@@|^(TfLite|LiteRt))")
        continue()
    endif()
    list(APPEND _extra "${_name}")
endforeach()
if(_extra)
    list(LENGTH _extra _count)
    list(SUBLIST _extra 0 40 _shown)
    string(REPLACE ";" "\n  " _report "${_shown}")
    message(FATAL_ERROR "abi-symbols: ${LIBRARY} exports ${_count} name(s) outside the promised and draft lists (the first 40):\n  ${_report}")
endif()
message(STATUS "abi-symbols: ${LIBRARY} exports exactly the ${_expected} promised and draft entry points")
