# ==============================================================================
# tanh-tooling · cmake/check-exports.cmake — CTest check of real dynamic export tables.
#
# One file, two entry points:
#
#   include(...)  defines
#     tanh_add_export_check(NAME <test> [LIBRARY <target>] [MODULE <target>]
#                           NAMESPACES <ns>... [FORBID_NAMESPACES <ns>...]
#                           [FORBID_PREFIXES <prefix>...] [ALLOW_REGEX <regex>...]
#                           [TOLERATE_REGEX_PE <regex>...])
#     which registers a test that runs this same file in script mode against the
#     binaries CMake actually produced. LIBRARY is the library under test (a shared
#     library must export exactly NAMESPACES — vtables, typeinfo, thunks and guards of
#     those classes included — plus ALLOW_REGEX, and nothing else; a STATIC library
#     has no export table and instead means "MODULE must not export NAMESPACES").
#     MODULE is a plugin-shaped shared object embedding the library: it must export
#     nothing matching FORBID_NAMESPACES / FORBID_PREFIXES, and — when there is no
#     shared LIBRARY to import from — nothing of NAMESPACES either (that would be the
#     statically linked library leaking). On ELF neither may define an STB_GNU_UNIQUE
#     symbol (glibc marks such objects NODELETE; the module must be dlclose-able).
#     TOLERATE_REGEX_PE lists exports accepted on PE only (headers that force
#     __declspec(dllexport) on template specialisations, e.g. LibTorch's c10::).
#     Skipped with a STATUS message where there is no dynamic export table (Wasm) or
#     no nm/dumpbin can be found.
#
#   cmake -P check-exports.cmake  with
#     -DFORMAT=ELF|MACHO|PE -DNM=<nm> | -DDUMPBIN=<dumpbin> -DLIBRARY=<path|""> -DMODULE=<path|"">
#     -DNAMESPACES=a;b -DFORBID_NAMESPACES=... -DFORBID_PREFIXES=... -DALLOW_REGEX=...
#     -DTOLERATE_REGEX=...
#   runs the check (what the registered test does).
#
# Inputs: target properties, CMAKE_NM, CMAKE_LINKER and the platform module (included
# explicitly). Requires CMake >= 3.18; include after project().
# ==============================================================================

if(NOT CMAKE_SCRIPT_MODE_FILE)
    # ----------------------------------------------------------------------------
    # Configure mode: registration.
    # ----------------------------------------------------------------------------
    include_guard(GLOBAL)
    include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/platform.cmake")

    # cmake_parse_arguments: an empty value after a keyword is a value, not an omission
    # (CMP0174 NEW); function bodies record the policy state of their definition.
    cmake_policy(PUSH)
    if(POLICY CMP0174)
        cmake_policy(SET CMP0174 NEW)
    endif()

    function(tanh_add_export_check)
        cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME;LIBRARY;MODULE"
            "NAMESPACES;FORBID_NAMESPACES;FORBID_PREFIXES;ALLOW_REGEX;TOLERATE_REGEX_PE")
        if(arg_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "tanh_add_export_check: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
        endif()
        if(NOT arg_NAME OR NOT arg_NAMESPACES OR (NOT arg_LIBRARY AND NOT arg_MODULE))
            message(FATAL_ERROR "tanh_add_export_check: NAME, NAMESPACES and at least one of LIBRARY/MODULE are required")
        endif()

        if(TANH_BINARY_FORMAT STREQUAL "ELF")
            set(_format ELF)
        elseif(TANH_BINARY_FORMAT STREQUAL "Mach-O")
            set(_format MACHO)
        elseif(TANH_BINARY_FORMAT STREQUAL "PE")
            set(_format PE)
        else()
            message(STATUS "tanh_add_export_check(${arg_NAME}): skipped, no dynamic export table on ${TANH_BINARY_FORMAT}")
            return()
        endif()

        if(_format STREQUAL "PE")
            # dumpbin lives next to link.exe; it is not on PATH outside a developer shell.
            get_filename_component(_linker_dir "${CMAKE_LINKER}" DIRECTORY)
            find_program(TANH_DUMPBIN dumpbin HINTS "${_linker_dir}")
            set(_tool "${TANH_DUMPBIN}")
            set(_tool_arg "-DDUMPBIN=${TANH_DUMPBIN}")
        else()
            if(CMAKE_NM)
                set(TANH_NM "${CMAKE_NM}")
            else()
                find_program(TANH_NM NAMES nm llvm-nm)
            endif()
            set(_tool "${TANH_NM}")
            set(_tool_arg "-DNM=${TANH_NM}")
        endif()
        if(NOT _tool)
            message(WARNING "tanh_add_export_check(${arg_NAME}): no nm/dumpbin found — test not registered")
            return()
        endif()

        set(_library "")
        if(arg_LIBRARY)
            get_target_property(_type ${arg_LIBRARY} TYPE)
            if(_type STREQUAL "SHARED_LIBRARY")
                set(_library "$<TARGET_FILE:${arg_LIBRARY}>")
            elseif(NOT _type STREQUAL "STATIC_LIBRARY")
                message(FATAL_ERROR "tanh_add_export_check(${arg_NAME}): LIBRARY must be a shared or static library (got ${_type})")
            endif()
        endif()
        set(_module "")
        if(arg_MODULE)
            set(_module "$<TARGET_FILE:${arg_MODULE}>")
        endif()

        add_test(NAME ${arg_NAME}
            COMMAND ${CMAKE_COMMAND}
                -DFORMAT=${_format}
                ${_tool_arg}
                "-DLIBRARY=${_library}"
                "-DMODULE=${_module}"
                "-DNAMESPACES=${arg_NAMESPACES}"
                "-DFORBID_NAMESPACES=${arg_FORBID_NAMESPACES}"
                "-DFORBID_PREFIXES=${arg_FORBID_PREFIXES}"
                "-DALLOW_REGEX=${arg_ALLOW_REGEX}"
                "-DTOLERATE_REGEX=$<$<STREQUAL:${_format},PE>:${arg_TOLERATE_REGEX_PE}>"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    endfunction()
    cmake_policy(POP)
    return()
endif()

# ------------------------------------------------------------------------------
# Script mode: the check.
# ------------------------------------------------------------------------------
if(NOT FORMAT MATCHES "^(ELF|MACHO|PE)$")
    message(FATAL_ERROR "check-exports: FORMAT must be ELF, MACHO or PE (got '${FORMAT}')")
endif()
if(NOT LIBRARY AND NOT MODULE)
    message(FATAL_ERROR "check-exports: LIBRARY or MODULE is required")
endif()
if(NOT NAMESPACES)
    message(FATAL_ERROR "check-exports: NAMESPACES is required")
endif()

# Itanium: "a::b" -> "1a1b"; MSVC: "a::b" -> "@b@a@@" (innermost first).
function(_tanh_ns_itanium ns out)
    string(REPLACE "::" ";" _parts "${ns}")
    set(_m "")
    foreach(_p IN LISTS _parts)
        string(LENGTH "${_p}" _l)
        string(APPEND _m "${_l}${_p}")
    endforeach()
    set(${out} "${_m}" PARENT_SCOPE)
endfunction()
function(_tanh_ns_msvc ns out)
    string(REPLACE "::" ";" _parts "${ns}")
    list(REVERSE _parts)
    string(JOIN "@" _m ${_parts})
    set(${out} "@${_m}@@" PARENT_SCOPE)
endfunction()

# Allow / forbid pattern lists for the format at hand.
set(_allow "")
set(_forbid "")
foreach(_ns IN LISTS NAMESPACES)
    if(FORMAT STREQUAL "PE")
        _tanh_ns_msvc("${_ns}" _m)
        list(APPEND _allow "${_m}")
    else()
        _tanh_ns_itanium("${_ns}" _m)
        # functions/data, const members, vtable/typeinfo/typeinfo-name/VTT, thunks, guards
        list(APPEND _allow "^_?_Z(NK?|T[VIST]N|T[hv][^N]*N|GVN)${_m}")
    endif()
endforeach()
list(APPEND _allow ${ALLOW_REGEX})
foreach(_ns IN LISTS FORBID_NAMESPACES)
    if(FORMAT STREQUAL "PE")
        _tanh_ns_msvc("${_ns}" _m)
        list(APPEND _forbid "${_m}")
    else()
        _tanh_ns_itanium("${_ns}" _m)
        list(APPEND _forbid "^_?_Z(NK?|T[VIST]N|T[hv][^N]*N|GVN)?${_m}")
    endif()
endforeach()
foreach(_p IN LISTS FORBID_PREFIXES)
    if(FORMAT STREQUAL "PE")
        list(APPEND _forbid "^${_p}")
    else()
        list(APPEND _forbid "^_?${_p}")
    endif()
endforeach()

# _tanh_exports(<file> <out> <unique-out>): defined dynamic exports of a shared object
# as a list of symbol names; on ELF also the STB_GNU_UNIQUE ones.
function(_tanh_exports file out unique_out)
    set(_names "")
    set(_unique "")
    if(FORMAT STREQUAL "PE")
        execute_process(COMMAND "${DUMPBIN}" /nologo /exports "${file}"
            OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "check-exports: dumpbin failed on ${file}: ${_err}")
        endif()
        string(REPLACE "\n" ";" _lines "${_raw}")
        foreach(_line IN LISTS _lines)
            # "   ordinal hint RVA      name" — forwarded exports ("name = ...") keep
            # their first token.
            if(_line MATCHES "^ +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +([^ \r]+)")
                list(APPEND _names "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    else()
        if(FORMAT STREQUAL "ELF")
            set(_args -D --defined-only)
        else()
            set(_args -g -U)
        endif()
        execute_process(COMMAND "${NM}" ${_args} "${file}"
            OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "check-exports: nm failed on ${file}: ${_err}")
        endif()
        string(REPLACE "\n" ";" _lines "${_raw}")
        foreach(_line IN LISTS _lines)
            # "address type name"; the type letter is what nm prints.
            if(_line MATCHES "^[0-9A-Fa-f]* +([A-Za-z?]) +([^ ]+)")
                list(APPEND _names "${CMAKE_MATCH_2}")
                if(CMAKE_MATCH_1 STREQUAL "u")
                    list(APPEND _unique "${CMAKE_MATCH_2}")
                endif()
            endif()
        endforeach()
    endif()
    set(${out} "${_names}" PARENT_SCOPE)
    set(${unique_out} "${_unique}" PARENT_SCOPE)
endfunction()

function(_tanh_matches_any name patterns out)
    set(_hit FALSE)
    foreach(_p IN LISTS patterns)
        if(name MATCHES "${_p}")
            set(_hit TRUE)
            break()
        endif()
    endforeach()
    set(${out} ${_hit} PARENT_SCOPE)
endfunction()

function(_tanh_report_list header items out_var)
    list(LENGTH items _n)
    list(SUBLIST items 0 40 _shown)
    string(REPLACE ";" "\n    " _shown "${_shown}")
    set(${out_var} "${header} (${_n}, first 40):\n    ${_shown}" PARENT_SCOPE)
endfunction()

set(_failures "")

if(LIBRARY)
    _tanh_exports("${LIBRARY}" _lib_exports _lib_unique)
    list(LENGTH _lib_exports _n)
    set(_bad "")
    foreach(_s IN LISTS _lib_exports)
        _tanh_matches_any("${_s}" "${_allow}" _ok)
        if(NOT _ok)
            _tanh_matches_any("${_s}" "${TOLERATE_REGEX}" _tolerated)
            if(NOT _tolerated)
                list(APPEND _bad "${_s}")
            endif()
        endif()
    endforeach()
    list(LENGTH _bad _nbad)
    message(STATUS "check-exports: ${LIBRARY}: ${_n} exports, ${_nbad} outside ${NAMESPACES}")
    if(_bad)
        _tanh_report_list("${LIBRARY} exports symbols outside ${NAMESPACES}" "${_bad}" _msg)
        list(APPEND _failures "${_msg}")
    endif()
    if(_lib_unique)
        _tanh_report_list("${LIBRARY} defines STB_GNU_UNIQUE symbols" "${_lib_unique}" _msg)
        list(APPEND _failures "${_msg}")
    endif()
endif()

if(MODULE)
    _tanh_exports("${MODULE}" _mod_exports _mod_unique)
    list(LENGTH _mod_exports _n)
    set(_bad "")
    set(_leaked "")
    foreach(_s IN LISTS _mod_exports)
        _tanh_matches_any("${_s}" "${_allow}" _is_api)
        if(_is_api)
            # The library's API inside the module: an import of the shared library's
            # inline members when there is one, a leak of the embedded static library
            # otherwise.
            if(NOT LIBRARY)
                list(APPEND _leaked "${_s}")
            endif()
            continue()
        endif()
        _tanh_matches_any("${_s}" "${_forbid}" _hit)
        if(_hit)
            _tanh_matches_any("${_s}" "${TOLERATE_REGEX}" _tolerated)
            if(NOT _tolerated)
                list(APPEND _bad "${_s}")
            endif()
        endif()
    endforeach()
    list(LENGTH _bad _nbad)
    list(LENGTH _leaked _nleaked)
    message(STATUS "check-exports: ${MODULE}: ${_n} exports, ${_nbad} forbidden, ${_nleaked} of the statically linked library")
    if(_bad)
        _tanh_report_list("${MODULE} exports forbidden symbols" "${_bad}" _msg)
        list(APPEND _failures "${_msg}")
    endif()
    if(_leaked)
        _tanh_report_list("${MODULE} exports symbols of the statically linked library" "${_leaked}" _msg)
        list(APPEND _failures "${_msg}")
    endif()
    if(_mod_unique)
        _tanh_report_list("${MODULE} defines STB_GNU_UNIQUE symbols" "${_mod_unique}" _msg)
        list(APPEND _failures "${_msg}")
    endif()
endif()

if(_failures)
    string(REPLACE ";" "\n" _failures "${_failures}")
    message(FATAL_ERROR "check-exports: symbol-visibility invariant violated\n${_failures}")
endif()
message(STATUS "check-exports: OK")
