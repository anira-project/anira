# ==============================================================================
# CheckExports.cmake — CTest check of the dynamic export tables (script mode).
#
# Verifies the symbol-visibility invariant described in the top-level
# CMakeLists.txt on the binaries CMake actually produced:
#
#   * LIBRARY (the shared libanira, empty for a static build): every exported
#     symbol belongs to namespace anira — the ANIRA_API surface plus the vtables,
#     typeinfo, thunks and guards of those classes — and nothing else. In particular
#     no std::, Ort*, torch::/c10::, executorch::, xnn_*, TfLite*/LiteRt* symbol.
#   * MODULE (the plugin-shaped test module embedding anira, shared or static):
#     exports no backend-runtime symbol. Its own entry points and whatever its own
#     translation units export are its business, as for any plugin.
#   * ELF only: neither defines an STB_GNU_UNIQUE symbol (glibc never unloads an
#     object that does; the module must be dlclose-able).
#
# Usage:
#   cmake -DFORMAT=ELF|MACHO|PE -DNM=<nm> -DDUMPBIN=<dumpbin>
#         -DLIBRARY=<path or empty> -DMODULE=<path> -P CheckExports.cmake
# ==============================================================================

if(NOT FORMAT MATCHES "^(ELF|MACHO|PE)$")
    message(FATAL_ERROR "CheckExports: FORMAT must be ELF, MACHO or PE (got '${FORMAT}')")
endif()
if(NOT MODULE)
    message(FATAL_ERROR "CheckExports: MODULE is required")
endif()

# Namespace anira in the Itanium mangling (ELF, Mach-O with a leading underscore):
# functions/data, const members, vtable/typeinfo/typeinfo-name/VTT, thunks, guards.
set(_itanium_allow "^_?_Z(NK?|T[VIST]N|T[hv][^N]*N|GVN)5anira")
# The same set in the MSVC decoration.
set(_msvc_allow "@anira@@")
# Backend runtimes, keyed on their namespace / prefix encodings so that anira's own
# LiteRtProcessor/ExecuTorchProcessor/... names never trip it.
set(_itanium_forbid
    "^_?_ZN?(3Ort|11onnxruntime|5torch|3c10|6caffe2|10executorch|6tflite|6litert|11flatbuffers|5Eigen|4absl)"
    "^_?(Ort|OrtApi|xnn_|pthreadpool_|cpuinfo_|kai_|TfLite|LiteRt)")
set(_msvc_forbid
    "@(Ort|onnxruntime|torch|c10|caffe2|executorch|tflite|litert|flatbuffers|Eigen|absl)@@"
    "^(Ort|OrtApi|xnn_|pthreadpool_|cpuinfo_|kai_|TfLite|LiteRt)")

# anira_exports(<file> <out-var> <unique-out-var>): defined dynamic exports of a
# shared object as a list of symbol names; on ELF also the STB_GNU_UNIQUE ones.
function(anira_exports file out unique_out)
    set(_names "")
    set(_unique "")
    if(FORMAT STREQUAL "PE")
        execute_process(COMMAND "${DUMPBIN}" /nologo /exports "${file}"
            OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "CheckExports: dumpbin failed on ${file}: ${_err}")
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
            message(FATAL_ERROR "CheckExports: nm failed on ${file}: ${_err}")
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

function(anira_matches_any name patterns out)
    set(_hit FALSE)
    foreach(_p IN LISTS patterns)
        if(name MATCHES "${_p}")
            set(_hit TRUE)
            break()
        endif()
    endforeach()
    set(${out} ${_hit} PARENT_SCOPE)
endfunction()

if(FORMAT STREQUAL "PE")
    set(_allow "${_msvc_allow}")
    set(_forbid "${_msvc_forbid}")
else()
    set(_allow "${_itanium_allow}")
    set(_forbid "${_itanium_forbid}")
endif()

set(_failures "")

if(LIBRARY)
    anira_exports("${LIBRARY}" _lib_exports _lib_unique)
    list(LENGTH _lib_exports _n)
    set(_bad "")
    foreach(_s IN LISTS _lib_exports)
        if(NOT _s MATCHES "${_allow}")
            list(APPEND _bad "${_s}")
        endif()
    endforeach()
    list(LENGTH _bad _nbad)
    message(STATUS "CheckExports: ${LIBRARY}: ${_n} exports, ${_nbad} outside namespace anira")
    if(_nbad GREATER 0)
        list(SUBLIST _bad 0 40 _shown)
        string(REPLACE ";" "\n    " _shown "${_shown}")
        list(APPEND _failures
            "${LIBRARY} exports ${_nbad} symbol(s) outside namespace anira (first 40):\n    ${_shown}")
    endif()
    if(_lib_unique)
        list(LENGTH _lib_unique _nu)
        list(SUBLIST _lib_unique 0 20 _shown)
        string(REPLACE ";" "\n    " _shown "${_shown}")
        list(APPEND _failures
            "${LIBRARY} defines ${_nu} STB_GNU_UNIQUE symbol(s) (first 20):\n    ${_shown}")
    endif()
endif()

anira_exports("${MODULE}" _mod_exports _mod_unique)
list(LENGTH _mod_exports _n)
set(_bad "")
foreach(_s IN LISTS _mod_exports)
    if(_s MATCHES "${_allow}")
        continue()
    endif()
    anira_matches_any("${_s}" "${_forbid}" _hit)
    if(_hit)
        list(APPEND _bad "${_s}")
    endif()
endforeach()
list(LENGTH _bad _nbad)
message(STATUS "CheckExports: ${MODULE}: ${_n} exports, ${_nbad} backend-runtime symbols")
if(_nbad GREATER 0)
    list(SUBLIST _bad 0 40 _shown)
    string(REPLACE ";" "\n    " _shown "${_shown}")
    list(APPEND _failures
        "${MODULE} exports ${_nbad} backend-runtime symbol(s) (first 40):\n    ${_shown}")
endif()
if(_mod_unique)
    list(LENGTH _mod_unique _nu)
    list(SUBLIST _mod_unique 0 20 _shown)
    string(REPLACE ";" "\n    " _shown "${_shown}")
    list(APPEND _failures
        "${MODULE} defines ${_nu} STB_GNU_UNIQUE symbol(s) (first 20):\n    ${_shown}")
endif()

if(_failures)
    string(REPLACE ";" "\n" _failures "${_failures}")
    message(FATAL_ERROR "CheckExports: symbol-visibility invariant violated\n${_failures}")
endif()
message(STATUS "CheckExports: OK")
