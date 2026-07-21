# Rename every defined external symbol in a static archive except the backend's
# public C API, by rewriting each member with llvm-objcopy --redefine-syms.
#
# This is the COFF flavor of the backend-localization step (see
# anira_localize_static_archive in AniraBackends.cmake): MSVC has no partial
# link, so instead of demoting the vendored internals (XNNPACK, cpuinfo,
# pthreadpool, ...) to local symbols they are renamed — definitions and
# references consistently, member by member — so they can no longer collide or
# cross-bind with another backend's copy of the same vendored libraries.
# Undefined externals (CRT/OS imports) are untouched: only symbols DEFINED in
# the archive are renamed.
#
# Script arguments (all required, passed as -D<name>=<value> with -P):
#   NM             llvm-nm executable
#   OBJCOPY        llvm-objcopy executable
#   ARCHIVE        input static archive (.lib/.a)
#   KEEP_PREFIX    symbol prefix to keep untouched (the public API, e.g. LiteRt)
#   RENAME_PREFIX  prefix prepended to every other defined symbol
#   OUTPUT         rewritten archive path
#
# Runs standalone (cmake -P), so it is testable outside a configure and usable
# from custom commands.

foreach(_arg NM OBJCOPY ARCHIVE KEEP_PREFIX RENAME_PREFIX OUTPUT)
    if(NOT DEFINED ${_arg})
        message(FATAL_ERROR "AniraRenameArchive: missing -D${_arg}=...")
    endif()
endforeach()

execute_process(
    COMMAND "${NM}" --defined-only --extern-only "${ARCHIVE}"
    OUTPUT_VARIABLE _nm_out
    ERROR_VARIABLE _nm_err
    RESULT_VARIABLE _nm_res
)
if(NOT _nm_res EQUAL 0)
    message(FATAL_ERROR "AniraRenameArchive: ${NM} failed on ${ARCHIVE}: ${_nm_err}")
endif()

# nm prints one "addr type name" line per defined external symbol, plus member
# header lines ("member.obj:") and blanks. Bulk list operations keep the whole
# pass in native code — a per-line foreach takes minutes on the ~200k lines a
# real backend archive produces. \r handles llvm-nm's CRLF output on Windows.
string(REPLACE "\r" "" _nm_out "${_nm_out}")
# Strip the "addr type " prefix off symbol lines; header/blank lines pass through.
string(REGEX REPLACE "(^|\n)[0-9a-fA-F]+ +[A-Za-z] +" "\\1" _nm_out "${_nm_out}")
string(REPLACE "\n" ";" _lines "${_nm_out}")
list(REMOVE_DUPLICATES _lines)
# Drop blanks, member headers (they end with ':'), and the kept API prefix.
list(FILTER _lines EXCLUDE REGEX "(^$|:$)")
list(FILTER _lines EXCLUDE REGEX "^${KEEP_PREFIX}")
list(LENGTH _lines _count)
list(TRANSFORM _lines REPLACE "^(.+)$" "\\1 ${RENAME_PREFIX}\\1")
list(JOIN _lines "\n" _map)
string(APPEND _map "\n")
if(_count EQUAL 0)
    message(FATAL_ERROR "AniraRenameArchive: no symbols to rename found in ${ARCHIVE} "
                        "(wrong archive, or nm output not understood)")
endif()

get_filename_component(_out_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")
set(_mapfile "${OUTPUT}.rename.map")
file(WRITE "${_mapfile}" "${_map}")

execute_process(
    COMMAND "${OBJCOPY}" "--redefine-syms=${_mapfile}" "${ARCHIVE}" "${OUTPUT}"
    ERROR_VARIABLE _oc_err
    RESULT_VARIABLE _oc_res
)
if(NOT _oc_res EQUAL 0)
    message(FATAL_ERROR "AniraRenameArchive: ${OBJCOPY} failed on ${ARCHIVE}: ${_oc_err}")
endif()

message(STATUS "AniraRenameArchive: renamed ${_count} defined symbols in "
               "${ARCHIVE} (kept ${KEEP_PREFIX}*) -> ${OUTPUT}")
