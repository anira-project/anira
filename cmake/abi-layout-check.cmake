# ==============================================================================
# anira · cmake/abi-layout-check.cmake — gate 3, script mode.
#
#   cmake -DLAYOUT_EXE=<anira_abi_layout> -DEXPECTED=<abi/layout-N.txt> -DACTUAL=<out> -P <this>
#       runs the layout printer and fails when its table differs from the committed one
#   cmake -DLAYOUT_EXE=<anira_abi_layout> -DWRITE_TO=<abi/layout-N.txt> -P <this>
#       rewrites the committed table from the built printer
#
# N is ANIRA_ABI_MAJOR, and the script reads it from the file name. While N is 0 (before the
# v3.0.0 freeze) the table grows with the registry: tools/abi/gen.py writes it from
# abi/anira.yml together with the headers, and this gate proves that the compiler lays the
# records out the way gen.py's natural-alignment model says. From the freeze on the table
# changes only in a commit that changes ANIRA_ABI_MAJOR.
# ==============================================================================
if(NOT LAYOUT_EXE)
    message(FATAL_ERROR "abi-layout-check: LAYOUT_EXE is required")
endif()
execute_process(COMMAND "${LAYOUT_EXE}" OUTPUT_VARIABLE _table RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "abi-layout-check: ${LAYOUT_EXE} failed (${_rc})")
endif()
string(REPLACE "\r\n" "\n" _table "${_table}")
if(WRITE_TO)
    file(WRITE "${WRITE_TO}" "${_table}")
    message(STATUS "abi-layout-check: wrote ${WRITE_TO}")
    return()
endif()
if(NOT EXPECTED OR NOT EXISTS "${EXPECTED}")
    message(FATAL_ERROR "abi-layout-check: no committed table at '${EXPECTED}'")
endif()
file(READ "${EXPECTED}" _expected)
string(REPLACE "\r\n" "\n" _expected "${_expected}")
if(ACTUAL)
    file(WRITE "${ACTUAL}" "${_table}")
endif()
if(NOT _table STREQUAL _expected)
    # The ABI major is the N of abi/layout-N.txt; an unreadable name gets the frozen rule.
    set(_major "")
    if(EXPECTED MATCHES "layout-([0-9]+)\\.txt$")
        set(_major "${CMAKE_MATCH_1}")
    endif()
    if(_major STREQUAL "0")
        string(CONCAT _rule
            "ANIRA_ABI_MAJOR is 0: the Tier-1 table is not frozen yet and grows with the registry. "
            "Run `python3 tools/abi/gen.py --repo . --write` and commit abi/layout-0.txt together "
            "with the regenerated headers. If `gen.py --check` is already clean, the compiler "
            "disagrees with gen.py's natural-alignment model (the built table above is the "
            "compiler's): fix the record in abi/anira.yml or the model in tools/abi/gen.py, never "
            "the table by hand.")
    else()
        string(CONCAT _rule
            "The Tier-1 layout is frozen: it may change only in a commit that changes "
            "ANIRA_ABI_MAJOR, which starts a new abi/layout-<major>.txt (tools/abi/gen.py writes "
            "it; the anira_abi_layout_regen target writes the compiler's view of it). A new shape "
            "inside a major is a new struct with new functions, never a changed one.")
    endif()
    message(FATAL_ERROR
        "abi-layout-check: the Tier-1 layout differs from ${EXPECTED}\n"
        "--- built ---\n${_table}--- committed ---\n${_expected}"
        "${_rule}")
endif()
message(STATUS "abi-layout-check: Tier-1 layout matches ${EXPECTED}")
