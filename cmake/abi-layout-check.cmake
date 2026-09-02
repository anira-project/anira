# ==============================================================================
# anira · cmake/abi-layout-check.cmake — gate 3, script mode.
#
#   cmake -DLAYOUT_EXE=<anira_abi_layout> -DEXPECTED=<abi/layout-N.txt> -DACTUAL=<out> -P <this>
#       runs the layout printer and fails when its table differs from the committed one
#   cmake -DLAYOUT_EXE=<anira_abi_layout> -DWRITE_TO=<abi/layout-N.txt> -P <this>
#       rewrites the committed table (only in a commit that changes ANIRA_ABI_MAJOR)
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
    message(FATAL_ERROR
        "abi-layout-check: the Tier-1 layout differs from ${EXPECTED}\n"
        "--- built ---\n${_table}--- committed ---\n${_expected}"
        "A Tier-1 layout may change only in a commit that changes ANIRA_ABI_MAJOR; "
        "then regenerate the table with the anira_abi_layout_regen target.")
endif()
message(STATUS "abi-layout-check: Tier-1 layout matches ${EXPECTED}")
