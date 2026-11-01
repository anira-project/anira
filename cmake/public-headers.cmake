# ==============================================================================
# The public header set: every file anira installs under include/anira, by name. The install
# rule (cmake/install.cmake) copies exactly these, and the per-file self-containment gates of
# test/abi compile each one alone; a header under include/anira that is not listed here fails
# the configure, so a header tools/abi/gen.py adds cannot be left out of the package or the gates.
#
#   ANIRA_PUBLIC_UMBRELLAS     the C umbrellas, at include/anira/ (generated)
#   ANIRA_PUBLIC_PLATFORM      the platform companions, at include/anira/ (generated; empty on
#                              every other target)
#   ANIRA_PUBLIC_ABI           the C ABI headers, at include/anira/abi/ (generated), the
#                              draft ones as draft/<name>
#   ANIRA_PUBLIC_CXX           the header-only C++ faces, relative to include/anira/
# ==============================================================================
include_guard(GLOBAL)

set(ANIRA_PUBLIC_UMBRELLAS anira anira_all)
set(ANIRA_PUBLIC_PLATFORM anira_em)
set(ANIRA_PUBLIC_ABI export status version enums log tensor config context core thread lifecycle
    processor engine handler draft/tensor_platform)
set(ANIRA_PUBLIC_CXX anira.hpp compat/v2.hpp)

# Every header file under include/anira must be one of the above.
set(_anira_public_expected "")
foreach(_h IN LISTS ANIRA_PUBLIC_UMBRELLAS ANIRA_PUBLIC_PLATFORM)
    list(APPEND _anira_public_expected "${_h}.h")
endforeach()
foreach(_h IN LISTS ANIRA_PUBLIC_ABI)
    list(APPEND _anira_public_expected "abi/${_h}.h")
endforeach()
list(APPEND _anira_public_expected ${ANIRA_PUBLIC_CXX})
file(GLOB_RECURSE _anira_public_found CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_LIST_DIR}/../include/anira"
    "${CMAKE_CURRENT_LIST_DIR}/../include/anira/*.h" "${CMAKE_CURRENT_LIST_DIR}/../include/anira/*.hpp")
list(SORT _anira_public_found)
list(SORT _anira_public_expected)
if(NOT _anira_public_found STREQUAL _anira_public_expected)
    set(_anira_unlisted ${_anira_public_found})
    list(REMOVE_ITEM _anira_unlisted ${_anira_public_expected})
    set(_anira_absent ${_anira_public_expected})
    list(REMOVE_ITEM _anira_absent ${_anira_public_found})
    message(FATAL_ERROR "cmake/public-headers.cmake: include/anira and the public header set differ "
                        "(not listed: ${_anira_unlisted}; listed but missing: ${_anira_absent})")
endif()
unset(_anira_public_expected)
unset(_anira_public_found)
