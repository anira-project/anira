# ==============================================================================
# tanh-tooling · cmake/modules-version.cmake — the version stamp of this module set.
#
# Every module includes this file first. It records which tanh-tooling release the
# loaded modules come from and warns when a second, different release is loaded into
# the same build: functions are global in CMake, so if anira carries cmake/tanh/ at
# one tag and the tanh-lib it fetches carries cmake/tanh/ at another, the definition
# included last silently wins. Pin every project in one build to the same tag.
#
# Outputs: TANH_CMAKE_MODULES_VERSION (plain variable, caller scope). The check itself
# uses a global property, because directory scopes do not see each other's variables.
# Bumped by the release commit together with the default REF in install.sh.
# ==============================================================================
set(TANH_CMAKE_MODULES_VERSION "0.2.7")

get_property(_tanh_loaded_version GLOBAL PROPERTY TANH_CMAKE_MODULES_VERSION)
if(_tanh_loaded_version)
    if(NOT _tanh_loaded_version STREQUAL TANH_CMAKE_MODULES_VERSION)
        get_property(_tanh_loaded_from GLOBAL PROPERTY TANH_CMAKE_MODULES_ORIGIN)
        message(WARNING "tanh-tooling cmake modules: ${CMAKE_CURRENT_LIST_DIR} is release "
                        "${TANH_CMAKE_MODULES_VERSION}, but release ${_tanh_loaded_version} was already "
                        "loaded from ${_tanh_loaded_from}. Pin both projects to the same tanh-tooling tag.")
    endif()
else()
    set_property(GLOBAL PROPERTY TANH_CMAKE_MODULES_VERSION "${TANH_CMAKE_MODULES_VERSION}")
    set_property(GLOBAL PROPERTY TANH_CMAKE_MODULES_ORIGIN "${CMAKE_CURRENT_LIST_DIR}")
endif()
unset(_tanh_loaded_version)
unset(_tanh_loaded_from)
