# ==============================================================================
# tanh-tooling · cmake/package.cmake — CPack Debian packaging of a library project.
#
#   tanh_cpack_debian(VENDOR <name> CONTACT <email> RUNTIME_DESCRIPTION <text> DEV_DESCRIPTION <text>
#                     [PACKAGE_NAME lib<project>] [MAINTAINER "<name> <email>"]
#                     [LICENSE_FILE <path>] [README_FILE <path>]
#                     [DEPS_DESCRIPTION <text>] [DEPS_COMPONENTS <component>...])
#
# One runtime package "<PACKAGE_NAME><major>" (component `runtime`, section libs), one
# "-dev" package (component `dev`, section libdevel, depends on runtime) and, if any
# DEPS_COMPONENTS are given, one "-deps" package grouping them. Component dependencies
# and shlibs are generated automatically; binaries are stripped. Ends with
# include(CPack) — call it once, after every install() rule of the project, from the
# top-level directory (it is a macro so CPack sees the variables it sets).
#
#   cd build && cpack -G DEB      # then: apt install ./lib<project>*.deb && ldconfig
#
# Requires CMake >= 3.18.
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

macro(tanh_cpack_debian)
    cmake_parse_arguments(_tcd "" "PACKAGE_NAME;VENDOR;CONTACT;MAINTAINER;LICENSE_FILE;README_FILE;RUNTIME_DESCRIPTION;DEV_DESCRIPTION;DEPS_DESCRIPTION" "DEPS_COMPONENTS" ${ARGN})
    if(_tcd_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tanh_cpack_debian: unexpected arguments: ${_tcd_UNPARSED_ARGUMENTS}")
    endif()
    foreach(_req VENDOR CONTACT RUNTIME_DESCRIPTION DEV_DESCRIPTION)
        if(NOT _tcd_${_req})
            message(FATAL_ERROR "tanh_cpack_debian: ${_req} is required")
        endif()
    endforeach()
    if(NOT _tcd_PACKAGE_NAME)
        set(_tcd_PACKAGE_NAME "lib${PROJECT_NAME}")
    endif()
    if(NOT _tcd_MAINTAINER)
        set(_tcd_MAINTAINER "${_tcd_VENDOR} <${_tcd_CONTACT}>")
    endif()
    if(NOT _tcd_LICENSE_FILE)
        set(_tcd_LICENSE_FILE "${PROJECT_SOURCE_DIR}/LICENSE")
    endif()
    if(NOT _tcd_README_FILE)
        set(_tcd_README_FILE "${PROJECT_SOURCE_DIR}/README.md")
    endif()
    if(NOT _tcd_DEPS_DESCRIPTION)
        set(_tcd_DEPS_DESCRIPTION "dependencies of ${_tcd_PACKAGE_NAME}${PROJECT_VERSION_MAJOR}")
    endif()

    include(CPackComponent)

    set(CPACK_THREADS 0)  # all cores
    set(CPACK_PACKAGE_NAME "${_tcd_PACKAGE_NAME}")
    set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
    set(CPACK_PACKAGE_VENDOR "${_tcd_VENDOR}")
    set(CPACK_VERBATIM_VARIABLES YES)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}")
    set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/packages")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_PACKAGE_CONTACT "${_tcd_CONTACT}")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${_tcd_MAINTAINER}")
    set(CPACK_RESOURCE_FILE_LICENSE "${_tcd_LICENSE_FILE}")
    set(CPACK_RESOURCE_FILE_README "${_tcd_README_FILE}")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

    # Each group (or ungrouped component) becomes its own package.
    set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
    set(CPACK_DEB_COMPONENT_INSTALL YES)
    cpack_add_component(runtime REQUIRED)
    cpack_add_component(dev DEPENDS runtime)
    foreach(_c IN LISTS _tcd_DEPS_COMPONENTS)
        cpack_add_component(${_c} GROUP deps)
    endforeach()

    # The runtime package is named after the major version instead of "-runtime".
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}${PROJECT_VERSION_MAJOR}")
    set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION libs)
    set(CPACK_DEBIAN_DEV_PACKAGE_SECTION libdevel)
    set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
    set(CPACK_STRIP_FILES YES)  # lintian: unstripped-binary-or-object
    set(CPACK_DEBIAN_RUNTIME_DESCRIPTION "${_tcd_RUNTIME_DESCRIPTION}")
    set(CPACK_DEBIAN_DEV_DESCRIPTION "${_tcd_DEV_DESCRIPTION}")
    set(CPACK_DEBIAN_DEPS_DESCRIPTION "${_tcd_DEPS_DESCRIPTION}")

    include(CPack)
endmacro()
