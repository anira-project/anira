# ==============================================================================
# tanh-tooling · cmake/compile-commands.cmake — clangd's compile database in one place.
#
#   tanh_link_compile_commands([DIRECTORY <dir>])
#
# Points <dir>/compile_commands.json at this build tree's compile_commands.json with a
# relative symbolic link. <dir> defaults to build/ in the top-level source tree, where
# the clang family's .clangd reads the database. A project with several build trees
# (presets under build/<platform>/<config>/) calls it in the trees clangd should follow;
# every configure of such a tree re-points the link, so clangd reads the tree that was
# configured last. The link is made at configure time and the database is written at
# generate time, so on a tree's first configure it dangles until generation finishes.
#
# Does nothing in a subproject (the top-level project owns its source tree's build/) or
# when <dir> is the build tree itself (the database is already there). Warns and leaves
# <dir> as it is when this tree writes no database (CMAKE_EXPORT_COMPILE_COMMANDS off, or
# a generator other than Makefiles/Ninja), when <dir>/compile_commands.json is a regular
# file (a tree configured into <dir> owns it), or when the link cannot be created
# (Windows without the symbolic-link privilege).
#
# Inputs: CMAKE_EXPORT_COMPILE_COMMANDS, CMAKE_GENERATOR, CMAKE_SOURCE_DIR,
# CMAKE_BINARY_DIR, PROJECT_SOURCE_DIR. Outputs: none (the link). Requires CMake >= 3.18;
# call after project().
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")

function(tanh_link_compile_commands)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "DIRECTORY" "")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tanh_link_compile_commands: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT PROJECT_SOURCE_DIR)
        message(FATAL_ERROR "tanh_link_compile_commands: call it after project()")
    endif()
    if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        return()
    endif()

    if(arg_DIRECTORY)
        get_filename_component(_dir "${arg_DIRECTORY}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    else()
        set(_dir "${CMAKE_SOURCE_DIR}/build")
    endif()
    get_filename_component(_dir_real "${_dir}" REALPATH)
    get_filename_component(_tree_real "${CMAKE_BINARY_DIR}" REALPATH)
    if(_dir_real STREQUAL _tree_real)
        return()
    endif()

    set(_link "${_dir}/compile_commands.json")
    if(NOT CMAKE_EXPORT_COMPILE_COMMANDS OR NOT CMAKE_GENERATOR MATCHES "Makefiles|Ninja")
        message(WARNING "tanh_link_compile_commands: this tree writes no compile_commands.json "
                        "(generator '${CMAKE_GENERATOR}', CMAKE_EXPORT_COMPILE_COMMANDS "
                        "'${CMAKE_EXPORT_COMPILE_COMMANDS}'); ${_link} is left as it is")
        return()
    endif()
    if(EXISTS "${_link}" AND NOT IS_SYMLINK "${_link}")
        message(WARNING "tanh_link_compile_commands: ${_link} is a file, not a link (a tree "
                        "configured into ${_dir} owns it); it is left as it is")
        return()
    endif()

    # Relative, so the link survives moving the checkout; computed between real paths, so
    # a symlinked source or build directory cannot skew it.
    file(RELATIVE_PATH _target "${_dir_real}" "${_tree_real}/compile_commands.json")
    if(CMAKE_HOST_WIN32)
        file(TO_NATIVE_PATH "${_target}" _target)
    endif()
    file(MAKE_DIRECTORY "${_dir}")
    file(CREATE_LINK "${_target}" "${_link}" SYMBOLIC RESULT _result)
    if(NOT _result EQUAL 0)
        message(WARNING "tanh_link_compile_commands: cannot link ${_link} -> ${_target}: ${_result}")
        return()
    endif()
    message(STATUS "compile_commands.json for clangd: ${_link} -> ${_target}")
endfunction()
