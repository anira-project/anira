cmake_minimum_required(VERSION 3.15)

# Use the dynamic MSVC runtime (/MD), matching anira (which uses the CMake default).
# A static anira is linked into this plugin, so a /MT plugin would clash with anira's
# /MD objects (LNK2038 "RuntimeLibrary mismatch").
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.11 CACHE STRING "Build for 10.1")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "You need to specify CMAKE_BUILD_TYPE")
endif()

project(anira-clap-plugin-example VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

FetchContent_Declare(clap
        GIT_REPOSITORY https://github.com/free-audio/clap.git
        GIT_TAG main
        GIT_PROGRESS TRUE
        EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(clap)

FetchContent_Declare(clap-helpers
        GIT_REPOSITORY https://github.com/free-audio/clap-helpers.git
        GIT_TAG main
        GIT_PROGRESS TRUE
        EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(clap-helpers)

add_library(${PROJECT_NAME} MODULE
        anira-clap-demo.cpp
        anira-clap-demo-pluginentry.cpp
        utils/DryWetMixer.cpp
)
target_link_libraries(${PROJECT_NAME} clap-core clap-helpers anira)

# What a plugin embedding anira should do (docs/sphinx/troubleshooting.rst, "Host application ships its own backend runtime"):
# export nothing but its entry point — CLAP_EXPORT marks clap_entry
# default-visibility/dllexport — so that the embedded anira and its backend runtimes
# stay private and cannot be interposed by a host that ships its own copy; under GCC
# bind no STB_GNU_UNIQUE symbol, so that the host can actually unload the plugin; and
# let the linker drop what the plugin never references (anira's objects are compiled
# with one section per function for exactly that).
# All of that is the shared policy (cmake/tanh/symbol-policy.cmake, available from the
# in-tree build and from the installed aniraConfig.cmake alike): hidden visibility,
# -fno-gnu-unique, dead-code stripping, and an export table generated at configure
# time and pinned to clap_entry.
tanh_apply_symbol_policy(${PROJECT_NAME})
tanh_set_export_allowlist(${PROJECT_NAME} SYMBOL clap_entry)

if(TANH_OPERATING_SYSTEM STREQUAL "macOS")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            BUNDLE True
            BUNDLE_EXTENSION clap
            MACOSX_BUNDLE_GUI_IDENTIFIER org.anira-project.${PROJECT_NAME}
            MACOSX_BUNDLE_BUNDLE_NAME ${PROJECT_NAME}
            MACOSX_BUNDLE_BUNDLE_VERSION "0.1"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
            MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_LIST_DIR}/cmake/anira-clap-demo.plist.in
    )
    target_link_libraries(${PROJECT_NAME} "-framework CoreFoundation" "-framework AppKit" "-framework CoreGraphics")
    target_compile_definitions(${PROJECT_NAME} PRIVATE IS_MAC=1)
elseif(TANH_OPERATING_SYSTEM STREQUAL "Linux")
    target_compile_definitions(${PROJECT_NAME} PRIVATE IS_LINUX=1)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".clap" PREFIX "")
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE IS_WIN=1)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".clap" PREFIX "")
endif()
