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
set_target_properties(${PROJECT_NAME} PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-gnu-unique)
endif()
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /Gy)
    target_link_options(${PROJECT_NAME} PRIVATE /OPT:REF)
elseif(APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${PROJECT_NAME} PRIVATE LINKER:-dead_strip)
elseif(UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${PROJECT_NAME} PRIVATE LINKER:--gc-sections)
endif()
# Finally pin the export table to the entry point at link time: hidden visibility
# cannot hide what a header stamps default-visibility itself (libstdc++'s std::
# instantiations, anira's inline members when libanira is shared), and on macOS
# nothing short of an export list keeps a force-loaded static ExecuTorch out. PE
# exports only what carries dllexport, so MSVC needs nothing.
if(APPLE)
    set(_clap_export_list "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clap.exports")
    target_link_options(${PROJECT_NAME} PRIVATE "LINKER:-exported_symbols_list,${_clap_export_list}")
elseif(UNIX)
    set(_clap_export_list "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clap.map")
    target_link_options(${PROJECT_NAME} PRIVATE "LINKER:--version-script=${_clap_export_list}")
endif()
if(_clap_export_list)
    # Relink when the allowlist changes.
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY LINK_DEPENDS "${_clap_export_list}")
endif()

if(APPLE)
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
elseif(UNIX)
    target_compile_definitions(${PROJECT_NAME} PRIVATE IS_LINUX=1)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".clap" PREFIX "")
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE IS_WIN=1)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".clap" PREFIX "")
endif()
