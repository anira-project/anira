# ==============================================================================
# tanh-tooling · cmake/symbol-policy.cmake — the symbol-export policy of a library.
#
# Background: memory-bridge/guides/about_libraries.md. On ELF and Mach-O every symbol
# is exported unless told otherwise; the policy makes them behave like PE — the
# export table is an allowlist — so that a library embedded in a plugin never leaks
# its bundled third-party code (ONNX Runtime, miniaudio, libstdc++ instantiations…)
# into the host process, where the dynamic linker would interpose it against the
# host's own copy.
#
# Functions (all take the target; the consumer decides what they apply to):
#
#   tanh_apply_symbol_policy(<target> [EXPORT_PREFIX <P>] [NO_GC_SECTIONS])
#     Compile side, same for static and shared:
#       - POSITION_INDEPENDENT_CODE, <LANG>_VISIBILITY_PRESET hidden, VISIBILITY_INLINES_HIDDEN
#       - <P>_BUILDING (PRIVATE) while compiling the target; <P>_STATIC (PUBLIC) when the
#         target is a static library (OBJECT libraries follow BUILD_SHARED_LIBS). The
#         library's export header selects on these: <P>_STATIC → empty macro,
#         <P>_BUILDING → the platform's export decoration (dllexport /
#         visibility("default")), else the import decoration; each library's own
#         export header spells the selector.
#         Omit EXPORT_PREFIX for a plugin/module or executable that has no export macro.
#       - -fno-gnu-unique for GNU C++ on ELF: GCC binds exported vague-linkage data as
#         STB_GNU_UNIQUE, which glibc answers by never unloading the object (NODELETE)
#         and by sharing those statics across RTLD_LOCAL plugins.
#       - one section per function/variable (-ffunction-sections -fdata-sections, /Gy)
#         so the link-time GC below can drop unreferenced code individually.
#       - /wd4251 (PUBLIC, MSVC): "class needs to have dll-interface" is inherent to
#         dllexport classes holding std::/template members and fires on the dllimport
#         side too; suppressed here once instead of in every export header.
#     Link side, SHARED/MODULE/EXECUTABLE targets only (a static archive has no link):
#       - dead-code stripping: --gc-sections (ELF) / -dead_strip (Mach-O) / /OPT:REF (MSVC),
#         unless NO_GC_SECTIONS.
#
#   tanh_set_export_allowlist(<target> [NAMESPACE <ns>...] [SYMBOL <glob>...])
#     Pins the dynamic export table of a SHARED/MODULE/EXECUTABLE target at link time to
#     the given C++ namespaces (functions, data, vtables, typeinfo, thunks, guard
#     variables — the nine Itanium pattern kinds) plus the given unmangled symbol globs
#     (C entry points such as clap_entry). Generates the ELF version script or the
#     Mach-O -exported_symbols_list into ${CMAKE_CURRENT_BINARY_DIR}/tanh-exports/ and
#     relinks when it changes. Why it is needed on top of hidden visibility: a header
#     can stamp default visibility itself (libstdc++ wraps namespace std in
#     visibility("default"), LibTorch's C10_API classes…). No-op for static libraries
#     (no export table), PE (dllexport is the allowlist) and Wasm.
#
#   tanh_hidden_archive_link_items(<archive> <out_libs> <out_opts>)
#     How to link a prebuilt static archive so that none of its symbols leaves the
#     consumer's shared object: ELF → the archive plus LINKER:--exclude-libs,<basename>
#     (matches the basename even when linked by full path; composes with
#     --whole-archive); Mach-O → -Wl,-load_hidden,<archive> instead of the plain path
#     (there is no hidden variant of -force_load); PE/Wasm → the archive as is.
#     <out_libs> goes to target_link_libraries, <out_opts> to target_link_options.
#
# Inputs: target properties, BUILD_SHARED_LIBS, MSVC, CMAKE_<LANG>_COMPILER_ID and the
# platform module (included here explicitly). Requires CMake >= 3.18; include after
# project(); usable inside a package config.
# ==============================================================================
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/modules-version.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/platform.cmake")

# cmake_parse_arguments: an empty value after a keyword is a value, not an omission
# (CMP0174 NEW); function bodies record the policy state of their definition.
cmake_policy(PUSH)
if(POLICY CMP0174)
    cmake_policy(SET CMP0174 NEW)
endif()

# _tanh_target_kind(<target> <out>): STATIC | SHARED | EXECUTABLE, resolving OBJECT
# libraries through BUILD_SHARED_LIBS (they become part of whatever they are merged into).
function(_tanh_target_kind target out)
    get_target_property(_type ${target} TYPE)
    if(_type STREQUAL "STATIC_LIBRARY")
        set(${out} STATIC PARENT_SCOPE)
    elseif(_type MATCHES "^(SHARED_LIBRARY|MODULE_LIBRARY)$")
        set(${out} SHARED PARENT_SCOPE)
    elseif(_type STREQUAL "EXECUTABLE")
        set(${out} EXECUTABLE PARENT_SCOPE)
    elseif(_type STREQUAL "OBJECT_LIBRARY")
        if(BUILD_SHARED_LIBS)
            set(${out} SHARED PARENT_SCOPE)
        else()
            set(${out} STATIC PARENT_SCOPE)
        endif()
    else()
        message(FATAL_ERROR "tanh symbol policy: target '${target}' is of type ${_type}; expected a "
                            "static, shared, module or object library, or an executable")
    endif()
endfunction()

function(tanh_apply_symbol_policy target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "NO_GC_SECTIONS" "EXPORT_PREFIX" "")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tanh_apply_symbol_policy: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "tanh_apply_symbol_policy: '${target}' is not a target")
    endif()
    _tanh_target_kind(${target} _kind)
    get_target_property(_type ${target} TYPE)

    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        OBJC_VISIBILITY_PRESET hidden
        OBJCXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

    if(arg_EXPORT_PREFIX)
        target_compile_definitions(${target} PRIVATE ${arg_EXPORT_PREFIX}_BUILDING)
        if(_kind STREQUAL "STATIC")
            # PUBLIC: a consumer of the archive must see the same empty macro — dllimport
            # would look for __imp_ stubs an archive never provides, and default
            # visibility would leak the API from the consumer's export table.
            target_compile_definitions(${target} PUBLIC ${arg_EXPORT_PREFIX}_STATIC)
        endif()
    endif()

    if(TANH_BINARY_FORMAT STREQUAL "ELF")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fno-gnu-unique>)
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE /Gy)
        target_compile_options(${target} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4251>)
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:-ffunction-sections;-fdata-sections>)
    endif()

    if(NOT arg_NO_GC_SECTIONS AND NOT _type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY)$")
        if(TANH_BINARY_FORMAT STREQUAL "PE")
            if(MSVC)
                # link.exe defaults to /OPT:REF — except under /DEBUG, where it flips to
                # /OPT:NOREF unless given explicitly.
                target_link_options(${target} PRIVATE /OPT:REF)
            else()
                target_link_options(${target} PRIVATE LINKER:--gc-sections)
            endif()
        elseif(TANH_BINARY_FORMAT STREQUAL "Mach-O")
            target_link_options(${target} PRIVATE LINKER:-dead_strip)
        elseif(TANH_BINARY_FORMAT STREQUAL "ELF")
            target_link_options(${target} PRIVATE LINKER:--gc-sections)
        endif()
    endif()
endfunction()

# _tanh_itanium_namespace(<ns> <out>): "anira" -> "5anira", "a::b" -> "1a1b" — the
# <length><name> pieces of a nested-name in the Itanium mangling.
function(_tanh_itanium_namespace ns out)
    string(REPLACE "::" ";" _parts "${ns}")
    set(_m "")
    foreach(_p IN LISTS _parts)
        if(NOT _p MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
            message(FATAL_ERROR "tanh symbol policy: '${ns}' is not a C++ namespace name")
        endif()
        string(LENGTH "${_p}" _l)
        string(APPEND _m "${_l}${_p}")
    endforeach()
    set(${out} "${_m}" PARENT_SCOPE)
endfunction()

function(tanh_set_export_allowlist target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "NAMESPACE;SYMBOL")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "tanh_set_export_allowlist: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT arg_NAMESPACE AND NOT arg_SYMBOL)
        message(FATAL_ERROR "tanh_set_export_allowlist(${target}): give at least one NAMESPACE or SYMBOL")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "tanh_set_export_allowlist: '${target}' is not a target")
    endif()
    _tanh_target_kind(${target} _kind)
    get_target_property(_type ${target} TYPE)
    if(_type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY)$")
        return()  # no link step, no export table: the consumer's link decides
    endif()
    if(NOT TANH_BINARY_FORMAT MATCHES "^(ELF|Mach-O)$")
        return()  # PE: dllexport is the allowlist; Wasm: no dynamic export table
    endif()

    # The nine kinds of symbol a namespace contributes: functions and data, const
    # members, vtables, typeinfo, typeinfo names, VTTs, non-virtual and virtual thunks,
    # guard variables of function-local statics.
    set(_patterns "")
    foreach(_ns IN LISTS arg_NAMESPACE)
        _tanh_itanium_namespace("${_ns}" _m)
        list(APPEND _patterns
            "_ZN${_m}*" "_ZNK${_m}*" "_ZTVN${_m}*" "_ZTIN${_m}*" "_ZTSN${_m}*" "_ZTTN${_m}*"
            "_ZTh*_N${_m}*" "_ZTv*_N${_m}*" "_ZGVN${_m}*")
    endforeach()
    list(APPEND _patterns ${arg_SYMBOL})

    set(_dir "${CMAKE_CURRENT_BINARY_DIR}/tanh-exports")
    file(MAKE_DIRECTORY "${_dir}")
    if(TANH_BINARY_FORMAT STREQUAL "ELF")
        set(_file "${_dir}/${target}.map")
        set(_content "/* generated by tanh_set_export_allowlist(${target}) — do not edit */\n{\n  global:\n")
        foreach(_p IN LISTS _patterns)
            string(APPEND _content "    ${_p};\n")
        endforeach()
        string(APPEND _content "  local:\n    *;\n};\n")
        set(_opt "LINKER:--version-script=${_file}")
    else()
        set(_file "${_dir}/${target}.exports")
        set(_content "# generated by tanh_set_export_allowlist(${target}) — do not edit\n")
        foreach(_p IN LISTS _patterns)
            string(APPEND _content "_${_p}\n")
        endforeach()
        set(_opt "LINKER:-exported_symbols_list,${_file}")
    endif()
    # file(CONFIGURE) leaves the file untouched when the content is unchanged, so an
    # unchanged allowlist does not trigger a relink.
    file(CONFIGURE OUTPUT "${_file}" CONTENT "${_content}" @ONLY)
    target_link_options(${target} PRIVATE "${_opt}")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_file}")
endfunction()

function(tanh_hidden_archive_link_items archive out_libs out_opts)
    get_filename_component(_basename "${archive}" NAME)
    if(TANH_BINARY_FORMAT STREQUAL "ELF")
        set(${out_libs} "${archive}" PARENT_SCOPE)
        set(${out_opts} "LINKER:--exclude-libs,${_basename}" PARENT_SCOPE)
    elseif(TANH_BINARY_FORMAT STREQUAL "Mach-O")
        set(${out_libs} "-Wl,-load_hidden,${archive}" PARENT_SCOPE)
        set(${out_opts} "" PARENT_SCOPE)
    else()
        set(${out_libs} "${archive}" PARENT_SCOPE)
        set(${out_opts} "" PARENT_SCOPE)
    endif()
endfunction()

cmake_policy(POP)
