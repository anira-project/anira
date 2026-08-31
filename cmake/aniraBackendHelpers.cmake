# ==============================================================================
# aniraBackendHelpers.cmake — the anira::<engine> targets
# ==============================================================================
#
# One imported target per inference engine, named anira::<engine> with <engine> =
# onnxruntime | tflite | litert | libtorch | executorch, defined by the one function
# below both in anira's build tree (cmake/backends.cmake) and in the installed
# package (aniraBackendTargets.cmake, generated from aniraBackendTargets.cmake.in
# and shipped next to this file), so that both trees expose the same names with the
# same usage requirements.
#
# anira links these targets PRIVATE: what links anira::anira gets anira's headers
# and USE_* definitions, but no engine header and no engine on its link line beyond
# what a static anira has to carry as $<LINK_ONLY:...>. A consumer whose own code
# calls an engine links anira::<engine> explicitly and gets the very file anira
# uses — one copy of the engine per process, never two.
#
# Shared engines are SHARED IMPORTED targets (IMPORTED_LOCATION, plus IMPORTED_IMPLIB
# on Windows), so that $<TARGET_FILE:anira::onnxruntime> and $<TARGET_RUNTIME_DLLS:>
# work for consumers. Static engines are INTERFACE IMPORTED targets whose
# INTERFACE_LINK_LIBRARIES carry the archive — linked on demand and hidden:
#
#  * Never whole-archive. The onnxruntime/tflite/litert archives vendor overlapping
#    copies of protobuf/absl/onnx and contain several members defining the same
#    symbols, which only resolve on demand; force-loading them yields thousands of
#    duplicate-symbol errors. anira drives the engines through their C API, which
#    the linker resolves on demand.
#  * Hidden, so that none of the archive's symbols becomes a dynamic export of the
#    module that links it. A host may ship its own copy of an engine (Ableton Live
#    12 bundles an ONNX Runtime dylib); any engine symbol exported from a plugin can
#    then be interposed (ELF) or weak-coalesced (Mach-O) against the host's copy,
#    which crashes the host on the first API call when the versions differ. Mach-O
#    links the archive as -Wl,-load_hidden,<archive> (ld64, Xcode >= 14: linked
#    exactly like a plain input, but every symbol it contributes is private_extern) —
#    which is why the static targets are INTERFACE rather than STATIC IMPORTED: the
#    archive must appear on the link line once, in that form, never additionally as
#    a plain path. ELF localizes the archive's symbols with --exclude-libs on its
#    basename (INTERFACE_LINK_OPTIONS, so it reaches whoever links the archive).
#    PE/COFF exports nothing without dllexport, so MSVC needs nothing; Emscripten
#    has no dynamic export table.
#
# The INTERFACE kind is for engines that come with their own CMake package
# (LibTorch via find_package(Torch)):
# anira::<engine> then wraps the package's targets through LINK_LIBRARIES and adds
# the include directories and definitions the package does not carry correctly.
# ==============================================================================

include_guard(GLOBAL)

# ------------------------------------------------------------------------------
# anira_define_backend_target(<engine> SHARED|STATIC|INTERFACE
#     [GLOBAL]                     make the imported target visible in every directory
#     [LOCATION <file>]            SHARED: the .so/.dylib/.dll (may be empty on Windows
#                                  when only the import library is at hand);
#                                  STATIC: the archive
#     [IMPLIB <file>]              SHARED on Windows: the import library
#     [INCLUDE_DIRS <dir>...]      the engine's headers (SYSTEM for consumers)
#     [DEFINITIONS <def>...]       INTERFACE_COMPILE_DEFINITIONS
#     [COMPILE_OPTIONS <opt>...]   INTERFACE_COMPILE_OPTIONS
#     [LINK_LIBRARIES <item>...]   further INTERFACE_LINK_LIBRARIES (targets, libs, flags)
#     [LINK_OPTIONS <opt>...]      further INTERFACE_LINK_OPTIONS
# )
# A no-op when anira::<engine> already exists (anira and a consumer in another
# directory may both define it).
# ------------------------------------------------------------------------------
function(anira_define_backend_target engine kind)
    # PARSE_ARGV keeps empty arguments (LOCATION "" on Windows when only the import
    # library is known); the plain form would drop them and mis-pair the keywords.
    cmake_parse_arguments(PARSE_ARGV 2 _abt
        "GLOBAL"
        "LOCATION;IMPLIB"
        "INCLUDE_DIRS;DEFINITIONS;COMPILE_OPTIONS;LINK_LIBRARIES;LINK_OPTIONS")
    if(_abt_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "anira_define_backend_target(${engine}): unexpected arguments "
                            "'${_abt_UNPARSED_ARGUMENTS}'")
    endif()

    set(_target "anira::${engine}")
    if(TARGET ${_target})
        return()
    endif()
    set(_global "")
    if(_abt_GLOBAL)
        set(_global GLOBAL)
    endif()

    set(_link_libs "")
    set(_link_opts "")
    if(kind STREQUAL "SHARED")
        add_library(${_target} SHARED IMPORTED ${_global})
        if(NOT _abt_LOCATION STREQUAL "")
            set_target_properties(${_target} PROPERTIES IMPORTED_LOCATION "${_abt_LOCATION}")
        endif()
        if(NOT _abt_IMPLIB STREQUAL "")
            set_target_properties(${_target} PROPERTIES IMPORTED_IMPLIB "${_abt_IMPLIB}")
        endif()
        if(_abt_LOCATION STREQUAL "" AND _abt_IMPLIB STREQUAL "")
            message(FATAL_ERROR "anira_define_backend_target(${engine} SHARED): LOCATION or IMPLIB required")
        endif()
    elseif(kind STREQUAL "STATIC")
        if(_abt_LOCATION STREQUAL "")
            message(FATAL_ERROR "anira_define_backend_target(${engine} STATIC): LOCATION required")
        endif()
        add_library(${_target} INTERFACE IMPORTED ${_global})
        _anira_static_archive_link_items("${_abt_LOCATION}" _link_libs _link_opts)
    elseif(kind STREQUAL "INTERFACE")
        add_library(${_target} INTERFACE IMPORTED ${_global})
    else()
        message(FATAL_ERROR "anira_define_backend_target(${engine}): kind must be SHARED, STATIC or INTERFACE (got '${kind}')")
    endif()

    list(APPEND _link_libs ${_abt_LINK_LIBRARIES})
    list(APPEND _link_opts ${_abt_LINK_OPTIONS})
    if(_link_libs)
        set_property(TARGET ${_target} PROPERTY INTERFACE_LINK_LIBRARIES "${_link_libs}")
    endif()
    if(_link_opts)
        set_property(TARGET ${_target} PROPERTY INTERFACE_LINK_OPTIONS "${_link_opts}")
    endif()
    if(_abt_INCLUDE_DIRS)
        set_target_properties(${_target} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_abt_INCLUDE_DIRS}"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_abt_INCLUDE_DIRS}")
    endif()
    if(_abt_DEFINITIONS)
        set_property(TARGET ${_target} PROPERTY INTERFACE_COMPILE_DEFINITIONS "${_abt_DEFINITIONS}")
    endif()
    if(_abt_COMPILE_OPTIONS)
        set_property(TARGET ${_target} PROPERTY INTERFACE_COMPILE_OPTIONS "${_abt_COMPILE_OPTIONS}")
    endif()
endfunction()

# ------------------------------------------------------------------------------
# _anira_static_archive_link_items(<archive> <out-libs> <out-opts>) — how a prebuilt
# static engine archive goes onto a link line on this platform (see the header
# comment): the archive itself, linked on demand and hidden, plus the system
# libraries it depends on.
# ------------------------------------------------------------------------------
function(_anira_static_archive_link_items archive out_libs out_opts)
    # The format decides how the archive is hidden (ELF: --exclude-libs on the basename,
    # Mach-O: -load_hidden, PE/Wasm: nothing to hide) — cmake/tanh/symbol-policy.cmake.
    tanh_hidden_archive_link_items("${archive}" _libs _opts)
    # The OS decides which system libraries the archive needs on top.
    if(TANH_BINARY_FORMAT STREQUAL "Mach-O")
        # macOS and iOS alike: static onnxruntime/tflite/litert pull in absl/CoreFoundation
        # time-zone and Apple logging code (Foundation/CoreFoundation), and static LiteRT
        # references Metal (LiteRtCreateMetalInfo -> MTLCreateSystemDefaultDevice).
        list(APPEND _libs "-framework Foundation" "-framework CoreFoundation" "-framework Metal")
    elseif(TANH_OPERATING_SYSTEM STREQUAL "Android")
        # Android's bionic folds pthread/dl/libm into libc, but the static LiteRT/TFLite
        # archives vendor the GPU (GL ES) delegate and use Android logging, whose symbols
        # (glClear, EGL*, __android_log_*) live in NDK system libs that must be linked.
        list(APPEND _libs EGL GLESv2 android log)
    elseif(TANH_BINARY_FORMAT STREQUAL "ELF")
        find_package(Threads REQUIRED)
        list(APPEND _libs Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
    set(${out_libs} "${_libs}" PARENT_SCOPE)
    set(${out_opts} "${_opts}" PARENT_SCOPE)
endfunction()


# ------------------------------------------------------------------------------
# anira_relocate_include_dirs(<from> <to> <target>...) — rewrite every interface
# include directory of the given imported targets that lies under <from> to the
# same path under <to>. For packages whose config hardwires its headers to
# <prefix>/include (TorchConfig.cmake) while anira installs them per engine under
# <prefix>/include/anira-backends/<engine>.
# ------------------------------------------------------------------------------
function(anira_relocate_include_dirs from to)
    get_filename_component(from "${from}" ABSOLUTE)
    foreach(_tgt IN LISTS ARGN)
        foreach(_prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            get_target_property(_dirs "${_tgt}" ${_prop})
            if(NOT _dirs)
                continue()
            endif()
            set(_rewritten "")
            foreach(_dir IN LISTS _dirs)
                get_filename_component(_abs "${_dir}" ABSOLUTE)
                if(_abs STREQUAL from)
                    list(APPEND _rewritten "${to}")
                elseif(_abs MATCHES "^${from}/(.*)$")
                    list(APPEND _rewritten "${to}/${CMAKE_MATCH_1}")
                else()
                    list(APPEND _rewritten "${_dir}")
                endif()
            endforeach()
            set_target_properties("${_tgt}" PROPERTIES ${_prop} "${_rewritten}")
        endforeach()
    endforeach()
endfunction()

