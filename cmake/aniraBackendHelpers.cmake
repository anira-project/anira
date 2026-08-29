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
# (LibTorch via find_package(Torch), desktop ExecuTorch via find_package(executorch)):
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
    set(_libs "")
    set(_opts "")
    get_filename_component(_basename "${archive}" NAME)
    if(EMSCRIPTEN OR EMSDK_VERSION)
        list(APPEND _libs "${archive}")
    elseif(WIN32)
        # PE/COFF exports nothing without __declspec(dllexport): no hiding needed.
        list(APPEND _libs "${archive}")
    elseif(APPLE)
        # Static onnxruntime/tflite/litert pull in absl/CoreFoundation time-zone and
        # Apple logging code (Foundation/CoreFoundation), and static LiteRT references
        # Metal (LiteRtCreateMetalInfo -> MTLCreateSystemDefaultDevice).
        list(APPEND _libs "-Wl,-load_hidden,${archive}"
            "-framework Foundation" "-framework CoreFoundation" "-framework Metal")
    elseif(ANDROID)
        # Android's bionic folds pthread/dl/libm into libc, but the static LiteRT/TFLite
        # archives vendor the GPU (GL ES) delegate and use Android logging, whose symbols
        # (glClear, EGL*, __android_log_*) live in NDK system libs that must be linked.
        list(APPEND _opts "LINKER:--exclude-libs,${_basename}")
        list(APPEND _libs "${archive}" EGL GLESv2 android log)
    else() # Linux / other ELF
        find_package(Threads REQUIRED)
        list(APPEND _opts "LINKER:--exclude-libs,${_basename}")
        list(APPEND _libs "${archive}" Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
    set(${out_libs} "${_libs}" PARENT_SCOPE)
    set(${out_opts} "${_opts}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# anira_sanitize_executorch_targets(<target>...) — clean up the targets the
# ExecuTorch CMake package imported, for use inside anira and by its consumers:
#
#  * They bake absolute system-library paths from the machine that built the
#    archives into their INTERFACE_LINK_LIBRARIES (e.g.
#    .../MacOSX15.5.sdk/usr/lib/libm.tbd, .../Frameworks/Foundation.framework,
#    or Debian's multiarch /usr/lib/aarch64-linux-gnu/libm.so). Those paths
#    rarely exist on the consuming machine (Fedora keeps libm in /usr/lib64), so
#    rewrite them into portable equivalents the consumer's toolchain resolves:
#    <sdk>/lib<name>.tbd -> <name>, /usr/lib[/<multiarch>]/lib<name>.so -> <name>,
#    and <path>/<Name>.framework -> -framework <Name>.
#
#  * They carry compile usage requirements (include dirs — among them ExecuTorch's
#    VENDORED c10 headers — compile definitions and options) that would leak into
#    every TU of a target linking them, and whose include paths do not even survive
#    anira's install layout. Strip all compile-side usage requirements; the compile
#    definitions the headers DO need are collected first and returned in <out-defs>,
#    and anira::executorch carries them together with the right include dirs.
# ------------------------------------------------------------------------------
function(anira_sanitize_executorch_targets out_defs)
    set(_defs "")
    foreach(_tgt IN LISTS ARGN)
        get_target_property(_tgt_defs "${_tgt}" INTERFACE_COMPILE_DEFINITIONS)
        if(_tgt_defs)
            list(APPEND _defs ${_tgt_defs})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _defs)
    set(${out_defs} "${_defs}" PARENT_SCOPE)
    foreach(_tgt IN LISTS ARGN)
        set_target_properties("${_tgt}" PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ""
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ""
            INTERFACE_COMPILE_OPTIONS ""
            INTERFACE_COMPILE_DEFINITIONS "")
        get_target_property(_libs "${_tgt}" INTERFACE_LINK_LIBRARIES)
        if(NOT _libs)
            continue()
        endif()
        set(_rewritten "")
        set(_changed FALSE)
        foreach(_lib IN LISTS _libs)
            if(_lib MATCHES "^/.*/([A-Za-z0-9_]+)\\.framework$")
                list(APPEND _rewritten "-framework ${CMAKE_MATCH_1}")
                set(_changed TRUE)
            elseif(_lib MATCHES "^/.*/lib([A-Za-z0-9_.+-]+)\\.tbd$")
                list(APPEND _rewritten "${CMAKE_MATCH_1}")
                set(_changed TRUE)
            elseif(_lib MATCHES "^/usr/lib(64)?(/[A-Za-z0-9_-]+)?/lib([A-Za-z0-9_+-]+)\\.so(\\.[0-9.]+)?$")
                # Linux system library referenced by absolute path from the build
                # machine (e.g. Debian multiarch /usr/lib/aarch64-linux-gnu/libm.so).
                # Keep only the library name so the consumer's toolchain resolves it
                # (-lm) and no bogus -L / rpath entries leak into the link line.
                list(APPEND _rewritten "${CMAKE_MATCH_3}")
                set(_changed TRUE)
            else()
                list(APPEND _rewritten "${_lib}")
            endif()
        endforeach()
        if(_changed)
            set_target_properties("${_tgt}" PROPERTIES INTERFACE_LINK_LIBRARIES "${_rewritten}")
        endif()
    endforeach()
endfunction()

# ------------------------------------------------------------------------------
# anira_find_executorch_package(<path> <out-targets> <out-archives>) — find_package
# the ExecuTorch CMake package at <path>, return the imported targets it defined
# (sanitized, see above) and the basenames of the static archives among them — the
# ELF --exclude-libs list of anira::executorch. Promotes the targets to GLOBAL:
# imported targets are directory-scoped, and a consumer of a static anira in another
# directory resolves $<LINK_ONLY:anira::executorch> -> executorch ... at its own link.
# ------------------------------------------------------------------------------
function(anira_find_executorch_package path out_targets out_archives out_defs)
    get_property(_before DIRECTORY PROPERTY IMPORTED_TARGETS)
    find_package(executorch REQUIRED CONFIG PATHS "${path}" NO_DEFAULT_PATH)
    get_property(_after DIRECTORY PROPERTY IMPORTED_TARGETS)
    if(_before)
        list(REMOVE_ITEM _after ${_before})
    endif()
    set(_archives "")
    foreach(_tgt IN LISTS _after)
        set_target_properties(${_tgt} PROPERTIES IMPORTED_GLOBAL TRUE)
        get_target_property(_type ${_tgt} TYPE)
        if(_type STREQUAL "STATIC_LIBRARY")
            get_target_property(_configs ${_tgt} IMPORTED_CONFIGURATIONS)
            set(_props IMPORTED_LOCATION)
            foreach(_cfg IN LISTS _configs)
                list(APPEND _props "IMPORTED_LOCATION_${_cfg}")
            endforeach()
            foreach(_prop IN LISTS _props)
                get_target_property(_loc ${_tgt} ${_prop})
                if(_loc)
                    get_filename_component(_loc "${_loc}" NAME)
                    list(APPEND _archives "${_loc}")
                endif()
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _archives)
    anira_sanitize_executorch_targets(_defs ${_after})
    set(${out_targets} "${_after}" PARENT_SCOPE)
    set(${out_archives} "${_archives}" PARENT_SCOPE)
    set(${out_defs} "${_defs}" PARENT_SCOPE)
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

# ------------------------------------------------------------------------------
# anira_define_executorch_target(<incdir> <archives> <defs> [GLOBAL]) — the desktop
# anira::executorch: the runtime plus the extension (Module/TensorPtr/data loader),
# kernel (optimized + quantized CPU ops) and XNNPACK-delegate targets of the package
# found by anira_find_executorch_package — the ops/backend targets carry the
# per-platform force-load options their static-initializer registration requires.
# The CoreML/MPS/MLX delegates are deliberately not linked: anira pins its backends
# to portable CPU execution, and their exported targets reference absolute SDK
# paths from the machine that built the archives.
#
# XNNPACK's runtime-dispatch config tables reference its microkernel symbols from
# data sections, which on-demand archive resolution does not satisfy (Apple's
# linker fails with "does not have address" fixup errors on archive members it
# never materialized). The microkernel archive is a leaf with no dependencies or
# colliding symbols, so it is linked whole-archive.
#
# The desktop archives are compiled with default visibility and (partly)
# force-loaded, so every symbol they contribute would become a dynamic export of
# whatever links them (executorch::, xnn_*, pthreadpool_*, cpuinfo_*, kai_*, the
# vendored c10:: and Eigen/BLAS). A host that ships its own XNNPACK (LibTorch does)
# could then interpose the delegate's kernels. ELF localizes them with
# --exclude-libs per archive basename, which composes with --whole-archive
# (basenames only, so the installed export stays relocatable). Mach-O has no hidden
# variant of -force_load: there a plugin embedding a static anira relies on its own
# -exported_symbols_list (docs/sphinx/troubleshooting.rst).
# ------------------------------------------------------------------------------
function(anira_define_executorch_target incdir archives defs)
    set(_libs executorch executorch_extensions executorch_kernels xnnpack_backend)
    if(TARGET xnnpack-microkernels-prod)
        list(APPEND _libs "$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack-microkernels-prod>")
    endif()
    set(_opts "")
    if(NOT APPLE AND NOT WIN32 AND NOT EMSCRIPTEN AND NOT EMSDK_VERSION)
        foreach(_archive IN LISTS archives)
            list(APPEND _opts "LINKER:--exclude-libs,${_archive}")
        endforeach()
    endif()
    anira_define_backend_target(executorch INTERFACE ${ARGN}
        LINK_LIBRARIES ${_libs}
        LINK_OPTIONS ${_opts}
        INCLUDE_DIRS "${incdir}" "${incdir}/executorch/runtime/core/portable_type/c10"
        DEFINITIONS ${defs})
endfunction()
