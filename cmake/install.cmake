# ==============================================================================
# Install the library
# ==============================================================================

# for CMAKE_INSTALL_INCLUDEDIR and others definition
include(GNUInstallDirs)

# include the public headers of the anira library for the install target
# TODO: File PR so that the concurrentqueue library does not have to be included that way
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/concurrentqueue/moodycamel>
)

# define the dircetory where the library will be installed CMAKE_INSTALL_PREFIX
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message( STATUS "CMAKE_INSTALL_PREFIX will be set to ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-${PROJECT_VERSION}" )
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-${PROJECT_VERSION}" CACHE PATH "Where the library will be installed to" FORCE)
else()
    message(STATUS "CMAKE_INSTALL_PREFIX was already set to ${CMAKE_INSTALL_PREFIX}")
endif()

# The install-time RPATH ($ORIGIN / @loader_path, cmake/tanh/install-helpers.cmake) so
# the installed shared library finds its siblings; the MKL path for Intel macs.
include(${CMAKE_CURRENT_LIST_DIR}/tanh/install-helpers.cmake)
set(_anira_apple_rpaths "")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_anira_apple_rpaths EXTRA_APPLE_PATHS "/opt/intel/oneapi/mkl/latest/lib")
endif()
tanh_set_install_rpath(${PROJECT_NAME} ${_anira_apple_rpaths})
if(ANIRA_WITH_BENCHMARK)
    # gtest_main must find gtest at runtime from the install tree.
    tanh_set_install_rpath(gtest_main)
endif()
unset(_anira_apple_rpaths)

# the variant with PUBLIC_HEADER property unfortunately does not preserve the folder structure therefore we use the simple install directory command
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/anira
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT dev
)

# install the target and create export-set
# GNUInstallDirs layout: shared libraries into the libdir; on Windows the DLL is a
# RUNTIME artifact and goes to the bindir next to the executables (where the loader
# looks), the import library into the libdir. tanh_core.dll and the engine DLLs
# (below) follow the same rule, so one directory — bin/ — holds every DLL.
install(TARGETS ${PROJECT_NAME} concurrentqueue nlohmann_json
    EXPORT "aniraTargets"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll files
    COMPONENT runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so or .dylib files
    COMPONENT runtime NAMELINK_COMPONENT dev
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .a or .lib (import library) files
    COMPONENT dev
)

# ==============================================================================
# The backends. Libraries go into the install libdir as they are (libanira's
# INSTALL_RPATH $ORIGIN and the consumers' runtime search depend on that layout);
# headers go per engine into <includedir>/anira-backends/<engine>/; and the
# anira::<engine> targets are defined again from the install prefix by
# aniraBackendTargets.cmake (generated below, included by aniraConfig.cmake before
# aniraTargets.cmake, so that the $<LINK_ONLY:anira::<engine>> entries of a static
# anira resolve). ANIRA_<ID>_ROOTDIR & co. are set by anira_setup_backend().
#
# The per-engine header directories are what makes "you need anira::<engine>" true
# at compile time in the installed tree: only that target carries the directory, so
# a consumer with anira::anira alone cannot include an engine header by accident.
# ==============================================================================
set(_anira_backend_incdir "${CMAKE_INSTALL_INCLUDEDIR}/anira-backends")

# ------------------------------------------------------------------------------
# _anira_install_engine_libs(<rootdir> [EXCLUDE_CMAKE]) — an engine's lib/ tree into
# the libdir. On Windows the DLLs go to the bindir instead — import libraries and
# static archives stay in lib/ — like anira.dll and tanh_core.dll, so that the
# loader finds every DLL next to the executables and a consumer needs one
# directory on PATH, not two. (A few archives keep their DLLs in a bin/ of their
# own; those go to the bindir as well.) EXCLUDE_CMAKE leaves out a lib/cmake
# package, which _anira_install_cmake_package installs separately.
# ------------------------------------------------------------------------------
function(_anira_install_engine_libs rootdir)
    cmake_parse_arguments(PARSE_ARGV 1 _ael "EXCLUDE_CMAKE" "" "")
    set(_exclude "")
    if(_ael_EXCLUDE_CMAKE)
        set(_exclude PATTERN "cmake" EXCLUDE)
    endif()
    if(TANH_BINARY_FORMAT STREQUAL "PE")
        install(DIRECTORY "${rootdir}/lib/" DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT deps-backends ${_exclude} PATTERN "*.dll" EXCLUDE)
        install(DIRECTORY "${rootdir}/lib/" DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT deps-backends FILES_MATCHING ${_exclude} PATTERN "*.dll")
        if(IS_DIRECTORY "${rootdir}/bin")
            install(DIRECTORY "${rootdir}/bin/" DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT deps-backends FILES_MATCHING PATTERN "*.dll")
        endif()
    else()
        install(DIRECTORY "${rootdir}/lib/" DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT deps-backends ${_exclude})
    endif()
endfunction()

# ------------------------------------------------------------------------------
# _anira_install_cmake_package(<id> <src> <dest>) — install an engine's own CMake
# package directory (LibTorch's share/cmake, ExecuTorch's lib/cmake) to <dest>
# under the prefix. Those packages locate their libraries relative to their own
# file as <prefix>/lib/... (Caffe2Targets/ExecuTorchTargets: ${_IMPORT_PREFIX}/lib/,
# TorchConfig: ${TORCH_INSTALL_PREFIX}/lib), which is only right when everything
# is in "lib". Two layouts differ, and anira installs patched copies of the
# package files for them instead of bending its own layout to the packages:
#  * hosts where GNUInstallDirs picks lib64 (Fedora): the engine libraries go to
#    lib64 like everything else — a second library directory would break
#    libanira's $ORIGIN rpath and every consumer's runtime search — so "lib"
#    becomes the libdir in the package files;
#  * Windows: the DLLs go to the bindir (_anira_install_engine_libs), so a DLL's
#    IMPORTED_LOCATION becomes ${_IMPORT_PREFIX}/bin/<name>.dll; import libraries
#    stay in lib/.
# The patch is a literal replacement of those spellings and a plain directory
# install where neither applies.
# ------------------------------------------------------------------------------
function(_anira_install_cmake_package id src dest)
    if(NOT TANH_BINARY_FORMAT STREQUAL "PE" AND CMAKE_INSTALL_LIBDIR STREQUAL "lib")
        install(DIRECTORY "${src}/" DESTINATION "${dest}" COMPONENT deps-backends)
        return()
    endif()
    set(_staged "${CMAKE_CURRENT_BINARY_DIR}/anira-backends-cmake/${id}")
    file(REMOVE_RECURSE "${_staged}")
    file(GLOB_RECURSE _files RELATIVE "${src}" "${src}/*")
    foreach(_rel IN LISTS _files)
        if(_rel MATCHES "\\.cmake$")
            file(READ "${src}/${_rel}" _content)
            if(TANH_BINARY_FORMAT STREQUAL "PE")
                string(REGEX REPLACE "\\$\\{_IMPORT_PREFIX\\}/lib/([^\"]*\\.dll\")"
                    "\${_IMPORT_PREFIX}/${CMAKE_INSTALL_BINDIR}/\\1" _content "${_content}")
            endif()
            string(REPLACE "\${_IMPORT_PREFIX}/lib/" "\${_IMPORT_PREFIX}/${CMAKE_INSTALL_LIBDIR}/"
                _content "${_content}")
            string(REPLACE "\${TORCH_INSTALL_PREFIX}/lib\"" "\${TORCH_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\""
                _content "${_content}")
            file(WRITE "${_staged}/${_rel}" "${_content}")
        else()
            configure_file("${src}/${_rel}" "${_staged}/${_rel}" COPYONLY)
        endif()
    endforeach()
    install(DIRECTORY "${_staged}/" DESTINATION "${dest}" COMPONENT deps-backends)
endfunction()

if(ANIRA_WITH_LIBTORCH)
    # LibTorch ships its own CMake package (share/cmake/Torch + Caffe2), which the
    # installed aniraBackendTargets.cmake re-resolves and re-points at the relocated
    # headers (TorchConfig.cmake hardwires them to <prefix>/include).
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/include/"
        DESTINATION "${_anira_backend_incdir}/libtorch" COMPONENT deps-backends)
    _anira_install_engine_libs("${LIBTORCH_ROOTDIR}")
    _anira_install_cmake_package(libtorch "${LIBTORCH_ROOTDIR}/share/cmake" "${CMAKE_INSTALL_LIBDIR}/cmake")
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    if(TANH_OPERATING_SYSTEM STREQUAL "iOS")
        # iOS ships an xcframework: install it whole (the static .a then sits at the
        # SUBPATH anira::onnxruntime expects) plus the active slice's headers.
        install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/onnxruntime.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/onnxruntime.xcframework/${ANIRA_ONNXRUNTIME_IOS_SLICE}/Headers/"
            DESTINATION "${_anira_backend_incdir}/onnxruntime" COMPONENT deps-backends)
    else()
        if(TANH_OPERATING_SYSTEM STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
            install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/include/onnxruntime/"
                DESTINATION "${_anira_backend_incdir}/onnxruntime" COMPONENT deps-backends)
        else()
            install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/include/"
                DESTINATION "${_anira_backend_incdir}/onnxruntime" COMPONENT deps-backends)
        endif()
        _anira_install_engine_libs("${ANIRA_ONNXRUNTIME_ROOTDIR}")
    endif()
endif()

if(ANIRA_WITH_TFLITE)
    if(TANH_OPERATING_SYSTEM STREQUAL "iOS")
        # iOS TFLite is a TensorFlowLiteC.framework xcframework: install it whole, plus
        # the framework's flat headers AND the generated <tensorflow/lite/...> shim that
        # forwards onto them (so the canonical include paths resolve for a consumer).
        install(DIRECTORY "${ANIRA_TFLITE_ROOTDIR}/TensorFlowLiteC.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_TFLITE_ROOTDIR}/TensorFlowLiteC.xcframework/${ANIRA_TFLITE_IOS_SLICE}/TensorFlowLiteC.framework/Headers/"
            DESTINATION "${_anira_backend_incdir}/tflite" COMPONENT deps-backends)
        if(NOT ANIRA_TFLITE_IOS_SHIM)
            message(FATAL_ERROR "ANIRA_TFLITE_IOS_SHIM is empty — refusing to install (would copy the filesystem root).")
        endif()
        install(DIRECTORY "${ANIRA_TFLITE_IOS_SHIM}/"
            DESTINATION "${_anira_backend_incdir}/tflite" COMPONENT deps-backends)
    else()
        install(DIRECTORY "${ANIRA_TFLITE_ROOTDIR}/include/"
            DESTINATION "${_anira_backend_incdir}/tflite" COMPONENT deps-backends)
        _anira_install_engine_libs("${ANIRA_TFLITE_ROOTDIR}")
    endif()
endif()

if(ANIRA_WITH_LITERT)
    if(TANH_OPERATING_SYSTEM STREQUAL "iOS")
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/LiteRt.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/LiteRt.xcframework/${ANIRA_LITERT_IOS_SLICE}/Headers/"
            DESTINATION "${_anira_backend_incdir}/litert" COMPONENT deps-backends)
    else()
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/include/"
            DESTINATION "${_anira_backend_incdir}/litert" COMPONENT deps-backends)
        _anira_install_engine_libs("${ANIRA_LITERT_ROOTDIR}")
    endif()
endif()

if(ANIRA_WITH_EXECUTORCH)
    if(TANH_OPERATING_SYSTEM STREQUAL "iOS")
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/executorch.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/executorch.xcframework/${ANIRA_EXECUTORCH_IOS_SLICE}/Headers/"
            DESTINATION "${_anira_backend_incdir}/executorch" COMPONENT deps-backends)
    else()
        # On desktop the lib/ tree carries ExecuTorch's CMake package (lib/cmake/
        # ExecuTorch, plus the KleidiAI one it depends on), which the installed
        # aniraBackendTargets.cmake re-resolves via find_package(executorch) —
        # analogous to LibTorch above, and installed through the same libdir patch.
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/include/"
            DESTINATION "${_anira_backend_incdir}/executorch" COMPONENT deps-backends)
        _anira_install_engine_libs("${ANIRA_EXECUTORCH_ROOTDIR}" EXCLUDE_CMAKE)
        _anira_install_cmake_package(executorch "${ANIRA_EXECUTORCH_ROOTDIR}/lib/cmake" "${CMAKE_INSTALL_LIBDIR}/cmake")
    endif()
endif()

# ------------------------------------------------------------------------------
# aniraBackendTargets.cmake — the anira::<engine> definitions of this install,
# generated from cmake/aniraBackendTargets.cmake.in with one
# anira_define_backend_target() call per enabled engine. Every path is relative to
# the package prefix (PACKAGE_PREFIX_DIR, set by aniraConfig.cmake), so the tree is
# relocatable; the file names are exactly those install(DIRECTORY ...) copies.
# ------------------------------------------------------------------------------
set(_anira_installed_targets "")
foreach(_engine onnxruntime tflite litert)
    string(TOUPPER "${_engine}" _ENGINE)
    if(NOT ANIRA_WITH_${_ENGINE})
        continue()
    endif()
    if(ANIRA_${_ENGINE}_IS_STATIC)
        set(_defs "")
        if(_engine STREQUAL "tflite")
            set(_defs "\n    DEFINITIONS TFL_COMPILE_LIBRARY")
        endif()
        string(APPEND _anira_installed_targets
            "anira_define_backend_target(${_engine} STATIC\n"
            "    LOCATION \"\${_anira_libdir}/${ANIRA_${_ENGINE}_STATIC_LIB_SUBPATH}\"\n"
            "    INCLUDE_DIRS \"\${_anira_incdir}/${_engine}\"${_defs})\n")
    else()
        # The shared library sits in the libdir — on Windows, where it is a DLL, in the
        # bindir (_anira_install_engine_libs); the import library stays in the libdir.
        if(TANH_BINARY_FORMAT STREQUAL "PE")
            set(_location_dir "\${_anira_bindir}")
        else()
            set(_location_dir "\${_anira_libdir}")
        endif()
        set(_location "")
        if(ANIRA_${_ENGINE}_SHARED_LIB_SUBPATH)
            set(_location "\n    LOCATION \"${_location_dir}/${ANIRA_${_ENGINE}_SHARED_LIB_SUBPATH}\"")
        endif()
        set(_implib "")
        if(ANIRA_${_ENGINE}_IMPLIB_SUBPATH)
            set(_implib "\n    IMPLIB \"\${_anira_libdir}/${ANIRA_${_ENGINE}_IMPLIB_SUBPATH}\"")
        endif()
        string(APPEND _anira_installed_targets
            "anira_define_backend_target(${_engine} SHARED${_location}${_implib}\n"
            "    INCLUDE_DIRS \"\${_anira_incdir}/${_engine}\")\n")
    endif()
endforeach()
unset(_engine)
unset(_ENGINE)
unset(_defs)
unset(_location)
unset(_location_dir)
unset(_implib)

if(ANIRA_WITH_LIBTORCH)
    string(APPEND _anira_installed_targets [=[
# LibTorch: its own CMake package, installed into lib/cmake/{Torch,Caffe2}. Its
# TorchConfig.cmake hardwires the package's include directories to
# <prefix>/include (+ torch/csrc/api/include), while the headers live under
# <includedir>/anira-backends/libtorch here — so every target the package imported
# (torch, torch_library, c10, ...) is re-pointed at the relocated directories.
get_property(_anira_imported_before DIRECTORY PROPERTY IMPORTED_TARGETS)
find_package(Torch REQUIRED CONFIG HINTS "${_anira_libdir}/cmake/Torch")
get_property(_anira_imported_after DIRECTORY PROPERTY IMPORTED_TARGETS)
if(_anira_imported_before)
    list(REMOVE_ITEM _anira_imported_after ${_anira_imported_before})
endif()
anira_relocate_include_dirs("${PACKAGE_PREFIX_DIR}/@CMAKE_INSTALL_INCLUDEDIR@"
    "${_anira_incdir}/libtorch" ${_anira_imported_after})
set(TORCH_INCLUDE_DIRS
    "${_anira_incdir}/libtorch"
    "${_anira_incdir}/libtorch/torch/csrc/api/include")
set(_anira_torch_libs torch torch_library ${TORCH_LIBRARIES})
list(REMOVE_DUPLICATES _anira_torch_libs)
anira_define_backend_target(libtorch INTERFACE
    LINK_LIBRARIES ${_anira_torch_libs}
    INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
unset(_anira_torch_libs)
unset(_anira_imported_before)
unset(_anira_imported_after)
]=])
endif()

if(ANIRA_WITH_EXECUTORCH)
    if(TANH_OPERATING_SYSTEM STREQUAL "Android" OR TANH_OPERATING_SYSTEM STREQUAL "iOS")
        string(APPEND _anira_installed_targets
            "anira_define_backend_target(executorch STATIC\n"
            "    LOCATION \"\${_anira_libdir}/${ANIRA_EXECUTORCH_STATIC_LIB_SUBPATH}\"\n"
            "    INCLUDE_DIRS \"\${_anira_incdir}/executorch\")\n")
    else()
        string(APPEND _anira_installed_targets [=[
# ExecuTorch: its own CMake package, installed into lib/cmake/ExecuTorch. Imported,
# sanitized and wrapped exactly as in anira's build tree (aniraBackendHelpers.cmake).
anira_find_executorch_package("${_anira_libdir}/cmake/ExecuTorch"
    _anira_executorch_targets _anira_executorch_archives _anira_executorch_defs)
anira_define_executorch_target("${_anira_incdir}/executorch"
    "${_anira_executorch_archives}" "${_anira_executorch_defs}")
unset(_anira_executorch_targets)
unset(_anira_executorch_archives)
unset(_anira_executorch_defs)
]=])
    endif()
endif()

set(ANIRA_INSTALLED_BACKEND_TARGETS "${_anira_installed_targets}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/aniraBackendTargets.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/aniraBackendTargets.cmake" @ONLY)
unset(_anira_installed_targets)
unset(_anira_backend_incdir)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/aniraBackendHelpers.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aniraBackendTargets.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    COMPONENT dev
)
# The shared modules aniraBackendHelpers.cmake needs at a consumer's configure time
# (platform axes, tanh_hidden_archive_link_items) — the same verbatim files as in the
# build tree, included from aniraConfig.cmake before the helpers.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tanh/modules-version.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tanh/platform.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tanh/symbol-policy.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/tanh
    COMPONENT dev
)

# ==============================================================================
# Generate cmake config files
# ==============================================================================

# generate and install export file in the folder cmake with the name of the project and namespace
# this generates files called aniraTargets.cmake, aniraTargets-debug.cmake, aniraTargets-release.cmake
install(EXPORT "aniraTargets"
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    COMPONENT dev
)

include(CMakePackageConfigHelpers)

# create config file from the template file Config.cmake.in and specify the install destination
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

# install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/aniraConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    COMPONENT dev
)