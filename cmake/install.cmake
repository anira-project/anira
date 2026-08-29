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

# at install the rpath is cleared by default so we have to set it again for the installed shared library to find the other libraries
# in this case we set the rpath to the directories where the other libraries are installed
# $ORIGIN in Linux is a special token that gets replaced by the directory of the library at runtime from that point we could navigate to the other libraries
# The same token for macOS is @loader_path
if(UNIX AND NOT APPLE)
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            INSTALL_RPATH "$ORIGIN"
    )
    # This is necessary for the gtest_main library to find the gtest library at runtime
    if (ANIRA_WITH_BENCHMARK)
    set_target_properties(gtest_main
        PROPERTIES
            INSTALL_RPATH "$ORIGIN"
    )
    endif()
elseif(APPLE)
    set(OSX_RPATHS "@loader_path")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        list(APPEND OSX_RPATHS "/opt/intel/oneapi/mkl/latest/lib")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    endif()
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            INSTALL_RPATH "${OSX_RPATHS}"
    )
    if (ANIRA_WITH_BENCHMARK)
    set_target_properties(gtest_main
        PROPERTIES
            INSTALL_RPATH "@loader_path"
    )
    endif()
endif()


# the variant with PUBLIC_HEADER property unfortunately does not preserve the folder structure therefore we use the simple install directory command
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/anira
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT dev
)

# install the target and create export-set
install(TARGETS ${PROJECT_NAME} concurrentqueue nlohmann_json
    EXPORT "aniraTargets"
    # these get default values from GNUInstallDirs
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} # .dll files 
    COMPONENT runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so or .dylib files
    COMPONENT runtime NAMELINK_COMPONENT dev
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .lib files
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

if(ANIRA_WITH_LIBTORCH)
    # LibTorch ships its own CMake package (lib/cmake/Torch + Caffe2), which the
    # installed aniraBackendTargets.cmake re-resolves and re-points at the relocated
    # headers (TorchConfig.cmake hardwires them to <prefix>/include).
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/include/"
        DESTINATION "${_anira_backend_incdir}/libtorch" COMPONENT deps-backends)
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/share/cmake/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake" COMPONENT deps-backends)
endif()

if(ANIRA_WITH_ONNXRUNTIME)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # iOS ships an xcframework: install it whole (the static .a then sits at the
        # SUBPATH anira::onnxruntime expects) plus the active slice's headers.
        install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/onnxruntime.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/onnxruntime.xcframework/${ANIRA_ONNXRUNTIME_IOS_SLICE}/Headers/"
            DESTINATION "${_anira_backend_incdir}/onnxruntime" COMPONENT deps-backends)
    else()
        if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
            install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/include/onnxruntime/"
                DESTINATION "${_anira_backend_incdir}/onnxruntime" COMPONENT deps-backends)
        else()
            install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/include/"
                DESTINATION "${_anira_backend_incdir}/onnxruntime" COMPONENT deps-backends)
        endif()
        install(DIRECTORY "${ANIRA_ONNXRUNTIME_ROOTDIR}/lib/"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
    endif()
endif()

if(ANIRA_WITH_TFLITE)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
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
        install(DIRECTORY "${ANIRA_TFLITE_ROOTDIR}/lib/"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
    endif()
endif()

if(ANIRA_WITH_LITERT)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/LiteRt.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/LiteRt.xcframework/${ANIRA_LITERT_IOS_SLICE}/Headers/"
            DESTINATION "${_anira_backend_incdir}/litert" COMPONENT deps-backends)
    else()
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/include/"
            DESTINATION "${_anira_backend_incdir}/litert" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_LITERT_ROOTDIR}/lib/"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
    endif()
endif()

if(ANIRA_WITH_EXECUTORCH)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/executorch.xcframework"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/executorch.xcframework/${ANIRA_EXECUTORCH_IOS_SLICE}/Headers/"
            DESTINATION "${_anira_backend_incdir}/executorch" COMPONENT deps-backends)
    else()
        # On desktop the installed lib/ tree includes ExecuTorch's CMake package
        # (lib/cmake/ExecuTorch), which the installed aniraBackendTargets.cmake
        # re-resolves via find_package(executorch) — analogous to LibTorch above.
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/include/"
            DESTINATION "${_anira_backend_incdir}/executorch" COMPONENT deps-backends)
        install(DIRECTORY "${ANIRA_EXECUTORCH_ROOTDIR}/lib/"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT deps-backends)
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
        set(_location "")
        if(ANIRA_${_ENGINE}_SHARED_LIB_SUBPATH)
            set(_location "\n    LOCATION \"\${_anira_libdir}/${ANIRA_${_ENGINE}_SHARED_LIB_SUBPATH}\"")
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
    if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
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