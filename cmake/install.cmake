# ==============================================================================
# Install the library
# ==============================================================================

# for CMAKE_INSTALL_INCLUDEDIR and others definition
include(GNUInstallDirs)

# include the public headers of the anira library for the install target
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_directories(${PROJECT_NAME} PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>
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
# $ORIGIN is a special token that gets replaced by the directory of the library at runtime from that point we could navigate to the other libraries
# it is a little strange but the onnxruntime library and libtorch work without this setting in the JUCE example... but tensorflowlite does not
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

# the variant with PUBLIC_HEADER property unfortunately does not preserve the folder structure therefore we use the simple install directory command
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/anira
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install the target and create export-set
install(TARGETS ${PROJECT_NAME}
    EXPORT "aniraTargets"
    # these get default values from GNUInstallDirs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
)

# libtorch has cmake config files that we can use to install the library later with find_package and then just link to it
# if we would implement the install manually we would have to include the include and lib directories to our install interface
if(ANIRA_WITH_LIBTORCH)
    # install(DIRECTORY "${LIBTORCH_ROOTDIR}/"
    #     DESTINATION "${CMAKE_INSTALL_PREFIX}/modules/${LIBTORCH_DIR_NAME}"
    # )
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
    install(DIRECTORY "${LIBTORCH_ROOTDIR}/share/cmake/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    )
endif()

# the other ones don't have cmake config files so we have to install them manually
if(ANIRA_WITH_ONNXRUNTIME)
    install(DIRECTORY "${ONNXRUNTIME_ROOTDIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    install(DIRECTORY "${ONNXRUNTIME_ROOTDIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endif()

if(ANIRA_WITH_TFLITE)
    install(DIRECTORY "${TENSORFLOWLITE_ROOTDIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    install(DIRECTORY "${TENSORFLOWLITE_ROOTDIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endif()

# set a debug postfix for the library
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

# ==============================================================================
# Generate cmake config files
# ==============================================================================

# generate and install export file in the folder cmake with the name of the project and namespace
# this generates files called aniraTargets.cmake, aniraTargets-debug.cmake, aniraTargets-release.cmake
install(EXPORT "aniraTargets"
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
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
)