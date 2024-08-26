# ==============================================================================
# Install the executable
# ==============================================================================

# for CMAKE_INSTALL_INCLUDEDIR and others definition
include(GNUInstallDirs)

# define the directory where the executable will be installed CMAKE_INSTALL_PREFIX
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-${PROJECT_VERSION}" CACHE PATH "Where the library will be installed to" FORCE)

set(CUSTOM_RPATH "/root/anira/lib")
if (ANIRA_WITH_INSTALL)
    list(APPEND CUSTOM_RPATH "$ORIGIN/../lib")
endif()
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "${CUSTOM_RPATH}"
)

# install the target
install(TARGETS ${PROJECT_NAME}
    # these get default values from GNUInstallDirs
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)
