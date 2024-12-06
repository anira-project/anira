# ==============================================================================
# Install the library
# ==============================================================================

# for CMAKE_INSTALL_INCLUDEDIR and others definition
include(GNUInstallDirs)

# define the dircetory where the library will be installed CMAKE_INSTALL_PREFIX
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-${PROJECT_VERSION}" CACHE PATH "Where the library will be installed to" FORCE)

# ==============================================================================
# Install specific compile definition
# ==============================================================================

if(ANIRA_WITH_INSTALL)
    target_compile_definitions(${TARGET_NAME}
        PUBLIC
        INSTALL_VERSION
    )
endif()

set(INSTALL_TARGETS ${TARGET_NAME}_Standalone ${TARGET_NAME}_VST3)

# at install the rpath is cleared by default so we have to set it again for the installed shared library to find the other libraries
# in this case we set the rpath to the directories where the other libraries are installed
# $ORIGIN in Linux is a special token that gets replaced by the directory of the library at runtime from that point we could navigate to the other libraries
# The same token for macOS is @loader_path
if(UNIX AND NOT APPLE)
    foreach(TARGET ${INSTALL_TARGETS})
        set_target_properties(${TARGET}
            PROPERTIES
                INSTALL_RPATH "$ORIGIN/../lib"
        )
    endforeach()
elseif(APPLE)
    set(OSX_RPATHS "@loader_path/../lib;@loader_path/../../../../lib;@loader_path/../../../")
    list(APPEND INSTALL_TARGETS ${TARGET_NAME}_AU)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        list(APPEND OSX_RPATHS "/opt/intel/oneapi/mkl/latest/lib")
    endif()
    foreach(TARGET ${INSTALL_TARGETS})
        set_target_properties(${TARGET}
            PROPERTIES
                INSTALL_RPATH "${OSX_RPATHS}"
        )
    endforeach()
endif()

# install the target and create export-set
install(TARGETS ${INSTALL_TARGETS}
    # these get default values from GNUInstallDirs
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)
