# to package:
# go into the build dir
# call "cpack -G DEB"
# sometimes it needs to be called twice...

set(CPACK_THREADS 10)

set(CPACK_PACKAGE_NAME "lib${PROJECT_NAME}")
set(CPACK_DEBIAN_PACKAGE_NAME ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VENDOR "anira-project")
set(CPACK_VERBATIM_VARIABLES YES)

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})

#TODO maybe change this to outside of buildtree?
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/packages")

set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")

set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CPACK_PACKAGE_CONTACT "fares.schulz@tu-berlin.de")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Fares Schulz <${CPACK_PACKAGE_CONTACT}>")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

#TODO add all actual deps
#TODO add changelog
#TODO add copyright file

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

# each group (or component, if not in group) is built as a seperate package
set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
set(CPACK_DEB_COMPONENT_INSTALL YES)


# setup components
cpack_add_component(runtime REQUIRED)
cpack_add_component(dev DEPENDS runtime)

# group all dependency components
cpack_add_component(deps-backends GROUP deps)
cpack_add_component(Devel GROUP deps)
cpack_add_component(Unspecified GROUP deps)

# remove -runtime suffix of runtime package, add major version number instead 
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME ${CPACK_PACKAGE_NAME}${PROJECT_VERSION_MAJOR})

# automatically generete dependencies between components
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)

set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION libs)
set(CPACK_DEBIAN_DEV_PACKAGE_SECTION libdevel)

# extremely slow, and doesn't work for the deps package
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)

# fix unstripped-binary-or-object error (probably to remove unwanted debug symbols)
set(CPACK_STRIP_FILES YES)

# set package descriptions, cmake variables 
set(CPACK_DEBIAN_RUNTIME_DESCRIPTION "library for real-time inference of neural networks")
set(CPACK_DEBIAN_DEV_DESCRIPTION "header files for libanira${PROJECT_VERSION_MAJOR}")
set(CPACK_DEBIAN_DEPS_DESCRIPTION "misc deps for libanira${PROJECT_VERSION_MAJOR}")

include(CPack)