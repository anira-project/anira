# CPack Debian packaging through the shared module (cmake/tanh/package.cmake): one
# runtime package libanira<major>, one -dev package, and one -deps package grouping the
# bundled inference engines. Included after every install() rule.
#
#   cd build && cpack -G DEB      # then: apt install ./libanira*.deb && ldconfig
include(${CMAKE_CURRENT_LIST_DIR}/tanh/package.cmake)

set(_anira_deps_components deps-backends Devel Unspecified)
if(ANIRA_WITH_BENCHMARK)
    list(APPEND _anira_deps_components gtest gmock)
endif()
tanh_cpack_debian(
    VENDOR "anira-project"
    CONTACT "fares.schulz@tu-berlin.de"
    MAINTAINER "Fares Schulz <fares.schulz@tu-berlin.de>"
    RUNTIME_DESCRIPTION "library for real-time inference of neural networks"
    DEV_DESCRIPTION "header files for libanira${PROJECT_VERSION_MAJOR}"
    DEPS_DESCRIPTION "misc deps for libanira${PROJECT_VERSION_MAJOR}"
    DEPS_COMPONENTS ${_anira_deps_components})
unset(_anira_deps_components)
