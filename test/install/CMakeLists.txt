# Consumer of an installed anira package. Not part of the main build: the
# build_install CI workflow installs anira into a fresh prefix and builds this
# project against it with find_package(anira), so a header, exported target or
# dependency (tanh::Core, the backends) missing from the install tree fails CI
# rather than a downstream project.
cmake_minimum_required(VERSION 3.25)
project(anira_install_consumer CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anira REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE anira::anira)
