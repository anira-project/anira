# Minimal downstream project that consumes an installed anira via find_package.
# Driven by test/install/run_install_test.cmake (the "InstallConsumer" CTest).
cmake_minimum_required(VERSION 3.15)
project(anira_install_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anira REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE anira::anira)
