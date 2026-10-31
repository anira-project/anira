# The fixture of the benchmark examples: a Google Benchmark fixture over anira.hpp
# (ProcessBlockFixture.h), the static library the benchmarks link. Not part of anira and not
# installed: a project of your own copies the two files.
include(${anira_SOURCE_DIR}/cmake/tanh/test-deps.cmake)
tanh_fetch_googlebenchmark(VERSION v1.9.5)

add_library(anira_benchmark_fixture STATIC
    ProcessBlockFixture.cpp
    ProcessBlockFixture.h
)
target_include_directories(anira_benchmark_fixture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(anira_benchmark_fixture PUBLIC cxx_std_20)
# The benchmarks are gtest executables (gtest_discover_tests registers each case): gtest_main
# comes with the fixture. They raise their thread's priority through tanh-lib's thread API, so
# they link tanh::Core themselves, as any application using tanh-lib does: anira's copy is
# private to libanira.
target_link_libraries(anira_benchmark_fixture PUBLIC anira::anira benchmark::benchmark gtest_main tanh::Core)
