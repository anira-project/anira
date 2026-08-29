# ==============================================================================
# Sources related to the benchmarking options
# ==============================================================================

target_sources(${PROJECT_NAME}
    PRIVATE
        # TODO: find out why we need to add the header files here, so that they can find the <benchmark/benchmark.h> and <gtest/gtest.h> files
        include/anira/benchmark/ProcessBlockFixture.h
        src/benchmark/ProcessBlockFixture.cpp
)

# google benchmark through the shared module (cmake/tanh/test-deps.cmake): testing and
# install off, warnings not errors, the Apple Silicon std::regex probe answered.
include(${CMAKE_CURRENT_LIST_DIR}/tanh/test-deps.cmake)
tanh_fetch_googlebenchmark(VERSION v1.9.5 INSTALL)  # exported with anira (find_dependency(benchmark))
target_link_libraries(${PROJECT_NAME} PUBLIC benchmark)
