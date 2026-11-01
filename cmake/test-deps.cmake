# googletest for the test suites and the benchmarks, through the shared module
# (cmake/tanh/test-deps.cmake). Dashboard targets are suppressed. Every binary that runs gtest
# links gtest_main itself (anira_add_test_binary, the benchmark fixture, the unload harness);
# nothing of googletest is linked into anira or installed with it.
include(CTest)
include(${CMAKE_CURRENT_LIST_DIR}/tanh/test-deps.cmake)
tanh_fetch_googletest(VERSION v1.17.0)
