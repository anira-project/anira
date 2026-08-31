# googletest for the test suites, through the shared module (cmake/tanh/test-deps.cmake).
# Dashboard targets are suppressed; gtest_main is linked into anira as a convenience
# for the in-process test binaries (test/unload links gtest directly).
include(CTest)
include(${CMAKE_CURRENT_LIST_DIR}/tanh/test-deps.cmake)
tanh_fetch_googletest(VERSION v1.17.0 INSTALL)  # exported with anira (find_dependency(GTest))
target_link_libraries(${PROJECT_NAME} PUBLIC gtest_main)
