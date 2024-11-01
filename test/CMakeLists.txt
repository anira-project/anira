set(PROJECT_NAME tests)
project (${PROJECT_NAME} VERSION 0.0.1)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    test_audiobuffer.cpp
)

target_link_libraries(${PROJECT_NAME} anira::anira)

# gtest_discover_tests will register a CTest test for each gtest and run them all in parallel with the rest of the Test.
gtest_discover_tests(${PROJECT_NAME} DISCOVERY_TIMEOUT 90)
