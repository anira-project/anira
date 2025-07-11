set(PROJECT_NAME tests)
project (${PROJECT_NAME} VERSION ${PROJECT_VERSION})


# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_compile_definitions(
        GUITARLSTM_MODELS_PATH_TENSORFLOW="${GUITARLSTM_MODELS_PATH_TENSORFLOW}"
        GUITARLSTM_MODELS_PATH_PYTORCH="${GUITARLSTM_MODELS_PATH_PYTORCH}"
        STEERABLENAFX_MODELS_PATH_TENSORFLOW="${STEERABLENAFX_MODELS_PATH_TENSORFLOW}"
        STEERABLENAFX_MODELS_PATH_PYTORCH="${STEERABLENAFX_MODELS_PATH_PYTORCH}"
        STATEFULLSTM_MODELS_PATH_TENSORFLOW="${STATEFULLSTM_MODELS_PATH_TENSORFLOW}"
        STATEFULLSTM_MODELS_PATH_PYTORCH="${STATEFULLSTM_MODELS_PATH_PYTORCH}"
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
	test_InferenceHandler.cpp
    utils/test_Buffer.cpp
	test_WavReader.cpp
)

target_link_libraries(${PROJECT_NAME} anira::anira)

# gtest_discover_tests will register a CTest test for each gtest and run them all in parallel with the rest of the Test.
gtest_discover_tests(${PROJECT_NAME} DISCOVERY_TIMEOUT 90)

if (MSVC)
	add_custom_command(TARGET ${PROJECT_NAME}
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${ANIRA_SHARED_LIBS_WIN}
			$<TARGET_FILE_DIR:${PROJECT_NAME}>
	)
endif (MSVC)