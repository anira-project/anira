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
		RAVE_MODEL_FUNK_DRUM_JSON_CONFIG_PATH="${RAVE_MODEL_FUNK_DRUM_JSON_CONFIG_PATH}"
		RAVE_MODEL_FUNK_DRUM_ENCODER_JSON_CONFIG_PATH="${RAVE_MODEL_FUNK_DRUM_ENCODER_JSON_CONFIG_PATH}"
		RAVE_MODEL_FUNK_DRUM_DECODER_JSON_CONFIG_PATH="${RAVE_MODEL_FUNK_DRUM_DECODER_JSON_CONFIG_PATH}"
		SIMPLE_GAIN_JSON_CONFIG_PATH="${SIMPLE_GAIN_JSON_CONFIG_PATH}"
		SIMPLEGAIN_MODEL_PATH="${SIMPLEGAIN_MODEL_PATH}"
		RAVE_MODEL_DIR="${RAVE_MODEL_DIR}"
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
	test_BackendLinkage.cpp
	test_InferenceHandler.cpp
	test_OneSidedStreaming.cpp
	test_StatefulOrdering.cpp
	backends/test_BackendBase.cpp
	backends/test_ExecuTorchModelFunction.cpp
    utils/test_Buffer.cpp
	utils/test_RingBuffer.cpp
	utils/test_PrePostProcessorWindow.cpp
	utils/test_Semaphore.cpp
	utils/test_JsonConfigLoader.cpp
	scheduler/test_ConcurrentLifecycle.cpp
	scheduler/test_ContextLifecycle.cpp
	scheduler/test_CreateSessionFailure.cpp
	scheduler/test_InferenceManager.cpp
	scheduler/test_ProcessorPooling.cpp
	scheduler/test_SessionElement.cpp
	scheduler/test_UserManagedThread.cpp
	test_WavReader.cpp
)

target_link_libraries(${PROJECT_NAME} anira::anira)

if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # On mobile the test binary runs on a device/emulator/simulator, reached over a
    # single adb/simctl invocation. Per-test registration is wrong here: gtest_discover_tests
    # would have to run the cross-built binary on the host just to enumerate the cases
    # (impossible), and even if it could, each case would re-spawn the target. Register
    # one CTest entry that runs the whole bundled binary in a single on-device launch.
    add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
else()
    # gtest_discover_tests will register a CTest test for each gtest and run them all in parallel with the rest of the Test.
    gtest_discover_tests(${PROJECT_NAME} DISCOVERY_TIMEOUT 90)
endif()

# Host-shaped library-unload test: a plugin-like module embedding anira is dlopen'ed,
# driven and dlclose'd by an executable that does not link anira. It proves that no
# anira thread survives the unload (and that the library really gets unmapped). Not
# on mobile (no dlopen story on device) and not under RTSan (the module inherits
# -fsanitize=realtime but an executable without the runtime cannot load it).
if(NOT CMAKE_SYSTEM_NAME MATCHES "Android|iOS" AND NOT EMSCRIPTEN AND NOT ANIRA_WITH_RTSAN)
    add_subdirectory(unload)
endif()

# Installed-package smoke test: install anira to a throwaway prefix and build +
# run a downstream find_package(anira) consumer against it. Only meaningful when
# install targets exist, so gate on ANIRA_WITH_INSTALL (needs both that and
# ANIRA_WITH_TESTS). This is the only coverage for the install/export path.
if(ANIRA_WITH_INSTALL)
    add_test(NAME InstallConsumer
        COMMAND ${CMAKE_COMMAND}
            -DANIRA_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DCONSUMER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/install/consumer
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install-test
            -DGENERATOR=${CMAKE_GENERATOR}
            -DBUILD_TYPE=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/install/run_install_test.cmake)
    set_tests_properties(InstallConsumer PROPERTIES TIMEOUT 600)
endif()

# Only copy runtime DLLs when there are any (a fully static build has none).
if (MSVC AND ANIRA_SHARED_LIBS_WIN)
	add_custom_command(TARGET ${PROJECT_NAME}
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${ANIRA_SHARED_LIBS_WIN}
			$<TARGET_FILE_DIR:${PROJECT_NAME}>
	)
endif ()
