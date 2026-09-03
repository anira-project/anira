project(anira-tests VERSION ${PROJECT_VERSION})


# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# The root of the model tree at run time (extras/CMakeLists.txt); every model path the tests
# spell is relative to it, and extras/models/model_files.h names the configuration files.
add_compile_definitions(ANIRA_EXTRAS_MODELS_DIR="${ANIRA_EXTRAS_MODELS_DIR}")


# One test binary per component, mirroring the source tree: test/<dir> covers
# src/<dir> and include/anira/<dir>, and the root files cover the root units. The
# test_* naming is load-bearing: it is what the shared ci-actions mobile runners
# auto-discover.
function(anira_add_test_binary TARGET)
    # support/CoreReclaim.cpp releases the context core after the binary's last test,
    # so the immortal-by-design core is not reported as a leak under LeakSanitizer.
    add_executable(${TARGET} ${ARGN} "${CMAKE_CURRENT_SOURCE_DIR}/support/CoreReclaim.cpp")
    target_link_libraries(${TARGET} anira::anira)

    if(TANH_OPERATING_SYSTEM MATCHES "^(Android|iOS)$")
        # On mobile the test binary runs on a device/emulator/simulator, reached over a
        # single adb/simctl invocation. Per-test registration is wrong here: gtest_discover_tests
        # would have to run the cross-built binary on the host just to enumerate the cases
        # (impossible), and even if it could, each case would re-spawn the target. Register
        # one CTest entry per binary that runs it in a single on-device launch.
        add_test(NAME ${TARGET} COMMAND ${TARGET})
    else()
        # One CTest entry per gtest, so ctest runs them all in parallel. Every test
        # carries its binary name as a label for component selection
        # (ctest -L test_scheduler). Every leg runs the full suite.
        #
        # PRE_TEST: ctest enumerates the binaries one after another when it starts,
        # instead of one post-build step per target that Ninja runs in parallel. CMake
        # 4.4.0's post-build discovery writes every target's --gtest_list_tests JSON to
        # the same file (the per-target hash is taken from an empty name; CMake issue
        # 27972, fixed in 4.4.1), so with the four binaries linking within the same
        # second a neighbour's fopen truncated a list between its writer's exit and
        # CMake's read, and the build failed with "Syntax error: value, object or array
        # expected" (Homebrew CMake 4.4.0 on the macOS runners; test_utils,
        # test_backends and test_scheduler each hit it). Serial discovery has no such
        # window on any CMake version.
        gtest_discover_tests(${TARGET}
            DISCOVERY_MODE PRE_TEST
            DISCOVERY_TIMEOUT 90
            PROPERTIES LABELS ${TARGET})
    endif()

    # Only copy runtime DLLs when there are any (a fully static build has none).
    if (MSVC AND ANIRA_SHARED_LIBS_WIN)
        add_custom_command(TARGET ${TARGET}
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ANIRA_SHARED_LIBS_WIN}
                $<TARGET_FILE_DIR:${TARGET}>
        )
    endif ()
endfunction()

anira_add_test_binary(test_utils
	utils/test_Buffer.cpp
	utils/test_RingBuffer.cpp
	utils/test_Semaphore.cpp
	utils/test_JsonConfigLoader.cpp
	utils/test_JsonConfigLoaderErrors.cpp
	utils/test_HelperFunctions.cpp
	utils/test_Logger.cpp
	system/test_HighPriorityThread.cpp
)

anira_add_test_binary(test_scheduler
	scheduler/test_ConcurrentLifecycle.cpp
	scheduler/test_ContextConfigMismatch.cpp
	scheduler/test_ContextLifecycle.cpp
	scheduler/test_CreateSessionFailure.cpp
	scheduler/test_FallbackPaths.cpp
	scheduler/test_InferenceManager.cpp
	scheduler/test_InferenceThread.cpp
	scheduler/test_LatencyCalculator.cpp
	scheduler/test_ProcessorPooling.cpp
	scheduler/test_SessionElement.cpp
	scheduler/test_StatefulOrdering.cpp
)

anira_add_test_binary(test_backends
	backends/test_BackendBase.cpp
	backends/test_ExecuTorchModelFunction.cpp
	backends/test_ExecuTorchProcessor.cpp
	backends/test_LibTorchProcessor.cpp
)

anira_add_test_binary(test_handler
	contracts/test_BackendLinkage.cpp
	test_ContextConfig.cpp
	test_InferenceConfig.cpp
	test_InferenceHandler.cpp
	test_InferenceHandlerApi.cpp
	test_OneSidedStreaming.cpp
	test_PrePostProcessor.cpp
)

# The C ABI (include/anira/abi/, src/capi/): version negotiation, status text, the
# exception firewall and the log entries. test/abi/ also holds the ABI gates.
anira_add_test_binary(test_abi
	abi/test_Version.cpp
	abi/test_Error.cpp
	abi/test_Firewall.cpp
	abi/test_Log.cpp
	abi/test_Handles.cpp
	abi/test_ExtRegistry.cpp
	abi/test_Json.cpp
	abi/test_JsonUpgrade.cpp
	abi/test_anira_hpp.cpp
	abi/test_Translate.cpp
	abi/test_Bridge.cpp
	abi/test_ExtrasFixtures.cpp
)
# capi_internal.h (the firewall probe) and the generated tables under src/capi/.
target_include_directories(test_abi PRIVATE "${anira_SOURCE_DIR}/src")
add_subdirectory(abi)

# Header-isolation check: anira's public headers must compile with anira's own include
# directory (plus those of its public dependencies — the queue, JSON and tanh-lib
# headers the public headers include) and no engine include directory at all. The
# engine state lives behind the named pimpl of each backend processor (see
# BackendBase.h); this is what lets anira link its backends PRIVATE and makes
# "include an engine header -> link anira::<engine>" true at compile time. The object
# is excluded from the default build and compiled by a CTest, so that a regression
# shows up as a failed test with the compiler's error instead of a broken build.
if(NOT TANH_OPERATING_SYSTEM MATCHES "^(Android|iOS|Emscripten)$")
    add_library(anira_header_isolation OBJECT EXCLUDE_FROM_ALL contracts/header_isolation.cpp)
    set_target_properties(anira_header_isolation PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)
    target_include_directories(anira_header_isolation PRIVATE
        "${anira_SOURCE_DIR}/include"
        $<TARGET_PROPERTY:concurrentqueue,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:tanh::Core,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(anira_header_isolation PRIVATE
        ${ANIRA_COMPILE_DEFINITIONS}
        $<TARGET_PROPERTY:tanh::Core,INTERFACE_COMPILE_DEFINITIONS>
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:ANIRA_STATIC>)
    add_test(NAME anira_header_isolation
        COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --config $<CONFIG>
                --target anira_header_isolation)
    # The test drives the build tree; keep build-invoking tests mutually
    # exclusive under a parallel ctest (-j).
    set_tests_properties(anira_header_isolation PROPERTIES RESOURCE_LOCK anira_build_tree)
endif()

# Host-shaped library-unload test: a plugin-like module embedding anira is dlopen'ed,
# driven and dlclose'd by an executable that does not link anira. It proves that no
# anira thread survives the unload (and that the library really gets unmapped). Not
# on mobile (no dlopen story on device) and not under RTSan (the module inherits
# -fsanitize=realtime but an executable without the runtime cannot load it).
# Not under ASan/TSan/LSan either: the suite's premise is that the library's memory goes
# away underneath code that may still touch it — LeakedThreadCrashesOnUnload leaks a
# thread on purpose, dlcloses the module and asserts the leaked thread dies on the freed
# queue. A sanitizer diagnoses exactly that and fails the run. The unload contract is
# covered on the uninstrumented build_test legs.
if(NOT TANH_OPERATING_SYSTEM MATCHES "^(Android|iOS|Emscripten)$"
   AND NOT ANIRA_WITH_RTSAN AND NOT ANIRA_WITH_ASAN AND NOT ANIRA_WITH_TSAN AND NOT ANIRA_WITH_LSAN)
    add_subdirectory(contracts/unload)
endif()

# Installed-package smoke test: install anira to a throwaway prefix and build +
# run the downstream find_package(anira) consumers of test/install against it —
# anira alone, and an engine-calling consumer that links anira::onnxruntime (plus
# the negative check that the engine header is unreachable without it). Only
# meaningful when install targets exist, so gate on ANIRA_WITH_INSTALL (needs both
# that and ANIRA_WITH_TESTS). This is the only coverage for the install/export path.
if(ANIRA_WITH_INSTALL)
    add_test(NAME InstallConsumer
        COMMAND ${CMAKE_COMMAND}
            -DANIRA_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DCONSUMER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/contracts/install
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install-test
            -DGENERATOR=${CMAKE_GENERATOR}
            -DBUILD_TYPE=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/contracts/install/run-install-test.cmake)
    # RESOURCE_LOCK: installs from the build tree — see anira_header_isolation.
    set_tests_properties(InstallConsumer PROPERTIES TIMEOUT 600 RESOURCE_LOCK anira_build_tree)
endif()
