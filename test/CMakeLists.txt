project(anira-tests VERSION ${PROJECT_VERSION})


# Sets the cpp language minimum
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# The root of the model tree at run time (extras/CMakeLists.txt); every model path the tests
# spell is relative to it, and extras/models/model_files.h names the configuration files.
add_compile_definitions(ANIRA_EXTRAS_MODELS_DIR="${ANIRA_EXTRAS_MODELS_DIR}")


# Windows: anira.dll and the runtime DLLs are searched next to the executable. Every test
# binary of this directory is built into the same directory, so one target copies them there
# once and each binary depends on it; a PRE_BUILD copy per binary had parallel links race on
# the same file ("Permission denied"). Only when there are DLLs (a fully static build has none).
if (MSVC AND ANIRA_SHARED_LIBS_WIN)
    get_property(_anira_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (_anira_multi_config)
        set(_anira_test_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
    else ()
        set(_anira_test_bin_dir "${CMAKE_CURRENT_BINARY_DIR}")
    endif ()
    add_custom_target(anira_test_runtime_dlls
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_anira_test_bin_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ANIRA_SHARED_LIBS_WIN}
                "${_anira_test_bin_dir}"
        COMMENT "Copying the runtime DLLs next to the test binaries"
        VERBATIM)
    # The list names files the build makes, not only prebuilt ones: anira.dll by its path and
    # tanh-lib's and gtest's DLLs as $<TARGET_FILE:...>. The copy waits for every target that
    # builds one (the per-binary copy ran after its binary's own link dependencies).
    if (BUILD_SHARED_LIBS)
        add_dependencies(anira_test_runtime_dlls anira)
    endif ()
    foreach (_anira_dll IN LISTS ANIRA_SHARED_LIBS_WIN)
        if (_anira_dll MATCHES "^\\$<TARGET_FILE:(.+)>$")
            set(_anira_dll_target "${CMAKE_MATCH_1}")
            get_target_property(_anira_aliased "${_anira_dll_target}" ALIASED_TARGET)
            if (_anira_aliased)
                set(_anira_dll_target "${_anira_aliased}")
            endif ()
            get_target_property(_anira_imported "${_anira_dll_target}" IMPORTED)
            if (NOT _anira_imported)
                add_dependencies(anira_test_runtime_dlls "${_anira_dll_target}")
            endif ()
        endif ()
    endforeach ()
    unset(_anira_dll)
    unset(_anira_dll_target)
    unset(_anira_aliased)
    unset(_anira_imported)
    unset(_anira_multi_config)
    unset(_anira_test_bin_dir)
endif ()

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
        # test_engines and test_scheduler each hit it). Serial discovery has no such
        # window on any CMake version.
        gtest_discover_tests(${TARGET}
            DISCOVERY_MODE PRE_TEST
            DISCOVERY_TIMEOUT 90
            PROPERTIES LABELS ${TARGET})
    endif()

    if (TARGET anira_test_runtime_dlls)
        add_dependencies(${TARGET} anira_test_runtime_dlls)
    endif ()
endfunction()

# The partition of the test suite for the 3.x cut-over (decision 22 of the M2 plan): every TU
# targets one API and moves once.
#   (a) through anira/compat/v2.hpp, the 2.x source-compatible face over the C ABI:
#       test_InferenceHandlerApi, test_OneSidedStreaming (PR 8).
#   (b) rewritten on anira/anira.hpp: test_InferenceHandler (PR 10, with the extras as Processors).
#   (c) white-box over the private headers (the src include directory): test_scheduler,
#       test_engines, test_utils, plus test_OneSidedStreamingInternals (the struct-pool cases of
#       test_OneSidedStreaming on the 2.x session, PR 8), and test_capi_internal
#       (test_abi's internal probes and the 2.x arms, PR 12).
#   (d) retired with their subject: test_Bridge (PR 10, with the bridge; what the C prepare
#       owns is in test_Prepare), test_HighPriorityThread (PR 12); with the 2.x configuration
#       types of the core (PR 10b): test_JsonConfigLoader and its Errors suite, test_CoreConfig,
#       test_InferenceConfig, test_PrePostProcessor, test_Buffer, test_HelperFunctions, and the
#       2.x configuration comparison of test_Translate, which is test_Validate now (what the
#       validator derives, which the session reads).
#   (e) API-free: test_EngineLinkage (gtest and preprocessor flags only; re-pointed from USE_*
#       onto the enabled-engines entry in PR 12).
# No suite drives a 2.x session any more (PR 10b): the 2.x handler and the 2.x objects of the
# white-box suites left the tests, the arms of test_abi check the 3.x side against closed
# forms, and the session runs on the 3.x configuration.

anira_add_test_binary(test_utils
	utils/test_RingBuffer.cpp
	utils/test_Semaphore.cpp
	utils/test_Logger.cpp
	system/test_HighPriorityThread.cpp
)
# test_Logger reads its C handlers white-box through test/abi/handler_support.h, which includes
# the private headers.
target_include_directories(test_utils PRIVATE "${anira_SOURCE_DIR}/src")

anira_add_test_binary(test_scheduler
	scheduler/test_PlanSwitch.cpp
	scheduler/test_ConcurrentLifecycle.cpp
	scheduler/test_ContextConfigMismatch.cpp
	scheduler/test_CopyPathOracle.cpp
	scheduler/test_CoreLifecycle.cpp
	scheduler/test_CreateSessionFailure.cpp
	scheduler/test_FallbackPaths.cpp
	scheduler/test_InferenceManager.cpp
	scheduler/test_InferenceThread.cpp
	scheduler/test_LatencyCalculator.cpp
	scheduler/test_OneSidedStreamingInternals.cpp
	scheduler/test_ProcessorPooling.cpp
	scheduler/test_SessionElement.cpp
	scheduler/test_StatefulOrdering.cpp
	scheduler/test_TensorStems.cpp
)
# The C handler's body (src/capi/handler.h: the session a handler prepared, its ports) and the
# engine room (src/engines/Adapter.h) the suites read white-box.
target_include_directories(test_scheduler PRIVATE "${anira_SOURCE_DIR}/src")

anira_add_test_binary(test_engines
	engines/test_Adapter.cpp
	engines/test_ExecuTorchModelFunction.cpp
	engines/test_AdapterExecuTorch.cpp
	engines/test_AdapterLibTorch.cpp
)
# The engine room (src/engines/Adapter.h and the adapters) for test_Adapter.
target_include_directories(test_engines PRIVATE "${anira_SOURCE_DIR}/src")

anira_add_test_binary(test_handler
	contracts/test_EngineLinkage.cpp
	test_InferenceHandler.cpp
	test_InferenceHandlerApi.cpp
	test_OneSidedStreaming.cpp
)

# The C ABI (include/anira/abi/, src/capi/): version negotiation, status text, the
# exception firewall and the log entries. test/abi/ also holds the ABI gates.
anira_add_test_binary(test_abi
	abi/test_Version.cpp
	abi/test_Error.cpp
	abi/test_Firewall.cpp
	abi/test_Log.cpp
	abi/test_Handles.cpp
	abi/test_ReadBack.cpp
	abi/test_ExtRegistry.cpp
	abi/test_Json.cpp
	abi/test_JsonUpgrade.cpp
	abi/test_anira_hpp.cpp
	abi/test_Validate.cpp
	abi/test_ExtrasFixtures.cpp
	abi/test_Extras.cpp
	abi/test_Context.cpp
	abi/test_Thread.cpp
	abi/test_Handler.cpp
	abi/test_HandlerCopyOracle.cpp
	abi/test_HandlerStatic.cpp
	abi/test_HandlerTensor.cpp
	abi/test_HandlerWait.cpp
	abi/test_InstanceDispatch.cpp
	abi/test_Prepare.cpp
	abi/test_RtError.cpp
	abi/test_Processor.cpp
	abi/test_Engine.cpp
	abi/test_State.cpp
	abi/test_Tensor.cpp
	abi/test_Compat.cpp
)
# capi_internal.h (the firewall probe) and the generated tables under src/capi/.
target_include_directories(test_abi PRIVATE "${anira_SOURCE_DIR}/src")
add_subdirectory(abi)

# Header-isolation check: anira's public headers must compile with anira's own include
# directory (plus those of its public dependencies — the queue, JSON and tanh-lib
# headers the public headers include) and no engine include directory at all. The
# engine state lives in each engine adapter's own translation unit under src/engines; this
# is what lets anira link its backends PRIVATE and makes
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
        "${ANIRA_GENERATED_INCLUDE_DIR}"
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
