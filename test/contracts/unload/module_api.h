#ifndef ANIRA_TEST_UNLOAD_MODULE_API_H
#define ANIRA_TEST_UNLOAD_MODULE_API_H

// C entry points of the plugin-shaped test module (module.cpp). The test executable
// (test_LibraryUnload.cpp) deliberately includes neither anira headers nor this file —
// it resolves these by name after dlopen/LoadLibrary and mirrors the signatures with
// function-pointer typedefs. Keep the two in sync.

#if defined(_WIN32)
#define ANIRA_TEST_EXPORT __declspec(dllexport)
#else
#define ANIRA_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

/// Creates an InferenceHandler-backed instance (CUSTOM backend, no model files);
/// nullptr if construction threw.
ANIRA_TEST_EXPORT void* unloadtest_create(void);
/// prepare() with a 512-sample / 48 kHz host configuration — starts the pool threads.
ANIRA_TEST_EXPORT void unloadtest_prepare(void* instance);
/// Runs num_blocks blocks through process().
ANIRA_TEST_EXPORT void unloadtest_process(void* instance, int num_blocks);
/// Destroys the instance (its InferenceHandler releases its session).
ANIRA_TEST_EXPORT void unloadtest_destroy(void* instance);
/// Constructs an InferenceHandler with a backend whose prepare() throws; returns 1 if
/// the constructor threw (as it must), 0 otherwise.
ANIRA_TEST_EXPORT int unloadtest_create_throwing(void);

ANIRA_TEST_EXPORT unsigned int unloadtest_num_inference_threads(void);
ANIRA_TEST_EXPORT int unloadtest_has_inference_threads(void);
ANIRA_TEST_EXPORT int unloadtest_num_sessions(void);
ANIRA_TEST_EXPORT int unloadtest_has_core(void);
/// anira::Context::shutdown() — what a plugin calls from its module-exit entry point.
ANIRA_TEST_EXPORT void unloadtest_shutdown(void);
/// Starts a user-managed inference thread and never stops it. The negative control:
/// a thread that is still alive when the module is unloaded must crash the process.
ANIRA_TEST_EXPORT void unloadtest_leak_thread(void);
}

#endif  // ANIRA_TEST_UNLOAD_MODULE_API_H
