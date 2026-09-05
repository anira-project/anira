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

/// Creates a context and a C handler over it (the custom engine row, no model file);
/// nullptr if a step failed.
ANIRA_TEST_EXPORT void* unloadtest_create(void);
/// anira_handler_prepare with a 512-sample / 48 kHz Hard contract — starts the pool threads.
ANIRA_TEST_EXPORT void unloadtest_prepare(void* instance);
/// Runs num_blocks blocks through anira_handler_process().
ANIRA_TEST_EXPORT void unloadtest_process(void* instance, int num_blocks);
/// Destroys the instance (the handler releases its session, then the context goes).
ANIRA_TEST_EXPORT void unloadtest_destroy(void* instance);
/// Creates a handler over a model that does not load (an engine at a path that does not
/// exist); returns 1 if create or prepare failed (as one must), 0 if both succeeded.
ANIRA_TEST_EXPORT int unloadtest_create_throwing(void);

ANIRA_TEST_EXPORT unsigned int unloadtest_num_inference_threads(void);
ANIRA_TEST_EXPORT int unloadtest_has_inference_threads(void);
ANIRA_TEST_EXPORT int unloadtest_num_sessions(void);
ANIRA_TEST_EXPORT int unloadtest_has_core(void);
/// anira::Core::shutdown() — what a plugin calls from its module-exit entry point.
ANIRA_TEST_EXPORT void unloadtest_shutdown(void);
/// Leaks a context and starts a user-managed inference thread on it that is never
/// stopped. The negative control: a thread that is still alive when the module is
/// unloaded must crash the process.
ANIRA_TEST_EXPORT void unloadtest_leak_thread(void);
}

#endif  // ANIRA_TEST_UNLOAD_MODULE_API_H
