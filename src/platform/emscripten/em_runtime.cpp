// The runtime of a module that links anira never exits on its own (WebAssembly only). With
// -pthread and without the noExitRuntime symbol, emsdk's keepRuntimeAlive() is
// `runtimeKeepaliveCounter > 0` alone (src/lib/libcore.js: pthreads must be able to exit), so the
// first callback into C that runs with no keepalive held (an emscripten_set_timeout) ends in
// maybeExit, exit and ABORT, after which every callback into C is dropped silently. Including
// noExitRuntime makes keepRuntimeAlive() `noExitRuntime || counter > 0`, and noExitRuntime is
// `!EXIT_RUNTIME`: true for a library module (AniraWeb) and for an app that leaves EXIT_RUNTIME
// at its default; an app that sets EXIT_RUNTIME=1 owns its runtime's lifetime. Compiled on the
// Wasm binary format only; every link against anira pulls it in (CMakeLists.txt).
#include <emscripten/em_js.h>

EM_JS_DEPS(anira_em_runtime, "$noExitRuntime");

// The symbol every WebAssembly link of anira pulls in, so this TU and its dependency are in the
// module whatever of the library the consumer calls. Not part of the ABI.
extern "C" void anira_em_runtime_anchor() {}
