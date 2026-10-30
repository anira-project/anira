/*
 * The node test of the runtime that never exits on its own (WebAssembly only): a -pthread
 * library module (--no-entry: a program with a main() includes noExitRuntime anyway) that links
 * the archive with none of AniraWeb's link flags (EXIT_RUNTIME at its default); its one entry,
 * gate_start, is called by em_runtime.pre.js once the runtime is up. An emscripten_set_timeout
 * callback runs and calls C; when it returns no keepalive is held, which without the library's
 * noExitRuntime dependency (em_runtime.cpp) ends the runtime there (maybeExit, exit, ABORT). A
 * later plain setTimeout then arms a second emscripten_set_timeout, whose callback runs only on
 * a live runtime (callUserCallback drops it after ABORT). em_runtime.pre.js fails the process
 * unless that second callback ran. Exit code 0 on success.
 */
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <emscripten/em_js.h>
#include <emscripten/em_macros.h>
#include <emscripten/eventloop.h>
#include <stdint.h>
#include <stdio.h>

EM_JS_DEPS(em_runtime_gate, "$getWasmTableEntry") /* no ";": -Wextra-semi */

/* A plain setTimeout that calls fn through the table: no keepalive, no maybeExit of its own. */
EM_JS(void, call_later, (uintptr_t fn, double ms), {
    setTimeout(function() { getWasmTableEntry(fn)(); }, ms);
}) /* EM_JS ends a definition itself; a ';' here is -Wextra-semi */

EM_JS(void, mark_second_callback, (void), { globalThis.aniraRuntimeGate = true; }) /* likewise */

static void second_timer(void* user_data) {
    (void)user_data;
    anira_log(ANIRA_LOG_INFO, "anira.test", "anira-runtime-gate: the second timer callback");
    mark_second_callback();
    printf("a second timer callback ran: the runtime is alive\nPASSED: 0 failed check(s)\n");
}

static void late(void) {
    emscripten_set_timeout(second_timer, 10.0, NULL);
}

static void timer(void* user_data) {
    (void)user_data;
    anira_log(ANIRA_LOG_INFO, "anira.test", "anira-runtime-gate: the first timer callback");
    printf("an emscripten_set_timeout callback ran\n");
}

EMSCRIPTEN_KEEPALIVE void gate_start(void) {
    emscripten_set_timeout(timer, 10.0, NULL);
    call_later((uintptr_t)&late, 200.0);
}
