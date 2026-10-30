/*
 * The node test of the console sink (WebAssembly only): a consumer that links the library
 * archive with no export list and calls anira_log alone, nothing of the sink. The record must
 * reach the console once, through console.warn: the sink's object is linked (the library's
 * --undefined anchor) and its static constructor switched it on and tanh-lib's stderr sink
 * (console.error) off. em_console.pre.js counts the console lines that carry the marker.
 * Exit code 0 on success.
 */
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <emscripten/em_js.h>
#include <stdio.h>

/* What em_console.pre.js counted: 0 console.warn, 1 console.error, 2 any other method. */
EM_JS(int, gate_count, (int which), {
    var counts = globalThis.aniraConsoleGate;
    return which == 0 ? counts.warn : which == 1 ? counts.error : counts.other;
}) /* EM_JS ends a definition itself; a ';' here is -Wextra-semi */

int main(void) {
    anira_log(ANIRA_LOG_WARNING, "anira.test", "anira-console-gate: a warning");
    const int warns = gate_count(0);
    const int errors = gate_count(1);
    const int others = gate_count(2);
    printf("the warning on the console: console.warn %d, console.error %d, other %d\n",
           warns,
           errors,
           others);
    const int ok = warns == 1 && errors == 0 && others == 0;
    printf("%s: %d failed check(s)\n", ok ? "PASSED" : "FAILED", ok ? 0 : 1);
    return ok ? 0 : 1;
}
