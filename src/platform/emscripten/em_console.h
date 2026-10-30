// The platform sink of WebAssembly: the browser console by level. tanh-lib's platform sink
// writes every record to stderr, which the module's glue prints as console.error, so a warning
// or an info record would show as an error; on WebAssembly that sink stays off and this one,
// a log sink of anira/utils/Logger.h, takes its place. Compiled on the Wasm binary format only.
#ifndef ANIRA_PLATFORM_EMSCRIPTEN_EM_CONSOLE_H
#define ANIRA_PLATFORM_EMSCRIPTEN_EM_CONSOLE_H

namespace anira::em {

/// Switches the console sink on or off (anira::detail::set_platform_sink_enabled on
/// WebAssembly: the hold of ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK and of the JavaScript log
/// hook). On from the module's start; tanh-lib's own platform sink is switched off the first
/// time. Not from inside a log sink.
void set_console_sink_enabled(bool enabled);

}  // namespace anira::em

#endif  // ANIRA_PLATFORM_EMSCRIPTEN_EM_CONSOLE_H
