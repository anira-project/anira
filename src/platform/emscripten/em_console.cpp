// The console sink of WebAssembly: see em_console.h.
#include "em_console.h"

#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/utils/Logger.h>
#include <emscripten/em_js.h>
#include <tanh/core/Logger.h>

#include <mutex>
#include <string>

namespace {

// One line on the console of the thread that logs, by level: console.debug, .info, .warn,
// .error. The line is tanh-lib's platform format, "[warn][rt][anira.scheduler] message". The
// body is kept from clang-format, which formats JavaScript as C++ (`===` becomes `== =`).
EM_JS_DEPS(anira_em_console, "$UTF8ToString");

// clang-format off
EM_JS(void, anira_em_js_console, (int level, const char* line), {
    var text = UTF8ToString(line);
    if (level == 3) {
        console.error(text);
    } else if (level == 2) {
        console.warn(text);
    } else if (level == 1) {
        console.info(text);
    } else {
        console.debug(text);
    }
});
// clang-format on

static_assert(ANIRA_LOG_DEBUG == 0 && ANIRA_LOG_INFO == 1 && ANIRA_LOG_WARNING == 2 &&
                  ANIRA_LOG_ERROR == 3,
              "the console sink passes anira_log_level as the JavaScript switch reads it");

const char* level_tag(uint32_t level) {
    switch (level) {
        case ANIRA_LOG_ERROR: return "error";
        case ANIRA_LOG_WARNING: return "warn";
        case ANIRA_LOG_INFO: return "info";
        default: return "debug";
    }
}

void ANIRA_CALL console_sink(const anira_log_record* record, void* /*user_data*/) {
    std::string line = "[";
    line += level_tag(record->level);
    line += (record->flags & ANIRA_LOG_RECORD_REALTIME) != 0 ? "][rt][" : "][native][";
    line += record->group != nullptr ? record->group : "";
    line += "] ";
    line += record->message != nullptr ? record->message : "";
    anira_em_js_console(static_cast<int>(record->level), line.c_str());
}

std::mutex s_mutex;
anira::detail::LogSinkId s_sink = 0;
bool s_platform_off = false;

// On from the module's start, before any record: the static constructor of this TU, which
// the library links whenever it links its log sinks (LogSinks.cpp calls in here).
[[maybe_unused]] const bool k_started = [] {
    anira::em::set_console_sink_enabled(true);
    return true;
}();

}  // namespace

namespace anira::em {

void set_console_sink_enabled(bool enabled) {
    const std::scoped_lock<std::mutex> lock(s_mutex);
    if (!s_platform_off) {
        // tanh-lib's platform sink: stderr, every level as console.error. Never its drain
        // thread (the core drains its own queue, see Core::start_log_drain_locked).
        thl::Logger::LoggerConfig config = thl::Logger::get_config();
        config.m_platform_enabled = false;
        config.m_rt_enabled = false;
        thl::Logger::set_config(config);
        s_platform_off = true;
    }
    if (enabled && s_sink == 0) {
        s_sink = anira::detail::add_log_sink(&console_sink, nullptr, ANIRA_LOG_DEBUG);
    } else if (!enabled && s_sink != 0) {
        anira::detail::remove_log_sink(s_sink);
        s_sink = 0;
    }
}

}  // namespace anira::em
