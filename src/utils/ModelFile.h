/*
 * The two things every backend shares on its load path: the model-file pre-check that
 * runs before an engine sees a path, and the one message shape every backend failure
 * carries, "<engine>: <path|memory>: <engine text>".
 *
 * Private, header-only: never included from a public header, never exported.
 */
#ifndef ANIRA_UTILS_MODELFILE_H
#define ANIRA_UTILS_MODELFILE_H

#include <anira/abi/status.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

#include "StatusError.h"

namespace anira::model_file {

/// The location of a model handed over as bytes (a binary ModelData), for the message.
inline constexpr const char* k_memory = "memory";

/// Formats a backend failure: "<engine>: <where>: <text>" while the three fit the
/// caller-owned anira_error record (ANIRA_ERROR_MESSAGE_CAPACITY bytes, head kept on
/// truncation). When they would not fit, the engine's text — the reason — comes first
/// and the location last, so that what survives the truncation is the diagnosis and
/// not the path (an engine's text often repeats the path anyway).
inline std::string message(const char* engine, const std::string& where, const std::string& text) {
    const std::string prefix = std::string(engine) + ": ";
    if (prefix.size() + where.size() + 2 + text.size() < ANIRA_ERROR_MESSAGE_CAPACITY) {
        return prefix + where + ": " + text;
    }
    return prefix + text + " (" + where + ")";
}

/// Checks that path names a readable regular file and returns its absolute, lexically
/// normalised form (the path every later message of that backend carries). Throws
/// StatusError(ANIRA_ERROR_NO_SUCH_FILE) with "<engine>: <absolute path>: no such file"
/// when nothing is there, "...: not a regular file" for a directory or a device, and
/// "...: not readable" when the file exists but cannot be opened for reading.
inline std::string require_readable(const std::string& path, const char* engine) {
    if (path.empty()) {
        throw StatusError(ANIRA_ERROR_NO_SUCH_FILE,
                          message(engine, "(empty path)", "no such file"));
    }
    std::error_code ec;
    std::filesystem::path absolute = std::filesystem::absolute(path, ec);
    if (ec) { absolute = path; }
    absolute = absolute.lexically_normal();
    const std::string where = absolute.string();

    const std::filesystem::file_status status = std::filesystem::status(absolute, ec);
    if (ec || !std::filesystem::exists(status)) {
        throw StatusError(ANIRA_ERROR_NO_SUCH_FILE, message(engine, where, "no such file"));
    }
    if (!std::filesystem::is_regular_file(status)) {
        throw StatusError(ANIRA_ERROR_NO_SUCH_FILE, message(engine, where, "not a regular file"));
    }
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) closed right below
    std::FILE* file = std::fopen(where.c_str(), "rb");
    if (file == nullptr) {
        throw StatusError(ANIRA_ERROR_NO_SUCH_FILE, message(engine, where, "not readable"));
    }
    static_cast<void>(std::fclose(file));
    return where;
}

}  // namespace anira::model_file

#endif  // ANIRA_UTILS_MODELFILE_H
