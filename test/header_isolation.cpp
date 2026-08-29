// Compiled by the anira_header_isolation CTest with anira's own include directory
// (plus those of its public dependencies) and nothing else on the include path: no
// engine include directory. It fails to compile as soon as a public header includes
// an engine header again — the engine state belongs behind each backend processor's
// named pimpl (see the note on anira::BackendBase). Deliberately the umbrella header,
// so that every public header is covered.
#include <anira/anira.h>  // NOLINT(misc-include-cleaner)

#include <type_traits>

static_assert(std::is_class_v<anira::InferenceConfig>);  // NOLINT(misc-include-cleaner)
