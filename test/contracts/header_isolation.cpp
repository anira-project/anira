// Compiled by the anira_header_isolation CTest with anira's own include directories and
// nothing else on the include path: no engine include directory and none of anira's private
// dependencies (tanh-lib, the queue, the JSON library). It fails to compile as soon as a public
// header includes an engine header or a private one again — the engine state belongs in each
// engine adapter's own translation unit under src/engines (src/engines/Adapter.h), and the
// core's C++ classes live under src/. Every public header is covered: the C umbrellas, the C++
// face and the 2.x compat shim.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/anira.h>
#include <anira/anira_all.h>

#include <anira/anira.hpp>
#include <anira/compat/v2.hpp>
#include <type_traits>

static_assert(std::is_class_v<anira::InferenceHandler>);
static_assert(std::is_class_v<anira::v2::InferenceHandler>);
// NOLINTEND(misc-include-cleaner)
