// Gate (internal): the library's private scheduler classes beside anira.hpp in one translation
// unit. libanira defines anira::scheduler::InferenceThread and anira::Core; anira.hpp defines its
// classes inline in namespace anira. A class of the same name in both (the ODR violation the 2.x
// anira::InferenceHandler and anira::InferenceThread were, a strong definition in the library
// against the inline one in every TU that includes anira.hpp, the library's own sources and the
// white-box tests among them) is a compile error here. Include hygiene is not the point of this
// TU, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include "scheduler/Core.h"
#include "scheduler/InferenceThread.h"
// anira.hpp second.
#include <anira/anira.hpp>
#include <type_traits>

namespace {

static_assert(!std::is_same_v<anira::InferenceThread, anira::scheduler::InferenceThread>);
static_assert(std::is_class_v<anira::Core>);
static_assert(std::is_class_v<anira::InferenceHandler>);

}  // namespace
// NOLINTEND(misc-include-cleaner)
