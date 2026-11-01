// Gate 4 (strict): compiled with exactly the usage requirements of anira::anira, on every preset
// (desktop, Android, iOS, WebAssembly), under the strict flags. Nothing private is reachable
// through them: none of anira's own private headers (src/), no engine header, none of anira's
// private dependencies (tanh-lib, the queue, the JSON library) and no test or benchmark library.
// Each probe below is an #error the moment such a header resolves (a system-wide installation of
// one of those libraries on the build machine would fire it too; anira's CI images carry none),
// and then every public header must compile: the C umbrellas, the C++ face and the 2.x shim.
// NOLINTBEGIN(misc-include-cleaner)
#if !defined(__has_include)
#error "anira_header_isolation needs __has_include (C++17)"
#endif

// anira's private headers: src/ is no include directory of a consumer.
#if __has_include(<scheduler/Core.h>) || __has_include(<capi/handler.h>) || \
    __has_include(<engines/Adapter.h>) || __has_include(<utils/Logger.h>)
#error "a private header of anira (src/) is reachable through anira::anira"
#endif
// The engines: a consumer that calls one links anira::<engine> for its headers.
#if __has_include(<onnxruntime_cxx_api.h>) || __has_include(<torch/script.h>) || \
    __has_include(<tensorflow/lite/c/c_api.h>) || __has_include(<litert/c/litert_common.h>) || \
    __has_include(<executorch/runtime/core/error.h>)
#error "an engine header is reachable through anira::anira"
#endif
// anira's private dependencies.
#if __has_include(<tanh/core/Logger.h>) || __has_include(<tanh/tanh.h>)
#error "tanh-lib is reachable through anira::anira"
#endif
#if __has_include(<nlohmann/json.hpp>)
#error "nlohmann_json is reachable through anira::anira"
#endif
#if __has_include(<concurrentqueue.h>) || __has_include(<moodycamel/concurrentqueue.h>) || \
    __has_include(<concurrentqueue/moodycamel/concurrentqueue.h>)
#error "concurrentqueue is reachable through anira::anira"
#endif
#if __has_include(<benchmark/benchmark.h>) || __has_include(<gtest/gtest.h>)
#error "a test or benchmark library is reachable through anira::anira"
#endif

#include <anira/anira.h>
#include <anira/anira_all.h>

#include <anira/anira.hpp>
#include <anira/compat/v2.hpp>
#include <type_traits>

static_assert(std::is_class_v<anira::InferenceHandler>);
static_assert(std::is_class_v<anira::v2::InferenceHandler>);
// NOLINTEND(misc-include-cleaner)
