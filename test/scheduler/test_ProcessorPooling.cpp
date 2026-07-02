#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Context.h>

#include <memory>

#include "../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "gtest/gtest.h"

using namespace anira;

// Pick one real, pooled backend that is compiled into this build. The CUSTOM backend
// is never pooled, so it cannot exercise the shared-processor lifetime path.
#if defined(USE_LIBTORCH)
#define POOL_PROCESSOR(session) ((session)->m_libtorch_processor)
#define ANIRA_HAS_POOLED_BACKEND 1
#elif defined(USE_ONNXRUNTIME)
#define POOL_PROCESSOR(session) ((session)->m_onnx_processor)
#define ANIRA_HAS_POOLED_BACKEND 1
#elif defined(USE_TFLITE)
#define POOL_PROCESSOR(session) ((session)->m_tflite_processor)
#define ANIRA_HAS_POOLED_BACKEND 1
#elif defined(USE_LITERT)
#define POOL_PROCESSOR(session) ((session)->m_litert_processor)
#define ANIRA_HAS_POOLED_BACKEND 1
#else
#define ANIRA_HAS_POOLED_BACKEND 0
#endif

#if ANIRA_HAS_POOLED_BACKEND

// Regression test for issue #76 (use-after-free).
//
// Backend processors are pooled and shared between sessions whose configs compare
// equal (the default, m_session_exclusive_processor == false). A shared processor
// can therefore outlive the session that first created it. The host owns each
// session's InferenceConfig, so releasing that first session destroys its config.
// If the pooled processor held an `InferenceConfig&` bound to that config, the
// reference would dangle and the next inference on the surviving session would
// dereference freed memory.
//
// This reproduces the scenario at the Context level — two independently-owned
// configs with equal values (e.g. two anira~ patches), sharing one pooled
// processor, then the first is released and its config freed.
//
// We compare storage *addresses* rather than dereferencing the processor's config:
// the whole point of the bug is that dereferencing it is undefined behaviour, so the
// test must detect the dangling alias without triggering it. With the bug,
// `&processor->m_inference_config` is the address of the now-freed config (the
// reference's referent), so it equals the released storage. With the fix, the
// processor owns its config in its own storage, so the addresses differ.
TEST(ProcessorPoolingTest, PooledProcessorDoesNotAliasReleasedSessionConfig) {
    ContextConfig const context_config;
    auto context = Context::get_instance(context_config);

    // Two hosts, each owning an equal-valued InferenceConfig. Session A's config is
    // heap-allocated so its storage can be freed deterministically mid-test.
    auto* config_a = new InferenceConfig(hybridnn_config);
    auto* pp_a = new PrePostProcessor(*config_a);
    auto session_a = context->create_session(*pp_a, *config_a, nullptr);

    auto config_b = std::make_unique<InferenceConfig>(hybridnn_config);
    auto pp_b = std::make_unique<PrePostProcessor>(*config_b);
    auto session_b = context->create_session(*pp_b, *config_b, nullptr);

    // Precondition: equal configs must actually share one pooled processor, otherwise
    // the test would not exercise the bug at all.
    ASSERT_NE(POOL_PROCESSOR(session_a), nullptr);
    ASSERT_EQ(POOL_PROCESSOR(session_a), POOL_PROCESSOR(session_b))
        << "Sessions with equal configs are expected to share one pooled processor";

    // Keep the pooled processor alive independently so it can be inspected after
    // session A is gone (this is what the Context's processor pool does internally).
    auto pooled = POOL_PROCESSOR(session_b);
    const void* released_config_storage = static_cast<const void*>(config_a);

    // Release session A and free its config — exactly as a host destroying one plugin
    // instance would. Session B and the pooled processor live on.
    context->release_session(session_a);
    session_a.reset();
    delete pp_a;
    delete config_a;  // Session A's InferenceConfig storage is now freed.

    // The pooled processor is still in use by session B. It must not have its config
    // sitting in session A's freed storage.
    const void* processor_config_storage = static_cast<const void*>(&pooled->m_inference_config);
    EXPECT_NE(processor_config_storage, released_config_storage)
        << "Pooled processor still aliases the released session's InferenceConfig "
           "(use-after-free, issue #76)";

    // Cleanup: releasing the last session resets the Context singleton.
    context->release_session(session_b);
}

#else

TEST(ProcessorPoolingTest, DISABLED_RequiresPooledBackend) {
    GTEST_SKIP() << "No pooled inference backend compiled in; nothing to exercise.";
}

#endif
