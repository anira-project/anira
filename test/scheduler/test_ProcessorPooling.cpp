#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/extras_fixtures.h"
#include "backends/Adapter.h"
#include "backends/Adapters.h"
#include "capi/handles.h"  // IWYU pragma: keep - defines anira_context_config
#include "gtest/gtest.h"

using namespace anira;

// The pooled adapter of a session: plan 0 is the first configured model row, an engine of
// the build with the bundled model behind it. The CUSTOM backend is never pooled, so an
// engine-free build cannot exercise the shared-adapter lifetime path.
#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME) || defined(USE_TFLITE) || \
    defined(USE_LITERT) || defined(USE_EXECUTORCH)
#define ANIRA_HAS_POOLED_BACKEND 1
#else
#define ANIRA_HAS_POOLED_BACKEND 0
#endif

#if ANIRA_HAS_POOLED_BACKEND

namespace {

// One inference through `prepared` over buffers of its loaded model's element counts, with
// descriptors over them: what the inference thread hands the prepared handle, built by hand
// for a session that was never prepared.
anira_status run_once(backend::Prepared& prepared) {
    const backend::Model& model = prepared.loaded().model();
    std::vector<BufferF> inputs;
    std::vector<BufferF> outputs;
    std::vector<anira_tensor> input_tensors(model.m_inputs.size());
    std::vector<anira_tensor> output_tensors(model.m_outputs.size());
    for (const backend::TensorInfo& tensor : model.m_inputs) {
        inputs.emplace_back(1, tensor.m_num_elements);
        inputs.back().clear();
    }
    for (const backend::TensorInfo& tensor : model.m_outputs) {
        outputs.emplace_back(1, tensor.m_num_elements);
        outputs.back().clear();
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        anira_tensor_init_host(&input_tensors[i],
                               inputs[i].data(),
                               ANIRA_DTYPE_F32,
                               static_cast<uint32_t>(model.m_inputs[i].m_dims.size()),
                               model.m_inputs[i].m_dims.data());
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        anira_tensor_init_host(&output_tensors[i],
                               outputs[i].data(),
                               ANIRA_DTYPE_F32,
                               static_cast<uint32_t>(model.m_outputs[i].m_dims.size()),
                               model.m_outputs[i].m_dims.data());
    }
    anira_engine_ctx ctx{};
    ctx.num_inputs = static_cast<uint32_t>(input_tensors.size());
    ctx.num_outputs = static_cast<uint32_t>(output_tensors.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.inputs = input_tensors.data();
    ctx.outputs = output_tensors.data();
    backend::ChunkBuffers buffers{.m_inputs = &inputs, .m_outputs = &outputs};
    return prepared.run(ctx, &buffers, false);
}

}  // namespace

// Regression test for issue #76 (use-after-free).
//
// Prepared models are pooled and shared between sessions whose plans describe the same model
// (the default, m_session_exclusive_processor == false). A shared loaded model can therefore
// outlive the session that first created it. The host owns each session's InferenceConfig,
// so releasing that first session destroys its config. If the pooled loaded model held a
// reference into that config, the reference would dangle and the next inference on the
// surviving session would dereference freed memory.
//
// This reproduces the scenario at the core level — two independently-owned configs with
// equal values (e.g. two anira~ patches), sharing one pooled loaded model, then the first is
// released and its config freed.
//
// The record the loaded model keeps is compared by storage address with the released config
// (the whole point of the bug is that dereferencing a dangling alias is undefined behaviour,
// so the test detects it without triggering it), and then the model runs one inference on
// the surviving session's behalf: with a dangling alias that is the read ASan catches.
TEST(ProcessorPoolingTest, PooledProcessorDoesNotAliasReleasedSessionConfig) {
    // The defaults: AUTO threads (resolved by the core), SPIN_BACKOFF, WARNING.
    const anira_context_config core_config;

    // Two hosts, each owning an equal-valued InferenceConfig. Session A's config is
    // heap-allocated so its storage can be freed deterministically mid-test.
    const InferenceConfig hybridnn_config =
        anira_test::bridged(k_hybridnn_model_json, k_hybridnn_contract_json);
    auto* config_a = new InferenceConfig(hybridnn_config);
    auto* pp_a = new PrePostProcessor(*config_a);
    auto session_a = Core::create_session(*pp_a,
                                          *config_a,
                                          backend::legacy_plan_requests(*config_a, nullptr),
                                          core_config);

    auto config_b = std::make_unique<InferenceConfig>(hybridnn_config);
    auto pp_b = std::make_unique<PrePostProcessor>(*config_b);
    auto session_b = Core::create_session(*pp_b,
                                          *config_b,
                                          backend::legacy_plan_requests(*config_b, nullptr),
                                          core_config);

    // Precondition: equal configs must actually share one pooled loaded model, otherwise the
    // test would not exercise the bug at all.
    ASSERT_FALSE(session_a->m_plans.empty());
    ASSERT_NE(session_a->m_plans[0].m_loaded, nullptr);
    ASSERT_EQ(session_a->m_plans[0].m_loaded, session_b->m_plans[0].m_loaded)
        << "Sessions with equal configs are expected to share one pooled loaded model";

    // Keep the pooled loaded model alive independently so it can be inspected after session A
    // is gone (this is what the core's pool does internally).
    const std::shared_ptr<backend::Loaded> pooled = session_b->m_plans[0].m_loaded;
    const void* released_config_storage = static_cast<const void*>(config_a);

    // Release session A and free its config — exactly as a host destroying one plugin
    // instance would. Session B and the pooled loaded model live on.
    Core::release_session(session_a);
    session_a.reset();
    delete pp_a;
    delete config_a;  // Session A's InferenceConfig storage is now freed.

    // The pooled loaded model is still in use by session B. Its record must not sit in session
    // A's freed storage, and an inference on it must read nothing of that storage.
    const void* record_storage = static_cast<const void*>(&pooled->model());
    EXPECT_NE(record_storage, released_config_storage)
        << "Pooled loaded model still aliases the released session's InferenceConfig "
           "(use-after-free, issue #76)";
    ASSERT_NE(session_b->m_plans[0].m_prepared, nullptr);
    EXPECT_EQ(run_once(*session_b->m_plans[0].m_prepared), ANIRA_OK);

    // Cleanup: releasing the last session tears the shared thread pool down.
    Core::release_session(session_b);
}

#else

TEST(ProcessorPoolingTest, DISABLED_RequiresPooledBackend) {
    GTEST_SKIP() << "No pooled inference backend compiled in; nothing to exercise.";
}

#endif
