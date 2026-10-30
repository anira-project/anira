#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../abi/handler_support.h"
#include "engines/Adapter.h"
#include "gtest/gtest.h"

using namespace anira;

// The pooled engine of a session: plan 0 is the bundled gain model on the first engine of the
// build. A custom engine is pooled per engine object, and a Handler adds an object of its own
// to every pipeline, so an engine-free build cannot exercise the shared-adapter lifetime path.
#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME) || defined(USE_TFLITE) || \
    defined(USE_LITERT) || defined(USE_EXECUTORCH)
#define ANIRA_HAS_POOLED_BACKEND 1
#else
#define ANIRA_HAS_POOLED_BACKEND 0
#endif

#if ANIRA_HAS_POOLED_BACKEND

// Regression test for issue #76 (use-after-free).
//
// Loaded models are pooled and shared between sessions whose plans describe the same model
// (a model that is not session-exclusive). A shared loaded model can therefore outlive the
// session that first created it. Each handler owns its copy of the model config, so
// destroying the first handler destroys its config. If the pooled loaded model held a
// reference into that config, the reference would dangle and the next inference on the
// surviving session would dereference freed memory.
//
// This reproduces the scenario on two handlers over equal model configs (e.g. two plugin
// instances), sharing one pooled loaded model; then the first is destroyed and its config
// freed.
//
// The record the loaded model keeps is compared by storage address with the destroyed
// handler's config (the whole point of the bug is that dereferencing a dangling alias is
// undefined behaviour, so the test detects it without triggering it), and then the surviving
// handler runs one inference on the model: with a dangling alias that is the read ASan catches.
TEST(ProcessorPoolingTest, PooledProcessorDoesNotAliasReleasedSessionConfig) {
    const anira_test::Context context(2);
    const std::vector<anira::BackendId> enabled = anira::enabled_engines();
    ASSERT_FALSE(enabled.empty());
    const std::vector<anira_backend_id> first_engine{{.struct_size = sizeof(anira_backend_id),
                                                      .engine = enabled.front().engine,
                                                      .provider = ANIRA_PROVIDER_CPU,
                                                      .engine_id = nullptr}};
    const anira::ModelConfig model = anira::ModelConfig::from_file(k_gain_model_json);
    const anira::ContractHandle contract = anira_test::file_contract(k_gain_contract_json, 512);

    // Handler A is heap-allocated so its config can be freed deterministically mid-test.
    auto handler_a = std::make_unique<anira_test::Handler>(context, model, first_engine);
    ASSERT_EQ(handler_a->prepare(contract), ANIRA_OK) << handler_a->m_err.message;
    anira_test::Handler handler_b(context, model, first_engine);
    ASSERT_EQ(handler_b.prepare(contract), ANIRA_OK) << handler_b.m_err.message;

    // Precondition: equal configs must actually share one pooled loaded model, otherwise the
    // test would not exercise the bug at all.
    const std::shared_ptr<SessionElement> session_a = anira_test::session_of(handler_a->m_handler);
    const std::shared_ptr<SessionElement> session_b = anira_test::session_of(handler_b.m_handler);
    ASSERT_NE(session_a, nullptr);
    ASSERT_NE(session_b, nullptr);
    ASSERT_FALSE(session_a->m_plans.empty());
    ASSERT_NE(session_a->m_plans[0].m_loaded, nullptr);
    ASSERT_EQ(session_a->m_plans[0].m_loaded, session_b->m_plans[0].m_loaded)
        << "Handlers over equal configs are expected to share one pooled loaded model";

    // Keep the pooled loaded model alive independently so it can be inspected after handler A
    // is gone (this is what the core's pool does internally).
    const std::shared_ptr<engine::Loaded> pooled = session_b->m_plans[0].m_loaded;
    const void* released_config_storage =
        static_cast<const void*>(&handler_a->m_handler->m_pipeline.m_variants[0]);

    // Destroy handler A, and with it its config — exactly as a host destroying one plugin
    // instance would. Handler B and the pooled loaded model live on.
    handler_a.reset();

    // The pooled loaded model is still in use by handler B. Its record must not sit in handler
    // A's freed storage, and an inference on it must read nothing of that storage.
    const void* record_storage = static_cast<const void*>(&pooled->model());
    EXPECT_NE(record_storage, released_config_storage)
        << "Pooled loaded model still aliases the released handler's model config "
           "(use-after-free, issue #76)";

    std::vector<float> audio(512, 0.25F);
    const std::array<float*, 1> channels{audio.data()};
    const std::array<int64_t, 2> shape{1, 512};
    anira_tensor block{};
    anira_tensor_init_host_planar(&block,
                                  static_cast<const void*>(channels.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    size_t delivered = 0;
    EXPECT_EQ(anira_handler_process_wait(handler_b.m_handler,
                                         &block,
                                         0,
                                         &block,
                                         0,
                                         &delivered,
                                         ANIRA_WAIT_FOREVER),
              ANIRA_OK);
    EXPECT_EQ(delivered, 512U);
}

#else

TEST(ProcessorPoolingTest, DISABLED_RequiresPooledBackend) {
    GTEST_SKIP() << "No pooled inference backend compiled in; nothing to exercise.";
}

#endif
