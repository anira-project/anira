// Gate 5: a host that brings its own tanh-lib and links it before anira. The host's tanh::Core (a
// shared libtanh_core the consumer project builds itself, in tanh-lib's default namespace
// thl::v<major>_<minor>) comes first on the link line and anira::anira second; anira's copy of
// tanh-lib is inside libanira, compiled hidden in thl::anira. The two copies must not bind to
// each other: the host's logger never sees one of anira's records, and anira's records reach
// anira's sink, not the host's. On every platform and with a shared or a static libanira;
// run-install-test.cmake adds the nm half of the proof for a shared libanira (it names no thl
// symbol in its dynamic table, defined or undefined).
#include <tanh/core/Logger.h>

#include <anira/anira.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace {

std::atomic<int> g_host_records{0};
std::atomic<int> g_host_saw_anira{0};
std::atomic<int> g_anira_records{0};
std::atomic<int> g_anira_saw_host{0};

void ANIRA_CALL anira_sink(const anira_log_record* record, void* /*user_data*/) {
    g_anira_records.fetch_add(1);
    if (record->group != nullptr && std::strncmp(record->group, "host.", 5) == 0) {
        g_anira_saw_host.fetch_add(1);
    }
}

/// A pass-through engine, so the handler prepares without a model file.
class CopyEngine final : public anira::Engine {
public:
    CopyEngine() : anira::Engine("org.anira.install.tanh-first") {}
    uint32_t flags() const noexcept override { return ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL; }
    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& /*info*/) override {
        return std::make_unique<Loaded>();
    }

private:
    class Prepared final : public anira::Engine::Prepared {
    public:
        anira_status process(anira::EngineContext& ctx) noexcept override {
            const anira::Tensor& in = ctx.inputs()[0];
            const anira::Tensor& out = ctx.outputs()[0];
            std::memcpy(out.data_f32(), in.data_f32(), out.num_elements() * sizeof(float));
            return ANIRA_OK;
        }
    };
    class Loaded final : public anira::Engine::Loaded {
    public:
        std::unique_ptr<anira::Engine::Prepared> prepare(
            const anira::PrepareInfo& /*info*/) override {
            return std::make_unique<Prepared>();
        }
    };
};

anira::TensorSpec streamed(const char* name) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, 64);
    spec.window(64, 64, 0);
    return spec;
}

}  // namespace

int main() {
    // The host's own logger, from the host's own tanh-lib.
    thl::Logger::set_callback([](const thl::Logger::LogRecord& record) {
        g_host_records.fetch_add(1);
        if (record.m_group.rfind("anira", 0) == 0) { g_host_saw_anira.fetch_add(1); }
    });
    // Error, the one level compiled into every build type of the host's tanh-lib.
    thl::Logger::error("host.consumer", "a record of the host's own logger");

    {
        // anira, logging everything into its own sink, drained by hand.
        anira::Context context{anira::ContextConfig{}
                                   .log_level(ANIRA_LOG_DEBUG)
                                   .log_drain(ANIRA_LOG_DRAIN_MANUAL)
                                   .log_sink(anira_sink)};
        anira::ModelConfig model;
        model.add_model("org.anira.install.tanh-first");
        model.input(streamed("audio_in"));
        model.output(streamed("audio_out"));
        anira::Pipeline pipeline;
        pipeline.add_engine(std::make_shared<CopyEngine>());
        pipeline.add_inference(model);
        anira::InferenceHandler handler(context, pipeline);
        handler.prepare(anira::Hard{.host_block_min = 64,
                                    .host_block_max = 64,
                                    .host_rate = 48000,
                                    .budget = std::chrono::milliseconds(5),
                                    .warmup = ANIRA_WARMUP_NONE});
        anira_drain_log();
    }
    anira_drain_log();
    thl::Logger::error("host.consumer", "a second record of the host's own logger");

    const bool ok = g_host_records.load() >= 2 && g_host_saw_anira.load() == 0 &&
                    g_anira_records.load() > 0 && g_anira_saw_host.load() == 0;
    std::printf(
        "%s: consumer_tanh_first, host records %d (of anira's: %d), anira records %d (of "
        "the host's: %d)\n",
        ok ? "PASSED" : "FAILED",
        g_host_records.load(),
        g_host_saw_anira.load(),
        g_anira_records.load(),
        g_anira_saw_host.load());
    return ok ? 0 : 1;
}
