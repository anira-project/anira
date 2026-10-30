// A consumer of the installed anira through anira/anira.hpp: a model config built in code, a
// custom engine that needs no model file, a Pipeline, an InferenceHandler on a Context, prepared
// with a Hard contract, one block through the Hard entry. The installed package must carry
// anira's headers, its exported target and what the library links.
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace {

/// A pass-through engine: the first input into the first output.
class CopyEngine final : public anira::Engine {
public:
    CopyEngine() : anira::Engine("org.anira.install.copy") {}
    uint32_t flags() const noexcept override {
        return ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL;
    }
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
    anira::Context context{anira::ContextConfig{}};
    anira::ModelConfig model;
    model.add_model("org.anira.install.copy");
    model.input(streamed("audio_in"));
    model.output(streamed("audio_out"));
    anira::Pipeline pipeline;
    pipeline.add_engine(std::make_shared<CopyEngine>());
    pipeline.add_inference(model);
    anira::InferenceHandler handler(context, pipeline);
    handler.prepare(anira::Hard{.block_min = 64,
                                .block_max = 64,
                                .rate = 48000,
                                .budget = std::chrono::milliseconds(5),
                                .warmup = ANIRA_WARMUP_NONE});

    std::vector<float> data(64, 0.25F);
    float* plane = data.data();
    const std::array<int64_t, 2> extents{1, 64};
    const anira::Tensor block =
        anira::Tensor::from_host_planar<float>(std::span<float* const>(&plane, 1), extents);
    const anira_status status = handler.process(block, 0, block, 0);

    std::printf("anira %s, status %d, latency %u, plans %u\n",
                anira_version_string(),
                static_cast<int>(status),
                handler.get_latency(),
                handler.plan_report().num_plans());
    return status == ANIRA_OK && handler.rt_error() == ANIRA_OK ? 0 : 1;
}
