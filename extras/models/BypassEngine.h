#ifndef ANIRA_EXTRAS_BYPASSENGINE_H
#define ANIRA_EXTRAS_BYPASSENGINE_H

// The bypass of the bundled audio models as a custom engine: for every row of the model's input
// window (the rows are the input's elements over the extent of its last axis: one row for the
// CNN's [1, 1, 15380], 256 for the HybridNN's [256, 1, 150]), the last samples of the row go to
// the output, as many as the output holds per row (2048 for the CNN, 1 for the HybridNN). That is
// the window's newest audio, so the model's output is its input, delayed by the handler's
// latency: what the examples' "bypass" setting runs, for any model shaped this way. The first
// input goes to the first output; every other output (a Static output) is zeros.
//
// A model config serves it with an entry without a source, which the engine's
// ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL allows (anira::ModelConfig::add_model(k_bypass_engine_id), or a
// JSON entry `{"engine": "org.anira.examples.bypass"}`); the pipeline carries the engine
// (Pipeline::add_engine). process allocates nothing
// (ANIRA_ENGINE_FLAG_REALTIME_SAFE); a shape it cannot read so (an output row longer than the
// input row) is zero-filled.

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>

inline constexpr const char* k_bypass_engine_id = "org.anira.examples.bypass";

class BypassEngine final : public anira::Engine {
public:
    BypassEngine() : anira::Engine(k_bypass_engine_id) {}

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
            const std::span<const anira::Tensor> inputs = ctx.inputs();
            const std::span<anira::Tensor> outputs = ctx.outputs();
            if (inputs.empty() || outputs.empty()) { return ANIRA_OK; }
            const float* in = inputs[0].data_f32();
            float* out = outputs[0].data_f32();
            if (in == nullptr || out == nullptr) { return ANIRA_ERROR_CONFIG; }
            const anira::Tensor& source = inputs[0];
            const std::size_t row = source.ndim == 0 ? 1 : source.extent(source.ndim - 1);
            const std::size_t in_elements = source.num_elements();
            const std::size_t out_elements = outputs[0].num_elements();
            const std::size_t rows = row > 0 ? in_elements / row : 0;
            const std::size_t per_row = rows > 0 ? out_elements / rows : 0;
            if (rows == 0 || per_row == 0 || per_row > row || per_row * rows != out_elements) {
                std::memset(out, 0, out_elements * sizeof(float));
                return ANIRA_OK;
            }
            for (std::size_t r = 0; r < rows; ++r) {
                std::memcpy(out + (r * per_row),
                            in + (r * row) + (row - per_row),
                            per_row * sizeof(float));
            }
            // Every other output (a Static output of the model, the gain models' peak) is zeros.
            for (std::size_t i = 1; i < outputs.size(); ++i) {
                void* data = outputs[i].data(outputs[i].dtype);
                if (data == nullptr) { continue; }
                std::memset(data,
                            0,
                            outputs[i].num_elements() * ANIRA_DTYPE_BITS(outputs[i].dtype) *
                                ANIRA_DTYPE_LANES(outputs[i].dtype) / 8);
            }
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

#endif  // ANIRA_EXTRAS_BYPASSENGINE_H
