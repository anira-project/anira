#ifndef ANIRA_EXAMPLES_CLEARENGINE_H
#define ANIRA_EXAMPLES_CLEARENGINE_H

// The advanced benchmark's baseline engine: it runs no model and writes zeros into every output,
// so a benchmark of it measures what anira's own path costs (the rings, the processor, the queue,
// the inference thread) without an inference. A model config serves it with an entry without a
// source (anira::ModelConfig::add_model(k_clear_engine_id)), which ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL
// allows; process allocates nothing.

#include <anira/anira.hpp>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>

inline constexpr const char* k_clear_engine_id = "org.anira.examples.clear";

class ClearEngine final : public anira::Engine {
public:
    ClearEngine() : anira::Engine(k_clear_engine_id) {}

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
            for (const anira::Tensor& out : ctx.outputs()) {
                void* data = out.data(out.dtype);
                if (data == nullptr) { continue; }
                const std::size_t bytes = out.num_elements() * ANIRA_DTYPE_BITS(out.dtype) *
                                          ANIRA_DTYPE_LANES(out.dtype) / 8;
                std::memset(data, 0, bytes);
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

#endif  // ANIRA_EXAMPLES_CLEARENGINE_H
