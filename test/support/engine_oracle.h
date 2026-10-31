#ifndef ANIRA_TEST_SUPPORT_ENGINE_ORACLE_H
#define ANIRA_TEST_SUPPORT_ENGINE_ORACLE_H

// The oracle of a stream through a neural model, which has no closed form: the model of one plan
// of a C handler, run directly in the engine room on windows built by hand from the input
// signal. The oracle loads the plan's own record (the engine, the file or the bytes, the engine's
// extents under the entry's layout, the entry point) on an executor of its own, outside the
// scheduler, and runs one inference per window. What a test compares against it is everything
// the handler adds: the rings, the window fill or the processor, the dispatch, the plan's
// executor and the latency priming.

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "../abi/handler_support.h"
#include "engines/Adapter.h"
#include "engines/Adapters.h"
#include "scheduler/Core.h"
#include "scheduler/SessionElement.h"
#include "utils/Logger.h"

namespace anira_test {

class EngineOracle {
public:
    /// The model of `plan` of the handler's session; ready() is false for a plan that does not
    /// run on a built-in engine (a registered engine has no record to reload here).
    EngineOracle(anira_handler* handler, uint32_t plan) {
        const std::shared_ptr<anira::SessionElement> session = session_of(handler);
        if (session == nullptr || plan >= session->m_plans.size()) { return; }
        anira::engine::Model record = session->m_plans[plan].m_loaded->model();
        if (record.m_engine == ANIRA_ENGINE_CUSTOM) { return; }
        record.m_instances = 0;  // an executor of the oracle's own, no shared slot
        record.m_warm_up = 0;
        m_loaded = anira::engine::make_builtin_loaded(anira::Core::builtin_engine(record.m_engine));
        if (m_loaded == nullptr) { return; }
        anira_init_info info = ANIRA_INIT_INFO_INIT;
        info.log_level = static_cast<uint32_t>(anira::get_log_level());
        m_loaded->init(info);
        m_loaded->load(record);
        m_prepared = m_loaded->prepare(
            anira::engine::PrepareRequest{.m_exclusive = true, .m_info = nullptr});
    }

    bool ready() const noexcept { return m_prepared != nullptr; }

    /// One inference over float32 inputs, one buffer per input slot in the engine's extents
    /// (row-major); the outputs, one buffer per output slot. Empty on a failed call.
    std::vector<std::vector<float>> run(std::vector<std::vector<float>> inputs) {
        const anira::engine::Model& record = m_loaded->model();
        std::vector<std::vector<float>> outputs;
        outputs.reserve(record.m_outputs.size());
        for (const anira::engine::TensorInfo& tensor : record.m_outputs) {
            outputs.emplace_back(tensor.m_num_elements, 0.0F);
        }
        const auto descriptors = [](std::vector<std::vector<float>>& buffers,
                                    const std::vector<anira::engine::TensorInfo>& tensors) {
            std::vector<anira_tensor> out(buffers.size());
            for (size_t slot = 0; slot < buffers.size() && slot < tensors.size(); ++slot) {
                anira_tensor_init_host(&out[slot],
                                       buffers[slot].data(),
                                       ANIRA_DTYPE_F32,
                                       static_cast<uint32_t>(tensors[slot].m_dims.size()),
                                       tensors[slot].m_dims.data());
            }
            return out;
        };
        const std::vector<anira_tensor> in_tensors = descriptors(inputs, record.m_inputs);
        std::vector<anira_tensor> out_tensors = descriptors(outputs, record.m_outputs);
        anira_engine_ctx ctx{};
        ctx.num_inputs = static_cast<uint32_t>(in_tensors.size());
        ctx.num_outputs = static_cast<uint32_t>(out_tensors.size());
        ctx.ticket = ANIRA_TICKET_INVALID;
        ctx.inputs = in_tensors.data();
        ctx.outputs = out_tensors.data();
        if (m_prepared->run(ctx, false) != ANIRA_OK) { return {}; }
        return outputs;
    }

private:
    std::shared_ptr<anira::engine::Loaded> m_loaded;  // outlives the prepared handle over it
    std::unique_ptr<anira::engine::Prepared> m_prepared;
};

/// The window anira's default fill hands a mono receptive-field model for inference `j`: the
/// `window` samples of `signal` that end with its hop `j` (hops of `hop` samples), silence before
/// the stream's first sample.
inline std::vector<float> receptive_window(std::span<const float> signal,
                                           size_t j,
                                           size_t hop,
                                           size_t window) {
    std::vector<float> out(window, 0.0F);
    const auto end = static_cast<long>((j + 1) * hop);
    for (size_t i = 0; i < window; ++i) {
        const long n = end - static_cast<long>(window) + static_cast<long>(i);
        if (n >= 0 && static_cast<size_t>(n) < signal.size()) {
            out[i] = signal[static_cast<size_t>(n)];
        }
    }
    return out;
}

/// The windows extras/models/hybrid-nn/HybridNNProcessor builds for inference `j`: `batches`
/// windows of `window` samples, window b ending with sample j * batches + b, silence before the
/// stream's first sample.
inline std::vector<float> batched_windows(std::span<const float> signal,
                                          size_t j,
                                          size_t batches,
                                          size_t window) {
    std::vector<float> out(batches * window, 0.0F);
    for (size_t b = 0; b < batches; ++b) {
        const auto last = static_cast<long>((j * batches) + b);
        for (size_t i = 0; i < window; ++i) {
            const long n = last - static_cast<long>(window) + 1 + static_cast<long>(i);
            if (n >= 0 && static_cast<size_t>(n) < signal.size()) {
                out[(b * window) + i] = signal[static_cast<size_t>(n)];
            }
        }
    }
    return out;
}

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_ENGINE_ORACLE_H
