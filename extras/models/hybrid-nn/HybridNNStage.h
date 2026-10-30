#ifndef ANIRA_EXTRAS_HYBRIDNNSTAGE_H
#define ANIRA_EXTRAS_HYBRIDNNSTAGE_H

// The windows GuitarLSTM (extras/models/hybrid-nn) takes, as the pre_process of an anira::Stage.
// The model's input is `batches` windows of 150 samples, each one sample further than the last
// ([batches, 1, 150]; 256 in hybridnn.model.json), and its output one sample per window: one
// inference consumes a hop of `batches` new samples with 149 samples of history ahead of them.
// anira's default fill writes one window per channel, so the batches are this stage's: one
// RingView::pop_windows of `batches` windows of 149 old and 1 new sample. The output side is
// anira's default push (phases() names pre_process only).
//
// The shapes are read off the prepare record (the model tensors at the pinned window), so the
// stage serves any batch count of the builder in HybridNNConfig.h. The stage sees the canonical
// [batch, channel, time] layout on every engine: a TensorFlow row's [batch, time, channel] is
// the engine's (the entry's layout), and with one channel the memory is the same anyway.

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

class HybridNNStage final : public anira::Stage {
public:
    uint32_t phases() const noexcept override { return k_pre_process; }
    // pre_process runs on the audio thread under a Hard contract: it pops and copies, nothing
    // else.
    uint32_t flags() const noexcept override { return ANIRA_STAGE_FLAG_REALTIME_PRE_POST; }

    std::unique_ptr<Prepared> prepare(const anira::PrepareInfo& info) override {
        const anira::Tensor& input = info.inputs()[0];
        const anira::Tensor& output = info.outputs()[0];
        const auto batches = static_cast<uint32_t>(input.extent(0));
        const std::size_t window = batches > 0 ? input.num_elements() / batches : 0;
        const std::size_t num_new = batches > 0 ? output.num_elements() / batches : 0;
        if (batches == 0 || num_new == 0 || num_new > window) {
            throw anira::Error(ANIRA_ERROR_CONFIG,
                               "HybridNNStage: the model is not [batches, 1, window] -> "
                               "[batches, new samples]");
        }
        return std::make_unique<Windows>(batches, window - num_new, num_new);
    }

private:
    class Windows final : public Prepared {
    public:
        Windows(uint32_t batches, std::size_t num_old, std::size_t num_new) noexcept
            : m_batches(batches), m_num_old(num_old), m_num_new(num_new) {}

        anira_status pre_process(anira::StageContext& ctx) noexcept override {
            anira::RingView ring;
            const anira_status has_ring = ctx.input_ring(0, ring);
            if (has_ring != ANIRA_OK) { return has_ring; }
            anira::Tensor tensor;
            const anira_status exposed = ctx.input_tensor(0, tensor);
            if (exposed != ANIRA_OK) { return exposed; }
            float* samples = tensor.data_f32();
            if (samples == nullptr) { return ANIRA_ERROR_CONFIG; }
            const std::span<float> windows(samples, tensor.num_elements());
            const std::size_t written =
                ring.pop_windows(0, windows, m_num_new, m_num_old, 0, m_batches);
            return written == 0 ? ANIRA_ERROR_CONFIG : ANIRA_OK;
        }

    private:
        uint32_t m_batches;
        std::size_t m_num_old;
        std::size_t m_num_new;
    };
};

#endif  // ANIRA_EXTRAS_HYBRIDNNSTAGE_H
