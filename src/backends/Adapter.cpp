#include "Adapter.h"

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../utils/StatusError.h"
#include "ProcessingGuard.h"

namespace anira::backend {

bool Model::operator==(const Model& other) const {
    return m_engine == other.m_engine && m_engine_id == other.m_engine_id &&
           m_path == other.m_path && m_bytes == other.m_bytes && m_num_bytes == other.m_num_bytes &&
           m_entry == other.m_entry && m_inputs == other.m_inputs && m_outputs == other.m_outputs &&
           m_instances == other.m_instances && m_warm_up == other.m_warm_up &&
           m_session_exclusive == other.m_session_exclusive;
}

void Adapter::prepare(const Model& model) {
    // A second prepare starts over: nothing of a failed or earlier one is kept.
    m_num_instances = 0;
    m_busy.clear();
    m_model = model;
    do_prepare(m_model);
    const uint32_t instances = m_model.m_instances > 0 ? m_model.m_instances : 1U;
    // Value-initialised: every flag starts clear. Sized once here and never grown (an atomic
    // cannot be moved).
    m_busy = std::vector<std::atomic<bool>>(instances);
    m_num_instances = instances;
}

anira_status Adapter::run(const anira_engine_ctx& ctx,
                          ChunkBuffers* chunk,
                          bool reset_first) noexcept {
    if (m_num_instances == 0) { return ANIRA_ERROR_INVALID_STATE; }
    anira_engine_ctx call = ctx;
    if (!claims_instances()) {
        call.instance = 0;
        if (reset_first) { reset(call); }
        return process(call, chunk);
    }
    // The spin-and-claim loop of the 2.x processors: the first instance whose busy flag was
    // clear runs this call, and the flag is released on every path out of it.
    while (true) {
        for (uint32_t i = 0; i < m_num_instances; ++i) {
            if (m_busy[i].exchange(true)) { continue; }
            const detail::ProcessingGuard guard(m_busy[i]);
            call.instance = i;
            if (reset_first) { reset(call); }
            return process(call, chunk);
        }
    }
}

float* host_f32_packed(const anira_tensor& tensor, size_t expected) noexcept {
    if (tensor.domain != static_cast<uint32_t>(ANIRA_DOMAIN_HOST) &&
        tensor.domain != static_cast<uint32_t>(ANIRA_DOMAIN_HOST_PINNED)) {
        return nullptr;
    }
    if (tensor.dtype != ANIRA_DTYPE_F32) { return nullptr; }
    if ((tensor.flags & static_cast<uint32_t>(ANIRA_TENSOR_PLANAR)) != 0U) { return nullptr; }
    if (tensor.ndim > ANIRA_MAX_RANK) { return nullptr; }
    size_t elements = 1;
    for (uint32_t axis = 0; axis < tensor.ndim; ++axis) {
        if (tensor.shape[axis] < 0) { return nullptr; }
        elements *= static_cast<size_t>(tensor.shape[axis]);
    }
    if (elements != expected) { return nullptr; }
    // Packed row-major: all-zero strides (this record's spelling of packed) or the row-major
    // strides themselves, an axis of extent 1 excepted (it is never stepped).
    bool all_zero = true;
    for (uint32_t axis = 0; axis < tensor.ndim; ++axis) {
        all_zero = all_zero && tensor.strides[axis] == 0;
    }
    if (!all_zero) {
        int64_t stride = 1;
        for (uint32_t axis = tensor.ndim; axis-- > 0;) {
            if (tensor.shape[axis] != 1 && tensor.strides[axis] != stride) { return nullptr; }
            stride *= tensor.shape[axis];
        }
    }
    if (tensor.handle.host.ptr == nullptr) { return nullptr; }
    auto* base = static_cast<unsigned char*>(tensor.handle.host.ptr);
    return reinterpret_cast<float*>(  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        base + static_cast<size_t>(tensor.byte_offset));
}

void require_f32(const Model& model, const char* engine) {
    const auto check = [engine](const std::vector<TensorInfo>& tensors, const char* side) {
        for (const TensorInfo& tensor : tensors) {
            if (tensor.m_dtype == ANIRA_DTYPE_F32) { continue; }
            std::array<char, 16> code{};
            static_cast<void>(std::snprintf(code.data(),
                                            code.size(),
                                            "0x%x",
                                            static_cast<unsigned int>(tensor.m_dtype)));
            throw StatusError(ANIRA_ERROR_CONFIG,
                              std::string(engine) + ": " + side + " tensor '" + tensor.m_name +
                                  "' has dtype " + code.data() + "; the " + engine +
                                  " adapter binds float32 tensors alone in this pre-release");
        }
    };
    check(model.m_inputs, "input");
    check(model.m_outputs, "output");
}

}  // namespace anira::backend
