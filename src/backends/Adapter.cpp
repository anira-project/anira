#include "Adapter.h"

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <algorithm>
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
    // By position for every slot until do_prepare says otherwise (set_bindings).
    m_bindings.m_inputs.assign(m_model.m_inputs.size(), ANIRA_BINDING_POSITION);
    m_bindings.m_outputs.assign(m_model.m_outputs.size(), ANIRA_BINDING_POSITION);
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

namespace {

// "'a', 'b'" of the engine's names; "none" for an empty side, and what a side without names
// says of itself (every name empty: it binds by position alone).
std::string list_names(const std::vector<std::string>& names) {
    std::string text;
    bool any_named = false;
    for (const std::string& name : names) {
        any_named = any_named || !name.empty();
        if (!text.empty()) { text += ", "; }
        text += "'" + name + "'";
    }
    if (names.empty()) { return "none"; }
    return any_named ? text : "unnamed: the side binds by position alone";
}

// "[1, 2, 64]"; a negative extent prints as it is (the engine's mark of a dynamic axis).
std::string shape_text(const std::vector<int64_t>& dims) {
    std::string text = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) { text += ", "; }
        text += std::to_string(dims[i]);
    }
    return text + "]";
}

std::string hex_dtype(anira_dtype dtype) {
    std::array<char, 16> code{};
    static_cast<void>(
        std::snprintf(code.data(), code.size(), "0x%x", static_cast<unsigned int>(dtype)));
    return code.data();
}

// "onnxruntime: input slot 0 'audio_in': ".
std::string at_slot(const char* engine, const char* side, size_t slot, const TensorInfo& info) {
    return std::string(engine) + ": " + side + " slot " + std::to_string(slot) + " '" +
           info.m_name + "': ";
}

}  // namespace

std::vector<SlotBinding> bind_slots(const std::vector<TensorInfo>& slots,
                                    const std::vector<std::string>& engine_names,
                                    size_t required,
                                    const char* engine,
                                    const char* side) {
    std::vector<SlotBinding> bindings(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        const TensorInfo& slot = slots[i];
        const bool named = !slot.m_engine_name.empty();
        const std::string& wanted = named ? slot.m_engine_name : slot.m_name;
        // A name binds where the side has it: the record's, else the canonical one.
        const auto found =
            wanted.empty() ? engine_names.end() : std::ranges::find(engine_names, wanted);
        if (found != engine_names.end()) {
            bindings[i] = SlotBinding{.m_index = static_cast<size_t>(found - engine_names.begin()),
                                      .m_binding = ANIRA_BINDING_NAME};
            continue;
        }
        if (named) {
            throw StatusError(ANIRA_ERROR_CONFIG,
                              at_slot(engine, side, i, slot) + "the tensors record names '" +
                                  wanted + "', which the " + engine + " model's " + side +
                                  "s do not have (they are " + list_names(engine_names) + ")");
        }
        if (i >= engine_names.size()) {
            throw StatusError(ANIRA_ERROR_CONFIG,
                              at_slot(engine, side, i, slot) + "binds by position, and the " +
                                  engine + " model has " + std::to_string(engine_names.size()) +
                                  " " + side + "(s); the model config declares " +
                                  std::to_string(slots.size()));
        }
        bindings[i] = SlotBinding{.m_index = i, .m_binding = ANIRA_BINDING_POSITION};
    }
    // Exactly once: no engine tensor bound by two slots, and none below `required` unbound.
    std::vector<size_t> bound_by(engine_names.size(), SIZE_MAX);
    for (size_t i = 0; i < bindings.size(); ++i) {
        const size_t index = bindings[i].m_index;
        if (bound_by[index] != SIZE_MAX) {
            const size_t first = bound_by[index];
            const auto how = [&bindings](size_t slot) {
                return bindings[slot].m_binding == ANIRA_BINDING_NAME ? "by name" : "by position";
            };
            throw StatusError(
                ANIRA_ERROR_CONFIG,
                std::string(engine) + ": " + side + " slots " + std::to_string(first) + " '" +
                    slots[first].m_name + "' (" + how(first) + ") and " + std::to_string(i) + " '" +
                    slots[i].m_name + "' (" + how(i) + ") both bind the " + engine + " model's " +
                    side + " '" + engine_names[index] + "' (index " + std::to_string(index) +
                    "); name the slots in the entry's tensors record");
        }
        bound_by[index] = i;
    }
    for (size_t index = 0; index < required && index < bound_by.size(); ++index) {
        if (bound_by[index] != SIZE_MAX) { continue; }
        throw StatusError(ANIRA_ERROR_CONFIG,
                          std::string(engine) + ": the " + engine + " model's " + side + " '" +
                              engine_names[index] + "' (index " + std::to_string(index) +
                              ") is bound by no slot; the model config declares " +
                              std::to_string(slots.size()) + " " + side + "(s) for " +
                              std::to_string(required) + " of the model's");
    }
    return bindings;
}

void check_engine_tensor(const TensorInfo& slot,
                         const EngineTensor& engine_tensor,
                         ExtentRule rule,
                         const char* engine,
                         const char* side) {
    const std::string where = std::string(engine) + ": " + side + " tensor '" + slot.m_name +
                              "' bound to the model's " +
                              (engine_tensor.m_name.empty() ? std::string("unnamed ") + side
                                                            : "'" + engine_tensor.m_name + "'") +
                              ": ";
    if (engine_tensor.m_dtype != slot.m_dtype) {
        throw StatusError(ANIRA_ERROR_CONFIG,
                          where + "the model's element type is " + engine_tensor.m_type_word +
                              ", the spec's dtype " + hex_dtype(slot.m_dtype));
    }
    const std::string shapes = "the model's tensor is " + shape_text(engine_tensor.m_dims) +
                               ", the spec's engine dims are " + shape_text(slot.m_dims);
    if (engine_tensor.m_dims.size() != slot.m_dims.size()) {
        throw StatusError(ANIRA_ERROR_CONFIG,
                          where + shapes +
                              " (the ranks differ; a layout in the entry's tensors record "
                              "re-views axes of extent 1)");
    }
    for (size_t axis = 0; axis < slot.m_dims.size(); ++axis) {
        const int64_t have = engine_tensor.m_dims[axis];
        const int64_t want = slot.m_dims[axis];
        const bool ok = rule == ExtentRule::UpperBound ? want <= have : have < 0 || have == want;
        if (ok) { continue; }
        throw StatusError(
            ANIRA_ERROR_CONFIG,
            where + shapes + " (axis " + std::to_string(axis) + ": " + std::to_string(have) +
                (rule == ExtentRule::UpperBound ? " is the planned upper bound of " : " against ") +
                std::to_string(want) + ")");
    }
}

std::vector<std::string> names_of(const std::vector<EngineTensor>& engine_tensors) {
    std::vector<std::string> names;
    names.reserve(engine_tensors.size());
    for (const EngineTensor& tensor : engine_tensors) { names.push_back(tensor.m_name); }
    return names;
}

std::vector<SlotBinding> bind_side(const std::vector<TensorInfo>& slots,
                                   const std::vector<EngineTensor>& engine_tensors,
                                   size_t required,
                                   ExtentRule rule,
                                   const char* engine,
                                   const char* side) {
    const std::vector<SlotBinding> bindings =
        bind_slots(slots, names_of(engine_tensors), required, engine, side);
    for (size_t i = 0; i < slots.size(); ++i) {
        check_engine_tensor(slots[i], engine_tensors[bindings[i].m_index], rule, engine, side);
    }
    return bindings;
}

Bindings bindings_of(const std::vector<SlotBinding>& inputs,
                     const std::vector<SlotBinding>& outputs) {
    Bindings bindings;
    bindings.m_inputs.reserve(inputs.size());
    bindings.m_outputs.reserve(outputs.size());
    for (const SlotBinding& binding : inputs) { bindings.m_inputs.push_back(binding.m_binding); }
    for (const SlotBinding& binding : outputs) { bindings.m_outputs.push_back(binding.m_binding); }
    return bindings;
}

}  // namespace anira::backend
