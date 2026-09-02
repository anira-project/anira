#include "layout.h"

#include <anira/abi/enums.h>
#include <anira/abi/status.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "handles.h"  // IWYU pragma: keep - the body of anira_tensor_spec

namespace anira::capi {

namespace {

bool material(int64_t extent) {
    return extent != 1;  // ANIRA_DYNAMIC counts: its runtime extent is unknown
}

}  // namespace

bool valid_layout_shape(const std::vector<uint32_t>& axes, std::string* why) {
    if (axes.empty() || axes.size() > ANIRA_MAX_RANK) {
        if (why != nullptr) {
            *why = "a layout has 1 to " + std::to_string(ANIRA_MAX_RANK) + " entries";
        }
        return false;
    }
    std::array<bool, ANIRA_MAX_RANK> seen{};
    for (size_t k = 0; k < axes.size(); ++k) {
        const uint32_t axis = axes[k];
        if (axis == ANIRA_AXIS_INSERT) { continue; }
        if (axis >= ANIRA_MAX_RANK) {
            if (why != nullptr) {
                *why = "entry " + std::to_string(k) + " is not a spec axis index or \"insert\"";
            }
            return false;
        }
        if (seen[axis]) {
            if (why != nullptr) { *why = "spec axis " + std::to_string(axis) + " is listed twice"; }
            return false;
        }
        seen[axis] = true;
    }
    return true;
}

LayoutKind classify_layout(const anira_tensor_spec& spec,
                           const std::vector<uint32_t>& axes,
                           std::string* why) {
    if (axes.empty()) { return LayoutKind::Identity; }
    if (!valid_layout_shape(axes, why)) { return LayoutKind::Invalid; }
    std::array<bool, ANIRA_MAX_RANK> listed{};
    for (const uint32_t axis : axes) {
        if (axis == ANIRA_AXIS_INSERT) { continue; }
        if (axis >= spec.m_ndim) {
            if (why != nullptr) {
                *why = "spec axis " + std::to_string(axis) + " does not exist; the spec has " +
                       std::to_string(spec.m_ndim) + " axes";
            }
            return LayoutKind::Invalid;
        }
        listed[axis] = true;
    }
    for (uint32_t i = 0; i < spec.m_ndim; ++i) {
        if (!listed[i] && material(spec.m_axes[i].m_extent)) {
            if (why != nullptr) {
                *why = "spec axis " + std::to_string(i) + " (extent " +
                       std::to_string(spec.m_axes[i].m_extent) + ") is left out";
            }
            return LayoutKind::Invalid;
        }
    }
    // Identity: the spec's own order, nothing inserted.
    bool identity = axes.size() == spec.m_ndim;
    for (size_t k = 0; identity && k < axes.size(); ++k) { identity = axes[k] == k; }
    if (identity) { return LayoutKind::Identity; }
    // A view iff the material spec axes appear in increasing order.
    uint32_t last_material = 0;
    bool any_material = false;
    for (const uint32_t axis : axes) {
        if (axis == ANIRA_AXIS_INSERT || !material(spec.m_axes[axis].m_extent)) { continue; }
        if (any_material && axis < last_material) { return LayoutKind::Transpose; }
        last_material = axis;
        any_material = true;
    }
    return LayoutKind::View;
}

std::vector<int64_t> engine_dims(const anira_tensor_spec& spec, const std::vector<uint32_t>& axes) {
    std::vector<int64_t> dims;
    if (axes.empty()) {
        dims.reserve(spec.m_ndim);
        for (uint32_t i = 0; i < spec.m_ndim; ++i) { dims.push_back(spec.m_axes[i].m_extent); }
        return dims;
    }
    dims.reserve(axes.size());
    for (const uint32_t axis : axes) {
        dims.push_back(
            axis == ANIRA_AXIS_INSERT || axis >= spec.m_ndim ? 1 : spec.m_axes[axis].m_extent);
    }
    return dims;
}

std::optional<std::vector<uint32_t>> stable_fill_layout(const std::vector<int64_t>& canonical,
                                                        const std::vector<int64_t>& engine) {
    if (canonical.empty() || engine.empty() || canonical.size() > ANIRA_MAX_RANK ||
        engine.size() > ANIRA_MAX_RANK) {
        return std::nullopt;
    }
    std::vector<uint32_t> axes(engine.size(), ANIRA_AXIS_INSERT);
    // Material extents must appear in the same order with the same values.
    size_t c = 0;
    for (size_t k = 0; k < engine.size(); ++k) {
        if (!material(engine[k])) { continue; }
        while (c < canonical.size() && !material(canonical[c])) { ++c; }
        if (c == canonical.size() || canonical[c] != engine[k]) { return std::nullopt; }
        axes[k] = static_cast<uint32_t>(c);
        ++c;
    }
    for (; c < canonical.size(); ++c) {
        if (material(canonical[c])) { return std::nullopt; }  // a material axis the engine lacks
    }
    // Unit positions take the canonical unit axes in order, then ANIRA_AXIS_INSERT.
    size_t next_unit = 0;
    for (size_t k = 0; k < engine.size(); ++k) {
        if (material(engine[k])) { continue; }
        while (next_unit < canonical.size() && material(canonical[next_unit])) { ++next_unit; }
        if (next_unit < canonical.size()) {
            axes[k] = static_cast<uint32_t>(next_unit);
            ++next_unit;
        }
    }
    return axes;
}

}  // namespace anira::capi
