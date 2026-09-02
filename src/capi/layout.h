/*
 * The per-entry axis layout of section 5: how one engine's file holds a tensor's axes when
 * that order differs from the spec's. Shared by the C entry (validation at set time), the JSON
 * loader and the version 2 upgrade (stable fill), and the translator (classification).
 */
#ifndef ANIRA_CAPI_LAYOUT_H
#define ANIRA_CAPI_LAYOUT_H

#include <anira/system/Exports.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "handles.h"

namespace anira::capi {

enum class LayoutKind {
    Identity,   ///< The spec's own order (or no layout).
    View,       ///< Only axes of extent 1 move: the same bytes with other dims.
    Transpose,  ///< An axis of another extent moves: a copy per inference (not at M1).
    Invalid,  ///< Does not fit the spec: rank, a spec axis out of range, a material axis left out.
};

/// Set-time shape of a layout, independent of any spec: 1..ANIRA_MAX_RANK entries, each
/// ANIRA_AXIS_INSERT or below ANIRA_MAX_RANK, no spec axis twice. `why` receives the reason.
ANIRA_API bool valid_layout_shape(const std::vector<uint32_t>& axes, std::string* why);

/// Classifies a layout against its spec; `why` receives the reason for Invalid.
ANIRA_API LayoutKind classify_layout(const anira_tensor_spec& spec,
                                     const std::vector<uint32_t>& axes,
                                     std::string* why);

/// The dims of the engine's tensor: the extent of spec axis axes[k], or 1 for
/// ANIRA_AXIS_INSERT. An empty layout yields the spec's own extents.
ANIRA_API std::vector<int64_t> engine_dims(const anira_tensor_spec& spec,
                                           const std::vector<uint32_t>& axes);

/// The layout that turns `canonical` dims into `engine` dims by moving axes of extent 1 only
/// (material extents matched in order, unit axes filling the remaining positions in canonical
/// order, ANIRA_AXIS_INSERT once they are exhausted), or nullopt when the material extents
/// differ. Identity is returned as such (never empty) when the dims are equal.
ANIRA_API std::optional<std::vector<uint32_t>> stable_fill_layout(
    const std::vector<int64_t>& canonical,
    const std::vector<int64_t>& engine);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_LAYOUT_H
