#ifndef ANIRA_CAPI_ENUMERATE_H
#define ANIRA_CAPI_ENUMERATE_H
/*
 * The enumeration convention of section 6a, shared by the context's capabilities entries
 * (context.cpp) and the pipeline's (handler.cpp): out == NULL asks for the count, a short
 * buffer is filled as far as it goes and returns ANIRA_INCOMPLETE. Records are written at the
 * caller's stride, min(element_size, the library's record size) bytes each, so the row's
 * struct_size tells a newer caller how much of its record the library filled.
 */
#include <anira/abi/status.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace anira::capi {

template <class T>
anira_status enumerate_records(const std::vector<T>& rows,
                               uint32_t element_size,
                               uint32_t* count,
                               void* out) {
    if (count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto total = static_cast<uint32_t>(rows.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    if (element_size < sizeof(uint32_t)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    const size_t bytes = std::min<size_t>(element_size, sizeof(T));
    auto* destination = static_cast<unsigned char*>(out);
    for (uint32_t i = 0; i < written; ++i) {
        std::memcpy(destination + static_cast<size_t>(i) * element_size, &rows[i], bytes);
    }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
}

template <class T>
anira_status enumerate_scalars(const std::vector<T>& rows, uint32_t* count, T* out) {
    if (count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto total = static_cast<uint32_t>(rows.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t i = 0; i < written; ++i) { out[i] = rows[i]; }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
}

}  // namespace anira::capi

#endif  // ANIRA_CAPI_ENUMERATE_H
