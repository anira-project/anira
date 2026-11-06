#ifndef ANIRA_CONTENTHASH_H
#define ANIRA_CONTENTHASH_H

#include <cstddef>
#include <cstdint>

/**
 * @file ContentHash.h
 * @brief A 64-bit hash of a byte range's content: XXH64 (Yann Collet's xxHash, 64-bit, seed 0).
 *
 * What tells two models loaded from bytes apart in the model's figure (FigureKey): identical
 * content is one model whoever owns the bytes, and a buffer whose address is reused for other
 * bytes never matches. Computed once per bytes object (capi::BytesCarrier::content_hash), at
 * prepare, on the main thread: four 64-bit lanes over 32-byte stripes, several GB/s, so a
 * 126 MB model takes a few tens of milliseconds once. Not cryptographic: two different models
 * of the same size share a figure only on a 64-bit collision, about 2^-64 per pair of models.
 */

namespace anira {

/// XXH64 of the `size` bytes at `data` with seed 0; `data` may be NULL for size 0.
uint64_t content_hash(const void* data, size_t size) noexcept;

}  // namespace anira

#endif  // ANIRA_CONTENTHASH_H
