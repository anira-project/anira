#ifndef ANIRA_MEMORYBLOCK_H
#define ANIRA_MEMORYBLOCK_H

#include <tanh/core/MemoryBlock.h>

namespace anira {

/**
 * @brief Contiguous, resizable memory block
 *
 * Provided by tanh-lib's Apache-2.0 core component (thl::core). Supports
 * non-copyable element types (e.g. std::atomic<float>) via raw storage
 * management, element access through operator[], data(), resize(), and
 * zero-copy swap_data().
 *
 * @see Buffer, PrePostProcessor
 */
template <typename T>
using MemoryBlock = thl::core::MemoryBlock<T>;

}  // namespace anira

#endif  // ANIRA_MEMORYBLOCK_H
