#ifndef ANIRA_MEMORYBLOCK_H
#define ANIRA_MEMORYBLOCK_H

#include <iostream>
#include <type_traits>

#include <anira/utils/Logger.h>

namespace anira {

/**
 * @brief Template class for managing contiguous memory blocks with automatic resource management
 * 
 * The MemoryBlock class provides a low-level, efficient container for managing contiguous
 * memory allocation with automatic cleanup and move semantics. It serves as a foundation
 * for higher-level data structures like Buffer.
 * 
 * Key features:
 * - Automatic memory management with proper cleanup
 * - Move semantics for efficient memory transfers
 * - Zero-copy data swapping for trivially copyable types
 * - Direct memory access with array-style indexing
 * - Resize capabilities with memory reallocation
 * - Template-based design supporting any data type
 * 
 * This class is designed for performance-critical applications where direct memory
 * control is needed while maintaining memory safety.
 * 
 * @tparam T The data type to store in the memory block
 * 
 * @note This class uses malloc/free for memory management to allow for efficient
 *       reallocation operations and to avoid constructor/destructor calls for POD types.
 * 
 * @see Buffer
 */
template <typename T>
class MemoryBlock {
public:
    /**
     * @brief Constructor that allocates a memory block of specified size
     * 
     * Allocates contiguous memory for the specified number of elements.
     * If allocation fails, an error is logged and the block remains in an invalid state.
     * 
     * @param size Number of elements to allocate (default: 0 for empty block)
     * 
     * @note Memory is allocated but not initialized. For non-POD types, consider
     *       using appropriate initialization after construction.
     */
    MemoryBlock(std::size_t size = 0) : m_size(size) {
        void* data = malloc(sizeof(T) * m_size);
        if (data != nullptr) {
            m_data = (T*) data;
        } else {
            LOG_ERROR << "Failed to allocate memory!" << std::endl;
        }
    }

    /**
     * @brief Destructor that automatically frees allocated memory
     * 
     * Safely deallocates the memory block using free(). Marked noexcept to
     * guarantee no exceptions during destruction, which is essential for RAII.
     */
    ~MemoryBlock() noexcept {
        free(m_data);
    }

    /**
     * @brief Copy constructor that creates a deep copy of another memory block
     * 
     * Allocates new memory and copies all data from the source block.
     * If allocation fails, an error is logged and the block remains in an invalid state.
     * 
     * @param other The source memory block to copy from
     */
    MemoryBlock(const MemoryBlock& other) : m_size(other.m_size) {
        void* data = malloc(sizeof(T) * m_size);
        if (data != nullptr) {
            m_data = (T*) data;
            memcpy(m_data, other.m_data, sizeof(T) * m_size);
        } else {
            LOG_ERROR << "Failed to allocate memory!" << std::endl;
        }
    }

    /**
     * @brief Copy assignment operator that replaces this block's content with another's data
     * 
     * Deallocates existing memory, allocates new memory matching the source size,
     * and copies all data. Self-assignment is safely handled.
     * 
     * @param other The source memory block to copy from
     * @return Reference to this memory block after the copy operation
     */
    MemoryBlock& operator=(const MemoryBlock& other) {
        if (this != &other) {
            free(m_data);
            m_size = other.m_size;
            void* data = malloc(sizeof(T) * m_size);
            if (data != nullptr) {
                m_data = (T*) data;
                memcpy(m_data, other.m_data, sizeof(T) * m_size);
            } else {
                LOG_ERROR << "Failed to allocate memory!" << std::endl;
            }
        }
        return *this;
    }

    /**
     * @brief Move constructor that transfers ownership from another memory block
     * 
     * Efficiently transfers the memory ownership from the source block to this block
     * without copying data. The source block is left in a valid but empty state.
     * 
     * @param other The source memory block to move from (will be left empty)
     * 
     * @note Marked noexcept to guarantee no exceptions, essential for move semantics
     */
    MemoryBlock(MemoryBlock&& other) noexcept : m_size(other.m_size), m_data(other.m_data) {
        other.m_size = 0;
        other.m_data = nullptr;
    }

    /**
     * @brief Move assignment operator that transfers ownership from another memory block
     * 
     * Deallocates any existing memory, then efficiently transfers ownership from
     * the source block. The source block is left in a valid but empty state.
     * 
     * @param other The source memory block to move from (will be left empty)
     * @return Reference to this memory block after the move operation
     * 
     * @note Marked noexcept to guarantee no exceptions, essential for move semantics
     */
    MemoryBlock& operator=(MemoryBlock&& other) noexcept {
        if (this != &other) {
            free(m_data);
            m_size = other.m_size;
            m_data = other.m_data;
            other.m_size = 0;
            other.m_data = nullptr;
        }
        return *this;
    }

    /**
     * @brief Array subscript operator for mutable element access
     * 
     * Provides direct access to elements in the memory block using array-style indexing.
     * No bounds checking is performed for performance reasons.
     * 
     * @param index The index of the element to access (0-based)
     * @return Reference to the element at the specified index
     * 
     * @warning No bounds checking is performed. Accessing out-of-bounds indices
     *          results in undefined behavior.
     */
    T& operator[](size_t index) {
        return m_data[index];
    }

    /**
     * @brief Array subscript operator for const element access
     * 
     * Provides direct read-only access to elements in the memory block using
     * array-style indexing. No bounds checking is performed for performance reasons.
     * 
     * @param index The index of the element to access (0-based)
     * @return Const reference to the element at the specified index
     * 
     * @warning No bounds checking is performed. Accessing out-of-bounds indices
     *          results in undefined behavior.
     */
    const T& operator[](size_t index) const {
        return m_data[index];
    }

    /**
     * @brief Gets a pointer to the raw memory data
     * 
     * Returns a direct pointer to the underlying memory block. This can be used
     * for interfacing with C APIs or for performance-critical operations that
     * need direct memory access.
     * 
     * @return Pointer to the first element in the memory block
     */
    T* data() {
        return m_data;
    }

    /**
     * @brief Gets the number of elements in the memory block
     * 
     * @return The number of elements currently allocated in the memory block
     */
    size_t size() const {
        return m_size;
    }

    /**
     * @brief Resizes the memory block to a new size
     * 
     * Changes the size of the memory block, potentially reallocating memory.
     * If the new size is larger, the additional memory is uninitialized.
     * If smaller, data beyond the new size is lost.
     * 
     * @param size New number of elements to allocate
     * 
     * @note This operation may invalidate existing pointers to the data.
     *       For performance, realloc is used when possible to avoid unnecessary copying.
     */
    void resize(size_t size) {
        m_size = size;
        void* data;
        if (m_size > 0) {
            data = realloc(m_data, sizeof(T) * size);
        } else {
            free(m_data);
            data = malloc(sizeof(T) * size);
        }

        if (data != nullptr) {
            m_data = (T*) data;
        } else {
            LOG_ERROR << "Failed to reallocate memory!" << std::endl;
        }
    }

    /**
     * @brief Clears the memory block by setting all bytes to zero
     * 
     * Efficiently zeros out all memory in the block using memset. This is a
     * fast operation suitable for clearing numerical data or preparing memory
     * for fresh use.
     * 
     * @note This operation sets all bytes to zero, which may not be appropriate
     *       for all data types (e.g., types with non-trivial constructors).
     */
    void clear() {
        memset(m_data, 0, sizeof(T) * m_size);
    }

    /**
     * @brief Swaps data with another memory block without copying (for trivially copyable types)
     * 
     * Efficiently exchanges the data pointers between this block and another block
     * of the same size. This is a zero-copy operation that only swaps pointers.
     * The function is only available for trivially copyable types to ensure safety.
     * 
     * Template parameter U is used with SFINAE (Substitution Failure Is Not An Error)
     * to enable this function only for trivially copyable types, ensuring memory
     * safety for complex types that may have special copying requirements.
     * 
     * @tparam U Template parameter defaulting to T, used for SFINAE type checking
     * @param other The memory block to swap data with
     * 
     * @note Both blocks must have identical sizes for the swap to succeed.
     *       This function is only available for trivially copyable types.
     */
    template <typename U = T, std::enable_if_t<std::is_trivially_copyable_v<U>, bool> = true>
    void swap_data(MemoryBlock& other) {
        if (this != &other) {
            if (m_size == other.m_size) {
                std::swap(m_data, other.m_data);
            } else {
                LOG_ERROR << "Cannot swap data with different sizes!" << std::endl;
            }
        }
    }

    /**
     * @brief Swaps data with a raw memory pointer without copying
     * 
     * Efficiently exchanges the internal data pointer with the provided raw pointer.
     * This is a zero-copy operation that transfers ownership of the memory.
     * 
     * @param data Reference to the raw memory pointer to swap with
     * @param size Size of the raw memory in number of elements
     * 
     * @note The provided memory size must match this block's current size.
     *       After the swap, the caller assumes ownership of this block's original memory.
     */
    void swap_data(T*& data, size_t size) {
        if (m_size == size) {
            std::swap(m_data, data);
        } else {
            LOG_ERROR << "Cannot swap data with different sizes!" << std::endl;
        }
    }
        
private:
    T* m_data = nullptr;      ///< Pointer to the allocated memory block
    size_t m_size;            ///< Number of elements in the memory block
};

} // namespace anira

#endif // ANIRA_MEMORYBLOCK_H