#ifndef ANIRA_AUDIO_BUFFER_H
#define ANIRA_AUDIO_BUFFER_H

#include <iostream>
#include <cstring>
#include <anira/utils/MemoryBlock.h>
#include <anira/system/AniraWinExports.h>

#include <anira/utils/Logger.h>

namespace anira {

/**
 * @brief Template class for managing multi-channel audio buffers with efficient memory layout
 * 
 * The Buffer class provides a high-performance, multi-channel audio buffer implementation
 * optimized for real-time audio processing. It manages memory in a contiguous block with
 * channel pointers for efficient access patterns, making it suitable for low-latency
 * audio applications and neural network inference preprocessing.
 * 
 * Key features:
 * - Contiguous memory layout for cache-friendly access patterns
 * - Channel-based indexing with both read and write pointer access
 * - Move semantics for efficient buffer transfers
 * - Memory swapping capabilities for zero-copy operations
 * - Template-based design supporting various numeric types
 * - Real-time safe operations (no allocations in critical paths)
 * 
 * Memory layout: [Channel0_Sample0, Channel0_Sample1, ..., Channel1_Sample0, Channel1_Sample1, ...]
 * 
 * @tparam T The data type for audio samples (typically float or double)
 * 
 * @note This class is designed for real-time audio processing and avoids memory
 *       allocations in performance-critical operations where possible.
 * 
 * @see MemoryBlock
 */
template <typename T>
class ANIRA_API Buffer
{
public:
    /**
     * @brief Default constructor creates an empty buffer with no allocated memory
     * 
     * Creates a buffer with zero channels and zero size. Memory allocation is deferred
     * until the buffer is resized or initialized with specific dimensions.
     */
    Buffer() = default;
    
    /**
     * @brief Constructor that creates a buffer with specified dimensions
     * 
     * Allocates memory for the specified number of channels and samples, initializes
     * channel pointers, and clears all data to zero.
     * 
     * @param num_channels Number of audio channels to allocate
     * @param size Number of samples per channel
     */
    Buffer(size_t num_channels, size_t size) : m_num_channels(num_channels), m_size(size), m_data(num_channels * size) {
        malloc_channels();
        clear();
    }

    /**
     * @brief Copy constructor that creates a deep copy of another buffer
     * 
     * Creates a new buffer with the same dimensions as the source buffer and copies
     * all audio data. The new buffer has its own independent memory allocation.
     * 
     * @param other The source buffer to copy from
     */
    Buffer(const Buffer& other) : m_num_channels(other.m_num_channels), m_size(other.m_size), m_data(other.m_data) {
        if (m_num_channels == 0 || m_size == 0) {
            return;
        } else {
            malloc_channels();
        }
    }

    /**
     * @brief Move constructor that transfers ownership from another buffer
     * 
     * Efficiently transfers all resources from the source buffer to this buffer,
     * leaving the source buffer in a valid but empty state. This operation is
     * optimized for performance and does not involve data copying.
     * 
     * @param other The source buffer to move from (will be left empty)
     * 
     * @note Marked noexcept to guarantee no exceptions are thrown, which is
     *       essential for move semantics and container operations
     */
    Buffer(Buffer&& other) noexcept : m_num_channels(other.m_num_channels), m_size(other.m_size), m_data(std::move(other.m_data)), m_channels(other.m_channels) {
        other.m_num_channels = 0;
        other.m_size = 0;
        other.m_channels = nullptr;
    }

    /**
     * @brief Destructor that cleans up allocated channel pointer memory
     * 
     * Automatically frees the memory allocated for channel pointers. The underlying
     * audio data memory is managed by the MemoryBlock and cleaned up automatically.
     */
    ~Buffer() {
        free(m_channels);
    }

    /**
     * @brief Copy assignment operator that replaces this buffer's content with another buffer's data
     * 
     * Performs a deep copy of the source buffer's data and dimensions. Any existing
     * data in this buffer is replaced. Self-assignment is safely handled.
     * 
     * @param other The source buffer to copy from
     * @return Reference to this buffer after the copy operation
     */
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            free(m_channels);
            m_num_channels = other.m_num_channels;
            m_size = other.m_size;
            m_data = other.m_data;
            malloc_channels();
        }
        return *this;
    }

    /**
     * @brief Move assignment operator that transfers ownership from another buffer
     * 
     * Efficiently transfers all resources from the source buffer to this buffer,
     * replacing any existing content. The source buffer is left in a valid but empty state.
     * 
     * @param other The source buffer to move from (will be left empty)
     * @return Reference to this buffer after the move operation
     * 
     * @note Marked noexcept to guarantee no exceptions, essential for move semantics
     */
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            free(m_channels);
            m_num_channels = other.m_num_channels;
            m_size = other.m_size;
            m_data = std::move(other.m_data);
            m_channels = other.m_channels;
            other.m_num_channels = 0;
            other.m_size = 0;
            other.m_channels = nullptr;
        }
        return *this;
    }

    /**
     * @brief Resizes the buffer to new dimensions
     * 
     * Changes the buffer's channel count and sample count, reallocating memory as needed.
     * All existing data is lost during the resize operation. The buffer is not automatically
     * cleared after resizing.
     * 
     * @param num_channels New number of audio channels
     * @param size New number of samples per channel
     * 
     * @note This operation involves memory allocation and should not be called in real-time contexts
     */
    void resize(size_t num_channels, size_t size) {
        m_num_channels = num_channels;
        m_size = size;
        m_data.resize(num_channels * size);
        free(m_channels);
        malloc_channels();
    }

    /**
     * @brief Gets the number of channels in the buffer
     * 
     * @return The number of audio channels currently allocated in the buffer
     */
    size_t get_num_channels() const {
        return m_num_channels;
    }

    /**
     * @brief Gets the number of samples per channel in the buffer
     * 
     * @return The number of samples per channel currently allocated in the buffer
     */
    size_t get_num_samples() const {
        return m_size;
    }

    /**
     * @brief Gets a read-only pointer to the start of a specific channel's data
     * 
     * Returns a const pointer to the beginning of the specified channel's sample data.
     * This pointer can be used for read-only access to the channel's samples.
     * 
     * @param channel The channel index (0-based)
     * @return Const pointer to the first sample of the specified channel
     */
    const T* get_read_pointer(size_t channel) const {
        return m_channels[channel];
    }

    /**
     * @brief Gets a read-only pointer to a specific sample within a channel
     * 
     * Returns a const pointer to a specific sample position within the specified channel.
     * This is useful for accessing data starting from a particular sample offset.
     * 
     * @param channel The channel index (0-based)
     * @param sample_index The sample index within the channel (0-based)
     * @return Const pointer to the specified sample position
     */
    const T* get_read_pointer(size_t channel, size_t sample_index) const {
        return m_channels[channel] + sample_index;
    }

    /**
     * @brief Gets a writable pointer to the start of a specific channel's data
     * 
     * Returns a mutable pointer to the beginning of the specified channel's sample data.
     * This pointer can be used for both reading and writing to the channel's samples.
     * 
     * @param channel The channel index (0-based)
     * @return Mutable pointer to the first sample of the specified channel
     */
    T* get_write_pointer(size_t channel) {
        return m_channels[channel];
    }

    /**
     * @brief Gets a writable pointer to a specific sample within a channel
     * 
     * Returns a mutable pointer to a specific sample position within the specified channel.
     * This is useful for writing data starting from a particular sample offset.
     * 
     * @param channel The channel index (0-based)
     * @param sample_index The sample index within the channel (0-based)
     * @return Mutable pointer to the specified sample position
     */
    T* get_write_pointer(size_t channel, size_t sample_index) {
        return m_channels[channel] + sample_index;
    }

    /**
     * @brief Gets an array of read-only pointers for all channels
     * 
     * Returns an array of const pointers, where each pointer points to the start
     * of a channel's data. This is useful for interfacing with audio processing
     * functions that expect a channel pointer array format.
     * 
     * @return Array of const pointers to all channel data
     */
    const T* const* get_array_of_read_pointers() const {
        return const_cast<const T**>(m_channels);
    }

    /**
     * @brief Gets an array of writable pointers for all channels
     * 
     * Returns an array of mutable pointers, where each pointer points to the start
     * of a channel's data. This is useful for interfacing with audio processing
     * functions that need to modify channel data.
     * 
     * @return Array of mutable pointers to all channel data
     */
    T* const* get_array_of_write_pointers() {
        return m_channels;
    }

    /**
     * @brief Gets a pointer to the raw contiguous data block
     * 
     * Returns a pointer to the underlying contiguous memory block containing all
     * audio data. Data is organized as: [Channel0_Samples..., Channel1_Samples..., etc.]
     * 
     * @return Pointer to the raw data block
     */
    T* data() {
        return m_data.data();
    }

    /**
     * @brief Gets a reference to the underlying memory block
     * 
     * Provides direct access to the MemoryBlock object that manages the buffer's
     * contiguous memory allocation. Useful for advanced memory management operations.
     * 
     * @return Reference to the internal MemoryBlock
     */
    MemoryBlock<T>& get_memory_block() {
        return m_data;
    }

    /**
     * @brief Swaps data with another buffer without copying
     * 
     * Efficiently exchanges the data content between this buffer and another buffer
     * of the same dimensions. This is a zero-copy operation that only swaps pointers.
     * Both buffers must have identical channel count and sample count.
     * 
     * @param other The buffer to swap data with
     * 
     * @note Buffers must have identical dimensions for the swap to succeed
     */
    void swap_data(Buffer& other) {
        if (this != &other) {
            if( m_num_channels == other.m_num_channels && m_size == other.m_size) {
                m_data.swap_data(other.m_data);
                T** temp_channels = m_channels;
                m_channels = other.m_channels;
                other.m_channels = temp_channels;
            } else {
                LOG_ERROR << "Cannot swap data, buffers have different number of channels or sizes!" << std::endl;
            }
        }
    }

    /**
     * @brief Swaps data with a MemoryBlock without copying
     * 
     * Efficiently exchanges the data content between this buffer and a MemoryBlock.
     * The MemoryBlock must have the same total size as this buffer's allocated memory.
     * Channel pointers are automatically updated after the swap.
     * 
     * @param other The MemoryBlock to swap data with
     * 
     * @note The MemoryBlock size must match this buffer's total allocated size
     */
    void swap_data(MemoryBlock<T>& other) {
        if (other.size() == m_num_channels * m_size) {
            m_data.swap_data(other);
            reset_channel_ptr();
        } else {
            LOG_ERROR << "Cannot swap data, MemoryBlock has a different size!" << std::endl;
        }
    }

    /**
     * @brief Swaps data with a raw memory pointer without copying
     * 
     * Efficiently exchanges the data content between this buffer and raw memory.
     * The provided memory must have the same total size as this buffer's allocated memory.
     * Channel pointers are automatically updated after the swap.
     * 
     * @param data Pointer to the raw memory to swap with
     * @param size Size of the raw memory in number of elements
     * 
     * @note The memory size must match this buffer's total allocated size
     */
    void swap_data(T*& data, size_t size) {
        if (size == m_num_channels * m_size) {
            m_data.swap_data(data, size);
            reset_channel_ptr();
        } else {
            LOG_ERROR << "Cannot swap data, MemoryBlock has a different size!" << std::endl;
        }
    }

    /**
     * @brief Resets channel pointers to point to the correct memory locations
     * 
     * Updates all channel pointers to point to their correct positions within
     * the contiguous data block. This method is called automatically after
     * operations that might change the underlying memory layout.
     */
    void reset_channel_ptr() {
        for (size_t i = 0; i < m_num_channels; i++) {
            m_channels[i] = m_data.data() + i * m_size;
        }
    }

    /**
     * @brief Gets a single sample value from the buffer
     * 
     * Retrieves the sample value at the specified channel and sample position.
     * This method provides bounds-safe access to individual samples.
     * 
     * @param channel The channel index (0-based)
     * @param sample_index The sample index within the channel (0-based)
     * @return The sample value at the specified position
     */
    T get_sample(size_t channel, size_t sample_index) const {
        return m_channels[channel][sample_index];
    }

    /**
     * @brief Sets a single sample value in the buffer
     * 
     * Writes a sample value to the specified channel and sample position.
     * This method provides bounds-safe access for modifying individual samples.
     * 
     * @param channel The channel index (0-based)
     * @param sample_index The sample index within the channel (0-based)
     * @param value The sample value to write
     */
    void set_sample(size_t channel, size_t sample_index, T value) {
        m_channels[channel][sample_index] = value;
    }

    /**
     * @brief Clears the buffer by setting all samples to zero
     * 
     * Efficiently zeros out all audio data in the buffer across all channels.
     * This operation is optimized for performance and is safe to call in real-time contexts.
     */
    void clear() {
        m_data.clear();
    }

private:

    /**
     * @brief Allocates and initializes channel pointer array
     * 
     * Allocates memory for the array of channel pointers and sets each pointer
     * to point to the correct position within the contiguous data block.
     * This method is called during construction and resize operations.
     */
    void malloc_channels() {
        void* channels = malloc(m_num_channels * sizeof(T*));
        if (channels != nullptr) {
            m_channels = (T**) channels;
        } else {
            LOG_ERROR << "Failed to allocate memory!" << std::endl;
        }
        for (size_t i = 0; i < m_num_channels; i++) {
            m_channels[i] = m_data.data() + i * m_size;
        }
    }

    size_t m_num_channels = 0;        ///< Number of audio channels in the buffer
    size_t m_size = 0;                ///< Number of samples per channel
    T** m_channels = nullptr;         ///< Array of pointers to each channel's data start

    MemoryBlock<T> m_data;            ///< Contiguous memory block holding all audio data
};

/**
 * @brief Type alias for float-based audio buffers
 * 
 * Convenience typedef for the most commonly used buffer type in audio processing,
 * using single-precision floating-point samples.
 */
using BufferF = Buffer<float>;

} // namespace anira

#endif //ANIRA_AUDIO_BUFFER_H