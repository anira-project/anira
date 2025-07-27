#ifndef ANIRA_RINGBUFFER_H
#define ANIRA_RINGBUFFER_H

#include <vector>
#include <cmath>
#include "Buffer.h"

namespace anira {

/**
 * @brief Circular buffer implementation for real-time audio processing with multi-channel support
 * 
 * The RingBuffer class extends the Buffer class to provide circular buffer functionality
 * optimized for real-time audio applications. It maintains separate read and write positions
 * for each channel, enabling efficient streaming audio processing with lookahead and
 * lookbehind capabilities.
 * 
 * Key features:
 * - Multi-channel circular buffer with independent channel positions
 * - Real-time safe push/pop operations for streaming audio
 * - Lookahead access to future samples in the buffer
 * - Lookbehind access to previously processed samples
 * - Automatic wrapping and overflow handling
 * - Full/empty state tracking for each channel
 * - Zero-copy access patterns for performance-critical code
 * 
 * @note This class inherits from Buffer<float> and adds circular buffer semantics
 *       while maintaining real-time safety guarantees.
 * 
 * @see Buffer, BufferF, MemoryBlock
 */
class ANIRA_API RingBuffer : public Buffer<float>
{
public:
    /**
     * @brief Default constructor that creates an empty ring buffer
     * 
     * Creates an uninitialized ring buffer with no allocated memory.
     * The buffer must be initialized using initialize_with_positions()
     * before it can be used for audio processing.
     */
    RingBuffer();

    /**
     * @brief Initializes the ring buffer with specified dimensions and position tracking
     * 
     * Allocates memory for the ring buffer and initializes read/write position vectors
     * for each channel. All positions are set to zero and the buffer is cleared.
     * This method must be called before using any other ring buffer operations.
     * 
     * @param num_channels Number of audio channels to allocate
     * @param num_samples Number of samples per channel (buffer size)
     * 
     * @note This method involves memory allocation and should not be called in real-time contexts
     */
    void initialize_with_positions(size_t num_channels, size_t num_samples);
    
    /**
     * @brief Clears the buffer content and resets all position counters
     * 
     * Sets all audio data to zero and resets read/write positions for all channels
     * to their initial state. The buffer size and channel count remain unchanged.
     * This operation is real-time safe.
     */
    void clear_with_positions();
    
    /**
     * @brief Pushes a single sample into the specified channel's ring buffer
     * 
     * Writes a sample to the current write position of the specified channel
     * and advances the write position with automatic wrapping. This operation
     * is optimized for real-time audio processing.
     * 
     * @param channel The channel index to write to (0-based)
     * @param sample The audio sample value to write
     * 
     * @note This method is real-time safe and performs automatic buffer wrapping
     */
    void push_sample(size_t channel, float sample);
    
    /**
     * @brief Pops a single sample from the specified channel's ring buffer
     * 
     * Reads a sample from the current read position of the specified channel
     * and advances the read position with automatic wrapping. Returns the
     * sample value that was read.
     * 
     * @param channel The channel index to read from (0-based)
     * @return The audio sample value that was read
     * 
     * @note This method is real-time safe and performs automatic buffer wrapping
     */
    float pop_sample(size_t channel);
    
    /**
     * @brief Gets a future sample from the ring buffer without advancing positions
     * 
     * Reads a sample from a position ahead of the current read position without
     * modifying any position counters. This enables lookahead processing for
     * algorithms that need access to upcoming samples.
     * 
     * @param channel The channel index to read from (0-based)
     * @param offset Number of samples ahead of the read position to access
     * @return The audio sample value at the future position
     * 
     * @note The offset should not exceed the number of available samples in the buffer
     */
    float get_future_sample(size_t channel, size_t offset);
    
    /**
     * @brief Gets a past sample from the ring buffer without advancing positions
     * 
     * Reads a sample from a position behind the current read position without
     * modifying any position counters. This enables lookbehind processing for
     * algorithms that need access to previously processed samples.
     * 
     * @param channel The channel index to read from (0-based)
     * @param offset Number of samples behind the read position to access
     * @return The audio sample value at the past position
     * 
     * @note The offset should not exceed the number of available past samples
     */
    float get_past_sample(size_t channel, size_t offset);
    
    /**
     * @brief Gets the number of samples available for reading from a channel
     * 
     * Calculates the number of samples that have been written to the channel
     * but not yet read. This is useful for determining how much data is
     * available for processing.
     * 
     * @param channel The channel index to query (0-based)
     * @return Number of samples available for reading
     */
    size_t get_available_samples(size_t channel);
    
    /**
     * @brief Gets the number of past samples available for lookbehind access
     * 
     * Calculates the number of samples that have been read and are available
     * for past sample access. This is useful for algorithms that need to
     * access historical data.
     * 
     * @param channel The channel index to query (0-based)
     * @return Number of past samples available for lookbehind access
     */
    size_t get_available_past_samples(size_t channel);

private:
    std::vector<size_t> m_read_pos;    ///< Read position for each channel in the ring buffer
    std::vector<size_t> m_write_pos;   ///< Write position for each channel in the ring buffer
    std::vector<bool> m_is_full;       ///< Track if each channel's buffer is full (write has wrapped around to read position)
};

} // namespace anira

#endif //ANIRA_RINGBUFFER_H