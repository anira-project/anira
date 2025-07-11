#ifndef ANIRA_AUDIO_BUFFER_H
#define ANIRA_AUDIO_BUFFER_H

#include <iostream>
#include <cstring>
#include <anira/utils/MemoryBlock.h>
#include <anira/system/AniraWinExports.h>

#include <anira/utils/Logger.h>

namespace anira {

template <typename T>
class ANIRA_API Buffer
{
public:
    // Default constructor creates an empty buffer with channelcount 0 and no memory allocated
    Buffer() = default;
    
    // Constructor creates a buffer with the given number of channels and samples
    Buffer(size_t num_channels, size_t size) : m_num_channels(num_channels), m_size(size), m_data(num_channels * size) {
        malloc_channels();
        clear();
    }

    // Copy constructor takes an lvalue reference to another buffer and creates a new buffer with the same number of channels and samples and copies the data from the other buffer to the new buffer
    Buffer(const Buffer& other) : m_num_channels(other.m_num_channels), m_size(other.m_size), m_data(other.m_data) {
        if (m_num_channels == 0 || m_size == 0) {
            return;
        } else {
            malloc_channels();
        }
    }

    // Move constructor takes an rvalue reference to another buffer and moves the data from the other buffer to the new buffer, then the other buffer is left in a valid but null state
    // marked as noexcept since it is not supposed to throw exceptions and if it does, the program will terminate, this is because the move constructor could corrupt the data in the other buffer if it fails
    Buffer(Buffer&& other) noexcept : m_num_channels(other.m_num_channels), m_size(other.m_size), m_data(std::move(other.m_data)), m_channels(other.m_channels) {
        other.m_num_channels = 0;
        other.m_size = 0;
        other.m_channels = nullptr;
    }

    ~Buffer() {
        free(m_channels);
    }

    // Copy assignment operator takes an lvalue reference to another buffer and copies the data from the other buffer to this buffer
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

    // Move assignment operator takes an rvalue reference to another buffer and moves the data from the other buffer to this buffer, then the other buffer is left in a valid but null state
    // marked as noexcept since it is not supposed to throw exceptions and if it does, the program will terminate, this is because the move operation could corrupt the data in the other buffer if it fails
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

    // Resizes the buffer to the given number of channels and samples   
    void resize(size_t num_channels, size_t size) {
        m_num_channels = num_channels;
        m_size = size;
        m_data.resize(num_channels * size);
        free(m_channels);
        malloc_channels();
    }

    // Returns the number of channels in the buffer, const since it is not supposed to modify any member variables
    size_t get_num_channels() const {
        return m_num_channels;
    }

    // Returns the number of samples in the buffer, const since it is not supposed to modify any member variables
    size_t get_num_samples() const {
        return m_size;
    }

    // Returns a read pointer to the data in the given channel, the pointer points to a const T since the data is not supposed to be modified
    const T* get_read_pointer(size_t channel) const {
        return m_channels[channel];
    }

    // Returns a read pointer to the data in the given channel at the given sample index
    const T* get_read_pointer(size_t channel, size_t sample_index) const {
        return m_channels[channel] + sample_index;
    }

    // Returns a write pointer to the data in the given channel
    T* get_write_pointer(size_t channel) {
        return m_channels[channel];
    }

    // Returns a write pointer to the data in the given channel at the given sample index
    T* get_write_pointer(size_t channel, size_t sample_index) {
        return m_channels[channel] + sample_index;
    }

    // Returns an array of read pointers to the data in all channels
    const T* const* get_array_of_read_pointers() const {
        return const_cast<const T**>(m_channels);
    }

    // Returns an array of write pointers to the data in all channels
    T* const* get_array_of_write_pointers() {
        return m_channels;
    }

    // Returns a pointer to the raw data in the buffer as a contiguous block of memory with first the samples of the first channel, then the samples of the second channel and so on
    T* data() {
        return m_data.data();
    }

    MemoryBlock<T>& get_memory_block() {
        return m_data;
    }

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

    void swap_data(MemoryBlock<T>& other) {
        if (other.size() == m_num_channels * m_size) {
            m_data.swap_data(other);
            reset_channel_ptr();
        } else {
            LOG_ERROR << "Cannot swap data, MemoryBlock has a different size!" << std::endl;
        }
    }

    void swap_data(T*& data, size_t size) {
        if (size == m_num_channels * m_size) {
            m_data.swap_data(data, size);
            reset_channel_ptr();
        } else {
            LOG_ERROR << "Cannot swap data, MemoryBlock has a different size!" << std::endl;
        }
    }

    void reset_channel_ptr() {
        for (size_t i = 0; i < m_num_channels; i++) {
            m_channels[i] = m_data.data() + i * m_size;
        }
    }

    // Returns a sample from the given channel at the given sample index
    T get_sample(size_t channel, size_t sample_index) const {
        return m_channels[channel][sample_index];
    }

    // Sets a sample in the given channel at the given sample index to the given value
    void set_sample(size_t channel, size_t sample_index, T value) {
        m_channels[channel][sample_index] = value;
    }

    // Clears the buffer by setting all samples to 0
    void clear() {
        m_data.clear();
    }

private:

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

    size_t m_num_channels = 0;
    size_t m_size = 0;
    T** m_channels = nullptr;

    MemoryBlock<T> m_data;
};


using BufferF = Buffer<float>;

} // namespace anira

#endif //ANIRA_AUDIO_BUFFER_H