#ifndef ANIRA_AUDIO_BUFFER_H
#define ANIRA_AUDIO_BUFFER_H

#include <iostream>
#include <cstring>
#include "anira/system/AniraConfig.h"

namespace anira {

template <typename T>
class ANIRA_API AudioBuffer
{
public:
    // Default constructor creates an empty buffer with channelcount 0 and no memory allocated
    AudioBuffer()
    {
    }
    
    // Constructor creates a buffer with the given number of channels and samples
    AudioBuffer(size_t number_of_channels, size_t size)
        : m_number_of_channels(number_of_channels), m_size(size)
    {
        allocateMemory();
        clear();
    }
    
    // Constructor creates a buffer with the given number of channels and samples and either copies the data from the given blocks of memory to the internal buffer data or reference the data from the given blocks of memory
    AudioBuffer(T* const* data, size_t number_of_channels, size_t size, bool copy_data = true)
        : m_number_of_channels(number_of_channels), m_size(size)
    {
        if (copy_data) {
            allocateMemory();
            for (size_t i = 0; i < m_number_of_channels; i++) {
                std::memcpy(m_p_channels[i], data[i], m_size * sizeof(T));
            }
        } else {
            // we allocate memory for the pointers to the channels but we do not allocate memory for the data itself
            // if we would directly reference the data with m_p_channels = data, then when we would delete the buffer, the data pointers would be deleted as well
            m_p_channels = new T*[m_number_of_channels];
            for (size_t i = 0; i < m_number_of_channels; i++) {
                m_p_channels[i] = data[i];
            }
            m_p_data = nullptr; // since we cannot garantee that the data will be in a contiguous block of memory when we reference it from blocks of memory
        }
    }

    // Copy constructor takes an lvalue reference to another buffer and creates a new buffer with the same number of channels and samples and copies the data from the other buffer to the new buffer
    AudioBuffer(const AudioBuffer& other)
        : m_number_of_channels(other.m_number_of_channels), m_size(other.m_size)
    {
        if (m_number_of_channels == 0 || m_size == 0) {
            return;
        } else {
            allocateMemory();
            std::memcpy(m_p_data, other.m_p_data, m_number_of_channels * m_size * sizeof(T));
        }
    }

    // Move constructor takes an rvalue reference to another buffer and moves the data from the other buffer to the new buffer, then the other buffer is left in a valid but null state
    // marked as noexcept since it is not supposed to throw exceptions and if it does, the program will terminate, this is because the move constructor could corrupt the data in the other buffer if it fails
    AudioBuffer(AudioBuffer&& other) noexcept
        : m_number_of_channels(other.m_number_of_channels), m_size(other.m_size), m_p_data(other.m_p_data), m_p_channels(other.m_p_channels)
    {
        other.m_number_of_channels = 0;
        other.m_size = 0;
        other.m_p_data = nullptr;
        other.m_p_channels = nullptr;
    }

    ~AudioBuffer()
    {
        delete[] m_p_data;
        delete[] m_p_channels;
    }

    // Copy assignment operator takes an lvalue reference to another buffer and copies the data from the other buffer to this buffer
    AudioBuffer& operator=(const AudioBuffer& other)
    {
        initialize(other.m_number_of_channels, other.m_size);
        std::memcpy(m_p_data, other.m_p_data, m_number_of_channels * m_size * sizeof(T));
        return *this;
    }

    // Move assignment operator takes an rvalue reference to another buffer and moves the data from the other buffer to this buffer, then the other buffer is left in a valid but null state
    // marked as noexcept since it is not supposed to throw exceptions and if it does, the program will terminate, this is because the move operation could corrupt the data in the other buffer if it fails
    AudioBuffer& operator=(AudioBuffer&& other) noexcept
    {
        if (this != &other) {
            delete[] m_p_data;
            delete[] m_p_channels;
            m_number_of_channels = other.m_number_of_channels;
            m_size = other.m_size;
            m_p_data = other.m_p_data;
            m_p_channels = other.m_p_channels;
            other.m_number_of_channels = 0;
            other.m_size = 0;
            other.m_p_data = nullptr;
            other.m_p_channels = nullptr;
        }
        return *this;
    }

    // Resets the buffer to the given number of channels and samples and either copies the data from the given blocks of memory to the internal buffer data or reference the data from the given blocks of memory
    void resetFromData(T* const* data, size_t number_of_channels, size_t size, bool copy_data = true)
    {
        delete[] m_p_data;
        delete[] m_p_channels;
        m_number_of_channels = number_of_channels;
        m_size = size;
        if (copy_data) {
            allocateMemory();
            for (size_t i = 0; i < m_number_of_channels; i++) {
                std::memcpy(m_p_channels[i], data[i], m_size * sizeof(T));
            }
        } else {
            // we allocate memory for the pointers to the channels but we do not allocate memory for the data itself see the respective constructor for more details
            m_p_channels = new T*[m_number_of_channels];
            for (size_t i = 0; i < m_number_of_channels; i++) {
                m_p_channels[i] = data[i];
            }
            m_p_data = nullptr; // since we cannot garantee that the data will be in a contiguous block of memory when we reference it from blocks of memory
        }
    }

    // Resizes the buffer to the given number of channels and samples, all data in the buffer is lost    
    void initialize(size_t number_of_channels, size_t size)
    {
        delete[] m_p_data;
        delete[] m_p_channels;
        m_number_of_channels = number_of_channels;
        m_size = size;
        allocateMemory();
    }

    // Returns the number of channels in the buffer, const since it is not supposed to modify any member variables
    size_t getNumChannels() const
    {
        return m_number_of_channels;
    }

    // Returns the number of samples in the buffer, const since it is not supposed to modify any member variables
    size_t getNumSamples() const
    {
        return m_size;
    }

    // Returns a read pointer to the data in the given channel, the pointer points to a const T since the data is not supposed to be modified
    const T* getReadPointer(size_t channelNumber) const
    {
        return m_p_channels[channelNumber];
    }

    // Returns a read pointer to the data in the given channel at the given sample index
    const T* getReadPointer(size_t channelNumber, size_t sampleIndex) const
    {
        return m_p_channels[channelNumber] + sampleIndex;
    }

    // Returns a write pointer to the data in the given channel
    T* getWritePointer(size_t channelNumber)
    {
        return m_p_channels[channelNumber];
    }

    // Returns a write pointer to the data in the given channel at the given sample index
    T* getWritePointer(size_t channelNumber, size_t sampleIndex)
    {
        return m_p_channels[channelNumber] + sampleIndex;
    }

    // Returns an array of read pointers to the data in all channels
    const T** getArrayOfReadPointers() const
    {
        return m_p_channels;
    }

    // Returns an array of write pointers to the data in all channels
    T** getArrayOfWritePointers()
    {
        return m_p_channels;
    }

    // Returns a pointer to the raw data in the buffer as a contiguous block of memory with first the samples of the first channel, then the samples of the second channel and so on
    T* getRawData()
    {   
        if (m_p_data == nullptr) {
            throw std::runtime_error("Either the buffer is empty or the data is referenced from external blocks of memory");
        } else {
            return m_p_data;
        }
    }

    // Returns a sample from the given channel at the given sample index
    T getSample(size_t channelNumber, size_t sampleIndex) const
    {
        return m_p_channels[channelNumber][sampleIndex];
    }

    // Sets a sample in the given channel at the given sample index to the given value
    void setSample(size_t channelNumber, size_t sampleIndex, T value)
    {
        m_p_channels[channelNumber][sampleIndex] = value;
    }

    // Clears the buffer by setting all samples to 0
    void clear()
    {
        for (size_t i = 0; i < m_number_of_channels * m_size; i++) {
            m_p_data[i] = 0;
        }
    }

    // Copy make copy of the buffer, the buffer will have the same number of channels and samples and the data from the other buffer will be copied to this buffer
    // Works with buffers of different types, the data from the other buffer will be converted to the type of this buffer
    template <typename U>
    void makeCopyOf(const AudioBuffer<U>& other)
    {
        initialize(other.m_number_of_channels, other.m_size);
        if (typeid(T) == typeid(U)) {
            std::memcpy(m_p_data, other.m_p_data, m_number_of_channels * m_size * sizeof(T));
        } else {
            const U* p_source_data = other.getRawData();
            for (size_t i = 0; i < m_number_of_channels * m_size; i++) {
                m_p_data[i] = static_cast<T>(p_source_data[i]);
            }
        }
    }

private:

    void allocateMemory()
    {
        m_p_data = new T[m_number_of_channels * m_size];
        m_p_channels = new T*[m_number_of_channels];
        for (size_t i = 0; i < m_number_of_channels; i++) {
            m_p_channels[i] = m_p_data + i * m_size;
        }
    }

    size_t m_number_of_channels = 0;
    size_t m_size = 0;
    T** m_p_channels = nullptr;
    T* m_p_data = nullptr;
};


using AudioBufferF = AudioBuffer<float>;

} // namespace anira

#endif //ANIRA_AUDIO_BUFFER_H