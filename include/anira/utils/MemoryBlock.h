#ifndef ANIRA_MEMORYBLOCK_H
#define ANIRA_MEMORYBLOCK_H

#include <iostream>
#include <type_traits>

#include <anira/utils/Logger.h>

namespace anira {

template <typename T>
class MemoryBlock {
public:
    MemoryBlock(std::size_t size = 0) : m_size(size) {
        void* data = malloc(sizeof(T) * m_size);
        if (data != nullptr) {
            m_data = (T*) data;
        } else {
            LOG_ERROR << "Failed to allocate memory!" << std::endl;
        }
    }

    ~MemoryBlock() noexcept {
        free(m_data);
    }

    MemoryBlock(const MemoryBlock& other) : m_size(other.m_size) {
        void* data = malloc(sizeof(T) * m_size);
        if (data != nullptr) {
            m_data = (T*) data;
            memcpy(m_data, other.m_data, sizeof(T) * m_size);
        } else {
            LOG_ERROR << "Failed to allocate memory!" << std::endl;
        }
    }

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

    MemoryBlock(MemoryBlock&& other) noexcept : m_size(other.m_size), m_data(other.m_data) {
        other.m_size = 0;
        other.m_data = nullptr;
    }

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

    T& operator[](size_t index) {
        return m_data[index];
    }

    const T& operator[](size_t index) const {
        return m_data[index];
    }

    T* data() {
        return m_data;
    }

    size_t size() const {
        return m_size;
    }

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

    void clear() {
        memset(m_data, 0, sizeof(T) * m_size);
    }

    // enable_if_t<expression, T> is a helper type that is equivalent to T if expression is true, otherwise it is not defined.
    // This is useful to check if a type is trivially copyable or not and enable or disable a function based on that.
    // remember that template parameters do not need to have a name, so we can use the following syntax:
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

    void swap_data(T*& data, size_t size) {
        if (m_size == size) {
            std::swap(m_data, data);
        } else {
            LOG_ERROR << "Cannot swap data with different sizes!" << std::endl;
        }
    }
        
private:
    T* m_data = nullptr;
    size_t m_size;
};

} // namespace anira

#endif // ANIRA_MEMORYBLOCK_H