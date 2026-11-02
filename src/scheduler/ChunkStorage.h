#ifndef ANIRA_CHUNKSTORAGE_H
#define ANIRA_CHUNKSTORAGE_H
/*
 * Where a chunk's slot buffers live (SessionElement::ThreadSafeStruct): one block per chunk,
 * allocated by the session's prepare on the control thread, never on a real-time path, and
 * released with the chunk. The storage decides the memory and its domain; the session decides
 * the block's size and its alignment (the largest any of its plans' engines needs to bind a
 * slot in place, engine::Loaded::buffer_alignment). In this pre-release the host storage is the
 * only one: aligned, zeroed host memory (ANIRA_DOMAIN_HOST). A provider whose engine binds
 * memory of another kind (a device buffer, host memory a GPU maps) brings its own storage
 * through this interface, with the domain the slot descriptors then name.
 */
#include <anira/abi/enums.h>

#include <cstddef>
#include <cstring>
#include <new>
#include <utility>

namespace anira {

/// The allocator of chunk blocks.
class ChunkStorage {
public:
    ChunkStorage() = default;
    virtual ~ChunkStorage() = default;
    ChunkStorage(const ChunkStorage&) = delete;
    ChunkStorage& operator=(const ChunkStorage&) = delete;
    ChunkStorage(ChunkStorage&&) = delete;
    ChunkStorage& operator=(ChunkStorage&&) = delete;

    /// A zeroed block of `bytes` bytes (at least one) whose first byte lies on an `alignment`
    /// boundary (a power of two). Throws std::bad_alloc when there is no memory.
    virtual std::byte* allocate(size_t bytes, size_t alignment) = 0;
    /// Gives back a block allocate handed out, with the size and alignment it was asked for.
    virtual void release(std::byte* block, size_t bytes, size_t alignment) noexcept = 0;
    /// The domain of the memory the storage hands out, what the slot descriptors name.
    virtual anira_domain domain() const noexcept = 0;
};

/// Aligned, zeroed host memory: the storage of every chunk in this pre-release.
class HostChunkStorage final : public ChunkStorage {
public:
    std::byte* allocate(size_t bytes, size_t alignment) override {
        auto* block = static_cast<std::byte*>(
            ::operator new(bytes > 0 ? bytes : 1, std::align_val_t{alignment}));
        std::memset(block, 0, bytes);
        return block;
    }
    void release(std::byte* block, size_t bytes, size_t alignment) noexcept override {
        ::operator delete(block, bytes > 0 ? bytes : 1, std::align_val_t{alignment});
    }
    anira_domain domain() const noexcept override { return ANIRA_DOMAIN_HOST; }
};

/// The process's one host storage (stateless).
inline ChunkStorage& host_chunk_storage() noexcept {
    static HostChunkStorage storage;
    return storage;
}

/// One block of a storage, owned: given back to its storage when it dies.
class ChunkBlock {
public:
    ChunkBlock() = default;
    ChunkBlock(ChunkStorage& storage, size_t bytes, size_t alignment)
        : m_storage(&storage)
        , m_data(storage.allocate(bytes, alignment))
        , m_bytes(bytes)
        , m_alignment(alignment) {}
    ~ChunkBlock() { reset(); }
    ChunkBlock(const ChunkBlock&) = delete;
    ChunkBlock& operator=(const ChunkBlock&) = delete;
    ChunkBlock(ChunkBlock&& other) noexcept
        : m_storage(std::exchange(other.m_storage, nullptr))
        , m_data(std::exchange(other.m_data, nullptr))
        , m_bytes(std::exchange(other.m_bytes, 0))
        , m_alignment(std::exchange(other.m_alignment, 0)) {}
    ChunkBlock& operator=(ChunkBlock&& other) noexcept {
        if (this != &other) {
            reset();
            m_storage = std::exchange(other.m_storage, nullptr);
            m_data = std::exchange(other.m_data, nullptr);
            m_bytes = std::exchange(other.m_bytes, 0);
            m_alignment = std::exchange(other.m_alignment, 0);
        }
        return *this;
    }

    std::byte* data() const noexcept { return m_data; }
    size_t size() const noexcept { return m_bytes; }
    size_t alignment() const noexcept { return m_alignment; }
    /// The domain of the block's memory; ANIRA_DOMAIN_HOST for an empty block.
    anira_domain domain() const noexcept {
        return m_storage != nullptr ? m_storage->domain() : ANIRA_DOMAIN_HOST;
    }

private:
    void reset() noexcept {
        if (m_storage != nullptr && m_data != nullptr) {
            m_storage->release(m_data, m_bytes, m_alignment);
        }
        m_storage = nullptr;
        m_data = nullptr;
    }

    ChunkStorage* m_storage = nullptr;
    std::byte* m_data = nullptr;
    size_t m_bytes = 0;
    size_t m_alignment = 0;
};

}  // namespace anira

#endif  // ANIRA_CHUNKSTORAGE_H
