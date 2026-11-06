#include "ContentHash.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace anira {

namespace {

// XXH64's five primes.
constexpr uint64_t k_prime1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t k_prime2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t k_prime3 = 0x165667B19E3779F9ULL;
constexpr uint64_t k_prime4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t k_prime5 = 0x27D4EB2F165667C5ULL;

// A word's bytes reversed (std::byteswap is C++23).
template <class Word>
constexpr Word swapped(Word value) noexcept {
    Word result = 0;
    for (size_t i = 0; i < sizeof(Word); ++i) {
        result = static_cast<Word>((result << 8) | (value & 0xFFU));
        value = static_cast<Word>(value >> 8);
    }
    return result;
}

// The input as little-endian words, at any alignment (one load on a little-endian host).
uint64_t read64(const unsigned char* bytes) noexcept {
    uint64_t value = 0;
    std::memcpy(&value, bytes, sizeof value);
    if constexpr (std::endian::native == std::endian::big) { value = swapped(value); }
    return value;
}

uint32_t read32(const unsigned char* bytes) noexcept {
    uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof value);
    if constexpr (std::endian::native == std::endian::big) { value = swapped(value); }
    return value;
}

constexpr uint64_t mix_round(uint64_t accumulator, uint64_t input) noexcept {
    accumulator += input * k_prime2;
    accumulator = std::rotl(accumulator, 31);
    return accumulator * k_prime1;
}

constexpr uint64_t merge_round(uint64_t accumulator, uint64_t value) noexcept {
    accumulator ^= mix_round(0, value);
    return (accumulator * k_prime1) + k_prime4;
}

}  // namespace

uint64_t content_hash(const void* data, size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    size_t at = 0;
    uint64_t hash = 0;
    if (size >= 32) {
        uint64_t v1 = k_prime1 + k_prime2;
        uint64_t v2 = k_prime2;
        uint64_t v3 = 0;
        uint64_t v4 = 0 - k_prime1;
        for (; at + 32 <= size; at += 32) {
            v1 = mix_round(v1, read64(bytes + at));
            v2 = mix_round(v2, read64(bytes + at + 8));
            v3 = mix_round(v3, read64(bytes + at + 16));
            v4 = mix_round(v4, read64(bytes + at + 24));
        }
        hash = std::rotl(v1, 1) + std::rotl(v2, 7) + std::rotl(v3, 12) + std::rotl(v4, 18);
        hash = merge_round(hash, v1);
        hash = merge_round(hash, v2);
        hash = merge_round(hash, v3);
        hash = merge_round(hash, v4);
    } else {
        hash = k_prime5;
    }
    hash += static_cast<uint64_t>(size);
    for (; at + 8 <= size; at += 8) {
        hash ^= mix_round(0, read64(bytes + at));
        hash = (std::rotl(hash, 27) * k_prime1) + k_prime4;
    }
    if (at + 4 <= size) {
        hash ^= static_cast<uint64_t>(read32(bytes + at)) * k_prime1;
        hash = (std::rotl(hash, 23) * k_prime2) + k_prime3;
        at += 4;
    }
    for (; at < size; ++at) {
        hash ^= static_cast<uint64_t>(bytes[at]) * k_prime5;
        hash = std::rotl(hash, 11) * k_prime1;
    }
    hash ^= hash >> 33;
    hash *= k_prime2;
    hash ^= hash >> 29;
    hash *= k_prime3;
    hash ^= hash >> 32;
    return hash;
}

}  // namespace anira
