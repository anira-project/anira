// anira::content_hash (src/utils/ContentHash.h) is XXH64 with seed 0: the published vectors,
// and vectors checked against LLVM's xxHash64 (llvm/Support/xxhash.h, every size 0..300 at eight
// alignments and 1 MiB agreed when the function was written); the result does not depend on the
// address's alignment, and one changed bit changes it.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string_view>
#include <vector>

#include "utils/ContentHash.h"

namespace {

std::vector<unsigned char> pattern(size_t size) {
    std::vector<unsigned char> bytes(size);
    for (size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<unsigned char>((i * 2654435761U) >> 13);
    }
    return bytes;
}

}  // namespace

TEST(ContentHash, IsXxh64WithSeedZero) {
    EXPECT_EQ(anira::content_hash(nullptr, 0), 0xEF46DB3751D8E999ULL);
    EXPECT_EQ(anira::content_hash("abc", 3), 0x44BC2CF5AD770999ULL);
    constexpr std::string_view k_fox = "The quick brown fox jumps over the lazy dog";
    EXPECT_EQ(anira::content_hash(k_fox.data(), k_fox.size()), 0x0B242D361FDA71BCULL);
    const std::vector<unsigned char> mib = pattern(size_t{1} << 20);
    EXPECT_EQ(anira::content_hash(mib.data(), mib.size()), 0xAE5BA942A34ABDCFULL);
}

TEST(ContentHash, IgnoresTheAlignmentAndSeesOneBit) {
    const std::vector<unsigned char> source = pattern(1031);
    std::vector<unsigned char> shifted(source.size() + 8);
    for (size_t offset = 1; offset < 8; ++offset) {
        std::memcpy(shifted.data() + offset, source.data(), source.size());
        EXPECT_EQ(anira::content_hash(shifted.data() + offset, source.size()),
                  anira::content_hash(source.data(), source.size()))
            << "offset " << offset;
    }
    std::vector<unsigned char> flipped = source;
    flipped[517] ^= 0x10U;
    EXPECT_NE(anira::content_hash(flipped.data(), flipped.size()),
              anira::content_hash(source.data(), source.size()));
    EXPECT_NE(anira::content_hash(source.data(), source.size() - 1),
              anira::content_hash(source.data(), source.size()));
}
