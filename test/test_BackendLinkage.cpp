// The two linkage shapes anira supports (cmake/validate-options.cmake): a shared anira
// links shared backends, a static anira links static backends. LibTorch ships
// shared-only and ExecuTorch static-only, so each can only be compiled into the
// matching shape — a mismatch here means the configure-time rule was bypassed.
// ANIRA_STATIC and the USE_* macros are the compile definitions anira propagates to
// everything that links it, so this test sees exactly what a consumer sees.
#include <gtest/gtest.h>

namespace {

constexpr bool k_static_anira =
#ifdef ANIRA_STATIC
    true;
#else
    false;
#endif

constexpr bool k_with_libtorch =
#ifdef USE_LIBTORCH
    true;
#else
    false;
#endif

constexpr bool k_with_executorch =
#ifdef USE_EXECUTORCH
    true;
#else
    false;
#endif

}  // namespace

TEST(BackendLinkage, LibTorchOnlyInSharedBuilds) {
    if (k_with_libtorch) {
        EXPECT_FALSE(k_static_anira)
            << "LibTorch is shared-only and must not be compiled into a static anira";
    }
}

TEST(BackendLinkage, ExecuTorchOnlyInStaticBuilds) {
    if (k_with_executorch) {
        EXPECT_TRUE(k_static_anira)
            << "ExecuTorch is static-only and must not be compiled into a shared anira";
    }
}
