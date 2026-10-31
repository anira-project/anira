// The two linkage shapes anira supports (cmake/validate-options.cmake): a shared anira
// links shared engines, a static anira links static engines. LibTorch ships shared-only and
// ExecuTorch static-only, so each can only be compiled into the matching shape — a mismatch
// here means the configure-time rule was bypassed. The test reads what a consumer can read: the
// engines this build compiled in come from anira_enabled_engines (anira passes no USE_* switch
// to its consumers), and the shape from ANIRA_STATIC, which a static anira gives everything that
// links it.
#include <anira/anira.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr bool k_static_anira =
#ifdef ANIRA_STATIC
    true;
#else
    false;
#endif

/// The engines of this build: the count query, then a stride-sized array.
std::vector<anira_backend_id> enabled_engines() {
    uint32_t count = 0;
    EXPECT_EQ(anira_enabled_engines(sizeof(anira_backend_id), &count, nullptr), ANIRA_OK);
    std::vector<anira_backend_id> rows(count, ANIRA_BACKEND_ID_INIT);
    if (count > 0) {
        EXPECT_EQ(anira_enabled_engines(sizeof(anira_backend_id), &count, rows.data()), ANIRA_OK);
    }
    rows.resize(count);
    return rows;
}

bool enabled(anira_engine engine) {
    const std::vector<anira_backend_id> rows = enabled_engines();
    return std::ranges::any_of(rows, [engine](const anira_backend_id& row) {
        return row.engine == engine;
    });
}

}  // namespace

TEST(EngineLinkage, LibTorchOnlyInSharedBuilds) {
    if (enabled(ANIRA_ENGINE_LIBTORCH)) {
        EXPECT_FALSE(k_static_anira)
            << "LibTorch is shared-only and must not be compiled into a static anira";
    }
}

TEST(EngineLinkage, ExecuTorchOnlyInStaticBuilds) {
    if (enabled(ANIRA_ENGINE_EXECUTORCH)) {
        EXPECT_TRUE(k_static_anira)
            << "ExecuTorch is static-only and must not be compiled into a shared anira";
    }
}
