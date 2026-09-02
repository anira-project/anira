#include <anira/abi/build_info.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

// The proc table the library was built from: every promised name must resolve.
namespace {
constexpr std::array k_promised_names{
#define ANIRA_PROC(name) #name,
#include "capi/generated/proc_table.inc"
#undef ANIRA_PROC
};
}  // namespace

TEST(AbiVersion, LibraryReportsTheHeaderPair) {
    EXPECT_EQ(anira_abi_version(), ANIRA_ABI_VERSION);
    EXPECT_EQ(ANIRA_ABI_VERSION_MAJOR(anira_abi_version()), ANIRA_ABI_MAJOR);
    EXPECT_EQ(ANIRA_ABI_VERSION_MINOR(anira_abi_version()), ANIRA_ABI_MINOR);
}

TEST(AbiVersion, CheckAbiAcceptsTheExactPair) {
    EXPECT_EQ(anira_check_abi(ANIRA_ABI_VERSION), ANIRA_OK);
}

TEST(AbiVersion, CheckAbiRefusesANewerMinorAndAnotherMajor) {
    EXPECT_EQ(anira_check_abi(ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR, ANIRA_ABI_MINOR + 1)),
              ANIRA_ERROR_ABI_VERSION);
    EXPECT_EQ(anira_check_abi(ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1, 0)),
              ANIRA_ERROR_ABI_VERSION);
    EXPECT_EQ(anira_check_abi(ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1, ANIRA_ABI_MINOR)),
              ANIRA_ERROR_ABI_VERSION);
}

TEST(AbiVersion, CheckAbiOfAnOlderMinorDependsOnTheMajor) {
    constexpr uint32_t k_major = ANIRA_ABI_MAJOR;
    constexpr uint32_t k_minor = ANIRA_ABI_MINOR;
    if (k_minor == 0) { GTEST_SKIP() << "no older minor exists at minor 0"; }
    const uint32_t older = ANIRA_MAKE_ABI_VERSION(k_major, k_minor - 1);
    // Exact match while the major is 0 (nothing promised); compatible from v3.0.0 on.
    if (k_major == 0) {
        EXPECT_EQ(anira_check_abi(older), ANIRA_ERROR_ABI_VERSION);
    } else {
        EXPECT_EQ(anira_check_abi(older), ANIRA_OK);
    }
}

TEST(AbiVersion, VersionPacksTheSemverTriple) {
    EXPECT_EQ(anira_version(),
              ANIRA_MAKE_VERSION(ANIRA_VERSION_MAJOR, ANIRA_VERSION_MINOR, ANIRA_VERSION_PATCH));
    EXPECT_EQ(anira_version() >> 22, static_cast<uint32_t>(ANIRA_VERSION_MAJOR));
}

TEST(AbiVersion, VersionStringIsTheBuildString) {
    ASSERT_NE(anira_version_string(), nullptr);
    EXPECT_STREQ(anira_version_string(), ANIRA_VERSION_STRING);
    EXPECT_EQ(anira_version_string(), anira_version_string()) << "static storage";
}

TEST(AbiVersion, EveryPromisedNameResolvesThroughGetProcAddress) {
    for (const char* name : k_promised_names) {
        EXPECT_NE(anira_get_proc_address(name), nullptr) << name;
    }
    EXPECT_EQ(anira_get_proc_address("anira_abi_version"),
              reinterpret_cast<void*>(&anira_abi_version));
    EXPECT_EQ(anira_get_proc_address("anira_status_string"),
              reinterpret_cast<void*>(&anira_status_string));
}

TEST(AbiVersion, UnknownOrNullNamesYieldNull) {
    EXPECT_EQ(anira_get_proc_address(nullptr), nullptr);
    EXPECT_EQ(anira_get_proc_address(""), nullptr);
    EXPECT_EQ(anira_get_proc_address("anira_no_such_entry"), nullptr);
    EXPECT_EQ(anira_get_proc_address("anira_abi_version "), nullptr) << "exact match only";
}
