#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>

namespace {
const anira_status k_every_status[] = {
#define ANIRA_STATUS_TEXT(name, text) name,
#include "capi/generated/status_strings.inc"
#undef ANIRA_STATUS_TEXT
};
}  // namespace

TEST(AbiStatus, FailedAndSucceededSplitOnTheSign) {
    EXPECT_FALSE(ANIRA_FAILED(ANIRA_OK));
    EXPECT_FALSE(ANIRA_FAILED(ANIRA_SUCCESS_UPGRADED));
    EXPECT_FALSE(ANIRA_FAILED(ANIRA_INCOMPLETE));
    EXPECT_TRUE(ANIRA_FAILED(ANIRA_ERROR_UNKNOWN));
    EXPECT_TRUE(ANIRA_FAILED(ANIRA_ERROR_INTERNAL));
    EXPECT_TRUE(ANIRA_SUCCEEDED(ANIRA_PENDING));
    EXPECT_FALSE(ANIRA_SUCCEEDED(ANIRA_ERROR_JSON));
}

TEST(AbiStatus, EveryStatusHasDistinctStaticText) {
    std::set<std::string> texts;
    for (const anira_status status : k_every_status) {
        const char* text = anira_status_string(status);
        ASSERT_NE(text, nullptr) << static_cast<int>(status);
        EXPECT_STRNE(text, "unknown status") << static_cast<int>(status);
        EXPECT_GT(std::strlen(text), 0u);
        EXPECT_TRUE(texts.insert(text).second) << "duplicate text: " << text;
        EXPECT_EQ(anira_status_string(status), text) << "static storage";
    }
    EXPECT_EQ(texts.size(), sizeof(k_every_status) / sizeof(k_every_status[0]));
}

TEST(AbiStatus, UnknownValuesYieldUnknownStatus) {
    EXPECT_STREQ(anira_status_string(static_cast<anira_status>(12345)), "unknown status");
    EXPECT_STREQ(anira_status_string(static_cast<anira_status>(-12345)), "unknown status");
    EXPECT_STREQ(anira_status_string(ANIRA_STATUS_FORCE32), "unknown status");
}

TEST(AbiStatus, ErrorInitIsOkWithAnEmptyMessage) {
    const anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(err.status, ANIRA_OK);
    EXPECT_EQ(err.reserved, 0u);
    EXPECT_EQ(err.message[0], '\0');
    static_assert(sizeof(anira_error) == 520, "Tier 1, frozen");
    static_assert(sizeof(err.message) == ANIRA_ERROR_MESSAGE_CAPACITY, "capacity");
}
