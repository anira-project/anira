#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "capi/capi_internal.h"

using anira::capi::firewall_probe;

TEST(AbiFirewall, SuccessWritesTheOutParameterAndLeavesErrAlone) {
    anira_error err = ANIRA_ERROR_INIT;
    int value = -1;
    EXPECT_EQ(firewall_probe(0, ANIRA_OK, nullptr, &err, &value), ANIRA_OK);
    EXPECT_EQ(value, 42);
    EXPECT_EQ(err.status, ANIRA_OK);
    EXPECT_EQ(err.message[0], '\0');
}

TEST(AbiFirewall, BadAllocBecomesOutOfMemory) {
    anira_error err = ANIRA_ERROR_INIT;
    int value = -1;
    EXPECT_EQ(firewall_probe(1, ANIRA_OK, nullptr, &err, &value), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(err.status, ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_STREQ(err.message, "out of memory");
    EXPECT_EQ(value, -1) << "out-parameters are written only on success";
}

TEST(AbiFirewall, StatusErrorCarriesItsStatusAndMessage) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "models[0].engine: unknown", &err, nullptr),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(err.status, ANIRA_ERROR_JSON);
    EXPECT_STREQ(err.message, "models[0].engine: unknown");
}

TEST(AbiFirewall, InvalidArgumentBecomesConfig) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(3, ANIRA_OK, "bad shape", &err, nullptr), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(err.status, ANIRA_ERROR_CONFIG);
    EXPECT_STREQ(err.message, "bad shape");
}

TEST(AbiFirewall, OtherExceptionsBecomeInternal) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(4, ANIRA_OK, "engine exploded", &err, nullptr), ANIRA_ERROR_INTERNAL);
    EXPECT_STREQ(err.message, "engine exploded");
    err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(5, ANIRA_OK, nullptr, &err, nullptr), ANIRA_ERROR_INTERNAL);
    EXPECT_STREQ(err.message, "unknown exception");
}

TEST(AbiFirewall, NullErrIsAccepted) {
    EXPECT_EQ(firewall_probe(1, ANIRA_OK, nullptr, nullptr, nullptr), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_DEVICE, "x", nullptr, nullptr), ANIRA_ERROR_DEVICE);
}

TEST(AbiFirewall, LongMessagesAreTruncatedAndTerminated) {
    anira_error err = ANIRA_ERROR_INIT;
    const std::string long_message(700, 'm');
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_CONFIG, long_message.c_str(), &err, nullptr),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(std::strlen(err.message), static_cast<size_t>(ANIRA_ERROR_MESSAGE_CAPACITY - 1));
    EXPECT_EQ(err.message[ANIRA_ERROR_MESSAGE_CAPACITY - 1], '\0');
}
