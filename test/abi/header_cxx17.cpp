// Gate 4 (docs/anira-v3-architecture.md, section 6a): the M1 headers as C++17 under the
// strict flags with no anira define at all; the C headers never need C++20 (anira.hpp may).
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>

#include <cstring>

namespace {

void on_record(const anira_log_record* /*record*/, void* /*user_data*/) {}

static_assert(sizeof(anira_error) == 520, "anira_error is frozen at 520 bytes");
static_assert(sizeof(anira_log_record) == 56, "anira_log_record is frozen at 56 bytes");
static_assert(sizeof(anira_status) == 4, "enums are 32-bit");
static_assert(ANIRA_DTYPE_F32 == 0x00012002u, "DLPack float32");

}  // namespace

int anira_header_cxx17_probe();
int anira_header_cxx17_probe() {
    anira_error err = ANIRA_ERROR_INIT;
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    anira_log_record record{};
    desc.callback = on_record;
    record.message = "message";
    int checks = 0;
    checks += ANIRA_SUCCEEDED(err.status) ? 1 : 0;
    checks += desc.abi_version == ANIRA_ABI_VERSION ? 1 : 0;
    checks += std::strlen(record.message) == 7 ? 1 : 0;
    checks += ANIRA_ABI_VERSION_MINOR(ANIRA_ABI_VERSION) == ANIRA_ABI_MINOR ? 1 : 0;
    return checks;
}
