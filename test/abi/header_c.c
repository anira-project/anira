/*
 * Gate 4 (docs/anira-v3-architecture.md, section 6a): every M1 header in one C11
 * translation unit under -std=c11 -Wall -Wextra -Werror -pedantic (or /std:c11 /W4 /WX),
 * with no anira define at all, exercising each _INIT initializer and the macros a C host
 * uses. The per-file wrappers test/abi/CMakeLists.txt generates cover self-containment.
 */
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <stddef.h>
#include <string.h>

static void on_record(const anira_log_record* record, void* user_data) {
    (void)record;
    (void)user_data;
}

int anira_header_c_probe(void);
int anira_header_c_probe(void) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    anira_log_record record;
    const anira_dtype f32 = ANIRA_DTYPE_F32;
    const uint32_t abi = ANIRA_ABI_VERSION;
    const uint32_t version =
        ANIRA_MAKE_VERSION(ANIRA_VERSION_MAJOR, ANIRA_VERSION_MINOR, ANIRA_VERSION_PATCH);
    int checks = 0;

    memset(&record, 0, sizeof(record));
    record.group = "anira.test";
    desc.callback = on_record;
    desc.level = ANIRA_LOG_ERROR;

    checks += ANIRA_SUCCEEDED(err.status) ? 1 : 0;
    checks += ANIRA_FAILED(ANIRA_ERROR_JSON) ? 1 : 0;
    checks += desc.struct_size == sizeof(anira_log_desc) ? 1 : 0;
    checks += desc.abi_version == abi ? 1 : 0;
    checks += ANIRA_DTYPE_CODE(f32) == ANIRA_DTYPE_FLOAT ? 1 : 0;
    checks += ANIRA_DTYPE_BITS(f32) == 32u ? 1 : 0;
    checks += ANIRA_ABI_VERSION_MAJOR(abi) == ANIRA_ABI_MAJOR ? 1 : 0;
    checks += version != 0u || ANIRA_VERSION_MAJOR == 0 ? 1 : 0;
    checks += record.group_bits != 0u ? 1 : 0;
    checks += err.message[0] == '\0' ? 1 : 0;
    checks += ANIRA_MAX_RANK == 8 ? 1 : 0;
    checks += ANIRA_DYNAMIC == ANIRA_UNBOUNDED ? 1 : 0;
    checks += (int)ANIRA_ANCHOR_FIRST_STREAMED == -1 ? 1 : 0;
    return checks;
}
