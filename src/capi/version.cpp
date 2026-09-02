#include <anira/abi/version.h>

#include <cstddef>
#include <cstring>

#include "capi_internal.h"
// Every ABI header: the proc table below names every promised entry point.
#include "generated/abi_headers.inc"

namespace {

struct ProcEntry {
    const char* m_name;
    void* m_address;
};

// One row per promised entry point (src/capi/generated/proc_table.inc): a promised name
// without a definition fails the link here, which is gate 1's presence half until the
// symbol baseline lands at M2. Function-local so the table exists before any static
// initializer that reaches the C API.
const ProcEntry* proc_table(std::size_t& count) {
    // reinterpret_cast of a function pointer to void* is conditionally supported and is
    // what every dlsym-style host does on the platforms anira ships on.
    static const ProcEntry k_table[] = {
#define ANIRA_PROC(name) {#name, reinterpret_cast<void*>(&name)},
#include "generated/proc_table.inc"
#undef ANIRA_PROC
    };
    count = sizeof(k_table) / sizeof(k_table[0]);
    return k_table;
}

}  // namespace

uint32_t ANIRA_CALL anira_abi_version(void) {
    return ANIRA_ABI_VERSION;
}

anira_status ANIRA_CALL anira_check_abi(uint32_t header_abi_version) {
    if (ANIRA_ABI_VERSION_MAJOR(header_abi_version) != ANIRA_ABI_MAJOR) {
        return ANIRA_ERROR_ABI_VERSION;
    }
    // While the major is 0 nothing is promised: the pair must match exactly.
    if (ANIRA_ABI_MAJOR == 0) {
        return header_abi_version == ANIRA_ABI_VERSION ? ANIRA_OK : ANIRA_ERROR_ABI_VERSION;
    }
    return ANIRA_ABI_VERSION_MINOR(header_abi_version) <= ANIRA_ABI_MINOR ? ANIRA_OK
                                                                          : ANIRA_ERROR_ABI_VERSION;
}

uint32_t ANIRA_CALL anira_version(void) {
    return ANIRA_MAKE_VERSION(ANIRA_VERSION_MAJOR, ANIRA_VERSION_MINOR, ANIRA_VERSION_PATCH);
}

const char* ANIRA_CALL anira_version_string(void) {
    return ANIRA_VERSION_STRING;
}

void* ANIRA_CALL anira_get_proc_address(const char* name) {
    if (name == nullptr) { return nullptr; }
    std::size_t count = 0;
    const ProcEntry* table = proc_table(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::strcmp(table[i].m_name, name) == 0) { return table[i].m_address; }
    }
    return nullptr;
}
