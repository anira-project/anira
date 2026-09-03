#include <anira/abi/build_info.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>

#include <cstdint>

uint32_t ANIRA_CALL anira_abi_version(void) ANIRA_NOEXCEPT {
    return ANIRA_ABI_VERSION;
}

anira_status ANIRA_CALL anira_check_abi(uint32_t header_abi_version) ANIRA_NOEXCEPT {
    constexpr uint32_t k_library_major = ANIRA_ABI_MAJOR;
    if (ANIRA_ABI_VERSION_MAJOR(header_abi_version) != k_library_major) {
        return ANIRA_ERROR_ABI_VERSION;
    }
    // While the major is 0 nothing is promised: the pair must match exactly.
    if (k_library_major == 0) {
        return header_abi_version == ANIRA_ABI_VERSION ? ANIRA_OK : ANIRA_ERROR_ABI_VERSION;
    }
    return ANIRA_ABI_VERSION_MINOR(header_abi_version) <= ANIRA_ABI_MINOR ? ANIRA_OK
                                                                          : ANIRA_ERROR_ABI_VERSION;
}

uint32_t ANIRA_CALL anira_version(void) ANIRA_NOEXCEPT {
    return ANIRA_MAKE_VERSION(ANIRA_VERSION_MAJOR, ANIRA_VERSION_MINOR, ANIRA_VERSION_PATCH);
}

const char* ANIRA_CALL anira_version_string(void) ANIRA_NOEXCEPT {
    return ANIRA_VERSION_STRING;
}
