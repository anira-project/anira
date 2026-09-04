/*
 * test/abi/generated/test_layout.c -- gate 3, generated from abi/anira.yml by tools/abi/gen.py.
 * _Static_asserts pin every enum's width and terminator, the ABI version packing, the dtype
 * packing and every Tier-1 layout; main() prints the Tier-1 table that abi/layout-<major>.txt
 * commits. Do not edit.
 */
#include <stddef.h>
#include <stdio.h>

#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/thread.h>

_Static_assert(sizeof(anira_status) == 4, "anira_status is a 32-bit enum");
_Static_assert(ANIRA_STATUS_FORCE32 == 0x7fffffff, "anira_status terminator");
_Static_assert(sizeof(anira_dtype_code) == 4, "anira_dtype_code is a 32-bit enum");
_Static_assert(ANIRA_DTYPE_CODE_FORCE32 == 0x7fffffff, "anira_dtype_code terminator");
_Static_assert(sizeof(anira_domain) == 4, "anira_domain is a 32-bit enum");
_Static_assert(ANIRA_DOMAIN_FORCE32 == 0x7fffffff, "anira_domain terminator");
_Static_assert(sizeof(anira_sync_kind) == 4, "anira_sync_kind is a 32-bit enum");
_Static_assert(ANIRA_SYNC_KIND_FORCE32 == 0x7fffffff, "anira_sync_kind terminator");
_Static_assert(sizeof(anira_tensor_flags) == 4, "anira_tensor_flags is a 32-bit enum");
_Static_assert(ANIRA_TENSOR_FLAGS_FORCE32 == 0x7fffffff, "anira_tensor_flags terminator");
_Static_assert(sizeof(anira_struct_id) == 4, "anira_struct_id is a 32-bit enum");
_Static_assert(ANIRA_STRUCT_ID_FORCE32 == 0x7fffffff, "anira_struct_id terminator");
_Static_assert(sizeof(anira_container) == 4, "anira_container is a 32-bit enum");
_Static_assert(ANIRA_CONTAINER_FORCE32 == 0x7fffffff, "anira_container terminator");
_Static_assert(sizeof(anira_pixel_format) == 4, "anira_pixel_format is a 32-bit enum");
_Static_assert(ANIRA_PIXEL_FORMAT_FORCE32 == 0x7fffffff, "anira_pixel_format terminator");
_Static_assert(sizeof(anira_color_matrix) == 4, "anira_color_matrix is a 32-bit enum");
_Static_assert(ANIRA_COLOR_MATRIX_FORCE32 == 0x7fffffff, "anira_color_matrix terminator");
_Static_assert(sizeof(anira_color_range) == 4, "anira_color_range is a 32-bit enum");
_Static_assert(ANIRA_COLOR_RANGE_FORCE32 == 0x7fffffff, "anira_color_range terminator");
_Static_assert(sizeof(anira_axis_tag) == 4, "anira_axis_tag is a 32-bit enum");
_Static_assert(ANIRA_AXIS_TAG_FORCE32 == 0x7fffffff, "anira_axis_tag terminator");
_Static_assert(sizeof(anira_role) == 4, "anira_role is a 32-bit enum");
_Static_assert(ANIRA_ROLE_FORCE32 == 0x7fffffff, "anira_role terminator");
_Static_assert(sizeof(anira_contract_kind) == 4, "anira_contract_kind is a 32-bit enum");
_Static_assert(ANIRA_CONTRACT_KIND_FORCE32 == 0x7fffffff, "anira_contract_kind terminator");
_Static_assert(sizeof(anira_budget_kind) == 4, "anira_budget_kind is a 32-bit enum");
_Static_assert(ANIRA_BUDGET_KIND_FORCE32 == 0x7fffffff, "anira_budget_kind terminator");
_Static_assert(sizeof(anira_warmup_mode) == 4, "anira_warmup_mode is a 32-bit enum");
_Static_assert(ANIRA_WARMUP_MODE_FORCE32 == 0x7fffffff, "anira_warmup_mode terminator");
_Static_assert(sizeof(anira_miss_policy) == 4, "anira_miss_policy is a 32-bit enum");
_Static_assert(ANIRA_MISS_POLICY_FORCE32 == 0x7fffffff, "anira_miss_policy terminator");
_Static_assert(sizeof(anira_late_policy) == 4, "anira_late_policy is a 32-bit enum");
_Static_assert(ANIRA_LATE_POLICY_FORCE32 == 0x7fffffff, "anira_late_policy terminator");
_Static_assert(sizeof(anira_priority) == 4, "anira_priority is a 32-bit enum");
_Static_assert(ANIRA_PRIORITY_FORCE32 == 0x7fffffff, "anira_priority terminator");
_Static_assert(sizeof(anira_delivery) == 4, "anira_delivery is a 32-bit enum");
_Static_assert(ANIRA_DELIVERY_FORCE32 == 0x7fffffff, "anira_delivery terminator");
_Static_assert(sizeof(anira_edge_cost) == 4, "anira_edge_cost is a 32-bit enum");
_Static_assert(ANIRA_EDGE_COST_FORCE32 == 0x7fffffff, "anira_edge_cost terminator");
_Static_assert(sizeof(anira_ownership) == 4, "anira_ownership is a 32-bit enum");
_Static_assert(ANIRA_OWNERSHIP_FORCE32 == 0x7fffffff, "anira_ownership terminator");
_Static_assert(sizeof(anira_exec_policy) == 4, "anira_exec_policy is a 32-bit enum");
_Static_assert(ANIRA_EXEC_POLICY_FORCE32 == 0x7fffffff, "anira_exec_policy terminator");
_Static_assert(sizeof(anira_gl_threads) == 4, "anira_gl_threads is a 32-bit enum");
_Static_assert(ANIRA_GL_THREADS_FORCE32 == 0x7fffffff, "anira_gl_threads terminator");
_Static_assert(sizeof(anira_wait_strategy) == 4, "anira_wait_strategy is a 32-bit enum");
_Static_assert(ANIRA_WAIT_STRATEGY_FORCE32 == 0x7fffffff, "anira_wait_strategy terminator");
_Static_assert(sizeof(anira_log_level) == 4, "anira_log_level is a 32-bit enum");
_Static_assert(ANIRA_LOG_LEVEL_FORCE32 == 0x7fffffff, "anira_log_level terminator");
_Static_assert(sizeof(anira_log_drain) == 4, "anira_log_drain is a 32-bit enum");
_Static_assert(ANIRA_LOG_DRAIN_FORCE32 == 0x7fffffff, "anira_log_drain terminator");
_Static_assert(sizeof(anira_edge_class) == 4, "anira_edge_class is a 32-bit enum");
_Static_assert(ANIRA_EDGE_CLASS_FORCE32 == 0x7fffffff, "anira_edge_class terminator");
_Static_assert(sizeof(anira_probe_rung) == 4, "anira_probe_rung is a 32-bit enum");
_Static_assert(ANIRA_RUNG_FORCE32 == 0x7fffffff, "anira_probe_rung terminator");
_Static_assert(sizeof(anira_model_state) == 4, "anira_model_state is a 32-bit enum");
_Static_assert(ANIRA_MODEL_STATE_FORCE32 == 0x7fffffff, "anira_model_state terminator");
_Static_assert(sizeof(anira_bytes_ownership) == 4, "anira_bytes_ownership is a 32-bit enum");
_Static_assert(ANIRA_BYTES_OWNERSHIP_FORCE32 == 0x7fffffff, "anira_bytes_ownership terminator");
_Static_assert(sizeof(anira_ticket_status) == 4, "anira_ticket_status is a 32-bit enum");
_Static_assert(ANIRA_TICKET_STATUS_FORCE32 == 0x7fffffff, "anira_ticket_status terminator");
_Static_assert(sizeof(anira_pad_policy) == 4, "anira_pad_policy is a 32-bit enum");
_Static_assert(ANIRA_PAD_POLICY_FORCE32 == 0x7fffffff, "anira_pad_policy terminator");
_Static_assert(sizeof(anira_engine) == 4, "anira_engine is a 32-bit enum");
_Static_assert(ANIRA_ENGINE_FORCE32 == 0x7fffffff, "anira_engine terminator");
_Static_assert(sizeof(anira_provider) == 4, "anira_provider is a 32-bit enum");
_Static_assert(ANIRA_PROVIDER_FORCE32 == 0x7fffffff, "anira_provider terminator");
_Static_assert(sizeof(anira_stage_phase) == 4, "anira_stage_phase is a 32-bit enum");
_Static_assert(ANIRA_STAGE_PHASE_FORCE32 == 0x7fffffff, "anira_stage_phase terminator");

_Static_assert(ANIRA_ABI_VERSION_MAJOR(ANIRA_ABI_VERSION) == ANIRA_ABI_MAJOR, "abi major round trip");
_Static_assert(ANIRA_ABI_VERSION_MINOR(ANIRA_ABI_VERSION) == ANIRA_ABI_MINOR, "abi minor round trip");
_Static_assert(ANIRA_DTYPE_F32 == 0x00012002u, "DLPack float32 packing");
_Static_assert(ANIRA_DTYPE_CODE(ANIRA_DTYPE_F32) == ANIRA_DTYPE_FLOAT, "dtype code");
_Static_assert(ANIRA_DTYPE_BITS(ANIRA_DTYPE_F32) == 32, "dtype bits");
_Static_assert(ANIRA_DTYPE_LANES(ANIRA_DTYPE_F32) == 1, "dtype lanes");

_Static_assert(sizeof(anira_error) == 520, "anira_error size");
_Static_assert(_Alignof(anira_error) == 4, "anira_error align");
_Static_assert(offsetof(anira_error, status) == 0, "anira_error.status offset");
_Static_assert(sizeof(((const anira_error*)0)->status) == 4, "anira_error.status size");
_Static_assert(offsetof(anira_error, reserved) == 4, "anira_error.reserved offset");
_Static_assert(sizeof(((const anira_error*)0)->reserved) == 4, "anira_error.reserved size");
_Static_assert(offsetof(anira_error, message) == 8, "anira_error.message offset");
_Static_assert(sizeof(((const anira_error*)0)->message) == 512, "anira_error.message size");

_Static_assert(sizeof(anira_log_record) == 56, "anira_log_record size");
_Static_assert(_Alignof(anira_log_record) == 8, "anira_log_record align");
_Static_assert(offsetof(anira_log_record, level) == 0, "anira_log_record.level offset");
_Static_assert(sizeof(((const anira_log_record*)0)->level) == 4, "anira_log_record.level size");
_Static_assert(offsetof(anira_log_record, flags) == 4, "anira_log_record.flags offset");
_Static_assert(sizeof(((const anira_log_record*)0)->flags) == 4, "anira_log_record.flags size");
_Static_assert(offsetof(anira_log_record, dropped_before) == 8, "anira_log_record.dropped_before offset");
_Static_assert(sizeof(((const anira_log_record*)0)->dropped_before) == 4, "anira_log_record.dropped_before size");
_Static_assert(offsetof(anira_log_record, reserved) == 12, "anira_log_record.reserved offset");
_Static_assert(sizeof(((const anira_log_record*)0)->reserved) == 4, "anira_log_record.reserved size");
_Static_assert(offsetof(anira_log_record, sequence) == 16, "anira_log_record.sequence offset");
_Static_assert(sizeof(((const anira_log_record*)0)->sequence) == 8, "anira_log_record.sequence size");
_Static_assert(offsetof(anira_log_record, timestamp_ms) == 24, "anira_log_record.timestamp_ms offset");
_Static_assert(sizeof(((const anira_log_record*)0)->timestamp_ms) == 8, "anira_log_record.timestamp_ms size");
_Static_assert(offsetof(anira_log_record, monotonic_ns) == 32, "anira_log_record.monotonic_ns offset");
_Static_assert(sizeof(((const anira_log_record*)0)->monotonic_ns) == 8, "anira_log_record.monotonic_ns size");
_Static_assert(offsetof(anira_log_record, group) == 40, "anira_log_record.group offset");
_Static_assert(sizeof(((const anira_log_record*)0)->group_bits) == 8, "anira_log_record.group is an 8-byte slot");
_Static_assert(offsetof(anira_log_record, message) == 48, "anira_log_record.message offset");
_Static_assert(sizeof(((const anira_log_record*)0)->message_bits) == 8, "anira_log_record.message is an 8-byte slot");

_Static_assert(offsetof(anira_log_desc, struct_size) == 0, "anira_log_desc.struct_size first");
_Static_assert(offsetof(anira_log_desc, abi_version) == 4, "anira_log_desc.abi_version second");
_Static_assert(offsetof(anira_log_desc, user_data) == 8, "anira_log_desc.user_data third");

_Static_assert(offsetof(anira_ext_header, struct_size) == 0, "anira_ext_header.struct_size first");

_Static_assert(offsetof(anira_ext_entry, header) == 0, "anira_ext_entry.header (an anira_ext_header) first");
_Static_assert(offsetof(anira_ext_entry, header.struct_size) == 0, "anira_ext_entry: struct_size first through the header");

_Static_assert(offsetof(anira_cuda_desc, struct_size) == 0, "anira_cuda_desc.struct_size first");

_Static_assert(offsetof(anira_gl_desc, struct_size) == 0, "anira_gl_desc.struct_size first");

_Static_assert(offsetof(anira_vulkan_desc, struct_size) == 0, "anira_vulkan_desc.struct_size first");

_Static_assert(offsetof(anira_metal_desc, struct_size) == 0, "anira_metal_desc.struct_size first");

_Static_assert(offsetof(anira_d3d12_desc, struct_size) == 0, "anira_d3d12_desc.struct_size first");

_Static_assert(offsetof(anira_webgpu_desc, struct_size) == 0, "anira_webgpu_desc.struct_size first");

_Static_assert(offsetof(anira_backend_id, struct_size) == 0, "anira_backend_id.struct_size first");

_Static_assert(offsetof(anira_edge_info, struct_size) == 0, "anira_edge_info.struct_size first");

int main(void) {
    printf("struct anira_error size %u align %u\n", (unsigned)sizeof(anira_error), (unsigned)_Alignof(anira_error));
    printf("field anira_error.status offset %u size %u\n", (unsigned)offsetof(anira_error, status), (unsigned)sizeof(((const anira_error*)0)->status));
    printf("field anira_error.reserved offset %u size %u\n", (unsigned)offsetof(anira_error, reserved), (unsigned)sizeof(((const anira_error*)0)->reserved));
    printf("field anira_error.message offset %u size %u\n", (unsigned)offsetof(anira_error, message), (unsigned)sizeof(((const anira_error*)0)->message));
    printf("struct anira_log_record size %u align %u\n", (unsigned)sizeof(anira_log_record), (unsigned)_Alignof(anira_log_record));
    printf("field anira_log_record.level offset %u size %u\n", (unsigned)offsetof(anira_log_record, level), (unsigned)sizeof(((const anira_log_record*)0)->level));
    printf("field anira_log_record.flags offset %u size %u\n", (unsigned)offsetof(anira_log_record, flags), (unsigned)sizeof(((const anira_log_record*)0)->flags));
    printf("field anira_log_record.dropped_before offset %u size %u\n", (unsigned)offsetof(anira_log_record, dropped_before), (unsigned)sizeof(((const anira_log_record*)0)->dropped_before));
    printf("field anira_log_record.reserved offset %u size %u\n", (unsigned)offsetof(anira_log_record, reserved), (unsigned)sizeof(((const anira_log_record*)0)->reserved));
    printf("field anira_log_record.sequence offset %u size %u\n", (unsigned)offsetof(anira_log_record, sequence), (unsigned)sizeof(((const anira_log_record*)0)->sequence));
    printf("field anira_log_record.timestamp_ms offset %u size %u\n", (unsigned)offsetof(anira_log_record, timestamp_ms), (unsigned)sizeof(((const anira_log_record*)0)->timestamp_ms));
    printf("field anira_log_record.monotonic_ns offset %u size %u\n", (unsigned)offsetof(anira_log_record, monotonic_ns), (unsigned)sizeof(((const anira_log_record*)0)->monotonic_ns));
    printf("field anira_log_record.group offset %u size %u\n", (unsigned)offsetof(anira_log_record, group), 8u);
    printf("field anira_log_record.message offset %u size %u\n", (unsigned)offsetof(anira_log_record, message), 8u);
    return 0;
}
