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
#include <anira/abi/tensor.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/thread.h>
#include <anira/abi/stage.h>
#include <anira/abi/handler.h>
#include <anira/abi/draft/tensor_platform.h>

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

_Static_assert(sizeof(anira_memory_handle) == 24, "anira_memory_handle size");
_Static_assert(_Alignof(anira_memory_handle) == 8, "anira_memory_handle align");
_Static_assert(offsetof(anira_memory_handle, host) == 0, "anira_memory_handle.host offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->host) == 8, "anira_memory_handle.host size");
_Static_assert(offsetof(anira_memory_handle, host.ptr) == 0, "anira_memory_handle.host.ptr offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->host.ptr_bits) == 8, "anira_memory_handle.host.ptr is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, cuda) == 0, "anira_memory_handle.cuda offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->cuda) == 16, "anira_memory_handle.cuda size");
_Static_assert(offsetof(anira_memory_handle, cuda.ptr) == 0, "anira_memory_handle.cuda.ptr offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->cuda.ptr_bits) == 8, "anira_memory_handle.cuda.ptr is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, cuda.device) == 8, "anira_memory_handle.cuda.device offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->cuda.device) == 4, "anira_memory_handle.cuda.device size");
_Static_assert(offsetof(anira_memory_handle, gl) == 0, "anira_memory_handle.gl offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->gl) == 8, "anira_memory_handle.gl size");
_Static_assert(offsetof(anira_memory_handle, gl.id) == 0, "anira_memory_handle.gl.id offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->gl.id) == 4, "anira_memory_handle.gl.id size");
_Static_assert(offsetof(anira_memory_handle, gl.target) == 4, "anira_memory_handle.gl.target offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->gl.target) == 4, "anira_memory_handle.gl.target size");
_Static_assert(offsetof(anira_memory_handle, vk) == 0, "anira_memory_handle.vk offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->vk) == 24, "anira_memory_handle.vk size");
_Static_assert(offsetof(anira_memory_handle, vk.buffer) == 0, "anira_memory_handle.vk.buffer offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->vk.buffer) == 8, "anira_memory_handle.vk.buffer size");
_Static_assert(offsetof(anira_memory_handle, vk.memory) == 8, "anira_memory_handle.vk.memory offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->vk.memory) == 8, "anira_memory_handle.vk.memory size");
_Static_assert(offsetof(anira_memory_handle, vk.offset) == 16, "anira_memory_handle.vk.offset offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->vk.offset) == 8, "anira_memory_handle.vk.offset size");
_Static_assert(offsetof(anira_memory_handle, opaque) == 0, "anira_memory_handle.opaque offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->opaque) == 16, "anira_memory_handle.opaque size");
_Static_assert(offsetof(anira_memory_handle, opaque.fd) == 0, "anira_memory_handle.opaque.fd offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->opaque.fd) == 4, "anira_memory_handle.opaque.fd size");
_Static_assert(offsetof(anira_memory_handle, opaque.reserved) == 4, "anira_memory_handle.opaque.reserved offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->opaque.reserved) == 4, "anira_memory_handle.opaque.reserved size");
_Static_assert(offsetof(anira_memory_handle, opaque.size) == 8, "anira_memory_handle.opaque.size offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->opaque.size) == 8, "anira_memory_handle.opaque.size size");
_Static_assert(offsetof(anira_memory_handle, mtl) == 0, "anira_memory_handle.mtl offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->mtl) == 8, "anira_memory_handle.mtl size");
_Static_assert(offsetof(anira_memory_handle, mtl.buffer) == 0, "anira_memory_handle.mtl.buffer offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->mtl.buffer_bits) == 8, "anira_memory_handle.mtl.buffer is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, iosurface) == 0, "anira_memory_handle.iosurface offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->iosurface) == 16, "anira_memory_handle.iosurface size");
_Static_assert(offsetof(anira_memory_handle, iosurface.surface) == 0, "anira_memory_handle.iosurface.surface offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->iosurface.surface_bits) == 8, "anira_memory_handle.iosurface.surface is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, iosurface.size) == 8, "anira_memory_handle.iosurface.size offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->iosurface.size) == 8, "anira_memory_handle.iosurface.size size");
_Static_assert(offsetof(anira_memory_handle, wgpu) == 0, "anira_memory_handle.wgpu offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->wgpu) == 16, "anira_memory_handle.wgpu size");
_Static_assert(offsetof(anira_memory_handle, wgpu.buffer) == 0, "anira_memory_handle.wgpu.buffer offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->wgpu.buffer_bits) == 8, "anira_memory_handle.wgpu.buffer is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, wgpu.offset) == 8, "anira_memory_handle.wgpu.offset offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->wgpu.offset) == 8, "anira_memory_handle.wgpu.offset size");
_Static_assert(offsetof(anira_memory_handle, dmabuf) == 0, "anira_memory_handle.dmabuf offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->dmabuf) == 24, "anira_memory_handle.dmabuf size");
_Static_assert(offsetof(anira_memory_handle, dmabuf.fd) == 0, "anira_memory_handle.dmabuf.fd offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->dmabuf.fd) == 4, "anira_memory_handle.dmabuf.fd size");
_Static_assert(offsetof(anira_memory_handle, dmabuf.reserved) == 4, "anira_memory_handle.dmabuf.reserved offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->dmabuf.reserved) == 4, "anira_memory_handle.dmabuf.reserved size");
_Static_assert(offsetof(anira_memory_handle, dmabuf.size) == 8, "anira_memory_handle.dmabuf.size offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->dmabuf.size) == 8, "anira_memory_handle.dmabuf.size size");
_Static_assert(offsetof(anira_memory_handle, dmabuf.offset) == 16, "anira_memory_handle.dmabuf.offset offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->dmabuf.offset) == 8, "anira_memory_handle.dmabuf.offset size");
_Static_assert(offsetof(anira_memory_handle, ahb) == 0, "anira_memory_handle.ahb offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->ahb) == 8, "anira_memory_handle.ahb size");
_Static_assert(offsetof(anira_memory_handle, ahb.buffer) == 0, "anira_memory_handle.ahb.buffer offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->ahb.buffer_bits) == 8, "anira_memory_handle.ahb.buffer is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, d3d12) == 0, "anira_memory_handle.d3d12 offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->d3d12) == 16, "anira_memory_handle.d3d12 size");
_Static_assert(offsetof(anira_memory_handle, d3d12.resource) == 0, "anira_memory_handle.d3d12.resource offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->d3d12.resource_bits) == 8, "anira_memory_handle.d3d12.resource is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, d3d12.shared_handle) == 8, "anira_memory_handle.d3d12.shared_handle offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->d3d12.shared_handle_bits) == 8, "anira_memory_handle.d3d12.shared_handle is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, planes) == 0, "anira_memory_handle.planes offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->planes) == 16, "anira_memory_handle.planes size");
_Static_assert(offsetof(anira_memory_handle, planes.ptrs) == 0, "anira_memory_handle.planes.ptrs offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->planes.ptrs_bits) == 8, "anira_memory_handle.planes.ptrs is an 8-byte slot");
_Static_assert(offsetof(anira_memory_handle, planes.count) == 8, "anira_memory_handle.planes.count offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->planes.count) == 4, "anira_memory_handle.planes.count size");
_Static_assert(offsetof(anira_memory_handle, planes.reserved) == 12, "anira_memory_handle.planes.reserved offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->planes.reserved) == 4, "anira_memory_handle.planes.reserved size");
_Static_assert(offsetof(anira_memory_handle, raw) == 0, "anira_memory_handle.raw offset");
_Static_assert(sizeof(((const anira_memory_handle*)0)->raw) == 24, "anira_memory_handle.raw size");

_Static_assert(sizeof(anira_sync_token) == 24, "anira_sync_token size");
_Static_assert(_Alignof(anira_sync_token) == 8, "anira_sync_token align");
_Static_assert(offsetof(anira_sync_token, kind) == 0, "anira_sync_token.kind offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->kind) == 4, "anira_sync_token.kind size");
_Static_assert(offsetof(anira_sync_token, flags) == 4, "anira_sync_token.flags offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->flags) == 4, "anira_sync_token.flags size");
_Static_assert(offsetof(anira_sync_token, u) == 8, "anira_sync_token.u offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u) == 16, "anira_sync_token.u size");
_Static_assert(offsetof(anira_sync_token, u.cuda_event) == 8, "anira_sync_token.u.cuda_event offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.cuda_event_bits) == 8, "anira_sync_token.u.cuda_event is an 8-byte slot");
_Static_assert(offsetof(anira_sync_token, u.vk) == 8, "anira_sync_token.u.vk offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.vk) == 16, "anira_sync_token.u.vk size");
_Static_assert(offsetof(anira_sync_token, u.vk.semaphore) == 8, "anira_sync_token.u.vk.semaphore offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.vk.semaphore) == 8, "anira_sync_token.u.vk.semaphore size");
_Static_assert(offsetof(anira_sync_token, u.vk.value) == 16, "anira_sync_token.u.vk.value offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.vk.value) == 8, "anira_sync_token.u.vk.value size");
_Static_assert(offsetof(anira_sync_token, u.gl_sync) == 8, "anira_sync_token.u.gl_sync offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.gl_sync_bits) == 8, "anira_sync_token.u.gl_sync is an 8-byte slot");
_Static_assert(offsetof(anira_sync_token, u.fd) == 8, "anira_sync_token.u.fd offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.fd) == 4, "anira_sync_token.u.fd size");
_Static_assert(offsetof(anira_sync_token, u.mtl) == 8, "anira_sync_token.u.mtl offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.mtl) == 16, "anira_sync_token.u.mtl size");
_Static_assert(offsetof(anira_sync_token, u.mtl.object) == 8, "anira_sync_token.u.mtl.object offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.mtl.object_bits) == 8, "anira_sync_token.u.mtl.object is an 8-byte slot");
_Static_assert(offsetof(anira_sync_token, u.mtl.value) == 16, "anira_sync_token.u.mtl.value offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.mtl.value) == 8, "anira_sync_token.u.mtl.value size");
_Static_assert(offsetof(anira_sync_token, u.d3d12) == 8, "anira_sync_token.u.d3d12 offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.d3d12) == 16, "anira_sync_token.u.d3d12 size");
_Static_assert(offsetof(anira_sync_token, u.d3d12.object) == 8, "anira_sync_token.u.d3d12.object offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.d3d12.object_bits) == 8, "anira_sync_token.u.d3d12.object is an 8-byte slot");
_Static_assert(offsetof(anira_sync_token, u.d3d12.value) == 16, "anira_sync_token.u.d3d12.value offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.d3d12.value) == 8, "anira_sync_token.u.d3d12.value size");
_Static_assert(offsetof(anira_sync_token, u.raw) == 8, "anira_sync_token.u.raw offset");
_Static_assert(sizeof(((const anira_sync_token*)0)->u.raw) == 16, "anira_sync_token.u.raw size");

_Static_assert(sizeof(anira_tensor) == 216, "anira_tensor size");
_Static_assert(_Alignof(anira_tensor) == 8, "anira_tensor align");
_Static_assert(offsetof(anira_tensor, domain) == 0, "anira_tensor.domain offset");
_Static_assert(sizeof(((const anira_tensor*)0)->domain) == 4, "anira_tensor.domain size");
_Static_assert(offsetof(anira_tensor, dtype) == 4, "anira_tensor.dtype offset");
_Static_assert(sizeof(((const anira_tensor*)0)->dtype) == 4, "anira_tensor.dtype size");
_Static_assert(offsetof(anira_tensor, ndim) == 8, "anira_tensor.ndim offset");
_Static_assert(sizeof(((const anira_tensor*)0)->ndim) == 4, "anira_tensor.ndim size");
_Static_assert(offsetof(anira_tensor, flags) == 12, "anira_tensor.flags offset");
_Static_assert(sizeof(((const anira_tensor*)0)->flags) == 4, "anira_tensor.flags size");
_Static_assert(offsetof(anira_tensor, shape) == 16, "anira_tensor.shape offset");
_Static_assert(sizeof(((const anira_tensor*)0)->shape) == 64, "anira_tensor.shape size");
_Static_assert(offsetof(anira_tensor, strides) == 80, "anira_tensor.strides offset");
_Static_assert(sizeof(((const anira_tensor*)0)->strides) == 64, "anira_tensor.strides size");
_Static_assert(offsetof(anira_tensor, byte_offset) == 144, "anira_tensor.byte_offset offset");
_Static_assert(sizeof(((const anira_tensor*)0)->byte_offset) == 8, "anira_tensor.byte_offset size");
_Static_assert(offsetof(anira_tensor, handle) == 152, "anira_tensor.handle offset");
_Static_assert(sizeof(((const anira_tensor*)0)->handle) == 24, "anira_tensor.handle size");
_Static_assert(offsetof(anira_tensor, manager_ctx) == 176, "anira_tensor.manager_ctx offset");
_Static_assert(sizeof(((const anira_tensor*)0)->manager_ctx_bits) == 8, "anira_tensor.manager_ctx is an 8-byte slot");
_Static_assert(offsetof(anira_tensor, release) == 184, "anira_tensor.release offset");
_Static_assert(sizeof(((const anira_tensor*)0)->release_bits) == 8, "anira_tensor.release is an 8-byte slot");
_Static_assert(offsetof(anira_tensor, acquire) == 192, "anira_tensor.acquire offset");
_Static_assert(sizeof(((const anira_tensor*)0)->acquire) == 24, "anira_tensor.acquire size");

_Static_assert(offsetof(anira_ext_header, struct_size) == 0, "anira_ext_header.struct_size first");

_Static_assert(offsetof(anira_ext_entry, header) == 0, "anira_ext_entry.header (an anira_ext_header) first");
_Static_assert(offsetof(anira_ext_entry, header.struct_size) == 0, "anira_ext_entry: struct_size first through the header");

_Static_assert(offsetof(anira_cuda_desc, struct_size) == 0, "anira_cuda_desc.struct_size first");

_Static_assert(offsetof(anira_gl_desc, struct_size) == 0, "anira_gl_desc.struct_size first");

_Static_assert(offsetof(anira_vulkan_desc, struct_size) == 0, "anira_vulkan_desc.struct_size first");
_Static_assert(sizeof(anira_vulkan_desc) ==
                   sizeof(((const anira_vulkan_desc*)0)->struct_size) +
                   sizeof(((const anira_vulkan_desc*)0)->ownership) +
                   sizeof(((const anira_vulkan_desc*)0)->queue_family) +
                   sizeof(((const anira_vulkan_desc*)0)->queue_index) +
                   sizeof(((const anira_vulkan_desc*)0)->instance) +
                   sizeof(((const anira_vulkan_desc*)0)->physical) +
                   sizeof(((const anira_vulkan_desc*)0)->device) +
                   sizeof(((const anira_vulkan_desc*)0)->device_index) +
                   sizeof(((const anira_vulkan_desc*)0)->reserved),
               "anira_vulkan_desc has no implicit padding");

_Static_assert(offsetof(anira_metal_desc, struct_size) == 0, "anira_metal_desc.struct_size first");

_Static_assert(offsetof(anira_d3d12_desc, struct_size) == 0, "anira_d3d12_desc.struct_size first");

_Static_assert(offsetof(anira_webgpu_desc, struct_size) == 0, "anira_webgpu_desc.struct_size first");

_Static_assert(offsetof(anira_backend_id, struct_size) == 0, "anira_backend_id.struct_size first");

_Static_assert(offsetof(anira_edge_info, struct_size) == 0, "anira_edge_info.struct_size first");

_Static_assert(sizeof(anira_stage_ctx) == 64, "anira_stage_ctx size");
_Static_assert(_Alignof(anira_stage_ctx) == 8, "anira_stage_ctx align");
_Static_assert(offsetof(anira_stage_ctx, phase) == 0, "anira_stage_ctx.phase offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->phase) == 4, "anira_stage_ctx.phase size");
_Static_assert(offsetof(anira_stage_ctx, engine) == 4, "anira_stage_ctx.engine offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->engine) == 4, "anira_stage_ctx.engine size");
_Static_assert(offsetof(anira_stage_ctx, provider) == 8, "anira_stage_ctx.provider offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->provider) == 4, "anira_stage_ctx.provider size");
_Static_assert(offsetof(anira_stage_ctx, variant) == 12, "anira_stage_ctx.variant offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->variant) == 4, "anira_stage_ctx.variant size");
_Static_assert(offsetof(anira_stage_ctx, num_inputs) == 16, "anira_stage_ctx.num_inputs offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->num_inputs) == 4, "anira_stage_ctx.num_inputs size");
_Static_assert(offsetof(anira_stage_ctx, num_outputs) == 20, "anira_stage_ctx.num_outputs offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->num_outputs) == 4, "anira_stage_ctx.num_outputs size");
_Static_assert(offsetof(anira_stage_ctx, ticket) == 24, "anira_stage_ctx.ticket offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->ticket) == 4, "anira_stage_ctx.ticket size");
_Static_assert(offsetof(anira_stage_ctx, reserved) == 28, "anira_stage_ctx.reserved offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->reserved) == 4, "anira_stage_ctx.reserved size");
_Static_assert(offsetof(anira_stage_ctx, input_rings) == 32, "anira_stage_ctx.input_rings offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->input_rings_bits) == 8, "anira_stage_ctx.input_rings is an 8-byte slot");
_Static_assert(offsetof(anira_stage_ctx, model_inputs) == 40, "anira_stage_ctx.model_inputs offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->model_inputs_bits) == 8, "anira_stage_ctx.model_inputs is an 8-byte slot");
_Static_assert(offsetof(anira_stage_ctx, model_outputs) == 48, "anira_stage_ctx.model_outputs offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->model_outputs_bits) == 8, "anira_stage_ctx.model_outputs is an 8-byte slot");
_Static_assert(offsetof(anira_stage_ctx, output_rings) == 56, "anira_stage_ctx.output_rings offset");
_Static_assert(sizeof(((const anira_stage_ctx*)0)->output_rings_bits) == 8, "anira_stage_ctx.output_rings is an 8-byte slot");

_Static_assert(offsetof(anira_stage_desc, struct_size) == 0, "anira_stage_desc.struct_size first");
_Static_assert(offsetof(anira_stage_desc, abi_version) == 4, "anira_stage_desc.abi_version second");
_Static_assert(offsetof(anira_stage_desc, user_data) == 8, "anira_stage_desc.user_data third");
_Static_assert(sizeof(anira_stage_desc) ==
                   sizeof(((const anira_stage_desc*)0)->struct_size) +
                   sizeof(((const anira_stage_desc*)0)->abi_version) +
                   sizeof(((const anira_stage_desc*)0)->user_data) +
                   sizeof(((const anira_stage_desc*)0)->name) +
                   sizeof(((const anira_stage_desc*)0)->domain_in) +
                   sizeof(((const anira_stage_desc*)0)->domain_out) +
                   sizeof(((const anira_stage_desc*)0)->consumed_kinds) +
                   sizeof(((const anira_stage_desc*)0)->num_consumed_kinds) +
                   sizeof(((const anira_stage_desc*)0)->reserved) +
                   sizeof(((const anira_stage_desc*)0)->pre_process) +
                   sizeof(((const anira_stage_desc*)0)->post_process) +
                   sizeof(((const anira_stage_desc*)0)->before_inference) +
                   sizeof(((const anira_stage_desc*)0)->after_inference) +
                   sizeof(((const anira_stage_desc*)0)->prepare) +
                   sizeof(((const anira_stage_desc*)0)->release),
               "anira_stage_desc has no implicit padding");

_Static_assert(offsetof(anira_plan_slot, struct_size) == 0, "anira_plan_slot.struct_size first");

_Static_assert(offsetof(anira_plan_ext, struct_size) == 0, "anira_plan_ext.struct_size first");

_Static_assert(offsetof(anira_plan_info, struct_size) == 0, "anira_plan_info.struct_size first");

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
    printf("union anira_memory_handle size %u align %u\n", (unsigned)sizeof(anira_memory_handle), (unsigned)_Alignof(anira_memory_handle));
    printf("field anira_memory_handle.host offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, host), (unsigned)sizeof(((const anira_memory_handle*)0)->host));
    printf("field anira_memory_handle.host.ptr offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, host.ptr), 8u);
    printf("field anira_memory_handle.cuda offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, cuda), (unsigned)sizeof(((const anira_memory_handle*)0)->cuda));
    printf("field anira_memory_handle.cuda.ptr offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, cuda.ptr), 8u);
    printf("field anira_memory_handle.cuda.device offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, cuda.device), (unsigned)sizeof(((const anira_memory_handle*)0)->cuda.device));
    printf("field anira_memory_handle.gl offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, gl), (unsigned)sizeof(((const anira_memory_handle*)0)->gl));
    printf("field anira_memory_handle.gl.id offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, gl.id), (unsigned)sizeof(((const anira_memory_handle*)0)->gl.id));
    printf("field anira_memory_handle.gl.target offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, gl.target), (unsigned)sizeof(((const anira_memory_handle*)0)->gl.target));
    printf("field anira_memory_handle.vk offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, vk), (unsigned)sizeof(((const anira_memory_handle*)0)->vk));
    printf("field anira_memory_handle.vk.buffer offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, vk.buffer), (unsigned)sizeof(((const anira_memory_handle*)0)->vk.buffer));
    printf("field anira_memory_handle.vk.memory offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, vk.memory), (unsigned)sizeof(((const anira_memory_handle*)0)->vk.memory));
    printf("field anira_memory_handle.vk.offset offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, vk.offset), (unsigned)sizeof(((const anira_memory_handle*)0)->vk.offset));
    printf("field anira_memory_handle.opaque offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, opaque), (unsigned)sizeof(((const anira_memory_handle*)0)->opaque));
    printf("field anira_memory_handle.opaque.fd offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, opaque.fd), (unsigned)sizeof(((const anira_memory_handle*)0)->opaque.fd));
    printf("field anira_memory_handle.opaque.reserved offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, opaque.reserved), (unsigned)sizeof(((const anira_memory_handle*)0)->opaque.reserved));
    printf("field anira_memory_handle.opaque.size offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, opaque.size), (unsigned)sizeof(((const anira_memory_handle*)0)->opaque.size));
    printf("field anira_memory_handle.mtl offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, mtl), (unsigned)sizeof(((const anira_memory_handle*)0)->mtl));
    printf("field anira_memory_handle.mtl.buffer offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, mtl.buffer), 8u);
    printf("field anira_memory_handle.iosurface offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, iosurface), (unsigned)sizeof(((const anira_memory_handle*)0)->iosurface));
    printf("field anira_memory_handle.iosurface.surface offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, iosurface.surface), 8u);
    printf("field anira_memory_handle.iosurface.size offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, iosurface.size), (unsigned)sizeof(((const anira_memory_handle*)0)->iosurface.size));
    printf("field anira_memory_handle.wgpu offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, wgpu), (unsigned)sizeof(((const anira_memory_handle*)0)->wgpu));
    printf("field anira_memory_handle.wgpu.buffer offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, wgpu.buffer), 8u);
    printf("field anira_memory_handle.wgpu.offset offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, wgpu.offset), (unsigned)sizeof(((const anira_memory_handle*)0)->wgpu.offset));
    printf("field anira_memory_handle.dmabuf offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, dmabuf), (unsigned)sizeof(((const anira_memory_handle*)0)->dmabuf));
    printf("field anira_memory_handle.dmabuf.fd offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, dmabuf.fd), (unsigned)sizeof(((const anira_memory_handle*)0)->dmabuf.fd));
    printf("field anira_memory_handle.dmabuf.reserved offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, dmabuf.reserved), (unsigned)sizeof(((const anira_memory_handle*)0)->dmabuf.reserved));
    printf("field anira_memory_handle.dmabuf.size offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, dmabuf.size), (unsigned)sizeof(((const anira_memory_handle*)0)->dmabuf.size));
    printf("field anira_memory_handle.dmabuf.offset offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, dmabuf.offset), (unsigned)sizeof(((const anira_memory_handle*)0)->dmabuf.offset));
    printf("field anira_memory_handle.ahb offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, ahb), (unsigned)sizeof(((const anira_memory_handle*)0)->ahb));
    printf("field anira_memory_handle.ahb.buffer offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, ahb.buffer), 8u);
    printf("field anira_memory_handle.d3d12 offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, d3d12), (unsigned)sizeof(((const anira_memory_handle*)0)->d3d12));
    printf("field anira_memory_handle.d3d12.resource offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, d3d12.resource), 8u);
    printf("field anira_memory_handle.d3d12.shared_handle offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, d3d12.shared_handle), 8u);
    printf("field anira_memory_handle.planes offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, planes), (unsigned)sizeof(((const anira_memory_handle*)0)->planes));
    printf("field anira_memory_handle.planes.ptrs offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, planes.ptrs), 8u);
    printf("field anira_memory_handle.planes.count offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, planes.count), (unsigned)sizeof(((const anira_memory_handle*)0)->planes.count));
    printf("field anira_memory_handle.planes.reserved offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, planes.reserved), (unsigned)sizeof(((const anira_memory_handle*)0)->planes.reserved));
    printf("field anira_memory_handle.raw offset %u size %u\n", (unsigned)offsetof(anira_memory_handle, raw), (unsigned)sizeof(((const anira_memory_handle*)0)->raw));
    printf("struct anira_sync_token size %u align %u\n", (unsigned)sizeof(anira_sync_token), (unsigned)_Alignof(anira_sync_token));
    printf("field anira_sync_token.kind offset %u size %u\n", (unsigned)offsetof(anira_sync_token, kind), (unsigned)sizeof(((const anira_sync_token*)0)->kind));
    printf("field anira_sync_token.flags offset %u size %u\n", (unsigned)offsetof(anira_sync_token, flags), (unsigned)sizeof(((const anira_sync_token*)0)->flags));
    printf("field anira_sync_token.u offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u), (unsigned)sizeof(((const anira_sync_token*)0)->u));
    printf("field anira_sync_token.u.cuda_event offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.cuda_event), 8u);
    printf("field anira_sync_token.u.vk offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.vk), (unsigned)sizeof(((const anira_sync_token*)0)->u.vk));
    printf("field anira_sync_token.u.vk.semaphore offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.vk.semaphore), (unsigned)sizeof(((const anira_sync_token*)0)->u.vk.semaphore));
    printf("field anira_sync_token.u.vk.value offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.vk.value), (unsigned)sizeof(((const anira_sync_token*)0)->u.vk.value));
    printf("field anira_sync_token.u.gl_sync offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.gl_sync), 8u);
    printf("field anira_sync_token.u.fd offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.fd), (unsigned)sizeof(((const anira_sync_token*)0)->u.fd));
    printf("field anira_sync_token.u.mtl offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.mtl), (unsigned)sizeof(((const anira_sync_token*)0)->u.mtl));
    printf("field anira_sync_token.u.mtl.object offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.mtl.object), 8u);
    printf("field anira_sync_token.u.mtl.value offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.mtl.value), (unsigned)sizeof(((const anira_sync_token*)0)->u.mtl.value));
    printf("field anira_sync_token.u.d3d12 offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.d3d12), (unsigned)sizeof(((const anira_sync_token*)0)->u.d3d12));
    printf("field anira_sync_token.u.d3d12.object offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.d3d12.object), 8u);
    printf("field anira_sync_token.u.d3d12.value offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.d3d12.value), (unsigned)sizeof(((const anira_sync_token*)0)->u.d3d12.value));
    printf("field anira_sync_token.u.raw offset %u size %u\n", (unsigned)offsetof(anira_sync_token, u.raw), (unsigned)sizeof(((const anira_sync_token*)0)->u.raw));
    printf("struct anira_tensor size %u align %u\n", (unsigned)sizeof(anira_tensor), (unsigned)_Alignof(anira_tensor));
    printf("field anira_tensor.domain offset %u size %u\n", (unsigned)offsetof(anira_tensor, domain), (unsigned)sizeof(((const anira_tensor*)0)->domain));
    printf("field anira_tensor.dtype offset %u size %u\n", (unsigned)offsetof(anira_tensor, dtype), (unsigned)sizeof(((const anira_tensor*)0)->dtype));
    printf("field anira_tensor.ndim offset %u size %u\n", (unsigned)offsetof(anira_tensor, ndim), (unsigned)sizeof(((const anira_tensor*)0)->ndim));
    printf("field anira_tensor.flags offset %u size %u\n", (unsigned)offsetof(anira_tensor, flags), (unsigned)sizeof(((const anira_tensor*)0)->flags));
    printf("field anira_tensor.shape offset %u size %u\n", (unsigned)offsetof(anira_tensor, shape), (unsigned)sizeof(((const anira_tensor*)0)->shape));
    printf("field anira_tensor.strides offset %u size %u\n", (unsigned)offsetof(anira_tensor, strides), (unsigned)sizeof(((const anira_tensor*)0)->strides));
    printf("field anira_tensor.byte_offset offset %u size %u\n", (unsigned)offsetof(anira_tensor, byte_offset), (unsigned)sizeof(((const anira_tensor*)0)->byte_offset));
    printf("field anira_tensor.handle offset %u size %u\n", (unsigned)offsetof(anira_tensor, handle), (unsigned)sizeof(((const anira_tensor*)0)->handle));
    printf("field anira_tensor.manager_ctx offset %u size %u\n", (unsigned)offsetof(anira_tensor, manager_ctx), 8u);
    printf("field anira_tensor.release offset %u size %u\n", (unsigned)offsetof(anira_tensor, release), 8u);
    printf("field anira_tensor.acquire offset %u size %u\n", (unsigned)offsetof(anira_tensor, acquire), (unsigned)sizeof(((const anira_tensor*)0)->acquire));
    printf("struct anira_stage_ctx size %u align %u\n", (unsigned)sizeof(anira_stage_ctx), (unsigned)_Alignof(anira_stage_ctx));
    printf("field anira_stage_ctx.phase offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, phase), (unsigned)sizeof(((const anira_stage_ctx*)0)->phase));
    printf("field anira_stage_ctx.engine offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, engine), (unsigned)sizeof(((const anira_stage_ctx*)0)->engine));
    printf("field anira_stage_ctx.provider offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, provider), (unsigned)sizeof(((const anira_stage_ctx*)0)->provider));
    printf("field anira_stage_ctx.variant offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, variant), (unsigned)sizeof(((const anira_stage_ctx*)0)->variant));
    printf("field anira_stage_ctx.num_inputs offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, num_inputs), (unsigned)sizeof(((const anira_stage_ctx*)0)->num_inputs));
    printf("field anira_stage_ctx.num_outputs offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, num_outputs), (unsigned)sizeof(((const anira_stage_ctx*)0)->num_outputs));
    printf("field anira_stage_ctx.ticket offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, ticket), (unsigned)sizeof(((const anira_stage_ctx*)0)->ticket));
    printf("field anira_stage_ctx.reserved offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, reserved), (unsigned)sizeof(((const anira_stage_ctx*)0)->reserved));
    printf("field anira_stage_ctx.input_rings offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, input_rings), 8u);
    printf("field anira_stage_ctx.model_inputs offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, model_inputs), 8u);
    printf("field anira_stage_ctx.model_outputs offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, model_outputs), 8u);
    printf("field anira_stage_ctx.output_rings offset %u size %u\n", (unsigned)offsetof(anira_stage_ctx, output_rings), 8u);
    return 0;
}
