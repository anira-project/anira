/*
 * test/abi/generated/link_probe.c -- the presence gate anira_abi_link, generated from
 * abi/anira.yml by tools/abi/gen.py. Takes the address of every promised and draft entry
 * point, so an entry without a definition fails this link, not a consumer's. Do not edit.
 */
#include <stdint.h>
#include <stdio.h>

#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/config.h>
#include <anira/abi/machine.h>
#include <anira/abi/thread.h>

struct anira_link_entry {
    const char* name;
    uintptr_t address;
};

static const struct anira_link_entry k_entries[] = {
    {"anira_abi_version", (uintptr_t)&anira_abi_version},
    {"anira_capabilities_backends", (uintptr_t)&anira_capabilities_backends},
    {"anira_capabilities_domains", (uintptr_t)&anira_capabilities_domains},
    {"anira_capabilities_edge", (uintptr_t)&anira_capabilities_edge},
    {"anira_capabilities_edges", (uintptr_t)&anira_capabilities_edges},
    {"anira_capabilities_ext_kinds", (uintptr_t)&anira_capabilities_ext_kinds},
    {"anira_check_abi", (uintptr_t)&anira_check_abi},
    {"anira_contract_async_set_deadline", (uintptr_t)&anira_contract_async_set_deadline},
    {"anira_contract_async_set_policy", (uintptr_t)&anira_contract_async_set_policy},
    {"anira_contract_create_async", (uintptr_t)&anira_contract_create_async},
    {"anira_contract_create_hard", (uintptr_t)&anira_contract_create_hard},
    {"anira_contract_destroy", (uintptr_t)&anira_contract_destroy},
    {"anira_contract_from_json", (uintptr_t)&anira_contract_from_json},
    {"anira_contract_get_kind", (uintptr_t)&anira_contract_get_kind},
    {"anira_contract_hard_set_budget", (uintptr_t)&anira_contract_hard_set_budget},
    {"anira_contract_hard_set_geometry", (uintptr_t)&anira_contract_hard_set_geometry},
    {"anira_contract_hard_set_on_miss", (uintptr_t)&anira_contract_hard_set_on_miss},
    {"anira_contract_hard_set_ring_dtype", (uintptr_t)&anira_contract_hard_set_ring_dtype},
    {"anira_contract_hard_set_wait_ratio", (uintptr_t)&anira_contract_hard_set_wait_ratio},
    {"anira_contract_hard_set_warmup", (uintptr_t)&anira_contract_hard_set_warmup},
    {"anira_contract_set_edge_cost", (uintptr_t)&anira_contract_set_edge_cost},
    {"anira_contract_set_ext", (uintptr_t)&anira_contract_set_ext},
    {"anira_contract_set_ext_json", (uintptr_t)&anira_contract_set_ext_json},
    {"anira_drain_log", (uintptr_t)&anira_drain_log},
    {"anira_enabled_backends", (uintptr_t)&anira_enabled_backends},
    {"anira_has_core", (uintptr_t)&anira_has_core},
    {"anira_inference_thread_create", (uintptr_t)&anira_inference_thread_create},
    {"anira_inference_thread_destroy", (uintptr_t)&anira_inference_thread_destroy},
    {"anira_inference_thread_execute", (uintptr_t)&anira_inference_thread_execute},
    {"anira_inference_thread_has_exited", (uintptr_t)&anira_inference_thread_has_exited},
    {"anira_inference_thread_is_running", (uintptr_t)&anira_inference_thread_is_running},
    {"anira_inference_thread_run_loop", (uintptr_t)&anira_inference_thread_run_loop},
    {"anira_inference_thread_should_exit", (uintptr_t)&anira_inference_thread_should_exit},
    {"anira_inference_thread_start", (uintptr_t)&anira_inference_thread_start},
    {"anira_inference_thread_stop", (uintptr_t)&anira_inference_thread_stop},
    {"anira_job_options_create", (uintptr_t)&anira_job_options_create},
    {"anira_job_options_destroy", (uintptr_t)&anira_job_options_destroy},
    {"anira_job_options_set_below_min", (uintptr_t)&anira_job_options_set_below_min},
    {"anira_job_options_set_ext", (uintptr_t)&anira_job_options_set_ext},
    {"anira_job_options_set_ext_json", (uintptr_t)&anira_job_options_set_ext_json},
    {"anira_job_options_set_head_trim", (uintptr_t)&anira_job_options_set_head_trim},
    {"anira_job_options_set_tail_flush", (uintptr_t)&anira_job_options_set_tail_flush},
    {"anira_log", (uintptr_t)&anira_log},
    {"anira_log_rt", (uintptr_t)&anira_log_rt},
    {"anira_machine_byte_image_bytes", (uintptr_t)&anira_machine_byte_image_bytes},
    {"anira_machine_capabilities", (uintptr_t)&anira_machine_capabilities},
    {"anira_machine_config_create", (uintptr_t)&anira_machine_config_create},
    {"anira_machine_config_destroy", (uintptr_t)&anira_machine_config_destroy},
    {"anira_machine_config_from_json", (uintptr_t)&anira_machine_config_from_json},
    {"anira_machine_config_set_cuda", (uintptr_t)&anira_machine_config_set_cuda},
    {"anira_machine_config_set_d3d12", (uintptr_t)&anira_machine_config_set_d3d12},
    {"anira_machine_config_set_ext", (uintptr_t)&anira_machine_config_set_ext},
    {"anira_machine_config_set_ext_json", (uintptr_t)&anira_machine_config_set_ext_json},
    {"anira_machine_config_set_gl", (uintptr_t)&anira_machine_config_set_gl},
    {"anira_machine_config_set_log", (uintptr_t)&anira_machine_config_set_log},
    {"anira_machine_config_set_log_drain", (uintptr_t)&anira_machine_config_set_log_drain},
    {"anira_machine_config_set_log_flags", (uintptr_t)&anira_machine_config_set_log_flags},
    {"anira_machine_config_set_log_level", (uintptr_t)&anira_machine_config_set_log_level},
    {"anira_machine_config_set_log_queue_capacity", (uintptr_t)&anira_machine_config_set_log_queue_capacity},
    {"anira_machine_config_set_log_sink", (uintptr_t)&anira_machine_config_set_log_sink},
    {"anira_machine_config_set_metal", (uintptr_t)&anira_machine_config_set_metal},
    {"anira_machine_config_set_threads", (uintptr_t)&anira_machine_config_set_threads},
    {"anira_machine_config_set_vulkan", (uintptr_t)&anira_machine_config_set_vulkan},
    {"anira_machine_config_set_webgpu", (uintptr_t)&anira_machine_config_set_webgpu},
    {"anira_machine_config_to_json", (uintptr_t)&anira_machine_config_to_json},
    {"anira_machine_create", (uintptr_t)&anira_machine_create},
    {"anira_machine_destroy", (uintptr_t)&anira_machine_destroy},
    {"anira_machine_drain_log", (uintptr_t)&anira_machine_drain_log},
    {"anira_machine_num_inference_threads", (uintptr_t)&anira_machine_num_inference_threads},
    {"anira_machine_probe", (uintptr_t)&anira_machine_probe},
    {"anira_model_config_add_input", (uintptr_t)&anira_model_config_add_input},
    {"anira_model_config_add_model_bytes", (uintptr_t)&anira_model_config_add_model_bytes},
    {"anira_model_config_add_model_bytes_custom", (uintptr_t)&anira_model_config_add_model_bytes_custom},
    {"anira_model_config_add_model_path", (uintptr_t)&anira_model_config_add_model_path},
    {"anira_model_config_add_model_path_custom", (uintptr_t)&anira_model_config_add_model_path_custom},
    {"anira_model_config_add_output", (uintptr_t)&anira_model_config_add_output},
    {"anira_model_config_create", (uintptr_t)&anira_model_config_create},
    {"anira_model_config_destroy", (uintptr_t)&anira_model_config_destroy},
    {"anira_model_config_from_json", (uintptr_t)&anira_model_config_from_json},
    {"anira_model_config_from_json_file", (uintptr_t)&anira_model_config_from_json_file},
    {"anira_model_config_model_bytes", (uintptr_t)&anira_model_config_model_bytes},
    {"anira_model_config_model_count", (uintptr_t)&anira_model_config_model_count},
    {"anira_model_config_model_engine", (uintptr_t)&anira_model_config_model_engine},
    {"anira_model_config_model_engine_id", (uintptr_t)&anira_model_config_model_engine_id},
    {"anira_model_config_model_path", (uintptr_t)&anira_model_config_model_path},
    {"anira_model_config_set_anchor", (uintptr_t)&anira_model_config_set_anchor},
    {"anira_model_config_set_default_engine", (uintptr_t)&anira_model_config_set_default_engine},
    {"anira_model_config_set_default_engine_custom", (uintptr_t)&anira_model_config_set_default_engine_custom},
    {"anira_model_config_set_ext", (uintptr_t)&anira_model_config_set_ext},
    {"anira_model_config_set_ext_json", (uintptr_t)&anira_model_config_set_ext_json},
    {"anira_model_config_set_max_instances", (uintptr_t)&anira_model_config_set_max_instances},
    {"anira_model_config_set_model_bytes", (uintptr_t)&anira_model_config_set_model_bytes},
    {"anira_model_config_set_model_ext", (uintptr_t)&anira_model_config_set_model_ext},
    {"anira_model_config_set_model_ext_json", (uintptr_t)&anira_model_config_set_model_ext_json},
    {"anira_model_config_set_state", (uintptr_t)&anira_model_config_set_state},
    {"anira_model_config_set_tensor_layout", (uintptr_t)&anira_model_config_set_tensor_layout},
    {"anira_model_config_set_tensor_name", (uintptr_t)&anira_model_config_set_tensor_name},
    {"anira_model_config_take_legacy_contract", (uintptr_t)&anira_model_config_take_legacy_contract},
    {"anira_model_config_to_json", (uintptr_t)&anira_model_config_to_json},
    {"anira_now_ms", (uintptr_t)&anira_now_ms},
    {"anira_now_ns", (uintptr_t)&anira_now_ns},
    {"anira_num_inference_threads", (uintptr_t)&anira_num_inference_threads},
    {"anira_registered_ext_kinds", (uintptr_t)&anira_registered_ext_kinds},
    {"anira_release_core_if_idle", (uintptr_t)&anira_release_core_if_idle},
    {"anira_shutdown", (uintptr_t)&anira_shutdown},
    {"anira_status_string", (uintptr_t)&anira_status_string},
    {"anira_tensor_spec_create", (uintptr_t)&anira_tensor_spec_create},
    {"anira_tensor_spec_destroy", (uintptr_t)&anira_tensor_spec_destroy},
    {"anira_tensor_spec_set_axis", (uintptr_t)&anira_tensor_spec_set_axis},
    {"anira_tensor_spec_set_ext", (uintptr_t)&anira_tensor_spec_set_ext},
    {"anira_tensor_spec_set_ext_json", (uintptr_t)&anira_tensor_spec_set_ext_json},
    {"anira_tensor_spec_set_latency", (uintptr_t)&anira_tensor_spec_set_latency},
    {"anira_tensor_spec_set_time_ratio", (uintptr_t)&anira_tensor_spec_set_time_ratio},
    {"anira_tensor_spec_set_window", (uintptr_t)&anira_tensor_spec_set_window},
    {"anira_version", (uintptr_t)&anira_version},
    {"anira_version_string", (uintptr_t)&anira_version_string},
};

int main(void) {
    const size_t count = sizeof(k_entries) / sizeof(k_entries[0]);
    size_t missing = 0;
    size_t i;
    for (i = 0; i < count; ++i) {
        if (k_entries[i].address == 0) {
            printf("missing: %s\n", k_entries[i].name);
            ++missing;
        }
    }
    printf("%zu of %zu entry points linked\n", count - missing, count);
    return missing == 0 ? 0 : 1;
}
