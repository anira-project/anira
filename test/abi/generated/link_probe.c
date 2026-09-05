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
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/thread.h>

struct anira_link_entry {
    const char* name;
    uintptr_t address;
};

#define ANIRA_LINK_PROBE_COUNT 114

/* The addresses are taken by assignment at run time, never in a static initializer:
   MSVC refuses the address of a dllimport there (C4232, identity not guaranteed). */
int main(void) {
    struct anira_link_entry entries[ANIRA_LINK_PROBE_COUNT];
    size_t missing = 0;
    size_t i;
    entries[0].name = "anira_abi_version";
    entries[0].address = (uintptr_t)&anira_abi_version;
    entries[1].name = "anira_capabilities_backends";
    entries[1].address = (uintptr_t)&anira_capabilities_backends;
    entries[2].name = "anira_capabilities_domains";
    entries[2].address = (uintptr_t)&anira_capabilities_domains;
    entries[3].name = "anira_capabilities_edge";
    entries[3].address = (uintptr_t)&anira_capabilities_edge;
    entries[4].name = "anira_capabilities_edges";
    entries[4].address = (uintptr_t)&anira_capabilities_edges;
    entries[5].name = "anira_capabilities_ext_kinds";
    entries[5].address = (uintptr_t)&anira_capabilities_ext_kinds;
    entries[6].name = "anira_check_abi";
    entries[6].address = (uintptr_t)&anira_check_abi;
    entries[7].name = "anira_context_byte_image_bytes";
    entries[7].address = (uintptr_t)&anira_context_byte_image_bytes;
    entries[8].name = "anira_context_capabilities";
    entries[8].address = (uintptr_t)&anira_context_capabilities;
    entries[9].name = "anira_context_config_create";
    entries[9].address = (uintptr_t)&anira_context_config_create;
    entries[10].name = "anira_context_config_destroy";
    entries[10].address = (uintptr_t)&anira_context_config_destroy;
    entries[11].name = "anira_context_config_from_json";
    entries[11].address = (uintptr_t)&anira_context_config_from_json;
    entries[12].name = "anira_context_config_set_cuda";
    entries[12].address = (uintptr_t)&anira_context_config_set_cuda;
    entries[13].name = "anira_context_config_set_d3d12";
    entries[13].address = (uintptr_t)&anira_context_config_set_d3d12;
    entries[14].name = "anira_context_config_set_ext";
    entries[14].address = (uintptr_t)&anira_context_config_set_ext;
    entries[15].name = "anira_context_config_set_ext_json";
    entries[15].address = (uintptr_t)&anira_context_config_set_ext_json;
    entries[16].name = "anira_context_config_set_gl";
    entries[16].address = (uintptr_t)&anira_context_config_set_gl;
    entries[17].name = "anira_context_config_set_log";
    entries[17].address = (uintptr_t)&anira_context_config_set_log;
    entries[18].name = "anira_context_config_set_log_drain";
    entries[18].address = (uintptr_t)&anira_context_config_set_log_drain;
    entries[19].name = "anira_context_config_set_log_flags";
    entries[19].address = (uintptr_t)&anira_context_config_set_log_flags;
    entries[20].name = "anira_context_config_set_log_level";
    entries[20].address = (uintptr_t)&anira_context_config_set_log_level;
    entries[21].name = "anira_context_config_set_log_queue_capacity";
    entries[21].address = (uintptr_t)&anira_context_config_set_log_queue_capacity;
    entries[22].name = "anira_context_config_set_log_sink";
    entries[22].address = (uintptr_t)&anira_context_config_set_log_sink;
    entries[23].name = "anira_context_config_set_metal";
    entries[23].address = (uintptr_t)&anira_context_config_set_metal;
    entries[24].name = "anira_context_config_set_threads";
    entries[24].address = (uintptr_t)&anira_context_config_set_threads;
    entries[25].name = "anira_context_config_set_vulkan";
    entries[25].address = (uintptr_t)&anira_context_config_set_vulkan;
    entries[26].name = "anira_context_config_set_webgpu";
    entries[26].address = (uintptr_t)&anira_context_config_set_webgpu;
    entries[27].name = "anira_context_config_to_json";
    entries[27].address = (uintptr_t)&anira_context_config_to_json;
    entries[28].name = "anira_context_create";
    entries[28].address = (uintptr_t)&anira_context_create;
    entries[29].name = "anira_context_destroy";
    entries[29].address = (uintptr_t)&anira_context_destroy;
    entries[30].name = "anira_context_probe";
    entries[30].address = (uintptr_t)&anira_context_probe;
    entries[31].name = "anira_contract_async_set_deadline";
    entries[31].address = (uintptr_t)&anira_contract_async_set_deadline;
    entries[32].name = "anira_contract_async_set_policy";
    entries[32].address = (uintptr_t)&anira_contract_async_set_policy;
    entries[33].name = "anira_contract_create_async";
    entries[33].address = (uintptr_t)&anira_contract_create_async;
    entries[34].name = "anira_contract_create_hard";
    entries[34].address = (uintptr_t)&anira_contract_create_hard;
    entries[35].name = "anira_contract_destroy";
    entries[35].address = (uintptr_t)&anira_contract_destroy;
    entries[36].name = "anira_contract_from_json";
    entries[36].address = (uintptr_t)&anira_contract_from_json;
    entries[37].name = "anira_contract_get_kind";
    entries[37].address = (uintptr_t)&anira_contract_get_kind;
    entries[38].name = "anira_contract_hard_set_budget";
    entries[38].address = (uintptr_t)&anira_contract_hard_set_budget;
    entries[39].name = "anira_contract_hard_set_geometry";
    entries[39].address = (uintptr_t)&anira_contract_hard_set_geometry;
    entries[40].name = "anira_contract_hard_set_on_miss";
    entries[40].address = (uintptr_t)&anira_contract_hard_set_on_miss;
    entries[41].name = "anira_contract_hard_set_ring_dtype";
    entries[41].address = (uintptr_t)&anira_contract_hard_set_ring_dtype;
    entries[42].name = "anira_contract_hard_set_wait_ratio";
    entries[42].address = (uintptr_t)&anira_contract_hard_set_wait_ratio;
    entries[43].name = "anira_contract_hard_set_warmup";
    entries[43].address = (uintptr_t)&anira_contract_hard_set_warmup;
    entries[44].name = "anira_contract_set_edge_cost";
    entries[44].address = (uintptr_t)&anira_contract_set_edge_cost;
    entries[45].name = "anira_contract_set_ext";
    entries[45].address = (uintptr_t)&anira_contract_set_ext;
    entries[46].name = "anira_contract_set_ext_json";
    entries[46].address = (uintptr_t)&anira_contract_set_ext_json;
    entries[47].name = "anira_drain_log";
    entries[47].address = (uintptr_t)&anira_drain_log;
    entries[48].name = "anira_enabled_backends";
    entries[48].address = (uintptr_t)&anira_enabled_backends;
    entries[49].name = "anira_has_core";
    entries[49].address = (uintptr_t)&anira_has_core;
    entries[50].name = "anira_inference_thread_create";
    entries[50].address = (uintptr_t)&anira_inference_thread_create;
    entries[51].name = "anira_inference_thread_destroy";
    entries[51].address = (uintptr_t)&anira_inference_thread_destroy;
    entries[52].name = "anira_inference_thread_execute";
    entries[52].address = (uintptr_t)&anira_inference_thread_execute;
    entries[53].name = "anira_inference_thread_has_exited";
    entries[53].address = (uintptr_t)&anira_inference_thread_has_exited;
    entries[54].name = "anira_inference_thread_is_running";
    entries[54].address = (uintptr_t)&anira_inference_thread_is_running;
    entries[55].name = "anira_inference_thread_run_loop";
    entries[55].address = (uintptr_t)&anira_inference_thread_run_loop;
    entries[56].name = "anira_inference_thread_should_exit";
    entries[56].address = (uintptr_t)&anira_inference_thread_should_exit;
    entries[57].name = "anira_inference_thread_start";
    entries[57].address = (uintptr_t)&anira_inference_thread_start;
    entries[58].name = "anira_inference_thread_stop";
    entries[58].address = (uintptr_t)&anira_inference_thread_stop;
    entries[59].name = "anira_job_options_create";
    entries[59].address = (uintptr_t)&anira_job_options_create;
    entries[60].name = "anira_job_options_destroy";
    entries[60].address = (uintptr_t)&anira_job_options_destroy;
    entries[61].name = "anira_job_options_set_below_min";
    entries[61].address = (uintptr_t)&anira_job_options_set_below_min;
    entries[62].name = "anira_job_options_set_ext";
    entries[62].address = (uintptr_t)&anira_job_options_set_ext;
    entries[63].name = "anira_job_options_set_ext_json";
    entries[63].address = (uintptr_t)&anira_job_options_set_ext_json;
    entries[64].name = "anira_job_options_set_head_trim";
    entries[64].address = (uintptr_t)&anira_job_options_set_head_trim;
    entries[65].name = "anira_job_options_set_tail_flush";
    entries[65].address = (uintptr_t)&anira_job_options_set_tail_flush;
    entries[66].name = "anira_log";
    entries[66].address = (uintptr_t)&anira_log;
    entries[67].name = "anira_log_rt";
    entries[67].address = (uintptr_t)&anira_log_rt;
    entries[68].name = "anira_model_config_add_input";
    entries[68].address = (uintptr_t)&anira_model_config_add_input;
    entries[69].name = "anira_model_config_add_model_bytes";
    entries[69].address = (uintptr_t)&anira_model_config_add_model_bytes;
    entries[70].name = "anira_model_config_add_model_bytes_custom";
    entries[70].address = (uintptr_t)&anira_model_config_add_model_bytes_custom;
    entries[71].name = "anira_model_config_add_model_path";
    entries[71].address = (uintptr_t)&anira_model_config_add_model_path;
    entries[72].name = "anira_model_config_add_model_path_custom";
    entries[72].address = (uintptr_t)&anira_model_config_add_model_path_custom;
    entries[73].name = "anira_model_config_add_output";
    entries[73].address = (uintptr_t)&anira_model_config_add_output;
    entries[74].name = "anira_model_config_create";
    entries[74].address = (uintptr_t)&anira_model_config_create;
    entries[75].name = "anira_model_config_destroy";
    entries[75].address = (uintptr_t)&anira_model_config_destroy;
    entries[76].name = "anira_model_config_from_json";
    entries[76].address = (uintptr_t)&anira_model_config_from_json;
    entries[77].name = "anira_model_config_from_json_file";
    entries[77].address = (uintptr_t)&anira_model_config_from_json_file;
    entries[78].name = "anira_model_config_model_bytes";
    entries[78].address = (uintptr_t)&anira_model_config_model_bytes;
    entries[79].name = "anira_model_config_model_count";
    entries[79].address = (uintptr_t)&anira_model_config_model_count;
    entries[80].name = "anira_model_config_model_engine";
    entries[80].address = (uintptr_t)&anira_model_config_model_engine;
    entries[81].name = "anira_model_config_model_engine_id";
    entries[81].address = (uintptr_t)&anira_model_config_model_engine_id;
    entries[82].name = "anira_model_config_model_path";
    entries[82].address = (uintptr_t)&anira_model_config_model_path;
    entries[83].name = "anira_model_config_set_anchor";
    entries[83].address = (uintptr_t)&anira_model_config_set_anchor;
    entries[84].name = "anira_model_config_set_default_engine";
    entries[84].address = (uintptr_t)&anira_model_config_set_default_engine;
    entries[85].name = "anira_model_config_set_default_engine_custom";
    entries[85].address = (uintptr_t)&anira_model_config_set_default_engine_custom;
    entries[86].name = "anira_model_config_set_ext";
    entries[86].address = (uintptr_t)&anira_model_config_set_ext;
    entries[87].name = "anira_model_config_set_ext_json";
    entries[87].address = (uintptr_t)&anira_model_config_set_ext_json;
    entries[88].name = "anira_model_config_set_max_instances";
    entries[88].address = (uintptr_t)&anira_model_config_set_max_instances;
    entries[89].name = "anira_model_config_set_model_bytes";
    entries[89].address = (uintptr_t)&anira_model_config_set_model_bytes;
    entries[90].name = "anira_model_config_set_model_ext";
    entries[90].address = (uintptr_t)&anira_model_config_set_model_ext;
    entries[91].name = "anira_model_config_set_model_ext_json";
    entries[91].address = (uintptr_t)&anira_model_config_set_model_ext_json;
    entries[92].name = "anira_model_config_set_state";
    entries[92].address = (uintptr_t)&anira_model_config_set_state;
    entries[93].name = "anira_model_config_set_tensor_layout";
    entries[93].address = (uintptr_t)&anira_model_config_set_tensor_layout;
    entries[94].name = "anira_model_config_set_tensor_name";
    entries[94].address = (uintptr_t)&anira_model_config_set_tensor_name;
    entries[95].name = "anira_model_config_take_legacy_contract";
    entries[95].address = (uintptr_t)&anira_model_config_take_legacy_contract;
    entries[96].name = "anira_model_config_to_json";
    entries[96].address = (uintptr_t)&anira_model_config_to_json;
    entries[97].name = "anira_now_ms";
    entries[97].address = (uintptr_t)&anira_now_ms;
    entries[98].name = "anira_now_ns";
    entries[98].address = (uintptr_t)&anira_now_ns;
    entries[99].name = "anira_num_inference_threads";
    entries[99].address = (uintptr_t)&anira_num_inference_threads;
    entries[100].name = "anira_registered_ext_kinds";
    entries[100].address = (uintptr_t)&anira_registered_ext_kinds;
    entries[101].name = "anira_release_core_if_idle";
    entries[101].address = (uintptr_t)&anira_release_core_if_idle;
    entries[102].name = "anira_shutdown";
    entries[102].address = (uintptr_t)&anira_shutdown;
    entries[103].name = "anira_status_string";
    entries[103].address = (uintptr_t)&anira_status_string;
    entries[104].name = "anira_tensor_spec_create";
    entries[104].address = (uintptr_t)&anira_tensor_spec_create;
    entries[105].name = "anira_tensor_spec_destroy";
    entries[105].address = (uintptr_t)&anira_tensor_spec_destroy;
    entries[106].name = "anira_tensor_spec_set_axis";
    entries[106].address = (uintptr_t)&anira_tensor_spec_set_axis;
    entries[107].name = "anira_tensor_spec_set_ext";
    entries[107].address = (uintptr_t)&anira_tensor_spec_set_ext;
    entries[108].name = "anira_tensor_spec_set_ext_json";
    entries[108].address = (uintptr_t)&anira_tensor_spec_set_ext_json;
    entries[109].name = "anira_tensor_spec_set_latency";
    entries[109].address = (uintptr_t)&anira_tensor_spec_set_latency;
    entries[110].name = "anira_tensor_spec_set_time_ratio";
    entries[110].address = (uintptr_t)&anira_tensor_spec_set_time_ratio;
    entries[111].name = "anira_tensor_spec_set_window";
    entries[111].address = (uintptr_t)&anira_tensor_spec_set_window;
    entries[112].name = "anira_version";
    entries[112].address = (uintptr_t)&anira_version;
    entries[113].name = "anira_version_string";
    entries[113].address = (uintptr_t)&anira_version_string;
    for (i = 0; i < ANIRA_LINK_PROBE_COUNT; ++i) {
        if (entries[i].address == 0) {
            printf("missing: %s\n", entries[i].name);
            ++missing;
        }
    }
    printf("%zu of %zu entry points linked\n", (size_t)ANIRA_LINK_PROBE_COUNT - missing,
           (size_t)ANIRA_LINK_PROBE_COUNT);
    return missing == 0 ? 0 : 1;
}
