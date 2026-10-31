// Generated from abi/anira.yml by tools/abi/gen.py; edit the registry, not this file.
// One typed method per entry point of the module: the C name, the C argument order without
// the out parameters and the anira_error, which the binding makes on the instance's stack. A
// string argument is copied to that stack, a written value is returned (several as an object),
// a failed anira_status throws an AniraError, an anira_bool is a boolean, a 64-bit integer a
// bigint and a pointer or a size a number. An enumeration entry (a uint32_t* count) keeps its
// pointers as arguments.

import { AniraError } from '../AniraError'
import { anira_status } from './enums'

/** The raw exports and runtime methods of one module instance that the bindings call. */
export interface AbiModule {
  _anira_status_string(_0: number): number
  _anira_abi_version(): number
  _anira_check_abi(_0: number): number
  _anira_version(): number
  _anira_version_string(): number
  _anira_drain_log(): number
  _anira_log_rt(_0: number, _1: number, _2: number, _3: number, _4: number): void
  _anira_log(_0: number, _1: number, _2: number): void
  _anira_tensor_init_host(_0: number, _1: number, _2: number, _3: number, _4: number): void
  _anira_tensor_init_pinned(_0: number, _1: number, _2: number, _3: number, _4: number): void
  _anira_tensor_init_host_planar(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): void
  _anira_tensor_init_cuda(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number): void
  _anira_tensor_init_gl_buffer(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number): void
  _anira_tensor_init_vulkan(_0: number, _1: bigint, _2: bigint, _3: bigint, _4: bigint, _5: bigint, _6: number, _7: number, _8: number): void
  _anira_tensor_init_opaque_fd(_0: number, _1: number, _2: bigint, _3: number, _4: number, _5: number): void
  _anira_tensor_init_wgpu_buffer(_0: number, _1: number, _2: bigint, _3: number, _4: number, _5: number, _6: number): void
  _anira_tensor_init_dmabuf(_0: number, _1: number, _2: bigint, _3: bigint, _4: number, _5: number, _6: number, _7: number): void
  _anira_tensor_init_dlpack(_0: number, _1: number, _2: number): number
  _anira_tensor_data_f32(_0: number): number
  _anira_tensor_data(_0: number, _1: number): number
  _anira_tensor_plane(_0: number, _1: number, _2: number): number
  _anira_tensor_num_elements(_0: number): number
  _anira_tensor_extent(_0: number, _1: number): number
  _anira_sizeof(_0: number): number
  _anira_sync_token_reset(_0: number): void
  _anira_sync_token_dup(_0: number, _1: number): number
  _anira_registered_ext_kinds(_0: number, _1: number): number
  _anira_tensor_spec_create(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_tensor_spec_set_axis(_0: number, _1: number, _2: number, _3: bigint): number
  _anira_tensor_spec_set_window(_0: number, _1: bigint, _2: bigint, _3: bigint): number
  _anira_tensor_spec_set_time_ratio(_0: number, _1: bigint, _2: bigint): number
  _anira_tensor_spec_set_latency(_0: number, _1: bigint): number
  _anira_tensor_spec_set_state_source(_0: number, _1: number): number
  _anira_tensor_spec_set_ext(_0: number, _1: number, _2: number): number
  _anira_tensor_spec_set_ext_json(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_tensor_spec_destroy(_0: number): void
  _anira_contract_create_hard(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_contract_create_async(_0: number, _1: number): number
  _anira_contract_hard_set_geometry(_0: number, _1: number, _2: number, _3: number): number
  _anira_contract_hard_set_budget(_0: number, _1: number, _2: number): number
  _anira_contract_hard_set_warmup(_0: number, _1: number, _2: number): number
  _anira_contract_hard_set_on_miss(_0: number, _1: number): number
  _anira_contract_hard_set_miss_fn(_0: number, _1: number, _2: number): number
  _anira_contract_hard_set_wait_ratio(_0: number, _1: number): number
  _anira_contract_hard_set_ring_dtype(_0: number, _1: number, _2: number): number
  _anira_contract_hard_set_latency(_0: number, _1: number, _2: number): number
  _anira_contract_async_set_deadline(_0: number, _1: number): number
  _anira_contract_async_set_policy(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_contract_set_edge_cost(_0: number, _1: number): number
  _anira_contract_set_host_domain(_0: number, _1: number, _2: number): number
  _anira_contract_set_ext(_0: number, _1: number, _2: number): number
  _anira_contract_set_ext_json(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_contract_get_kind(_0: number): number
  _anira_contract_destroy(_0: number): void
  _anira_context_config_create(_0: number, _1: number): number
  _anira_context_config_set_threads(_0: number, _1: number, _2: number): number
  _anira_context_config_set_log_level(_0: number, _1: number): number
  _anira_context_config_set_log_drain(_0: number, _1: number, _2: number): number
  _anira_context_config_set_log_queue_capacity(_0: number, _1: number): number
  _anira_context_config_set_log_flags(_0: number, _1: number): number
  _anira_context_config_set_log_sink(_0: number, _1: number, _2: number): number
  _anira_context_config_set_log(_0: number, _1: number): number
  _anira_context_config_set_cuda(_0: number, _1: number): number
  _anira_context_config_set_gl(_0: number, _1: number): number
  _anira_context_config_set_vulkan(_0: number, _1: number): number
  _anira_context_config_set_metal(_0: number, _1: number): number
  _anira_context_config_set_d3d12(_0: number, _1: number): number
  _anira_context_config_set_webgpu(_0: number, _1: number): number
  _anira_context_config_set_ext(_0: number, _1: number, _2: number): number
  _anira_context_config_set_ext_json(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_context_config_destroy(_0: number): void
  _anira_model_config_create(_0: number, _1: number): number
  _anira_model_config_add_model_path(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_model_config_add_model_bytes(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number, _7: number, _8: number, _9: number): number
  _anira_model_config_set_model_bytes(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number, _7: number): number
  _anira_model_config_model_count(_0: number): number
  _anira_model_config_model_engine(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_set_model_provider(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_model_config_model_provider(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_model_path(_0: number, _1: number): number
  _anira_model_config_model_bytes(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_set_tensor_name(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_set_tensor_layout(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_model_config_set_model_ext(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_set_model_ext_json(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_model_config_add_input(_0: number, _1: number): number
  _anira_model_config_add_output(_0: number, _1: number): number
  _anira_model_config_set_default_engine(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_set_default_provider(_0: number, _1: number, _2: number): number
  _anira_model_config_set_state(_0: number, _1: number): number
  _anira_model_config_set_max_instances(_0: number, _1: number): number
  _anira_model_config_set_anchor(_0: number, _1: number): number
  _anira_model_config_set_ext(_0: number, _1: number, _2: number): number
  _anira_model_config_set_ext_json(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_model_config_destroy(_0: number): void
  _anira_job_options_create(_0: number, _1: number): number
  _anira_job_options_set_head_trim(_0: number, _1: number, _2: number): number
  _anira_job_options_set_tail_flush(_0: number, _1: number): number
  _anira_job_options_set_below_min(_0: number, _1: number): number
  _anira_job_options_set_ext(_0: number, _1: number): number
  _anira_job_options_set_ext_json(_0: number, _1: number, _2: number, _3: number): number
  _anira_job_options_destroy(_0: number): void
  _anira_model_config_from_json(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_model_config_from_json_file(_0: number, _1: number, _2: number): number
  _anira_model_config_to_json(_0: number, _1: number, _2: number, _3: number): number
  _anira_model_config_take_legacy_contract(_0: number, _1: number): number
  _anira_context_config_from_json(_0: number, _1: number, _2: number, _3: number): number
  _anira_context_config_to_json(_0: number, _1: number, _2: number, _3: number): number
  _anira_contract_from_json(_0: number, _1: number, _2: number, _3: number): number
  _anira_tensor_spec_name(_0: number): number
  _anira_tensor_spec_dtype(_0: number): number
  _anira_tensor_spec_role(_0: number): number
  _anira_tensor_spec_ndim(_0: number): number
  _anira_tensor_spec_axis(_0: number, _1: number, _2: number, _3: number): number
  _anira_tensor_spec_window(_0: number, _1: number, _2: number, _3: number): number
  _anira_tensor_spec_time_ratio(_0: number, _1: number, _2: number): number
  _anira_tensor_spec_latency(_0: number): bigint
  _anira_tensor_spec_state_source(_0: number): number
  _anira_model_config_num_inputs(_0: number): number
  _anira_model_config_num_outputs(_0: number): number
  _anira_model_config_input(_0: number, _1: number): number
  _anira_model_config_output(_0: number, _1: number): number
  _anira_model_config_default_engine(_0: number, _1: number, _2: number): number
  _anira_model_config_default_provider(_0: number, _1: number, _2: number): number
  _anira_model_config_state(_0: number): number
  _anira_model_config_max_instances(_0: number): number
  _anira_model_config_anchor(_0: number): number
  _anira_model_config_tensor_name(_0: number, _1: number, _2: number): number
  _anira_model_config_tensor_layout(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_model_config_model_ext(_0: number, _1: number, _2: number): number
  _anira_contract_hard_geometry(_0: number, _1: number, _2: number, _3: number): number
  _anira_contract_hard_budget(_0: number, _1: number, _2: number): number
  _anira_contract_hard_warmup(_0: number, _1: number, _2: number): number
  _anira_contract_hard_on_miss(_0: number, _1: number): number
  _anira_contract_hard_miss_fn(_0: number, _1: number, _2: number): number
  _anira_contract_hard_wait_ratio(_0: number, _1: number): number
  _anira_contract_hard_num_ring_dtypes(_0: number): number
  _anira_contract_hard_ring_dtype(_0: number, _1: number, _2: number, _3: number): number
  _anira_contract_hard_num_latencies(_0: number): number
  _anira_contract_hard_latency(_0: number, _1: number, _2: number, _3: number): number
  _anira_contract_edge_cost(_0: number): number
  _anira_context_config_threads(_0: number, _1: number, _2: number): number
  _anira_context_config_log(_0: number, _1: number): number
  _anira_context_create(_0: number, _1: number, _2: number): number
  _anira_context_destroy(_0: number): void
  _anira_context_probe(_0: number, _1: number, _2: number): number
  _anira_context_capabilities(_0: number): number
  _anira_capabilities_backends(_0: number, _1: number, _2: number, _3: number): number
  _anira_capabilities_domains(_0: number, _1: number, _2: number): number
  _anira_capabilities_ext_kinds(_0: number, _1: number, _2: number): number
  _anira_capabilities_edges(_0: number, _1: number, _2: number, _3: number): number
  _anira_capabilities_edge(_0: number, _1: number, _2: number, _3: number): number
  _anira_enabled_engines(_0: number, _1: number, _2: number): number
  _anira_context_byte_image_bytes(_0: number, _1: bigint, _2: number): bigint
  _anira_now_ms(): number
  _anira_now_ns(): bigint
  _anira_shutdown(): number
  _anira_release_core_if_idle(): number
  _anira_has_core(): number
  _anira_inference_thread_create(_0: number, _1: number, _2: number): number
  _anira_inference_thread_run_loop(_0: number): void
  _anira_inference_thread_execute(_0: number): number
  _anira_inference_thread_start(_0: number, _1: number): number
  _anira_inference_thread_stop(_0: number): void
  _anira_inference_thread_has_exited(_0: number): number
  _anira_inference_thread_should_exit(_0: number): number
  _anira_inference_thread_is_running(_0: number): number
  _anira_inference_thread_destroy(_0: number): void
  _anira_num_inference_threads(): number
  _anira_ring_dtype(_0: number): number
  _anira_ring_num_channels(_0: number): number
  _anira_ring_available(_0: number, _1: number): number
  _anira_ring_available_past(_0: number, _1: number): number
  _anira_ring_pop_block(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_ring_peek_past_block(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_ring_push_block(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_ring_push_fill(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_ring_discard(_0: number, _1: number, _2: number): number
  _anira_ring_pop_windows(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number, _7: number): number
  _anira_processor_input_role(_0: number, _1: number, _2: number): number
  _anira_processor_output_role(_0: number, _1: number, _2: number): number
  _anira_processor_input_ring(_0: number, _1: number, _2: number): number
  _anira_processor_output_ring(_0: number, _1: number, _2: number): number
  _anira_processor_input_tensor(_0: number, _1: number, _2: number): number
  _anira_processor_output_tensor(_0: number, _1: number, _2: number): number
  _anira_processor_engine(_0: number, _1: number, _2: number): number
  _anira_processor_provider(_0: number, _1: number, _2: number): number
  _anira_processor_default_pre_process(_0: number): number
  _anira_processor_default_post_process(_0: number): number
  _anira_pipeline_create(_0: number, _1: number): number
  _anira_pipeline_add_inference(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_pipeline_add_processor(_0: number, _1: number, _2: number): number
  _anira_custom_engine_create(_0: number, _1: number, _2: number, _3: number): number
  _anira_custom_engine_destroy(_0: number): void
  _anira_custom_engine_detach(_0: number): void
  _anira_pipeline_add_engine(_0: number, _1: number, _2: number): number
  _anira_pipeline_capabilities_backends(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_pipeline_capabilities_edge(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_pipeline_destroy(_0: number): void
  _anira_handler_create(_0: number, _1: number, _2: number, _3: number): number
  _anira_handler_destroy(_0: number): void
  _anira_handler_prepare(_0: number, _1: number, _2: number): number
  _anira_handler_plan_report(_0: number): number
  _anira_handler_num_entries(_0: number): number
  _anira_plan_report_num_plans(_0: number): number
  _anira_plan_report_plans(_0: number, _1: number, _2: number, _3: number): number
  _anira_plan_report_slots(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_plan_report_exts(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_plan_report_find(_0: number, _1: number, _2: number): number
  _anira_handler_set_plan(_0: number, _1: number): number
  _anira_handler_get_plan(_0: number): number
  _anira_handler_process(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_handler_process_multi(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_handler_push_data(_0: number, _1: number, _2: number): number
  _anira_handler_push_data_multi(_0: number, _1: number, _2: number): number
  _anira_handler_pop_data(_0: number, _1: number, _2: number, _3: number): number
  _anira_handler_pop_data_multi(_0: number, _1: number, _2: number, _3: number): number
  _anira_handler_set_static_input(_0: number, _1: number, _2: number): number
  _anira_handler_get_static_output(_0: number, _1: number, _2: number): number
  _anira_handler_get_latency(_0: number, _1: number): number
  _anira_handler_get_latencies(_0: number, _1: number, _2: number): number
  _anira_handler_get_available_samples(_0: number, _1: number, _2: number, _3: number): number
  _anira_handler_reset(_0: number): void
  _anira_handler_rt_error(_0: number): number
  _anira_handler_process_wait(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number): number
  _anira_handler_process_multi_wait(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number): number
  _anira_handler_pop_data_wait(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_handler_pop_data_multi_wait(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_tensor_init_metal(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): void
  _anira_tensor_init_iosurface(_0: number, _1: number, _2: bigint, _3: number, _4: number, _5: number, _6: number): void
  _anira_tensor_init_ahardwarebuffer(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): void
  _anira_tensor_init_d3d12(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number): void
  _anira_em_processor_desc_init(_0: number, _1: number, _2: number, _3: number): number
  _anira_em_pipeline_add_js_processor(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_em_engine_desc_init(_0: number, _1: number, _2: number, _3: number): number
  _anira_em_pipeline_add_js_engine(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number): number
  _anira_em_set_log_hook(_0: number): void
  _anira_em_context_config_set_worker_module(_0: number, _1: number, _2: number): number
  _anira_em_inference_threads_in_loop(): number
  _anira_em_audio_node_create(_0: number, _1: number, _2: number, _3: number, _4: number, _5: number, _6: number, _7: number, _8: number, _9: number, _10: number, _11: number, _12: number, _13: number, _14: number): number
  _anira_em_audio_node_set_static_input(_0: number, _1: number, _2: number, _3: number, _4: number): number
  _anira_em_audio_node_prepare(_0: number, _1: number, _2: number): number
  _anira_em_audio_node_hold(_0: number, _1: number, _2: number): number
  _anira_em_audio_node_destroy(_0: number): void
  _malloc(bytes: number): number
  _free(ptr: number): void
  stackSave(): number
  stackRestore(ptr: number): void
  stackAlloc(bytes: number): number
  stringToUTF8OnStack(text: string): number
  lengthBytesUTF8(text: string): number
  UTF8ToString(ptr: number, maxBytes?: number): string
  HEAPU8: Uint8Array
  HEAP32: Int32Array
  HEAPU32: Uint32Array
  HEAPF32: Float32Array
  HEAPF64: Float64Array
  HEAP64: BigInt64Array
  HEAPU64: BigUint64Array
  emscriptenRegisterAudioObject(object: unknown): number
  emscriptenGetAudioObject(handle: number): unknown
  /** The instance's shared memory: passed on the main thread, set by anira's boot elsewhere. */
  wasmMemory?: WebAssembly.Memory
}

/**
 * `m`, its heap views over the memory as it is now. The memory is shared and grows (any
 * thread's malloc may grow it): a growth gives `memory.buffer` a new, longer SharedArrayBuffer,
 * and views made before it cover only the old length. Every heap access of the package goes
 * through this helper, which rebuilds the views once after a growth (seven typed arrays: a
 * small allocation, made in JavaScript, never by the C audio path).
 */
export const heap = (m: AbiModule): AbiModule => {
  const memory = m.wasmMemory
  if (memory !== undefined && m.HEAPU8.buffer !== memory.buffer) {
    const b = memory.buffer
    m.HEAPU8 = new Uint8Array(b)
    m.HEAP32 = new Int32Array(b)
    m.HEAPU32 = new Uint32Array(b)
    m.HEAPF32 = new Float32Array(b)
    m.HEAPF64 = new Float64Array(b)
    m.HEAP64 = new BigInt64Array(b)
    m.HEAPU64 = new BigUint64Array(b)
  }
  return m
}

// An 8-byte out slot on the stack, zeroed.
const stackSlot = (m: AbiModule): number => {
  const ptr = m.stackAlloc(8)
  heap(m).HEAPU32[ptr >>> 2] = 0
  heap(m).HEAPU32[(ptr >>> 2) + 1] = 0
  return ptr
}

// An anira_error on the stack as ANIRA_ERROR_INIT leaves it: status 0, an empty message.
const stackError = (m: AbiModule): number => {
  const ptr = m.stackAlloc(520)
  heap(m).HEAPU32.fill(0, ptr >>> 2, (ptr >>> 2) + 3)
  return ptr
}

const readCString = (m: AbiModule, ptr: number): string | null =>
  ptr === 0 ? null : m.UTF8ToString(ptr >>> 0)

// A failed status throws: the entry's message when it wrote one, the status text otherwise.
const checkStatus = (m: AbiModule, status: number, entry: string, err = 0): number => {
  if (status >= 0) return status
  const message = err === 0 ? '' : m.UTF8ToString(err + 8, 512)
  const text = m.UTF8ToString(m._anira_status_string(status) >>> 0)
  throw new AniraError(status, `${entry}: ${message || text}`)
}

/**
 * The C ABI over one module instance: `new AniraBindings(module)` where `module` is the
 * instance's Module (the main thread's, a Wasm Worker's or the audio worklet's).
 */
export class AniraBindings {
  readonly module: AbiModule

  constructor(module: AbiModule) {
    this.module = module
  }

  /**
   * Static text for a status code, for diagnostics; never NULL. An unknown value yields
   * "unknown status".
   *
   * @param status Any status, known or not.
   * @returns A NUL-terminated string in static storage.
   * Thread: [thread-safe]
   */
  anira_status_string(status: number): string | null {
    const m = this.module
    return readCString(m, m._anira_status_string(status))
  }

  /**
   * The ANIRA_ABI_VERSION the library was built with.
   *
   * @returns The packed pair; unpack with ANIRA_ABI_VERSION_MAJOR /
   *   ANIRA_ABI_VERSION_MINOR.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_abi_version(): number {
    const m = this.module
    return m._anira_abi_version() >>> 0
  }

  /**
   * Negotiates the header the caller compiled against with the library it loaded:
   * ANIRA_OK iff the majors are equal and the library's minor is at least the header's;
   * while the major is 0 (before v3.0.0) the pair must match exactly.
   *
   * @param header_abi_version The caller's ANIRA_ABI_VERSION.
   * @returns ANIRA_OK, or ANIRA_ERROR_ABI_VERSION. A failed status throws an AniraError.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_check_abi(header_abi_version: number): number {
    const m = this.module
    return checkStatus(m, m._anira_check_abi(header_abi_version), 'anira_check_abi')
  }

  /**
   * The library's release version, packed as ANIRA_MAKE_VERSION(major, minor, patch).
   *
   * @returns The packed semver triple of the library.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_version(): number {
    const m = this.module
    return m._anira_version() >>> 0
  }

  /**
   * The library's full version string (git describe without the leading v), e.g.
   * "3.0.0-alpha.1-12-gabc123"; what ANIRA_VERSION_STRING was for the build that produced
   * the library.
   *
   * @returns A NUL-terminated string in static storage.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_version_string(): string | null {
    const m = this.module
    return readCString(m, m._anira_version_string())
  }

  /**
   * Delivers the queued real-time records of this copy's core to the sinks; the host's
   * pump under ANIRA_LOG_DRAIN_MANUAL (and on Wasm, where no drain thread exists; there
   * the records reach the C sinks and the JavaScript hook of anira_em_set_log_hook alike,
   * on the instance that drains). Returns 0 while no core exists. Real-time records are
   * also delivered on the thread of a failing main-thread entry: before a negative status
   * is returned the firewall drains the queue there, so the records that preceded the
   * failure are in front of the host before it acts.
   *
   * @returns The number of records delivered.
   * Thread: [thread-safe, !audio-thread]
   */
  anira_drain_log(): number {
    const m = this.module
    return m._anira_drain_log() >>> 0
  }

  /**
   * Real-time logging for callers on an ANIRA_NONBLOCKING path: writes "<message> [arg0
   * arg1]" into the real-time queue without formatting or allocating; a full queue drops
   * and counts. A no-op while no core exists.
   *
   * @param level Severity.
   * @param group "anira.<component>" or the host's own group; static storage.
   * @param static_message A static string, never a format string.
   * @param arg0 First argument, appended as "[arg0 arg1]".
   * @param arg1 Second argument.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_log_rt(
    level: number,
    group: string,
    static_message: string,
    arg0: number,
    arg1: number
  ): void {
    const m = this.module
    const sp = m.stackSave()
    try {
      const group_ = group === null ? 0 : m.stringToUTF8OnStack(group)
      const static_message_ = static_message === null ? 0 : m.stringToUTF8OnStack(static_message)
      m._anira_log_rt(level, group_, static_message_, arg0, arg1)
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Control-path logging through anira's private logger; formats and allocates like any
   * other control-path call and reaches every sink synchronously.
   *
   * @param level Severity.
   * @param group "anira.<component>" or the host's own group.
   * @param message The message; copied.
   * Thread: [thread-safe, !audio-thread]
   */
  anira_log(level: number, group: string, message: string): void {
    const m = this.module
    const sp = m.stackSave()
    try {
      const group_ = group === null ? 0 : m.stringToUTF8OnStack(group)
      const message_ = message === null ? 0 : m.stringToUTF8OnStack(message)
      m._anira_log(level, group_, message_)
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills a tensor over host memory. Like every anira_tensor_init_* factory it zeroes the
   * record, then fills it: domain ANIRA_DOMAIN_HOST, dtype, ndim, the first ndim extents,
   * handle.host.ptr. Everything else stays zero: strides (packed row-major), byte_offset,
   * flags, manager_ctx and release (NULL = borrowed), acquire (ANIRA_SYNC_NONE). The
   * factories return nothing: a dtype of 0, ndim above ANIRA_MAX_RANK, a NULL shape with
   * ndim above 0, or a negative extent leave the record all-zero, which reads as dtype 0
   * (never a filled record), makes the data accessors return NULL and
   * anira_tensor_num_elements return 0. Every argument is read before the record is
   * zeroed, so shape (and a token handed by pointer) may point into the record being
   * filled: anira_tensor_init_host(&t, p, t.dtype, t.ndim, t.shape) re-points a tensor.
   * The previous content is overwritten, never released: end an owning acquire token with
   * anira_sync_token_reset and call a non-NULL release before re-initialising a record. A
   * field fill: legal from a render thread and from inside a processor callback.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param data The first byte of pageable host memory; borrowed.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_host(
    tensor: number,
    data: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_host(tensor, data, dtype, ndim, shape)
  }

  /**
   * anira_tensor_init_host for page-locked memory: domain ANIRA_DOMAIN_HOST_PINNED, the
   * same host arm, the same refusals.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param data The first byte of page-locked host memory; borrowed.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_pinned(
    tensor: number,
    data: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_pinned(tensor, data, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over planar host memory, one pointer per plane of axis 0 (the channel
   * pointers of an audio host): domain ANIRA_DOMAIN_HOST, flags ANIRA_TENSOR_PLANAR,
   * handle.planes. The pointer array travels as const void* so that a float** and a const
   * float* const* both convert without a cast, in C and in C++; it is read as an array of
   * void*, which every supported target represents like any other object pointer. There
   * is no flags parameter: a caller that built the tensor over read-only planes ORs
   * ANIRA_TENSOR_READ_ONLY into flags afterwards, and one over page-locked planes assigns
   * domain ANIRA_DOMAIN_HOST_PINNED. Zero-then-fill and the refusals of
   * anira_tensor_init_host, and two more: a rank of 0 (there is no plane axis) and a
   * count other than shape[0] leave the record all-zero. anira_tensor_data and
   * anira_tensor_data_f32 return NULL for a planar tensor; anira_tensor_plane reads it.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param planes An array of count object pointers of one pointee type (a float**, a
   *   const int16_t* const*), read as void* objects; borrowed like the memory it names.
   *   NULL is accepted: an empty tensor.
   * @param count The number of planes; must equal shape[0].
   * @param dtype The element type.
   * @param ndim The rank, 1 to ANIRA_MAX_RANK: axis 0 is the plane axis.
   * @param shape ndim extents, each 0 or more.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_host_planar(
    tensor: number,
    planes: number,
    count: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_host_planar(tensor, planes, count, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over CUDA device memory: domain ANIRA_DOMAIN_CUDA, handle.cuda. A
   * non-NULL cuda_event becomes acquire (ANIRA_SYNC_CUDA_EVENT, non-owning); NULL leaves
   * ANIRA_SYNC_NONE. Zero-then-fill and the refusals of anira_tensor_init_host. In this
   * pre-release a field fill only: no adapter consumes the arm.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param ptr The CUDA device pointer.
   * @param device CUDA device ordinal.
   * @param cuda_event cudaEvent_t the data is valid after, or NULL: already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_cuda(
    tensor: number,
    ptr: number,
    device: number,
    cuda_event: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_cuda(tensor, ptr, device, cuda_event, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over an OpenGL buffer object: domain ANIRA_DOMAIN_GL_BUFFER,
   * handle.gl. A non-NULL gl_sync becomes acquire (ANIRA_SYNC_GL_SYNC, non-owning); NULL
   * leaves ANIRA_SYNC_NONE. Consuming it requires the GL thread policy of the context
   * config (anira_gl_desc.threads). Zero-then-fill and the refusals of
   * anira_tensor_init_host. In this pre-release a field fill only: no adapter consumes
   * the arm.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param id GLuint buffer name.
   * @param target GLenum binding target.
   * @param gl_sync GLsync the data is valid after, or NULL: already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_gl_buffer(
    tensor: number,
    id: number,
    target: number,
    gl_sync: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_gl_buffer(tensor, id, target, gl_sync, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over a native Vulkan buffer of the same process: domain
   * ANIRA_DOMAIN_VULKAN_BUFFER, handle.vk. A non-zero timeline_semaphore becomes acquire
   * (ANIRA_SYNC_VK_TIMELINE with value, non-owning); 0 leaves ANIRA_SYNC_NONE. The
   * handles travel at their wire width, which makes this one of the allowlisted 64-bit
   * ANIRA_NONBLOCKING declarations: nothing that produces a VkBuffer runs in JS.
   * Zero-then-fill and the refusals of anira_tensor_init_host. In this pre-release a
   * field fill only: no adapter consumes the arm.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param buffer VkBuffer.
   * @param memory VkDeviceMemory.
   * @param offset Byte offset of the buffer in the memory.
   * @param timeline_semaphore VkSemaphore (timeline) the data is valid after, or 0
   *   (VK_NULL_HANDLE): already visible.
   * @param value The timeline value that signals; ignored without a semaphore.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_vulkan(
    tensor: number,
    buffer: bigint,
    memory: bigint,
    offset: bigint,
    timeline_semaphore: bigint,
    value: bigint,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_vulkan(
      tensor,
      buffer,
      memory,
      offset,
      timeline_semaphore,
      value,
      dtype,
      ndim,
      shape
    )
  }

  /**
   * Fills a tensor over exported opaque memory: domain ANIRA_DOMAIN_OPAQUE_FD,
   * handle.opaque. It takes no fence: acquire stays ANIRA_SYNC_NONE, and a producer with
   * one fills acquire after the call. Allowlisted 64-bit ANIRA_NONBLOCKING declaration (a
   * byte size at its wire width). Zero-then-fill and the refusals of
   * anira_tensor_init_host. In this pre-release a field fill only: no adapter consumes
   * the arm.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param fd The exported opaque fd; on Windows the NT handle. Not owned by the tensor.
   * @param size Size of the exported allocation in bytes.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_opaque_fd(
    tensor: number,
    fd: number,
    size: bigint,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_opaque_fd(tensor, fd, size, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over a WebGPU buffer: domain ANIRA_DOMAIN_WGPU_BUFFER, handle.wgpu.
   * The token is copied into acquire and that copy is the hand-off: an owned fd moves
   * with it and the caller does not reset its source. anira never fabricates a fence:
   * NULL is ANIRA_SYNC_NONE. When the call is refused the record is all-zero, the token
   * was not copied and stays the caller's. Allowlisted 64-bit ANIRA_NONBLOCKING
   * declaration. Zero-then-fill and the refusals of anira_tensor_init_host. In this
   * pre-release a field fill only: no adapter consumes the arm.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param wgpu_buffer WGPUBuffer of the core's Dawn device.
   * @param offset Byte offset into the buffer.
   * @param fence Copied into acquire, or NULL: already visible (ANIRA_SYNC_NONE). Pass
   *   kind ANIRA_SYNC_QUEUE_ORDERED for work still on the producing queue.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_wgpu_buffer(
    tensor: number,
    wgpu_buffer: number,
    offset: bigint,
    fence: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_wgpu_buffer(
      tensor,
      wgpu_buffer,
      offset,
      fence,
      dtype,
      ndim,
      shape
    )
  }

  /**
   * Fills a tensor over exported buffer memory (Vulkan external memory, a GBM linear bo,
   * a dma-heap): domain ANIRA_DOMAIN_DMABUF, handle.dmabuf, typed by the descriptor like
   * every other arm. A sync_fd of 0 or more becomes acquire (ANIRA_SYNC_SYNC_FILE_FD):
   * the token owns it, so the producer must not close it and the holder of the record
   * ends it with anira_sync_token_reset; a negative sync_fd leaves ANIRA_SYNC_NONE. When
   * the call is refused the record is all-zero and sync_fd stays the caller's.
   * Allowlisted 64-bit ANIRA_NONBLOCKING declaration. Zero-then-fill and the refusals of
   * anira_tensor_init_host. In this pre-release a field fill only: no adapter consumes
   * the arm.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param fd The dma-buf fd of exported buffer memory. Not owned by the tensor.
   * @param size Size of the exported memory in bytes.
   * @param offset Byte offset into the exported memory.
   * @param sync_fd A sync_file fd the data is valid after, owned by the tensor's acquire
   *   token from here on; negative: already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_dmabuf(
    tensor: number,
    fd: number,
    size: bigint,
    offset: bigint,
    sync_fd: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_dmabuf(tensor, fd, size, offset, sync_fd, dtype, ndim, shape)
  }

  /**
   * Fills a tensor from a DLPack 1.x managed tensor. kDLCPU maps to ANIRA_DOMAIN_HOST and
   * kDLCUDAHost to ANIRA_DOMAIN_HOST_PINNED; the DLDataType is the anira_dtype (codes 0
   * to 6, any bits and lanes); shape is copied, strides are copied or, when NULL (a
   * producer older than DLPack 1.2), left all-zero = packed row-major; strides that are
   * present and all zero over more than one element (a fully broadcast view) are refused,
   * because all-zero is this record's spelling of packed; data and byte_offset are
   * copied; DLPACK_FLAG_BITMASK_READ_ONLY sets ANIRA_TENSOR_READ_ONLY; acquire is
   * ANIRA_SYNC_NONE; the result is never planar. Ownership: on success the managed tensor
   * belongs to the record: manager_ctx holds it and release is a trampoline that disarms
   * the record (release and manager_ctx NULL) and then calls the producer's deleter
   * exactly once; a NULL deleter gives a borrowed tensor (release NULL). The trampoline
   * guards one descriptor, not its bitwise copies: call release on one copy only, and
   * expect it to be as blocking as the producer's deleter. On failure nothing is
   * consumed: anira never calls the deleter of a tensor it refused, the caller keeps
   * ownership, and *tensor is untouched.
   *
   * @param tensor The caller's record; written on success only.
   * @param dl_managed_tensor_versioned A DLManagedTensorVersioned* of DLPack major 1,
   *   passed as void* so that this header never includes dlpack.h.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL tensor or managed tensor,
   *   a rank below 0 or above ANIRA_MAX_RANK, a NULL shape with a rank above 0, a negative
   *   extent, or a dtype of 0 bits or 0 lanes; ANIRA_ERROR_NOT_SUPPORTED for a DLPack major
   *   other than 1 (nothing past the version is read), a device other than kDLCPU and
   *   kDLCUDAHost, a dtype code above 6, or all-zero strides over more than one element. A
   *   failed status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_tensor_init_dlpack(tensor: number, dl_managed_tensor_versioned: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_tensor_init_dlpack(tensor, dl_managed_tensor_versioned, err),
        'anira_tensor_init_dlpack',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The first element of a host float tensor: handle.host.ptr plus byte_offset. NULL for
   * every domain but ANIRA_DOMAIN_HOST and ANIRA_DOMAIN_HOST_PINNED, for every dtype but
   * ANIRA_DTYPE_F32, for a planar tensor (ANIRA_TENSOR_PLANAR: anira_tensor_plane reads
   * it) and for a NULL base pointer: a processor that receives a device tensor or a uint8
   * tensor learns so from the NULL, not from a crash. It never converts and never looks
   * at strides: they are returned as declared, the caller reads by them (all-zero =
   * packed row-major).
   *
   * @param tensor The tensor; NULL returns NULL.
   * @returns The element pointer, or NULL.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_data_f32(tensor: number): number {
    const m = this.module
    return m._anira_tensor_data_f32(tensor) >>> 0
  }

  /**
   * The typed read path of a host tensor: anira_tensor_data_f32 for any element type.
   * NULL unless dtype equals the tensor's own dtype (and is not 0); never converts.
   *
   * @param tensor The tensor; NULL returns NULL.
   * @param dtype The element type the caller is about to read.
   * @returns The element pointer, or NULL.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_data(tensor: number, dtype: number): number {
    const m = this.module
    return m._anira_tensor_data(tensor, dtype) >>> 0
  }

  /**
   * The first element of one plane of a planar host tensor: handle.planes.ptrs[plane]
   * plus byte_offset. NULL for a tensor without ANIRA_TENSOR_PLANAR, for every domain but
   * ANIRA_DOMAIN_HOST and ANIRA_DOMAIN_HOST_PINNED, for a plane at or above
   * handle.planes.count, for a NULL pointer array or plane pointer, and unless dtype
   * equals the tensor's own dtype (and is not 0); never converts. It is how a callback
   * that is handed planar tensors reaches their memory. The caller reads by the strides
   * from axis 1 on (all-zero = packed).
   *
   * @param tensor The tensor; NULL returns NULL.
   * @param plane The plane, below handle.planes.count.
   * @param dtype The element type the caller is about to read.
   * @returns The element pointer, or NULL.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_plane(tensor: number, plane: number, dtype: number): number {
    const m = this.module
    return m._anira_tensor_plane(tensor, plane, dtype) >>> 0
  }

  /**
   * The number of elements: the product of the first ndim extents, whatever the domain
   * and whether planar or not. 1 at rank 0 (the empty product), 0 when any extent is 0. 0
   * as well for what is not a count: a record of dtype 0 (the all-zero record of a
   * refused factory, never a filled one), ndim above ANIRA_MAX_RANK, a negative extent,
   * or a product that does not fit size_t.
   *
   * @param tensor The tensor; NULL returns 0.
   * @returns The element count; a plain number on wasm32.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_num_elements(tensor: number): number {
    const m = this.module
    return m._anira_tensor_num_elements(tensor) >>> 0
  }

  /**
   * One extent of the shape. Runtime extents are counts, never ANIRA_DYNAMIC; the raw
   * int64_t shape stays readable through the record.
   *
   * @param tensor The tensor; NULL returns 0.
   * @param axis The axis, below ndim.
   * @returns shape[axis]; 0 for an axis at or above ndim or ANIRA_MAX_RANK, a negative
   *   extent, or one that does not fit size_t.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_extent(tensor: number, axis: number): number {
    const m = this.module
    return m._anira_tensor_extent(tensor, axis) >>> 0
  }

  /**
   * sizeof of a record of this library, for an allocator that cannot see the header
   * (TypeScript, a binding): the Tier-1 records, whose size is the same on every target,
   * and the enumerated Tier-2 records, whose size is the target's and is the element
   * stride of the array enumerators.
   *
   * @param id The record.
   * @returns The size in bytes; 0 for an id this library does not know.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_sizeof(id: number): number {
    const m = this.module
    return m._anira_sizeof(id) >>> 0
  }

  /**
   * Ends a token: closes the descriptor of an owning kind (ANIRA_SYNC_SYNC_FILE_FD,
   * ANIRA_SYNC_OPAQUE_FD_SEMAPHORE with an fd of 0 or more; on Windows an NT handle above
   * 0, closed with CloseHandle) and zeroes the record: kind becomes ANIRA_SYNC_NONE,
   * under which the payload is never read, so a second reset closes nothing. Test kind,
   * never the payload: a token moved out of by a later pre-release carries fd -1 instead.
   * Every other kind is a non-owning handle and is only zeroed. Closing is a system call,
   * hence never on the driver thread. On WebAssembly no sync kind owns an
   * operating-system object and nothing is closed.
   *
   * @param token The token; NULL is a no-op.
   * Thread: [thread-safe, !audio-thread]
   */
  anira_sync_token_reset(token: number): void {
    const m = this.module
    m._anira_sync_token_reset(token)
  }

  /**
   * Duplicates a token so that it outlives its source: for the two owning fd kinds with
   * an fd of 0 or more a new descriptor of the same open file, close-on-exec (on Windows
   * DuplicateHandle for an NT handle above 0); the duplicate owns it and ends with
   * anira_sync_token_reset. A plain copy for every other kind. Duplicating is a system
   * call, hence never on the driver thread.
   *
   * @param token The token to duplicate, typically a non-owning view read from a ticket.
   * @param out Receives the duplicate on success; whatever it held is overwritten, not
   *   reset. Must not alias token.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or aliasing argument or a
   *   descriptor that is not open; ANIRA_ERROR_OUT_OF_MEMORY when the process is out of
   *   descriptors; ANIRA_ERROR_NOT_SUPPORTED for an owning kind with an fd of 0 or more on
   *   WebAssembly, where nothing can own one. *out is untouched on failure. A failed status
   *   throws an AniraError.
   * Thread: [thread-safe, !audio-thread]
   */
  anira_sync_token_dup(token: number, out: number): number {
    const m = this.module
    return checkStatus(m, m._anira_sync_token_dup(token, out), 'anira_sync_token_dup')
  }

  /**
   * Scalar enumeration of the extension kinds this build understands, without a context:
   * NULL out returns the count, a short buffer is filled as far as it goes and returns
   * ANIRA_INCOMPLETE.
   *
   * @param count In: the capacity of out; out: the number of registered kinds.
   * @param out Receives the kind names (static storage), or NULL to ask for the count
   *   only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL
   *   count. A failed status throws an AniraError.
   * Thread: [thread-safe]
   */
  anira_registered_ext_kinds(count: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_registered_ext_kinds(count, out),
      'anira_registered_ext_kinds'
    )
  }

  /**
   * Creates a tensor spec: no axes yet, window 0/0/0, time ratio (0, 0) = derive, latency
   * 0.
   *
   * @param name The canonical name: your name for this tensor, UTF-8, copied. Every other
   *   part of the configuration refers to the tensor by it (the per-entry name and layout
   *   records, the anchor, error messages); it is never handed to an engine. Unique across
   *   the inputs and outputs of one model config.
   * @param dtype The model's true dtype (section 1).
   * @param role STREAMED, BUFFER, STATIC or STATE (both halves of a declared state pair
   *   carry STATE).
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_tensor_spec_create(name: string, dtype: number, role: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const name_ = name === null ? 0 : m.stringToUTF8OnStack(name)
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_tensor_spec_create(name_, dtype, role, out_, err),
        'anira_tensor_spec_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Sets one axis; axis index order is model memory order (NCHW vs NHWC is just axis
   * order).
   *
   * @param spec The spec.
   * @param i Axis index in model memory order; i < ANIRA_MAX_RANK; ndim becomes max(i +
   *   1).
   * @param tag The axis' meaning.
   * @param extent Extent > 0, or ANIRA_DYNAMIC on the Time axis of a Streamed or Buffer
   *   spec.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for i >= ANIRA_MAX_RANK, an
   *   unknown tag, or an extent that is neither positive nor ANIRA_DYNAMIC. A failed status
   *   throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_axis(
    spec: number,
    i: number,
    tag: number,
    extent: bigint
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_tensor_spec_set_axis(spec, i, tag, extent),
      'anira_tensor_spec_set_axis'
    )
  }

  /**
   * The window of a Streamed spec, in elements along the Time axis; the fixed case is
   * window_min == window_max. Defaults 0, 0, 0. Cross-field legality (context <
   * window_min, window_max >= window_min) is checked at prepare.
   *
   * @param spec The spec.
   * @param window_min The model's smallest legal Time extent, in elements.
   * @param window_max The largest, or ANIRA_UNBOUNDED.
   * @param overlap Overlap of consecutive windows: the elements kept from the previous
   *   window. The advance per inference, the hop, is window_used - overlap.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a negative value other than
   *   ANIRA_UNBOUNDED as window_max. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_window(
    spec: number,
    window_min: bigint,
    window_max: bigint,
    overlap: bigint
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_tensor_spec_set_window(spec, window_min, window_max, overlap),
      'anira_tensor_spec_set_window'
    )
  }

  /**
   * The tensor's Time advance relative to the anchor tensor. A flexible window is pinned
   * at prepare from the anchor hop, the anchor elements one inference consumes, chosen
   * once for the whole model: the tensor's hop is the anchor hop times this ratio, so the
   * anchor hop is a multiple of den (the ratio reduced); a ratio of 0 is
   * ANIRA_ERROR_CONFIG at prepare.
   *
   * @param spec The spec.
   * @param num This tensor advances num elements ...
   * @param den ... per den anchor elements; (0, 0) = derive (default): a fixed window's
   *   ratio from its hop, a flexible window's 1/1.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a negative value or den == 0
   *   with num != 0. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_time_ratio(spec: number, num: bigint, den: bigint): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_tensor_spec_set_time_ratio(spec, num, den),
      'anira_tensor_spec_set_time_ratio'
    )
  }

  /**
   * Outputs only: the model's internal delay along the Time axis, which the reported
   * latency adds.
   *
   * @param spec The spec.
   * @param latency Model-internal delay along Time, in elements; default 0.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a negative latency. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_latency(spec: number, latency: bigint): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_tensor_spec_set_latency(spec, latency),
      'anira_tensor_spec_set_latency'
    )
  }

  /**
   * Declared state passing: pairs a state input with the state output it is fed from on
   * the next inference. Both halves carry ANIRA_ROLE_STATE; the pairing is stated once,
   * on the input (JSON: "state_source" on the input spec). anira keeps one value per pair
   * in the handler, on the port of the state input, in the spec's shape and dtype (zeroed
   * at create and at prepare), copies it into the input tensor as the first step before
   * the engine call, ahead of the processor's before_inference, and copies the output
   * tensor into it as the last step after the engine call, behind the processor's
   * after_inference; a failed inference (an engine failure, a non-OK processor) skips the
   * capture, so the state keeps its last good value. anira_handler_reset re-initialises
   * the state at the first inference of the new stream; the state survives
   * anira_handler_set_plan. A model with a pair runs as ANIRA_MODEL_STATEFUL whatever its
   * state says. Validation (ANIRA_ERROR_CONFIG naming the tensor, at create and at
   * prepare): every state input names exactly one state output and every state output is
   * named by exactly one state input; the two halves have equal dtype and shape; a state
   * spec has no Time axis, window, time ratio or latency, and neither a ring dtype nor
   * the anchor may name it. ANIRA_ERROR_NOT_SUPPORTED: a state tensor of another dtype
   * than float32 (as every model tensor of this pre-release; the value takes the spec's
   * dtype), or with a transposed layout. A second call replaces the name.
   *
   * @param spec The spec of a state INPUT (role ANIRA_ROLE_STATE).
   * @param output_canonical The canonical name of the state output this input is fed
   *   from, UTF-8, copied; resolved when the model is validated (anira_handler_create,
   *   anira_handler_prepare), like the anchor's name.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec, a NULL or empty
   *   name, or a spec whose role is not ANIRA_ROLE_STATE. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_state_source(spec: number, output_canonical: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const output_canonical_ = output_canonical === null ? 0 : m.stringToUTF8OnStack(output_canonical)
      return checkStatus(
        m,
        m._anira_tensor_spec_set_state_source(spec, output_canonical_),
        'anira_tensor_spec_set_state_source'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Sets an extension on the spec (section 1b); one slot per kind, a second set replaces
   * the first.
   *
   * @param spec The spec.
   * @param ext The payload; deep-copied through the registry row.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short header;
   *   ANIRA_ERROR_EXTENSION_VERSION for a known kind at an unregistered version. A failed
   *   status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_ext(spec: number, ext: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_tensor_spec_set_ext(spec, ext, err),
        'anira_tensor_spec_set_ext',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The JSON twin of set_ext: a known kind is parsed through its registry row, an unknown
   * kind keeps the text.
   *
   * @param spec The spec.
   * @param kind The extension kind.
   * @param utf8 The extension object as JSON text, optionally with a "version" member
   *   (default 1).
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL kind or text;
   *   ANIRA_ERROR_JSON for malformed text; ANIRA_ERROR_EXTENSION_VERSION for a known kind
   *   at an unregistered version. A failed status throws an AniraError with the entry's
   *   message.
   * Thread: [main-thread]
   */
  anira_tensor_spec_set_ext_json(spec: number, kind: string, utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_tensor_spec_set_ext_json(
          spec,
          kind_,
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          err
        ),
        'anira_tensor_spec_set_ext_json',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Destroys a spec; NULL-safe. A spec added to a model config was copied and may be
   * destroyed right after.
   *
   * @param spec The spec, or NULL.
   * Thread: [main-thread]
   */
  anira_tensor_spec_destroy(spec: number): void {
    const m = this.module
    m._anira_tensor_spec_destroy(spec)
  }

  /**
   * Creates a Hard (real-time) contract with the stream geometry; the fixed-block host
   * earns the tight latency. The defaults: MEASURED budget, UNTIL_STABLE warmup, BYPASS
   * on miss, wait_ratio 0, PERMISSIVE edge cost. The blocks are both 0 (the geometry left
   * unset, completed by hard_set_geometry or refused at prepare: 0, 0, 0 is legal here)
   * or both finite and > 0 with block_min <= block_max; block_min < block_max allows
   * every smaller block. A block need not be whole: a fractional block (for a model with
   * no stream at the host rate, e.g. a control-rate decoder at samplesPerBlock / 2048.0)
   * is taken as the rational the latency calculation recovers from it (1e-6 relative,
   * denominator at most 2^20), and the host moves an element of a stream only once a
   * whole one has accumulated: after its k-th call it has pushed and popped floor(k * b)
   * elements of a stream whose block is b, so one call carries floor(b) or ceil(b) of
   * them (0 or 1 for 0.25, the 1 on every fourth call).
   *
   * @param block_min Smallest block the host callback delivers, in Time-axis elements of
   *   the anchor tensor; may be fractional (0.25: one element every fourth callback).
   * @param block_max Largest; block_min == block_max is the fixed-block host. May be
   *   fractional like block_min.
   * @param rate Anchor elements per second (48000 for audio).
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_contract_create_hard(block_min: number, block_max: number, rate: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_contract_create_hard(block_min, block_max, rate, out_, err),
        'anira_contract_create_hard',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Creates an Async contract: no deadline (the offline posture), FINISH on late, AUTO
   * priority, auto lanes and depth, POLLED delivery, PERMISSIVE edge cost.
   *
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_contract_create_async(): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_contract_create_async(out_, err),
        'anira_contract_create_async',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Patches the stream geometry, e.g. of a contract loaded from a file (section 8), under
   * the rules of anira_contract_create_hard; a refusal leaves the geometry as it was.
   *
   * @param contract A Hard contract.
   * @param block_min See anira_contract_create_hard; may be fractional.
   * @param block_max See anira_contract_create_hard; may be fractional.
   * @param rate See anira_contract_create_hard.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a block that is NaN, infinite or
   *   negative, one block 0 and the other not, block_min > block_max, or a negative rate. A
   *   failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_geometry(
    contract: number,
    block_min: number,
    block_max: number,
    rate: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_hard_set_geometry(contract, block_min, block_max, rate),
      'anira_contract_hard_set_geometry'
    )
  }

  /**
   * The per-inference budget of a Hard contract (v2's max_inference_time when EXPLICIT).
   *
   * @param contract A Hard contract.
   * @param kind MEASURED (default) derives the budget during warmup; EXPLICIT reads
   *   explicit_ms.
   * @param explicit_ms Per-inference budget in milliseconds, read for EXPLICIT only.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for an unknown kind or, with EXPLICIT, explicit_ms <= 0.
   *   A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_budget(
    contract: number,
    kind: number,
    explicit_ms: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_hard_set_budget(contract, kind, explicit_ms),
      'anira_contract_hard_set_budget'
    )
  }

  /**
   * The warmup policy of a Hard contract (v2's warm_up when FIXED).
   *
   * @param contract A Hard contract.
   * @param mode UNTIL_STABLE (default), FIXED, or NONE (legal only with an EXPLICIT
   *   budget, checked at prepare).
   * @param iterations Iterations for FIXED only.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for an unknown mode. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_warmup(
    contract: number,
    mode: number,
    iterations: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_hard_set_warmup(contract, mode, iterations),
      'anira_contract_hard_set_warmup'
    )
  }

  /**
   * What the handler delivers when an inference misses its deadline.
   *
   * @param contract A Hard contract.
   * @param policy BYPASS (default; requires shape-compatible I/O along the anchored Time
   *   axis), HOLD_LAST, ZEROS or CALLBACK (needs anira_contract_hard_set_miss_fn, in either
   *   order; checked at prepare).
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for an unknown policy. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_on_miss(contract: number, policy: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_hard_set_on_miss(contract, policy),
      'anira_contract_hard_set_on_miss'
    )
  }

  /**
   * The backup function of ANIRA_MISS_CALLBACK. The pair is copied with the contract at
   * anira_handler_prepare and must outlive the prepared handler; it is read only under
   * the CALLBACK policy, in whichever order the two setters ran. A contract file can say
   * "on_miss": "callback" but cannot carry a function: set the pair on the parsed
   * contract before prepare.
   *
   * @param contract A Hard contract.
   * @param fn The backup function, or NULL to clear the pair.
   * @param user_data Handed to fn as it is; never read by anira.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_miss_fn(
    contract: number,
    fn: number,
    user_data: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_hard_set_miss_fn(contract, fn, user_data),
      'anira_contract_hard_set_miss_fn'
    )
  }

  /**
   * The wait ratio consumed by the _wait twins only: how long, as a fraction of the block
   * duration, a wait entry may block for a result.
   *
   * @param contract A Hard contract.
   * @param ratio 0 (default) = never wait; v2's blocking_ratio one-to-one.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a negative ratio. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_wait_ratio(contract: number, ratio: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_hard_set_wait_ratio(contract, ratio),
      'anira_contract_hard_set_wait_ratio'
    )
  }

  /**
   * The ring dtype of one tensor under a Hard contract: the element type the tensor forms
   * of the Hard entries carry across the ABI and anira_ring_dtype reports, held by the
   * ring as is. Nothing in anira converts: the Hard entries copy between the host and the
   * ring, a ring dtype that differs from the spec's dtype (the model's) is
   * ANIRA_ERROR_CONFIG at prepare, unless a processor of the pipeline fills the phase
   * that moves that ring (pre_process for an input, post_process for an output) and so
   * takes the difference on itself; the default bodies of anira/abi/processor.h still
   * refuse such a slot at run time. Set per tensor by canonical name, so an input and an
   * output may differ; ANIRA_DTYPE_F32 for every tensor never set. A name that matches no
   * Streamed tensor is checked at prepare, not here.
   *
   * @param contract A Hard contract.
   * @param canonical The tensor's canonical name (the one its spec was created with).
   * @param dtype The element type of the host's samples for that tensor; ANIRA_DTYPE_F32
   *   for every tensor that was never set.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL or empty name, or a dtype of
   *   0. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_ring_dtype(
    contract: number,
    canonical: string,
    dtype: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      return checkStatus(
        m,
        m._anira_contract_hard_set_ring_dtype(contract, canonical_, dtype),
        'anira_contract_hard_set_ring_dtype'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The declared stream latency of one output under a Hard contract, in samples of that
   * output: what anira_handler_get_latency reports for the slot and what its receive ring
   * is primed with, REPLACING the figure anira computes (the buffer-adaptation delay, the
   * queue term, the wait credit and the model's internal latency;
   * docs/sphinx/latency.rst). A host that needs one figure across block sizes, or one
   * that aligns this stream with another, sets it; every output never named keeps the
   * computed figure. Never below the model's internal latency
   * (anira_tensor_spec_set_latency: a stream cannot deliver before the model does).
   * Checked at anira_handler_prepare, not here: a figure below that floor, and a name
   * that matches no Streamed output (an input has no stream latency), are
   * ANIRA_ERROR_CONFIG there. Set per output by canonical name, a second set of a name
   * replacing the first. In a contract file the key is "latencies": {"<name>": samples}
   * in the hard object.
   *
   * @param contract A Hard contract.
   * @param canonical The canonical name of a Streamed output (the one its spec was
   *   created with).
   * @param samples The stream latency in samples of that output, at most INT32_MAX.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL or empty name, or samples
   *   above INT32_MAX (the scheduler's per-output figure is a long: one bound on every
   *   platform). A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_set_latency(
    contract: number,
    canonical: string,
    samples: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      return checkStatus(
        m,
        m._anira_contract_hard_set_latency(contract, canonical_, samples),
        'anira_contract_hard_set_latency'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The per-job deadline of an Async contract; an absolute per-job override is the
   * deadline_ms argument of anira_handler_submit.
   *
   * @param contract An Async contract.
   * @param deadline_ms < 0 (default) = none, the offline posture; the clock starts at
   *   submit.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on a Hard contract. A failed status
   *   throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_async_set_deadline(contract: number, deadline_ms: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_async_set_deadline(contract, deadline_ms),
      'anira_contract_async_set_deadline'
    )
  }

  /**
   * The scheduling policy of an Async contract.
   *
   * @param contract An Async contract.
   * @param on_late FINISH (default) or DROP (cancels at chunk boundaries, enables
   *   admission control).
   * @param priority AUTO (default): INTERACTIVE iff a deadline is set, else BATCH.
   * @param lanes Parallel plan instances; 0 = auto (1 if STATEFUL, else
   *   min(max_instances, pool-derived)).
   * @param max_in_flight Per-lane pipelining; 0 = auto (shallow iff deadline, else deep).
   * @param delivery POLLED (default) or IMMEDIATE.
   * @returns ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on a Hard contract;
   *   ANIRA_ERROR_INVALID_ARGUMENT for an unknown enum value. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_async_set_policy(
    contract: number,
    on_late: number,
    priority: number,
    lanes: number,
    max_in_flight: number,
    delivery: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_async_set_policy(
        contract,
        on_late,
        priority,
        lanes,
        max_in_flight,
        delivery
      ),
      'anira_contract_async_set_policy'
    )
  }

  /**
   * Plan validation policy for the edges a pipeline uses (section 7); not scheduling.
   *
   * @param contract Either contract kind.
   * @param cost PERMISSIVE (default) or STRICT.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown value. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_set_edge_cost(contract: number, cost: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_contract_set_edge_cost(contract, cost),
      'anira_contract_set_edge_cost'
    )
  }

  /**
   * The host-end domain of one tensor, common to both contract kinds like
   * anira_contract_set_edge_cost: the domain anira allocates the ring, the model tensor
   * and the Static store of the slot in, and the domain all four phases of a processor,
   * the feed and the capture work in; for a State input it overrides the domain of the
   * pair's two buffers, which is the backend's domain of the plans that run the pair
   * otherwise (host memory for every backend of this pre-release), the domain the bound
   * State descriptors report; the planner joins it to the backend's domain per slot with
   * the edge of the registry's rows (anira_plan_slot.domain_in / domain_out / edge_class
   * / recipe). A tensor declared in the backend's own domain has no edge to cross. The
   * ring, the model tensor and the tensors the context accessors fill share the declared
   * domain, readable from the domain field of any tensor an accessor fills or from the
   * plan row. Set per tensor by canonical name, resolved at anira_handler_prepare: a name
   * that matches no tensor is ANIRA_ERROR_CONFIG there, and in this pre-release any
   * domain but ANIRA_DOMAIN_HOST is ANIRA_ERROR_NOT_SUPPORTED there, naming the tensor
   * (the declaration is data; anira's runtime allocates in host memory only). In a
   * contract file the key is a top-level "host_domains": {"<name>": "<domain word>"}, the
   * words the lower-case suffixes of anira_domain ("host", "host_pinned", "cuda", ...).
   *
   * @param contract Either contract kind.
   * @param canonical The tensor's canonical name (the one its spec was created with): any
   *   tensor of either side, Streamed, Buffer, Static and State alike.
   * @param domain The domain of the tensor's host end; ANIRA_DOMAIN_HOST for every tensor
   *   that was never set.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL or empty
   *   name, or a domain that is no anira_domain value. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_set_host_domain(
    contract: number,
    canonical: string,
    domain: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      return checkStatus(
        m,
        m._anira_contract_set_host_domain(contract, canonical_, domain),
        'anira_contract_set_host_domain'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Sets an extension on the contract (section 1b); v3.0.0 registers none for this host.
   *
   * @param contract Either contract kind.
   * @param ext The payload; deep-copied through the registry row.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short header;
   *   ANIRA_ERROR_EXTENSION_VERSION for a known kind at an unregistered version. A failed
   *   status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_contract_set_ext(contract: number, ext: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_contract_set_ext(contract, ext, err),
        'anira_contract_set_ext',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The JSON twin of anira_contract_set_ext.
   *
   * @param contract Either contract kind.
   * @param kind The extension kind.
   * @param utf8 The extension object as JSON text.
   * @returns As anira_tensor_spec_set_ext_json. A failed status throws an AniraError with
   *   the entry's message.
   * Thread: [main-thread]
   */
  anira_contract_set_ext_json(contract: number, kind: string, utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_contract_set_ext_json(
          contract,
          kind_,
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          err
        ),
        'anira_contract_set_ext_json',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Whether the contract is Hard or Async.
   *
   * @param contract The contract.
   * @returns ANIRA_CONTRACT_HARD or ANIRA_CONTRACT_ASYNC.
   * Thread: [main-thread]
   */
  anira_contract_get_kind(contract: number): number {
    const m = this.module
    return m._anira_contract_get_kind(contract) >>> 0
  }

  /**
   * Destroys a contract; NULL-safe.
   *
   * @param contract The contract, or NULL.
   * Thread: [main-thread]
   */
  anira_contract_destroy(contract: number): void {
    const m = this.module
    m._anira_contract_destroy(contract)
  }

  /**
   * Creates a context config with the defaults: ANIRA_THREADS_AUTO threads, SPIN_BACKOFF,
   * log level WARNING, the drain thread every 10 ms, a 512-record queue, no sink, no
   * device blocks.
   *
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_context_config_create(): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_context_config_create(out_, err),
        'anira_context_config_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The inference thread pool the first context sizes.
   *
   * @param config The config.
   * @param num_threads Pool size; ANIRA_THREADS_AUTO = the library default, 0 = bring
   *   your own threads.
   * @param wait SPIN_BACKOFF (default) or BLOCKING.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown wait strategy. A
   *   failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_threads(
    config: number,
    num_threads: number,
    wait: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_threads(config, num_threads, wait),
      'anira_context_config_set_threads'
    )
  }

  /**
   * The runtime log level.
   *
   * @param config The config.
   * @param level Default WARNING; the most verbose request across contexts wins.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown level. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_log_level(config: number, level: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_log_level(config, level),
      'anira_context_config_set_log_level'
    )
  }

  /**
   * Who drains the real-time log queue and how often.
   *
   * @param config The config.
   * @param drain THREAD (default) or MANUAL (anira_drain_log).
   * @param interval_ms Drain-thread period in milliseconds; 0 keeps the default of 10.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown drain. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_log_drain(
    config: number,
    drain: number,
    interval_ms: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_log_drain(config, drain, interval_ms),
      'anira_context_config_set_log_drain'
    )
  }

  /**
   * The real-time log queue's capacity.
   *
   * @param config The config.
   * @param capacity Records; clamped to [64, 65536]; fixed for the life of the core.
   * @returns ANIRA_OK. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_log_queue_capacity(config: number, capacity: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_log_queue_capacity(config, capacity),
      'anira_context_config_set_log_queue_capacity'
    )
  }

  /**
   * Log flags, e.g. ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK.
   *
   * @param config The config.
   * @param flags ANIRA_LOG_FLAG_* bits.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown bit. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_log_flags(config: number, flags: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_log_flags(config, flags),
      'anira_context_config_set_log_flags'
    )
  }

  /**
   * The context's log sink, on every target: on WebAssembly a C sink compiled into the
   * module receives the records like any other, and anira_em_set_log_hook
   * (anira/anira_em.h) adds the JavaScript hook beside it.
   *
   * @param config The config.
   * @param callback The sink, or NULL for none.
   * @param user_data Passed to callback.
   * @returns ANIRA_OK. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_log_sink(
    config: number,
    callback: number,
    user_data: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_log_sink(config, callback, user_data),
      'anira_context_config_set_log_sink'
    )
  }

  /**
   * The C one-shot convenience equal to the five scalar log setters.
   *
   * @param config The config.
   * @param desc The log block; read within min(struct_size, sizeof(anira_log_desc)).
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short descriptor;
   *   ANIRA_ERROR_ABI_VERSION when desc->abi_version fails anira_check_abi. A failed status
   *   throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_log(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_log(config, desc),
      'anira_context_config_set_log'
    )
  }

  /**
   * Declares the CUDA device block; presence is the user's declaration, no implicit
   * probing.
   *
   * @param config The config.
   * @param desc The block, or NULL = domain unavailable, edges pruned.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_cuda(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_cuda(config, desc),
      'anira_context_config_set_cuda'
    )
  }

  /**
   * Declares the OpenGL device block.
   *
   * @param config The config.
   * @param desc The block, or NULL.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_gl(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_gl(config, desc),
      'anira_context_config_set_gl'
    )
  }

  /**
   * Declares the Vulkan device block.
   *
   * @param config The config.
   * @param desc The block, or NULL.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_vulkan(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_vulkan(config, desc),
      'anira_context_config_set_vulkan'
    )
  }

  /**
   * Declares the Metal device block.
   *
   * @param config The config.
   * @param desc The block, or NULL.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_metal(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_metal(config, desc),
      'anira_context_config_set_metal'
    )
  }

  /**
   * Declares the Direct3D 12 device block.
   *
   * @param config The config.
   * @param desc The block, or NULL.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_d3d12(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_d3d12(config, desc),
      'anira_context_config_set_d3d12'
    )
  }

  /**
   * Declares the WebGPU device block (native Dawn); ANIRA_ERROR_NOT_SUPPORTED under
   * Emscripten in 3.0.
   *
   * @param config The config.
   * @param desc The block, or NULL.
   * @returns ANIRA_OK, ANIRA_ERROR_NOT_SUPPORTED, or ANIRA_ERROR_INVALID_ARGUMENT for a
   *   short descriptor. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_set_webgpu(config: number, desc: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_set_webgpu(config, desc),
      'anira_context_config_set_webgpu'
    )
  }

  /**
   * Sets an extension on the context config (section 1b).
   *
   * @param config The config.
   * @param ext The payload; deep-copied through the registry row.
   * @returns As anira_tensor_spec_set_ext. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_context_config_set_ext(config: number, ext: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_context_config_set_ext(config, ext, err),
        'anira_context_config_set_ext',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The JSON twin of anira_context_config_set_ext.
   *
   * @param config The config.
   * @param kind The extension kind.
   * @param utf8 The extension object as JSON text.
   * @returns As anira_tensor_spec_set_ext_json. A failed status throws an AniraError with
   *   the entry's message.
   * Thread: [main-thread]
   */
  anira_context_config_set_ext_json(config: number, kind: string, utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_context_config_set_ext_json(
          config,
          kind_,
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          err
        ),
        'anira_context_config_set_ext_json',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Destroys a context config; NULL-safe.
   *
   * @param config The config, or NULL.
   * Thread: [main-thread]
   */
  anira_context_config_destroy(config: number): void {
    const m = this.module
    m._anira_context_config_destroy(config)
  }

  /**
   * Creates an empty model config: no models, no tensors, default engine NONE (= plan 0,
   * the first plan of the table), STATELESS, max_instances ANIRA_INSTANCES_AUTO, anchor =
   * the first Streamed input.
   *
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_create(): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_model_config_create(out_, err),
        'anira_model_config_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Appends a model entry that loads from a file, for the engine the pair names (the pair
   * rule of anira_engine). Whether a built-in engine is in this build is decided at
   * anira_handler_create (an entry of an engine the build lacks is skipped when the
   * candidates leave it out, refused when they name it), so a config can name every
   * engine a deployment might have. A custom engine is named by the id
   * anira_custom_engine_create gave it; the entry and the engine's addition to the
   * pipeline (anira_pipeline_add_engine) may come in either order, and an id no engine of
   * the pipeline has is ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create, naming the id.
   *
   * @param config The config.
   * @param engine A built-in engine, or ANIRA_ENGINE_CUSTOM for a custom engine (not
   *   NONE).
   * @param engine_id The custom engine's reverse-URI id (must contain a '.'), copied,
   *   beside ANIRA_ENGINE_CUSTOM; NULL beside a built-in engine.
   * @param utf8_path Model file path, UTF-8, copied; NULL for an entry without a source,
   *   which anira_handler_create admits for a custom engine that declares
   *   ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL alone (the JSON form: an entry without "path").
   * @returns What the entry writes to out_index. A failed status throws an AniraError
   *   with the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_add_model_path(
    config: number,
    engine: number,
    engine_id: string | null,
    utf8_path: string | null
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_id_ = engine_id === null ? 0 : m.stringToUTF8OnStack(engine_id)
      const utf8_path_ = utf8_path === null ? 0 : m.stringToUTF8OnStack(utf8_path)
      const out_index_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_model_config_add_model_path(
          config,
          engine,
          engine_id_,
          utf8_path_,
          out_index_,
          err
        ),
        'anira_model_config_add_model_path',
        err
      )
      return heap(m).HEAPU32[out_index_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Appends a model entry that loads from memory, for the engine the pair names (as
   * anira_model_config_add_model_path).
   *
   * @param config The config.
   * @param engine A built-in engine, or ANIRA_ENGINE_CUSTOM for a custom engine (not
   *   NONE).
   * @param engine_id The custom engine's reverse-URI id, copied, beside
   *   ANIRA_ENGINE_CUSTOM; NULL beside a built-in engine.
   * @param bytes The model bytes.
   * @param size Their size; > 0.
   * @param ownership COPY, or BORROW (the plugin default for embedded blobs).
   * @param release Optional release callback for BORROW; fires once when the last carrier
   *   dies.
   * @param ctx Passed to release.
   * @returns What the entry writes to out_index. A failed status throws an AniraError
   *   with the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_add_model_bytes(
    config: number,
    engine: number,
    engine_id: string | null,
    bytes: number,
    size: number,
    ownership: number,
    release: number,
    ctx: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_id_ = engine_id === null ? 0 : m.stringToUTF8OnStack(engine_id)
      const out_index_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_model_config_add_model_bytes(
          config,
          engine,
          engine_id_,
          bytes,
          size,
          ownership,
          release,
          ctx,
          out_index_,
          err
        ),
        'anira_model_config_add_model_bytes',
        err
      )
      return heap(m).HEAPU32[out_index_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Replaces an entry's source with bytes, e.g. to patch a path entry a JSON file
   * produced; the path is kept for anira_model_config_to_json, model_path() then returns
   * NULL.
   *
   * @param config The config.
   * @param model_index An existing entry.
   * @param bytes The model bytes.
   * @param size Their size; > 0.
   * @param ownership COPY or BORROW.
   * @param release Optional release callback for BORROW.
   * @param ctx Passed to release.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an index out of range, NULL
   *   bytes, a zero size or an unknown ownership. A failed status throws an AniraError with
   *   the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_model_bytes(
    config: number,
    model_index: number,
    bytes: number,
    size: number,
    ownership: number,
    release: number,
    ctx: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_model_bytes(
          config,
          model_index,
          bytes,
          size,
          ownership,
          release,
          ctx,
          err
        ),
        'anira_model_config_set_model_bytes',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The number of model entries.
   *
   * @param config The config.
   * @returns The count; 0 for NULL.
   * Thread: [main-thread]
   */
  anira_model_config_model_count(config: number): number {
    const m = this.module
    return m._anira_model_config_model_count(config) >>> 0
  }

  /**
   * The entry's engine as the pair (anira_engine's pair rule). A status getter of the
   * read-back: its out-parameters are written on ANIRA_OK only; the id out-parameter may
   * be NULL when the caller wants the value alone, and the id comes back NULL unless the
   * value is CUSTOM (the pair rule).
   *
   * @param config The config.
   * @param model_index An entry.
   * @returns What the entry writes to engine, engine_id. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_model_engine(
    config: number,
    model_index: number
  ): { engine: number; engine_id: string | null } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_ = stackSlot(m)
      const engine_id_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_model_config_model_engine(config, model_index, engine_, engine_id_),
        'anira_model_config_model_engine'
      )
      return {
        engine: heap(m).HEAPU32[engine_ >>> 2],
        engine_id: readCString(m, heap(m).HEAPU32[engine_id_ >>> 2]),
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Pins the entry to a provider: its file is built for one (an ExecuTorch export lowered
   * for a provider (an ExecuTorch delegate), an ONNX Runtime .ort compiled for an
   * execution provider), so only a candidate naming that provider runs it, and a
   * candidate naming it picks this entry. An entry without a pin is neutral: it runs on
   * any provider of its engine, the candidate deciding. Two entries of one engine may
   * coexist when their pins differ. JSON: the entry's "provider" key beside its "engine",
   * "coreml" or "cpu" (the enum's spellings) or "com.example.npu" (a custom name); no key
   * is a neutral entry, and the words "default" and "none" are refused.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param provider A provider of the enum, or ANIRA_PROVIDER_CUSTOM with a provider_id
   *   for a custom one; ANIRA_PROVIDER_NONE (with a NULL provider_id) unpins the entry;
   *   ANIRA_PROVIDER_CPU pins it to the CPU path.
   * @param provider_id A custom provider's name in the engine's vocabulary (the runtime's
   *   own name for a built-in engine, an entry of a custom engine's providers list),
   *   copied, beside ANIRA_PROVIDER_CUSTOM; NULL beside a provider of the enum.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config, an index out of
   *   range, an unknown provider value, or a pair against the pair rule of anira_provider
   *   (a provider_id beside a provider other than ANIRA_PROVIDER_CUSTOM, CUSTOM without a
   *   provider_id or with an empty one). A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_model_provider(
    config: number,
    model_index: number,
    provider: number,
    provider_id: string | null
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const provider_id_ = provider_id === null ? 0 : m.stringToUTF8OnStack(provider_id)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_model_provider(
          config,
          model_index,
          provider,
          provider_id_,
          err
        ),
        'anira_model_config_set_model_provider',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The provider the entry is pinned to, as the pair (anira_provider's pair rule). A
   * status getter of the read-back: its out-parameters are written on ANIRA_OK only; the
   * id out-parameter may be NULL when the caller wants the value alone, and the id comes
   * back NULL unless the value is CUSTOM (the pair rule).
   *
   * @param config The config.
   * @param model_index An entry.
   * @returns What the entry writes to provider, provider_id. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_model_provider(
    config: number,
    model_index: number
  ): { provider: number; provider_id: string | null } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const provider_ = stackSlot(m)
      const provider_id_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_model_config_model_provider(
          config,
          model_index,
          provider_,
          provider_id_
        ),
        'anira_model_config_model_provider'
      )
      return {
        provider: heap(m).HEAPU32[provider_ >>> 2],
        provider_id: readCString(m, heap(m).HEAPU32[provider_id_ >>> 2]),
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The entry's model path.
   *
   * @param config The config.
   * @param model_index An entry.
   * @returns Object-owned; NULL for a bytes entry or an index out of range.
   * Thread: [main-thread]
   */
  anira_model_config_model_path(config: number, model_index: number): string | null {
    const m = this.module
    return readCString(m, m._anira_model_config_model_path(config, model_index))
  }

  /**
   * The entry's model bytes.
   *
   * @param config The config.
   * @param model_index An entry.
   * @returns What the entry writes to bytes, size. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_model_bytes(
    config: number,
    model_index: number
  ): { bytes: number; size: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const bytes_ = stackSlot(m)
      const size_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_model_config_model_bytes(config, model_index, bytes_, size_),
        'anira_model_config_model_bytes'
      )
      return {
        bytes: heap(m).HEAPU32[bytes_ >>> 2],
        size: heap(m).HEAPU32[size_ >>> 2],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Records what this entry's file calls the tensor you named canonical, and switches
   * that tensor of this entry from positional binding (input slot i to the file's input
   * i, the slot order being the add_input/add_output order; ONNX Runtime's session order,
   * the primary subgraph's order on TFLite and LiteRT) to binding by that name. An entry
   * without a record for a tensor binds it positionally. A name the engine cannot
   * resolve, or an engine that binds only positionally on that side, fails prepare with
   * ANIRA_ERROR_CONFIG naming what the file has. Together with
   * anira_model_config_set_tensor_layout this is the per-entry tensor record, the JSON
   * file's models[].tensors.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param canonical Your canonical name of the tensor (the spec's name); the spec may be
   *   added later, the name is resolved at prepare.
   * @param export_name The export's name for that tensor, copied: ONNX Runtime the
   *   graph's input or output name; TFLite and LiteRT the signature key ("args_0",
   *   "output_0"), or the tensor name for a file without signatures; LibTorch the method's
   *   argument name (inputs only); ExecuTorch the tensor name when the export carries one.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an index out of range or a
   *   NULL or empty name. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_set_tensor_name(
    config: number,
    model_index: number,
    canonical: string,
    export_name: string
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      const export_name_ = export_name === null ? 0 : m.stringToUTF8OnStack(export_name)
      return checkStatus(
        m,
        m._anira_model_config_set_tensor_name(
          config,
          model_index,
          canonical_,
          export_name_
        ),
        'anira_model_config_set_tensor_name'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The axis order in which this entry's file holds the tensor you named canonical, when
   * it differs from the spec's (a TensorFlow export holding batch, time, channel where
   * the spec says batch, channel, time is {0, 2, 1}). A spec axis left out must have
   * extent 1. A layout that moves only axes of extent 1 is a view: the same bytes with
   * other dims, at no cost. One that moves an axis of another extent is a transpose,
   * refused at prepare in this pre-release with ANIRA_ERROR_NOT_SUPPORTED. Agreement with
   * the spec's rank and extents is checked at prepare, and against the file where the
   * engine reports its dims.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param canonical Your canonical name of the tensor (the spec's name); resolved at
   *   prepare.
   * @param axes ndim entries, copied: axes[k] is the spec axis (an index into
   *   anira_tensor_spec_set_axis's order, each at most once) that this entry's file holds
   *   at position k, or ANIRA_AXIS_INSERT for an axis of extent 1 the file has and the spec
   *   does not. NULL with ndim 0 clears.
   * @param ndim 1..ANIRA_MAX_RANK, or 0 to clear.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an index out of range, a NULL
   *   or empty canonical, ndim > ANIRA_MAX_RANK, NULL axes with ndim > 0, a spec axis index
   *   >= ANIRA_MAX_RANK that is not ANIRA_AXIS_INSERT, or a spec axis listed twice. A
   *   failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_set_tensor_layout(
    config: number,
    model_index: number,
    canonical: string,
    axes: number,
    ndim: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      return checkStatus(
        m,
        m._anira_model_config_set_tensor_layout(
          config,
          model_index,
          canonical_,
          axes,
          ndim
        ),
        'anira_model_config_set_tensor_layout'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Sets an extension on one model entry (host "model"), e.g. anira_ext_entry.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param ext The payload; deep-copied through the registry row.
   * @returns As anira_tensor_spec_set_ext, plus ANIRA_ERROR_INVALID_ARGUMENT for an index
   *   out of range. A failed status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_model_ext(
    config: number,
    model_index: number,
    ext: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_model_ext(config, model_index, ext, err),
        'anira_model_config_set_model_ext',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The JSON twin of anira_model_config_set_model_ext.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param kind The extension kind.
   * @param utf8 The extension object as JSON text.
   * @returns As anira_tensor_spec_set_ext_json, plus ANIRA_ERROR_INVALID_ARGUMENT for an
   *   index out of range. A failed status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_model_ext_json(
    config: number,
    model_index: number,
    kind: string,
    utf8: string
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_model_ext_json(
          config,
          model_index,
          kind_,
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          err
        ),
        'anira_model_config_set_model_ext_json',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Appends an input tensor spec (copied; the caller keeps ownership of spec). The order
   * of the add_input calls is the slot order an entry without a name record binds
   * positionally.
   *
   * @param config The config.
   * @param spec The spec; copied.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec or a spec whose
   *   canonical name an input or output of this config already has. A failed status throws
   *   an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_add_input(config: number, spec: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_model_config_add_input(config, spec),
      'anira_model_config_add_input'
    )
  }

  /**
   * Appends an output tensor spec (copied). The order of the add_output calls is the slot
   * order an entry without a name record binds positionally.
   *
   * @param config The config.
   * @param spec The spec; copied.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec or a spec whose
   *   canonical name an input or output of this config already has. A failed status throws
   *   an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_add_output(config: number, spec: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_model_config_add_output(config, spec),
      'anira_model_config_add_output'
    )
  }

  /**
   * The engine the handler starts on, the pair of anira_engine's pair rule: its first
   * plan in plan-table order, on the default provider when one is set
   * (anira_model_config_set_default_provider), so an engine with several plans (one entry
   * under several providers, or several entries) starts on the pinned provider. One rule
   * for both defaults (this one and anira_model_config_set_default_provider): what the
   * configuration alone decides is checked at anira_handler_create and at prepare,
   * ANIRA_ERROR_CONFIG for a default engine that names no model entry ("default_engine
   * 'x' names no model entry") and for a default provider no entry of the default engine
   * could run (every one pinned to another provider); whether a plan runs the default
   * here is the candidates' and the context's, and when none does the handler starts on
   * the default engine's first plan, else on plan 0, with one Warning at
   * anira_handler_prepare naming what was asked and the plan it starts on; never a
   * refusal. anira_handler_get_plan says which plan runs.
   *
   * @param config The config.
   * @param engine A built-in engine, ANIRA_ENGINE_CUSTOM for a custom engine, or
   *   ANIRA_ENGINE_NONE = plan 0, the first plan of the table (default).
   * @param engine_id The custom engine's reverse-URI id, copied, beside
   *   ANIRA_ENGINE_CUSTOM; NULL beside a built-in engine and NONE.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config, an unknown
   *   engine or a pair against the pair rule (an id beside a built-in engine or NONE,
   *   ANIRA_ENGINE_CUSTOM without an id or with an id without a '.'). A failed status
   *   throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_default_engine(
    config: number,
    engine: number,
    engine_id: string | null
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_id_ = engine_id === null ? 0 : m.stringToUTF8OnStack(engine_id)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_default_engine(config, engine, engine_id_, err),
        'anira_model_config_set_default_engine',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The provider the handler starts on, beside the default engine: the handler starts on
   * the first plan in plan-table order whose engine is the default engine (any engine
   * without one) and whose provider is this one, so a default engine with several plans
   * (one entry under several providers, or several entries) starts on the pinned provider
   * rather than on the first of its plans. The rule of
   * anira_model_config_set_default_engine holds for both defaults: ANIRA_ERROR_CONFIG at
   * anira_handler_create when every entry of the default engine (every entry without one)
   * is pinned to another provider ("default_provider 'x' runs no model entry ..."; a
   * neutral entry may run on it); when no plan runs on it here (not a candidate, not
   * usable here) the handler starts as without it, with the one Warning at
   * anira_handler_prepare.
   *
   * @param config The config.
   * @param provider A provider of the enum, or ANIRA_PROVIDER_CUSTOM beside a
   *   provider_id; ANIRA_PROVIDER_NONE (with a NULL provider_id) sets none (default).
   * @param provider_id A custom provider's name in the engine's own vocabulary
   *   (anira_provider), copied, beside ANIRA_PROVIDER_CUSTOM; NULL beside a provider of the
   *   enum.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config, a provider this
   *   header does not name, or a pair against the pair rule of anira_provider (a
   *   provider_id beside a provider other than ANIRA_PROVIDER_CUSTOM, CUSTOM without a
   *   provider_id or with an empty one). A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_set_default_provider(
    config: number,
    provider: number,
    provider_id: string | null
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const provider_id_ = provider_id === null ? 0 : m.stringToUTF8OnStack(provider_id)
      return checkStatus(
        m,
        m._anira_model_config_set_default_provider(config, provider, provider_id_),
        'anira_model_config_set_default_provider'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Whether the model carries state across inferences (v2's session_exclusive_processor).
   *
   * @param config The config.
   * @param state STATELESS (default) or STATEFUL (session-exclusive, lanes forced to 1).
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown state. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_set_state(config: number, state: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_model_config_set_state(config, state),
      'anira_model_config_set_state'
    )
  }

  /**
   * The ceiling within which the planner allocates lanes and pool instances (v2's
   * num_parallel_processors). Each instance is a full copy of the model in memory; set 1
   * to keep one. A stateful model (or one with a declared State pair) has no shared
   * instance whatever it says: each of its handlers runs on its own.
   *
   * @param config The config.
   * @param max_instances Memory ceiling of loaded model instances; >= 1, or
   *   ANIRA_INSTANCES_AUTO (the default): the inference pool's thread count.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for 0. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_set_max_instances(config: number, max_instances: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_model_config_set_max_instances(config, max_instances),
      'anira_model_config_set_max_instances'
    )
  }

  /**
   * The anchor: the streamed tensor whose Time axis is the model's clock. A Hard
   * contract's block range and rate are counted in its elements, and every other streamed
   * tensor's time ratio is stated against it (v2's HostConfig reference stream). Default:
   * the first streamed input, or, for a model without one (a generator), the first
   * streamed output. Resolved at prepare, where a name that is not a streamed tensor of
   * this config fails with ANIRA_ERROR_CONFIG.
   *
   * @param config The config.
   * @param canonical Your canonical name of a streamed tensor, copied; NULL or empty
   *   restores the default.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config. A failed status
   *   throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_set_anchor(config: number, canonical: string | null): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      return checkStatus(
        m,
        m._anira_model_config_set_anchor(config, canonical_),
        'anira_model_config_set_anchor'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Sets an extension on the whole config (host "model_config").
   *
   * @param config The config.
   * @param ext The payload; deep-copied through the registry row.
   * @returns As anira_tensor_spec_set_ext. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_ext(config: number, ext: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_ext(config, ext, err),
        'anira_model_config_set_ext',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The JSON twin of anira_model_config_set_ext.
   *
   * @param config The config.
   * @param kind The extension kind.
   * @param utf8 The extension object as JSON text.
   * @returns As anira_tensor_spec_set_ext_json. A failed status throws an AniraError with
   *   the entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_set_ext_json(config: number, kind: string, utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_model_config_set_ext_json(
          config,
          kind_,
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          err
        ),
        'anira_model_config_set_ext_json',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Destroys a model config; NULL-safe. Borrowed model bytes are released (their callback
   * fires) when the last carrier dies.
   *
   * @param config The config, or NULL.
   * Thread: [main-thread]
   */
  anira_model_config_destroy(config: number): void {
    const m = this.module
    m._anira_model_config_destroy(config)
  }

  /**
   * Creates job options with the defaults: no head trim, tail flush on, REJECT below the
   * window minimum. Build once, reuse across submits; never mutate concurrently with a
   * submit that reads it.
   *
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_job_options_create(): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_job_options_create(out_, err),
        'anira_job_options_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Per-output head trim of an Async job's outputs.
   *
   * @param options The options.
   * @param count Number of trims, one per output.
   * @param trims Elements to drop at the head of each output; -1 = that output's latency
   *   (input-aligned). Copied.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL trims with count > 0,
   *   or a value below -1. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_job_options_set_head_trim(options: number, count: number, trims: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_job_options_set_head_trim(options, count, trims),
      'anira_job_options_set_head_trim'
    )
  }

  /**
   * ViewChunker reassembly semantics: whether the tail of a job is flushed through the
   * model.
   *
   * @param options The options.
   * @param tail_flush Default true.
   * @returns ANIRA_OK. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_job_options_set_tail_flush(options: number, tail_flush: boolean): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_job_options_set_tail_flush(options, tail_flush ? 1 : 0),
      'anira_job_options_set_tail_flush'
    )
  }

  /**
   * What happens to a submitted buffer shorter than the window minimum.
   *
   * @param options The options.
   * @param policy REJECT (default) or ZEROS.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown policy. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_job_options_set_below_min(options: number, policy: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_job_options_set_below_min(options, policy),
      'anira_job_options_set_below_min'
    )
  }

  /**
   * Sets a per-job extension (host "job"): borrowed, not copied, because submit is
   * ANIRA_NONBLOCKING and copies what it needs into the job record; the consumed-or-fail
   * check runs at submit and fails the ticket, not the handler.
   *
   * @param options The options.
   * @param ext The payload; BORROWED until every submit that reads the options has
   *   returned.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short header. A
   *   failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_job_options_set_ext(options: number, ext: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_job_options_set_ext(options, ext),
      'anira_job_options_set_ext'
    )
  }

  /**
   * The JSON twin every config handle carries; the parsed payload is owned by the
   * options.
   *
   * @param options The options.
   * @param kind The extension kind.
   * @param utf8 The extension object as JSON text.
   * @returns ANIRA_OK, ANIRA_ERROR_INVALID_ARGUMENT for a NULL kind or text,
   *   ANIRA_ERROR_JSON for malformed text, ANIRA_ERROR_EXTENSION_VERSION for a known kind
   *   at an unregistered version. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_job_options_set_ext_json(options: number, kind: string, utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      return checkStatus(
        m,
        m._anira_job_options_set_ext_json(
          options,
          kind_,
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8)
        ),
        'anira_job_options_set_ext_json'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Destroys job options; NULL-safe.
   *
   * @param options The options, or NULL.
   * Thread: [main-thread]
   */
  anira_job_options_destroy(options: number): void {
    const m = this.module
    m._anira_job_options_destroy(options)
  }

  /**
   * Loads a model config from JSON text (section 8.1). A version 2 document is upgraded
   * (section 8.4): the result carries the models, tensors and scalars, holds back
   * max_inference_time / warm_up / blocking_ratio as a legacy Hard contract for
   * anira_model_config_take_legacy_contract, logs one warning per process, and returns
   * ANIRA_SUCCESS_UPGRADED. Unknown keys are stored as extensions and fail prepare by
   * name.
   *
   * @param utf8 The document text (a v3 model file, or a v2 document with an
   *   inference_config root).
   * @param base_dir Directory relative model paths resolve against (the joined path uses
   *   forward slashes on every platform; a rooted path stays as written), or NULL to keep
   *   every path as written.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_from_json(utf8: string, base_dir: string | null): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const base_dir_ = base_dir === null ? 0 : m.stringToUTF8OnStack(base_dir)
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_model_config_from_json(
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          base_dir_,
          out_,
          err
        ),
        'anira_model_config_from_json',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Reads a file and loads it as anira_model_config_from_json with base_dir = the file's
   * directory.
   *
   * @param utf8_path Path of the file; its directory is the base_dir of the model paths.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_model_config_from_json_file(utf8_path: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const utf8_path_ = utf8_path === null ? 0 : m.stringToUTF8OnStack(utf8_path)
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_model_config_from_json_file(utf8_path_, out_, err),
        'anira_model_config_from_json_file',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Writes the config in v3 spelling (section 8.1), fixed key order; reading a v2 file
   * and writing it out is the migration tool. Bytes entries keep the path they were
   * loaded with.
   *
   * @param config The config.
   * @returns What the entry writes to buf. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_to_json(config: number): string {
    const m = this.module
    const sp = m.stackSave()
    try {
      const len = stackSlot(m)
      let buf = 0
      let cap = 0
      const sized = m._anira_model_config_to_json(config, buf, cap, len)
      if (sized !== anira_status.ANIRA_ERROR_BUFFER_TOO_SMALL) {
        checkStatus(m, sized, 'anira_model_config_to_json')
      }
      cap = heap(m).HEAPU32[len >>> 2] + 1
      buf = m._malloc(cap)
      if (buf === 0) {
        throw new AniraError(anira_status.ANIRA_ERROR_OUT_OF_MEMORY, 'anira_model_config_to_json')
      }
      try {
        checkStatus(
          m,
          m._anira_model_config_to_json(config, buf, cap, len),
          'anira_model_config_to_json'
        )
        return m.UTF8ToString(buf)
      } finally {
        m._free(buf)
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Hands out the Hard contract a version 2 upgrade held back (budget, warmup and wait
   * ratio; no geometry). Non-NULL only once after an upgrade: a second call, or a v3
   * document, yields NULL.
   *
   * @param config The config.
   * @returns What the entry writes to out. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_take_legacy_contract(config: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_model_config_take_legacy_contract(config, out_),
        'anira_model_config_take_legacy_contract'
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Loads a context config from JSON text (section 8.2). Device blocks in JSON imply
   * ANIRA_OWNERSHIP_OWNED; borrowed handles are code-only and patched afterwards with the
   * device setters. A version 2 document is upgraded (context_config; the bare log_level
   * key becomes log.level) and returns ANIRA_SUCCESS_UPGRADED.
   *
   * @param utf8 The document text (a v3 context file, or a v2 document with a
   *   context_config root).
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_context_config_from_json(utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_context_config_from_json(
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          out_,
          err
        ),
        'anira_context_config_from_json',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Writes the config in v3 spelling (section 8.2), fixed key order; the sink is
   * code-only and not written, device blocks are written without their borrowed handles.
   *
   * @param config The config.
   * @returns What the entry writes to buf. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_to_json(config: number): string {
    const m = this.module
    const sp = m.stackSave()
    try {
      const len = stackSlot(m)
      let buf = 0
      let cap = 0
      const sized = m._anira_context_config_to_json(config, buf, cap, len)
      if (sized !== anira_status.ANIRA_ERROR_BUFFER_TOO_SMALL) {
        checkStatus(m, sized, 'anira_context_config_to_json')
      }
      cap = heap(m).HEAPU32[len >>> 2] + 1
      buf = m._malloc(cap)
      if (buf === 0) {
        throw new AniraError(anira_status.ANIRA_ERROR_OUT_OF_MEMORY, 'anira_context_config_to_json')
      }
      try {
        checkStatus(
          m,
          m._anira_context_config_to_json(config, buf, cap, len),
          'anira_context_config_to_json'
        )
        return m.UTF8ToString(buf)
      } finally {
        m._free(buf)
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Loads a contract from JSON text (section 8.3): budget is "measured" or {"ms": x},
   * warmup is "until_stable", "none" or {"fixed": n}; a file with both roots or neither
   * is ANIRA_ERROR_JSON. A version 2 document yields its legacy Hard contract directly
   * (ANIRA_SUCCESS_UPGRADED).
   *
   * @param utf8 The document text: {"hard": {...}} or {"async": {...}} with an optional
   *   top-level edge_cost and an optional top-level host_domains ({"<name>": "<domain
   *   word>"}, anira_contract_set_host_domain), or a v2 document.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_contract_from_json(utf8: string): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const utf8_ = utf8 === null ? 0 : m.stringToUTF8OnStack(utf8)
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_contract_from_json(
          utf8_,
          utf8 === null ? 0 : m.lengthBytesUTF8(utf8),
          out_,
          err
        ),
        'anira_contract_from_json',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The canonical name the spec was created with.
   *
   * @param spec The spec.
   * @returns Object-owned; NULL for a NULL spec.
   * Thread: [main-thread]
   */
  anira_tensor_spec_name(spec: number): string | null {
    const m = this.module
    return readCString(m, m._anira_tensor_spec_name(spec))
  }

  /**
   * The element type the spec was created with: the model's.
   *
   * @param spec The spec.
   * @returns The dtype; 0, which is no dtype, for a NULL spec.
   * Thread: [main-thread]
   */
  anira_tensor_spec_dtype(spec: number): number {
    const m = this.module
    return m._anira_tensor_spec_dtype(spec) >>> 0
  }

  /**
   * The role the spec was created with.
   *
   * @param spec The spec.
   * @returns The role; ANIRA_ROLE_FORCE32, which is no role, for a NULL spec.
   * Thread: [main-thread]
   */
  anira_tensor_spec_role(spec: number): number {
    const m = this.module
    return m._anira_tensor_spec_role(spec) >>> 0
  }

  /**
   * The rank: one more than the highest axis anira_tensor_spec_set_axis set. An axis
   * below it that was never set is a hole, which anira_tensor_spec_axis reads as
   * {ANIRA_AXIS_ANY, 0}.
   *
   * @param spec The spec.
   * @returns The rank; 0 for a spec without axes and for a NULL spec.
   * Thread: [main-thread]
   */
  anira_tensor_spec_ndim(spec: number): number {
    const m = this.module
    return m._anira_tensor_spec_ndim(spec) >>> 0
  }

  /**
   * One axis as anira_tensor_spec_set_axis stored it. A hole below the rank reads
   * {ANIRA_AXIS_ANY, 0}, the value of an axis never set.
   *
   * @param spec The spec.
   * @param i The axis, below anira_tensor_spec_ndim.
   * @returns What the entry writes to tag, extent. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_axis(spec: number, i: number): { tag: number; extent: bigint } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const tag_ = stackSlot(m)
      const extent_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_tensor_spec_axis(spec, i, tag_, extent_),
        'anira_tensor_spec_axis'
      )
      return {
        tag: heap(m).HEAPU32[tag_ >>> 2],
        extent: heap(m).HEAP64[extent_ >>> 3],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The window as anira_tensor_spec_set_window stored it; 0, 0, 0 until it ran, on every
   * role.
   *
   * @param spec The spec.
   * @returns What the entry writes to window_min, window_max, overlap. A failed status
   *   throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_window(
    spec: number
  ): { window_min: bigint; window_max: bigint; overlap: bigint } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const window_min_ = stackSlot(m)
      const window_max_ = stackSlot(m)
      const overlap_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_tensor_spec_window(spec, window_min_, window_max_, overlap_),
        'anira_tensor_spec_window'
      )
      return {
        window_min: heap(m).HEAP64[window_min_ >>> 3],
        window_max: heap(m).HEAP64[window_max_ >>> 3],
        overlap: heap(m).HEAP64[overlap_ >>> 3],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The time ratio as anira_tensor_spec_set_time_ratio stored it.
   *
   * @param spec The spec.
   * @returns What the entry writes to num, den. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_tensor_spec_time_ratio(spec: number): { num: bigint; den: bigint } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const num_ = stackSlot(m)
      const den_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_tensor_spec_time_ratio(spec, num_, den_),
        'anira_tensor_spec_time_ratio'
      )
      return {
        num: heap(m).HEAP64[num_ >>> 3],
        den: heap(m).HEAP64[den_ >>> 3],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The model-internal latency anira_tensor_spec_set_latency stored.
   *
   * @param spec The spec.
   * @returns The latency in elements; 0 for a spec without one and for a NULL spec.
   * Thread: [main-thread]
   */
  anira_tensor_spec_latency(spec: number): bigint {
    const m = this.module
    return m._anira_tensor_spec_latency(spec)
  }

  /**
   * The canonical name of the State output a State input is fed from
   * (anira_tensor_spec_set_state_source).
   *
   * @param spec The spec.
   * @returns Object-owned; NULL for a spec without one and for a NULL spec.
   * Thread: [main-thread]
   */
  anira_tensor_spec_state_source(spec: number): string | null {
    const m = this.module
    return readCString(m, m._anira_tensor_spec_state_source(spec))
  }

  /**
   * The number of input specs, State specs included.
   *
   * @param config The config.
   * @returns The count; 0 for NULL.
   * Thread: [main-thread]
   */
  anira_model_config_num_inputs(config: number): number {
    const m = this.module
    return m._anira_model_config_num_inputs(config) >>> 0
  }

  /**
   * The number of output specs, State specs included.
   *
   * @param config The config.
   * @returns The count; 0 for NULL.
   * Thread: [main-thread]
   */
  anira_model_config_num_outputs(config: number): number {
    const m = this.module
    return m._anira_model_config_num_outputs(config) >>> 0
  }

  /**
   * The config's own copy of an input spec (anira_model_config_add_input copied it), for
   * the anira_tensor_spec getters only: never a spec to set or to destroy. Valid until
   * the config's next anira_model_config_add_input or anira_model_config_add_output,
   * which moves its specs, or its destruction.
   *
   * @param config The config.
   * @param index The slot: the spec's position in the input list.
   * @returns Object-owned; NULL for a NULL config or an index out of range.
   * Thread: [main-thread]
   */
  anira_model_config_input(config: number, index: number): number {
    const m = this.module
    return m._anira_model_config_input(config, index) >>> 0
  }

  /**
   * The output twin of anira_model_config_input, with the same lifetime.
   *
   * @param config The config.
   * @param index The slot: the spec's position in the output list.
   * @returns Object-owned; NULL for a NULL config or an index out of range.
   * Thread: [main-thread]
   */
  anira_model_config_output(config: number, index: number): number {
    const m = this.module
    return m._anira_model_config_output(config, index) >>> 0
  }

  /**
   * The pair anira_model_config_set_default_engine stored. A status getter of the
   * read-back: its out-parameters are written on ANIRA_OK only; the id out-parameter may
   * be NULL when the caller wants the value alone, and the id comes back NULL unless the
   * value is CUSTOM (the pair rule).
   *
   * @param config The config.
   * @returns What the entry writes to engine, engine_id. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_default_engine(
    config: number
  ): { engine: number; engine_id: string | null } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_ = stackSlot(m)
      const engine_id_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_model_config_default_engine(config, engine_, engine_id_),
        'anira_model_config_default_engine'
      )
      return {
        engine: heap(m).HEAPU32[engine_ >>> 2],
        engine_id: readCString(m, heap(m).HEAPU32[engine_id_ >>> 2]),
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The pair anira_model_config_set_default_provider stored. A status getter of the
   * read-back: its out-parameters are written on ANIRA_OK only; the id out-parameter may
   * be NULL when the caller wants the value alone, and the id comes back NULL unless the
   * value is CUSTOM (the pair rule).
   *
   * @param config The config.
   * @returns What the entry writes to provider, provider_id. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_default_provider(
    config: number
  ): { provider: number; provider_id: string | null } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const provider_ = stackSlot(m)
      const provider_id_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_model_config_default_provider(config, provider_, provider_id_),
        'anira_model_config_default_provider'
      )
      return {
        provider: heap(m).HEAPU32[provider_ >>> 2],
        provider_id: readCString(m, heap(m).HEAPU32[provider_id_ >>> 2]),
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The state as anira_model_config_set_state stored it; a config with a declared State
   * pair runs as ANIRA_MODEL_STATEFUL whatever it reads.
   *
   * @param config The config.
   * @returns The state; ANIRA_MODEL_STATE_FORCE32, which is no state, for NULL.
   * Thread: [main-thread]
   */
  anira_model_config_state(config: number): number {
    const m = this.module
    return m._anira_model_config_state(config) >>> 0
  }

  /**
   * The instance limit anira_model_config_set_max_instances stored, unresolved:
   * ANIRA_INSTANCES_AUTO when never set (the handler resolves it to the pool's thread
   * count when it is prepared; anira_engine_load_info.instances reports the resolved
   * count).
   *
   * @param config The config.
   * @returns The limit, at least 1, or ANIRA_INSTANCES_AUTO; 0 for NULL.
   * Thread: [main-thread]
   */
  anira_model_config_max_instances(config: number): number {
    const m = this.module
    return m._anira_model_config_max_instances(config) >>> 0
  }

  /**
   * The anchor's canonical name as anira_model_config_set_anchor wrote it, unresolved:
   * whether it names a streamed tensor is prepare's question.
   *
   * @param config The config.
   * @returns Object-owned; NULL for the default anchor and for NULL.
   * Thread: [main-thread]
   */
  anira_model_config_anchor(config: number): string | null {
    const m = this.module
    return readCString(m, m._anira_model_config_anchor(config))
  }

  /**
   * The export's name of the tensor in the entry's tensor record
   * (anira_model_config_set_tensor_name, the JSON file's models[].tensors).
   *
   * @param config The config.
   * @param model_index An entry.
   * @param canonical The tensor's canonical name.
   * @returns Object-owned; NULL when the entry binds the tensor positionally, and for a
   *   NULL config or canonical or an index out of range.
   * Thread: [main-thread]
   */
  anira_model_config_tensor_name(
    config: number,
    model_index: number,
    canonical: string
  ): string | null {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      const raw =
        m._anira_model_config_tensor_name(config, model_index, canonical_)
      return readCString(m, raw)
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The axis order of the tensor in the entry's tensor record, with the count protocol of
   * anira_handler_get_latencies.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param canonical The tensor's canonical name.
   * @param count In: the capacity of axes; out: the length of the layout, 0 when the
   *   entry holds the tensor in the spec's order.
   * @param axes Receives the layout as anira_model_config_set_tensor_layout stored it
   *   (per position of the file, a spec axis or ANIRA_AXIS_INSERT); or NULL to ask for the
   *   length.
   * @returns ANIRA_OK, count 0 for a tensor without a layout; ANIRA_INCOMPLETE when axes
   *   is too short (count holds the length, the first capacity entries are written);
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL config or count, an index out of range, or a
   *   NULL or empty canonical. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_model_config_tensor_layout(
    config: number,
    model_index: number,
    canonical: string,
    count: number,
    axes: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = canonical === null ? 0 : m.stringToUTF8OnStack(canonical)
      return checkStatus(
        m,
        m._anira_model_config_tensor_layout(config, model_index, canonical_, count, axes),
        'anira_model_config_tensor_layout'
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The typed record of a known extension kind on one entry (host "model"), set as a
   * struct or as JSON alike: the header of the kind's payload, read within
   * header->struct_size (for "entry", ((const anira_ext_entry*)header)->name). The record
   * and its strings are object-owned, valid until the entry's next set of that kind or
   * the config's destruction.
   *
   * @param config The config.
   * @param model_index An entry.
   * @param kind The extension kind, e.g. "entry".
   * @returns Object-owned; NULL for a kind the entry does not carry, for a kind this
   *   build does not register (stored for prepare to name), and for a NULL config or kind
   *   or an index out of range.
   * Thread: [main-thread]
   */
  anira_model_config_model_ext(
    config: number,
    model_index: number,
    kind: string
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = kind === null ? 0 : m.stringToUTF8OnStack(kind)
      return m._anira_model_config_model_ext(config, model_index, kind_) >>> 0
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The geometry as anira_contract_create_hard or anira_contract_hard_set_geometry stored
   * it; 0, 0, 0 on the legacy contract of a version 2 upgrade until set.
   *
   * @param contract A Hard contract.
   * @returns What the entry writes to block_min, block_max, rate. A failed status throws
   *   an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_geometry(
    contract: number
  ): { block_min: number; block_max: number; rate: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const block_min_ = stackSlot(m)
      const block_max_ = stackSlot(m)
      const rate_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_geometry(contract, block_min_, block_max_, rate_),
        'anira_contract_hard_geometry'
      )
      return {
        block_min: heap(m).HEAPF64[block_min_ >>> 3],
        block_max: heap(m).HEAPF64[block_max_ >>> 3],
        rate: heap(m).HEAPF64[rate_ >>> 3],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The budget as anira_contract_hard_set_budget stored it.
   *
   * @param contract A Hard contract.
   * @returns What the entry writes to kind, explicit_ms. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_budget(contract: number): { kind: number; explicit_ms: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const kind_ = stackSlot(m)
      const explicit_ms_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_budget(contract, kind_, explicit_ms_),
        'anira_contract_hard_budget'
      )
      return {
        kind: heap(m).HEAPU32[kind_ >>> 2],
        explicit_ms: heap(m).HEAPF64[explicit_ms_ >>> 3],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The warmup as anira_contract_hard_set_warmup stored it.
   *
   * @param contract A Hard contract.
   * @returns What the entry writes to mode, iterations. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_warmup(contract: number): { mode: number; iterations: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const mode_ = stackSlot(m)
      const iterations_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_warmup(contract, mode_, iterations_),
        'anira_contract_hard_warmup'
      )
      return {
        mode: heap(m).HEAPU32[mode_ >>> 2],
        iterations: heap(m).HEAPU32[iterations_ >>> 2],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The miss policy as anira_contract_hard_set_on_miss stored it.
   *
   * @param contract A Hard contract.
   * @returns What the entry writes to policy. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_on_miss(contract: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const policy_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_on_miss(contract, policy_),
        'anira_contract_hard_on_miss'
      )
      return heap(m).HEAPU32[policy_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The pair anira_contract_hard_set_miss_fn stored, whatever the policy; a contract file
   * carries none.
   *
   * @param contract A Hard contract.
   * @returns What the entry writes to fn, user_data. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_miss_fn(contract: number): { fn: number; user_data: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const fn_ = stackSlot(m)
      const user_data_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_miss_fn(contract, fn_, user_data_),
        'anira_contract_hard_miss_fn'
      )
      return {
        fn: heap(m).HEAPU32[fn_ >>> 2],
        user_data: heap(m).HEAPU32[user_data_ >>> 2],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The wait ratio as anira_contract_hard_set_wait_ratio stored it.
   *
   * @param contract A Hard contract.
   * @returns What the entry writes to ratio. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_wait_ratio(contract: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const ratio_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_wait_ratio(contract, ratio_),
        'anira_contract_hard_wait_ratio'
      )
      return heap(m).HEAPF64[ratio_ >>> 3]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The number of tensors anira_contract_hard_set_ring_dtype (or a contract file's
   * ring_dtypes) named; every other tensor's ring dtype is ANIRA_DTYPE_F32.
   *
   * @param contract A Hard contract.
   * @returns The count; 0 on an Async contract and for NULL.
   * Thread: [main-thread]
   */
  anira_contract_hard_num_ring_dtypes(contract: number): number {
    const m = this.module
    return m._anira_contract_hard_num_ring_dtypes(contract) >>> 0
  }

  /**
   * One ring dtype the contract names, enumerated by index in bytewise order of the
   * canonical names: what lists what was set, which a lookup by name could not (it would
   * answer ANIRA_DTYPE_F32 for a name never set and for one set to it alike).
   *
   * @param contract A Hard contract.
   * @param index Below anira_contract_hard_num_ring_dtypes.
   * @returns What the entry writes to canonical, dtype. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_ring_dtype(
    contract: number,
    index: number
  ): { canonical: string | null; dtype: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = stackSlot(m)
      const dtype_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_ring_dtype(contract, index, canonical_, dtype_),
        'anira_contract_hard_ring_dtype'
      )
      return {
        canonical: readCString(m, heap(m).HEAPU32[canonical_ >>> 2]),
        dtype: heap(m).HEAPU32[dtype_ >>> 2],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The number of outputs anira_contract_hard_set_latency (or a contract file's
   * latencies) named; every other output keeps the computed figure.
   *
   * @param contract A Hard contract.
   * @returns The count; 0 on an Async contract and for NULL.
   * Thread: [main-thread]
   */
  anira_contract_hard_num_latencies(contract: number): number {
    const m = this.module
    return m._anira_contract_hard_num_latencies(contract) >>> 0
  }

  /**
   * One declared stream latency the contract names, enumerated by index in bytewise order
   * of the canonical names, the shape of anira_contract_hard_ring_dtype.
   *
   * @param contract A Hard contract.
   * @param index Below anira_contract_hard_num_latencies.
   * @returns What the entry writes to canonical, samples. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_contract_hard_latency(
    contract: number,
    index: number
  ): { canonical: string | null; samples: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const canonical_ = stackSlot(m)
      const samples_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_contract_hard_latency(contract, index, canonical_, samples_),
        'anira_contract_hard_latency'
      )
      return {
        canonical: readCString(m, heap(m).HEAPU32[canonical_ >>> 2]),
        samples: heap(m).HEAPU32[samples_ >>> 2],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The edge cost policy anira_contract_set_edge_cost stored.
   *
   * @param contract Either contract kind.
   * @returns The policy; ANIRA_EDGE_COST_PERMISSIVE, the default, for NULL.
   * Thread: [main-thread]
   */
  anira_contract_edge_cost(contract: number): number {
    const m = this.module
    return m._anira_contract_edge_cost(contract) >>> 0
  }

  /**
   * The pool size and the wait strategy as anira_context_config_set_threads (or a context
   * file) stored them.
   *
   * @param config The config.
   * @returns What the entry writes to num_threads, wait. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_threads(config: number): { num_threads: number; wait: number } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const num_threads_ = stackSlot(m)
      const wait_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_context_config_threads(config, num_threads_, wait_),
        'anira_context_config_threads'
      )
      return {
        num_threads: heap(m).HEAPU32[num_threads_ >>> 2],
        wait: heap(m).HEAPU32[wait_ >>> 2],
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Every log field the config stores, in the record anira_context_config_set_log takes:
   * the sink and its user_data, the level, the drain mode and interval, the queue
   * capacity (clamped as the setters clamp it) and the flags.
   *
   * @param config The config.
   * @param out Receives the log settings, written within its struct_size (set it before
   *   the call), which it keeps; abi_version receives this build's ANIRA_ABI_VERSION.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument or a struct_size
   *   below the three leading slots (struct_size, abi_version, user_data). A failed status
   *   throws an AniraError.
   * Thread: [main-thread]
   */
  anira_context_config_log(config: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_context_config_log(config, out),
      'anira_context_config_log'
    )
  }

  /**
   * Creates a context over this copy's core: reconciles the config into the core (see the
   * file comment), registers the config's log sink so that it receives every record from
   * here on, applies the config's ANIRA_LOG_FLAG_* switches while the context lives, and
   * probes the capabilities. A device block on the config is refused in this pre-release.
   * On WebAssembly the pool's threads are Wasm Workers
   * (anira_em_context_config_set_worker_module names the module they import);
   * ANIRA_WAIT_BLOCKING and ANIRA_LOG_DRAIN_THREAD are coerced to SPIN_BACKOFF and MANUAL
   * with a warning.
   *
   * @param config The context config; copied, destroyable right after.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_context_create(config: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_context_create(config, out_, err),
        'anira_context_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Drops the user's reference: flushes the real-time log queue when this was the last
   * user of the core, unregisters the context's log sink and waits for its in-flight
   * calls, withdraws its ANIRA_LOG_FLAG_* switches, and invalidates the handle for the
   * caller whatever the internal count (a handler that added a reference keeps the memory
   * alive). Joins nothing. A destroy issued from inside the context's own sink is refused
   * with one Error record and nothing happens. A plugin must destroy its handlers, then
   * its contexts, before its host unloads it (its instance teardown or module-exit entry
   * point), and never from its own static destructors or DllMain, which run inside the
   * unload under the loader lock.
   *
   * @param context The handle; NULL is a no-op.
   * Thread: [main-thread & !loader-lock]
   */
  anira_context_destroy(context: number): void {
    const m = this.module
    m._anira_context_destroy(context)
  }

  /**
   * Re-runs the probe and refreshes the capabilities the handle reports: the compiled-in
   * engines on the providers their runtimes report usable now
   * (anira_capabilities_backends), one host edge per backend row, the host domain.
   * Nothing is cached in this pre-release, so force changes nothing.
   *
   * @param context The handle.
   * @param force Nonzero re-runs every rung even where a cached answer exists.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL context. A failed
   *   status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_context_probe(context: number, force: boolean): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_context_probe(context, force ? 1 : 0, err),
        'anira_context_probe',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The capabilities the last probe established; context-owned, valid while the context
   * is, replaced in place by anira_context_probe.
   *
   * @param context The handle.
   * @returns The capabilities, or NULL for a NULL context.
   * Thread: [thread-safe]
   */
  anira_context_capabilities(context: number): number {
    const m = this.module
    return m._anira_context_capabilities(context) >>> 0
  }

  /**
   * The backends that are compiled in and usable here, one record per (engine, provider)
   * the build and its runtimes report: the enum's providers by provider, any other as
   * ANIRA_PROVIDER_CUSTOM with provider_id in the runtime's own words (an ONNX Runtime
   * execution provider by its name); a custom engine's rows are its pipeline's
   * (anira_pipeline_capabilities_backends), since an engine belongs to a pipeline, not to
   * the context. Stride-explicit enumeration: min(element_size, the library's record
   * size) bytes are written per element. The strings the rows point to (provider_id) are
   * the context's, valid until its next anira_context_probe replaces them or the context
   * is destroyed: copy what must outlive a probe.
   *
   * @param capabilities The capabilities.
   * @param element_size sizeof(anira_backend_id) of the caller's header, the stride of
   *   out.
   * @param count In: the capacity of out in elements; out: the number of backends.
   * @param out Receives the rows at the caller's stride, or NULL to ask for the count
   *   only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or
   *   ANIRA_ERROR_INVALID_ARGUMENT for NULL capabilities or count or an element_size below
   *   the struct_size slot. A failed status throws an AniraError.
   * Thread: [thread-safe]
   */
  anira_capabilities_backends(
    capabilities: number,
    element_size: number,
    count: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_capabilities_backends(capabilities, element_size, count, out),
      'anira_capabilities_backends'
    )
  }

  /**
   * The memory domains a tensor may live in here; ANIRA_DOMAIN_HOST alone in this
   * pre-release.
   *
   * @param capabilities The capabilities.
   * @param count In: the capacity of out; out: the number of domains.
   * @param out Receives the domains, or NULL to ask for the count only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or
   *   ANIRA_ERROR_INVALID_ARGUMENT for NULL capabilities or count. A failed status throws
   *   an AniraError.
   * Thread: [thread-safe]
   */
  anira_capabilities_domains(capabilities: number, count: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_capabilities_domains(capabilities, count, out),
      'anira_capabilities_domains'
    )
  }

  /**
   * The extension kinds this build understands, beside the probed edges and the enabled
   * backends; the same list as anira_registered_ext_kinds.
   *
   * @param capabilities The capabilities.
   * @param count In: the capacity of out; out: the number of kinds.
   * @param out Receives the kind names (static storage), or NULL to ask for the count
   *   only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or
   *   ANIRA_ERROR_INVALID_ARGUMENT for NULL capabilities or count. A failed status throws
   *   an AniraError.
   * Thread: [thread-safe]
   */
  anira_capabilities_ext_kinds(capabilities: number, count: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_capabilities_ext_kinds(capabilities, count, out),
      'anira_capabilities_ext_kinds'
    )
  }

  /**
   * Every row of the edge registry, available or not; one host edge to each backend row
   * of this pre-release (zero-copy to a CPU provider, a host copy to any other).
   * Stride-explicit enumeration. The strings the rows point to (to_provider_id, reason)
   * are the context's, valid until its next anira_context_probe replaces them or the
   * context is destroyed.
   *
   * @param capabilities The capabilities.
   * @param element_size sizeof(anira_edge_info) of the caller's header, the stride of
   *   out.
   * @param count In: the capacity of out in elements; out: the number of edges.
   * @param out Receives the rows at the caller's stride, or NULL to ask for the count
   *   only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or
   *   ANIRA_ERROR_INVALID_ARGUMENT for NULL capabilities or count or an element_size below
   *   the struct_size slot. A failed status throws an AniraError.
   * Thread: [thread-safe]
   */
  anira_capabilities_edges(
    capabilities: number,
    element_size: number,
    count: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_capabilities_edges(capabilities, element_size, count, out),
      'anira_capabilities_edges'
    )
  }

  /**
   * One row of the edge registry, by domain and backend: the engine on the provider, a
   * custom provider (ANIRA_PROVIDER_CUSTOM) by its provider_id (read when the caller's
   * record has the slot). A custom engine (ANIRA_ENGINE_CUSTOM) has no row here: its rows
   * are its pipeline's (anira_pipeline_capabilities_edge).
   *
   * @param capabilities The capabilities.
   * @param from The tensor's domain.
   * @param to The backend; read within its struct_size.
   * @param out Receives the row, read and written within its struct_size (set it before
   *   the call).
   * @returns ANIRA_OK with the row; ANIRA_ERROR_EDGE_UNREACHABLE when the registry has no
   *   row for the pair (out untouched); ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument or
   *   a struct_size below the fixed fields. A failed status throws an AniraError.
   * Thread: [thread-safe]
   */
  anira_capabilities_edge(
    capabilities: number,
    from: number,
    to: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_capabilities_edge(capabilities, from, to, out),
      'anira_capabilities_edge'
    )
  }

  /**
   * What this build compiled in, without a context: every engine with an adapter, on
   * ANIRA_PROVIDER_CPU, in anira_engine order. Whether a backend is usable here is
   * anira_capabilities_backends on a probed context.
   *
   * @param element_size sizeof(anira_backend_id) of the caller's header, the stride of
   *   out.
   * @param count In: the capacity of out in elements; out: the number of backends.
   * @param out Receives the rows at the caller's stride, or NULL to ask for the count
   *   only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, or
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL count or an element_size below the
   *   struct_size slot. A failed status throws an AniraError.
   * Thread: [thread-safe]
   */
  anira_enabled_engines(element_size: number, count: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_enabled_engines(element_size, count, out),
      'anira_enabled_engines'
    )
  }

  /**
   * The size in bytes of a tensor's byte image under the edges this context probed
   * (section 7): the dense encoding, num_elements times the element size, in this
   * pre-release.
   *
   * @param context The handle.
   * @param num_elements Elements of the tensor.
   * @param dtype The element type.
   * @returns The byte count, 0 for a NULL context or a dtype without a size.
   * Thread: [thread-safe]
   */
  anira_context_byte_image_bytes(
    context: number,
    num_elements: bigint,
    dtype: number
  ): bigint {
    const m = this.module
    const raw =
      m._anira_context_byte_image_bytes(context, num_elements, dtype)
    return BigInt.asUintN(64, raw)
  }

  /**
   * The steady clock in milliseconds, for deadlines and submit timestamps; the same clock
   * as anira_now_ns.
   *
   * @returns Milliseconds since an unspecified steady epoch.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_now_ms(): number {
    const m = this.module
    return m._anira_now_ms()
  }

  /**
   * The steady clock in nanoseconds; the one allowlisted 64-bit return on an
   * ANIRA_NONBLOCKING declaration.
   *
   * @returns Nanoseconds since an unspecified steady epoch.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_now_ns(): bigint {
    const m = this.module
    return BigInt.asUintN(64, m._anira_now_ns())
  }

  /**
   * Stops and joins the core's inference threads and its log drain thread and flushes the
   * queue, for a static embedding that is about to be unloaded (called from clap_deinit
   * or ExitDll). Idempotent, never creates the core, and effective only when no context
   * and no handler exist in this copy: otherwise nothing happens, so one client of a
   * shared library cannot silence another's sessions. A plugin must destroy its handlers,
   * then its contexts, before its host unloads it (its instance teardown or module-exit
   * entry point), and never from its own static destructors or DllMain, which run inside
   * the unload under the loader lock. On ELF and Mach-O the library-unload hook is the
   * backstop for a host that unloads with a handler alive: it joins an idle pool; with an
   * inference in flight it tells the threads to stop, waits up to two seconds for them to
   * leave their loops and for the inferences, and when one is still running (it can never
   * finish while the unload holds the loader lock) writes 'anira: the library is being
   * unloaded while an inference is still running (N in flight, M inference threads not
   * stopped); ...' to stderr and aborts instead of hanging. At process exit it waits
   * without a bound. Windows has no such hook: destroy the handlers and call this from
   * the module-exit entry point.
   *
   * @returns ANIRA_OK (also without a core); ANIRA_ERROR_INVALID_STATE while a context or
   *   a handler lives. A failed status throws an AniraError.
   * Thread: [main-thread & !loader-lock]
   */
  anira_shutdown(): number {
    const m = this.module
    return checkStatus(m, m._anira_shutdown(), 'anira_shutdown')
  }

  /**
   * Frees the core when nothing uses it: no context, no handler, no pool thread, no
   * user-driven inference thread and, on WebAssembly, no inference loop still running.
   * Never blocks; the unload hook's call, after it stopped the pool (see anira_shutdown
   * for an unload with an inference in flight).
   *
   * @returns Nonzero when the core was freed.
   * Thread: [main-thread]
   */
  anira_release_core_if_idle(): boolean {
    const m = this.module
    return m._anira_release_core_if_idle() !== 0
  }

  /**
   * Whether this copy of anira holds a core.
   *
   * @returns Nonzero while the core exists.
   * Thread: [thread-safe]
   */
  anira_has_core(): boolean {
    const m = this.module
    return m._anira_has_core() !== 0
  }

  /**
   * Creates an inference thread bound to the core's queue. Nothing runs until
   * anira_inference_thread_start or anira_inference_thread_run_loop.
   *
   * @param context The context whose core the thread serves; the thread waits with the
   *   core's reconciled wait strategy, the first user's.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_inference_thread_create(context: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_inference_thread_create(context, out_, err),
        'anira_inference_thread_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Runs the inference loop on the calling thread until anira_inference_thread_stop; sets
   * has_exited on return. The caller's thread is an inference thread while it is inside.
   *
   * @param thread The object.
   * Thread: [inference-thread]
   */
  anira_inference_thread_run_loop(thread: number): void {
    const m = this.module
    m._anira_inference_thread_run_loop(thread)
  }

  /**
   * One step of the loop: dequeues and dispatches one inference if any is pending. The
   * dequeue is allocation-free; the engine call and host callbacks may allocate and
   * block.
   *
   * @param thread The object.
   * @returns Nonzero when an inference was dispatched.
   * Thread: [inference-thread]
   */
  anira_inference_thread_execute(thread: number): boolean {
    const m = this.module
    return m._anira_inference_thread_execute(thread) !== 0
  }

  /**
   * Spawns a thread that runs the loop: natively an OS thread at the inference-thread
   * scheduling class (real-time where granted), on WebAssembly a Wasm Worker on the
   * module's memory, which imports the context's worker module
   * (anira_em_context_config_set_worker_module) first. A second start while the loop
   * runs, and an operating system or browser that refuses the thread, are returned, never
   * silent: is_running stays 0 then.
   *
   * @param thread The object.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL thread;
   *   ANIRA_ERROR_INVALID_STATE when it is already running; ANIRA_ERROR_OUT_OF_MEMORY when
   *   the operating system refused to create the thread (a thread limit or its stack). A
   *   failed status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_inference_thread_start(thread: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_inference_thread_start(thread, err),
        'anira_inference_thread_start',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Requests the loop to exit and joins the thread anira_inference_thread_start started
   * (WebAssembly: waits for its Wasm Worker to leave the loop, then terminates it; a
   * Worker still booting, its module's import or its engines' init in flight, is never
   * terminated there: the stop cancels its boot and returns, the Worker finishes the boot
   * without entering the loop, and a main-thread timer frees its region once it is done,
   * terminating one that has not finished within 10 s and leaving its region allocated).
   * A loop the host runs itself (anira_inference_thread_run_loop) is asked to return
   * only; poll anira_inference_thread_has_exited.
   *
   * @param thread The object.
   * Thread: [main-thread]
   */
  anira_inference_thread_stop(thread: number): void {
    const m = this.module
    m._anira_inference_thread_stop(thread)
  }

  /**
   * Nonzero once a run of the loop returned (a shared-memory atomic, readable from every
   * WebAssembly instance); required before anira_inference_thread_destroy on WebAssembly.
   *
   * @param thread The object.
   * @returns Nonzero after the loop returned; 0 before it ran and while it runs.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_inference_thread_has_exited(thread: number): boolean {
    const m = this.module
    return m._anira_inference_thread_has_exited(thread) !== 0
  }

  /**
   * Nonzero once a stop was requested.
   *
   * @param thread The object.
   * @returns The stop flag.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_inference_thread_should_exit(thread: number): boolean {
    const m = this.module
    return m._anira_inference_thread_should_exit(thread) !== 0
  }

  /**
   * Nonzero while the loop is running (native: the OS thread exists and has not returned;
   * WebAssembly: between start and stop, the Wasm Worker's import included).
   *
   * @param thread The object.
   * @returns The running flag.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_inference_thread_is_running(thread: number): boolean {
    const m = this.module
    return m._anira_inference_thread_is_running(thread) !== 0
  }

  /**
   * Destroys a stopped object. Native: stops and joins first if the caller did not.
   * WebAssembly: refused with one Error record unless has_exited (the Worker may still be
   * inside the loop).
   *
   * @param thread The object; NULL is a no-op.
   * Thread: [main-thread]
   */
  anira_inference_thread_destroy(thread: number): void {
    const m = this.module
    m._anira_inference_thread_destroy(thread)
  }

  /**
   * The size of the default inference thread pool in this copy of anira: the pool exists
   * while a handler does, so 0 before the first handler and when the context brought its
   * own threads. User-driven threads are not counted.
   *
   * @returns The pool size.
   * Thread: [thread-safe]
   */
  anira_num_inference_threads(): number {
    const m = this.module
    return m._anira_num_inference_threads() >>> 0
  }

  /**
   * The element type the ring stores: the ring dtype the host declared for the slot on
   * the Hard contract (anira_contract_hard_set_ring_dtype), ANIRA_DTYPE_F32 when nothing
   * was declared. It is never inferred from the tensor spec's dtype, which may differ
   * when the processor converts; every data accessor states the dtype it believes it is
   * reading or writing.
   *
   * @param ring A ring of the processor context, or NULL.
   * @returns The ring's dtype; 0, which is no dtype, for a NULL ring.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_dtype(ring: number): number {
    const m = this.module
    return m._anira_ring_dtype(ring) >>> 0
  }

  /**
   * The channel count of the ring: the extent of the Channel axis of its Streamed tensor
   * spec, 1 without one.
   *
   * @param ring A ring of the processor context, or NULL.
   * @returns The number of channels; 0 for a NULL ring.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_num_channels(ring: number): number {
    const m = this.module
    return m._anira_ring_num_channels(ring) >>> 0
  }

  /**
   * The unread elements of one channel.
   *
   * @param ring A ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @returns The count; 0 for a NULL ring, and 0 with ANIRA_ERROR_INVALID_ARGUMENT
   *   recorded in anira_handler_rt_error for a channel out of range.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_available(ring: number, channel: number): number {
    const m = this.module
    return m._anira_ring_available(ring, channel) >>> 0
  }

  /**
   * The consumed elements of one channel the ring still retains as history: what
   * anira_ring_peek_past_block can address. 0 right after prepare or reset, growing as
   * elements are popped, bounded by the ring's capacity.
   *
   * @param ring A ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @returns The count; 0 for a NULL ring, and 0 with ANIRA_ERROR_INVALID_ARGUMENT
   *   recorded in anira_handler_rt_error for a channel out of range.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_available_past(ring: number, channel: number): number {
    const m = this.module
    return m._anira_ring_available_past(ring, channel) >>> 0
  }

  /**
   * Pops count elements of one channel into out, oldest first. Exactly count elements are
   * written: those beyond the available ones are value-initialised (zero).
   *
   * @param ring An input ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @param out Room for count elements of dtype, packed.
   * @param dtype The dtype of the memory at out; it must be anira_ring_dtype(ring).
   * @param count Elements to pop.
   * @returns count; 0 with nothing popped and nothing written for a NULL ring, a NULL out
   *   with a count above 0 (ANIRA_ERROR_INVALID_ARGUMENT recorded), a channel out of range
   *   (ANIRA_ERROR_INVALID_ARGUMENT recorded) or a dtype that is not the ring's
   *   (ANIRA_ERROR_CONFIG recorded in anira_handler_rt_error with one latched record;
   *   nothing converts).
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_pop_block(
    ring: number,
    channel: number,
    out: number,
    dtype: number,
    count: number
  ): number {
    const m = this.module
    return m._anira_ring_pop_block(ring, channel, out, dtype, count) >>> 0
  }

  /**
   * Copies the count most recently consumed elements of one channel into out without
   * popping, oldest first, so that out[count - 1] is the element popped last: the
   * receptive field of a model whose window is longer than its hop. The range is not
   * checked against anira_ring_available_past: history the ring never held reads as zero
   * after prepare or reset.
   *
   * @param ring An input ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @param out Room for count elements of dtype, packed.
   * @param dtype The dtype of the memory at out; it must be anira_ring_dtype(ring).
   * @param count Elements of history to copy.
   * @returns count; 0 with nothing written under the refusals of anira_ring_pop_block.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_peek_past_block(
    ring: number,
    channel: number,
    out: number,
    dtype: number,
    count: number
  ): number {
    const m = this.module
    return m._anira_ring_peek_past_block(ring, channel, out, dtype, count) >>> 0
  }

  /**
   * Pushes count elements into one channel. A full ring overwrites its oldest elements;
   * anira sizes an output ring so that one hop per submitted chunk always fits.
   *
   * @param ring An output ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @param in_ count elements of dtype, packed.
   * @param dtype The dtype of the memory at in; it must be anira_ring_dtype(ring).
   * @param count Elements to push.
   * @returns count; 0 with nothing pushed for a NULL ring, a NULL in with a count above 0
   *   (ANIRA_ERROR_INVALID_ARGUMENT recorded), a channel out of range
   *   (ANIRA_ERROR_INVALID_ARGUMENT recorded) or a dtype that is not the ring's
   *   (ANIRA_ERROR_CONFIG recorded in anira_handler_rt_error with one latched record;
   *   nothing converts).
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_push_block(
    ring: number,
    channel: number,
    in_: number,
    dtype: number,
    count: number
  ): number {
    const m = this.module
    return m._anira_ring_push_block(ring, channel, in_, dtype, count) >>> 0
  }

  /**
   * Pushes count copies of one element into one channel.
   *
   * @param ring An output ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @param value One element of dtype.
   * @param dtype The dtype of the element at value; it must be anira_ring_dtype(ring).
   * @param count Copies to push.
   * @returns count; 0 with nothing pushed under the refusals of anira_ring_push_block (a
   *   NULL value is the NULL in).
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_push_fill(
    ring: number,
    channel: number,
    value: number,
    dtype: number,
    count: number
  ): number {
    const m = this.module
    return m._anira_ring_push_fill(ring, channel, value, dtype, count) >>> 0
  }

  /**
   * Drops up to count unread elements of one channel; they become history like popped
   * ones. No element crosses the call, so it takes no dtype and has no dtype refusal.
   *
   * @param ring An input ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @param count Unread elements to drop.
   * @returns The elements dropped, at most the available ones; 0 for a NULL ring, and 0
   *   with ANIRA_ERROR_INVALID_ARGUMENT recorded in anira_handler_rt_error for a channel
   *   out of range.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_discard(ring: number, channel: number, count: number): number {
    const m = this.module
    return m._anira_ring_discard(ring, channel, count) >>> 0
  }

  /**
   * Pops num_batches overlapping windows of one channel: window b lands at out + offset +
   * b * (num_new + num_old) and holds the num_old most recently consumed elements
   * followed by num_new freshly popped ones, the batched sliding-window layout a model
   * with an input shape [num_batches, ..., num_new + num_old] expects. It pops
   * num_batches * num_new elements in all; with num_batches 1 and offset 0 it is
   * anira_ring_peek_past_block followed by anira_ring_pop_block.
   *
   * @param ring An input ring of the processor context, or NULL.
   * @param channel The channel, below anira_ring_num_channels.
   * @param out The base of the destination, in elements of dtype: room for offset +
   *   num_batches * (num_new + num_old) elements.
   * @param dtype The dtype of the memory at out; it must be anira_ring_dtype(ring).
   * @param num_new Freshly popped elements per window.
   * @param num_old Elements of history ahead of them per window.
   * @param offset Elements to skip at out before the first window.
   * @param num_batches The number of windows.
   * @returns The elements written, num_batches * (num_new + num_old); 0 with nothing
   *   popped and nothing written under the refusals of anira_ring_pop_block.
   * Thread: [driver-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_ring_pop_windows(
    ring: number,
    channel: number,
    out: number,
    dtype: number,
    num_new: number,
    num_old: number,
    offset: number,
    num_batches: number
  ): number {
    const m = this.module
    const raw =
      m._anira_ring_pop_windows(
        ring,
        channel,
        out,
        dtype,
        num_new,
        num_old,
        offset,
        num_batches
      )
    return raw >>> 0
  }

  /**
   * Fills out with what the tensor of an input slot is: the anira_role of its spec, the
   * same answer in all four phases. It is the first question of a processor about a slot
   * it does not know from its model config: a Streamed tensor has a ring in pre_process
   * (anira_processor_input_ring) and a model end, a Static and a State tensor have their
   * model end only (anira_processor_input_tensor). Every context accessor answers the
   * same way: a status, an out-parameter filled on ANIRA_OK and reset on anything else,
   * and every status but ANIRA_OK recorded in anira_handler_rt_error with one latched
   * record per kind naming the entry, the slot and the phase, since asking for what a
   * slot does not have is the processor's bug and not an answer. A NULL ctx and a ctx
   * without a frame name no handler to record into.
   *
   * @param ctx The context the phase callback received.
   * @param slot The tensor's position in the model config's input list, below num_inputs.
   * @returns What the entry writes to out. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_input_role(ctx: number, slot: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_processor_input_role(ctx, slot, out_),
        'anira_processor_input_role'
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills out with what the tensor of an output slot is: the anira_role of its spec, the
   * same answer in all four phases. A Streamed tensor has a ring in post_process
   * (anira_processor_output_ring) and a model end, a Static and a State tensor have their
   * model end only (anira_processor_output_tensor).
   *
   * @param ctx The context the phase callback received.
   * @param slot The tensor's position in the model config's output list, below
   *   num_outputs.
   * @returns What the entry writes to out. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_output_role(ctx: number, slot: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_processor_output_role(ctx, slot, out_),
        'anira_processor_output_role'
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills out with the ring of an input slot: the ring of a Streamed tensor in
   * ANIRA_PHASE_PRE_PROCESS, the one phase that moves the input rings. The pointer is
   * valid until the callback returns, and the processor advances the ring through the
   * anira_ring accessors. In every other case the slot has no ring here, and asking for
   * one is the processor's bug: a tensor of another role has none (ask
   * anira_processor_input_role first, or know the slot from the model config), and no
   * input ring is handed out in another phase than pre_process.
   *
   * @param ctx The context the phase callback received.
   * @param slot The tensor's position in the model config's input list, below num_inputs.
   * @returns What the entry writes to out. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_input_ring(ctx: number, slot: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_processor_input_ring(ctx, slot, out_),
        'anira_processor_input_ring'
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills out with the ring of an output slot: the ring of a Streamed tensor in
   * ANIRA_PHASE_POST_PROCESS, the one phase that moves the output rings; valid until the
   * callback returns. A tensor of another role has no ring, and no output ring is handed
   * out in another phase: asking is the processor's bug, as for
   * anira_processor_input_ring.
   *
   * @param ctx The context the phase callback received.
   * @param slot The tensor's position in the model config's output list, below
   *   num_outputs.
   * @returns What the entry writes to out. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_output_ring(ctx: number, slot: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_processor_output_ring(ctx, slot, out_),
        'anira_processor_output_ring'
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills out with the model end of an input slot: the tensor the engine binds, in the
   * spec's shape, memory owned by anira in the slot's host domain (readable from the
   * descriptor's domain field), for the processor to write. Available in
   * ANIRA_PHASE_PRE_PROCESS for a Streamed and a Static input (every Static input is
   * already materialised there; a State input has no host end and is fed after
   * pre_process, so there is no model end to hand out), and in
   * ANIRA_PHASE_BEFORE_INFERENCE for a tensor of every role, State included: there every
   * State input is already fed from the handler's state value, a processor may read or
   * alter it, and what it leaves is what the engine gets. Asking in after_inference or
   * post_process, which expose the model's outputs, or for a State input in pre_process,
   * is the processor's bug. The descriptor is built by this call over the memory the
   * tensor has right now; an engine may swap that memory between two inferences, so a
   * processor asks again in every callback and keeps neither the descriptor nor its data
   * pointer across calls.
   *
   * @param ctx The context the phase callback received.
   * @param slot The tensor's position in the model config's input list, below num_inputs.
   * @param out The caller's record, filled with a borrowed descriptor (release NULL); an
   *   all-zero record, which reads as dtype 0, whenever the status is not ANIRA_OK.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_STATE, recorded in anira_handler_rt_error with
   *   one latched record naming the entry, the slot and the phase, in a phase that does not
   *   expose the model's inputs (after_inference, post_process, reset) and for a State
   *   input in pre_process. ANIRA_ERROR_INVALID_ARGUMENT for a NULL ctx, a ctx without a
   *   frame, a NULL out, or a slot at or beyond num_inputs; the NULL out and the slot out
   *   of range are recorded too. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_input_tensor(ctx: number, slot: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_processor_input_tensor(ctx, slot, out),
      'anira_processor_input_tensor'
    )
  }

  /**
   * Fills out with the model end of an output slot: the tensor the engine wrote, in the
   * spec's shape, memory owned by anira in the slot's host domain. Available in
   * ANIRA_PHASE_AFTER_INFERENCE for a tensor of every role, State included (there a State
   * output is not yet captured: what the processor leaves in it is what the next
   * inference is fed), and in ANIRA_PHASE_POST_PROCESS for a Streamed and a Static output
   * (a State output was captured ahead of post_process and has no host end). Asking in
   * pre_process or before_inference, or for a State output in post_process, is the
   * processor's bug. A chunk that completed without an inference (dropped, failed) reads
   * as zeros. The descriptor is built by this call and is not kept across calls, as for
   * anira_processor_input_tensor.
   *
   * @param ctx The context the phase callback received.
   * @param slot The tensor's position in the model config's output list, below
   *   num_outputs.
   * @param out The caller's record, filled with a borrowed descriptor (release NULL); an
   *   all-zero record, which reads as dtype 0, whenever the status is not ANIRA_OK.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_STATE, recorded, in a phase that does not
   *   expose the model's outputs (pre_process, before_inference, reset) and for a State
   *   output in post_process; ANIRA_ERROR_INVALID_ARGUMENT under the refusals of
   *   anira_processor_input_tensor, against num_outputs. A failed status throws an
   *   AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_output_tensor(ctx: number, slot: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_processor_output_tensor(ctx, slot, out),
      'anira_processor_output_tensor'
    )
  }

  /**
   * Fills the pair of the engine of the plan the chunk was submitted under
   * (anira_engine's pair rule): the value anira_processor_ctx.engine carries, and the id
   * anira_plan_info.engine_id carries for that plan, the reverse-URI id of a custom
   * engine beside ANIRA_ENGINE_CUSTOM, NULL for a built-in engine. The id out-parameter
   * may be NULL when the caller wants the value alone. The same answer in every phase of
   * a chunk. The string is anira's, valid for the duration of the callback.
   *
   * @param ctx The context the phase callback received.
   * @returns What the entry writes to engine, engine_id. A failed status throws an
   *   AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_engine(ctx: number): { engine: number; engine_id: string | null } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_ = stackSlot(m)
      const engine_id_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_processor_engine(ctx, engine_, engine_id_),
        'anira_processor_engine'
      )
      return {
        engine: heap(m).HEAPU32[engine_ >>> 2],
        engine_id: readCString(m, heap(m).HEAPU32[engine_id_ >>> 2]),
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The provider twin of anira_processor_engine: the pair of the provider of the plan the
   * chunk was submitted under (anira_plan_info.provider and provider_id), the name of a
   * custom provider beside ANIRA_PROVIDER_CUSTOM, NULL for a provider of the enum.
   *
   * @param ctx The context the phase callback received.
   * @returns What the entry writes to provider, provider_id. A failed status throws an
   *   AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_provider(
    ctx: number
  ): { provider: number; provider_id: string | null } {
    const m = this.module
    const sp = m.stackSave()
    try {
      const provider_ = stackSlot(m)
      const provider_id_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_processor_provider(ctx, provider_, provider_id_),
        'anira_processor_provider'
      )
      return {
        provider: heap(m).HEAPU32[provider_ >>> 2],
        provider_id: readCString(m, heap(m).HEAPU32[provider_id_ >>> 2]),
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The default pre_process, for a processor that wants it ("call super"): for every
   * Streamed input it pops one hop per channel from the ring into the model tensor, which
   * it reads as packed row-major memory: channel c of a tensor of n elements per channel
   * starts at element c * n. Where n is the hop, that is the plain pop. Where n is above
   * the hop, the receptive field of a sliding-window model, the n minus hop elements at
   * the head of the channel are the ring's history (anira_ring_peek_past_block) and the
   * hop is popped behind them. It never touches a tensor of another role. Real-time by
   * construction: no allocation, no lock, no system call. anira runs it itself, once per
   * chunk, when the processor's pre_process slot is NULL (or the pipeline has no
   * processor).
   *
   * @param ctx The context the processor's pre_process received.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL ctx or a ctx of another
   *   phase than ANIRA_PHASE_PRE_PROCESS; ANIRA_ERROR_CONFIG when the ring dtype of a
   *   Streamed input is not its tensor's dtype, with that tensor untouched and the others
   *   filled (nothing converts: a processor that declared another ring dtype pops that ring
   *   itself). The status is returned, never recorded by the default itself: a processor
   *   that returns it fails the chunk with it. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_default_pre_process(ctx: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_processor_default_pre_process(ctx),
      'anira_processor_default_pre_process'
    )
  }

  /**
   * The default post_process, for a processor that wants it ("call super"): for every
   * Streamed output it pushes one hop per channel from the model tensor, read as packed
   * row-major memory, into the ring: channel c from element c * hop. It never touches a
   * tensor of another role. Real-time by construction. anira runs it itself, once per
   * chunk, when the processor's post_process slot is NULL (or the pipeline has no
   * processor).
   *
   * @param ctx The context the processor's post_process received.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL ctx or a ctx of another
   *   phase than ANIRA_PHASE_POST_PROCESS; ANIRA_ERROR_CONFIG when the ring dtype of a
   *   Streamed output is not its tensor's dtype, with that ring untouched and the others
   *   pushed. The status is returned, never recorded by the default itself: a processor
   *   that returns it fails the chunk with it. A failed status throws an AniraError.
   * Thread: [driver-thread | inference-thread] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_processor_default_post_process(ctx: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_processor_default_post_process(ctx),
      'anira_processor_default_post_process'
    )
  }

  /**
   * Creates an empty pipeline. A pipeline holds exactly one inference stage
   * (anira_pipeline_add_inference) and at most one processor, the custom pre- and
   * post-processing (anira_pipeline_add_processor). Value-like: copied by
   * anira_handler_create, destroyable right after; the copy shares the carrier of the
   * processor and the custom engines added to it.
   *
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_pipeline_create(): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(m, m._anira_pipeline_create(out_, err), 'anira_pipeline_create', err)
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Adds the inference stage: the variants and the candidate backends. One plan is
   * compiled per model entry and candidate that names its engine and a provider it
   * accepts (anira_handler_prepare; the plan table is in entry order and, within an
   * entry, candidate order, an equal provider once), so one entry under two providers of
   * its engine is two plans, which anira_handler_set_plan switches between; a candidate
   * without an entry is not a plan and not an error; a pipeline whose candidates match no
   * entry is ANIRA_ERROR_CONFIG at anira_handler_create. The structure of the variant
   * (axes, roles, windows, layouts, the engines of the named candidates against this
   * build, the extensions on the model and its specs) is checked at anira_handler_create;
   * the contract rules at prepare. A custom engine is part of this processor and never a
   * processor of its own: it is one more implementation a candidate's engine_id resolves
   * to, added to the pipeline with anira_pipeline_add_engine (before or after this call),
   * and its call runs in ANIRA_PHASE_INFERENCE like a built-in engine's; an entry whose
   * id no engine of the pipeline serves is ANIRA_ERROR_NOT_SUPPORTED at
   * anira_handler_create: every custom id, anira.v2.custom included, needs an engine on
   * the pipeline.
   *
   * @param pipeline The pipeline.
   * @param variants The model configurations the processor may run, copied; exactly one
   *   in this pre-release.
   * @param num_variants The number of variants; 1 in this pre-release.
   * @param candidates The candidate backends, copied, or NULL (with num_candidates 0) for
   *   the default set: every engine this build carries on ANIRA_PROVIDER_CPU, every custom
   *   engine an entry names (ANIRA_ENGINE_CUSTOM with its id), and every provider a model
   *   entry of the variant is pinned to, on that entry's engine. Under it every entry is
   *   one plan: a pinned entry on its pin, a neutral one on its engine's home provider,
   *   ANIRA_PROVIDER_CPU for a built-in engine and for a custom engine that serves it (no
   *   providers list, or "cpu" in it), else the custom engine's first listed provider.
   *   Under the default set a model entry for an engine this build lacks is skipped, not
   *   refused, and a plan whose provider its engine cannot use here (a built-in engine's
   *   provider the context's capabilities lack, a custom engine's declared provider its
   *   query cleared) is dropped at anira_handler_create and anira_handler_prepare, with its
   *   entry when no plan of the entry is left; a variant left with no plan is
   *   ANIRA_ERROR_CONFIG. Name a candidate to have it checked: a named candidate the engine
   *   cannot serve here is refused. A candidate names a built-in engine by its value or a
   *   custom engine as ANIRA_ENGINE_CUSTOM with its id; ANIRA_ENGINE_NONE names no engine.
   *   A candidate's provider (a value of the enum, or ANIRA_PROVIDER_CUSTOM with a
   *   provider_id for a custom one; never ANIRA_PROVIDER_NONE) is what its plans run on: a
   *   model entry without a pin runs on it, a pinned entry on its pin alone (a candidate
   *   naming another provider skips it); its syntax is checked here (a known value on both
   *   axes and the pair rule of anira_engine and anira_provider), whether the engine serves
   *   it at anira_handler_create.
   * @param num_candidates The number of candidates; 0 with a NULL list.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL pipeline, a NULL or empty
   *   variant list, a NULL entry in it, a NULL candidates with num_candidates above 0, a
   *   candidate whose struct_size is below the record's head or unlike the first
   *   candidate's (the array's one stride), or a candidate whose engine is neither a
   *   built-in engine nor ANIRA_ENGINE_CUSTOM, whose provider is ANIRA_PROVIDER_NONE or no
   *   value of anira_provider, or whose pair breaks the pair rule of anira_engine and
   *   anira_provider (an id beside a value other than CUSTOM, CUSTOM without an id, an
   *   empty provider_id, a custom engine id without a '.'); ANIRA_ERROR_CONFIG for a second
   *   inference stage; ANIRA_ERROR_NOT_SUPPORTED for more than one variant. A failed status
   *   throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_pipeline_add_inference(
    pipeline: number,
    variants: number,
    num_variants: number,
    candidates: number,
    num_candidates: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_pipeline_add_inference(
          pipeline,
          variants,
          num_variants,
          candidates,
          num_candidates,
          err
        ),
        'anira_pipeline_add_inference',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Sets the pipeline's one processor: the descriptor is copied into a refcounted
   * carrier, which the pipeline and every handler created from it share, and release
   * fires exactly once, when the last of them is destroyed, after every unprepare. One
   * registration, prepared per handler: every anira_handler_prepare calls the processor's
   * prepare with an anira_prepare_info and keeps the prepared pointer it hands back for
   * that handler, which its phase calls, its reset and its unprepare receive beside
   * user_data (two handlers of one pipeline never share a prepared pointer; user_data
   * they do). A pipeline holds at most one processor, and the processor owns every phase
   * it fills for every slot (it composes what it does not handle itself by calling the
   * default bodies); a second call is refused. The order relative to
   * anira_pipeline_add_inference means nothing. A refused call creates no carrier and
   * never calls release. The descriptor is 104 bytes on LP64 and 60 on ILP32 without
   * implicit padding; its slots, from the innermost level out: the four phases, reset,
   * prepare, unprepare, init, release; init runs once per registration, at the first
   * anira_handler_prepare of a handler of the pipeline, with an anira_init_info.
   *
   * @param pipeline The pipeline.
   * @param desc The processor; min(struct_size, sizeof(anira_processor_desc)) bytes are
   *   copied, with the consumed kinds, so the record and its strings may die when the call
   *   returns. The slots a shorter struct_size does not cover read as in
   *   ANIRA_PROCESSOR_DESC_INIT.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_STATE when the pipeline already has a
   *   processor; ANIRA_ERROR_INVALID_ARGUMENT for a NULL pipeline or desc, a struct_size
   *   below the three leading slots {struct_size, abi_version, user_data}, a flags bit this
   *   header does not define, or a NULL consumed_kinds (or a NULL entry in it) with a count
   *   above 0; ANIRA_ERROR_ABI_VERSION for an abi_version this library does not serve
   *   (anira_check_abi). A failed status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_pipeline_add_processor(pipeline: number, desc: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_pipeline_add_processor(pipeline, desc, err),
        'anira_pipeline_add_processor',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Creates a custom engine: the id and the descriptor are copied into a refcounted
   * object, which anira_pipeline_add_engine adds to any number of pipelines. The object
   * is the engine's identity and the id its name, both fixed for the object's life:
   * handlers running the same object on an equal model configuration share one loaded
   * model (one load), whichever pipelines they come from; two objects never share, even
   * under one id over the same callbacks and user_data. The handle holds one reference
   * and may be destroyed right after the last anira_pipeline_add_engine; release fires
   * exactly once, when the last reference dies (the handle, the pipelines, their
   * handlers, the loaded models), after every unload. A refused call creates nothing and
   * never calls release.
   *
   * @param engine_id The engine's id, copied: the reverse-URI name model entries name the
   *   engine by (anira_model_config_add_model_path_engine_id), which
   *   anira_backend_id.engine_id, the plan report and every message about the engine carry;
   *   it must contain a '.', and the prefix "anira." is anira's own, with one exception:
   *   "anira.v2.custom", the id anira/compat/v2.hpp creates a 2.x custom backend under,
   *   legal like any other id (a 2.x model file's "CUSTOM" row names it). The id belongs to
   *   the engine, in every pipeline it is added to, and is unique per pipeline
   *   (anira_pipeline_add_engine), not per process: two engine objects may carry one id in
   *   two pipelines.
   * @param desc The engine; min(struct_size, sizeof(anira_engine_desc)) bytes are copied,
   *   with the consumed kinds, so the record and its strings may die when the call returns.
   *   The slots a shorter struct_size does not cover read as in ANIRA_ENGINE_DESC_INIT.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_custom_engine_create(engine_id: string, desc: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_id_ = engine_id === null ? 0 : m.stringToUTF8OnStack(engine_id)
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_custom_engine_create(engine_id_, desc, out_, err),
        'anira_custom_engine_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Drops the handle's reference (or frees a detached handle). The pipelines the engine
   * was added to and the handlers created from them keep theirs; release fires here when
   * no one holds the engine any more.
   *
   * @param engine The handle; NULL is a no-op.
   * Thread: [main-thread]
   */
  anira_custom_engine_destroy(engine: number): void {
    const m = this.module
    m._anira_custom_engine_destroy(engine)
  }

  /**
   * Detaches the handle from the engine's lifetime: from now on the handle keeps the
   * engine alive no longer and names it for anira_pipeline_add_engine only while a
   * pipeline, a handler created from one or a loaded model holds it; once they are gone
   * the engine is released and the addition is ANIRA_ERROR_INVALID_STATE.
   * anira_custom_engine_destroy still frees the handle. For a caller that keeps a handle
   * to add the engine to pipelines yet to come without keeping it alive itself
   * (anira.hpp's Engine object does, so that one object has one C engine for as long as
   * anything holds it). A handle nothing else holds the engine through is released here.
   *
   * @param engine The handle; NULL, or a handle detached already, is a no-op.
   * Thread: [main-thread]
   */
  anira_custom_engine_detach(engine: number): void {
    const m = this.module
    m._anira_custom_engine_detach(engine)
  }

  /**
   * Adds a custom engine to the pipeline, under the id it was created with
   * (anira_custom_engine_create): the pipeline's model entries name it so. The engines of
   * one pipeline have distinct ids. Legal before or after anira_pipeline_add_inference; a
   * handler copies the pipeline at anira_handler_create, so a later addition does not
   * reach that handler. An added engine no entry names is not a plan and not an error; an
   * entry whose id no engine of the pipeline has is ANIRA_ERROR_NOT_SUPPORTED at
   * anira_handler_create, naming the id.
   *
   * @param pipeline The pipeline.
   * @param engine The engine; the pipeline takes a reference, so the handle may be
   *   destroyed right after.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL pipeline or engine;
   *   ANIRA_ERROR_INVALID_STATE when the pipeline already has an engine with the engine's
   *   id, this engine or another, or when a detached handle's engine was released
   *   (anira_custom_engine_detach). A refused call takes no reference. A failed status
   *   throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_pipeline_add_engine(pipeline: number, engine: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_pipeline_add_engine(pipeline, engine, err),
        'anira_pipeline_add_engine',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The backends usable here for a handler of this pipeline on this context: the
   * context's rows first (anira_capabilities_backends: the built-in engines on the
   * providers their runtimes report, as the last anira_context_probe left them), then,
   * for every custom engine added to the pipeline, in the order they were added, one row
   * per provider usable here: ANIRA_PROVIDER_CPU alone for a descriptor without a
   * providers list, else each provider of the list whose query bit is set
   * (anira_engine_query_fn; every listed provider for an engine without a query), engine
   * ANIRA_ENGINE_CUSTOM with the engine's id in engine_id. Every call runs the custom
   * engines' queries; the built-in rows are refreshed by anira_context_probe alone. The
   * strings point into the pipeline's engines (engine_id, a custom provider's
   * provider_id) and into the context's store (a built-in engine's provider_id): valid
   * until the pipeline is destroyed and until the context's next probe. Stride-explicit
   * enumeration: min(element_size, the library's record size) bytes are written per
   * element.
   *
   * @param pipeline The pipeline whose custom engines are asked.
   * @param context The context whose probed rows lead the list.
   * @param element_size sizeof(anira_backend_id) of the caller's header, the stride of
   *   out.
   * @param count In: the capacity of out in elements; out: the number of backends.
   * @param out Receives the rows at the caller's stride, or NULL to ask for the count
   *   only.
   * @returns ANIRA_OK, ANIRA_INCOMPLETE for a short buffer, ANIRA_ERROR_INVALID_ARGUMENT
   *   for a NULL pipeline, context or count or an element_size below the struct_size slot,
   *   or the status a custom engine's query returned when it failed. A failed status throws
   *   an AniraError.
   * Thread: [main-thread]
   */
  anira_pipeline_capabilities_backends(
    pipeline: number,
    context: number,
    element_size: number,
    count: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_pipeline_capabilities_backends(
        pipeline,
        context,
        element_size,
        count,
        out
      ),
      'anira_pipeline_capabilities_backends'
    )
  }

  /**
   * One row of the edge registry a handler of this pipeline sees on this context: for a
   * built-in engine the context's row (anira_capabilities_edge); for a custom engine
   * added to the pipeline, the edge from ANIRA_DOMAIN_HOST to the engine on a provider
   * usable here (its query runs, as anira_pipeline_capabilities_backends runs it):
   * zero-copy to ANIRA_PROVIDER_CPU and to XNNPACK, the CPU providers host memory reaches
   * without a copy, a host copy the engine makes for itself to every other provider, the
   * reason saying the engine declared the provider and its query reported it usable. The
   * strings' lifetime is anira_pipeline_capabilities_backends's.
   *
   * @param pipeline The pipeline whose custom engines are asked.
   * @param context The context whose edge registry answers for a built-in engine.
   * @param from The tensor's domain.
   * @param to The backend; read within its struct_size. A custom engine by its engine_id
   *   (engine ANIRA_ENGINE_CUSTOM), a custom provider by its provider_id (provider
   *   ANIRA_PROVIDER_CUSTOM).
   * @param out Receives the row, read and written within its struct_size (set it before
   *   the call).
   * @returns ANIRA_OK with the row; ANIRA_ERROR_EDGE_UNREACHABLE when neither registry
   *   has a row for the pair (out untouched: a custom engine the pipeline does not add, a
   *   provider it does not declare or its query cleared, a domain other than the host's);
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument or a struct_size below the fixed
   *   fields; or the status a custom engine's query returned when it failed. A failed
   *   status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_pipeline_capabilities_edge(
    pipeline: number,
    context: number,
    from: number,
    to: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_pipeline_capabilities_edge(pipeline, context, from, to, out),
      'anira_pipeline_capabilities_edge'
    )
  }

  /**
   * Destroys a pipeline; handlers created from it keep their copy. The release function
   * of a processor fires here when no handler shares its carrier any more, and a custom
   * engine's when neither its handle, another pipeline, a handler nor a loaded model
   * holds it any more.
   *
   * @param pipeline The handle; NULL is a no-op.
   * Thread: [main-thread]
   */
  anira_pipeline_destroy(pipeline: number): void {
    const m = this.module
    m._anira_pipeline_destroy(pipeline)
  }

  /**
   * Creates a handler over a context from a pipeline, copying everything: the three
   * handles may be destroyed when the call returns. Validates the variant's structure,
   * walks the extensions on the model and its specs, checks the named candidates against
   * this build (a named candidate whose engine is not in the build is
   * ANIRA_ERROR_NOT_SUPPORTED, and so is an entry whose custom id no engine of the
   * pipeline serves; under the default candidate set an entry for an absent engine is
   * skipped, and a variant left with no entry is ANIRA_ERROR_CONFIG) and checks every
   * plan's provider against what its engine serves: a built-in engine's against this
   * context's capabilities (anira_capabilities_backends, what its runtime reports), a
   * custom engine's against its descriptor's providers list (a descriptor without one
   * serves ANIRA_PROVIDER_CPU alone); a provider the engine does not serve is
   * ANIRA_ERROR_NOT_SUPPORTED naming the entry, the engine, the provider and the served
   * list (the engine's load may still refuse one at prepare, a device missing at run
   * time). Models are loaded at anira_handler_prepare in this pre-release, so a file that
   * will not load is reported there. The handler is unprepared until prepare succeeds:
   * every Hard entry returns 0 with ANIRA_ERROR_NOT_PREPARED in anira_handler_rt_error.
   * The handler counts as a user of the core until destroy.
   *
   * @param context The context the handler runs on; the handler adds a reference and
   *   drops it at destroy.
   * @param pipeline The pipeline, copied.
   * @returns What the entry writes to out. A failed status throws an AniraError with the
   *   entry's message.
   * Thread: [main-thread]
   */
  anira_handler_create(context: number, pipeline: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      const err = stackError(m)
      checkStatus(
        m,
        m._anira_handler_create(context, pipeline, out_, err),
        'anira_handler_create',
        err
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Destroys a handler: releases its session (in-flight inferences are drained, the
   * thread pool joins with the last session of this copy, which is why the call must not
   * run under a loader lock), then calls the processor's unprepare for this handler when
   * its prepare succeeded (no phase call runs afterwards), frees the plan report, stops
   * counting as a user of the core, drops the handler's reference on its context and its
   * share of the processor and engine carriers (the release function of a processor or an
   * engine fires here when this handler held the last one). The driver thread must have
   * stopped calling the Hard entries. A plugin must destroy its handlers, then its
   * contexts, before its host unloads it (its instance teardown or module-exit entry
   * point), and never from its own static destructors or DllMain, which run inside the
   * unload under the loader lock.
   *
   * @param handler The handle; NULL is a no-op.
   * Thread: [main-thread & !loader-lock]
   */
  anira_handler_destroy(handler: number): void {
    const m = this.module
    m._anira_handler_destroy(handler)
  }

  /**
   * The blocking quiescence point, and the one call no other handler entry may overlap:
   * validates the variant against the contract (geometry, the explicit budget, the
   * warm-up mode, the miss policy against the anchor, the ring dtypes by canonical name,
   * the contract's extensions), checks every plan's provider against what its engine
   * serves (a built-in engine's runtime as the context reports it now, so a context
   * probed again since the create is seen; a custom engine's query as it answered at
   * anira_handler_create, which prepare never calls again; ANIRA_ERROR_NOT_SUPPORTED as
   * at anira_handler_create), runs the processor's init when it has not run yet (once per
   * registration, before any model loads), loads the model of every candidate with an
   * entry, warms up as the contract says, sizes the rings for the contract's block range
   * and the latency, builds the plan report, selects the plan of the variant's default
   * engine when that engine has a plan (else plan 0), logs the report (Info records of
   * the group anira.capi: the counts and the selected plan, then one record per plan, per
   * slot and per consumed extension) re-arms the real-time latches, logging the count of
   * failures suppressed since the last prepare or reset, and last calls the processor's
   * prepare function, when the pipeline has a processor with one, with an
   * anira_prepare_info (the handler, the report, the entry count, a template of the model
   * end of every slot and the canonical names) and keeps the prepared pointer it hands
   * back for this handler (a status other than ANIRA_OK fails this call with it, and
   * unprepare is not called for a refused prepare). A second prepare replaces the
   * previous session whole: the processor's unprepare of the previous prepare runs once
   * the old session is released, before the new prepare. A failed prepare leaves the
   * handler unprepared, the previous prepare's unprepare called all the same. Refused in
   * this pre-release: an Async contract, ANIRA_BUDGET_MEASURED and
   * ANIRA_WARMUP_UNTIL_STABLE (ANIRA_ERROR_NOT_SUPPORTED; set an explicit budget and
   * FIXED or NONE warm-up), ANIRA_MISS_BYPASS when the anchor is an output or when a
   * streamed output's channel count or ring dtype differs from the anchored input's,
   * ANIRA_MISS_CALLBACK without a function (anira_contract_hard_set_miss_fn), a ring
   * dtype that names no Streamed tensor, or one that differs from its spec's dtype while
   * the processor does not fill the phase that moves that ring (ANIRA_ERROR_CONFIG naming
   * the field), and a processor whose filled pre_process, post_process or reset carries
   * no ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST in its flags, since under a Hard contract
   * those three phases run on the driving thread (ANIRA_ERROR_CONFIG naming the flag).
   *
   * @param handler The handler.
   * @param contract A Hard contract, copied; the handle may be destroyed when the call
   *   returns.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or contract;
   *   ANIRA_ERROR_CONFIG for a rule the configuration breaks, with the offending field
   *   named in the message; ANIRA_ERROR_NOT_SUPPORTED for what this pre-release cannot do,
   *   a Buffer spec under a Hard contract included (Buffer tensors arrive with the Async
   *   contract; the message names the tensor); ANIRA_ERROR_EXTENSION_UNKNOWN or
   *   ANIRA_ERROR_EXTENSION_UNCONSUMED for an extension on the contract, the model or a
   *   spec that this build does not know or nothing consumes; ANIRA_ERROR_NO_SUCH_FILE,
   *   ANIRA_ERROR_MODEL_LOAD or ANIRA_ERROR_ENGINE when a model does not load; the status a
   *   processor's prepare function returned; ANIRA_ERROR_INVALID_STATE while an audio node
   *   streams through the handler (WebAssembly: its render callback may be inside a process
   *   call; re-prepare through anira_em_audio_node_prepare). A failed status throws an
   *   AniraError with the entry's message.
   * Thread: [main-thread & !processing]
   */
  anira_handler_prepare(handler: number, contract: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_handler_prepare(handler, contract, err),
        'anira_handler_prepare',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The handler-owned plan report of the last successful prepare, walked by
   * anira_plan_report_num_plans / plans / slots / exts; valid until the next prepare or
   * destroy.
   *
   * @param handler A prepared handler.
   * @returns The report, or NULL for a NULL or unprepared handler.
   * Thread: [main-thread & prepared]
   */
  anira_handler_plan_report(handler: number): number {
    const m = this.module
    return m._anira_handler_plan_report(handler) >>> 0
  }

  /**
   * The entries of the prepared handler: the chunks that can be in flight at once, each
   * holding one chunk from its pre_process to its post_process. anira_processor_ctx.entry
   * names one of them, 0 .. num_entries - 1, in every phase callback: the size of a
   * processor's per-chunk scratch, allocated in its prepare function (the handler counts
   * as prepared there, and anira_prepare_info.num_entries carries the same count) and
   * indexed by the entry at run time. Fixed by anira_handler_prepare for the life of the
   * session; the next prepare may change it.
   *
   * @param handler A prepared handler.
   * @returns The count; 0 for a NULL or unprepared handler.
   * Thread: [main-thread & prepared]
   */
  anira_handler_num_entries(handler: number): number {
    const m = this.module
    return m._anira_handler_num_entries(handler) >>> 0
  }

  /**
   * The number of plans; the dense indices 0..num_plans-1 are what anira_handler_set_plan
   * takes.
   *
   * @param report The report.
   * @returns The count; 0 for a NULL report.
   * Thread: [main-thread]
   */
  anira_plan_report_num_plans(report: number): number {
    const m = this.module
    return m._anira_plan_report_num_plans(report) >>> 0
  }

  /**
   * Enumerates the plans in dense-index order; min(element_size, sizeof(anira_plan_info))
   * bytes are written per row.
   *
   * @param report The report.
   * @param element_size sizeof(anira_plan_info) of the caller's header; rows are written
   *   at this stride.
   * @param count In: the capacity of out; out: the number of plans.
   * @param out Receives the rows, or NULL to ask for the count.
   * @returns ANIRA_OK; ANIRA_INCOMPLETE when out is too short (count holds the total);
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL report or count, or an element_size below the
   *   record's fixed head. A failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_plan_report_plans(
    report: number,
    element_size: number,
    count: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_plan_report_plans(report, element_size, count, out),
      'anira_plan_report_plans'
    )
  }

  /**
   * Enumerates the input or output slots of one plan in tensor order; min(element_size,
   * sizeof(anira_plan_slot)) bytes are written per row.
   *
   * @param report The report.
   * @param plan The dense plan index.
   * @param inputs Nonzero for the input slots, 0 for the output slots.
   * @param element_size sizeof(anira_plan_slot) of the caller's header.
   * @param count In: the capacity of out; out: the number of slots.
   * @param out Receives the rows, or NULL to ask for the count.
   * @returns ANIRA_OK; ANIRA_INCOMPLETE when out is too short;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL report or count, a plan index out of range,
   *   or an element_size below the record's fixed head. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_plan_report_slots(
    report: number,
    plan: number,
    inputs: boolean,
    element_size: number,
    count: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_plan_report_slots(report, plan, inputs ? 1 : 0, element_size, count, out),
      'anira_plan_report_slots'
    )
  }

  /**
   * Enumerates the extensions one plan consumes; min(element_size,
   * sizeof(anira_plan_ext)) bytes are written per row.
   *
   * @param report The report.
   * @param plan The dense plan index.
   * @param element_size sizeof(anira_plan_ext) of the caller's header.
   * @param count In: the capacity of out; out: the number of rows.
   * @param out Receives the rows, or NULL to ask for the count.
   * @returns ANIRA_OK; ANIRA_INCOMPLETE when out is too short;
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL report or count, a plan index out of range,
   *   or an element_size below the record's fixed head. A failed status throws an
   *   AniraError.
   * Thread: [main-thread]
   */
  anira_plan_report_exts(
    report: number,
    plan: number,
    element_size: number,
    count: number,
    out: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_plan_report_exts(report, plan, element_size, count, out),
      'anira_plan_report_exts'
    )
  }

  /**
   * The first plan, in dense-index order, that has every property the query names: the
   * variant unless ANIRA_VARIANT_ANY, the engine unless ANIRA_ENGINE_NONE (a custom
   * engine by ANIRA_ENGINE_CUSTOM and its engine_id, compared as a string), the provider
   * unless ANIRA_PROVIDER_NONE (a custom provider by ANIRA_PROVIDER_CUSTOM and its
   * provider_id, compared as a string). What a host looks the plan of a backend choice up
   * with, from the thread that picks it: the index goes to anira_handler_set_plan. Reads
   * the report's rows and nothing else: no allocation, no lock. Callable from any thread
   * while the report is valid (until the handler's next prepare or its destroy).
   *
   * @param report The report.
   * @param query What the plan must have; ANIRA_PLAN_QUERY_INIT matches every plan.
   * @returns What the entry writes to plan. A failed status throws an AniraError.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_plan_report_find(report: number, query: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const plan_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_plan_report_find(report, query, plan_),
        'anira_plan_report_find'
      )
      return heap(m).HEAPU32[plan_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Selects the plan the next submitted chunk runs on: one relaxed store, never planning.
   * The selection is a single atomic value, the dense index itself, so callers on several
   * threads cannot leave anira_handler_get_plan and the running engine in disagreement,
   * and two plans on one engine (two variants, two providers) stay distinct. A chunk
   * keeps the plan it was submitted under from its pre-processing to its post-processing:
   * one that is queued or in flight when the call lands finishes on the old plan, and
   * exactly one engine runs for it. A plan switch keeps the model's state: every plan of
   * the one variant runs the same declared State pairs, and the next chunk's inference
   * reads the state the last one produced, whichever plan produced it and whether or not
   * that plan's engine keeps its own aliasing (ANIRA_ENGINE_FLAG_STATE_ALIAS). An index
   * out of range is a no-op recorded as ANIRA_ERROR_CONFIG in anira_handler_rt_error; a
   * call on an unprepared handler is a no-op recorded as ANIRA_ERROR_NOT_PREPARED. Not
   * while anira_handler_prepare runs (prepare is the quiescence point).
   *
   * @param handler The handler.
   * @param plan A dense plan index the report handed out.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler;
   *   ANIRA_ERROR_NOT_PREPARED before a successful prepare; ANIRA_ERROR_CONFIG for an index
   *   out of range. A failed status throws an AniraError.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_handler_set_plan(handler: number, plan: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_handler_set_plan(handler, plan),
      'anira_handler_set_plan'
    )
  }

  /**
   * The plan selected last: the one the next submitted chunk runs on (one relaxed load of
   * the value anira_handler_set_plan stores). Not while anira_handler_prepare runs.
   *
   * @param handler The handler.
   * @returns The dense plan index; 0 for a NULL or unprepared handler.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_handler_get_plan(handler: number): number {
    const m = this.module
    return m._anira_handler_get_plan(handler) >>> 0
  }

  /**
   * Pushes one block into the input ring of in_slot, submits whatever inferences are due,
   * collects the completed ones without waiting and pops one block from the output ring
   * of out_slot into the memory of out; the two sample counts may differ under a time
   * ratio, and the two slots are independent (a model whose stream is input 1 and output
   * 0 calls anira_handler_process(handler, in, 1, out, 0, delivered)). Accepts a planar
   * tensor (ANIRA_TENSOR_PLANAR) as well as one block read by strides (interleaved is
   * strides {1, channels}) or packed, for any channel count; each side has its own
   * description and its own dtype, which must be its slot's ring dtype (nothing
   * converts). A single form is the form of a Streamed slot: the single form of a Static
   * slot is anira_handler_set_static_input / anira_handler_get_static_output, a State
   * tensor is carried by no Hard entry, and an in_slot or out_slot that names a Static or
   * a State tensor is refused (a generator with Static input 0 calls
   * anira_handler_set_static_input(0) and anira_handler_pop_data(0)). Every tensor is
   * validated before anything is pushed, so a refusal pushes nothing. Each tensor carries
   * one host block of the geometry's block_max at most, ceil(b) samples of the block b on
   * its slot's stream (the header doc): a larger shape[1] is refused, never cut to its
   * tail. A block whose inference has not completed is an on_miss event: the contract's
   * policy fills out (ANIRA_MISS_ZEROS zeros, ANIRA_MISS_HOLD_LAST the last delivered
   * block, ANIRA_MISS_BYPASS min(shape[1] of in, shape[1] of out) samples per channel of
   * the input block when in_slot is the anchored input's slot and zeros past it; any
   * other slot zeros, ANIRA_MISS_CALLBACK what the contract's anira_miss_fn writes) and
   * the call returns ANIRA_MISSED, a success: the memory is valid and the stream stays
   * time-aligned. A refusal is a failure status (ANIRA_FAILED), recorded in
   * anira_handler_rt_error; a miss is not recorded. One tensor handed over as in and out
   * is in place, and ANIRA_MISS_BYPASS leaves it where it is; input and output memory
   * that overlap under two different descriptions (an interleaved input over planar views
   * of the same bytes) are undefined on an ANIRA_MISS_BYPASS miss. release, manager_ctx
   * and acquire of the tensors are not read: the memory is borrowed until the call
   * returns. A model with several Streamed inputs is driven by one call per host block
   * over all of them (anira_handler_process_multi); separate single-slot calls per stream
   * keep the streams aligned but lose ANIRA_MISS_BYPASS (the call that does not carry the
   * anchored slot fills zeros on a miss) and the wait credit of ANIRA_WAIT_CONTRACT for
   * that call.
   *
   * @param handler The handler.
   * @param in_ The input block of in_slot: a host tensor of the logical shape [channels,
   *   samples] in the slot's dtype, planar, read by strides or packed; shape[1] is the
   *   number of samples pushed. Never written.
   * @param in_slot The slot of in: the tensor's position in the model config's input
   *   list; a Streamed tensor. A slot is that one number everywhere: every slot, in_slot
   *   and out_slot of this header, the positions of the arrays and delivered counts of the
   *   _multi forms and of anira_miss_fn, the latency vector, the plan report's slot rows
   *   and the slot of a processor's context accessors (anira_processor_input_role and its
   *   siblings) use it, each against the list of its side. The role of the tensor decides
   *   which entries take its slot: a Streamed one the block calls, a Static one
   *   anira_handler_set_static_input / anira_handler_get_static_output or an element of a
   *   _multi form, a State one no Hard entry. A host that does not want to hard-code the
   *   number resolves it by the tensor's canonical name at setup.
   * @param out The output block of out_slot, described the same way; shape[1] is the
   *   number of samples requested. The descriptor is never written, the memory it names is.
   *   The same tensor as in for in place.
   * @param out_slot The slot of out: the tensor's position in the model config's output
   *   list; a Streamed tensor. Independent of in_slot: the two lists are unrelated, and a
   *   model's stream need not stand at the same position on both sides.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_process(
    handler: number,
    in_: number,
    in_slot: number,
    out: number,
    out_slot: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_process(handler, in_, in_slot, out, out_slot, delivered_),
        'anira_handler_process'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * anira_handler_process over every slot at once, Static slots included. The call IS the
   * sequence it abbreviates, over the same functions: anira_handler_set_static_input for
   * every non-empty Static element of inputs, the streamed call over the Streamed
   * elements, anira_handler_get_static_output for every non-empty Static element of
   * outputs. So a Static input of the call is seen by every inference the call submits,
   * and a Static output holds the value of the latest inference the call or an earlier
   * one collected, whatever the block's status: ANIRA_MISS_ZEROS, ANIRA_MISS_HOLD_LAST
   * and ANIRA_MISS_BYPASS define nothing for a Static output, which is the stored value.
   * Under ANIRA_MISS_CALLBACK a missed block's non-empty Static output elements are
   * filled with the stored value first, then anira_miss_fn is called with these arrays
   * and may overwrite them; the get step is skipped for that block, and what the function
   * wrote never enters the store. The two arrays are handed to the copy path as they are;
   * every element is validated before anything is set or pushed, so one malformed tensor
   * refuses the whole call, no ring is written and no Static value changes. Accepts
   * planar tensors for the Streamed elements, and each tensor has its own description and
   * dtype; a Static element is one block (ANIRA_TENSOR_PLANAR is refused). On a miss
   * every requested Streamed output holds what the policy says (ANIRA_MISS_BYPASS copies
   * the anchored input's block of this call into every requested streamed output of its
   * dtype).
   *
   * @param handler The handler.
   * @param inputs One tensor per input slot, in slot order and covering every slot. The
   *   element of a Streamed slot is its host block, an empty tensor (shape[1] == 0) leaves
   *   the slot out. The element of a Static slot is the whole tensor in the spec's shape
   *   and dtype, exactly what anira_handler_set_static_input takes, or an empty tensor. The
   *   element of a State slot must be an empty tensor: anira feeds declared state itself,
   *   and anything else there is ANIRA_ERROR_INVALID_ARGUMENT. Empty is a rank of 1 or more
   *   with an extent of 0 (a spec extent is never 0; a zeroed record has rank 0 and is not
   *   empty); nothing else of it is read. Never written.
   * @param num_inputs The length of inputs; must be the number of tensors of the model
   *   config's input list, State tensors included.
   * @param outputs One tensor per output slot, in slot order and covering every slot;
   *   shape[1] of a Streamed element is the request, a Static element is the whole tensor
   *   in the spec's shape and dtype (what anira_handler_get_static_output takes), an empty
   *   tensor leaves its slot untouched, and the element of a State slot must be an empty
   *   tensor. The descriptors are never written.
   * @param num_outputs The length of outputs; must be the number of tensors of the model
   *   config's output list, State tensors included.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_process_multi(
    handler: number,
    inputs: number,
    num_inputs: number,
    outputs: number,
    num_outputs: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_process_multi(
          handler,
          inputs,
          num_inputs,
          outputs,
          num_outputs,
          delivered_
        ),
        'anira_handler_process_multi'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Pushes one block into the input ring of the slot and submits the inferences that are
   * due; nothing is popped (anira_handler_pop_data). Accepts a planar tensor. A Static
   * slot is refused (its single form is anira_handler_set_static_input) and so is a State
   * slot, which no Hard entry carries, so a generator (no streamed input) has no slot
   * this entry takes.
   *
   * @param handler The handler.
   * @param in_ The input block of the slot, as in anira_handler_process.
   * @param slot The slot: the tensor's position in the model config's input list; a
   *   Streamed tensor.
   * @returns ANIRA_OK, or a failure as in anira_handler_process (the checks of in),
   *   recorded in anira_handler_rt_error. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_push_data(handler: number, in_: number, slot: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_handler_push_data(handler, in_, slot),
      'anira_handler_push_data'
    )
  }

  /**
   * anira_handler_push_data over every input slot at once: anira_handler_set_static_input
   * for every non-empty Static element, then the push of the Streamed elements. Accepts
   * planar tensors for the Streamed elements.
   *
   * @param handler The handler.
   * @param inputs One tensor per input slot, as in anira_handler_process_multi.
   * @param num_inputs The length of inputs; must be the number of tensors of the model
   *   config's input list, State tensors included.
   * @returns ANIRA_OK, or a failure as in anira_handler_process_multi (the checks of
   *   inputs), recorded in anira_handler_rt_error. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_push_data_multi(
    handler: number,
    inputs: number,
    num_inputs: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_handler_push_data_multi(handler, inputs, num_inputs),
      'anira_handler_push_data_multi'
    )
  }

  /**
   * Collects the completed inferences without waiting and pops one block from the output
   * ring of the slot; on a generator the request also pulls the next inference. Accepts a
   * planar tensor. A missed block follows the miss policy and returns ANIRA_MISSED, where
   * ANIRA_MISS_BYPASS delivers zeros (a pop has no input block to pass through).
   *
   * @param handler The handler.
   * @param out The output block of the slot, as in anira_handler_process; shape[1] is the
   *   request.
   * @param slot The slot: the tensor's position in the model config's output list; a
   *   Streamed tensor (a Static slot is refused: its single form is
   *   anira_handler_get_static_output; a State slot is refused: no Hard entry carries it).
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_pop_data(handler: number, out: number, slot: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_pop_data(handler, out, slot, delivered_),
        'anira_handler_pop_data'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * anira_handler_pop_data over every output slot at once: the pop of the Streamed
   * elements, then anira_handler_get_static_output for every non-empty Static element
   * (under ANIRA_MISS_CALLBACK as in anira_handler_process_multi); an output whose tensor
   * is empty is left untouched. Accepts planar tensors for the Streamed elements.
   *
   * @param handler The handler.
   * @param outputs One tensor per output slot, as in anira_handler_process_multi.
   * @param num_outputs The length of outputs; must be the number of tensors of the model
   *   config's output list, State tensors included.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_pop_data_multi(
    handler: number,
    outputs: number,
    num_outputs: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_pop_data_multi(handler, outputs, num_outputs, delivered_),
        'anira_handler_pop_data_multi'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Stores the value of a Static input in the handler: the whole tensor, copied under the
   * slot's latch, so an inference never sees a torn tensor. Every inference submitted
   * after the call sees the new value, materialised into the model's input tensor ahead
   * of the processor's pre_process; a value set between two Hard calls applies from the
   * next submitted inference on. The store is the handler's, zeroed at
   * anira_handler_create, never resized, untouched by anira_handler_prepare and
   * anira_handler_reset: the entry is legal from create on, a value set before prepare
   * survives it, and a processor's prepare may call it. 'Fits' is a check, never a clamp:
   * there is no partial Static tensor and no count. One writer at a time; the driver
   * thread is the writer (and the reader, under a Hard contract, so the latch is never
   * contended). A host that writes from another thread breaks the tag: the tensor still
   * never tears, but a writer preempted inside the call can make the driver thread wait
   * for it.
   *
   * @param handler The handler; prepared or not.
   * @param slot The slot: the tensor's position in the model config's input list; a
   *   Static tensor. A slot of any other role is refused.
   * @param tensor The whole tensor: a host tensor of exactly the spec's rank, extents and
   *   dtype (any dtype, the spec's; a Channel axis of any extent is one of its axes), over
   *   one block of host memory read by its strides in elements, all-zero strides being
   *   packed row-major. Never written; release, manager_ctx and acquire are not read, the
   *   memory is borrowed until the call returns.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or tensor, a slot
   *   out of range, a Streamed or a State slot, a domain other than ANIRA_DOMAIN_HOST and
   *   ANIRA_DOMAIN_HOST_PINNED, a rank or an extent other than the spec's (an empty tensor
   *   and a zeroed record included), NULL memory, a negative stride, a stride of 0 on an
   *   axis longer than 1 unless every stride is 0, memory that does not start on a multiple
   *   of the element size; ANIRA_ERROR_NOT_SUPPORTED for ANIRA_TENSOR_PLANAR (a Static
   *   tensor is one block) and for a flag bit this library does not know;
   *   ANIRA_ERROR_CONFIG for a dtype other than the spec's (nothing converts). Checked in
   *   the order slot, domain, flags, rank and extents, dtype, memory and strides. Every
   *   failure but the NULL handler is recorded in anira_handler_rt_error and logged once
   *   through the real-time queue; a refused call stores nothing. A failed status throws an
   *   AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_set_static_input(handler: number, slot: number, tensor: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_handler_set_static_input(handler, slot, tensor),
      'anira_handler_set_static_input'
    )
  }

  /**
   * Reads the value of a Static output: the whole tensor the latest collected inference
   * produced, copied under the slot's latch, never torn. The handler captures it from the
   * model's output tensor behind the last processor's post_process, when a Hard entry
   * collects the inference; after a Hard call it is the value of the latest inference
   * that call or an earlier one collected. All zeros before the first capture. A chunk
   * that completed as zeros (dropped, or failed in a processor or in the engine) captures
   * nothing, and anira_handler_reset and anira_handler_prepare leave the store alone: it
   * holds what the model produced last. The miss policies define nothing for a Static
   * output, and what an anira_miss_fn writes into a caller's tensor never enters the
   * store. Legal from anira_handler_create on.
   *
   * @param handler The handler; prepared or not. Not const: a refusal is recorded in its
   *   rt_error.
   * @param slot The slot: the tensor's position in the model config's output list; a
   *   Static tensor. A slot of any other role is refused.
   * @param out The whole tensor, described as for anira_handler_set_static_input; the
   *   descriptor is never written, the memory it names is, by its strides.
   * @returns ANIRA_OK; the failures of anira_handler_set_static_input, and
   *   ANIRA_ERROR_INVALID_ARGUMENT for ANIRA_TENSOR_READ_ONLY on out. Every failure but the
   *   NULL handler is recorded in anira_handler_rt_error; a refused call writes nothing. A
   *   failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_get_static_output(handler: number, slot: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_handler_get_static_output(handler, slot, out),
      'anira_handler_get_static_output'
    )
  }

  /**
   * The stream latency of one output in samples of that output, valid from prepare on and
   * not while anira_handler_prepare runs: the 2.x arithmetic including the wait_ratio
   * credit, so a host that calls the ANIRA_NONBLOCKING entries on a wait_ratio above 0
   * handler gets the same figure and more on_miss events. A generator counts from its
   * first process or pop after prepare or reset. Where the Hard contract declared a
   * stream latency for the slot (anira_contract_hard_set_latency), that figure replaces
   * the computed one.
   *
   * @param handler A prepared handler.
   * @param slot The slot: the tensor's position in the model config's output list.
   * @returns The latency; 0 for an output that is not Streamed (Static, State), a NULL or
   *   unprepared handler or an index out of range.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_handler_get_latency(handler: number, slot: number): number {
    const m = this.module
    return m._anira_handler_get_latency(handler, slot) >>> 0
  }

  /**
   * The latency vector, index-aligned with the output list; valid from prepare on and not
   * while anira_handler_prepare runs. A slot the Hard contract declared a stream latency
   * for (anira_contract_hard_set_latency) reports that figure.
   *
   * @param handler A prepared handler.
   * @param count In: the capacity of out; out: the number of output tensors.
   * @param out Receives one latency per output slot, in slot order: one entry per tensor
   *   of the model config's output list, 0 for one that is not Streamed (Static, State); or
   *   NULL to ask for the count.
   * @returns ANIRA_OK; ANIRA_INCOMPLETE when out is too short (count holds the total);
   *   ANIRA_ERROR_INVALID_ARGUMENT for a NULL handler or count; ANIRA_ERROR_NOT_PREPARED
   *   before a successful prepare. A failed status throws an AniraError.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_handler_get_latencies(handler: number, count: number, out: number): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_handler_get_latencies(handler, count, out),
      'anira_handler_get_latencies'
    )
  }

  /**
   * Collects the completed inferences without waiting (running the post-processing of
   * each) and reports the samples waiting in the output ring of one channel; right after
   * prepare that is the reported latency.
   *
   * @param handler The handler; not const, the call collects completed inferences.
   * @param slot The slot: the tensor's position in the model config's output list.
   * @param channel The channel, below the output's channel count.
   * @returns What the entry writes to out. A failed status throws an AniraError.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_get_available_samples(
    handler: number,
    slot: number,
    channel: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const out_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_get_available_samples(handler, slot, channel, out_),
        'anira_handler_get_available_samples'
      )
      return heap(m).HEAPU32[out_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Wait-free stream reset: the rings return to their post-prepare state (the latency
   * zeros re-seeded), in-flight results are discarded when they complete, the held block
   * of ANIRA_MISS_HOLD_LAST is dropped. Declared state (ANIRA_ROLE_STATE) is
   * re-initialised to zeros: not by this call, which stays one generation bump, but on
   * the inference thread at the first inference of the new stream, so an inference still
   * in flight across the reset can never seed the new stream with its state. The state a
   * session-exclusive engine keeps inside itself and the state a processor keeps between
   * chunks are re-initialised through their reset slots (anira_engine_desc.reset on the
   * inference thread, anira_processor_desc.reset on the thread that runs pre_process) at
   * the first chunk of the new stream; without a reset slot they stay untouched. Clears
   * anira_handler_rt_error and re-arms the real-time latches, logging the count of
   * failures suppressed since the last prepare or reset through the real-time queue.
   *
   * @param handler The handler; NULL is a no-op.
   * Thread: [driver-thread] ANIRA_NONBLOCKING
   */
  anira_handler_reset(handler: number): void {
    const m = this.module
    m._anira_handler_reset(handler)
  }

  /**
   * The last real-time failure recorded on the handler: a contract violation of an
   * ANIRA_NONBLOCKING or _wait entry (ANIRA_ERROR_NOT_PREPARED,
   * ANIRA_ERROR_WRONG_CONTRACT, ANIRA_ERROR_CONFIG, ANIRA_ERROR_INVALID_STATE,
   * ANIRA_ERROR_INVALID_ARGUMENT, ANIRA_ERROR_NOT_SUPPORTED for a host tensor with a flag
   * this library does not know) or ANIRA_ERROR_ENGINE after a failed inference (its
   * output is zeros), or ANIRA_ERROR_CAPACITY after a hop the scheduler dropped because
   * no inference slot was free (its output is the miss policy's fill, at its stream
   * position). An entry's ANIRA_ERROR_CAPACITY return is back-pressure and never lands
   * here. Last-wins; cleared by prepare and reset. Each kind is logged once per prepare
   * or reset through the real-time queue (a violation flagged
   * ANIRA_LOG_RECORD_CONTRACT_VIOLATION), later occurrences are counted and the drain
   * reports a persisting condition at most every 10 seconds. One relaxed load: readable
   * from any thread, a callback, a crash handler.
   *
   * @param handler The handler.
   * @returns ANIRA_OK when nothing was recorded since the last prepare or reset (or for a
   *   NULL handler); else the status. A failed status throws an AniraError.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_handler_rt_error(handler: number): number {
    const m = this.module
    return checkStatus(m, m._anira_handler_rt_error(handler), 'anira_handler_rt_error')
  }

  /**
   * anira_handler_process that waits for the block's inference: on the completion
   * semaphore when the contract's wait_ratio is above 0, else by polling the done flag
   * every millisecond. A block not completed at the timeout is an on_miss event as in the
   * ANIRA_NONBLOCKING entry, ANIRA_MISSED. Without an inference thread inside its loop
   * (anira_inference_thread_run_loop, or the core's pool) the call does what the
   * ANIRA_NONBLOCKING entry does, records ANIRA_ERROR_INVALID_STATE in
   * anira_handler_rt_error and returns it at once; a thread leaving during a poll is
   * noticed at the next poll, one leaving during a semaphore wait at the timeout. A host
   * that pumps anira_inference_thread_execute itself is not counted and uses the
   * ANIRA_NONBLOCKING entries. Accepts a planar tensor. Legal from the driver thread only
   * if the host accepts a wait there; on WebAssembly every wait spins.
   *
   * @param handler The handler.
   * @param in_ The input block of in_slot, as in anira_handler_process.
   * @param in_slot The slot of in: the tensor's position in the model config's input
   *   list; a Streamed tensor, as in anira_handler_process.
   * @param out The output block of out_slot, as in anira_handler_process; the same tensor
   *   as in for in place.
   * @param out_slot The slot of out: the tensor's position in the model config's output
   *   list; a Streamed tensor, independent of in_slot, as in anira_handler_process.
   * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds
   *   (a value at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio times
   *   the duration of this call's block on the anchor, measured by shape[1] of the anchored
   *   tensor (the 2.x blocking_ratio); ANIRA_WAIT_FOREVER, or any other negative value,
   *   without limit.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [any-thread, blocking]
   */
  anira_handler_process_wait(
    handler: number,
    in_: number,
    in_slot: number,
    out: number,
    out_slot: number,
    timeout_ms: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_process_wait(
          handler,
          in_,
          in_slot,
          out,
          out_slot,
          delivered_,
          timeout_ms
        ),
        'anira_handler_process_wait'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * anira_handler_process_multi that waits for the block's inference, as
   * anira_handler_process_wait does. Accepts planar tensors.
   *
   * @param handler The handler.
   * @param inputs One tensor per input slot, as in anira_handler_process_multi.
   * @param num_inputs The length of inputs; must be the number of tensors of the model
   *   config's input list, State tensors included.
   * @param outputs One tensor per output slot, as in anira_handler_process_multi.
   * @param num_outputs The length of outputs; must be the number of tensors of the model
   *   config's output list, State tensors included.
   * @param timeout_ms As in anira_handler_process_wait.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [any-thread, blocking]
   */
  anira_handler_process_multi_wait(
    handler: number,
    inputs: number,
    num_inputs: number,
    outputs: number,
    num_outputs: number,
    timeout_ms: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_process_multi_wait(
          handler,
          inputs,
          num_inputs,
          outputs,
          num_outputs,
          delivered_,
          timeout_ms
        ),
        'anira_handler_process_multi_wait'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * anira_handler_pop_data that waits for the block's inference, as
   * anira_handler_process_wait does. A block not completed at the timeout is ANIRA_MISSED
   * (ANIRA_MISS_BYPASS delivers zeros, a pop has no input block to pass through). Accepts
   * a planar tensor.
   *
   * @param handler The handler.
   * @param out The output block of the slot, as in anira_handler_process; shape[1] is the
   *   request.
   * @param slot The slot: the tensor's position in the model config's output list.
   * @param timeout_ms How long to wait for the block's inference: 0 or more milliseconds
   *   (a value at or above 1e12 is without limit); ANIRA_WAIT_CONTRACT for wait_ratio times
   *   the contract's block_max duration (a pop has no input block to measure);
   *   ANIRA_WAIT_FOREVER, or any other negative value, without limit.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [any-thread, blocking]
   */
  anira_handler_pop_data_wait(
    handler: number,
    out: number,
    slot: number,
    timeout_ms: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_pop_data_wait(handler, out, slot, delivered_, timeout_ms),
        'anira_handler_pop_data_wait'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * anira_handler_pop_data_multi that waits for the block's inference, as
   * anira_handler_pop_data_wait does. Accepts planar tensors.
   *
   * @param handler The handler.
   * @param outputs One tensor per output slot, as in anira_handler_process_multi.
   * @param num_outputs The length of outputs; must be the number of tensors of the model
   *   config's output list, State tensors included.
   * @param timeout_ms As in anira_handler_pop_data_wait.
   * @returns What the entry writes to delivered. A failed status throws an AniraError.
   * Thread: [any-thread, blocking]
   */
  anira_handler_pop_data_multi_wait(
    handler: number,
    outputs: number,
    num_outputs: number,
    timeout_ms: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const delivered_ = stackSlot(m)
      checkStatus(
        m,
        m._anira_handler_pop_data_multi_wait(
          handler,
          outputs,
          num_outputs,
          delivered_,
          timeout_ms
        ),
        'anira_handler_pop_data_multi_wait'
      )
      return heap(m).HEAPU32[delivered_ >>> 2]
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills a tensor over a Metal buffer: domain ANIRA_DOMAIN_METAL_BUFFER, handle.mtl. The
   * token is copied as anira_tensor_init_wgpu_buffer copies its fence. Zero-then-fill and
   * the refusals of anira_tensor_init_host.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param buffer id<MTLBuffer>, not retained.
   * @param shared_event Copied into acquire (kind ANIRA_SYNC_MTL_SHARED_EVENT: the
   *   id<MTLSharedEvent> and its value), or NULL: already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_metal(
    tensor: number,
    buffer: number,
    shared_event: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_metal(tensor, buffer, shared_event, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over an IOSurface whose plane 0 is the byte-image encoding of packed
   * floats: domain ANIRA_DOMAIN_IOSURFACE, handle.iosurface. Allowlisted 64-bit
   * ANIRA_NONBLOCKING declaration: size mirrors the arm. Zero-then-fill and the refusals
   * of anira_tensor_init_host.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param surface IOSurfaceRef, not retained; plane 0 holds the byte image.
   * @param size Size of plane 0 in bytes.
   * @param shared_event Copied into acquire (kind ANIRA_SYNC_MTL_SHARED_EVENT), or NULL:
   *   already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_iosurface(
    tensor: number,
    surface: number,
    size: bigint,
    shared_event: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_iosurface(
      tensor,
      surface,
      size,
      shared_event,
      dtype,
      ndim,
      shape
    )
  }

  /**
   * Fills a tensor over an AHardwareBuffer: domain ANIRA_DOMAIN_AHARDWAREBUFFER,
   * handle.ahb. fence_fd follows the sync_fd rule of anira_tensor_init_dmabuf
   * (ANIRA_SYNC_SYNC_FILE_FD, owned; the caller's again when the call is refused).
   * Zero-then-fill and the refusals of anira_tensor_init_host.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param buffer AHardwareBuffer* of format BLOB, not acquired.
   * @param fence_fd A sync_file fd the data is valid after, owned by the tensor's acquire
   *   token from here on; negative: already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_ahardwarebuffer(
    tensor: number,
    buffer: number,
    fence_fd: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_ahardwarebuffer(tensor, buffer, fence_fd, dtype, ndim, shape)
  }

  /**
   * Fills a tensor over a Direct3D 12 resource: domain ANIRA_DOMAIN_D3D12, handle.d3d12.
   * The token is copied as anira_tensor_init_wgpu_buffer copies its fence. Zero-then-fill
   * and the refusals of anira_tensor_init_host.
   *
   * @param tensor The caller's record; NULL is a no-op.
   * @param resource ID3D12Resource*, no reference taken.
   * @param shared_handle The NT shared handle of the resource, or NULL; not owned.
   * @param fence Copied into acquire (kind ANIRA_SYNC_D3D12_FENCE: the ID3D12Fence* and
   *   its value), or NULL: already visible.
   * @param dtype The element type.
   * @param ndim The rank, at most ANIRA_MAX_RANK.
   * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
   * Thread: [thread-safe] [callback-safe] ANIRA_NONBLOCKING
   */
  anira_tensor_init_d3d12(
    tensor: number,
    resource: number,
    shared_handle: number,
    fence: number,
    dtype: number,
    ndim: number,
    shape: number
  ): void {
    const m = this.module
    m._anira_tensor_init_d3d12(tensor, resource, shared_handle, fence, dtype, ndim, shape)
  }

  /**
   * Fills a processor descriptor whose slots are the trampolines into the scope
   * registry's processor: one per bit of hook_mask, every other slot NULL, the release
   * slot always (it frees the record behind user_data, after the hook's release when
   * ANIRA_EM_PROCESSOR_HOOK_RELEASE is set), and flags processor_flags. The promise is
   * stated, never derived from the hook bits: under a Hard contract a processor with a
   * pre_process, post_process or reset hook runs it on the driving thread (the audio
   * worklet), and anira_handler_prepare refuses it with ANIRA_ERROR_CONFIG unless
   * processor_flags carries ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST. The descriptor is for
   * anira_pipeline_add_processor; one that is never added keeps its record, so
   * anira_em_pipeline_add_js_processor is the call JavaScript makes.
   *
   * @param desc The caller's descriptor, overwritten from ANIRA_PROCESSOR_DESC_INIT.
   * @param js_key The key the hook receives, the processor's entry in the instance's
   *   registry.
   * @param hook_mask An OR of the ANIRA_EM_PROCESSOR_HOOK_* bits: the slots that cross
   *   into the scope registry's processor.
   * @param processor_flags The descriptor's flags, the processor's real-time promise,
   *   which JavaScript cannot write into the descriptor itself: an OR of
   *   ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST (the pre_process, post_process and reset hooks
   *   allocate nothing, lock nothing and block on nothing) and
   *   ANIRA_PROCESSOR_FLAG_REALTIME_HOOKS (before_inference and after_inference likewise),
   *   or 0 for no promise. Stored as given: a bit the header does not define is refused by
   *   anira_pipeline_add_processor.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL desc or a bit hook_mask
   *   does not define; ANIRA_ERROR_OUT_OF_MEMORY when the record cannot be allocated. A
   *   failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_em_processor_desc_init(
    desc: number,
    js_key: number,
    hook_mask: number,
    processor_flags: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_em_processor_desc_init(desc, js_key, hook_mask, processor_flags),
      'anira_em_processor_desc_init'
    )
  }

  /**
   * anira_em_processor_desc_init followed by anira_pipeline_add_processor: the pipeline's
   * one processor is a JavaScript processor. A refused addition frees the record; the
   * hook's release is not called then, as anira_pipeline_add_processor never calls
   * release on a refusal.
   *
   * @param pipeline The pipeline.
   * @param js_key As in anira_em_processor_desc_init.
   * @param hook_mask As in anira_em_processor_desc_init.
   * @param processor_flags As in anira_em_processor_desc_init: the processor's real-time
   *   promise.
   * @returns As anira_em_processor_desc_init, then as anira_pipeline_add_processor
   *   (ANIRA_ERROR_INVALID_ARGUMENT for a flags bit the header does not define,
   *   ANIRA_ERROR_INVALID_STATE when the pipeline already has a processor). A failed status
   *   throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_pipeline_add_js_processor(
    pipeline: number,
    js_key: number,
    hook_mask: number,
    processor_flags: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_pipeline_add_js_processor(
          pipeline,
          js_key,
          hook_mask,
          processor_flags,
          err
        ),
        'anira_em_pipeline_add_js_processor',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Fills an engine descriptor whose process (always), reset and query (per bit) are
   * trampolines into the scope registry's engine, and whose release frees the record
   * behind user_data without crossing into JavaScript. The init, load, unload, prepare
   * and unprepare slots stay NULL: a JavaScript engine is ready when its Worker's
   * registry booted (its init resolved), before the Worker enters its loop. flags is
   * engine_flags; providers stay empty (the CPU path alone), for a C caller to fill
   * before anira_custom_engine_create. A descriptor that is never created keeps its
   * record.
   *
   * @param desc The caller's descriptor, overwritten from ANIRA_ENGINE_DESC_INIT.
   * @param js_key The key the hook receives, the engine's entry in the instance's
   *   registry.
   * @param hook_mask An OR of the ANIRA_EM_ENGINE_HOOK_* bits; process crosses without
   *   one.
   * @param engine_flags The descriptor's flags, the engine's promises, which JavaScript
   *   cannot write into the descriptor itself: an OR of ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
   *   ANIRA_ENGINE_FLAG_REALTIME_SAFE, ANIRA_ENGINE_FLAG_DYNAMIC_TIME and
   *   ANIRA_ENGINE_FLAG_STATE_ALIAS, or 0 for none. Stored as given: a bit the header does
   *   not define is refused by anira_custom_engine_create.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL desc or a bit hook_mask
   *   does not define; ANIRA_ERROR_OUT_OF_MEMORY when the record cannot be allocated. A
   *   failed status throws an AniraError.
   * Thread: [main-thread]
   */
  anira_em_engine_desc_init(
    desc: number,
    js_key: number,
    hook_mask: number,
    engine_flags: number
  ): number {
    const m = this.module
    return checkStatus(
      m,
      m._anira_em_engine_desc_init(desc, js_key, hook_mask, engine_flags),
      'anira_em_engine_desc_init'
    )
  }

  /**
   * anira_em_engine_desc_init with engine_flags, anira_custom_engine_create under
   * engine_id, anira_pipeline_add_engine and anira_custom_engine_destroy of the handle:
   * the pipeline holds the JavaScript engine, and model entries name it as
   * ANIRA_ENGINE_CUSTOM with engine_id. The descriptor has no providers list, so the
   * engine serves ANIRA_PROVIDER_CPU alone and a query bit has nothing to answer; an
   * engine with providers is built with anira_em_engine_desc_init. A refused create frees
   * the record; a refused addition destroys the handle, which releases the engine and
   * frees it.
   *
   * @param pipeline The pipeline.
   * @param engine_id The engine's reverse-URI id, as anira_custom_engine_create takes it;
   *   copied.
   * @param js_key As in anira_em_engine_desc_init.
   * @param hook_mask As in anira_em_engine_desc_init.
   * @param engine_flags The descriptor's flags: an OR of
   *   ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, ANIRA_ENGINE_FLAG_REALTIME_SAFE,
   *   ANIRA_ENGINE_FLAG_DYNAMIC_TIME and ANIRA_ENGINE_FLAG_STATE_ALIAS, which JavaScript
   *   cannot write into the descriptor itself.
   * @returns As anira_em_engine_desc_init, then as anira_custom_engine_create
   *   (ANIRA_ERROR_INVALID_ARGUMENT for an id without a '.', with the prefix "anira." or a
   *   flags bit the header does not define), then as anira_pipeline_add_engine
   *   (ANIRA_ERROR_INVALID_STATE when the pipeline already has an engine under the id). A
   *   failed status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_pipeline_add_js_engine(
    pipeline: number,
    engine_id: string,
    js_key: number,
    hook_mask: number,
    engine_flags: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const engine_id_ = engine_id === null ? 0 : m.stringToUTF8OnStack(engine_id)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_pipeline_add_js_engine(
          pipeline,
          engine_id_,
          js_key,
          hook_mask,
          engine_flags,
          err
        ),
        'anira_em_pipeline_add_js_engine',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Registers the scope registries' log as a log sink of this module, beside the C sinks
   * of the contexts (anira_context_config_set_log_sink works on WebAssembly too): every
   * record at any level the core logs reaches the registry of the scope that logs or
   * drains (anira_drain_log), as an anira_log_record pointer it reads through layout.ts
   * while it runs. One registration for the whole module: the threads share one memory
   * and so one sink list. While registered the hook takes the place of the platform sink,
   * which on WebAssembly is the console by level (console.debug, .info, .warn, .error;
   * tanh-lib's own stderr sink stays off), so a record reaches the console once: the same
   * hold on the platform sink a context's ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK takes,
   * released with the hook.
   *
   * @param enable ANIRA_TRUE registers the hook as a log sink, ANIRA_FALSE removes it; a
   *   repeated call changes nothing.
   * Thread: [main-thread]
   */
  anira_em_set_log_hook(enable: boolean): void {
    const m = this.module
    m._anira_em_set_log_hook(enable ? 1 : 0)
  }

  /**
   * The one web-only field of a context config: the ES module every inference Worker
   * imports before it enters its loop. The context's pool threads are Wasm Workers (as
   * natively they are OS threads), and anira_inference_thread_start starts one for a
   * user-driven thread; each imports this module first, which defines the JavaScript
   * engines and processors of its scope on the registry
   * globalThis[Symbol.for('anira.registry')] (the package's defineEngine and
   * defineProcessor), and awaits the registry's boot (the engines' async init) before the
   * loop. A module that fails to load is reported on the console and the Worker runs
   * without it. Like the wait strategy, the first context's module is the one in effect
   * for the process's pool; a later context naming another is warned about. Not part of
   * the context config's JSON.
   *
   * @param config The config.
   * @param url The absolute URL of an ES module, copied; NULL or empty for none.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config. A failed status
   *   throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_context_config_set_worker_module(config: number, url: string | null): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const url_ = url === null ? 0 : m.stringToUTF8OnStack(url)
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_context_config_set_worker_module(config, url_, err),
        'anira_em_context_config_set_worker_module',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * The inference threads inside their loops now, the pool's and the user-driven ones
   * alike. A Wasm Worker anira starts enters its loop asynchronously, once the browser
   * has started it (the main thread must yield to its event loop first) and its worker
   * module is imported and booted: a host that waits for the inferences of a prepared
   * handler on the main thread (an offline render, anira_handler_process_wait) first
   * waits, yielding, until this reaches anira_num_inference_threads; natively the pool is
   * inside its loops by the time anira_handler_prepare returns.
   *
   * @returns The count.
   * Thread: [thread-safe] ANIRA_NONBLOCKING
   */
  anira_em_inference_threads_in_loop(): number {
    const m = this.module
    return m._anira_em_inference_threads_in_loop() >>> 0
  }

  /**
   * Creates a Wasm Audio Worklet node that streams through the handler, asynchronously:
   * the first node of a context starts the context's worklet thread (once; its stack
   * stays the context's), a node of n AudioParams registers the processor anira-node-n
   * once per context, and the node is created after both, from the main thread's event
   * loop; out_status turns from ANIRA_INCOMPLETE to the result. Per render quantum, on
   * the audio thread, the C render callback hands the node's buffers to the handler as
   * planar host tensors (no copy: the output tensor is the node's output), pushes an
   * AudioParam's value to its Static input when it changed and a tensor slot's value the
   * main thread wrote since the last push, and calls anira_handler_process_multi (or
   * anira_handler_process_multi_wait) once for every Streamed slot it serves, an empty
   * tensor at every other position, so a model with several streams keeps every guarantee
   * of the multi form (ANIRA_MISS_BYPASS passes the anchored input through to every
   * output, the wait credit of ANIRA_WAIT_CONTRACT, one anira_miss_fn call per block);
   * the miss rule: ANIRA_MISSED keeps the miss policy's fill, a failure is silence, a
   * success silences what an output was not delivered. Every Streamed slot of the model
   * is served by exactly one input or output of the node, unless
   * ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS leaves some out (their positions stay
   * empty). Driving one handler from several nodes, one slot each, is not the supported
   * form: the streams stay aligned, but a non-anchored call loses BYPASS (zeros on a
   * miss), the wait credit and the miss callback's view of the block. A quantum other
   * than the declared one is silence. A JavaScript processor of the handler runs its
   * worklet phases on the worklet scope's registry, which the caller fills before
   * (audioWorklet.addModule of a module that calls defineProcessor) and anira boots over
   * the worklet's instance at every create. The render callback allocates nothing and
   * locks nothing; nodes are at most 256 at once.
   *
   * @param handler A handler prepared under a Hard contract whose block range holds the
   *   context's render quantum; it must outlive the node (anira_em_audio_node_destroy
   *   first).
   * @param audio_context The audio context: what emscriptenRegisterAudioObject returned
   *   for an AudioContext or an OfflineAudioContext.
   * @param input_slots num_input_slots Streamed input slots, input k of the node feeding
   *   the k-th; each input has the slot's channels (the node's channelCount is the largest
   *   of them, explicit, discrete; a slot reads its first channels). NULL: every Streamed
   *   input slot of the model, in slot order.
   * @param num_input_slots Their count, at most 32; ignored for NULL. 0 with a non-NULL
   *   list: a node without an input.
   * @param output_slots num_output_slots Streamed output slots, output k of the node
   *   playing the k-th with the slot's channels. NULL: every Streamed output slot of the
   *   model, in slot order.
   * @param num_output_slots Their count, at most 32; ignored for NULL.
   * @param param_slots num_param_slots input slots of float32 Static scalars (rank 1, one
   *   element), each driven by the AudioParam of its index ('0', '1', ...; k-rate, default
   *   0); NULL for none.
   * @param num_param_slots Their count, at most 32.
   * @param tensor_slots num_tensor_slots Static input slots the main thread writes whole
   *   with anira_em_audio_node_set_static_input (any dtype, concrete extents); NULL for
   *   none.
   * @param num_tensor_slots Their count.
   * @param flags ANIRA_EM_AUDIO_NODE_FLAG_*; 0 for none.
   * @param wait_ms 0: every quantum calls anira_handler_process_multi (a live context);
   *   above 0: anira_handler_process_multi_wait with this bound (an offline render that
   *   waits for every inference).
   * @param out_status Written later: ANIRA_INCOMPLETE now, the creation's status once it
   *   completed (written last, with release ordering); the caller keeps the word alive
   *   until then.
   * @param out_node Written later, before out_status: the node's handle
   *   (emscriptenGetAudioObject gives the AudioWorkletNode).
   * @returns ANIRA_OK when the creation is under way; ANIRA_ERROR_INVALID_ARGUMENT for a
   *   NULL handler or out word, a slot list longer than 32 or without its pointer, a slot
   *   listed twice, a slot that is not a Streamed slot of its side with a float32 ring, a
   *   Streamed slot no input or output serves (without
   *   ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS), a node without any input or output, a slot
   *   of more than 64 channels; ANIRA_ERROR_INVALID_STATE for an unprepared handler;
   *   ANIRA_ERROR_CONFIG for a contract that is not Hard or whose block range excludes the
   *   quantum, a parameter slot that is not a float32 Static scalar, a tensor slot that is
   *   not Static or has a dynamic extent; ANIRA_ERROR_NOT_SUPPORTED when the context's
   *   worklet thread failed to start (later failures land in out_status:
   *   ANIRA_ERROR_NOT_SUPPORTED for a thread or processor that failed, ANIRA_ERROR_UNKNOWN
   *   for a node the browser refused); ANIRA_ERROR_OUT_OF_MEMORY. A failed status throws an
   *   AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_audio_node_create(
    handler: number,
    audio_context: number,
    input_slots: number,
    num_input_slots: number,
    output_slots: number,
    num_output_slots: number,
    param_slots: number,
    num_param_slots: number,
    tensor_slots: number,
    num_tensor_slots: number,
    flags: number,
    wait_ms: number,
    out_status: number,
    out_node: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_audio_node_create(
          handler,
          audio_context,
          input_slots,
          num_input_slots,
          output_slots,
          num_output_slots,
          param_slots,
          num_param_slots,
          tensor_slots,
          num_tensor_slots,
          flags,
          wait_ms,
          out_status,
          out_node,
          err
        ),
        'anira_em_audio_node_create',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Writes a Static input a node serves, for the render callback to push before its next
   * process call: a seqlock per slot (the main thread the one writer), so the audio
   * thread never pushes a torn value and never waits; a write it overlaps is pushed at a
   * later quantum, the handler keeping the previous value until then.
   *
   * @param node A node of anira_em_audio_node_create.
   * @param slot One of its tensor slots.
   * @param bytes The whole value, the spec's bytes in its layout; copied.
   * @param size Their count: exactly the spec's.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown node, a slot it
   *   does not serve, NULL bytes or a size other than the spec's. A failed status throws an
   *   AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_audio_node_set_static_input(
    node: number,
    slot: number,
    bytes: number,
    size: number
  ): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_audio_node_set_static_input(node, slot, bytes, size, err),
        'anira_em_audio_node_set_static_input',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Re-prepares the node's handler under contract while the node lives and its context
   * runs: the way to change the geometry of a running node (a new block_max after the
   * output device changed). The render callback is held off first: from its next quantum
   * it outputs silence and leaves the handler alone, and this call waits (at most 5000
   * ms, on the main thread; the callback never waits) until the quantum in flight
   * returned; then anira_handler_prepare's work runs, the node's state that depends on
   * the prepare is rebuilt off the audio thread (the handler's empty tensors; its Static
   * inputs are pushed again at the next quantum), and the callback resumes. The quanta in
   * between are silence. anira_handler_prepare itself refuses while a node streams
   * through the handler. Nothing allocates on the audio thread.
   *
   * @param node A node of anira_em_audio_node_create.
   * @param contract A Hard contract whose block range holds the context's render quantum;
   *   copied by the prepare.
   * @returns ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for an unknown node or a NULL
   *   contract; ANIRA_ERROR_CONFIG for a contract that is not Hard or whose block range
   *   excludes the render quantum (the handler untouched); ANIRA_ERROR_INVALID_STATE when
   *   the render callback did not return within the bound (the handler untouched);
   *   otherwise anira_handler_prepare's statuses, after which a failed prepare leaves the
   *   handler unprepared and the node silent until a prepare succeeds. A failed status
   *   throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_audio_node_prepare(node: number, contract: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_audio_node_prepare(node, contract, err),
        'anira_em_audio_node_prepare',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Holds the node's render callback off, or releases it: from its next quantum a held
   * node outputs silence and leaves its handler alone (nothing pushed, nothing pulled, no
   * inference started), as anira_em_audio_node_prepare holds it while it re-prepares;
   * released, it streams on from where it was held. For a quiet audio thread while the
   * page measures the context's render geometry. Does not wait for the quantum in flight.
   * A hold and a re-prepare do not release each other's hold.
   *
   * @param node A node of anira_em_audio_node_create.
   * @param held Nonzero holds the node, 0 releases it.
   * @returns ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown node. A failed
   *   status throws an AniraError with the entry's message.
   * Thread: [main-thread]
   */
  anira_em_audio_node_hold(node: number, held: number): number {
    const m = this.module
    const sp = m.stackSave()
    try {
      const err = stackError(m)
      return checkStatus(
        m,
        m._anira_em_audio_node_hold(node, held, err),
        'anira_em_audio_node_hold',
        err
      )
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Stops the node's render callback calling into the handler (a quantum inside it
   * finishes first, at most its wait bound), disconnects the node and releases its
   * handle; the callback outputs silence and lets the browser drop the node. Destroy the
   * nodes before the handler they stream through.
   *
   * @param node A node of anira_em_audio_node_create; an unknown one is a no-op.
   * Thread: [main-thread]
   */
  anira_em_audio_node_destroy(node: number): void {
    const m = this.module
    m._anira_em_audio_node_destroy(node)
  }
}
