// Generated from abi/anira.yml by tools/abi/gen.py; edit the registry, not this file.
// The C names are kept verbatim so a value can be searched across languages.

export const anira_status = {
  ANIRA_OK: 0,
  ANIRA_SUCCESS_UPGRADED: 1,
  ANIRA_INCOMPLETE: 2,
  ANIRA_TIMEOUT: 3,
  ANIRA_PENDING: 4,
  ANIRA_ERROR_UNKNOWN: -1,
  ANIRA_ERROR_INVALID_ARGUMENT: -2,
  ANIRA_ERROR_INVALID_STATE: -3,
  ANIRA_ERROR_OUT_OF_MEMORY: -4,
  ANIRA_ERROR_NOT_SUPPORTED: -5,
  ANIRA_ERROR_NO_SUCH_FILE: -6,
  ANIRA_ERROR_MODEL_LOAD: -7,
  ANIRA_ERROR_ENGINE: -8,
  ANIRA_ERROR_CONFIG: -9,
  ANIRA_ERROR_EXTENSION_UNCONSUMED: -10,
  ANIRA_ERROR_EXTENSION_UNKNOWN: -11,
  ANIRA_ERROR_EDGE_UNREACHABLE: -12,
  ANIRA_ERROR_BUDGET: -13,
  ANIRA_ERROR_CAPACITY: -14,
  ANIRA_ERROR_TICKET_STALE: -15,
  ANIRA_ERROR_WRONG_CONTRACT: -16,
  ANIRA_ERROR_NOT_PREPARED: -17,
  ANIRA_ERROR_JSON: -18,
  ANIRA_ERROR_ABI_VERSION: -19,
  ANIRA_ERROR_BUFFER_TOO_SMALL: -20,
  ANIRA_ERROR_DEVICE: -21,
  ANIRA_ERROR_EXTENSION_VERSION: -22,
  ANIRA_ERROR_INTERNAL: -100,
} as const
export type anira_status = (typeof anira_status)[keyof typeof anira_status]

export const anira_dtype_code = {
  ANIRA_DTYPE_INT: 0,
  ANIRA_DTYPE_UINT: 1,
  ANIRA_DTYPE_FLOAT: 2,
  ANIRA_DTYPE_OPAQUE: 3,
  ANIRA_DTYPE_BFLOAT: 4,
  ANIRA_DTYPE_COMPLEX: 5,
  ANIRA_DTYPE_BOOL: 6,
} as const
export type anira_dtype_code = (typeof anira_dtype_code)[keyof typeof anira_dtype_code]

export const anira_domain = {
  ANIRA_DOMAIN_HOST: 0,
  ANIRA_DOMAIN_HOST_PINNED: 1,
  ANIRA_DOMAIN_CUDA: 2,
  ANIRA_DOMAIN_GL_BUFFER: 3,
  ANIRA_DOMAIN_VULKAN_BUFFER: 4,
  ANIRA_DOMAIN_OPAQUE_FD: 5,
  ANIRA_DOMAIN_METAL_BUFFER: 6,
  ANIRA_DOMAIN_WGPU_BUFFER: 7,
  ANIRA_DOMAIN_DMABUF: 8,
  ANIRA_DOMAIN_IOSURFACE: 9,
  ANIRA_DOMAIN_AHARDWAREBUFFER: 10,
  ANIRA_DOMAIN_D3D12: 11,
  ANIRA_DOMAIN_FRAME: 12,
} as const
export type anira_domain = (typeof anira_domain)[keyof typeof anira_domain]

export const anira_sync_kind = {
  ANIRA_SYNC_NONE: 0,
  ANIRA_SYNC_QUEUE_ORDERED: 1,
  ANIRA_SYNC_CUDA_EVENT: 2,
  ANIRA_SYNC_VK_TIMELINE: 3,
  ANIRA_SYNC_GL_SYNC: 4,
  ANIRA_SYNC_SYNC_FILE_FD: 5,
  ANIRA_SYNC_OPAQUE_FD_SEMAPHORE: 6,
  ANIRA_SYNC_MTL_SHARED_EVENT: 7,
  ANIRA_SYNC_D3D12_FENCE: 8,
} as const
export type anira_sync_kind = (typeof anira_sync_kind)[keyof typeof anira_sync_kind]

export const anira_tensor_flags = {
  ANIRA_TENSOR_READ_ONLY: 1,
  ANIRA_TENSOR_DISCARD_CONTENTS: 2,
  ANIRA_TENSOR_HOST_COHERENT: 4,
} as const
export type anira_tensor_flags = (typeof anira_tensor_flags)[keyof typeof anira_tensor_flags]

export const anira_struct_id = {
  ANIRA_STRUCT_TENSOR: 1,
  ANIRA_STRUCT_SYNC_TOKEN: 2,
  ANIRA_STRUCT_MEMORY_HANDLE: 3,
  ANIRA_STRUCT_STAGE_CTX: 4,
  ANIRA_STRUCT_LOG_RECORD: 5,
  ANIRA_STRUCT_ERROR: 6,
  ANIRA_STRUCT_EDGE_INFO: 7,
  ANIRA_STRUCT_PLAN_SLOT: 8,
  ANIRA_STRUCT_PLAN_EXT: 9,
  ANIRA_STRUCT_PLAN_INFO: 10,
  ANIRA_STRUCT_BACKEND_ID: 11,
} as const
export type anira_struct_id = (typeof anira_struct_id)[keyof typeof anira_struct_id]

export const anira_container = {
  ANIRA_CONTAINER_DMABUF: 0,
  ANIRA_CONTAINER_AHARDWAREBUFFER: 1,
  ANIRA_CONTAINER_IOSURFACE: 2,
  ANIRA_CONTAINER_DXGI: 3,
  ANIRA_CONTAINER_VK_IMAGE: 4,
  ANIRA_CONTAINER_GL_TEXTURE: 5,
  ANIRA_CONTAINER_WGPU_TEXTURE: 6,
  ANIRA_CONTAINER_MTL_TEXTURE: 7,
  ANIRA_CONTAINER_HOST: 8,
} as const
export type anira_container = (typeof anira_container)[keyof typeof anira_container]

export const anira_pixel_format = {
  ANIRA_PIXEL_NV12: 842094158,
  ANIRA_PIXEL_YUYV: 1448695129,
  ANIRA_PIXEL_UYVY: 1498831189,
  ANIRA_PIXEL_R8: 538982482,
  ANIRA_PIXEL_RGBA8: 875708754,
  ANIRA_PIXEL_BGRA8: 875708738,
  ANIRA_PIXEL_XRGB8: 875713112,
} as const
export type anira_pixel_format = (typeof anira_pixel_format)[keyof typeof anira_pixel_format]

export const anira_color_matrix = {
  ANIRA_COLOR_BT601: 0,
  ANIRA_COLOR_BT709: 1,
  ANIRA_COLOR_BT2020: 2,
} as const
export type anira_color_matrix = (typeof anira_color_matrix)[keyof typeof anira_color_matrix]

export const anira_color_range = {
  ANIRA_RANGE_LIMITED: 0,
  ANIRA_RANGE_FULL: 1,
} as const
export type anira_color_range = (typeof anira_color_range)[keyof typeof anira_color_range]

export const anira_axis_tag = {
  ANIRA_AXIS_BATCH: 0,
  ANIRA_AXIS_CHANNEL: 1,
  ANIRA_AXIS_TIME: 2,
  ANIRA_AXIS_HEIGHT: 3,
  ANIRA_AXIS_WIDTH: 4,
  ANIRA_AXIS_FEATURE: 5,
  ANIRA_AXIS_ANY: 6,
} as const
export type anira_axis_tag = (typeof anira_axis_tag)[keyof typeof anira_axis_tag]

export const anira_role = {
  ANIRA_ROLE_STREAMED: 0,
  ANIRA_ROLE_BUFFER: 1,
  ANIRA_ROLE_STATIC: 2,
} as const
export type anira_role = (typeof anira_role)[keyof typeof anira_role]

export const anira_contract_kind = {
  ANIRA_CONTRACT_HARD: 1,
  ANIRA_CONTRACT_ASYNC: 2,
} as const
export type anira_contract_kind = (typeof anira_contract_kind)[keyof typeof anira_contract_kind]

export const anira_budget_kind = {
  ANIRA_BUDGET_MEASURED: 0,
  ANIRA_BUDGET_EXPLICIT: 1,
} as const
export type anira_budget_kind = (typeof anira_budget_kind)[keyof typeof anira_budget_kind]

export const anira_warmup_mode = {
  ANIRA_WARMUP_NONE: 0,
  ANIRA_WARMUP_FIXED: 1,
  ANIRA_WARMUP_UNTIL_STABLE: 2,
} as const
export type anira_warmup_mode = (typeof anira_warmup_mode)[keyof typeof anira_warmup_mode]

export const anira_miss_policy = {
  ANIRA_MISS_BYPASS: 0,
  ANIRA_MISS_HOLD_LAST: 1,
  ANIRA_MISS_ZEROS: 2,
} as const
export type anira_miss_policy = (typeof anira_miss_policy)[keyof typeof anira_miss_policy]

export const anira_late_policy = {
  ANIRA_LATE_FINISH: 0,
  ANIRA_LATE_DROP: 1,
} as const
export type anira_late_policy = (typeof anira_late_policy)[keyof typeof anira_late_policy]

export const anira_priority = {
  ANIRA_PRIORITY_AUTO: 0,
  ANIRA_PRIORITY_INTERACTIVE: 1,
  ANIRA_PRIORITY_BATCH: 2,
} as const
export type anira_priority = (typeof anira_priority)[keyof typeof anira_priority]

export const anira_delivery = {
  ANIRA_DELIVERY_POLLED: 0,
  ANIRA_DELIVERY_IMMEDIATE: 1,
} as const
export type anira_delivery = (typeof anira_delivery)[keyof typeof anira_delivery]

export const anira_edge_cost = {
  ANIRA_EDGE_COST_PERMISSIVE: 0,
  ANIRA_EDGE_COST_STRICT: 1,
} as const
export type anira_edge_cost = (typeof anira_edge_cost)[keyof typeof anira_edge_cost]

export const anira_ownership = {
  ANIRA_OWNERSHIP_BORROWED: 0,
  ANIRA_OWNERSHIP_OWNED: 1,
} as const
export type anira_ownership = (typeof anira_ownership)[keyof typeof anira_ownership]

export const anira_exec_policy = {
  ANIRA_EXEC_WORKER: 0,
  ANIRA_EXEC_USER_DRIVEN: 1,
} as const
export type anira_exec_policy = (typeof anira_exec_policy)[keyof typeof anira_exec_policy]

export const anira_gl_threads = {
  ANIRA_GL_CALLER_THREAD: 0,
  ANIRA_GL_SHARED_CONTEXT: 1,
} as const
export type anira_gl_threads = (typeof anira_gl_threads)[keyof typeof anira_gl_threads]

export const anira_wait_strategy = {
  ANIRA_WAIT_SPIN_BACKOFF: 0,
  ANIRA_WAIT_BLOCKING: 1,
} as const
export type anira_wait_strategy = (typeof anira_wait_strategy)[keyof typeof anira_wait_strategy]

export const anira_log_level = {
  ANIRA_LOG_DEBUG: 0,
  ANIRA_LOG_INFO: 1,
  ANIRA_LOG_WARNING: 2,
  ANIRA_LOG_ERROR: 3,
} as const
export type anira_log_level = (typeof anira_log_level)[keyof typeof anira_log_level]

export const anira_log_drain = {
  ANIRA_LOG_DRAIN_THREAD: 0,
  ANIRA_LOG_DRAIN_MANUAL: 1,
} as const
export type anira_log_drain = (typeof anira_log_drain)[keyof typeof anira_log_drain]

export const anira_edge_class = {
  ANIRA_EDGE_ZERO_COPY: 0,
  ANIRA_EDGE_DEVICE_COPY: 1,
  ANIRA_EDGE_HOST_COPY: 2,
  ANIRA_EDGE_UNAVAILABLE: 3,
} as const
export type anira_edge_class = (typeof anira_edge_class)[keyof typeof anira_edge_class]

export const anira_probe_rung = {
  ANIRA_RUNG_STATIC: 0,
  ANIRA_RUNG_IDENTITY: 1,
  ANIRA_RUNG_FUNCTIONAL: 2,
} as const
export type anira_probe_rung = (typeof anira_probe_rung)[keyof typeof anira_probe_rung]

export const anira_model_state = {
  ANIRA_MODEL_STATELESS: 0,
  ANIRA_MODEL_STATEFUL: 1,
} as const
export type anira_model_state = (typeof anira_model_state)[keyof typeof anira_model_state]

export const anira_bytes_ownership = {
  ANIRA_BYTES_COPY: 0,
  ANIRA_BYTES_BORROW: 1,
} as const
export type anira_bytes_ownership = (typeof anira_bytes_ownership)[keyof typeof anira_bytes_ownership]

export const anira_ticket_status = {
  ANIRA_TICKET_PENDING: 0,
  ANIRA_TICKET_MET: 1,
  ANIRA_TICKET_LATE: 2,
  ANIRA_TICKET_DROPPED: 3,
  ANIRA_TICKET_FAILED: 4,
} as const
export type anira_ticket_status = (typeof anira_ticket_status)[keyof typeof anira_ticket_status]

export const anira_pad_policy = {
  ANIRA_PAD_REJECT: 0,
  ANIRA_PAD_ZEROS: 1,
} as const
export type anira_pad_policy = (typeof anira_pad_policy)[keyof typeof anira_pad_policy]

export const anira_engine = {
  ANIRA_ENGINE_NONE: 0,
  ANIRA_ENGINE_ONNXRUNTIME: 1,
  ANIRA_ENGINE_LIBTORCH: 2,
  ANIRA_ENGINE_TFLITE: 3,
  ANIRA_ENGINE_LITERT: 4,
  ANIRA_ENGINE_EXECUTORCH: 5,
} as const
export type anira_engine = (typeof anira_engine)[keyof typeof anira_engine]

export const anira_provider = {
  ANIRA_PROVIDER_DEFAULT: 0,
  ANIRA_PROVIDER_CUDA: 1,
  ANIRA_PROVIDER_WEBGPU: 2,
  ANIRA_PROVIDER_DIRECTML: 3,
  ANIRA_PROVIDER_COREML: 4,
  ANIRA_PROVIDER_XNNPACK: 5,
  ANIRA_PROVIDER_VULKAN: 6,
} as const
export type anira_provider = (typeof anira_provider)[keyof typeof anira_provider]

export const anira_stage_phase = {
  ANIRA_PHASE_PRE_PROCESS: 0,
  ANIRA_PHASE_POST_PROCESS: 1,
  ANIRA_PHASE_BEFORE_INFERENCE: 2,
  ANIRA_PHASE_AFTER_INFERENCE: 3,
  ANIRA_PHASE_PREPARE: 4,
  ANIRA_PHASE_RELEASE: 5,
} as const
export type anira_stage_phase = (typeof anira_stage_phase)[keyof typeof anira_stage_phase]

export const ANIRA_FAILED = (s: number): boolean => (s | 0) < 0
export const ANIRA_SUCCEEDED = (s: number): boolean => (s | 0) >= 0
export const ANIRA_ERROR_MESSAGE_CAPACITY = 512
export const ANIRA_MAKE_ABI_VERSION = (major: number, minor: number): number => ((major << 16) | minor) >>> 0
export const ANIRA_ABI_VERSION_MAJOR = (v: number): number => (v >>> 16) & 0xffff
export const ANIRA_ABI_VERSION_MINOR = (v: number): number => v & 0xffff
export const ANIRA_MAKE_DTYPE = (code: number, bits: number, lanes: number): number => (code | (bits << 8) | (lanes << 16)) >>> 0
export const ANIRA_DTYPE_CODE = (d: number): number => d & 0xff
export const ANIRA_DTYPE_BITS = (d: number): number => (d >>> 8) & 0xff
export const ANIRA_DTYPE_LANES = (d: number): number => d >>> 16
export const ANIRA_DTYPE_F32 = 0x00012002
export const ANIRA_DTYPE_F16 = 0x00011002
export const ANIRA_DTYPE_BF16 = 0x00011004
export const ANIRA_DTYPE_I8 = 0x00010800
export const ANIRA_DTYPE_U8 = 0x00010801
export const ANIRA_DTYPE_I16 = 0x00011000
export const ANIRA_DTYPE_I32 = 0x00012000
export const ANIRA_DTYPE_I64 = 0x00014000
export const ANIRA_DTYPE_BOOL8 = 0x00010806
export const ANIRA_MAX_RANK = 8
export const ANIRA_DYNAMIC = -1n
export const ANIRA_UNBOUNDED = -1n
export const ANIRA_FOURCC = (a: number, b: number, c: number, d: number): number => (a | (b << 8) | (c << 16) | (d << 24)) >>> 0
export const ANIRA_THREADS_AUTO = 4294967295
export const ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK = 1
export const ANIRA_ANCHOR_FIRST_STREAMED = 4294967295
export const ANIRA_TICKET_INVALID = 0
export const ANIRA_LOG_RECORD_REALTIME = 1
export const ANIRA_LOG_RECORD_CONTRACT_VIOLATION = 2
