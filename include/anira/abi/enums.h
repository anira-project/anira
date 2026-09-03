/*
 * anira/abi/enums.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_ENUMS_H
#define ANIRA_ABI_ENUMS_H

/**
 * @file enums.h
 * @brief Every enum and pinned sentinel of the C ABI.
 *
 * Every value is pinned independently of build options; every enum is a C enum with explicit
 * values and a _FORCE32 = 0x7fffffff terminator, appears as the enum type in parameters and as
 * uint32_t in struct fields, and is exactly four bytes. Engine and execution provider are two
 * independent axes. Reserved ranges are stated per enum.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief Boolean at the boundary: 0 is false, any other value true.
 */
typedef uint32_t anira_bool;

/**
 * @brief Packed element type: code | bits << 8 | lanes << 16, whose little-endian bytes are
 * exactly DLPack's DLDataType {uint8 code; uint8 bits; uint16 lanes}.
 */
typedef uint32_t anira_dtype;

/**
 * @brief A job ticket: slot in the low 16 bits, generation in the high 16; a value type, never
 * a heap object.
 */
typedef uint32_t anira_ticket;

/**
 * @brief The DLPack type codes carried in the low byte of an anira_dtype.
 */
typedef enum anira_dtype_code {
    ANIRA_DTYPE_INT = 0,  /**< Signed integer. */
    ANIRA_DTYPE_UINT = 1,  /**< Unsigned integer. */
    ANIRA_DTYPE_FLOAT = 2,  /**< IEEE floating point. */
    ANIRA_DTYPE_OPAQUE = 3,  /**< Opaque handle. */
    ANIRA_DTYPE_BFLOAT = 4,  /**< bfloat. */
    ANIRA_DTYPE_COMPLEX = 5,  /**< Complex. */
    ANIRA_DTYPE_BOOL = 6,  /**< Boolean. */
    ANIRA_DTYPE_CODE_FORCE32 = 0x7fffffff
} anira_dtype_code;

/**
 * @brief Packs an anira_dtype from a DLPack code, a bit width and a lane count.
 */
#define ANIRA_MAKE_DTYPE(code, bits, lanes) ((uint32_t)(code) | ((uint32_t)(bits) << 8) | ((uint32_t)(lanes) << 16))

/**
 * @brief The anira_dtype_code of a packed dtype.
 */
#define ANIRA_DTYPE_CODE(d) ((d) & 0xffu)

/**
 * @brief The bit width of a packed dtype.
 */
#define ANIRA_DTYPE_BITS(d) (((d) >> 8) & 0xffu)

/**
 * @brief The lane count of a packed dtype.
 */
#define ANIRA_DTYPE_LANES(d) ((d) >> 16)

/**
 * @brief 32-bit float, one lane: the type of every v2 stream and the only one the Hard entries
 * carry.
 */
#define ANIRA_DTYPE_F32 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 32, 1)

/**
 * @brief 16-bit float.
 */
#define ANIRA_DTYPE_F16 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 16, 1)

/**
 * @brief bfloat16.
 */
#define ANIRA_DTYPE_BF16 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_BFLOAT, 16, 1)

/**
 * @brief Signed 8-bit integer.
 */
#define ANIRA_DTYPE_I8 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_INT, 8, 1)

/**
 * @brief Unsigned 8-bit integer.
 */
#define ANIRA_DTYPE_U8 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_UINT, 8, 1)

/**
 * @brief Signed 16-bit integer.
 */
#define ANIRA_DTYPE_I16 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_INT, 16, 1)

/**
 * @brief Signed 32-bit integer.
 */
#define ANIRA_DTYPE_I32 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_INT, 32, 1)

/**
 * @brief Signed 64-bit integer.
 */
#define ANIRA_DTYPE_I64 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_INT, 64, 1)

/**
 * @brief 8-bit boolean.
 */
#define ANIRA_DTYPE_BOOL8 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_BOOL, 8, 1)

/**
 * @brief Maximum tensor rank: the axis arrays of anira_tensor and anira_tensor_spec hold this
 * many entries.
 */
#define ANIRA_MAX_RANK 8

/**
 * @brief A spec extent that is decided at run time; legal on the Time axis of a Streamed or
 * Buffer spec only.
 */
#define ANIRA_DYNAMIC ((int64_t)-1)

/**
 * @brief window_max without an upper limit.
 */
#define ANIRA_UNBOUNDED ((int64_t)-1)

/**
 * @brief Where a tensor's bytes live. Values 13..63 are reserved for later domains; every arm
 * named here is pinned now even where no engine reads it before M4.
 */
typedef enum anira_domain {
    ANIRA_DOMAIN_HOST = 0,  /**< Pageable host memory. */
    ANIRA_DOMAIN_HOST_PINNED = 1,  /**< Page-locked host memory. */
    ANIRA_DOMAIN_CUDA = 2,  /**< CUDA device memory. */
    ANIRA_DOMAIN_GL_BUFFER = 3,  /**< An OpenGL buffer object. */
    ANIRA_DOMAIN_VULKAN_BUFFER = 4,  /**< A VkBuffer with its VkDeviceMemory. */
    ANIRA_DOMAIN_OPAQUE_FD = 5,  /**< Exported opaque memory (fd, or the NT handle on Windows). */
    ANIRA_DOMAIN_METAL_BUFFER = 6,  /**< An id<MTLBuffer>. */
    ANIRA_DOMAIN_WGPU_BUFFER = 7,  /**< A WGPUBuffer of the machine's Dawn device. */
    ANIRA_DOMAIN_DMABUF = 8,  /**< Exported buffer memory as a dma-buf. */
    ANIRA_DOMAIN_IOSURFACE = 9,  /**< An IOSurfaceRef, plane 0, byte-image encoded. */
    ANIRA_DOMAIN_AHARDWAREBUFFER = 10,  /**< An AHardwareBuffer* (BLOB). */
    ANIRA_DOMAIN_D3D12 = 11,  /**< An ID3D12Resource*. */
    /**
     * Reserved: the Frame side of a FrameToTensor edge (section 1a); nothing consumes it at
     * v3.0.0.
     */
    ANIRA_DOMAIN_FRAME = 12,
    ANIRA_DOMAIN_FORCE32 = 0x7fffffff
} anira_domain;

/**
 * @brief The kind of fence an anira_sync_token carries; NONE means the data is already visible.
 */
typedef enum anira_sync_kind {
    ANIRA_SYNC_NONE = 0,  /**< No fence: the data is visible. */
    /**
     * Complete when the producing queue reaches it; a same-queue consumer waits for nothing.
     */
    ANIRA_SYNC_QUEUE_ORDERED = 1,
    ANIRA_SYNC_CUDA_EVENT = 2,  /**< A cudaEvent_t. */
    ANIRA_SYNC_VK_TIMELINE = 3,  /**< A Vulkan timeline semaphore plus value. */
    ANIRA_SYNC_GL_SYNC = 4,  /**< A GLsync. */
    /**
     * A sync_file fd (dma-buf / AHardwareBuffer world); owned by the token.
     */
    ANIRA_SYNC_SYNC_FILE_FD = 5,
    /**
     * A Vulkan semaphore exported OPAQUE_FD, the only fence CUDA imports; owned by the token.
     */
    ANIRA_SYNC_OPAQUE_FD_SEMAPHORE = 6,
    ANIRA_SYNC_MTL_SHARED_EVENT = 7,  /**< An id<MTLSharedEvent> plus value. */
    ANIRA_SYNC_D3D12_FENCE = 8,  /**< An ID3D12Fence* plus value. */
    ANIRA_SYNC_KIND_FORCE32 = 0x7fffffff
} anira_sync_kind;

/**
 * @brief Bit flags of anira_tensor::flags.
 */
typedef enum anira_tensor_flags {
    ANIRA_TENSOR_READ_ONLY = 1,  /**< The producer will not write; anira must not either. */
    ANIRA_TENSOR_DISCARD_CONTENTS = 2,  /**< The previous contents need not be preserved. */
    ANIRA_TENSOR_HOST_COHERENT = 4,  /**< Host writes are visible without an explicit flush. */
    ANIRA_TENSOR_FLAGS_FORCE32 = 0x7fffffff
} anira_tensor_flags;

/**
 * @brief Identifies a record for anira_sizeof (M2): the six Tier-1 PODs and the enumerated
 * Tier-2 records. Ids 0x0001xxxx are reserved for extension payloads, 0x0004xxxx for
 * Emscripten-only structs. The document pins 1..3 and 7..11; 4..6 follow its Tier-1
 * table order.
 */
typedef enum anira_struct_id {
    ANIRA_STRUCT_TENSOR = 1,  /**< anira_tensor (216 bytes). */
    ANIRA_STRUCT_SYNC_TOKEN = 2,  /**< anira_sync_token (24 bytes). */
    ANIRA_STRUCT_MEMORY_HANDLE = 3,  /**< anira_memory_handle (24 bytes). */
    ANIRA_STRUCT_STAGE_CTX = 4,  /**< anira_stage_ctx (64 bytes). */
    ANIRA_STRUCT_LOG_RECORD = 5,  /**< anira_log_record (56 bytes). */
    ANIRA_STRUCT_ERROR = 6,  /**< anira_error (520 bytes). */
    ANIRA_STRUCT_EDGE_INFO = 7,  /**< anira_edge_info (Tier 2). */
    ANIRA_STRUCT_PLAN_SLOT = 8,  /**< anira_plan_slot (Tier 2). */
    ANIRA_STRUCT_PLAN_EXT = 9,  /**< anira_plan_ext (Tier 2). */
    ANIRA_STRUCT_PLAN_INFO = 10,  /**< anira_plan_info (Tier 2). */
    ANIRA_STRUCT_BACKEND_ID = 11,  /**< anira_backend_id (Tier 2). */
    ANIRA_STRUCT_ID_FORCE32 = 0x7fffffff
} anira_struct_id;

/**
 * @brief The container of a Frame (section 1a; values reserved at v3.0.0, consumed by
 * abi/draft/frame.h).
 */
typedef enum anira_container {
    ANIRA_CONTAINER_DMABUF = 0,  /**< dma-buf planes. */
    ANIRA_CONTAINER_AHARDWAREBUFFER = 1,  /**< AHardwareBuffer. */
    ANIRA_CONTAINER_IOSURFACE = 2,  /**< IOSurface / CVPixelBuffer. */
    ANIRA_CONTAINER_DXGI = 3,  /**< DXGI shared resource. */
    ANIRA_CONTAINER_VK_IMAGE = 4,  /**< VkImage. */
    ANIRA_CONTAINER_GL_TEXTURE = 5,  /**< GL texture. */
    ANIRA_CONTAINER_WGPU_TEXTURE = 6,  /**< WGPUTexture. */
    ANIRA_CONTAINER_MTL_TEXTURE = 7,  /**< id<MTLTexture>. */
    ANIRA_CONTAINER_HOST = 8,  /**< Host planes. */
    ANIRA_CONTAINER_FORCE32 = 0x7fffffff
} anira_container;

/**
 * @brief A DRM-style fourcc code from four characters.
 */
#define ANIRA_FOURCC(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

/**
 * @brief Pixel formats of a Frame; every value is the literal DRM fourcc constant of the format
 * the name abbreviates.
 */
typedef enum anira_pixel_format {
    ANIRA_PIXEL_NV12 = 0x3231564e,  /**< ANIRA_FOURCC('N','V','1','2'). */
    ANIRA_PIXEL_YUYV = 0x56595559,  /**< ANIRA_FOURCC('Y','U','Y','V'). */
    ANIRA_PIXEL_UYVY = 0x59565955,  /**< ANIRA_FOURCC('U','Y','V','Y'). */
    ANIRA_PIXEL_R8 = 0x20203852,  /**< ANIRA_FOURCC('R','8',' ',' '). */
    ANIRA_PIXEL_RGBA8 = 0x34324152,  /**< ANIRA_FOURCC('R','A','2','4'). */
    ANIRA_PIXEL_BGRA8 = 0x34324142,  /**< ANIRA_FOURCC('B','A','2','4'). */
    ANIRA_PIXEL_XRGB8 = 0x34325258,  /**< ANIRA_FOURCC('X','R','2','4'). */
    ANIRA_PIXEL_FORMAT_FORCE32 = 0x7fffffff
} anira_pixel_format;

/**
 * @brief YCbCr matrix of a Frame.
 */
typedef enum anira_color_matrix {
    ANIRA_COLOR_BT601 = 0,  /**< BT.601. */
    ANIRA_COLOR_BT709 = 1,  /**< BT.709. */
    ANIRA_COLOR_BT2020 = 2,  /**< BT.2020. */
    ANIRA_COLOR_MATRIX_FORCE32 = 0x7fffffff
} anira_color_matrix;

/**
 * @brief Sample range of a Frame.
 */
typedef enum anira_color_range {
    ANIRA_RANGE_LIMITED = 0,  /**< Limited (video) range. */
    ANIRA_RANGE_FULL = 1,  /**< Full range. */
    ANIRA_COLOR_RANGE_FORCE32 = 0x7fffffff
} anira_color_range;

/**
 * @brief The meaning of one axis of a tensor spec. Axis index order is model memory order (NCHW
 * vs NHWC is just axis order); chunkers find the Time axis by tag.
 */
typedef enum anira_axis_tag {
    ANIRA_AXIS_BATCH = 0,  /**< Batch. */
    ANIRA_AXIS_CHANNEL = 1,  /**< Channel. */
    ANIRA_AXIS_TIME = 2,  /**< Time: the axis a Streamed spec is consumed along. */
    ANIRA_AXIS_HEIGHT = 3,  /**< Image height. */
    ANIRA_AXIS_WIDTH = 4,  /**< Image width. */
    ANIRA_AXIS_FEATURE = 5,  /**< Feature. */
    ANIRA_AXIS_ANY = 6,  /**< No semantics. */
    ANIRA_AXIS_TAG_FORCE32 = 0x7fffffff
} anira_axis_tag;

/**
 * @brief How a tensor spec is fed and drained.
 */
typedef enum anira_role {
    ANIRA_ROLE_STREAMED = 0,  /**< Has a Time axis consumed window-wise (inputs and outputs). */
    /**
     * The whole submitted buffer is one model tensor, no Time axis (frames, images).
     */
    ANIRA_ROLE_BUFFER = 1,
    /**
     * No time semantics: conditioning in, scalar or embedding out; one value per job.
     */
    ANIRA_ROLE_STATIC = 2,
    ANIRA_ROLE_FORCE32 = 0x7fffffff
} anira_role;

/**
 * @brief The scheduling regime of a contract.
 */
typedef enum anira_contract_kind {
    /**
     * Real-time: the host callback drives fixed or ranged blocks; the v2 regime.
     */
    ANIRA_CONTRACT_HARD = 1,
    ANIRA_CONTRACT_ASYNC = 2,  /**< Asynchronous: jobs with tickets, an optional deadline. */
    ANIRA_CONTRACT_KIND_FORCE32 = 0x7fffffff
} anira_contract_kind;

/**
 * @brief How a Hard contract's per-inference budget is set.
 */
typedef enum anira_budget_kind {
    ANIRA_BUDGET_MEASURED = 0,  /**< Derived during warmup (default). */
    /**
     * The explicit_ms of anira_contract_hard_set_budget; the v2 max_inference_time.
     */
    ANIRA_BUDGET_EXPLICIT = 1,
    ANIRA_BUDGET_KIND_FORCE32 = 0x7fffffff
} anira_budget_kind;

/**
 * @brief Warmup policy of a Hard contract.
 */
typedef enum anira_warmup_mode {
    ANIRA_WARMUP_NONE = 0,  /**< No warmup; legal only with an explicit budget. */
    ANIRA_WARMUP_FIXED = 1,  /**< A fixed number of iterations; the v2 warm_up. */
    ANIRA_WARMUP_UNTIL_STABLE = 2,  /**< Until the measured time stabilises (default). */
    ANIRA_WARMUP_MODE_FORCE32 = 0x7fffffff
} anira_warmup_mode;

/**
 * @brief What a Hard contract delivers when an inference misses its deadline.
 */
typedef enum anira_miss_policy {
    /**
     * Pass the input through (default); requires shape-compatible I/O along the anchored Time
     * axis.
     */
    ANIRA_MISS_BYPASS = 0,
    ANIRA_MISS_HOLD_LAST = 1,  /**< Repeat the last output. */
    ANIRA_MISS_ZEROS = 2,  /**< Deliver zeros. */
    ANIRA_MISS_POLICY_FORCE32 = 0x7fffffff
} anira_miss_policy;

/**
 * @brief What an Async contract does with a job past its deadline.
 */
typedef enum anira_late_policy {
    ANIRA_LATE_FINISH = 0,  /**< Finish it anyway (default). */
    ANIRA_LATE_DROP = 1,  /**< Cancel at chunk boundaries; enables admission control at dispatch. */
    ANIRA_LATE_POLICY_FORCE32 = 0x7fffffff
} anira_late_policy;

/**
 * @brief Scheduling priority of Async jobs.
 */
typedef enum anira_priority {
    ANIRA_PRIORITY_AUTO = 0,  /**< INTERACTIVE iff a deadline is set, else BATCH (default). */
    ANIRA_PRIORITY_INTERACTIVE = 1,  /**< Interactive. */
    ANIRA_PRIORITY_BATCH = 2,  /**< Batch. */
    ANIRA_PRIORITY_FORCE32 = 0x7fffffff
} anira_priority;

/**
 * @brief Where an Async completion callback runs.
 */
typedef enum anira_delivery {
    ANIRA_DELIVERY_POLLED = 0,  /**< In the caller of anira_handler_poll / ticket_wait (default). */
    ANIRA_DELIVERY_IMMEDIATE = 1,  /**< On the inference thread. */
    ANIRA_DELIVERY_FORCE32 = 0x7fffffff
} anira_delivery;

/**
 * @brief Plan validation policy for the edges a pipeline uses (section 7); not scheduling.
 */
typedef enum anira_edge_cost {
    ANIRA_EDGE_COST_PERMISSIVE = 0,  /**< Any available edge (default). */
    ANIRA_EDGE_COST_STRICT = 1,  /**< Zero-copy edges only. */
    ANIRA_EDGE_COST_FORCE32 = 0x7fffffff
} anira_edge_cost;

/**
 * @brief timeout_ms sentinel of the _wait twins: wait without limit (the v2 set_non_realtime
 * behaviour).
 */
#define ANIRA_WAIT_FOREVER (-1.0)

/**
 * @brief timeout_ms sentinel of the _wait twins: wait_ratio times the block duration (the v2
 * blocking_ratio).
 */
#define ANIRA_WAIT_CONTRACT (-2.0)

/**
 * @brief Who owns the device a descriptor names.
 */
typedef enum anira_ownership {
    ANIRA_OWNERSHIP_BORROWED = 0,  /**< The user's handles; anira never destroys them. */
    ANIRA_OWNERSHIP_OWNED = 1,  /**< anira creates and destroys the device. */
    ANIRA_OWNERSHIP_FORCE32 = 0x7fffffff
} anira_ownership;

/**
 * @brief Who pumps a WebGPU device (ProcessEvents / WaitAny).
 */
typedef enum anira_exec_policy {
    ANIRA_EXEC_WORKER = 0,  /**< anira's worker (default). */
    ANIRA_EXEC_USER_DRIVEN = 1,  /**< The user pumps. */
    ANIRA_EXEC_POLICY_FORCE32 = 0x7fffffff
} anira_exec_policy;

/**
 * @brief How anira reaches a GL context.
 */
typedef enum anira_gl_threads {
    /**
     * Only inside allocate_*, submit and bind_output, on the calling thread where the user's
     * context is current (default).
     */
    ANIRA_GL_CALLER_THREAD = 0,
    /**
     * A second context of the same share group that anira's worker makes current (additive).
     */
    ANIRA_GL_SHARED_CONTEXT = 1,
    ANIRA_GL_THREADS_FORCE32 = 0x7fffffff
} anira_gl_threads;

/**
 * @brief How idle inference threads wait; the v2 WaitStrategy.
 */
typedef enum anira_wait_strategy {
    ANIRA_WAIT_SPIN_BACKOFF = 0,  /**< Spin with backoff (default). */
    ANIRA_WAIT_BLOCKING = 1,  /**< Block on a semaphore. */
    ANIRA_WAIT_STRATEGY_FORCE32 = 0x7fffffff
} anira_wait_strategy;

/**
 * @brief Log severity; the four v2 levels, numerically as the engines count them.
 */
typedef enum anira_log_level {
    ANIRA_LOG_DEBUG = 0,  /**< Debug. */
    ANIRA_LOG_INFO = 1,  /**< Info. */
    ANIRA_LOG_WARNING = 2,  /**< Warning (the default level). */
    ANIRA_LOG_ERROR = 3,  /**< Error. */
    ANIRA_LOG_LEVEL_FORCE32 = 0x7fffffff
} anira_log_level;

/**
 * @brief Who drains the real-time log queue.
 */
typedef enum anira_log_drain {
    /**
     * anira's low-priority drain thread, every drain_interval_ms (default).
     */
    ANIRA_LOG_DRAIN_THREAD = 0,
    ANIRA_LOG_DRAIN_MANUAL = 1,  /**< The host, through anira_drain_log; forced on Wasm. */
    ANIRA_LOG_DRAIN_FORCE32 = 0x7fffffff
} anira_log_drain;

/**
 * @brief Cost class of an edge between a domain and a backend (section 7).
 */
typedef enum anira_edge_class {
    ANIRA_EDGE_ZERO_COPY = 0,  /**< No copy. */
    ANIRA_EDGE_DEVICE_COPY = 1,  /**< A device-side copy. */
    ANIRA_EDGE_HOST_COPY = 2,  /**< A copy through host memory. */
    ANIRA_EDGE_UNAVAILABLE = 3,  /**< No edge. */
    ANIRA_EDGE_CLASS_FORCE32 = 0x7fffffff
} anira_edge_class;

/**
 * @brief The rung a machine probe reached for an edge (section 4).
 */
typedef enum anira_probe_rung {
    ANIRA_RUNG_STATIC = 0,  /**< Compiled-in knowledge only. */
    ANIRA_RUNG_IDENTITY = 1,  /**< The device identified itself. */
    ANIRA_RUNG_FUNCTIONAL = 2,  /**< A round trip succeeded. */
    ANIRA_RUNG_FORCE32 = 0x7fffffff
} anira_probe_rung;

/**
 * @brief num_threads sentinel: the library's default pool size; 0 means bring your own threads.
 */
#define ANIRA_THREADS_AUTO 0xffffffffu

/**
 * @brief Log flag: switch the platform sink (stderr, logcat, os_log) off while this machine
 * lives.
 */
#define ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK 1u

/**
 * @brief Log flag: every failed status of every C entry also emits one Error record whose text
 * is the anira_error message prefixed by the entry and the status (the boundary trace,
 * for an application that swallowed the status and a developer who only has the device
 * log). Off by default on every platform. The machine config stores the flag; the
 * machine of the 3.x runtime applies it, in this pre-release the process-wide switch is
 * anira::capi::set_trace_failures.
 */
#define ANIRA_LOG_FLAG_TRACE_FAILURES 2u

/**
 * @brief Whether a model carries state across inferences.
 */
typedef enum anira_model_state {
    ANIRA_MODEL_STATELESS = 0,  /**< Stateless (default). */
    /**
     * Stateful: session-exclusive, lanes forced to 1; the v2 session_exclusive_processor.
     */
    ANIRA_MODEL_STATEFUL = 1,
    ANIRA_MODEL_STATE_FORCE32 = 0x7fffffff
} anira_model_state;

/**
 * @brief Whether model bytes handed to a config are copied or borrowed.
 */
typedef enum anira_bytes_ownership {
    ANIRA_BYTES_COPY = 0,  /**< Copied into the config. */
    /**
     * Borrowed until the last carrier dies (the plugin default for embedded blobs); the release
     * callback fires once.
     */
    ANIRA_BYTES_BORROW = 1,
    ANIRA_BYTES_OWNERSHIP_FORCE32 = 0x7fffffff
} anira_bytes_ownership;

/**
 * @brief In the axes of anira_model_config_set_tensor_layout: an axis of extent 1 that the
 * entry's file has and the spec does not (a TensorFlow export's rank-3 scalar against a
 * rank-1 spec).
 */
#define ANIRA_AXIS_INSERT 0xffffffffu

/**
 * @brief The state of an Async job.
 */
typedef enum anira_ticket_status {
    ANIRA_TICKET_PENDING = 0,  /**< Not yet terminal. */
    ANIRA_TICKET_MET = 1,  /**< Completed within the deadline. */
    ANIRA_TICKET_LATE = 2,  /**< Completed past the deadline. */
    ANIRA_TICKET_DROPPED = 3,  /**< Cancelled by the late policy. */
    ANIRA_TICKET_FAILED = 4,  /**< Failed; anira_handler_ticket_error has the reason. */
    ANIRA_TICKET_STATUS_FORCE32 = 0x7fffffff
} anira_ticket_status;

/**
 * @brief What happens to a submitted buffer shorter than the window minimum.
 */
typedef enum anira_pad_policy {
    ANIRA_PAD_REJECT = 0,  /**< Refuse the job (default). */
    ANIRA_PAD_ZEROS = 1,  /**< Pad with zeros. */
    ANIRA_PAD_POLICY_FORCE32 = 0x7fffffff
} anira_pad_policy;

/**
 * @brief The ticket value that names no job; every failed submit writes it.
 */
#define ANIRA_TICKET_INVALID 0u

/**
 * @brief Inference engine, one of the two independent backend axes. Values 6..0x0fff are
 * reserved for later anira engines; a registered custom engine is assigned a value from
 * 0x1000 up at prepare, scoped to its pipeline.
 */
typedef enum anira_engine {
    ANIRA_ENGINE_NONE = 0,  /**< No engine; as a default engine it means models[0]. */
    ANIRA_ENGINE_ONNXRUNTIME = 1,  /**< ONNX Runtime (v2 ONNX, JSON "onnxruntime"). */
    ANIRA_ENGINE_LIBTORCH = 2,  /**< LibTorch (JSON "libtorch"). */
    ANIRA_ENGINE_TFLITE = 3,  /**< TensorFlow Lite, legacy C API (JSON "tflite"). */
    ANIRA_ENGINE_LITERT = 4,  /**< LiteRT (JSON "litert"). */
    ANIRA_ENGINE_EXECUTORCH = 5,  /**< ExecuTorch (JSON "executorch"). */
    ANIRA_ENGINE_FORCE32 = 0x7fffffff
} anira_engine;

/**
 * @brief Execution provider, the other backend axis; a provider name means the same thing
 * across engines.
 */
typedef enum anira_provider {
    ANIRA_PROVIDER_DEFAULT = 0,  /**< The engine's own CPU path (JSON: no suffix). */
    ANIRA_PROVIDER_CUDA = 1,  /**< CUDA (JSON ":cuda"). */
    ANIRA_PROVIDER_WEBGPU = 2,  /**< WebGPU (JSON ":webgpu"). */
    ANIRA_PROVIDER_DIRECTML = 3,  /**< DirectML (JSON ":directml"). */
    ANIRA_PROVIDER_COREML = 4,  /**< Core ML (JSON ":coreml"). */
    ANIRA_PROVIDER_XNNPACK = 5,  /**< XNNPACK (JSON ":xnnpack"). */
    ANIRA_PROVIDER_VULKAN = 6,  /**< Vulkan (JSON ":vulkan"). */
    ANIRA_PROVIDER_FORCE32 = 0x7fffffff
} anira_provider;

/**
 * @brief The phase a stage callback runs in (abi/stage.h, M2); pinned now.
 */
typedef enum anira_stage_phase {
    ANIRA_PHASE_PRE_PROCESS = 0,  /**< Before the model inputs are formed. */
    ANIRA_PHASE_POST_PROCESS = 1,  /**< After the model outputs arrive. */
    ANIRA_PHASE_BEFORE_INFERENCE = 2,  /**< On the inference thread, before the engine call. */
    ANIRA_PHASE_AFTER_INFERENCE = 3,  /**< On the inference thread, after the engine call. */
    ANIRA_PHASE_PREPARE = 4,  /**< At anira_handler_prepare. */
    ANIRA_PHASE_RELEASE = 5,  /**< When the last carrier dies. */
    ANIRA_STAGE_PHASE_FORCE32 = 0x7fffffff
} anira_stage_phase;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_ENUMS_H */
