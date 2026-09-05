/*
 * anira/abi/config.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_CONFIG_H
#define ANIRA_ABI_CONFIG_H

/**
 * @file config.h
 * @brief The configuration handles: tensor specs, model, context, contract and job options, their scalar setters, the device descriptors and the extension slots.
 *
 * Every entry is [main-thread] and may allocate; a rejected value is ANIRA_FAILED(status), with
 * anira_error::message filled where the entry takes one, and the handle is left as it was. The
 * handle layouts never enter the ABI. Extensions (section 1b) arrive through a set_ext /
 * set_ext_json pair on every handle, one slot per kind, a second set of the same kind replacing
 * the first: a known kind at a registered version is deep-copied during the call, a known kind
 * at an unregistered version is ANIRA_ERROR_EXTENSION_VERSION, an unknown kind is stored and
 * fails prepare by name (ANIRA_ERROR_EXTENSION_UNKNOWN). Strings in are UTF-8, NUL-terminated
 * and copied; strings out are owned by the handle and valid until it is destroyed or mutated.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief The first member of every extension payload: the payload's size, the revision of the
 * kind's layout and the kind, a stable string id that doubles as the JSON key (anira's
 * bare kinds such as "entry"; third-party kinds carry a reverse-URI prefix).
 */
typedef struct anira_ext_header {
    uint32_t struct_size;  /**< sizeof the payload struct of the caller's header. */
    uint32_t version;  /**< Revision of this kind's layout; 1 for every kind anira ships at 3.0. */
    const char* kind;  /**< The kind, NUL-terminated; e.g. "entry". */
} anira_ext_header;

/**
 * @brief Extension "entry", version 1, on a model entry: the entry point a program is run
 * through (v2's model_function; absent means "forward"). Consumed by the LibTorch and
 * ExecuTorch adapters.
 */
typedef struct anira_ext_entry {
    anira_ext_header header;  /**< {sizeof(anira_ext_entry), 1, "entry"}. */
    const char* name;  /**< The entry point's name; copied by the set call. */
} anira_ext_entry;
/**
 * @brief An anira_ext_entry with its header filled and no name yet.
 */
#define ANIRA_EXT_ENTRY_INIT ANIRA_INIT(anira_ext_entry, {sizeof(anira_ext_entry), 1, "entry"}, NULL)

/**
 * @brief Scalar enumeration of the extension kinds this build understands, without a context:
 * NULL out returns the count, a short buffer is filled as far as it goes and returns
 * ANIRA_INCOMPLETE.
 * @param count In: the capacity of out; out: the number of registered kinds.
 * @param out Receives the kind names (static storage), or NULL to ask for the count only.
 * @return ANIRA_OK, ANIRA_INCOMPLETE, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL count.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_registered_ext_kinds(uint32_t* count,
                                                             const char** out) ANIRA_NOEXCEPT;

/**
 * @brief The CUDA device block of a context config. Nothing to hand over: the primary context
 * is process-wide, so a pointer, stream or event on it is anira's as much as the user's.
 */
typedef struct anira_cuda_desc {
    uint32_t struct_size;  /**< sizeof(anira_cuda_desc) of the caller's header. */
    uint32_t ownership;  /**< anira_ownership. */
    int32_t device;  /**< CUDA device ordinal. */
    uint32_t reserved;  /**< Zero. */
    uint64_t pinned_pool_limit;  /**< Cap on cudaHostAlloc staging in bytes; 0 = planner-sized. */
} anira_cuda_desc;
/**
 * @brief OWNED, device 0, planner-sized staging.
 */
#define ANIRA_CUDA_DESC_INIT ANIRA_INIT(anira_cuda_desc, sizeof(anira_cuda_desc), ANIRA_OWNERSHIP_OWNED, 0, 0u, 0u)

/**
 * @brief The OpenGL block of a context config; GL is always borrowed. CALLER_THREAD: anira
 * touches GL only inside allocate_*, submit and bind_output, on the calling thread where
 * the user's context is current. SHARED_CONTEXT (additive): a second context of the same
 * share group that anira's worker makes current.
 */
typedef struct anira_gl_desc {
    uint32_t struct_size;  /**< sizeof(anira_gl_desc) of the caller's header. */
    uint32_t threads;  /**< anira_gl_threads. */
    void* display;  /**< EGLDisplay (or the GLX equivalent). */
    void* context;  /**< EGLContext (or the GLX equivalent). */
    void* gbm;  /**< gbm_device*: lets allocate_* back GL storage with a dma-buf; NULL otherwise. */
} anira_gl_desc;
/**
 * @brief CALLER_THREAD, no handles.
 */
#define ANIRA_GL_DESC_INIT ANIRA_INIT(anira_gl_desc, sizeof(anira_gl_desc), ANIRA_GL_CALLER_THREAD, NULL, NULL, NULL)

/**
 * @brief The Vulkan block of a context config; thread-agnostic, anira serializes its own
 * submissions on the queue.
 */
typedef struct anira_vulkan_desc {
    uint32_t struct_size;  /**< sizeof(anira_vulkan_desc) of the caller's header. */
    uint32_t ownership;  /**< anira_ownership. */
    uint32_t queue_family;  /**< Queue family index. */
    uint32_t queue_index;  /**< Queue index within the family. */
    void* instance;  /**< VkInstance. */
    void* physical;  /**< VkPhysicalDevice. */
    void* device;  /**< VkDevice. */
} anira_vulkan_desc;
/**
 * @brief OWNED, queue family 0, index 0, no handles.
 */
#define ANIRA_VULKAN_DESC_INIT ANIRA_INIT(anira_vulkan_desc, sizeof(anira_vulkan_desc), ANIRA_OWNERSHIP_OWNED, 0u, 0u, NULL, NULL, NULL)

/**
 * @brief The Metal block of a context config.
 */
typedef struct anira_metal_desc {
    uint32_t struct_size;  /**< sizeof(anira_metal_desc) of the caller's header. */
    uint32_t reserved;  /**< Zero. */
    void* device;  /**< id<MTLDevice>; NULL = the default device. */
} anira_metal_desc;
/**
 * @brief The default device.
 */
#define ANIRA_METAL_DESC_INIT ANIRA_INIT(anira_metal_desc, sizeof(anira_metal_desc), 0u, NULL)

/**
 * @brief The Direct3D 12 block of a context config.
 */
typedef struct anira_d3d12_desc {
    uint32_t struct_size;  /**< sizeof(anira_d3d12_desc) of the caller's header. */
    uint32_t ownership;  /**< anira_ownership. */
    void* device;  /**< ID3D12Device*. */
} anira_d3d12_desc;
/**
 * @brief OWNED, no device handle.
 */
#define ANIRA_D3D12_DESC_INIT ANIRA_INIT(anira_d3d12_desc, sizeof(anira_d3d12_desc), ANIRA_OWNERSHIP_OWNED, NULL)

/**
 * @brief The WebGPU block of a context config (native Dawn); someone must pump ProcessEvents /
 * WaitAny, which exec selects.
 */
typedef struct anira_webgpu_desc {
    uint32_t struct_size;  /**< sizeof(anira_webgpu_desc) of the caller's header. */
    uint32_t ownership;  /**< anira_ownership. */
    uint32_t exec;  /**< anira_exec_policy. */
    uint32_t reserved;  /**< Zero. */
    void* instance;  /**< WGPUInstance. */
    void* device;  /**< WGPUDevice. */
    void* queue;  /**< WGPUQueue. */
} anira_webgpu_desc;
/**
 * @brief OWNED, WORKER, no handles.
 */
#define ANIRA_WEBGPU_DESC_INIT ANIRA_INIT(anira_webgpu_desc, sizeof(anira_webgpu_desc), ANIRA_OWNERSHIP_OWNED, ANIRA_EXEC_WORKER, 0u, NULL, NULL, NULL)

/**
 * @brief Creates a tensor spec: no axes yet, window 0/0/0, time ratio (0, 0) = derive, latency
 * 0.
 * @param name The canonical name: your name for this tensor, UTF-8, copied. Every other part of
 *        the configuration refers to the tensor by it (the per-entry name and layout
 *        records, the anchor, error messages); it is never handed to an engine. Unique
 *        across the inputs and outputs of one model config.
 * @param dtype The model's true dtype (section 1).
 * @param role STREAMED, BUFFER or STATIC.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL or empty name, an unknown role,
 *         or a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_create(const char* name,
                                                           anira_dtype dtype,
                                                           anira_role role,
                                                           anira_tensor_spec** out,
                                                           anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Sets one axis; axis index order is model memory order (NCHW vs NHWC is just axis
 * order).
 * @param spec The spec.
 * @param i Axis index in model memory order; i < ANIRA_MAX_RANK; ndim becomes max(i + 1).
 * @param tag The axis' meaning.
 * @param extent Extent > 0, or ANIRA_DYNAMIC on the Time axis of a Streamed or Buffer spec.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for i >= ANIRA_MAX_RANK, an unknown tag, or
 *         an extent that is neither positive nor ANIRA_DYNAMIC.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_axis(anira_tensor_spec* spec,
                                                             uint32_t i,
                                                             anira_axis_tag tag,
                                                             int64_t extent) ANIRA_NOEXCEPT;

/**
 * @brief The window of a Streamed spec, in elements along the Time axis; the fixed case is
 * window_min == window_max. Defaults 0, 0, 0. Cross-field legality (context <
 * window_min, window_max >= window_min) is checked at prepare.
 * @param spec The spec.
 * @param window_min The model's smallest legal Time extent, in elements.
 * @param window_max The largest, or ANIRA_UNBOUNDED.
 * @param overlap Overlap of consecutive windows: the elements kept from the previous window.
 *        The advance per inference, the hop, is window_used - overlap.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a negative value other than
 *         ANIRA_UNBOUNDED as window_max.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_window(anira_tensor_spec* spec,
                                                               int64_t window_min,
                                                               int64_t window_max,
                                                               int64_t overlap) ANIRA_NOEXCEPT;

/**
 * @brief The tensor's Time advance relative to the anchor tensor.
 * @param spec The spec.
 * @param num This tensor advances num elements ...
 * @param den ... per den anchor elements; (0, 0) = derive (default).
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a negative value or den == 0 with num
 *         != 0.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_time_ratio(anira_tensor_spec* spec,
                                                                   int64_t num,
                                                                   int64_t den) ANIRA_NOEXCEPT;

/**
 * @brief Outputs only: the model's internal delay along the Time axis, which the reported
 * latency adds.
 * @param spec The spec.
 * @param latency Model-internal delay along Time, in elements; default 0.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a negative latency.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_latency(anira_tensor_spec* spec,
                                                                int64_t latency) ANIRA_NOEXCEPT;

/**
 * @brief Sets an extension on the spec (section 1b); one slot per kind, a second set replaces
 * the first.
 * @param spec The spec.
 * @param ext The payload; deep-copied through the registry row.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short header;
 *         ANIRA_ERROR_EXTENSION_VERSION for a known kind at an unregistered version.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_ext(anira_tensor_spec* spec,
                                                            const anira_ext_header* ext,
                                                            anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The JSON twin of set_ext: a known kind is parsed through its registry row, an unknown
 * kind keeps the text.
 * @param spec The spec.
 * @param kind The extension kind.
 * @param utf8 The extension object as JSON text, optionally with a "version" member (default
 *        1).
 * @param len Length of utf8 in bytes.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL kind or text; ANIRA_ERROR_JSON for
 *         malformed text; ANIRA_ERROR_EXTENSION_VERSION for a known kind at an unregistered
 *         version.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_ext_json(anira_tensor_spec* spec,
                                                                 const char* kind,
                                                                 const char* utf8,
                                                                 size_t len,
                                                                 anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Destroys a spec; NULL-safe. A spec added to a model config was copied and may be
 * destroyed right after.
 * @param spec The spec, or NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API void ANIRA_CALL anira_tensor_spec_destroy(anira_tensor_spec* spec) ANIRA_NOEXCEPT;

/**
 * @brief Creates a Hard (real-time) contract with the stream geometry; the fixed-block host
 * earns the tight latency. The defaults: MEASURED budget, UNTIL_STABLE warmup, BYPASS on
 * miss, wait_ratio 0, PERMISSIVE edge cost. A geometry of 0, 0, 0 is legal here and
 * completed by hard_set_geometry or refused at prepare.
 * @param block_min Smallest block the host callback delivers, in Time-axis elements of the
 *        anchor tensor.
 * @param block_max Largest; block_min == block_max is the fixed-block host.
 * @param rate Anchor elements per second (48000 for audio).
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for block_min > block_max, a negative rate,
 *         or a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_create_hard(uint32_t block_min,
                                                             uint32_t block_max,
                                                             double rate,
                                                             anira_contract** out,
                                                             anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Creates an Async contract: no deadline (the offline posture), FINISH on late, AUTO
 * priority, auto lanes and depth, POLLED delivery, PERMISSIVE edge cost.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_create_async(anira_contract** out,
                                                              anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Patches the stream geometry, e.g. of a contract loaded from a file (section 8).
 * @param contract A Hard contract.
 * @param block_min See anira_contract_create_hard.
 * @param block_max See anira_contract_create_hard.
 * @param rate See anira_contract_create_hard.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for block_min > block_max or a negative rate.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_geometry(anira_contract* contract,
                                                                   uint32_t block_min,
                                                                   uint32_t block_max,
                                                                   double rate) ANIRA_NOEXCEPT;

/**
 * @brief The per-inference budget of a Hard contract (v2's max_inference_time when EXPLICIT).
 * @param contract A Hard contract.
 * @param kind MEASURED (default) derives the budget during warmup; EXPLICIT reads explicit_ms.
 * @param explicit_ms Per-inference budget in milliseconds, read for EXPLICIT only.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for an unknown kind or, with EXPLICIT, explicit_ms <= 0.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_budget(anira_contract* contract,
                                                                 anira_budget_kind kind,
                                                                 double explicit_ms) ANIRA_NOEXCEPT;

/**
 * @brief The warmup policy of a Hard contract (v2's warm_up when FIXED).
 * @param contract A Hard contract.
 * @param mode UNTIL_STABLE (default), FIXED, or NONE (legal only with an EXPLICIT budget,
 *        checked at prepare).
 * @param iterations Iterations for FIXED only.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for an unknown mode.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_warmup(anira_contract* contract,
                                                                 anira_warmup_mode mode,
                                                                 uint32_t iterations) ANIRA_NOEXCEPT;

/**
 * @brief What the handler delivers when an inference misses its deadline.
 * @param contract A Hard contract.
 * @param policy BYPASS (default; requires shape-compatible I/O along the anchored Time axis),
 *        HOLD_LAST or ZEROS.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for an unknown policy.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_on_miss(anira_contract* contract,
                                                                  anira_miss_policy policy) ANIRA_NOEXCEPT;

/**
 * @brief The wait ratio consumed by the _wait twins only: how long, as a fraction of the block
 * duration, a wait entry may block for a result.
 * @param contract A Hard contract.
 * @param ratio 0 (default) = never wait; v2's blocking_ratio one-to-one.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a negative ratio.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_wait_ratio(anira_contract* contract,
                                                                     double ratio) ANIRA_NOEXCEPT;

/**
 * @brief The ring dtype of one tensor under a Hard contract: the element type the typed Hard
 * entries carry across the ABI and anira_ring_dtype reports, held by the ring as is.
 * Nothing in anira converts: the Hard entries copy between the host and the ring, a ring
 * dtype that differs from the spec's dtype (the model's) is ANIRA_ERROR_CONFIG at
 * prepare. Set per tensor by canonical name, so an input and an output may differ;
 * ANIRA_DTYPE_F32 for every tensor never set, which is what the float entries are legal
 * on. A name that matches no Streamed tensor is checked at prepare, not here.
 * @param contract A Hard contract.
 * @param canonical The tensor's canonical name (the one its spec was created with).
 * @param dtype The element type of the host's samples for that tensor; ANIRA_DTYPE_F32 for
 *        every tensor that was never set.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL or empty name, or a dtype of
 *         0.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_ring_dtype(anira_contract* contract,
                                                                     const char* canonical,
                                                                     anira_dtype dtype) ANIRA_NOEXCEPT;

/**
 * @brief The per-job deadline of an Async contract; an absolute per-job override is the
 * deadline_ms argument of anira_handler_submit.
 * @param contract An Async contract.
 * @param deadline_ms < 0 (default) = none, the offline posture; the clock starts at submit.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on a Hard contract.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_async_set_deadline(anira_contract* contract,
                                                                    double deadline_ms) ANIRA_NOEXCEPT;

/**
 * @brief The scheduling policy of an Async contract.
 * @param contract An Async contract.
 * @param on_late FINISH (default) or DROP (cancels at chunk boundaries, enables admission
 *        control).
 * @param priority AUTO (default): INTERACTIVE iff a deadline is set, else BATCH.
 * @param lanes Parallel plan instances; 0 = auto (1 if STATEFUL, else min(max_instances,
 *        pool-derived)).
 * @param max_in_flight Per-lane pipelining; 0 = auto (shallow iff deadline, else deep).
 * @param delivery POLLED (default) or IMMEDIATE.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on a Hard contract; ANIRA_ERROR_INVALID_ARGUMENT
 *         for an unknown enum value.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_async_set_policy(anira_contract* contract,
                                                                  anira_late_policy on_late,
                                                                  anira_priority priority,
                                                                  uint32_t lanes,
                                                                  uint32_t max_in_flight,
                                                                  anira_delivery delivery) ANIRA_NOEXCEPT;

/**
 * @brief Plan validation policy for the edges a pipeline uses (section 7); not scheduling.
 * @param contract Either contract kind.
 * @param cost PERMISSIVE (default) or STRICT.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown value.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_set_edge_cost(anira_contract* contract,
                                                               anira_edge_cost cost) ANIRA_NOEXCEPT;

/**
 * @brief Sets an extension on the contract (section 1b); v3.0.0 registers none for this host.
 * @param contract Either contract kind.
 * @param ext The payload; deep-copied through the registry row.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short header;
 *         ANIRA_ERROR_EXTENSION_VERSION for a known kind at an unregistered version.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_set_ext(anira_contract* contract,
                                                         const anira_ext_header* ext,
                                                         anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The JSON twin of anira_contract_set_ext.
 * @param contract Either contract kind.
 * @param kind The extension kind.
 * @param utf8 The extension object as JSON text.
 * @param len Length of utf8 in bytes.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext_json.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_set_ext_json(anira_contract* contract,
                                                              const char* kind,
                                                              const char* utf8,
                                                              size_t len,
                                                              anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Whether the contract is Hard or Async.
 * @param contract The contract.
 * @return ANIRA_CONTRACT_HARD or ANIRA_CONTRACT_ASYNC.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_contract_kind ANIRA_CALL anira_contract_get_kind(const anira_contract* contract)
                                                                 ANIRA_NOEXCEPT;

/**
 * @brief Destroys a contract; NULL-safe.
 * @param contract The contract, or NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API void ANIRA_CALL anira_contract_destroy(anira_contract* contract) ANIRA_NOEXCEPT;

/**
 * @brief Creates a context config with the defaults: ANIRA_THREADS_AUTO threads, SPIN_BACKOFF,
 * log level WARNING, the drain thread every 10 ms, a 512-record queue, no sink, no
 * device blocks.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_create(anira_context_config** out,
                                                              anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The inference thread pool the first context sizes.
 * @param config The config.
 * @param num_threads Pool size; ANIRA_THREADS_AUTO = the library default, 0 = bring your own
 *        threads.
 * @param wait SPIN_BACKOFF (default) or BLOCKING.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown wait strategy.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_threads(anira_context_config* config,
                                                                   uint32_t num_threads,
                                                                   anira_wait_strategy wait) ANIRA_NOEXCEPT;

/**
 * @brief The runtime log level.
 * @param config The config.
 * @param level Default WARNING; the most verbose request across contexts wins.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown level.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_log_level(anira_context_config* config,
                                                                     anira_log_level level) ANIRA_NOEXCEPT;

/**
 * @brief Who drains the real-time log queue and how often.
 * @param config The config.
 * @param drain THREAD (default) or MANUAL (anira_drain_log).
 * @param interval_ms Drain-thread period in milliseconds; 0 keeps the default of 10.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown drain.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_log_drain(anira_context_config* config,
                                                                     anira_log_drain drain,
                                                                     uint32_t interval_ms) ANIRA_NOEXCEPT;

/**
 * @brief The real-time log queue's capacity.
 * @param config The config.
 * @param capacity Records; clamped to [64, 65536]; fixed for the life of the core.
 * @return ANIRA_OK.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_log_queue_capacity(anira_context_config* config,
                                                                              uint32_t capacity) ANIRA_NOEXCEPT;

/**
 * @brief Log flags, e.g. ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK.
 * @param config The config.
 * @param flags ANIRA_LOG_FLAG_* bits.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown bit.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_log_flags(anira_context_config* config,
                                                                     uint32_t flags) ANIRA_NOEXCEPT;

/**
 * @brief The context's log sink; ignored on Wasm, where anira_em_set_log_hook is the sink.
 * @param config The config.
 * @param callback The sink, or NULL for none.
 * @param user_data Passed to callback.
 * @return ANIRA_OK.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_log_sink(anira_context_config* config,
                                                                    anira_log_fn callback,
                                                                    void* user_data) ANIRA_NOEXCEPT;

/**
 * @brief The C one-shot convenience equal to the five scalar log setters.
 * @param config The config.
 * @param desc The log block; read within min(struct_size, sizeof(anira_log_desc)).
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short descriptor;
 *         ANIRA_ERROR_ABI_VERSION when desc->abi_version fails anira_check_abi.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_log(anira_context_config* config,
                                                               const anira_log_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Declares the CUDA device block; presence is the user's declaration, no implicit
 * probing.
 * @param config The config.
 * @param desc The block, or NULL = domain unavailable, edges pruned.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_cuda(anira_context_config* config,
                                                                const anira_cuda_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Declares the OpenGL device block.
 * @param config The config.
 * @param desc The block, or NULL.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_gl(anira_context_config* config,
                                                              const anira_gl_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Declares the Vulkan device block.
 * @param config The config.
 * @param desc The block, or NULL.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_vulkan(anira_context_config* config,
                                                                  const anira_vulkan_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Declares the Metal device block.
 * @param config The config.
 * @param desc The block, or NULL.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_metal(anira_context_config* config,
                                                                 const anira_metal_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Declares the Direct3D 12 device block.
 * @param config The config.
 * @param desc The block, or NULL.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a short descriptor.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_d3d12(anira_context_config* config,
                                                                 const anira_d3d12_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Declares the WebGPU device block (native Dawn); ANIRA_ERROR_NOT_SUPPORTED under
 * Emscripten in 3.0.
 * @param config The config.
 * @param desc The block, or NULL.
 * @return ANIRA_OK, ANIRA_ERROR_NOT_SUPPORTED, or ANIRA_ERROR_INVALID_ARGUMENT for a short
 *         descriptor.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_webgpu(anira_context_config* config,
                                                                  const anira_webgpu_desc* desc) ANIRA_NOEXCEPT;

/**
 * @brief Sets an extension on the context config (section 1b).
 * @param config The config.
 * @param ext The payload; deep-copied through the registry row.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_ext(anira_context_config* config,
                                                               const anira_ext_header* ext,
                                                               anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The JSON twin of anira_context_config_set_ext.
 * @param config The config.
 * @param kind The extension kind.
 * @param utf8 The extension object as JSON text.
 * @param len Length of utf8 in bytes.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext_json.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_set_ext_json(anira_context_config* config,
                                                                    const char* kind,
                                                                    const char* utf8,
                                                                    size_t len,
                                                                    anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Destroys a context config; NULL-safe.
 * @param config The config, or NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_context_config_destroy(anira_context_config* config) ANIRA_NOEXCEPT;

/**
 * @brief Release callback of borrowed model bytes: fires exactly once, when the last carrier of
 * the bytes dies, on the thread that destroys it.
 * @param bytes The bytes handed to add_model_bytes / set_model_bytes.
 * @param ctx The ctx handed with them.
 * @par Thread contract
 * [main-thread]
 */
typedef void (ANIRA_CALL* anira_bytes_release_fn)(const void* bytes, void* ctx);

/**
 * @brief Creates an empty model config: no models, no tensors, default engine NONE (=
 * models[0]), STATELESS, max_instances 1, anchor = the first Streamed input.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_create(anira_model_config** out,
                                                            anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Appends a model entry that loads from a file. Whether the engine is in this build is
 * decided at prepare, so a config can name every engine a deployment might have.
 * @param config The config.
 * @param engine A built-in engine (not NONE).
 * @param utf8_path Model file path, UTF-8, copied.
 * @param out_index Receives the model index, or NULL.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for ANIRA_ENGINE_NONE, an unknown engine,
 *         or a NULL or empty path.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_model_path(anira_model_config* config,
                                                                    anira_engine engine,
                                                                    const char* utf8_path,
                                                                    uint32_t* out_index,
                                                                    anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Appends a model entry that loads from memory.
 * @param config The config.
 * @param engine A built-in engine (not NONE).
 * @param bytes The model bytes.
 * @param size Their size; > 0.
 * @param ownership COPY, or BORROW (the plugin default for embedded blobs).
 * @param release Optional release callback for BORROW; fires once when the last carrier dies.
 * @param ctx Passed to release.
 * @param out_index Receives the model index, or NULL.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for ANIRA_ENGINE_NONE, an unknown engine or
 *         ownership, NULL bytes or a zero size.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_model_bytes(anira_model_config* config,
                                                                     anira_engine engine,
                                                                     const void* bytes,
                                                                     size_t size,
                                                                     anira_bytes_ownership ownership,
                                                                     anira_bytes_release_fn release,
                                                                     void* ctx,
                                                                     uint32_t* out_index,
                                                                     anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Appends a model entry for a custom engine registered by name
 * (anira_pipeline_register_engine, M2).
 * @param config The config.
 * @param engine_id A registered custom engine's name, reverse-URI (must contain a '.').
 * @param utf8_path Model file path, UTF-8, copied.
 * @param out_index Receives the model index, or NULL.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an id without a '.', or a NULL or empty
 *         path.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_model_path_custom(anira_model_config* config,
                                                                           const char* engine_id,
                                                                           const char* utf8_path,
                                                                           uint32_t* out_index,
                                                                           anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The bytes twin of anira_model_config_add_model_path_custom.
 * @param config The config.
 * @param engine_id A registered custom engine's name, reverse-URI.
 * @param bytes The model bytes.
 * @param size Their size; > 0.
 * @param ownership COPY or BORROW.
 * @param release Optional release callback for BORROW.
 * @param ctx Passed to release.
 * @param out_index Receives the model index, or NULL.
 * @param err Nullable.
 * @return As anira_model_config_add_model_bytes, plus ANIRA_ERROR_INVALID_ARGUMENT for an id
 *         without a '.'.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_model_bytes_custom(anira_model_config* config,
                                                                            const char* engine_id,
                                                                            const void* bytes,
                                                                            size_t size,
                                                                            anira_bytes_ownership ownership,
                                                                            anira_bytes_release_fn release,
                                                                            void* ctx,
                                                                            uint32_t* out_index,
                                                                            anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Replaces an entry's source with bytes, e.g. to patch a path entry a JSON file
 * produced; the path is kept for anira_model_config_to_json, model_path() then returns
 * NULL.
 * @param config The config.
 * @param model_index An existing entry.
 * @param bytes The model bytes.
 * @param size Their size; > 0.
 * @param ownership COPY or BORROW.
 * @param release Optional release callback for BORROW.
 * @param ctx Passed to release.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an index out of range, NULL bytes, a
 *         zero size or an unknown ownership.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_model_bytes(anira_model_config* config,
                                                                     uint32_t model_index,
                                                                     const void* bytes,
                                                                     size_t size,
                                                                     anira_bytes_ownership ownership,
                                                                     anira_bytes_release_fn release,
                                                                     void* ctx,
                                                                     anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The number of model entries.
 * @param config The config.
 * @return The count; 0 for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API uint32_t ANIRA_CALL anira_model_config_model_count(const anira_model_config* config)
                                                             ANIRA_NOEXCEPT;

/**
 * @brief The entry's engine.
 * @param config The config.
 * @param model_index An entry.
 * @return The engine; ANIRA_ENGINE_NONE for a custom entry or an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_engine ANIRA_CALL anira_model_config_model_engine(const anira_model_config* config,
                                                                  uint32_t model_index) ANIRA_NOEXCEPT;

/**
 * @brief The entry's custom engine name.
 * @param config The config.
 * @param model_index An entry.
 * @return Object-owned; NULL for a built-in engine or an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_model_engine_id(const anira_model_config* config,
                                                                    uint32_t model_index) ANIRA_NOEXCEPT;

/**
 * @brief The entry's model path.
 * @param config The config.
 * @param model_index An entry.
 * @return Object-owned; NULL for a bytes entry or an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_model_path(const anira_model_config* config,
                                                               uint32_t model_index) ANIRA_NOEXCEPT;

/**
 * @brief The entry's model bytes.
 * @param config The config.
 * @param model_index An entry.
 * @param bytes Receives the bytes (object-owned).
 * @param size Receives their size.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_STATE for a path entry; ANIRA_ERROR_INVALID_ARGUMENT
 *         for an index out of range or NULL out-parameters.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_model_bytes(const anira_model_config* config,
                                                                 uint32_t model_index,
                                                                 const void** bytes,
                                                                 size_t* size) ANIRA_NOEXCEPT;

/**
 * @brief Records what this entry's file calls the tensor you named canonical, and switches that
 * tensor of this entry from positional binding (input slot i to the file's input i, the
 * slot order being the add_input/add_output order; ONNX Runtime's session order, the
 * primary subgraph's order on TFLite and LiteRT) to binding by that name. An entry
 * without a record for a tensor binds it positionally. A name the engine cannot resolve,
 * or an engine that binds only positionally on that side, fails prepare with
 * ANIRA_ERROR_CONFIG naming what the file has. Together with
 * anira_model_config_set_tensor_layout this is the per-entry tensor record, the JSON
 * file's models[].tensors.
 * @param config The config.
 * @param model_index An entry.
 * @param canonical Your canonical name of the tensor (the spec's name); the spec may be added
 *        later, the name is resolved at prepare.
 * @param engine_name The export's name for that tensor, copied: ONNX Runtime the graph's input
 *        or output name; TFLite and LiteRT the signature key ("args_0",
 *        "output_0"), or the tensor name for a file without signatures; LibTorch
 *        the method's argument name (inputs only); ExecuTorch the tensor name when
 *        the export carries one.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an index out of range or a NULL or
 *         empty name.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_tensor_name(anira_model_config* config,
                                                                     uint32_t model_index,
                                                                     const char* canonical,
                                                                     const char* engine_name) ANIRA_NOEXCEPT;

/**
 * @brief The axis order in which this entry's file holds the tensor you named canonical, when
 * it differs from the spec's (a TensorFlow export holding batch, time, channel where the
 * spec says batch, channel, time is {0, 2, 1}). A spec axis left out must have extent 1.
 * A layout that moves only axes of extent 1 is a view: the same bytes with other dims,
 * at no cost. One that moves an axis of another extent is a transpose, refused at
 * prepare in this pre-release with ANIRA_ERROR_NOT_SUPPORTED. Agreement with the spec's
 * rank and extents is checked at prepare, and against the file where the engine reports
 * its dims.
 * @param config The config.
 * @param model_index An entry.
 * @param canonical Your canonical name of the tensor (the spec's name); resolved at prepare.
 * @param axes ndim entries, copied: axes[k] is the spec axis (an index into
 *        anira_tensor_spec_set_axis's order, each at most once) that this entry's file
 *        holds at position k, or ANIRA_AXIS_INSERT for an axis of extent 1 the file has
 *        and the spec does not. NULL with ndim 0 clears.
 * @param ndim 1..ANIRA_MAX_RANK, or 0 to clear.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an index out of range, a NULL or empty
 *         canonical, ndim > ANIRA_MAX_RANK, NULL axes with ndim > 0, a spec axis index >=
 *         ANIRA_MAX_RANK that is not ANIRA_AXIS_INSERT, or a spec axis listed twice.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_tensor_layout(anira_model_config* config,
                                                                       uint32_t model_index,
                                                                       const char* canonical,
                                                                       const uint32_t* axes,
                                                                       uint32_t ndim) ANIRA_NOEXCEPT;

/**
 * @brief Sets an extension on one model entry (host "model"), e.g. anira_ext_entry.
 * @param config The config.
 * @param model_index An entry.
 * @param ext The payload; deep-copied through the registry row.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext, plus ANIRA_ERROR_INVALID_ARGUMENT for an index out of
 *         range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_model_ext(anira_model_config* config,
                                                                   uint32_t model_index,
                                                                   const anira_ext_header* ext,
                                                                   anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The JSON twin of anira_model_config_set_model_ext.
 * @param config The config.
 * @param model_index An entry.
 * @param kind The extension kind.
 * @param utf8 The extension object as JSON text.
 * @param len Length of utf8 in bytes.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext_json, plus ANIRA_ERROR_INVALID_ARGUMENT for an index out
 *         of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_model_ext_json(anira_model_config* config,
                                                                        uint32_t model_index,
                                                                        const char* kind,
                                                                        const char* utf8,
                                                                        size_t len,
                                                                        anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Appends an input tensor spec (copied; the caller keeps ownership of spec). The order
 * of the add_input calls is the slot order an entry without a name record binds
 * positionally.
 * @param config The config.
 * @param spec The spec; copied.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec or a spec whose canonical
 *         name an input or output of this config already has.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_input(anira_model_config* config,
                                                               const anira_tensor_spec* spec) ANIRA_NOEXCEPT;

/**
 * @brief Appends an output tensor spec (copied). The order of the add_output calls is the slot
 * order an entry without a name record binds positionally.
 * @param config The config.
 * @param spec The spec; copied.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec or a spec whose canonical
 *         name an input or output of this config already has.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_output(anira_model_config* config,
                                                                const anira_tensor_spec* spec) ANIRA_NOEXCEPT;

/**
 * @brief The engine the handler starts on; whether it names an entry is checked at prepare.
 * @param config The config.
 * @param engine A built-in engine, or ANIRA_ENGINE_NONE = models[0] (default).
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown engine.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_default_engine(anira_model_config* config,
                                                                        anira_engine engine) ANIRA_NOEXCEPT;

/**
 * @brief The custom twin of anira_model_config_set_default_engine.
 * @param config The config.
 * @param engine_id A registered custom engine's name, reverse-URI.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an id without a '.'.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_default_engine_custom(anira_model_config* config,
                                                                               const char* engine_id) ANIRA_NOEXCEPT;

/**
 * @brief Whether the model carries state across inferences (v2's session_exclusive_processor).
 * @param config The config.
 * @param state STATELESS (default) or STATEFUL (session-exclusive, lanes forced to 1).
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown state.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_state(anira_model_config* config,
                                                               anira_model_state state) ANIRA_NOEXCEPT;

/**
 * @brief The ceiling within which the planner allocates lanes and pool instances (v2's
 * num_parallel_processors).
 * @param config The config.
 * @param max_instances Memory ceiling of loaded model instances; >= 1; default 1.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for 0.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_max_instances(anira_model_config* config,
                                                                       uint32_t max_instances) ANIRA_NOEXCEPT;

/**
 * @brief The anchor: the streamed tensor whose Time axis is the model's clock. A Hard
 * contract's block range and rate are counted in its elements, and every other streamed
 * tensor's time ratio is stated against it (v2's HostConfig reference stream). Default:
 * the first streamed input, or, for a model without one (a generator), the first
 * streamed output. Resolved at prepare, where a name that is not a streamed tensor of
 * this config fails with ANIRA_ERROR_CONFIG.
 * @param config The config.
 * @param canonical Your canonical name of a streamed tensor, copied; NULL or empty restores the
 *        default.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_anchor(anira_model_config* config,
                                                                const char* canonical) ANIRA_NOEXCEPT;

/**
 * @brief Sets an extension on the whole config (host "model_config").
 * @param config The config.
 * @param ext The payload; deep-copied through the registry row.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_ext(anira_model_config* config,
                                                             const anira_ext_header* ext,
                                                             anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The JSON twin of anira_model_config_set_ext.
 * @param config The config.
 * @param kind The extension kind.
 * @param utf8 The extension object as JSON text.
 * @param len Length of utf8 in bytes.
 * @param err Nullable.
 * @return As anira_tensor_spec_set_ext_json.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_ext_json(anira_model_config* config,
                                                                  const char* kind,
                                                                  const char* utf8,
                                                                  size_t len,
                                                                  anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Destroys a model config; NULL-safe. Borrowed model bytes are released (their callback
 * fires) when the last carrier dies.
 * @param config The config, or NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API void ANIRA_CALL anira_model_config_destroy(anira_model_config* config) ANIRA_NOEXCEPT;

/**
 * @brief Creates job options with the defaults: no head trim, tail flush on, REJECT below the
 * window minimum. Build once, reuse across submits; never mutate concurrently with a
 * submit that reads it.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_job_options_create(anira_job_options** out,
                                                           anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Per-output head trim of an Async job's outputs.
 * @param options The options.
 * @param count Number of trims, one per output.
 * @param trims Elements to drop at the head of each output; -1 = that output's latency
 *        (input-aligned). Copied.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL trims with count > 0, or a value
 *         below -1.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_job_options_set_head_trim(anira_job_options* options,
                                                                  uint32_t count,
                                                                  const int64_t* trims) ANIRA_NOEXCEPT;

/**
 * @brief ViewChunker reassembly semantics: whether the tail of a job is flushed through the
 * model.
 * @param options The options.
 * @param tail_flush Default true.
 * @return ANIRA_OK.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_job_options_set_tail_flush(anira_job_options* options,
                                                                   anira_bool tail_flush) ANIRA_NOEXCEPT;

/**
 * @brief What happens to a submitted buffer shorter than the window minimum.
 * @param options The options.
 * @param policy REJECT (default) or ZEROS.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown policy.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_job_options_set_below_min(anira_job_options* options,
                                                                  anira_pad_policy policy) ANIRA_NOEXCEPT;

/**
 * @brief Sets a per-job extension (host "job"): borrowed, not copied, because submit is
 * ANIRA_NONBLOCKING and copies what it needs into the job record; the consumed-or-fail
 * check runs at submit and fails the ticket, not the handler.
 * @param options The options.
 * @param ext The payload; BORROWED until every submit that reads the options has returned.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL or short header.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_job_options_set_ext(anira_job_options* options,
                                                            const anira_ext_header* ext) ANIRA_NOEXCEPT;

/**
 * @brief The JSON twin every config handle carries; the parsed payload is owned by the options.
 * @param options The options.
 * @param kind The extension kind.
 * @param utf8 The extension object as JSON text.
 * @param len Length of utf8 in bytes.
 * @return ANIRA_OK, ANIRA_ERROR_INVALID_ARGUMENT for a NULL kind or text, ANIRA_ERROR_JSON for
 *         malformed text, ANIRA_ERROR_EXTENSION_VERSION for a known kind at an unregistered
 *         version.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_job_options_set_ext_json(anira_job_options* options,
                                                                 const char* kind,
                                                                 const char* utf8,
                                                                 size_t len) ANIRA_NOEXCEPT;

/**
 * @brief Destroys job options; NULL-safe.
 * @param options The options, or NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API void ANIRA_CALL anira_job_options_destroy(anira_job_options* options) ANIRA_NOEXCEPT;

/**
 * @brief Loads a model config from JSON text (section 8.1). A version 2 document is upgraded
 * (section 8.4): the result carries the models, tensors and scalars, holds back
 * max_inference_time / warm_up / blocking_ratio as a legacy Hard contract for
 * anira_model_config_take_legacy_contract, logs one warning per process, and returns
 * ANIRA_SUCCESS_UPGRADED. Unknown keys are stored as extensions and fail prepare by
 * name.
 * @param utf8 The document text (a v3 model file, or a v2 document with an inference_config
 *        root).
 * @param len Length of utf8 in bytes.
 * @param base_dir Directory relative model paths resolve against (the joined path uses forward
 *        slashes on every platform; a rooted path stays as written), or NULL to keep
 *        every path as written.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK or ANIRA_SUCCESS_UPGRADED; ANIRA_ERROR_JSON with the key path and the
 *         offending value in err for malformed text, a wrong type, or a string outside a key's
 *         vocabulary; ANIRA_ERROR_EXTENSION_VERSION for a known extension at an unregistered
 *         version.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_from_json(const char* utf8,
                                                               size_t len,
                                                               const char* base_dir,
                                                               anira_model_config** out,
                                                               anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Reads a file and loads it as anira_model_config_from_json with base_dir = the file's
 * directory.
 * @param utf8_path Path of the file; its directory is the base_dir of the model paths.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return As anira_model_config_from_json, plus ANIRA_ERROR_NO_SUCH_FILE.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_from_json_file(const char* utf8_path,
                                                                    anira_model_config** out,
                                                                    anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Writes the config in v3 spelling (section 8.1), fixed key order; reading a v2 file and
 * writing it out is the migration tool. Bytes entries keep the path they were loaded
 * with.
 * @param config The config.
 * @param buf Receives the text, NUL-terminated; may be NULL with cap 0 to size.
 * @param cap Capacity of buf in bytes.
 * @param out_len Receives the text length without the NUL; always written.
 * @return ANIRA_OK, or ANIRA_ERROR_BUFFER_TOO_SMALL (out_len holds the required length) or
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL config or out_len.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_to_json(const anira_model_config* config,
                                                             char* buf,
                                                             size_t cap,
                                                             size_t* out_len) ANIRA_NOEXCEPT;

/**
 * @brief Hands out the Hard contract a version 2 upgrade held back (budget, warmup and wait
 * ratio; no geometry). Non-NULL only once after an upgrade: a second call, or a v3
 * document, yields NULL.
 * @param config The config.
 * @param out Receives the contract (caller destroys it), or NULL when there is none.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config or out.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_take_legacy_contract(anira_model_config* config,
                                                                          anira_contract** out) ANIRA_NOEXCEPT;

/**
 * @brief Loads a context config from JSON text (section 8.2). Device blocks in JSON imply
 * ANIRA_OWNERSHIP_OWNED; borrowed handles are code-only and patched afterwards with the
 * device setters. A version 2 document is upgraded (context_config; the bare log_level
 * key becomes log.level) and returns ANIRA_SUCCESS_UPGRADED.
 * @param utf8 The document text (a v3 context file, or a v2 document with a context_config
 *        root).
 * @param len Length of utf8 in bytes.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK or ANIRA_SUCCESS_UPGRADED; ANIRA_ERROR_JSON with the key path for malformed
 *         text, a wrong type or an unknown vocabulary value.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_from_json(const char* utf8,
                                                                 size_t len,
                                                                 anira_context_config** out,
                                                                 anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief Writes the config in v3 spelling (section 8.2), fixed key order; the sink is code-only
 * and not written, device blocks are written without their borrowed handles.
 * @param config The config.
 * @param buf Receives the text, NUL-terminated; may be NULL with cap 0 to size.
 * @param cap Capacity of buf in bytes.
 * @param out_len Receives the text length without the NUL; always written.
 * @return ANIRA_OK, or ANIRA_ERROR_BUFFER_TOO_SMALL (out_len holds the required length) or
 *         ANIRA_ERROR_INVALID_ARGUMENT.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_to_json(const anira_context_config* config,
                                                               char* buf,
                                                               size_t cap,
                                                               size_t* out_len) ANIRA_NOEXCEPT;

/**
 * @brief Loads a contract from JSON text (section 8.3): budget is "measured" or {"ms": x},
 * warmup is "until_stable", "none" or {"fixed": n}; a file with both roots or neither is
 * ANIRA_ERROR_JSON. A version 2 document yields its legacy Hard contract directly
 * (ANIRA_SUCCESS_UPGRADED).
 * @param utf8 The document text: {"hard": {...}} or {"async": {...}} with an optional top-level
 *        edge_cost, or a v2 document.
 * @param len Length of utf8 in bytes.
 * @param out Receives the handle on success.
 * @param err Nullable.
 * @return ANIRA_OK or ANIRA_SUCCESS_UPGRADED, or ANIRA_ERROR_JSON with the key path.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_from_json(const char* utf8,
                                                           size_t len,
                                                           anira_contract** out,
                                                           anira_error* err) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_CONFIG_H */
