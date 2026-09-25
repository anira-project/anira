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
 * and copied; strings, spec pointers and extension records out are owned by the handle and
 * valid until it is destroyed or mutated (a config's next add_input or add_output moves its
 * specs). The read-back entries at the end answer what the setters and the loaders stored.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/tensor.h>

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
 * @brief One set of anira_ext_provider_options: a backend (an engine on a provider, the
 * two-axis id as anira_backend_id spells it) and the options its runtime takes for the
 * provider, as string pairs in the runtime's own vocabulary (ONNX Runtime's execution
 * provider option keys: device_id for CUDA, backend_path for QNN, ...), kept in key
 * order (two sets with the same pairs are one set, whatever their order). Tier 2,
 * struct_size first, read within it; copied by the set call, which refuses a malformed
 * set with ANIRA_ERROR_INVALID_ARGUMENT naming it: a struct_size below the fixed head or
 * unlike the first record's (an array has one stride), an engine or a provider value
 * this header does not name, no provider (the default provider takes no options), a
 * provider of the enum beside a provider_id, an empty provider_id, or a count of options
 * without both arrays or with a NULL entry. Nothing is dropped or ignored on the quiet.
 */
typedef struct anira_provider_option_set {
    uint32_t struct_size;  /**< sizeof(anira_provider_option_set) of the caller's header. */
    /**
     * anira_engine of the backend; ANIRA_ENGINE_NONE with an engine_id for a custom engine,
     * which receives the set in its load record (anira_engine_load_info.option_keys) when its
     * descriptor lists "context:provider_options".
     */
    uint32_t engine;
    /**
     * anira_provider of the backend; ANIRA_PROVIDER_DEFAULT beside a provider_id, never alone:
     * the default provider takes no options, and a set for it is refused.
     */
    uint32_t provider;
    uint32_t num_options;  /**< The number of key/value pairs. */
    const char* engine_id;  /**< NULL for a built-in engine; the id of a custom engine, copied. */
    /**
     * NULL for a provider the enum names; a custom provider's name in the engine's vocabulary,
     * copied.
     */
    const char* provider_id;
    const char* const* keys;  /**< num_options option names, NUL-terminated, copied. */
    const char* const* values;  /**< num_options option values, NUL-terminated, copied. */
} anira_provider_option_set;
/**
 * @brief No backend and no options.
 */
#define ANIRA_PROVIDER_OPTION_SET_INIT ANIRA_INIT(anira_provider_option_set, sizeof(anira_provider_option_set), ANIRA_ENGINE_NONE, ANIRA_PROVIDER_DEFAULT, 0u, NULL, NULL, NULL, NULL)

/**
 * @brief Extension "provider_options", version 1, on the context config: the options an
 * engine's runtime takes for a provider, one set per backend
 * (anira_provider_option_set). Each set is checked against the backend it names. Its
 * engine must consume the kind: the ONNX Runtime adapter, which reads the set of the
 * plan's backend at load into the execution provider's options (CUDA's own entry and the
 * generic one alike), or a custom engine whose descriptor lists
 * "context:provider_options", which receives the set in its load record; a set for
 * another engine is ANIRA_ERROR_EXTENSION_UNCONSUMED naming the backend, at
 * anira_context_create for a built-in engine (no adapter of the build reads the kind for
 * it) and at anira_handler_create for a custom engine of the pipeline. Its provider must
 * be one the engine serves here, the context's capabilities for a built-in engine and
 * the descriptor's list for a custom one, else ANIRA_ERROR_NOT_SUPPORTED at
 * anira_handler_create, so a misspelled provider word is refused and never silently
 * ignored. A set for a custom engine no pipeline of the handler adds, or for a backend
 * no plan runs on, is not an error. The options are part of the loaded model where its
 * engine reads them: two contexts with different options for one backend load twice. A
 * device block of the context (anira_cuda_desc.device and its siblings) and a provider
 * option that selects a device ("device_id" for CUDA) name the same device: anira does
 * not reconcile the two in this pre-release. JSON: {"version": 1, "sets": [{"engine":
 * "onnxruntime", "provider": "cuda", "options": {"device_id": "0"}}]}: the pair as its
 * two keys, as a model entry spells them (a built-in engine's word or a custom engine's
 * id; the enum's spelling or a custom provider's name, never the default provider, which
 * takes no options), every option value a string.
 */
typedef struct anira_ext_provider_options {
    anira_ext_header header;  /**< {sizeof(anira_ext_provider_options), 1, "provider_options"}. */
    /**
     * num_sets records at the stride of the first record's struct_size (an array has one), each
     * read within it; copied. NULL with a count of 0 for none.
     */
    const anira_provider_option_set* sets;
    uint32_t num_sets;  /**< The number of sets. */
    uint32_t reserved;  /**< 0. */
} anira_ext_provider_options;
/**
 * @brief An anira_ext_provider_options with its header filled and no set yet.
 */
#define ANIRA_EXT_PROVIDER_OPTIONS_INIT ANIRA_INIT(anira_ext_provider_options, {sizeof(anira_ext_provider_options), 1, "provider_options"}, NULL, 0u, 0u)

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
    /**
     * Index of the physical device anira picks when it owns the device (ANIRA_OWNERSHIP_OWNED);
     * the JSON key vulkan.device. A tail slot appended at ABI 0.2: a shorter record of an older
     * caller reads as 0.
     */
    int32_t device_index;
    /**
     * Zero. Keeps the record free of tail padding on 64-bit targets: a setter copies within
     * struct_size, so padding a caller never wrote could not become a slot later.
     */
    uint32_t reserved;
} anira_vulkan_desc;
/**
 * @brief OWNED, queue family 0, queue index 0, no handles, device index 0.
 */
#define ANIRA_VULKAN_DESC_INIT ANIRA_INIT(anira_vulkan_desc, sizeof(anira_vulkan_desc), ANIRA_OWNERSHIP_OWNED, 0u, 0u, NULL, NULL, NULL, 0, 0u)

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
 * @param role STREAMED, BUFFER, STATIC or STATE (both halves of a declared state pair carry
 *        STATE).
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
 * @brief Declared state passing: pairs a state input with the state output it is fed from on
 * the next inference. Both halves carry ANIRA_ROLE_STATE; the pairing is stated once, on
 * the input (JSON: "state_source" on the input spec). anira keeps one value per pair in
 * the handler, on the port of the state input, in the spec's shape and dtype (zeroed at
 * create and at prepare), copies it into the input tensor as the first step before the
 * engine call, ahead of the stage's before_inference, and copies the output tensor into
 * it as the last step after the engine call, behind the stage's after_inference; a
 * failed inference (an engine failure, a non-OK stage) skips the capture, so the state
 * keeps its last good value. anira_handler_reset re-initialises the state at the first
 * inference of the new stream; the state survives anira_handler_set_plan. A model with a
 * pair runs as ANIRA_MODEL_STATEFUL whatever its state says. Validation
 * (ANIRA_ERROR_CONFIG naming the tensor, at create and at prepare): every state input
 * names exactly one state output and every state output is named by exactly one state
 * input; the two halves have equal dtype and shape; a state spec has no Time axis,
 * window, time ratio or latency, and neither a ring dtype nor the anchor may name it.
 * ANIRA_ERROR_NOT_SUPPORTED: a state tensor of another dtype than float32 (as every
 * model tensor of this pre-release; the value takes the spec's dtype), or with a
 * transposed layout. A second call replaces the name.
 * @param spec The spec of a state INPUT (role ANIRA_ROLE_STATE).
 * @param output_canonical The canonical name of the state output this input is fed from, UTF-8,
 *        copied; resolved when the model is validated (anira_handler_create,
 *        anira_handler_prepare), like the anchor's name.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec, a NULL or empty name, or a
 *         spec whose role is not ANIRA_ROLE_STATE.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_set_state_source(anira_tensor_spec* spec,
                                                                     const char* output_canonical) ANIRA_NOEXCEPT;

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
 *        HOLD_LAST, ZEROS or CALLBACK (needs anira_contract_hard_set_miss_fn, in either
 *        order; checked at prepare).
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for an unknown policy.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_on_miss(anira_contract* contract,
                                                                  anira_miss_policy policy) ANIRA_NOEXCEPT;

/**
 * @brief The backup function of ANIRA_MISS_CALLBACK: fills a whole missed block, every slot at
 * once and whatever the slots' dtypes. Called once per missed block on the thread that
 * called the Hard entry: the driver thread, or the thread inside a _wait twin. Under a
 * tensor entry the tensors are the caller's own (a single-tensor form on a side with
 * several slots passes the handler's array, the caller's descriptor in its slot and
 * empty tensors beside it, which name no memory: no pointer of an earlier call stays in
 * a slot). Return ANIRA_OK when the outputs are filled; any other status makes anira
 * zero-fill every requested Streamed output. Either way the block counts as missed: the
 * entry returns ANIRA_MISSED with a delivered count of 0 for every Streamed slot, and
 * the stream stays time-aligned. The function has the last word on every output of the
 * block, the Static ones of a _multi form included: anira fills them with the stored
 * value before the call and does not touch them afterwards. What the function writes
 * there reaches the caller's memory only, never the handler's store:
 * anira_handler_get_static_output keeps returning what the model produced. The function
 * must not call a Hard entry, anira_handler_reset or anira_handler_prepare of the same
 * handler, and it is real-time code: no allocation, no lock, no system call. Under clang
 * a function converted to this type must itself be declared ANIRA_NONBLOCKING.
 * @param handler The handler whose block was missed.
 * @param inputs One tensor per input slot, in slot order and covering every slot (a slot is the
 *        tensor's position in the model config's input list): the arrays of the running
 *        call, the caller's own under a _multi form. A slot the call did not carry, and
 *        the position of a State tensor always, is an empty tensor (shape[1] == 0, its
 *        memory arm not to be read); a pop passes empty inputs. The block of a process
 *        form is already pushed and still intact in host memory, in place too.
 * @param num_inputs The number of tensors of the model config's input list.
 * @param outputs One tensor per output slot (the tensor's position in the model config's output
 *        list); shape[1] of a Streamed slot is the request, and the memory is the
 *        function's to fill. A slot whose tensor is empty (an extent of 0) was not
 *        requested. A non-empty Static element of a _multi form is the whole tensor in
 *        the spec's shape and already holds the stored value (the latest the model
 *        produced): leave it, or overwrite it. The descriptors are const: write through
 *        anira_tensor_data, anira_tensor_plane or the handle, by the tensor's strides.
 * @param num_outputs The number of tensors of the model config's output list.
 * @param user_data The user_data of anira_contract_hard_set_miss_fn.
 * @par Thread contract
 * [driver-thread] ANIRA_NONBLOCKING
 */
typedef anira_status (ANIRA_CALL* anira_miss_fn)(anira_handler* handler,
                                                 const anira_tensor* inputs,
                                                 uint32_t num_inputs,
                                                 const anira_tensor* outputs,
                                                 uint32_t num_outputs,
                                                 void* user_data) ANIRA_NONBLOCKING;

/**
 * @brief The backup function of ANIRA_MISS_CALLBACK. The pair is copied with the contract at
 * anira_handler_prepare and must outlive the prepared handler; it is read only under the
 * CALLBACK policy, in whichever order the two setters ran. A contract file can say
 * "on_miss": "callback" but cannot carry a function: set the pair on the parsed contract
 * before prepare.
 * @param contract A Hard contract.
 * @param fn The backup function, or NULL to clear the pair.
 * @param user_data Handed to fn as it is; never read by anira.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_miss_fn(anira_contract* contract,
                                                                  anira_miss_fn fn,
                                                                  void* user_data) ANIRA_NOEXCEPT;

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
 * @brief The ring dtype of one tensor under a Hard contract: the element type the tensor forms
 * of the Hard entries carry across the ABI and anira_ring_dtype reports, held by the
 * ring as is. Nothing in anira converts: the Hard entries copy between the host and the
 * ring, a ring dtype that differs from the spec's dtype (the model's) is
 * ANIRA_ERROR_CONFIG at prepare, unless a stage of the pipeline fills the phase that
 * moves that ring (pre_process for an input, post_process for an output) and so takes
 * the difference on itself; the default bodies of anira/abi/stage.h still refuse such a
 * slot at run time. Set per tensor by canonical name, so an input and an output may
 * differ; ANIRA_DTYPE_F32 for every tensor never set. A name that matches no Streamed
 * tensor is checked at prepare, not here.
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
 * @brief The declared stream latency of one output under a Hard contract, in samples of that
 * output: what anira_handler_get_latency reports for the slot and what its receive ring
 * is primed with, REPLACING the figure anira computes (the buffer-adaptation delay, the
 * queue term, the wait credit and the model's internal latency;
 * docs/sphinx/latency.rst). A host that needs one figure across block sizes, or one that
 * aligns this stream with another, sets it; every output never named keeps the computed
 * figure. Never below the model's internal latency (anira_tensor_spec_set_latency: a
 * stream cannot deliver before the model does). Checked at anira_handler_prepare, not
 * here: a figure below that floor, and a name that matches no Streamed output (an input
 * has no stream latency), are ANIRA_ERROR_CONFIG there. Set per output by canonical
 * name, a second set of a name replacing the first. In a contract file the key is
 * "latencies": {"<name>": samples} in the hard object.
 * @param contract A Hard contract.
 * @param canonical The canonical name of a Streamed output (the one its spec was created with).
 * @param samples The stream latency in samples of that output, at most INT32_MAX.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL or empty name, or samples
 *         above INT32_MAX (the scheduler's per-output figure is a long: one bound on every
 *         platform).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_set_latency(anira_contract* contract,
                                                                  const char* canonical,
                                                                  uint32_t samples) ANIRA_NOEXCEPT;

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
 * @brief The host-end domain of one tensor, common to both contract kinds like
 * anira_contract_set_edge_cost: the domain anira allocates the ring, the model tensor
 * and the Static store of the slot in, and the domain all four phases of a stage, the
 * feed and the capture work in; for a State input it overrides the domain of the pair's
 * two buffers, which is the backend's domain of the plans that run the pair otherwise
 * (host memory for every backend of this pre-release), the domain the bound State
 * descriptors report; the planner joins it to the backend's domain per slot with the
 * edge of the registry's rows (anira_plan_slot.domain_in / domain_out / edge_class /
 * recipe). A tensor declared in the backend's own domain has no edge to cross. The ring,
 * the model tensor and the tensors the context accessors fill share the declared domain,
 * readable from the domain field of any tensor an accessor fills or from the plan row.
 * Set per tensor by canonical name, resolved at anira_handler_prepare: a name that
 * matches no tensor is ANIRA_ERROR_CONFIG there, and in this pre-release any domain but
 * ANIRA_DOMAIN_HOST is ANIRA_ERROR_NOT_SUPPORTED there, naming the tensor (the
 * declaration is data; anira's runtime allocates in host memory only). In a contract
 * file the key is a top-level "host_domains": {"<name>": "<domain word>"}, the words the
 * lower-case suffixes of anira_domain ("host", "host_pinned", "cuda", ...).
 * @param contract Either contract kind.
 * @param canonical The tensor's canonical name (the one its spec was created with): any tensor
 *        of either side, Streamed, Buffer, Static and State alike.
 * @param domain The domain of the tensor's host end; ANIRA_DOMAIN_HOST for every tensor that
 *        was never set.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL or empty name, or
 *         a domain that is no anira_domain value.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_set_host_domain(anira_contract* contract,
                                                                 const char* canonical,
                                                                 anira_domain domain) ANIRA_NOEXCEPT;

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
 * @brief Creates an empty model config: no models, no tensors, default engine NONE (= plan 0,
 * the first plan of the table), STATELESS, max_instances 1, anchor = the first Streamed
 * input.
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
 * decided at anira_handler_create (an entry of an engine the build lacks is skipped when
 * the candidates leave it out, refused when they name it), so a config can name every
 * engine a deployment might have.
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
 * @brief Appends a model entry for a custom engine, named by its reverse-URI id (the one
 * anira_custom_engine_create gave it; the entry and the engine's addition to the
 * pipeline, anira_pipeline_add_engine, may come in either order). An id no engine of the
 * pipeline has is ANIRA_ERROR_NOT_SUPPORTED at anira_handler_create, naming the id.
 * @param config The config.
 * @param engine_id A registered custom engine's name, reverse-URI (must contain a '.').
 * @param utf8_path Model file path, UTF-8, copied.
 * @param out_index Receives the model index, or NULL.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an id without a '.', or a NULL or empty
 *         path.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_model_path_engine_id(anira_model_config* config,
                                                                              const char* engine_id,
                                                                              const char* utf8_path,
                                                                              uint32_t* out_index,
                                                                              anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The bytes twin of anira_model_config_add_model_path_engine_id.
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
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_add_model_bytes_engine_id(anira_model_config* config,
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
 * @brief Pins the entry to a provider: its file is built for one (an ExecuTorch export lowered
 * for a provider (an ExecuTorch delegate), an ONNX Runtime .ort compiled for an
 * execution provider), so only a candidate naming that provider runs it, and a candidate
 * naming it picks this entry. An entry without a pin is neutral: it runs on any provider
 * of its engine, the candidate deciding. Two entries of one engine may coexist when
 * their pins differ. JSON: the entry's "provider" key beside its "engine", "coreml" (the
 * enum's spellings) or "com.example.npu" (a custom name); "default" or no key is a
 * neutral entry.
 * @param config The config.
 * @param model_index An entry.
 * @param provider A provider of the enum, or ANIRA_PROVIDER_DEFAULT with a provider_id for a
 *        custom one; DEFAULT with a NULL provider_id unpins the entry.
 * @param provider_id A custom provider's name in the engine's vocabulary (the runtime's own
 *        name for a built-in engine, an entry of a custom engine's providers list),
 *        copied; NULL for a provider of the enum.
 * @param err Nullable.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config, an index out of range,
 *         an unknown provider value, both a provider of the enum and a provider_id, or an empty
 *         provider_id.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_model_provider(anira_model_config* config,
                                                                        uint32_t model_index,
                                                                        anira_provider provider,
                                                                        const char* provider_id,
                                                                        anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The provider the entry is pinned to.
 * @param config The config.
 * @param model_index An entry.
 * @return The provider; ANIRA_PROVIDER_DEFAULT for a neutral entry, for an entry pinned to a
 *         custom provider (anira_model_config_model_provider_id names it) and for an index out
 *         of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_provider ANIRA_CALL anira_model_config_model_provider(const anira_model_config* config,
                                                                      uint32_t model_index) ANIRA_NOEXCEPT;

/**
 * @brief The custom provider the entry is pinned to.
 * @param config The config.
 * @param model_index An entry.
 * @return Object-owned; NULL for a neutral entry, for an entry pinned to a provider of the enum
 *         and for an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_model_provider_id(const anira_model_config* config,
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
 * @param export_name The export's name for that tensor, copied: ONNX Runtime the graph's input
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
                                                                     const char* export_name) ANIRA_NOEXCEPT;

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
 * @brief The engine the handler starts on: its first plan in plan-table order, on the default
 * provider when one is set (anira_model_config_set_default_provider), so an engine with
 * several plans (one entry under several providers, or several entries) starts on the
 * pinned provider. One rule for both defaults (this one and
 * anira_model_config_set_default_provider): what the configuration alone decides is
 * checked at anira_handler_create and at prepare, ANIRA_ERROR_CONFIG for a default
 * engine that names no model entry ("default_engine 'x' names no model entry") and for a
 * default provider no entry of the default engine could run (every one pinned to another
 * provider); whether a plan runs the default here is the candidates' and the context's,
 * and when none does the handler starts on the default engine's first plan, else on plan
 * 0, with one Warning at anira_handler_prepare naming what was asked and the plan it
 * starts on; never a refusal. anira_handler_get_plan says which plan runs.
 * @param config The config.
 * @param engine A built-in engine, or ANIRA_ENGINE_NONE = plan 0, the first plan of the table
 *        (default).
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an unknown engine.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_default_engine(anira_model_config* config,
                                                                        anira_engine engine) ANIRA_NOEXCEPT;

/**
 * @brief The custom twin of anira_model_config_set_default_engine: the engine named by its id.
 * An engine with several plans starts as anira_model_config_set_default_engine says.
 * @param config The config.
 * @param engine_id A registered custom engine's name, reverse-URI.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for an id without a '.'.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_default_engine_id(anira_model_config* config,
                                                                           const char* engine_id) ANIRA_NOEXCEPT;

/**
 * @brief The provider the handler starts on, beside the default engine: the handler starts on
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
 * @param config The config.
 * @param provider A provider of the enum, or ANIRA_PROVIDER_DEFAULT beside a provider_id;
 *        DEFAULT with a NULL provider_id sets none (default).
 * @param provider_id NULL, or a custom provider's name in the engine's own vocabulary
 *        (anira_provider), copied; only beside ANIRA_PROVIDER_DEFAULT.
 * @return ANIRA_OK, or ANIRA_ERROR_INVALID_ARGUMENT for a NULL config, a provider this header
 *         does not name, a provider of the enum and a provider_id at once, or an empty
 *         provider_id.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_set_default_provider(anira_model_config* config,
                                                                          anira_provider provider,
                                                                          const char* provider_id) ANIRA_NOEXCEPT;

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
 *        edge_cost and an optional top-level host_domains ({"<name>": "<domain word>"},
 *        anira_contract_set_host_domain), or a v2 document.
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

/**
 * @brief The canonical name the spec was created with.
 * @param spec The spec.
 * @return Object-owned; NULL for a NULL spec.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_tensor_spec_name(const anira_tensor_spec* spec)
                                                        ANIRA_NOEXCEPT;

/**
 * @brief The element type the spec was created with: the model's.
 * @param spec The spec.
 * @return The dtype; 0, which is no dtype, for a NULL spec.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_dtype ANIRA_CALL anira_tensor_spec_dtype(const anira_tensor_spec* spec)
                                                         ANIRA_NOEXCEPT;

/**
 * @brief The role the spec was created with.
 * @param spec The spec.
 * @return The role; ANIRA_ROLE_FORCE32, which is no role, for a NULL spec.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_role ANIRA_CALL anira_tensor_spec_role(const anira_tensor_spec* spec)
                                                       ANIRA_NOEXCEPT;

/**
 * @brief The rank: one more than the highest axis anira_tensor_spec_set_axis set. An axis below
 * it that was never set is a hole, which anira_tensor_spec_axis reads as
 * {ANIRA_AXIS_ANY, 0}.
 * @param spec The spec.
 * @return The rank; 0 for a spec without axes and for a NULL spec.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_tensor_spec_ndim(const anira_tensor_spec* spec) ANIRA_NOEXCEPT;

/**
 * @brief One axis as anira_tensor_spec_set_axis stored it. A hole below the rank reads
 * {ANIRA_AXIS_ANY, 0}, the value of an axis never set.
 * @param spec The spec.
 * @param i The axis, below anira_tensor_spec_ndim.
 * @param tag Receives the axis tag.
 * @param extent Receives the extent, ANIRA_DYNAMIC where it was set so.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec, a NULL out-parameter or an
 *         axis at or beyond the rank.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_axis(const anira_tensor_spec* spec,
                                                         uint32_t i,
                                                         anira_axis_tag* tag,
                                                         int64_t* extent) ANIRA_NOEXCEPT;

/**
 * @brief The window as anira_tensor_spec_set_window stored it; 0, 0, 0 until it ran, on every
 * role.
 * @param spec The spec.
 * @param window_min Receives the smallest legal Time extent.
 * @param window_max Receives the largest, or ANIRA_UNBOUNDED.
 * @param overlap Receives the overlap of consecutive windows.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec or a NULL out-parameter.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_window(const anira_tensor_spec* spec,
                                                           int64_t* window_min,
                                                           int64_t* window_max,
                                                           int64_t* overlap) ANIRA_NOEXCEPT;

/**
 * @brief The time ratio as anira_tensor_spec_set_time_ratio stored it.
 * @param spec The spec.
 * @param num Receives the numerator.
 * @param den Receives the denominator; (0, 0) = derive.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL spec or a NULL out-parameter.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_spec_time_ratio(const anira_tensor_spec* spec,
                                                               int64_t* num,
                                                               int64_t* den) ANIRA_NOEXCEPT;

/**
 * @brief The model-internal latency anira_tensor_spec_set_latency stored.
 * @param spec The spec.
 * @return The latency in elements; 0 for a spec without one and for a NULL spec.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API int64_t ANIRA_CALL anira_tensor_spec_latency(const anira_tensor_spec* spec)
                                                       ANIRA_NOEXCEPT;

/**
 * @brief The canonical name of the State output a State input is fed from
 * (anira_tensor_spec_set_state_source).
 * @param spec The spec.
 * @return Object-owned; NULL for a spec without one and for a NULL spec.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_tensor_spec_state_source(const anira_tensor_spec* spec)
                                                                ANIRA_NOEXCEPT;

/**
 * @brief The number of input specs, State specs included.
 * @param config The config.
 * @return The count; 0 for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_model_config_num_inputs(const anira_model_config* config)
                                                            ANIRA_NOEXCEPT;

/**
 * @brief The number of output specs, State specs included.
 * @param config The config.
 * @return The count; 0 for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_model_config_num_outputs(const anira_model_config* config)
                                                             ANIRA_NOEXCEPT;

/**
 * @brief The config's own copy of an input spec (anira_model_config_add_input copied it), for
 * the anira_tensor_spec getters only: never a spec to set or to destroy. Valid until the
 * config's next anira_model_config_add_input or anira_model_config_add_output, which
 * moves its specs, or its destruction.
 * @param config The config.
 * @param index The slot: the spec's position in the input list.
 * @return Object-owned; NULL for a NULL config or an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const anira_tensor_spec* ANIRA_CALL anira_model_config_input(const anira_model_config* config,
                                                                       uint32_t index) ANIRA_NOEXCEPT;

/**
 * @brief The output twin of anira_model_config_input, with the same lifetime.
 * @param config The config.
 * @param index The slot: the spec's position in the output list.
 * @return Object-owned; NULL for a NULL config or an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const anira_tensor_spec* ANIRA_CALL anira_model_config_output(const anira_model_config* config,
                                                                        uint32_t index) ANIRA_NOEXCEPT;

/**
 * @brief The built-in engine anira_model_config_set_default_engine stored.
 * @param config The config.
 * @return The engine; ANIRA_ENGINE_NONE for the default (plan 0), for a custom default
 *         (anira_model_config_default_engine_id names it) and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_engine ANIRA_CALL anira_model_config_default_engine(const anira_model_config* config)
                                                                    ANIRA_NOEXCEPT;

/**
 * @brief The custom engine anira_model_config_set_default_engine_id stored.
 * @param config The config.
 * @return Object-owned; NULL for a built-in default, for no default and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_default_engine_id(const anira_model_config* config)
                                                                      ANIRA_NOEXCEPT;

/**
 * @brief The provider of the enum anira_model_config_set_default_provider stored.
 * @param config The config.
 * @return The provider; ANIRA_PROVIDER_DEFAULT for none, for a custom default provider
 *         (anira_model_config_default_provider_id names it) and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_provider ANIRA_CALL anira_model_config_default_provider(const anira_model_config* config)
                                                                        ANIRA_NOEXCEPT;

/**
 * @brief The custom provider anira_model_config_set_default_provider stored.
 * @param config The config.
 * @return Object-owned; NULL for a provider of the enum, for none and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_default_provider_id(const anira_model_config* config)
                                                                        ANIRA_NOEXCEPT;

/**
 * @brief The state as anira_model_config_set_state stored it; a config with a declared State
 * pair runs as ANIRA_MODEL_STATEFUL whatever it reads.
 * @param config The config.
 * @return The state; ANIRA_MODEL_STATE_FORCE32, which is no state, for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_model_state ANIRA_CALL anira_model_config_state(const anira_model_config* config)
                                                                ANIRA_NOEXCEPT;

/**
 * @brief The instance limit anira_model_config_set_max_instances stored.
 * @param config The config.
 * @return The limit, at least 1; 0 for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_model_config_max_instances(const anira_model_config* config)
                                                               ANIRA_NOEXCEPT;

/**
 * @brief The anchor's canonical name as anira_model_config_set_anchor wrote it, unresolved:
 * whether it names a streamed tensor is prepare's question.
 * @param config The config.
 * @return Object-owned; NULL for the default anchor and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_anchor(const anira_model_config* config)
                                                           ANIRA_NOEXCEPT;

/**
 * @brief The export's name of the tensor in the entry's tensor record
 * (anira_model_config_set_tensor_name, the JSON file's models[].tensors).
 * @param config The config.
 * @param model_index An entry.
 * @param canonical The tensor's canonical name.
 * @return Object-owned; NULL when the entry binds the tensor positionally, and for a NULL
 *         config or canonical or an index out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const char* ANIRA_CALL anira_model_config_tensor_name(const anira_model_config* config,
                                                                uint32_t model_index,
                                                                const char* canonical) ANIRA_NOEXCEPT;

/**
 * @brief The axis order of the tensor in the entry's tensor record, with the count protocol of
 * anira_handler_get_latencies.
 * @param config The config.
 * @param model_index An entry.
 * @param canonical The tensor's canonical name.
 * @param count In: the capacity of axes; out: the length of the layout, 0 when the entry holds
 *        the tensor in the spec's order.
 * @param axes Receives the layout as anira_model_config_set_tensor_layout stored it (per
 *        position of the file, a spec axis or ANIRA_AXIS_INSERT); or NULL to ask for the
 *        length.
 * @return ANIRA_OK, count 0 for a tensor without a layout; ANIRA_INCOMPLETE when axes is too
 *         short (count holds the length, the first capacity entries are written);
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL config or count, an index out of range, or a
 *         NULL or empty canonical.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_model_config_tensor_layout(const anira_model_config* config,
                                                                   uint32_t model_index,
                                                                   const char* canonical,
                                                                   uint32_t* count,
                                                                   uint32_t* axes) ANIRA_NOEXCEPT;

/**
 * @brief The typed record of a known extension kind on one entry (host "model"), set as a
 * struct or as JSON alike: the header of the kind's payload, read within
 * header->struct_size (for "entry", ((const anira_ext_entry*)header)->name). The record
 * and its strings are object-owned, valid until the entry's next set of that kind or the
 * config's destruction.
 * @param config The config.
 * @param model_index An entry.
 * @param kind The extension kind, e.g. "entry".
 * @return Object-owned; NULL for a kind the entry does not carry, for a kind this build does
 *         not register (stored for prepare to name), and for a NULL config or kind or an index
 *         out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API const anira_ext_header* ANIRA_CALL anira_model_config_model_ext(const anira_model_config* config,
                                                                          uint32_t model_index,
                                                                          const char* kind) ANIRA_NOEXCEPT;

/**
 * @brief The geometry as anira_contract_create_hard or anira_contract_hard_set_geometry stored
 * it; 0, 0, 0 on the legacy contract of a version 2 upgrade until set.
 * @param contract A Hard contract.
 * @param block_min Receives the smallest block.
 * @param block_max Receives the largest block.
 * @param rate Receives the rate.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract or a NULL out-parameter.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_geometry(const anira_contract* contract,
                                                               uint32_t* block_min,
                                                               uint32_t* block_max,
                                                               double* rate) ANIRA_NOEXCEPT;

/**
 * @brief The budget as anira_contract_hard_set_budget stored it.
 * @param contract A Hard contract.
 * @param kind Receives the budget kind.
 * @param explicit_ms Receives the explicit budget in milliseconds; 0 unless the kind is
 *        ANIRA_BUDGET_EXPLICIT.
 * @return As anira_contract_hard_geometry.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_budget(const anira_contract* contract,
                                                             anira_budget_kind* kind,
                                                             double* explicit_ms) ANIRA_NOEXCEPT;

/**
 * @brief The warmup as anira_contract_hard_set_warmup stored it.
 * @param contract A Hard contract.
 * @param mode Receives the warmup mode.
 * @param iterations Receives the iterations; 0 unless the mode is ANIRA_WARMUP_FIXED.
 * @return As anira_contract_hard_geometry.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_warmup(const anira_contract* contract,
                                                             anira_warmup_mode* mode,
                                                             uint32_t* iterations) ANIRA_NOEXCEPT;

/**
 * @brief The miss policy as anira_contract_hard_set_on_miss stored it.
 * @param contract A Hard contract.
 * @param policy Receives the miss policy.
 * @return As anira_contract_hard_geometry.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_on_miss(const anira_contract* contract,
                                                              anira_miss_policy* policy) ANIRA_NOEXCEPT;

/**
 * @brief The pair anira_contract_hard_set_miss_fn stored, whatever the policy; a contract file
 * carries none.
 * @param contract A Hard contract.
 * @param fn Receives the backup function; NULL when none is set.
 * @param user_data Receives its user_data; NULL when no function is set.
 * @return As anira_contract_hard_geometry.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_miss_fn(const anira_contract* contract,
                                                              anira_miss_fn* fn,
                                                              void** user_data) ANIRA_NOEXCEPT;

/**
 * @brief The wait ratio as anira_contract_hard_set_wait_ratio stored it.
 * @param contract A Hard contract.
 * @param ratio Receives the wait ratio.
 * @return As anira_contract_hard_geometry.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_wait_ratio(const anira_contract* contract,
                                                                 double* ratio) ANIRA_NOEXCEPT;

/**
 * @brief The number of tensors anira_contract_hard_set_ring_dtype (or a contract file's
 * ring_dtypes) named; every other tensor's ring dtype is ANIRA_DTYPE_F32.
 * @param contract A Hard contract.
 * @return The count; 0 on an Async contract and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_contract_hard_num_ring_dtypes(const anira_contract* contract)
                                                                  ANIRA_NOEXCEPT;

/**
 * @brief One ring dtype the contract names, enumerated by index in bytewise order of the
 * canonical names: what lists what was set, which a lookup by name could not (it would
 * answer ANIRA_DTYPE_F32 for a name never set and for one set to it alike).
 * @param contract A Hard contract.
 * @param index Below anira_contract_hard_num_ring_dtypes.
 * @param canonical Receives the tensor's canonical name (object-owned).
 * @param dtype Receives its ring dtype.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL out-parameter or an index
 *         out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_ring_dtype(const anira_contract* contract,
                                                                 uint32_t index,
                                                                 const char** canonical,
                                                                 anira_dtype* dtype) ANIRA_NOEXCEPT;

/**
 * @brief The number of outputs anira_contract_hard_set_latency (or a contract file's latencies)
 * named; every other output keeps the computed figure.
 * @param contract A Hard contract.
 * @return The count; 0 on an Async contract and for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_contract_hard_num_latencies(const anira_contract* contract)
                                                                ANIRA_NOEXCEPT;

/**
 * @brief One declared stream latency the contract names, enumerated by index in bytewise order
 * of the canonical names, the shape of anira_contract_hard_ring_dtype.
 * @param contract A Hard contract.
 * @param index Below anira_contract_hard_num_latencies.
 * @param canonical Receives the output's canonical name (object-owned).
 * @param samples Receives its declared stream latency in samples.
 * @return ANIRA_OK; ANIRA_ERROR_WRONG_CONTRACT on an Async contract;
 *         ANIRA_ERROR_INVALID_ARGUMENT for a NULL contract, a NULL out-parameter or an index
 *         out of range.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_contract_hard_latency(const anira_contract* contract,
                                                              uint32_t index,
                                                              const char** canonical,
                                                              uint32_t* samples) ANIRA_NOEXCEPT;

/**
 * @brief The edge cost policy anira_contract_set_edge_cost stored.
 * @param contract Either contract kind.
 * @return The policy; ANIRA_EDGE_COST_PERMISSIVE, the default, for NULL.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_edge_cost ANIRA_CALL anira_contract_edge_cost(const anira_contract* contract)
                                                              ANIRA_NOEXCEPT;

/**
 * @brief The pool size and the wait strategy as anira_context_config_set_threads (or a context
 * file) stored them.
 * @param config The config.
 * @param num_threads Receives the pool size; ANIRA_THREADS_AUTO when never set.
 * @param wait Receives the wait strategy.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL config or a NULL out-parameter.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_threads(const anira_context_config* config,
                                                               uint32_t* num_threads,
                                                               anira_wait_strategy* wait) ANIRA_NOEXCEPT;

/**
 * @brief Every log field the config stores, in the record anira_context_config_set_log takes:
 * the sink and its user_data, the level, the drain mode and interval, the queue capacity
 * (clamped as the setters clamp it) and the flags.
 * @param config The config.
 * @param out Receives the log settings, written within its struct_size (set it before the
 *        call), which it keeps; abi_version receives this build's ANIRA_ABI_VERSION.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument or a struct_size below the
 *         three leading slots (struct_size, abi_version, user_data).
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_context_config_log(const anira_context_config* config,
                                                           anira_log_desc* out) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_CONFIG_H */
