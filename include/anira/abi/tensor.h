/*
 * anira/abi/tensor.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_TENSOR_H
#define ANIRA_ABI_TENSOR_H

/**
 * @file tensor.h
 * @brief The runtime tensor: the three frozen Tier-1 PODs, their factories and accessors.
 *
 * anira_tensor is the unit of data between a host and anira: a plain descriptor of memory,
 * trivially copyable through lock-free FIFOs, 216 bytes on every target. It carries
 * user-to-anira information only. Every arm of anira_memory_handle is typeless memory; the
 * descriptor (dtype, shape, strides, byte_offset) is the only type, and pixel formats never
 * appear on a tensor. The three records are Tier 1: no struct_size, no version field, the ABI
 * major is their version, and the layout table committed under abi/ pins every offset. Every
 * pointer sits in an ANIRA_PTR slot, so a producer zeroes the record before filling it; the
 * factories do. The factories are field fills: they check the dtype, the rank and the extents,
 * nothing else, and consume nothing. The accessors read the two host domains
 * (ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST_PINNED) only; the factories of the device domains fill
 * the record for adapters that arrive later. Planar memory (ANIRA_TENSOR_PLANAR: one pointer
 * per plane of axis 0 instead of one block) is a boundary representation of host memory: an
 * entry accepts a planar tensor only where its own documentation says so, which is where it
 * copies a host block; every other consumer refuses the flag with ANIRA_ERROR_NOT_SUPPORTED, as
 * does every domain but the two host domains, and anira never hands a planar tensor to a stage,
 * a backend or JavaScript. The factories of the unmeasured platform arms live in
 * anira/abi/draft/tensor_platform.h.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief The memory of a tensor, one arm per anira_domain (HOST and HOST_PINNED share host); 24
 * bytes on every target. Tier 1: frozen. Every arm is typeless memory and every vendor
 * type is replaced by its wire width; image-typed handles (a pixel CVPixelBuffer, a
 * multi-plane dma-buf) are Frames, not tensors. All thirteen arms are declared on every
 * platform; which domains a build consumes is a run-time property
 * (anira_capabilities_domains).
 */
typedef union anira_memory_handle {
    struct {
        /**
         * The base of the allocation; the tensor's byte_offset counts from it.
         */
        ANIRA_PTR(void, ptr);
    } host;  /**< ANIRA_DOMAIN_HOST and ANIRA_DOMAIN_HOST_PINNED. */
    struct {
        ANIRA_PTR(void, ptr);  /**< The device pointer. */
        int32_t device;  /**< CUDA device ordinal. */
    } cuda;  /**< ANIRA_DOMAIN_CUDA. */
    struct {
        uint32_t id;  /**< GLuint buffer name. */
        uint32_t target;  /**< GLenum binding target. */
    } gl;  /**< ANIRA_DOMAIN_GL_BUFFER. */
    struct {
        uint64_t buffer;  /**< VkBuffer (non-dispatchable, 64-bit on every target). */
        uint64_t memory;  /**< VkDeviceMemory. */
        uint64_t offset;  /**< Byte offset of the buffer in the memory. */
    } vk;  /**< ANIRA_DOMAIN_VULKAN_BUFFER: native handles of the same process. */
    struct {
        int32_t fd;  /**< The exported fd; on Windows the NT handle (32 bits significant). */
        uint32_t reserved;  /**< Zero. */
        uint64_t size;  /**< Size of the exported allocation in bytes. */
    } opaque;  /**< ANIRA_DOMAIN_OPAQUE_FD: exported opaque memory. */
    struct {
        ANIRA_PTR(void, buffer);  /**< id<MTLBuffer>. */
    } mtl;  /**< ANIRA_DOMAIN_METAL_BUFFER. */
    struct {
        ANIRA_PTR(void, surface);  /**< IOSurfaceRef. */
        uint64_t size;  /**< Size of plane 0 in bytes. */
    } iosurface;  /**< ANIRA_DOMAIN_IOSURFACE: plane 0, byte-image encoded packed floats. */
    struct {
        ANIRA_PTR(void, buffer);  /**< WGPUBuffer. */
        uint64_t offset;  /**< Byte offset into the buffer. */
    } wgpu;  /**< ANIRA_DOMAIN_WGPU_BUFFER: a buffer of the core's Dawn device, same process. */
    struct {
        int32_t fd;  /**< The dma-buf fd. */
        uint32_t reserved;  /**< Zero. */
        uint64_t size;  /**< Size of the exported memory in bytes. */
        uint64_t offset;  /**< Byte offset into the exported memory. */
    } dmabuf;  /**< ANIRA_DOMAIN_DMABUF: exported buffer memory, typeless. */
    struct {
        ANIRA_PTR(void, buffer);  /**< AHardwareBuffer*. */
    } ahb;  /**< ANIRA_DOMAIN_AHARDWAREBUFFER: buffer-typed (BLOB). */
    struct {
        ANIRA_PTR(void, resource);  /**< ID3D12Resource*. */
        ANIRA_PTR(void, shared_handle);  /**< The NT shared handle of the resource, or NULL. */
    } d3d12;  /**< ANIRA_DOMAIN_D3D12. */
    struct {
        /**
         * count plane pointers, plane i at ptrs[i]; the array is borrowed like the memory it
         * names.
         */
        ANIRA_PTR(void* const, ptrs);
        uint32_t count;  /**< The number of planes; equals shape[0]. */
        uint32_t reserved;  /**< Zero. */
    } planes;
    /**< ANIRA_DOMAIN_HOST and ANIRA_DOMAIN_HOST_PINNED under ANIRA_TENSOR_PLANAR: one pointer
     *   per plane of axis 0. Not a domain of its own: domain stays the planes' real domain.
     */
    /**
     * The handle as words: copy, compare, key a plan cache (never by the words of a planar
     * handle, which name the caller's pointer array and not the memory).
     */
    uint64_t raw[3];
} anira_memory_handle;

/**
 * @brief A fence on a tensor's memory; kind ANIRA_SYNC_NONE means the data is already visible.
 * 24 bytes on every target. Tier 1: frozen. The two fd kinds (ANIRA_SYNC_SYNC_FILE_FD,
 * ANIRA_SYNC_OPAQUE_FD_SEMAPHORE) own their descriptor: the token closes it when reset
 * (anira_sync_token_reset), every hand-off is a transfer, and a holder that needs it
 * longer duplicates it (anira_sync_token_dup). Every other kind is a non-owning handle.
 */
typedef struct anira_sync_token {
    uint32_t kind;  /**< anira_sync_kind: which member of u is live. */
    uint32_t flags;  /**< No bit is defined at v3.0.0; zero. */
    union {
        ANIRA_PTR(void, cuda_event);  /**< ANIRA_SYNC_CUDA_EVENT: cudaEvent_t. */
        struct {
            uint64_t semaphore;  /**< VkSemaphore (timeline). */
            uint64_t value;  /**< The timeline value that signals. */
        } vk;  /**< ANIRA_SYNC_VK_TIMELINE. */
        ANIRA_PTR(void, gl_sync);  /**< ANIRA_SYNC_GL_SYNC: GLsync. */
        /**
         * ANIRA_SYNC_SYNC_FILE_FD and ANIRA_SYNC_OPAQUE_FD_SEMAPHORE: the fd, owned by the
         * token; negative = none. On Windows the NT handle (32 bits significant); 0 or negative
         * = none.
         */
        int32_t fd;
        struct {
            ANIRA_PTR(void, object);  /**< id<MTLSharedEvent>. */
            uint64_t value;  /**< The event value that signals. */
        } mtl;  /**< ANIRA_SYNC_MTL_SHARED_EVENT. */
        struct {
            ANIRA_PTR(void, object);  /**< ID3D12Fence*. */
            uint64_t value;  /**< The fence value that signals. */
        } d3d12;  /**< ANIRA_SYNC_D3D12_FENCE. */
        uint64_t raw[2];  /**< The payload as words. */
    } u;
    /**< The payload of kind; never read under ANIRA_SYNC_NONE and ANIRA_SYNC_QUEUE_ORDERED.
     */
} anira_sync_token;

/* Forward declaration: struct anira_tensor is defined below. */
typedef struct anira_tensor anira_tensor;

/**
 * @brief The release callback of a tensor. A function type, not a pointer type: the slot is
 * ANIRA_PTR(anira_tensor_release_proc, release). It unmaps, unregisters, recycles or
 * frees what the producer attached through manager_ctx. anira calls a non-NULL release
 * exactly once per submitted copy: on an inference thread when the job reaches a
 * terminal state, on the caller of the poll and wait entries under polled delivery, or
 * on the caller of anira_handler_prepare or anira_handler_destroy for a job still
 * outstanding then; never on the driver thread, and never for a tensor it did not
 * receive through a submit or a bind. Not real-time: it may block and free. In this
 * pre-release no entry point takes ownership of a tensor, so anira never calls it and
 * the holder of a descriptor with a non-NULL release calls it once itself.
 * @param tensor The descriptor being released; the callback reads its manager_ctx.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 */
typedef void (ANIRA_CALL anira_tensor_release_proc)(anira_tensor* tensor);

/**
 * @brief The runtime tensor: a descriptor of memory, never the memory itself. 216 bytes, align
 * 8, identical on wasm32, LP64 and LLP64, trivially copyable. Tier 1: frozen; a new
 * shape is a new struct with new functions. Zero the record before filling it by hand
 * (the factories do): the high half of an ANIRA_PTR slot is otherwise undefined on a
 * 32-bit target.
 */
struct anira_tensor {
    uint32_t domain;  /**< anira_domain: which arm of handle is live. */
    anira_dtype dtype;  /**< The element type; 0 is not a type. */
    uint32_t ndim;  /**< The rank, 0 to ANIRA_MAX_RANK; rank 0 is one element. */
    /**
     * anira_tensor_flags bits; the factories leave it 0, except anira_tensor_init_host_planar,
     * which sets ANIRA_TENSOR_PLANAR.
     */
    uint32_t flags;
    /**
     * The extents: counts, never ANIRA_DYNAMIC; zero from ndim on.
     */
    int64_t shape[ANIRA_MAX_RANK];
    /**
     * In elements; all-zero = packed row-major. Under ANIRA_TENSOR_PLANAR strides[0] is ignored
     * and the rest apply inside each plane.
     */
    int64_t strides[ANIRA_MAX_RANK];
    /**
     * Byte offset of the first element from the base of handle (under ANIRA_TENSOR_PLANAR: from
     * each plane pointer); what a view slices with.
     */
    uint64_t byte_offset;
    anira_memory_handle handle;  /**< The memory: the arm domain names. */
    /**
     * Producer bookkeeping only (a pool slot, a refcounted view parent): what release reads,
     * and nothing else reads it; never edge state.
     */
    ANIRA_PTR(void, manager_ctx);
    /**
     * NULL = borrowed: the memory is the caller's and stays valid until the ticket is terminal
     * (Async) or the call returns (Hard). Else unmaps, unregisters, recycles or frees; see
     * anira_tensor_release_proc.
     */
    ANIRA_PTR(anira_tensor_release_proc, release);
    /**
     * An input: the data is valid once this signals. A bound output: the buffer is free to
     * write once this signals.
     */
    anira_sync_token acquire;
};

/**
 * @brief Fills a tensor over host memory. Like every anira_tensor_init_* factory it zeroes the
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
 * field fill: legal from a render thread and from inside a stage callback.
 * @param tensor The caller's record; NULL is a no-op.
 * @param data The first byte of pageable host memory; borrowed.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_host(anira_tensor* tensor,
                                                 void* data,
                                                 anira_dtype dtype,
                                                 uint32_t ndim,
                                                 const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief anira_tensor_init_host for page-locked memory: domain ANIRA_DOMAIN_HOST_PINNED, the
 * same host arm, the same refusals.
 * @param tensor The caller's record; NULL is a no-op.
 * @param data The first byte of page-locked host memory; borrowed.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_pinned(anira_tensor* tensor,
                                                   void* data,
                                                   anira_dtype dtype,
                                                   uint32_t ndim,
                                                   const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over planar host memory, one pointer per plane of axis 0 (the channel
 * pointers of an audio host): domain ANIRA_DOMAIN_HOST, flags ANIRA_TENSOR_PLANAR,
 * handle.planes. The pointer array travels as const void* so that a float** and a const
 * float* const* both convert without a cast, in C and in C++; it is read as an array of
 * void*, which every supported target represents like any other object pointer. There is
 * no flags parameter: a caller that built the tensor over read-only planes ORs
 * ANIRA_TENSOR_READ_ONLY into flags afterwards, and one over page-locked planes assigns
 * domain ANIRA_DOMAIN_HOST_PINNED. Zero-then-fill and the refusals of
 * anira_tensor_init_host, and two more: a rank of 0 (there is no plane axis) and a count
 * other than shape[0] leave the record all-zero. anira_tensor_data and
 * anira_tensor_data_f32 return NULL for a planar tensor; anira_tensor_plane reads it.
 * @param tensor The caller's record; NULL is a no-op.
 * @param planes An array of count object pointers of one pointee type (a float**, a const
 *        int16_t* const*), read as void* objects; borrowed like the memory it names.
 *        NULL is accepted: an empty tensor.
 * @param count The number of planes; must equal shape[0].
 * @param dtype The element type.
 * @param ndim The rank, 1 to ANIRA_MAX_RANK: axis 0 is the plane axis.
 * @param shape ndim extents, each 0 or more.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_host_planar(anira_tensor* tensor,
                                                        const void* planes,
                                                        uint32_t count,
                                                        anira_dtype dtype,
                                                        uint32_t ndim,
                                                        const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over CUDA device memory: domain ANIRA_DOMAIN_CUDA, handle.cuda. A
 * non-NULL cuda_event becomes acquire (ANIRA_SYNC_CUDA_EVENT, non-owning); NULL leaves
 * ANIRA_SYNC_NONE. Zero-then-fill and the refusals of anira_tensor_init_host. In this
 * pre-release a field fill only: no adapter consumes the arm.
 * @param tensor The caller's record; NULL is a no-op.
 * @param ptr The CUDA device pointer.
 * @param device CUDA device ordinal.
 * @param cuda_event cudaEvent_t the data is valid after, or NULL: already visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_cuda(anira_tensor* tensor,
                                                 void* ptr,
                                                 int32_t device,
                                                 void* cuda_event,
                                                 anira_dtype dtype,
                                                 uint32_t ndim,
                                                 const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over an OpenGL buffer object: domain ANIRA_DOMAIN_GL_BUFFER, handle.gl.
 * A non-NULL gl_sync becomes acquire (ANIRA_SYNC_GL_SYNC, non-owning); NULL leaves
 * ANIRA_SYNC_NONE. Consuming it requires the GL thread policy of the context config
 * (anira_gl_desc.threads). Zero-then-fill and the refusals of anira_tensor_init_host. In
 * this pre-release a field fill only: no adapter consumes the arm.
 * @param tensor The caller's record; NULL is a no-op.
 * @param id GLuint buffer name.
 * @param target GLenum binding target.
 * @param gl_sync GLsync the data is valid after, or NULL: already visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_gl_buffer(anira_tensor* tensor,
                                                      uint32_t id,
                                                      uint32_t target,
                                                      void* gl_sync,
                                                      anira_dtype dtype,
                                                      uint32_t ndim,
                                                      const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over a native Vulkan buffer of the same process: domain
 * ANIRA_DOMAIN_VULKAN_BUFFER, handle.vk. A non-zero timeline_semaphore becomes acquire
 * (ANIRA_SYNC_VK_TIMELINE with value, non-owning); 0 leaves ANIRA_SYNC_NONE. The handles
 * travel at their wire width, which makes this one of the allowlisted 64-bit
 * ANIRA_NONBLOCKING declarations: nothing that produces a VkBuffer runs in JS.
 * Zero-then-fill and the refusals of anira_tensor_init_host. In this pre-release a field
 * fill only: no adapter consumes the arm.
 * @param tensor The caller's record; NULL is a no-op.
 * @param buffer VkBuffer.
 * @param memory VkDeviceMemory.
 * @param offset Byte offset of the buffer in the memory.
 * @param timeline_semaphore VkSemaphore (timeline) the data is valid after, or 0
 *        (VK_NULL_HANDLE): already visible.
 * @param value The timeline value that signals; ignored without a semaphore.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_vulkan(anira_tensor* tensor,
                                                   uint64_t buffer,
                                                   uint64_t memory,
                                                   uint64_t offset,
                                                   uint64_t timeline_semaphore,
                                                   uint64_t value,
                                                   anira_dtype dtype,
                                                   uint32_t ndim,
                                                   const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over exported opaque memory: domain ANIRA_DOMAIN_OPAQUE_FD,
 * handle.opaque. It takes no fence: acquire stays ANIRA_SYNC_NONE, and a producer with
 * one fills acquire after the call. Allowlisted 64-bit ANIRA_NONBLOCKING declaration (a
 * byte size at its wire width). Zero-then-fill and the refusals of
 * anira_tensor_init_host. In this pre-release a field fill only: no adapter consumes the
 * arm.
 * @param tensor The caller's record; NULL is a no-op.
 * @param fd The exported opaque fd; on Windows the NT handle. Not owned by the tensor.
 * @param size Size of the exported allocation in bytes.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_opaque_fd(anira_tensor* tensor,
                                                      int32_t fd,
                                                      uint64_t size,
                                                      anira_dtype dtype,
                                                      uint32_t ndim,
                                                      const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over a WebGPU buffer: domain ANIRA_DOMAIN_WGPU_BUFFER, handle.wgpu. The
 * token is copied into acquire and that copy is the hand-off: an owned fd moves with it
 * and the caller does not reset its source. anira never fabricates a fence: NULL is
 * ANIRA_SYNC_NONE. When the call is refused the record is all-zero, the token was not
 * copied and stays the caller's. Allowlisted 64-bit ANIRA_NONBLOCKING declaration.
 * Zero-then-fill and the refusals of anira_tensor_init_host. In this pre-release a field
 * fill only: no adapter consumes the arm.
 * @param tensor The caller's record; NULL is a no-op.
 * @param wgpu_buffer WGPUBuffer of the core's Dawn device.
 * @param offset Byte offset into the buffer.
 * @param fence Copied into acquire, or NULL: already visible (ANIRA_SYNC_NONE). Pass kind
 *        ANIRA_SYNC_QUEUE_ORDERED for work still on the producing queue.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_wgpu_buffer(anira_tensor* tensor,
                                                        void* wgpu_buffer,
                                                        uint64_t offset,
                                                        const anira_sync_token* fence,
                                                        anira_dtype dtype,
                                                        uint32_t ndim,
                                                        const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor over exported buffer memory (Vulkan external memory, a GBM linear bo, a
 * dma-heap): domain ANIRA_DOMAIN_DMABUF, handle.dmabuf, typed by the descriptor like
 * every other arm. A sync_fd of 0 or more becomes acquire (ANIRA_SYNC_SYNC_FILE_FD): the
 * token owns it, so the producer must not close it and the holder of the record ends it
 * with anira_sync_token_reset; a negative sync_fd leaves ANIRA_SYNC_NONE. When the call
 * is refused the record is all-zero and sync_fd stays the caller's. Allowlisted 64-bit
 * ANIRA_NONBLOCKING declaration. Zero-then-fill and the refusals of
 * anira_tensor_init_host. In this pre-release a field fill only: no adapter consumes the
 * arm.
 * @param tensor The caller's record; NULL is a no-op.
 * @param fd The dma-buf fd of exported buffer memory. Not owned by the tensor.
 * @param size Size of the exported memory in bytes.
 * @param offset Byte offset into the exported memory.
 * @param sync_fd A sync_file fd the data is valid after, owned by the tensor's acquire token
 *        from here on; negative: already visible.
 * @param dtype The element type.
 * @param ndim The rank, at most ANIRA_MAX_RANK.
 * @param shape ndim extents, each 0 or more; may be NULL when ndim is 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_tensor_init_dmabuf(anira_tensor* tensor,
                                                   int32_t fd,
                                                   uint64_t size,
                                                   uint64_t offset,
                                                   int32_t sync_fd,
                                                   anira_dtype dtype,
                                                   uint32_t ndim,
                                                   const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Fills a tensor from a DLPack 1.x managed tensor. kDLCPU maps to ANIRA_DOMAIN_HOST and
 * kDLCUDAHost to ANIRA_DOMAIN_HOST_PINNED; the DLDataType is the anira_dtype (codes 0 to
 * 6, any bits and lanes); shape is copied, strides are copied or, when NULL (a producer
 * older than DLPack 1.2), left all-zero = packed row-major; strides that are present and
 * all zero over more than one element (a fully broadcast view) are refused, because
 * all-zero is this record's spelling of packed; data and byte_offset are copied;
 * DLPACK_FLAG_BITMASK_READ_ONLY sets ANIRA_TENSOR_READ_ONLY; acquire is ANIRA_SYNC_NONE;
 * the result is never planar. Ownership: on success the managed tensor belongs to the
 * record: manager_ctx holds it and release is a trampoline that disarms the record
 * (release and manager_ctx NULL) and then calls the producer's deleter exactly once; a
 * NULL deleter gives a borrowed tensor (release NULL). The trampoline guards one
 * descriptor, not its bitwise copies: call release on one copy only, and expect it to be
 * as blocking as the producer's deleter. On failure nothing is consumed: anira never
 * calls the deleter of a tensor it refused, the caller keeps ownership, and *tensor is
 * untouched.
 * @param tensor The caller's record; written on success only.
 * @param dl_managed_tensor_versioned A DLManagedTensorVersioned* of DLPack major 1, passed as
 *        void* so that this header never includes dlpack.h.
 * @param err Nullable.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL tensor or managed tensor, a rank
 *         below 0 or above ANIRA_MAX_RANK, a NULL shape with a rank above 0, a negative extent,
 *         or a dtype of 0 bits or 0 lanes; ANIRA_ERROR_NOT_SUPPORTED for a DLPack major other
 *         than 1 (nothing past the version is read), a device other than kDLCPU and
 *         kDLCUDAHost, a dtype code above 6, or all-zero strides over more than one element.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_tensor_init_dlpack(anira_tensor* tensor,
                                                           void* dl_managed_tensor_versioned,
                                                           anira_error* err) ANIRA_NOEXCEPT;

/**
 * @brief The first element of a host float tensor: handle.host.ptr plus byte_offset. NULL for
 * every domain but ANIRA_DOMAIN_HOST and ANIRA_DOMAIN_HOST_PINNED, for every dtype but
 * ANIRA_DTYPE_F32, for a planar tensor (ANIRA_TENSOR_PLANAR: anira_tensor_plane reads
 * it) and for a NULL base pointer: a stage that receives a device tensor or a uint8
 * tensor learns so from the NULL, not from a crash. It never converts and never looks at
 * strides: they are returned as declared, the caller reads by them (all-zero = packed
 * row-major).
 * @param tensor The tensor; NULL returns NULL.
 * @return The element pointer, or NULL.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API float* ANIRA_CALL anira_tensor_data_f32(const anira_tensor* tensor)
                                                  ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The typed read path of a host tensor: anira_tensor_data_f32 for any element type. NULL
 * unless dtype equals the tensor's own dtype (and is not 0); never converts.
 * @param tensor The tensor; NULL returns NULL.
 * @param dtype The element type the caller is about to read.
 * @return The element pointer, or NULL.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void* ANIRA_CALL anira_tensor_data(const anira_tensor* tensor,
                                             anira_dtype dtype) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The first element of one plane of a planar host tensor: handle.planes.ptrs[plane] plus
 * byte_offset. NULL for a tensor without ANIRA_TENSOR_PLANAR, for every domain but
 * ANIRA_DOMAIN_HOST and ANIRA_DOMAIN_HOST_PINNED, for a plane at or above
 * handle.planes.count, for a NULL pointer array or plane pointer, and unless dtype
 * equals the tensor's own dtype (and is not 0); never converts. It is how a callback
 * that is handed planar tensors reaches their memory. The caller reads by the strides
 * from axis 1 on (all-zero = packed).
 * @param tensor The tensor; NULL returns NULL.
 * @param plane The plane, below handle.planes.count.
 * @param dtype The element type the caller is about to read.
 * @return The element pointer, or NULL.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API void* ANIRA_CALL anira_tensor_plane(const anira_tensor* tensor,
                                              uint32_t plane,
                                              anira_dtype dtype) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The number of elements: the product of the first ndim extents, whatever the domain and
 * whether planar or not. 1 at rank 0 (the empty product), 0 when any extent is 0. 0 as
 * well for what is not a count: a record of dtype 0 (the all-zero record of a refused
 * factory, never a filled one), ndim above ANIRA_MAX_RANK, a negative extent, or a
 * product that does not fit size_t.
 * @param tensor The tensor; NULL returns 0.
 * @return The element count; a plain number on wasm32.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_tensor_num_elements(const anira_tensor* tensor)
                                                      ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief One extent of the shape. Runtime extents are counts, never ANIRA_DYNAMIC; the raw
 * int64_t shape stays readable through the record.
 * @param tensor The tensor; NULL returns 0.
 * @param axis The axis, below ndim.
 * @return shape[axis]; 0 for an axis at or above ndim or ANIRA_MAX_RANK, a negative extent, or
 *         one that does not fit size_t.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API size_t ANIRA_CALL anira_tensor_extent(const anira_tensor* tensor,
                                                uint32_t axis) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief sizeof of a record of this library, for an allocator that cannot see the header
 * (TypeScript, a binding): the Tier-1 records, whose size is the same on every target,
 * and the enumerated Tier-2 records, whose size is the target's and is the element
 * stride of the array enumerators.
 * @param id The record.
 * @return The size in bytes; 0 for an id this library does not know. In this pre-release
 *         ANIRA_STRUCT_STAGE_CTX is not registered yet and returns 0.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint32_t ANIRA_CALL anira_sizeof(anira_struct_id id) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Ends a token: closes the descriptor of an owning kind (ANIRA_SYNC_SYNC_FILE_FD,
 * ANIRA_SYNC_OPAQUE_FD_SEMAPHORE with an fd of 0 or more; on Windows an NT handle above
 * 0, closed with CloseHandle) and zeroes the record: kind becomes ANIRA_SYNC_NONE, under
 * which the payload is never read, so a second reset closes nothing. Test kind, never
 * the payload: a token moved out of by a later pre-release carries fd -1 instead. Every
 * other kind is a non-owning handle and is only zeroed. Closing is a system call, hence
 * never on the driver thread. On WebAssembly no sync kind owns an operating-system
 * object and nothing is closed.
 * @param token The token; NULL is a no-op.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 * @since ABI 0.2
 */
ANIRA_API void ANIRA_CALL anira_sync_token_reset(anira_sync_token* token) ANIRA_NOEXCEPT;

/**
 * @brief Duplicates a token so that it outlives its source: for the two owning fd kinds with an
 * fd of 0 or more a new descriptor of the same open file, close-on-exec (on Windows
 * DuplicateHandle for an NT handle above 0); the duplicate owns it and ends with
 * anira_sync_token_reset. A plain copy for every other kind. Duplicating is a system
 * call, hence never on the driver thread.
 * @param token The token to duplicate, typically a non-owning view read from a ticket.
 * @param out Receives the duplicate on success; whatever it held is overwritten, not reset.
 *        Must not alias token.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL or aliasing argument or a
 *         descriptor that is not open; ANIRA_ERROR_OUT_OF_MEMORY when the process is out of
 *         descriptors; ANIRA_ERROR_NOT_SUPPORTED for an owning kind with an fd of 0 or more on
 *         WebAssembly, where nothing can own one. *out is untouched on failure.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_sync_token_dup(const anira_sync_token* token,
                                                       anira_sync_token* out) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_TENSOR_H */
