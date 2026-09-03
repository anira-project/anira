# anira v3: Universal Tensor Architecture

2026-08-30. Reconciled with the anira and tanh-lib trees as of 29.08.2026.

Consolidated design, with real-time audio and real-time video as equally first-class targets. One passive tensor type carrying memory, ownership and readiness across CPU, GPU and NPU; one spec carrying model meaning; two scheduling contracts; a prepare-time planner that compiles declared capabilities into a fixed, validated pipeline; one stable, versioned C ABI. The hard real-time audio path (`process` / `push_data` / `pop_data`) is frozen by design: the `ANIRA_NONBLOCKING` entries are one-line forwarders to the v2 code paths and never wait; v2's in-callback wait lives in their `_wait` twins.

The C header set `include/anira/abi/*.h` (umbrella `anira.h`, C11) is the library's only binary contract, byte-identical on every platform -- native shared, native static (plugins, iOS) and WebAssembly; `anira.hpp` is a header-only C++20 wrapper over it with no binary promise, as is the TypeScript package. Every entry point is a directly exported `anira_*` function; the promise is numbered by `ANIRA_ABI_MAJOR`/`ANIRA_ABI_MINOR`, append-only inside a major, stable from v3.0.0. tanh-lib, concurrentqueue and nlohmann_json are private on every platform and never appear in a public header.

Layering rule: `MachineConfig` owns machine resources, `ModelConfig` owns model semantics, the `Contract` owns the scheduling regime of one handler, per-`submit` options own job particulars; the C ABI owns the binary contract, `anira.hpp` owns ergonomics. Semantics live in exactly one place; storage may live wherever engineering needs it. Section 6a states the ABI conventions, section 6b the boundary to tanh-lib and the browser. Section 11 is the order in which this is built: one last v2-line release (v2.3.0, no binary promise), then the v3 milestones. Version shorthand below: "v1" is the first v3 release, v3.0.0; "post-3.0" is a later minor inside ABI major 3.

---

## 1. Runtime Tensor (user <-> anira data unit)

The Tensor is a POD descriptor, trivially copyable through lock-free FIFOs. It carries only user-to-anira information; every anira-to-user signal lives on the Ticket. Graphics handles stay representable through the pipeline and are erased only at the backend adapter (enables accelerator switching under the same input and GPU pre/post stages).

Typing rule: every `anira_memory_handle` arm is typeless memory; the tensor descriptor (`dtype`, `shape`, `strides`, `byte_offset`) is the only type. Pixel formats never appear on a Tensor. Image-typed data (camera buffers, decoder output, rendered frames) enters as a `Frame` (section 1a) and exists as a Tensor only after a `FrameToTensor` stage.

The Tensor is a C struct. It is declared in `abi/tensor.h`, frozen at M2 with the other Tier-1 PODs, and `anira.hpp` adds names and factories to it, never members. Enum values come from `abi/enums.h`, where every value is pinned independently of any build option.

```c
/* abi/enums.h (excerpt) -- every value pinned; 12..63 reserved for later domains */
typedef enum anira_domain { ANIRA_DOMAIN_HOST = 0, ANIRA_DOMAIN_HOST_PINNED = 1, ANIRA_DOMAIN_CUDA = 2, ANIRA_DOMAIN_GL_BUFFER = 3,
    ANIRA_DOMAIN_VULKAN_BUFFER = 4, ANIRA_DOMAIN_OPAQUE_FD = 5, ANIRA_DOMAIN_METAL_BUFFER = 6, ANIRA_DOMAIN_WGPU_BUFFER = 7,
    ANIRA_DOMAIN_DMABUF = 8, ANIRA_DOMAIN_IOSURFACE = 9, ANIRA_DOMAIN_AHARDWAREBUFFER = 10, ANIRA_DOMAIN_D3D12 = 11,
    ANIRA_DOMAIN_FRAME = 12,   /* reserved: the Frame side of a FrameToTensor edge, section 1a; nothing consumes it at v3.0.0 */
    ANIRA_DOMAIN_FORCE32 = 0x7fffffff } anira_domain;
typedef enum anira_sync_kind { ANIRA_SYNC_NONE = 0, ANIRA_SYNC_QUEUE_ORDERED = 1, ANIRA_SYNC_CUDA_EVENT = 2, ANIRA_SYNC_VK_TIMELINE = 3,
    ANIRA_SYNC_GL_SYNC = 4, ANIRA_SYNC_SYNC_FILE_FD = 5, ANIRA_SYNC_OPAQUE_FD_SEMAPHORE = 6, ANIRA_SYNC_MTL_SHARED_EVENT = 7,
    ANIRA_SYNC_D3D12_FENCE = 8, ANIRA_SYNC_KIND_FORCE32 = 0x7fffffff } anira_sync_kind;
/* QUEUE_ORDERED: complete when the producing queue reaches it (a WGPUBuffer; GL after
   cudaGraphicsUnmapResources) -- a same-queue consumer waits for nothing, the host waits
   on the queue; never a fabricated fence. OPAQUE_FD_SEMAPHORE: a Vulkan semaphore exported
   OPAQUE_FD, the only fence CUDA imports. */
typedef enum anira_tensor_flags { ANIRA_TENSOR_READ_ONLY = 1, ANIRA_TENSOR_DISCARD_CONTENTS = 2, ANIRA_TENSOR_HOST_COHERENT = 4,
    ANIRA_TENSOR_FLAGS_FORCE32 = 0x7fffffff } anira_tensor_flags;
typedef enum anira_dtype_code { ANIRA_DTYPE_INT = 0, ANIRA_DTYPE_UINT = 1, ANIRA_DTYPE_FLOAT = 2, ANIRA_DTYPE_OPAQUE = 3,
    ANIRA_DTYPE_BFLOAT = 4, ANIRA_DTYPE_COMPLEX = 5, ANIRA_DTYPE_BOOL = 6, ANIRA_DTYPE_CODE_FORCE32 = 0x7fffffff } anira_dtype_code;   /* DLPack codes */
typedef uint32_t anira_dtype;                              /* code | bits << 8 | lanes << 16: the little-endian bytes of DLDataType */
#define ANIRA_MAKE_DTYPE(code, bits, lanes) ((uint32_t)(code) | ((uint32_t)(bits) << 8) | ((uint32_t)(lanes) << 16))
#define ANIRA_DTYPE_CODE(d)  ((d) & 0xffu)
#define ANIRA_DTYPE_BITS(d)  (((d) >> 8) & 0xffu)
#define ANIRA_DTYPE_LANES(d) ((d) >> 16)
#define ANIRA_DTYPE_F32 ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 32, 1)   /* also F16, BF16, I8, U8, I16, I32, I64, BOOL8 */
#define ANIRA_MAX_RANK 8

/* abi/tensor.h -- Tier-1 PODs, frozen at M2. ANIRA_API / ANIRA_CALL elided; NB = ANIRA_NONBLOCKING;
   [tag] = thread tag, [cs] = callback-safe (additive). ANIRA_PTR(T, name) = union { T* name; uint64_t name##_bits; } */
typedef union anira_memory_handle {                        /* 24 bytes; every arm typeless; vendor typedefs replaced by their wire width */
    struct { ANIRA_PTR(void, ptr); }                                          host;
    struct { ANIRA_PTR(void, ptr); int32_t device; }                          cuda;
    struct { uint32_t id; uint32_t target; }                                  gl;        /* GLuint + GLenum */
    struct { uint64_t buffer; uint64_t memory; uint64_t offset; }             vk;        /* VkBuffer, VkDeviceMemory (non-dispatchable, uint64); native, same-process */
    struct { int32_t fd; uint32_t reserved; uint64_t size; }                  opaque;    /* exported opaque fd; on Windows the NT handle (32-bit significant) */
    struct { ANIRA_PTR(void, buffer); }                                       mtl;       /* id<MTLBuffer> */
    struct { ANIRA_PTR(void, surface); uint64_t size; }                       iosurface; /* IOSurfaceRef, plane 0, BYTE-IMAGE-ENCODED packed floats
                                                                                            (pixel CVPixelBuffers are Frames) */
    struct { ANIRA_PTR(void, buffer); uint64_t offset; }                      wgpu;      /* WGPUBuffer; same process, the Machine's Dawn device */
    struct { int32_t fd; uint32_t reserved; uint64_t size; uint64_t offset; } dmabuf;    /* EXPORTED BUFFER MEMORY, typeless
                                                                                            (multi-plane image dma-bufs are Frames) */
    struct { ANIRA_PTR(void, buffer); }                                       ahb;       /* AHardwareBuffer*, buffer-typed (BLOB) */
    struct { ANIRA_PTR(void, resource); ANIRA_PTR(void, shared_handle); }     d3d12;
    uint64_t raw[3];                                                                     /* the handle as words: copy, compare, key a plan cache */
} anira_memory_handle;

typedef struct anira_sync_token {                          /* 24 bytes; kind NONE = data already visible */
    uint32_t kind;                                         /* anira_sync_kind */
    uint32_t flags;                                        /* no bit defined at v3.0.0; bits are appended by a minor */
    union {
        ANIRA_PTR(void, cuda_event);                                /* cudaEvent_t */
        struct { uint64_t semaphore; uint64_t value; } vk;          /* VkSemaphore (timeline) + value */
        ANIRA_PTR(void, gl_sync);                                   /* GLsync */
        int32_t fd;                                                 /* SYNC_FILE_FD (dmabuf / AHB world) and OPAQUE_FD_SEMAPHORE; both owned */
        struct { ANIRA_PTR(void, object); uint64_t value; } mtl;    /* id<MTLSharedEvent> + value */
        struct { ANIRA_PTR(void, object); uint64_t value; } d3d12;  /* ID3D12Fence* + value */
        uint64_t raw[2];
    } u;
} anira_sync_token;

typedef struct anira_tensor anira_tensor;
typedef void (ANIRA_CALL anira_tensor_release_proc)(anira_tensor*);
struct anira_tensor {                                      /* 216 bytes, align 8, identical on wasm32, LP64 and LLP64; trivially copyable */
    uint32_t            domain;                            /* anira_domain */
    anira_dtype         dtype;
    uint32_t            ndim;
    uint32_t            flags;                             /* anira_tensor_flags */
    int64_t             shape  [ANIRA_MAX_RANK];
    int64_t             strides[ANIRA_MAX_RANK];           /* in elements; all-zero = packed row-major */
    uint64_t            byte_offset;                       /* what the ViewChunker slices with */
    anira_memory_handle handle;
    ANIRA_PTR(void, manager_ctx);                          /* PRODUCER bookkeeping only (pool slot, refcounted view parent); never edge state */
    ANIRA_PTR(anira_tensor_release_proc, release);         /* unmap / unregister / recycle / free; NULL = borrowed (valid until the ticket completes) */
    anira_sync_token    acquire;                           /* input: data valid once this signals; bound output: buffer free to write once this signals */
};

/* factories: fill caller memory -- zero the struct, then fill it; all [thread-safe] [cs] NB except init_dlpack */
void anira_tensor_init_host       (anira_tensor*, void* data, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_pinned     (anira_tensor*, void* data, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_cuda       (anira_tensor*, void* ptr, int32_t device, void* cuda_event, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_gl_buffer  (anira_tensor*, uint32_t id, uint32_t target, void* gl_sync, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_vulkan     (anira_tensor*, uint64_t buffer, uint64_t memory, uint64_t offset, uint64_t timeline_semaphore, uint64_t value,
                                   anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_opaque_fd  (anira_tensor*, int32_t fd, uint64_t size, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_wgpu_buffer(anira_tensor*, void* wgpu_buffer, uint64_t offset, const anira_sync_token* fence, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
void anira_tensor_init_dmabuf     (anira_tensor*, int32_t fd, uint64_t size, uint64_t offset, int32_t sync_fd, anira_dtype, uint32_t ndim, const int64_t* shape) NB;
anira_status anira_tensor_init_dlpack(anira_tensor*, void* dl_managed_tensor_versioned);   /* [main-thread]; DLManagedTensorVersioned*; deleter wired into release/manager_ctx */
/* abi/draft/tensor_platform.h (anira_all.h only): anira_tensor_init_metal, _iosurface, _ahardwarebuffer, _d3d12 -- declared, outside the promise until measured */

/* accessors for JS and for stages: [thread-safe] [cs] NB */
float*   anira_tensor_data_f32(const anira_tensor*) NB;               /* HOST / HOST_PINNED and dtype F32 only, honours byte_offset; NULL for every other domain and every other dtype */
void*    anira_tensor_data(const anira_tensor*, anira_dtype) NB;      /* the typed read path: HOST / HOST_PINNED, NULL unless the argument equals the tensor's dtype; never converts */
size_t   anira_tensor_num_elements(const anira_tensor*) NB;           /* size_t: a number on wasm32 */
size_t   anira_tensor_extent(const anira_tensor*, uint32_t axis) NB;  /* runtime extents are counts, never ANIRA_DYNAMIC */
uint32_t anira_sizeof(anira_struct_id) NB;                            /* ANIRA_STRUCT_TENSOR = 1, _SYNC_TOKEN = 2, _MEMORY_HANDLE = 3 */
void         anira_sync_token_reset(anira_sync_token*);                              /* [thread-safe, !audio-thread]; closes an owned fd, kind = NONE */
anira_status anira_sync_token_dup(const anira_sync_token*, anira_sync_token* out);   /* [thread-safe, !audio-thread]; dup() for fd kinds, copy otherwise */
```

The `anira.hpp` view is the same struct with names and factories on it, not a class over it: a `Tensor*` is an `anira_tensor*`, and the wrapper asserts as much.

```cpp
namespace anira {
using Domain = anira_domain;  using SyncKind = anira_sync_kind;  using DType = anira_dtype;  using Engine = anira_engine;  using Provider = anira_provider;  using BackendId = anira_backend_id;   // Domain::Host, SyncKind::SyncFileFd as inline constants
struct SyncToken : anira_sync_token { void reset() noexcept; SyncToken dup() const; };
struct Tensor : anira_tensor {                                   // the C struct: 216 bytes, no members added, trivially copyable
    static Tensor from_host(void* data, DType, std::span<const int64_t> shape) noexcept;   // -> anira_tensor_init_host
    static Tensor from_pinned(...), from_cuda(void* ptr, int32_t device, void* event, ...), from_gl_buffer(uint32_t id, uint32_t target, void* glsync, ...),
                  from_vulkan(uint64_t buf, uint64_t mem, uint64_t off, uint64_t timeline, uint64_t value, ...), from_opaque_fd(int32_t fd, uint64_t size, ...),
                  from_wgpu_buffer(void* buf, uint64_t off, const SyncToken* fence, ...), from_dmabuf(int32_t fd, uint64_t size, uint64_t off, int32_t sync_fd, ...);
    static Tensor from_dlpack(void* dl_managed_tensor_versioned);   // throws Error, or Result<Tensor> under ANIRA_CXX_NO_EXCEPTIONS
    float* data_f32() const noexcept;  size_t num_elements() const noexcept;  size_t extent(uint32_t axis) const noexcept;
};                                                               // from_metal, from_iosurface, from_ahardwarebuffer, from_d3d12: anira_all.hpp
static_assert(sizeof(Tensor) == sizeof(anira_tensor) && std::is_trivially_copyable_v<Tensor>);
}
```

The prose below names domains and sync kinds by that spelling -- `Host`, `WgpuBuffer`, `SyncFileFd` -- and the C constants are the `ANIRA_DOMAIN_*` / `ANIRA_SYNC_*` values above.

Layout identity. `anira_tensor` is 216 bytes on every target, committed in `abi/layout-3.txt`. Every pointer sits in an `ANIRA_PTR` slot -- a union of the pointer with a `uint64_t`, 8 bytes on ILP32 and LP64 alike -- and every count is `uint32_t` or `uint64_t`, so `sizeof` and every `offsetof` are the same on wasm32, LP64 and LLP64; `anira_memory_handle` and `anira_sync_token` are 24 bytes by the same rule. The table of sizes and offsets is committed data, emitted by the layout executable and diffed natively and under node on every CI leg; the file may change only in a commit that changes `ANIRA_ABI_MAJOR`. That is the whole stability mechanism for the three structs: no `struct_size`, no version field, no chain -- a descriptor copied through a FIFO and read per block cannot afford a size switch, and a new shape is a new struct with new functions. Three consequences. A producer zeroes the struct before filling it (the factories do; the high half of an `ANIRA_PTR` is otherwise undefined on ILP32). JS reads a tensor through the generated `web/src/abi/layout.ts`, never through a hand-computed offset, and an allocator that cannot see the header asks `anira_sizeof(ANIRA_STRUCT_TENSOR)`. And `anira_dtype` is a packed `uint32_t` whose bytes are DLPack's `DLDataType`, so every signature that carries a type stays scalar (a 4-byte struct is passed by pointer on wasm32) and `anira_tensor_init_dlpack` copies the type with a `memcpy`. The pump's FIFO element is this struct; v2 queues `shared_ptr` payloads (`include/anira/scheduler/SessionElement.h:597-627`), and a C payload is what lets the native and the Wasm build share one queue element layout.

Which arms are enabled is decided by one rule, applied per platform -- never by a producer's preference. A domain is enabled when an engine reads it natively on hardware anira shares (`WgpuBuffer` with the WebGPU EP, `Cuda` with the CUDA EP, `D3D12` with DirectML, `AHardwareBuffer`/`GlBuffer` with LiteRT on Android), or when it is a producer API anira allocates for (`VulkanBuffer`, `GlBuffer`: section 6, `allocate_input`). Beside those sit the platform's *crossing currencies* -- not APIs that compute, but the shape memory travels in: `DmaBuf` and `OpaqueFd` on Linux, an NT handle on Windows (the same `opaque` arm), `AHardwareBuffer` on Android, `IOSurface` on Apple. Two of these are image-typed primitives -- `IOSurface` always carries a pixel format, `AHardwareBuffer` and `DmaBuf` can -- and they therefore appear in two roles, which the boundary rule keeps apart: as a *Tensor domain* when the format is the edge's own byte-image encoding of packed floats (section 7) and the payload is bytes in order, and as a *Frame container* when the format is real and must be interpreted. `Domain::IOSurface` is the first; a camera's NV12 `CVPixelBuffer` is the second. Boundary rule: a Tensor is bytes in order, a Frame is bytes with a layout -- a handle that would need a modifier, a pitch or a format on the Tensor is a Frame (section 1a).

What that yields, v1:

| platform | enabled domains | crossing currency | fences |
|---|---|---|---|
| Linux | `Host`, `HostPinned`, `Cuda`, `WgpuBuffer`, `VulkanBuffer`, `GlBuffer`, `DmaBuf`, `OpaqueFd` | dma-buf fd; opaque fd for CUDA | `SyncFileFd`, `OpaqueFdSemaphore`, `CudaEvent`, `QueueOrdered` |
| Windows | `Host`, `HostPinned`, `Cuda`, `WgpuBuffer`, `D3D12`, `VulkanBuffer`, `GlBuffer` | NT shared handle (`opaque`) | `D3D12Fence`, `CudaEvent`, `QueueOrdered` |
| macOS / iOS | `Host`, `WgpuBuffer`, `MetalBuffer`, `IOSurface` | `IOSurface` (`IOSurfaceRef`, public on both since iOS 11); `MTLBuffer` is reach-in on the shared `MTLDevice` | `MtlSharedEvent`, `QueueOrdered` |
| Android | `Host`, `WgpuBuffer`, `GlBuffer`, `VulkanBuffer`, `AHardwareBuffer`, `DmaBuf` | `AHardwareBuffer` (BLOB for tensors, image for Frames) | `SyncFileFd`, `GlSync`, `QueueOrdered` |

Two things hold everywhere. `Host` is the floor on every platform -- the v2 path, every CPU engine, no regression in coverage. And `WgpuBuffer` is the portable fast domain: Dawn has Vulkan, Metal and D3D12 backends, so `WgpuBuffer -> WebGPU EP` is `ZeroCopy` on all four platforms even where every other row differs.

Absences are decisions, not omissions. `GlBuffer` is enabled wherever its one reach-in row exists -- `cudaGraphicsGLRegisterBuffer` on Linux, Windows and Android -- and absent on Apple, where GL is deprecated, frozen below compute (4.1) and has no CUDA to register into: every remaining GL row there is `glGetBufferSubData`, which is `from_host` under another name, so the domain would earn nothing. A GL texture backed by an `IOSurface` is a Frame, not a `GlBuffer`. `Cuda` and `HostPinned` are absent on Apple for the same hard reason (no CUDA since 10.2, none on Apple silicon). **OpenCL is excluded outright, on every platform and not merely deferred**: it is first-party nowhere -- deprecated on Apple since 2018, absent from the Android NDK with no guarantee that a device ships an ICD, vendor-supplied elsewhere -- so it would add a `dlopen`'d dependency and a context to borrow without adding reach, since every accelerator that offers a CL path (LiteRT's GPU delegate on Android) reaches the same hardware through `AHardwareBuffer` or `GlBuffer`, which are first-party. An engine may use CL internally; anira neither takes a `cl_mem` nor hands one out. It is absent from `anira_domain`, `anira_sync_kind` and `anira_machine_config`, and is not on the additive list. Two Windows wrinkles worth stating before they surprise anyone: Dawn's default Windows backend is D3D12, so `D3D12` is the natural WebGPU pairing there while `VulkanBuffer -> WebGPU` would require Dawn forced onto its Vulkan backend -- `VulkanBuffer` is enabled on Windows for the CUDA row (opaque NT handle), not the WebGPU one; and `GlBuffer -> WebGPU` is `HostCopy` on Windows, since no Dawn D3D12 path imports GL memory.

Measurement status. Only the Linux column is measured -- Mesa Honeykrisp for the 50-cell matrix and an NVIDIA Turing box for the CUDA rows (section 7). The others follow from the rule and each needs its own measured cell before its rows ship -- the doc's standard is that no row exists without a measurement, and a per-platform matrix run is what turns these from declarations into rows. Two asymmetries to expect rather than assume. Dawn's `SharedBufferMemory` exists on D3D12, so Windows may get a true `ZeroCopy` buffer import into WebGPU where Linux needs the byte image (section 7). And Apple is expected to mirror Linux almost exactly: Dawn's Metal backend imports `SharedTextureMemoryIOSurface` -- textures, like the dma-buf path -- so `IOSurface -> WgpuBuffer` should be the same `DeviceCopy` byte-image relayout, with `MtlSharedEvent` in place of the sync file. The open question there is the *alias*, the Apple twin of the prototype's `VkImage` + `VkBuffer` pair: whether an `MTLBuffer` can be made to view the same `IOSurface` memory (`IOSurfaceGetBaseAddress` is page-aligned on UMA, which `newBufferWithBytesNoCopy:` wants) so the user's compute shader writes an ordinary buffer while Dawn imports a texture. If it can, `MetalBuffer` and `IOSurface` are two views of one allocation as on Linux; if not, `allocate_*` hands back a texture-writing path instead. To be measured, not assumed.

The measurement standard has a header-level consequence. The union arms for every platform are in the frozen `anira_memory_handle` from M2, their domain and sync values pinned in `abi/enums.h` from M1, because a Tier-1 layout cannot grow later; the *factories* that fill the unmeasured arms -- `anira_tensor_init_metal(buf, shared_event)`, `anira_tensor_init_iosurface(surface, size, shared_event)`, `anira_tensor_init_ahardwarebuffer(ahb, fence_fd)`, `anira_tensor_init_d3d12(resource, shared_handle, fence)` -- are declared in `abi/draft/tensor_platform.h`, included by `anira_all.h` and never by `anira.h`, and listed in `abi/symbols-draft.txt` rather than `abi/symbols-3.txt`. Promotion, when a platform column has its measured cells, moves a name between the two baselines and never renames it; the draft header exists so that an unmeasured surface is a declaration, not a promise, without a second naming scheme.

Sync tokens and WebGPU: no WebGPU-specific `anira_sync_kind`. Dawn exposes readiness as a `SharedFence`, which is a sync file on the Vulkan backend (`SyncFileFd`) -- or, where the consumer is CUDA, a `VkSemaphore` exported `OPAQUE_FD` (`OpaqueFdSemaphore`, the only fence CUDA imports; Dawn's Vulkan backend exports both kinds) -- a `D3D12Fence` on D3D12 and an `MtlSharedEvent` on Metal; the WebGPU adapter converts internally. Verified end to end in the prototype: one sync file crosses Vulkan (`vkGetSemaphoreFdKHR` out, `vkImportSemaphoreFdKHR` with `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` back in), Dawn (`ImportSharedFence` in, `EndAccess` fence out) and EGL (`EGL_SYNC_NATIVE_FENCE_ANDROID`) with no API learning that the others exist.

Sync token ownership, the two fd kinds (`SyncFileFd`, `OpaqueFdSemaphore`) only: the token owns its fd and closes it when reset or replaced; every hand-off is a transfer, and an importer that needs the fd past the call dups it (Dawn does). A producer must not close an fd it handed to a Tensor, and an adapter importing into an API that takes ownership -- Vulkan's temporary semaphore import -- clears the token instead of closing it. Unstated, this is a double close or a leak on every frame; the other `anira_sync_kind` arms are non-owning handles and need no such rule. The hand-off into anira is a move: `anira_handler_submit` and `anira_handler_bind_output` take `anira_tensor*`, non-const, and move every owning `acquire` token into the slot -- when the call returns, the caller's `acquire.kind` is `ANIRA_SYNC_NONE` and `u.fd` is `-1`, so a later `anira_sync_token_reset` on the caller's copy closes nothing. Owned fds inside a slot are closed by the inference thread when the slot is recycled, never by `anira_handler_ticket_release`, which therefore stays `ANIRA_NONBLOCKING`; `anira_handler_destroy` and `anira_handler_prepare` close the rest on the calling thread. Tokens read back through `anira_handler_ticket_input_released`/`ticket_output_ready` are non-owning views valid until `ticket_release`; a caller that needs one longer calls `anira_sync_token_dup`, and `anira_sync_token_reset` closes an owned fd -- both `[thread-safe, !audio-thread]`, because `dup` and `close` are syscalls. A factory that takes a token by pointer (`anira_tensor_init_wgpu_buffer`) copies it into `acquire`; that copy is the hand-off, and the caller does not reset its source.

Construction (edge factories, all producing this one struct, all filling caller memory -- zero, then fill): `anira_tensor_init_host`, `anira_tensor_init_pinned`, `anira_tensor_init_cuda(ptr, device, event)`, `anira_tensor_init_gl_buffer(id, target, glsync)`, `anira_tensor_init_vulkan(buf, mem, off, timeline, value)`, `anira_tensor_init_opaque_fd(fd, size)`, `anira_tensor_init_wgpu_buffer(buf, off, fence)`, `anira_tensor_init_dmabuf(fd, size, off, sync_fd)`, `anira_tensor_init_dlpack(dl_managed_tensor_versioned)`, plus the four draft platform factories above. All but one are `ANIRA_NONBLOCKING` field fills, legal from a render thread and from inside a stage callback. `anira_tensor_init_dlpack` is the exception, `[main-thread]` and fallible: it takes a `DLManagedTensorVersioned*` as `void*` so that `abi/tensor.h` never includes `dlpack.h`, maps device and dtype (the bytes are already `anira_dtype`), and wires the DLPack deleter into `release`/`manager_ctx` -- the 20-line bridge the DLPack-compatible codes and the inline shape/stride arrays exist for. The GPU factories take Vulkan handles and byte offsets at their wire width, `uint64_t`; they are the allowlisted exception to the no-64-bit-on-hot-paths rule of section 6a beside `anira_now_ns`, because nothing that produces a `VkBuffer` runs in JS.

Rules: one tensor per submit; multi-buffering is a producer-side pattern over tokens (rotate descriptors, reuse a slot when its `input_released` token -- `anira_handler_ticket_input_released` -- signals) -- and it stays on the producer's side of the edge: an engine whose graph is captured replays the buffers it was captured with (section 7, completion), so the tensors bound to the engine are fixed and the edge moves each rotating slot into them. `anira_tensor_init_gl_buffer` requires the GL thread policy from `anira_machine_config` (`anira_gl_desc.threads`, an `anira_gl_threads`). `anira_tensor_init_dmabuf` builds a Tensor from exported *buffer* memory (Vulkan `VK_KHR_external_memory_fd`, GBM linear bo, dma-heap) and is typed by the descriptor like every other arm. Images of any API (GL textures, `VkImage`, `WGPUTexture`, `MTLTexture`) are not Tensors: they are `Frame` containers and enter via a `FrameToTensor` stage, which is a zero-copy import plus one shader pass or an explicit device copy; the plan reports which.

`release` and `manager_ctx`, the thread rule. `release == NULL` means borrowed: the memory is the caller's and stays valid until the ticket is terminal (Async) or the call returns (Hard); nothing is called back. A non-NULL `release` is invoked **exactly once per submitted copy -- one submit, one release** -- on an inference thread when the job reaches a terminal state, inside `anira_handler_poll`/`anira_handler_ticket_wait` under Polled delivery, or on the calling thread of `anira_handler_destroy`/`anira_handler_prepare` for jobs still outstanding then; never on the driver thread, and never for a tensor anira did not receive through `submit` or `bind_output`. `manager_ctx` is what `release` reads and nothing else reads it. Tensors obtained from `anira_handler_allocate_input`/`anira_handler_allocate_output` are anira-owned pool tensors: anira never invokes their `release`, the producer submits the same descriptor repeatedly under the multi-buffering pattern above, and returns it with `anira_handler_free_tensor(h, t)` (`[main-thread & prepared]`); `prepare` and `destroy` free every pool tensor still out. A JS-produced tensor is always borrowed -- JS cannot supply a function pointer -- and the TS `Tensor` frees its heap block after a terminal `status()`.

Accessors, for JS and for stages, all `[callback-safe] ANIRA_NONBLOCKING`: `anira_tensor_data_f32(t)` returns the element pointer for `Host` and `HostPinned` when the tensor's `dtype` is `ANIRA_DTYPE_F32`, `byte_offset` applied, and NULL for every other domain and every other dtype -- a stage that receives a GPU tensor, or a `uint8` tensor, learns so from the NULL, not from a crash; `anira_tensor_data(t, dtype)` is the same read for any other element type and returns NULL unless the argument equals the tensor's own `dtype`, so the tensor path converts as little as the ring path does; `anira_tensor_num_elements(t)` and `anira_tensor_extent(t, axis)` are `size_t`, a plain number on wasm32, because runtime extents are counts and never `ANIRA_DYNAMIC` (the raw `int64_t shape[]` stays readable through `layout.ts` for a JS caller that wants it); `anira_sizeof(id)` for the three struct ids. These are the whole read surface: the worklet's `TensorView` is a `Float32Array` over `anira_tensor_data_f32` sized by `anira_tensor_num_elements`, and a native stage reads its `anira_stage_ctx.model_inputs` the same way (section 7).

Where edge state lives: not on the Tensor. An edge that caches an expensive import (a Dawn `SharedTextureMemory` over a dma-buf, a CUDA graphics registration, an NPU registration) keeps that cache **in the compiled plan**, keyed by the incoming memory handle, because the same plan sees a rotating set of descriptors under producer-side multi-buffering and must hit the cache for each slot rather than rebuild on every alternation. `manager_ctx` stays the producer's, and only the producer's. The prototype put edge state on the tensor (no plan exists there to hold it) and immediately hit the collision: a Vulkan tensor needs per-tensor producer state of its own, so the two uses fight over one pointer and are separable only while Vulkan tensors are never an edge *destination* -- which v3 does not guarantee.

Deliberately absent: stream/queue affinity (resolved at the adapter against the consumer's stream, DLPack-style), quantization and NCHW/NHWC semantics (spec), tensor names (spec, binding is positional), conversion methods (stages act, data flows), callbacks/mutexes/refcounts in the public struct, pixel formats (Frame), extensions (section 1b: config-time structs only -- a POD that travels through a FIFO owns no pointer), a `struct_size` or version field (layout identity above: the ABI major is the version).

---

## 1a. Frame (image data, user -> FrameToTensor stage)

A `Frame` is image data in a platform image-sharing container, described by a pixel format instead of a tensor descriptor: planes, per-plane pitch, chroma subsampling and colorimetry cannot be expressed as `dtype/shape/strides`, and a `VkBuffer`/SSBO cannot carry a format. It is a POD like `anira_tensor`, carries the same sync/ownership fields, and is accepted by exactly one kind of stage: `FrameToTensor`. A Frame is never chunked, windowed or bound as an output.

Deferred past v1, vocabulary declared here. `Frame`, its factories, `submit_frame` and `FrameToTensor` are named in this document so the vocabulary is complete, and their values are reserved from M1: the container and pixel-format enums and the Frame side of a `FrameToTensor` edge (`ANIRA_DOMAIN_FRAME`, one value of the `12..63` block) sit in `abi/enums.h` at v3.0.0 with nothing consuming them. The C struct itself is not in `anira.h` at v3.0.0. It arrives as `abi/draft/frame.h` -- `anira_frame`, `anira_frame_init_*`, `anira_handler_submit_frame` -- included by `anira_all.h` only, carried by the draft baselines (`abi/symbols-draft.txt`, `abi/abidiff-draft.supp`), and is promoted into `anira.h` by a later minor, additive under ABI major 3, once its platform column is measured; promotion moves names between baselines and never renames them. Freezing an unmeasured union in the 3.0.0 header is the one ABI mistake a minor cannot correct; reserving its values costs nothing. In v1 the user turns pixels into a float Tensor before `submit` -- their own shader writing an `allocate_input` buffer, or a host loop -- and the boundary a Tensor crosses is unchanged when the stage arrives.

```c
/* abi/enums.h (values reserved at v3.0.0) */
typedef enum anira_container { ANIRA_CONTAINER_DMABUF = 0, ANIRA_CONTAINER_AHARDWAREBUFFER = 1, ANIRA_CONTAINER_IOSURFACE = 2,
    ANIRA_CONTAINER_DXGI = 3, ANIRA_CONTAINER_VK_IMAGE = 4, ANIRA_CONTAINER_GL_TEXTURE = 5, ANIRA_CONTAINER_WGPU_TEXTURE = 6,
    ANIRA_CONTAINER_MTL_TEXTURE = 7, ANIRA_CONTAINER_HOST = 8, ANIRA_CONTAINER_FORCE32 = 0x7fffffff } anira_container;
#define ANIRA_FOURCC(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))
typedef enum anira_pixel_format { ANIRA_PIXEL_NV12 = ANIRA_FOURCC('N','V','1','2'), ANIRA_PIXEL_YUYV = ANIRA_FOURCC('Y','U','Y','V'),
    ANIRA_PIXEL_UYVY = ANIRA_FOURCC('U','Y','V','Y'), ANIRA_PIXEL_R8 = ANIRA_FOURCC('R','8',' ',' '),
    ANIRA_PIXEL_RGBA8 = ANIRA_FOURCC('R','A','2','4'), ANIRA_PIXEL_BGRA8 = ANIRA_FOURCC('B','A','2','4'), ANIRA_PIXEL_XRGB8 = ANIRA_FOURCC('X','R','2','4'),
    /* every value is the literal DRM fourcc constant of the format the name abbreviates */ ANIRA_PIXEL_FORMAT_FORCE32 = 0x7fffffff } anira_pixel_format;
typedef enum anira_color_matrix { ANIRA_COLOR_BT601 = 0, ANIRA_COLOR_BT709 = 1, ANIRA_COLOR_BT2020 = 2, ANIRA_COLOR_MATRIX_FORCE32 = 0x7fffffff } anira_color_matrix;
typedef enum anira_color_range  { ANIRA_RANGE_LIMITED = 0, ANIRA_RANGE_FULL = 1, ANIRA_COLOR_RANGE_FORCE32 = 0x7fffffff } anira_color_range;

/* abi/draft/frame.h -- the shape the draft ships; sizeof pinned in abi/layout-3.txt only on promotion */
typedef struct anira_color_info { uint32_t matrix; uint32_t range; uint32_t chroma_siting; uint32_t reserved; } anira_color_info;
typedef struct anira_frame anira_frame;
typedef void (ANIRA_CALL anira_frame_release_proc)(anira_frame*);
struct anira_frame {                         /* Tier 1 once promoted: ANIRA_PTR slots, wire widths, no struct_size */
    uint32_t container; uint32_t format; uint32_t planes; uint32_t width; uint32_t height; uint32_t reserved;
    anira_color_info color;
    union {
        struct { int32_t fd[4]; uint64_t off[4]; uint32_t pitch[4]; uint64_t modifier; }          dmabuf;
        struct { ANIRA_PTR(void, buffer); }                                                      ahb;        /* AHardwareBuffer* */
        struct { ANIRA_PTR(void, surface); }                                                     iosurface;  /* IOSurfaceRef / CVPixelBufferRef */
        struct { ANIRA_PTR(void, handle); ANIRA_PTR(void, resource); }                           dxgi;
        struct { uint64_t image; uint64_t memory; int32_t layout; uint32_t reserved; uint64_t modifier; } vk;   /* VkImage, VkDeviceMemory as uint64 */
        struct { uint32_t id; uint32_t target; }                                                 gl;
        struct { ANIRA_PTR(void, texture); }                                                     wgpu;       /* WGPUTexture */
        struct { ANIRA_PTR(void, texture); }                                                     mtl;        /* id<MTLTexture> */
        struct { uint64_t plane[4]; uint32_t pitch[4]; }                                         host;       /* plane pointers as bits */
        uint64_t raw[9];
    } handle;
    ANIRA_PTR(void, manager_ctx);
    ANIRA_PTR(anira_frame_release_proc, release);
    anira_sync_token acquire;
};
```

Factories, C functions filling caller memory like `anira_tensor_init_*`: `anira_frame_init_dmabuf(f, fds, offs, pitches, planes, modifier, fourcc, w, h, const anira_color_info*, sync_fd)`, `_init_ahardwarebuffer`, `_init_iosurface`, `_init_dxgi`, `_init_vk_image(f, image, memory, layout, modifier, ...)`, `_init_gl_texture(f, id, target, gl_sync, ...)`, `_init_wgpu_texture`, `_init_mtl_texture`, `_init_host_planes`; `anira.hpp` spells them `Frame::from_dmabuf(...)` and so on over the same struct. `anira_handler_submit_frame(h, anira_frame*, opts_or_null, deadline_ms, job_user_data, &ticket)` is the Frame twin of `anira_handler_submit`, with the same token move, the same per-job `void*` and the same `ANIRA_NONBLOCKING` class.

Containers vs. formats: the container is the sharing primitive of the platform (Linux dma-buf from V4L2 / VA-API / PipeWire; Android `AHardwareBuffer`; Apple `IOSurface`/`CVPixelBuffer`; Windows DXGI handles; same-process API images). anira carries every container and every format; whether a driver can *import* a given (container, format, modifier) is a probed edge (section 7). Same-process API images are importable only by their own API unless the user exports them (`VK_EXT_external_memory_dma_buf`, `eglExportDMABUFImageMESA`); the planner does not export on the user's behalf. Vulkan image layout and Dawn `BeginAccess/EndAccess` state are edge-internal.

Interpretation is a stage, not a property: `stage::FrameToTensor` (section 7) ships kernels for `{NV12, YUYV, UYVY, RGBA8, BGRA8}`; any other format is a user stage with the same edge, an `anira_stage_desc` whose `domain_in` is `ANIRA_DOMAIN_FRAME` and whose `domain_out` is the domain of its choice.

Implicit sync is part of the container, not of the token. A dma-buf from V4L2, VA-API or PipeWire carries fences on its reservation object that appear in no `anira_sync_token`: `DMA_BUF_IOCTL_SYNC` (CPU access) and every driver-side import wait on all of them, so a Frame whose `acquire` is `ANIRA_SYNC_NONE` may still block on entry. The planner therefore treats a dma-buf container as implicitly synchronised -- correct without an explicit token, and never a licence to skip one when the producer has it -- and the plan report states the cost, because it is measurable: the prototype's `INFER_DMABUF_NOSYNC=1` exists to measure exactly this bracket. Explicit tokens remain preferred: they say *what* is waited for, where implicit sync waits for everything the buffer ever saw.

---

## 1b. Extensions (capability slots on config handles)

Some of what a config carries is read by the core -- the Machine, the planner, the chunkers, the pump -- and some is read by exactly one stage or one backend adapter. The first kind are fields, which on a C handle means scalar setters. The second kind are *extensions*: they arrive with their consumer, leave with it, and a build that lacks the consumer must refuse them rather than ignore them. Every config handle a stage or adapter reads accepts them through one `set_ext`/`set_ext_json` pair -- `anira_tensor_spec_set_ext`, `anira_model_config_set_model_ext(cfg, model_index, ...)` for a `ModelData` entry, `anira_model_config_set_ext` for the whole config, `anira_machine_config_set_ext`, `anira_contract_set_ext` (Hard and Async alike), `anira_job_options_set_ext` -- one entry per kind, a second set of the same kind replaces the first. Never `anira_tensor` or `anira_frame`: those are PODs copied through lock-free FIFOs, and an owning pointer on them is a lifetime and an allocation. The layout rule has three tiers: Tier-1 RT PODs are fixed for the life of the ABI major, a new shape being a new struct with new functions; Tier-2 descriptors grow only at the tail, after the last v3.0.0 slot, behind their leading `struct_size`; an extension's layout is fixed per revision -- fields may be appended within a revision (`struct_size` says how much the caller filled), a changed field is a new revision. A new capability is therefore a new registry row, never a new field on a config, and the config handles have no public layout at all.

```c
/* abi/config.h */
typedef struct anira_ext_header { uint32_t struct_size; uint32_t version; const char* kind; } anira_ext_header;   /* first member of every payload; kind doubles as the JSON key */
typedef struct anira_ext_entry  { anira_ext_header header; const char* name; } anira_ext_entry;                  /* kind "entry", version 1 */
#define ANIRA_EXT_ENTRY_INIT ANIRA_INIT(anira_ext_entry, { sizeof(anira_ext_entry), 1, "entry" }, NULL)

anira_status anira_model_config_set_model_ext     (anira_model_config*, uint32_t model_index, const anira_ext_header*, anira_error*);   /* [main-thread]; deep copy through the registry row */
anira_status anira_model_config_set_model_ext_json(anira_model_config*, uint32_t model_index, const char* kind, const char* utf8, size_t len, anira_error*);
/* the same pair on anira_tensor_spec, anira_model_config, anira_machine_config, anira_contract, anira_job_options */
anira_status anira_registered_ext_kinds(uint32_t* count, const char** out);   /* [thread-safe]; scalar enumeration: NULL out = count, ANIRA_INCOMPLETE when short */
```

```cpp
// anira.hpp -- the C++ view: aggregates that mint the C header for the duration of the set call
namespace anira::ext { struct Entry { std::string name; }; }     // -> anira_ext_entry{ {sizeof(anira_ext_entry), 1, "entry"}, name.c_str() }
cfg.model_ext(1, anira::ext::Entry{ "forward_stream" });          // template <class Ext> ModelConfig& model_ext(uint32_t, const Ext&); likewise
spec.ext(anira::ext::Quant{ ... });                                 // TensorSpec::ext, ModelConfig::ext, MachineConfig::ext, JobOptions::ext;
                                                                    // Hard/Async, being aggregates, carry theirs in a vector applied at prepare
```

The registry is internal in v3.0.0: one row per `(kind, version)` -- `{kind, version, struct_size, clone, destroy, from_json, to_json}` -- registered by the extension's own translation unit; `from_json == NULL` marks a code-only kind. `kind` is the stable string id and the JSON key; `version` is the revision of that kind's layout, in JSON the optional `"version"` member of the extension object (default 1). CLAP names its extensions `clap.$NAME/$REV` with `$REVERSE_URI.$NAME/$REV` for third parties, and anira keeps the string id and the integer revision, with the revision as a header field rather than a suffix because the kind doubles as the JSON key. A known kind at a registered version is deep-copied by its `clone` during the set call, so the caller's struct and every string it points to may die when the call returns; an extension is self-contained by rule, owning everything it references and pointing into neither its host nor another extension. A known kind at an unregistered version is rejected at set time with `ANIRA_ERROR_EXTENSION_VERSION`, from code and from JSON alike, because both can be checked there. An unknown kind is *not* rejected at set: from code the header is stored, from JSON the header and the raw text (the internal `ext::Unknown` carrier), and `anira_handler_prepare` fails by name.

Consumed or fail. Every stage and backend adapter declares what it reads: `anira_stage_desc` and `anira_backend_desc` carry `const char* const* consumed_kinds; uint32_t num_consumed_kinds;` as v3.0.0 fields, and anira's own stages and adapters declare theirs the same way, so the walk is over declared data. Each entry of `consumed_kinds` names the host it is read from as `"<host>:<kind>"` -- `"tensor_spec:quant"`, `"model:entry"`, `"job:crop_affine"`, `"machine:artifact_cache"` -- so the walk matches host and kind together and an entry set on the wrong handle fails with `ANIRA_ERROR_EXTENSION_UNCONSUMED` instead of being absorbed by a stage that consumes that kind elsewhere. After the plan is compiled, `anira_handler_prepare` walks every extension on every handle it touched and checks each entry against the union of what the stages and adapters actually in the plan consume. An entry no stage consumes fails prepare with `ANIRA_ERROR_EXTENSION_UNCONSUMED` and the name in `anira_error.message` -- `extension 'quant' on tensor 'audio_in' is not consumed by any stage in this build` -- and an unknown kind fails the same way with `ANIRA_ERROR_EXTENSION_UNKNOWN`, which is why the JSON loader does not drop it but carries it, precisely so that a typo or a missing backend fails here with the name in the message, from code and from JSON at one failure point. This is the one place the design inverts Vulkan's `pNext`, whose contract is to skip what it does not recognise: right for a driver ABI, wrong for a config, where a skipped quantization block means an int8 model runs and produces garbage; it is also why there is no `pNext` chain anywhere in the header. Two stages may consume the same entry. After `prepare` the bags are frozen and read-only; a stage that needs an extension read it at prepare and cached what it needs, and the pump never sees the carrier -- `anira_stage_ctx` is a 64-byte POD with no room for one. The plan report lists, per handle, each extension present and the stage that consumed it as `anira_plan_ext {struct_size, index, host, kind, consumer}` rows through `anira_plan_report_exts`, so a sweep log says `quant -> QuantStage(int8)` rather than leaving the reader to assume. Host-side consumption is post-3.0: the `consumed_kinds` fields exist from v3.0.0 so the freeze needs no later slot, but `anira_register_ext_kind` and the prepare-time payload accessor a custom stage would read through arrive, additively, with the first external consumer.

Per-job extensions are borrowed, not owned: the payload behind `anira_job_options_set_ext` must stay valid until every `anira_handler_submit` that reads the options has returned, the stage copies what it needs into its fixed-size job record at submit (`submit` is `ANIRA_NONBLOCKING` and allocates nothing), and the consumed-or-fail check runs at submit and fails the ticket -- `ANIRA_TICKET_FAILED`, the kind name readable through `anira_handler_ticket_error` -- not the handler.

`anira_capabilities_ext_kinds` on `anira_machine_capabilities(m)` reports the registered kinds beside the probed edges and the enabled backends, and `anira_registered_ext_kinds` reports them without a machine, so a deployment can ask whether this anira understands `npu` before loading a model that needs it.

Reserved now, used later. Third-party kinds are post-3.0; the reverse-URI prefix (`"de.tu-berlin.ak.mel"`) is reserved at v3.0.0 so anira's bare kinds (`"entry"`, `"quant"`, `"ort_session"`) never collide. Written into `abi/enums.h` at M1 with the same intent: `anira_engine` values `6..0x0fff` for later anira engines, a registered custom engine assigned a value from `0x1000` up at prepare under a reverse-URI name of its own; domain values `12..63` for later domains, `ANIRA_DOMAIN_FRAME` among them; `anira_struct_id` `0x0001xxxx` for extension payloads (so `anira_sizeof` can size one for an allocator that cannot see the header), `0x0004xxxx` for Emscripten-only structs. Every later kind is a new row, a new payload struct with a new struct id, and one `ANIRA_ABI_MINOR` bump.

v1 ships the carrier, the registry, the walk, `ext::Unknown` -- and one extension: `anira_ext_entry` (`ext::Entry { std::string name; }` in `anira.hpp`) on a `ModelData` entry, the entry point a program is run through (v2's `model_function`, `include/anira/InferenceConfig.h:154`; absent means `forward`), consumed by the two adapters that have one -- LibTorch (`get_method`, `src/backends/LibTorchProcessor.cpp:153,188`) and ExecuTorch (`load_method`, `src/backends/ExecuTorchProcessor.cpp:156`, which defaults to `"forward"` itself) -- so the rule that two consumers may read one entry is exercised from day one. It is the proof of the mechanism against backends that already have users. A v2 model file carrying `model_function` inside `model_data[]` loads into the extension through the auto-upgrade path of `anira_model_config_from_json` (section 8) with its `ANIRA_SUCCESS_UPGRADED` status and one-time warning; the same file on a build with neither backend fails prepare by name instead of carrying a field that means nothing there -- today `ModelData` logs an error at construction when a function name meets another backend (`InferenceConfig.h:53-64`) and then keeps the field; a v2 file that never set one migrates with an empty bag, which proves the absent path; and the plan report shows the row `ModelData[1].ext: entry -> LibTorchAdapter`.

Deferred, each arriving with its consumer as a new `(kind, version)` row, all additive inside ABI major 3: `ext::Quant` on `TensorSpec` (scales, zero points, channel axis; consumed by a (de)quant stage), `ext::Artifacts` on `ModelData` and `ext::ArtifactCache` on `MachineConfig` (precompiled EP/NPU binaries keyed by device; consumed by adapters that compile -- TensorRT, QNN, CoreML), `ext::Npu` on `MachineConfig` and `ext::NpuHard` on `Hard` (plugin directories, performance hint, full-offload-or-reject and performance pinning; consumed by an NPU adapter), `ext::OrtSession` on `ModelData` (graph capture, validation mode, layout preference, intra-op spinning; consumed by the ORT adapter -- the prototype's `INFER_ORT_OPTS` given a home; v2 hard-codes these in `src/backends/OnnxRuntimeProcessor.cpp`), `ext::CropAffine` on `JobOptions` (the per-job 2x3 affine of `FrameToTensor`, section 7), and `ext::JobBackend` / `ext::JobModel` on `JobOptions` (per-job backend and variant selection, section 7). Whether something belongs here rather than in a setter is decided by the question above and nothing else: who reads it. the block range, `rate` and `anchor` are read by the core and are setters; the device blocks of `MachineConfig` are read by the Machine's probe and are descriptors (`anira_cuda_desc` and its siblings); the entry point of a LibTorch or ExecuTorch program is read by that adapter and is an extension.

---

## 2. TensorSpec (model truth, per I/O slot)

A spec is a configuration object, so it is an opaque handle mutated by scalar setters, never a struct the caller lays out: its layout never enters the ABI, and JS builds one with one call per field instead of marshalling a struct. `anira_tensor_spec_create` fixes the three things every spec has -- name, dtype, role -- and every other field has a setter whose default is the value a field would have been initialised to. The name is copied. The spec is a value: `anira_model_config_add_input`/`add_output` (section 5) copy it, the handle may be destroyed right after, and one spec may be copied into two configs. There is no `anira_tensor_spec_from_json`: a spec exists only inside a model config, and `anira_model_config_from_json` (section 8) builds it there.

```c
/* abi/enums.h */
#define ANIRA_MAX_RANK 8
#define ANIRA_DYNAMIC   ((int64_t)-1)                    /* legal on the Time axis of a Streamed spec (the extent is the pinned window) and of a Buffer spec */
#define ANIRA_UNBOUNDED ((int64_t)-1)                    /* for window_max */

typedef enum anira_axis_tag { ANIRA_AXIS_BATCH = 0, ANIRA_AXIS_CHANNEL = 1, ANIRA_AXIS_TIME = 2, ANIRA_AXIS_HEIGHT = 3,
    ANIRA_AXIS_WIDTH = 4, ANIRA_AXIS_FEATURE = 5, ANIRA_AXIS_ANY = 6, ANIRA_AXIS_TAG_FORCE32 = 0x7fffffff } anira_axis_tag;
/* axis index order = model memory order; NCHW vs NHWC is just axis order; layout conversion = tag-sequence
   matching. Chunkers find the Time axis by tag. */

typedef enum anira_role {
    ANIRA_ROLE_STREAMED = 0,   /* has a Time axis consumed window-wise           (in + out) */
    ANIRA_ROLE_BUFFER   = 1,   /* whole submitted buffer = one model tensor,
                                  no Time axis (frames, images)                 (in + out) */
    ANIRA_ROLE_STATIC   = 2,   /* no time semantics: conditioning in,
                                  scalar/embedding out; one value per job */
    ANIRA_ROLE_FORCE32  = 0x7fffffff
} anira_role;

/* abi/config.h -- opaque handle + scalar setters; every entry [main-thread], may allocate; ANIRA_API/ANIRA_CALL elided */
typedef struct anira_tensor_spec anira_tensor_spec;
anira_status anira_tensor_spec_create(const char* name, anira_dtype dtype, anira_role role,
                                      anira_tensor_spec** out, anira_error* err);
        /* name:  canonical, UTF-8, copied; the model config maps it to engine names (section 5)
           dtype: the model's true dtype (section 1). A quantized model takes float I/O only
                  through the `quant` extension and its (de)quant stage (section 1b, deferred) */
anira_status anira_tensor_spec_set_axis      (anira_tensor_spec*, uint32_t i, anira_axis_tag tag, int64_t extent);
        /* i < ANIRA_MAX_RANK; ndim = max(i + 1) */
/* ANIRA_ROLE_STREAMED only, in elements along the Time axis: */
anira_status anira_tensor_spec_set_window    (anira_tensor_spec*, int64_t window_min, int64_t window_max, int64_t context);
        /* window_min: model's smallest legal Time extent; window_max: largest, ANIRA_UNBOUNDED = no upper limit;
           context: left-context retained across inferences; consumed per inference = window_used - context;
           fixed case: window_min == window_max; default 0, 0, 0 */
anira_status anira_tensor_spec_set_time_ratio(anira_tensor_spec*, int64_t num, int64_t den);
        /* vs. anchor tensor; (0, 0) = derive (default); this tensor advances num elements per den anchor elements */
/* Outputs only: */
anira_status anira_tensor_spec_set_latency   (anira_tensor_spec*, int64_t latency);
        /* model-internal delay along Time (per tensor); default 0 */
anira_status anira_tensor_spec_set_ext       (anira_tensor_spec*, const anira_ext_header*, anira_error* err);   /* section 1b */
anira_status anira_tensor_spec_set_ext_json  (anira_tensor_spec*, const char* kind, const char* utf8, size_t len, anira_error* err);
void         anira_tensor_spec_destroy       (anira_tensor_spec*);
```

A setter judges one argument (`i < ANIRA_MAX_RANK`, `den != 0` unless both are zero) and returns `ANIRA_ERROR_INVALID_ARGUMENT`; every rule that reads two fields, two tensors or the plan is prepare-time, below. `set_ext` deep-copies a known `(kind, version)` through its registry row, stores an unknown kind as header plus raw JSON so that `anira_handler_prepare` can fail by name, and rejects a known kind with an unregistered version at once with `ANIRA_ERROR_EXTENSION_VERSION`; `set_ext_json` is the same call for a JSON payload, which is how the TS builder registers one. v1 registers no kind for the spec; `quant` arrives with its consumer. The `int64_t` extents and windows are the one place a JS caller passes `BigInt` -- configuration time only; no `ANIRA_NONBLOCKING` entry carries a 64-bit integer except the allowlisted GPU factories of section 1 and `anira_now_ns`, and runtime extents come back as `size_t` counts through `anira_tensor_extent` (section 1).

The `anira.hpp` spelling is a builder with one method per C setter, so C++ and TS are the same call sequence and a field nobody writes carries exactly the default its setter documents:

```cpp
// anira.hpp -- not ABI-stable; one method per C setter, chained
class TensorSpec {
public:
    TensorSpec(std::string name, DType dtype, anira_role role);       // = anira_tensor_spec_create; DType = anira_dtype, section 1
    TensorSpec& axis(uint32_t i, anira_axis_tag tag, int64_t extent); // model memory order; ndim = max(i + 1) <= ANIRA_MAX_RANK

    // ANIRA_ROLE_STREAMED only, in elements along the Time axis:
    TensorSpec& window(int64_t window_min, int64_t window_max, int64_t context);   // unwritten: 0, 0, 0
    TensorSpec& time_ratio(int64_t num, int64_t den);                              // unwritten: (0, 0) = derive

    // Outputs only:
    TensorSpec& latency(int64_t latency);                                          // unwritten: 0

    template <class Ext> TensorSpec& ext(const Ext&);                              // = set_ext, which copies; section 1b
};
```

The object owns the handle from its constructor, every method is its setter and throws `anira::Error` on failure, and the destructor is `anira_tensor_spec_destroy`; `ModelConfig::input`/`output` (section 5) is `add_input`/`add_output`, which copies, so the builder may die on the next line. No C++ type crosses the ABI -- the handle does, and its layout is promised to nobody. The TS `TensorSpec` builder in `config/` is the same calls in the same order, with `destroy()` after the model config has copied it.

Window semantics: incremental arrival accumulates until `window_min`, then runs greedily clamped to `window_max`, retaining `context`. Complete buffers within range run in one shot; above `window_max` the ViewChunker slices, rebalancing the last two chunks into range; below `window_min` a JobOptions policy decides pad vs reject. Hard pins one effective window at prepare (host cadence clamped into range) and measures the budget at exactly that window. One ring element is one Time-axis element of its slot: `window_min`, `window_max`, `context` and `latency` count the elements that slot's own ring counts, `block_min`/`block_max` and `rate` the anchor's, `time_ratio` relating the two, whatever the element type. Typing changes the width of an element, never the count -- a transform that changes the count is not a dtype conversion and does not belong on the ring path.

Prepare-time legality, every violation reported by `anira_handler_prepare` as `ANIRA_ERROR_CONFIG` with the tensor and the rule named in `anira_error.message` (nothing throws across the ABI; `anira.hpp` rethrows it as `anira::Error`): exactly one Time axis for Streamed; window fields iff Streamed; `ANIRA_DYNAMIC` only on Buffer Time extent; every axis slot below `ndim` written; ratios and window ranges jointly satisfiable across streamed tensors; every extension consumed by a stage in the plan (section 1b) -- the one rule with its own status, `ANIRA_ERROR_EXTENSION_UNCONSUMED`, so a host can tell a missing consumer from a malformed spec; and, for a Streamed tensor whose chunker is a RingChunker -- incremental arrival, which is the Hard path (section 3) -- a `dtype` that differs from the element type of its ring requires a stage in the plan that consumes the difference, in `pre_process` on the way in and in `post_process` on the way out, reported with the tensor and both dtypes named. A ring is typed by the host stream on that side of the handler -- what the host pushes into an input ring, what the host pops from an output ring -- and the host declares that element type per streamed slot on the Hard contract (`anira_contract_hard_set_ring_dtype`, section 3; `ANIRA_DTYPE_F32` when unset). **Hard is not float-only**: a Streamed tensor of any `dtype` whose stream is declared in the same type needs no stage, an `int16` stream into an `int16` model as much as the float default; only a stream and a model that differ -- a float host feeding an `int8` model -- need the converting stage, which is what `ext::Quant` and its (de)quant stage bring (section 1b, deferred). The rule is about the consumer, not about the dtype, and it governs the ring path only: on the `submit` path there is no ring, a Streamed tensor submitted complete is sliced by the ViewChunker into views carrying the tensor's own dtype, and a dtype that disagrees with its spec is refused at `submit` (section 6a), which is where that check already lives. Streamed tensors may sit on one side only: a generator (Static or Buffer inputs, Streamed outputs) and an analyser (Streamed inputs, Static or Buffer outputs) are first-class, not edge cases -- the tree already has the v2 half: the reference stream is an input or an output (`ReferenceStream{m_is_input, m_index}`, `include/anira/utils/HostConfig.h:22-24`; `k_first_streamable` at :59 resolves to the first streamable input, else output, `resolve_reference()` at :153-190), the redo of the reverted #101 (#98, #99, #110) that stopped `SessionElement::prepare()` hanging for generators and crashing for analysers; v3 carries that resolution as the anchor rule of section 5. Under Hard at least one Streamed tensor must exist on either side, because the anchor is the clock; Async requires none, having no anchor, but admits them: a Streamed tensor submitted complete is sliced by the ViewChunker, which is what `anira_job_options_set_head_trim` and `set_tail_flush` are for and what offline file rendering is. A frame rate in a Hard contract therefore says nothing about images: it is legal only where the anchor is a Streamed tensor whose Time axis counts frames -- a per-frame streaming model, the video analogue of an audio stream -- and a Buffer-role image, having no Time axis, can never be an anchor and belongs to Async.

---

## 3. Contract (scheduling regime of one handler)

A contract is one opaque handle, `anira_contract`, tagged Hard or Async at creation and mutated by scalar setters; `anira_handler_prepare` takes it by pointer and copies it, so the handle may be destroyed when the call returns. The kind chosen at `prepare` is the one decision no later call revisits -- the analogue of CLAP's render mode, `set(REALTIME | OFFLINE)` on the main thread before processing. Every duration crossing the ABI is a `double` in milliseconds (v2's `max_inference_time` unit; no 64-bit integer at the boundary), an absent deadline is a negative value rather than a flag, and every default equals the initialiser of the `anira.hpp` aggregate below.

```c
/* abi/enums.h -- values pinned, independent of build options */
typedef enum anira_contract_kind { ANIRA_CONTRACT_HARD = 1, ANIRA_CONTRACT_ASYNC = 2, ANIRA_CONTRACT_KIND_FORCE32 = 0x7fffffff } anira_contract_kind;
typedef enum anira_budget_kind   { ANIRA_BUDGET_MEASURED = 0, ANIRA_BUDGET_EXPLICIT = 1, ANIRA_BUDGET_KIND_FORCE32 = 0x7fffffff } anira_budget_kind;
typedef enum anira_warmup_mode   { ANIRA_WARMUP_NONE = 0, ANIRA_WARMUP_FIXED = 1, ANIRA_WARMUP_UNTIL_STABLE = 2, ANIRA_WARMUP_MODE_FORCE32 = 0x7fffffff } anira_warmup_mode;
typedef enum anira_miss_policy   { ANIRA_MISS_BYPASS = 0, ANIRA_MISS_HOLD_LAST = 1, ANIRA_MISS_ZEROS = 2, ANIRA_MISS_POLICY_FORCE32 = 0x7fffffff } anira_miss_policy;
typedef enum anira_late_policy   { ANIRA_LATE_FINISH = 0, ANIRA_LATE_DROP = 1, ANIRA_LATE_POLICY_FORCE32 = 0x7fffffff } anira_late_policy;
typedef enum anira_priority      { ANIRA_PRIORITY_AUTO = 0, ANIRA_PRIORITY_INTERACTIVE = 1, ANIRA_PRIORITY_BATCH = 2, ANIRA_PRIORITY_FORCE32 = 0x7fffffff } anira_priority;
typedef enum anira_delivery      { ANIRA_DELIVERY_POLLED = 0, ANIRA_DELIVERY_IMMEDIATE = 1, ANIRA_DELIVERY_FORCE32 = 0x7fffffff } anira_delivery;
typedef enum anira_edge_cost     { ANIRA_EDGE_COST_PERMISSIVE = 0, ANIRA_EDGE_COST_STRICT = 1, ANIRA_EDGE_COST_FORCE32 = 0x7fffffff } anira_edge_cost;
#define ANIRA_WAIT_FOREVER  (-1.0)      /* timeout_ms sentinel: v2 set_non_realtime behaviour */
#define ANIRA_WAIT_CONTRACT (-2.0)      /* timeout_ms sentinel: wait_ratio x block duration, v2 blocking_ratio */

/* abi/config.h -- every entry [main-thread], may allocate; a rejected value is ANIRA_FAILED(status) + anira_error.message */
typedef struct anira_contract anira_contract;
anira_status anira_contract_create_hard (uint32_t block_min, uint32_t block_max, double rate, anira_contract** out, anira_error* err);
        /* stream geometry, what the host callback delivers: the range of n passed to anira_handler_process, in Time-axis elements
           of the anchor tensor; block_min == block_max is the fixed-block host and earns the tight latency, because the reported
           figure carries a buffer-adaptation term that depends on the block size and vanishes when the block divides the hop;
           block_min < block_max reports the worst case across the range; rate = anchor elements per second (48000 for audio);
           the anchor is the ModelConfig's anchor */
anira_status anira_contract_create_async(anira_contract** out, anira_error* err);
anira_status anira_contract_hard_set_geometry  (anira_contract*, uint32_t block_min, uint32_t block_max, double rate);   /* the host patches a file's geometry (section 8) */
anira_status anira_contract_hard_set_budget    (anira_contract*, anira_budget_kind, double explicit_ms);  /* MEASURED (default): derived during warmup; explicit_ms read for EXPLICIT only */
anira_status anira_contract_hard_set_warmup    (anira_contract*, anira_warmup_mode, uint32_t iterations); /* UNTIL_STABLE (default); iterations for FIXED only; NONE legal only with EXPLICIT */
anira_status anira_contract_hard_set_on_miss   (anira_contract*, anira_miss_policy);                      /* BYPASS (default) requires shape-compatible I/O along the anchored Time axis */
anira_status anira_contract_hard_set_wait_ratio(anira_contract*, double ratio);                           /* 0 (default); v2 blocking_ratio one-to-one; consumed by the _wait twins only */
anira_status anira_contract_hard_set_ring_dtype(anira_contract*, const char* canonical, anira_dtype dtype); /* the host stream's element type on that streamed slot (the tensor's canonical name, resolved at prepare); ANIRA_DTYPE_F32 when unset; the ring beneath the slot takes it (section 7) and the typed Hard entries check against it (section 6); a model dtype that differs needs a converting stage in the plan (section 2). Hard is not float-only. */
anira_status anira_contract_async_set_deadline (anira_contract*, double deadline_ms);                     /* < 0 (default) = none, the offline posture; the clock starts at submit; an absolute
                                                                                                             per-job override is the deadline_ms argument of anira_handler_submit (section 6) */
anira_status anira_contract_async_set_policy   (anira_contract*, anira_late_policy on_late, anira_priority, uint32_t lanes, uint32_t max_in_flight, anira_delivery);
        /* on_late FINISH (default); DROP cancels at chunk boundaries and enables admission control at dispatch
           priority AUTO (default): INTERACTIVE iff deadline, else BATCH
           lanes: parallel plan instances; 0 = auto: 1 if ANIRA_MODEL_STATEFUL, else min(max_instances, pool-derived)
           max_in_flight: per-lane pipelining; 0 = auto: shallow iff deadline, else deep
           delivery POLLED (default): on_complete runs in the caller of anira_handler_poll/ticket_wait; IMMEDIATE: on the inference thread */
anira_status anira_contract_set_edge_cost(anira_contract*, anira_edge_cost);                              /* PERMISSIVE (default); plan validation, not scheduling; prepare takes one object (section 6) */
anira_status anira_contract_set_ext      (anira_contract*, const anira_ext_header*, anira_error* err);    /* section 1b; v3.0.0 registers none; ext::NpuHard arrives with an NPU adapter */
anira_status anira_contract_set_ext_json (anira_contract*, const char* kind, const char* utf8, size_t len, anira_error* err);
anira_contract_kind anira_contract_get_kind(const anira_contract*);
anira_status anira_contract_from_json(const char* utf8, size_t len, anira_contract** out, anira_error* err);   /* {"hard": {}} | {"async": {}} + "edge_cost", section 8 */
void         anira_contract_destroy(anira_contract*);
```

```cpp
// anira.hpp -- header-only, not ABI-stable; plain aggregates with designated initialisers, minted into an anira_contract at prepare()
struct Hard {
    uint32_t          block_min = 0;  uint32_t block_max = 0;   double rate = 0;
    anira_budget_kind budget  = ANIRA_BUDGET_MEASURED;      std::chrono::nanoseconds budget_value{};   // Explicit only
    anira_warmup_mode warmup  = ANIRA_WARMUP_UNTIL_STABLE;  uint32_t warmup_iterations = 0;            // Fixed only
    anira_miss_policy on_miss = ANIRA_MISS_BYPASS;
    double            wait_ratio = 0;                                                                  // v2 blocking_ratio
    anira_edge_cost   edge_cost  = ANIRA_EDGE_COST_PERMISSIVE;
};
struct Async {
    std::optional<std::chrono::nanoseconds> deadline;       // absent = offline posture; crosses as deadline_ms < 0
    anira_late_policy on_late  = ANIRA_LATE_FINISH;         anira_priority priority = ANIRA_PRIORITY_AUTO;
    uint32_t          lanes = 0;                            uint32_t max_in_flight = 0;
    anira_delivery    delivery  = ANIRA_DELIVERY_POLLED;
    anira_edge_cost   edge_cost = ANIRA_EDGE_COST_PERMISSIVE;
};
using Contract = std::variant<Hard, Async>;                 // prepare(const Contract&), prepare(const Hard&), prepare(const Async&)
```

Entry-point coupling: Hard enables `anira_handler_process`/`push_data`/`pop_data` and their `_wait` twins and disables `anira_handler_submit`; Async the reverse. The C layer enforces it at runtime -- a Hard entry on an Async handler returns 0 / `ANIRA_ERROR_WRONG_CONTRACT`, records it in `anira_handler_rt_error` and logs once through the RT queue (clap-helpers' "offline refused when hard-realtime", as a status instead of a bool) -- and `anira.hpp` documents it, with the `prepare(const Hard&)`/`prepare(const Async&)` overloads making the regime visible at the call site. Soft real-time and offline are documentation vocabulary over Async (with / without deadline), not presets.

`ANIRA_DELIVERY_IMMEDIATE` runs the job's `anira_job_complete_fn` (set once on the frame-invariant `anira_job_options`, section 6) on the inference thread that completed the job, `[inference-thread]` and `ANIRA_NONBLOCKING`, restricted to the `[callback-safe]` entries: `anira_handler_ticket_release` is legal from inside it, `anira_handler_submit` is not. `ANIRA_DELIVERY_POLLED` runs the same callback in the caller of `anira_handler_poll` or `anira_handler_ticket_wait`, which is where the render thread of section 9 wants it.

Waiting. The `ANIRA_NONBLOCKING` Hard entries never wait: a block whose inference has not completed when its output is collected is an `on_miss` event, and nothing else can happen on that path. v2's two in-callback waits -- the `blocking_ratio` wait inside `process()` (`src/scheduler/InferenceManager.cpp:59-73`) and the unbounded wait of `set_non_realtime` -- live in the `_wait` twins of section 6, and the two sentinels above are their contract-side values: `ANIRA_WAIT_CONTRACT` is `wait_ratio` times the block duration, `ANIRA_WAIT_FOREVER` is v2's `set_non_realtime` behaviour, a `timeout_ms >= 0` is neither. `wait_ratio` is v2's `m_blocking_ratio` one-to-one and belongs in the contract because it decides at `prepare` which completion primitive the handler builds, which is a scheduling decision no per-call timeout may revisit. `set_non_realtime` is not a C entry: `compat/v2.hpp` routes the v2 overloads to the twins with `ANIRA_WAIT_CONTRACT` when the legacy contract carries `blocking_ratio > 0` and with `ANIRA_WAIT_FOREVER` while its wrapper-side `set_non_realtime(true)` flag is set (section 10). The twins themselves -- `[any-thread, blocking]`, not `ANIRA_NONBLOCKING` -- their timeouts and their refusal without an inference thread are section 6.

A v2 file's `max_inference_time`, `warm_up` and `blocking_ratio` become a legacy Hard contract -- `ANIRA_BUDGET_EXPLICIT`, `ANIRA_WARMUP_FIXED`, `wait_ratio = blocking_ratio` -- that `anira_model_config_take_legacy_contract` hands out after the auto-upgrade (section 8); `wait_ratio` is the one mapping under which such a file keeps both its latency figure and its in-callback wait. Its geometry, like that of a v3 contract file written without a block range or `rate`, is patched by the host with `hard_set_geometry` before `prepare`.

Prepare validation. Hard: warmup, budget vs block cadence, no waits or allocation reachable from `anira_handler_process`, plus whatever an adapter adds for the extensions it consumes (an NPU adapter's full-offload-or-reject and performance pinning arrive with `ext::NpuHard`, section 1b); with multiple enabled backends all of this holds per plan, and the reported latency covers the slowest enabled plan (section 7, plan sets). Async: deadline feasibility vs measured time, `lanes = 1` for stateful models, no warmup required without deadline. A failed budget is `ANIRA_ERROR_BUDGET`, every other rejected combination `ANIRA_ERROR_CONFIG`, each with the offending field named in `anira_error.message`.

Deadline effects: prepare posture (latency vs throughput defaults), dispatch ordering (EDF ahead of batch) and early rejection under `ANIRA_LATE_DROP`, chunk-boundary cancellation, honest ticket reporting (`ANIRA_TICKET_MET` / `LATE` / `DROPPED`). It is advisory information, not a promise; only Hard's budget changes what code may exist.

---

## 4. MachineConfig (machine and process resources)

`MachineConfig` is an opaque handle, `anira_machine_config`, mutated by scalar setters: threads and wait strategy, the log block, one Tier-2 descriptor per device API, extensions. The handle's layout never enters the ABI; the device descriptors are `struct_size`-first records handed once to a setter and read only inside `min(struct_size, sizeof(lib's))`, so a block grows at the tail inside ABI major 3, and JS -- which never writes a struct field by hand -- builds a machine config with one call per field. A NULL descriptor is an absent block: the domain is unavailable and its edges are pruned. Presence is the user's declaration; the Machine probes what is declared and nothing else.

```c
/* abi/enums.h -- every value pinned, independent of USE_* */
typedef enum anira_ownership     { ANIRA_OWNERSHIP_BORROWED = 0, ANIRA_OWNERSHIP_OWNED = 1, ANIRA_OWNERSHIP_FORCE32 = 0x7fffffff } anira_ownership;   /* BORROWED = user's handles */
typedef enum anira_exec_policy   { ANIRA_EXEC_WORKER = 0, ANIRA_EXEC_USER_DRIVEN = 1, ANIRA_EXEC_POLICY_FORCE32 = 0x7fffffff } anira_exec_policy;
typedef enum anira_gl_threads    { ANIRA_GL_CALLER_THREAD = 0, ANIRA_GL_SHARED_CONTEXT = 1, ANIRA_GL_THREADS_FORCE32 = 0x7fffffff } anira_gl_threads;
typedef enum anira_wait_strategy { ANIRA_WAIT_SPIN_BACKOFF = 0, ANIRA_WAIT_BLOCKING = 1, ANIRA_WAIT_STRATEGY_FORCE32 = 0x7fffffff } anira_wait_strategy;
typedef enum anira_log_level     { ANIRA_LOG_DEBUG = 0, ANIRA_LOG_INFO = 1, ANIRA_LOG_WARNING = 2, ANIRA_LOG_ERROR = 3, ANIRA_LOG_LEVEL_FORCE32 = 0x7fffffff } anira_log_level;
typedef enum anira_log_drain     { ANIRA_LOG_DRAIN_THREAD = 0, ANIRA_LOG_DRAIN_MANUAL = 1, ANIRA_LOG_DRAIN_FORCE32 = 0x7fffffff } anira_log_drain;
typedef enum anira_edge_class    { ANIRA_EDGE_ZERO_COPY = 0, ANIRA_EDGE_DEVICE_COPY = 1, ANIRA_EDGE_HOST_COPY = 2, ANIRA_EDGE_UNAVAILABLE = 3,
                                   ANIRA_EDGE_CLASS_FORCE32 = 0x7fffffff } anira_edge_class;
typedef enum anira_probe_rung    { ANIRA_RUNG_STATIC = 0, ANIRA_RUNG_IDENTITY = 1, ANIRA_RUNG_FUNCTIONAL = 2, ANIRA_RUNG_FORCE32 = 0x7fffffff } anira_probe_rung;
#define ANIRA_THREADS_AUTO 0xffffffffu                          /* library default pool size; 0 = bring your own threads */
#define ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK 1u

/* abi/config.h -- device descriptors: Tier 2, struct_size first, handed once to a setter; NULL = block absent */
typedef struct anira_cuda_desc   { uint32_t struct_size; uint32_t ownership; int32_t device; uint32_t reserved;
                                   uint64_t pinned_pool_limit; } anira_cuda_desc;         /* 0 = planner-sized; cap on cudaHostAlloc staging */
                                   /* nothing to hand over: the primary context is process-wide, so a pointer, stream or
                                      event on it is anira's as much as the user's */
typedef struct anira_gl_desc     { uint32_t struct_size; uint32_t threads;                /* anira_gl_threads; GL is always borrowed */
                                   void* display; void* context;                          /* EGL (or GLX equivalents) */
                                   void* gbm; } anira_gl_desc;                            /* gbm_device*: lets allocate_* back GL storage with a dma-buf */
                                   /* CALLER_THREAD (v1): anira touches GL only inside allocate_*, submit and bind_output, on the
                                      calling thread, where the user's context is current; a call from another thread is a
                                      contract error. SHARED_CONTEXT (additive): the user passes a second context of the same
                                      share group and anira's worker makes it current. */
typedef struct anira_vulkan_desc { uint32_t struct_size; uint32_t ownership; uint32_t queue_family; uint32_t queue_index;
                                   void* instance; void* physical; void* device; } anira_vulkan_desc;
                                   /* thread-agnostic: anira serializes its own submissions on the queue */
typedef struct anira_metal_desc  { uint32_t struct_size; uint32_t reserved; void* device; } anira_metal_desc;   /* NULL = default device */
typedef struct anira_d3d12_desc  { uint32_t struct_size; uint32_t ownership; void* device; } anira_d3d12_desc;
typedef struct anira_webgpu_desc { uint32_t struct_size; uint32_t ownership; uint32_t exec; uint32_t reserved;   /* anira_exec_policy: someone must pump ProcessEvents / WaitAny */
                                   void* instance; void* device; void* queue; } anira_webgpu_desc;   /* WGPUInstance / WGPUDevice / WGPUQueue, typeless at the boundary */
/* ANIRA_CUDA_DESC_INIT .. ANIRA_WEBGPU_DESC_INIT: ANIRA_INIT with the defaults -- OWNED, device 0, CALLER_THREAD, WORKER */

/* abi/log.h -- the log block crosses as scalars in and one frozen record out */
typedef struct anira_log_record {            /* 56 bytes, Tier 1, frozen; valid for the duration of the callback */
    uint32_t level; uint32_t flags; uint32_t dropped_before; uint32_t reserved;   /* flags: ANIRA_LOG_RECORD_REALTIME, ANIRA_LOG_RECORD_CONTRACT_VIOLATION */
    uint64_t sequence; int64_t timestamp_ms; uint64_t monotonic_ns;
    ANIRA_PTR(const char, group); ANIRA_PTR(const char, message);                /* "anira.<component>" */
} anira_log_record;
typedef void (ANIRA_CALL* anira_log_fn)(const anira_log_record*, void* user_data);   /* never the driver thread; must not call anira */
typedef struct anira_log_desc { uint32_t struct_size; uint32_t abi_version; void* user_data; anira_log_fn callback;
    uint32_t level; uint32_t drain; uint32_t queue_capacity; uint32_t drain_interval_ms; uint32_t flags; uint32_t reserved; } anira_log_desc;
#define ANIRA_LOG_DESC_INIT ANIRA_INIT(anira_log_desc, sizeof(anira_log_desc), ANIRA_ABI_VERSION, NULL, NULL, ANIRA_LOG_WARNING, ANIRA_LOG_DRAIN_THREAD, 512, 10, 0, 0)
size_t anira_drain_log(void);   /* [thread-safe, !audio-thread]; process-wide, the core of this copy; returns records delivered */
void   anira_log_rt(anira_log_level, const char* group, const char* static_message, int32_t arg0, int32_t arg1);   /* [thread-safe] [callback-safe] ANIRA_NONBLOCKING */
void   anira_log(anira_log_level, const char* group, const char* message);   /* [thread-safe, !audio-thread]; control-path logging */

/* abi/config.h -- the handle; every setter [main-thread], may allocate; ANIRA_API / ANIRA_CALL elided (section 6a) */
typedef struct anira_machine_config anira_machine_config;
anira_status anira_machine_config_create(anira_machine_config** out, anira_error* err);
anira_status anira_machine_config_set_threads(anira_machine_config*, uint32_t num_threads, anira_wait_strategy);   /* defaults: ANIRA_THREADS_AUTO, SPIN_BACKOFF */
anira_status anira_machine_config_set_log_level(anira_machine_config*, anira_log_level);                     /* default WARNING; most verbose request wins */
anira_status anira_machine_config_set_log_drain(anira_machine_config*, anira_log_drain, uint32_t interval_ms);
anira_status anira_machine_config_set_log_queue_capacity(anira_machine_config*, uint32_t capacity);          /* clamped [64, 65536]; fixed for the life of the core */
anira_status anira_machine_config_set_log_flags(anira_machine_config*, uint32_t flags);
anira_status anira_machine_config_set_log_sink(anira_machine_config*, anira_log_fn, void* user_data);        /* ignored on Wasm: anira_em_set_log_hook is the sink there */
anira_status anira_machine_config_set_log(anira_machine_config*, const anira_log_desc*);                     /* C one-shot convenience = the five above */
anira_status anira_machine_config_set_cuda  (anira_machine_config*, const anira_cuda_desc*);     /* NULL = domain unavailable, edges pruned */
anira_status anira_machine_config_set_gl    (anira_machine_config*, const anira_gl_desc*);       /* presence is the user's declaration, no implicit probing */
anira_status anira_machine_config_set_vulkan(anira_machine_config*, const anira_vulkan_desc*);
anira_status anira_machine_config_set_metal (anira_machine_config*, const anira_metal_desc*);
anira_status anira_machine_config_set_d3d12 (anira_machine_config*, const anira_d3d12_desc*);
anira_status anira_machine_config_set_webgpu(anira_machine_config*, const anira_webgpu_desc*);  /* native Dawn; Emscripten: ANIRA_ERROR_NOT_SUPPORTED in 3.0 */
anira_status anira_machine_config_set_ext(anira_machine_config*, const anira_ext_header*, anira_error* err);           /* section 1b */
anira_status anira_machine_config_set_ext_json(anira_machine_config*, const char* kind, const char* utf8, size_t len, anira_error* err);
anira_status anira_machine_config_from_json(const char* utf8, size_t len, anira_machine_config** out, anira_error* err);   /* section 8; JSON device blocks = OWNED */
anira_status anira_machine_config_to_json(const anira_machine_config*, char* buf, size_t cap, size_t* out_len);
void         anira_machine_config_destroy(anira_machine_config*);

/* abi/machine.h -- the handle over the core, its capabilities, the probe */
typedef struct anira_machine anira_machine;  typedef struct anira_capabilities anira_capabilities;
typedef struct anira_edge_info { uint32_t struct_size; uint32_t from_domain; uint32_t to_engine; uint32_t to_provider; uint32_t edge_class; uint32_t rung; uint32_t available;
                                 const char* reason; } anira_edge_info;                    /* {available, class, reason, rung}; row valid for the call */
anira_status anira_machine_create(const anira_machine_config*, anira_machine** out, anira_error* err);   /* [main-thread]; probes, validates borrowed devices, Dawn assertion, registers the sink */
void         anira_machine_destroy(anira_machine*);            /* [main-thread & !loader-lock]; drops the user's reference, unregisters the sink, joins nothing */
anira_status anira_machine_probe(anira_machine*, anira_bool force, anira_error* err);   /* [main-thread]; re-runs the rungs, refreshes capabilities */
const anira_capabilities* anira_machine_capabilities(const anira_machine*);              /* [thread-safe]; machine-owned */
anira_status anira_capabilities_backends (const anira_capabilities*, uint32_t element_size, uint32_t* count, anira_backend_id* out);   /* compiled in AND usable here */
anira_status anira_capabilities_domains  (const anira_capabilities*, uint32_t* count, anira_domain* out);
anira_status anira_capabilities_ext_kinds(const anira_capabilities*, uint32_t* count, const char** out);
anira_status anira_capabilities_edges    (const anira_capabilities*, uint32_t element_size, uint32_t* count, anira_edge_info* out);   /* stride-explicit */
anira_status anira_capabilities_edge     (const anira_capabilities*, anira_domain from, const anira_backend_id* to, anira_edge_info* out);
anira_status anira_enabled_backends(uint32_t element_size, uint32_t* count, anira_backend_id* out);   /* [thread-safe]; no machine needed: what this build compiled in */
uint64_t     anira_machine_byte_image_bytes(const anira_machine*, uint64_t num_elements, anira_dtype);   /* [thread-safe]; the edge's encoding, section 7 */
size_t       anira_machine_drain_log(anira_machine*);         /* [thread-safe, !audio-thread]; ANIRA_LOG_DRAIN_MANUAL */
uint32_t     anira_machine_num_inference_threads(const anira_machine*);   /* [thread-safe]; the pool behind this handle */
double       anira_now_ms(void) NB;  uint64_t anira_now_ns(void) NB;   /* [thread-safe] [cs]; steady clock for deadlines and submit;
                                                                          now_ns is the one allowlisted 64-bit return on a NB declaration */
anira_status anira_shutdown(void);                            /* [main-thread & !loader-lock]; idempotent, never creates the core;
                                                                 effective only when no machine handle and no handler exists in this copy,
                                                                 else nothing happens and ANIRA_ERROR_INVALID_STATE */
anira_bool   anira_release_core_if_idle(void);                /* [main-thread]; never blocks; on Wasm also requires zero loop-active threads */
anira_bool   anira_has_core(void);                            /* [thread-safe] */
```

Threads. `num_threads == 0` keeps the meaning it has in the tree: the caller opts out of the auto-managed pool and drives inference through `anira_inference_thread` objects (section 6); the Wasm build depends on it, because its inference threads are Workers and a nonzero value is coerced to 0 with a warning there. `ANIRA_THREADS_AUTO` is the library default: `hardware_concurrency() / 2` (minimum 1) natively, 0 on Wasm. The two are distinct values because a zero that meant "auto" would silently start threads in the browser. `wait_strategy` defaults to `ANIRA_WAIT_SPIN_BACKOFF`; `ANIRA_WAIT_BLOCKING` is coerced to it on Wasm, where every wait spins (section 6a).

The machine handle. `anira_machine` is a refcounted handle over the immortal core -- one core per copy of anira (a shared `libanira.so.3` has one, every static embedding has its own), the object that is `Context` today. `anira_machine_create` `[main-thread]` reconciles the config into the core, validates borrowed devices, runs the Dawn version assertion and the probe below, registers the log sink and returns the handle; `anira_handler_create` adds a reference, so the handle's memory (reconciled config, device blocks, capabilities) outlives the user's `anira_machine_destroy`, which drops the user's reference, unregisters the sink, invalidates the handle for the caller regardless of the count, and joins nothing. Thread pool and inference queue are core-owned and exist exactly while any handler in this copy exists: the first machine sizes the pool; later machines reconcile per field as today -- wait strategy first wins, log level most verbose wins, the pool only shrinks and never to zero (`0` = no preference while handlers exist), drain mode, queue capacity and drain interval first win with a warning on mismatch -- and once the last handler is gone the next machine's config takes effect whole. `anira_num_inference_threads` (`abi/thread.h`, section 6) reports the size of that default pool and, like `anira_enabled_backends`, takes no machine; `anira_machine_num_inference_threads` is the per-handle answer, the same number in 3.0.0, where every machine is served by the default pool. Two machine handles in one copy are two views of one core with two log sinks, sharing the default pool rather than owning one each; `anira_shutdown` (section 6) is refused while any of them or any handler exists. `ContextConfig::m_anira_version` and `m_enabled_backends` have no successor fields: what the build compiled in is `anira_enabled_backends`, a function that needs no machine; what is usable *here* -- a compiled-in backend whose driver is absent is enabled but not usable -- is `anira_capabilities_backends` on the probed `anira_capabilities`; the cross-session version compare is dropped, because inside one copy every session shares one header and across two copies nothing can see the other.

The same object in C++ is a builder over the handle, and `anira::Machine` owns one `anira_machine*`:

```cpp
class MachineConfig { static MachineConfig from_json(std::string_view); static MachineConfig from_file(const std::filesystem::path&);
    MachineConfig& threads(uint32_t num_threads, anira_wait_strategy = ANIRA_WAIT_SPIN_BACKOFF);
    MachineConfig& log_level(anira_log_level); MachineConfig& log_sink(LogSink); MachineConfig& log(const anira_log_desc&);
    MachineConfig& cuda(const anira_cuda_desc&); MachineConfig& gl(const anira_gl_desc&); MachineConfig& vulkan(const anira_vulkan_desc&);
    MachineConfig& metal(const anira_metal_desc&); MachineConfig& d3d12(const anira_d3d12_desc&); MachineConfig& webgpu(const anira_webgpu_desc&); };
class Machine { explicit Machine(const MachineConfig&); const Capabilities& capabilities() const; void probe(bool force = false);
                size_t drain_log(); anira_machine* native() const; };

anira::MachineConfig mc;
mc.threads(ANIRA_THREADS_AUTO).log_level(ANIRA_LOG_WARNING)
  .gl(anira_gl_desc{ sizeof(anira_gl_desc), ANIRA_GL_CALLER_THREAD, egl_display, egl_context, gbm });   // borrowed by definition
anira::Machine machine{mc};                                                                               // anira::Error on ANIRA_FAILED
```

`anira_machine_config_set_ext`/`set_ext_json` carry the section 1b payloads (`ext::Npu` and `ext::ArtifactCache` arrive with their adapters); the device blocks are setters, not extensions, because the Machine's probe reads them, not an adapter. `anira_machine_config_from_json` reads the `machine.json` of section 8, whose device blocks imply `ANIRA_OWNERSHIP_OWNED`; borrowed handles are code-only and are patched onto the loaded handle by the setters afterwards.

WebGPU ownership: one WebGPU implementation per process, and anira owns it. The Machine links its own Dawn (`libwebgpu_dawn.so`, monolithic shared build from a pinned Dawn revision) and creates or borrows the instance, device and queue on it -- `WGPUInstance`/`WGPUDevice`/`WGPUQueue`, carried as `void*` in `anira_webgpu_desc` because the header names no vendor type. Engines never bring their own Dawn; they receive anira's through a proc table: a `WGPUDevice` is a C++ object of one build, not an ABI-stable handle, and `DawnProcTable` is a struct whose layout is fixed by the Dawn revision, so engine and Machine must be built against the same Dawn source tree. Per engine:

- ORT: built with `onnxruntime_USE_EXTERNAL_DAWN=ON` and `onnxruntime_CUSTOM_DAWN_SRC_PATH=<anira's Dawn tree>` (ORT then links only `dawn_proc` thunks, hidden behind its version script); the adapter passes `dawnProcTable = dawn::native::GetProcs()` of anira's library plus `webgpuInstance`/`webgpuDevice` with `deviceId >= 1` when creating the session. The proc table is installed once per process (ORT `call_once`), which matches one core, and so one Dawn, per copy of anira.
- An engine that statically embeds Dawn (ORT's default build, LiteRT's prebuilt WebGPU accelerator) cannot share the device and is treated as a `{Host}` consumer.
- The host app's renderer, if it uses WebGPU, links anira's Dawn and borrows the device (`ANIRA_OWNERSHIP_BORROWED`).
- The browser is not a WebGPU device of v3.0.0 -- on schedule and on measurement, not on architecture. Every browser WebGPU implementation reachable from Wasm completes only when the calling agent returns to its event loop, and the inference Worker never does: it is parked in `anira_inference_thread_run_loop` for the whole session. That rules out a completion observed *on that thread*, not one observed at all -- a GPU proxy Worker whose own loop stays free, rendezvousing through `Atomics.wait` on the shared memory, blocks only an agent that is allowed to block and fits `anira_backend_desc.process` exactly as it is already tagged. Every such path is freeze-neutral, which is what makes it a later minor rather than a v3.0.0 risk. `ONNXRUNTIME` on `ANIRA_PROVIDER_WEBGPU`, `anira_machine_config_set_webgpu` and `anira_tensor_init_wgpu_buffer` stay in the header as the native Dawn path; on Emscripten `set_webgpu` returns `ANIRA_ERROR_NOT_SUPPORTED`. Browser WebGPU arrives post-3.0 as an explicitly asynchronous JS backend -- an `anira_backend_desc` tail slot with a completion callback, driven from an unblocked Worker pump -- additive under ABI major 3 (section 10).

Borrowed devices are validated at `anira_machine_create` for the features the edges need (`SharedTextureMemoryDmaBuf`, `SharedFenceSyncFD`, `DawnMultiPlanarFormats`, `HostMappedPointer`).

Borrowing differs per API, because the APIs do. WebGPU: anira owns the implementation (above). Vulkan: device and queue are usable from any thread under external synchronization; hand them over once. GL: a context is current on exactly one thread, so `anira_gl_threads` above is the whole story, and v1 is `ANIRA_GL_CALLER_THREAD`. CUDA: the primary context is shared by every library in the process; anira retains it (`cudaSetDevice` / `cuDevicePrimaryCtxRetain`) and asserts at construction that user pointers belong to it (`cudaPointerGetAttributes` fails with `cudaErrorIncompatibleDriverContext` for a pointer from a context created with `cuCtxCreate`). Toolkit versions need not match between anira and an engine: every CUDA runtime funnels into the driver's one `libcuda.so`, and only the driver's minimum for the CUDA major family matters -- the opposite of the Dawn revision lock.

Why external Dawn, and what it costs. ORT's external-Dawn mode is the only one in which ORT is a *consumer* of a device rather than its provider; with ORT owning Dawn, no second WebGPU engine and no renderer could ever share the device. The price is a revision lock: `DawnProcTable` is a plain struct of ~280 function pointers whose order changes between Dawn revisions, so Dawn, ORT and every other WebGPU engine form one versioned triple per anira release (Dawn revision pinned to ORT's `deps.txt`; an ORT upgrade means re-pinning and rebuilding Dawn, rebuilding ORT from that tree, re-validating the other engines). The proc table is process-global (ORT `call_once`), consistent with one core per copy of anira; it also means anira cannot coexist with another library installing a different Dawn in the same process, which is true of any arrangement but here fails loudly. Build complexity is two builds instead of one and no `find_package` path in ORT (`onnxruntime_CUSTOM_DAWN_SRC_PATH` is what ORT's own CI uses for custom Dawn). Debuggability improves: one Dawn, one validation layer, one toggle set, one device-lost callback.

Version assertion (required): the Machine compares the Dawn it loaded against the revision anira was built with -- `kDawnVersion` from `dawn/dawn_version.h` baked at build time versus the version reported by the loaded library -- and `anira_machine_create` fails with `ANIRA_ERROR_DEVICE` and both revisions in `anira_error.message` on mismatch. This turns the one real failure mode of this arrangement, a silently mismatched proc table, into an immediate, readable error. The same check runs for every engine that consumes the proc table.

Probing: domain and edge availability are *driver* facts, not platform facts. At `anira_machine_create` the Machine enumerates Vulkan device extensions, `wgpuAdapterHasFeature`, EGL/GL extension strings, CUDA attributes and each engine's buffer requirements, and fills the edge registry (section 7) from the answers. Every cross-API row also requires that the two devices are the same physical GPU -- an exported allocation is memory on one adapter, and a second API on another adapter cannot see it -- so the Machine compares device identity across the enabled blocks at construction (Vulkan `VkPhysicalDeviceIDProperties::deviceUUID`, CUDA `cudaDeviceProp::uuid`, Dawn's adapter info, the D3D12 adapter LUID on Windows) and marks a cross-API row unavailable, with the mismatch in the report, when the identities differ; a machine with two GPUs therefore needs its device blocks to name the same one, and the Vulkan block that exists only to mint exportable memory for the `OpaqueFd` rows (section 7) must land on the GPU Dawn and CUDA use. For the GL rows the check is CUDA's own: `cudaGLGetDevices` on the thread where the borrowed context is current (`ANIRA_GL_CALLER_THREAD`, so inside `anira_machine_create`) names the CUDA device backing that context, and the `GlBuffer -> CUDA` registration row is enabled only when it names the CUDA block's device -- no device (the context is on an iGPU), a different one, or a failure on the right GPU because the GL driver is not the vendor's (`cudaGraphicsGLRegisterBuffer` lives in NVIDIA's GL, not in Mesa) each disables the row, not the block, and the plan falls back to `glGetBufferSubData` with the reason in the report. `GL_EXT_memory_object`'s `GL_DEVICE_UUID_EXT` would give the same identity driver-neutrally, but `cudaGLGetDevices` proves the interop path exists and not merely that the GPUs match. The dma-buf rows (`GL -> WebGPU`, `Vulkan -> WebGPU`) can in principle cross GPUs, since a dma-buf is a kernel object; that is the importing driver mapping foreign memory, a copy in disguise behind a modifier the other device may not understand, so same-GPU is their precondition too and the probe is the import succeeding on a test allocation. Measured example: a UMA device (Apple M1 under Mesa Honeykrisp) lacks `VK_EXT_external_memory_host`, so host-pointer-import edges (persistent map, Dawn `HostMappedPointer`) are absent although the memory is unified; the software device on the same machine has them. "UMA" therefore never appears as a planner condition; only probed edges do. The probed registry is readable before any handler exists through `anira_machine_capabilities`, and `anira_handler_prepare` reports it again with the plan.

### Probing: the three rungs

A row enters the registry only after it has passed three rungs, and the prototype's history is the argument for the third. *Static* (`ANIRA_RUNG_STATIC`): the extension and feature bits above -- cheap, necessary, never sufficient. *Identity* (`ANIRA_RUNG_IDENTITY`): the two APIs sit on the same physical GPU (above). *Functional* (`ANIRA_RUNG_FUNCTIONAL`): the Machine runs the row once, end to end, and checks the bytes. Every silent failure the prototype met passed a feature check: the dma-buf that imported without complaint and was written at a driver-rounded pitch (`max_abs_err` 175 on every dma-heap output), the `imageStore` into an EGLImage-backed texture that satisfied `glReadPixels` and left the dma-buf zeroed behind a tiled shadow, the tensor whose `OrtMemoryInfo` device id was off by one and was copied into an EP-owned buffer under a row that said `ZeroCopy`, the captured graph replaying the buffers of run 0. None of those is an extension bit, and each was found by a round trip with a pattern and a compare. The functional rung is that discovery made routine and moved onto the user's machine, because the matrix of section 7 measured *this* driver and the user's may differ in exactly these ways.

The functional rung, per row: allocate a few KB the row's own way -- the exact recipe production will use, exportable memory with the linear image bound, a rendered-into `gbm_bo`, a `WGPUBuffer` on the Machine's Dawn; write a known pattern from the *producer* side through the producer's API; execute the edge as the plan would, import, pass, registration, fence hand-off included; read back from the *consumer* side through the consumer's API and compare bit-exact. For engine rows the pattern is an identity model run twice with different inputs whose outputs must differ -- the `hello_inference` stale check, which is what proves the engine read the caller's buffer rather than a private copy, and what decides whether graph capture is permitted on this driver. The result is an `anira_edge_info` row `{available, class, reason, rung}`, never a bool -- `edge_class` an `anira_edge_class`, `rung` the last `anira_probe_rung` passed, `reason` static text owned by the capabilities object -- enumerated by `anira_capabilities_edges` (stride-explicit, so a later minor may append fields) or queried one at a time by `anira_capabilities_edge(caps, from, to, &info)`: `unavailable: import ok, readback mismatch at byte 72 (pitch)` is what the capability report and the plan report show, the runtime twin of "a foreign handle that works but slower is data, never a log line". It measures correctness and cost *class*, not time; time is the Hard warmup's and the benchmark sweep's.

| row (Linux v1) | static | functional |
|---|---|---|
| `WgpuBuffer -> WebGPU EP`, `-> WgpuBuffer` | Dawn present; the adapter's memory info compares equal to the EP allocator's | identity model, varying input, stale check; repeated with capture on, which is what enables capture |
| `DmaBuf -> WebGPU`, `WebGPU EP -> DmaBuf` | `SharedTextureMemoryDmaBuf`; `GetProperties` usage includes `TextureBinding` / `StorageBinding` for RGBA8 at that modifier | mmap-write pattern -> import -> texels->buffer pass -> readback, at the byte-image pitch |
| `VulkanBuffer -> WebGPU` (via `allocate_*`) | `external_memory_fd`, `external_memory_dma_buf`, `image_drm_format_modifier`; image+buffer alias accepted, else the `imageStore` writer | as the dma-buf row, written through the alias |
| GL renderbuffer `-> WebGPU` | GBM device; EGL dma-buf export/import; the modifier round-trips | *render* into the renderbuffer, export, import, compare; readback needs `eglWaitSync` + re-target |
| `WgpuBuffer <-> OpaqueFd`, `OpaqueFd -> CUDA` | `external_memory_fd` / `external_semaphore_fd` with the OPAQUE bit; Dawn's opaque-fd shared-texture feature; `cudaImportExternalMemory` | CUDA write -> pass into `WGPUBuffer` -> compare, and back; a fence signalled on one side and waited on the other with a timeout |
| `GlBuffer -> CUDA` | `cudaGLGetDevices` names the CUDA block's device | register, map, write, unmap, `glGetBufferSubData`, compare |
| `Cuda -> CUDA EP` | pointer on the primary context; EP memory info device id equals the block's | identity model, stale check |
| `DmaBuf -> CPU EP` | `DMA_BUF_IOCTL_SYNC` accepted | device write -> sync bracket -> host read, compare |
| host-pointer import rows | `VK_EXT_external_memory_host`, Dawn `HostMappedPointer` | import a page-aligned host block, device write, host read |
| fences, every kind | `SharedFenceSyncFD` / opaque-fd semaphore export; `EGL_ANDROID_native_fence_sync` | an empty job's fence signals within a bound -- a fence that signals 8 ms late is a row that lies about its class |

Cost and caching. Functional probes are milliseconds each and there are dozens, and a plugin constructs its Machine inside `prepareToPlay`. Results are therefore cached on the box, keyed by `(anira version, driver versions, device UUIDs, enabled blocks)`, next to where an artifact cache would live: a matching key loads the registry in microseconds, a mismatch re-probes. `anira_machine_probe(m, force, err)` re-runs on demand and refreshes the `anira_capabilities` object that `anira_machine_capabilities` returns, and a benchmark run always forces, since a row cached under last month's driver measures nothing.

Thread and device contracts. GL rungs need the borrowed context current and run on the thread that calls `anira_machine_create` under `ANIRA_GL_CALLER_THREAD`, as every GL touch does. Functional probes submit once to the Vulkan, Dawn and CUDA queues at construction; on an `ANIRA_OWNERSHIP_BORROWED` device that is a submission the application did not make, which is part of what borrowing means and is stated here so that it is not a surprise.

How a thread waits for a GPU is a machine-level decision, not a detail: `anira_wait_strategy` governs GPU completion waits as much as queue waits. A thread that blocks in a fence wait for the length of an inference leaves its core idle, and the CPU frequency governor clocks it down; the *next* thing that runs there -- the pre-processing of the following frame, the host callback -- then executes slow. Measured on the M1 under `schedutil`: identical producer code took 46 us when it followed a CPU inference and 208 us when it followed a 10 ms GPU fence wait, edges 113 us versus 28 us, and even the engine's own submission overhead grew by ~0.7 ms; pinning the governor to `performance` collapsed every column onto the busy-core value. Consequences: `ANIRA_WAIT_SPIN_BACKOFF` is the correct default for deadline-carrying contracts (Hard always, Async with a deadline), a blocking wait is for offline postures, and the plan report names the wait strategy used per edge (`anira_plan_slot.wait_strategy`). Benchmarks record the governor (see section 7) -- otherwise the two regimes differ by 3-5x on everything except the inference itself and the numbers are not comparable.

The core also owns one runtime environment per engine (`Ort::Env`, LibTorch/c10 globals, the TFLite interpreter's shared state, the LiteRT environment, the ExecuTorch runtime and its XNNPACK thread pool; a later engine brings its own), created lazily for enabled backends and shared by every model, plan and handler in the copy. Today the environment is per processor *instance* -- `Ort::Env` and `LiteRtEnvironment` are members of the pimpl `Instance` (`src/backends/OnnxRuntimeProcessor.cpp:71`, `src/backends/LiteRtProcessor.cpp:87`), `num_parallel_processors` of them per processor, and sharing happens only through the pooling of processors with equal configs -- so the consolidation is v3 work, not a description of the tree. Adapter-level session sharing (ORT shared allocators and prepacked-weight sharing) rides on these shared environments and is where the memory win for multi-model setups lands. No engine type crosses the ABI: the environments are reachable through no C function.

Logging is `thl::Logger` (tanh-lib's core component, adopted in v2.3.0 -- section 11), and anira's is a *private* copy: tanh_core's objects are absorbed into libanira, no public header includes it, and an application that links tanh-lib itself has a second, invisible logger that anira never touches -- the sentence "anira never calls `set_config`" survives as "the two loggers cannot see each other". Logging crosses the ABI as `anira_log_desc` in and `anira_log_record` out. `level` maps onto the private logger's runtime level (anira `DEBUG 0..ERROR 3`, thl `Error 1..Debug 4`; the compiled-in ceiling is pinned to 4 in every build type through tanh-lib's `TANH_LOG_COMPILED_MAX_LEVEL` option, set as a plain variable before the fetch (section 6b) -- tanh-lib itself compiles in `Error` only in Release (`tanh-lib/CMakeLists.txt:136-137`) -- so `ANIRA_LOG_WARNING` means the same in Release) and is still forwarded to the engines (ORT, LiteRT, LibTorch; TFLite and ExecuTorch have no runtime level); default `ANIRA_LOG_WARNING`, most verbose request wins across machines. Sinks are machine-scoped behind one trampoline: anira installs `thl::Logger::set_callback` once per core and fans out to a per-copy registry `{callback, user_data, level, in_flight}` with one entry per machine that set a sink; `anira_machine_destroy` unregisters its entry and blocks until that entry's in-flight count is zero, and a destroy issued from inside a sink is refused with an `ANIRA_LOG_ERROR` record, so two instances of one plugin in one DSO each keep their own sink and neither is called after its destroy. The platform sink (stderr, logcat, os_log) stays on beside a callback sink, v2 behaviour, until any live machine sets `ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK`. `anira_log_fn` runs on whichever thread logs -- the caller of any `[main-thread]` entry, the caller of `anira_machine_destroy`/`anira_shutdown` for the final flush, the `[drain-thread]` for real-time records -- never on the driver thread, possibly under anira's lifecycle lock, and therefore must not call anira. `anira_log_record` is a projection of `thl::Logger::LogRecord`: `level`, `flags` (`ANIRA_LOG_RECORD_REALTIME` for a record that came through the queue, `ANIRA_LOG_RECORD_CONTRACT_VIOLATION` for one raised by the C layer or the wrapper), `dropped_before` (the queue's drop counter), `sequence`, `timestamp_ms`, `monotonic_ns`, `group` (`"anira.<component>"`, verbatim), `message`; Tier 1, 56 bytes, valid for the duration of the callback. Nothing reachable from an `ANIRA_NONBLOCKING` entry -- `anira_handler_process`, `push_data`, `pop_data`, `submit` -- calls the logger; it allocates and locks, and RTSan enforces that without a suppression. Real-time paths format on the caller's stack into fixed-size records (256-byte message, 32-byte group) of the core-owned lock-free queue -- `queue_capacity` records, clamped to `[64, 65536]`, created once per core and never resized, a full queue drops and counts -- and the queue is drained by the private low-priority drain thread `"anira-log"` every `drain_interval_ms` (`ANIRA_LOG_DRAIN_THREAD`) or by the host through `anira_drain_log()` / `anira_machine_drain_log(m)` (`ANIRA_LOG_DRAIN_MANUAL`, forced on Wasm and pumped from a main-thread timer there) -- never by the inference threads, whose loop is the inference. Host RT code logs the same way: `anira_log_rt(level, group, static_message, arg0, arg1)` is `[thread-safe] [callback-safe] ANIRA_NONBLOCKING`, writes `"%s [%d %d]"` into the same queue, and takes static strings and two `int32_t` -- the message is never a format string. It exists only for callers on an `ANIRA_NONBLOCKING` path; everything else, host and library alike, logs through `anira_log(level, group, message)` `[thread-safe, !audio-thread]`, which formats and allocates like any other control-path call. On Wasm `anira_em_set_log_hook(1)` is the only sink installer and every record reaches `Module.anira.log` on the instance that emits or drains it; `set_log_sink` is ignored there.

What is logged and what is returned (the error-and-log strategy, `docs/anira-v3-error-and-log-strategy.md`, decided 2026-09-03): a failure that can be returned is returned as a status plus the caller's `anira_error` and is **not** logged (Abseil's rule: a low-level routine passes the status up); a failure that has no channel -- an anira-owned thread, a void or destroy entry, a sink -- is logged once at Error; a real-time refusal does both, the status into `rt_error` and one latched record flagged `ANIRA_LOG_RECORD_CONTRACT_VIOLATION` (SQLite's `SQLITE_MISUSE` argument: the return value is presumed unread). The one returned failure the firewall also logs is `ANIRA_ERROR_INTERNAL`, the non-fatal CHECK. `ANIRA_LOG_FLAG_TRACE_FAILURES` on the machine config turns every failed status into one Error record besides, for the application that swallowed the status; off by default on every platform. Delivery: a control-path record runs every sink on the calling thread before the entry returns (the console sink flushes per record; `__android_log_write` is one datagram to logd; `os_log` lands in logd-shared memory), so a record emitted before a status is at the sink before the caller sees the status; a real-time record reaches the sinks at the next drain, and additionally **on the thread of a failing `[main-thread]` entry**, which drains the queue before returning a negative status (never from inside a sink, never from an `ANIRA_NONBLOCKING` entry). anira installs no crash or terminate handler; crash-time preservation is the host's, through its `anira_log_fn`. Level table: Error -> `ANDROID_LOG_ERROR` / `OS_LOG_TYPE_ERROR` (persisted), Warning -> `WARN` / `DEFAULT` (persisted), Info -> `INFO` / `INFO` (memory only), Debug -> `DEBUG` / `DEBUG` (only when streamed); every failure record is Error. Identity: the private copy files records under the Android tag `anira` and an Apple subsystem/category of anira's own (tanh-lib's identity fields), so a device filter on `anira` finds them.

---

## 5. ModelConfig (model semantics)

One `ModelConfig` is one opaque handle, `anira_model_config`, built by scalar setters and destroyed by its creator. The entries v2 spells as a `std::vector<ModelData>` are rows of that handle, addressed by the `uint32_t` index the `add_model_*` call returned; there is no `ModelData` struct in the ABI. Nothing about the handle's layout is a binary promise, which is why a row may gain a field in a minor without a new struct, and why JS builds one with one call per field instead of assembling a libc++ vector by hand (section 6a). Every entry below is `[main-thread]`, may allocate, and lives in `abi/config.h` (M1).

```c
/* abi/enums.h -- values pinned, independent of USE_* */
typedef enum anira_model_state     { ANIRA_MODEL_STATELESS = 0, ANIRA_MODEL_STATEFUL = 1, ANIRA_MODEL_STATE_FORCE32 = 0x7fffffff } anira_model_state;
typedef enum anira_bytes_ownership { ANIRA_BYTES_COPY = 0, ANIRA_BYTES_BORROW = 1, ANIRA_BYTES_OWNERSHIP_FORCE32 = 0x7fffffff } anira_bytes_ownership;
/* no anchor sentinel: set_anchor(cfg, canonical) names the tensor; an empty name is the default (first Streamed input, else output) */

/* abi/config.h -- ANIRA_API / ANIRA_CALL elided; every entry [main-thread], may allocate */
typedef struct anira_model_config anira_model_config;
typedef void (ANIRA_CALL* anira_bytes_release_fn)(const void* bytes, void* ctx);   /* [main-thread]; once, when the last carrier dies */
anira_status anira_model_config_create(anira_model_config** out, anira_error* err);
anira_status anira_model_config_add_model_path (anira_model_config*, anira_engine, const char* utf8_path, uint32_t* out_index, anira_error* err);
anira_status anira_model_config_add_model_bytes(anira_model_config*, anira_engine, const void* bytes, size_t size, anira_bytes_ownership,
                                                anira_bytes_release_fn release, void* ctx, uint32_t* out_index, anira_error* err);
anira_status anira_model_config_add_model_path_custom (anira_model_config*, const char* engine_id, const char* utf8_path, uint32_t* out_index, anira_error* err);
anira_status anira_model_config_add_model_bytes_custom(anira_model_config*, const char* engine_id, const void* bytes, size_t size, anira_bytes_ownership,
                                                       anira_bytes_release_fn release, void* ctx, uint32_t* out_index, anira_error* err);
anira_status anira_model_config_set_model_bytes(anira_model_config*, uint32_t model_index, const void* bytes, size_t size, anira_bytes_ownership,
                                                anira_bytes_release_fn release, void* ctx, anira_error* err);   /* patches a JSON-produced path entry */
uint32_t      anira_model_config_model_count(const anira_model_config*);
anira_engine  anira_model_config_model_engine(const anira_model_config*, uint32_t model_index);
const char*   anira_model_config_model_engine_id(const anira_model_config*, uint32_t model_index);   /* object-owned; NULL for a built-in engine */
const char*   anira_model_config_model_path(const anira_model_config*, uint32_t model_index);   /* object-owned; NULL for a bytes entry */
anira_status  anira_model_config_model_bytes(const anira_model_config*, uint32_t model_index, const void** bytes, size_t* size);   /* INVALID_STATE for a path entry */
anira_status anira_model_config_set_tensor_name(anira_model_config*, uint32_t model_index, const char* canonical, const char* engine_name);
anira_status anira_model_config_set_model_ext(anira_model_config*, uint32_t model_index, const anira_ext_header*, anira_error* err);
anira_status anira_model_config_set_model_ext_json(anira_model_config*, uint32_t model_index, const char* kind, const char* utf8, size_t len, anira_error* err);
anira_status anira_model_config_add_input (anira_model_config*, const anira_tensor_spec*);    /* copied */
anira_status anira_model_config_add_output(anira_model_config*, const anira_tensor_spec*);
anira_status anira_model_config_set_default_engine(anira_model_config*, anira_engine);        /* ANIRA_ENGINE_NONE = models[0] */
anira_status anira_model_config_set_default_engine_custom(anira_model_config*, const char* engine_id);
anira_status anira_model_config_set_state(anira_model_config*, anira_model_state);            /* STATEFUL = session-exclusive: forces lanes = 1 */
anira_status anira_model_config_set_max_instances(anira_model_config*, uint32_t);             /* default 1; memory ceiling, the planner allocates lanes/pool within */
anira_status anira_model_config_set_anchor(anira_model_config*, uint32_t index_or_first_streamed, anira_bool is_input);
anira_status anira_model_config_set_ext(anira_model_config*, const anira_ext_header*, anira_error* err);
anira_status anira_model_config_set_ext_json(anira_model_config*, const char* kind, const char* utf8, size_t len, anira_error* err);
anira_status anira_model_config_from_json(const char* utf8, size_t len, const char* base_dir, anira_model_config** out, anira_error* err);  /* ANIRA_SUCCESS_UPGRADED for a v2 document */
anira_status anira_model_config_from_json_file(const char* utf8_path, anira_model_config** out, anira_error* err);
anira_status anira_model_config_to_json(const anira_model_config*, char* buf, size_t cap, size_t* out_len);
anira_status anira_model_config_take_legacy_contract(anira_model_config*, anira_contract** out);   /* non-NULL only after a v2 upgrade; caller destroys */
void         anira_model_config_destroy(anira_model_config*);
```

Model entries. One row per engine able to run this model, keyed by `anira_engine`, or by the registered name through `add_model_path_custom`/`add_model_bytes_custom` for an engine the host registers, whose numeric value is assigned per pipeline at prepare and does not exist here (section 6a). No provider appears in a model config, nor in `model.json`: which file to load is a model semantic and which device runs it is a machine resource, so one `.onnx` file is named once and serves every provider candidate of its engine. `add_model_path` copies the UTF-8 path, widened internally on Windows. `add_model_bytes` takes `ANIRA_BYTES_COPY` or `ANIRA_BYTES_BORROW` plus an optional `release(bytes, ctx)`; borrow is the plugin default, because a `tanh_add_binary_data` blob lives as long as the DSO. v2 borrows binary blobs silently and never frees them (`include/anira/InferenceConfig.h:145-149`, the `m_is_binary` flag); v3 says so in the signature. Borrowed bytes sit in a refcounted carrier that every copy shares -- pipeline, handler, and the pooled processor whose lifetime is decoupled from the handler (`include/anira/backends/BackendBase.h:99`) -- and `release` fires exactly once, when the last carrier dies, on the `[main-thread]` caller that destroys it (section 6a). `set_model_bytes` patches the entry a JSON file produced: the browser has no filesystem and a plugin embeds its model beside its `model.json`, so the loader's `path` row becomes a bytes row without a second config. `model_path` and `model_bytes` exist because a custom backend, native or JS, reads its own model through the config it was given at `prepare` (today `ONNXRuntimeWebBackend.init()` does exactly this through a getter on the v2 config, `web/src/backends/ONNXRuntimeWebBackend.ts:126-160`); the config is the one place a model lives, and no side channel carries it. A row whose engine this build lacks is accepted, from code or JSON, and refused at `anira_handler_prepare` with `ANIRA_ERROR_NOT_SUPPORTED`: presence is a runtime query (`anira_enabled_backends`), and one config serves every build.

Tensor names. `set_tensor_name(cfg, i, canonical, engine_name)` maps a spec's canonical name to what model `i`'s graph calls it. A spec is written once; per model only the names differ. Today engines bind tensors positionally -- the ONNX adapter sizes its name arrays from `GetInputCount()`/`GetOutputCount()` and fills them by index (`src/backends/OnnxRuntimeProcessor.cpp:172-183`) -- and per-backend `TensorShape` overrides exist because nothing else could express a graph that orders its tensors differently (`InferenceConfig.h:246-252`); in v3 the name map is the whole per-model difference and there is one canonical spec list.

Entry point. `set_model_ext(cfg, i, &entry.header, err)` with an `anira_ext_entry` is `ext::Entry{name}` of section 1b -- v2's `model_function` (`InferenceConfig.h:154`), absent means `forward`, consumed by LibTorch and ExecuTorch; `set_model_ext_json(cfg, i, "entry", utf8, len, err)` is the JS spelling of the same row. A known kind with an unregistered `version` is refused at set time with `ANIRA_ERROR_EXTENSION_VERSION`; an unknown kind is stored and fails `prepare` by name (section 1b).

Specs, selection, state, instances, anchor. `add_input`/`add_output` copy the `anira_tensor_spec`; the spec handle may be destroyed right after, and the order of the calls is the slot order every handler entry indexes (`tensor_index`, `slot`). `set_default_engine`: `ANIRA_ENGINE_NONE` means `models[0]`, and `set_default_engine_custom` names a registered engine. `set_state`: `ANIRA_MODEL_STATEFUL` is session-exclusive and forces `lanes = 1`, v2's `m_session_exclusive_processor` (`InferenceConfig.h:713`). `set_max_instances` defaults to 1: a memory ceiling, within which the planner allocates lanes and pool; v2's `m_num_parallel_processors` defaults to `hardware_concurrency() / 2` (`InferenceConfig.h:456-459`), which is the wrong default for a ceiling. `set_anchor(cfg, canonical)` names the clock for `time_ratio` and Hard geometry by the tensor's canonical name (an empty name restores the default), moved here from v2's `HostConfig` (`k_first_streamable`, `m_tensor_index`, `m_tensor_is_input`, `include/anira/utils/HostConfig.h:59,102-106`) because the clock is a property of the model, not of the host; resolution is the tree's: first Streamed input, else first Streamed output (generator).

JSON. `from_json`/`from_json_file` are the second constructor of the handle, produced by the same private `nlohmann::json` code that never reaches a header; `base_dir` resolves relative paths. A v2 document (an `inference_config` root) is upgraded in place -- the key table is in section 8 -- returns `ANIRA_SUCCESS_UPGRADED` (a positive status: the stable test is `ANIRA_FAILED`, never `!= ANIRA_OK`) and logs one `ANIRA_LOG_WARNING` per process. `take_legacy_contract` hands out the Hard contract that document implied, carrying `max_inference_time`, `warm_up` and `blocking_ratio` as `budget`, `warmup` and `wait_ratio` (section 3); it is non-NULL only after an upgrade and the caller destroys it. `to_json` round-trips a handle into a caller-owned buffer (`ANIRA_ERROR_BUFFER_TOO_SMALL` with the needed length). Unknown keys are not dropped: they are stored as extension rows and fail `prepare` by name, as section 1b requires.

Lifetime. The handle is value-like: `anira_handler_create` copies it, so it may be destroyed right after the call. v2 keeps an `InferenceConfig&` for the handler's whole life (`include/anira/InferenceHandler.h:455`); that trap does not exist in v3.

The C++ view (`anira.hpp`, header-only, not ABI-stable) is a value class over the handle with the same operations as members, chained on the setters:

```cpp
class ModelConfig {
    static ModelConfig from_json(std::string_view, std::string_view base_dir = {});  static ModelConfig from_file(const std::filesystem::path&);
    uint32_t add_model_path(Engine, const std::filesystem::path&);   uint32_t add_model_path(std::string_view engine_id, const std::filesystem::path&);
    uint32_t add_model_bytes(Engine, std::span<const std::byte>, anira_bytes_ownership = ANIRA_BYTES_COPY);
    void set_model_bytes(uint32_t, std::span<const std::byte>, anira_bytes_ownership = ANIRA_BYTES_COPY);
    std::string_view model_path(uint32_t) const;  std::span<const std::byte> model_bytes(uint32_t) const;
    ModelConfig& input(const TensorSpec&);  ModelConfig& output(const TensorSpec&);
    ModelConfig& default_engine(Engine);    ModelConfig& state(anira_model_state);
    ModelConfig& max_instances(uint32_t);   ModelConfig& anchor(uint32_t, bool is_input);
    std::string to_json() const;  std::optional<ContractHandle> take_legacy_contract();   // as built: a handle, not the aggregate
};
```

Nothing else is on the handle: the private quantization arena and its span-rebinding rule left with `ext::Quant`, which owns its own vectors (section 1b). Quantization is not on the config in v1 at all: a quantized model's true `dtype` is in its spec, and a float producer meeting an int8 spec fails prepare with the extension named, never a silent conversion.

One `ModelConfig` describes one logical model. A variant set references several configs inside one handler; running models in sequence is several handlers on one Machine, composed through tickets (section 7, Multi-model support).

One-sided streaming. The anchor is whichever Streamed tensor is the clock, input or output, and everything time-related is stated in its elements. In a generator (no Streamed input) `Hard`'s block range and `rate` are in elements of the Streamed *output*, `process()` is a pull -- Static inputs arrive through `set_input`, `push_data` has nothing to push -- and `get_latency()` counts from the first `process()`. In an analyser (no Streamed output) `process()` pushes, Static outputs leave through `get_output` (a ticket under Async), and `get_latency()` covers Streamed outputs only: a Static output carries no stream latency and reports 0. The latency vector (`anira_handler_get_latencies`) stays index-aligned with the output list, which is what the tree does since the one-sided streaming fix (`include/anira/InferenceHandler.h:357-360`) and the only representation under which a C array indexed by slot cannot misalign. Both are the same RingChunker with one side empty (section 7).

Removed, with destinations: `ModelData::model_function` -> `anira_ext_entry` through `set_model_ext` (section 1b); `ModelData::is_binary` -> `anira_bytes_ownership`, stated instead of implied; `ModelData::operator=` dropping `m_model_function` on assignment (`InferenceConfig.h:125-139`) cannot recur, because a handle is copied whole by anira and never by the caller; `max_inference_time` -> `Hard::inference_budget` (`anira_contract_hard_set_budget`, `ANIRA_BUDGET_EXPLICIT` on the upgrade path); `warm_up` -> `Hard::warmup` (`anira_contract_hard_set_warmup`, `ANIRA_WARMUP_FIXED` on the upgrade path); `blocking_ratio` -> `Hard::wait_ratio` (`anira_contract_hard_set_wait_ratio`) plus the `_wait` twins of section 6 -- the `ANIRA_NONBLOCKING` `process` never waits; `HostConfig` block size / rate (both `float` today, `HostConfig.h:96-99`) -> `Hard` geometry (`anira_contract_create_hard`, `uint32_t` and `double`); `HostConfig::allow_smaller_buffers` -> the block range: the flag clear is `block_min == block_max`, the flag set is `block_min` 1 and `block_max` the v2 buffer size, and a host that knows its real range states it and pays less than v2's sweep, which always walked down to 1; `HostConfig::tensor_index` / `tensor_is_input` / `k_first_streamable` -> `set_anchor` / `ANIRA_ANCHOR_FIRST_STREAMED`; `num_parallel_processors` -> `max_instances`; `session_exclusive_processor` -> `state`; `TensorShapeList` + `ProcessingSpec` -> `TensorSpec` axes and streaming fields, with `ProcessingSpec::internal_model_latency` -> per-output `latency` (`anira_tensor_spec_set_latency`); per-backend `TensorShape` overrides -> `set_tensor_name` per model plus one canonical spec list; `InferenceConfig::Defaults` -> the defaults of the setters, one per setter, with no mutable static (`Defaults::m_num_parallel_processors` exists only to dodge `STB_GNU_UNIQUE`, `InferenceConfig.h:456-459`); `JsonConfigLoader::get_inference_config()` -> `from_json`/`from_json_file` (section 8), with `anira::v2::JsonConfigLoader` in `compat/v2.hpp` for one minor (section 10).

---

## 6. Handler API surface

The handler surface is four C headers -- `abi/machine.h`, `abi/thread.h`, `abi/handler.h` (M2) and `abi/ticket.h` (M3) -- plus the job-options setters of `abi/config.h`; `anira.hpp` is the C++ spelling over them and adds nothing the C header cannot do. `abi/machine.h` is section 4's block; the handler surface starts here, at the inference threads a host drives itself, which exist only when the machine was configured with `num_threads == 0` (section 4) and which are the Wasm build's only inference threads (section 6b). Export macro, error model, ownership, enumeration convention and the thread-tag vocabulary are stated once in section 6a and assumed here; in the blocks below `[tag]` is the thread tag, `[cs]` = `[callback-safe]`, `NB` = `ANIRA_NONBLOCKING` after the declarator, and the `ANIRA_API`/`ANIRA_CALL` pair is elided for readability and present in the real header.

```c
/* ================= abi/enums.h — values pinned, independent of build options ============== */
typedef enum anira_ticket_status { ANIRA_TICKET_PENDING = 0, ANIRA_TICKET_MET = 1, ANIRA_TICKET_LATE = 2, ANIRA_TICKET_DROPPED = 3,
                                   ANIRA_TICKET_FAILED = 4, ANIRA_TICKET_STATUS_FORCE32 = 0x7fffffff } anira_ticket_status;
typedef enum anira_pad_policy   { ANIRA_PAD_REJECT = 0, ANIRA_PAD_ZEROS = 1, ANIRA_PAD_POLICY_FORCE32 = 0x7fffffff } anira_pad_policy;

/* ================= abi/thread.h — user-driven inference threads; the Wasm Worker's primitive ===== */
typedef struct anira_inference_thread anira_inference_thread;
anira_status anira_inference_thread_create(anira_machine*, anira_inference_thread** out, anira_error* err);   /* [main-thread]; bound to the core's queue, outlives the machine; Wasm: main instance only */
void         anira_inference_thread_run_loop(anira_inference_thread*);    /* [inference-thread]; blocks until stop; sets has_exited on return */
anira_bool   anira_inference_thread_execute(anira_inference_thread*);     /* [inference-thread]; dequeue + dispatch, allocation-free; the engine call and host callbacks may allocate and block */
void         anira_inference_thread_start(anira_inference_thread*);       /* [main-thread]; native: spawns thl::core::Thread at the machine's inference-thread scheduling class (RealTime by default) */
void         anira_inference_thread_stop(anira_inference_thread*);        /* [main-thread]; native: joins; Wasm: requests stop only */
anira_bool   anira_inference_thread_has_exited(const anira_inference_thread*) NB;    /* [thread-safe]; true once run_loop returned (shared-memory atomic); required before destroy on Wasm */
anira_bool   anira_inference_thread_should_exit(const anira_inference_thread*) NB;   /* [thread-safe] */
anira_bool   anira_inference_thread_is_running(const anira_inference_thread*) NB;    /* [thread-safe] */
void         anira_inference_thread_destroy(anira_inference_thread*);     /* [main-thread]; must be stopped and, on Wasm, has_exited */
uint32_t     anira_num_inference_threads(void);                           /* [thread-safe]; the size of the default pool in this copy */

/* ================= abi/handler.h ========================================================= */
typedef struct anira_pipeline anira_pipeline;  typedef struct anira_handler anira_handler;  typedef struct anira_plan_report anira_plan_report;
anira_status anira_pipeline_create(anira_pipeline** out, anira_error* err);
anira_status anira_pipeline_add_stage(anira_pipeline*, const anira_stage_desc*, anira_error* err);            /* copied into a refcounted carrier */
anira_status anira_pipeline_add_inference(anira_pipeline*, const anira_model_config* const* variants, uint32_t num_variants,
                                          const anira_backend_id* candidates, uint32_t num_candidates, anira_error* err);   /* NULL/0 candidates = every engine the variants name, on ANIRA_PROVIDER_DEFAULT */
anira_status anira_pipeline_register_engine(anira_pipeline*, const char* engine_id, const anira_backend_desc*, anira_error* err);   /* engine_id reverse-URI; twice in one pipeline is ANIRA_ERROR_CONFIG; carrier */
void         anira_pipeline_destroy(anira_pipeline*);
anira_status anira_handler_create(anira_machine*, const anira_pipeline*, anira_handler** out, anira_error* err);   /* [main-thread]; loads every candidate's model whose prepare != NULL; add-refs the machine; copies everything */
void         anira_handler_destroy(anira_handler*);           /* [main-thread]; releases the session, blocks for quiescence like v2's dtor; releases outstanding user tensors and frees pool tensors on this thread */
anira_status anira_handler_prepare(anira_handler*, const anira_contract*, anira_error* err);   /* [main-thread & !processing]; blocking quiescence point; one plan per surviving (variant, candidate) pair, dense-indexed; calls every stage prepare(h, report, ud) */
const anira_plan_report* anira_handler_plan_report(const anira_handler*);   /* [main-thread & prepared]; valid until next prepare/destroy */
typedef struct anira_plan_slot { uint32_t struct_size; uint32_t slot; uint32_t is_input; uint32_t domain_in; uint32_t domain_out; uint32_t edge_class;
                                 uint32_t allocate_class; uint32_t wait_strategy; const char* recipe; const char* reason; } anira_plan_slot;
typedef struct anira_plan_ext  { uint32_t struct_size; uint32_t index; const char* host; const char* kind; const char* consumer; } anira_plan_ext;   /* "entry -> LibTorchAdapter" */
typedef struct anira_plan_info { uint32_t struct_size; uint32_t variant; uint32_t engine; uint32_t provider; const char* engine_id; double budget_ms; } anira_plan_info;
                                 /* engine_id: NULL for a built-in engine, the registered name for a custom one; budget_ms is the measured budget of that
                                    one plan, and the contract's Hard promise is the worst case across every plan (section 7) */
/* A plan is addressed by a dense index 0..num_plans-1, never by a composite key: a dense index cannot name a plan
   that does not exist, enumerating the set is one loop, a later dimension (batch, device) appends a field to the row
   instead of moving every signature, and the index the report hands out is the same index selection takes. */
uint32_t     anira_plan_report_num_plans(const anira_plan_report*);
anira_status anira_plan_report_plans(const anira_plan_report*, uint32_t element_size, uint32_t* count, anira_plan_info* out);
anira_status anira_plan_report_slots(const anira_plan_report*, uint32_t plan, anira_bool inputs, uint32_t element_size, uint32_t* count, anira_plan_slot* out);
anira_status anira_plan_report_exts (const anira_plan_report*, uint32_t plan, uint32_t element_size, uint32_t* count, anira_plan_ext* out);
anira_status anira_plan_report_to_json(const anira_plan_report*, char* buf, size_t cap, size_t* out_len);
/* ---- runtime selection (both contracts): one relaxed atomic store, never planning ---- */
anira_status anira_handler_set_plan(anira_handler*, uint32_t plan) NB;   /* [thread-safe]; the dense index anira_plan_report_plans hands out; relaxed store, InferenceManager.cpp:36-38; effective at the next chunk under Hard, the next job under Async; stores the request, and a Stateful switch clears stream state on the driving thread at the next Hard entry / job boundary */
uint32_t     anira_handler_get_plan(const anira_handler*) NB;            /* [thread-safe] */
/* ---- Hard entries: v2-identical semantics minus the wait (InferenceHandler.h:177-338,452); [driver-thread] NB unless noted; host owns the memory ---- */
size_t       anira_handler_process(anira_handler*, float* const* data, size_t num_samples, uint32_t tensor_index) NB;
size_t       anira_handler_process_separate(anira_handler*, const float* const* in, size_t num_in, float* const* out, size_t num_out, uint32_t tensor_index) NB;
anira_status anira_handler_process_multi(anira_handler*, const float* const* const* in, const size_t* num_in, float* const* const* out, size_t* num_out /* in: capacity, out: written */) NB;
anira_status anira_handler_push_data(anira_handler*, const float* const* in, size_t num_in, uint32_t tensor_index) NB;
anira_status anira_handler_push_data_multi(anira_handler*, const float* const* const* in, const size_t* num_in) NB;
size_t       anira_handler_pop_data(anira_handler*, float* const* out, size_t num_out, uint32_t tensor_index) NB;
anira_status anira_handler_pop_data_multi(anira_handler*, float* const* const* out, size_t* num_out) NB;
/* The typed twins: void* const* where the entries above take float* const*. The element type of each slot is the ring dtype
   declared on the Hard contract (anira_contract_hard_set_ring_dtype, F32 when unset): the ring beneath the slot holds it, the
   call is checked against it and never converts (a disagreement is 0 / ANIRA_ERROR_CONFIG in rt_error). The float entries above
   are forwarders onto these and are legal on F32 slots only. Same [driver-thread] tag and ANIRA_NONBLOCKING class. */
size_t       anira_handler_process_typed(anira_handler*, void* const* data, size_t num_samples, uint32_t tensor_index) NB;
size_t       anira_handler_process_separate_typed(anira_handler*, const void* const* in, size_t num_in, void* const* out, size_t num_out, uint32_t tensor_index) NB;
anira_status anira_handler_process_multi_typed(anira_handler*, const void* const* const* in, const size_t* num_in, void* const* const* out, size_t* num_out) NB;
anira_status anira_handler_push_data_typed(anira_handler*, const void* const* in, size_t num_in, uint32_t tensor_index) NB;
anira_status anira_handler_push_data_multi_typed(anira_handler*, const void* const* const* in, const size_t* num_in) NB;
size_t       anira_handler_pop_data_typed(anira_handler*, void* const* out, size_t num_out, uint32_t tensor_index) NB;
anira_status anira_handler_pop_data_multi_typed(anira_handler*, void* const* const* out, size_t* num_out) NB;
uint32_t     anira_handler_get_latency(const anira_handler*, uint32_t tensor_index) NB;      /* [thread-safe]; valid from prepare on; v2 arithmetic incl. the wait_ratio credit; 0 for Static outputs, index-aligned (InferenceHandler.h:357-360) */
anira_status anira_handler_get_latencies(const anira_handler*, uint32_t* count, uint32_t* out) NB;   /* [thread-safe]; valid from prepare on; replaces the thread_local vector export */
size_t       anira_handler_get_available_samples(anira_handler*, uint32_t tensor_index, uint32_t channel) NB;   /* [driver-thread]; runs collect_completed and therefore post_process (InferenceManager.cpp:267-276) */
anira_status anira_handler_set_input (anira_handler*, uint32_t slot, uint32_t element, float value) NB;   /* Static tensors; materialised into model_inputs before pre_process; float scalar -- a non-F32 slot is REFUSED (ANIRA_ERROR_CONFIG in rt_error), never converted */
float        anira_handler_get_output(const anira_handler*, uint32_t slot, uint32_t element) NB;          /* float scalar -- a non-F32 slot is REFUSED the same way and returns 0.f; per-element, no snapshot */
anira_status anira_handler_set_static_input(anira_handler*, uint32_t slot, const anira_tensor*) NB;    /* whole Static tensor, HOST, copied */
anira_status anira_handler_get_static_output(const anira_handler*, uint32_t slot, const anira_tensor* dst) NB;   /* whole Static tensor, HOST, copied into the caller's dst under one latch, so a classifier vector cannot mix two inferences; the typed read twin of set_static_input; dst's dtype and element count must equal the slot's, else ANIRA_ERROR_CONFIG */
void         anira_handler_reset(anira_handler*) NB;           /* [driver-thread]; wait-free; in-flight results discarded; model state untouched */
anira_status anira_handler_rt_error(const anira_handler*) NB;  /* [thread-safe] [cs]; last RT contract violation, relaxed atomic; ANIRA_OK if none */
/* ---- _wait twins: v2's blocking_ratio / set_non_realtime waits; [any-thread, blocking], NOT NB; timeout_ms >= 0 explicit, ANIRA_WAIT_CONTRACT, ANIRA_WAIT_FOREVER;
        semaphore wait when wait_ratio > 0, 1 ms atomic poll otherwise (Context.cpp:948-956); 0 / ANIRA_ERROR_INVALID_STATE at once without an active inference thread; Wasm: spins ---- */
size_t       anira_handler_process_wait(anira_handler*, float* const* data, size_t num_samples, double timeout_ms, uint32_t tensor_index);
size_t       anira_handler_process_separate_wait(anira_handler*, const float* const* in, size_t num_in, float* const* out, size_t num_out, double timeout_ms, uint32_t tensor_index);
anira_status anira_handler_process_multi_wait(anira_handler*, const float* const* const* in, const size_t* num_in, float* const* const* out, size_t* num_out, double timeout_ms);
size_t       anira_handler_pop_data_wait(anira_handler*, float* const* out, size_t num_out, double timeout_ms, uint32_t tensor_index);   /* InferenceHandler.h:303-306 */
anira_status anira_handler_pop_data_multi_wait(anira_handler*, float* const* const* out, size_t* num_out, double timeout_ms);
size_t       anira_handler_process_wait_typed(anira_handler*, void* const* data, size_t num_samples, double timeout_ms, uint32_t tensor_index);
size_t       anira_handler_process_separate_wait_typed(anira_handler*, const void* const* in, size_t num_in, void* const* out, size_t num_out, double timeout_ms, uint32_t tensor_index);
anira_status anira_handler_process_multi_wait_typed(anira_handler*, const void* const* const* in, const size_t* num_in, void* const* const* out, size_t* num_out, double timeout_ms);
size_t       anira_handler_pop_data_wait_typed(anira_handler*, void* const* out, size_t num_out, double timeout_ms, uint32_t tensor_index);
anira_status anira_handler_pop_data_multi_wait_typed(anira_handler*, void* const* const* out, size_t* num_out, double timeout_ms);

/* ================= abi/config.h — job options: the frame-invariant part of a job; the per-job deadline is a submit parameter;
                     the completion typedef and its setter land with abi/ticket.h (M3), the frame-invariant setters are M1 ======= */
typedef struct anira_job_options anira_job_options;
typedef void (ANIRA_CALL* anira_job_complete_fn)(anira_handler* h, anira_ticket t, anira_ticket_status s, void* user_data) NB;   /* [inference-thread] (IMMEDIATE) or the caller of poll/ticket_wait (POLLED); may call [cs] entries incl. ticket_release; never submit */
anira_status anira_job_options_create(anira_job_options** out, anira_error* err);              /* [main-thread]; build once, reuse across submits; never mutate concurrently with a submit that reads it */
anira_status anira_job_options_set_head_trim(anira_job_options*, uint32_t count, const int64_t* trims);   /* -1 = per-output latency (input-aligned) */
anira_status anira_job_options_set_tail_flush(anira_job_options*, anira_bool);                 /* ViewChunker reassembly semantics; default true */
anira_status anira_job_options_set_below_min(anira_job_options*, anira_pad_policy);            /* default ANIRA_PAD_REJECT */
anira_status anira_job_options_set_on_complete(anira_job_options*, anira_job_complete_fn, void* user_data);
anira_status anira_job_options_set_ext(anira_job_options*, const anira_ext_header*);            /* borrowed until submit returns; ext "crop_affine", "job_backend", "job_model" when their consumers arrive */
anira_status anira_job_options_set_ext_json(anira_job_options*, const char* kind, const char* utf8, size_t len);   /* the JSON twin every config handle carries, section 1b */
void         anira_job_options_destroy(anira_job_options*);

/* ================= abi/ticket.h — Async entries (M3) ====================================== */
typedef uint32_t anira_ticket;               /* slot (low 16) | generation (high 16); value type */
#define ANIRA_TICKET_INVALID 0u                  /* with the other pinned sentinels in abi/enums.h, so abi/stage.h can name it at M2 */
anira_status anira_handler_allocate_input (anira_handler*, uint32_t slot, anira_domain, anira_tensor* out, anira_error* err);   /* [main-thread & prepared]; anira-owned pool tensor; anira never calls its release */
anira_status anira_handler_allocate_output(anira_handler*, uint32_t slot, anira_domain, anira_tensor* out, anira_error* err);
anira_status anira_handler_free_tensor(anira_handler*, anira_tensor*);                          /* [main-thread & prepared]; returns an allocate_* tensor to the pool; prepare/destroy free the rest */
anira_status anira_handler_bind_output(anira_handler*, uint32_t slot, anira_tensor*, anira_error* err);   /* [main-thread & prepared]; user-owned destination; acquire = writable-when; owning tokens move */
anira_status anira_handler_submit(anira_handler*, anira_tensor* inputs, uint32_t num_inputs, const anira_job_options* opts_or_null, double deadline_ms, void* job_user_data, anira_ticket* out) NB;
        /* [thread-safe]; owning acquire tokens move into the slot (caller's kind = NONE); deadline_ms absolute on anira_now_ms(), < 0 = the contract's; job_user_data stored by value, never dereferenced, read back with ticket_user_data; ANIRA_ERROR_CAPACITY when no slot is free; ANIRA_ERROR_CONFIG when a tensor's dtype or axis-tag sequence disagrees with its spec, recorded in anira_handler_rt_error; *out is ANIRA_TICKET_INVALID on every failure; never blocks, never allocates */
uint32_t            anira_handler_poll(anira_handler*);                                          /* [main-thread]; drains Polled completions (on_complete, tensor release) on the calling thread; returns count */
anira_ticket_status anira_handler_ticket_status(const anira_handler*, anira_ticket) NB;          /* [thread-safe] [cs]; PENDING | MET | LATE | DROPPED | FAILED */
anira_ticket_status anira_handler_ticket_wait(anira_handler*, anira_ticket, double timeout_ms);  /* [any-thread, blocking]; any timeout_ms < 0 waits forever -- ANIRA_WAIT_FOREVER, and ANIRA_WAIT_CONTRACT, which has no block cadence to scale under Async; a stale or released ticket returns ANIRA_TICKET_FAILED; Wasm: spins */
anira_status anira_handler_ticket_input_released(const anira_handler*, anira_ticket, uint32_t i, anira_sync_token* out) NB;   /* [thread-safe] [cs]; non-owning view valid until ticket_release; dup with anira_sync_token_dup */
anira_status anira_handler_ticket_output_ready (const anira_handler*, anira_ticket, uint32_t i, anira_sync_token* out) NB;
anira_status anira_handler_ticket_error(const anira_handler*, anira_ticket, anira_error* out) NB;   /* [thread-safe] [cs]; FAILED reason, e.g. the kind name of an unconsumed job extension */
anira_status anira_handler_ticket_user_data(const anira_handler*, anira_ticket, void** out) NB;   /* [thread-safe] [cs]; the job_user_data of the submit that minted this ticket; ANIRA_ERROR_TICKET_STALE for a recycled slot */
void         anira_handler_ticket_release(anira_handler*, anira_ticket) NB;                      /* [thread-safe] [cs]; marks the slot for recycling (deferred until terminal; fds closed by the inference thread); legal from on_complete */
```

The same surface in `anira.hpp` (header-only, C++20, `namespace anira`, not ABI-stable; nothing in it is exported from libanira). `Hard`, `Async` and `Contract` are section 3's aggregates, `Machine` and `MachineConfig` are section 4's; what follows is the handler half and the types a stage or a custom backend subclasses:

```cpp
struct Error : std::runtime_error { anira_status status; };          // thrown by control paths on ANIRA_FAILED unless ANIRA_CXX_NO_EXCEPTIONS
template <class T> struct Result { T value; anira_error error; bool ok() const { return ANIRA_SUCCEEDED(error.status); } };   // returned instead, expected-shaped
struct JobOptions { std::vector<int64_t> head_trim; bool tail_flush = true; anira_pad_policy below_min = ANIRA_PAD_REJECT;
                    std::function<void(Ticket, anira_ticket_status)> on_complete; };   // frame-invariant; the deadline and the per-job void* are submit arguments
class Capabilities { std::vector<BackendId> backends() const; std::vector<Domain> domains() const; std::vector<std::string> ext_kinds() const;
                     std::vector<anira_edge_info> edges() const; anira_edge_info edge(Domain, const BackendId&) const; };
class Stage { virtual anira_status pre_process(StageContext&) noexcept ANIRA_NONBLOCKING { return anira_stage_default_pre_process(...); }
              // and post_process, before_inference, after_inference, prepare(InferenceHandler&, const PlanReport&), release(), consumed_kinds()
};                                                                     // the defaults forward to anira_stage_default_*: "call super" is the base class
class StageContext { anira_stage_phase phase() const; Engine engine() const; Provider provider() const; uint32_t variant() const; anira_ticket ticket() const; RingView input_ring(uint32_t);
                     Tensor& model_input(uint32_t); Tensor& model_output(uint32_t); RingView output_ring(uint32_t); };   // over the anira_stage_ctx POD
class RingView { DType dtype() const; uint32_t num_channels() const; size_t available(uint32_t) const; size_t available_past(uint32_t) const;   // T's anira_dtype must equal dtype(): a mismatch is 0 + ANIRA_ERROR_CONFIG, never a conversion
                 template <class T> size_t pop_block(uint32_t, T*, size_t);  template <class T> size_t peek_past_block(uint32_t, T*, size_t) const;
                 template <class T> size_t push_block(uint32_t, const T*, size_t);  template <class T> size_t push_fill(uint32_t, const T&, size_t);
                 size_t discard(uint32_t, size_t);                            // no dtype: discard moves the read position only
                 template <class T> size_t pop_windows(uint32_t, T*, size_t, size_t, size_t, uint32_t); };  // over anira_ring_*, T -> anira_dtype at compile time
class BackendImpl { /* virtual prepare/process/release; consumed_kinds() -- over anira_backend_desc */ };
namespace stage { struct Inference { Inference(const ModelConfig&, std::initializer_list<BackendId> = {}); Inference(std::initializer_list<std::reference_wrapper<const ModelConfig>>, std::initializer_list<BackendId>); };
                  struct Custom { Custom(std::shared_ptr<Stage>, Domain in = ANIRA_DOMAIN_HOST, Domain out = ANIRA_DOMAIN_HOST); template <class F> Custom(F pre_process, Domain, Domain); };   // control block in user_data, deleted in release
                  struct CustomBackend { explicit CustomBackend(std::shared_ptr<BackendImpl>); }; }
class Pipeline { Pipeline(std::initializer_list<std::variant<stage::Inference, stage::Custom, stage::CustomBackend>>); Pipeline& inference(const ModelConfig&, std::initializer_list<BackendId> = {}); ... };
                                                                       // pre/post stages around exactly ONE Inference stage
class PlanReport { uint32_t num_plans() const; std::vector<anira_plan_info> plans() const;   // the budget is the row field anira_plan_info::budget_ms
                   std::vector<anira_plan_slot> slots(uint32_t plan, bool inputs) const; std::vector<anira_plan_ext> extensions(uint32_t plan) const;
                   std::string to_json() const; };   // plan 0 is the only plan of a single-variant, single-candidate pipeline
class Ticket { anira_ticket_status status() const noexcept ANIRA_NONBLOCKING; anira_ticket_status wait(); anira_ticket_status wait_for(std::chrono::nanoseconds);
               SyncToken input_released(uint32_t) const; SyncToken output_ready(uint32_t) const; anira_error error() const;
               void* user_data() const noexcept ANIRA_NONBLOCKING;   // anira_handler_ticket_user_data: the per-job void* this ticket was submitted with
               ~Ticket(); /* releases the slot */ };
class InferenceHandler {
    InferenceHandler(Machine&, const Pipeline&);
    PlanReport prepare(const Contract&);  PlanReport prepare(const Hard&);  PlanReport prepare(const Async&);   // throws Error / Result<PlanReport>
    const PlanReport& plan_report() const;
    // Hard entries: noexcept ANIRA_NONBLOCKING inline forwarders, v2 signatures; never wait
    size_t process(float* const* data, size_t n, uint32_t tensor_index = 0) noexcept ANIRA_NONBLOCKING;
    size_t process(const float* const* in, size_t n_in, float* const* out, size_t n_out, uint32_t tensor_index = 0) noexcept ANIRA_NONBLOCKING;
    anira_status process(const float* const* const* in, const size_t* n_in, float* const* const* out, size_t* n_out) noexcept ANIRA_NONBLOCKING;
    anira_status push_data(const float* const* in, size_t n, uint32_t tensor_index = 0) noexcept ANIRA_NONBLOCKING;  anira_status push_data(const float* const* const*, const size_t*) noexcept ANIRA_NONBLOCKING;
    size_t pop_data(float* const* out, size_t n, uint32_t tensor_index = 0) noexcept ANIRA_NONBLOCKING;  anira_status pop_data(float* const* const*, size_t*) noexcept ANIRA_NONBLOCKING;
    // _wait twins: noexcept, blocking; std::chrono::nanoseconds timeout or the ANIRA_WAIT_* sentinels
    size_t process_wait(float* const* data, size_t n, double timeout_ms, uint32_t tensor_index = 0) noexcept;   // and the separate/multi forms
    size_t pop_data_wait(float* const* out, size_t n, double timeout_ms, uint32_t tensor_index = 0) noexcept;  anira_status pop_data_wait(float* const* const*, size_t*, double timeout_ms) noexcept;
    uint32_t get_latency(uint32_t tensor_index = 0) const noexcept;  std::vector<uint32_t> get_latency_vector() const;  size_t get_available_samples(uint32_t, uint32_t = 0) noexcept;   // non-const: runs post_process
    void set_input(float, uint32_t slot, uint32_t element) noexcept ANIRA_NONBLOCKING;  float get_output(uint32_t slot, uint32_t element) const noexcept ANIRA_NONBLOCKING;
    void set_static_input(uint32_t slot, const Tensor&) noexcept ANIRA_NONBLOCKING;  void get_static_output(uint32_t slot, Tensor& dst) const noexcept ANIRA_NONBLOCKING;   // the typed Static pair; the two above are the float scalars
    void reset() noexcept ANIRA_NONBLOCKING;  anira_status rt_error() const noexcept;
    void set_plan(uint32_t plan) noexcept ANIRA_NONBLOCKING;  uint32_t get_plan() const noexcept ANIRA_NONBLOCKING;   // the whole C selection surface: one relaxed store
    void set_model(uint32_t variant) noexcept ANIRA_NONBLOCKING;  void set_backend(Engine, Provider = ANIRA_PROVIDER_DEFAULT) noexcept ANIRA_NONBLOCKING;   // conveniences over set_plan, O(1) through the (variant, engine, provider) -> plan table built at prepare
    // Async entries
    Tensor allocate_input(uint32_t slot, Domain);  Tensor allocate_output(uint32_t slot, Domain);  void free_tensor(Tensor&);  void bind_output(uint32_t slot, Tensor&);
    Ticket submit(std::span<Tensor> inputs, const JobOptions& = {}, std::optional<std::chrono::steady_clock::time_point> deadline = {}, void* job_user_data = nullptr) noexcept(false);   // tokens move; ANIRA_ERROR_CAPACITY -> Error/Result; the C call is non-blocking; the pointer comes back as Ticket::user_data()
    uint32_t poll();
    static uint32_t get_num_inference_threads() noexcept;  size_t drain_log();   // v2 survivors
    anira_handler* native() const; };
class InferenceThread { explicit InferenceThread(Machine&); void start(); void stop(); bool execute(); void run_loop();
                        bool should_exit() const noexcept; bool is_running() const noexcept; bool has_exited() const noexcept; };
anira_status shutdown() noexcept;  bool release_core_if_idle() noexcept;  bool has_core() noexcept;  size_t drain_log();
inline void log_rt(anira_log_level, const char* group, const char* static_message, int32_t = 0, int32_t = 0) noexcept ANIRA_NONBLOCKING;
```

`prepare` takes one object, the contract, and the edge-cost policy travels inside it: `anira_contract_set_edge_cost(c, ANIRA_EDGE_COST_PERMISSIVE | STRICT)` is plan validation, not scheduling, and it rides on the contract handle because the contract file already carries `edge_cost` (section 8) and a second `prepare` parameter would exist only to mirror a separate policy object; `anira.hpp` keeps it as the `edge_cost` member of `Hard` and `Async`, so `handler.prepare(Hard{ .block_min = 512, .block_max = 512, .rate = 48000 })` compiles and validates every candidate's plan and returns the `PlanReport` (section 7). `prepare` is the blocking quiescence point, never from the driver thread; a `prepare` while processing is a contract error. The report is handler-owned, valid until the next `prepare` or `destroy`, and walked by stride-explicit enumerators (`anira_plan_report_plans`, `anira_plan_report_slots`, `anira_plan_report_exts`); `anira.hpp` copies the rows into vectors.

The Hard entries are one-line `noexcept` forwarders to the v2 code paths (`InferenceManager.cpp:53-134`), carry a truthful `ANIRA_NONBLOCKING`, and none of them waits. v2's in-callback wait -- `blocking_ratio > 0` inside `process()` (`InferenceManager.cpp:59-73`), and every Hard entry under `set_non_realtime` (`Context.cpp:948-956`) -- lives in the `_wait` twins, which are not `ANIRA_NONBLOCKING`: `timeout_ms >= 0` is explicit, `ANIRA_WAIT_CONTRACT` (-2.0) waits `wait_ratio × block duration` and is v2's `blocking_ratio` behaviour, `ANIRA_WAIT_FOREVER` (-1.0) is v2's `set_non_realtime` behaviour; with `wait_ratio > 0` they wait on the completion semaphore, with `wait_ratio == 0` they poll the done atomic with 1 ms sleeps, and they return 0 / `ANIRA_ERROR_INVALID_STATE` at once when no inference thread is active. They are `[any-thread, blocking]`: legal from the driver thread only if the host accepts a wait there. `anira_handler_set_non_realtime` is not a C entry: `compat/v2.hpp` implements `v2::InferenceHandler::set_non_realtime(bool)` as a wrapper-side flag that routes the v2 overloads to the `_wait` twins with `ANIRA_WAIT_FOREVER`, and to `ANIRA_WAIT_CONTRACT` when the legacy contract carries `blocking_ratio > 0` -- v2-identical from the compat layer, including the latency figure (`get_latency` reports the v2 arithmetic with the wait credit; a host that calls the NB `process` on a `wait_ratio > 0` handler gets the same latency figure and more `on_miss` events). On Wasm every `_wait` twin spins; calling one inside the worklet is the host's decision, as v2's `blocking_ratio` was.

Three forced changes against today's signatures: the multi variants write counts into the caller's in/out `num_out` instead of returning a handler-owned `size_t*` (`InferenceHandler.h:229-232`, which leaked libc++ layout into the web wrapper); `tensor_index` is `uint32_t`; `anira_handler_get_available_samples` takes a non-const handler and is `[driver-thread]` because it forwards to `InferenceManager::get_available_samples`, which runs `collect_completed` and therefore the stage's `post_process` (`InferenceManager.cpp:267-276`). The latency vector is index-aligned with the output list and reports 0 for a Static output (`InferenceHandler.h:357-360`): a C array indexed by slot is the only representation that cannot misalign. Static tensors are materialised by anira, not by a stage: values set through `anira_handler_set_input`/`set_static_input` are copied into `model_inputs` before any stage's `pre_process` runs, and Static outputs are copied out of `model_outputs` after `post_process`, independent of the stage -- the store is the handler's, and `anira_stage_ctx` is a frozen 64-byte POD without a handler pointer, so no stage can reach it otherwise (v2 did this inside the default `PrePostProcessor`, `src/PrePostProcessor.cpp:13-31, 60-71, 85-93`). `anira_handler_set_input`/`get_output` are float scalars and refuse rather than convert: on a non-F32 Static slot they record `ANIRA_ERROR_CONFIG` in `anira_handler_rt_error`, and `get_output` additionally returns 0.f. Refusing is only honest with a typed read beside it, so v3.0.0 adds `anira_handler_get_static_output`, the whole-tensor read twin of `set_static_input` -- without it an `int64` class index would be configurable, runnable and unreadable. It is also the non-tearing read: the per-element getter has no snapshot, so a classifier vector read across a frame boundary can mix two inferences, while a whole-tensor read is one.

RT entries carry no `anira_error` (section 6a); `anira_handler_rt_error(h)` is where a worklet reads why `process` returned 0. Calling a Hard entry on an Async handler returns 0 / `ANIRA_ERROR_WRONG_CONTRACT`, records it in `rt_error` (last-wins, cleared by `prepare` and `reset`; `CAPACITY` is back-pressure and never lands there) and emits one latched record through the RT queue: per handler and per status kind the first occurrence after `prepare`/`reset` is logged at Error with `ANIRA_LOG_RECORD_CONTRACT_VIOLATION`, later ones increment a suppressed counter with one relaxed RMW, the drain thread reports changed counters at most every 10 s, and the re-arm logs the final count; the type-level Hard/Async coupling of section 3 is documented by `anira.hpp`'s types and enforced here at runtime. Hard's payload is host memory the caller owns -- `float` arrays through `process`/`push_data`/`pop_data`, or a Host-domain `anira_tensor` through `set_static_input`/`get_static_output` -- and every device-handle entry (`allocate_input`, `allocate_output`, `bind_output`, `submit`) is an Async entry. That, not any dtype rule, is why real-time video is Async with a deadline and `ANIRA_LATE_DROP`: a frame that stays on the GPU has no Hard entry to arrive through, and one that reaches host memory has no shape to arrive in, because the Hard entries and the rings beneath them carry `(channel, time)` scalars and a two-dimensional frame has nowhere to live in a ring -- the accumulate-to-full column of the chunker table (section 7) is the `submit` path, not this one.

Runtime selection is one store, the same for both contracts: `anira_handler_set_plan(h, plan)` takes the dense plan index `anira_plan_report_plans` handed out, is a relaxed atomic store (`InferenceManager.cpp:36-38`), `[thread-safe] ANIRA_NONBLOCKING`, atomic selection among the precompiled plan set (section 7), effective at a safe boundary -- next chunk under Hard, next job under Async -- and never triggering planning. Two independent stores on two axes could land on a pair that has no plan, on the driver thread, with no error channel; one store cannot. `set_plan` stores the request only: a Stateful variant's stream-state clear is ring and struct-pool work that must run on the session's driving thread (`Context.cpp:1180-1186`, `SessionElement.cpp:63-72`), so it executes at the next Hard entry on the driver thread, or at the next job boundary on the inference thread under Async (defined, logged); the setter itself never touches a ring. `anira.hpp` keeps `set_model(variant)` and `set_backend(engine, provider)` as conveniences resolving through a (variant, engine, provider) -> plan table built at `prepare`, so both stay O(1) and `ANIRA_NONBLOCKING`, and `compat/v2.hpp` implements v2's `set_inference_backend` over the same table.

Async entries. `anira_ticket` is a `uint32_t` value -- slot in the low 16 bits, generation in the high 16, `ANIRA_TICKET_INVALID = 0` -- never a heap object: `submit` never allocates and never blocks, `ticket_release` marks the slot for recycling, `ANIRA_ERROR_CAPACITY` arrives when no slot is free and `ANIRA_ERROR_TICKET_STALE` when a released slot is addressed again; `anira.hpp`'s `Ticket` is the RAII owner whose destructor releases. Ownership, token moves and the `[callback-safe]` list are section 6a's; `submit` and `bind_output` take `anira_tensor*` (non-const) because that is where the owning tokens move. The per-job deadline is a scalar parameter of `submit` (`double deadline_ms`, absolute on the `anira_now_ms()` clock of `abi/machine.h`, `< 0` = the contract's relative deadline) because it is a presentation timestamp set every frame from the render thread, and a render loop with a 33 ms deadline must not be the thread that touches an allocating setter; `anira_job_options` therefore holds only the frame-invariant part, is built once and reused across submits, and `anira.hpp`'s `JobOptions` mints its handle once for the same reason. The per-job `void* job_user_data` is a scalar parameter of `submit` for exactly that reason: it is the per-frame channel for a presentation timestamp, a crop rectangle or frame identity, which `anira_job_options` cannot carry because it is frame-invariant and its setters allocate on the main thread. anira stores it by value in the slot and never dereferences it; `anira_handler_ticket_user_data(h, t, &p)` `[thread-safe]` `[callback-safe]` `ANIRA_NONBLOCKING` reads it back, so a completion running on the inference thread learns which frame it just finished. `anira_job_complete_fn` is unchanged -- its `user_data` stays the options-level one, and the two never merge. A submitted tensor whose `anira_dtype` or `anira_axis_tag` sequence disagrees with its `anira_tensor_spec` is refused at `submit` with `ANIRA_ERROR_CONFIG`, recorded in `anira_handler_rt_error` and minting no ticket; conversion, if it ever arrives, is a stage. Delivery splits where the completion runs, not what it may do: `on_complete` fires on the inference thread under `ANIRA_DELIVERY_IMMEDIATE` and on the caller of `poll`/`ticket_wait` under `ANIRA_DELIVERY_POLLED`, and `ticket_release` from inside it is legal while a re-entrant `submit` is not. `request_output` (an anira-owned output pool with recycling on release) stays deferred past v1; when it arrives it is one appended function under ABI major 3.

Any handle is accepted at `submit` and `bind_output`; the plan decides what it costs. `allocate_input`/`allocate_output` exist because the fast rows of section 7 are decided at allocation time and anira can retrofit nothing: they hand back a handle in the user's own API -- a `VkBuffer` the user's shader writes as an SSBO, a GL buffer object, or for a WebGPU candidate a dma-buf-backed renderbuffer the user renders into -- allocated so that every enabled candidate gets its best row. The `PlanReport` states per slot the edge taken, its class, the class an `allocate_*` handle would have gotten, and why they differ (`anira_plan_slot.edge_class`, `.allocate_class`, `.recipe`, `.reason`). A foreign handle that works but slower is data there, never a log line; one that cannot work at all (a same-process image never exported, a `WGPUBuffer` of another device) fails `prepare()` with the recipe in `anira_error.message`. `ANIRA_EDGE_COST_STRICT` makes the first case fail too -- the harness rule that an unavailable zero-copy row is a bug, not a fallback, promoted to a library contract; development runs strict, production ships permissive.

Push/pop vs submit/poll: the same exchange addressed by stream position (quantitative, continuous, miss policy fabricates) versus ticket identity (transactional, out-of-order across lanes, never fabricates). Below the API both feed the same pump.

---

## 6a. The C ABI

The binary contract of anira v3 is a set of C11 headers, `include/anira/abi/*.h`, and the flat `anira_*` functions they declare. Everything else that ships -- `anira.hpp`, `compat/v2.hpp`, the TypeScript package -- is a convenience over that set and promises nothing on its own. This section is the contract's rulebook: what is frozen, by which number, in which file, and which gate catches a violation. Section 6b states the boundary to tanh-lib and the browser; section 11 the gates' schedule.

### Principles

Each rule is enforceable; the gate that enforces it (section 11) is named in brackets.

1. The header set `include/anira/abi/*.h` is the library's only binary contract, byte-identical for native shared, native static (plugins, iOS) and Emscripten; `anira.hpp` and the TS package are conveniences over it. [gate 4: the header set compiles as C11 on every CI leg; gate 5: a pure-C consumer links both library shapes]
2. Every entry point is a directly exported `anira_*` C function with a frozen signature; no function table, no `uintptr_t` handle, no exported C++ symbol from v3.0.0-alpha.2 on, on ELF, Mach-O, PE and Wasm alike. [gate 1: symbol baseline on ELF/Mach-O/PE; the `--emit-tsd` export diff on Wasm]
3. The Hard path is frozen: the `ANIRA_NONBLOCKING` entries `anira_handler_process/process_separate/process_multi/push_data*/pop_data*/reset` are one-line forwarders to the v2 code paths (`src/scheduler/InferenceManager.cpp:53-134`) and never wait; v2's in-callback wait (`blocking_ratio`, `set_non_realtime`) lives in the `_wait` twins, which are not `ANIRA_NONBLOCKING`. [the v2 suite runs through `anira.hpp`/`compat/v2.hpp` from alpha.2; the RTSan leg instruments only the NB entries]
4. Nothing on an `ANIRA_NONBLOCKING` path allocates, locks, waits, formats into heap memory, touches thread-local storage, returns a heap object, or invokes a callback that may; `rt_vsnprintf` into a fixed record is the one allowed RT formatting, and on Wasm `thread_local` is forbidden in any code reachable from the Worker or the worklet. [gate 6: RTSan runtime over the NB entries; `-Werror=function-effects` on the consumer-shaped test targets; the Wasm layout test asserts `__builtin_wasm_tls_size() <= 1024`]
5. RT data is fixed-layout POD with one `sizeof`/offset table on every target (`ANIRA_PTR` pins every pointer slot to 8 bytes), and that table is committed data (`abi/layout-<major>.txt`) that only an ABI-major commit may change; everything with a lifetime is an opaque handle mutated by scalar setters. [gate 3: layout executable natively and under node, diffed against the committed table]
6. No enum value, struct layout or function set depends on a build option: `USE_*`, `ENABLE_LOGGING`, `ANIRA_VERSION` become PRIVATE at alpha.2 (PUBLIC today, `CMakeLists.txt:338-356`); presence is a runtime query (`anira_enabled_backends`). [gate 4 compiles the header with no anira define at all]
7. Errors are `anira_status` integers; `status < 0` is the only stable failure test; messages land in a caller-owned `anira_error`; exceptions stop in `src/capi/`. No status objects, no thread-local last-error, no `errno`; `== ANIRA_OK` is not a stable test because a minor may add positive statuses. [gate 5; the exception firewall macro is the only way to define a C entry; `anira.hpp` and TS `Status` use `ANIRA_FAILED`]
8. Append-only inside an ABI major: new functions, enum values, extension revisions, `anira_em_*` companions and descriptor tail slots appended after the last v3.0.0 slot bump `ANIRA_ABI_MINOR`; a changed layout, signature, value, thread contract or RT class is an ABI major bump. [gate 2: abidiff bit 8 always fails; bit 4 fails unless the minor moved since the last tag; the Tier-1 abidiff fails on bit 4 regardless. From M1 the registry diff, `tools/abi/gen.py --diff-against <last tag>`, classifies every change to `abi/anira.yml` at release time: an appended function, value or tail field is a minor or pre-release change, a removed or renamed entity, a changed value, signature, thread tag or Tier-1 layout is never admitted -- advisory while the major is 0, gating from v3.0.0]
9. Every declaration carries one thread tag from the fixed vocabulary below, `[callback-safe]` as an additive property where it applies, and `ANIRA_NONBLOCKING` where the body is RT. [the generator `tools/abi/gen.py` refuses a registry entry without a thread tag; the headers are generated from `abi/anira.yml` and never hand-edited]
10. tanh-lib, concurrentqueue and nlohmann_json are PRIVATE and invisible on every platform: tanh_core's objects are absorbed into libanira, no public header includes them, logging crosses as a C callback, tanh-lib gets no C ABI. [`anira_exports` forbids `thl`, `moodycamel`, `nlohmann`; strict `anira_header_isolation`; `consumer_tanh_first` binds locally]
11. JS never hand-computes a struct offset on the control path and never writes a struct field by hand: every descriptor JS must produce has an `anira_em_*_init` or a scalar-setter twin; RT structs are read through generated `layout.ts`. [generated `enums.ts`/`layout.ts`/`exports_wasm.txt` diffed in `build_web.yml`]
12. No struct-by-value parameter or return anywhere; no 64-bit integer in an `ANIRA_NONBLOCKING` declaration (allowlist: `anira_now_ns` and the GPU tensor factories `anira_tensor_init_vulkan`, `_init_opaque_fd`, `_init_wgpu_buffer`, `_init_dmabuf`, whose parameters are vendor handles and byte offsets at their wire width -- nothing that produces a `VkBuffer` runs in JS; `tools/abi/gen.py` carries the list); timeouts and deadlines are `double` milliseconds; indices and sentinels are `uint32_t`; sample and element counts are `size_t`. [`gen.py` validation]
13. Stability is declared once, at v3.0.0 (`ANIRA_ABI_MAJOR 3`); before it `ANIRA_ABI_MAJOR` is 0 (CLAP's "0.X.Y is development", `clap/include/clap/version.h:11-13`), every gate runs, abidiff is advisory. [`anira_check_abi` demands an exact match while major is 0]
14. Unmeasured surfaces live in `include/anira/abi/draft/` with their own baselines (`abi/symbols-draft.txt`, `abi/abidiff-draft.supp`), never included by `anira.h`; promotion moves a name between baselines and never renames it. [gate 1 tolerates the draft list in both directions]
15. Every CMake change uses the existing tanh-tooling modules; what they lack is an anira-local script under `anira/cmake/` (never `cmake/tanh/`, pin-owned) until a tanh-tooling tag after v3.0.0 that anira and tanh-lib re-pin together. [`config-check.yml` drift check]

### Header set

Files installed under `include/anira/`:

| file | content | frozen at |
|---|---|---|
| `anira.h` | C11 umbrella: includes every `abi/*.h` below in dependency order, never `abi/draft/`; exists from alpha.2 -- through alpha.1 `include/anira/anira.h` stays the v2 C++ umbrella (`include/anira/anira.h:4-22`) and the M1 headers are included per file | v3.0.0 |
| `anira_all.h` | `anira.h` plus `abi/draft/*.h` | -- |
| `abi/export.h` | `ANIRA_API`, `ANIRA_CALL`, `ANIRA_NONBLOCKING`, `ANIRA_NOEXCEPT`, `ANIRA_DEPRECATED`, `ANIRA_INIT`, `ANIRA_PTR` | M1 |
| `abi/version.h`, `abi/build_info.h` | ABI constants and negotiation (`version.h` generated from the registry; `build_info.h` configured by `cmake/build-info.cmake`, which derives the ABI pair and the semver triple from the git tag -- the one source) | M1 / generated |
| `abi/status.h`, `abi/enums.h`, `abi/config.h`, `abi/log.h` | status codes and `anira_error`; every enum; config handles and descriptors; log sink | M1 |
| `abi/tensor.h`, `abi/machine.h`, `abi/thread.h`, `abi/stage.h`, `abi/handler.h` | RT PODs; machine, capabilities, lifecycle; user-driven inference threads; rings, stage and backend descriptors; pipeline, handler, Hard entries and their `_wait` twins | M2 |
| `abi/ticket.h` | Async entries | M3 |
| `abi/draft/*.h` | unmeasured platform arms (`anira_tensor_init_metal/iosurface/ahardwarebuffer/d3d12`), later `frame.h`; included by `anira_all.h` only | never (until promoted) |
| `anira_em.h` | Emscripten companion, body under `#ifdef __EMSCRIPTEN__` (section 6b) | with `anira.h` |
| `anira.hpp` | header-only C++20 wrapper, declared not ABI-stable | -- |
| `compat/v2.hpp` | `anira::v2::*` constructor shims and `anira::v2::PrePostProcessor` views (section 10); removed one minor after 3.0 | -- |

Per-file headers exist so the abidiff baseline and "which file froze when" are readable per milestone; naming the umbrella only at alpha.2 keeps the v2 suite compiling through alpha.1 without a rename. Naming: functions `anira_<object>_<verb>`, types `anira_<noun>` (`typedef struct anira_machine anira_machine;` -- handles are `anira_machine*`, never the `uintptr_t` today's web wrapper passes, `src/emscripten-wrappers/InferenceHandler.cpp:10-42`), enum constants `ANIRA_<ENUM>_<VALUE>` with `ANIRA_<ENUM>_FORCE32 = 0x7fffffff`, macros `ANIRA_*`, one `ANIRA_<DESC>_INIT` default initialiser per descriptor. A nested `include/anira/abi/.clang-tidy` (`InheritParentConfig: true`, struct and enum-constant naming relaxed) keeps `clang_tidy.yml` green without touching the pin-owned root `.clang-tidy`; the workflow's positive `SOURCES` list gains `src/capi/` and `test/abi/` and drops `src/emscripten-wrappers/` when that directory is deleted. The headers carry no `_Static_assert` (those live in `test/abi/test_layout.c`), no bitfield and no function-like macro inside a declaration beyond `ANIRA_API`/`ANIRA_CALL`/`ANIRA_NONBLOCKING`/`ANIRA_NOEXCEPT`/`ANIRA_PTR`, so a `ctypes`/`cffi` binding needs the preprocessor and the committed `abi/layout-<major>.txt`, never a code generator.

### Export macros and what a consumer links

```c
/* abi/export.h */
#if defined(__EMSCRIPTEN__)
#  define ANIRA_API __attribute__((visibility("default")))   /* + generated -sEXPORTED_FUNCTIONS (`used` unnecessary) */
#elif defined(ANIRA_STATIC)                                   /* PUBLIC on static targets, symbol-policy.cmake:110-115 */
#  define ANIRA_API                                           /* hidden inside the embedding plugin */
#elif defined(_WIN32)
#  if defined(ANIRA_BUILDING)                                 /* PRIVATE, symbol-policy.cmake:109 */
#    define ANIRA_API __declspec(dllexport)                   /* dllexport IS the allowlist on PE, symbol-policy.cmake:178-180 */
#  else
#    define ANIRA_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define ANIRA_API __attribute__((visibility("default")))    /* + version script from tanh_set_export_allowlist SYMBOL anira_* */
#else
#  define ANIRA_API
#endif
#if defined(_WIN32) && !defined(_WIN64)
#  define ANIRA_CALL __cdecl                                  /* CLAP_ABI, clap/include/clap/private/macros.h:20-26; fixed forever */
#else
#  define ANIRA_CALL
#endif
#if defined(__clang__) && defined(__has_attribute)
#  if __has_attribute(nonblocking)
#    define ANIRA_NONBLOCKING __attribute__((nonblocking))    /* verified clang 22 + emcc 4.0.23 in C11 and C++20; confirm on the clang-20 CI leg */
#  endif
#endif
#ifndef ANIRA_NONBLOCKING
#  define ANIRA_NONBLOCKING
#endif
#if defined(__cplusplus)
#  define ANIRA_NOEXCEPT noexcept
#  define ANIRA_INIT(type, ...) (type{__VA_ARGS__})
#else
#  define ANIRA_NOEXCEPT
#  define ANIRA_INIT(type, ...) ((type){__VA_ARGS__})
#endif
#define ANIRA_PTR(T, name) union { T* name; uint64_t name##_bits; }   /* 8-byte pointer slot on ILP32 and LP64; zero the struct before use */
#define ANIRA_DEPRECATED(msg) __attribute__((deprecated(msg)))       /* MSVC: __declspec(deprecated(msg)) */
```

Every declaration is `ANIRA_API <ret> ANIRA_CALL name(...)`, with `ANIRA_NONBLOCKING` after the declarator where the body is RT. `ANIRA_STATIC`/`ANIRA_BUILDING` keep coming from `tanh_apply_symbol_policy(anira EXPORT_PREFIX ANIRA)` (`CMakeLists.txt:292`); `abi/export.h` is self-contained and does not include `<tanh/core/ExportMacros.h>` as `include/anira/system/Exports.h:30` does today. `ANIRA_NONBLOCKING` is gated on `__clang__` + `__has_attribute`, never on `ANIRA_WITH_RTSAN`, and replaces the internal `ANIRA_REALTIME` (`include/anira/utils/RealtimeSanitizer.h:4-14`) so the public and the internal attribute are one definition. At alpha.2 the internal `ANIRA_API` of `system/Exports.h` is redefined to nothing on every platform: on PE `dllexport` is the allowlist, and a decorated internal class would otherwise stay in `anira-3.dll`'s export table after the version script has stripped it from ELF and Mach-O.

A static-embedded consumer links `anira::anira`, a self-contained archive with tanh_core's objects inside it (section 6b). The archive's `INTERFACE_LINK_OPTIONS` carry `--exclude-libs` for `libanira.a` and the engine archives (extending `cmake/aniraBackendHelpers.cmake:139-160`; the `libtanh_core.a` entry of `CMakeLists.txt:301-303` goes, because no such archive is linked any more), so the plugin's own `tanh_set_export_allowlist(<plugin> SYMBOL clap_entry)` is the only export decision left: `anira_*` never appears in a plugin's export table unless the plugin lists it. A shared consumer links `libanira.so.3` / `libanira.3.dylib` / `anira-3.lib`; a `dlopen` host resolves `anira_check_abi` first and everything newer through `anira_get_proc_address`. For such a host `anira.hpp` has two modes: `ANIRA_CXX_MANUAL_INIT` routes every call through `anira::detail::api()`, a struct of `PFN_anira_*` pointers filled by `anira::init_api(void* (*loader)(const char*))` or `anira::init_api_from(void* dl_handle)` and resolved lazily, so a function this library lacks throws `anira::Error{ANIRA_ERROR_NOT_SUPPORTED, "anira_x requires ABI 3.N"}`; `ANIRA_NO_PROTOTYPES` makes `anira.h` emit only the `PFN_anira_*` typedefs (Vulkan's `VK_NO_PROTOTYPES` rule). The default mode calls the C functions directly, so the driver thread pays no indirection.

### Versioning

Two integers own the promise, independent of the semver triple:

```c
/* abi/version.h (generated from abi/anira.yml by tools/abi/gen.py, like every abi/*.h; never hand-edited) */
#include "build_info.h"                  /* ANIRA_ABI_MAJOR/MINOR and ANIRA_VERSION_*, configured from git by cmake/build-info.cmake */
#define ANIRA_MAKE_ABI_VERSION(major, minor) (((uint32_t)(major) << 16) | (uint32_t)(minor))
#define ANIRA_ABI_VERSION ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR, ANIRA_ABI_MINOR)
#define ANIRA_ABI_VERSION_MAJOR(v) ((uint32_t)(v) >> 16)
#define ANIRA_ABI_VERSION_MINOR(v) ((uint32_t)(v) & 0xffffu)
/* abi/build_info.h (configured by cmake/build-info.cmake from the git tag -- one source, the tag) */
#define ANIRA_ABI_MAJOR 3                     /* 0 until v3.0.0, then the tag's major; SOVERSION; MACHO compat major; DLL suffix */
#define ANIRA_ABI_MINOR 0                     /* 0.N for the N-th v3.0.0 pre-release, X.Y from vX.Y.Z, the next minor past a tag; never moves in a patch */
#define ANIRA_VERSION_MAJOR 3
#define ANIRA_VERSION_MINOR 0
#define ANIRA_VERSION_PATCH 0
#define ANIRA_VERSION_STRING "3.0.0-alpha.2-12-gabc123"
#define ANIRA_MAKE_VERSION(maj, min, pat) (((uint32_t)(maj) << 22) | ((uint32_t)(min) << 12) | (uint32_t)(pat))   /* Vulkan packing */

uint32_t     anira_abi_version(void);                         /* [thread-safe] NB; the library's ANIRA_ABI_VERSION */
anira_status anira_check_abi(uint32_t header_abi_version);    /* [thread-safe] NB; OK iff same major and lib minor >= header minor (exact while major 0) */
uint32_t     anira_version(void);                             /* [thread-safe] NB; ANIRA_MAKE_VERSION(...) of the library */
const char*  anira_version_string(void);                      /* [thread-safe] NB; git describe, static storage */
void*        anira_get_proc_address(const char* name);        /* [thread-safe]; NULL = not in this build; feature detection for dlopen hosts */
```

`anira_check_abi` is the one negotiation call; `ANIRA_ERROR_ABI_VERSION` otherwise. `anira_version_string` replaces `anira_get_version()` of `src/emscripten-wrappers/InferenceConfig.cpp:28-30`. Static builds, the Wasm module and Windows have no soname, so a function-level negotiation exists beside it; packing `(major << 16) | minor` keeps the check one integer that TS compares without `BigInt`.

The promise inside ABI major 3, from v3.0.0: existing functions keep name, signature, semantics, thread tag and RT class; existing enum values and Tier-1 POD layouts never change; a minor may append functions, enum values (below `_FORCE32`), positive statuses, extension revisions, descriptor tail slots (after the last v3.0.0 slot) and new `anira_em_*` companions, each bumping `ANIRA_ABI_MINOR`; a patch changes nothing in the header. Deprecation is `ANIRA_DEPRECATED("use anira_x")` plus a `**Deprecated:**` CHANGELOG entry; the symbol stays exported until the next major. A major bump happens only for an RT POD layout change, a removed or renamed function, a changed enum value, or a changed thread or RT contract. Everything section 10 lists as deferred-additive fits inside major 3.

CMake: `cmake/build-info.cmake` derives `ANIRA_ABI_MAJOR`/`ANIRA_ABI_MINOR` from the git tag (`0.N` for the N-th `v3.0.0-*` pre-release reachable from HEAD, one running counter over alphas and betas; `X.Y` from `vX.Y.Z`; the next minor when the checkout is past a tag, so a development header never claims a released promise it exceeds; `0.0` without a reachable v3 tag) and writes them with the semver triple into the generated `abi/build_info.h`, which the generated `abi/version.h` includes -- the tag is the one source, nothing is bumped by hand. From alpha.2 it also sets `SOVERSION ${ANIRA_ABI_MAJOR}` (today `SOVERSION ${PROJECT_VERSION_MAJOR}`, `CMakeLists.txt:291`), `MACHO_COMPATIBILITY_VERSION ${ANIRA_ABI_MAJOR}.${ANIRA_ABI_MINOR}` and `MACHO_CURRENT_VERSION ${ANIRA_ABI_MAJOR}.${ANIRA_ABI_MINOR}.${PROJECT_VERSION_PATCH}` so dyld refuses a library older than the one a client linked, `OUTPUT_NAME anira-${ANIRA_ABI_MAJOR}` when `TANH_BINARY_FORMAT STREQUAL "PE"` (`anira-3.dll`, `anira-3.lib`; PE has no soname), and `aniraConfigVersion.cmake` `COMPATIBILITY SameMajorVersion` (today `AnyNewerVersion`, `cmake/install.cmake:375-379`). Semver major and ABI major are kept equal by policy from 3.0.0, so the Debian runtime package is `libanira3`; during the alphas that package wraps `libanira.so.0` and `find_package(anira 3.0)` accepts mutually incompatible alphas -- accepted, `anira_check_abi` is the gate that matters then. Prerelease tags: tanh-tooling v0.2.7's `tanh_git_version` strips the pre-release identifier for `project()` and `write_basic_package_version_file` itself and reports it as `TANH_VERSION_PRERELEASE` beside `TANH_VERSION_DISTANCE`; anira's `v3` branch calls it with `MATCH "v3*"`, so the v2 tags the branch also reaches never name a build, and keeps the full describe string for `ANIRA_VERSION_STRING`. It landed in the first M1 PR, before the first alpha tag. The v2 cross-session version-string compare (`src/scheduler/Context.cpp:325-348`) is dropped: two static embeddings have two cores and cannot see each other, and inside one core every session shares one header.

ABI 0 before the freeze: the header exists in full, `ANIRA_ABI_MAJOR` is 0, `ANIRA_ABI_MINOR` is the alpha/beta ordinal, `anira_check_abi` demands an exact match, gates 1, 3, 4, 5, 6 fail the build once online, gate 2 reports, the CHANGELOG prefixes header changes with `**ABI (unstable):**`, npm publishes under `next`. The freeze is the `v3.0.0` tag, from which `cmake/build-info.cmake` derives `3.0`, on the commit that commits `abi/symbols-3.txt`, `abi/layout-3.txt` and `abi/anira-3.0.<arch>.abi`.

### The three object classes

| class | mechanism | why |
|---|---|---|
| Tier-1 RT PODs: `anira_tensor`, `anira_sync_token`, `anira_memory_handle`, `anira_stage_ctx`, `anira_log_record`, `anira_error` | fixed layout, no `struct_size`, pinned `sizeof`, versioned by ABI major only; a new shape is a new struct plus new functions | copied through lock-free FIFOs, read per block, or value-initialised by `anira::Result<T>`; no size switch, no chain walk |
| config objects: `anira_machine_config`, `anira_model_config`, `anira_tensor_spec`, `anira_contract`, `anira_job_options`, `anira_pipeline` | opaque handle + scalar setters/getters (ORT `SessionOptions` shape), all `[main-thread]`, all may allocate | the layout never enters the ABI; JS builds a config with one call per field instead of reproducing libc++ vector layout (`web/src/wrappers/InferenceHandler.ts:201-212` today) |
| Tier-2 descriptors handed once to a setter or filled by an enumerator: device blocks, `anira_log_desc`, `anira_stage_desc`, `anira_backend_desc`, `anira_backend_id`, extension payloads, `anira_edge_info`, `anira_plan_slot`, `anira_plan_ext`, `anira_plan_info` | `uint32_t struct_size` first; descriptors with callbacks are `{struct_size, abi_version, user_data, ...slots}`; tail growth appends after the last v3.0.0 slot; the library reads or fills only inside `min(struct_size, sizeof(lib's))` | tail growth without `pNext`; `user_data`'s offset can never move; TS writes no field by hand |

Tier-1 layouts, identical on wasm32, LP64 and LLP64 because every pointer sits in an `ANIRA_PTR` slot and every count is `uint32_t`/`uint64_t` (verified with clang 22 and emcc 4.0.23):

| struct | `anira_struct_id` | bytes | align |
|---|---|---|---|
| `anira_tensor` | `ANIRA_STRUCT_TENSOR` | 216 | 8 |
| `anira_sync_token` | `ANIRA_STRUCT_SYNC_TOKEN` | 24 | 8 |
| `anira_memory_handle` | `ANIRA_STRUCT_MEMORY_HANDLE` | 24 | 8 |
| `anira_stage_ctx` | `ANIRA_STRUCT_STAGE_CTX` | 64 | 8 |
| `anira_log_record` | `ANIRA_STRUCT_LOG_RECORD` | 56 | 8 |
| `anira_error` | `ANIRA_STRUCT_ERROR` | 520 | 4 |

The table, with every `offsetof`, is `abi/layout-<major>.txt`, emitted by the layout executable of `test/abi/test_layout.c` and diffed by gate 3; it changes only in a commit that changes `ANIRA_ABI_MAJOR`. `anira_sizeof(anira_struct_id)` exists for allocators that cannot see the header; beside the six Tier-1 ids it answers for the enumerated Tier-2 records, `ANIRA_STRUCT_EDGE_INFO = 7`, `ANIRA_STRUCT_PLAN_SLOT = 8`, `ANIRA_STRUCT_PLAN_EXT = 9`, `ANIRA_STRUCT_PLAN_INFO = 10` and `ANIRA_STRUCT_BACKEND_ID = 11`, which is how TS gets the stride for `anira_capabilities_edges`, `anira_capabilities_backends`, `anira_enabled_backends`, `anira_plan_report_plans`, `anira_plan_report_slots` and `anira_plan_report_exts` without computing a descriptor layout by hand; ids `0x0001xxxx` are reserved for extension payloads, `0x0004xxxx` for Emscripten-only structs. `abi_version` in a descriptor is the packed `ANIRA_ABI_VERSION` the host compiled against, so anira can gate calls into an older host. Descriptors are target-layout (they hold pointers) and are handed to a setter once, never read per block; TS never writes one, it uses the setters, the `_set_ext_json` twins and the `anira_em_*_init` fills.

Enumeration convention: array enumerators take the caller's element stride explicitly -- `(uint32_t element_size, uint32_t* count, T* out)`, `element_size = sizeof(T)` in C, `anira_sizeof(id)` from TS -- and write `min(element_size, sizeof(lib's T))` bytes per element at that stride (`anira_capabilities_edges`, `anira_capabilities_backends`, `anira_enabled_backends`, `anira_plan_report_plans`, `anira_plan_report_slots`, `anira_plan_report_exts` -- a backend is an engine paired with a provider, so it enumerates as a record like the rest); scalar enumerators (`anira_capabilities_domains/ext_kinds`, `anira_registered_ext_kinds`, `anira_handler_get_latencies`) are `(uint32_t* count, T* out)` without a stride. `out == NULL` returns the count; a short buffer fills what fits and returns `ANIRA_INCOMPLETE`. The explicit stride is what makes `struct_size` growth usable for arrays without per-element pre-initialisation.

### Error model

```c
typedef enum anira_status {
    ANIRA_OK = 0, ANIRA_SUCCESS_UPGRADED = 1 /* v2 JSON auto-upgraded */, ANIRA_INCOMPLETE = 2 /* enumeration buffer short */,
    ANIRA_TIMEOUT = 3, ANIRA_PENDING = 4,
    ANIRA_ERROR_UNKNOWN = -1, ANIRA_ERROR_INVALID_ARGUMENT = -2, ANIRA_ERROR_INVALID_STATE = -3, ANIRA_ERROR_OUT_OF_MEMORY = -4,
    ANIRA_ERROR_NOT_SUPPORTED = -5 /* backend/domain/kind not in this build */, ANIRA_ERROR_NO_SUCH_FILE = -6, ANIRA_ERROR_MODEL_LOAD = -7,
    ANIRA_ERROR_ENGINE = -8, ANIRA_ERROR_CONFIG = -9 /* prepare-time legality (section 2), and the real-time refusals: a submitted tensor's dtype or axis-tag sequence, a ring or tensor accessor's dtype, and a non-F32 Static slot met by the float scalars anira_handler_set_input / get_output, all recorded in anira_handler_rt_error */, ANIRA_ERROR_EXTENSION_UNCONSUMED = -10,
    ANIRA_ERROR_EXTENSION_UNKNOWN = -11, ANIRA_ERROR_EDGE_UNREACHABLE = -12, ANIRA_ERROR_BUDGET = -13 /* Hard validation */,
    ANIRA_ERROR_CAPACITY = -14 /* RT: no free slot / ring full */, ANIRA_ERROR_TICKET_STALE = -15, ANIRA_ERROR_WRONG_CONTRACT = -16,
    ANIRA_ERROR_NOT_PREPARED = -17, ANIRA_ERROR_JSON = -18, ANIRA_ERROR_ABI_VERSION = -19, ANIRA_ERROR_BUFFER_TOO_SMALL = -20,
    ANIRA_ERROR_DEVICE = -21, ANIRA_ERROR_EXTENSION_VERSION = -22, ANIRA_ERROR_INTERNAL = -100, ANIRA_STATUS_FORCE32 = 0x7fffffff
} anira_status;
#define ANIRA_FAILED(s)    ((int32_t)(s) < 0)          /* the only stable failure test */
#define ANIRA_SUCCEEDED(s) ((int32_t)(s) >= 0)
#define ANIRA_ERROR_MESSAGE_CAPACITY 512
typedef struct anira_error { int32_t status; uint32_t reserved; char message[ANIRA_ERROR_MESSAGE_CAPACITY]; } anira_error;   /* Tier 1, 520 bytes, frozen */
#define ANIRA_ERROR_INIT ANIRA_INIT(anira_error, ANIRA_OK, 0u, {0})
```

Every fallible function returns `anira_status`; created handles come back through an `out` parameter; functions that can produce a message take `anira_error* err` (nullable) as their last parameter and fill it on failure, out-parameters untouched. Positive statuses are informational successes -- `ANIRA_SUCCESS_UPGRADED` lets TS surface the one-time v2-JSON warning, `ANIRA_INCOMPLETE` serves the enumeration convention -- and the set a function may return is not frozen: a minor may add one, which is why the stable idiom is `ANIRA_FAILED(s)` and never `s != ANIRA_OK`; the section 9 sketches, `anira.hpp` and the TS `Status` helper all use it. `anira_status_string(status)` returns static text and is tagged `[thread-safe]` only: not `ANIRA_NONBLOCKING`, and not `[callback-safe]` because that list is closed.

RT entries never take `anira_error`: they return a count or an `anira_status`, log through the RT queue as v2 does (`include/anira/utils/Logger.h:123-134`), and record the last contract violation in a per-handler relaxed atomic readable through `anira_handler_rt_error(h)` `[thread-safe] [callback-safe] ANIRA_NONBLOCKING` -- a worklet learns why `process` returned 0 without a message channel. Ticket failures live on the ticket, `anira_handler_ticket_error(h, t, err)`, not on the handler (section 1b). Exceptions never cross: `src/capi/*.cpp` wraps every control-path body in `ANIRA_CAPI_BEGIN/END` -- `std::bad_alloc` to `OUT_OF_MEMORY`, `std::invalid_argument` to `CONFIG`/`INVALID_ARGUMENT`, engine exceptions such as `throw_if_foreign_onnxruntime` (`src/backends/OnnxRuntimeProcessor.cpp:43-55`) to `ENGINE`/`MODEL_LOAD`, `...` to `INTERNAL`; RT bodies are `noexcept` with no catch, because nothing inside them throws by the v2 contract. The Wasm build keeps exception catching enabled (`-sNO_DISABLE_EXCEPTION_CATCHING`, `cmake/build-wasm.cmake:55`): Emscripten's default compiles `throw` but disables every `catch`, and the firewall is a `catch`. A caller-owned fixed buffer works identically in a static plugin, in a `dlclose`d DSO and per Wasm instance, where a thread-local buffer would be per module instance; the frozen 520 bytes let `anira::Result<T>` value-initialise it without an `_INIT` and let TS allocate it with `anira_sizeof(ANIRA_STRUCT_ERROR)` forever.

### Ownership and lifetime

- Create/destroy pairs: `anira_status anira_<x>_create(..., anira_<x>** out, anira_error*)`; `void anira_<x>_destroy(anira_<x>*)`, NULL-safe, a second destroy of the same pointer undefined. Config handles are value-like: a handler copies the pipeline and every config at `anira_handler_create` (v2 keeps `InferenceConfig&` for life, `include/anira/InferenceHandler.h:455`), so configs and pipelines may be destroyed right after the call and TS frees its builders eagerly.
- Descriptor and bytes carriers are refcounted internally: `anira_pipeline_add_stage`, `anira_pipeline_register_engine` and `add_model_bytes`/`set_model_bytes` wrap the descriptor or the `BORROW`ed bytes in a carrier that every copy (pipeline, handler, pooled processor whose lifetime is decoupled from the handler, `src/scheduler/Context.cpp:441-487`) shares; `release(user_data)` and `anira_bytes_release_fn` fire exactly once, when the last carrier reference dies, on the thread that destroys it -- the caller of `anira_pipeline_destroy`, `anira_handler_destroy`, `anira_model_config_destroy` or the pool's release, all `[main-thread]`. A pipeline destroyed without ever creating a handler releases once; two handlers from one pipeline release once, after the second is destroyed. One descriptor registered under two engine ids is two registrations and two carriers, so `release` fires once per registration and a host that reuses one descriptor is written for that.
- Strings in: UTF-8, NUL-terminated, copied by the callee (paths widened internally on Windows). Strings out: `const char*` owned by the object and valid until it is destroyed or mutated, or `(char* buf, size_t cap, size_t* out_len)` with `ANIRA_ERROR_BUFFER_TOO_SMALL`; never on RT paths.
- Model bytes: `ANIRA_BYTES_COPY` or `ANIRA_BYTES_BORROW` plus an optional `release(bytes, ctx)`; borrow is the plugin default (`tanh_add_binary_data` blobs live for the DSO). v2 borrows binary blobs silently (`include/anira/InferenceConfig.h:103-114`); v3 says so.
- Tensor: `release` and `manager_ctx` as in section 1. `release == NULL` means borrowed until the ticket is terminal (Async) or the call returns (Hard). For a user tensor with a non-NULL `release`, one submit is one release call -- exactly once per submitted copy, on an inference thread when the job reaches a terminal state, inside `anira_handler_poll`/`ticket_wait` under Polled delivery, or on the calling thread of `anira_handler_destroy`/`anira_handler_prepare` for jobs still outstanding then; never on the driver thread. Tensors from `anira_handler_allocate_input/allocate_output` are anira-owned pool tensors: anira never invokes their `release`, the producer resubmits the same descriptor over `input_released`, returns it with `anira_handler_free_tensor(h, t)` `[main-thread & prepared]`, and `prepare`/`destroy` free every pool tensor still out. A JS-produced tensor is always borrowed.
- Sync-token fds (`ANIRA_SYNC_SYNC_FILE_FD`, `ANIRA_SYNC_OPAQUE_FD_SEMAPHORE`): the token owns its fd. `anira_handler_submit` and `anira_handler_bind_output` take `anira_tensor*` (non-const) and move every owning `acquire` token into the slot -- the caller's `acquire.kind` becomes `ANIRA_SYNC_NONE` and `u.fd` becomes `-1` before the call returns, so a later `anira_sync_token_reset` on the caller's copy closes nothing. Tokens read through `ticket_input_released/output_ready` are non-owning views valid until `ticket_release` (no `dup`, so the accessors stay `ANIRA_NONBLOCKING`); a caller that needs one longer calls `anira_sync_token_dup(src, out)` `[thread-safe, !audio-thread]`. Owned fds inside a slot are closed by the inference thread when the slot is recycled (deferred close), never by `ticket_release`, which therefore stays `ANIRA_NONBLOCKING`; `anira_handler_destroy`/`prepare` close the rest on the calling thread. Other kinds are non-owning.
- Tickets are `uint32_t` values (`slot & 0xffff`, `generation << 16`; `ANIRA_TICKET_INVALID = 0`), never heap objects: `submit` never allocates, `anira_handler_ticket_release` marks the slot for recycling (a still-pending ticket's slot recycles only after the job is terminal; the value is invalid for the caller immediately), `ANIRA_ERROR_CAPACITY` after `lanes * max_in_flight + 8` unreleased tickets, `ANIRA_ERROR_TICKET_STALE` for a recycled slot. A forgotten `ticket_release` surfaces as `ANIRA_ERROR_CAPACITY`; the wrapper's RAII `Ticket` hides it.
- Machine: a refcounted handle over the immortal core (`include/anira/scheduler/Context.h:79-81`, one core per copy of anira). `anira_handler_create` adds a reference so the handle's memory outlives the user's `anira_machine_destroy`; `anira_machine_destroy` drops the user's reference, invalidates the handle for the caller regardless of the count, unregisters its log sink and waits for that sink's in-flight calls, and joins nothing. Thread pool and inference queue are core-owned: the pool exists exactly while any handler in this copy exists; user-driven `anira_inference_thread` objects bind to the core's queue and outlive machines. `num_threads == 0` keeps meaning "bring your own threads" (`ContextConfig.h:221-231`); `ANIRA_THREADS_AUTO = 0xffffffff` is the library default.
- Shutdown, C-visible: `anira_shutdown(void)` is `Context::shutdown()` (`Context.cpp:682-704`, idempotent, never creates the core), effective only when no machine handle and no handler exist in this copy; otherwise it does nothing and returns `ANIRA_ERROR_INVALID_STATE`, so a client of a shared `libanira.so.3` cannot silence another client's sessions. It is meant for static embeddings and is called from `clap_deinit`/`ExitDll` as `examples/clap-audio-plugin/anira-clap-demo-pluginentry.cpp:42-49` does today. `anira_release_core_if_idle()` never blocks (on Wasm it also requires zero loop-active threads) and `anira_has_core()` stay. The unload rule: the ELF/Mach-O destructor hook (`Context.cpp:131-146`) stays as a backstop for hosts that skipped `anira_handler_destroy`, but v3 runs host code (`anira_backend_desc.process`, `before/after_inference`, `on_complete`, tensor `release`) on pool threads, so a host that installs custom stages or backends must destroy its handlers before its DSO is unloaded -- the hook joins under the loader lock and host code that binds symbols there deadlocks. Windows keeps the non-joining `UnloadGuard`; `anira_machine_destroy` and `anira_shutdown` are `[main-thread & !loader-lock]`. User-managed `anira_inference_thread` objects must be stopped before unload; on Wasm `anira_inference_thread_destroy` additionally requires `has_exited`.

### Callbacks

`void* user_data` is the last parameter of every callback signature and the third member of every descriptor, `{struct_size, abi_version, user_data, ...}`; one slot suffices for both wrappers (the C++ wrapper stores a heap control block, the JS bridge a registry key) and its offset survives every tail growth. Every callback fires exactly once per event; re-entrant `submit` from `on_complete` is forbidden in v1. Function-pointer typedefs on RT slots (`anira_stage_fn`, `anira_job_complete_fn`) carry `ANIRA_NONBLOCKING`, so `-Wfunction-effects` flags a blocking body in the consumer's own build.

| callback | thread tag | RT class | descriptor |
|---|---|---|---|
| stage `pre_process`, `post_process` | `[driver-thread]` under Hard, `[inference-thread]` under Async | `ANIRA_NONBLOCKING` | `anira_stage_desc` |
| stage `before_inference`, `after_inference` | `[inference-thread]`, between dequeue and engine call | `ANIRA_NONBLOCKING` | `anira_stage_desc` |
| stage `prepare(anira_handler*, const anira_plan_report*, user_data)`, `release` | `[main-thread]`: the caller of `anira_handler_prepare`; `release` on the caller of the last destroy | may allocate; must not call lock-taking entries | `anira_stage_desc` |
| custom backend `process` | `[inference-thread]` | may block (engine call), must not allocate per call | `anira_backend_desc` |
| custom backend `prepare(const anira_model_config*, variant, instances, user_data)`, `release` | `[main-thread]`, under the lifecycle mutex | may allocate; must not call lock-taking entries | `anira_backend_desc` |
| `on_complete` | `[inference-thread]` under `ANIRA_DELIVERY_IMMEDIATE`; the caller of `anira_handler_poll`/`ticket_wait` under `POLLED` | `ANIRA_NONBLOCKING` | `anira_job_options_set_on_complete` |
| log sink `anira_log_fn` | any thread that logs: the caller of every `[main-thread]` entry (sync records), the caller of `anira_machine_destroy`/`anira_shutdown` (final flush), `[drain-thread]` for RT records; never the driver thread; may run with anira's lifecycle lock held | not RT, may allocate, must not call anira | `anira_log_desc` |
| tensor `release`, model-bytes `release`, descriptor `release` | `[inference-thread]` or the caller of `poll`/`ticket_wait`/`prepare`/`destroy` / `[main-thread]` / `[main-thread]` | not RT | on the tensor / carriers |

A non-OK status from `pre_process` completes that chunk as zeros at its stream position (v2 `complete_with_zeros`, `src/scheduler/SessionElement.cpp:244-248`), records the status in `rt_error` and emits one RT log record; non-OK from `post_process` zeroes that chunk's outputs; non-OK from `before/after_inference` marks the job `ANIRA_TICKET_FAILED` (Async) or zeros (Hard).

`[callback-safe]` is an additive property: from inside any callback a host may call exactly the entries so tagged and nothing else -- `anira_tensor_init_*`, `anira_tensor_data_f32`, `anira_tensor_data` (a converting stage needs the typed read from inside a callback), `anira_tensor_num_elements`, `anira_tensor_extent`, `anira_sizeof`, `anira_ring_*`, `anira_stage_default_pre_process/post_process`, `anira_log_rt`, `anira_now_ms`/`anira_now_ns`, `anira_handler_rt_error`, `anira_handler_ticket_status`, `ticket_input_released`, `ticket_output_ready`, `ticket_error`, `ticket_user_data`, `ticket_release`; a stage's `prepare` may additionally use the handler getters and `set_input`/`get_output`. The reason is a real lock: `anira_backend_desc.prepare` runs inside `create_session` under `m_lifecycle_mutex` (`Context.cpp:530-535`) and sync log records are dispatched while that non-recursive mutex is held, so a `prepare` or log callback that calls `anira_machine_create`, `anira_handler_create/destroy` or `anira_shutdown` deadlocks. `ticket_release` from `on_complete` is legal (the slot recycles after the callback returns); `submit` is not. On Wasm JS supplies no function pointers; `anira_em.h` ships trampolines (`anira_em_stage_desc_init`, `anira_em_backend_desc_init`, `anira_em_job_options_set_on_complete_js`, `anira_em_set_log_hook`) that fill the same slots with C functions compiled into anira (section 6b).

### Fixed-width types and pinned values

Every enum is a C `enum` with explicit values and a `_FORCE32 = 0x7fffffff` terminator; enums appear as the enum type in parameters and as `uint32_t` fields in structs. `typedef uint32_t anira_bool;`. `anira_dtype` is a packed `uint32_t` = `code | bits << 8 | lanes << 16` (`ANIRA_MAKE_DTYPE`, `ANIRA_DTYPE_F32` and the other constants), whose little-endian bytes are exactly DLPack's `DLDataType {uint8 code; uint8 bits; uint16 lanes}` -- a 4-byte struct is passed `byval` through a pointer on wasm32, a `uint32_t` is not, and `from_dlpack` is a `memcpy`. `anira_ticket` is `uint32_t`. Indices, slots, counts of tensors and channels, and sentinels are `uint32_t` (`ANIRA_ANCHOR_FIRST_STREAMED = 0xffffffff`, `ANIRA_THREADS_AUTO = 0xffffffff`, `ANIRA_TICKET_INVALID = 0`); sample and element counts are `size_t` (v2-identical on the Hard entries; a JS `number` on wasm32); durations, deadlines and timeouts are `double` milliseconds (`ANIRA_WAIT_FOREVER = -1.0`, `ANIRA_WAIT_CONTRACT = -2.0`). "Hot" is every `ANIRA_NONBLOCKING` declaration: no `int64_t`/`uint64_t` parameter or return there; the allowlist is `anira_now_ns` and the four GPU factories `anira_tensor_init_vulkan`/`_init_opaque_fd`/`_init_wgpu_buffer`/`_init_dmabuf`, whose parameters are vendor handles and byte offsets at their wire width and which no JS caller reaches (`scan_header.py` carries the list); `int64_t` otherwise appears only in config-time shape/window/extent setters (`BigInt` at config time only) and in the `anira_tensor` fields, which JS reads through `layout.ts`. Any `uint32_t`/`size_t` value >= 2^31 returned to JS arrives as a negative i32 and needs `>>> 0`; that rule is documented once, in `web/src/runtime/Heap.ts`, never claimed away.

Every enum value is pinned independently of `USE_*` (today `enum InferenceBackend` shifts with the build, `include/anira/utils/InferenceBackend.h:32-110`, and the web wrapper exports `get_inference_backend_onnx()` to cope). Engine and execution provider are two independent enums, never one packed id: a provider name means the same thing across engines -- CUDA is CUDA whether ONNX Runtime or LibTorch runs it -- which the packed form hid. Engine values `6..0x0fff` are reserved for later anira engines; a custom engine is registered by name and assigned a value from `0x1000` up at `anira_handler_prepare`, so its number is pipeline-scoped and meaningless outside that pipeline, and its registered name must be reverse-URI -- the reservation section 1b already makes for third-party extension kinds -- so anira's short engine names can never collide. Domain values `12-63` are reserved for later domains.

```c
typedef enum anira_engine   { ANIRA_ENGINE_NONE = 0, ANIRA_ENGINE_ONNXRUNTIME = 1, ANIRA_ENGINE_LIBTORCH = 2, ANIRA_ENGINE_TFLITE = 3,
    ANIRA_ENGINE_LITERT = 4, ANIRA_ENGINE_EXECUTORCH = 5,
    /* 6..0x0fff reserved for anira engines; a registered custom engine is assigned a value from 0x1000 up at prepare */
    ANIRA_ENGINE_FORCE32 = 0x7fffffff } anira_engine;
typedef enum anira_provider { ANIRA_PROVIDER_DEFAULT = 0 /* the engine's own CPU path */, ANIRA_PROVIDER_CUDA = 1, ANIRA_PROVIDER_WEBGPU = 2,
    ANIRA_PROVIDER_DIRECTML = 3, ANIRA_PROVIDER_COREML = 4, ANIRA_PROVIDER_XNNPACK = 5, ANIRA_PROVIDER_VULKAN = 6,
    ANIRA_PROVIDER_FORCE32 = 0x7fffffff } anira_provider;
/* where the pair must travel as one item, a Tier-2 record */
typedef struct anira_backend_id { uint32_t struct_size; uint32_t engine; uint32_t provider; const char* engine_id; } anira_backend_id;
    /* engine_id: NULL for a built-in engine, the registered name for a custom one */
```

| `anira_engine` | value | v2 name / JSON string |
|---|---|---|
| `ANIRA_ENGINE_NONE` | `0` | -- |
| `ANIRA_ENGINE_ONNXRUNTIME` | `1` | `ONNX` / `"onnxruntime"` |
| `ANIRA_ENGINE_LIBTORCH` | `2` | `LIBTORCH` / `"libtorch"` |
| `ANIRA_ENGINE_TFLITE` | `3` | `TFLITE` / `"tflite"` |
| `ANIRA_ENGINE_LITERT` | `4` | `LITERT` / `"litert"` |
| `ANIRA_ENGINE_EXECUTORCH` | `5` | `EXECUTORCH` / `"executorch"` |
| `ANIRA_ENGINE_FORCE32` | `0x7fffffff` | -- |

| `anira_provider` | value | JSON suffix |
|---|---|---|
| `ANIRA_PROVIDER_DEFAULT` | `0` | -- (omitted) |
| `ANIRA_PROVIDER_CUDA` | `1` | `:cuda` |
| `ANIRA_PROVIDER_WEBGPU` | `2` | `:webgpu` |
| `ANIRA_PROVIDER_DIRECTML` | `3` | `:directml` |
| `ANIRA_PROVIDER_COREML` | `4` | `:coreml` |
| `ANIRA_PROVIDER_XNNPACK` | `5` | `:xnnpack` |
| `ANIRA_PROVIDER_VULKAN` | `6` | `:vulkan` |
| `ANIRA_PROVIDER_FORCE32` | `0x7fffffff` | -- |

A v2 `InferenceBackend` names an engine and never a provider, so v2 `ONNX` is `ANIRA_ENGINE_ONNXRUNTIME` with `ANIRA_PROVIDER_DEFAULT`, `LIBTORCH` is `ANIRA_ENGINE_LIBTORCH` with `ANIRA_PROVIDER_DEFAULT`, and so on through the five; v2 `CUSTOM` is a registered engine name, not a value.

Upper-case v2 strings are accepted on the JSON upgrade path (section 8). Enum values, status codes and `ANIRA_ABI_VERSION` reach TS as `web/src/abi/enums.ts`, generated by `tools/abi/gen.py` from the registry `abi/anira.yml`, the same source those headers are generated from.

### Thread tags

Every declaration carries one tag; `[callback-safe]` is added where it applies. The vocabulary is CLAP's (`clap/include/clap/plugin.h:41-110`, `ext/thread-check.h:37-41`: "functions marked with [audio-thread] ARE NOT CONCURRENT"), with CLAP's `[audio-thread]` renamed `[driver-thread]` because anira drives frames as well as samples and the tag never named the medium, extended by the roles anira has and CLAP does not:

- `[main-thread]` -- control path, externally serialised per object, may allocate and block; `[main-thread & !prepared]`, `[main-thread & prepared]`, `[main-thread & !processing]` -- state-qualified; `[main-thread & !loader-lock]` -- never from `DllMain`/`dlclose` paths.
- `[driver-thread]` -- the thread that drives a Hard contract: the host's real-time callback, whatever the medium -- an audio device callback, a video frame callback, a render loop. `ANIRA_NONBLOCKING`, not concurrent per handler: two driver threads may drive two handlers of one machine, never one handler. "Not concurrent" holds per handler, not per machine. The tag names the role, not the medium, which is why `pre_process` carries it under Hard and `[inference-thread]` under Async.
- `[inference-thread]` -- anira's pool or a user `anira_inference_thread`.
- `[thread-safe]` -- any thread, concurrently; `[thread-safe, !audio-thread]` -- any thread except the driver thread (fd-closing token calls, control-path logging, `anira_drain_log`).
- `[callback-safe]` (`[cs]`) -- additive: also legal from inside any anira callback.
- `[drain-thread]` -- the log sink's thread for RT records; a real-time record may also be delivered on the thread of a failing `[main-thread]` entry, which drains the queue before returning a negative status.
- `[any-thread, blocking]` -- may wait (the `_wait` twins, `anira_handler_ticket_wait`); on Wasm every wait spins on `emscripten_get_now`, so inside the worklet calling one is a host policy decision.

### What anira borrows from CLAP, and what it does not

Applied: per-slot thread tags on every declaration (the vocabulary above, with `[inference-thread]`, `[drain-thread]`, `[callback-safe]`, `[any-thread, blocking]` and the state qualifiers added because anira has more thread roles than CLAP's two); version-first host-provided vtables with the user pointer second (`clap_host.clap_version`/`host_data`, `clap_plugin.plugin_data`) -- `anira_stage_desc`, `anira_backend_desc`, `anira_log_desc` are `{struct_size, abi_version, user_data, ...}`, with one packed `uint32_t` instead of `{major, minor, revision}` because every other version in the header is packed and TS writes one field; the compat macros (`CLAP_ABI` `__cdecl` on 32-bit Windows, one header for C and C++) as `ANIRA_CALL`, `ANIRA_INIT`, the `extern "C"` guards and the C11-plus-C++17 compile test; the `init`/`deinit` refcount rule of `clap_entry`, applied per DSO as CLAP means it, while `anira_shutdown` additionally refuses while any machine or handler exists in the copy -- a rule CLAP does not need because a CLAP DSO is never shared, and the one that protects a shared `libanira.so.3` between two clients; string ids with a revision and a draft folder (`conventions/extension-id.md`, `clap.h` vs `all.h`) -- extension kinds are strings with a `version` field in `anira_ext_header`, the registry key is `(kind, version)` with the revision as a field rather than a `/REV` suffix because the kind doubles as the JSON key, `abi/draft/` plus `anira_all.h` carry unmeasured surfaces with their own baselines, promotion never renames, no `_COMPAT` ids ever, reverse-URI third-party kinds reserved for after 3.0; size-first records for what the host enumerates (`events.h`) -- `anira_edge_info`, `anira_plan_slot`, `anira_plan_ext` are `struct_size`-first, walked by stride and count, rows with pointers valid for the call only, whereas `anira_log_record` and `anira_error` are frozen Tier-1 PODs because one is read per drained record and the other is value-initialised by `Result<T>`; allocation confined to activation (`plugin.h:61`) restated as no allocation after `prepare` on any `[driver-thread]`/`[inference-thread]` path -- `anira_handler_create` loads models, `allocate_*` allocates by name, `anira_machine_create` probes, all `[main-thread]`, and `get_latency` is valid from `prepare` on; render mode as contract kind (`ext/render.h`) -- `anira_contract_kind` at `prepare` is CLAP's `set(REALTIME | OFFLINE)`, `ANIRA_ERROR_WRONG_CONTRACT` its refusal; log severities (`ext/log.h` `HOST_MISBEHAVING`/`PLUGIN_MISBEHAVING`) with a deviation -- the four v2 levels stay and a contract violation is a record with `ANIRA_LOG_RECORD_CONTRACT_VIOLATION` set, so "most verbose wins" ordering is unchanged; `clap_id`/`CLAP_INVALID_ID` and caller-owned text buffers (`ext/params.h`) as `uint32_t` indices, `ANIRA_TICKET_INVALID = 0` and `(buf, cap, *out_len)`.

Deliberately not borrowed: string-keyed `get_extension` near the driver thread (everything string-keyed resolves at `prepare`; `anira_get_proc_address` covers "functions not all builds provide"); the plugin-side vtable as the library boundary (entry points are flat functions; vtables exist only for host-provided services and are not callable from JS otherwise); `bool` as the only error channel; bundling the process call into one struct (the hot path is frozen at `float* const*`); host-pumped main-thread machinery (`request_callback`, timer-support, posix-fd-support -- anira has no event loop, only `anira_handler_poll`); fixed 256-byte name arrays in hot structs (names live on `anira_tensor_spec` only); `clap_version_is_compatible = major >= 1` (anira majors may break; `anira_check_abi` tests major equality); blocking `thread-pool.request_exec`, a fork-join inside `process()` (anira's threads are long-lived `anira_inference_thread` loops; a host-provided thread-provider vtable is deferred); the export-only `CLAP_EXPORT` (anira needs import/export and an Emscripten branch); convention-only stability (CLAP ships no checker; anira ships six gates).

### Gates

Six CTests own the contract, each covering one failure mode on every platform (section 11 lists what comes online at which milestone): `anira_symbol_baseline` diffs the real export table against `abi/symbols-<major>.txt` plus the draft list (gate 1); `anira_abi_diff` runs libabigail against the last tag's baseline, once over the whole header set and once over the Tier-1 headers alone (gate 2); `anira_abi_layout` emits every Tier-1 `sizeof`/`offsetof` natively and under node and diffs it against `abi/layout-<major>.txt` (gate 3); `anira_header_c11` compiles `anira.h` and `anira_em.h` as C11 and as C++17 with no anira define on every preset, and the strict `anira_header_isolation` proves no `tanh/`, `nlohmann/`, `concurrentqueue` or `benchmark/` include is reachable (gate 4); `test/install/consumer_c` links the installed package from pure C in both library shapes and runs the raw-C sketch of section 9, beside `consumer_tanh_first` (gate 5); the RTSan leg instruments every `ANIRA_NONBLOCKING` entry and callback typedef at runtime, with `-Werror=function-effects` on the consumer-shaped targets only (gate 6). The presence check and the abidiff are the two things tanh-tooling lacks today; until a tag after v3.0.0 they live as anira-local scripts under `anira/cmake/`.

---

## 6b. The boundary: tanh-lib, third parties, and the WebAssembly build

One copy of anira is one self-contained world: hidden visibility, `-fno-gnu-unique`, a private thread pool, a private logger, and tanh-lib's objects inside `libanira`. tanh-lib, concurrentqueue and nlohmann_json are PRIVATE and invisible on every platform -- no public header includes them, logging crosses as a C callback, and tanh-lib gets no C ABI of its own. The WebAssembly build is the fifth static-embedding target, with a JavaScript host instead of a plugin host; nothing in it is a second ABI. `anira_exports` forbids `thl`, `moodycamel` and `nlohmann` in the export table, `anira_header_isolation` proves no third-party include is reachable through `anira::anira`, and `consumer_tanh_first` proves the binding is local (section 6a).

### tanh-lib: absorbed, private, no C ABI

Every place the v2 headers leak tanh-lib or another dependency, and its closure:

| leak today | closure |
|---|---|
| `include/anira/system/Exports.h:30` includes `<tanh/core/ExportMacros.h>` | `abi/export.h` is self-contained; `Exports.h` becomes a private header whose `ANIRA_API` expands to nothing on every platform from alpha.2 (section 6a) |
| `utils/Logger.h:4` and `scheduler/Context.h:4,542` include `<tanh/core/Logger.h>`; `utils/Logger.h:112` exports `rt_log_queue_slot()` returning `std::atomic<thl::Logger::rt::Queue*>&` | both headers private; logging crosses as `anira_log_desc` in and `anira_log_record` out (section 4); the slot stays constant-initialised per copy |
| `scheduler/InferenceThread.h:9,226` holds a `thl::core::Thread` member; `system/HighPriorityThread.h:4` | opaque `anira_inference_thread`; `HighPriorityThread` removed -- it is already `[[deprecated]]` for one minor |
| `utils/Buffer.h:4,29`, `utils/RingBuffer.h:4,39-43`, `utils/MemoryBlock.h:4,19` are aliases over `thl::core` templates | deleted from the public tree, used internally; consumers see `anira_ring*` and `anira_tensor`, v2 subclasses see the `compat/v2.hpp` views (section 7) |
| `scheduler/SessionElement.h:4-7,311,344-349` includes `<concurrentqueue.h>` and names `moodycamel::*`; `utils/Semaphore.h:34-35` | private; the pump's FIFO payload becomes a POD task reference |
| `utils/JsonConfigLoader.h:8` includes `<nlohmann/json.hpp>` | private; JSON crosses as UTF-8 text (section 8) |
| `benchmark/ProcessBlockFixture.h:4,8` includes `<benchmark/benchmark.h>` and the v2 umbrella; `cmake/benchmark-src.cmake:16` and `cmake/test-deps.cmake:7` link `benchmark`/`gtest_main` PUBLIC; `cmake/install.cmake:39-42` installs the whole `include/anira` tree; `Config.cmake.in:29-35` looks up GTest and benchmark | the fixture moves to `examples/benchmark/` over `anira.hpp`; `benchmark` and `gtest_main` are never linked into `anira`; `install(DIRECTORY)` becomes the explicit file list below; `Config.cmake.in` loses both lookups (M2) |
| `CMakeLists.txt:363` links `concurrentqueue nlohmann_json::nlohmann_json tanh::Core` PUBLIC; `Config.cmake.in:20-27` runs `find_package(concurrentqueue)`, `find_package(nlohmann_json)`, `find_dependency(tanh COMPONENTS Core)`; `cmake/install.cmake:49` installs all three targets | none installed, no `find_dependency`; `TANH_WITH_INSTALL OFF`, `JSON_Install OFF` |
| tanh_core's PUBLIC defines `TANH_VERSION`, `THL_LOG_COMPILED_MAX_LEVEL`, `THL_PLATFORM_*` (`tanh-lib/CMakeLists.txt:129-150`) and the PUBLIC `-fsanitize=realtime` of `tanh_add_sanitizer` | stop at anira; `ANIRA_WITH_RTSAN` re-exports only `-fsanitize=realtime`, to anira's own consumers |
| `TORCH_CXX_FLAGS` PUBLIC (`CMakeLists.txt:402-411`), pinning the `std::string` ABI of the public API | PRIVATE: a C ABI has no `std::string` to pin |

Embedding. tanh_core's objects are absorbed into `libanira`. tanh-lib stays a `FetchContent` subtree at its tag (`GIT_TAG v0.1.0`, `CMakeLists.txt:227-237`), its own CMake runs -- `tanh_apply_symbol_policy(tanh_core EXPORT_PREFIX TANH)` and the `NAMESPACE thl` allowlist -- and the same-tag warning of `modules-version.cmake` stays alive; compiling tanh's translation units directly is rejected, because it would fork tanh-lib's build. The shape is anira-local and fifteen lines:

```cmake
# CMakeLists.txt -- tanh_core built by its own CMake as a hidden static archive, its objects absorbed
set(_anira_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)                       # scoped: TANH_STATIC PUBLIC, TANH_API empty (symbol-policy.cmake:110-115)
set(TANH_WITH_INSTALL OFF)
set(TANH_LOG_COMPILED_MAX_LEVEL 4)               # one compiled log ceiling in every build type (a tanh-lib cache option; plain variable = CMP0077 override)
FetchContent_MakeAvailable(tanh-lib)
set(BUILD_SHARED_LIBS ${_anira_shared})
target_sources(anira PRIVATE $<TARGET_OBJECTS:tanh_core>)           # core.cpp, Dispatcher, Logger, Thread, RCU
target_link_libraries(anira PRIVATE $<BUILD_INTERFACE:tanh_core> $<BUILD_INTERFACE:concurrentqueue>
                                    $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)   # include dirs and defines at build time; nothing in the export
if(TANH_OPERATING_SYSTEM STREQUAL "Android")
    target_link_libraries(anira PRIVATE log)                         # tanh_core's one PRIVATE system link, repeated
endif()
```

`$<TARGET_OBJECTS:tanh_core>` puts the five objects of `tanh-lib/CMakeLists.txt:108-114` into `libanira.{a,so}`; the `$<BUILD_INTERFACE>` links carry include directories and defines (`TANH_STATIC`, `THL_PLATFORM_*`) while building and evaluate to nothing in the install export, so `install(EXPORT aniraTargets)` never demands `tanh_core` in an export set -- the reason today's tree installs `libtanh_core.a` and exports `concurrentqueue` and `nlohmann_json` disappears. `concurrentqueue` and `nlohmann_json` are header-only, `SYSTEM`, build-interface only. The `LINKER:--exclude-libs,libtanh_core.a` entry of `CMakeLists.txt:301-303` is dropped, since no such archive is linked any more; `tanh_hidden_archive_link_items` stays for the engine archives. `THL_LOG_COMPILED_MAX_LEVEL` is consumed only inside `Logger.cpp` (`tanh-lib/src/core/Logger.cpp:46-62`) and is `1` in Release by tanh's default (`TANH_LOG_COMPILED_MAX_LEVEL=AUTO`); anira sets the option to `4` before the fetch, so `anira_log_desc.level` means the same thing in every build type and a v2 file's `ANIRA_LOG_WARNING` reaches a Release sink.

Why this closes the two-copies hazard. anira's translation units see `TANH_STATIC`, so `TANH_API` is empty and, under `-fvisibility=hidden`, every reference from anira to `thl::Logger::set_callback`, `thl::Logger::rt::Queue` or `thl::core::Thread` is an STV_HIDDEN undefined reference, which the static linker may satisfy only from a relocatable object: a `libtanh_core.so` that appears earlier on a plugin's link line -- a tanh-lab plugin using `tanh::State` -- cannot interpose it on ELF or Mach-O regardless of link order, and the objects that satisfy it are the ones inside `libanira.a`. A shared `libanira.so.3` carries tanh_core under `local: *` of its version script; a static `libanira.a` carries it hidden; on Mach-O the plugin's `-exported_symbols_list` does the same job. PE has no visibility on undefined references, so that guarantee does not exist there: the rule "list `anira::anira` before `tanh::Core`" is documented for Windows, and `test/install/consumer_tanh_first` -- a consumer that links a shared `tanh::Core` first and `anira::anira` second and checks with `nm`/`dumpbin` that the module binds `thl` locally -- runs on the ELF and Mach-O install legs. Two copies of tanh_core in one process (two plugins, or a plugin beside a host that uses tanh-lib) each own their `g_runtime_level`, `state()`, `default_queue()`, `atexit` hook and `rcu_thread_state()`; with the objects absorbed and hidden nothing of either copy is in a dynamic symbol table and, on ELF and Mach-O, nothing can bind across. The only visible effect is two `"anira-log"` drain threads, which is accepted: merging them would require tanh-lib to become a shared dependency again, the situation being left.

Logging. anira keeps a private `thl::Logger` -- the copy inside `libanira` -- and installs one trampoline on it (`thl::Logger::set_callback`, once per core) that fans out to a per-copy sink registry with one entry `{callback, user_data, level, in_flight}` per machine that set a sink; `anira_machine_destroy` unregisters its entry and blocks until the entry's in-flight count is zero, because thl dispatches a callback from a copy taken outside its mutex (`tanh-lib/src/core/Logger.cpp:466-472, 501-505`) and a clear alone would not suffice. An application that links tanh-lib has a second, invisible logger that anira never touches. `anira_log_record` is a projection of `thl::Logger::LogRecord`; the level mapping, the RT queue, the drain thread and the platform-sink flag are stated once, in section 4.

`Thread`, `RCU`, `Dispatcher` and `LockFreeQueue` are internal. `thl::RCU<T>` (namespace `thl`, not `thl::core`) is M5's selection primitive natively only: it keys reader identity on a `static thread_local` (`tanh-lib/src/core/RCU.cpp:26-29`), which the WebAssembly build cannot honour; on Wasm selection stays the relaxed-atomic index of `src/scheduler/InferenceManager.cpp:36-38`, sufficient with one pump instance. The two `thread_local` objects tanh_core carries (`Logger.cpp:411`, `RCU.cpp:27`) are tolerated on Wasm because sync log dispatch runs on main or the Worker only and RCU is never instantiated there.

Pin rule. anira's tanh-lib tag may move freely; anira and the fetched tanh-lib must carry the same tanh-tooling release in `cmake/tanh/` (0.1.5 today, `modules-version.cmake:14-23` warns on a mismatch), so a tooling bump is a two-repository PR pair. Every CMake change of this section uses the modules that exist; what they lack is an anira-local script under `anira/cmake/` until the tanh-tooling tag after v3.0.0 that both repositories re-pin together (section 11). tanh-lib gets no C ABI: nothing of it is a binary contract of anira's, and a tanh-lab plugin that wants `thl` links tanh-lib itself.

The installed package, as an explicit file list:

| path | content |
|---|---|
| `include/anira/{anira.h, anira_all.h, anira_em.h, anira.hpp, compat/v2.hpp}`, `include/anira/abi/*.h`, `include/anira/abi/draft/*.h` | the header set (section 6a); nothing else under `include/anira` |
| `lib/libanira.so.3.x.y` + soname/namelink, `lib/libanira.3.dylib`, `lib/libanira.a`; `bin/anira-3.dll` + `lib/anira-3.lib` | the library, `SOVERSION`/`MACHO_*`/`OUTPUT_NAME` from `ANIRA_ABI_MAJOR` |
| `include/anira-backends/<engine>/`, engine libraries | unchanged |
| `lib/cmake/anira/{aniraConfig.cmake, aniraConfigVersion.cmake, aniraTargets.cmake, aniraBackendTargets.cmake, tanh/*.cmake}` | `Config.cmake` finds engines only; `SameMajorVersion` |
| -- | no `include/tanh`, no `libtanh_core`, no concurrentqueue, no nlohmann, no benchmark or gtest headers |

### WebAssembly: the C ABI with a JavaScript host

Deleted: `src/emscripten-wrappers/` -- eleven files, 1865 lines, 226 `EMSCRIPTEN_KEEPALIVE` entry points with `uintptr_t` handles, `int` enums, 43 `vector_*` shims (48 `EMSCRIPTEN_KEEPALIVE` entries in `src/emscripten-wrappers/utils/Vectors.cpp`, five of them `buffer_f_*`) and no error path -- at M2, together with the `anira_wasm_wrappers` archive and the `src/emscripten-wrappers/` exclusion in `clang_tidy.yml`. The C ABI implementation in `src/capi/` is compiled into `libanira.a` on every target, Emscripten included; the only Emscripten-specific code is `src/platform/emscripten/em_hooks.cpp` (about 180 lines: the `anira_em_*` functions, the `EM_JS` thunks, `anira_em_alloc_thread_stack`). JavaScript reaches every entry as `Module._anira_<x>`, exactly as it reaches `_inferencehandler_process` today.

Link line. `cmake/build-wasm.cmake` keeps the `--no-entry` `AniraWeb` module, drops the whole-archive link (`build-wasm.cmake:68-69`) and `EXPORT_KEEPALIVE=1`, and links `libanira.a` against a generated export list:

```cmake
add_executable(AniraWeb "${CMAKE_CURRENT_BINARY_DIR}/aniraweb_stub.cpp")
target_link_libraries(AniraWeb PRIVATE anira::anira)                # no wrapper archive, no --whole-archive
#   --no-entry --emit-tsd=web/wasm/AniraWeb.d.ts                    # MainModule typing: _malloc/_free/HEAP* and every _anira_* signature
#   -sIMPORTED_MEMORY=1 -sINITIAL_MEMORY=536870912 -sSHARED_MEMORY=1 -sALLOW_MEMORY_GROWTH=0 -sMALLOC=emmalloc
#   -sEXPORT_ES6=1 -sMODULARIZE=1 -sENVIRONMENT=worklet,web -sSTACK_SIZE=33554432 -sSTACK_OVERFLOW_CHECK=0 -sASSERTIONS=1
#   -sNO_DISABLE_EXCEPTION_CATCHING                                  # the exception firewall's catch needs it (section 6a)
#   -sEXPORTED_FUNCTIONS=@${CMAKE_BINARY_DIR}/abi/exports_wasm.txt   # every ANIRA_API name of abi/*.h + anira_em.h, _-prefixed, plus _malloc, _free
#   -sEXPORTED_RUNTIME_METHODS=UTF8ToString,HEAPU32,HEAPF32,stackSave,stackRestore
```

The export list alone pulls the archive members (verified on emsdk 4.0.23), and an undefined name fails the link, which is the Wasm presence gate; no `EMSCRIPTEN_KEEPALIVE` and no `used` attribute exist in the header. The committed copy of the list under `web/src/abi/` is used when Python is absent at configure time. Exception catching stays on because Emscripten's default compiles `throw` and disables every `catch`, including the firewall's; `-fwasm-exceptions` is a size and speed choice once the ORT wasm archive in the backends release is built with it. `EXPORTED_RUNTIME_METHODS` is unchanged: no `HEAP64` -- TypeScript builds `BigInt64Array` views over `memory.buffer` itself, valid because memory never grows -- and no `addFunction`.

Generated artefacts, all diffed in `build_web.yml`, which today only builds:

| artefact | source | produced by | checked by |
|---|---|---|---|
| `web/src/abi/enums.ts` | `abi/anira.yml`, the registry the C headers are generated from: enum values, status codes, `ANIRA_ABI_VERSION`, struct ids | `tools/abi/gen.py` | diff against the committed file (the `anira_abi_generate` CTest and `build_web.yml`) |
| `web/src/abi/exports_wasm.txt`, `abi/symbols-<major>.txt` | every function of `abi/anira.yml` (promised and draft), `anira_em.h` | `gen.py` | the link (presence); `AniraWeb.d.ts` export names diffed against `symbols-<major>.txt` ∪ `symbols-draft.txt` ∪ `anira_em_*` (extras, until a `wasm-objdump` branch lands in tanh-tooling) |
| `web/src/abi/layout.ts` | `sizeof`/`offsetof` of every Tier-1 POD | the layout executable, run under node | diff against `abi/layout-<major>.txt`; asserts `__builtin_wasm_tls_size() <= 1024` |

`validate-options.cmake:128-134` forbids tests on Wasm; the layout executable and the C11 header compile are the one explicit exception, built under `ANIRA_WITH_ABI_TESTS` as OBJECT targets plus one node-run executable. JavaScript never hand-computes a struct offset on the control path and never writes a struct field by hand: every descriptor it must produce has an `anira_em_*_init` or a scalar-setter twin, and the RT structs it reads (`anira_stage_ctx`, `anira_tensor`, `anira_log_record`) are read through `layout.ts`. The hand-mirrored constants of today (`get_inference_backend_onnx()`, the libc++ vector layout read in `InferenceHandler.ts:201-212`) are the drift the ABI exists to end.

Threading model, unchanged. No `-pthread`. One shared `WebAssembly.Memory`, 512 MiB, fixed; one module instance per JS thread -- main, each inference Worker, the AudioWorklet -- each with its own 4 MiB stack carved out of the heap by main through `anira_em_alloc_thread_stack` (today `AniraWeb.allocateWorkerStack`, `web/src/AniraWeb.ts:256-266`). `anira_inference_thread_create` runs on the main instance; the Worker calls `_anira_inference_thread_run_loop(t)` synchronously inside `onmessage` (`web/src/workers/inferenceWorkerHandler.ts:222-226`), so its event loop is blocked for the life of the loop. Stop protocol: main calls `_anira_inference_thread_stop`, which on Wasm only requests the stop (`src/scheduler/InferenceThread.cpp:82-87` flips the flags while the Worker is still inside the loop), then awaits `_anira_inference_thread_has_exited` -- a shared-memory atomic set on `run_loop` exit -- before `_anira_inference_thread_destroy`; this replaces the `'stopped'` message the TypeScript `InferenceThread` relies on today (`AniraWeb.ts:454-458`). A counter of loop-active threads, maintained at `run_loop` entry and exit, is consulted by `anira_release_core_if_idle`, so the core is never freed under a Worker that is still dequeuing. `num_threads` is coerced to 0, `LogDrain` to `ANIRA_LOG_DRAIN_MANUAL` and the wait strategy to `ANIRA_WAIT_SPIN_BACKOFF`, each with a warning, as `src/scheduler/Context.cpp:207-247` does today. The real-time path stays one `_anira_handler_process_separate` call per render quantum on the worklet's instance (`web/src/workers/worklet-base.ts:266-272`).

No per-instance TLS exists. Only the instance that runs `__wasm_init_memory` gets `__tls_base = 1024`; every later hand-instantiated instance gets `__tls_base = 0`, so the Worker and the worklet share one `thread_local` block at address 0 that overlaps the main instance's static data once the TLS segment exceeds 1024 bytes (verified on emsdk 4.0.23). Rule: no `thread_local` in anira code reachable from the Worker or the worklet -- the `thread_local` vector behind `inferencehandler_get_latency_vector` (`src/emscripten-wrappers/InferenceHandler.cpp:226`) dies with its file and `anira_handler_get_latencies` writes into caller memory -- and the layout executable asserts `__builtin_wasm_tls_size() <= 1024` on every build.

Waits spin. In the no-pthread `SHARED_MEMORY` libc the futex is a no-op stub and `try_acquire_until` busy-waits on `emscripten_get_now` (verified), so every `[any-thread, blocking]` entry spins on Wasm: the `_wait` twins, `anira_handler_ticket_wait`, `anira_machine_destroy`'s in-flight wait. The worklet may call a `_wait` twin only if the host accepts spinning inside the render quantum -- which is exactly what v2's `blocking_ratio` did there, and why `worklet-base.ts:9-24` polyfills `performance.now` for a scope that lacks it; the polyfill stays. `AniraAudioWorkletBase` calls the `ANIRA_NONBLOCKING` entries by default and exposes `processWait` as an explicit opt-in. Allocation across instances is safe -- emmalloc under `SHARED_MEMORY` spinlocks every operation -- and forbidden on `ANIRA_NONBLOCKING` paths and in the worklet by real-time policy, the same policy as native; the compiled-in engine's `process` on the Worker allocates as it does natively.

The JS hook object. JavaScript cannot supply a C function pointer without `addFunction`, which needs `ALLOW_TABLE_GROWTH` and a per-instance table in this multi-instance build. `anira_em.h` therefore ships trampolines: C functions compiled into anira that forward through one `EM_JS` hook object per module instance, replacing the undocumented `Module.processPrePost`/`processBuffers` of `web/src/factory.ts:36-41`.

```c
/* include/anira/anira_em.h -- body under #ifdef __EMSCRIPTEN__; installed with anira.h; appended-to like every other header */
#define ANIRA_EM_HOOKS_VERSION 1                 /* layout of Module.anira; checked at instantiation */
#define ANIRA_EM_HOOK_PRE 1u
#define ANIRA_EM_HOOK_POST 2u
#define ANIRA_EM_HOOK_BEFORE 4u
#define ANIRA_EM_HOOK_AFTER 8u
#define ANIRA_EM_HOOK_PREPARE 16u
#define ANIRA_EM_HOOK_RELEASE 32u
uint32_t     anira_em_hooks_version(void);                                                  /* [thread-safe] */
anira_status anira_em_stage_desc_init(anira_stage_desc*, uint32_t js_key, uint32_t hook_mask);   /* fills the six slots with EM_JS trampolines; unset bits stay NULL and never cross */
anira_status anira_em_pipeline_add_js_stage(anira_pipeline*, uint32_t domain_in, uint32_t domain_out, uint32_t js_key, uint32_t hook_mask, anira_error* err);
                                                       /* == anira_em_stage_desc_init + anira_pipeline_add_stage: the same registration path as native */
anira_status anira_em_backend_desc_init(anira_backend_desc*, const char* engine_id, uint32_t js_key);   /* process only; prepare/release NULL: readiness is the TS Worker protocol; the trampoline captures engine_id and hands it to JS */
anira_status anira_em_pipeline_register_js_engine(anira_pipeline*, const char* engine_id, uint32_t js_key, anira_error* err);   /* == desc_init + anira_pipeline_register_engine under the same name; engine_id is reverse-URI */
void         anira_em_set_log_hook(anira_bool enable);                                      /* the only Wasm sink installer: every record -> Module.anira.log(recordPtr) on the emitting/draining instance */
anira_status anira_em_job_options_set_on_complete_js(anira_job_options*, uint32_t js_key);  /* the key travels with submit: Module.anira.complete(key, handlerPtr, ticket) on the Worker; TS forwards by postMessage */
void*        anira_em_alloc_thread_stack(size_t bytes);  void anira_em_free_thread_stack(void*);   /* [main-thread]; the Worker/worklet stack carving */
/* reserved for post-3.0: anira_em_machine_config_set_js_webgpu(mc, js_key, err) -- the browser GPUDevice import, with the asynchronous backend contract below */
/* Module.anira = { version: number, stage(key, phase, a, b), backend(key, engineIdPtr, inPtr, nIn, outPtr, nOut, instance), log(recordPtr), complete(key, handlerPtr, ticket) }
   phases 0-3: a = ctxPtr, b = 0; phase 4 PREPARE: a = handlerPtr, b = planReportPtr; phase 5 RELEASE: a = b = 0.
   engineIdPtr is the UTF-8 name the engine was registered under, so a JS backend can find its own models[] row. */
```

`js_key` is the JS-side registry key carried in `user_data`; the registries are per instance, keyed by `js_key`, as today's `Map<ptr, instance>` registries are keyed by the C++ `this` pointer. `engine_id` is the second identity and independent of it: the key finds the JS instance, the name finds its `models[]` row, so two JS engines in one Worker read two rows instead of both resolving the singleton today's `ONNXRuntimeWebBackend.ts:139-143` hardcodes. `hook_mask` replaces the `m_js_inference_hooks` atomic gate of `src/emscripten-wrappers/JSPrePostProcessor.cpp:53,63,101`: an unset bit stays NULL in the descriptor and never crosses into JS, so an unregistered phase costs nothing on the inference thread. Each hook fires on the JS thread whose module instance executes the C call -- the worklet for `pre_process`/`post_process`, the Worker for `before_inference`/`after_inference` and backend `process`, main for `prepare`/`release` -- identical to today's phase routing (`JSPrePostProcessor.cpp:6-19`). A JS backend's `prepare` and `release` are NULL: readiness is the protocol that exists today (Worker `registerProcessor` → `await init()` → `'processorRegistered'`, `inferenceWorkerHandler.ts:143-166`), which the TypeScript side completes before calling `anira_handler_prepare`, and `anira_handler_create` loads nothing for a registered custom engine whose `prepare` is NULL (section 7). JS completion keys travel with the job through `anira_em_job_options_set_on_complete_js`, so no completion can precede its registration; `Module.anira.complete` fires inside the Worker's blocked loop and the TypeScript worker forwards it to the main-thread `Ticket` by `postMessage`. `Module.anira.version` is checked against `ANIRA_EM_HOOKS_VERSION` at instantiation; the hook object's layout is versioned by that number, appended to like the header.

Two engine paths behind one ABI. (a) Today's `onnxruntime-web` backend is a custom engine registered by name, `anira_em_pipeline_register_js_engine(pipe, "org.anira.onnxruntime-web", js_key, err)`, and its `models[]` row carries that name; it lives in onnxruntime-web's own WebAssembly memory, so the heap-to-heap copy per inference remains (`web/src/backends/ONNXRuntimeWebBackend.ts:269-357`), now over `anira_tensor_data_f32` and `anira_tensor_num_elements` instead of per-channel buffer calls, and it reads its model through `anira_model_config_model_path`/`model_bytes` in its Worker-side `init()` (section 5). (b) The compiled-in ORT is `ANIRA_ENGINE_ONNXRUNTIME` on `ANIRA_PROVIDER_DEFAULT`, single-threaded through the `USE_ANIRA_WEB` global thread pool (`src/backends/OnnxRuntimeProcessor.cpp:124-141`). Browser WebGPU is not a v3.0.0 deliverable, for schedule and measurement rather than architecture. ORT-Web ships four Wasm artefacts -- plain (CPU only), `.jsep` (TypeScript kernels, Asyncify-linked), `.asyncify` and `.jspi` (both the native Dawn WebGPU EP) -- and in every one capable of WebGPU `_OrtRun` returns a Promise, which settles only when the calling agent returns to its event loop. anira's inference Worker never returns to its loop (it is parked in `run_loop` for the whole session), so the completion cannot be observed *on that thread*; it can be observed on another. Two paths are live and neither touches the header: a **GPU proxy Worker** owning the WebGPU session, rendezvousing with the inference thread through `Atomics.wait` on the shared memory -- legal in a dedicated worker agent, illegal only on the main thread and in worklets -- which fits `anira_backend_desc.process` as tagged (`[inference-thread]`, may block); and a **JSPI-linked module**, which rewrites nothing (the real-time export is byte-identical to the plain build; only the listed async exports are wrapped) but is absent from Safari and contested on Chrome for Android, so it would reach fewer browsers than WebGPU itself. Asyncify is the mechanism that would rewrite the `ANIRA_NONBLOCKING` forwarders, and only Asyncify. What is missing is a number, not a design: no admission rule can be written before end-to-end block latency is measured, and the pinned `onnxruntime-*-WASM-static` archive is CPU-only, so this build cannot produce that number today. `ANIRA_PROVIDER_WEBGPU` on `ANIRA_ENGINE_ONNXRUNTIME`, `anira_machine_config_set_webgpu` and `anira_tensor_init_wgpu_buffer` stay in the header as the native Dawn path of M4 (`set_webgpu` returns `ANIRA_ERROR_NOT_SUPPORTED` on Emscripten in 3.0); the browser candidate arrives post-3.0 as an explicitly asynchronous JS backend -- an `anira_backend_desc` tail slot `process_async` plus a completion callback, driven by `anira_inference_thread_execute` from an unblocked Worker pump -- or as a JSPI-linked second module variant sharing the header, additive under ABI major 3 (section 10).

Log and version from JS. `anira_em_set_log_hook(1)` is the only sink installer on Wasm; `anira_log_desc.callback` and `anira_machine_config_set_log_sink` are ignored there. Every record, real-time or not, reaches `Module.anira.log(recordPtr)` of the instance that emits or drains it: main for drained RT records (a main-thread `setInterval(() => _anira_drain_log(), 10)`, which today's `drainAniraLog` helper exists for but nothing pumps), the Worker for control-path records emitted there (an ORT error inside `process`), which the TypeScript worker forwards to main by `postMessage`. TypeScript reads `group` and `message` through `layout.ts` and `UTF8ToString`. `_anira_abi_version()` and `_anira_check_abi(v)` are numbers, `UTF8ToString(_anira_version_string())` replaces the 256-byte scan of `web/src/helpers.ts:58-64`; `AniraWeb.create()` calls `_anira_check_abi(ANIRA_ABI_VERSION)` first and checks `Module.anira.version === ANIRA_EM_HOOKS_VERSION`.

The TypeScript package `@anira-project/anira` 3.x, ESM, same subpath exports as today:

| directory | content |
|---|---|
| `abi/` | `enums.ts`, `layout.ts`, `exports_wasm.txt` -- generated, never edited |
| `runtime/` | `createAniraWasm` over the `--emit-tsd` `MainModule` plus the `Module.anira` hook object; `Memory`; worker stacks via `_anira_em_alloc_thread_stack`; `Heap` with malloc/free, f32/u32/i64 views, string in/out and the one documented `>>> 0` rule for values ≥ 2^31; `Status`, throwing `AniraError(code, message)` on `ANIRA_FAILED` |
| `config/` | `TensorSpec`, `ModelConfig`, `MachineConfig`, `Contract` (`Hard`/`Async`), `JobOptions` -- builders, one C call per setter, `fromJson()`/`fromUrl()`, `destroy()` eagerly after handler creation |
| `core/` | `Machine`, `Pipeline`, `InferenceHandler` with `process*` and `processWait*`, `PlanReport`, `Capabilities`, `Ticket` (`submit(inputs, opts?, deadlineMs?, jobUserData?)`, `userData` reading the pointer back as a number), `Tensor` (host: owns its heap block, freed after a terminal `status()`), `InferenceThread` |
| `stages/` | `CustomStage` with `preProcess/postProcess/beforeInference/afterInference(ctx: StageContext)` and `prepare(handler, report)/release()`; `StageContext` reading `anira_stage_ctx` through `layout.ts`; `RingView` over `_anira_ring_*`, its element view the typed array `_anira_ring_dtype` names (`Float32Array` over an f32 ring, `Int16Array` over an i16 one, and so on) rather than a bare `Float32Array`; `TensorView` = `Float32Array` over `_anira_tensor_data_f32` sized by `_anira_tensor_num_elements` |
| `backends/` | `JSBackend` over `Module.anira.backend`, reading its `engineId` from the trampoline and its model row with it; `ONNXRuntimeWebBackend` |
| `workers/` | unchanged protocol; registries keyed by `js_key` |
| `AniraWeb.ts`, `index.ts` | entry points |

Fate of every current class:

| today | v3 |
|---|---|
| `VectorSizeT`, `VectorUnsignedInt`, `VectorFloat`, `VectorInt64T`, `VectorVectorInt64`/`TensorShapeList`, `VectorModelData`, `VectorTensorShape`, `VectorRingBuffer`, `VectorBufferF` -- ten classes, 43 `vector_*` exports (48 `EMSCRIPTEN_KEEPALIVE` entries in `src/emscripten-wrappers/utils/Vectors.cpp`, five of them `buffer_f_*`) | deleted; they existed only because the v2 constructors take `std::vector` by reference, and every builder call is one scalar |
| `ModelData`, `TensorShape`, `ProcessingSpec`, `InferenceConfig` | replaced by the `config/` builders `ModelConfig` and `TensorSpec` |
| `HostConfig` | deleted: `Contract.hard({ maxBlockSize, rate })`; the anchor moved to `ModelConfig` (section 5) |
| `BufferF` | `TensorView` |
| `RingBuffer` | `RingView` |
| `PrePostProcessor`, `JSPrePostProcessor` | `CustomStage`; `setInferenceHooks` becomes `hook_mask` at registration |
| `JSBackendBase` | `JSBackend` |
| `createInferenceBackend`, `createFactory`/`Factory<C>` (`web/src/utils.ts:71-87`), `resolvePtr`/`PossiblePointer` | deleted; engine and provider ids are constants of `enums.ts` and a custom engine is the name it was registered under, builders need no arity games |
| `InferenceHandler`, `InferenceThread` | stay, rewritten over the C entries; `getLatencyVector` reads caller memory, `processWait*` added, `setNonRealtime` gone (section 10) |
| `ONNXRuntimeWebBackend`, `AniraWeb`, `AniraAudioWorkletBase`, `setupInferenceWorker`, `bundleAudioWorklet`, the message protocol, the `performance` polyfill | stay |

Raw `_anira_ring_*` and `_anira_handler_process*` exports stay blessed on the real-time path, the rule `docs/sphinx/web-api/custom_pre_post_processing.rst:274-302` states today: a wrapper object per block is allocation pressure on the render thread. The worklet reads `anira_stage_ctx` with two `HEAPU32` loads from `layout.ts` and makes one `_anira_ring_pop_windows` call per block -- the one batched windowing call `anira-web-example/src/guitar-lstm/audio-worklet.ts:40-48` already makes today, reached through two further `_vector_*` crossings that `anira_stage_ctx` removes.

What the consumer writes. The current example builds fourteen handles of ten classes to describe one model (`anira-web-example/src/simple-gain-stereo/index.ts:26-61`):

```ts
const vectorModelData  = aniraWeb.VectorModelData([aniraWeb.ModelData(modelBuffer, aniraWeb.InferenceBackend.ONNX)])
const tensorShape      = aniraWeb.TensorShape(aniraWeb.TensorShapeList([[1, 2, 512], [1]]), aniraWeb.TensorShapeList([[1, 2, 512], [1]]))
const processingSpec   = aniraWeb.ProcessingSpec(aniraWeb.VectorSizeT([2, 1]), aniraWeb.VectorSizeT([2, 1]), aniraWeb.VectorSizeT([512, 0]), aniraWeb.VectorSizeT([512, 0]))
const inferenceConfig  = aniraWeb.InferenceConfig(vectorModelData, aniraWeb.VectorTensorShape([tensorShape]), processingSpec, 5, 10, false, 0, 1)
const hostAudioConfig  = aniraWeb.HostConfig(128, 48000, false, 0)
```

In v3 that is one call when the model ships its `model.json`, or one builder chain when it does not -- one call per model row and per tensor slot, each a single `_anira_model_config_*` crossing:

```ts
const cfg = await ModelConfig.fromUrl(anira, '/models/simple-gain-stereo.json')   // _anira_model_config_from_json; bytes injected with cfg.setModelBytes(0, modelBuffer)
// or, without a file:
const cfg = new ModelConfig(anira)
  .addModelBytesCustom('org.anira.onnxruntime-web', modelBuffer)
  .input (new TensorSpec('audio_in',  Abi.DTYPE_F32, Abi.ROLE_STREAMED).axis(0, Abi.AXIS_BATCH, 1).axis(1, Abi.AXIS_CHANNEL, 2).axis(2, Abi.AXIS_TIME, 512))
  .input (new TensorSpec('gain',      Abi.DTYPE_F32, Abi.ROLE_STATIC  ).axis(0, Abi.AXIS_FEATURE, 1))
  .output(new TensorSpec('audio_out', Abi.DTYPE_F32, Abi.ROLE_STREAMED).axis(0, Abi.AXIS_BATCH, 1).axis(1, Abi.AXIS_CHANNEL, 2).axis(2, Abi.AXIS_TIME, 512))
  .output(new TensorSpec('gain_out',  Abi.DTYPE_F32, Abi.ROLE_STATIC  ).axis(0, Abi.AXIS_FEATURE, 1))
const handler = machine.createHandler(Pipeline.inference(cfg, ['org.anira.onnxruntime-web']).registerEngine('org.anira.onnxruntime-web', new ONNXRuntimeWebBackend(cfg)))
cfg.destroy()                                                                      // copied at handler creation; the builder is free to go
handler.prepare(Contract.hard({ maxBlockSize: 128, rate: ctx.sampleRate }))       // HostConfig's four arguments, two of which survive
```

---

## 7. Planner, stages, chunkers

Stages declare a contract, the tensor stays passive: data flows, stages act, the planner decides where and when. A stage is a descriptor, not a class: `anira_stage_desc` -- `{struct_size, abi_version, user_data, domain_in, domain_out, consumed_kinds, num_consumed_kinds, reserved, pre_process, post_process, before_inference, after_inference, prepare, release}` -- handed once to `anira_pipeline_add_stage` and copied into a refcounted carrier; the inference stage is `anira_pipeline_add_inference` with its variant list and candidate set; the chain order is the order of the calls. Nothing about a stage is a vtable in libanira: the callbacks are the host's, `user_data` is the host's one slot (third member, its offset never moves; `anira.hpp` keeps a heap control block there, JS a registry key), and `release` fires exactly once, when the last carrier dies.

An execution-context requirement -- "this body needs the GL context current" -- has no field in v3.0.0, because every v3.0.0 stage is `{ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST}` and runs wherever its phase runs. It arrives as an appended `anira_stage_desc` tail slot with the first device-domain stage, `stage::GlCompute` (a GL compute pre-processor) being the candidate, and then `anira_handler_prepare` rejects a GL-context stage on a handler whose `anira_gl_threads` is `ANIRA_GL_CALLER_THREAD` under a Hard contract -- `pre_process` runs on the `[driver-thread]` there, and the borrowed context is current on the thread that called `anira_machine_create` (section 4). Without the slot that mismatch is a runtime failure instead of a prepare-time one, which is what the deferral costs and why the slot is named now.

`abi/stage.h` in full; conventions and thread tags as in section 6a.

```c
/* ================= abi/stage.h -- rings, stage context, stage and backend descriptors ==== */
typedef enum anira_stage_phase { ANIRA_PHASE_PRE_PROCESS = 0, ANIRA_PHASE_POST_PROCESS = 1, ANIRA_PHASE_BEFORE_INFERENCE = 2,   /* abi/enums.h, pinned at M1 */
    ANIRA_PHASE_AFTER_INFERENCE = 3, ANIRA_PHASE_PREPARE = 4, ANIRA_PHASE_RELEASE = 5, ANIRA_STAGE_PHASE_FORCE32 = 0x7fffffff } anira_stage_phase;
typedef struct anira_ring anira_ring;        /* anira-owned per streamed slot; storage typed by the host stream, never by the model tensor; accessors [cs] NB */
anira_dtype anira_ring_dtype(const anira_ring*) NB;                    /* the ring's element type, from the host stream and never inferred from the slot's spec dtype; every data accessor states the one it believes it is reading */
uint32_t anira_ring_num_channels(const anira_ring*) NB;
size_t   anira_ring_available(const anira_ring*, uint32_t ch) NB;      size_t anira_ring_available_past(const anira_ring*, uint32_t ch) NB;
/* dt != anira_ring_dtype(r): 0 returned and ANIRA_ERROR_CONFIG recorded in anira_handler_rt_error -- never a conversion, in either direction */
size_t   anira_ring_pop_block(anira_ring*, uint32_t ch, void* out, anira_dtype dt, size_t n) NB;              /* thl::core::RingBuffer::pop_block, tanh-lib RingBuffer.h:111 */
size_t   anira_ring_peek_past_block(const anira_ring*, uint32_t ch, void* out, anira_dtype dt, size_t n) NB;  /* receptive field */
size_t   anira_ring_push_block(anira_ring*, uint32_t ch, const void* in, anira_dtype dt, size_t n) NB;
size_t   anira_ring_push_fill(anira_ring*, uint32_t ch, const void* value, anira_dtype dt, size_t n) NB;      /* value points at one element of dt */
size_t   anira_ring_discard(anira_ring*, uint32_t ch, size_t n) NB;                                           /* no element crosses, so no dtype and no refusal */
size_t   anira_ring_pop_windows(anira_ring*, uint32_t ch, void* out, anira_dtype dt, size_t num_new, size_t num_old, size_t offset, uint32_t num_batches) NB;   /* elements written; PrePostProcessor.h:291-296 */
typedef struct anira_stage_ctx {             /* 64 bytes, Tier 1, no struct_size; on anira's stack for the callback's duration; tensor arrays re-pointed per call */
    uint32_t phase; uint32_t engine; uint32_t provider; uint32_t variant; uint32_t num_inputs; uint32_t num_outputs; uint32_t ticket; uint32_t reserved;   /* phase is an anira_stage_phase; engine and provider are the running plan's pair, engine a pipeline-scoped value for a registered custom engine; ticket is the submitting job's anira_ticket under Async, ANIRA_TICKET_INVALID under Hard */
    ANIRA_PTR(anira_ring* const, input_rings);      /* PRE only; NULL entries for non-streamed slots, and NULL throughout under Async, which has no ring (section 2) */
    ANIRA_PTR(anira_tensor, model_inputs);          /* PRE/BEFORE: write; Host domain in v3.0.0, the stage's domain_out from the minor that enables device-domain stages; model-shaped, anira-owned; Statics already materialised */
    ANIRA_PTR(anira_tensor, model_outputs);         /* POST/AFTER: read */
    ANIRA_PTR(anira_ring* const, output_rings);     /* POST only; NULL under Async for the same reason */
} anira_stage_ctx;
typedef anira_status (ANIRA_CALL* anira_stage_fn)(const anira_stage_ctx*, void* user_data) NB;   /* non-OK: pre = chunk completes as zeros + rt_error + one RT record; post = outputs zeroed; before/after = job FAILED (Async) or zeros (Hard) */
typedef struct anira_stage_desc {            /* {struct_size, abi_version, user_data, ...}: user_data's offset never moves; tail growth after release */
    uint32_t struct_size; uint32_t abi_version; void* user_data;
    uint32_t domain_in; uint32_t domain_out;        /* Host/Host in v3.0.0 */
    const char* const* consumed_kinds; uint32_t num_consumed_kinds; uint32_t reserved;   /* what this stage reads at prepare (section 1b) */
    anira_stage_fn pre_process, post_process, before_inference, after_inference;   /* NULL = anira's default for that phase */
    anira_status (ANIRA_CALL* prepare)(anira_handler*, const anira_plan_report*, void* user_data);   /* [main-thread]; may allocate; only [cs] and handler getters/set_input/get_output; cache what you need */
    void (ANIRA_CALL* release)(void* user_data);                                     /* [main-thread]; exactly once, when the last carrier dies */
} anira_stage_desc;
#define ANIRA_STAGE_DESC_INIT ANIRA_INIT(anira_stage_desc, sizeof(anira_stage_desc), ANIRA_ABI_VERSION, NULL, ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST, NULL, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL)
anira_status anira_stage_default_pre_process (const anira_stage_ctx*) NB;   /* [cs]; "call super": the receptive-field fill; ANIRA_ERROR_CONFIG where ring dtype != spec dtype, never a conversion */
anira_status anira_stage_default_post_process(const anira_stage_ctx*) NB;
#define ANIRA_BACKEND_NEEDS_NO_MODEL 0x1u      /* anira_backend_desc.flags: this engine forms a plan for every variant, model entry or not -- a passthrough, a mock, a benchmark baseline */
typedef struct anira_backend_desc {           /* custom engine: replaces BackendBase subclassing (backends/BackendBase.h:95-97) */
    uint32_t struct_size; uint32_t abi_version; void* user_data;
    const char* const* consumed_kinds; uint32_t num_consumed_kinds; uint32_t flags;      /* ANIRA_BACKEND_NEEDS_NO_MODEL, in the spare the layout already had */
    anira_status (ANIRA_CALL* prepare)(const anira_model_config*, uint32_t variant, uint32_t instances, void* user_data);   /* [main-thread], under the lifecycle mutex; once per variant, that variant's index beside its config; may allocate/load; NULL = JS engine, readiness is the TypeScript protocol */
    anira_status (ANIRA_CALL* process)(const anira_tensor* inputs, uint32_t num_in, anira_tensor* outputs, uint32_t num_out, uint32_t instance, void* user_data);   /* [inference-thread]; may block, no per-call allocation */
    void (ANIRA_CALL* release)(void* user_data);                                     /* [main-thread]; exactly once */
} anira_backend_desc;
#define ANIRA_BACKEND_DESC_INIT ANIRA_INIT(anira_backend_desc, sizeof(anira_backend_desc), ANIRA_ABI_VERSION, NULL, NULL, 0, 0, NULL, NULL, NULL)
```

```c
/* a host post-processing stage and a three-candidate inference stage; every test is ANIRA_FAILED, never != ANIRA_OK */
anira_stage_desc peak = ANIRA_STAGE_DESC_INIT;          /* domain_in = domain_out = ANIRA_DOMAIN_HOST in v3.0.0 */
peak.user_data = &pp; peak.post_process = peak_pick; peak.prepare = peak_prepare;
peak.consumed_kinds = NULL; peak.num_consumed_kinds = 0; /* a stage that reads a tensor or job extension names its kind here (section 1b) */

anira_pipeline* pipe; anira_pipeline_create(&pipe, &err);
const anira_model_config* variants[] = { cfg };
anira_backend_id candidates[] = {                                       /* engine and provider are independent axes; engine_id is NULL for a built-in engine */
    { sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU,  NULL },
    { sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA,    NULL },
    { sizeof(anira_backend_id), ANIRA_ENGINE_TFLITE,      ANIRA_PROVIDER_DEFAULT, NULL },
};
anira_pipeline_add_inference(pipe, variants, 1, candidates, 3, &err);   /* candidates NULL/0 = every engine the config names, on its default provider */
anira_pipeline_add_stage(pipe, &peak, &err);
/* a GL compute pre-processor is the same descriptor with {ANIRA_DOMAIN_GL_BUFFER, ANIRA_DOMAIN_GL_BUFFER}: the domain
   fields and values are pinned now; device-domain stages arrive additively, with the engine that reads the domain */
```

```cpp
// anira.hpp: the same pipeline, the same stage as a Stage subclass -- Custom's single-callable
// form binds pre_process and has no slot for a prepare, so a post_process stage is a class
anira::Pipeline pipe { anira::stage::Inference(cfg, {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU}, {ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA}, {ANIRA_ENGINE_TFLITE, ANIRA_PROVIDER_DEFAULT}}),
                       anira::stage::Custom(std::make_shared<PeakStage>(pp), ANIRA_DOMAIN_HOST, ANIRA_DOMAIN_HOST) };
```

At `anira_handler_prepare` the planner intersects producer domain, stage chain, backend capability report and terminal output domain against a sparse conversion-edge registry (GL<->CUDA register/map cached, GL->host readback or persistent map where the driver has host-pointer import, Vulkan external memory, dma-buf image import into Vulkan/EGL/Dawn, registration-cached NPU edges, pinned-staging fallback always available with fused dtype/layout conversion). The registry is filled by the Machine's driver probe (section 4), never by platform assumptions. Paths are composed, not looked up: between a producer domain and a consumer the planner searches the registry for every chain of rows, allocating any intermediate from its own pool with the fast-path recipe (a crossing currency of section 1 -- `OpaqueFd`, `DmaBuf`, an NT handle -- is the usual intermediate), and ranks the candidates by their worst edge class, so a `WgpuBuffer` bound for the CUDA EP goes `WgpuBuffer -> DmaBuf -> OpaqueFd -> CUDA EP` at `DeviceCopy` (two pool intermediates, because on NVIDIA one allocation cannot export both ways -- measured, below) when the Machine has a Vulkan device to mint them and `WgpuBuffer -> Host -> Cuda` at `HostCopy` when it has not; the plan report states the chain either way (`anira_plan_slot.recipe`), and the user never names the intermediate. Unreachable domains fail `anira_handler_prepare` with `ANIRA_ERROR_EDGE_UNREACHABLE` and the recipe in `anira_error.message`. The plan is fixed before processing; `process`/`submit` replay it.

The plan report is data with a fixed row shape, never a log line. `anira_handler_plan_report(h)` returns the handler-owned `anira_plan_report`, valid until the next prepare or destroy, and `anira_plan_report_slots(rep, plan, is_input, element_size, count, out)` fills one `anira_plan_slot` per (plan, slot): `{struct_size, slot, is_input, domain_in, domain_out, edge_class, allocate_class, wait_strategy, recipe, reason}` -- the edge taken and its class, the class an `allocate_*` handle would have gotten, the wait the edge implies (a GL map is a full wait, completion below), and why. `anira_plan_report_exts(rep, element_size, count, out)` lists `anira_plan_ext {struct_size, index, host, kind, consumer}` rows, one per consumed extension (`"entry" -> LibTorchAdapter`, section 1b); `anira_plan_info.budget_ms` carries the measured Hard budget of each plan; `anira_plan_report_to_json` is the dump a benchmark records. Rows follow the enumeration convention of section 6a: `out == NULL` returns the count, the caller passes its `sizeof` as `element_size` and the library writes `min(element_size, sizeof(lib's))` bytes per row at that stride, a short buffer fills what fits and returns `ANIRA_INCOMPLETE`; string members are valid for the call only. Tests assert classes per slot by walking these rows, which is also what `ANIRA_EDGE_COST_STRICT` (section 6) checks at prepare.

### Edge classes and the measured Linux GPU registry

Every edge carries a cost class, stated per slot in `anira_plan_slot.edge_class` so tests can assert it per edge: `ZeroCopy` (handle hand-over or memory import, no data movement), `DeviceCopy` (one GPU pass or copy-engine operation), `HostCopy` (readback and/or upload through staging) -- `ANIRA_EDGE_ZERO_COPY`, `ANIRA_EDGE_DEVICE_COPY`, `ANIRA_EDGE_HOST_COPY`, with `ANIRA_EDGE_UNAVAILABLE` for a row the probe refused. The class in the registry is the *functional* rung's result (section 4, the three rungs), never the static rung's: a row whose feature bits are present but whose round trip failed is unavailable, with the reason attached in `anira_edge_info.reason`, which `anira_capabilities_edge(caps, from, to, &info)` returns for any pair -- a row, never a bool.

Two crossing mechanisms, and everything about a row follows from which one it uses. *Reach-in*: same-process, nothing travels, no allocation-time decision -- a CUDA pointer on the primary context, a `WGPUBuffer` on the shared Dawn device, a GL buffer object registered with `cudaGraphicsGLRegisterBuffer`; any user handle is first-class. *Export*: the handle leaves its API as the platform's crossing currency -- a dma-buf fd or an opaque fd on Linux, an NT shared handle on Windows, an `AHardwareBuffer` on Android, an `IOSurface` on Apple -- decided at allocation time and never retrofittable, which is where `allocate_*` earns its place. The byte-image encoding below is the general answer wherever the importer takes textures and not buffers, which is Dawn's Vulkan *and* Metal backends both; only D3D12 is expected to escape it. A domain enters the enabled set with the first engine that reads it natively on shared hardware, or as a producer API anira allocates for; never for a producer's convenience alone. The distinction is what makes the registry portable while its rows are not: every platform has both mechanisms, and which one a given pair uses is a probed fact per driver. The rows below are Linux; the other platforms need the same matrix run before their rows are claimed (section 1, measurement status), and two are expected to differ structurally -- D3D12's `SharedBufferMemory` may make the Windows buffer import into WebGPU a true `ZeroCopy`, and Apple's image-typed currency may or may not need the byte image. Consumers measured as `{Host}` today, adding no domain: ExecuTorch's Vulkan backend (`execute()` copies every input from host `EValue`s through a staging buffer and its `Context` creates its own `VkDevice`; zero-copy I/O is pytorch/executorch#13382, open), ORT's CoreML EP, LiteRT's prebuilt WebGPU accelerator. Native CoreML/ANE reads `IOSurface`, Apple's crossing currency; an `IOSurface` buffer arm joins with the first engine that reads it, once measured.

What the user hands in, per engine (Linux, v1; measured on ORT 1.29 / LiteRT 2.2 out of tree -- the tree pins ORT 1.26.0 / LiteRt 2.1.5, `cmake/backends.cmake:86,90` -- and every GPU row in this section is such a measurement, not code in the tree):

| user hands in | CPU EP | WebGPU EP | CUDA EP |
|---|---|---|---|
| host floats | ZeroCopy | HostCopy (`writeBuffer`) | HostCopy (pinned staging) |
| dma-buf fd (`DmaBuf`) | ZeroCopy (mmap) | DeviceCopy (import + relayout) | unavailable on desktop (`DMABUF_FD` import is Jetson Thor only) |
| opaque fd (`OpaqueFd`) | -- | -- | ZeroCopy (`cudaImportExternalMemory`) |
| `WGPUBuffer` on anira's device | HostCopy (`MapAsync`) | ZeroCopy | DeviceCopy, measured as the three-hop bridge (Dawn relayout into a dma-buf pool tensor, `vkCmdCopy` into an opaque-fd pool buffer, CUDA maps that; Vulkan block required), else HostCopy (`MapAsync` + pinned staging) |
| CUDA pointer, primary context | HostCopy | -- | ZeroCopy |
| plain `VkBuffer` | HostCopy (map / staging) | HostCopy | HostCopy |
| `VkBuffer` from `allocate_*` | ZeroCopy where the dma-buf mmaps (UMA), else HostCopy | DeviceCopy | ZeroCopy when allocated opaque-fd (measured) -- one export kind per allocation: `compatibleHandleTypes` is one bit on NVIDIA (measured), so a plan set with both GPU candidates needs two allocations, and a dma-buf-exported allocation reaches CUDA at DeviceCopy through a second opaque-fd buffer and one `vkCmdCopy` |
| GL buffer object | HostCopy (`glGetBufferSubData`) | HostCopy: GL has no buffer export | ZeroCopy by registration + map (borrowed GL context; implicit sync in map/unmap) -- **unmeasured**: the prototype's GL domain is the dma-buf-backed renderbuffer of the next row, which reaches CUDA via host |
| GL renderbuffer from `allocate_*` (dma-buf-backed) | HostCopy (`glReadPixels`) | DeviceCopy | -- (a different object, not a differently allocated one) |

Output rows mirror these (bound outputs, below). The CUDA cells were measured on 2026-08-27 on an NVIDIA Turing GPU (ORT 1.29 built `--use_cuda`, CUDA 13, cuDNN 9; `hello_inference --src/--ep/--dst cuda`, every cell bit-exact and stale-checked): `OpaqueFd -> CUDA` is `ZeroCopy` (`cudaImportExternalMemory` on an `OPAQUE_FD` allocation, opaque-fd binary semaphores in both directions); a Vulkan allocation exports as dma-buf *or* opaque fd, never both, so the WebGPU-to-CUDA hand-off is the **three-hop bridge** -- Dawn's relayout pass into a dma-buf pool tensor, one `vkCmdCopy` into an opaque-fd pool buffer, CUDA mapping that -- at `DeviceCopy` with no host bytes, cached per tensor. Dawn's own opaque-fd `SharedTextureMemory` import, which would collapse the three hops into one, was not exercised and stays to be measured as an optimization, not a row. The `GlBuffer -> CUDA` registration row is unmeasured (table). Engine facts from the same run, for the adapter: the CUDA EP is handed anira's non-blocking stream as `user_compute_stream` and run with `disable_synchronize_execution_providers`, so `Run` is submission there too and the stream is the token; `enable_cuda_graph` records on the third run (warm-up >= 3) and is refused for a graph with any node off the CUDA EP, so the adapter recreates the session without it and says so; `cudnn_conv_algo_search=HEURISTIC` by default (exhaustive autotunes every conv shape per session). ORT's CUDA memory info carries the device ordinal as its `OrtDevice` id -- unlike the WebGPU EP's constant 0 (completion, below) -- and the same silent-copy trap applies when it disagrees with the EP's `device_id`.

Reference rows, measured on Linux (Mesa Honeykrisp, Dawn from ORT 1.29, LiteRT 2.2 prebuilts, out of tree; the tree pins ORT 1.26.0 / LiteRt 2.1.5); other drivers produce other rows, which is the point of probing:

| edge | mechanism | class |
|---|---|---|
| `WgpuBuffer -> ORT WebGPU EP` (same device) | pass the `WGPUBuffer` as a `WebGPU_Buffer` OrtValue; ordering by queue submission | ZeroCopy |
| `Host -> ORT WebGPU EP` | `writeBuffer` | HostCopy |
| `VulkanBuffer` / `GlBuffer -> WebGPU` | Dawn's Vulkan backend has no buffer import (`SharedBufferMemory` exists for D3D12 only) and imports dma-bufs as textures in 8/10/16-bit formats only, so the memory travels as an **RGBA8 byte image**: one texel holds the four little-endian bytes of one float, written `unpackUnorm4x8(floatBitsToUint(f))` and restored `bitcast<f32>(pack4x8unorm(t))` — exact, because unorm8 round-trips k/255. Import as `SharedTextureMemory`, one texel->buffer pass. Requires memory allocated exportable (below) | DeviceCopy |
| `Frame(DmaBuf, NV12) -> WebGPU` | `SharedTextureMemory` import (`R8BG8Biplanar420Unorm`, per-plane views) + the `FrameToTensor` pass | ZeroCopy import; the stage's own pass |
| `Frame(DmaBuf, NV12) -> Vulkan` | `VkImage` with DRM modifier + YCbCr sampler | ZeroCopy import; stage pass |
| `OpaqueFd` (Vulkan memory exported `OPAQUE_FD`) / `GlBuffer -> CUDA` | `cudaImportExternalMemory` (measured on NVIDIA) / `cudaGraphicsGLRegisterBuffer` + map (unmeasured) | ZeroCopy |
| `Cuda -> WgpuBuffer` (measured, NVIDIA) | the three-hop bridge: CUDA writes the opaque-fd pool buffer on its stream, one `vkCmdCopy` moves it into a dma-buf pool tensor, Dawn imports that as `SharedTextureMemory` and runs the texels -> buffer pass into the engine's fixed `WGPUBuffer`; ordered by opaque-fd semaphores and the sync file. Dawn's opaque-fd descriptor would make it one hop (to be measured). What makes a CUDA producer reach the WebGPU EP without the host | DeviceCopy |
| `DmaBuf -> CPU EP` | the buffer's own `mmap`; the CPU engine reads the pages in place (sync bracket only) | ZeroCopy |
| any other -> CPU EP | map / `MapAsync` / `glGetBufferSubData` | HostCopy |
| `ORT WebGPU EP -> WgpuBuffer` (bound output, same device) | the consumer's `WGPUBuffer` bound as the output OrtValue; the engine writes it in place; `output_ready` is queue order (see completion, below) | ZeroCopy |
| `ORT WebGPU EP -> VulkanBuffer` / `GlBuffer` / `DmaBuf` (bound output) | the consumer's dma-buf imported as `SharedTextureMemory` with `StorageBinding`, one packed-floats -> texels pass, `EndAccess`'s fence handed to the consumer as `output_ready` (a sync file: Vulkan waits on it as a semaphore, GL as an `EGLSync`) | DeviceCopy |
| `WgpuBuffer -> Cuda` (measured, NVIDIA) | the mirror of the row above: the packed-floats -> texels pass into a dma-buf pool tensor, `EndAccess`'s sync file waited on by Vulkan, one `vkCmdCopy` into the opaque-fd pool buffer, an opaque-fd `VkSemaphore` that CUDA waits on with `cudaWaitExternalSemaphoresAsync`. The source may be any `WGPUBuffer` on the Machine's Dawn -- the engine's fixed output or a user's -- so this is how a WebGPU EP output reaches the CUDA EP at `DeviceCopy`, the planner inserting both pool tensors (path composition, above). On Windows the same hand-off may be `ZeroCopy`: a shared D3D12 resource is a real buffer to Dawn (`SharedBufferMemory`) and importable by CUDA -- section 1, measurement status | DeviceCopy |
| `ORT CPU EP -> DmaBuf` (bound output) | the consumer's `mmap` bound as the host output; "ready" is closing the CPU write window (`DMA_BUF_IOCTL_SYNC` end + the x86 flush) | ZeroCopy |
| `ORT CPU EP -> VulkanBuffer` / `GlBuffer` / `WgpuBuffer` | map or staging upload / texture upload + blit into the bo / `writeBuffer` | HostCopy |

The output rows are the mirror image of the input rows and were measured the same way (prototype: all 50 (source, EP, destination) cells bit-exact on the synthetic model, error 1e-4 on the palm detector; the two `DeviceCopy` bodies are one import used in one direction each, with the fence that gates Dawn's access being the producer's `ready` on the way in and the consumer's `released` on the way out). Two facts the mirror exposed. The byte image's rows must be 64-byte aligned: a linear dma-buf packed at 18-float rows imported without complaint and was written at the pitch the driver rounded to, so the consumer read garbage (`max_abs_err` 175 on every dma-heap output) -- the image is the edge's encoding, not the tensor's shape, so the edge picks aligned rows (an exact factorisation of the element count, or a padded tail no pass reads) and the floats stay packed. The image is computed from the element count alone, so nothing about it lives on the Tensor; the memory must be packed at that pitch (a driver that refuses the aligned pitch makes the row unavailable -- padded rows are image rows, not tensor rows, and no `strides` can say so), and it can be larger than `n * 4`: `allocate_*` sizes for it, and a user-exported buffer must be at least `anira_machine_byte_image_bytes(m, n, dtype)`. And the headline cell `VulkanBuffer -> WebGPU EP -> VulkanBuffer` -- a camera-fed model rendered by Vulkan, the stage-five shape -- costs `DeviceCopy` twice, one texture round trip each way; that number on paper is the case for the upstream contribution below.

Exportability is a producer-side precondition, not an edge capability. The two dma-buf export rows above exist only for memory that was *allocated* exportable, and no edge can retrofit that: a `VkBuffer` on ordinary `VkDeviceMemory` has no fd to hand out, and the allocation must carry `VkExportMemoryAllocateInfo` plus — since Dawn imports textures — a linear DRM-modifier `VkImage` bound to the same memory, with the row pitch the driver accepts, before the tensor is ever written. The prototype allocates image and optional buffer alias together and writes through the alias, so the generator is an ordinary SSBO shader that never learns an image exists; where the driver refuses the alias (dedicated allocation, disjoint memory types) the writer falls back to `imageStore` into the same memory. Consequences for v3: `anira_tensor_init_vulkan(t, buf, mem, off, timeline, value, ...)` on user memory reaches WebGPU only through `HostCopy` and the plan report says so; `anira_handler_allocate_input`/`allocate_output` (section 6) are how a Vulkan user gets the row without learning the recipe -- anira allocates on the user's device, exportable, image tag bound, buffer alias, aligned rows, image-sized, and the user's shader writes an ordinary SSBO. anira's own pool tensors are allocated the same way.

GL has no buffer export at all, so `Domain::GlBuffer -> WebGPU` does not exist as written: `glGetBufferSubData`/`glMapBufferRange` is the only way out of a GL buffer object, i.e. `HostCopy`. A GL producer that wants the DeviceCopy row must allocate its storage as a `gbm_bo` and import it back into GL (EGLImage + renderbuffer + FBO) — which makes it a `Frame{Container::DmaBuf}` with a GL view, not a `Domain::GlBuffer`. Two traps, both measured: on Mesa/agx an `imageStore` into an EGLImage-backed *texture* passes `glReadPixels` and leaves the dma-buf zeroed (the driver keeps a tiled shadow it never flushes back — Apple GPUs do not write linear images), so the producer must *render* into a renderbuffer; and the modifier GBM chose travels with the fd and must reach the importer, since a "linear" assumption is wrong on exactly those drivers. `GlBuffer -> CUDA` is unaffected: `cudaGraphicsGLRegisterBuffer` is same-process registration, not export.

Upgrade path noted, not assumed: a `SharedBufferMemory` dma-buf importer for Dawn's Vulkan backend would turn the `VulkanBuffer -> WebGPU` row into ZeroCopy and remove the byte-image encoding entirely; tracked as an upstream contribution.

Two-reader ownership: `input_released` (`anira_handler_ticket_input_released`, a non-owning view until `ticket_release`) signals when *all* consumers in the plan are done with an input (a camera Frame read by both the app's presenter and the inference edge); the planner composes the fences. Across handlers the composition is the user's: two handlers reading one buffer (the composed-models pattern below -- a detector and a landmark model sampling the same camera frame) each signal their own `input_released`, and the producer's reuse condition is their conjunction. The prototype ANDs the tracker's hold into the capture stream's per-buffer fence and `cam_stream` never learns that a second reader exists; that is the seam, not a new API.

### Stage catalogue: FrameToTensor

`stage::FrameToTensor(PixelFormat, target_size, Crop | Letterbox, normalization, layout)` with built-in kernels for `{NV12, YUYV, UYVY, RGBA8, BGRA8}` on each consumer API `{WebGPU (WGSL), Vulkan (SPIR-V), GL (GLSL), CUDA}`; multi-planar sampling is per API (Dawn plane-aspect views, Vulkan YCbCr sampler, GL `samplerExternalOES`). Any other format is a user stage with the same declaration. The stage output is a Tensor in the consumer's buffer domain; everything downstream is buffers.

Deferred past v1 (section 1a); it arrives with `abi/draft/frame.h` and `submit_frame`, additive under ABI major 3. When it ships, its kernels run where anira already has a device -- WGSL on the Machine's Dawn, C on the host -- so the stage adds no domain; a SPIR-V or CUDA kernel arrives with an engine that reads that domain. A same-process image the user never exported stays theirs to convert.

Two things the prototype's two-model pipeline forces on the declaration before it ships. **The crop is per job, not per stage.** The palm detector wants the whole frame letterboxed or centre-cropped into 192x192; the landmark model wants a rotated, expanded square around the previous result in 224x224; and both are one kernel -- walk the destination, map through a 2x3 affine, gather the source -- handed different six floats. `Crop | Letterbox` are therefore presets that compute that affine from the frame and target sizes, and the affine itself is a job extension, `ext::CropAffine` set through `anira_job_options_set_ext` on the options of `submit_frame` (section 1b), named in the stage's `consumed_kinds` and rejected at submit on a handler without one; without it the second handler of every two-model vision pipeline cannot use the built-in stage. The direction is destination -> source, which is also the direction that maps a model's output back onto the frame, so decoding needs no inverse. **`border_mode` is per model, not per stage type:** MediaPipe's palm graph fills outside the frame with zero (the letterbox bars *are* black) and its landmark graph replicates the edge, and using zero for the landmark crop collapses presence exactly when the hand comes close enough for its ROI to run off the frame -- a flicker that arrives with proximity and nothing else. Both measured: the C and WGSL kernels handed the same affine and the same YUV coefficients agree to max |d| ~2e-5, which is what makes a downstream disagreement between two execution providers attributable to the model rather than the input.

### Plan sets and runtime backend selection

The inference stage declares a candidate set (above: `{ONNXRUNTIME, WEBGPU}`, `{ONNXRUNTIME, CUDA}`, `{TFLITE, DEFAULT}`), and `anira_handler_prepare` compiles **one plan per candidate**: each with its own conversion chain (GL->dma-buf->Dawn for the WebGPU EP, GL->CUDA registration for the CUDA EP, GL->host readback for CPU), its own registrations and staging pools, all validated and preallocated upfront. The set is sparse: a (variant, candidate) pair whose variant carries no model entry for that candidate is not a plan and is not an error, so a wide candidate set beside a config that names one engine simply yields fewer plans. A backend declaring `ANIRA_BACKEND_NEEDS_NO_MODEL` in its `flags` is exempt and forms a plan for every variant regardless -- the shape of a passthrough, a mock, or the zero-inference baseline a benchmark measures against -- and a real engine whose named model file will not load is still `ANIRA_ERROR_MODEL_LOAD` at prepare: needing no model and missing a model are different, and neither may be read as the other. `anira_handler_set_plan` therefore means *selection, not reconfiguration*: a relaxed atomic store (`[thread-safe] ANIRA_NONBLOCKING`; v2 does exactly this today, `InferenceManager.cpp:37`), an atomic switch among precompiled plans, effective at the next chunk (Hard) or the next job (Async). No planning ever happens at runtime; determinism survives because only selection does. The input feeds every plan whose engine its domain reaches -- which is what keeping graphics handles representable until the adapter boundary was for -- but not necessarily every plan at the same class, and on NVIDIA not always at all: a Vulkan allocation exports as dma-buf *or* opaque fd, and a GL producer reaches CUDA with a buffer object but WebGPU only with a rendered-into dma-buf. `anira_handler_allocate_*` therefore resolves against the whole enabled candidate set: one object that serves every candidate when such an object exists, otherwise the one serving the most, with the degraded candidates named per slot in `anira_plan_slot.allocate_class` and `reason` -- the price of a wide candidate set, visible at prepare like the Hard budget of its slowest member. A user who needs both fast rows submits one handle per plan. With model variant sets, candidates generalize to (variant, backend) pairs; see Multi-model support below.

Costs, stated explicitly at prepare: every enabled plan's resources stay resident. v2 already constructs and `prepare()`s a processor for every backend that has a `ModelData` entry, in the handler constructor (`Context.cpp:531-556`), pooled across sessions of equal config unless session-exclusive; v3 extends this to staging and registration state. Under Hard, worst-case honesty applies across the set: the budget is measured per plan during warmup (`anira_plan_info.budget_ms`; today it is the user-supplied `max_inference_time` and nothing is measured -- a v2 file's figure survives as `ANIRA_BUDGET_EXPLICIT`), the single latency reported to the host (`anira_handler_get_latency`) covers the slowest enabled plan, and all Hard validations (no-wait reachability, and whatever an adapter adds for the extensions it consumes) must pass for every candidate, else `ANIRA_ERROR_BUDGET`. A Hard handler that wants live switching buys headroom for its worst candidate; shrink the candidate set to shrink the price.

Async additionally admits per-job selection as a job extension (`ext::JobBackend` through `anira_job_options_set_ext`, section 1b; default = handler-level choice): live A/B against identical frames, automatic fallback when a device saturates, and closed-loop adaptation driven by ticket telemetry (miss rate climbs under thermal throttling, the app shifts inference to the NPU mid-session, the render loop never notices). Handler-level switching at safe boundaries is the v1 commitment; the per-job extension waits for a demonstrated need (reversibility rule).

Benchmarking follows from declarativeness: the contract JSON sweeps scheduling policy, the model JSON (or a `default_engine` override) sweeps engines, one unmodified binary loops over the grid, and each run emits met/late/dropped rates, per-window inference times from warmup, and the compiled plan (`anira_plan_report_to_json`: which edges, where staging landed) so results explain themselves. In-process backend cycling gives perfect code-path comparability but couples runs through GPU clocks and thermal state; randomize order or restart per run (the file sweep does this for free) for publishable numbers. Every report records the CPU frequency governor and the wait strategy alongside the plan: a blocking GPU wait under `schedutil` inflates everything the CPU does around the inference by 3-5x (section 4), so a run without those two fields cannot be compared with another. Reports also state, per job, whether the result was verified fresh -- a stale-output check (completion, below) is what separates a measurement from an artefact. The v2 google-benchmark fixture (`ProcessBlockFixture`) drives this grid from `examples/benchmark/` over `anira.hpp`; it is not part of the library and links nothing into it.

### Multi-model support

Two meanings of "multiple models", handled at two different levels, plus the shared infrastructure both use. A handler owns exactly one inference stage; that stage may hold several *variants*, and running several models in *sequence* is several handlers. Part of this is v2: one model per backend per config (`InferenceConfig(std::vector<ModelData>, ...)`, `InferenceConfig.h:486`), selected at runtime by `set_inference_backend`, and several handlers on one `Context` with pooled processors. What is missing, and what follows adds: several variants on the same backend, `set_model`, tensor names, in-plan chaining and device-memory hand-off.

**Model variants (alternatives) -- inside one handler.** `anira_pipeline_add_inference` takes a variant list, several configs on the same or different backends, and the candidate set in one call:

```c
const anira_model_config* variants[] = { cfg_small, cfg_large };
anira_backend_id candidates[] = { { sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA,    NULL },
                                  { sizeof(anira_backend_id), ANIRA_ENGINE_TFLITE,      ANIRA_PROVIDER_DEFAULT, NULL } };
anira_pipeline_add_inference(pipe, variants, 2, candidates, 2, &err);
// anira.hpp: anira::stage::Inference({cfg_small, cfg_large}, {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA}, {ANIRA_ENGINE_TFLITE, ANIRA_PROVIDER_DEFAULT}})
```

The plan-set machinery generalizes unchanged: candidates become (variant, engine, provider) triples, `anira_handler_prepare` compiles one plan per triple that has a model, and `anira_handler_set_plan(h, plan)` is the same atomic safe-boundary selection over the dense index the report enumerates -- selection, never planning, and one store rather than one per axis, so no sequence of calls can land on a triple that has no plan. The setter stores the request and never touches a ring; the clear a Stateful switch needs (rings and struct pool, work that must run on the session's driving thread) executes at the next Hard entry on the driver thread, or at the next job boundary on the inference thread under Async -- defined, logged through the RT queue. The selection index is a relaxed atomic today and, natively, `thl::RCU` at M5 (internal; on Wasm the atomic stays, because `thl::RCU` keys its readers on thread-local state the Wasm build has none of). Validation at prepare: variants must agree on the external I/O contract (axes, roles, dtypes after de/quantization); windows and latencies may differ per variant since each owns a plan, but under Hard the worst case across all enabled (variant, backend) pairs sets the budget and the reported latency. Per-job variant choice arrives as a job extension, `ext::JobModel` beside `ext::JobBackend` (section 1b), not as a reserved field.

**Composed models (sequencing) -- across handlers, through tickets.** Running model B on model A's output is two handlers on one `Machine`, sequenced by the user's loop, and it costs nothing an in-plan chain would have saved. A's `anira_handler_allocate_output(h_a, slot, domain, &t, err)` is a handle on the Machine's shared device, so it is a legal `submit` input on B (section 6: any handle is accepted), and A's `output_ready(i)` (`anira_handler_ticket_output_ready`, a view valid until A's ticket is released; `anira_sync_token_dup` when B's job outlives it) becomes that tensor's `acquire`. The edge B takes is the same registry row as any other -- `WgpuBuffer -> WebGPU EP` ZeroCopy, `Cuda -> CUDA EP` ZeroCopy, a cross-engine pair `HostCopy` through pinned staging -- and B's plan report states it. On one queue the token is `QueueOrdered` and B waits for nothing; the host drains once, at the end of the sequence. Under Hard, `anira_handler_get_latency` values add. Under Async each handler carries its own contract, which is itself a reason to compose this way: a detector with a deadline feeding a landmark stage that has none of its own. Whatever runs between the two models -- a decode, a non-maximum suppression, a crop affine derived from A's output -- is the user's code, either on the host after `anira_handler_ticket_wait`/`anira_handler_poll` (a priced `HostCopy`, chosen, never imposed) or as the user's own pass on the Machine's queue between the two submits (queue-ordered, no token needed).

Two things this gives that a chain inside one plan cannot. **Control flow:** whether B runs at all, how many times, and against which of A's results is application logic. MediaPipe's hand graph gates its detector on the previous cycle's landmark count and runs the landmark model once per associated rect; no linear stage list expresses that, and a stage graph with conditionals and loops is a graph runtime, not an inference library. **A visible intermediate:** what settles a disagreement between two models is dumping what sat between them -- every state-machine bug in the prototype's tracker was found that way, and an intermediate that is a plan-internal buffer is exactly the thing that cannot be printed.

Chained inference stages inside one pipeline -- A's output as a plan-internal buffer feeding B -- are deferred past v1, on the additive list. Their zero-copy benefit is already free above. Two same-engine graphs with fixed dataflow are better merged offline, where the engine fuses across the boundary and a runtime chain never can. The one thing only an in-plan chain can do -- joint window selection and latency composition for two *streamed* models with different hops under Hard, so that one re-buffering disappears -- waits for a demonstrated model pair (reversibility rule, as for per-job backend selection above). If it returns it is an edge between two plans that already exist, which is why deferring it costs nothing.

**Shared engine environments** (section 4): one runtime environment per engine, Machine-owned, shared by every handler in the process. That is not how v2 works -- each `OnnxRuntimeProcessor::Instance` owns its own `Ort::Env` (`OnnxRuntimeProcessor.cpp:71`) and sharing happens only through processor pooling for identical configs -- and it changes with the machine at M2; per-engine session sharing (ORT shared allocators, prepacked weights) is the adapter-level optimization that makes many-model setups memory-viable. One measured session option belongs here rather than in any single handler: ORT's intra-op pool spin-waits after its work is done, so two sessions that never overlap in time still halve each other -- the palm detector 11.8 ms alone, 23.4 ms beside an *idle* landmark session, ~10 ms with `session.intra_op.allow_spinning = 0` (or one pool for all sessions via `CreateEnvWithGlobalThreadPools`). Invisible to any single-model benchmark, which is why it was found only when the second handler arrived; the shared environment is where anira sets it. The tree pins `SetIntraOpNumThreads(1)` (`OnnxRuntimeProcessor.cpp:146`) and, on the web, `CreateEnvWithGlobalThreadPools` (`:134`), so the measurement describes the multi-threaded intra-op pool a v3 contract may open, not today's configuration.

Chunker selection (spec Role + arrival mode; contract fixes only the entry point):

| arrival \\ spec        | Streamed                                                                                      | Buffer                       |
|-----------------------|-----------------------------------------------------------------------------------------------|------------------------------|
| incremental (process) | RingChunker (today's ring buffers)                                                            | accumulate-to-full or reject |
| complete (submit)     | ViewChunker (byte_offset views, refcounted parent, head trim + tail flush in its reassembler) | IdentityChunker              |

Hard uses RingChunker only, with input rings and output rings sized independently so that either side may be empty (section 5, one-sided streaming; today `calculate_send_buffer_sizes`/`calculate_receive_buffer_sizes`, `SessionElement.cpp:443-444`, which push 0 for a non-streamable tensor): a generator has output rings only and `process()` pulls, an analyser has input rings only and its Static outputs bypass the rings. Below the chunker everything is uniform: lanes, pump, stages and backends see model-shaped tensors and never know their origin.

Rings stay inside anira. The RingChunker's storage is the `thl::core::RingBuffer<T>` behind `SessionElement::m_send_buffer`/`m_receive_buffer`, `T` the element type of the host stream on that slot -- what the driver pushed into an input ring, what the driver will pop out of an output ring -- and never the model tensor's `dtype`, which stays on the spec. A ring's element type is the ring dtype the host declared for its slot on the Hard contract (`anira_contract_hard_set_ring_dtype`, `ANIRA_DTYPE_F32` when unset), and the typed Hard entries of section 6 carry that type across the ABI; the `float` entries are forwarders for the F32 default. The ring crosses the ABI only as an opaque `anira_ring*` with the ten accessors declared above -- among them `anira_ring_pop_windows`, the batched windowed pop that is v2's `pop_samples_from_buffer(..., num_batches)` (`PrePostProcessor.h:291-296`) -- all `[callback-safe] ANIRA_NONBLOCKING`. The accessors are typed with the storage, which is the host's stream: the ring knows its element type, `anira_ring_dtype(r)` reports it, and every data accessor takes a `void*` plus the `anira_dtype` the caller believes it is reading or writing. **A stage reads `anira_ring_dtype(r)` and never infers a ring's element type from its slot's spec `dtype`** -- the two are independent quantities, and it is the sentence third-party stage code compiled once against 3.0.0 freezes against. Disagreement is refused -- 0 returned, `ANIRA_ERROR_CONFIG` recorded in `anira_handler_rt_error`, nothing written -- and never repaired: **no ring accessor converts between element types, in either direction, ever**, so a stage that wants floats out of an `int16` capture stream converts in its own body, where the cost is visible and the rounding is its own; the same body in the other direction -- a float host stream feeding an `int8` model -- is what `ext::Quant`'s (de)quant stage will be. A converting stage cannot pop straight into `model_inputs`, because the typed accessors refuse a foreign dtype on both sides: it pops into scratch it sized and allocated in its own `prepare` (`[main-thread]`, may allocate) and writes the model tensor through `anira_tensor_data(t, dtype)`, so nothing allocates on the `[driver-thread]` and Hard's no-allocation rule (section 3) holds. The cost is not a constant: the receptive-field history stays in the ring and is re-read on every inference (`peek_past_block` then `pop_block`, `PrePostProcessor.cpp:114-127`), so a differing dtype re-converts `window`/`hop` times the fresh hop -- 32x at a 4096-element window and a 128-element hop -- unless the stage keeps its own converted history, which duplicates exactly the machinery `anira_ring_pop_windows`'s `num_old` exists to avoid. `anira_ring_discard` is the one data-free entry and takes no dtype. A live view over the ring is not offered because none could be stable: `thl::core::RingBuffer` keeps its positions in a `std::vector<size_t>` and `m_is_full` in a `std::vector<bool>`, and tanh-lib is invisible on every platform (section 6b). A stage sees its world through `anira_stage_ctx`, a 64-byte Tier-1 POD without `struct_size`, on anira's stack for the duration of the callback: `{phase, engine, provider, variant, num_inputs, num_outputs, ticket, reserved, input_rings, model_inputs, model_outputs, output_rings}`, `phase` one of the six `anira_stage_phase` values and the same number the browser hook object's `phase` argument carries (section 6b) -- `input_rings` for `PRE_PROCESS` only (NULL entries for non-streamed slots), `model_inputs` written by `PRE_PROCESS`/`BEFORE_INFERENCE`, `model_outputs` read by `AFTER_INFERENCE`/`POST_PROCESS`, `output_rings` for `POST_PROCESS` only; the tensors are Host domain in v3.0.0, the stage's `domain_out` from the minor that enables device-domain stages, model-shaped, anira-owned. `ticket` is the submitting job's `anira_ticket` under Async and `ANIRA_TICKET_INVALID` under Hard -- job identity is what lets a stage reach its own per-job record, the one it filled from the job's extensions at submit (a third-party `FrameToTensor`-style stage reading its crop affine, section 1b), which a context without it cannot address; it takes the spare `uint32_t` the layout already had, while unpacking `backend` into `engine` and `provider` grows the struct from 56 to 64 bytes -- a Tier-1 change, made before the freeze, and `abi/layout-3.txt` records 64. **`model_inputs`/`model_outputs` are re-pointed on every call**, with `anira_tensor_init_host` on the calling thread from one preallocated `anira_tensor` array per pooled struct: LibTorch and TFLite `swap_data` the struct's memory with the engine's scratch on every inference (`LibTorchProcessor.cpp:175`, `TFLiteProcessor.cpp:146`) and the struct a call sees is chosen per call, so nothing about a tensor descriptor is cached across calls, by anira or by any wrapper over it. Static tensors are the handler's, not the stage's: values set through `anira_handler_set_input`/`set_static_input` are materialised into `model_inputs` before any `pre_process` runs and Static outputs are copied out of `model_outputs` after `post_process`, independent of the stage (v2 did this inside the default `PrePostProcessor`; a frozen 64-byte context has no handler pointer to do it through). `anira_stage_default_pre_process(ctx)`/`_post_process(ctx)` are the "call super" hatch -- the receptive-field fill and the plain pop -- so a stage that touches one slot forwards the rest. They convert no more than the accessors beneath them: on a slot whose ring dtype and spec `dtype` differ they touch nothing and return `ANIRA_ERROR_CONFIG`, so a configuration that needs a conversion needs a real stage to do it. The worklet reads the context with two `HEAPU32` loads from `layout.ts` and makes one `_anira_ring_pop_windows` call per block, the single crossing it has today.

Phase placement and failure are fixed by the descriptor. `pre_process`/`post_process` run on the `[driver-thread]` under Hard and the `[inference-thread]` under Async; `before_inference`/`after_inference` on the `[inference-thread]` between dequeue and engine call, where recurrent-state splicing under Stateful belongs; `prepare(handler, report, user_data)` on the caller of `anira_handler_prepare`, may allocate, and is where a stage caches what it needs (its slot indices, the classes the report assigned); `release` once, on the caller of the last destroy. All four phase slots are `anira_stage_fn`, `ANIRA_NONBLOCKING` in the typedef so a blocking body is flagged in the consumer's own build. A non-OK status from `pre_process` completes that chunk as zeros at its stream position (v2's `complete_with_zeros`, `SessionElement.cpp:237`), records the status in `anira_handler_rt_error` and emits one RT log record; from `post_process` it zeroes that chunk's outputs; from `before`/`after_inference` it marks the job `ANIRA_TICKET_FAILED` under Async and zeros under Hard. From inside any of them a stage may call exactly the `[callback-safe]` entries -- the `anira_tensor_*` accessors, `anira_ring_*`, `anira_stage_default_*`, `anira_log_rt`, `anira_now_ms`/`anira_now_ns`, `anira_handler_rt_error`, the ticket accessors -- and nothing else; `prepare` additionally the handler getters and `set_input`/`get_output`, never a lock-taking entry, because stage and backend `prepare` run under the lifecycle mutex. A stage that reads a tensor or job extension names its kind in `consumed_kinds`; the consumed-or-fail walk of section 1b is over that list.

Custom engines are the same shape one level down: `anira_backend_desc {struct_size, abi_version, user_data, consumed_kinds, num_consumed_kinds, flags, prepare, process, release}`, registered by name with `anira_pipeline_register_engine(pipe, engine_id, &desc, err)` and named as a candidate by that id. `prepare(model_config, variant, instances, user_data)` runs `[main-thread]` under the lifecycle mutex and may load -- it fires once per variant, the variant's index beside that variant's config, so a custom engine can serve a multi-variant plan set, one plan row per variant; it reads its own model through `anira_model_config_model_path`/`model_bytes`; `process(inputs, num_in, outputs, num_out, instance, user_data)` runs on the `[inference-thread]` over `anira_tensor` arrays, may block on its engine and must not allocate per call; `release` fires once. `flags` carries `ANIRA_BACKEND_NEEDS_NO_MODEL` where an engine has no model entry to find -- the built-in roundtrip (`src/scheduler/SessionElement.cpp:52-53`), a test mock, the advanced benchmark's `ClearCustomProcessor` -- and then forms a plan for every variant, where a descriptor without it forms a plan only for the variants whose config names its id. It replaces subclassing `BackendBase` -- `process(std::vector<BufferF>&, std::vector<BufferF>&, std::shared_ptr<SessionElement>)` (`backends/BackendBase.h:95-97`), whose session argument no engine uses. A `prepare` of NULL means readiness is the host's business, which is how the browser's `onnxruntime-web` backend registers (section 4). Descriptors are copied at registration into a refcounted carrier that pipeline, handler and pooled processor share, so a descriptor may live on the caller's stack and `release` fires exactly once per registration, when the last copy of that carrier dies, on the thread that destroys it -- one descriptor registered under two ids is two carriers and two `release` calls, and a host that reuses one descriptor is written for that. The pooled processor's key is (config, engine id, carrier) and never the config alone: two pipelines may register different descriptors under one id with equal configs, and v2 pools by config equality (`src/scheduler/Context.cpp:1131-1145`), which would hand the second handler the first's engine.

v2 `PrePostProcessor` subclasses run through `anira::v2::LegacyProcessorStage(pp)` in `compat/v2.hpp`: consumer side, header-only, deprecated on arrival, removed one minor after 3.0. `anira::v2::PrePostProcessor` keeps the four v2 virtuals over `std::vector<v2::RingBuffer>&`/`std::vector<v2::BufferF>&`, where `v2::RingBuffer` and `v2::BufferF` are non-owning views with the v2 method names over the C accessors, reserved at `prepare` and re-pointed from `anira_stage_ctx` on every call; `set_input`/`get_output` forward to the handler received in `prepare`. Source-compatible with subclasses that use the block API and `set_input`/`get_output`; not with code that reached into `thl::core` storage (`swap_data`, `get_memory_block`) or per-sample `push_sample`/`pop_sample` -- those callers migrate. It includes no tanh header, which is why `utils/{Buffer,RingBuffer,MemoryBlock}.h` leave the public tree.

Completion is a fence, never a return value. Measured with ORT's WebGPU EP and graph capture enabled: `Run()` returns ~8 ms before the GPU finishes and host-bound outputs contain the previous job's results on every iteration; without capture `Run()` blocks correctly. The adapter therefore never derives `output_ready` from the engine call returning; it binds outputs in the engine's device domain, obtains the engine's completion fence (queue work-done future, `cudaEvent`, a sync file) and exposes that as the token. Engine features that make execution asynchronous (graph capture, replay, deferred readback) are allowed only through this path.

That path was then built and measured, and the stale outputs turned out to be the adapter's, not the engine's. A replay run in ORT (`inference_session.cc`) skips the framework entirely — no feed copy, no fetch, no host download — and re-dispatches only the captured compute bind groups, against the exact device buffers they were recorded with. The contract a captured graph therefore imposes on the adapter is stricter than "await a fence": **every byte the graph touches must be device-resident, at a stable address, and written by nobody but the caller and the graph.** Three ways to violate it, each measured as stale outputs: host-bound outputs (their download is framework work); an input whose `OrtMemoryInfo` device id does not match the EP's internal `OrtDevice` (id 0 for the WebGPU EP, regardless of the `deviceId` option, which only selects the `WGPUDevice`) — the framework then `MemCpy`s it into an ORT-owned buffer before the graph, a copy neither captured nor repeated, so the graph reads run 0's data forever; and producer-side multi-buffering, since the graph references one of the rotating buffers. With all three satisfied — device-bound outputs fetched after the queue work-done fence, input memory info on the EP's device id, one input buffer — ORT 1.29's WebGPU capture is exact and never stale, across the identity and the dma-buf edges. It is still not worth enabling on this hardware: ~3% on the large model, slower on the small one. Consequences for the planner: capture is a per-plan mode that *disables* multi-buffering on the input side (the two optimizations are mutually exclusive, and the plan report says which one it chose); the adapter's device-id mapping is queried from the EP's allocator, never taken from a session option; and a feature is enabled only when the warmup's stale check has *demonstrated* freshness under it, recorded as a measured bit in the capability report rather than assumed from the completion contract. Test harnesses vary the input per job and flag outputs identical to the previous job's (the hello_inference "stale" check); a benchmark that cannot detect a stale frame measures nothing.

Measured with the outputs bound on the device and completion taken from the token (prototype, palm detector, M1): the engine call is submission, not completion -- ORT's WebGPU EP spends 2-3 ms of *host* time per `Run()` encoding the graph and the GPU another ~7 ms. Graph capture, the trap above, becomes legitimate in this arrangement under one further condition: **the buffers bound to the engine are fixed**. ORT replays the bind groups it captured, i.e. the buffers of the run it captured on, exactly as a CUDA graph replays addresses; with the engine's input and output tensors rotating between two slots, replay alternates fresh and stale outputs (10 of 23 iterations, caught only by a stale check that compares against the last *N* outputs, not the last one). With one fixed slot at the engine, `Run()` falls from 3.3 ms to 1.1 ms of host time, the GPU time is unchanged and the outputs are fresh. Consequence for the planner: the producer-side multi-buffering of section 1 stays on the producer's side of the edge -- under capture the edge moves each rotating slot into fixed engine tensors, and a hand-over edge (the same buffer on both sides) cannot rotate. The 4x capture seemed to give was the missing wait; the 3x it gives is real, and it is submission overhead, not GPU time.

A trap in the ORT adapter that the same check exposed: the OrtDevice id on a `WebGPU_Buffer` memory info must be 0 -- the constant the EP's allocator reports (`webgpu/allocator.h`: `WebGpuDevice{GPU, DEFAULT, NONE, 0}`) -- and not the `deviceId` provider option that selects the Dawn device. A tensor labelled with any other id is on a foreign device as far as the session is concerned: it is copied into an EP-owned buffer before every run and copied back after, a `DeviceCopy` hidden under a `ZeroCopy` row that no cost table shows, and one a captured graph does not replay (the replay read ORT's private copy from capture time; every capture-on cell was stale until the ids were told apart). The version assertion of section 4 has a sibling here: an adapter should assert that its memory info compares equal to the EP allocator's, at session creation, and refuse otherwise -- the shape of today's `throw_if_foreign_onnxruntime()` (`OnnxRuntimeProcessor.cpp:43-55`), which the exception firewall turns into `ANIRA_ERROR_ENGINE` from `anira_handler_create`.

What `output_ready` is for a WebGPU consumer: Dawn exports `SharedFence`s only from `SharedTextureMemory::EndAccess`, so for a plain `WGPUBuffer` there is no fence to hand out. The host-side token is the queue's work-done future; a consumer on the same queue needs no token at all, because submission order is the guarantee. Neither is `SyncKind::None` ("already visible" -- it is not, to the host), so `anira_sync_kind` has `ANIRA_SYNC_QUEUE_ORDERED` beside `ANIRA_SYNC_NONE` (section 1): `anira_handler_poll`/`ticket_wait` block on the future, a same-queue GPU consumer proceeds without waiting, and the adapter never fabricates a fence it does not have. CUDA: the token is a `cudaEvent` recorded on the engine's stream after the run (`user_compute_stream` puts ORT on anira's stream, so one event covers edge and engine). A GL consumer reached by registration has no fence either: `cudaGraphicsUnmapResources` orders CUDA's writes before GL's subsequent commands, so `output_ready` is `QueueOrdered` after the unmap, and the map on the way in is a full wait on GL's pending work for that buffer -- coarser than a fence, stated in the plan report (`anira_plan_slot.wait_strategy`).

Backends and capability reports, v1. Every backend is a pinned `anira_engine` value paired with a pinned `anira_provider` value, both independent of `USE_*` (today `InferenceBackend`'s values shift with the build, `utils/InferenceBackend.h:32-110`) -- and a new engine or a new provider is an appended value below `_FORCE32` plus an `ANIRA_ABI_MINOR` bump, never a layout change; `anira_enabled_backends` says which of them this build compiled in, `anira_capabilities_backends` which are usable on this machine. The five v2 engines on the CPU -- ONNX Runtime, LibTorch, TFLite, LiteRT, ExecuTorch (`ANIRA_ENGINE_ONNXRUNTIME`, `LIBTORCH`, `TFLITE`, `LITERT`, `EXECUTORCH`, each on `ANIRA_PROVIDER_DEFAULT`) -- so that no deployment loses coverage (section 10), plus the GPU providers where the Machine has the device: `ONNXRUNTIME` on `ANIRA_PROVIDER_WEBGPU` (IOBinding on `WebGPU_Buffer` memory info, the Machine's native Dawn; the browser is a different path, where the engine is `onnxruntime-web`, registered by name through `anira_em_pipeline_register_js_engine`, and browser WebGPU is post-3.0, section 4), `ONNXRUNTIME` on `ANIRA_PROVIDER_CUDA` (IOBinding on `user_compute_stream`), `ONNXRUNTIME` on `ANIRA_PROVIDER_DIRECTML` (`ID3D12Resource` through the `DML` memory info -- DirectML *is* the D3D12 consumer; there is no separate Direct3D provider), and `LIBTORCH` on `ANIRA_PROVIDER_CUDA` (DLPack both directions). `ANIRA_DOMAIN_METAL_BUFFER` is a pinned value whose rows are enabled with the first engine measured to read an `MTLBuffer` in place -- LibTorch MPS is the candidate, ORT's CoreML EP is host-only -- and not before, per the rule that no row exists without a measurement; until then `anira_tensor_init_metal` lives in `abi/draft/`. Capability reports are *queried*, not tabulated: ORT via EP memory infos, LibTorch via device and DLPack support, at `anira_machine_create` and `anira_machine_probe`, read back through `anira_capabilities_edges`. Later minors, each bringing the extensions it consumes (section 1b): LiteRT non-host TensorBuffers (GL / AHB / Metal, fence-based async, on Android and Apple -- the CompiledModel + TensorBuffer API is already what the LiteRT adapter uses for host memory, `src/backends/LiteRtProcessor.cpp`, so only the buffer types are pending; measured on desktop Linux its prebuilt WebGPU accelerator accepts only `WebGpuBufferPacked`, rejects dma-buf and cannot adopt the Machine's Dawn, so it is a `{Host}` consumer there, and its CL path is never taken -- section 1), TensorRT and CoreML (`ext::Artifacts`, `ext::ArtifactCache`), and NPU adapters (`ext::Npu`, `ext::NpuHard`, plus registration edges and the artifact cache). Quantized I/O likewise waits for `ext::Quant` and its (de)quant stage; in v1 a quantized model runs only when the producer hands in the model's true dtype.

Kernel quality is per (engine, model class, driver) and is the reason plan sets exist. Same GPU, same Vulkan driver, same model (MediaPipe palm detection, fp32, all nodes on the GPU, no fallback): LiteRT's ML Drift accelerator runs it in 2.2 ms while ORT's WebGPU EP needs 8.8-9.9 ms -- the same as ORT's CPU EP on that machine's eight cores. ML Drift is tuned for exactly this class of mobile convolution network (PHWC4 layouts, fused ops, per-GPU-family kernels); the WebGPU EP's strengths are matmul/attention shapes, and its WGSL passes through Tint -> SPIR-V -> the platform driver. Neither engine is "the GPU backend": which one wins is a measurement per model class and platform, which is what a candidate set plus the benchmarking sweep is for. Two corollaries for the planner: never present a GPU backend as faster than a CPU backend without a measured budget for that model, and remember that an equal-time GPU plan is still a win when the point is *offloading* the CPU -- but measure the host share, it is not the ~150 us a bare queue submit suggests: on the palm detector ORT's WebGPU EP costs the host 2-3 ms of submission for ~7 ms of GPU time, 1 ms with graph capture, so the plan hands back ~7 of ~9.5 ms, not ~9.4 (completion, above). That is still the usual case under a Hard contract; the budget just has to be the measured one.

---

## 8. JSON schemas

Three files, three lifetimes, no field in two of them. Loaders are dumb (strings to enums, numbers, construct); all semantic validation happens once in `anira_handler_prepare` / `anira_machine_create`, identical for JSON and code. Handles and host-discovered geometry are patched from code, last write wins.

A loader is a C function over UTF-8 text, the second constructor of the handle it fills: `anira_model_config_from_json(utf8, len, base_dir, &cfg, err)` and `anira_model_config_from_json_file(path, &cfg, err)`, `anira_machine_config_from_json(utf8, len, &mc, err)`, `anira_contract_from_json(utf8, len, &c, err)`. The model file has two entry points because its `path` entries are relative to the file, and `base_dir` is where they resolve (`from_json_file` derives it); the other two files carry no paths and read text only, the file open being the caller's -- the browser has no file to open and the plugin's JSON is a `tanh_add_binary_data` blob. All three are produced by the same private `nlohmann::json` code that never reaches a header, on every target, and all three return `ANIRA_SUCCESS_UPGRADED` for a version 2 document (section 8.4): a positive status, which is why the stable test after a loader is `ANIRA_FAILED(st)` and never `st != ANIRA_OK` (section 6a). A loader failure -- unparseable text, a wrong JSON type for a known key, a string outside a key's vocabulary -- is `ANIRA_ERROR_JSON` with the key path and the offending value in `anira_error.message`; nothing is logged and dropped, which is what the tree does today (`src/utils/JsonConfigLoader.cpp:280-327`). `anira_model_config_to_json` and `anira_machine_config_to_json` write the handle back into a caller-owned buffer in v3 spelling only, so reading a v2 file and writing it out is the migration tool.

As built at M1, `anira.hpp` declares no `anira::JsonConfigLoader` (the v2 class of that name is still in every example TU until the shims of M2): the loaders are `ModelConfig::from_json(text, base_dir)` / `from_file(path)`, `MachineConfig::from_json` / `from_file` and `ContractHandle::from_json` / `from_file`, throwing `anira::Error` (or returning `anira::Result`) on `ANIRA_FAILED`; `upgraded()` reports a version 2 document and `take_legacy_contract()` returns `std::optional<ContractHandle>`. The `JsonConfigLoader::model/machine/contract` spelling of the sketches may return with the handler half. Nothing about JSON is exported from libanira except the C functions above.

### 8.1 Model file (`model.json`, travels with the model)

```json
{
  "models": [
    { "engine": "onnxruntime", "path": "model.onnx",
      "tensors": { "audio_in": "input_0", "mask_out": "output_0" } },
    { "engine": "libtorch", "path": "model.pt",
      "tensors": { "audio_in": "x", "mask_out": { "name": "y", "layout": [0, 2, 1] } },
      "entry": { "name": "forward_streaming" } },
    { "engine": "de.tu-berlin.coreml", "path": "model.mlpackage",
      "tensors": { "audio_in": "input_0", "mask_out": "output_0" } }
  ],
  "default_engine": "onnxruntime",
  "state": "stateless",
  "max_instances": 4,
  "anchor": { "output": "mask_out" },

  "inputs": [
    { "name": "audio_in", "dtype": "float32", "role": "streamed",
      "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
      "window": { "min": 2048, "max": 8192 }, "context": 1024 }
  ],
  "outputs": [
    { "name": "mask_out", "dtype": "float32", "role": "streamed",
      "axes": [ ["batch", 1], ["channel", 2], ["time", "dynamic"] ],
      "window": { "min": 2048, "max": 8192 }, "context": 1024, "latency": 512 }
  ]
}
```

A `models[]` entry is tagged by engine alone. `"engine"` is a lower-case built-in name -- `"onnxruntime"`, `"libtorch"`, `"tflite"`, `"litert"`, `"executorch"` -- or the reverse-URI name a custom engine was registered under, `"de.tu-berlin.coreml"` above, the required dot being the same reservation third-party extension kinds make (section 1b), so a custom name can never collide with anira's own short ones. No provider is ever written in this file: which file to load is a model semantic and which device runs it is a machine resource, so one `.onnx` file is named once and serves every provider candidate of its engine. `"default_engine"` names an engine too, never a pair -- `anira_model_config_set_default_engine` is the code half -- and absent it is `ANIRA_ENGINE_NONE`, `models[0]`. Where a *candidate* is named as a string -- the plan report `anira_plan_report_to_json` writes, an error message naming one -- the spelling is `engine` or `engine:provider`, a bare name meaning `ANIRA_PROVIDER_DEFAULT`: `"onnxruntime:cuda"`, `"onnxruntime:webgpu"`, `"onnxruntime:directml"`, `"libtorch:cuda"`, `"de.tu-berlin.coreml:coreml"`. A dotless string outside the five is `ANIRA_ERROR_JSON` in a v3 document, because only a built-in name can have been meant; the upper-case v2 names are accepted on the upgrade path only (section 8.4). A custom name is not checkable at load -- registration is a code call, and the loader has no pipeline -- so an unregistered one fails `anira_handler_prepare` by name, exactly as an unknown extension kind does. An engine that is in the list but not in this build is stored, not dropped, and fails `anira_handler_prepare` with `ANIRA_ERROR_NOT_SUPPORTED` if it is a candidate -- with no candidate list every engine named in the file is one, so a host that wants one file to serve every build names its candidates from `anira_enabled_backends` (section 7).

Extension keys (section 1b) sit beside the core keys of the object they extend -- `"entry"` on a model entry above; later `"quant"` on a tensor, `"artifacts"` on a model entry, `"ort_session"` options -- and resolve through the extension registry, each key being the extension's `kind` and the optional `"version"` member of the object its layout revision (default 1): `"entry": { "version": 1, "name": "forward_streaming" }` is the explicit spelling of the example. A known kind at an unregistered version is the one thing a loader rejects on semantic grounds, `ANIRA_ERROR_EXTENSION_VERSION` at load, because it can be decided without a build. A key the loader does not know is neither dropped nor a load error: it is stored as `{anira_ext_header, raw text}` (`ext::Unknown` internally) and `anira_handler_prepare` fails by name -- `ANIRA_ERROR_EXTENSION_UNKNOWN` for a kind no registry row has, `ANIRA_ERROR_EXTENSION_UNCONSUMED` for a kind no stage in the plan reads -- with the key and its host in `anira_error.message`. A typo in `"max_instances"` therefore fails prepare with the misspelt key named, not silently with a default. A v2 file's `model_function` is upgraded into `"entry": { "name": ... }` with the one-time warning of section 10; a v2 file without one produces an empty bag.

Byte injection. A `models[]` entry loaded from JSON is a path entry; `anira_model_config_set_model_bytes(cfg, index, bytes, size, ownership, release, ctx, err)` patches it into a bytes entry -- `anira_model_config_model_path` returns NULL for it afterwards, nothing opens the file, and `anira_model_config_model_bytes` hands the bytes to any backend, native or JS, that reads its own model. The rule exists because the two deployments the file travels to most have no filesystem to speak of: the browser fetches `model.json` and `model.onnx` as two responses (`ModelConfig.fromUrl` then `setModelBytes`), and a plugin embeds both with `tanh_add_binary_data` and borrows the blob for the life of the DSO (`ANIRA_BYTES_BORROW`, section 5). The `"path"` string stays in the file so that `to_json` round-trips it and a desktop build of the same plugin still finds the model on disk:

```c
/* model_json and model_onnx are the blobs tanh_add_binary_data compiled into the DSO */
anira_model_config* cfg = NULL;
anira_status st = anira_model_config_from_json(model_json, model_json_size, NULL, &cfg, &err);
if (ANIRA_FAILED(st)) { host_log(err.message); return false; }        /* ANIRA_SUCCESS_UPGRADED passes */
anira_model_config_set_model_bytes(cfg, 0, model_onnx, model_onnx_size,
                                   ANIRA_BYTES_BORROW, NULL, NULL, &err);   /* models[0] now reads from memory */
```

### 8.2 Machine file (`machine.json`, lives on the box)

```json
{
  "num_threads": 0,
  "wait_strategy": "spin_backoff",
  "log": { "level": "warning", "drain": "thread", "queue_capacity": 512, "drain_interval_ms": 10 },

  "cuda":   { "device": 0, "pinned_pool_limit": 67108864 },
  "vulkan": { "device": 0 },
  "metal":  { },

  "gl":     { "threads": "caller_thread" }
}
```

`vulkan.device` names a device index that `anira_vulkan_desc` has no slot for (open item 2 of section 12): the M1 loader keeps it on the handle, and the M4 machine selects the physical device by it when it creates the device (a borrowed device carries its own).

`"num_threads": 0` means bring your own threads, as it does in the tree and as the Wasm build requires; the library default is the absent key (`ANIRA_THREADS_AUTO`, section 4). `"log"` is the block the tree already parses (`level`, `drain`, `queue_capacity`, `drain_interval_ms`, the four scalar setters of section 4); the bare `log_level` key is the version 2 spelling and is upgraded, not accepted, in a v3 document. The sink is a callback and is code-only (`anira_machine_config_set_log_sink`; `anira_em_set_log_hook` on Wasm). Device blocks in JSON imply `ANIRA_OWNERSHIP_OWNED` (anira creates); `"d3d12"` and `"webgpu"` are spelled the same way and a block for a device this build cannot drive fails `anira_machine_create` with `ANIRA_ERROR_NOT_SUPPORTED`, which is the Machine-construction half of the validation rule above. `"npu"` and `"artifact_cache_dir"` return as extension keys with the adapters that consume them (section 1b); until then a file that carries them fails `anira_handler_prepare` by name. Borrowed handles are code-only, patched onto the loaded handle by the device setters, which replace the block they name:

```c
anira_machine_config* mc = NULL;
anira_machine_config_from_json(text, len, &mc, &err);
anira_gl_desc gl = ANIRA_GL_DESC_INIT;                 /* threads = ANIRA_GL_CALLER_THREAD, what the file said */
gl.display = user_egl_display;                         /* code completes what JSON declared */
gl.context = user_egl_context;
gl.gbm     = user_gbm_device;                          /* optional: dma-buf-backed GL storage from allocate_* */
anira_machine_config_set_gl(mc, &gl);
anira_machine* m = NULL;
anira_machine_create(mc, &m, &err);                    /* validates: indices exist, borrowed non-null */
anira_machine_config_destroy(mc);                      /* the machine copied it */
```

```cpp
auto mc = anira::JsonConfigLoader::machine("machine.json");   // == anira::MachineConfig::from_file
mc.gl(gl);                                                    // the same anira_gl_desc
anira::Machine machine(mc);
```

### 8.3 Contract files (name the run; the file you sweep in experiments)

```json
{ "async": {
    "deadline_ms": 33.3,
    "on_late": "drop",
    "priority": "auto",
    "lanes": 0,
    "max_in_flight": 0,
    "delivery": "polled"
} }
```

```json
{ "hard": {
    "block_min": 512,
    "block_max": 512,
    "rate": 48000,
    "budget": "measured",
    "warmup": "until_stable",
    "on_miss": "bypass",
    "wait_ratio": 0
} }
```

Dual encodings: `"budget"` is `"measured"` or `{"ms": 1.8}`; `"warmup"` is `"until_stable"`, `"none"`, or `{"fixed": 200}`; omitted `"deadline_ms"` is the offline posture (`{"async": {}}`). `"wait_ratio"` is v2's `blocking_ratio` (section 3): `0` is the default and `> 0` selects the semaphore wait of the `_wait` twins; it is a contract key because it decides a completion primitive at `prepare`, not a property of the model. The root names the kind (`anira_contract_get_kind`); a file with both roots or neither is `ANIRA_ERROR_JSON`. Hard geometry keys are optional: fixed-rate deployments write them, plugins patch from the host:

```c
anira_contract* c = NULL;
anira_contract_from_json(text, len, &c, &err);
anira_contract_hard_set_geometry(c, host_block_size, host_sample_rate);   /* the host patches what the file left out */
anira_handler_prepare(h, c, &err);
anira_contract_destroy(c);                                                /* the handler copied it */
```

```cpp
auto c = anira::JsonConfigLoader::contract("session.json");   // anira::Contract = std::variant<Hard, Async>
std::get<anira::Hard>(c).block_min = host_block_size;   // a fixed-block host: min == max earns the tight latency
 std::get<anira::Hard>(c).block_max = host_block_size;
std::get<anira::Hard>(c).rate           = host_sample_rate;
handler.prepare(c);                                           // mints the anira_contract at prepare
```

Optional in either contract file, top level: `"edge_cost": "strict" | "permissive"` -- `anira_contract_set_edge_cost` (section 6), because this is the file a test sweep runs and `prepare` takes one object.

Never in JSON: per-submit overrides and JobOptions, callbacks, planner-derived values, runtime Tensors.

### 8.4 Version 2 documents

A version 2 document is one file with two roots, `context_config` and `inference_config`; either root marks it, no schema-version key exists in the tree and none is added, because a v3 file is recognised by its own roots. Each loader reads the block it owns and returns `ANIRA_SUCCESS_UPGRADED`: `anira_model_config_from_json` reads `inference_config` and holds back the three keys that are contract fields, which `anira_model_config_take_legacy_contract(cfg, &c)` hands out afterwards as a Hard contract without geometry (non-NULL only after an upgrade; the caller destroys it); `anira_machine_config_from_json` reads `context_config`; `anira_contract_from_json` given the same document yields that legacy contract directly, for a host that never touches the model config. The upgrade logs one `ANIRA_LOG_WARNING` per process (group `anira.config`) naming the file and the three keys that moved, so a TS or C host can surface it once through the status and the sink sees it once. The v2 loader's tolerance -- unknown keys ignored, disabled backends dropped -- does not survive the upgrade path either: an unknown v2 key is stored and fails `prepare` by name like any other.

| version 2 key | version 3 destination | rule |
|---|---|---|
| `inference_config` root | read by `anira_model_config_from_json` | marks a v2 document |
| `context_config` root | read by `anira_machine_config_from_json` | marks a v2 document |
| `model_data[]` | `models[]` | |
| `model_data[].model_path` | `models[].path` | relative to `base_dir` |
| `model_data[].inference_backend`: `"ONNX"`, `"LIBTORCH"`, `"TFLITE"`, `"LITERT"`, `"EXECUTORCH"` | `models[].engine`: `"onnxruntime"`, `"libtorch"`, `"tflite"`, `"litert"`, `"executorch"` | upper-case accepted here only; a v2 name is an engine and only an engine, because v2 had no provider to name |
| `model_data[].inference_backend`: `"CUSTOM"` | `models[].engine`: `"anira.v2.custom"` | the reserved name `compat/v2.hpp` registers a v2 custom backend under, so a v2 file keeps loading; a v3 file names its own engine (section 7) |
| `model_data[].model_function` | `models[].entry.name` | absent = `forward` (LibTorch, ExecuTorch) |
| `tensor_shape[].input_shape` / `output_shape` (universal entry) | `inputs[].axes` / `outputs[].axes` | the axis that carries the per-channel element count (the window) is `time`, else the trailing axis (v2 lays channel `i` at `[i * size, (i + 1) * size)`); the last other axis whose extent equals the channel count is `channel`; every other axis `any` |
| `tensor_shape[]` entry with `inference_backend` | a `layout` in the `tensors` record of that backend's model entries | one canonical spec per tensor in v3; an entry that holds the same axes in another order (only unit axes moved) becomes a layout; one that changes an extent other than 1 fails the upgrade by name (`ANIRA_ERROR_JSON`) |
| `processing_spec.preprocess_input_channels[i]` / `postprocess_output_channels[i]` | the `channel` extent of `inputs[i]` / `outputs[i]` | |
| `processing_spec.preprocess_input_size[i]` / `postprocess_output_size[i]` | `inputs[i].window` / `outputs[i].window` with `min == max` = the per-channel element count (elements / channels); `context` = that count minus the size (the receptive-field rule) | `0` = `"role": "static"` |
| `processing_spec.internal_model_latency[i]` | `outputs[i].latency` | |
| `num_parallel_processors` | `max_instances` | absent: the 2.x constructor default (half the hardware threads, at least 1), so an upgraded document runs as it ran |
| `session_exclusive_processor` | `state`: `true` = `"stateful"`, `false` = `"stateless"` | |
| `max_inference_time` (ms) | legacy contract `hard.budget: {"ms": ...}` | `ANIRA_BUDGET_EXPLICIT` |
| `warm_up` (count) | legacy contract `hard.warmup: {"fixed": ...}` | `ANIRA_WARMUP_FIXED`; absent: `{"fixed": 0}`, the 2.x constructor default |
| `blocking_ratio` | legacy contract `hard.wait_ratio` | section 3 |
| `context_config.num_threads`, `wait_strategy` | `num_threads`, `wait_strategy` | unchanged spelling and meaning |
| `context_config.log_level` (anira <= 2.2) | `log.level` | |
| `context_config.log{}` | `log{}` | unchanged |

`anchor` is the model's clock: the canonical name of one streamed tensor, `"anchor": "mask_out"` (canonical names are unique across inputs and outputs, so no side is needed), absent meaning the default, the first streamed input, else the first streamed output; a name that is not a streamed tensor of this model fails `anira_handler_prepare` with `ANIRA_ERROR_CONFIG`. `time_ratio` is the per-tensor `[num, den]` beside `axes` when a tensor's Time axis runs at a rate other than the anchor's; absent is 1:1. `anchor` has no v2 key: v2's clock lived in `HostConfig` (code-only) and the upgrade leaves the default, which is the tree's resolution. `HostConfig::allow_smaller_buffers` maps to the block range of section 3: clear upgrades to `block_min == block_max`, set upgrades to `block_min` 1 with `block_max` the v2 buffer size, which is exactly what the v2 sweep computed. The v2 C++ loader, `anira::v2::JsonConfigLoader(path)` with `get_context_config()` / `get_inference_config()`, is a `compat/v2.hpp` shim over these functions for one minor (section 10).

---

## 9. Usage sketches

Every sketch is written against `anira.hpp`, the header-only C++20 wrapper over `anira.h`; each line of it is one C call, and the C call is the contract. The Hard sketch is additionally given in raw C, as a CLAP plugin embedding static libanira, and in TypeScript, because those two hosts never see `anira.hpp`. Conventions that hold in every sketch: a failure is `ANIRA_FAILED(status)` and its text is in a caller-owned `anira_error` (the wrapper throws `anira::Error`, or returns `anira::Result<T>` under `ANIRA_CXX_NO_EXCEPTIONS`); `submit` moves every owning `acquire` token into the job slot; the per-job deadline and the per-job `void*` are arguments of `submit`, never of `JobOptions`; a ticket is a `uint32_t` value that `anira::Ticket` wraps so the slot is released when the wrapper dies; tensors from `allocate_input`/`allocate_output` are pool tensors that anira never releases -- they go back through `free_tensor` or die with the next `prepare()`/destroy.

### Hard: audio plugin (v2-identical hot path)

```cpp
// C++ (anira.hpp)
auto cfg = anira::ModelConfig::from_file("model.json");                       // v2 file: auto-upgraded, one warning
anira::Machine machine{anira::MachineConfig::from_file("machine.json")};
anira::InferenceHandler handler{machine, anira::Pipeline{ anira::stage::Inference(cfg) }};

// prepareToPlay:
anira::PlanReport report = handler.prepare(anira::Hard{ .block_min = host_block, .block_max = host_block, .rate = host_rate });   // throws anira::Error("extension 'quant' on tensor 'audio_in' ...")
set_latency(handler.get_latency());

// audio callback:
handler.process(channel_ptrs, n);                                              // inline -> anira_handler_process, ANIRA_NONBLOCKING, never waits
```

The same sequence in raw C, placed into the CLAP slots it belongs to. `ANIRA_STATIC` is on, so nothing of `anira_*` reaches the plugin's export table; `clap_entry` stays the DSO's one symbol.

```c
/* raw C -- a CLAP plugin embedding static libanira. Every test is ANIRA_FAILED, never != ANIRA_OK. */
anira_error err = ANIRA_ERROR_INIT;   /* one per slot, never static: the initialiser is a compound literal, not a constant expression */

/* clap_entry.init [main-thread]: the header this plugin compiled against must be served by this library */
bool clap_init(const char* path) { return ANIRA_SUCCEEDED(anira_check_abi(ANIRA_ABI_VERSION)); }

/* clap_plugin.init [main-thread]: */
anira_model_config* cfg; anira_machine_config* mc; anira_machine* m; anira_pipeline* pipe; anira_handler* h;
anira_model_config_from_json(model_json, model_json_size, NULL, &cfg, &err);                 /* v2 file: ANIRA_SUCCESS_UPGRADED */
anira_model_config_set_model_bytes(cfg, 0, model_onnx, model_onnx_size, ANIRA_BYTES_BORROW, NULL, NULL, &err);   /* the embedded blob lives for the DSO */
anira_machine_config_create(&mc, &err);
anira_machine_config_set_log_level(mc, ANIRA_LOG_WARNING);
anira_machine_config_set_log_sink(mc, log_sink, host);                                       /* anira_log_fn: never the driver thread; must not call anira */
if (ANIRA_FAILED(anira_machine_create(mc, &m, &err))) { host_log(err.message); return false; }
anira_pipeline_create(&pipe, &err);
const anira_model_config* variants[] = { cfg };
anira_backend_id candidates[] = {                                                            /* engine and provider are independent axes; engine_id is NULL for a built-in engine */
    { sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_DEFAULT, NULL },
    { sizeof(anira_backend_id), ANIRA_ENGINE_LITERT,      ANIRA_PROVIDER_DEFAULT, NULL },
};
anira_pipeline_add_inference(pipe, variants, 1, candidates, 2, &err);
anira_status st = anira_handler_create(m, pipe, &h, &err);                                   /* copies everything: the three handles may go now */
anira_pipeline_destroy(pipe); anira_model_config_destroy(cfg); anira_machine_config_destroy(mc);
if (ANIRA_FAILED(st)) { host_log(err.message); anira_machine_destroy(m); return false; }

/* clap_plugin.activate(sample_rate, min_frames_count, max_frames_count) [main-thread & !active]: */
anira_contract* c;
anira_contract_create_hard(min_frames_count, max_frames_count, sample_rate, &c, &err);
st = anira_handler_prepare(h, c, &err);                                                      /* the blocking quiescence point; err.message names the offending tensor or kind */
anira_contract_destroy(c);
if (ANIRA_FAILED(st)) { host_log(err.message); return false; }
host_set_latency(anira_handler_get_latency(h, 0));                                           /* constant until deactivate, as CLAP requires */

/* clap_plugin.process [audio-thread]: in place, after the plugin's in -> out copy; ANIRA_NONBLOCKING, never waits */
anira_handler_process(h, process->audio_outputs[0].data32, process->frames_count, 0);
/* elsewhere, off the driver thread: anira_handler_rt_error(h) says why a chunk came back as zeros */

/* clap_plugin.destroy [main-thread]: */
anira_handler_destroy(h);                                                                    /* blocks for quiescence; the pool dies with the last handler in this copy */
anira_machine_destroy(m);                                                                    /* drops the reference, joins nothing */

/* clap_entry.deinit [main-thread & !loader-lock]: the host's last call before it unloads the DSO */
void clap_deinit(void) { anira_shutdown(); }                                                 /* idempotent; refuses (ANIRA_ERROR_INVALID_STATE, nothing happens) while any machine or handler of this copy still lives */
```

The same sequence from the browser, over the TS package. Every call below is a `Module._anira_*` export; the worklet's per-quantum call is the same `anira_handler_process_separate` the native audio thread makes.

```ts
// TypeScript (main thread + worklet)
const anira = await AniraWeb.create();                                           // _anira_check_abi + Module.anira.version check
const cfg = await ModelConfig.fromUrl(anira, '/models/model.json');              // _anira_model_config_from_json; bytes injected with setModelBytes when the page has them
const machine = anira.createMachine({ log: { level: Abi.LOG_WARNING, sink: r => console.log(r.group, r.message) } });   // _anira_machine_config_set_log_level + _anira_em_set_log_hook + drain timer
const handler = machine.createHandler(Pipeline.inference(cfg, ['org.anira.onnxruntime-web']).registerEngine('org.anira.onnxruntime-web', new ONNXRuntimeWebBackend(cfg)));   // _anira_em_pipeline_register_js_engine
cfg.destroy();
await anira.spinUpInferenceWorker();                                             // _anira_inference_thread_create / _run_loop in the Worker; backend init() awaited
handler.prepare(Contract.hard({ maxBlockSize: 128, rate: ctx.sampleRate }));    // throws AniraError(status, message) on ANIRA_FAILED
await anira.configureAudioWorklet(ctx, handler);                                 // worklet: one _anira_handler_process_separate per quantum; processWait opt-in
```

### Async with deadline: GL video frames, GPU-resident round trip

```cpp
auto mc = anira::MachineConfig::from_file("machine.json");
mc.gl(anira_gl_desc{ .struct_size = sizeof(anira_gl_desc), .threads = ANIRA_GL_CALLER_THREAD,
                     .display = egl_dpy, .context = egl_ctx, .gbm = gbm });   // gbm only matters for a WebGPU candidate; GL is always borrowed
anira::Machine machine{mc};
anira::InferenceHandler handler{machine, pipe};                    // Inference(cfg, {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CUDA}})
handler.prepare(anira::Async{ .deadline = 33ms, .on_late = ANIRA_LATE_DROP });   // the contract's relative deadline; a submit may name an absolute one per job

// the app's own SSBOs: to the CUDA EP this is ZeroCopy (registration, cached in the plan);
// to a WebGPU candidate it would be HostCopy -- allocate_input would hand back a dma-buf-backed
// renderbuffer to render into instead, and the plan report says which row each slot got
const int64_t shape[] = {1, 3, h, w};
auto out = anira::Tensor::from_gl_buffer(out_ssbo, GL_SHADER_STORAGE_BUFFER, draw_done_fence, ANIRA_DTYPE_F32, shape);
handler.bind_output(0, out);                                       // acquire = writable-when; a GLsync token is non-owning, nothing moves
// per frame, on the GL thread (ANIRA_GL_CALLER_THREAD: anira touches GL only inside allocate_*, submit, bind_output):
auto in = anira::Tensor::from_gl_buffer(in_ssbo, GL_SHADER_STORAGE_BUFFER, render_fence, ANIRA_DTYPE_F32, shape);
anira::Ticket t = handler.submit({&in, 1}, {}, present_time);      // deadline per job, absolute on the anira_now_ms clock; the C call never blocks, never allocates
// next frame:
if (t.status() == ANIRA_TICKET_MET) draw_from(out_ssbo);           // output_ready(0) is QueueOrdered after the unmap
else reuse_last_frame();
// t leaves scope: anira_handler_ticket_release; a forgotten release would surface as ANIRA_ERROR_CAPACITY at submit
```

### Async with deadline: camera frame, v1 (the app converts the pixels)

```cpp
auto mc = anira::MachineConfig::from_file("machine.json");
mc.vulkan(anira_vulkan_desc{ .struct_size = sizeof(anira_vulkan_desc), .ownership = ANIRA_OWNERSHIP_BORROWED,
                             .queue_family = qf, .queue_index = 0, .instance = inst, .physical = phys, .device = dev });
anira::Machine machine{mc};
anira::InferenceHandler handler{machine, anira::Pipeline{ anira::stage::Inference(palm_cfg, {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU}}) }};
handler.prepare(anira::Async{ .deadline = 33ms, .on_late = ANIRA_LATE_DROP });
anira::Tensor in  = handler.allocate_input (0, ANIRA_DOMAIN_VULKAN_BUFFER);   // exportable + image tag: the DeviceCopy row
anira::Tensor out = handler.allocate_output(0, ANIRA_DOMAIN_VULKAN_BUFFER);   // pool tensors: anira never calls their release
handler.bind_output(0, out);
// per V4L2 frame: the app's own NV12 -> float pass (its VkImage import, its compute shader)
// writes in.handle.vk.buffer as an SSBO and signals a timeline value; the same descriptor is resubmitted every frame
in.acquire.kind = ANIRA_SYNC_VK_TIMELINE;  in.acquire.u.vk = { timeline, value };   // a timeline token is non-owning: nothing to move
anira::Ticket t = handler.submit({&in, 1});
// t.output_ready(0) is a sync file owned by the slot: the view dies with the ticket, so dup it, import it as a semaphore and draw from out
anira::SyncToken ready = t.output_ready(0).dup();
// teardown: handler.free_tensor(in); handler.free_tensor(out); -- or the next prepare()/destroy frees what is still out
```

### Async with deadline: two models composed by tickets (the tracking case)

```cpp
anira::Machine machine{mc};                                          // one Dawn device, shared (native WebGPU; the browser GPUDevice is post-3.0)
anira::InferenceHandler palm{machine, anira::Pipeline{ anira::stage::Inference(palm_cfg, {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU}}) }};
anira::InferenceHandler lmk {machine, anira::Pipeline{ anira::stage::Inference(lmk_cfg,  {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU}}) }};
palm.prepare(anira::Async{ .deadline = 33ms, .on_late = ANIRA_LATE_DROP });
lmk.prepare (anira::Async{});                                        // no deadline of its own: the loop has one

auto palm_in  = palm.allocate_input (0, ANIRA_DOMAIN_WGPU_BUFFER);   // the app's NV12 -> float pass writes these
auto lmk_in   = lmk.allocate_input  (0, ANIRA_DOMAIN_WGPU_BUFFER);   // (post-3.0: FrameToTensor, per-job affine)
auto palm_out = palm.allocate_output(0, ANIRA_DOMAIN_WGPU_BUFFER);
auto lmk_out  = lmk.allocate_output (0, ANIRA_DOMAIN_WGPU_BUFFER);
palm.bind_output(0, palm_out);  lmk.bind_output(0, lmk_out);

// per camera frame, in the app's tracking loop -- MediaPipe's graph, node for node:
if (prev_rects.size() < num_hands) {                                 // the gate: application logic
    write_letterbox(palm_in, frame);                                 // the app's pass on the Machine's queue
    anira::Ticket t = palm.submit({&palm_in, 1});
    t.wait_for(budget);                                              // HostCopy, priced: the decode needs the bytes
    rects = associate(decode_nms(palm_out), prev_rects);             // user code, user state
}
prev_rects.clear();
for (auto& r : rects) {                                              // 0..num_hands runs: application logic
    write_crop(lmk_in, affine_of(r));                                // same frame, different affine
    anira::Ticket t = lmk.submit({&lmk_in, 1}, {}, {}, &r);          // WgpuBuffer -> WebGPU EP: ZeroCopy; &r rides the job as job_user_data, back out of t.user_data()
    t.wait_for(budget);
    if (presence(lmk_out) >= 0.5f) prev_rects.push_back(rect_of(lmk_out));
}                                                                    // each Ticket releases its slot at the end of its iteration
// Where B reads A's output DIRECTLY (no host logic between), nothing is waited on at all:
//   mid = a.allocate_output(0, ANIRA_DOMAIN_WGPU_BUFFER);  a.bind_output(0, mid);
//   anira::Ticket ta = a.submit({&in, 1});  mid.acquire = ta.output_ready(0);   // QueueOrdered: non-owning, nothing to dup, nothing moves
//   anira::Ticket tb = b.submit({&mid, 1});                                     // same queue: no wait, one drain at the end; ta outlives tb's submit
// The camera buffer is read by both handlers and the presenter: requeue it when all three have released it.
```

### Async with deadline: camera frame, all on one GPU queue (post-v1: Frame + FrameToTensor, `abi/draft/frame.h`)

`Frame`, `stage::FrameToTensor` and `submit_frame` are vocabulary in v3.0.0; the C struct and the entry arrive in `abi/draft/frame.h` with their own symbol baseline, additive under ABI major 3, and are promoted without renaming once measured. The sketch shows the intended shape.

```cpp
auto mc = anira::MachineConfig::from_file("machine.json");
mc.webgpu(anira_webgpu_desc{ .struct_size = sizeof(anira_webgpu_desc), .ownership = ANIRA_OWNERSHIP_BORROWED,
                             .exec = ANIRA_EXEC_WORKER, .instance = inst, .device = dev, .queue = q });
anira::Machine machine{mc};
anira::Pipeline pipe {
    anira::stage::FrameToTensor(anira::PixelFormat::NV12, {192, 192}, anira::Letterbox,
                                anira::Normalize01, anira::Layout::NHWC),
    anira::stage::Inference(palm_cfg, {{ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_WEBGPU}}),
};
anira::InferenceHandler handler{machine, pipe};
handler.prepare(anira::Async{ .deadline = 33ms, .on_late = ANIRA_LATE_DROP });

// per V4L2 frame (dma-buf exported once per buffer):
auto f = anira::Frame::from_dmabuf(fds, offs, pitches, DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_NV12,
                                   w, h, {anira::BT709, anira::Limited}, /*sync_fd*/ -1);
anira::Ticket t = handler.submit_frame(f);
// requeue the V4L2 buffer when t.input_released(0) signals; the app's own presenter read of
// the same buffer is part of that token (two-reader ownership, section 7)
```

### Async without deadline: offline file rendering (the former offline branch)

```cpp
handler.prepare(anira::Async{});                                     // no deadline: the offline posture; lanes auto, deep pipelining
const int64_t shape[] = {1, channels, num_samples};
auto in  = anira::Tensor::from_host(samples, ANIRA_DTYPE_F32, shape); // borrowed: release == NULL -- the memory stays the caller's and must stay valid until the ticket is terminal
auto out = anira::Tensor::from_host(out_buf, ANIRA_DTYPE_F32, shape);
handler.bind_output(0, out);                                         // request_output is post-3.0
anira::Ticket t = handler.submit({&in, 1}, anira::JobOptions{ .head_trim = {-1}, .tail_flush = true });
t.wait();                                                            // anira_handler_ticket_wait with a negative timeout: forever; input-aligned result
```

v2's `set_non_realtime(bool)` (released in v2.0.0) is neither a C entry nor an `anira.hpp` method: it survives only in `compat/v2.hpp`, where `anira::v2::InferenceHandler::set_non_realtime(true)` routes the v2 `process`/`pop_data` overloads to the `_wait` twins with `ANIRA_WAIT_FOREVER`. A direct v3 caller renders offline either through Async and a ticket, as above, or -- when the v2 pull shape is wanted -- through the Hard `_wait` twins (`anira_handler_process_wait(h, data, n, ANIRA_WAIT_FOREVER, 0)`), which are not `ANIRA_NONBLOCKING` and spin on Wasm.

---

## 10. Migration (v2 -> v3)

v3.0.0 is a major version whose one-time source break introduces the versioned C ABI -- `anira/anira.h`, `ANIRA_ABI_MAJOR 3` -- that does not break thereafter. The 2.x line never carried a binary promise (no ABI gate, an `AnyNewerVersion` package, `USE_*`-shaped enum values and struct layouts; v2.3.0's CHANGELOG preamble says so), and its eight `**Breaking:**` entries are source renames with deprecated forwards. `anira.hpp` and `compat/v2.hpp` are header-only source-level conveniences over the C header with no binary promise of their own; nothing in them is exported from libanira. Breakage concentrates in construction-time code; the hot path keeps its pointer shapes.

### 10.1 What is promised, and by which number

| number | lives in | owns |
|---|---|---|
| `ANIRA_ABI_MAJOR`, `ANIRA_ABI_MINOR`, packed as `ANIRA_ABI_VERSION` | `abi/build_info.h`, generated at configure by `cmake/build-info.cmake` from the git tag; the generated `abi/version.h` packs them -- the tag is the one source | the binary promise |
| `ANIRA_VERSION_MAJOR/MINOR/PATCH/STRING`, `ANIRA_MAKE_VERSION` | `abi/build_info.h`, configured from git by `cmake/build-info.cmake` over tanh-tooling v0.2.7's prerelease-aware `tanh_git_version` (`MATCH "v3*"` on the `v3` branch; full describe string kept) | release identity; semver major equals ABI major by policy from 3.0.0 |
| `SOVERSION ${ANIRA_ABI_MAJOR}`, `MACHO_COMPATIBILITY_VERSION ${ANIRA_ABI_MAJOR}.${ANIRA_ABI_MINOR}`, `OUTPUT_NAME anira-${ANIRA_ABI_MAJOR}` on PE | `cmake/build-info.cmake` (today `SOVERSION ${PROJECT_VERSION_MAJOR}`, `CMakeLists.txt:291`; flips at alpha.2) | `libanira.so.3`, `libanira.3.dylib` (dyld refuses a library older than the one a client linked), `anira-3.dll` + `anira-3.lib`; Debian runtime package `libanira3` |
| `aniraConfigVersion.cmake` `COMPATIBILITY SameMajorVersion` | `cmake/install.cmake` (today `AnyNewerVersion`, `install.cmake:378`) | `find_package(anira 3.x)` |

Inside ABI major 3, from v3.0.0: every existing function keeps its name, signature, semantics, thread tag and RT class; existing enum values and Tier-1 POD layouts (`anira_tensor`, `anira_sync_token`, `anira_memory_handle`, `anira_stage_ctx`, `anira_log_record`, `anira_error`; committed as `abi/layout-3.txt`) never change. A minor may append -- functions, enum values below `_FORCE32`, positive `anira_status` values, extension revisions, descriptor tail slots after the last v3.0.0 slot, `anira_em_*` companions -- and every such addition bumps `ANIRA_ABI_MINOR`; a patch changes nothing in the header. A major is forced only by an RT POD layout change, a removed or renamed function, a changed enum value, or a changed thread or RT contract. Deprecation inside a major is `ANIRA_DEPRECATED("use anira_x")` plus a `**Deprecated:**` CHANGELOG entry; the symbol stays exported until the next major. Everything listed as deferred below is a new function, value or tail slot, so it lands inside major 3 as a minor.

Before the freeze the header exists in full under ABI 0: `ANIRA_ABI_MAJOR` is 0, `ANIRA_ABI_MINOR` is the alpha/beta ordinal, header changes are CHANGELOG-prefixed `**ABI (unstable):**`, npm publishes under `next`, the symbol baselines, the layout table, the C11 header compile, the pure-C install consumer and the RTSan leg fail the build once online, abidiff reports. `find_package(anira 3.0)` accepts mutually incompatible alphas during that window -- accepted, because `anira_check_abi` is the gate that matters. `anira_check_abi(ANIRA_ABI_VERSION)` is the first call of every consumer -- the raw-C plugin, `anira.hpp`, `AniraWeb.create()` -- and returns `ANIRA_OK` iff the library's major equals the header's and its minor is at least the header's (exact match while major is 0), `ANIRA_ERROR_ABI_VERSION` otherwise; `anira_get_proc_address(name)` is feature detection for `dlopen` hosts (NULL = not in this build). The v2 cross-session version-string compare (`src/scheduler/Context.cpp:325-348`) is dropped: two static embeddings have two cores and cannot see each other, and inside one core every session shares one header.

### 10.2 Survives verbatim

Behind C entries with the same pointer shapes: the `process` / `push_data` / `pop_data` semantics in every pointer-triple variant (`anira_handler_process`, `process_separate`, `process_multi`, `push_data`, `push_data_multi`, `pop_data`, `pop_data_multi` -- semantics, not signatures, for the three `_multi` forms and for `get_available_samples` below: those four change shape, section 10.3), `get_latency`, `get_latency_vector` (`anira_handler_get_latencies`; index-aligned with the output list, 0 for Static outputs -- a C array indexed by slot is the only representation that cannot misalign), `get_available_samples`, `reset`, `set_input` / `get_output` (Static tensors underneath, materialised into `model_inputs` by the handler before any stage runs), `drain_log`, `get_num_inference_threads`, thread-pool sharing (one core per copy of anira; the first machine sizes the pool, later machines reconcile per field), `WaitStrategy` (`anira_wait_strategy`), `LogLevel` (`anira_log_level`), `LogDrain` and `LogConfig` (`anira_log_desc` and the scalar `set_log_*` setters). `set_inference_backend` / `get_inference_backend` survive in `compat/v2.hpp` only, resolved through the `(variant, engine, provider)` -> plan table built at `prepare`: two independent stores on two axes could land on a pair that has no plan, on the driver thread, with no error channel, and one store cannot, so the C surface is `anira_handler_set_plan` alone (section 6). `set_non_realtime` -- released in v2.0.0 (`CHANGELOG.md:197`, `InferenceHandler.h:392`) -- survives in `compat/v2.hpp` only, not as a C entry. No tanh type appears in any survivor.

### 10.3 Behaviour changes for direct v3 callers -- none for `compat/v2.hpp` users

- The `ANIRA_NONBLOCKING` `process` never waits. v2's `blocking_ratio` wait inside `process()` and the `set_non_realtime` wait inside every Hard entry are the `_wait` twins -- `anira_handler_process_wait`, `process_separate_wait`, `process_multi_wait`, `pop_data_wait`, `pop_data_multi_wait` -- with `timeout_ms` (`ANIRA_WAIT_CONTRACT` = `wait_ratio` x block duration, v2's `blocking_ratio`; `ANIRA_WAIT_FOREVER` = v2's `set_non_realtime`); `anira_contract_hard_set_wait_ratio` is v2's `m_blocking_ratio` one-to-one, and `get_latency` keeps the v2 arithmetic including the wait credit. The twins spin on Wasm.
- The JSON loader never drops entries: an unknown key is stored and fails `anira_handler_prepare` by name (`anira_error.message`), a backend string not in this build is `ANIRA_ERROR_NOT_SUPPORTED` at `prepare`, malformed text is `ANIRA_ERROR_JSON`. v2 ignored unknown keys and silently dropped disabled backends (`src/utils/JsonConfigLoader.cpp:280-327`).
- `anira_shutdown` is effective only when no machine handle and no handler exist in this copy of anira; otherwise it does nothing and returns `ANIRA_ERROR_INVALID_STATE`, so one client of a shared `libanira.so.3` cannot silence another's sessions.
- The multi variants write counts into the caller's in/out `num_out` (in: capacity, out: written) instead of returning a handler-owned `size_t*`; `tensor_index` and every index is `uint32_t`; `get_available_samples` takes a non-const handler and is `[driver-thread]` because it runs `post_process`.
- A handler copies the pipeline and every config at `anira_handler_create`; v2 kept the `InferenceConfig&` for life. Configs may be destroyed right after the call.
- Failures are `anira_status` (`ANIRA_FAILED(s)` is the only stable test) with the message in a caller-owned `anira_error`; `prepare` no longer throws at the C level (`anira.hpp` throws `anira::Error`, or returns `anira::Result<T>` under `ANIRA_CXX_NO_EXCEPTIONS`). A Hard entry on an Async handler returns 0 / `ANIRA_ERROR_WRONG_CONTRACT` and records it in `anira_handler_rt_error`.

### 10.4 Mechanical -> `compat/v2.hpp`

The v2 renames -- `InferenceConfig` -> `ModelConfig`, `ContextConfig` -> `MachineConfig`, `Context` -> `Machine`; `TensorShapeList` + `ProcessingSpec` -> `TensorSpec` (shapes -> tagged axes, `preprocess_input_size` -> `window_min == window_max`, hop -> `context`); `prepare(HostConfig)` -> `prepare(Hard{...})` (the custom-latency overloads -> per-output `latency` on the spec); `max_inference_time` / `warm_up` / `blocking_ratio` -> the Hard fields `budget`, `warmup`, `wait_ratio`; `model_function` -> `ext::Entry{name}` (`anira_ext_entry`, kind `"entry"`) on the model entry, the one extension v3.0.0 ships -- are constructor-level shims in `namespace anira::v2`, every entity `[[deprecated]]`, bridging one minor cycle and removed one minor after 3.0:

| `anira::v2` | is |
|---|---|
| `InferenceConfig(std::vector<ModelData>, std::vector<TensorShape>, ProcessingSpec, float max_inference_time, unsigned warm_up = 0, bool session_exclusive = false, float blocking_ratio = 0, unsigned num_parallel = Defaults::m_num_parallel_processors)`, `ModelData`, `TensorShape`, `ProcessingSpec` | a `ModelConfig` plus a Hard contract with `wait_ratio = blocking_ratio`; `num_parallel` keeps v2's default, `hardware_concurrency() / 2` (minimum 1, `InferenceConfig.h:456-459`), deliberately not v3's `max_instances` default of 1 (section 5) |
| `ContextConfig(unsigned num_threads = default_num_threads(), WaitStrategy = SpinBackoff, LogLevel = default_log_level())`; `using Context = anira::Machine` | a `MachineConfig`, with v2's defaults reproduced, not v3's: `default_num_threads()` is `hardware_concurrency() / 2` (minimum 1) natively and 0 on Emscripten, `default_log_level()` is `Info` in debug and `Error` in release (`ContextConfig.h:88-99,167-179`), where `anira_machine_config` defaults to `ANIRA_THREADS_AUTO` and `ANIRA_LOG_WARNING` (section 4) |
| `HostConfig(float buffer_size, float sample_rate, bool allow_smaller = false, size_t tensor_index = k_first_streamable, bool tensor_is_input = true)` | the Hard geometry and the anchor |
| `InferenceHandler(PrePostProcessor&, InferenceConfig&, const ContextConfig& = {})` with `prepare(HostConfig)` and the three overloads, the v2 `process` / `push_data` / `pop_data` overloads, `set_non_realtime(bool)` | the v2 overloads routed to the NB entries, to the `_wait` twins with `ANIRA_WAIT_CONTRACT` when `blocking_ratio > 0`, and with `ANIRA_WAIT_FOREVER` while the wrapper-side `set_non_realtime` flag is set -- v2-identical from the compat layer, latency figure included |
| `JsonConfigLoader(path)` with `get_context_config()` / `get_inference_config()` | `anira_machine_config_from_json` / `anira_model_config_from_json` + `take_legacy_contract` |
| `enum InferenceBackend { ONNX = ANIRA_ENGINE_ONNXRUNTIME, LIBTORCH = ANIRA_ENGINE_LIBTORCH, TFLITE = ANIRA_ENGINE_TFLITE, LITERT = ANIRA_ENGINE_LITERT, EXECUTORCH = ANIRA_ENGINE_EXECUTORCH, CUSTOM = -1 }` | the pinned `anira_engine` values (section 6a), independent of `USE_*` and of the provider axis v2 never had -- a v2 backend is that engine at `ANIRA_PROVIDER_DEFAULT`; every enumerator is initialised, because a continued value would land on another engine's id, and `CUSTOM` is negative because it is not an engine value at all: the shim registers the v2 processor under a reverse-URI name and the name resolves to its value at `prepare` |
| `RingBuffer`, `BufferF`, `PrePostProcessor`, `LegacyProcessorStage` | section 10.5 |

The v2 JSON schema (`inference_config` / `context_config` roots) is auto-upgraded by `anira_model_config_from_json` -- `model_data` -> `models`, `inference_backend "ONNX"` -> `"onnxruntime"`, `model_function` -> `entry.name`, `preprocess_input_size` / `postprocess_output_size` / `internal_model_latency` -> `window` / `context` / `latency`, `num_parallel_processors` -> `max_instances`, `session_exclusive_processor` -> `state`, `max_inference_time` / `warm_up` / `blocking_ratio` -> the legacy contract (section 8) -- returning `ANIRA_SUCCESS_UPGRADED` with one `ANIRA_LOG_WARNING` per process; the contract-shaped v2 fields come back through `anira_model_config_take_legacy_contract`. Already-deprecated v2 leftovers leave with the public tree they lived in: `HighPriorityThread` (`[[deprecated]]` since one minor), the `AniraExports.h` / `AniraWinExports.h` forwarders, the `utils/{Buffer,RingBuffer,MemoryBlock}.h` aliases over `thl::core`.

### 10.5 Semantic -> `anira::v2::LegacyProcessorStage`

Custom `PrePostProcessor` subclasses (the four virtuals over `std::vector<RingBuffer>&` / `std::vector<BufferF>&`, `PrePostProcessor.h:83-146`) become host-domain stage descriptors (`anira_stage_desc` over `anira_stage_ctx` and `anira_ring*`; section 7). `compat/v2.hpp` provides `anira::v2::PrePostProcessor` with the same four virtuals over `std::vector<v2::RingBuffer>&` / `std::vector<v2::BufferF>&`, where `v2::RingBuffer` and `v2::BufferF` are non-owning views with the v2 method names over the C accessors, reserved at `prepare` and re-pointed from `anira_stage_ctx` on every call (allocation-free, never cached across calls); `set_input` / `get_input` / `set_output` / `get_output` forward to the handler the stage received in `prepare`; `anira::v2::LegacyProcessorStage(pp)` yields the descriptor. Source-compatible: subclasses using the block API (`pop_block`, `push_block`, `peek_past_block`, `push_fill`, `discard`, `get_available_samples`, `pop_samples_from_buffer`, `push_samples_to_buffer`), `set_input` / `get_output`, `before_inference` / `after_inference`. Not source-compatible: code that reached into `thl::core` storage (`swap_data`, `get_memory_block`) or used per-sample `push_sample` / `pop_sample` -- those callers migrate to a stage. Custom `BackendBase` subclasses have no compat shim: `anira_backend_desc.process(inputs, n_in, outputs, n_out, instance, user_data)` replaces `BackendBase::process(std::vector<BufferF>&, ..., std::shared_ptr<SessionElement>)`, and no engine used the session.

### 10.6 Free

The unreleased lanes/callback/poll branch folds into Async before its release: lanes survive on `anira_contract_async_set_policy`, tickets subsume callback and poll (`ANIRA_DELIVERY_IMMEDIATE` / `ANIRA_DELIVERY_POLLED`), per-job non-streamable values become Static input tensors, dissolving the exclusive-scheduling rule. The web/TS package `@anira-project/anira` is a new major (3.x) over the C ABI, not a migration: `src/emscripten-wrappers/` (226 `EMSCRIPTEN_KEEPALIVE` handle functions over `uintptr_t`) is deleted, and with it the `Vector*` classes, `ModelData`, `TensorShape`, `ProcessingSpec`, `InferenceConfig`, `HostConfig`, `BufferF`, `PrePostProcessor` / `JSPrePostProcessor`, `JSBackendBase`, `createInferenceBackend`, `Factory`, `resolvePtr`; `ONNXRuntimeWebBackend`, `InferenceThread`, `AniraWeb`, `AniraAudioWorkletBase`, `setupInferenceWorker`, `bundleAudioWorklet` and the Worker message protocol stay; there is no TS compat layer.

Platform coverage is unchanged: `Host` (`ANIRA_DOMAIN_HOST`) is enabled on every platform v2 runs on (section 1), so every v2 deployment migrates without touching a domain. The GPU domains are additions on top, per platform.

Purely additive, inside ABI major 3 (no migration impact): variant sets with `anira_handler_set_plan`, composition of models across handlers on one Machine through tickets, shared engine environments, the native WebGPU machine resource (`anira_machine_config_set_webgpu`, Dawn), `allocate_input` / `allocate_output` / `free_tensor`, the extension slots of section 1b (`set_ext` / `set_ext_json` on every config handle). Deferred past v3.0.0, each arriving as a minor: `Frame` + `FrameToTensor` + `submit_frame` in `abi/draft/frame.h` (including `Container::IOSurface`, the pixel-carrying role of the same handle, and `ext::CropAffine`; the domain and container values are reserved in `abi/enums.h`), chained `Inference` stages inside one Pipeline (section 7, Multi-model support), `request_output`, `GlThreads::SharedContext` (value pinned, behaviour deferred), a per-run anchor override on `Hard` (the field waits for a multi-rate model), every `Domain` arm outside the v1 set of section 1 (values pinned in `abi/enums.h`; the unmeasured factories `anira_tensor_init_metal` / `_iosurface` / `_ahardwarebuffer` / `_d3d12` sit in `abi/draft/tensor_platform.h` with their own baselines until their platform column is measured -- promotion moves a name between baselines and never renames it), browser WebGPU as an explicitly asynchronous JS backend (an `anira_backend_desc` tail slot `process_async` plus a completion callback driven from an unblocked Worker pump, or a JSPI-linked second module variant sharing the header; section 4), third-party extension kinds (`anira_register_ext_kind` plus a prepare-time payload accessor; the reverse-URI prefix is reserved now), a host thread-provider vtable (`num_threads = 0` plus `anira_inference_thread_*` already lets a host bring its threads), and every entry of section 1b's deferred catalogue -- `ext::Quant`, `ext::Artifacts` / `ext::ArtifactCache`, `ext::Npu` / `ext::NpuHard`, `ext::OrtSession`, `ext::JobBackend` / `ext::JobModel` -- each with the stage or adapter that consumes it. Quantization and NPU support are therefore out of v3.0.0 scope entirely, not partially. Excluded, not deferred: OpenCL (section 1). The former multi-plane `dmabuf` arm of `MemoryHandle` is removed before any release that ships it: `anira_memory_handle.dmabuf` is the single exported-buffer arm `{fd, size, offset}`, and `anira_tensor_init_dmabuf` keeps its name with the buffer meaning.

---

## 11. Implementation roadmap

The design above lands in anira as a sequence of milestones, each a set of pull requests that merges with the full test suite green, so that the work proceeds one step at a time and can stop at any step with a coherent library. Two rules hold throughout. The v2 test suite is the oracle for the Hard path: `process`/`push_data`/`pop_data` survive verbatim behind the `ANIRA_NONBLOCKING` C entries (section 10), and every milestone that touches the pump keeps those tests passing through whatever API exists at that point -- through `anira.hpp`/`compat/v2.hpp` from `v3.0.0-alpha.2` on. And no registry row, cost class or platform column is coded before it is measured (sections 1 and 7): a milestone that needs a measurement lists it as a precondition rather than assuming it. A third rule applies to the ABI: an `abi/*.h` file freezes at the milestone that ships it, and from then on it changes only by appending (section 6a); every freeze is enforced by a gate that comes online in the same milestone, listed in the table below.

Strategy: replace subsystems in place, never rewrite from scratch. v2's ring buffers become the RingChunker's storage, its inference threads and lock-free queue become the pump, its `PrePostProcessor` subclasses keep running behind `anira::v2::LegacyProcessorStage`. The public API changes at the seams section 10 names; the internals are swapped one at a time underneath a green suite, and the C entries of `src/capi/` are thin forwarders onto the v2 code paths until the internals move.

Branching: M0 ships from `main` as v2.3.0. Afterwards `main` carries v2 maintenance until v3.0.0 merges, and v3 work lands on a long-lived `v3` integration branch through PRs, tagged `v3.0.0-alpha.N` / `-beta.N` at the milestone boundaries below. Every alpha and beta carries `ANIRA_ABI_MAJOR 0` with `ANIRA_ABI_MINOR` as the ordinal, `anira_check_abi` demands an exact match, and header changes are logged as `**ABI (unstable):**`; the freeze is one commit, in M6. Topic branches are `feat/v3-<topic>` (git cannot hold `refs/heads/v3` and `refs/heads/v3/x` at once). The first PR on `v3` adds `v3` to the `main`-only branch filters of `build_test`, `build_sanitizer`, `build_install` and `build_test_mobile` (`push` and `pull_request`) and of `build_examples` and `build_benchmark` (`pull_request` only) -- without it no gate runs where M1-M5 live (`build_web`, `clang_check`, `clang_format` and `clang_tidy` are the only unrestricted workflows). anira's repository rules apply to every PR: Doxygen on public headers, `docs/sphinx`, `CHANGELOG.md` with breaking entries prefixed `**Breaking:**`; `.clang-*` and `cmake/tanh/` are installed from tanh-tooling and never edited by hand, and anira and the fetched tanh-lib pin the same tanh-tooling tag (v0.2.8 today), so every CMake addition of M1-M6 lives in `anira/cmake/` until a tooling tag after v3.0.0. Target dates: M0 30.08.2026 (Vallo); M1–M3 07.09.2026 and M4–M6 25.09.2026 (Fares).

### M0 -- v2.3.0: the last release of the 2.x line (target 30.08.2026)

What `main` already holds is the `[Unreleased]` section of `CHANGELOG.md`, and it is the v2 parity baseline every later milestone measures against: the ExecuTorch backend and its `model_function` support; containers over `thl::core` (`Buffer<T>`, `RingBuffer`, `RingBufferT<T>`, `MemoryBlock<T>` as aliases, the block API on every real-time ring access); logging through `thl::Logger` with `LogConfig`, `LogDrain` and `drain_log()`; one-sided streaming with the input-or-output reference stream (#98, #99, #110, the redo of the reverted #101); the symbol policy through `tanh_apply_symbol_policy`/`tanh_set_export_allowlist` and the `anira_exports` CTest; engine pimpls, the `anira_header_isolation` CTest and the `anira::<engine>` imported targets with linkage following `BUILD_SHARED_LIBS`; the immortal `Context` core with `shutdown()`, `release_core_if_idle()`, `has_core()` and the `test/unload` suite; the `build_install` workflow and `InstallConsumer` CTest; the Windows `bin/` install layout and the `lib64` patching; `HighPriorityThread` deprecated for `thl::core::Thread`; the wait-free `reset()`; the concurrent-lifecycle fixes; the receptive-field default pre-processor; the Fedora ExecuTorch link fix. LiteRT as the default TensorFlow-family backend is not on this list: it shipped in v2.2.0. Added in this milestone:

1. **The tag is `v2.3.0`**, with a CHANGELOG preamble stating that the 2.x line never carried a binary-compatibility promise (no ABI gate, an `AnyNewerVersion` package, `USE_*`-shaped class layouts, `sizeof(HostConfig)` and the `Buffer` layout changed in this very release), that its eight `**Breaking:**` entries are source-level breaks -- renames carrying deprecated forwards where a rename was possible, layout changes where not, and that v3.0.0 is the first release that promises anything about its ABI. `v3.0.0-alpha.0` is the alternative, listed under the open decisions; `v2.3.0` keeps the "v2 parity" name, the backends release `v2.3.0` and the npm 2.x line aligned and gives plugin authors a pin.
2. **The `unloadtest_*` rename.** The unload module's `extern "C"` entry points `anira_test_*` (`test/contracts/unload/module_api.h`, `SYMBOL anira_test_*` in `test/contracts/unload/CMakeLists.txt`, the test resolving them by name) become `unloadtest_*`, because from M1 the export check allows `^_?anira_[a-z0-9_]+$` and a static-build module exporting names inside that regex is reported as exporting the statically linked library. Ten minutes, anira-local, and the only code change the tag waits for.
3. **CI on `tanh-lab/ci-actions`.** anira's own `setup`/`build`/`test` composites (`.github/actions/`) are replaced by the shared actions in preset mode; today three workflows use `tanh-lab/ci-actions` (`build_sanitizer`, `clang_format`, `clang_tidy`), every use is `@main`, and `build_test` still runs the local composites. anira gains per-platform presets (sanitizers, Android, iOS, Windows-arm64, macOS-universal, shared/static x backend sets) beside today's `desktop-*`, `clang-tidy`, `wasm-*` and `docs`, following tanh-lib's `android-*`, `ios-*`, `windows-arm64-debug` and `desktop-debug-{asan,tsan,msan,lsan,rtsan}`; the implicit gcc coverage of today's unpinned Linux legs becomes one explicit gcc job, because the `desktop-*` presets force clang; `GITHUB_TOKEN` moves from step level inside the build composite to job level everywhere, as `on_tag` and `build_test_mobile` already have it; the mobile test actions gain inputs for the extra files anira pushes (backend `.so`s, `libc++_shared`, `extras/models`), or anira keeps its mobile workflow; the `install`/codesign/release action stays anira's; ci-actions pinned to a tag like tanh-tooling. The `.clang-*` files and `cmake/tanh/` stay as they are: installed from tanh-tooling, drift-checked by the `tooling-config` job of `lint.yml` through tanh-tooling's `config-check.yml` (v0.2.8 pinned).
4. **Test reorganisation, anira.** The single `tests` binary (`test/CMakeLists.txt:25`) becomes per-component binaries over the sources that are already grouped under `test/backends/`, `test/scheduler/` and `test/utils/` -- `test_utils`, which also takes the root file `test_WavReader.cpp` because it exercises `thl::core::read_wav` and not the handler; `test_scheduler`; `test_backends`; `test_handler` (the integration suite: the other four root files `test_InferenceHandler`, `test_OneSidedStreaming`, `test_StatefulOrdering`, `test_BackendLinkage` and the model fixtures). WAV fixtures already come from the header-only `thl::core::read_wav` and model paths are compile definitions from `extras/CMakeLists.txt`; both stay where they are. The separate CTests that already exist (`anira_header_isolation`, `anira_unload_test`, `anira_exports`, `InstallConsumer`) are the pattern the ABI gates follow, and `test/abi/` is added beside them in M1. The `test_*` naming is what ci-actions' Android runner auto-discovers (tanh-lib's `test_core`, `test_dsp`, `test_modulation`, `test_state`), and per-component binaries are where M2's adapter tests and M4's probe suite slot in. No behaviour change; the suites' test counts before and after are the acceptance. The tanh-lib side is done: the core containers' tests live in `test/core/`, `benchmark_state`/`benchmark_modulation` are separate targets, and the DSP fixtures are `test_DspFixture_*`.
5. **Two v2 scheduler defects**, fixed here because this is the last release of the line they belong to. `SessionElement::least_common_multiple(int a, int b)` returns `a * b / greatest_common_divisor(a, b)` (`src/scheduler/SessionElement.cpp:773-775`), so the product overflows signed `int` once the host block times the hop exceeds 2^31; both call sites pass the floored host block and a stream hop (`:628` buffer adaptation, `:730` inference count), which puts it out of reach at audio block sizes and reachable only on the streaming path when the host block is itself frame-sized. It widens to `int64_t` and divides before multiplying, `a / gcd(a, b) * b`. The `allow_smaller_buffers` sweep counts down from the greatest relative buffer size with a float loop counter, `while (--greatest_buffer_size > 0)` (`:315`), recomputing latencies and struct counts on every step: millions of iterations when that size is large, and above 2^24 a decrement of one is no longer representable, so the counter can round back to its own value and the sweep stops making progress. It becomes a closed form over the same quantity. v3 drops `allow_smaller_buffers` for a declared block range (section 3), so that fix matters only for the 2.x line, while the overflow travels into v3 with the ring sizing M2 carries across. Neither defect is reachable from an Async, Buffer-role video path: a Buffer-role tensor has no hop, both sweeps skip it, and Async sizes no ring.
6. `feat/offline-inference` does **not** ship: it folds into `Async` in M3 (section 10).

Items 3 and 4 touch `main` only and merge into `v3` from there; the tag waits for neither. The stable, versioned C ABI itself is not an M0 deliverable: it is the spine of M1-M6 and freezes in M6 (section 6a, and the ABI rows of every milestone below); all M0 carries of it is the `unloadtest_*` rename it makes necessary. Effort beyond the tag itself: 0.1 person-days.

Exit: tag `v2.3.0`, release artifacts from `on_tag.yml`, npm 2.3.0 from `publish_web.yml`. "v2 parity" everywhere in this document means this tag.

### M1 -- configuration layer (`v3` branch; no runtime change; target 07.09.2026)

Everything in sections 1b, 2, 3, 4 (the struct), 5, 6 (`edge_cost`) and 8 that is data, spelled as C handles with scalar setters: `anira_tensor_spec` with tagged axes, roles, window/context/latency; `anira_model_config` with models by index, the per-entry tensor record (`set_tensor_name`, `set_tensor_layout`), `set_model_bytes`, `set_anchor` by canonical name; `anira_machine_config` with the device descriptors and the log scalars; `anira_contract` as the tagged handle behind `anira::Contract = std::variant<Hard, Async>`, `hard_set_wait_ratio` included; `anira_job_options` with its frame-invariant setters; the extension registry keyed `(kind, version)` with `anira_ext_entry` as its one row, `anira_registered_ext_kinds`, the consumed-or-fail walk; `anira_model_config_from_json`/`anira_machine_config_from_json`/`anira_contract_from_json` with the v2 auto-upgrade (`ANIRA_SUCCESS_UPGRADED`, one warning per process, `take_legacy_contract`). Internally a translator from the new handles to v2's `InferenceConfig`/`ContextConfig`/`HostConfig`, so the v2 engine runs unchanged behind the v3 configuration; the v2 classes stay public and exported through this milestone, `include/anira/anira.h` stays the v2 C++ umbrella, and `USE_*`/`ANIRA_VERSION` stay PUBLIC because the v2 headers still need them. The deprecated aliases (`InferenceConfig`, `ContextConfig`, `Context`, `JsonConfigLoader`) are not a v3 deliverable; they become the constructor shims of `compat/v2.hpp` in M2.

ABI work in this milestone:

- **First PR**: the `v3` CI branch filters (above) and `cmake/build-info.cmake` over tanh-tooling v0.2.7's prerelease-aware `tanh_git_version` (`MATCH "v3*"`; so `v3.0.0-alpha.1` configures and the full describe string stays `ANIRA_VERSION_STRING`), which also derives `ANIRA_ABI_MAJOR`/`MINOR` from the tag into the generated `abi/build_info.h` -- before the first alpha tag.
- **Frozen at M1**, included per file (no C umbrella yet): `abi/export.h`, `abi/version.h`, `abi/build_info.h` (generated), `abi/status.h`, `abi/enums.h` with every value pinned: `anira_engine` and `anira_provider` as the two independent axes, and every domain and sync arm M4-M6 will populate; `abi/config.h`, `abi/log.h`.
- **Implementation**: `src/capi/{version,error,config}.cpp` with the exception firewall; `tanh_set_export_allowlist(anira NAMESPACE anira SYMBOL "anira_*")` and the export check's `ALLOW_REGEX "^_?anira_[a-z0-9_]+$"`.
- **Tooling**: the registry `abi/anira.yml` and its generator `tools/abi/gen.py`, which emit the C headers themselves (never hand-edited; the Vulkan-Headers pattern), `web/src/abi/enums.ts`, `exports_wasm.txt`, both symbol lists, the proc table behind `anira_get_proc_address` and `test/abi/test_layout.c`, and validate thread tags, the 64-bit rule and the descriptor field order while doing so; `abi/layout-0.txt` committed; `gen.py --diff-against <tag>` for the release-time classification of principle 8.
- **Tests**: the prepare-time legality rules of section 2, JSON round trips, the v2 upgrade, consumed-or-fail by name, `test/abi/test_layout.c` and `test/abi/header_c.c`.
- **Gates online**: 3 (layout table, native), 4 (per-file C11 and C++17 compile; the strict-isolation half and the Wasm arm follow the umbrella at M2) and the registry check `anira_abi_generate`. Effort beyond the configuration layer itself: 2.0 person-days.
- **As built** (`v3.0.0-alpha.1`, ABI 0.1, 2026-09-03; PRs #159-#169 on `v3`): everything above, plus two items pulled forward from M2 because they depend on the C handles alone -- the configuration half of `anira.hpp` (C++20, header-only, not ABI-stable) and a transitional bridge, `anira/compat/v3_to_v2.h` (`namespace anira::v3compat`: `to_inference_config(model, contract, candidates)`, `to_context_config(machine)`, `to_host_config(contract, model)`, `enabled_engines()`), which the examples use until the 3.x handler lands and which is removed with it; the per-entry tensor record (`models[].tensors`: the export's name and the axis `layout`, section 5) instead of names alone; `anira_contract_hard_set_ring_dtype` (data only; the bridge takes float32). The bundled models ship as configuration files (`extras/models/**/*.model.json`, `*.contract.json`, named in `extras/models/model_files.h`), every example loads them the same way, and the 2.x fixture headers are gone; the model tree is addressed from the one root `ANIRA_EXTRAS_MODELS_DIR`.

Why first: driver-free, the largest public surface, and it makes every later milestone testable through the final API.

Exit: every example builds against the v3 configuration types; `v3.0.0-alpha.1`.

### M2 -- Tensor and the Host-only pump (target 07.09.2026)

`anira_tensor`, `anira_sync_token`, `anira_memory_handle` as the frozen 216/24/24-byte PODs, `ANIRA_DOMAIN_HOST` only; the edge registry, `anira_plan_report` and cost classes with Host rows only; `anira_machine` as a refcounted handle over the immortal `Core` behind `s_core` (`src/scheduler/Context.cpp:129`) -- thread pool core-owned, the first machine sizes it, later machines reconcile per field as `Context` does today, `anira_machine_destroy` joins nothing; `anira_handler_create(machine, pipeline)`, `anira_pipeline_add_inference`, `anira_handler_prepare(h, hard)` compiling a trivial plan over the RingChunker, which is v2's ring buffers moved off the `float` instantiation the pipeline still uses (`anira::RingBuffer`, `include/anira/utils/RingBuffer.h:39`; the send and receive vectors of `SessionElement.h:203-204`) and onto the type parameter through the `RingBufferT<T>` alias (`:42-43`), `T` the element type of the host stream on that slot -- what the driver pushes into an input ring, what the driver pops out of an output ring -- which is the ring dtype the host declared for the slot on the Hard contract (`anira_contract_hard_set_ring_dtype`, `ANIRA_DTYPE_F32` when unset), instantiated per slot at `prepare`; crossing only as `anira_ring*` with the typed block accessors; `anira_backend_desc` and the pinned `anira_engine` and `anira_provider` values replacing the five `#ifdef` paths (`InferenceBackend` members, `SessionElement`'s per-backend pointers, `InferenceThread::inference`), so nothing in the header depends on which engines a build enabled and presence is `anira_enabled_backends`; the shared engine environments of section 4 with `session.intra_op.allow_spinning = 0`; `anira::v2::LegacyProcessorStage` around existing `PrePostProcessor` subclasses; `set_input`/`get_output` as Static tensors materialised by the handler before `pre_process` and after `post_process`; the LibTorch and ExecuTorch adapters consuming `anira_ext_entry`; the core owning the real-time log drain, machine-scoped C sinks behind one trampoline on the private `thl::Logger`.

ABI work in this milestone, the largest of the roadmap:

- **Frozen at M2**: `abi/tensor.h`, `abi/machine.h`, `abi/thread.h` (with `has_exited`), `abi/stage.h` with the 64-byte `anira_stage_ctx` -- `backend` became `engine` plus `provider`, and `abi/layout-3.txt` records 64 -- `abi/handler.h` including the `_wait` twins and the typed (`void* const*`) twins of every Hard entry beside the float forwarders (section 6; the per-slot ring dtype setter `anira_contract_hard_set_ring_dtype` reaches `abi/config.h` before the alpha.1 tag), the dense plan index with `anira_handler_set_plan` / `anira_handler_get_plan` as the entire selection surface, and `anira_status anira_handler_get_static_output(const anira_handler*, uint32_t slot, const anira_tensor* dst)`, the typed whole-tensor read twin of `anira_handler_set_static_input` (section 6): the scalar `anira_handler_get_output` refuses a non-F32 Static slot instead of converting, and a per-element read has no snapshot, so a classifier vector read across a frame boundary can mix two inferences where the whole-tensor read is one. `include/anira/anira.h` becomes the C11 umbrella and `anira_all.h` adds `abi/draft/`; `abi/draft/tensor_platform.h` declares the unmeasured arms under `abi/symbols-draft.txt`.
- **Implementation**: `src/capi/{machine,handler,stage,tensor}.cpp` with the per-call `anira_stage_ctx` fill, the Static materialisation and refcounted carriers behind every descriptor and `BORROW`ed model bytes; capabilities; user-driven inference threads; the `anira_shutdown` family retargeted from `Context::shutdown()`; the internal `ANIRA_REALTIME` replaced by `ANIRA_NONBLOCKING` so the public and the internal attribute are one definition.
- **Build**: tanh_core's objects absorbed into libanira, concurrentqueue and nlohmann_json build-interface only, the `THL_LOG_COMPILED_MAX_LEVEL` override; the benchmark fixture moved to `examples/benchmark/`, the explicit install file list, `Config.cmake.in` without GTest/benchmark/tanh/concurrentqueue lookups; the internal `ANIRA_API` redefined to nothing (a PE `dumpbin /exports` check that only `anira_*` remains); `USE_*`/`ENABLE_LOGGING`/`ANIRA_VERSION`/`TORCH_CXX_FLAGS` PRIVATE; the allowlist reduced to `SYMBOL "anira_*"`; `SOVERSION`/`MACHO_*`/`OUTPUT_NAME` from the tag-derived ABI pair of `cmake/build-info.cmake`.
- **Web**: `src/platform/emscripten/em_hooks.cpp` and `anira_em.h` with the six-phase hook object; `src/emscripten-wrappers/` deleted; the wasm link on the generated `exports_wasm.txt` with exception catching kept and the `__builtin_wasm_tls_size() <= 1024` assertion; the TS `runtime/`, `config/`, `core/` (Hard half), `stages/` and `backends/JSBackend` over the C ABI, including the `has_exited` stop protocol. One addition here that is not ABI work and pays for itself immediately: the Wasm build has no blocking queue (`WaitStrategy::Blocking` is coerced away), so `run_loop`'s backoff spins a core per idle inference Worker for the whole session. An atomic wake word with `memory.atomic.notify` at the two enqueue sites lets it park instead -- invisible to the ABI, a win in every browser today, and the primitive any later externally-driven pump needs.
- **Wrappers and tests**: `anira.hpp` (configuration, Hard half, `_wait` twins) and `compat/v2.hpp` (constructor shims, `PrePostProcessor` views re-pointed per call, `LegacyProcessorStage`, `set_non_realtime` routing); the v2 suite compiled through them with the `#ifndef ANIRA_WITH_RTSAN` exclusions removed; JUCE and CLAP examples on `anira.hpp`; the `test/unload` death test retargeted at the C ABI; `test/install/consumer_c` and `consumer_tanh_first`.
- **Gates online**: 1 (symbol baselines with the draft list; the Wasm `.d.ts` diff), 3 under node emitting `layout.ts`, 5, 6. Effort beyond the pump work: 5.0 person-days.

Exit: the complete v2 test suite passes through the v3 API on the Hard path via `anira.hpp`/`compat/v2.hpp`; the JUCE and CLAP examples build on `anira.hpp`; `src/emscripten-wrappers/` is gone and the TS core runs over the C ABI; `consumer_c` runs the raw-C sketch of section 9 against the installed package, shared and static; every input and output ring in the pipeline is an instantiation of `RingBufferT<T>` and `anira_ring_dtype` reports the element type that ring actually holds; `v3.0.0-alpha.2`.

### M3 -- Async (target 07.09.2026)

`anira_handler_submit`/`anira_ticket` (`ticket_status`, `ticket_wait`, `poll`, `ticket_input_released`, `ticket_output_ready`, `ticket_error`, `ticket_user_data`, `ticket_release`), `anira_job_options` gains `set_on_complete` (M1 shipped the handle and its frame-invariant setters; the per-job deadline is the `deadline_ms` argument of `submit`), `ViewChunker` and `IdentityChunker`, lanes, `max_in_flight`, EDF ordering and `ANIRA_LATE_DROP`, `ANIRA_DELIVERY_IMMEDIATE` with `anira_job_complete_fn`, `anira_handler_bind_output`, borrowed `from_host` lifetimes with one release per submit, the submit path writing `anira_stage_ctx.ticket` on every phase of the job it submitted (the field is frozen with `abi/stage.h` at M2; the Hard pump writes `ANIRA_TICKET_INVALID`); `feat/offline-inference` absorbed (lanes survive, tickets subsume callback + poll). Still Host-only. The stale-output check (section 7, completion) enters the test harness here, not in M4. The streaming-direction precondition is met (open decisions below).

ABI work:

- **Frozen at M3**: `abi/ticket.h` -- tickets as `uint32_t` slot|generation with `ANIRA_TICKET_INVALID = 0` and `ANIRA_ERROR_CAPACITY` after `lanes * max_in_flight + 8` unreleased tickets, `submit` moving owning `acquire` tokens into the slot and carrying its per-job `void* job_user_data` back out through `anira_handler_ticket_user_data`, `allocate_input`/`allocate_output`/`free_tensor` for Host, `anira_sync_token_dup`, `anira_em_job_options_set_on_complete_js`.
- **Wrappers**: the Async half of `anira.hpp` (`Ticket` releasing its slot in the destructor, `submit` over `std::span<Tensor>`, `Ticket::user_data()`); TS `Ticket` (`userData` reading back the per-job number `submit` was given) and the host `Tensor`, completion forwarded from the Worker by `postMessage`.
- **Gates**: none new; the registry diff (`gen.py --diff-against`) classifies the appended `abi/ticket.h` names as a minor-or-pre-release change and the tag-derived `ANIRA_ABI_MINOR` moves with the tag -- the first exercise of the append rule. Effort beyond the Async work: 1.5 person-days.

Exit: offline file rendering matches v2's non-real-time output; met/late/dropped accounting tested; `v3.0.0-alpha.3`.

M1-M3 carry 8.5 of the roadmap's 12.5 ABI person-days inside a nine-day window that already holds the configuration layer, the pump and Async. M2 and M3 are therefore the milestones most likely to slip past 07.09; that is accepted rather than thinning M2, which carries the freeze-relevant half, and the cut order below is what gives if the date does not.

### M4 -- Machine probing and the Linux GPU domains (target 25.09.2026)

The prototype's `infer/` layer ported into the Machine, file by file: `infer_ctx_{vk,gl,wgpu,cuda}.c` -> the device blocks, the identity checks and the functional probes (section 4, three rungs, cached on the box); `infer_edge.c` and `infer_edge_cache` -> the registry and the plan-owned edge cache; `infer_vk_cuda.c` -> the opaque-fd rows and the three-hop bridge; `infer_engine_ort.c` -> the WebGPU and CUDA EP adapters (IOBinding, `user_compute_stream`, the EP-allocator device-id assertion, capture as a per-plan mode gated by the stale check, `ANIRA_SYNC_QUEUE_ORDERED`); `allocate_input`/`allocate_output` for the new domains; sync-token fd ownership (move at `submit`, deferred close by the inference thread); the byte image and `anira_machine_byte_image_bytes`. Order by what is measured: `WgpuBuffer`/WebGPU EP on Mesa first, then the Vulkan, dma-buf and GL export rows, then CUDA on the NVIDIA box. Build: `build-deps.sh` becomes the external-Dawn ORT variant of the anira-project/backends release, with the Dawn revision assertion of section 4. Tests: `hello_inference`'s matrix becomes the functional-probe suite under ctest, `--strict` = `ANIRA_EDGE_COST_STRICT`. This milestone needs a GPU CI runner (Mesa and NVIDIA), which anira's CI does not have.

ABI work: additive only. The bodies of `anira_tensor_init_cuda/gl_buffer/vulkan/opaque_fd/wgpu_buffer/dmabuf`, `anira_capabilities_edges`/`anira_capabilities_edge`, `anira_machine_probe`, the domain arms of `allocate_*` -- every declaration already froze in M2 and every enum value was pinned in M1, so this milestone changes no struct, no value and no signature. That is the proof that domains are additive under the ABI, and it is why a missing GPU runner cannot endanger the freeze: it only leaves values unconsumed on CI builds. Gate 2 comes online advisory, both invocations, with the first `abidw` baselines taken at `beta.1`. Effort beyond the port: 0.5 person-days.

Exit: every row of section 7's Linux tables is green against anira's Machine, strict, and the prototype's `hello_inference` runs against anira instead of its own `infer/`; `v3.0.0-beta.1`.

### M5 -- plan sets, variants, the TS package, documentation (target 25.09.2026)

One plan per (variant, candidate) pair, which is also when `anira_plan_report_plans` starts returning more than one row -- the dense plan index froze with `abi/handler.h` at M2 and `variant` is a field of the row, so no signature moves here; `anira_handler_set_plan` as atomic selection -- the setter stores the plan index, a Stateful switch clears stream state on the driving thread at the next Hard entry or job boundary; `thl::RCU` (namespace `thl`, not `thl::core`) is the primitive natively, readers registering at `prepare`, and the relaxed-atomic index of `InferenceManager.cpp:36-38` on Wasm, where RCU's `thread_local` reader state cannot exist; per-plan Hard warm-up and budget, worst case across the set; the JSON benchmark sweep with governor and wait strategy in every report; the plan report per extension consumed (`anira_plan_ext` rows); Doxygen on the C header as the reference and a Sphinx C-API page beside the C++ and web pages; the changelog; deprecation attributes on every entity of `compat/v2.hpp`.

ABI work: the TS package rewrite completed -- `abi/` generated, `CustomStage` with `prepare`/`release`, `ONNXRuntimeWebBackend` over `JSBackend` reading `anira_model_config_model_path`/`model_bytes`, typedoc; the abidiff baseline `abi/anira-0.<minor>.<arch>.abi` refreshed at `beta.2` so the M6 diff is against a known state. Effort beyond the plan-set work: 3.0 person-days.

Exit: `v3.0.0-beta.2`.

### M6 -- freeze (target 25.09.2026)

The freeze is one commit: `ANIRA_ABI_MAJOR 3`, `ANIRA_ABI_MINOR 0`; `SOVERSION 3`, `MACHO_COMPATIBILITY_VERSION 3.0`, `anira-3.dll`; `abi/symbols-3.txt`, `abi/symbols-draft.txt`, `abi/layout-3.txt` and `abi/anira-3.0.<arch>.abi` committed; `aniraConfigVersion.cmake` `SameMajorVersion`; `anira_check_abi` switching from exact match to same-major-and-newer-minor; gate 2 switching from advisory to gating; npm 3.0.0 on `latest`; the CHANGELOG entry "**ABI:** stable from this tag". Beside it, the web CI grows a test step: `build_web.yml` today only builds, and from M6 it runs the identity model with the stale check in a headless browser through the TS package on the CPU path, both engine paths (the `onnxruntime-web` JS engine registered as `"org.anira.onnxruntime-web"` and the compiled-in ORT as `ANIRA_ENGINE_ONNXRUNTIME` at `ANIRA_PROVIDER_DEFAULT`); the browser joins section 1's table with that run, at `Host`. Effort: 0.5 person-days.

Browser WebGPU is not a v3.0.0 deliverable and moves to "After v3.0" for the reason section 6b gives -- schedule and an unmeasured admission rule, with every candidate path freeze-neutral; `anira_machine_config_set_webgpu` returns `ANIRA_ERROR_NOT_SUPPORTED` under Emscripten in 3.0. Two things ride with M6 because they cannot be added later. The headless-browser step gains a one-day spike that runs ORT-Web's WebGPU execution provider behind a GPU proxy Worker and reports end-to-end block latency and rendezvous wake latency -- the number the whole post-3.0 design turns on, and the only deliverable here that is not a design. And the thread rule of `anira_inference_thread_execute` is written into the tag table: a thread that pumps must not be a thread that calls a `_wait` twin, `anira_handler_ticket_wait`, `anira_handler_prepare` or `anira_handler_destroy`, because a pump driven from a waiting thread never delivers its own completion. The rule constrains an entry point that does freeze in 3.0.0; the asynchronous slot itself does not need reserving, because descriptor tail growth already covers it.

Exit: `v3.0.0`.

### ABI gates

| gate | name | what it checks | where it runs | online |
|---|---|---|---|---|
| 1 | `anira_symbol_baseline` | the real export table (`nm -D` / `nm -gU` / `dumpbin`, through `_tanh_exports` copied into `anira/cmake/abi-symbols.cmake`) against `abi/symbols-<major>.txt` ∪ `abi/symbols-draft.txt`: a promised name missing fails, a name outside both lists fails, the draft list is tolerated in both directions, an added promised name is admitted by the registry diff (`gen.py --diff-against <last tag>`: appended = minor or pre-release, removed, renamed or changed = never), the ABI minor being derived from the tag rather than bumped by hand; on PE additionally "`dumpbin /exports` lists only `anira_*`" | every native shared leg of `build_test`; on Wasm the link against `exports_wasm.txt` is the presence check and the `--emit-tsd` `.d.ts` export diff in `build_web` is the extras check | M2 |
| 2 | `anira_abi_diff` | `abidiff` of `abidw` baselines `abi/anira-<major>.<minor>.<arch>.abi` with `--headers-dir` on `include/anira/abi`, `abi/abidiff-draft.supp`, `--no-added-syms`, `--fail-no-debug-info`: exit bit 8 always fails, bit 4 fails unless the minor was bumped since the last tag; a second invocation restricted to the Tier-1 headers (`tensor.h`, `stage.h`, `log.h`, `status.h`) fails on bit 4 regardless | Linux x86_64 and aarch64, `RelWithDebInfo`, apt `abigail-tools`, one extra `ubuntu-24.04` job | M4 advisory, M6 gating |
| 3 | `anira_abi_layout` | `test/abi/test_layout.c`: `_Static_assert` of every `_FORCE32` and of `ANIRA_ABI_VERSION`, plus an executable emitting every Tier-1 `sizeof`/`offsetof` and diffing it against the committed `abi/layout-<major>.txt`, which only a commit changing `ANIRA_ABI_MAJOR` may touch; under node it also emits `web/src/abi/layout.ts` and asserts `__builtin_wasm_tls_size() <= 1024` | every leg natively; OBJECT target under the `wasm-*` and mobile presets (`ANIRA_WITH_ABI_TESTS`), run under node in `build_web` | M1 native, M2 node |
| 4 | `anira_header_c11` | `test/abi/header_c.c` = `#include <anira/anira.h>` + `<anira/anira_em.h>` under `-std=c11 -Wall -Wextra -Werror -pedantic` and `/std:c11 /W4 /WX`, with no anira define at all; `header_cxx17.cpp` (`anira.h` never needs C++20, `anira.hpp` may); `anira_header_isolation` made strict -- negative `try_compile` that no `tanh/`, `nlohmann/`, `concurrentqueue` or `benchmark/` include is reachable through `anira::anira` | OBJECT target on every preset including mobile and Wasm (today `anira_header_isolation` is desktop-only) | M1 (per file), M2 (umbrella) |
| 5 | `consumer_c`, `consumer_tanh_first` | a pure-C consumer of the installed package (`LANGUAGES C CXX` with `LINKER_LANGUAGE CXX` on the static leg, because anira itself is C++) running the raw-C sketch of section 9 with one registered engine carrying `ANIRA_BACKEND_NEEDS_NO_MODEL` substituted for its engine candidates, so the consumer needs neither an engine nor a model file, plus `anira_shutdown`, shared and static legs; `consumer_tanh_first` links a shared `tanh::Core` before `anira::anira` and proves with `nm`/`dumpbin` that the module binds `thl` locally; the `consumer_engine_module` export scan additionally forbids `anira_` | `build_install` (Linux, macOS, Windows; `consumer_tanh_first` on ELF and Mach-O) | M2 |
| 6 | RTSan leg | every `ANIRA_NONBLOCKING` entry and callback typedef instrumented at runtime through the v2 suite (the `blocking_ratio`/`set_non_realtime` tests now call the `_wait` twins and lose their `#ifndef ANIRA_WITH_RTSAN`); `-Werror=function-effects` PRIVATE on the consumer-shaped targets only -- `test/abi/*.c`, `test/abi/test_rt_contract.c` (a `nonblocking` body calling every `[callback-safe]` entry), `consumer_c`, the header compile -- never on target `anira` | `build_sanitizer` (clang 20 via ci-actions; the leg confirms `__has_attribute(nonblocking)`) | M2 |

`tools/abi/gen.py` generates the headers, the TypeScript mirror, the symbol lists, the proc table and the layout test from `abi/anira.yml`, validating the header conventions of section 6a while doing so; its outputs are committed and the `anira_abi_generate` CTest (and `build_web.yml`) regenerate and diff them, so a missing Python never blocks a build. tanh-tooling stays off the critical path: `ALLOW_PREFIXES`, `EXPECT_SYMBOLS`, a Wasm branch that checks instead of skipping (`cmake/tanh/check-exports.cmake:68` skips today) and `tanh_add_abi_check()` are the v0.1.6 items that fold gates 1-2 back into the shared modules after v3.0.0; `_tanh_exports` is already a function (`cmake/tanh/check-exports.cmake:186`) and `ALLOW_REGEX` already a parameter of `tanh_add_export_check` (`:53,115`), which is what lets anira copy the one and pass the other from M1.

### Cut order

If the schedule slips, cuts happen in this order, first cut first:

1. The abidiff job -- the symbol baselines and the layout table remain and catch every removal and every RT-layout change, on Wasm too.
2. `anira_get_proc_address` with `ANIRA_CXX_MANUAL_INIT`/`ANIRA_NO_PROTOTYPES` -- `dlopen` hosts use `dlsym` per function.
3. `anira_model_config_to_json`/`anira_plan_report_to_json` -- the accessors suffice.
4. `anira_em_job_options_set_on_complete_js` -- polled tickets in the browser first.
5. The TS package rewrite beyond `abi/` and the raw exports, which the docs already bless on the RT path.
6. `compat/v2.hpp` beyond `LegacyProcessorStage` -- the migration is documented instead.
7. `consumer_tanh_first` -- the link-order rule stays documented.

Never cut:

- the pinned enum values (M1);
- the frozen `anira_tensor` and `abi/layout-3.txt` (M2);
- `SYMBOL "anira_*"`, the C11 header compile and the symbol baselines;
- the tanh absorption and the deletion of `src/emscripten-wrappers/` (M2);
- `anira_check_abi` and the exception firewall;
- the `_wait` twins -- without them the `ANIRA_NONBLOCKING` attribute on `process` is false.

### After v3.0 (additive under ABI major 3; section 10)

Every item below lands as an appended function, enum value, extension revision or descriptor tail slot with an `ANIRA_ABI_MINOR` bump; none needs a major.

- `anira_frame` + `FrameToTensor` + `submit_frame` as `abi/draft/frame.h`, promoted when measured; `request_output`; `ANIRA_GL_SHARED_CONTEXT`.
- Typed Hard entries -- **moved into M2 on 2026-09-03: Hard contracts are not float-only, and a rule that made every v3.0.0 ring F32 would have been read as one on the model**: a per-slot declaration of the host stream's element type on the contract -- `anira_contract_hard_set_ring_dtype(c, canonical, dtype)`, defaulting to `ANIRA_DTYPE_F32`, on `abi/config.h` before the alpha.1 tag -- plus typed twins of the Hard entries (`anira_handler_process_typed` / `_process_separate_typed` / `_push_data_typed` / `_pop_data_typed` and their `_wait` forms) taking `void* const*` where today's take `float* const*`. The dtype belongs on the slot at `prepare` and not on the call: one `tensor_index` addresses an input slot and an output slot at once, their rings may hold different element types, and a single per-call dtype argument cannot describe both. The float entries stay as forwarders -- `anira_handler_process(h, d, n, ti)` calls the typed twin with `(void* const*)d` and is legal on F32 slots only -- and both families freeze together in `abi/handler.h` at M2, same `[driver-thread]` tag and `ANIRA_NONBLOCKING` class.
- The Windows, Apple and Android GPU columns, each after its own matrix run, promoting `anira_tensor_init_metal/iosurface/ahardwarebuffer/d3d12` from `abi/draft/tensor_platform.h` by moving names between baselines, never renaming.
- The deferred extension catalogue; third-party extension kinds (`anira_register_ext_kind` plus a prepare-time payload accessor, the reverse-URI prefix reserved now).
- Browser WebGPU as an explicitly asynchronous JS backend -- an `anira_backend_desc` tail slot `process_async` with a completion callback driven by `anira_inference_thread_execute` from an unblocked Worker pump, or a JSPI-linked second module variant sharing the header, with `anira_em_machine_config_set_js_webgpu` reserved for the `GPUDevice` import; the Machine borrows the browser's `GPUDevice` through Emscripten's WebGPU bindings instead of linking Dawn, so that device is always `ANIRA_OWNERSHIP_BORROWED` and the Dawn revision assertion of section 4 does not apply there; `WgpuBuffer` tensors are `GPUBuffer`s the page's own compute passes write; the TS package exposes `allocate_input`/`allocate_output` beside the tickets it already has for that case; and the browser joins section 1's table only after the M4 functional-probe suite has run there, like every other platform column.
- A browser video path: WebCodecs `VideoFrame` and `GPUExternalTexture` into the WebAssembly build. Neither has a representation in the header or in the TypeScript package today, so video input is absent end to end on the one platform whose decoders are already present.
- Chained `Inference` stages if a model pair demands it; a host thread-provider vtable; engine-level batching of Async jobs.
- Per-class inference-thread groups: a machine that requests a scheduling class other than the default is served by its own thread group, the default group staying RealTime so 3.0.0 clients are unaffected -- `anira_machine_config_set_thread_class` plus a `thread_class` key in `machine.json`. This is what removes both the real-time default and the head-of-line blocking of a long job in front of a short one.
- `compat/v2.hpp` removed one minor after 3.0.
- the tanh-tooling tag that folds gates 1-2 into the shared modules (v0.2.8 is pinned today), re-pinned by anira and tanh-lib together. (The tanh-lib option `TANH_LOG_COMPILED_MAX_LEVEL` landed early, with tanh-lab/tanh-lib#37, and anira uses it.)
- `-Werror=function-effects` widened to `src/capi/` once `InferenceManager`/`Context`/`SessionElement`, `thl::core::RingBuffer` and a moodycamel wrapper are annotated.
- A Python `ctypes`/`cffi` package over `anira.h` -- the header has no function-like macro in a declaration except `ANIRA_API`/`ANIRA_CALL`/`ANIRA_NONBLOCKING`/`ANIRA_NOEXCEPT`, no bitfields, no `_Static_assert`.

### Batching, as a worked example of the tier split

Batching is not a v3.0.0 deliverable and does not compete with M1-M6 for schedule: it can land at 3.1 or 3.4 without anyone relinking. It is worth stating here because it is the first substantial feature to arrive after the freeze, and tracing it through the surface is the test of whether the tier split of section 6a was drawn in the right place.

| what batching needs | where it lives | verdict |
|---|---|---|
| declare a batch extent | `anira_tensor_spec_set_axis(s, 0, ANIRA_AXIS_BATCH, 4)` | exists; `ANIRA_AXIS_BATCH` is pinned at M1 |
| carry it at runtime | `anira_tensor.shape[]` and `ndim` | no change -- a batch axis is one shape entry, and the 216 bytes do not move |
| batched window extraction | `anira_ring_pop_windows(..., num_batches)` | exists; it is v2's `pop_samples_from_buffer(..., num_batches)` |
| batch sizes as alternatives | `anira_pipeline_add_inference(variants, ...)` with `anira_handler_set_plan` | exists; a batch size is a variant like any other |
| report which batch a plan chose | `anira_plan_info` -- Tier 2, `struct_size` first | append `uint32_t batch;` after the last v3.0.0 field, because a batch is a property of the plan and not of one slot; `ANIRA_ABI_MINOR` bump |
| an on/off or cap knob | a setter on an opaque config handle | a new function; the handle has no public layout to disturb |

Nothing lands on a Tier-1 POD, so nothing forces a major. That is the tier split doing what it was drawn for rather than luck: the batch extent rides in a frozen array that already has eight slots, the declaration rides on a handle with no layout at all, the reporting rides on a descriptor built to grow at the tail, and the selection reuses the variant machinery.

The one judgement call sits past that, at cross-ticket coalescing with a genuinely variable batch, which would relax `ANIRA_DYNAMIC`'s "Buffer-role Time axis only" rule (section 2). No symbol, layout or enum value changes, and no previously-working program breaks -- configurations that used to be rejected start being accepted, which is a minor with a prominent `**ABI:**` changelog entry rather than a major. The blocker there is not the ABI at all: a variable batch has no single answer to "what is the budget", and under Hard the budget is the promise.

Batching also names the gap the plan report would otherwise have had. The report is one row per plan and `budget_ms` is a field of that row, so batch-1, batch-4 and batch-16 each report their own number. A row per engine could not: a small and a large model on one engine differ in cost by construction, and with batch-1 / batch-4 / batch-16 the variants are identical *except* in cost, so a report that collapses them reports nothing. That is why the row carries `variant` beside `budget_ms` from M2 (section 6), before any variant set exists to need it.

### Open decisions, and the milestone that needs them

- **Streaming direction**: done. Streamed tensors on one side only are first-class -- section 2's legality rule, section 5's anchor that may be an output, section 7's RingChunker with independently sized sides. The v2 half is in `[Unreleased]` (#98, #99, #110: `HostConfig::k_first_streamable`, `m_tensor_is_input`, `resolve_reference()`, `calculate_{send,receive}_buffer_sizes`, `test/test_OneSidedStreaming.cpp`); M2 carries it across unchanged as one `anira_ring` per streamed slot.
- **Chunker element type**: done as a decision; the storage work is M2. What landed with the containers entry of `[Unreleased]` is the type-parameterised alias (`RingBufferT<T>`, `include/anira/utils/RingBuffer.h:42-43`, block API on every real-time access), instantiated nowhere in `src/` or `include/` outside that header; the pipeline's input and output rings are still the `float` instantiation (`:39`, `SessionElement.h:203-204`) and v2 has no `dtype` concept at all, so moving them onto the type parameter is real M2 work and not a description of the tree. The ring is typed by the host stream and not by the model tensor -- an input ring holds what the driver pushed, an output ring holds what the driver will pop (section 7). A ring takes the ring dtype the host declares for its slot on the Hard contract (`anira_contract_hard_set_ring_dtype`, `ANIRA_DTYPE_F32` when unset), and the typed Hard entries ship at M2 beside the float forwarders (decision 2026-09-03: Hard is not float-only, and no dtype rule falls on the model from the entry point). The `anira_ring_*` accessors are nevertheless typed rather than `float`-typed with an integer family bolted on later, because the accessor family cannot be retyped after M6: every data entry takes a `void*` plus the `anira_dtype` the caller believes it is reading, `anira_ring_dtype` reports the ring's own, and disagreement is 0 / `ANIRA_ERROR_CONFIG` in `anira_handler_rt_error` with no conversion either way (section 7). What that freezes is what a stage may assume: it reads `anira_ring_dtype(r)` and never infers the ring's element type from its slot's spec.
- **Engine and provider**: done. Two independent enums, `anira_engine` and `anira_provider`, replace the packed `anira_backend`, because a provider name means the same thing across engines -- CUDA is CUDA whether ONNX Runtime or LibTorch runs it -- which the packed form hid; the pair travels as one item only as the Tier-2 `anira_backend_id`. Both are pinned at M1, and `anira_stage_ctx` grows from 56 to 64 bytes for it, pre-freeze, recorded as 64 in `abi/layout-3.txt`. `anira_backend_id` itself lands in `abi/machine.h` at M2 with its first consumers; M1 pins only `ANIRA_STRUCT_BACKEND_ID = 11`.
- **Custom engines**: done. A custom engine is registered by name, `anira_pipeline_register_engine(pipe, engine_id, desc, err)`, never by slot number; the name must be reverse-URI, the same reservation third-party extension kinds already make, so anira's own short engine names can never collide, and registering one name twice in a pipeline is `ANIRA_ERROR_CONFIG` rather than a silent replace. It resolves to an `anira_engine` value from `0x1000` up at `anira_handler_prepare`, pipeline-scoped and meaningless outside it, which is why `anira_backend_id` and `anira_plan_info` both carry `engine_id` as a string. `anira_pipeline_set_custom_backend` and `ANIRA_BACKEND_CUSTOM` are gone; the Emscripten twin is `anira_em_pipeline_register_js_engine`, which passes the id the JS engine needs to find its own model row.
- **Plan addressing and selection**: done. A plan is a dense index `0..num_plans-1`, its row is `anira_plan_info` with `budget_ms` as a field, and `anira_handler_set_plan` / `anira_handler_get_plan` is the entire C selection surface -- a dense index cannot name a plan that does not exist, enumerating the set is one loop, a later dimension appends a field to the row instead of moving every signature, and the index the report hands out is the index selection takes. `anira.hpp` keeps `set_model(variant)` and `set_backend(engine, provider)` as O(1) conveniences over the `(variant, engine, provider)` -> plan table built at `prepare`; M2 freezes both entries, M5 is when more than one row exists.
- **Sparse plan sets**: done. A (variant, candidate) pair whose variant carries no model entry for that candidate is not a plan and is not an error, with one declared exemption: `ANIRA_BACKEND_NEEDS_NO_MODEL` in `anira_backend_desc.flags` forms a plan for every variant regardless of model entries, which is the normal shape of a passthrough or a mock. A real engine whose model file is missing is still `ANIRA_ERROR_MODEL_LOAD` at `prepare` -- needs-no-model and model-missing are different and never collapse.
- **GPU CI hardware** for M4: which machines run the functional-probe suite. Because the M4 surface is additive, its absence cannot break the M6 freeze -- it leaves enum values unconsumed on CI builds. Recommendation: ship the probe cache as a fixture so CPU-only CI exercises the planner against a recorded registry.
- **`ext::Entry`** (`anira_ext_entry`): chosen here (section 1b) over one extension per adapter; revisited only if a third adapter needs a different shape.
- **Anchor override on `Hard`** stays deferred until a multi-rate model exists.

Owner decisions on the ABI, each with the recommendation:

1. **M0 tag name**: `v2.3.0` with the no-binary-promise preamble (recommended) or `v3.0.0-alpha.0`. `alpha.0` renames `on_tag.yml` artefacts, the backends-release vocabulary and every "v2 parity" reference the day before the tag, and pulls `cmake/build-info.cmake` into M0.
2. **Browser WebGPU**: out of v3.0.0 as the asynchronous backend above (recommended), or a JSPI-linked second module variant at M6, which needs an ORT wasm archive built for JSPI that the backends release does not carry.
3. **Ticket width**: `uint32_t` slot16|gen16 (recommended) or `uint64_t`. 65,535 slots and a 16-bit generation are ample for `lanes * max_in_flight + 8`, and `uint32_t` keeps `bigint` off every TS Async call.
4. **Layout identity via `ANIRA_PTR`** (recommended) or native pointers. Identity gives one `layout.ts` and one offset table on every target for one cast per producer field and the zero-the-struct rule; native pointers give 216/208 sizes and two tables.
5. **`anira_dtype` as packed `uint32_t`** (recommended) or `const anira_dtype*`. Packed keeps every signature scalar and its bytes equal `DLDataType`, so `from_dlpack` is a `memcpy`.
6. **`anira.hpp` requires C++20** (recommended); `anira.h` compiles as C11 and C++17. Designated initialisers and `std::span` are the reason; a C++17 host uses the C header.
7. **The gates-1-2 tanh-tooling tag after v3.0.0** (recommended) or bundled with M2. The M1-M3 window is the scarce resource and the anira-local scripts total about eighty lines.
8. **`ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK` default off** (recommended, v2 behaviour: the platform sink stays on beside a callback) or on, which avoids duplicate lines in DAWs that show stderr.
9. **Host thread-provider vtable**: deferred (recommended). `num_threads = 0` plus `anira_inference_thread_*` already lets a host bring its threads; a provider that spawns threads for anira needs a `[thread-safe]` spawn/join contract not yet measured.
10. **Wasm exception mode**: keep `-sNO_DISABLE_EXCEPTION_CATCHING` (recommended) or move to `-fwasm-exceptions`, which is smaller and faster but needs the ORT wasm archive built with it.
11. **Two "anira-log" drain threads** when a host also links tanh-lib: accept (recommended); a merge would make tanh-lib a shared dependency again, which is the situation being left.
12. **`anira_frame` absent from the 3.0.0 header** (recommended): values reserved now, the struct ships with `submit_frame` in `abi/draft/frame.h`; a frozen unmeasured union cannot be corrected in a minor.
13. **Unmeasured platform arms in `abi/draft/tensor_platform.h` with `symbols-draft.txt`** (recommended) or in `abi/tensor.h` outside the promise by comment. The draft folder makes the rule mechanical.
14. **GPU CI hardware for M4**: as above; recommendation: the probe cache as a fixture.
15. **Where the edge-cost policy lives**: `anira_contract_set_edge_cost` on the contract handle (recommended) or a separate handle. The contract file carries `edge_cost` and `prepare` takes one object; `anira.hpp` keeps the `edge_cost` member of `Hard`/`Async`.
16. **Third-party extension kinds post-3.0** (recommended) or `anira_register_ext_kind` plus a payload accessor in v3.0.0. Post-3.0 is additive and no external consumer exists; in-3.0 costs about a day inside the M2 week.

## TODO

- Batching: settled as an outlook, section 11 ("Batching, as a worked example of the tier split"). Every piece of it is additive inside ABI major 3, so it is out of the v3.0.0 schedule; the one open question is engine-level coalescing of Async jobs across tickets, whose blocker is the budget of a variable batch rather than anything in the ABI.
