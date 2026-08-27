# anira v3: Universal Tensor Architecture

Consolidated design. One passive tensor type carrying memory, ownership and readiness across CPU, GPU and NPU; one spec carrying model meaning; two scheduling contracts; a prepare-time planner that compiles declared capabilities into a fixed, validated pipeline. The hard real-time audio path (`process` / `push_data` / `pop_data`) is frozen by design.

Layering rule: `MachineConfig` owns machine resources, `ModelConfig` owns model semantics, the `Contract` owns the scheduling regime of one handler, per-`submit` options own job particulars. Semantics live in exactly one place; storage may live wherever engineering needs it. Section 11 is the order in which this is built: one non-breaking v2.3.0 release, then the v3 milestones.

---

## 1. Runtime Tensor (user <-> anira data unit)

The Tensor is a POD descriptor, trivially copyable through lock-free FIFOs. It carries only user-to-anira information; every anira-to-user signal lives on the Ticket. Graphics handles stay representable through the pipeline and are erased only at the backend adapter (enables accelerator switching under the same input and GPU pre/post stages).

Typing rule: every `MemoryHandle` arm is typeless memory; the tensor descriptor (`dtype`, `shape`, `strides`, `byte_offset`) is the only type. Pixel formats never appear on a Tensor. Image-typed data (camera buffers, decoder output, rendered frames) enters as a `Frame` (section 1a) and exists as a Tensor only after a `FrameToTensor` stage.

```cpp
inline constexpr int k_max_rank = 8;

enum class Domain : uint8_t { Host, HostPinned, Cuda, GlBuffer,
    VulkanBuffer, OpaqueFd, MetalBuffer, WgpuBuffer,
    DmaBuf, IOSurface, AHardwareBuffer, D3D12 };

union MemoryHandle {
    void* host;
    struct { void* ptr; int32_t device; }                       cuda;
    struct { uint32_t id; uint32_t target; }                    gl;      // GLuint + GLenum
    struct { VkBuffer buf; VkDeviceMemory mem; uint64_t off; }  vk;      // native, same-process
    struct { int fd; uint64_t size; }                           opaque;  // exported opaque fd / NT handle
    void*                                                       mtl;     // id<MTLBuffer>
    struct { void* surface; uint64_t size; }                    iosurface;
                                                                         // IOSurfaceRef, plane 0,
                                                                         // BYTE-IMAGE-ENCODED packed
                                                                         // floats (pixel CVPixelBuffers
                                                                         // are Frames)
    struct { WGPUBuffer buf; uint64_t off; }                    wgpu;    // same process, the Machine's Dawn device
    struct { int fd; uint64_t size; uint64_t off; }             dmabuf;  // EXPORTED BUFFER MEMORY, typeless
                                                                         // (multi-plane image dma-bufs are Frames)
    AHardwareBuffer*                                            ahb;     // buffer-typed AHB (BLOB)
    struct { void* resource; void* shared_handle; }             d3d12;
};

enum class SyncKind : uint8_t { None, QueueOrdered, CudaEvent, VkTimeline, GlSync,
    SyncFileFd, OpaqueFdSemaphore, MtlSharedEvent, D3D12Fence };
// QueueOrdered: complete when the producing queue reaches it (a WGPUBuffer; GL after
// cudaGraphicsUnmapResources) -- a same-queue consumer waits for nothing, the host waits
// on the queue; never a fabricated fence. OpaqueFdSemaphore: a Vulkan semaphore exported
// OPAQUE_FD, the only fence CUDA imports.

struct SyncToken {
    SyncKind kind = SyncKind::None;                 // None = data already visible
    union { cudaEvent_t cuda;
            struct { VkSemaphore sem; uint64_t value; } vk;
            void* gl;                               // GLsync
            int fd;                                 // SyncFileFd (dmabuf / AHB world) and
                                                    // OpaqueFdSemaphore; both owned
            struct { void* obj; uint64_t value; } mtl, d3d12; };
};

enum TensorFlags : uint32_t { ReadOnly = 1, DiscardContents = 2, HostCoherent = 4 };

struct Tensor {                       // trivially copyable, ~220 bytes
    Domain       domain;
    MemoryHandle handle;
    DType        dtype;               // code (int/uint/float/bf16/bool) + bits + lanes
    uint8_t      ndim;
    int64_t      shape  [k_max_rank];
    int64_t      strides[k_max_rank]; // in elements; all-zero = packed row-major
    uint64_t     byte_offset;         // what the ViewChunker slices with
    uint32_t     flags;

    void*        manager_ctx;         // PRODUCER bookkeeping only (pool slot,
                                      // refcounted view parent); never edge state
    void       (*release)(Tensor*);   // unmap / unregister / recycle / free;
                                      // nullptr = borrowed (valid until ticket completes)

    SyncToken    acquire;             // input: data valid once this signals
                                      // bound output: buffer free to write once this signals
};
```

Which arms are enabled is decided by one rule, applied per platform -- never by a producer's preference. A domain is enabled when an engine reads it natively on hardware anira shares (`WgpuBuffer` with the WebGPU EP, `Cuda` with the CUDA EP, `D3D12` with DirectML, `AHardwareBuffer`/`GlBuffer` with LiteRT on Android), or when it is a producer API anira allocates for (`VulkanBuffer`, `GlBuffer`: section 6, `allocate_input`). Beside those sit the platform's *crossing currencies* -- not APIs that compute, but the shape memory travels in: `DmaBuf` and `OpaqueFd` on Linux, an NT handle on Windows (the same `opaque` arm), `AHardwareBuffer` on Android, `IOSurface` on Apple. Two of these are image-typed primitives -- `IOSurface` always carries a pixel format, `AHardwareBuffer` and `DmaBuf` can -- and they therefore appear in two roles, which the boundary rule keeps apart: as a *Tensor domain* when the format is the edge's own byte-image encoding of packed floats (section 7) and the payload is bytes in order, and as a *Frame container* when the format is real and must be interpreted. `Domain::IOSurface` is the first; a camera's NV12 `CVPixelBuffer` is the second. Boundary rule: a Tensor is bytes in order, a Frame is bytes with a layout -- a handle that would need a modifier, a pitch or a format on the Tensor is a Frame (section 1a).

What that yields, v1:

| platform | enabled domains | crossing currency | fences |
|---|---|---|---|
| Linux | `Host`, `HostPinned`, `Cuda`, `WgpuBuffer`, `VulkanBuffer`, `GlBuffer`, `DmaBuf`, `OpaqueFd` | dma-buf fd; opaque fd for CUDA | `SyncFileFd`, `OpaqueFdSemaphore`, `CudaEvent`, `QueueOrdered` |
| Windows | `Host`, `HostPinned`, `Cuda`, `WgpuBuffer`, `D3D12`, `VulkanBuffer`, `GlBuffer` | NT shared handle (`opaque`) | `D3D12Fence`, `CudaEvent`, `QueueOrdered` |
| macOS / iOS | `Host`, `WgpuBuffer`, `MetalBuffer`, `IOSurface` | `IOSurface` (`IOSurfaceRef`, public on both since iOS 11); `MTLBuffer` is reach-in on the shared `MTLDevice` | `MtlSharedEvent`, `QueueOrdered` |
| Android | `Host`, `WgpuBuffer`, `GlBuffer`, `VulkanBuffer`, `AHardwareBuffer`, `DmaBuf` | `AHardwareBuffer` (BLOB for tensors, image for Frames) | `SyncFileFd`, `GlSync`, `QueueOrdered` |

Two things hold everywhere. `Host` is the floor on every platform -- the v2 path, every CPU engine, no regression in coverage. And `WgpuBuffer` is the portable fast domain: Dawn has Vulkan, Metal and D3D12 backends, so `WgpuBuffer -> WebGPU EP` is `ZeroCopy` on all four platforms even where every other row differs.

Absences are decisions, not omissions. `GlBuffer` is enabled wherever its one reach-in row exists -- `cudaGraphicsGLRegisterBuffer` on Linux, Windows and Android -- and absent on Apple, where GL is deprecated, frozen below compute (4.1) and has no CUDA to register into: every remaining GL row there is `glGetBufferSubData`, which is `from_host` under another name, so the domain would earn nothing. A GL texture backed by an `IOSurface` is a Frame, not a `GlBuffer`. `Cuda` and `HostPinned` are absent on Apple for the same hard reason (no CUDA since 10.2, none on Apple silicon). **OpenCL is excluded outright, on every platform and not merely deferred**: it is first-party nowhere -- deprecated on Apple since 2018, absent from the Android NDK with no guarantee that a device ships an ICD, vendor-supplied elsewhere -- so it would add a `dlopen`'d dependency and a context to borrow without adding reach, since every accelerator that offers a CL path (LiteRT's GPU delegate on Android) reaches the same hardware through `AHardwareBuffer` or `GlBuffer`, which are first-party. An engine may use CL internally; anira neither takes a `cl_mem` nor hands one out. It is absent from `Domain`, `SyncKind` and `MachineConfig`, and is not on the additive list. Two Windows wrinkles worth stating before they surprise anyone: Dawn's default Windows backend is D3D12, so `D3D12` is the natural WebGPU pairing there while `VulkanBuffer -> WebGPU` would require Dawn forced onto its Vulkan backend -- `VulkanBuffer` is enabled on Windows for the CUDA row (opaque NT handle), not the WebGPU one; and `GlBuffer -> WebGPU` is `HostCopy` on Windows, since no Dawn D3D12 path imports GL memory.

Measurement status. Only the Linux column is measured -- Mesa Honeykrisp for the 50-cell matrix and an NVIDIA Turing box for the CUDA rows (section 7). The others follow from the rule and each needs its own measured cell before its rows ship -- the doc's standard is that no row exists without a measurement, and a per-platform matrix run is what turns these from declarations into rows. Two asymmetries to expect rather than assume. Dawn's `SharedBufferMemory` exists on D3D12, so Windows may get a true `ZeroCopy` buffer import into WebGPU where Linux needs the byte image (section 7). And Apple is expected to mirror Linux almost exactly: Dawn's Metal backend imports `SharedTextureMemoryIOSurface` -- textures, like the dma-buf path -- so `IOSurface -> WgpuBuffer` should be the same `DeviceCopy` byte-image relayout, with `MtlSharedEvent` in place of the sync file. The open question there is the *alias*, the Apple twin of the prototype's `VkImage` + `VkBuffer` pair: whether an `MTLBuffer` can be made to view the same `IOSurface` memory (`IOSurfaceGetBaseAddress` is page-aligned on UMA, which `newBufferWithBytesNoCopy:` wants) so the user's compute shader writes an ordinary buffer while Dawn imports a texture. If it can, `MetalBuffer` and `IOSurface` are two views of one allocation as on Linux; if not, `allocate_*` hands back a texture-writing path instead. To be measured, not assumed.

Sync tokens and WebGPU: no WebGPU-specific `SyncKind`. Dawn exposes readiness as a `SharedFence`, which is a sync file on the Vulkan backend (`SyncFileFd`) -- or, where the consumer is CUDA, a `VkSemaphore` exported `OPAQUE_FD` (`OpaqueFdSemaphore`, the only fence CUDA imports; Dawn's Vulkan backend exports both kinds) -- a `D3D12Fence` on D3D12 and an `MtlSharedEvent` on Metal; the WebGPU adapter converts internally. Verified end to end in the prototype: one sync file crosses Vulkan (`vkGetSemaphoreFdKHR` out, `vkImportSemaphoreFdKHR` with `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` back in), Dawn (`ImportSharedFence` in, `EndAccess` fence out) and EGL (`EGL_SYNC_NATIVE_FENCE_ANDROID`) with no API learning that the others exist.

Sync token ownership, the two fd kinds (`SyncFileFd`, `OpaqueFdSemaphore`) only: the token owns its fd and closes it when reset or replaced; every hand-off is a transfer, and an importer that needs the fd past the call dups it (Dawn does). A producer must not close an fd it handed to a Tensor, and an adapter importing into an API that takes ownership — Vulkan's temporary semaphore import — clears the token instead of closing it. Unstated, this is a double close or a leak on every frame; the other `SyncKind` arms are non-owning handles and need no such rule.

Construction (edge factories, all producing this one struct):
`from_host`, `from_pinned`, `from_cuda(ptr, device, event)`, `from_gl_buffer(id, target, glsync)`, `from_vulkan(buf, mem, off, timeline)`, `from_opaque_fd(fd, size)`, `from_metal(buf, shared_event)`, `from_wgpu_buffer(buf, off, fence)`, `from_dmabuf(fd, size, off, sync_fd)`, `from_ahardwarebuffer(ahb, fence_fd)`, `from_d3d12(resource, shared_handle, fence)`, `from_dlpack(capsule)`.

Rules: one tensor per submit; multi-buffering is a producer-side pattern over tokens (rotate descriptors, reuse a slot when its `input_released` token signals) -- and it stays on the producer's side of the edge: an engine whose graph is captured replays the buffers it was captured with (section 7, completion), so the tensors bound to the engine are fixed and the edge moves each rotating slot into them. GL factories require the context thread policy from `MachineConfig`. `from_dmabuf` builds a Tensor from exported *buffer* memory (Vulkan `VK_KHR_external_memory_fd`, GBM linear bo, dma-heap) and is typed by the descriptor like every other arm. Images of any API (GL textures, `VkImage`, `WGPUTexture`, `MTLTexture`) are not Tensors: they are `Frame` containers and enter via a `FrameToTensor` stage, which is a zero-copy import plus one shader pass or an explicit device copy; the plan reports which.

Where edge state lives: not on the Tensor. An edge that caches an expensive import (a Dawn `SharedTextureMemory` over a dma-buf, a CUDA graphics registration, an NPU registration) keeps that cache **in the compiled plan**, keyed by the incoming memory handle, because the same plan sees a rotating set of descriptors under producer-side multi-buffering and must hit the cache for each slot rather than rebuild on every alternation. `manager_ctx` stays the producer's, and only the producer's. The prototype put edge state on the tensor (no plan exists there to hold it) and immediately hit the collision: a Vulkan tensor needs per-tensor producer state of its own, so the two uses fight over one pointer and are separable only while Vulkan tensors are never an edge *destination* — which v3 does not guarantee.

Deliberately absent: stream/queue affinity (resolved at the adapter against the consumer's stream, DLPack-style), quantization and NCHW/NHWC semantics (spec), tensor names (spec, binding is positional), conversion methods (stages act, data flows), callbacks/mutexes/refcounts in the public struct, pixel formats (Frame), extensions (section 1b: config-time structs only -- a POD that travels through a FIFO owns no pointer).

---

## 1a. Frame (image data, user -> FrameToTensor stage)

A `Frame` is image data in a platform image-sharing container, described by a pixel format instead of a tensor descriptor: planes, per-plane pitch, chroma subsampling and colorimetry cannot be expressed as `dtype/shape/strides`, and a `VkBuffer`/SSBO cannot carry a format. It is a POD like `Tensor`, carries the same sync/ownership fields, and is accepted by exactly one kind of stage: `FrameToTensor`. A Frame is never chunked, windowed or bound as an output.

Deferred past v1: `Frame`, its factories, `submit_frame` and `FrameToTensor` are declared here so the vocabulary is complete and ship in a later minor. In v1 the user turns pixels into a float Tensor before `submit` -- their own shader writing an `allocate_input` buffer, or a host loop -- and the boundary a Tensor crosses is unchanged when the stage arrives.

```cpp
enum class Container : uint8_t { DmaBuf, AHardwareBuffer, IOSurface, DxgiHandle,
    VkImage, GlTexture, WgpuTexture, MtlTexture, Host };

enum class PixelFormat : uint32_t { NV12, YUYV, UYVY, RGBA8, BGRA8, XRGB8, R8 /* DRM fourcc values */ };

struct ColorInfo { enum Matrix : uint8_t { BT601, BT709, BT2020 } matrix;
                   enum Range  : uint8_t { Limited, Full }        range;
                   uint8_t chroma_siting; };

struct Frame {
    Container   container;
    union {
        struct { int fd[4]; uint64_t off[4]; uint32_t pitch[4]; uint64_t modifier; }  dmabuf;
        AHardwareBuffer*                                                             ahb;
        void*                                                                        iosurface;  // IOSurfaceRef / CVPixelBufferRef
        struct { void* handle; void* resource; }                                     dxgi;
        struct { VkImage img; VkDeviceMemory mem; int32_t layout; uint64_t modifier; } vk;
        struct { uint32_t id; uint32_t target; }                                     gl;
        WGPUTexture                                                                  wgpu;
        void*                                                                        mtl;
        struct { void* plane[4]; uint32_t pitch[4]; }                                host;
    } handle;
    PixelFormat format;  uint8_t planes;  uint32_t width, height;  ColorInfo color;

    void*        manager_ctx;
    void       (*release)(Frame*);
    SyncToken    acquire;
};
```

Factories: `Frame::from_dmabuf(fds, offs, pitches, modifier, fourcc, w, h, color, sync_fd)`, `from_ahardwarebuffer`, `from_iosurface`, `from_dxgi`, `from_vk_image(img, mem, layout, modifier)`, `from_gl_texture(id, target, glsync)`, `from_wgpu_texture`, `from_mtl_texture`, `from_host_planes`.

Containers vs. formats: the container is the sharing primitive of the platform (Linux dma-buf from V4L2 / VA-API / PipeWire; Android `AHardwareBuffer`; Apple `IOSurface`/`CVPixelBuffer`; Windows DXGI handles; same-process API images). anira carries every container and every format; whether a driver can *import* a given (container, format, modifier) is a probed edge (section 7). Same-process API images are importable only by their own API unless the user exports them (`VK_EXT_external_memory_dma_buf`, `eglExportDMABUFImageMESA`); the planner does not export on the user's behalf. Vulkan image layout and Dawn `BeginAccess/EndAccess` state are edge-internal.

Interpretation is a stage, not a property: `stage::FrameToTensor` (section 7) ships kernels for `{NV12, YUYV, UYVY, RGBA8, BGRA8}`; any other format is a user stage with the same `StageDecl { Frame -> Domain::X }`.

Implicit sync is part of the container, not of the token. A dma-buf from V4L2, VA-API or PipeWire carries fences on its reservation object that appear in no `SyncToken`: `DMA_BUF_IOCTL_SYNC` (CPU access) and every driver-side import wait on all of them, so a `Frame` whose `acquire` is `SyncKind::None` may still block on entry. The planner therefore treats a dma-buf container as implicitly synchronised — correct without an explicit token, and never a licence to skip one when the producer has it — and the plan report states the cost, because it is measurable: the prototype's `INFER_DMABUF_NOSYNC=1` exists to measure exactly this bracket. Explicit tokens remain preferred: they say *what* is waited for, where implicit sync waits for everything the buffer ever saw.

---

## 1b. Extensions (capability slots on config structs)

Some of what a config carries is read by the core -- the Machine, the planner, the chunkers, the pump -- and some is read by exactly one stage or one backend adapter. The first kind are fields. The second kind are *extensions*: they arrive with their consumer, leave with it, and a build that lacks the consumer must refuse them rather than ignore them. Every config-time struct a stage or adapter reads ends in one `Extensions ext` member -- `TensorSpec`, `ModelData`, `ModelConfig`, `MachineConfig`, `Hard`, `Async` -- and `JobOptions` carries a borrowed view of the same thing. Never `Tensor` or `Frame`: those are PODs copied through lock-free FIFOs, and an owning pointer on them is a lifetime and an allocation. The struct layouts are then fixed for the life of a major: a new capability is a new registry entry, never a new field.

```cpp
struct ExtBase {
    const char* kind;                  // stable id; doubles as the JSON key
    uint32_t    version = 1;           // of this extension's own layout
    virtual ~ExtBase() = default;
};

class Extensions {                     // owning; one entry per type; frozen at prepare()
public:
    template <class T> void     set(T v);           // replaces an existing T
    template <class T> const T* get() const;        // nullptr if absent
    template <class T> bool     has() const;
    std::span<const ExtBase* const> all() const;    // what prepare() walks
private:
    std::vector<std::unique_ptr<const ExtBase>> m_items;
};

struct ExtInfo {                       // registry row, one per kind, registered by the
    const char* kind;                  // extension's own translation unit
    std::unique_ptr<ExtBase> (*from_json)(const Json&);   // nullptr = code-only
    void (*to_json)(const ExtBase&, Json&);               // plan report, round trip
};
```

An extension is an ordinary struct deriving from `ExtBase`, self-contained by rule: it owns everything it references and points into neither its host struct nor another extension. Every stage and backend adapter reports, per host struct, the kinds it consumes.

Consumed or fail. After the plan is compiled, `prepare()` walks every `ext` on every struct it touched and checks each entry against the union of what the stages actually in the plan consume. An entry no stage consumes fails prepare by name -- `extension 'quant' on tensor 'audio_in' is not consumed by any stage in this build` -- and so does an unknown kind, which the JSON loader does not drop but carries as `ext::Unknown{kind, raw}` precisely so that a typo or a missing backend fails here with the name in the message. This is the one place the design inverts Vulkan's `pNext`, whose contract is to skip what it does not recognise: right for a driver ABI, wrong for a config, where a skipped quantization block means an int8 model runs and produces garbage. Two stages may consume the same entry. After `prepare()` the bags are frozen and read-only; a stage that needs an extension read it at prepare and cached what it needs, and the pump never sees `Extensions`. The `PlanReport` lists, per struct, each extension present and the stage that consumed it, so a sweep log says `quant -> QuantStage(int8)` rather than leaving the reader to assume.

Per-job extensions are borrowed, not owned: `JobOptions::ext` is a `std::span<const ExtBase* const>` valid for the duration of the `submit` call, the stage copies what it needs into its job record, and the consumed-or-fail check runs at `submit` and fails the ticket (`Failed`, with the kind name), not the handler.

`Machine::capabilities()` reports the registered extension kinds beside the probed edges and the enabled backends, so a deployment can ask whether this anira understands `npu` before loading a model that needs it.

v1 ships the carrier, the registry, the walk, `ext::Unknown` -- and one extension: `ext::Entry { std::string name; }` on `ModelData`, the entry point a program is run through (v2's `model_function`; absent means `forward`), consumed by the two adapters that have one -- LibTorch (`get_method`) and ExecuTorch (`load_method`) -- so the rule that two consumers may read one entry is exercised from day one. It is the proof of the mechanism against backends that already have users. A v2 model file carrying a function name loads into the extension through the auto-upgrade path (section 10) with its one-time warning; the same file on a build with neither backend fails prepare by name instead of carrying a field that means nothing there; a v2 file that never set one migrates with an empty bag, which proves the absent path; and the plan report shows `ModelData[1].ext: entry -> LibTorchAdapter`.

Deferred, each arriving with its consumer, all additive: `ext::Quant` on `TensorSpec` (scales, zero points, channel axis; consumed by a (de)quant stage), `ext::Artifacts` on `ModelData` and `ext::ArtifactCache` on `MachineConfig` (precompiled EP/NPU binaries keyed by device; consumed by adapters that compile -- TensorRT, QNN, CoreML), `ext::Npu` on `MachineConfig` and `ext::NpuHard` on `Hard` (plugin directories, performance hint, full-offload-or-reject and performance pinning; consumed by an NPU adapter), `ext::OrtSession` on `ModelData` (graph capture, validation mode, layout preference, intra-op spinning; consumed by the ORT adapter -- the prototype's `INFER_ORT_OPTS`, given a home), `ext::CropAffine` on `JobOptions` (the per-job 2x3 affine of `FrameToTensor`, section 7), and `ext::JobBackend` / `ext::JobModel` on `JobOptions` (per-job backend and variant selection, section 7). Whether something belongs here rather than in a field is decided by the question above and nothing else: who reads it. `max_block_size`, `rate` and `anchor` are read by the core and are fields; the device blocks of `MachineConfig` are read by the Machine's probe and are fields; the entry point of a LibTorch or ExecuTorch program is read by that adapter and is an extension.

---

## 2. TensorSpec (model truth, per I/O slot)

```cpp
inline constexpr int64_t k_dynamic   = -1;   // legal only on a Buffer-role Time axis
inline constexpr int64_t k_unbounded = -1;   // for window_max

enum class AxisTag : uint8_t { Batch, Channel, Time, Height, Width, Feature, Any };

struct Axis { AxisTag tag; int64_t extent; }; // array order = model memory order
// NCHW vs NHWC is just axis order; layout conversion = tag-sequence matching.
// Chunkers find the Time axis by tag.

enum class Role : uint8_t {
    Streamed,   // has a Time axis consumed window-wise           (in + out)
    Buffer,     // whole submitted buffer = one model tensor,
                //   no Time axis (frames, images)                 (in + out)
    Static      // no time semantics: conditioning in,
                //   scalar/embedding out; one value per job
};

struct Ratio { int64_t num, den; };

struct TensorSpec {
    const char*  name;                      // canonical; ModelData maps to engine names
    DType        dtype;                     // the model's true dtype. A quantized model
                                            // takes float I/O only through ext::Quant and
                                            // its (de)quant stage (section 1b, deferred)

    uint8_t      ndim;
    Axis         axes[k_max_rank];

    Role         role;

    // Role::Streamed only, in elements along the Time axis:
    int64_t      window_min = 0;            // model's smallest legal Time extent
    int64_t      window_max = 0;            // largest; k_unbounded = no upper limit
    int64_t      context    = 0;            // left-context retained across inferences
                                            // consumed per inference = window_used - context
                                            // fixed case: window_min == window_max
    Ratio        time_ratio = {0, 0};       // vs. anchor tensor; {0,0} = derive
                                            // this tensor advances num elements per
                                            // den anchor elements

    // Outputs only:
    int64_t      latency = 0;               // model-internal delay along Time (per tensor)

    Extensions   ext;                       // section 1b; v1 registers none for TensorSpec
};
```

Window semantics: incremental arrival accumulates until `window_min`, then runs greedily clamped to `window_max`, retaining `context`. Complete buffers within range run in one shot; above `window_max` the ViewChunker slices, rebalancing the last two chunks into range; below `window_min` a JobOptions policy decides pad vs reject. Hard pins one effective window at prepare (host cadence clamped into range) and measures the budget at exactly that window.

Prepare-time legality: exactly one Time axis for Streamed; window fields iff Streamed; `k_dynamic` only on Buffer Time extent; ratios and window ranges jointly satisfiable across streamed tensors; every `ext` entry consumed by a stage in the plan (section 1b). Streamed tensors may sit on one side only: a generator (Static or Buffer inputs, Streamed outputs) and an analyser (Streamed inputs, Static or Buffer outputs) are first-class, not edge cases -- v2's reference stream is by construction an input, which is the root of the `prepare()` hang and segfault of anira PR #101 (section 11, M0). Under Hard at least one Streamed tensor must exist on either side, because the anchor is the clock; Async admits none (IdentityChunker throughout).

---

## 3. Contract (scheduling regime of one handler)

```cpp
using Duration = std::chrono::nanoseconds;

struct Budget {
    enum class Kind : uint8_t { Measured, Explicit } kind = Kind::Measured;
    Duration value = {};                   // Kind::Explicit only
};

struct WarmUp {
    enum class Mode : uint8_t { None, Fixed, UntilStable } mode = Mode::UntilStable;
    uint32_t iterations = 0;               // Mode::Fixed only
};

enum class MissPolicy : uint8_t { Bypass, HoldLast, Zeros };
enum class LatePolicy : uint8_t { Finish, Drop };
enum class Priority   : uint8_t { Auto, Interactive, Batch };
enum class Delivery   : uint8_t { Polled, Immediate };

struct Hard {
    // Stream geometry: what the host callback delivers
    uint32_t max_block_size;               // largest n passed to process(), in Time-axis
                                           // elements of the anchor tensor
    double   rate;                         // anchor elements per second (48000 audio, 30 fps);
                                           // the anchor is ModelConfig::anchor

    // The guarantee
    Budget     inference_budget = {};      // Measured: derived during warmup
    WarmUp     warmup           = {};      // Mode::None legal only with Kind::Explicit
    MissPolicy on_miss          = MissPolicy::Bypass;  // Bypass requires shape-compatible
                                                       // I/O along the anchored Time axis

    Extensions ext;                        // section 1b; e.g. ext::NpuHard with an NPU adapter
};

struct Async {
    std::optional<Duration> deadline = {}; // absent = offline posture; clock starts at
                                           // submit(); per-submit override incl. absolute
                                           // time points
    LatePolicy on_late  = LatePolicy::Finish;  // Drop: cancel at chunk boundaries,
                                               // enables admission control at dispatch
    Priority   priority = Priority::Auto;      // Auto: Interactive iff deadline, else Batch
    uint32_t   lanes         = 0;              // parallel plan instances; 0 = auto:
                                               // 1 if Stateful, else min(max_instances,
                                               // pool-derived)
    uint32_t   max_in_flight = 0;              // per-lane pipelining; 0 = auto:
                                               // shallow iff deadline, else deep
    Delivery   delivery = Delivery::Polled;    // Immediate: callback on worker thread

    Extensions ext;                            // section 1b; v1 registers none for Async
};

using Contract = std::variant<Hard, Async>;
```

Entry-point coupling is type-level: Hard enables `process`/`push_data`/`pop_data` and disables `submit`; Async the reverse. Soft real-time and offline are documentation vocabulary over Async (with / without deadline), not presets.

Prepare validation. Hard: warmup, budget vs block cadence, no waits or allocation reachable from `process()`, plus whatever an adapter adds for the extensions it consumes (an NPU adapter's full-offload-or-reject and performance pinning arrive with `ext::NpuHard`, section 1b); with multiple enabled backends all of this holds per plan, and the reported latency covers the slowest enabled plan (section 7, plan sets). Async: deadline feasibility vs measured time, `lanes = 1` for stateful models, no warmup required without deadline.

Deadline effects: prepare posture (latency vs throughput defaults), dispatch ordering (EDF ahead of batch) and early rejection under Drop, chunk-boundary cancellation, honest ticket reporting (met / late / dropped). It is advisory information, not a promise; only Hard's budget changes what code may exist.

---

## 4. MachineConfig (machine and process resources)

```cpp
enum class Ownership  : uint8_t { Borrowed, Owned };  // Borrowed = user's handles
enum class ExecPolicy : uint8_t { Worker, UserDriven };
enum class GlThreads  : uint8_t { CallerThread, SharedContext };

struct MachineConfig {
    // Inference thread pool (process-global; first Machine wins, as today)
    uint32_t     num_threads   = 0;                   // 0 = hardware_concurrency - 1
    WaitStrategy wait_strategy = WaitStrategy::SpinBackoff;
    LogLevel     log_level     = LogLevel::Warning;   // most verbose request wins

    struct Cuda   { Ownership own = Ownership::Owned; int32_t device = 0;
                    size_t pinned_pool_limit = 0; };          // 0 = planner-sized; cap on
                                                              // cudaHostAlloc staging
                    // nothing to hand over: the primary context is process-wide, so a
                    // pointer, stream or event on it is anira's as much as the user's
    struct Gl     { Ownership own = Ownership::Borrowed;      // GL is always borrowed
                    void* display; void* context;             // EGL (or GLX equivalents)
                    void* gbm = nullptr;                      // gbm_device*: lets allocate_*
                                                              // back GL storage with a dma-buf
                    GlThreads threads = GlThreads::CallerThread; };
                    // CallerThread (v1): anira touches GL only inside allocate_*, submit and
                    // bind_output, on the calling thread, where the user's context is
                    // current; a call from another thread is a contract error.
                    // SharedContext (additive): the user passes a second context of the
                    // same share group and anira's worker makes it current.
    struct Vulkan { Ownership own = Ownership::Owned;
                    void* instance; void* physical; void* device;
                    uint32_t queue_family; uint32_t queue_index; };
                    // thread-agnostic: anira serializes its own submissions on the queue
    struct Metal  { void* device; };                          // nullptr = default device
    struct D3D12  { Ownership own; void* device; };
    struct WebGpu { Ownership own = Ownership::Owned;
                    void* instance; void* device; void* queue;   // WGPUInstance / WGPUDevice / WGPUQueue
                    ExecPolicy exec = ExecPolicy::Worker; };     // someone must pump ProcessEvents / WaitAny

    std::optional<Cuda>   cuda;      // nullopt = domain unavailable, edges pruned
    std::optional<Gl>     gl;        // presence is the user's declaration, no implicit probing
    std::optional<Vulkan> vulkan;
    std::optional<Metal>  metal;
    std::optional<D3D12>  d3d12;
    std::optional<WebGpu> webgpu;

    Extensions ext;                  // section 1b: ext::Npu and ext::ArtifactCache arrive
                                     // with their adapters. The device blocks above stay
                                     // fields: the Machine's probe reads them, not an adapter
};
```

WebGPU ownership: one WebGPU implementation per process, and anira owns it. The Machine links its own Dawn (`libwebgpu_dawn.so`, monolithic shared build from a pinned Dawn revision) and creates or borrows the `WGPUInstance/Device/Queue` on it. Engines never bring their own Dawn; they receive anira's through a proc table: a `WGPUDevice` is a C++ object of one build, not an ABI-stable handle, and `DawnProcTable` is a struct whose layout is fixed by the Dawn revision, so engine and Machine must be built against the same Dawn source tree. Per engine:

- ORT: built with `onnxruntime_USE_EXTERNAL_DAWN=ON` and `onnxruntime_CUSTOM_DAWN_SRC_PATH=<anira's Dawn tree>` (ORT then links only `dawn_proc` thunks, hidden behind its version script); the adapter passes `dawnProcTable = dawn::native::GetProcs()` of anira's library plus `webgpuInstance`/`webgpuDevice` with `deviceId >= 1` when creating the session. The proc table is installed once per process (ORT `call_once`), which matches one Machine per process.
- An engine that statically embeds Dawn (ORT's default build, LiteRT's prebuilt WebGPU accelerator) cannot share the device and is treated as a `{Host}` consumer.
- The host app's renderer, if it uses WebGPU, links anira's Dawn and borrows the device (`Ownership::Borrowed`).

Borrowed devices are validated at Machine construction for the features the edges need (`SharedTextureMemoryDmaBuf`, `SharedFenceSyncFD`, `DawnMultiPlanarFormats`, `HostMappedPointer`).

Borrowing differs per API, because the APIs do. WebGPU: anira owns the implementation (above). Vulkan: device and queue are usable from any thread under external synchronization; hand them over once. GL: a context is current on exactly one thread, so `GlThreads` above is the whole story, and v1 is `CallerThread`. CUDA: the primary context is shared by every library in the process; anira retains it (`cudaSetDevice` / `cuDevicePrimaryCtxRetain`) and asserts at construction that user pointers belong to it (`cudaPointerGetAttributes` fails with `cudaErrorIncompatibleDriverContext` for a pointer from a context created with `cuCtxCreate`). Toolkit versions need not match between anira and an engine: every CUDA runtime funnels into the driver's one `libcuda.so`, and only the driver's minimum for the CUDA major family matters -- the opposite of the Dawn revision lock.

Why external Dawn, and what it costs. ORT's external-Dawn mode is the only one in which ORT is a *consumer* of a device rather than its provider; with ORT owning Dawn, no second WebGPU engine and no renderer could ever share the device. The price is a revision lock: `DawnProcTable` is a plain struct of ~280 function pointers whose order changes between Dawn revisions, so Dawn, ORT and every other WebGPU engine form one versioned triple per anira release (Dawn revision pinned to ORT's `deps.txt`; an ORT upgrade means re-pinning and rebuilding Dawn, rebuilding ORT from that tree, re-validating the other engines). The proc table is process-global (ORT `call_once`), consistent with one Machine per process; it also means anira cannot coexist with another library installing a different Dawn in the same process, which is true of any arrangement but here fails loudly. Build complexity is two builds instead of one and no `find_package` path in ORT (`onnxruntime_CUSTOM_DAWN_SRC_PATH` is what ORT's own CI uses for custom Dawn). Debuggability improves: one Dawn, one validation layer, one toggle set, one device-lost callback.

Version assertion (required): the Machine compares the Dawn it loaded against the revision anira was built with -- `kDawnVersion` from `dawn/dawn_version.h` baked at build time versus the version reported by the loaded library -- and refuses construction on mismatch. This turns the one real failure mode of this arrangement, a silently mismatched proc table, into an immediate, readable error. The same check runs for every engine that consumes the proc table.

Probing: domain and edge availability are *driver* facts, not platform facts. At construction the Machine enumerates Vulkan device extensions, `wgpuAdapterHasFeature`, EGL/GL extension strings, CUDA attributes and each engine's buffer requirements, and fills the edge registry (section 7) from the answers. Every cross-API row also requires that the two devices are the same physical GPU -- an exported allocation is memory on one adapter, and a second API on another adapter cannot see it -- so the Machine compares device identity across the enabled blocks at construction (Vulkan `VkPhysicalDeviceIDProperties::deviceUUID`, CUDA `cudaDeviceProp::uuid`, Dawn's adapter info, the D3D12 adapter LUID on Windows) and marks a cross-API row unavailable, with the mismatch in the report, when the identities differ; a machine with two GPUs therefore needs its device blocks to name the same one, and the Vulkan block that exists only to mint exportable memory for the `OpaqueFd` rows (section 7) must land on the GPU Dawn and CUDA use. For the GL rows the check is CUDA's own: `cudaGLGetDevices` on the thread where the borrowed context is current (`GlThreads::CallerThread`, so inside Machine construction) names the CUDA device backing that context, and the `GlBuffer -> CUDA` registration row is enabled only when it names the CUDA block's device -- no device (the context is on an iGPU), a different one, or a failure on the right GPU because the GL driver is not the vendor's (`cudaGraphicsGLRegisterBuffer` lives in NVIDIA's GL, not in Mesa) each disables the row, not the block, and the plan falls back to `glGetBufferSubData` with the reason in the report. `GL_EXT_memory_object`'s `GL_DEVICE_UUID_EXT` would give the same identity driver-neutrally, but `cudaGLGetDevices` proves the interop path exists and not merely that the GPUs match. The dma-buf rows (`GL -> WebGPU`, `Vulkan -> WebGPU`) can in principle cross GPUs, since a dma-buf is a kernel object; that is the importing driver mapping foreign memory, a copy in disguise behind a modifier the other device may not understand, so same-GPU is their precondition too and the probe is the import succeeding on a test allocation. Measured example: a UMA device (Apple M1 under Mesa Honeykrisp) lacks `VK_EXT_external_memory_host`, so host-pointer-import edges (persistent map, Dawn `HostMappedPointer`) are absent although the memory is unified; the software device on the same machine has them. "UMA" therefore never appears as a planner condition; only probed edges do. `prepare()` reports the probed registry with the plan.

### Probing: the three rungs

A row enters the registry only after it has passed three rungs, and the prototype's history is the argument for the third. *Static*: the extension and feature bits above -- cheap, necessary, never sufficient. *Identity*: the two APIs sit on the same physical GPU (above). *Functional*: the Machine runs the row once, end to end, and checks the bytes. Every silent failure the prototype met passed a feature check: the dma-buf that imported without complaint and was written at a driver-rounded pitch (`max_abs_err` 175 on every dma-heap output), the `imageStore` into an EGLImage-backed texture that satisfied `glReadPixels` and left the dma-buf zeroed behind a tiled shadow, the tensor whose `OrtMemoryInfo` device id was off by one and was copied into an EP-owned buffer under a row that said `ZeroCopy`, the captured graph replaying the buffers of run 0. None of those is an extension bit, and each was found by a round trip with a pattern and a compare. The functional rung is that discovery made routine and moved onto the user's machine, because the matrix of section 7 measured *this* driver and the user's may differ in exactly these ways.

The functional rung, per row: allocate a few KB the row's own way -- the exact recipe production will use, exportable memory with the linear image bound, a rendered-into `gbm_bo`, a `WGPUBuffer` on the Machine's Dawn; write a known pattern from the *producer* side through the producer's API; execute the edge as the plan would, import, pass, registration, fence hand-off included; read back from the *consumer* side through the consumer's API and compare bit-exact. For engine rows the pattern is an identity model run twice with different inputs whose outputs must differ -- the `hello_inference` stale check, which is what proves the engine read the caller's buffer rather than a private copy, and what decides whether graph capture is permitted on this driver. The result is `{available, class, reason, rung}`, never a bool: `unavailable: import ok, readback mismatch at byte 72 (pitch)` is what the capability report and the plan report show, the runtime twin of "a foreign handle that works but slower is data, never a log line". It measures correctness and cost *class*, not time; time is the Hard warmup's and the benchmark sweep's.

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

Cost and caching. Functional probes are milliseconds each and there are dozens, and a plugin constructs its Machine inside `prepareToPlay`. Results are therefore cached on the box, keyed by `(anira version, driver versions, device UUIDs, enabled blocks)`, next to where an artifact cache would live: a matching key loads the registry in microseconds, a mismatch re-probes. `Machine::probe(force)` re-runs on demand, and a benchmark run always forces, since a row cached under last month's driver measures nothing.

Thread and device contracts. GL rungs need the borrowed context current and run on the constructing thread under `CallerThread`, as every GL touch does. Functional probes submit once to the Vulkan, Dawn and CUDA queues at construction; on a `Borrowed` device that is a submission the application did not make, which is part of what `Ownership::Borrowed` means and is stated here so that it is not a surprise.

How a thread waits for a GPU is a machine-level decision, not a detail: `WaitStrategy` governs GPU completion waits as much as queue waits. A thread that blocks in a fence wait for the length of an inference leaves its core idle, and the CPU frequency governor clocks it down; the *next* thing that runs there — the pre-processing of the following frame, the host callback — then executes slow. Measured on the M1 under `schedutil`: identical producer code took 46 us when it followed a CPU inference and 208 us when it followed a 10 ms GPU fence wait, edges 113 us versus 28 us, and even the engine's own submission overhead grew by ~0.7 ms; pinning the governor to `performance` collapsed every column onto the busy-core value. Consequences: `SpinBackoff` is the correct default for deadline-carrying contracts (Hard always, Async with a deadline), a blocking wait is for offline postures, and the plan report names the wait strategy used per edge. Benchmarks record the governor (see section 7) -- otherwise the two regimes differ by 3-5x on everything except the inference itself and the numbers are not comparable.

The Machine also owns one runtime environment per engine (`Ort::Env`, LibTorch/c10 globals, the TFLite interpreter's shared state, the LiteRT environment, the ExecuTorch runtime and its XNNPACK thread pool; a later engine brings its own), created lazily for enabled backends and shared by every model, plan and handler in the process; v2 duplicated these per handler. Adapter-level session sharing (ORT shared allocators and prepacked-weight sharing) rides on these shared environments and is where the memory win for multi-model setups lands.

Logging is `thl::Logger` (tanh-lib's core component, adopted in v2.3.0 -- section 11): `log_level` maps onto its runtime level and is still forwarded to the engines; anira never calls its `set_config`, because the process's logger belongs to the application, and the callback sink is how a plugin host receives anira's messages. Nothing reachable from `process()`, `push_data()`, `pop_data()` or `submit()` calls it -- it allocates and locks -- and RTSan enforces that without a suppression. Real-time paths log through a lock-free front of fixed-size records (level, group, static message, a few integer arguments) that the Machine's inference threads drain into the logger off the audio thread.

---

## 5. ModelConfig (model semantics)

```cpp
enum class ModelState : uint8_t { Stateless, Stateful };  // Stateful = session-exclusive:
                                                          // forces lanes = 1

struct ModelData {                       // one entry per backend able to run this model
    Source           source;             // path | owned blob
    InferenceBackend backend;
    NameMap          tensor_names;       // canonical spec name -> engine tensor name
    Extensions       ext;                // section 1b. v1: ext::Entry{name}, the entry
                                         // point (absent = "forward"; LibTorch, ExecuTorch).
                                         // Later: ext::OrtSession, ext::Artifacts
};

struct ModelConfig {
    std::vector<ModelData>  models;                       // >= 1
    InferenceBackend        default_backend = InferenceBackend::None;  // None = models[0]

    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;

    ModelState              state         = ModelState::Stateless;
    uint32_t                max_instances = 1;            // memory ceiling; planner
                                                          // allocates lanes/pool within
    size_t                  anchor = k_first_streamed;    // the clock for time_ratio and Hard
                                                          // geometry: first Streamed input, else
                                                          // first Streamed output (generator)

    Extensions              ext;                          // section 1b; v1 registers none
};
```

A plain aggregate again: the private quantization arena and its span-rebinding rule left with `ext::Quant`, which owns its own vectors (section 1b). Quantization is not on the config in v1 at all: a quantized model's true `dtype` is in its spec, and a float producer meeting an int8 spec fails prepare with the extension named, never a silent conversion.

One `ModelConfig` describes one logical model. A variant set references several configs inside one handler; running models in sequence is several handlers on one Machine, composed through tickets (section 7, Multi-model support).

One-sided streaming. The anchor is whichever Streamed tensor is the clock, input or output, and everything time-related is stated in its elements. In a generator (no Streamed input) `Hard::max_block_size` and `rate` are in elements of the Streamed *output*, `process()` is a pull -- Static inputs arrive through `set_input`, `push_data` has nothing to push -- and `get_latency()` counts from the first `process()`. In an analyser (no Streamed output) `process()` pushes, Static outputs leave through `get_output` (a ticket under Async), and `get_latency()` covers Streamed outputs only: a Static output carries no stream latency and never enters the latency vector, which is the index misalignment v2's fix had to patch around. Both are the same RingChunker with one side empty (section 7).

Removed, with destinations: `ModelData::model_function` -> `ext::Entry{name}` (section 1b); `max_inference_time` -> `Hard::inference_budget`; `warm_up` -> `Hard::warmup`; `HostConfig` block size / rate -> `Hard` geometry; `TensorShapeList` + `ProcessingSpec` -> `TensorSpec` axes and streaming fields.

---

## 6. Handler API surface

```cpp
anira::Machine          machine(machine_config);
anira::InferenceHandler handler(machine, pipeline);          // pipeline: pre/post stages around
                                                         // exactly ONE Inference stage
handler.prepare(contract, PlanPolicy{});                 // compiles and validates the plan;
                                                         // returns the PlanReport (section 7)

// ---- Hard entries (unchanged from v2, copy semantics: host owns callback memory) ----
handler.process(float** data, size_t n);                 // fused push + pop
handler.push_data(...);  handler.pop_data(...);          // decoupled, stream-position addressed

// ---- Async entries (borrow semantics, token-defined lifetimes) ----
anira::Tensor in  = handler.allocate_input (slot, Domain::VulkanBuffer);  // anira allocates on
anira::Tensor out = handler.allocate_output(slot, Domain::VulkanBuffer);  // the USER's API with the
                                                         // fast-path recipe for every enabled
                                                         // candidate; the user writes/reads it
                                                         // natively and never sees the recipe
handler.bind_output(slot, tensor);                       // user-owned destination, any handle;
                                                         // tensor.acquire = writable-when
handler.request_output(slot, Domain, DType);             // deferred past v1: anira-owned pool,
                                                         // release recycles

anira::Ticket t = handler.submit(std::span<const Tensor> inputs,
                                 const JobOptions& = {});
t.poll();                        // completion state, non-blocking
t.wait(); t.wait_for(dur);       // blocking
t.status();                      // Pending | Met | Late | Dropped | Failed
t.input_released(i);             // SyncToken: producer may reuse/unmap input i
t.output_ready(i);               // SyncToken: consumer may read bound output i
handler.poll();                  // drains Polled-delivery completions on calling thread

// ---- Runtime backend selection (both contracts) ----
handler.set_inference_backend(InferenceBackend);
// Atomic selection among the precompiled plan set (section 7). Takes effect at a safe
// boundary: next chunk under Hard, next job under Async. Never triggers planning.
handler.set_model(size_t variant);
// Same semantics for model variant sets: atomic selection among precompiled plans.
// Switching a Stateful variant clears the stream state (defined, logged).

struct JobOptions {
    std::optional<TimePoint> deadline;     // absolute override (presentation timestamp)
    std::vector<long> head_trim;           // -1 = trim per-output latency (input-aligned)
    bool tail_flush = true;                // ViewChunker reassembly semantics
    PadPolicy below_min = PadPolicy::Reject;
    JobCallback on_complete;               // Delivery::Immediate only
    std::span<const ExtBase* const> ext = {};   // section 1b: borrowed for the call.
                                                // ext::CropAffine, ext::JobBackend,
                                                // ext::JobModel when their consumers arrive
};

enum class EdgeCost : uint8_t { Permissive, Strict };
struct PlanPolicy { EdgeCost edge_cost = EdgeCost::Permissive; };  // plan validation, not
                                                                  // scheduling: beside the
                                                                  // Contract, not in it
```

Any handle is accepted at `submit` and `bind_output`; the plan decides what it costs. `allocate_input`/`allocate_output` exist because the fast rows of section 7 are decided at allocation time and anira can retrofit nothing: they hand back a handle in the user's own API -- a `VkBuffer` the user's shader writes as an SSBO, a GL buffer object, or for a WebGPU candidate a dma-buf-backed renderbuffer the user renders into -- allocated so that every enabled candidate gets its best row. The `PlanReport` states per slot the edge taken, its class, the class an `allocate_*` handle would have gotten, and why they differ. A foreign handle that works but slower is data there, never a log line; one that cannot work at all (a same-process image never exported, a `WGPUBuffer` of another device) fails `prepare()` with the recipe in the message. `EdgeCost::Strict` makes the first case fail too -- the harness rule that an unavailable zero-copy row is a bug, not a fallback, promoted to a library contract; development runs strict, production ships permissive.

Push/pop vs submit/poll: the same exchange addressed by stream position (quantitative, continuous, miss policy fabricates) versus ticket identity (transactional, out-of-order across lanes, never fabricates). Below the API both feed the same pump.

---

## 7. Planner, stages, chunkers

Stages declare a contract, the tensor stays passive: data flows, stages act, the planner decides where and when.

```cpp
struct StageDecl { Domain in; Domain out; ExecContext where; };
// e.g. GL compute pre-processor: { GL, GL, needs-GL-context }
// backend adapter (terminal):    { accepts {Cuda, Host, ...} per capability report }

anira::Pipeline pipe {
    anira::stage::GlCompute(mel_shader,  Domain::GlBuffer, Domain::GlBuffer),
    anira::stage::Inference(cfg, {Backend::OrtWebGpu, Backend::OrtCuda, Backend::TfLiteCpu}),
    anira::stage::Custom(peak_pick,      Domain::Host,     Domain::Host)
};
```

At `prepare()` the planner intersects producer domain, stage chain, backend capability report and terminal output domain against a sparse conversion-edge registry (GL<->CUDA register/map cached, GL->host readback or persistent map where the driver has host-pointer import, Vulkan external memory, dma-buf image import into Vulkan/EGL/Dawn, registration-cached NPU edges, pinned-staging fallback always available with fused dtype/layout conversion). The registry is filled by the Machine's driver probe (section 4), never by platform assumptions. Paths are composed, not looked up: between a producer domain and a consumer the planner searches the registry for every chain of rows, allocating any intermediate from its own pool with the fast-path recipe (a crossing currency of section 1 -- `OpaqueFd`, `DmaBuf`, an NT handle -- is the usual intermediate), and ranks the candidates by their worst edge class, so a `WgpuBuffer` bound for the CUDA EP goes `WgpuBuffer -> DmaBuf -> OpaqueFd -> CUDA EP` at `DeviceCopy` (two pool intermediates, because on NVIDIA one allocation cannot export both ways -- measured, below) when the Machine has a Vulkan device to mint them and `WgpuBuffer -> Host -> Cuda` at `HostCopy` when it has not; the plan report shows the chain either way, and the user never names the intermediate. Unreachable domains fail at prepare with a capability error. The plan is fixed before processing; `process`/`submit` replay it.

### Edge classes and the measured Linux GPU registry

Every edge carries a cost class, stated in the plan report so tests can assert it per edge: `ZeroCopy` (handle hand-over or memory import, no data movement), `DeviceCopy` (one GPU pass or copy-engine operation), `HostCopy` (readback and/or upload through staging). The class in the registry is the *functional* rung's result (section 4, the three rungs), never the static rung's: a row whose feature bits are present but whose round trip failed is unavailable, with the reason attached.

Two crossing mechanisms, and everything about a row follows from which one it uses. *Reach-in*: same-process, nothing travels, no allocation-time decision -- a CUDA pointer on the primary context, a `WGPUBuffer` on the shared Dawn device, a GL buffer object registered with `cudaGraphicsGLRegisterBuffer`; any user handle is first-class. *Export*: the handle leaves its API as the platform's crossing currency -- a dma-buf fd or an opaque fd on Linux, an NT shared handle on Windows, an `AHardwareBuffer` on Android, an `IOSurface` on Apple -- decided at allocation time and never retrofittable, which is where `allocate_*` earns its place. The byte-image encoding below is the general answer wherever the importer takes textures and not buffers, which is Dawn's Vulkan *and* Metal backends both; only D3D12 is expected to escape it. A domain enters the enabled set with the first engine that reads it natively on shared hardware, or as a producer API anira allocates for; never for a producer's convenience alone. The distinction is what makes the registry portable while its rows are not: every platform has both mechanisms, and which one a given pair uses is a probed fact per driver. The rows below are Linux; the other platforms need the same matrix run before their rows are claimed (section 1, measurement status), and two are expected to differ structurally -- D3D12's `SharedBufferMemory` may make the Windows buffer import into WebGPU a true `ZeroCopy`, and Apple's image-typed currency may or may not need the byte image. Consumers measured as `{Host}` today, adding no domain: ExecuTorch's Vulkan backend (`execute()` copies every input from host `EValue`s through a staging buffer and its `Context` creates its own `VkDevice`; zero-copy I/O is pytorch/executorch#13382, open), ORT's CoreML EP, LiteRT's prebuilt WebGPU accelerator. Native CoreML/ANE reads `IOSurface`, Apple's crossing currency; an `IOSurface` buffer arm joins with the first engine that reads it, once measured.

What the user hands in, per engine (Linux, v1):

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

Reference rows, measured on Linux (Mesa Honeykrisp, Dawn from ORT 1.29, LiteRT 2.2 prebuilts); other drivers produce other rows, which is the point of probing:

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

The output rows are the mirror image of the input rows and were measured the same way (prototype: all 50 (source, EP, destination) cells bit-exact on the synthetic model, error 1e-4 on the palm detector; the two `DeviceCopy` bodies are one import used in one direction each, with the fence that gates Dawn's access being the producer's `ready` on the way in and the consumer's `released` on the way out). Two facts the mirror exposed. The byte image's rows must be 64-byte aligned: a linear dma-buf packed at 18-float rows imported without complaint and was written at the pitch the driver rounded to, so the consumer read garbage (`max_abs_err` 175 on every dma-heap output) -- the image is the edge's encoding, not the tensor's shape, so the edge picks aligned rows (an exact factorisation of the element count, or a padded tail no pass reads) and the floats stay packed. The image is computed from the element count alone, so nothing about it lives on the Tensor; the memory must be packed at that pitch (a driver that refuses the aligned pitch makes the row unavailable -- padded rows are image rows, not tensor rows, and no `strides` can say so), and it can be larger than `n * 4`: `allocate_*` sizes for it, and a user-exported buffer must be at least `Machine::byte_image_bytes(n)`. And the headline cell `VulkanBuffer -> WebGPU EP -> VulkanBuffer` -- a camera-fed model rendered by Vulkan, the stage-five shape -- costs `DeviceCopy` twice, one texture round trip each way; that number on paper is the case for the upstream contribution below.

Exportability is a producer-side precondition, not an edge capability. The two dma-buf export rows above exist only for memory that was *allocated* exportable, and no edge can retrofit that: a `VkBuffer` on ordinary `VkDeviceMemory` has no fd to hand out, and the allocation must carry `VkExportMemoryAllocateInfo` plus — since Dawn imports textures — a linear DRM-modifier `VkImage` bound to the same memory, with the row pitch the driver accepts, before the tensor is ever written. The prototype allocates image and optional buffer alias together and writes through the alias, so the generator is an ordinary SSBO shader that never learns an image exists; where the driver refuses the alias (dedicated allocation, disjoint memory types) the writer falls back to `imageStore` into the same memory. Consequences for v3: `from_vulkan(buf, mem, off, timeline)` on user memory reaches WebGPU only through `HostCopy` and the plan report says so; `allocate_input`/`allocate_output` (section 6) are how a Vulkan user gets the row without learning the recipe -- anira allocates on the user's device, exportable, image tag bound, buffer alias, aligned rows, image-sized, and the user's shader writes an ordinary SSBO. anira's own pool tensors are allocated the same way.

GL has no buffer export at all, so `Domain::GlBuffer -> WebGPU` does not exist as written: `glGetBufferSubData`/`glMapBufferRange` is the only way out of a GL buffer object, i.e. `HostCopy`. A GL producer that wants the DeviceCopy row must allocate its storage as a `gbm_bo` and import it back into GL (EGLImage + renderbuffer + FBO) — which makes it a `Frame{Container::DmaBuf}` with a GL view, not a `Domain::GlBuffer`. Two traps, both measured: on Mesa/agx an `imageStore` into an EGLImage-backed *texture* passes `glReadPixels` and leaves the dma-buf zeroed (the driver keeps a tiled shadow it never flushes back — Apple GPUs do not write linear images), so the producer must *render* into a renderbuffer; and the modifier GBM chose travels with the fd and must reach the importer, since a "linear" assumption is wrong on exactly those drivers. `GlBuffer -> CUDA` is unaffected: `cudaGraphicsGLRegisterBuffer` is same-process registration, not export.

Upgrade path noted, not assumed: a `SharedBufferMemory` dma-buf importer for Dawn's Vulkan backend would turn the `VulkanBuffer -> WebGPU` row into ZeroCopy and remove the byte-image encoding entirely; tracked as an upstream contribution.

Two-reader ownership: `input_released` signals when *all* consumers in the plan are done with an input (a camera Frame read by both the app's presenter and the inference edge); the planner composes the fences. Across handlers the composition is the user's: two handlers reading one buffer (the composed-models pattern below -- a detector and a landmark model sampling the same camera frame) each signal their own `input_released`, and the producer's reuse condition is their conjunction. The prototype ANDs the tracker's hold into the capture stream's per-buffer fence and `cam_stream` never learns that a second reader exists; that is the seam, not a new API.

### Stage catalogue: FrameToTensor

`stage::FrameToTensor(PixelFormat, target_size, Crop | Letterbox, normalization, layout)` with built-in kernels for `{NV12, YUYV, UYVY, RGBA8, BGRA8}` on each consumer API `{WebGPU (WGSL), Vulkan (SPIR-V), GL (GLSL), CUDA}`; multi-planar sampling is per API (Dawn plane-aspect views, Vulkan YCbCr sampler, GL `samplerExternalOES`). Any other format is a user stage with the same declaration. The stage output is a Tensor in the consumer's buffer domain; everything downstream is buffers.

Deferred past v1 (section 1a). When it ships, its kernels run where anira already has a device -- WGSL on the Machine's Dawn, C on the host -- so the stage adds no domain; a SPIR-V or CUDA kernel arrives with an engine that reads that domain. A same-process image the user never exported stays theirs to convert.

Two things the prototype's two-model pipeline forces on the declaration before it ships. **The crop is per job, not per stage.** The palm detector wants the whole frame letterboxed or centre-cropped into 192x192; the landmark model wants a rotated, expanded square around the previous result in 224x224; and both are one kernel -- walk the destination, map through a 2x3 affine, gather the source -- handed different six floats. `Crop | Letterbox` are therefore presets that compute that affine from the frame and target sizes, and the affine itself is a job extension, `ext::CropAffine` on the `JobOptions` of `submit_frame` (section 1b), consumed by the stage and rejected at submit on a handler without one; without it the second handler of every two-model vision pipeline cannot use the built-in stage. The direction is destination -> source, which is also the direction that maps a model's output back onto the frame, so decoding needs no inverse. **`border_mode` is per model, not per stage type:** MediaPipe's palm graph fills outside the frame with zero (the letterbox bars *are* black) and its landmark graph replicates the edge, and using zero for the landmark crop collapses presence exactly when the hand comes close enough for its ROI to run off the frame -- a flicker that arrives with proximity and nothing else. Both measured: the C and WGSL kernels handed the same affine and the same YUV coefficients agree to max |d| ~2e-5, which is what makes a downstream disagreement between two execution providers attributable to the model rather than the input.

### Plan sets and runtime backend selection

The Inference stage declares a candidate set (above: `{OrtWebGpu, OrtCuda, TfLiteCpu}`), and `prepare()` compiles **one plan per candidate**: each with its own conversion chain (GL->dma-buf->Dawn for the WebGPU EP, GL->CUDA registration for the CUDA EP, GL->host readback for CPU), its own registrations and staging pools, all validated and preallocated upfront. `set_inference_backend` therefore means *selection, not reconfiguration*: an atomic switch among precompiled plans, effective at the next chunk (Hard) or the next job (Async). No planning ever happens at runtime; determinism survives because only selection does. The input feeds every plan whose engine its domain reaches -- which is what keeping graphics handles representable until the adapter boundary was for -- but not necessarily every plan at the same class, and on NVIDIA not always at all: a Vulkan allocation exports as dma-buf *or* opaque fd, and a GL producer reaches CUDA with a buffer object but WebGPU only with a rendered-into dma-buf. `allocate_*` therefore resolves against the whole enabled candidate set: one object that serves every candidate when such an object exists, otherwise the one serving the most, with the degraded candidates named in the plan report -- the price of a wide candidate set, visible at prepare like the Hard budget of its slowest member. A user who needs both fast rows submits one handle per plan. With model variant sets, candidates generalize to (variant, backend) pairs; see Multi-model support below.

Costs, stated explicitly at prepare: every enabled plan's resources stay resident (v2 already loads all backends at construction; v3 extends this to staging and registration state). Under Hard, worst-case honesty applies across the set: the budget is measured per plan during warmup, the single latency reported to the host covers the slowest enabled plan, and all Hard validations (no-wait reachability, and whatever an adapter adds for the extensions it consumes) must pass for every candidate. A Hard handler that wants live switching buys headroom for its worst candidate; shrink the candidate set to shrink the price.

Async additionally admits per-job selection as a job extension (`ext::JobBackend` on `JobOptions::ext`, section 1b; default = handler-level choice): live A/B against identical frames, automatic fallback when a device saturates, and closed-loop adaptation driven by ticket telemetry (miss rate climbs under thermal throttling, the app shifts inference to the NPU mid-session, the render loop never notices). Handler-level switching at safe boundaries is the v1 commitment; the per-job extension waits for a demonstrated need (reversibility rule).

Benchmarking follows from declarativeness: the contract JSON sweeps scheduling policy, the model JSON (or a `default_backend` override) sweeps engines, one unmodified binary loops over the grid, and each run emits met/late/dropped rates, per-window inference times from warmup, and the compiled plan (which edges, where staging landed) so results explain themselves. In-process backend cycling gives perfect code-path comparability but couples runs through GPU clocks and thermal state; randomize order or restart per run (the file sweep does this for free) for publishable numbers. Every report records the CPU frequency governor and the wait strategy alongside the plan: a blocking GPU wait under `schedutil` inflates everything the CPU does around the inference by 3-5x (section 4), so a run without those two fields cannot be compared with another. Reports also state, per job, whether the result was verified fresh -- a stale-output check (section 7, completion) is what separates a measurement from an artefact.

### Multi-model support (new in v3, missing since v2)

Two meanings of "multiple models", handled at two different levels, plus the shared infrastructure both use. A handler owns exactly one `Inference` stage; that stage may hold several *variants*, and running several models in *sequence* is several handlers.

**Model variants (alternatives) -- inside one handler.** `stage::Inference` accepts a variant list, several configs on the same or different backends:

```cpp
anira::stage::Inference({cfg_small, cfg_large}, {Backend::OrtCuda, Backend::TfLiteCpu})
```

The plan-set machinery generalizes unchanged: candidates become (variant, backend) pairs, `prepare()` compiles one plan per pair, and `set_model(variant)` is the same atomic safe-boundary selection as `set_inference_backend`, selection, never planning. Validation at prepare: variants must agree on the external I/O contract (axes, roles, dtypes after de/quantization); windows and latencies may differ per variant since each owns a plan, but under Hard the worst case across all enabled (variant, backend) pairs sets the budget and the reported latency. Switching a Stateful variant clears the stream state (defined, logged). Per-job variant choice arrives as a job extension, `ext::JobModel` beside `ext::JobBackend` (section 1b), not as a reserved field.

**Composed models (sequencing) -- across handlers, through tickets.** Running model B on model A's output is two handlers on one `Machine`, sequenced by the user's loop, and it costs nothing an in-plan chain would have saved. A's `allocate_output(slot, Domain::X)` is a handle on the Machine's shared device, so it is a legal `submit` input on B (section 6: any handle is accepted), and A's `output_ready(i)` becomes that tensor's `acquire`. The edge B takes is the same registry row as any other -- `WgpuBuffer -> WebGPU EP` ZeroCopy, `Cuda -> CUDA EP` ZeroCopy, a cross-engine pair `HostCopy` through pinned staging -- and B's plan report states it. On one queue the token is `QueueOrdered` and B waits for nothing; the host drains once, at the end of the sequence. Under Hard, `get_latency()` values add. Under Async each handler carries its own contract, which is itself a reason to compose this way: a detector with a deadline feeding a landmark stage that has none of its own. Whatever runs between the two models -- a decode, a non-maximum suppression, a crop affine derived from A's output -- is the user's code, either on the host after `wait_for`/`poll` (a priced `HostCopy`, chosen, never imposed) or as the user's own pass on the Machine's queue between the two submits (queue-ordered, no token needed).

Two things this gives that a chain inside one plan cannot. **Control flow:** whether B runs at all, how many times, and against which of A's results is application logic. MediaPipe's hand graph gates its detector on the previous cycle's landmark count and runs the landmark model once per associated rect; no linear stage list expresses that, and a stage graph with conditionals and loops is a graph runtime, not an inference library. **A visible intermediate:** what settles a disagreement between two models is dumping what sat between them -- every state-machine bug in the prototype's tracker was found that way, and an intermediate that is a plan-internal buffer is exactly the thing that cannot be printed.

Chained `Inference` stages inside one Pipeline -- A's output as a plan-internal buffer feeding B -- are deferred past v1, on the additive list. Their zero-copy benefit is already free above. Two same-engine graphs with fixed dataflow are better merged offline, where the engine fuses across the boundary and a runtime chain never can. The one thing only an in-plan chain can do -- joint window selection and latency composition for two *streamed* models with different hops under Hard, so that one re-buffering disappears -- waits for a demonstrated model pair (reversibility rule, as for per-job backend selection above). If it returns it is an edge between two plans that already exist, which is why deferring it costs nothing.

**Shared engine environments** (section 4): one runtime environment per engine, Machine-owned, shared by every handler in the process; per-engine session sharing (ORT shared allocators, prepacked weights) is the adapter-level optimization that makes many-model setups memory-viable. One measured session option belongs here rather than in any single handler: ORT's intra-op pool spin-waits after its work is done, so two sessions that never overlap in time still halve each other -- the palm detector 11.8 ms alone, 23.4 ms beside an *idle* landmark session, ~10 ms with `session.intra_op.allow_spinning = 0` (or one pool for all sessions via `CreateEnvWithGlobalThreadPools`). Invisible to any single-model benchmark, which is why it was found only when the second handler arrived; the shared environment is where anira sets it.

Chunker selection (spec Role + arrival mode; contract fixes only the entry point):

| arrival \\ spec        | Streamed                                                                                      | Buffer                       |
|-----------------------|-----------------------------------------------------------------------------------------------|------------------------------|
| incremental (process) | RingChunker (today's ring buffers)                                                            | accumulate-to-full or reject |
| complete (submit)     | ViewChunker (byte_offset views, refcounted parent, head trim + tail flush in its reassembler) | IdentityChunker              |

Hard uses RingChunker only, with input rings and output rings sized independently so that either side may be empty (section 5, one-sided streaming): a generator has output rings only and `process()` pulls, an analyser has input rings only and its Static outputs bypass the rings. Below the chunker everything is uniform: lanes, pump, stages and backends see model-shaped tensors and never know their origin.

Completion is a fence, never a return value. Measured with ORT's WebGPU EP and graph capture enabled: `Run()` returns ~8 ms before the GPU finishes and host-bound outputs contain the previous job's results on every iteration; without capture `Run()` blocks correctly. The adapter therefore never derives `output_ready` from the engine call returning; it binds outputs in the engine's device domain, obtains the engine's completion fence (queue work-done future, `cudaEvent`, a sync file) and exposes that as the token. Engine features that make execution asynchronous (graph capture, replay, deferred readback) are allowed only through this path.

That path was then built and measured, and the stale outputs turned out to be the adapter's, not the engine's. A replay run in ORT (`inference_session.cc`) skips the framework entirely — no feed copy, no fetch, no host download — and re-dispatches only the captured compute bind groups, against the exact device buffers they were recorded with. The contract a captured graph therefore imposes on the adapter is stricter than "await a fence": **every byte the graph touches must be device-resident, at a stable address, and written by nobody but the caller and the graph.** Three ways to violate it, each measured as stale outputs: host-bound outputs (their download is framework work); an input whose `OrtMemoryInfo` device id does not match the EP's internal `OrtDevice` (id 0 for the WebGPU EP, regardless of the `deviceId` option, which only selects the `WGPUDevice`) — the framework then `MemCpy`s it into an ORT-owned buffer before the graph, a copy neither captured nor repeated, so the graph reads run 0's data forever; and producer-side multi-buffering, since the graph references one of the rotating buffers. With all three satisfied — device-bound outputs fetched after the queue work-done fence, input memory info on the EP's device id, one input buffer — ORT 1.29's WebGPU capture is exact and never stale, across the identity and the dma-buf edges. It is still not worth enabling on this hardware: ~3% on the large model, slower on the small one. Consequences for the planner: capture is a per-plan mode that *disables* multi-buffering on the input side (the two optimizations are mutually exclusive, and the plan report says which one it chose); the adapter's device-id mapping is queried from the EP's allocator, never taken from a session option; and a feature is enabled only when the warmup's stale check has *demonstrated* freshness under it, recorded as a measured bit in the capability report rather than assumed from the completion contract. Test harnesses vary the input per job and flag outputs identical to the previous job's (the hello_inference "stale" check); a benchmark that cannot detect a stale frame measures nothing.

Measured with the outputs bound on the device and completion taken from the token (prototype, palm detector, M1): the engine call is submission, not completion -- ORT's WebGPU EP spends 2-3 ms of *host* time per `Run()` encoding the graph and the GPU another ~7 ms. Graph capture, the trap above, becomes legitimate in this arrangement under one further condition: **the buffers bound to the engine are fixed**. ORT replays the bind groups it captured, i.e. the buffers of the run it captured on, exactly as a CUDA graph replays addresses; with the engine's input and output tensors rotating between two slots, replay alternates fresh and stale outputs (10 of 23 iterations, caught only by a stale check that compares against the last *N* outputs, not the last one). With one fixed slot at the engine, `Run()` falls from 3.3 ms to 1.1 ms of host time, the GPU time is unchanged and the outputs are fresh. Consequence for the planner: the producer-side multi-buffering of section 1 stays on the producer's side of the edge -- under capture the edge moves each rotating slot into fixed engine tensors, and a hand-over edge (the same buffer on both sides) cannot rotate. The 4x capture seemed to give was the missing wait; the 3x it gives is real, and it is submission overhead, not GPU time.

A trap in the ORT adapter that the same check exposed: the OrtDevice id on a `WebGPU_Buffer` memory info must be 0 -- the constant the EP's allocator reports (`webgpu/allocator.h`: `WebGpuDevice{GPU, DEFAULT, NONE, 0}`) -- and not the `deviceId` provider option that selects the Dawn device. A tensor labelled with any other id is on a foreign device as far as the session is concerned: it is copied into an EP-owned buffer before every run and copied back after, a `DeviceCopy` hidden under a `ZeroCopy` row that no cost table shows, and one a captured graph does not replay (the replay read ORT's private copy from capture time; every capture-on cell was stale until the ids were told apart). The version assertion of section 4 has a sibling here: an adapter should assert that its memory info compares equal to the EP allocator's, at session creation, and refuse otherwise.

What `output_ready` is for a WebGPU consumer: Dawn exports `SharedFence`s only from `SharedTextureMemory::EndAccess`, so for a plain `WGPUBuffer` there is no fence to hand out. The host-side token is the queue's work-done future; a consumer on the same queue needs no token at all, because submission order is the guarantee. Neither is `SyncKind::None` ("already visible" -- it is not, to the host), so the enum has a `QueueOrdered` kind (section 1): `poll`/`wait` block on the future, a same-queue GPU consumer proceeds without waiting, and the adapter never fabricates a fence it does not have. CUDA: the token is a `cudaEvent` recorded on the engine's stream after the run (`user_compute_stream` puts ORT on anira's stream, so one event covers edge and engine). A GL consumer reached by registration has no fence either: `cudaGraphicsUnmapResources` orders CUDA's writes before GL's subsequent commands, so `output_ready` is `QueueOrdered` after the unmap, and the map on the way in is a full wait on GL's pending work for that buffer -- coarser than a fence, stated in the plan report.

Backends and capability reports, v1. The five v2 engines on the CPU -- ONNX Runtime, LibTorch, TFLite, LiteRT, ExecuTorch -- so that no deployment loses coverage (section 10), plus the GPU providers where the Machine has the device: ORT WebGPU EP (IOBinding on `WebGPU_Buffer` memory info, the Machine's Dawn), ORT CUDA EP (IOBinding on `user_compute_stream`), ORT DirectML EP (`ID3D12Resource` through the `DML` memory info -- DirectML *is* the D3D12 consumer; there is no separate Direct3D provider), and LibTorch CUDA (DLPack both directions). `MetalBuffer` is enabled with the first engine measured to read an `MTLBuffer` in place -- LibTorch MPS is the candidate, ORT's CoreML EP is host-only -- and not before, per the rule that no row exists without a measurement. Capability reports are *queried*, not tabulated: ORT via EP memory infos, LibTorch via device and DLPack support, at prepare. Later minors, each bringing the extensions it consumes (section 1b): LiteRT CompiledModel + TensorBuffer zero-copy (GL / AHB / Metal, fence-based async, on Android and Apple; measured on desktop Linux its prebuilt WebGPU accelerator accepts only `WebGpuBufferPacked`, rejects dma-buf and cannot adopt the Machine's Dawn, so it is a `{Host}` consumer there, and its CL path is never taken -- section 1), TensorRT and CoreML (`ext::Artifacts`, `ext::ArtifactCache`), and NPU adapters (`ext::Npu`, `ext::NpuHard`, plus registration edges and the artifact cache). Quantized I/O likewise waits for `ext::Quant` and its (de)quant stage; in v1 a quantized model runs only when the producer hands in the model's true dtype.

Kernel quality is per (engine, model class, driver) and is the reason plan sets exist. Same GPU, same Vulkan driver, same model (MediaPipe palm detection, fp32, all nodes on the GPU, no fallback): LiteRT's ML Drift accelerator runs it in 2.2 ms while ORT's WebGPU EP needs 8.8-9.9 ms -- the same as ORT's CPU EP on that machine's eight cores. ML Drift is tuned for exactly this class of mobile convolution network (PHWC4 layouts, fused ops, per-GPU-family kernels); the WebGPU EP's strengths are matmul/attention shapes, and its WGSL passes through Tint -> SPIR-V -> the platform driver. Neither engine is "the GPU backend": which one wins is a measurement per model class and platform, which is what a candidate set plus the benchmarking sweep is for. Two corollaries for the planner: never present a GPU backend as faster than a CPU backend without a measured budget for that model, and remember that an equal-time GPU plan is still a win when the point is *offloading* the CPU -- but measure the host share, it is not the ~150 us a bare queue submit suggests: on the palm detector ORT's WebGPU EP costs the host 2-3 ms of submission for ~7 ms of GPU time, 1 ms with graph capture, so the plan hands back ~7 of ~9.5 ms, not ~9.4 (section 7, completion). That is still the usual case under a Hard contract; the budget just has to be the measured one.

---

## 8. JSON schemas

Three files, three lifetimes, no field in two of them. Loaders are dumb (strings to enums, numbers, construct); all semantic validation happens once in `prepare()` / `Machine` construction, identical for JSON and code. Handles and host-discovered geometry are patched from code, last write wins.

### 8.1 Model file (`model.json`, travels with the model)

```json
{
  "models": [
    { "backend": "onnxruntime", "path": "model.onnx",
      "tensor_names": { "audio_in": "input_0", "mask_out": "output_0" } },
    { "backend": "libtorch", "path": "model.pt",
      "tensor_names": { "audio_in": "x", "mask_out": "y" },
      "entry": { "name": "forward_streaming" } }
  ],
  "default_backend": "onnxruntime",
  "state": "stateless",
  "max_instances": 4,

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

Extension keys (section 1b) sit beside the core keys of the struct they extend -- `"entry"` on a model entry above; later `"quant"` on a tensor, `"artifacts"` on a model entry, `"onnxruntime"` session options -- and resolve through the extension registry, each key being the extension's `kind`. A key the loader does not know is neither dropped nor a load error: it becomes `ext::Unknown` and `prepare()` fails by name. A v2 file's `model_function` is upgraded into `"entry": { "name": ... }` with the one-time warning of section 10; a v2 file without one produces an empty bag.

### 8.2 Machine file (`machine.json`, lives on the box)

```json
{
  "num_threads": 0,
  "wait_strategy": "spin_backoff",
  "log_level": "warning",

  "cuda":   { "device": 0, "pinned_pool_limit": 67108864 },
  "vulkan": { "device": 0 },
  "metal":  { },

  "gl":     { "threads": "caller_thread" }
}
```

Device blocks in JSON imply `Ownership::Owned` (anira creates). `"npu"` and `"artifact_cache_dir"` return as extension keys with the adapters that consume them (section 1b); until then a file that carries them fails `prepare()` by name. Borrowed handles are code-only:

```cpp
auto cc = anira::JsonConfigLoader::machine("machine.json");
cc.gl->display = user_egl_display;      // code completes what JSON declared
cc.gl->context = user_egl_context;
cc.gl->gbm     = user_gbm_device;       // optional: dma-buf-backed GL storage from allocate_*
anira::Machine machine(cc);              // validates: indices exist, borrowed non-null
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
    "max_block_size": 512,
    "rate": 48000,
    "budget": "measured",
    "warmup": "until_stable",
    "on_miss": "bypass"
} }
```

Dual encodings: `"budget"` is `"measured"` or `{"ms": 1.8}`; `"warmup"` is `"until_stable"`, `"none"`, or `{"fixed": 200}`; omitted `"deadline_ms"` is the offline posture (`{"async": {}}`). Hard geometry keys are optional: fixed-rate deployments write them, plugins patch from the host:

```cpp
auto c = anira::JsonConfigLoader::contract("session.json");
std::get<anira::Hard>(c).max_block_size = host_block_size;
std::get<anira::Hard>(c).rate           = host_sample_rate;
handler.prepare(c);
```

Optional in either contract file, top level: `"edge_cost": "strict" | "permissive"` -- the `PlanPolicy` of section 6, because this is the file a test sweep runs.

Never in JSON: per-submit overrides and JobOptions, callbacks, planner-derived values, runtime Tensors.

---

## 9. Usage sketches

### Hard: audio plugin (v2-identical hot path)

```cpp
auto cfg = anira::JsonConfigLoader::model("model.json");
anira::Machine machine(anira::JsonConfigLoader::machine("machine.json"));
anira::InferenceHandler handler(machine, anira::Pipeline{ anira::stage::Inference(cfg) });

// prepareToPlay:
anira::Hard h; h.max_block_size = host_block; h.rate = host_rate;
handler.prepare(h);
set_latency(handler.get_latency());

// audio callback:
handler.process(channel_ptrs, n);
```

### Async with deadline: GL video frames, GPU-resident round trip

```cpp
auto cc = anira::JsonConfigLoader::machine("machine.json");
cc.gl = { .display = egl_dpy, .context = egl_ctx, .gbm = gbm };   // gbm only matters for a WebGPU candidate
anira::Machine machine(cc);
anira::InferenceHandler handler(machine, pipe);            // Inference(cfg, {Backend::OrtCuda})
handler.prepare(anira::Async{ .deadline = 33ms, .on_late = anira::LatePolicy::Drop });

// the app's own SSBOs: to the CUDA EP this is ZeroCopy (registration, cached in the plan);
// to a WebGPU candidate it would be HostCopy -- allocate_input would hand back a dma-buf-backed
// renderbuffer to render into instead, and the plan report says which row each slot got
handler.bind_output(0, anira::Tensor::from_gl_buffer(out_ssbo, GL_SHADER_STORAGE_BUFFER,
                                                     draw_done_fence));
// per frame, on the GL thread (GlThreads::CallerThread):
auto in = anira::Tensor::from_gl_buffer(in_ssbo, GL_SHADER_STORAGE_BUFFER, render_fence);
auto t  = handler.submit({&in, 1});
// next frame:
if (t.poll() == anira::Status::Met) draw_from(out_ssbo);   // output_ready(0) is QueueOrdered after the unmap
else reuse_last_frame();
```

### Async with deadline: camera frame, v1 (the app converts the pixels)

```cpp
auto cc = anira::JsonConfigLoader::machine("machine.json");
cc.vulkan = { .own = anira::Ownership::Borrowed, .instance = inst, .physical = phys,
              .device = dev, .queue_family = qf, .queue_index = 0 };
anira::Machine machine(cc);
anira::InferenceHandler handler(machine, { anira::stage::Inference(palm_cfg, {anira::Backend::OrtWebGpu}) });
handler.prepare(anira::Async{ .deadline = 33ms, .on_late = anira::LatePolicy::Drop });
auto in  = handler.allocate_input (0, anira::Domain::VulkanBuffer);   // exportable + image tag: the DeviceCopy row
auto out = handler.allocate_output(0, anira::Domain::VulkanBuffer);
// per V4L2 frame: the app's own NV12 -> float pass (its VkImage import, its compute shader)
// writes in.handle.vk.buf as an SSBO and signals a timeline value; in.acquire = {VkTimeline, ...}
auto t = handler.submit({&in, 1});
// t.output_ready(0) is a sync file: import it as a semaphore and draw from out
```

### Async with deadline: two models composed by tickets (the tracking case)

```cpp
anira::Machine machine(cc);                                          // one Dawn device, shared
anira::InferenceHandler palm(machine, { anira::stage::Inference(palm_cfg, {anira::Backend::OrtWebGpu}) });
anira::InferenceHandler lmk (machine, { anira::stage::Inference(lmk_cfg,  {anira::Backend::OrtWebGpu}) });
palm.prepare(anira::Async{ .deadline = 33ms, .on_late = anira::LatePolicy::Drop });
lmk.prepare (anira::Async{});                                        // no deadline of its own: the loop has one

auto palm_in  = palm.allocate_input (0, anira::Domain::WgpuBuffer);  // the app's NV12 -> float pass writes these
auto lmk_in   = lmk.allocate_input  (0, anira::Domain::WgpuBuffer);  // (post-v1: FrameToTensor, per-job affine)
auto palm_out = palm.allocate_output(0, anira::Domain::WgpuBuffer);
auto lmk_out  = lmk.allocate_output (0, anira::Domain::WgpuBuffer);
palm.bind_output(0, palm_out);  lmk.bind_output(0, lmk_out);

// per camera frame, in the app's tracking loop -- MediaPipe's graph, node for node:
if (prev_rects.size() < num_hands) {                                 // the gate: application logic
    write_letterbox(palm_in, frame);                                 // the app's pass on the Machine's queue
    auto t = palm.submit({&palm_in, 1});
    t.wait_for(budget);                                              // HostCopy, priced: the decode needs the bytes
    rects = associate(decode_nms(palm_out), prev_rects);             // user code, user state
}
prev_rects.clear();
for (auto& r : rects) {                                              // 0..num_hands runs: application logic
    write_crop(lmk_in, affine_of(r));                                // same frame, different affine
    auto t = lmk.submit({&lmk_in, 1});                               // WgpuBuffer -> WebGPU EP: ZeroCopy
    t.wait_for(budget);
    if (presence(lmk_out) >= 0.5f) prev_rects.push_back(rect_of(lmk_out));
}
// Where B reads A's output DIRECTLY (no host logic between), nothing is waited on at all:
//   mid = a.allocate_output(0, Domain::WgpuBuffer);  a.bind_output(0, mid);
//   auto ta = a.submit({&in, 1});  mid.acquire = ta.output_ready(0);   // QueueOrdered
//   auto tb = b.submit({&mid, 1});                                       // same queue: no wait, one drain at the end
// The camera buffer is read by both handlers and the presenter: requeue it when all three have released it.
```

### Async with deadline: camera frame, all on one GPU queue (post-v1: Frame + FrameToTensor)

```cpp
auto cc = anira::JsonConfigLoader::machine("machine.json");
cc.webgpu = { .own = anira::Ownership::Borrowed, .instance = inst, .device = dev, .queue = q };
anira::Machine machine(cc);
anira::Pipeline pipe {
    anira::stage::FrameToTensor(anira::PixelFormat::NV12, {192, 192}, anira::Letterbox,
                                anira::Normalize01, anira::Layout::NHWC),
    anira::stage::Inference(palm_cfg, {anira::Backend::OrtWebGpu}),
};
anira::InferenceHandler handler(machine, pipe);
handler.prepare(anira::Async{ .deadline = 33ms, .on_late = anira::LatePolicy::Drop });

// per V4L2 frame (dma-buf exported once per buffer):
auto f = anira::Frame::from_dmabuf(fds, offs, pitches, DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_NV12,
                                   w, h, {anira::BT709, anira::Limited}, /*sync_fd*/ -1);
auto t = handler.submit_frame(f);
// requeue the V4L2 buffer when t.input_released(0) signals; the app's own presenter read of
// the same buffer is part of that token (two-reader ownership, section 7)
```

### Async without deadline: offline file rendering (the former offline branch)

```cpp
handler.prepare(anira::Async{});                       // lanes auto, deep pipelining
auto in  = anira::Tensor::from_host(samples, ...);     // borrowed: release = nullptr
handler.bind_output(0, anira::Tensor::from_host(out_buf, ...));   // request_output is post-v1
auto t = handler.submit({&in, 1}, { .head_trim = {-1}, .tail_flush = true });
t.wait();                                              // input-aligned result
```

---

## 10. Migration (v2 -> v3)

Major version; ABI breaks entirely; breakage concentrates in construction-time code.

Survives verbatim: `process`, `push_data`, `pop_data` (all pointer-triple variants), `set_inference_backend`, `get_latency`, `set_input`/`get_output` (now Static tensors underneath), thread-pool sharing, `WaitStrategy`, `LogLevel`.

Mechanical: renames with deprecated aliases for one minor cycle (`InferenceConfig` -> `ModelConfig`, `ContextConfig` -> `MachineConfig`, `Context` -> `Machine`, `JsonConfigLoader::context` -> `::machine`); `TensorShapeList` + `ProcessingSpec` -> `TensorSpec` (shapes -> tagged axes, `preprocess_input_size` -> `window_min == window_max`, hop -> `context`); `prepare(HostConfig)` -> `prepare(Hard{...})` (custom-latency overloads -> per-output `latency`); `max_inference_time` / `warm_up` -> Hard fields; `model_function` -> `ext::Entry{name}` on `ModelData` (section 1b), the one extension v1 ships. JSON v2 schema auto-upgraded by the loader with a one-time warning. Deprecated compat constructors bridge one minor cycle.

Semantic: custom `PrePostProcessor` subclasses (RingBuffer/BufferF virtuals) become host-domain stages; a `LegacyProcessorStage` adapter keeps existing subclasses running through v3.

Free: the offline API is unreleased; it folds into Async before its release (lanes survive, tickets subsume callback + poll, per-job non-streamable values become Static input tensors, dissolving the exclusive-scheduling rule). The web/TS wrapper migrates in lockstep.

Platform coverage is unchanged: `Host` is enabled on every platform v2 runs on (section 1), so every v2 deployment migrates without touching a domain. The GPU domains are additions on top, per platform.

Purely additive (no migration impact): variant sets with `set_model`, composition of models across handlers on one Machine through tickets, shared engine environments, the WebGPU machine resource, `allocate_input`/`allocate_output`, the `Extensions` slots of section 1b. Deferred past v1, additive when they come: `Frame` + `FrameToTensor` + `submit_frame` (including `Container::IOSurface`, the pixel-carrying role of the same handle, and `ext::CropAffine`), chained `Inference` stages inside one Pipeline (section 7, Multi-model support), `request_output`, `GlThreads::SharedContext`, a per-run anchor override on `Hard` (which input is the clock, for multi-rate models -- the field waits for such a model), every `Domain` arm outside the v1 set of section 1, and every entry of section 1b's deferred catalogue -- `ext::Quant`, `ext::Artifacts`/`ext::ArtifactCache`, `ext::Npu`/`ext::NpuHard`, `ext::OrtSession`, `ext::JobBackend`/`ext::JobModel` -- each with the stage or adapter that consumes it. Quantization and NPU support are therefore out of v1 scope entirely, not partially. Excluded, not deferred: OpenCL (section 1). The former multi-plane `dmabuf` arm of `MemoryHandle` is removed before any release that ships it; `from_dmabuf` keeps its name with the buffer meaning.

---

## 11. Implementation roadmap

The design above lands in anira as a sequence of milestones, each a set of pull requests that merges with the full test suite green, so that the work proceeds one step at a time and can stop at any step with a coherent library. Two rules hold throughout. The v2 test suite is the oracle for the Hard path: `process`/`push_data`/`pop_data` survive verbatim (section 10), and every milestone that touches the pump keeps those tests passing through whatever API exists at that point. And no registry row, cost class or platform column is coded before it is measured (sections 1 and 7): a milestone that needs a measurement lists it as a precondition rather than assuming it.

Strategy: replace subsystems in place, never rewrite from scratch. v2's ring buffers become the RingChunker's storage, its inference threads and lock-free queue become the pump, its `PrePostProcessor` subclasses keep running behind `LegacyProcessorStage`. The public API changes at the seams section 10 names; the internals are swapped one at a time underneath a green suite.

Branching: M0 ships from `main` as v2.3.0. Afterwards `main` carries v2 maintenance until v3.0.0 merges, and v3 work lands on a long-lived `v3` integration branch through PRs, tagged `v3.0.0-alpha.N` / `-beta.N` at the milestone boundaries below. anira's repository rules apply to every PR: Doxygen on public headers, `docs/sphinx`, `CHANGELOG.md` with breaking entries prefixed `**Breaking:**`.

### M0 -- v2.3.0: the last v2 release, no API break

What `main` already holds unreleased: the ExecuTorch backend and its `model_function` support, LiteRT as the default TensorFlow-family backend, the wait-free `reset()`, hidden backend symbols, the concurrent-lifecycle fixes, the receptive-field default pre-processor, the Fedora ExecuTorch link fix. Added in this milestone, in order:

1. **tanh-lib core containers** (anira `feat/tanh-core-buffers`, tanh-lib#12). `anira::Buffer<T>`, `RingBuffer` and `MemoryBlock<T>` become aliases over `thl::core`; `RingBufferT<T>` gives integer and other element types -- the v3 chunker's typed storage (open decisions, below). Preconditions on the tanh-lib side, all before the anira PR leaves draft: `tanh::Core` links nothing platform-specific by default (the journald sink opt-in or `dlopen`ed, and the Emscripten build must not fall into the Linux branch); the containers do not depend on the logger; the ring wraps with a compare rather than `%` (armv7l has no integer divide and anira pushes per sample); tanh-lib's install/export works, so `ANIRA_WITH_INSTALL` consumers get `find_dependency(tanh-lib COMPONENTS Core)`; a tagged tanh-lib release replaces the branch pin. Acceptance in anira: Linux, RTSan, WASM and install tests green; the ring-overflow entry leaves the RTSan suppressions file; the cnn-size benchmark within noise of v2.2.1.
2. **Logging through `thl::Logger`** (section 4). The `LOG_*` ostream macros move to `thl::Logger` with `anira.<component>` groups; `ContextConfig::log_level` maps onto the logger's runtime level -- which tanh-lib gains for this, its filtering being compile-time only today -- and still forwards to the engines. anira never calls `set_config`. Nothing reachable from an `ANIRA_REALTIME` entry calls the logger: RTSan enforces this with no suppression, and the sites it flags get the lock-free record front that `thl::core` grows for its own audio threads.
3. **CI on `tanh-lab/ci-actions`.** anira's `setup`/`build`/`test` composites are replaced by the shared actions in preset mode; anira gains per-platform presets (sanitizers, Android, iOS, Windows-arm64, macOS-universal, shared/static x backend sets); one gcc job stays; `GITHUB_TOKEN` is set at job level for the backends download; the mobile test actions gain inputs for the extra files anira pushes (backend `.so`s, `libc++_shared`, `extras/models`), or anira keeps its mobile workflow; the `install`/codesign/release action stays anira's; ci-actions pinned to a tag like tanh-tooling. The `.clang-*` files stay as they are: installed from tanh-tooling, drift-checked in CI.
4. **Test reorganisation, both repositories.** anira's single `tests` binary with root-level files becomes per-component binaries mirroring `src/` -- `test_utils`, `test_scheduler`, `test_backends`, `test_handler` (the integration suite: `InferenceHandler`, stateful ordering, the model fixtures) -- with the shared fixtures (`WavReader`, model paths) in `test/fixtures/` and the install smoke test kept apart. The `test_*` naming is what ci-actions' Android runner auto-discovers, and per-component binaries are where M2's adapter tests and M4's probe suite slot in. tanh-lib: the containers that moved to core in #12 take their tests with them (`test_RingBuffer`, `test_AudioBuffer*` from `test/dsp/` to `test/core/`), `benchmark_*` targets separated from `test_*`, the DSP fixtures under `test/modulation/` named for what they test. No behaviour change; the suites' test counts before and after are the acceptance.
5. **One-sided streaming, redone** (the former "one way streaming only" note). anira PR #101 fixed the `prepare()` hang for generator-style configs -- no streamable input, so the reference tensor's size is 0, the buffer ratio `inf`, and the smaller-buffer countdown never ends -- and the segfault for analyser-style configs -- no streamable output, so `sync_latencies()` indexes an empty latency vector -- with a reference-size fallback and index-aligned latencies, and was reverted (#110) because it patched the symptoms: v2's reference stream is by construction an input, `HostConfig::m_tensor_index`. The redo makes the reference stream a first-class notion in v2 -- an input or an output, chosen the way section 5's anchor is chosen -- and keeps the two regression configs (params-in/audio-out, audio-in/counter-out, each with and without `allow_smaller_buffers`) as tests. It ships in v2.3.0 because M2's RingChunker inherits whatever v2 does here.
6. `fix/pooled-processor-dangling-config` lands. `feat/offline-inference` does **not** ship: it folds into `Async` in M3 (section 10).

Exit: tag `v2.3.0`, release artifacts from `on_tag.yml`. "v2 parity" everywhere in this document means this tag.

### M1 -- configuration layer (`v3` branch; no runtime change)

Everything in sections 1b, 2, 3, 4 (the struct), 5, 6 (`PlanPolicy`) and 8 that is data: `TensorSpec` with tagged axes, roles, window/context/latency; `ModelConfig`/`ModelData`; `MachineConfig`; `Contract = std::variant<Hard, Async>`; `PlanPolicy`; the `Extensions` carrier, registry, `ext::Unknown`, the consumed-or-fail walk and `ext::Entry`; `JsonConfigLoader::model/machine/contract` with the v2 auto-upgrade and its one-time warning; deprecated aliases (`InferenceConfig`, `ContextConfig`, `Context`, `JsonConfigLoader::context`). Internally a translator from the new structs to v2's `InferenceConfig`/`ContextConfig`/`HostConfig`, so the v2 engine runs unchanged behind the v3 configuration. Tests: the prepare-time legality rules of section 2, JSON round trips, the v2 upgrade, consumed-or-fail by name. Why first: driver-free, the largest public surface, and it makes every later milestone testable through the final API.

Exit: every example builds against the v3 configuration types; `v3.0.0-alpha.1`.

### M2 -- Tensor and the Host-only pump

`Tensor`, `SyncToken`, `Domain::Host` only; the edge registry, `PlanReport` and cost classes with Host rows only; `Machine` replacing the `inline static` `Context` (thread pool process-global, first Machine wins); `InferenceHandler(machine, pipeline)`, `stage::Inference(cfg)`, `prepare(Hard)` compiling a trivial plan over the RingChunker, which is v2's ring buffers on `thl::core::RingBufferT`; one type-erased backend adapter interface replacing the five `#ifdef` paths (`InferenceBackend` members, `SessionElement`'s per-backend pointers, `InferenceThread::inference`), so the ABI stops depending on which engines a build enabled; the shared engine environments of section 4 with `session.intra_op.allow_spinning = 0`; `LegacyProcessorStage` around existing `PrePostProcessor` subclasses; `set_input`/`get_output` as Static tensors; the LibTorch and ExecuTorch adapters consuming `ext::Entry`; the Machine owning the real-time log drain.

Exit: the complete v2 test suite passes through the v3 API on the Hard path; the JUCE and CLAP examples and the WASM wrapper build; `v3.0.0-alpha.2`.

### M3 -- Async

`submit`/`Ticket` (`poll`, `wait`, `wait_for`, `status`, `input_released`, `output_ready`), `JobOptions`, `ViewChunker` and `IdentityChunker`, lanes, `max_in_flight`, EDF ordering and `LatePolicy::Drop`, `Delivery::Immediate`, `bind_output`, borrowed `from_host` lifetimes; `feat/offline-inference` absorbed (lanes survive, tickets subsume callback + poll). Still Host-only. The stale-output check (section 7, completion) enters the test harness here, not in M4. Precondition: the streaming-direction decision below.

Exit: offline file rendering matches v2's non-real-time output; met/late/dropped accounting tested; `v3.0.0-alpha.3`.

### M4 -- Machine probing and the Linux GPU domains

The prototype's `infer/` layer ported into the Machine, file by file: `infer_ctx_{vk,gl,wgpu,cuda}.c` -> the device blocks, the identity checks and the functional probes (section 4, three rungs, cached on the box); `infer_edge.c` and `infer_edge_cache` -> the registry and the plan-owned edge cache; `infer_vk_cuda.c` -> the opaque-fd rows and the three-hop bridge; `infer_engine_ort.c` -> the WebGPU and CUDA EP adapters (IOBinding, `user_compute_stream`, the EP-allocator device-id assertion, capture as a per-plan mode gated by the stale check, `QueueOrdered`); `allocate_input`/`allocate_output`; `SyncToken` fd ownership; the byte image and `Machine::byte_image_bytes`. Order by what is measured: `WgpuBuffer`/WebGPU EP on Mesa first, then the Vulkan, dma-buf and GL export rows, then CUDA on the NVIDIA box. Build: `build-deps.sh` becomes the external-Dawn ORT variant of the anira-project/backends release, with the Dawn revision assertion of section 4. Tests: `hello_inference`'s matrix becomes the functional-probe suite under ctest, `--strict` = `EdgeCost::Strict`. This milestone needs a GPU CI runner (Mesa and NVIDIA), which anira's CI does not have.

Exit: every row of section 7's Linux tables is green against anira's Machine, strict, and the prototype's `hello_inference` runs against anira instead of its own `infer/`; `v3.0.0-beta.1`.

### M5 -- plan sets, variants, documentation

One plan per (variant, backend) candidate; `set_inference_backend` and `set_model` as atomic selection (`thl::core::RCU` is the candidate primitive; readers register at `prepare`); per-plan Hard warm-up and budget, worst case across the set; the JSON benchmark sweep with governor and wait strategy in every report; the plan report per extension consumed; Doxygen, Sphinx, changelog, the web/TS wrapper at Host level.

Exit: `v3.0.0-beta.2`.

### M6 -- WebGPU acceleration in the browser

The ORT WebGPU EP in anira's Emscripten build. The Machine borrows the browser's `GPUDevice` through Emscripten's WebGPU bindings instead of linking Dawn (`MachineConfig::WebGpu` is always `Borrowed` there, and the Dawn revision assertion of section 4 is replaced by the browser's own implementation), `WgpuBuffer` tensors are `GPUBuffer`s the page's own compute passes write, and the audio worklet keeps `Host`. The web/TS wrapper exposes `allocate_input`/`allocate_output` and tickets for the WebGPU case; the WASM CI job runs the identity model with the stale check in a headless browser. Measured like every other column before its rows exist: the browser joins section 1's table with that run, and the M4 probe suite is what runs there.

Exit: `v3.0.0`.

### After v3.0 (additive; section 10)

`Frame` + `FrameToTensor` + `submit_frame`; `request_output`; `GlThreads::SharedContext`; the Windows, Apple and Android GPU columns, each after its own matrix run; the deferred extension catalogue; chained `Inference` stages if a model pair demands it.

### Open decisions, and the milestone that needs them

- **Streaming direction** (the former "one way streaming only" note): resolved. Streamed tensors on one side only are first-class -- section 2's legality rule, section 5's anchor that may be an output, section 7's RingChunker with independently sized sides. The v2 half is M0's redo of PR #101; the v3 half is M2.
- **Chunker element type**: resolved. `RingBufferT<T>` from `thl::core` (M0); the Time-axis storage of a Streamed tensor is typed by its `dtype`, so int8/int16 models stream exact values. M2 consumes it.
- **GPU CI hardware** for M4: which machines run the functional-probe suite, and whether the probe cache is shipped as a fixture so CPU-only CI can exercise the planner against a recorded registry.
- **`ext::Entry`**: chosen in this revision (section 1b) over one extension per adapter; revisited only if a third adapter needs a different shape.
- **Anchor override on `Hard`** stays deferred until a multi-rate model exists.
