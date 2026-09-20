// anira/abi/tensor.h and anira/abi/draft/tensor_platform.h: the tensor factories (field fills
// over caller memory), the accessors, anira_sizeof, the descriptor ownership of the sync
// token and the DLPack bridge. Only anira_tensor_init_dlpack sits behind the exception
// firewall of capi_internal.h; the factories and accessors are real-time entries and have no
// handler.
#include <anira/abi/context.h>  // IWYU pragma: keep - a record of generated/struct_sizes.inc
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>  // IWYU pragma: keep - a record of generated/struct_sizes.inc
#include <anira/abi/log.h>      // IWYU pragma: keep - a record of generated/struct_sizes.inc
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

// The two owning sync kinds hold an operating-system object: an fd on POSIX, an NT handle on
// Windows, nothing on WebAssembly (a module has no descriptor to own; 0, 1 and 2 are its stdio).
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#endif

#include "capi_internal.h"

using anira::capi::translate_exception;

namespace {

// ==== the fill every factory shares =========================================================

/// Zero, then fill: the part every factory shares. False leaves the record all-zero (dtype 0,
/// ndim 0): a NULL record, dtype 0, a rank above ANIRA_MAX_RANK, a NULL shape with a rank above
/// 0, or a negative extent. The extents are read into a local before the record is zeroed, so
/// shape may point into the record itself (init_host(&t, p, t.dtype, t.ndim, t.shape)).
/// `admitted` is a factory's own refusal, evaluated by the caller before the zeroing.
/// Real-time: no allocation, no lock, no system call.
bool begin_tensor(anira_tensor* tensor,
                  anira_domain domain,
                  anira_dtype dtype,
                  uint32_t ndim,
                  const int64_t* shape,
                  bool admitted = true) noexcept {
    if (tensor == nullptr) { return false; }
    std::array<int64_t, ANIRA_MAX_RANK> extents{};
    bool accepted =
        admitted && dtype != 0 && ndim <= ANIRA_MAX_RANK && (ndim == 0 || shape != nullptr);
    for (uint32_t axis = 0; accepted && axis < ndim; ++axis) {
        extents.at(axis) = shape[axis];
        accepted = extents.at(axis) >= 0;
    }
    std::memset(tensor, 0, sizeof(*tensor));
    if (!accepted) { return false; }
    tensor->domain = static_cast<uint32_t>(domain);
    tensor->dtype = dtype;
    tensor->ndim = ndim;
    for (uint32_t axis = 0; axis < ndim; ++axis) { tensor->shape[axis] = extents.at(axis); }
    return true;
}

/// A token handed by pointer, read BEFORE the record is zeroed: the fence may be the record's
/// own acquire. NULL = visible (an all-zero token, ANIRA_SYNC_NONE).
anira_sync_token snapshot(const anira_sync_token* fence) noexcept {
    anira_sync_token token;
    std::memset(&token, 0, sizeof(token));
    if (fence != nullptr) { token = *fence; }
    return token;
}

/// A sync_file fd becomes an owning ANIRA_SYNC_SYNC_FILE_FD token; negative = visible.
void own_sync_file(anira_tensor& tensor, int32_t fd) noexcept {
    if (fd < 0) { return; }
    tensor.acquire.kind = static_cast<uint32_t>(ANIRA_SYNC_SYNC_FILE_FD);
    tensor.acquire.u.fd = fd;
}

// ==== the accessors' rules ==================================================================

bool is_host(const anira_tensor& tensor) noexcept {
    return tensor.domain == static_cast<uint32_t>(ANIRA_DOMAIN_HOST) ||
           tensor.domain == static_cast<uint32_t>(ANIRA_DOMAIN_HOST_PINNED);
}

bool is_planar(const anira_tensor& tensor) noexcept {
    return (tensor.flags & static_cast<uint32_t>(ANIRA_TENSOR_PLANAR)) != 0;
}

/// base + byte_offset, or NULL for a NULL base or an offset beyond size_t (wasm32).
void* offset_into(void* base, uint64_t byte_offset) noexcept {
    if (base == nullptr) { return nullptr; }
    const auto offset = static_cast<size_t>(byte_offset);
    if (static_cast<uint64_t>(offset) != byte_offset) { return nullptr; }
    return static_cast<unsigned char*>(base) + offset;
}

/// handle.host.ptr + byte_offset for a one-block tensor of the two host domains, else NULL.
void* host_data(const anira_tensor& tensor) noexcept {
    if (!is_host(tensor) || is_planar(tensor)) { return nullptr; }
    return offset_into(tensor.handle.host.ptr, tensor.byte_offset);
}

/// An extent as a count: false when it is negative or does not fit size_t (wasm32).
bool to_count(int64_t extent, size_t& count) noexcept {
    if (extent < 0) { return false; }
    const auto wide = static_cast<uint64_t>(extent);
    count = static_cast<size_t>(wide);
    return static_cast<uint64_t>(count) == wide;
}

/// Whether the token holds a descriptor of one of the two owning kinds: an fd of 0 or more; on
/// Windows an NT handle above 0 (0 is NULL, negative values are pseudo-handles and
/// INVALID_HANDLE_VALUE). What owning means is the platform's: see reset and dup.
bool holds_descriptor(const anira_sync_token& token) noexcept {
    const bool owning_kind = token.kind == static_cast<uint32_t>(ANIRA_SYNC_SYNC_FILE_FD) ||
                             token.kind == static_cast<uint32_t>(ANIRA_SYNC_OPAQUE_FD_SEMAPHORE);
#if defined(_WIN32)
    return owning_kind && token.u.fd > 0;
#else
    return owning_kind && token.u.fd >= 0;
#endif
}

#if defined(_WIN32)
// An NT handle has 32 significant bits and is sign-extended to the pointer width (the rule of
// the 64-bit Windows interoperability guarantee), which is what lets it sit in int32_t fd.
HANDLE to_handle(int32_t fd) noexcept {
    // NOLINTNEXTLINE(performance-no-int-to-ptr) the documented representation of a HANDLE
    return reinterpret_cast<HANDLE>(static_cast<intptr_t>(fd));
}
int32_t from_handle(HANDLE handle) noexcept {
    return static_cast<int32_t>(reinterpret_cast<intptr_t>(handle));
}
#endif

// ==== the private DLPack mirror =============================================================

// Mirror of dmlc/dlpack include/dlpack/dlpack.h, DLPack 1.3 (tag v1.3, commit 84d107b). The
// ABI identity is the major (1): a minor adds enumerators only. Track
// DLManagedTensorVersioned exactly; test/abi/test_Tensor.cpp spells it a second time.
// Natural C types, never ANIRA_PTR: the layout is the platform's DLPack layout.
constexpr uint32_t k_dlpack_major = 1;                       // DLPACK_MAJOR_VERSION
constexpr int32_t k_dl_cpu = 1;                              // kDLCPU
constexpr int32_t k_dl_cuda_host = 3;                        // kDLCUDAHost
constexpr uint8_t k_dl_code_max = 6;                         // kDLBool: the last shared code
constexpr uint64_t k_dl_flag_read_only = uint64_t{1} << 0U;  // DLPACK_FLAG_BITMASK_READ_ONLY

// NOLINTBEGIN(readability-identifier-naming) the members spell DLPack's field names
struct DlpackVersion {
    uint32_t major;
    uint32_t minor;
};
struct DlpackDevice {
    int32_t device_type;  // DLDeviceType: a plain enum in C, enum : int32_t in C++
    int32_t device_id;
};
struct DlpackDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};
struct DlpackTensor {
    void* data;
    DlpackDevice device;
    int32_t ndim;
    DlpackDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
};
struct DlpackManagedTensorVersioned {
    DlpackVersion version;
    void* manager_ctx;
    void (*deleter)(DlpackManagedTensorVersioned* self);
    uint64_t flags;
    DlpackTensor dl_tensor;
};
// NOLINTEND(readability-identifier-naming)
static_assert(sizeof(DlpackDataType) == 4 && sizeof(DlpackDevice) == 8 &&
              sizeof(DlpackVersion) == 8);
static_assert(sizeof(void*) != 8 ||
              (sizeof(DlpackTensor) == 48 && sizeof(DlpackManagedTensorVersioned) == 80));
static_assert(sizeof(void*) != 8 || (offsetof(DlpackManagedTensorVersioned, deleter) == 16 &&
                                     offsetof(DlpackManagedTensorVersioned, dl_tensor) == 32));

/// anira_tensor::release of a DLPack-backed tensor: disarms this descriptor, then calls the
/// producer's deleter, which frees the managed tensor. A second call on the same descriptor is a
/// no-op; bitwise copies are the caller's to keep to one release.
#if defined(__clang__)
#define ANIRA_DLPACK_FOREIGN_CALL __attribute__((no_sanitize("function")))
#else
#define ANIRA_DLPACK_FOREIGN_CALL
#endif
ANIRA_DLPACK_FOREIGN_CALL void ANIRA_CALL dlpack_release(anira_tensor* tensor) noexcept {
    if (tensor == nullptr) { return; }
    auto* managed = static_cast<DlpackManagedTensorVersioned*>(tensor->manager_ctx);
    tensor->manager_ctx = nullptr;
    tensor->release = nullptr;
    if (managed != nullptr && managed->deleter != nullptr) { managed->deleter(managed); }
}

}  // namespace

// ==== the factories =========================================================================

void ANIRA_CALL anira_tensor_init_host(anira_tensor* tensor,
                                       void* data,
                                       anira_dtype dtype,
                                       uint32_t ndim,
                                       const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_HOST, dtype, ndim, shape)) { return; }
    tensor->handle.host.ptr = data;
}

void ANIRA_CALL anira_tensor_init_pinned(anira_tensor* tensor,
                                         void* data,
                                         anira_dtype dtype,
                                         uint32_t ndim,
                                         const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_HOST_PINNED, dtype, ndim, shape)) { return; }
    tensor->handle.host.ptr = data;
}

void ANIRA_CALL anira_tensor_init_host_planar(anira_tensor* tensor,
                                              const void* planes,
                                              uint32_t count,
                                              anira_dtype dtype,
                                              uint32_t ndim,
                                              const int64_t* shape)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    // Axis 0 is the plane axis: no rank 0, and one pointer per plane. Read before the zeroing.
    // std::cmp_equal compares the values: a negative extent equals no count.
    const bool one_pointer_per_plane =
        ndim > 0 && shape != nullptr && std::cmp_equal(shape[0], count);
    if (!begin_tensor(tensor, ANIRA_DOMAIN_HOST, dtype, ndim, shape, one_pointer_per_plane)) {
        return;
    }
    tensor->flags = static_cast<uint32_t>(ANIRA_TENSOR_PLANAR);
    tensor->handle.planes.ptrs = static_cast<void* const*>(planes);
    tensor->handle.planes.count = count;
}

void ANIRA_CALL anira_tensor_init_cuda(anira_tensor* tensor,
                                       void* ptr,
                                       int32_t device,
                                       void* cuda_event,
                                       anira_dtype dtype,
                                       uint32_t ndim,
                                       const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_CUDA, dtype, ndim, shape)) { return; }
    tensor->handle.cuda.ptr = ptr;
    tensor->handle.cuda.device = device;
    if (cuda_event != nullptr) {
        tensor->acquire.kind = static_cast<uint32_t>(ANIRA_SYNC_CUDA_EVENT);
        tensor->acquire.u.cuda_event = cuda_event;
    }
}

void ANIRA_CALL anira_tensor_init_gl_buffer(anira_tensor* tensor,
                                            uint32_t id,
                                            uint32_t target,
                                            void* gl_sync,
                                            anira_dtype dtype,
                                            uint32_t ndim,
                                            const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_GL_BUFFER, dtype, ndim, shape)) { return; }
    tensor->handle.gl.id = id;
    tensor->handle.gl.target = target;
    if (gl_sync != nullptr) {
        tensor->acquire.kind = static_cast<uint32_t>(ANIRA_SYNC_GL_SYNC);
        tensor->acquire.u.gl_sync = gl_sync;
    }
}

void ANIRA_CALL anira_tensor_init_vulkan(anira_tensor* tensor,
                                         uint64_t buffer,
                                         uint64_t memory,
                                         uint64_t offset,
                                         uint64_t timeline_semaphore,
                                         uint64_t value,
                                         anira_dtype dtype,
                                         uint32_t ndim,
                                         const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_VULKAN_BUFFER, dtype, ndim, shape)) { return; }
    tensor->handle.vk.buffer = buffer;
    tensor->handle.vk.memory = memory;
    tensor->handle.vk.offset = offset;
    if (timeline_semaphore != 0) {
        tensor->acquire.kind = static_cast<uint32_t>(ANIRA_SYNC_VK_TIMELINE);
        tensor->acquire.u.vk.semaphore = timeline_semaphore;
        tensor->acquire.u.vk.value = value;
    }
}

void ANIRA_CALL anira_tensor_init_opaque_fd(anira_tensor* tensor,
                                            int32_t fd,
                                            uint64_t size,
                                            anira_dtype dtype,
                                            uint32_t ndim,
                                            const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_OPAQUE_FD, dtype, ndim, shape)) { return; }
    tensor->handle.opaque.fd = fd;
    tensor->handle.opaque.size = size;
}

void ANIRA_CALL anira_tensor_init_wgpu_buffer(anira_tensor* tensor,
                                              void* wgpu_buffer,
                                              uint64_t offset,
                                              const anira_sync_token* fence,
                                              anira_dtype dtype,
                                              uint32_t ndim,
                                              const int64_t* shape)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const anira_sync_token token = snapshot(fence);
    if (!begin_tensor(tensor, ANIRA_DOMAIN_WGPU_BUFFER, dtype, ndim, shape)) { return; }
    tensor->handle.wgpu.buffer = wgpu_buffer;
    tensor->handle.wgpu.offset = offset;
    tensor->acquire = token;
}

void ANIRA_CALL anira_tensor_init_dmabuf(anira_tensor* tensor,
                                         int32_t fd,
                                         uint64_t size,
                                         uint64_t offset,
                                         int32_t sync_fd,
                                         anira_dtype dtype,
                                         uint32_t ndim,
                                         const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_DMABUF, dtype, ndim, shape)) { return; }
    tensor->handle.dmabuf.fd = fd;
    tensor->handle.dmabuf.size = size;
    tensor->handle.dmabuf.offset = offset;
    own_sync_file(*tensor, sync_fd);
}

// ==== the draft platform factories (anira/abi/draft/tensor_platform.h) ======================

void ANIRA_CALL anira_tensor_init_metal(anira_tensor* tensor,
                                        void* buffer,
                                        const anira_sync_token* shared_event,
                                        anira_dtype dtype,
                                        uint32_t ndim,
                                        const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const anira_sync_token token = snapshot(shared_event);
    if (!begin_tensor(tensor, ANIRA_DOMAIN_METAL_BUFFER, dtype, ndim, shape)) { return; }
    tensor->handle.mtl.buffer = buffer;
    tensor->acquire = token;
}

void ANIRA_CALL anira_tensor_init_iosurface(anira_tensor* tensor,
                                            void* surface,
                                            uint64_t size,
                                            const anira_sync_token* shared_event,
                                            anira_dtype dtype,
                                            uint32_t ndim,
                                            const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const anira_sync_token token = snapshot(shared_event);
    if (!begin_tensor(tensor, ANIRA_DOMAIN_IOSURFACE, dtype, ndim, shape)) { return; }
    tensor->handle.iosurface.surface = surface;
    tensor->handle.iosurface.size = size;
    tensor->acquire = token;
}

void ANIRA_CALL anira_tensor_init_ahardwarebuffer(anira_tensor* tensor,
                                                  void* buffer,
                                                  int32_t fence_fd,
                                                  anira_dtype dtype,
                                                  uint32_t ndim,
                                                  const int64_t* shape)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!begin_tensor(tensor, ANIRA_DOMAIN_AHARDWAREBUFFER, dtype, ndim, shape)) { return; }
    tensor->handle.ahb.buffer = buffer;
    own_sync_file(*tensor, fence_fd);
}

void ANIRA_CALL anira_tensor_init_d3d12(anira_tensor* tensor,
                                        void* resource,
                                        void* shared_handle,
                                        const anira_sync_token* fence,
                                        anira_dtype dtype,
                                        uint32_t ndim,
                                        const int64_t* shape) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const anira_sync_token token = snapshot(fence);
    if (!begin_tensor(tensor, ANIRA_DOMAIN_D3D12, dtype, ndim, shape)) { return; }
    tensor->handle.d3d12.resource = resource;
    tensor->handle.d3d12.shared_handle = shared_handle;
    tensor->acquire = token;
}

// ==== the DLPack bridge =====================================================================

anira_status ANIRA_CALL anira_tensor_init_dlpack(anira_tensor* tensor,
                                                 void* dl_managed_tensor_versioned,
                                                 anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(tensor != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "dlpack: NULL tensor");
    ANIRA_CAPI_REQUIRE(dl_managed_tensor_versioned != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "dlpack: NULL managed tensor");
    auto* managed = static_cast<DlpackManagedTensorVersioned*>(dl_managed_tensor_versioned);
    // The major first: under another major nothing past `flags` has a known layout.
    ANIRA_CAPI_REQUIRE(managed->version.major == k_dlpack_major,
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "dlpack: major version %u, this build reads major %u",
                       static_cast<unsigned>(managed->version.major),
                       static_cast<unsigned>(k_dlpack_major));
    const DlpackTensor& dl = managed->dl_tensor;
    const bool pinned = dl.device.device_type == k_dl_cuda_host;
    ANIRA_CAPI_REQUIRE(pinned || dl.device.device_type == k_dl_cpu,
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "dlpack: device type %d is neither kDLCPU nor kDLCUDAHost",
                       static_cast<int>(dl.device.device_type));
    ANIRA_CAPI_REQUIRE(dl.dtype.code <= k_dl_code_max,
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "dlpack: dtype code %u has no anira_dtype_code",
                       static_cast<unsigned>(dl.dtype.code));
    ANIRA_CAPI_REQUIRE(dl.dtype.bits != 0 && dl.dtype.lanes != 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "dlpack: a dtype of %u bits and %u lanes",
                       static_cast<unsigned>(dl.dtype.bits),
                       static_cast<unsigned>(dl.dtype.lanes));
    ANIRA_CAPI_REQUIRE(dl.ndim >= 0 && dl.ndim <= ANIRA_MAX_RANK,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "dlpack: rank %d is outside 0..%d",
                       static_cast<int>(dl.ndim),
                       ANIRA_MAX_RANK);
    const auto ndim = static_cast<uint32_t>(dl.ndim);
    ANIRA_CAPI_REQUIRE(ndim == 0 || dl.shape != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "dlpack: NULL shape at rank %u",
                       static_cast<unsigned>(ndim));
    // All-zero strides are this record's spelling of packed row-major; in DLPack they are a
    // fully broadcast view. Over at most one element the two readings agree.
    bool broadcast = dl.strides != nullptr && ndim > 0;
    bool empty = false;
    bool plural = false;
    for (uint32_t axis = 0; axis < ndim; ++axis) {
        ANIRA_CAPI_REQUIRE(dl.shape[axis] >= 0,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "dlpack: negative extent on axis %u",
                           static_cast<unsigned>(axis));
        empty = empty || dl.shape[axis] == 0;
        plural = plural || dl.shape[axis] > 1;
        broadcast = broadcast && dl.strides[axis] == 0;
    }
    ANIRA_CAPI_REQUIRE(!broadcast || empty || !plural,
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "dlpack: all-zero strides over more than one element (a fully broadcast "
                       "view) have no anira spelling");
    // Validated: zero, then fill. A refused call has left *tensor untouched.
    const anira_dtype dtype = ANIRA_MAKE_DTYPE(dl.dtype.code, dl.dtype.bits, dl.dtype.lanes);
    const anira_domain domain = pinned ? ANIRA_DOMAIN_HOST_PINNED : ANIRA_DOMAIN_HOST;
    static_cast<void>(begin_tensor(tensor, domain, dtype, ndim, dl.shape));
    if (dl.strides != nullptr) {
        for (uint32_t axis = 0; axis < ndim; ++axis) { tensor->strides[axis] = dl.strides[axis]; }
    }
    tensor->byte_offset = dl.byte_offset;
    tensor->handle.host.ptr = dl.data;
    if ((managed->flags & k_dl_flag_read_only) != 0) {
        tensor->flags |= static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY);
    }
    if (managed->deleter != nullptr) {
        tensor->manager_ctx = managed;
        tensor->release = &dlpack_release;
    }
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

// ==== the accessors =========================================================================

float* ANIRA_CALL anira_tensor_data_f32(const anira_tensor* tensor)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (tensor == nullptr || tensor->dtype != ANIRA_DTYPE_F32) { return nullptr; }
    return static_cast<float*>(host_data(*tensor));
}

void* ANIRA_CALL anira_tensor_data(const anira_tensor* tensor,
                                   anira_dtype dtype) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (tensor == nullptr || dtype == 0 || tensor->dtype != dtype) { return nullptr; }
    return host_data(*tensor);
}

void* ANIRA_CALL anira_tensor_plane(const anira_tensor* tensor,
                                    uint32_t plane,
                                    anira_dtype dtype) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (tensor == nullptr || dtype == 0 || tensor->dtype != dtype) { return nullptr; }
    if (!is_host(*tensor) || !is_planar(*tensor)) { return nullptr; }
    const auto& planes = tensor->handle.planes;
    if (planes.ptrs == nullptr || plane >= planes.count) { return nullptr; }
    // The array holds the caller's pointer objects (float*, const int16_t*, ...): copy the
    // bytes of one, never read it through an lvalue of another pointer type.
    void* base = nullptr;
    std::memcpy(static_cast<void*>(&base),
                static_cast<const void*>(planes.ptrs + plane),
                sizeof(base));
    return offset_into(base, tensor->byte_offset);
}

size_t ANIRA_CALL anira_tensor_num_elements(const anira_tensor* tensor)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    // dtype 0 is the all-zero record of a refused factory, never a filled one: no elements.
    if (tensor == nullptr || tensor->dtype == 0 || tensor->ndim > ANIRA_MAX_RANK) { return 0; }
    size_t total = 1;
    for (uint32_t axis = 0; axis < tensor->ndim; ++axis) {
        size_t extent = 0;
        if (!to_count(tensor->shape[axis], extent)) { return 0; }
        if (extent != 0 && total > std::numeric_limits<size_t>::max() / extent) { return 0; }
        total *= extent;
    }
    return total;
}

size_t ANIRA_CALL anira_tensor_extent(const anira_tensor* tensor,
                                      uint32_t axis) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (tensor == nullptr || axis >= tensor->ndim || axis >= ANIRA_MAX_RANK) { return 0; }
    size_t extent = 0;
    return to_count(tensor->shape[axis], extent) ? extent : 0;
}

uint32_t ANIRA_CALL anira_sizeof(anira_struct_id id) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    switch (id) {
#define ANIRA_STRUCT_SIZE(id_name, type) \
    case id_name: return static_cast<uint32_t>(sizeof(type));
#include "generated/struct_sizes.inc"
#undef ANIRA_STRUCT_SIZE
        default: return 0U;
    }
}

// ==== sync tokens ===========================================================================

void ANIRA_CALL anira_sync_token_reset(anira_sync_token* token) ANIRA_NOEXCEPT {
    if (token == nullptr) { return; }
#if defined(_WIN32)
    if (holds_descriptor(*token)) { static_cast<void>(::CloseHandle(to_handle(token->u.fd))); }
#elif !defined(__EMSCRIPTEN__)
    if (holds_descriptor(*token)) { static_cast<void>(::close(token->u.fd)); }
#endif
    std::memset(token, 0, sizeof(*token));
}

anira_status ANIRA_CALL anira_sync_token_dup(const anira_sync_token* token,
                                             anira_sync_token* out) ANIRA_NOEXCEPT {
    if (token == nullptr || out == nullptr || token == out) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira_sync_token copy = *token;
    if (holds_descriptor(copy)) {
#if defined(_WIN32)
        const HANDLE process = ::GetCurrentProcess();
        HANDLE duplicate = nullptr;
        if (::DuplicateHandle(process,
                              to_handle(copy.u.fd),
                              process,
                              &duplicate,
                              0,
                              FALSE,
                              DUPLICATE_SAME_ACCESS) == 0) {
            return ::GetLastError() == ERROR_INVALID_HANDLE ? ANIRA_ERROR_INVALID_ARGUMENT
                                                            : ANIRA_ERROR_OUT_OF_MEMORY;
        }
        copy.u.fd = from_handle(duplicate);
#elif defined(__EMSCRIPTEN__)
        return ANIRA_ERROR_NOT_SUPPORTED;  // a module owns no descriptor; 0, 1 and 2 are its stdio
#else
        // Close-on-exec: the duplicate is kept by its holder, and a plugin host forks helpers.
        const int duplicate = ::fcntl(copy.u.fd, F_DUPFD_CLOEXEC, 0);
        if (duplicate < 0) {
            return errno == EBADF ? ANIRA_ERROR_INVALID_ARGUMENT : ANIRA_ERROR_OUT_OF_MEMORY;
        }
        copy.u.fd = duplicate;
#endif
    }
    *out = copy;
    return ANIRA_OK;
}
