// anira/abi/tensor.h and anira/abi/draft/tensor_platform.h: the zero-then-fill factories, the
// planar host tensor, the NULL rules of the accessors, anira_sizeof, the descriptor ownership of
// the sync token and the DLPack bridge. The DLPack cases build their managed tensor from the tests'
// own spelling of dlpack.h (dlpack_producer.h): the mirror inside src/capi/tensor.cpp is private,
// and a second, independent spelling is the only guard it has.
#include <anira/abi/context.h>
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

#include "dlpack_producer.h"

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
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace {

using anira_test::device_type_of;
using anira_test::DLDataType;
using anira_test::DLPACK_FLAG_BITMASK_IS_COPIED;
using anira_test::DLPACK_FLAG_BITMASK_READ_ONLY;
using anira_test::DlpackProducer;
using anira_test::kDLBfloat;
using anira_test::kDLBool;
using anira_test::kDLComplex;
using anira_test::kDLCUDA;
using anira_test::kDLCUDAHost;
using anira_test::kDLFloat;
using anira_test::kDLFloat4_e2m1fn;
using anira_test::kDLFloat8_e3m4;
using anira_test::kDLInt;
using anira_test::kDLROCMHost;
using anira_test::kDLUInt;

// ---- helpers ---------------------------------------------------------------------------------

/// An enum constant as the uint32_t a record field carries.
template <class Enum>
constexpr uint32_t u32(Enum value) {
    return static_cast<uint32_t>(value);
}

/// A pointer as the 64-bit word of its ANIRA_PTR slot: on a 32-bit target the high half is 0.
uint64_t slot_bits(const void* pointer) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
}

/// The pointer array of a planar tensor as the C entry takes it. The conversion is implicit in
/// C and in C++; it is spelled here because clang-tidy's
/// bugprone-multi-level-implicit-pointer-conversion asks for it in this repository's C++.
template <class Planes>
const void* planes_of(Planes& planes) {
    return static_cast<const void*>(planes.data());
}

/// 0xAB in every byte: what a factory has to overwrite and what a refused DLPack call leaves.
void poison(anira_tensor& tensor) {
    std::memset(&tensor, 0xAB, sizeof(tensor));
}

/// The object representation of a record. The records hold unions, so tidy refuses a plain
/// memory compare over them (bugprone-suspicious-memory-comparison); byte for byte is
/// nevertheless the contract of zero-then-fill, and an array of unsigned char compares that.
template <class Record>
std::array<unsigned char, sizeof(Record)> bytes_of(const Record& record) {
    std::array<unsigned char, sizeof(Record)> bytes{};
    std::memcpy(bytes.data(), &record, sizeof(Record));
    return bytes;
}

template <class Record>
bool same_bytes(const Record& left, const Record& right) {
    return bytes_of(left) == bytes_of(right);
}

template <class Record>
bool all_zero(const Record& record) {
    return bytes_of(record) == std::array<unsigned char, sizeof(Record)>{};
}

/// The record every factory starts from, written by hand: all-zero, then the descriptor. The
/// case adds the arm and the fence and compares byte for byte, which covers the padding, the
/// unused tail of the handle union and the high half of every pointer slot.
void fill_expected(anira_tensor& expected,
                   anira_domain domain,
                   anira_dtype dtype,
                   std::span<const int64_t> shape) {
    std::memset(&expected, 0, sizeof(expected));
    expected.domain = u32(domain);
    expected.dtype = dtype;
    expected.ndim = static_cast<uint32_t>(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) { expected.shape[axis] = shape[axis]; }
}

/// Every field of the descriptor a factory fills, and everything it leaves zero.
void expect_descriptor(const anira_tensor& tensor,
                       anira_domain domain,
                       anira_dtype dtype,
                       std::span<const int64_t> shape) {
    EXPECT_EQ(tensor.domain, u32(domain));
    EXPECT_EQ(tensor.dtype, dtype);
    EXPECT_EQ(tensor.ndim, shape.size());
    EXPECT_EQ(tensor.flags, 0U);
    for (size_t axis = 0; axis < ANIRA_MAX_RANK; ++axis) {
        EXPECT_EQ(tensor.shape[axis], axis < shape.size() ? shape[axis] : 0) << "axis " << axis;
        EXPECT_EQ(tensor.strides[axis], 0) << "packed row-major, axis " << axis;
    }
    EXPECT_EQ(tensor.byte_offset, 0U);
    EXPECT_EQ(tensor.manager_ctx, nullptr);
    EXPECT_EQ(tensor.manager_ctx_bits, 0U);
    EXPECT_EQ(tensor.release, nullptr) << "NULL = borrowed";
    EXPECT_EQ(tensor.release_bits, 0U);
}

void expect_no_fence(const anira_tensor& tensor) {
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_NONE));
    EXPECT_EQ(tensor.acquire.flags, 0U);
    EXPECT_EQ(tensor.acquire.u.raw[0], 0U);
    EXPECT_EQ(tensor.acquire.u.raw[1], 0U);
}

constexpr std::array<int64_t, 2> k_shape{2, 3};
constexpr anira_dtype k_u16 = ANIRA_MAKE_DTYPE(ANIRA_DTYPE_UINT, 16, 1);

// One call per factory with plain arguments and no fence, for the rules all twelve share.
using Fill = void (*)(anira_tensor*, void*, anira_dtype, uint32_t, const int64_t*);

struct Factory {
    const char* m_name;
    anira_domain m_domain;
    Fill m_fill;
};

const std::array<Factory, 12> k_factories{{
    {.m_name = "init_host",
     .m_domain = ANIRA_DOMAIN_HOST,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_host(t, p, d, n, s);
         }},
    {.m_name = "init_pinned",
     .m_domain = ANIRA_DOMAIN_HOST_PINNED,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_pinned(t, p, d, n, s);
         }},
    {.m_name = "init_cuda",
     .m_domain = ANIRA_DOMAIN_CUDA,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_cuda(t, p, 0, nullptr, d, n, s);
         }},
    {.m_name = "init_gl_buffer",
     .m_domain = ANIRA_DOMAIN_GL_BUFFER,
     .m_fill =
         [](anira_tensor* t, void* /*p*/, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_gl_buffer(t, 7, 0x90D2, nullptr, d, n, s);
         }},
    {.m_name = "init_vulkan",
     .m_domain = ANIRA_DOMAIN_VULKAN_BUFFER,
     .m_fill =
         [](anira_tensor* t, void* /*p*/, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_vulkan(t, 1, 2, 3, 0, 0, d, n, s);
         }},
    {.m_name = "init_opaque_fd",
     .m_domain = ANIRA_DOMAIN_OPAQUE_FD,
     .m_fill =
         [](anira_tensor* t, void* /*p*/, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_opaque_fd(t, 5, 4096, d, n, s);
         }},
    {.m_name = "init_wgpu_buffer",
     .m_domain = ANIRA_DOMAIN_WGPU_BUFFER,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_wgpu_buffer(t, p, 0, nullptr, d, n, s);
         }},
    {.m_name = "init_dmabuf",
     .m_domain = ANIRA_DOMAIN_DMABUF,
     .m_fill =
         [](anira_tensor* t, void* /*p*/, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_dmabuf(t, 5, 4096, 0, -1, d, n, s);
         }},
    {.m_name = "init_metal",
     .m_domain = ANIRA_DOMAIN_METAL_BUFFER,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_metal(t, p, nullptr, d, n, s);
         }},
    {.m_name = "init_iosurface",
     .m_domain = ANIRA_DOMAIN_IOSURFACE,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_iosurface(t, p, 4096, nullptr, d, n, s);
         }},
    {.m_name = "init_ahardwarebuffer",
     .m_domain = ANIRA_DOMAIN_AHARDWAREBUFFER,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_ahardwarebuffer(t, p, -1, d, n, s);
         }},
    {.m_name = "init_d3d12",
     .m_domain = ANIRA_DOMAIN_D3D12,
     .m_fill =
         [](anira_tensor* t, void* p, anira_dtype d, uint32_t n, const int64_t* s) {
             anira_tensor_init_d3d12(t, p, nullptr, nullptr, d, n, s);
         }},
}};

/// A refused anira_tensor_init_dlpack: the status, a message, *tensor untouched, no deleter call.
void expect_refused(DlpackProducer& producer, anira_status expected, const char* fragment) {
    anira_tensor tensor;
    anira_tensor before;
    poison(tensor);
    poison(before);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, &err), expected);
    EXPECT_EQ(err.status, expected);
    EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
    EXPECT_TRUE(same_bytes(tensor, before)) << "a refused call leaves *tensor untouched";
    EXPECT_EQ(producer.m_deleted, 0) << "anira never calls the deleter of a tensor it refused";
    EXPECT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), expected)
        << "err is nullable";
    EXPECT_TRUE(same_bytes(tensor, before));
    EXPECT_EQ(producer.m_deleted, 0);
}

#if defined(_WIN32)
/// The Windows twin of the POSIX fixture below: an anonymous pipe whose write HANDLE a token
/// owns (an NT handle has 32 significant bits, so it sits in int32_t fd). PeekNamedPipe on the
/// read end succeeds while any write handle is open and fails with ERROR_BROKEN_PIPE once the
/// last one is closed: it observes CloseHandle and DuplicateHandle without asking whether a
/// handle VALUE is open.
struct Pipe {
    Pipe() { EXPECT_NE(::CreatePipe(&m_read, &m_write, nullptr, 0), 0); }
    ~Pipe() {
        static_cast<void>(::CloseHandle(m_read));
        if (m_write != nullptr) { static_cast<void>(::CloseHandle(m_write)); }
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    /// The write end leaves the fixture: a token owns it from here on.
    int32_t release_write_end() {
        const auto fd = static_cast<int32_t>(reinterpret_cast<intptr_t>(m_write));
        m_write = nullptr;
        return fd;
    }

    bool writer_open() const {
        DWORD available = 0;
        return ::PeekNamedPipe(m_read, nullptr, 0, nullptr, &available, nullptr) != 0;
    }

    HANDLE m_read = nullptr;
    HANDLE m_write = nullptr;
};
#elif !defined(__EMSCRIPTEN__)
/// A pipe whose write end a token owns. The read end is non-blocking and tells whether any
/// write end is still open: EAGAIN while one is, end-of-file (0) once the last one is closed.
/// That observes close() and dup() without asking whether an fd NUMBER is open, which another
/// thread of the process could reuse at any moment.
struct Pipe {
    Pipe() {
        std::array<int, 2> ends{-1, -1};
        EXPECT_EQ(::pipe(ends.data()), 0);
        m_read = ends[0];
        m_write = ends[1];
        EXPECT_EQ(::fcntl(m_read, F_SETFL, O_NONBLOCK), 0);
    }
    ~Pipe() {
        static_cast<void>(::close(m_read));
        if (m_write >= 0) { static_cast<void>(::close(m_write)); }
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    /// The write end leaves the fixture: a token owns it from here on.
    int32_t release_write_end() {
        const int fd = m_write;
        m_write = -1;
        return fd;
    }

    bool writer_open() const {
        char byte = 0;
        const ssize_t got = ::read(m_read, &byte, 1);
        return got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
    }

    int m_read = -1;
    int m_write = -1;
};
#endif

}  // namespace

// ---- factories ---------------------------------------------------------------------------------

TEST(AbiTensor, InitHostAndPinnedZeroThenFill) {
    std::array<float, 6> data{};
    for (const bool pinned : {false, true}) {
        SCOPED_TRACE(pinned ? "init_pinned" : "init_host");
        const anira_domain domain = pinned ? ANIRA_DOMAIN_HOST_PINNED : ANIRA_DOMAIN_HOST;
        anira_tensor tensor;
        poison(tensor);
        if (pinned) {
            anira_tensor_init_pinned(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
        } else {
            anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
        }
        expect_descriptor(tensor, domain, ANIRA_DTYPE_F32, k_shape);
        expect_no_fence(tensor);
        EXPECT_EQ(tensor.handle.host.ptr, data.data());
        EXPECT_EQ(tensor.handle.host.ptr_bits, slot_bits(data.data()));
        anira_tensor expected;
        fill_expected(expected, domain, ANIRA_DTYPE_F32, k_shape);
        expected.handle.host.ptr = data.data();
        EXPECT_TRUE(same_bytes(tensor, expected)) << "zero, then fill: nothing else is set";
    }
}

TEST(AbiTensor, InitCudaFillsTheArmAndANonOwningEvent) {
    std::array<float, 6> data{};
    int event = 0;
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_cuda(&tensor, data.data(), 3, &event, ANIRA_DTYPE_F16, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_CUDA, ANIRA_DTYPE_F16, k_shape);
    EXPECT_EQ(tensor.handle.cuda.ptr, data.data());
    EXPECT_EQ(tensor.handle.cuda.ptr_bits, slot_bits(data.data()));
    EXPECT_EQ(tensor.handle.cuda.device, 3);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_CUDA_EVENT));
    EXPECT_EQ(tensor.acquire.flags, 0U);
    EXPECT_EQ(tensor.acquire.u.cuda_event, &event);
    EXPECT_EQ(tensor.acquire.u.cuda_event_bits, slot_bits(&event));
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_CUDA, ANIRA_DTYPE_F16, k_shape);
    expected.handle.cuda.ptr = data.data();
    expected.handle.cuda.device = 3;
    expected.acquire.kind = u32(ANIRA_SYNC_CUDA_EVENT);
    expected.acquire.u.cuda_event = &event;
    EXPECT_TRUE(same_bytes(tensor, expected));

    poison(tensor);
    anira_tensor_init_cuda(&tensor, data.data(), 3, nullptr, ANIRA_DTYPE_F16, 2, k_shape.data());
    expect_no_fence(tensor);
}

TEST(AbiTensor, InitGlBufferFillsTheArmAndANonOwningSync) {
    int sync = 0;
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_gl_buffer(&tensor, 7, 0x90D2, &sync, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_GL_BUFFER, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.gl.id, 7U);
    EXPECT_EQ(tensor.handle.gl.target, 0x90D2U);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_GL_SYNC));
    EXPECT_EQ(tensor.acquire.u.gl_sync, &sync);
    EXPECT_EQ(tensor.acquire.u.gl_sync_bits, slot_bits(&sync));
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_GL_BUFFER, ANIRA_DTYPE_F32, k_shape);
    expected.handle.gl.id = 7;
    expected.handle.gl.target = 0x90D2;
    expected.acquire.kind = u32(ANIRA_SYNC_GL_SYNC);
    expected.acquire.u.gl_sync = &sync;
    EXPECT_TRUE(same_bytes(tensor, expected));

    poison(tensor);
    anira_tensor_init_gl_buffer(&tensor, 7, 0x90D2, nullptr, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_no_fence(tensor);
}

TEST(AbiTensor, InitVulkanFillsTheArmAndTheTimeline) {
    constexpr uint64_t k_buffer = 0x1111222233334444ULL;
    constexpr uint64_t k_memory = 0x5555666677778888ULL;
    constexpr uint64_t k_semaphore = 0x9999AAAABBBBCCCCULL;
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_vulkan(&tensor,
                             k_buffer,
                             k_memory,
                             64,
                             k_semaphore,
                             42,
                             ANIRA_DTYPE_F32,
                             2,
                             k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_VULKAN_BUFFER, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.vk.buffer, k_buffer);
    EXPECT_EQ(tensor.handle.vk.memory, k_memory);
    EXPECT_EQ(tensor.handle.vk.offset, 64U);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_VK_TIMELINE));
    EXPECT_EQ(tensor.acquire.u.vk.semaphore, k_semaphore);
    EXPECT_EQ(tensor.acquire.u.vk.value, 42U);
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_VULKAN_BUFFER, ANIRA_DTYPE_F32, k_shape);
    expected.handle.vk.buffer = k_buffer;
    expected.handle.vk.memory = k_memory;
    expected.handle.vk.offset = 64;
    expected.acquire.kind = u32(ANIRA_SYNC_VK_TIMELINE);
    expected.acquire.u.vk.semaphore = k_semaphore;
    expected.acquire.u.vk.value = 42;
    EXPECT_TRUE(same_bytes(tensor, expected));

    poison(tensor);
    anira_tensor_init_vulkan(&tensor,
                             k_buffer,
                             k_memory,
                             64,
                             0,
                             42,
                             ANIRA_DTYPE_F32,
                             2,
                             k_shape.data());
    expect_no_fence(tensor);  // VK_NULL_HANDLE: the value is ignored, no fence is fabricated
}

TEST(AbiTensor, InitOpaqueFdFillsTheArmAndTakesNoFence) {
    constexpr uint64_t k_size = 0x100000000ULL;  // above 4 GiB: the size travels at 64 bits
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_opaque_fd(&tensor, 5, k_size, ANIRA_DTYPE_U8, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_OPAQUE_FD, ANIRA_DTYPE_U8, k_shape);
    expect_no_fence(tensor);
    EXPECT_EQ(tensor.handle.opaque.fd, 5);
    EXPECT_EQ(tensor.handle.opaque.reserved, 0U);
    EXPECT_EQ(tensor.handle.opaque.size, k_size);
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_OPAQUE_FD, ANIRA_DTYPE_U8, k_shape);
    expected.handle.opaque.fd = 5;
    expected.handle.opaque.size = k_size;
    EXPECT_TRUE(same_bytes(tensor, expected));
}

TEST(AbiTensor, InitWgpuBufferCopiesTheTokenVerbatim) {
    int buffer = 0;
    anira_sync_token fence;
    std::memset(&fence, 0, sizeof(fence));
    fence.kind = u32(ANIRA_SYNC_QUEUE_ORDERED);
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_wgpu_buffer(&tensor,
                                  &buffer,
                                  128,
                                  &fence,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_WGPU_BUFFER, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.wgpu.buffer, &buffer);
    EXPECT_EQ(tensor.handle.wgpu.buffer_bits, slot_bits(&buffer));
    EXPECT_EQ(tensor.handle.wgpu.offset, 128U);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_QUEUE_ORDERED));
    EXPECT_TRUE(same_bytes(tensor.acquire, fence)) << "copied verbatim, no kind check";
    EXPECT_EQ(fence.kind, u32(ANIRA_SYNC_QUEUE_ORDERED)) << "the source is not reset";
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_WGPU_BUFFER, ANIRA_DTYPE_F32, k_shape);
    expected.handle.wgpu.buffer = &buffer;
    expected.handle.wgpu.offset = 128;
    expected.acquire = fence;
    EXPECT_TRUE(same_bytes(tensor, expected));

    poison(tensor);
    anira_tensor_init_wgpu_buffer(&tensor,
                                  &buffer,
                                  128,
                                  nullptr,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  k_shape.data());
    expect_no_fence(tensor);  // anira never fabricates ANIRA_SYNC_QUEUE_ORDERED
}

TEST(AbiTensor, InitDmabufFillsTheArmAndOwnsTheSyncFile) {
    anira_tensor tensor;
    poison(tensor);
    // 9 is never opened, closed or duplicated here: the fill stores the number, nothing else.
    anira_tensor_init_dmabuf(&tensor, 5, 4096, 512, 9, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_DMABUF, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.dmabuf.fd, 5);
    EXPECT_EQ(tensor.handle.dmabuf.reserved, 0U);
    EXPECT_EQ(tensor.handle.dmabuf.size, 4096U);
    EXPECT_EQ(tensor.handle.dmabuf.offset, 512U);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_SYNC_FILE_FD));
    EXPECT_EQ(tensor.acquire.u.fd, 9);
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_DMABUF, ANIRA_DTYPE_F32, k_shape);
    expected.handle.dmabuf.fd = 5;
    expected.handle.dmabuf.size = 4096;
    expected.handle.dmabuf.offset = 512;
    expected.acquire.kind = u32(ANIRA_SYNC_SYNC_FILE_FD);
    expected.acquire.u.fd = 9;
    EXPECT_TRUE(same_bytes(tensor, expected));

    poison(tensor);
    anira_tensor_init_dmabuf(&tensor, 5, 4096, 512, 0, ANIRA_DTYPE_F32, 2, k_shape.data());
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_SYNC_FILE_FD)) << "fd 0 is an fd";
    EXPECT_EQ(tensor.acquire.u.fd, 0);

    poison(tensor);
    anira_tensor_init_dmabuf(&tensor, 5, 4096, 512, -1, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_no_fence(tensor);
}

TEST(AbiTensor, DraftInitMetalAndIosurface) {
    int object = 0;
    int event = 0;
    anira_sync_token shared_event;
    std::memset(&shared_event, 0, sizeof(shared_event));
    shared_event.kind = u32(ANIRA_SYNC_MTL_SHARED_EVENT);
    shared_event.u.mtl.object = &event;
    shared_event.u.mtl.value = 17;

    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_metal(&tensor, &object, &shared_event, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_METAL_BUFFER, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.mtl.buffer, &object);
    EXPECT_EQ(tensor.handle.mtl.buffer_bits, slot_bits(&object));
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_MTL_SHARED_EVENT));
    EXPECT_EQ(tensor.acquire.u.mtl.object, &event);
    EXPECT_EQ(tensor.acquire.u.mtl.value, 17U);
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_METAL_BUFFER, ANIRA_DTYPE_F32, k_shape);
    expected.handle.mtl.buffer = &object;
    expected.acquire = shared_event;
    EXPECT_TRUE(same_bytes(tensor, expected));
    poison(tensor);
    anira_tensor_init_metal(&tensor, &object, nullptr, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_no_fence(tensor);

    constexpr uint64_t k_size = 0x100000000ULL;  // the sixth allowlisted 64-bit parameter
    poison(tensor);
    anira_tensor_init_iosurface(&tensor,
                                &object,
                                k_size,
                                &shared_event,
                                ANIRA_DTYPE_F32,
                                2,
                                k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_IOSURFACE, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.iosurface.surface, &object);
    EXPECT_EQ(tensor.handle.iosurface.surface_bits, slot_bits(&object));
    EXPECT_EQ(tensor.handle.iosurface.size, k_size);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_MTL_SHARED_EVENT));
    fill_expected(expected, ANIRA_DOMAIN_IOSURFACE, ANIRA_DTYPE_F32, k_shape);
    expected.handle.iosurface.surface = &object;
    expected.handle.iosurface.size = k_size;
    expected.acquire = shared_event;
    EXPECT_TRUE(same_bytes(tensor, expected));
    poison(tensor);
    anira_tensor_init_iosurface(&tensor,
                                &object,
                                k_size,
                                nullptr,
                                ANIRA_DTYPE_F32,
                                2,
                                k_shape.data());
    expect_no_fence(tensor);
}

TEST(AbiTensor, DraftInitAhardwarebufferAndD3d12) {
    int object = 0;
    int handle = 0;
    int fence_object = 0;
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_ahardwarebuffer(&tensor, &object, 9, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_AHARDWAREBUFFER, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.ahb.buffer, &object);
    EXPECT_EQ(tensor.handle.ahb.buffer_bits, slot_bits(&object));
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_SYNC_FILE_FD)) << "the sync_fd rule of dmabuf";
    EXPECT_EQ(tensor.acquire.u.fd, 9);
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_AHARDWAREBUFFER, ANIRA_DTYPE_F32, k_shape);
    expected.handle.ahb.buffer = &object;
    expected.acquire.kind = u32(ANIRA_SYNC_SYNC_FILE_FD);
    expected.acquire.u.fd = 9;
    EXPECT_TRUE(same_bytes(tensor, expected));
    poison(tensor);
    anira_tensor_init_ahardwarebuffer(&tensor, &object, -1, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_no_fence(tensor);

    anira_sync_token fence;
    std::memset(&fence, 0, sizeof(fence));
    fence.kind = u32(ANIRA_SYNC_D3D12_FENCE);
    fence.u.d3d12.object = &fence_object;
    fence.u.d3d12.value = 23;
    poison(tensor);
    anira_tensor_init_d3d12(&tensor, &object, &handle, &fence, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_D3D12, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.d3d12.resource, &object);
    EXPECT_EQ(tensor.handle.d3d12.resource_bits, slot_bits(&object));
    EXPECT_EQ(tensor.handle.d3d12.shared_handle, &handle);
    EXPECT_EQ(tensor.handle.d3d12.shared_handle_bits, slot_bits(&handle));
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_D3D12_FENCE));
    EXPECT_EQ(tensor.acquire.u.d3d12.object, &fence_object);
    EXPECT_EQ(tensor.acquire.u.d3d12.value, 23U);
    fill_expected(expected, ANIRA_DOMAIN_D3D12, ANIRA_DTYPE_F32, k_shape);
    expected.handle.d3d12.resource = &object;
    expected.handle.d3d12.shared_handle = &handle;
    expected.acquire = fence;
    EXPECT_TRUE(same_bytes(tensor, expected));
    poison(tensor);
    anira_tensor_init_d3d12(&tensor, &object, nullptr, nullptr, ANIRA_DTYPE_F32, 2, k_shape.data());
    expect_no_fence(tensor);
    EXPECT_EQ(tensor.handle.d3d12.shared_handle, nullptr);
}

TEST(AbiTensor, EveryFactoryNamesItsDomainAndKeepsRankZeroAndRankEight) {
    std::array<float, 6> data{};
    constexpr std::array<int64_t, ANIRA_MAX_RANK> k_full{1, 2, 1, 3, 1, 1, 1, 1};
    for (const Factory& factory : k_factories) {
        SCOPED_TRACE(factory.m_name);
        anira_tensor tensor;
        poison(tensor);
        factory.m_fill(&tensor, data.data(), ANIRA_DTYPE_F32, ANIRA_MAX_RANK, k_full.data());
        expect_descriptor(tensor, factory.m_domain, ANIRA_DTYPE_F32, k_full);
        expect_no_fence(tensor);
        EXPECT_EQ(anira_tensor_num_elements(&tensor), 6U);

        poison(tensor);
        factory.m_fill(&tensor, data.data(), k_u16, 0, nullptr);  // rank 0: shape may be NULL
        expect_descriptor(tensor, factory.m_domain, k_u16, {});
        EXPECT_EQ(anira_tensor_num_elements(&tensor), 1U) << "rank 0 is one element";
    }
}

TEST(AbiTensor, EveryFactoryRefusesByLeavingTheRecordAllZero) {
    std::array<float, 6> data{};
    constexpr std::array<int64_t, 9> k_nine{1, 1, 1, 1, 1, 1, 1, 1, 1};
    constexpr std::array<int64_t, 2> k_negative{4, -1};
    for (const Factory& factory : k_factories) {
        SCOPED_TRACE(factory.m_name);
        anira_tensor tensor;
        poison(tensor);
        factory.m_fill(&tensor, data.data(), ANIRA_DTYPE_F32, ANIRA_MAX_RANK + 1, k_nine.data());
        EXPECT_TRUE(all_zero(tensor)) << "a rank above ANIRA_MAX_RANK";
        poison(tensor);
        factory.m_fill(&tensor, data.data(), ANIRA_DTYPE_F32, 2, nullptr);
        EXPECT_TRUE(all_zero(tensor)) << "a NULL shape with a rank above 0";
        poison(tensor);
        factory.m_fill(&tensor, data.data(), 0, 2, k_shape.data());
        EXPECT_TRUE(all_zero(tensor)) << "dtype 0 is not a type";
        poison(tensor);
        factory.m_fill(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_negative.data());
        EXPECT_TRUE(all_zero(tensor)) << "a negative extent (never ANIRA_DYNAMIC at run time)";
        factory.m_fill(nullptr, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());  // a no-op
    }
    // What the all-zero record reads as: dtype 0, rank 0, no elements, no data.
    anira_tensor refused;
    poison(refused);
    anira_tensor_init_host(&refused, data.data(), ANIRA_DTYPE_F32, 9, k_nine.data());
    EXPECT_EQ(refused.dtype, 0U);
    EXPECT_EQ(anira_tensor_num_elements(&refused), 0U) << "dtype 0 is never a filled record";
    EXPECT_EQ(anira_tensor_data_f32(&refused), nullptr);
    EXPECT_EQ(anira_tensor_data(&refused, 0), nullptr) << "0 equals the record's dtype, still NULL";
}

TEST(AbiTensor, AFactoryReadsItsArgumentsBeforeItZeroesTheRecord) {
    std::array<float, 6> first{};
    std::array<float, 6> second{};
    anira_tensor tensor;
    anira_tensor_init_host(&tensor, first.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
    // Re-point the record at new memory with its own dtype, rank and shape: the shape argument
    // points into the record the factory is about to zero.
    anira_tensor_init_host(&tensor, second.data(), tensor.dtype, tensor.ndim, tensor.shape);
    expect_descriptor(tensor, ANIRA_DOMAIN_HOST, ANIRA_DTYPE_F32, k_shape);
    EXPECT_EQ(tensor.handle.host.ptr, second.data());
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 6U);

    // The same for a token handed by pointer: the record's own acquire is a legal fence.
    int event = 0;
    int buffer = 0;
    tensor.acquire.kind = u32(ANIRA_SYNC_CUDA_EVENT);
    tensor.acquire.u.cuda_event = &event;
    anira_tensor_init_wgpu_buffer(&tensor,
                                  &buffer,
                                  0,
                                  &tensor.acquire,
                                  tensor.dtype,
                                  tensor.ndim,
                                  tensor.shape);
    EXPECT_EQ(tensor.domain, u32(ANIRA_DOMAIN_WGPU_BUFFER));
    EXPECT_EQ(tensor.shape[1], 3);
    EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_CUDA_EVENT)) << "read before the zeroing";
    EXPECT_EQ(tensor.acquire.u.cuda_event, &event);

    // A refused call does not take the token: the record is all-zero, the source untouched.
    anira_sync_token fence;
    std::memset(&fence, 0, sizeof(fence));
    fence.kind = u32(ANIRA_SYNC_QUEUE_ORDERED);
    anira_tensor_init_wgpu_buffer(&tensor, &buffer, 0, &fence, 0, 2, k_shape.data());
    EXPECT_TRUE(all_zero(tensor));
    EXPECT_EQ(fence.kind, u32(ANIRA_SYNC_QUEUE_ORDERED));
}

TEST(AbiTensor, InitHostAcceptsNullDataOverAnEmptyBlock) {
    constexpr std::array<int64_t, 2> k_empty{2, 0};
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_host(&tensor, nullptr, ANIRA_DTYPE_F32, 2, k_empty.data());
    expect_descriptor(tensor, ANIRA_DOMAIN_HOST, ANIRA_DTYPE_F32, k_empty);
    EXPECT_EQ(tensor.handle.host.ptr, nullptr) << "a field fill: the memory is never looked at";
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U);
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr);
}

// ---- planar host tensors -----------------------------------------------------------------------

TEST(AbiTensor, InitHostPlanarFillsThePlanesArmAndSetsTheFlag) {
    std::array<float, 3> left{};
    std::array<float, 3> right{};
    std::array<float*, 2> channels{left.data(), right.data()};
    anira_tensor tensor;
    poison(tensor);
    // A float** converts to the const void* parameter with no cast, in C++ as in C (header_c.c
    // proves the C half); this one call leaves the conversion implicit to show it.
    // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
    anira_tensor_init_host_planar(&tensor, channels.data(), 2, ANIRA_DTYPE_F32, 2, k_shape.data());
    EXPECT_EQ(tensor.domain, u32(ANIRA_DOMAIN_HOST));
    EXPECT_EQ(tensor.dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(tensor.ndim, 2U);
    EXPECT_EQ(tensor.flags, u32(ANIRA_TENSOR_PLANAR));
    EXPECT_EQ(tensor.handle.planes.ptrs, static_cast<void*>(channels.data()));
    EXPECT_EQ(tensor.handle.planes.ptrs_bits, slot_bits(planes_of(channels)));
    EXPECT_EQ(tensor.handle.planes.count, 2U);
    EXPECT_EQ(tensor.handle.planes.reserved, 0U);
    expect_no_fence(tensor);
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_HOST, ANIRA_DTYPE_F32, k_shape);
    expected.flags = u32(ANIRA_TENSOR_PLANAR);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) float** read as void* const*
    expected.handle.planes.ptrs = reinterpret_cast<void* const*>(channels.data());
    expected.handle.planes.count = 2;
    EXPECT_TRUE(same_bytes(tensor, expected)) << "zero, then fill: nothing else is set";

    // Read-only planes convert as well; the caller ORs the flag in afterwards.
    const std::array<const float*, 2> read{left.data(), right.data()};
    anira_tensor_init_host_planar(&tensor, planes_of(read), 2, ANIRA_DTYPE_F32, 2, k_shape.data());
    tensor.flags |= u32(ANIRA_TENSOR_READ_ONLY);
    EXPECT_EQ(anira_tensor_plane(&tensor, 1, ANIRA_DTYPE_F32), right.data());
}

TEST(AbiTensor, InitHostPlanarRefusesByLeavingTheRecordAllZero) {
    std::array<float, 3> left{};
    std::array<float, 3> right{};
    std::array<float*, 2> channels{left.data(), right.data()};
    constexpr std::array<int64_t, 9> k_nine{2, 1, 1, 1, 1, 1, 1, 1, 1};
    constexpr std::array<int64_t, 2> k_negative{2, -1};
    anira_tensor tensor;
    poison(tensor);
    anira_tensor_init_host_planar(&tensor,
                                  planes_of(channels),
                                  3,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  k_shape.data());
    EXPECT_TRUE(all_zero(tensor)) << "count must equal shape[0]: more pointers than planes";
    poison(tensor);
    anira_tensor_init_host_planar(&tensor,
                                  planes_of(channels),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  k_shape.data());
    EXPECT_TRUE(all_zero(tensor)) << "count must equal shape[0]: fewer pointers than planes";
    poison(tensor);
    anira_tensor_init_host_planar(&tensor, planes_of(channels), 0, ANIRA_DTYPE_F32, 0, nullptr);
    EXPECT_TRUE(all_zero(tensor)) << "rank 0 has no plane axis";
    poison(tensor);
    anira_tensor_init_host_planar(&tensor, planes_of(channels), 2, 0, 2, k_shape.data());
    EXPECT_TRUE(all_zero(tensor)) << "dtype 0 is not a type";
    poison(tensor);
    anira_tensor_init_host_planar(&tensor,
                                  planes_of(channels),
                                  2,
                                  ANIRA_DTYPE_F32,
                                  9,
                                  k_nine.data());
    EXPECT_TRUE(all_zero(tensor)) << "a rank above ANIRA_MAX_RANK";
    poison(tensor);
    anira_tensor_init_host_planar(&tensor, planes_of(channels), 2, ANIRA_DTYPE_F32, 2, nullptr);
    EXPECT_TRUE(all_zero(tensor)) << "a NULL shape";
    poison(tensor);
    anira_tensor_init_host_planar(&tensor,
                                  planes_of(channels),
                                  2,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  k_negative.data());
    EXPECT_TRUE(all_zero(tensor)) << "a negative extent";
    anira_tensor_init_host_planar(nullptr,
                                  planes_of(channels),
                                  2,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  k_shape.data());

    // The empty tensor: NULL planes over a block of no samples is a filled record.
    constexpr std::array<int64_t, 2> k_empty{2, 0};
    poison(tensor);
    anira_tensor_init_host_planar(&tensor, nullptr, 2, ANIRA_DTYPE_F32, 2, k_empty.data());
    EXPECT_EQ(tensor.dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(tensor.flags, u32(ANIRA_TENSOR_PLANAR));
    EXPECT_EQ(tensor.handle.planes.ptrs, nullptr);
    EXPECT_EQ(tensor.handle.planes.count, 2U);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U);
    EXPECT_EQ(anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_F32), nullptr);
}

TEST(AbiTensor, PlaneReadsOnePlaneOfAPlanarHostTensorAndNothingElse) {
    std::array<int16_t, 4> left{};
    std::array<int16_t, 4> right{};
    std::array<int16_t*, 2> channels{left.data(), right.data()};
    constexpr std::array<int64_t, 2> k_block{2, 4};
    anira_tensor tensor;
    anira_tensor_init_host_planar(&tensor,
                                  planes_of(channels),
                                  2,
                                  ANIRA_DTYPE_I16,
                                  2,
                                  k_block.data());
    EXPECT_EQ(anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_I16), left.data());
    EXPECT_EQ(anira_tensor_plane(&tensor, 1, ANIRA_DTYPE_I16), right.data());
    EXPECT_EQ(anira_tensor_plane(&tensor, 2, ANIRA_DTYPE_I16), nullptr) << "at handle.planes.count";
    EXPECT_EQ(anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_F32), nullptr) << "another dtype";
    EXPECT_EQ(anira_tensor_plane(&tensor, 0, 0), nullptr);
    EXPECT_EQ(anira_tensor_plane(nullptr, 0, ANIRA_DTYPE_I16), nullptr);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 8U) << "the shape, planar or not";
    EXPECT_EQ(anira_tensor_extent(&tensor, 1), 4U);
    // A planar tensor is never one block: the two block reads refuse it.
    EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_I16), nullptr);
    anira_tensor floats = tensor;
    floats.dtype = ANIRA_DTYPE_F32;
    EXPECT_EQ(anira_tensor_data_f32(&floats), nullptr);

    tensor.byte_offset = 2;  // bytes, inside each plane
    EXPECT_EQ(anira_tensor_plane(&tensor, 1, ANIRA_DTYPE_I16), &right[1]);
    tensor.byte_offset = 0;
    tensor.domain = u32(ANIRA_DOMAIN_HOST_PINNED);
    EXPECT_EQ(anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_I16), left.data()) << "pinned planes";
    tensor.domain = u32(ANIRA_DOMAIN_CUDA);
    EXPECT_EQ(anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_I16), nullptr) << "host domains only";
    tensor.domain = u32(ANIRA_DOMAIN_HOST);
    channels[1] = nullptr;
    EXPECT_EQ(anira_tensor_plane(&tensor, 1, ANIRA_DTYPE_I16), nullptr) << "a NULL plane";

    // A one-block tensor has no planes.
    anira_tensor block;
    anira_tensor_init_host(&block, left.data(), ANIRA_DTYPE_I16, 1, k_block.data() + 1);
    EXPECT_EQ(anira_tensor_plane(&block, 0, ANIRA_DTYPE_I16), nullptr);
}

// ---- accessors ---------------------------------------------------------------------------------

TEST(AbiTensor, DataF32IsHostFloat32OnlyAndHonoursByteOffset) {
    std::array<float, 6> data{};
    for (const Factory& factory : k_factories) {
        SCOPED_TRACE(factory.m_name);
        anira_tensor tensor;
        factory.m_fill(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
        const bool host =
            factory.m_domain == ANIRA_DOMAIN_HOST || factory.m_domain == ANIRA_DOMAIN_HOST_PINNED;
        EXPECT_EQ(anira_tensor_data_f32(&tensor), host ? data.data() : nullptr);
        EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_F32), host ? data.data() : nullptr);
    }
    anira_tensor tensor;
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
    tensor.byte_offset = 2 * sizeof(float);
    EXPECT_EQ(anira_tensor_data_f32(&tensor), &data[2]) << "byte_offset counts bytes";
    tensor.strides[0] = 1;  // strides are returned as declared, never looked at
    tensor.strides[1] = 2;
    EXPECT_EQ(anira_tensor_data_f32(&tensor), &data[2]);
    anira_tensor_init_host(&tensor, nullptr, ANIRA_DTYPE_F32, 2, k_shape.data());
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr) << "a NULL base pointer";
    tensor.byte_offset = 8;
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr) << "NULL plus an offset is still NULL";
    EXPECT_EQ(anira_tensor_data_f32(nullptr), nullptr);
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
        anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
        tensor.byte_offset = uint64_t{1} << 32U;
        EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr) << "an offset beyond size_t";
    }
}

TEST(AbiTensor, DataReturnsNullUnlessTheDtypeIsTheTensorsOwn) {
    std::array<int16_t, 6> data{};
    anira_tensor tensor;
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_I16, 2, k_shape.data());
    EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_I16), data.data());
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr) << "an int16 tensor is no float tensor";
    EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_F32), nullptr) << "never converts";
    EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_U8), nullptr);
    EXPECT_EQ(anira_tensor_data(&tensor, k_u16), nullptr) << "the same width is not the same dtype";
    EXPECT_EQ(anira_tensor_data(&tensor, 0), nullptr);
    EXPECT_EQ(anira_tensor_data(nullptr, ANIRA_DTYPE_I16), nullptr);
    tensor.byte_offset = 3;  // bytes, not elements
    EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_I16),
              static_cast<void*>(reinterpret_cast<unsigned char*>(data.data()) + 3));
    anira_tensor_init_cuda(&tensor, data.data(), 0, nullptr, ANIRA_DTYPE_I16, 2, k_shape.data());
    EXPECT_EQ(anira_tensor_data(&tensor, ANIRA_DTYPE_I16), nullptr) << "a device domain";
}

TEST(AbiTensor, NumElementsIsTheProductOfTheExtents) {
    std::array<float, 6> data{};
    anira_tensor tensor;
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 6U);
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 0, nullptr);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 1U) << "rank 0: the empty product";
    EXPECT_EQ(anira_tensor_data_f32(&tensor), data.data());
    constexpr std::array<int64_t, 3> k_empty{4, 0, 5};
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 3, k_empty.data());
    EXPECT_EQ(tensor.ndim, 3U) << "a zero extent is a legal shape";
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U);
    EXPECT_EQ(anira_tensor_num_elements(nullptr), 0U);

    // Records no factory produces: written by hand, answered with 0, never read past the shape.
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_shape.data());
    tensor.ndim = ANIRA_MAX_RANK + 1;
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U) << "a rank above ANIRA_MAX_RANK";
    tensor.ndim = 2;
    tensor.shape[1] = -1;
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U) << "a negative extent is not a count";
    tensor.shape[0] = std::numeric_limits<int64_t>::max();
    tensor.shape[1] = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U) << "a product that does not fit size_t";
}

TEST(AbiTensor, ExtentIsZeroAtAndBeyondTheRank) {
    std::array<float, 6> data{};
    anira_tensor tensor;
    anira_tensor_init_cuda(&tensor, data.data(), 0, nullptr, ANIRA_DTYPE_F32, 2, k_shape.data());
    EXPECT_EQ(anira_tensor_extent(&tensor, 0), 2U) << "whatever the domain";
    EXPECT_EQ(anira_tensor_extent(&tensor, 1), 3U);
    EXPECT_EQ(anira_tensor_extent(&tensor, 2), 0U) << "axis == ndim";
    EXPECT_EQ(anira_tensor_extent(&tensor, ANIRA_MAX_RANK), 0U);
    EXPECT_EQ(anira_tensor_extent(&tensor, std::numeric_limits<uint32_t>::max()), 0U);
    EXPECT_EQ(anira_tensor_extent(nullptr, 0), 0U);
    tensor.shape[2] = 7;  // beyond ndim: not an extent of this tensor
    EXPECT_EQ(anira_tensor_extent(&tensor, 2), 0U);
    tensor.shape[1] = -1;
    EXPECT_EQ(anira_tensor_extent(&tensor, 1), 0U) << "a negative extent is not a count";
    tensor.ndim = ANIRA_MAX_RANK + 1;
    EXPECT_EQ(anira_tensor_extent(&tensor, ANIRA_MAX_RANK), 0U) << "never read past shape[]";
    constexpr std::array<int64_t, 2> k_empty{4, 0};
    anira_tensor_init_host(&tensor, data.data(), ANIRA_DTYPE_F32, 2, k_empty.data());
    EXPECT_EQ(anira_tensor_extent(&tensor, 0), 4U);
    EXPECT_EQ(anira_tensor_extent(&tensor, 1), 0U) << "a zero extent";
}

TEST(AbiTensor, SizeofAnswersEveryRegisteredRecord) {
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_TENSOR), sizeof(anira_tensor));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_SYNC_TOKEN), sizeof(anira_sync_token));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_MEMORY_HANDLE), sizeof(anira_memory_handle));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_LOG_RECORD), sizeof(anira_log_record));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_ERROR), sizeof(anira_error));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_EDGE_INFO), sizeof(anira_edge_info));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_PLAN_SLOT), sizeof(anira_plan_slot));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_PLAN_EXT), sizeof(anira_plan_ext));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_PLAN_INFO), sizeof(anira_plan_info));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_BACKEND_ID), sizeof(anira_backend_id));
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_ENGINE_CTX), sizeof(anira_engine_ctx));
    // The Tier-1 answers are the same number on every target.
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_TENSOR), 216U);
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_SYNC_TOKEN), 24U);
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_MEMORY_HANDLE), 24U);
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_STAGE_CTX), 64U);
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_LOG_RECORD), 56U);
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_ERROR), 520U);
    EXPECT_EQ(anira_sizeof(ANIRA_STRUCT_ENGINE_CTX), 64U);
    // Not registered: every id nobody pinned (0, the first free id, the block reserved for
    // extension payloads).
    for (const int unknown : {0, 13, 0x00010000}) {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) an unknown id on purpose
        EXPECT_EQ(anira_sizeof(static_cast<anira_struct_id>(unknown)), 0U) << unknown;
    }
}

// ---- sync tokens -------------------------------------------------------------------------------

TEST(AbiTensorSync, ANonOwningKindIsCopiedByDupAndZeroedByReset) {
    int event = 0;
    anira_sync_token token;
    std::memset(&token, 0, sizeof(token));
    token.kind = u32(ANIRA_SYNC_VK_TIMELINE);
    token.u.vk.semaphore = 0x1111222233334444ULL;
    token.u.vk.value = 42;
    anira_sync_token out;
    std::memset(&out, 0xAB, sizeof(out));
    EXPECT_EQ(anira_sync_token_dup(&token, &out), ANIRA_OK);
    EXPECT_TRUE(same_bytes(out, token)) << "whatever out held is overwritten";

    token.kind = u32(ANIRA_SYNC_CUDA_EVENT);
    token.u.raw[1] = 0;
    token.u.cuda_event = &event;
    EXPECT_EQ(anira_sync_token_dup(&token, &out), ANIRA_OK);
    EXPECT_EQ(out.u.cuda_event, &event);

    // A negative fd under an owning kind owns nothing: a copy, and nothing to close.
    token.kind = u32(ANIRA_SYNC_OPAQUE_FD_SEMAPHORE);
    token.u.raw[0] = 0;
    token.u.fd = -1;
    EXPECT_EQ(anira_sync_token_dup(&token, &out), ANIRA_OK);
    EXPECT_EQ(out.u.fd, -1);

    anira_sync_token_reset(&out);
    EXPECT_TRUE(all_zero(out)) << "kind NONE, flags 0, u all-zero";
    anira_sync_token_reset(&out);  // a second reset of a zeroed token
    EXPECT_EQ(out.kind, u32(ANIRA_SYNC_NONE));
    anira_sync_token_reset(nullptr);  // a no-op
}

TEST(AbiTensorSync, DupRefusesNullAndAliasingAndLeavesOutUntouched) {
    anira_sync_token token;
    std::memset(&token, 0, sizeof(token));
    token.kind = u32(ANIRA_SYNC_QUEUE_ORDERED);
    anira_sync_token out;
    anira_sync_token before;
    std::memset(&out, 0xAB, sizeof(out));
    std::memset(&before, 0xAB, sizeof(before));
    EXPECT_EQ(anira_sync_token_dup(nullptr, &out), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_sync_token_dup(&token, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_sync_token_dup(&token, &token), ANIRA_ERROR_INVALID_ARGUMENT)
        << "an in-place dup would lose the source fd";
    EXPECT_TRUE(same_bytes(out, before));
    EXPECT_EQ(token.kind, u32(ANIRA_SYNC_QUEUE_ORDERED));
}

#if !defined(__EMSCRIPTEN__)

TEST(AbiTensorSync, ResetClosesTheOwnedFdExactlyOnce) {
    for (const anira_sync_kind kind : {ANIRA_SYNC_SYNC_FILE_FD, ANIRA_SYNC_OPAQUE_FD_SEMAPHORE}) {
        Pipe pipe;
        ASSERT_TRUE(pipe.writer_open());
        anira_sync_token token;
        std::memset(&token, 0, sizeof(token));
        token.kind = u32(kind);
        token.u.fd = pipe.release_write_end();
        anira_sync_token_reset(&token);
        EXPECT_FALSE(pipe.writer_open()) << "the token owned the last write end and closed it";
        EXPECT_EQ(token.kind, u32(ANIRA_SYNC_NONE));
        EXPECT_TRUE(all_zero(token))
            << "the record is zeroed; under NONE the payload is never read";
#if !defined(_WIN32)
        // The zeroed token names fd 0 under kind NONE: a second reset must close nothing.
        const int stdin_flags = ::fcntl(0, F_GETFD);
        anira_sync_token_reset(&token);
        EXPECT_EQ(::fcntl(0, F_GETFD), stdin_flags);
#endif
    }
}

TEST(AbiTensorSync, DupDuplicatesTheOwnedFdAndEachTokenClosesItsOwn) {
    Pipe pipe;
    anira_sync_token token;
    std::memset(&token, 0, sizeof(token));
    token.kind = u32(ANIRA_SYNC_SYNC_FILE_FD);
    token.u.fd = pipe.release_write_end();
    anira_sync_token copy;
    std::memset(&copy, 0xAB, sizeof(copy));
    ASSERT_EQ(anira_sync_token_dup(&token, &copy), ANIRA_OK);
    EXPECT_EQ(copy.kind, u32(ANIRA_SYNC_SYNC_FILE_FD));
    EXPECT_EQ(copy.flags, 0U);
    EXPECT_GE(copy.u.fd, 0);
    EXPECT_NE(copy.u.fd, token.u.fd) << "a second descriptor, not a second name for the first";
#if !defined(_WIN32)
    const int descriptor_flags = ::fcntl(copy.u.fd, F_GETFD);
    ASSERT_NE(descriptor_flags, -1);
    EXPECT_NE(descriptor_flags & FD_CLOEXEC, 0) << "the duplicate does not leak into a child";
#endif
    anira_sync_token_reset(&token);
    EXPECT_TRUE(pipe.writer_open()) << "the duplicate outlives its source";
    anira_sync_token_reset(&copy);
    EXPECT_FALSE(pipe.writer_open()) << "and ends with its own reset";
}

TEST(AbiTensorSync, DupOfAnFdThatIsNotOpenIsInvalidArgument) {
    anira_sync_token token;
    std::memset(&token, 0, sizeof(token));
    token.kind = u32(ANIRA_SYNC_SYNC_FILE_FD);
    // Never an open descriptor: EBADF on POSIX, ERROR_INVALID_HANDLE on Windows.
    token.u.fd = std::numeric_limits<int32_t>::max();
    anira_sync_token out;
    anira_sync_token before;
    std::memset(&out, 0xAB, sizeof(out));
    std::memset(&before, 0xAB, sizeof(before));
    EXPECT_EQ(anira_sync_token_dup(&token, &out), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_TRUE(same_bytes(out, before)) << "*out is untouched on failure";
}

TEST(AbiTensorSync, AFactoryHandsTheSyncFileToTheTokenAndARefusedCallDoesNot) {
    constexpr std::array<int64_t, 9> k_nine{1, 1, 1, 1, 1, 1, 1, 1, 1};
    int object = 0;
    for (const bool draft : {false, true}) {
        SCOPED_TRACE(draft ? "init_ahardwarebuffer" : "init_dmabuf");
        Pipe pipe;
        const int32_t fd = pipe.release_write_end();
        anira_tensor tensor;
        // Refused (rank 9): the record is all-zero, so no token owns the fd: still the caller's.
        if (draft) {
            anira_tensor_init_ahardwarebuffer(&tensor,
                                              &object,
                                              fd,
                                              ANIRA_DTYPE_F32,
                                              9,
                                              k_nine.data());
        } else {
            anira_tensor_init_dmabuf(&tensor, 5, 64, 0, fd, ANIRA_DTYPE_F32, 9, k_nine.data());
        }
        EXPECT_TRUE(all_zero(tensor));
        anira_sync_token_reset(&tensor.acquire);
        EXPECT_TRUE(pipe.writer_open()) << "a refused call consumed nothing";
        // Accepted: acquire owns the fd, and the holder of the record ends it with reset.
        if (draft) {
            anira_tensor_init_ahardwarebuffer(&tensor,
                                              &object,
                                              fd,
                                              ANIRA_DTYPE_F32,
                                              2,
                                              k_shape.data());
        } else {
            anira_tensor_init_dmabuf(&tensor, 5, 64, 0, fd, ANIRA_DTYPE_F32, 2, k_shape.data());
        }
        EXPECT_EQ(tensor.acquire.kind, u32(ANIRA_SYNC_SYNC_FILE_FD));
        EXPECT_EQ(tensor.acquire.u.fd, fd);
        EXPECT_TRUE(pipe.writer_open()) << "a field fill closes and duplicates nothing";
        anira_sync_token_reset(&tensor.acquire);
        EXPECT_FALSE(pipe.writer_open());
    }
}

TEST(AbiTensorSync, AnOwningKindWithoutADescriptorOwnsNothing) {
    // Negative on every platform; 0 as well on Windows, where 0 is the NULL handle.
    for (const int32_t none : {-1, std::numeric_limits<int32_t>::min()}) {
        anira_sync_token token;
        std::memset(&token, 0, sizeof(token));
        token.kind = u32(ANIRA_SYNC_OPAQUE_FD_SEMAPHORE);
        token.u.fd = none;
        anira_sync_token out;
        std::memset(&out, 0xAB, sizeof(out));
        EXPECT_EQ(anira_sync_token_dup(&token, &out), ANIRA_OK) << "a plain copy";
        EXPECT_TRUE(same_bytes(out, token));
        anira_sync_token_reset(&out);
        EXPECT_TRUE(all_zero(out));
    }
}

#endif  // not Emscripten

// ---- DLPack ------------------------------------------------------------------------------------

TEST(AbiTensorDlpack, AHostTensorIsBridgedFieldByField) {
    DlpackProducer producer;
    anira_tensor tensor;
    poison(tensor);
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(tensor.domain, u32(ANIRA_DOMAIN_HOST));
    EXPECT_EQ(tensor.dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(tensor.ndim, 2U);
    EXPECT_EQ(tensor.flags, 0U);
    EXPECT_EQ(tensor.shape[0], 2);
    EXPECT_EQ(tensor.shape[1], 3);
    EXPECT_EQ(tensor.strides[0], 3) << "given strides are copied as declared";
    EXPECT_EQ(tensor.strides[1], 1);
    for (size_t axis = 2; axis < ANIRA_MAX_RANK; ++axis) {
        EXPECT_EQ(tensor.shape[axis], 0) << axis;
        EXPECT_EQ(tensor.strides[axis], 0) << axis;
    }
    EXPECT_EQ(tensor.byte_offset, 0U);
    EXPECT_EQ(tensor.handle.host.ptr, producer.m_data.data());
    EXPECT_EQ(tensor.handle.host.ptr_bits, slot_bits(producer.m_data.data()));
    expect_no_fence(tensor);
    EXPECT_EQ(tensor.manager_ctx, &producer.m_managed)
        << "the managed tensor itself, never the producer's own manager_ctx";
    EXPECT_NE(tensor.release, nullptr);
    EXPECT_EQ(producer.m_deleted, 0) << "init consumes on success and calls nothing";
    EXPECT_EQ(anira_tensor_data_f32(&tensor), producer.m_data.data());
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 6U);
    // Zero-then-fill holds for this factory too: nothing but the fields above is set.
    anira_tensor expected;
    fill_expected(expected, ANIRA_DOMAIN_HOST, ANIRA_DTYPE_F32, producer.m_shape);
    expected.strides[0] = 3;
    expected.strides[1] = 1;
    expected.handle.host.ptr = producer.m_data.data();
    expected.manager_ctx = &producer.m_managed;
    expected.release = tensor.release;
    EXPECT_TRUE(same_bytes(tensor, expected));
    tensor.release(&tensor);
}

TEST(AbiTensorDlpack, ReleaseCallsTheDeleterExactlyOnceAndDisarmsItself) {
    DlpackProducer producer;
    anira_tensor tensor;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    anira_tensor_release_proc* const release = tensor.release;
    ASSERT_NE(release, nullptr);
    release(&tensor);
    EXPECT_EQ(producer.m_deleted, 1);
    EXPECT_EQ(producer.m_last, &producer.m_managed) << "the deleter receives the managed tensor";
    EXPECT_EQ(tensor.release, nullptr) << "disarmed before the deleter ran";
    EXPECT_EQ(tensor.release_bits, 0U);
    EXPECT_EQ(tensor.manager_ctx, nullptr);
    EXPECT_EQ(tensor.manager_ctx_bits, 0U);
    EXPECT_EQ(tensor.ndim, 2U) << "the descriptor keeps its own copy of the shape";
    EXPECT_EQ(tensor.shape[1], 3);
    release(&tensor);  // a second call on the same descriptor
    EXPECT_EQ(producer.m_deleted, 1);
    release(nullptr);  // a no-op
    EXPECT_EQ(producer.m_deleted, 1);
}

TEST(AbiTensorDlpack, ANullDeleterIsABorrowedTensor) {
    DlpackProducer producer;
    producer.m_managed.deleter = nullptr;
    anira_tensor tensor;
    poison(tensor);
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.release, nullptr) << "NULL = borrowed";
    EXPECT_EQ(tensor.manager_ctx, nullptr) << "nothing for a release to read";
    EXPECT_EQ(anira_tensor_data_f32(&tensor), producer.m_data.data());
}

TEST(AbiTensorDlpack, PinnedHostMemoryMapsToHostPinned) {
    DlpackProducer producer;
    producer.m_managed.dl_tensor.device = {.device_type = kDLCUDAHost, .device_id = 0};
    anira_tensor tensor;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.domain, u32(ANIRA_DOMAIN_HOST_PINNED));
    EXPECT_EQ(anira_tensor_data_f32(&tensor), producer.m_data.data());
    tensor.release(&tensor);
    EXPECT_EQ(producer.m_deleted, 1);
}

TEST(AbiTensorDlpack, EveryOtherDeviceIsNotSupported) {
    for (const int32_t device : {int32_t{kDLCUDA}, int32_t{kDLROCMHost}, 0, 99}) {
        SCOPED_TRACE(device);
        DlpackProducer producer;
        producer.m_managed.dl_tensor.device.device_type = device_type_of(device);
        expect_refused(producer, ANIRA_ERROR_NOT_SUPPORTED, "dlpack: device type");
    }
}

TEST(AbiTensorDlpack, AnotherMajorVersionIsRefusedBeforeAnythingElseIsRead) {
    for (const uint32_t major : {0U, 2U}) {
        SCOPED_TRACE(major);
        DlpackProducer producer;
        producer.m_managed.version.major = major;
        // Under another major nothing past `flags` has a known layout: were anira to read the
        // tensor, these would be the first things it trips over (and ASan the second).
        producer.m_managed.dl_tensor.ndim = 2;
        producer.m_managed.dl_tensor.shape = nullptr;
        producer.m_managed.dl_tensor.device.device_type = device_type_of(99);
        expect_refused(producer, ANIRA_ERROR_NOT_SUPPORTED, "dlpack: major version");
    }
    DlpackProducer minor;  // a newer minor adds enumerators only: accepted
    minor.m_managed.version.minor = 99;
    anira_tensor tensor;
    EXPECT_EQ(anira_tensor_init_dlpack(&tensor, &minor.m_managed, nullptr), ANIRA_OK);
}

TEST(AbiTensorDlpack, TheDataTypeBytesAreTheAniraDtype) {
    struct Row {
        DLDataType m_dl;
        anira_dtype m_dtype;
    };
    const std::array<Row, 13> rows{{
        {.m_dl = {.code = kDLFloat, .bits = 32, .lanes = 1}, .m_dtype = ANIRA_DTYPE_F32},
        {.m_dl = {.code = kDLFloat, .bits = 64, .lanes = 1}, .m_dtype = ANIRA_DTYPE_F64},
        {.m_dl = {.code = kDLFloat, .bits = 16, .lanes = 1}, .m_dtype = ANIRA_DTYPE_F16},
        {.m_dl = {.code = kDLBfloat, .bits = 16, .lanes = 1}, .m_dtype = ANIRA_DTYPE_BF16},
        {.m_dl = {.code = kDLInt, .bits = 8, .lanes = 1}, .m_dtype = ANIRA_DTYPE_I8},
        {.m_dl = {.code = kDLUInt, .bits = 8, .lanes = 1}, .m_dtype = ANIRA_DTYPE_U8},
        {.m_dl = {.code = kDLInt, .bits = 16, .lanes = 1}, .m_dtype = ANIRA_DTYPE_I16},
        {.m_dl = {.code = kDLInt, .bits = 32, .lanes = 1}, .m_dtype = ANIRA_DTYPE_I32},
        {.m_dl = {.code = kDLInt, .bits = 64, .lanes = 1}, .m_dtype = ANIRA_DTYPE_I64},
        {.m_dl = {.code = kDLBool, .bits = 8, .lanes = 1}, .m_dtype = ANIRA_DTYPE_BOOL8},
        // No named constant, still a dtype: the codes 0..6 pass through with any bits and lanes.
        {.m_dl = {.code = kDLUInt, .bits = 16, .lanes = 1}, .m_dtype = k_u16},
        {.m_dl = {.code = kDLComplex, .bits = 64, .lanes = 1},
         .m_dtype = ANIRA_MAKE_DTYPE(ANIRA_DTYPE_COMPLEX, 64, 1)},
        {.m_dl = {.code = kDLFloat, .bits = 32, .lanes = 4},
         .m_dtype = ANIRA_MAKE_DTYPE(ANIRA_DTYPE_FLOAT, 32, 4)},
    }};
    for (const Row& row : rows) {
        SCOPED_TRACE(row.m_dtype);
        DlpackProducer producer;
        producer.m_managed.deleter = nullptr;
        producer.m_managed.dl_tensor.dtype = row.m_dl;
        anira_tensor tensor;
        ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
        EXPECT_EQ(tensor.dtype, row.m_dtype);
        anira_dtype bytes = 0;  // the claim of enums.h: the little-endian bytes are DLDataType
        std::memcpy(&bytes, &row.m_dl, sizeof(bytes));
        EXPECT_EQ(tensor.dtype, bytes);
        EXPECT_EQ(anira_tensor_data(&tensor, row.m_dtype), producer.m_data.data());
        EXPECT_EQ(anira_tensor_data_f32(&tensor) != nullptr, row.m_dtype == ANIRA_DTYPE_F32);
    }
}

TEST(AbiTensorDlpack, ADataTypeAniraHasNoCodeForIsRefused) {
    for (const uint8_t code : {kDLFloat8_e3m4, kDLFloat4_e2m1fn, uint8_t{255}}) {
        SCOPED_TRACE(int{code});
        DlpackProducer producer;
        producer.m_managed.dl_tensor.dtype = {.code = code, .bits = 8, .lanes = 1};
        expect_refused(producer, ANIRA_ERROR_NOT_SUPPORTED, "dlpack: dtype code");
    }
    DlpackProducer no_bits;
    no_bits.m_managed.dl_tensor.dtype = {.code = kDLFloat, .bits = 0, .lanes = 1};
    expect_refused(no_bits, ANIRA_ERROR_INVALID_ARGUMENT, "dlpack: a dtype of 0 bits");
    DlpackProducer no_lanes;
    no_lanes.m_managed.dl_tensor.dtype = {.code = kDLFloat, .bits = 32, .lanes = 0};
    expect_refused(no_lanes, ANIRA_ERROR_INVALID_ARGUMENT, "and 0 lanes");
}

TEST(AbiTensorDlpack, TheRankIsZeroToEight) {
    DlpackProducer nine;  // m_shape holds two extents: a read of nine would be ASan's to report
    nine.m_managed.dl_tensor.ndim = ANIRA_MAX_RANK + 1;
    expect_refused(nine, ANIRA_ERROR_INVALID_ARGUMENT, "dlpack: rank 9");
    DlpackProducer negative;
    negative.m_managed.dl_tensor.ndim = -1;
    expect_refused(negative, ANIRA_ERROR_INVALID_ARGUMENT, "dlpack: rank -1");
    DlpackProducer no_shape;
    no_shape.m_managed.dl_tensor.shape = nullptr;
    expect_refused(no_shape, ANIRA_ERROR_INVALID_ARGUMENT, "dlpack: NULL shape");
    DlpackProducer negative_extent;
    negative_extent.m_shape[1] = -1;
    expect_refused(negative_extent, ANIRA_ERROR_INVALID_ARGUMENT, "dlpack: negative extent");

    std::array<int64_t, ANIRA_MAX_RANK> full{1, 2, 1, 3, 1, 1, 1, 1};
    std::array<int64_t, ANIRA_MAX_RANK> full_strides{6, 3, 3, 1, 1, 1, 1, 1};
    DlpackProducer eight;
    eight.m_managed.deleter = nullptr;
    eight.m_managed.dl_tensor.ndim = ANIRA_MAX_RANK;
    eight.m_managed.dl_tensor.shape = full.data();
    eight.m_managed.dl_tensor.strides = full_strides.data();
    anira_tensor tensor;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &eight.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.ndim, 8U);
    EXPECT_EQ(tensor.shape[3], 3);
    EXPECT_EQ(tensor.strides[0], 6);
    EXPECT_EQ(tensor.strides[7], 1);

    DlpackProducer scalar;  // rank 0: shape and strides may be NULL (dlpack.h)
    scalar.m_managed.deleter = nullptr;
    scalar.m_managed.dl_tensor.ndim = 0;
    scalar.m_managed.dl_tensor.shape = nullptr;
    scalar.m_managed.dl_tensor.strides = nullptr;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &scalar.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.ndim, 0U);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 1U);
}

TEST(AbiTensorDlpack, NullStridesArePackedRowMajor) {
    DlpackProducer producer;  // a producer older than DLPack 1.2
    producer.m_managed.deleter = nullptr;
    producer.m_managed.dl_tensor.strides = nullptr;
    anira_tensor tensor;
    poison(tensor);
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    for (size_t axis = 0; axis < ANIRA_MAX_RANK; ++axis) {
        EXPECT_EQ(tensor.strides[axis], 0) << axis;
    }
    EXPECT_EQ(tensor.shape[0], 2);
    EXPECT_EQ(tensor.shape[1], 3);
}

TEST(AbiTensorDlpack, AllZeroStridesOverMoreThanOneElementAreRefused) {
    // DLPack spells a fully broadcast view with strides of 0; in anira_tensor all-zero strides
    // mean packed row-major. Over more than one element the two readings differ: refused.
    DlpackProducer broadcast;
    broadcast.m_strides = {0, 0};
    expect_refused(broadcast, ANIRA_ERROR_NOT_SUPPORTED, "dlpack: all-zero strides");

    DlpackProducer partial;  // {0, 1} is not all-zero: read as declared
    partial.m_managed.deleter = nullptr;
    partial.m_strides = {0, 1};
    anira_tensor tensor;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &partial.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.strides[0], 0);
    EXPECT_EQ(tensor.strides[1], 1);

    DlpackProducer single;  // one element: packed and broadcast are the same read
    single.m_managed.deleter = nullptr;
    single.m_shape = {1, 1};
    single.m_strides = {0, 0};
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &single.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 1U);

    DlpackProducer none;  // no element at all
    none.m_managed.deleter = nullptr;
    none.m_shape = {0, 3};
    none.m_strides = {0, 0};
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &none.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U);
}

TEST(AbiTensorDlpack, ByteOffsetAndANullDataPointerAreCopied) {
    DlpackProducer producer;
    producer.m_managed.deleter = nullptr;
    producer.m_managed.dl_tensor.byte_offset = 2 * sizeof(float);
    anira_tensor tensor;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.byte_offset, 8U);
    EXPECT_EQ(tensor.handle.host.ptr, producer.m_data.data()) << "the base, not base + offset";
    EXPECT_EQ(anira_tensor_data_f32(&tensor), &producer.m_data[2]);
    producer.m_shape = {0, 3};  // DLPack: the data pointer of a size-zero tensor is NULL
    producer.m_managed.dl_tensor.data = nullptr;
    producer.m_managed.dl_tensor.byte_offset = 0;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(anira_tensor_num_elements(&tensor), 0U);
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr);
}

TEST(AbiTensorDlpack, TheReadOnlyFlagIsTheOnlyOneCarried) {
    DlpackProducer producer;
    producer.m_managed.deleter = nullptr;
    producer.m_managed.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    anira_tensor tensor;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.flags, u32(ANIRA_TENSOR_READ_ONLY));
    producer.m_managed.flags = DLPACK_FLAG_BITMASK_IS_COPIED;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.flags, 0U) << "IS_COPIED has no anira counterpart";
    producer.m_managed.flags = DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED;
    ASSERT_EQ(anira_tensor_init_dlpack(&tensor, &producer.m_managed, nullptr), ANIRA_OK);
    EXPECT_EQ(tensor.flags, u32(ANIRA_TENSOR_READ_ONLY));
}

TEST(AbiTensorDlpack, NullArgumentsAreInvalid) {
    DlpackProducer producer;
    anira_tensor tensor;
    anira_tensor before;
    poison(tensor);
    poison(before);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_tensor_init_dlpack(nullptr, &producer.m_managed, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "dlpack: NULL tensor"), nullptr) << err.message;
    EXPECT_EQ(anira_tensor_init_dlpack(&tensor, nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "dlpack: NULL managed tensor"), nullptr) << err.message;
    EXPECT_EQ(anira_tensor_init_dlpack(nullptr, nullptr, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_TRUE(same_bytes(tensor, before));
    EXPECT_EQ(producer.m_deleted, 0);
}
