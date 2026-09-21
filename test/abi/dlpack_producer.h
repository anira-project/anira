// The tests' own spelling of dlpack.h and a producer built on it (test_Tensor.cpp and the AbiCxx
// tensor cases). Written from dmlc/dlpack include/dlpack/dlpack.h at tag v1.3 (commit 84d107b),
// field by field and in its order, NOT from the private mirror inside src/capi/tensor.cpp:
// when the two spellings disagree the DLPack cases fail, and that second, independent spelling
// is the only guard the mirror has. Only what a producer fills is declared (no legacy
// DLManagedTensor, no exchange API).
#ifndef ANIRA_TEST_ABI_DLPACK_PRODUCER_H
#define ANIRA_TEST_ABI_DLPACK_PRODUCER_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace anira_test {

// NOLINTBEGIN(readability-identifier-naming) DLPack's own names
struct DLPackVersion {
    uint32_t major;
    uint32_t minor;
};

enum DLDeviceType : int32_t {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLROCMHost = 11,
};

struct DLDevice {
    DLDeviceType device_type;
    int32_t device_id;
};

// DLDataTypeCode. dlpack.h declares an enum and stores it in a uint8_t ("we keep it uint8_t
// instead of DLDataTypeCode"); constants of the stored type keep every braced initializer
// below free of an enum-to-integer conversion (MSVC C2397).
inline constexpr uint8_t kDLInt = 0U;
inline constexpr uint8_t kDLUInt = 1U;
inline constexpr uint8_t kDLFloat = 2U;
inline constexpr uint8_t kDLOpaqueHandle = 3U;
inline constexpr uint8_t kDLBfloat = 4U;
inline constexpr uint8_t kDLComplex = 5U;
inline constexpr uint8_t kDLBool = 6U;
inline constexpr uint8_t kDLFloat8_e3m4 = 7U;     // the first code anira has no counterpart for
inline constexpr uint8_t kDLFloat4_e2m1fn = 17U;  // the last code of DLPack 1.3

struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
};

struct DLManagedTensorVersioned {
    DLPackVersion version;
    void* manager_ctx;
    void (*deleter)(DLManagedTensorVersioned* self);
    uint64_t flags;
    DLTensor dl_tensor;
};

inline constexpr uint64_t DLPACK_FLAG_BITMASK_READ_ONLY = 1ULL << 0U;
inline constexpr uint64_t DLPACK_FLAG_BITMASK_IS_COPIED = 1ULL << 1U;
// NOLINTEND(readability-identifier-naming)

// The layout dlpack.h has on an LP64 or LLP64 target; on a 32-bit one the natural layout is the
// platform's, as it is for a real producer.
static_assert(sizeof(DLPackVersion) == 8 && sizeof(DLDevice) == 8 && sizeof(DLDataType) == 4);
static_assert(sizeof(void*) != 8 ||
              (sizeof(DLTensor) == 48 && sizeof(DLManagedTensorVersioned) == 80));
static_assert(sizeof(void*) != 8 || (offsetof(DLManagedTensorVersioned, flags) == 24 &&
                                     offsetof(DLManagedTensorVersioned, dl_tensor) == 32 &&
                                     offsetof(DLTensor, byte_offset) == 40));

/// A DLPack producer: six floats as [2, 3] with strides [3, 1], and the count of deleter calls.
/// The deleter reaches the producer through the managed tensor's own manager_ctx, as a real
/// producer does, so a call with anything but the managed tensor itself would crash here.
/// Neither copied nor moved: m_managed points into the object.
struct DlpackProducer {
    DlpackProducer() = default;
    DlpackProducer(const DlpackProducer&) = delete;
    DlpackProducer& operator=(const DlpackProducer&) = delete;

    static void on_delete(DLManagedTensorVersioned* self) {
        auto* producer = static_cast<DlpackProducer*>(self->manager_ctx);
        producer->m_deleted += 1;
        producer->m_last = self;
    }

    std::array<float, 6> m_data{};
    std::array<int64_t, 2> m_shape{2, 3};
    std::array<int64_t, 2> m_strides{3, 1};
    int m_deleted = 0;
    const DLManagedTensorVersioned* m_last = nullptr;
    DLManagedTensorVersioned m_managed{
        .version = {.major = 1, .minor = 3},
        .manager_ctx = this,
        .deleter = &DlpackProducer::on_delete,
        .flags = 0,
        .dl_tensor = {.data = m_data.data(),
                      .device = {.device_type = kDLCPU, .device_id = 0},
                      .ndim = 2,
                      .dtype = {.code = kDLFloat, .bits = 32, .lanes = 1},
                      .shape = m_shape.data(),
                      .strides = m_strides.data(),
                      .byte_offset = 0},
    };
};

/// A device type this header does not name: the enum has a fixed underlying type, so every
/// int32_t is a value of it.
inline DLDeviceType device_type_of(int32_t value) {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) an unknown device on purpose
    return static_cast<DLDeviceType>(value);
}

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_DLPACK_PRODUCER_H
