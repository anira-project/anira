#include <anira/abi/build_info.h>
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "capi/handles.h"

namespace {

struct Spec {
    Spec(const char* name, anira_role role = ANIRA_ROLE_STREAMED) {
        EXPECT_EQ(anira_tensor_spec_create(name, ANIRA_DTYPE_F32, role, &m_spec, &m_err), ANIRA_OK)
            << m_err.message;
    }
    ~Spec() { anira_tensor_spec_destroy(m_spec); }
    Spec(const Spec&) = delete;
    Spec& operator=(const Spec&) = delete;
    anira_tensor_spec* m_spec = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

struct Model {
    Model() { EXPECT_EQ(anira_model_config_create(&m_config, &m_err), ANIRA_OK); }
    ~Model() { anira_model_config_destroy(m_config); }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    anira_model_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

struct Machine {
    Machine() { EXPECT_EQ(anira_machine_config_create(&m_config, &m_err), ANIRA_OK); }
    ~Machine() { anira_machine_config_destroy(m_config); }
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;
    anira_machine_config* m_config = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

struct Release {
    static void fire(const void* bytes, void* ctx) {
        auto* self = static_cast<Release*>(ctx);
        self->m_count += 1;
        self->m_last = bytes;
    }
    int m_count = 0;
    const void* m_last = nullptr;
};

// An out-of-range value on purpose: every setter must refuse what a newer header or a
// corrupt caller may hand it.
template <class Enum>
Enum bad_enum(int value) {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    return static_cast<Enum>(value);
}

}  // namespace

// ---- tensor spec ---------------------------------------------------------------------------

TEST(AbiTensorSpec, CreateRejectsBadArguments) {
    anira_tensor_spec* spec = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_tensor_spec_create(nullptr, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, &spec, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "name"), nullptr);
    EXPECT_EQ(anira_tensor_spec_create("", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, &spec, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_create("x", ANIRA_DTYPE_F32, bad_enum<anira_role>(7), &spec, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "role"), nullptr);
    EXPECT_EQ(anira_tensor_spec_create("x", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, nullptr, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(spec, nullptr) << "out-parameters are written only on success";
    anira_tensor_spec_destroy(nullptr);
}

TEST(AbiTensorSpec, DefaultsAndSetters) {
    const Spec s("audio_in");
    EXPECT_EQ(s.m_spec->m_name, "audio_in");
    EXPECT_EQ(s.m_spec->m_dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(s.m_spec->m_ndim, 0u);
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, 2, ANIRA_AXIS_TIME, 512), ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, 0, ANIRA_AXIS_BATCH, 1), ANIRA_OK);
    EXPECT_EQ(s.m_spec->m_ndim, 3u);
    EXPECT_TRUE(s.m_spec->m_axes[2].m_written);
    EXPECT_FALSE(s.m_spec->m_axes[1].m_written) << "slot 1 stays unwritten until set";
    EXPECT_EQ(s.m_spec->m_axes[2].m_extent, 512);
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, 1, ANIRA_AXIS_TIME, ANIRA_DYNAMIC), ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_set_window(s.m_spec, 512, ANIRA_UNBOUNDED, 128), ANIRA_OK);
    EXPECT_EQ(s.m_spec->m_window_max, ANIRA_UNBOUNDED);
    EXPECT_EQ(anira_tensor_spec_set_time_ratio(s.m_spec, 1, 2), ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_set_time_ratio(s.m_spec, 0, 0), ANIRA_OK) << "(0, 0) = derive";
    EXPECT_EQ(anira_tensor_spec_set_latency(s.m_spec, 2048), ANIRA_OK);
    EXPECT_EQ(s.m_spec->m_latency, 2048);
}

TEST(AbiTensorSpec, SingleArgumentRejections) {
    const Spec s("x");
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, ANIRA_MAX_RANK, ANIRA_AXIS_TIME, 1),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, 0, bad_enum<anira_axis_tag>(99), 1),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, 0, ANIRA_AXIS_TIME, 0),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_axis(s.m_spec, 0, ANIRA_AXIS_TIME, -2),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(s.m_spec->m_ndim, 0u) << "a rejected call leaves the spec as it was";
    EXPECT_EQ(anira_tensor_spec_set_window(s.m_spec, -1, 0, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_window(s.m_spec, 1, -2, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_window(s.m_spec, 1, 1, -1), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_time_ratio(s.m_spec, 1, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_time_ratio(s.m_spec, -1, 1), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_latency(s.m_spec, -1), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_axis(nullptr, 0, ANIRA_AXIS_TIME, 1),
              ANIRA_ERROR_INVALID_ARGUMENT);
}

// ---- model config ----------------------------------------------------------------------------

TEST(AbiModelConfig, DefaultsAndEntries) {
    Model m;
    EXPECT_EQ(anira_model_config_model_count(m.m_config), 0u);
    EXPECT_EQ(anira_model_config_model_count(nullptr), 0u);
    uint32_t index = 99;
    EXPECT_EQ(anira_model_config_add_model_path(m.m_config,
                                                ANIRA_ENGINE_ONNXRUNTIME,
                                                "model.onnx",
                                                &index,
                                                &m.m_err),
              ANIRA_OK);
    EXPECT_EQ(index, 0u);
    const std::array<unsigned char, 4> blob{1, 2, 3, 4};
    EXPECT_EQ(anira_model_config_add_model_bytes(m.m_config,
                                                 ANIRA_ENGINE_LIBTORCH,
                                                 blob.data(),
                                                 blob.size(),
                                                 ANIRA_BYTES_COPY,
                                                 nullptr,
                                                 nullptr,
                                                 &index,
                                                 &m.m_err),
              ANIRA_OK);
    EXPECT_EQ(index, 1u);
    EXPECT_EQ(anira_model_config_add_model_path_custom(m.m_config,
                                                       "com.example.engine",
                                                       "model.bin",
                                                       &index,
                                                       &m.m_err),
              ANIRA_OK);
    EXPECT_EQ(index, 2u);
    EXPECT_EQ(anira_model_config_model_count(m.m_config), 3u);
    EXPECT_EQ(anira_model_config_model_engine(m.m_config, 0), ANIRA_ENGINE_ONNXRUNTIME);
    EXPECT_STREQ(anira_model_config_model_path(m.m_config, 0), "model.onnx");
    EXPECT_EQ(anira_model_config_model_engine_id(m.m_config, 0), nullptr);
    EXPECT_EQ(anira_model_config_model_path(m.m_config, 1), nullptr) << "a bytes entry has no path";
    const void* bytes = nullptr;
    size_t size = 0;
    EXPECT_EQ(anira_model_config_model_bytes(m.m_config, 1, &bytes, &size), ANIRA_OK);
    EXPECT_EQ(size, blob.size());
    EXPECT_NE(bytes, blob.data()) << "COPY holds its own bytes";
    EXPECT_EQ(std::memcmp(bytes, blob.data(), size), 0);
    EXPECT_EQ(anira_model_config_model_bytes(m.m_config, 0, &bytes, &size),
              ANIRA_ERROR_INVALID_STATE)
        << "a path entry";
    EXPECT_EQ(anira_model_config_model_engine(m.m_config, 2), ANIRA_ENGINE_NONE);
    EXPECT_STREQ(anira_model_config_model_engine_id(m.m_config, 2), "com.example.engine");
    EXPECT_EQ(anira_model_config_model_engine(m.m_config, 3), ANIRA_ENGINE_NONE) << "out of range";
    EXPECT_EQ(anira_model_config_model_path(m.m_config, 3), nullptr);
}

TEST(AbiModelConfig, EntryRejections) {
    Model m;
    uint32_t index = 0;
    EXPECT_EQ(
        anira_model_config_add_model_path(m.m_config, ANIRA_ENGINE_NONE, "x", &index, &m.m_err),
        ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(m.m_err.message, "engine"), nullptr);
    EXPECT_EQ(anira_model_config_add_model_path(m.m_config,
                                                bad_enum<anira_engine>(0x1000),
                                                "x",
                                                &index,
                                                &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_add_model_path(m.m_config,
                                                ANIRA_ENGINE_ONNXRUNTIME,
                                                "",
                                                &index,
                                                &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_add_model_bytes(m.m_config,
                                                 ANIRA_ENGINE_ONNXRUNTIME,
                                                 nullptr,
                                                 4,
                                                 ANIRA_BYTES_COPY,
                                                 nullptr,
                                                 nullptr,
                                                 &index,
                                                 &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    const std::array<unsigned char, 4> blob{};
    EXPECT_EQ(anira_model_config_add_model_bytes(m.m_config,
                                                 ANIRA_ENGINE_ONNXRUNTIME,
                                                 blob.data(),
                                                 0,
                                                 ANIRA_BYTES_COPY,
                                                 nullptr,
                                                 nullptr,
                                                 &index,
                                                 &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_add_model_bytes(m.m_config,
                                                 ANIRA_ENGINE_ONNXRUNTIME,
                                                 blob.data(),
                                                 4,
                                                 bad_enum<anira_bytes_ownership>(5),
                                                 nullptr,
                                                 nullptr,
                                                 &index,
                                                 &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_add_model_path_custom(m.m_config, "noDot", "x", &index, &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(m.m_err.message, "reverse-URI"), nullptr);
    EXPECT_EQ(anira_model_config_set_model_bytes(m.m_config,
                                                 0,
                                                 blob.data(),
                                                 4,
                                                 ANIRA_BYTES_COPY,
                                                 nullptr,
                                                 nullptr,
                                                 &m.m_err),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "no entry yet";
    EXPECT_EQ(anira_model_config_model_count(m.m_config), 0u);
}

TEST(AbiModelConfig, SetModelBytesPatchesAPathEntry) {
    Model m;
    uint32_t index = 0;
    ASSERT_EQ(anira_model_config_add_model_path(m.m_config,
                                                ANIRA_ENGINE_EXECUTORCH,
                                                "model.pte",
                                                &index,
                                                &m.m_err),
              ANIRA_OK);
    const std::array<unsigned char, 3> blob{7, 8, 9};
    EXPECT_EQ(anira_model_config_set_model_bytes(m.m_config,
                                                 0,
                                                 blob.data(),
                                                 blob.size(),
                                                 ANIRA_BYTES_BORROW,
                                                 nullptr,
                                                 nullptr,
                                                 &m.m_err),
              ANIRA_OK);
    EXPECT_EQ(anira_model_config_model_path(m.m_config, 0), nullptr)
        << "model_path() is NULL for a bytes entry";
    EXPECT_EQ(m.m_config->m_models[0].m_path, "model.pte") << "the path is kept for to_json";
    const void* bytes = nullptr;
    size_t size = 0;
    EXPECT_EQ(anira_model_config_model_bytes(m.m_config, 0, &bytes, &size), ANIRA_OK);
    EXPECT_EQ(bytes, blob.data()) << "BORROW points at the caller's memory";
}

TEST(AbiModelConfig, BorrowedBytesReleaseFiresOnceWhenTheLastCarrierDies) {
    Release release;
    const std::array<unsigned char, 2> blob{1, 2};
    {
        Model m;
        uint32_t index = 0;
        ASSERT_EQ(anira_model_config_add_model_bytes(m.m_config,
                                                     ANIRA_ENGINE_LIBTORCH,
                                                     blob.data(),
                                                     blob.size(),
                                                     ANIRA_BYTES_BORROW,
                                                     &Release::fire,
                                                     &release,
                                                     &index,
                                                     &m.m_err),
                  ANIRA_OK);
        // A second carrier shares the bytes (what a handler copy would do); no release yet.
        const std::shared_ptr<anira::capi::BytesCarrier> shared = m.m_config->m_models[0].m_bytes;
        EXPECT_EQ(release.m_count, 0);
        anira_model_config_destroy(m.m_config);
        m.m_config = nullptr;
        EXPECT_EQ(release.m_count, 0) << "the shared carrier still holds the bytes";
    }
    EXPECT_EQ(release.m_count, 1);
    EXPECT_EQ(release.m_last, blob.data());
    // COPY never calls a release callback.
    Release copy_release;
    {
        Model m;
        uint32_t index = 0;
        ASSERT_EQ(anira_model_config_add_model_bytes(m.m_config,
                                                     ANIRA_ENGINE_LIBTORCH,
                                                     blob.data(),
                                                     blob.size(),
                                                     ANIRA_BYTES_COPY,
                                                     &Release::fire,
                                                     &copy_release,
                                                     &index,
                                                     &m.m_err),
                  ANIRA_OK);
    }
    EXPECT_EQ(copy_release.m_count, 0);
}

TEST(AbiModelConfig, SpecsAreCopiedAndTheRestIsScalar) {
    Model m;
    {
        const Spec in("audio_in");
        EXPECT_EQ(anira_tensor_spec_set_axis(in.m_spec, 0, ANIRA_AXIS_TIME, 512), ANIRA_OK);
        EXPECT_EQ(anira_model_config_add_input(m.m_config, in.m_spec), ANIRA_OK);
        EXPECT_EQ(anira_tensor_spec_set_axis(in.m_spec, 0, ANIRA_AXIS_TIME, 1024), ANIRA_OK);
        EXPECT_EQ(m.m_config->m_inputs[0].m_axes[0].m_extent, 512) << "add_input copied";
        const Spec out("audio_out", ANIRA_ROLE_STATIC);
        EXPECT_EQ(anira_model_config_add_output(m.m_config, out.m_spec), ANIRA_OK);
    }
    EXPECT_EQ(m.m_config->m_inputs.size(), 1u);
    EXPECT_EQ(m.m_config->m_outputs[0].m_role, ANIRA_ROLE_STATIC);
    EXPECT_EQ(anira_model_config_add_input(m.m_config, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_set_default_engine(m.m_config, ANIRA_ENGINE_TFLITE), ANIRA_OK);
    EXPECT_EQ(anira_model_config_set_default_engine(m.m_config, bad_enum<anira_engine>(42)),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_set_default_engine_custom(m.m_config, "com.example.x"), ANIRA_OK);
    EXPECT_EQ(m.m_config->m_default_engine, ANIRA_ENGINE_NONE);
    EXPECT_EQ(anira_model_config_set_default_engine_custom(m.m_config, "nodot"),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_set_state(m.m_config, ANIRA_MODEL_STATEFUL), ANIRA_OK);
    EXPECT_EQ(anira_model_config_set_state(m.m_config, bad_enum<anira_model_state>(3)),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_set_max_instances(m.m_config, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_set_max_instances(m.m_config, 4), ANIRA_OK);
    EXPECT_EQ(anira_model_config_set_anchor(m.m_config, 1, 0), ANIRA_OK);
    EXPECT_FALSE(m.m_config->m_anchor_is_input);
    uint32_t index = 0;
    ASSERT_EQ(anira_model_config_add_model_path(m.m_config,
                                                ANIRA_ENGINE_ONNXRUNTIME,
                                                "m.onnx",
                                                &index,
                                                &m.m_err),
              ANIRA_OK);
    EXPECT_EQ(anira_model_config_set_tensor_name(m.m_config, 0, "audio_in", "input_0"), ANIRA_OK);
    EXPECT_EQ(anira_model_config_set_tensor_name(m.m_config, 0, "", "x"),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_model_config_set_tensor_name(m.m_config, 1, "a", "b"),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(m.m_config->m_models[0].m_tensor_names.at("audio_in"), "input_0");
}

// ---- machine config --------------------------------------------------------------------------

TEST(AbiMachineConfig, DefaultsScalarsAndClamps) {
    const Machine m;
    EXPECT_EQ(m.m_config->m_num_threads, ANIRA_THREADS_AUTO);
    EXPECT_EQ(m.m_config->m_queue_capacity, 512u);
    EXPECT_EQ(anira_machine_config_set_threads(m.m_config, 2, ANIRA_WAIT_BLOCKING), ANIRA_OK);
    EXPECT_EQ(anira_machine_config_set_threads(m.m_config, 2, bad_enum<anira_wait_strategy>(9)),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_machine_config_set_log_level(m.m_config, ANIRA_LOG_DEBUG), ANIRA_OK);
    EXPECT_EQ(anira_machine_config_set_log_level(m.m_config, bad_enum<anira_log_level>(4)),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_machine_config_set_log_drain(m.m_config, ANIRA_LOG_DRAIN_MANUAL, 0), ANIRA_OK);
    EXPECT_EQ(m.m_config->m_drain_interval_ms, 10u) << "0 keeps the default";
    EXPECT_EQ(anira_machine_config_set_log_queue_capacity(m.m_config, 63), ANIRA_OK);
    EXPECT_EQ(m.m_config->m_queue_capacity, 64u);
    EXPECT_EQ(anira_machine_config_set_log_queue_capacity(m.m_config, 70000), ANIRA_OK);
    EXPECT_EQ(m.m_config->m_queue_capacity, 65536u);
    EXPECT_EQ(anira_machine_config_set_log_flags(m.m_config, ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK),
              ANIRA_OK);
    EXPECT_EQ(anira_machine_config_set_log_flags(m.m_config, 2u), ANIRA_ERROR_INVALID_ARGUMENT);
    int user_data = 0;
    EXPECT_EQ(anira_machine_config_set_log_sink(m.m_config, nullptr, &user_data), ANIRA_OK);
    EXPECT_EQ(m.m_config->m_sink_user_data, nullptr) << "no sink, no user data";
}

TEST(AbiMachineConfig, LogDescriptorIsReadWithinItsSizeAndChecksTheAbi) {
    const Machine m;
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    desc.level = ANIRA_LOG_ERROR;
    desc.queue_capacity = 32;
    EXPECT_EQ(anira_machine_config_set_log(m.m_config, &desc), ANIRA_OK);
    EXPECT_EQ(m.m_config->m_log_level, ANIRA_LOG_ERROR);
    EXPECT_EQ(m.m_config->m_queue_capacity, 64u) << "clamped";
    desc.abi_version = ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1, 0);
    EXPECT_EQ(anira_machine_config_set_log(m.m_config, &desc), ANIRA_ERROR_ABI_VERSION);
    desc = ANIRA_LOG_DESC_INIT;
    desc.struct_size = 8;
    EXPECT_EQ(anira_machine_config_set_log(m.m_config, &desc), ANIRA_ERROR_INVALID_ARGUMENT)
        << "shorter than {struct_size, abi_version, user_data}";
    EXPECT_EQ(anira_machine_config_set_log(m.m_config, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiMachineConfig, DeviceDescriptorsAreCopiedWithinStructSize) {
    const Machine m;
    anira_cuda_desc cuda{};
    cuda.struct_size = 12;  // struct_size, ownership, device: an older, shorter header
    cuda.ownership = ANIRA_OWNERSHIP_BORROWED;
    cuda.device = 3;
    EXPECT_EQ(anira_machine_config_set_cuda(m.m_config, &cuda), ANIRA_OK);
    ASSERT_TRUE(m.m_config->m_cuda.has_value());
    const anira_cuda_desc stored = m.m_config->m_cuda.value_or(anira_cuda_desc{});
    EXPECT_EQ(stored.struct_size, sizeof(anira_cuda_desc)) << "normalized to this build's size";
    EXPECT_EQ(stored.ownership, static_cast<uint32_t>(ANIRA_OWNERSHIP_BORROWED));
    EXPECT_EQ(stored.device, 3);
    EXPECT_EQ(stored.pinned_pool_limit, 0u) << "the tail keeps the default";
    EXPECT_EQ(anira_machine_config_set_cuda(m.m_config, nullptr), ANIRA_OK);
    EXPECT_FALSE(m.m_config->m_cuda.has_value()) << "NULL clears the block";
    anira_gl_desc gl = ANIRA_GL_DESC_INIT;
    gl.struct_size = 2;
    EXPECT_EQ(anira_machine_config_set_gl(m.m_config, &gl), ANIRA_ERROR_INVALID_ARGUMENT);
    const anira_vulkan_desc vulkan = ANIRA_VULKAN_DESC_INIT;
    const anira_metal_desc metal = ANIRA_METAL_DESC_INIT;
    const anira_d3d12_desc d3d12 = ANIRA_D3D12_DESC_INIT;
    const anira_webgpu_desc webgpu = ANIRA_WEBGPU_DESC_INIT;
    EXPECT_EQ(anira_machine_config_set_vulkan(m.m_config, &vulkan), ANIRA_OK);
    EXPECT_EQ(anira_machine_config_set_metal(m.m_config, &metal), ANIRA_OK);
    EXPECT_EQ(anira_machine_config_set_d3d12(m.m_config, &d3d12), ANIRA_OK);
    EXPECT_EQ(anira_machine_config_set_webgpu(m.m_config, &webgpu), ANIRA_OK);
    EXPECT_TRUE(m.m_config->m_vulkan && m.m_config->m_metal && m.m_config->m_d3d12 &&
                m.m_config->m_webgpu);
}

// ---- contract ---------------------------------------------------------------------------------

TEST(AbiContract, HardAndAsyncGateTheirSetters) {
    anira_contract* hard = nullptr;
    anira_contract* async_contract = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_contract_create_hard(0, 0, 0.0, &hard, &err), ANIRA_OK)
        << "0, 0, 0 is legal here";
    ASSERT_EQ(anira_contract_create_async(&async_contract, &err), ANIRA_OK);
    EXPECT_EQ(anira_contract_get_kind(hard), ANIRA_CONTRACT_HARD);
    EXPECT_EQ(anira_contract_get_kind(async_contract), ANIRA_CONTRACT_ASYNC);
    EXPECT_EQ(anira_contract_hard_set_geometry(hard, 1, 512, 48000.0), ANIRA_OK);
    EXPECT_EQ(anira_contract_hard_set_geometry(hard, 512, 1, 48000.0),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_set_budget(hard, ANIRA_BUDGET_EXPLICIT, 42.66), ANIRA_OK);
    EXPECT_EQ(anira_contract_hard_set_budget(hard, ANIRA_BUDGET_EXPLICIT, 0.0),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_set_warmup(hard, ANIRA_WARMUP_FIXED, 5), ANIRA_OK);
    EXPECT_EQ(hard->hard()->m_warmup_iterations, 5u);
    EXPECT_EQ(anira_contract_hard_set_on_miss(hard, ANIRA_MISS_ZEROS), ANIRA_OK);
    EXPECT_EQ(anira_contract_hard_set_wait_ratio(hard, -0.5), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_hard_set_wait_ratio(hard, 0.5), ANIRA_OK);
    EXPECT_EQ(anira_contract_async_set_deadline(hard, 10.0), ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_async_set_policy(hard,
                                              ANIRA_LATE_DROP,
                                              ANIRA_PRIORITY_AUTO,
                                              0,
                                              0,
                                              ANIRA_DELIVERY_POLLED),
              ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_set_budget(async_contract, ANIRA_BUDGET_EXPLICIT, 1.0),
              ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_hard_set_geometry(async_contract, 1, 1, 1.0),
              ANIRA_ERROR_WRONG_CONTRACT);
    EXPECT_EQ(anira_contract_async_set_deadline(async_contract, 10.0), ANIRA_OK);
    EXPECT_EQ(anira_contract_async_set_policy(async_contract,
                                              ANIRA_LATE_DROP,
                                              ANIRA_PRIORITY_INTERACTIVE,
                                              2,
                                              3,
                                              ANIRA_DELIVERY_IMMEDIATE),
              ANIRA_OK);
    EXPECT_EQ(async_contract->asynchronous()->m_lanes, 2u);
    EXPECT_EQ(anira_contract_async_set_policy(async_contract,
                                              bad_enum<anira_late_policy>(5),
                                              ANIRA_PRIORITY_AUTO,
                                              0,
                                              0,
                                              ANIRA_DELIVERY_POLLED),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_set_edge_cost(async_contract, ANIRA_EDGE_COST_STRICT), ANIRA_OK);
    EXPECT_EQ(anira_contract_set_edge_cost(hard, bad_enum<anira_edge_cost>(2)),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_contract_create_hard(2, 1, 48000.0, &hard, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "block_min"), nullptr);
    anira_contract_destroy(hard);
    anira_contract_destroy(async_contract);
    anira_contract_destroy(nullptr);
}

// ---- job options --------------------------------------------------------------------------------

TEST(AbiJobOptions, ScalarsAndBorrowedExtensions) {
    anira_job_options* options = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_job_options_create(&options, &err), ANIRA_OK);
    EXPECT_TRUE(options->m_tail_flush);
    const std::array<int64_t, 2> trims{-1, 128};
    EXPECT_EQ(anira_job_options_set_head_trim(options, 2, trims.data()), ANIRA_OK);
    EXPECT_EQ(options->m_head_trim.size(), 2u);
    const std::array<int64_t, 1> bad{-2};
    EXPECT_EQ(anira_job_options_set_head_trim(options, 1, bad.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_job_options_set_head_trim(options, 1, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_job_options_set_tail_flush(options, 0), ANIRA_OK);
    EXPECT_FALSE(options->m_tail_flush);
    EXPECT_EQ(anira_job_options_set_below_min(options, ANIRA_PAD_ZEROS), ANIRA_OK);
    EXPECT_EQ(anira_job_options_set_below_min(options, bad_enum<anira_pad_policy>(2)),
              ANIRA_ERROR_INVALID_ARGUMENT);
    const anira_ext_entry first = ANIRA_EXT_ENTRY_INIT;
    const anira_ext_entry second = ANIRA_EXT_ENTRY_INIT;
    EXPECT_EQ(anira_job_options_set_ext(options, &first.header), ANIRA_OK);
    EXPECT_EQ(options->m_borrowed_ext.size(), 1u);
    EXPECT_EQ(options->m_borrowed_ext[0], &first.header) << "borrowed: pointer identity";
    EXPECT_EQ(anira_job_options_set_ext(options, &second.header), ANIRA_OK);
    EXPECT_EQ(options->m_borrowed_ext.size(), 1u) << "a second set of the same kind replaces";
    EXPECT_EQ(options->m_borrowed_ext[0], &second.header);
    const anira_ext_header short_header{.struct_size = 2, .version = 1, .kind = "entry"};
    EXPECT_EQ(anira_job_options_set_ext(options, &short_header), ANIRA_ERROR_INVALID_ARGUMENT);
    const std::string json = R"({"name": "decode"})";
    EXPECT_EQ(anira_job_options_set_ext_json(options, "entry", json.c_str(), json.size()),
              ANIRA_OK);
    EXPECT_NE(options->m_json_ext.find("entry"), nullptr);
    anira_job_options_destroy(options);
    anira_job_options_destroy(nullptr);
}

// ---- registry enumeration --------------------------------------------------------------------

TEST(AbiExtKinds, ScalarEnumerationConvention) {
    uint32_t count = 0;
    EXPECT_EQ(anira_registered_ext_kinds(&count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, 1u) << "v3.0.0 registers one kind: entry";
    std::array<const char*, 4> kinds{};
    count = 0;
    EXPECT_EQ(anira_registered_ext_kinds(&count, kinds.data()), ANIRA_INCOMPLETE) << "capacity 0";
    EXPECT_EQ(count, 1u);
    count = kinds.size();
    EXPECT_EQ(anira_registered_ext_kinds(&count, kinds.data()), ANIRA_OK);
    EXPECT_EQ(count, 1u);
    EXPECT_STREQ(kinds[0], "entry");
    EXPECT_EQ(anira_registered_ext_kinds(nullptr, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
}
