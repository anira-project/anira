/*
 * The LiteRT adapter: one compiled model per instance in its own environment, run through
 * the model's first signature (LiteRT synthesizes one for a file without signatures, keyed by
 * the tensor names), the slots bound to the signature's inputs and outputs by name where the
 * entry's tensors record or the canonical name matches a key and by the signature's index
 * order otherwise, checked against the signature tensors' ranked types at prepare, and one
 * managed host buffer per signature tensor in the signature's index order (what
 * LiteRtRunCompiledModel takes), filled from and read into the descriptors' memory through a
 * lock and a memcpy per call (the managed buffers need LiteRT's own alignment; binding
 * anira's memory in place waits for the aligned chunk storage). File-local over
 * anira::backend::Model: nothing of LiteRT enters a public header.
 */
#ifdef USE_LITERT

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace anira::backend {

namespace {

constexpr const char* k_engine = "litert";

// The status names of litert/c/litert_common.h, so a message reads "kLiteRtStatusErrorFileIO"
// and not a number.
const char* litert_status_name(LiteRtStatus status) {
    switch (status) {
        case kLiteRtStatusOk: return "kLiteRtStatusOk";
        case kLiteRtStatusErrorInvalidArgument: return "kLiteRtStatusErrorInvalidArgument";
        case kLiteRtStatusErrorMemoryAllocationFailure:
            return "kLiteRtStatusErrorMemoryAllocationFailure";
        case kLiteRtStatusErrorRuntimeFailure: return "kLiteRtStatusErrorRuntimeFailure";
        case kLiteRtStatusErrorMissingInputTensor: return "kLiteRtStatusErrorMissingInputTensor";
        case kLiteRtStatusErrorUnsupported: return "kLiteRtStatusErrorUnsupported";
        case kLiteRtStatusErrorNotFound: return "kLiteRtStatusErrorNotFound";
        case kLiteRtStatusErrorTimeoutExpired: return "kLiteRtStatusErrorTimeoutExpired";
        case kLiteRtStatusErrorFileIO: return "kLiteRtStatusErrorFileIO";
        case kLiteRtStatusErrorInvalidFlatbuffer: return "kLiteRtStatusErrorInvalidFlatbuffer";
        default: return "LiteRtStatus";
    }
}

// Every LiteRT C API call returns a LiteRtStatus. A failure here means a setup or runtime
// problem, so it becomes a StatusError with the failing call and the status named; the status
// is ANIRA_ERROR_ENGINE unless the caller passes another.
void litert_check(LiteRtStatus status,
                  const char* what,
                  anira_status failure = ANIRA_ERROR_ENGINE,
                  const std::string& where = "") {
    if (status != kLiteRtStatusOk) {
        const std::string text = std::string(what) + " failed with " + litert_status_name(status) +
                                 " (" + std::to_string(static_cast<int>(status)) + ")";
        throw StatusError(failure,
                          model_file::message(k_engine, where.empty() ? what : where, text));
    }
}

// The anira dtype of a LiteRT element type; 0 for one anira has no code for.
anira_dtype dtype_of(LiteRtElementType type) {
    switch (type) {
        case kLiteRtElementTypeFloat32: return ANIRA_DTYPE_F32;
        case kLiteRtElementTypeFloat64: return ANIRA_DTYPE_F64;
        case kLiteRtElementTypeFloat16: return ANIRA_DTYPE_F16;
        case kLiteRtElementTypeBFloat16: return ANIRA_DTYPE_BF16;
        case kLiteRtElementTypeInt8: return ANIRA_DTYPE_I8;
        case kLiteRtElementTypeUInt8: return ANIRA_DTYPE_U8;
        case kLiteRtElementTypeInt16: return ANIRA_DTYPE_I16;
        case kLiteRtElementTypeInt32: return ANIRA_DTYPE_I32;
        case kLiteRtElementTypeInt64: return ANIRA_DTYPE_I64;
        default: return 0;
    }
}

// A ranked float32 tensor type of the record's engine dims: what a managed buffer is created
// with (the signature's dynamic extents resolved to the pinned window's).
LiteRtRankedTensorType float32_type_of(const std::vector<int64_t>& dims) {
    LiteRtRankedTensorType type{};
    type.element_type = kLiteRtElementTypeFloat32;
    const size_t rank = dims.size() < LITERT_TENSOR_MAX_RANK ? dims.size() : LITERT_TENSOR_MAX_RANK;
    type.layout.rank = static_cast<unsigned int>(rank);
    type.layout.has_strides = false;
    for (size_t d = 0; d < rank; ++d) { type.layout.dimensions[d] = static_cast<int32_t>(dims[d]); }
    return type;
}

// One tensor of the signature as LiteRT describes it: the key it is reached by, the ranked
// type's extents (a negative one dynamic) and its element type.
EngineTensor engine_tensor_of(LiteRtSignature signature, const char* name, bool is_input) {
    EngineTensor tensor;
    tensor.m_name = name;
    LiteRtTensor handle = nullptr;
    litert_check(is_input ? LiteRtGetSignatureInputTensor(signature, name, &handle)
                          : LiteRtGetSignatureOutputTensor(signature, name, &handle),
                 is_input ? "LiteRtGetSignatureInputTensor" : "LiteRtGetSignatureOutputTensor",
                 ANIRA_ERROR_MODEL_LOAD);
    LiteRtTensorTypeId type_id = kLiteRtRankedTensorType;
    litert_check(LiteRtGetTensorTypeId(handle, &type_id),
                 "LiteRtGetTensorTypeId",
                 ANIRA_ERROR_MODEL_LOAD);
    if (type_id != kLiteRtRankedTensorType) {
        tensor.m_type_word = "an unranked tensor";
        return tensor;
    }
    LiteRtRankedTensorType type{};
    litert_check(LiteRtGetRankedTensorType(handle, &type),
                 "LiteRtGetRankedTensorType",
                 ANIRA_ERROR_MODEL_LOAD);
    for (unsigned int d = 0; d < type.layout.rank && d < LITERT_TENSOR_MAX_RANK; ++d) {
        tensor.m_dims.push_back(type.layout.dimensions[d]);
    }
    tensor.m_dtype = dtype_of(type.element_type);
    tensor.m_type_word =
        "LiteRT element type " + std::to_string(static_cast<int>(type.element_type));
    return tensor;
}

/// One compiled model in its own environment: what one instance of the adapter runs on.
/// Loads and compiles at construction (a file or the bytes of the record); creates its managed
/// buffers at bind; runs every inference over the context's descriptors through them.
class Instance {
public:
    explicit Instance(const Model& model);
    ~Instance();
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The first signature's tensors of either side, in the signature's index order.
    std::vector<EngineTensor> inputs() { return side_of(true); }
    std::vector<EngineTensor> outputs() { return side_of(false); }

    /// Creates one managed host buffer per signature tensor, in the signature's index order,
    /// typed with the record's engine dims of the slot bound to it; keeps per slot the buffer
    /// it fills or reads.
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The FIXED warm-up of the record over the managed buffers (zeros in); a failing run is
    /// StatusError(ANIRA_ERROR_ENGINE) with the status named.
    void warm_up(uint32_t iterations);

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing call.
    anira_status process(const anira_engine_ctx& ctx) noexcept;

private:
    std::vector<EngineTensor> side_of(bool inputs);
    /// Copies every input descriptor into its buffer, runs, copies every output buffer into
    /// its descriptor; throws StatusError.
    void run(const anira_engine_ctx& ctx);
    /// Destroys every LiteRT handle this instance owns (no-op on nulls): the destructor and
    /// a throw during construction.
    void release() noexcept;

    LiteRtEnvironment m_env = nullptr;
    LiteRtModel m_model = nullptr;
    LiteRtOptions m_options = nullptr;
    LiteRtCompiledModel m_compiled_model = nullptr;

    std::vector<LiteRtTensorBuffer> m_input_buffers;   ///< in the signature's index order
    std::vector<LiteRtTensorBuffer> m_output_buffers;  ///< likewise
    std::vector<size_t> m_buffer_of_input_slot;        ///< per input slot: its buffer's index
    std::vector<size_t> m_buffer_of_output_slot;       ///< per output slot
    std::vector<size_t> m_input_elements;              ///< per input slot
    std::vector<size_t> m_output_elements;             ///< per output slot
};

Instance::Instance(const Model& model) {
    // Any litert_check below can throw; if it does mid-construction the destructor never
    // runs, so release the handles created so far before propagating.
    try {
        // Forward anira's log level to LiteRT's default logger, otherwise LiteRT spams INFO
        // logs on every environment creation and model compilation. The logger C API is not
        // exported from the prebuilt runtime, so this env option is the only route. Severity
        // values follow litert/c/internal/litert_logging.h: verbose=0, info=1, warning=2,
        // error=3 — numerically identical to anira's LogLevel enum, with Debug mapping to
        // LiteRT's verbose severity.
        const auto litert_severity = static_cast<int64_t>(model.m_log_level);
        const std::array<LiteRtEnvOption, 1> env_options = {{
            {.tag = kLiteRtEnvOptionTagMinLoggerSeverity,
             .value = {.type = kLiteRtAnyTypeInt, .int_value = litert_severity}},
        }};
        litert_check(LiteRtCreateEnvironment(1, env_options.data(), &m_env),
                     "LiteRtCreateEnvironment");

        if (model.m_bytes != nullptr) {
            litert_check(
                LiteRtCreateModelFromBuffer(m_env, model.m_bytes, model.m_num_bytes, &m_model),
                "LiteRtCreateModelFromBuffer",
                ANIRA_ERROR_MODEL_LOAD,
                model_file::k_memory);
        } else {
            const std::string modelpath = model_file::require_readable(model.m_path, k_engine);
            litert_check(LiteRtCreateModelFromFile(m_env, modelpath.c_str(), &m_model),
                         "LiteRtCreateModelFromFile",
                         ANIRA_ERROR_MODEL_LOAD,
                         modelpath);
        }

        // CPU compilation, pinned to a single thread to match the other engines (anira gets
        // its parallelism from running several adapter instances). The prebuilt LiteRT
        // runtime does not export the LrtCpuOptions helper symbols, so the payload it would
        // emit is built directly: an "xnnpack"-identified opaque-options blob carrying
        // num_threads. This depends only on the core exported API
        // (LiteRtCreateOpaqueOptions / AddOpaqueOptions).
        litert_check(LiteRtCreateOptions(&m_options), "LiteRtCreateOptions");
        litert_check(LiteRtSetOptionsHardwareAccelerators(m_options, kLiteRtHwAcceleratorCpu),
                     "LiteRtSetOptionsHardwareAccelerators");

        LiteRtOpaqueOptions cpu_opaque = nullptr;
        const char* const cpu_opts_toml = "num_threads = 1\n";  // freed by the deleter below
        const size_t cpu_opts_len = std::strlen(cpu_opts_toml) + 1;
        char* cpu_payload = static_cast<char*>(std::malloc(cpu_opts_len));
        if (cpu_payload == nullptr) {
            throw std::runtime_error(
                "[anira][LiteRT] out of memory allocating CPU options payload");
        }
        std::memcpy(cpu_payload, cpu_opts_toml, cpu_opts_len);
        // The opaque-options handle and its payload are locals until LiteRtAddOpaqueOptions
        // transfers ownership to m_options, so free them by hand on the failure branches
        // (release() only knows about the member handles).
        if (LiteRtCreateOpaqueOptions(
                "xnnpack",
                cpu_payload,
                [](void* p) { std::free(p); },
                &cpu_opaque) != kLiteRtStatusOk) {
            std::free(cpu_payload);  // the deleter never took ownership
            throw std::runtime_error("[anira][LiteRT] LiteRtCreateOpaqueOptions failed");
        }
        if (LiteRtAddOpaqueOptions(m_options, cpu_opaque) != kLiteRtStatusOk) {
            LiteRtDestroyOpaqueOptions(cpu_opaque);  // frees cpu_opaque + its payload
            throw std::runtime_error("[anira][LiteRT] LiteRtAddOpaqueOptions failed");
        }

        litert_check(LiteRtCreateCompiledModel(m_env, m_model, m_options, &m_compiled_model),
                     "LiteRtCreateCompiledModel");
    } catch (...) {
        release();
        throw;
    }
}

Instance::~Instance() {
    release();
}

void Instance::release() noexcept {
    for (auto& buffer : m_input_buffers) {
        if (buffer) { LiteRtDestroyTensorBuffer(buffer); }
    }
    m_input_buffers.clear();
    for (auto& buffer : m_output_buffers) {
        if (buffer) { LiteRtDestroyTensorBuffer(buffer); }
    }
    m_output_buffers.clear();
    if (m_compiled_model) {
        LiteRtDestroyCompiledModel(m_compiled_model);
        m_compiled_model = nullptr;
    }
    if (m_options) {
        LiteRtDestroyOptions(m_options);
        m_options = nullptr;
    }
    if (m_model) {
        LiteRtDestroyModel(m_model);
        m_model = nullptr;
    }
    if (m_env) {
        LiteRtDestroyEnvironment(m_env);
        m_env = nullptr;
    }
}

std::vector<EngineTensor> Instance::side_of(bool inputs) {
    LiteRtSignature signature = nullptr;
    litert_check(LiteRtGetModelSignature(m_model, 0, &signature),
                 "LiteRtGetModelSignature",
                 ANIRA_ERROR_MODEL_LOAD);
    LiteRtParamIndex count = 0;
    litert_check(inputs ? LiteRtGetNumSignatureInputs(signature, &count)
                        : LiteRtGetNumSignatureOutputs(signature, &count),
                 inputs ? "LiteRtGetNumSignatureInputs" : "LiteRtGetNumSignatureOutputs",
                 ANIRA_ERROR_MODEL_LOAD);
    std::vector<EngineTensor> tensors;
    tensors.reserve(count);
    for (LiteRtParamIndex i = 0; i < count; ++i) {
        const char* name = nullptr;
        litert_check(inputs ? LiteRtGetSignatureInputName(signature, i, &name)
                            : LiteRtGetSignatureOutputName(signature, i, &name),
                     inputs ? "LiteRtGetSignatureInputName" : "LiteRtGetSignatureOutputName",
                     ANIRA_ERROR_MODEL_LOAD);
        tensors.push_back(engine_tensor_of(signature, name, inputs));
    }
    return tensors;
}

void Instance::bind(const Model& model,
                    const std::vector<SlotBinding>& inputs,
                    const std::vector<SlotBinding>& outputs) {
    for (auto& buffer : m_input_buffers) {
        if (buffer) { LiteRtDestroyTensorBuffer(buffer); }
    }
    for (auto& buffer : m_output_buffers) {
        if (buffer) { LiteRtDestroyTensorBuffer(buffer); }
    }
    // One buffer per signature tensor, at the tensor's index: the exactly-once rule of the
    // binding made every index some slot's.
    m_input_buffers.assign(inputs.size(), nullptr);
    m_output_buffers.assign(outputs.size(), nullptr);
    m_buffer_of_input_slot.clear();
    m_buffer_of_output_slot.clear();
    m_input_elements.clear();
    m_output_elements.clear();
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        const TensorInfo& tensor = model.m_inputs[slot];
        const LiteRtRankedTensorType type = float32_type_of(tensor.m_dims);
        litert_check(LiteRtCreateManagedTensorBuffer(m_env,
                                                     kLiteRtTensorBufferTypeHostMemory,
                                                     &type,
                                                     tensor.m_num_elements * sizeof(float),
                                                     &m_input_buffers[inputs[slot].m_index]),
                     "LiteRtCreateManagedTensorBuffer (input)");
        m_buffer_of_input_slot.push_back(inputs[slot].m_index);
        m_input_elements.push_back(tensor.m_num_elements);
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        const TensorInfo& tensor = model.m_outputs[slot];
        const LiteRtRankedTensorType type = float32_type_of(tensor.m_dims);
        litert_check(LiteRtCreateManagedTensorBuffer(m_env,
                                                     kLiteRtTensorBufferTypeHostMemory,
                                                     &type,
                                                     tensor.m_num_elements * sizeof(float),
                                                     &m_output_buffers[outputs[slot].m_index]),
                     "LiteRtCreateManagedTensorBuffer (output)");
        m_buffer_of_output_slot.push_back(outputs[slot].m_index);
        m_output_elements.push_back(tensor.m_num_elements);
    }
}

void Instance::warm_up(uint32_t iterations) {
    if (iterations == 0) { return; }
    for (size_t slot = 0; slot < m_buffer_of_input_slot.size(); ++slot) {
        void* host = nullptr;
        LiteRtTensorBuffer buffer = m_input_buffers[m_buffer_of_input_slot[slot]];
        litert_check(LiteRtLockTensorBuffer(buffer, &host, kLiteRtTensorBufferLockModeWrite),
                     "LiteRtLockTensorBuffer (warm-up)");
        std::memset(host, 0, m_input_elements[slot] * sizeof(float));
        litert_check(LiteRtUnlockTensorBuffer(buffer), "LiteRtUnlockTensorBuffer (warm-up)");
    }
    for (uint32_t i = 0; i < iterations; ++i) {
        litert_check(LiteRtRunCompiledModel(m_compiled_model,
                                            /*signature_index=*/0,
                                            m_input_buffers.size(),
                                            m_input_buffers.data(),
                                            m_output_buffers.size(),
                                            m_output_buffers.data()),
                     "LiteRtRunCompiledModel (warm-up)");
    }
}

void Instance::run(const anira_engine_ctx& ctx) {
    if (ctx.inputs == nullptr || ctx.outputs == nullptr ||
        ctx.num_inputs != m_buffer_of_input_slot.size() ||
        ctx.num_outputs != m_buffer_of_output_slot.size()) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          std::string(k_engine) + ": the context carries " +
                              std::to_string(ctx.num_inputs) + " inputs and " +
                              std::to_string(ctx.num_outputs) + " outputs for a model of " +
                              std::to_string(m_buffer_of_input_slot.size()) + " and " +
                              std::to_string(m_buffer_of_output_slot.size()));
    }
    // In: every input block from its descriptor's memory, read on this call, into the managed
    // buffer of the signature tensor its slot binds.
    for (size_t slot = 0; slot < m_buffer_of_input_slot.size(); ++slot) {
        const float* data = host_f32_packed(ctx.inputs[slot], m_input_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_input_elements[slot]) + " elements");
        }
        LiteRtTensorBuffer buffer = m_input_buffers[m_buffer_of_input_slot[slot]];
        void* host = nullptr;
        litert_check(LiteRtLockTensorBuffer(buffer, &host, kLiteRtTensorBufferLockModeWrite),
                     "LiteRtLockTensorBuffer (input)");
        std::memcpy(host, data, m_input_elements[slot] * sizeof(float));
        litert_check(LiteRtUnlockTensorBuffer(buffer), "LiteRtUnlockTensorBuffer (input)");
    }

    litert_check(LiteRtRunCompiledModel(m_compiled_model,
                                        /*signature_index=*/0,
                                        m_input_buffers.size(),
                                        m_input_buffers.data(),
                                        m_output_buffers.size(),
                                        m_output_buffers.data()),
                 "LiteRtRunCompiledModel");

    // Out: every output buffer into its slot's descriptor memory.
    for (size_t slot = 0; slot < m_buffer_of_output_slot.size(); ++slot) {
        float* data = host_f32_packed(ctx.outputs[slot], m_output_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_output_elements[slot]) + " elements");
        }
        LiteRtTensorBuffer buffer = m_output_buffers[m_buffer_of_output_slot[slot]];
        void* host = nullptr;
        litert_check(LiteRtLockTensorBuffer(buffer, &host, kLiteRtTensorBufferLockModeRead),
                     "LiteRtLockTensorBuffer (output)");
        std::memcpy(data, host, m_output_elements[slot] * sizeof(float));
        litert_check(LiteRtUnlockTensorBuffer(buffer), "LiteRtUnlockTensorBuffer (output)");
    }
}

anira_status Instance::process(const anira_engine_ctx& ctx) noexcept {
    try {
        run(ctx);
        return ANIRA_OK;
    } catch (const StatusError& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_litert, "%s", e.what());
        return e.status();
    } catch (const std::exception& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_litert, "%s", e.what());
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_litert,
                           "litert threw a non-std exception out of LiteRtRunCompiledModel");
        return ANIRA_ERROR_ENGINE;
    }
}

class LiteRtAdapter final : public Adapter {
protected:
    void do_prepare(const Model& model) override {
        require_f32(model, k_engine);
        m_instances.clear();
        const uint32_t count = model.m_instances > 0 ? model.m_instances : 1U;
        for (uint32_t i = 0; i < count; ++i) {
            m_instances.push_back(std::make_unique<Instance>(model));
        }
        // The binding rule over the first instance's signature (every instance loaded the
        // same file), checked against the signature tensors' ranked types: a dynamic extent
        // of the file matches anything, and a static one must be the record's.
        Instance& first = *m_instances.front();
        const std::vector<EngineTensor> inputs = first.inputs();
        const std::vector<EngineTensor> outputs = first.outputs();
        const std::vector<SlotBinding> input_bindings =
            bind_side(model.m_inputs, inputs, inputs.size(), ExtentRule::Exact, k_engine, "input");
        const std::vector<SlotBinding> output_bindings = bind_side(model.m_outputs,
                                                                   outputs,
                                                                   outputs.size(),
                                                                   ExtentRule::Exact,
                                                                   k_engine,
                                                                   "output");
        for (const std::unique_ptr<Instance>& instance : m_instances) {
            instance->bind(model, input_bindings, output_bindings);
            instance->warm_up(model.m_warm_up);
        }
        set_bindings(bindings_of(input_bindings, output_bindings));
    }

    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept override {
        if (ctx.instance >= m_instances.size()) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        return m_instances[ctx.instance]->process(ctx);
    }

private:
    std::vector<std::unique_ptr<Instance>> m_instances;
};

}  // namespace

std::shared_ptr<Adapter> make_litert_adapter() {
    return std::make_shared<LiteRtAdapter>();
}

}  // namespace anira::backend

#endif  // USE_LITERT
