/*
 * The LiteRT adapter: one environment (carrying the log level), one model and one
 * compilation options object per loaded model, shared by every executor, and one compiled
 * model per executor, run through the model's first signature (LiteRT synthesizes one for a
 * file without signatures, keyed by the tensor names), the slots bound to the signature's
 * inputs and outputs by name where the entry's tensors record or the canonical name matches a
 * key and by the signature's index order otherwise, checked against the signature tensors'
 * ranked types at load, and one managed host buffer per signature tensor in the signature's
 * index order (what LiteRtRunCompiledModel takes), filled from and read into the descriptors'
 * memory through a lock and a memcpy per call (the managed buffers need LiteRT's own
 * alignment; binding anira's memory in place waits for the aligned chunk storage). File-local
 * over anira::backend::Model: nothing of LiteRT enters a public header.
 */
#ifdef USE_LITERT

#include <anira/CoreConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

// The accelerator query is internal to LiteRT and exported by every package but the Windows
// DLL (cmake/backends/litert.cmake defines ANIRA_LITERT_ACCELERATOR_QUERY where it is).
#if defined(ANIRA_LITERT_ACCELERATOR_QUERY)
#include "litert/c/internal/litert_accelerator.h"
#endif

namespace anira::backend {

namespace {

constexpr const char* k_engine = "litert";

// The accelerators LiteRT can take beyond the CPU, by the hardware they support, as the
// provider names anira gives them (custom providers beside ANIRA_PROVIDER_DEFAULT): "gpu",
// "npu" and, in the web build, "webnn". The CPU accelerator is the default provider.
#if defined(__EMSCRIPTEN__)
constexpr size_t k_hardware_count = 3;
#else
constexpr size_t k_hardware_count = 2;
#endif
constexpr std::array<std::pair<int, const char*>, k_hardware_count> k_hardware{{
    {kLiteRtHwAcceleratorGpu, "gpu"},
    {kLiteRtHwAcceleratorNpu, "npu"},
#if defined(__EMSCRIPTEN__)
    {kLiteRtHwAcceleratorWebNn, "webnn"},
#endif
}};

// The hardware bit of a provider name; 0 for a name the table lacks.
int hardware_of(std::string_view provider_id) noexcept {
    for (const auto& [hardware, name] : k_hardware) {
        if (provider_id == name) { return hardware; }
    }
    return 0;
}

#if defined(ANIRA_LITERT_ACCELERATOR_QUERY)
// Whether this LiteRT library can be asked which accelerators an environment registered.
constexpr bool k_accelerator_query = true;

// The hardware every accelerator registered to an environment supports, as one set.
LiteRtHwAcceleratorSet registered_hardware(LiteRtEnvironment env) noexcept {
    LiteRtHwAcceleratorSet registered = 0;
    LiteRtParamIndex count = 0;
    if (LiteRtGetNumAccelerators(env, &count) != kLiteRtStatusOk) { return registered; }
    for (LiteRtParamIndex index = 0; index < count; ++index) {
        LiteRtAccelerator accelerator = nullptr;
        LiteRtHwAcceleratorSet support = 0;
        if (LiteRtGetAccelerator(env, index, &accelerator) == kLiteRtStatusOk &&
            LiteRtGetAcceleratorHardwareSupport(accelerator, &support) == kLiteRtStatusOk) {
            registered |= support;
        }
    }
    return registered;
}
#else
constexpr bool k_accelerator_query = false;

// The CPU accelerator alone: this LiteRT library does not export its accelerator query, so no
// accelerator beyond the one every environment registers is known here.
LiteRtHwAcceleratorSet registered_hardware(LiteRtEnvironment /*env*/) noexcept {
    return kLiteRtHwAcceleratorCpu;
}
#endif

// What a message adds where the library cannot be asked for its accelerators.
constexpr const char* k_no_query_note =
    "; this LiteRT library does not export its accelerator query, so no accelerator beyond "
    "the CPU is known here";

// The names of a hardware set, as a message lists them ("cpu" for the CPU accelerator).
std::string hardware_names(LiteRtHwAcceleratorSet hardware) {
    std::string text;
    if ((hardware & kLiteRtHwAcceleratorCpu) != 0) { text = "cpu"; }
    for (const auto& [bit, name] : k_hardware) {
        if ((hardware & bit) == 0) { continue; }
        if (!text.empty()) { text += ", "; }
        text += name;
    }
    return text.empty() ? "none" : text;
}

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

/// What one load shares between its executors: the environment (with anira's log level), the
/// model (a file or the bytes of the record, which its owner keeps alive for the loaded
/// model's life, as LiteRT asks) and the compilation options (the CPU, one thread), which the
/// C API leaves to the caller once a compiled model was created over them. Immutable after
/// load, held by the loaded model and by every executor, freed with the last of them.
struct SharedModel {
    explicit SharedModel(const Model& model);
    ~SharedModel() { release(); }
    SharedModel(const SharedModel&) = delete;
    SharedModel& operator=(const SharedModel&) = delete;
    SharedModel(SharedModel&&) = delete;
    SharedModel& operator=(SharedModel&&) = delete;

    LiteRtEnvironment m_env = nullptr;
    LiteRtModel m_model = nullptr;
    LiteRtOptions m_options = nullptr;

private:
    /// Destroys the handles created so far (no-op on nulls): the destructor and a throw
    /// during construction.
    void release() noexcept;
};

SharedModel::SharedModel(const Model& model) {
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

        // The accelerator of the record: the CPU for the default provider, the hardware a
        // custom name spells else, and the environment must have registered one that supports
        // it (its automatic registration loads the accelerator libraries it finds), or the
        // load is refused here rather than at the compiled model.
        const int wanted = model.m_provider_id.empty() ? kLiteRtHwAcceleratorCpu
                                                       : hardware_of(model.m_provider_id);
        if (wanted != kLiteRtHwAcceleratorCpu) {
            const LiteRtHwAcceleratorSet registered = registered_hardware(m_env);
            if ((registered & wanted) == 0) {
                std::string message = "litert: no registered accelerator supports '";
                message += model.m_provider_id;
                message += "' here (the accelerators registered support: ";
                message += hardware_names(registered);
                if (!k_accelerator_query) { message += k_no_query_note; }
                message += ")";
                throw StatusError(ANIRA_ERROR_NOT_SUPPORTED, message);
            }
        }

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
        // its parallelism from running several executors). The prebuilt LiteRT runtime does
        // not export the LrtCpuOptions helper symbols, so the payload it would emit is built
        // directly: an "xnnpack"-identified opaque-options blob carrying num_threads. This
        // depends only on the core exported API (LiteRtCreateOpaqueOptions /
        // AddOpaqueOptions).
        litert_check(LiteRtCreateOptions(&m_options), "LiteRtCreateOptions");
        litert_check(LiteRtSetOptionsHardwareAccelerators(m_options, wanted),
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
    } catch (...) {
        release();
        throw;
    }
}

void SharedModel::release() noexcept {
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

/// One compiled model over the shared environment, model and options: what one executor of
/// the adapter runs on. Compiles at construction; creates its managed buffers at bind; runs
/// every inference over the context's descriptors through them.
class Instance final : public Executor {
public:
    explicit Instance(std::shared_ptr<const SharedModel> shared);
    ~Instance() override;
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
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing call.
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;

private:
    std::vector<EngineTensor> side_of(bool inputs);
    /// Copies every input descriptor into its buffer, runs, copies every output buffer into
    /// its descriptor; throws StatusError.
    void run(const anira_engine_ctx& ctx);
    /// Destroys every LiteRT handle this executor owns (no-op on nulls).
    void release() noexcept;

    std::shared_ptr<const SharedModel> m_shared;  ///< the environment, the model, the options
    LiteRtCompiledModel m_compiled_model = nullptr;

    std::vector<LiteRtTensorBuffer> m_input_buffers;   ///< in the signature's index order
    std::vector<LiteRtTensorBuffer> m_output_buffers;  ///< likewise
    std::vector<size_t> m_buffer_of_input_slot;        ///< per input slot: its buffer's index
    std::vector<size_t> m_buffer_of_output_slot;       ///< per output slot
    std::vector<size_t> m_input_elements;              ///< per input slot
    std::vector<size_t> m_output_elements;             ///< per output slot
};

Instance::Instance(std::shared_ptr<const SharedModel> shared) : m_shared(std::move(shared)) {
    litert_check(LiteRtCreateCompiledModel(m_shared->m_env,
                                           m_shared->m_model,
                                           m_shared->m_options,
                                           &m_compiled_model),
                 "LiteRtCreateCompiledModel");
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
}

std::vector<EngineTensor> Instance::side_of(bool inputs) {
    LiteRtSignature signature = nullptr;
    litert_check(LiteRtGetModelSignature(m_shared->m_model, 0, &signature),
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
        litert_check(LiteRtCreateManagedTensorBuffer(m_shared->m_env,
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
        litert_check(LiteRtCreateManagedTensorBuffer(m_shared->m_env,
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

anira_status Instance::process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept {
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

/// The loaded model: the environment, the model and the options every executor shares
/// (SharedModel, loaded once), the binding rule over a probe's signature, checked against the
/// signature tensors' ranked types (a dynamic extent of the file matches anything, a static
/// one must be the record's); the probe is the executor of shared slot 0 or, for a record
/// without a shared slot, the spare of the first exclusive session.
class LiteRtLoaded final : public ExecutorLoaded {
public:
    /// The default provider (the CPU accelerator), or an accelerator by its hardware's name
    /// beside ANIRA_PROVIDER_DEFAULT; whether the environment registers one is load's
    /// question. No provider of the enum names a LiteRT accelerator.
    bool serves(anira_provider provider, std::string_view provider_id) const noexcept override {
        if (provider != ANIRA_PROVIDER_DEFAULT) { return false; }
        return provider_id.empty() || hardware_of(provider_id) != 0;
    }

    std::string provider_reason() const override {
        std::string names;
        for (const auto& [hardware, name] : k_hardware) {
            if (!names.empty()) { names += ", "; }
            names += "'";
            names += name;
            names += "'";
        }
        std::string reason =
            "LiteRT takes an accelerator by the hardware's name beside ANIRA_PROVIDER_DEFAULT (";
        reason += names;
        reason += "); no provider of the enum names one";
        if (!k_accelerator_query) { reason += k_no_query_note; }
        return reason;
    }

protected:
    void do_load(const Model& model) override {
        require_f32(model, k_engine);
        m_shared = std::make_shared<const SharedModel>(model);
        auto probe = std::make_unique<Instance>(m_shared);
        const std::vector<EngineTensor> inputs = probe->inputs();
        const std::vector<EngineTensor> outputs = probe->outputs();
        m_input_bindings =
            bind_side(model.m_inputs, inputs, inputs.size(), ExtentRule::Exact, k_engine, "input");
        m_output_bindings = bind_side(model.m_outputs,
                                      outputs,
                                      outputs.size(),
                                      ExtentRule::Exact,
                                      k_engine,
                                      "output");
        probe->bind(model, m_input_bindings, m_output_bindings);
        set_bindings(bindings_of(m_input_bindings, m_output_bindings));
        adopt(std::move(probe));
    }

    std::unique_ptr<Executor> make_executor() override {
        auto executor = std::make_unique<Instance>(m_shared);
        executor->bind(model(), m_input_bindings, m_output_bindings);
        return executor;
    }

private:
    std::shared_ptr<const SharedModel> m_shared;  ///< freed with the last executor over it
    std::vector<SlotBinding> m_input_bindings;
    std::vector<SlotBinding> m_output_bindings;
};

}  // namespace

std::shared_ptr<Loaded> make_litert_loaded() {
    return std::make_shared<LiteRtLoaded>();
}

std::vector<ProviderInfo> litert_providers(anira::LogLevel level) {
    std::vector<ProviderInfo> providers;
    // A library that cannot be asked lists no accelerator: an environment is not worth creating.
    if (!k_accelerator_query) { return providers; }
    // A fresh environment registers the accelerators it can load (its automatic registration).
    // An accelerator it cannot load is this query's answer, not a warning: the environment's
    // logger keeps errors alone, whatever anira's level; the adapter's environments at load
    // keep anira's level, so a load that asks for an accelerator says why it failed.
    const auto severity =
        std::max(static_cast<int64_t>(level), static_cast<int64_t>(anira::LogLevel::Error));
    const std::array<LiteRtEnvOption, 1> env_options = {{
        {.tag = kLiteRtEnvOptionTagMinLoggerSeverity,
         .value = {.type = kLiteRtAnyTypeInt, .int_value = severity}},
    }};
    LiteRtEnvironment env = nullptr;
    if (LiteRtCreateEnvironment(1, env_options.data(), &env) != kLiteRtStatusOk || env == nullptr) {
        ANIRA_LOG_WARNING(anira::log_group::k_backend_litert,
                          "litert: no environment could be created to ask for the accelerators; "
                          "the capabilities list the default provider alone");
        return providers;
    }
    const LiteRtHwAcceleratorSet registered = registered_hardware(env);
    LiteRtDestroyEnvironment(env);
    for (const auto& [hardware, name] : k_hardware) {
        if ((registered & hardware) == 0) { continue; }
        ProviderInfo info;
        info.m_provider_id = name;
        providers.push_back(std::move(info));
    }
    return providers;
}

}  // namespace anira::backend

#endif  // USE_LITERT
