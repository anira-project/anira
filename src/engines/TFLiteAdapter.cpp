/*
 * The TensorFlow Lite adapter: one model (the flatbuffer parsed once) and one options object
 * per loaded model, shared by every executor, and one interpreter per executor, run through
 * the model's first signature (TfLiteSignatureRunner) when the file has one and through the
 * interpreter itself otherwise, the slots bound to the signature's inputs and outputs by name
 * where the entry's tensors record or the canonical name matches a key (a tensor name for a
 * file without signatures) and by the signature's key order (the runner's names, sorted)
 * otherwise, the inputs resized to the record's engine dims where the file's differ, every
 * tensor checked against its allocated dims and dtype at load, and every slot bound in place
 * per call: the slot's memory is the tensor's custom allocation
 * (TfLiteInterpreterSetCustomAllocationForTensor, then AllocateTensors, which re-plans nothing
 * when only an allocation moved) where it starts on TFLite's 64-byte boundary, as a chunk's
 * buffer does, and an aligned block of the executor's own, copied into or out of, where it does
 * not. reset re-initialises the interpreter's variable tensors.
 * File-local over anira::engine::Model: nothing of TensorFlow Lite enters a public header.
 */
#ifdef USE_TFLITE

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../capi/words.h"
#include "../scheduler/ChunkStorage.h"
#include "../scheduler/TensorRun.h"
#include "../utils/Logger.h"
#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

#ifdef _WIN32
#include <comdef.h>
#endif

// TfLiteInterpreterResetVariableTensors, the tensor indices of the interpreter's inputs and
// outputs and TfLiteInterpreterSetCustomAllocationForTensor live in
// tensorflow/lite/core/c/c_api_experimental.h, which includes tensorflow/lite/c/c_api_types.h,
// a path the pinned 2.17 include tree does not carry; the symbols are exported by the prebuilt
// library (checked with nm), so their prototypes are declared here as that header declares them.
// NOLINTBEGIN(readability-identifier-naming) TFLite's C API spells the names
extern "C" TFL_CAPI_EXPORT TfLiteStatus
    TfLiteInterpreterResetVariableTensors(TfLiteInterpreter* interpreter);
extern "C" TFL_CAPI_EXPORT int32_t
    TfLiteInterpreterGetInputTensorIndex(const TfLiteInterpreter* interpreter, int32_t input_index);
extern "C" TFL_CAPI_EXPORT int32_t
    TfLiteInterpreterGetOutputTensorIndex(const TfLiteInterpreter* interpreter,
                                          int32_t output_index);
extern "C" TFL_CAPI_EXPORT TfLiteStatus
    TfLiteInterpreterSetCustomAllocationForTensor(TfLiteInterpreter* interpreter,
                                                  int tensor_index,
                                                  const TfLiteCustomAllocation* allocation,
                                                  int64_t flags);
// NOLINTEND(readability-identifier-naming)

namespace anira::engine {

namespace {

constexpr const char* k_engine = anira::capi::engine_word(ANIRA_ENGINE_TFLITE);

// The TensorFlow Lite C API reports failure as a TfLiteStatus; on the control path it becomes
// a StatusError with the failing call named (ANIRA_ERROR_ENGINE unless the caller passes
// another).
void tflite_check(TfLiteStatus status,
                  const char* what,
                  anira_status failure = ANIRA_ERROR_ENGINE) {
    if (status != kTfLiteOk) {
        throw StatusError(failure,
                          model_file::message(
                              k_engine,
                              what,
                              "returned TfLiteStatus " + std::to_string(static_cast<int>(status))));
    }
}

// The anira dtype of a TFLite element type; 0 for one anira has no code for.
anira_dtype dtype_of(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32: return ANIRA_DTYPE_F32;
        case kTfLiteFloat64: return ANIRA_DTYPE_F64;
        case kTfLiteFloat16: return ANIRA_DTYPE_F16;
        case kTfLiteBFloat16: return ANIRA_DTYPE_BF16;
        case kTfLiteInt8: return ANIRA_DTYPE_I8;
        case kTfLiteUInt8: return ANIRA_DTYPE_U8;
        case kTfLiteInt16: return ANIRA_DTYPE_I16;
        case kTfLiteInt32: return ANIRA_DTYPE_I32;
        case kTfLiteInt64: return ANIRA_DTYPE_I64;
        case kTfLiteBool: return ANIRA_DTYPE_BOOL8;
        default: return 0;
    }
}

// Whether the runtime has an element type for an anira dtype: every dtype anira names; the
// validator refuses any other before a record reaches the adapter.
bool carries(anira_dtype dtype) noexcept {
    switch (dtype) {
        case ANIRA_DTYPE_F32:
        case ANIRA_DTYPE_F64:
        case ANIRA_DTYPE_F16:
        case ANIRA_DTYPE_BF16:
        case ANIRA_DTYPE_I8:
        case ANIRA_DTYPE_U8:
        case ANIRA_DTYPE_I16:
        case ANIRA_DTYPE_I32:
        case ANIRA_DTYPE_I64:
        case ANIRA_DTYPE_BOOL8: return true;
        default: return false;
    }
}

// TFLite's boundary for a custom allocation (kDefaultTensorAlignment of lite/util.h, which the
// C API does not carry; the doc of TfLiteInterpreterSetCustomAllocationForTensor names it).
constexpr size_t k_tensor_alignment = 64;

// One tensor as the interpreter holds it: the key it is reached by, its dims and its type.
// The dims are the signature's where the tensor has unknown extents (dims_signature, -1 for
// each of them: an output whose extent the interpreter resolves only at invoke reports its
// placeholder dims until then) and the current dims otherwise (after a resize and an
// allocation these are what the record's dims must be). The dims_signature field is read off
// the struct: TFLite 2.17's C API exposes it as a documented field and has no accessor.
EngineTensor engine_tensor_of(const TfLiteTensor* tensor, const std::string& name) {
    EngineTensor engine_tensor;
    engine_tensor.m_name = name;
    if (tensor == nullptr) {
        engine_tensor.m_type_word = "no tensor";
        return engine_tensor;
    }
    if (tensor->dims_signature != nullptr && tensor->dims_signature->size > 0) {
        for (int d = 0; d < tensor->dims_signature->size; ++d) {
            engine_tensor.m_dims.push_back(tensor->dims_signature->data[d]);
        }
    } else {
        for (int32_t d = 0; d < TfLiteTensorNumDims(tensor); ++d) {
            engine_tensor.m_dims.push_back(TfLiteTensorDim(tensor, d));
        }
    }
    engine_tensor.m_dtype = dtype_of(TfLiteTensorType(tensor));
    engine_tensor.m_type_word =
        "TfLiteType " + std::to_string(static_cast<int>(TfLiteTensorType(tensor)));
    // Quantisation parameters (the struct's field, as dims_signature is read, and the per-layer
    // parameters the C API reports) make the tensor's values scale * (q - zero point).
    engine_tensor.m_quantised = tensor->quantization.type != kTfLiteNoQuantization ||
                                TfLiteTensorQuantizationParams(tensor).scale != 0.F;
    return engine_tensor;
}

/// What one load shares between its executors: the model (the flatbuffer parsed once; the C
/// API leaves it to the caller once an interpreter was created over it, so every interpreter
/// is created over the one) and the interpreter options (one thread). Immutable after load,
/// held by the loaded model and by every executor, freed with the last of them; the bytes a
/// record loads from stay the record's owner's for the loaded model's life, as TFLite asks.
struct SharedModel {
    explicit SharedModel(const Model& model);
    ~SharedModel();
    SharedModel(const SharedModel&) = delete;
    SharedModel& operator=(const SharedModel&) = delete;
    SharedModel(SharedModel&&) = delete;
    SharedModel& operator=(SharedModel&&) = delete;

    TfLiteModel* m_model = nullptr;
    TfLiteInterpreterOptions* m_options = nullptr;
};

SharedModel::SharedModel(const Model& model) {
    // Note: the log level cannot be forwarded to this engine. TFLite logs through its
    // internal MinimalLogger, and the prebuilt TFLite C library does not export any symbol
    // to adjust its severity (checked libtensorflowlite_c.so 2.17).
    if (model.m_bytes != nullptr) {
        m_model = TfLiteModelCreate(model.m_bytes, model.m_num_bytes);
        if (m_model == nullptr) {
            throw StatusError(ANIRA_ERROR_MODEL_LOAD,
                              model_file::message(k_engine,
                                                  model_file::k_memory,
                                                  "TfLiteModelCreate returned NULL (not a "
                                                  "TensorFlow Lite flatbuffer)"));
        }
    } else {
        const std::string modelpath = model_file::require_readable(model.m_path, k_engine);
        m_model = TfLiteModelCreateFromFile(modelpath.c_str());
        if (m_model == nullptr) {
            throw StatusError(ANIRA_ERROR_MODEL_LOAD,
                              model_file::message(k_engine,
                                                  modelpath,
                                                  "TfLiteModelCreateFromFile returned NULL (not "
                                                  "a TensorFlow Lite flatbuffer)"));
        }
    }
    m_options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(m_options, 1);
}

SharedModel::~SharedModel() {
    if (m_options != nullptr) { TfLiteInterpreterOptionsDelete(m_options); }
    if (m_model != nullptr) { TfLiteModelDelete(m_model); }
}

/// One interpreter over the shared model, with the signature runner of the file's first
/// signature when it has one: what one executor of the adapter runs on. Created over the
/// shared model at construction; resizes, allocates and binds at bind; runs every inference
/// over the context's descriptors.
class Instance final : public Executor {
public:
    explicit Instance(std::shared_ptr<const SharedModel> shared);
    ~Instance() override;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The names of either side, in the signature's index order (the runner's keys), or the
    /// interpreter's order with the tensor names for a file without a signature.
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

    /// Resizes every input the record's engine dims differ from (a dynamic extent of the file
    /// takes the pinned window's), allocates the tensors, checks every tensor's allocated dims
    /// and type against the record (the check of the binding rule:
    /// StatusError(ANIRA_ERROR_CONFIG) naming both shapes; a quantised tensor is
    /// ANIRA_ERROR_NOT_SUPPORTED), and keeps per slot the tensor, its index among the
    /// interpreter's tensors and an aligned block of the executor's own that is the tensor's
    /// custom allocation until a call binds the slot's memory (a tensor that takes none, a
    /// signature tensor that is not one of the interpreter's inputs or outputs, is copied
    /// through instead).
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// Whether the slot's memory is bound in place per call (bind found its tensor's index and
    /// the tensor took a custom allocation).
    bool bound(bool is_input, size_t slot) const noexcept {
        return (is_input ? m_input_index : m_output_index)[slot] >= 0;
    }

    /// The FIXED warm-up of the record (zeros in); a failing invoke is
    /// StatusError(ANIRA_ERROR_ENGINE).
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing call.
    anira_status process(const anira_engine_ctx& ctx) noexcept override;

    /// Re-initialises the model's variable tensors (the state a stateful export keeps between
    /// invocations); logged, unlatched, when the interpreter refuses.
    void reset(const anira_engine_ctx& ctx) noexcept override;

private:
    bool has_signature() const noexcept { return m_runner != nullptr; }
    TfLiteTensor* input_tensor(const std::string& name, size_t index);
    const TfLiteTensor* output_tensor(const std::string& name, size_t index);
    TfLiteStatus resize_input(const std::string& name, size_t index, const std::vector<int>& dims);
    TfLiteStatus invoke();
    TfLiteStatus allocate();
    /// The index among the interpreter's tensors of one of its inputs (or outputs), matched by
    /// pointer (the runner's tensors are the interpreter's for the first signature's subgraph);
    /// -1 for none.
    int interpreter_index(const TfLiteTensor* tensor, bool is_input) const;
    /// Makes `memory` the custom allocation of the slot's tensor; false when the interpreter
    /// refuses it.
    bool set_allocation(int index, void* memory, size_t bytes);
    /// Binds every slot's memory as its tensor's allocation (an aligned block of the
    /// executor's own, copied into or out of, where the memory is off TFLite's boundary; a copy
    /// through the tensor for a slot bind could not bind), invokes, copies what was not bound
    /// in place out; throws StatusError.
    void run(const anira_engine_ctx& ctx);

    std::shared_ptr<const SharedModel> m_shared;  ///< the model and the options, shared
    TfLiteInterpreter* m_interpreter = nullptr;
    TfLiteSignatureRunner* m_runner = nullptr;  ///< the first signature's; NULL without one

    std::vector<TfLiteTensor*> m_inputs;         ///< per input slot: the tensor it fills
    std::vector<const TfLiteTensor*> m_outputs;  ///< per output slot: the tensor it reads
    std::vector<size_t> m_input_elements;
    std::vector<size_t> m_output_elements;
    std::vector<anira_dtype> m_input_dtypes;  ///< per input slot: the record's dtype
    std::vector<anira_dtype> m_output_dtypes;
    std::vector<size_t> m_input_bytes;  ///< per input slot: elements times the dtype's size
    std::vector<size_t> m_output_bytes;
    std::vector<int> m_input_index;  ///< per input slot: its tensor's index; -1 copies through it
    std::vector<int> m_output_index;
    std::vector<ChunkBlock> m_input_blocks;  ///< per input slot: the executor's aligned block
    std::vector<ChunkBlock> m_output_blocks;
    /// Per output slot, this call's memory the result must end up in: the slot's own where it
    /// was bound in place (nothing to copy unless the runtime moved the tensor), else the
    /// descriptor's memory a copy goes to.
    std::vector<void*> m_output_target;
};

Instance::Instance(std::shared_ptr<const SharedModel> shared) : m_shared(std::move(shared)) {
    m_interpreter = TfLiteInterpreterCreate(m_shared->m_model, m_shared->m_options);
    if (m_interpreter == nullptr) {
        throw StatusError(ANIRA_ERROR_ENGINE,
                          model_file::message(k_engine,
                                              "TfLiteInterpreterCreate",
                                              "returned NULL (the model's operators are not "
                                              "supported by this runtime)"));
    }
    // The first signature when the file has one: its keys are the names the slots bind to.
    if (TfLiteInterpreterGetSignatureCount(m_interpreter) > 0) {
        const char* key = TfLiteInterpreterGetSignatureKey(m_interpreter, 0);
        m_runner = TfLiteInterpreterGetSignatureRunner(m_interpreter, key);
        if (m_runner == nullptr) {
            TfLiteInterpreterDelete(m_interpreter);  // no destructor runs for a throw from here
            m_interpreter = nullptr;
            throw StatusError(ANIRA_ERROR_ENGINE,
                              model_file::message(k_engine,
                                                  "TfLiteInterpreterGetSignatureRunner",
                                                  std::string("returned NULL for the signature '") +
                                                      (key != nullptr ? key : "") + "'"));
        }
    }
}

Instance::~Instance() {
    if (m_runner != nullptr) { TfLiteSignatureRunnerDelete(m_runner); }
    if (m_interpreter != nullptr) { TfLiteInterpreterDelete(m_interpreter); }
}

std::vector<std::string> Instance::input_names() const {
    std::vector<std::string> names;
    if (has_signature()) {
        const size_t count = TfLiteSignatureRunnerGetInputCount(m_runner);
        for (size_t i = 0; i < count; ++i) {
            names.emplace_back(
                TfLiteSignatureRunnerGetInputName(m_runner, static_cast<int32_t>(i)));
        }
        return names;
    }
    const int32_t count = TfLiteInterpreterGetInputTensorCount(m_interpreter);
    for (int32_t i = 0; i < count; ++i) {
        const char* name = TfLiteTensorName(TfLiteInterpreterGetInputTensor(m_interpreter, i));
        names.emplace_back(name != nullptr ? name : "");
    }
    return names;
}

std::vector<std::string> Instance::output_names() const {
    std::vector<std::string> names;
    if (has_signature()) {
        const size_t count = TfLiteSignatureRunnerGetOutputCount(m_runner);
        for (size_t i = 0; i < count; ++i) {
            names.emplace_back(
                TfLiteSignatureRunnerGetOutputName(m_runner, static_cast<int32_t>(i)));
        }
        return names;
    }
    const int32_t count = TfLiteInterpreterGetOutputTensorCount(m_interpreter);
    for (int32_t i = 0; i < count; ++i) {
        const char* name = TfLiteTensorName(TfLiteInterpreterGetOutputTensor(m_interpreter, i));
        names.emplace_back(name != nullptr ? name : "");
    }
    return names;
}

TfLiteTensor* Instance::input_tensor(const std::string& name, size_t index) {
    return has_signature()
               ? TfLiteSignatureRunnerGetInputTensor(m_runner, name.c_str())
               : TfLiteInterpreterGetInputTensor(m_interpreter, static_cast<int32_t>(index));
}

const TfLiteTensor* Instance::output_tensor(const std::string& name, size_t index) {
    return has_signature()
               ? TfLiteSignatureRunnerGetOutputTensor(m_runner, name.c_str())
               : TfLiteInterpreterGetOutputTensor(m_interpreter, static_cast<int32_t>(index));
}

TfLiteStatus Instance::resize_input(const std::string& name,
                                    size_t index,
                                    const std::vector<int>& dims) {
    return has_signature()
               ? TfLiteSignatureRunnerResizeInputTensor(m_runner,
                                                        name.c_str(),
                                                        dims.data(),
                                                        static_cast<int32_t>(dims.size()))
               : TfLiteInterpreterResizeInputTensor(m_interpreter,
                                                    static_cast<int32_t>(index),
                                                    dims.data(),
                                                    static_cast<int32_t>(dims.size()));
}

TfLiteStatus Instance::invoke() {
    return has_signature() ? TfLiteSignatureRunnerInvoke(m_runner)
                           : TfLiteInterpreterInvoke(m_interpreter);
}

TfLiteStatus Instance::allocate() {
    return has_signature() ? TfLiteSignatureRunnerAllocateTensors(m_runner)
                           : TfLiteInterpreterAllocateTensors(m_interpreter);
}

int Instance::interpreter_index(const TfLiteTensor* tensor, bool is_input) const {
    const int32_t count = is_input ? TfLiteInterpreterGetInputTensorCount(m_interpreter)
                                   : TfLiteInterpreterGetOutputTensorCount(m_interpreter);
    for (int32_t i = 0; i < count; ++i) {
        const TfLiteTensor* candidate = is_input
                                            ? TfLiteInterpreterGetInputTensor(m_interpreter, i)
                                            : TfLiteInterpreterGetOutputTensor(m_interpreter, i);
        if (candidate == tensor) {
            return is_input ? TfLiteInterpreterGetInputTensorIndex(m_interpreter, i)
                            : TfLiteInterpreterGetOutputTensorIndex(m_interpreter, i);
        }
    }
    return -1;
}

bool Instance::set_allocation(int index, void* memory, size_t bytes) {
    const TfLiteCustomAllocation allocation{.data = memory, .bytes = bytes};
    return TfLiteInterpreterSetCustomAllocationForTensor(m_interpreter,
                                                         index,
                                                         &allocation,
                                                         kTfLiteCustomAllocationFlagsNone) ==
           kTfLiteOk;
}

void Instance::bind(const Model& model,
                    const std::vector<SlotBinding>& inputs,
                    const std::vector<SlotBinding>& outputs) {
    const std::vector<std::string> in_names = input_names();
    const std::vector<std::string> out_names = output_names();
    // Every input the record's engine dims differ from is resized to them (a dynamic extent
    // of the file takes the pinned window's; a static one that differs is refused by the
    // runtime, and the check below says so with both shapes); then the tensors are allocated.
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        const std::string& name = in_names[inputs[slot].m_index];
        const TfLiteTensor* tensor = input_tensor(name, inputs[slot].m_index);
        if (tensor == nullptr) { continue; }  // the check below names it
        const std::vector<int64_t>& dims = model.m_inputs[slot].m_dims;
        bool same = static_cast<size_t>(TfLiteTensorNumDims(tensor)) == dims.size();
        for (int32_t d = 0; same && d < TfLiteTensorNumDims(tensor); ++d) {
            same = TfLiteTensorDim(tensor, d) == dims[d];
        }
        if (same) { continue; }
        const std::vector<int> wanted(dims.begin(), dims.end());
        if (resize_input(name, inputs[slot].m_index, wanted) != kTfLiteOk) {
            check_engine_tensor(model.m_inputs[slot],
                                engine_tensor_of(tensor, name),
                                ExtentRule::Exact,
                                k_engine,
                                "input");
            throw StatusError(ANIRA_ERROR_CONFIG,
                              std::string(k_engine) + ": input tensor '" +
                                  model.m_inputs[slot].m_name +
                                  "': the interpreter refused to resize '" + name +
                                  "' to the spec's engine dims");
        }
    }
    tflite_check(allocate(),
                 has_signature() ? "TfLiteSignatureRunnerAllocateTensors"
                                 : "TfLiteInterpreterAllocateTensors");
    // The check of the binding rule against the allocated tensors, and the per-slot tensors.
    m_inputs.clear();
    m_outputs.clear();
    m_input_elements.clear();
    m_output_elements.clear();
    m_input_dtypes.clear();
    m_output_dtypes.clear();
    m_input_bytes.clear();
    m_output_bytes.clear();
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        const std::string& name = in_names[inputs[slot].m_index];
        TfLiteTensor* tensor = input_tensor(name, inputs[slot].m_index);
        check_engine_tensor(model.m_inputs[slot],
                            engine_tensor_of(tensor, name),
                            ExtentRule::Exact,
                            k_engine,
                            "input");
        m_inputs.push_back(tensor);
        m_input_elements.push_back(model.m_inputs[slot].m_num_elements);
        m_input_dtypes.push_back(model.m_inputs[slot].m_dtype);
        m_input_bytes.push_back(model.m_inputs[slot].m_num_elements *
                                tensor_run::dtype_size(model.m_inputs[slot].m_dtype));
    }
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        const std::string& name = out_names[outputs[slot].m_index];
        const TfLiteTensor* tensor = output_tensor(name, outputs[slot].m_index);
        check_engine_tensor(model.m_outputs[slot],
                            engine_tensor_of(tensor, name),
                            ExtentRule::Exact,
                            k_engine,
                            "output");
        m_outputs.push_back(tensor);
        m_output_elements.push_back(model.m_outputs[slot].m_num_elements);
        m_output_dtypes.push_back(model.m_outputs[slot].m_dtype);
        m_output_bytes.push_back(model.m_outputs[slot].m_num_elements *
                                 tensor_run::dtype_size(model.m_outputs[slot].m_dtype));
    }
    // In place: each tensor's index among the interpreter's, and the executor's aligned block as
    // its custom allocation until a call binds the slot's memory; a tensor without an index or
    // one the interpreter refuses the allocation for is copied through instead.
    const auto bind_side = [this](bool is_input) {
        const size_t count = is_input ? m_inputs.size() : m_outputs.size();
        std::vector<int>& indices = is_input ? m_input_index : m_output_index;
        std::vector<ChunkBlock>& blocks = is_input ? m_input_blocks : m_output_blocks;
        const std::vector<size_t>& bytes = is_input ? m_input_bytes : m_output_bytes;
        indices.assign(count, -1);
        blocks.clear();
        blocks.reserve(count);
        for (size_t slot = 0; slot < count; ++slot) {
            blocks.emplace_back(host_chunk_storage(), bytes[slot], k_tensor_alignment);
            const TfLiteTensor* tensor =
                is_input ? static_cast<const TfLiteTensor*>(m_inputs[slot]) : m_outputs[slot];
            const int index = interpreter_index(tensor, is_input);
            if (index >= 0 && set_allocation(index, blocks[slot].data(), bytes[slot])) {
                indices[slot] = index;
            }
        }
    };
    bind_side(true);
    bind_side(false);
    tflite_check(allocate(),
                 has_signature() ? "TfLiteSignatureRunnerAllocateTensors (custom allocations)"
                                 : "TfLiteInterpreterAllocateTensors (custom allocations)");
    m_output_target.assign(m_outputs.size(), nullptr);
}

void Instance::warm_up(uint32_t iterations) {
    if (iterations == 0) { return; }
    for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
        const std::vector<std::byte> zeros(m_input_bytes[slot], std::byte{0});
        tflite_check(TfLiteTensorCopyFromBuffer(m_inputs[slot], zeros.data(), zeros.size()),
                     "TfLiteTensorCopyFromBuffer (warm-up)");
    }
    for (uint32_t i = 0; i < iterations; ++i) {
        // A warm-up that fails fails prepare: the model cannot run.
        tflite_check(invoke(), "TfLiteInterpreterInvoke (warm-up)");
    }
}

void Instance::run(const anira_engine_ctx& ctx) {
    if (ctx.inputs == nullptr || ctx.outputs == nullptr || ctx.num_inputs != m_inputs.size() ||
        ctx.num_outputs != m_outputs.size()) {
        throw StatusError(
            ANIRA_ERROR_INVALID_ARGUMENT,
            std::string(k_engine) + ": the context carries " + std::to_string(ctx.num_inputs) +
                " inputs and " + std::to_string(ctx.num_outputs) + " outputs for a model of " +
                std::to_string(m_inputs.size()) + " and " + std::to_string(m_outputs.size()));
    }
    const auto packed = [this](const anira_tensor& tensor, bool is_input, size_t slot) {
        const anira_dtype dtype = is_input ? m_input_dtypes[slot] : m_output_dtypes[slot];
        const size_t elements = is_input ? m_input_elements[slot] : m_output_elements[slot];
        void* data = host_packed(tensor, dtype, elements);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": " + (is_input ? "input" : "output") +
                                  " slot " + std::to_string(slot) + " is not a packed " +
                                  capi::dtype_word(dtype) + " host block of " +
                                  std::to_string(elements) + " elements");
        }
        return data;
    };
    // In: the slot's memory is the tensor's allocation where it is on TFLite's boundary, else
    // it is copied into the executor's block (the tensor's allocation then), or through the
    // tensor for a slot bind could not bind.
    for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
        void* data = packed(ctx.inputs[slot], true, slot);
        const size_t bytes = m_input_bytes[slot];
        const int index = m_input_index[slot];
        if (index >= 0 && aligned_to(data, k_tensor_alignment) &&
            set_allocation(index, data, bytes)) {
            continue;
        }
        if (index >= 0) {
            std::memcpy(m_input_blocks[slot].data(), data, bytes);
            if (!set_allocation(index, m_input_blocks[slot].data(), bytes)) {
                throw StatusError(ANIRA_ERROR_ENGINE,
                                  std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                      ": the interpreter refused the executor's own block");
            }
        } else {
            tflite_check(TfLiteTensorCopyFromBuffer(m_inputs[slot], data, bytes),
                         "TfLiteTensorCopyFromBuffer");
        }
        count_copy_in();
    }
    // Out: likewise; what is not bound in place is copied out after the invoke.
    for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
        void* data = packed(ctx.outputs[slot], false, slot);
        m_output_target[slot] = data;
        const int index = m_output_index[slot];
        if (index < 0) { continue; }
        if (aligned_to(data, k_tensor_alignment) &&
            set_allocation(index, data, m_output_bytes[slot])) {
            continue;
        }
        if (!set_allocation(index, m_output_blocks[slot].data(), m_output_bytes[slot])) {
            throw StatusError(ANIRA_ERROR_ENGINE,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  ": the interpreter refused the executor's own block");
        }
    }
    tflite_check(allocate(), "AllocateTensors (the call's allocations)");
    tflite_check(invoke(), "TfLiteInterpreterInvoke");
    // A result the runtime left elsewhere than the slot's memory (the executor's block, the
    // interpreter's own tensor, or a tensor an in-place operator re-pointed) is copied.
    for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
        const void* result = TfLiteTensorData(m_outputs[slot]);
        if (result == m_output_target[slot]) { continue; }
        std::memcpy(m_output_target[slot], result, m_output_bytes[slot]);
        count_copy_out();
    }
}

anira_status Instance::process(const anira_engine_ctx& ctx) noexcept {
    try {
        run(ctx);
        m_failures.succeeded();
        return ANIRA_OK;
    } catch (const StatusError& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_tflite, "%s", e.what());
        }
        return e.status();
    } catch (const std::exception& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_tflite, "%s", e.what());
        }
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_tflite,
                               "tflite threw a non-std exception out of the invoke");
        }
        return ANIRA_ERROR_ENGINE;
    }
}

void Instance::reset(const anira_engine_ctx& /*ctx*/) noexcept {
    if (TfLiteInterpreterResetVariableTensors(m_interpreter) != kTfLiteOk) {
        ANIRA_LOG_RT_ERROR(log_group::k_engine_tflite,
                           "TfLiteInterpreterResetVariableTensors failed; the model's variable "
                           "tensors keep their values into the new stream");
    }
}

/// The engine object: nothing per process, since the prebuilt TensorFlow Lite runtime has no
/// global state anira sets (no runtime logger; the thread count is the interpreter options').
/// The core keeps one object per engine (Core::builtin_engine); every loaded model holds it.
class TFLiteEngine final : public BuiltinEngine {
public:
    TFLiteEngine() : BuiltinEngine(ANIRA_ENGINE_TFLITE) {}
};

/// The loaded model: the model and the options every executor shares (SharedModel, loaded
/// once), the names of a probe interpreter over them bind the slots, the check against the
/// dims runs in bind, after the resize and the allocation, on every executor; the probe is
/// the executor of shared slot 0 or, for a record without a shared slot, the spare of the
/// first exclusive session.
class TFLiteLoaded final : public ExecutorLoaded {
public:
    explicit TFLiteLoaded(std::shared_ptr<TFLiteEngine> engine)
        : ExecutorLoaded(std::move(engine)) {}

    /// A slot is bound in place (its memory the tensor's custom allocation per call) where the
    /// probe's bind could; one whose tensor takes no custom allocation is copied through it.
    SlotEdge slot_edge(bool is_input, uint32_t slot) const noexcept override {
        const std::vector<bool>& bound = is_input ? m_input_bound : m_output_bound;
        if (slot < bound.size() && bound[slot]) { return SlotEdge{}; }
        return SlotEdge{.m_class = ANIRA_EDGE_HOST_COPY,
                        .m_reason =
                            "tflite: copied through the interpreter's own tensor (the "
                            "tensor takes no custom allocation)"};
    }

    /// TFLite's boundary for a custom allocation, which every chunk's buffer meets.
    size_t buffer_alignment() const noexcept override { return k_tensor_alignment; }

    std::string provider_reason() const override {
        return "the TensorFlow Lite C API of this build ships no delegate factory, so the "
               "adapter runs the CPU path alone in this pre-release";
    }

protected:
    void do_load(const Model& model) override {
        require_dtypes(model, k_engine, carries);
        m_shared = std::make_shared<const SharedModel>(model);
        auto probe = std::make_unique<Instance>(m_shared);
        const std::vector<std::string> in_names = probe->input_names();
        const std::vector<std::string> out_names = probe->output_names();
        m_input_bindings = bind_slots(model.m_inputs, in_names, in_names.size(), k_engine, "input");
        m_output_bindings =
            bind_slots(model.m_outputs, out_names, out_names.size(), k_engine, "output");
        probe->bind(model, m_input_bindings, m_output_bindings);
        m_input_bound.assign(model.m_inputs.size(), false);
        m_output_bound.assign(model.m_outputs.size(), false);
        for (size_t slot = 0; slot < model.m_inputs.size(); ++slot) {
            m_input_bound[slot] = probe->bound(true, slot);
        }
        for (size_t slot = 0; slot < model.m_outputs.size(); ++slot) {
            m_output_bound[slot] = probe->bound(false, slot);
        }
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
    std::vector<bool> m_input_bound;  ///< per input slot: bound in place (the probe's bind)
    std::vector<bool> m_output_bound;
};

}  // namespace

std::shared_ptr<BuiltinEngine> make_tflite_engine() {
    return std::make_shared<TFLiteEngine>();
}

std::shared_ptr<Loaded> make_tflite_loaded(const std::shared_ptr<BuiltinEngine>& engine) {
    std::shared_ptr<TFLiteEngine> own = std::dynamic_pointer_cast<TFLiteEngine>(engine);
    if (own == nullptr) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          "tflite: the engine object is not this adapter's");
    }
    return std::make_shared<TFLiteLoaded>(std::move(own));
}

}  // namespace anira::engine

#endif  // USE_TFLITE
