/*
 * The TensorFlow Lite adapter: one model (the flatbuffer parsed once) and one options object
 * per loaded model, shared by every executor, and one interpreter per executor, run through
 * the model's first signature (TfLiteSignatureRunner) when the file has one and through the
 * interpreter itself otherwise, the slots bound to the signature's inputs and outputs by name
 * where the entry's tensors record or the canonical name matches a key (a tensor name for a
 * file without signatures) and by the signature's index order (the interpreter's order)
 * otherwise, the inputs resized to the record's engine dims where the file's differ, every
 * tensor checked against its allocated dims at load, and every block copied from and into the
 * descriptors' memory with TfLiteTensorCopyFromBuffer / CopyToBuffer per call (the
 * interpreter's tensors need TFLite's own alignment; binding anira's memory in place waits
 * for the aligned chunk storage). reset re-initialises the interpreter's variable tensors.
 * File-local over anira::backend::Model: nothing of TensorFlow Lite enters a public header.
 */
#ifdef USE_TFLITE

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/Logger.h>
#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/core/c/c_api_types.h>
#include <tensorflow/lite/core/c/common.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

#ifdef _WIN32
#include <comdef.h>
#endif

// TfLiteInterpreterResetVariableTensors lives in tensorflow/lite/core/c/c_api_experimental.h,
// which includes tensorflow/lite/c/c_api_types.h, a path the pinned 2.17 include tree does not
// carry; the symbol is exported by the prebuilt library (checked with nm), so its prototype is
// declared here as that header declares it.
// NOLINTBEGIN(readability-identifier-naming) TFLite's C API spells the name
extern "C" TFL_CAPI_EXPORT TfLiteStatus
    TfLiteInterpreterResetVariableTensors(TfLiteInterpreter* interpreter);
// NOLINTEND(readability-identifier-naming)

namespace anira::backend {

namespace {

constexpr const char* k_engine = "tflite";

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
        default: return 0;
    }
}

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
    /// takes the pinned window's), allocates the tensors, keeps per slot the tensor it copies
    /// from or into, and checks every tensor's allocated dims and type against the record
    /// (the check of the binding rule: StatusError(ANIRA_ERROR_CONFIG) naming both shapes).
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The FIXED warm-up of the record (zeros in); a failing invoke is
    /// StatusError(ANIRA_ERROR_ENGINE).
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing call.
    anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept override;

    /// Re-initialises the model's variable tensors (the state a stateful export keeps between
    /// invocations); logged, unlatched, when the interpreter refuses.
    void reset(const anira_engine_ctx& ctx) noexcept override;

private:
    bool has_signature() const noexcept { return m_runner != nullptr; }
    TfLiteTensor* input_tensor(const std::string& name, size_t index);
    const TfLiteTensor* output_tensor(const std::string& name, size_t index);
    TfLiteStatus resize_input(const std::string& name, size_t index, const std::vector<int>& dims);
    TfLiteStatus invoke();
    /// Copies every input descriptor into its tensor, invokes, copies every output tensor into
    /// its descriptor; throws StatusError.
    void run(const anira_engine_ctx& ctx);

    std::shared_ptr<const SharedModel> m_shared;  ///< the model and the options, shared
    TfLiteInterpreter* m_interpreter = nullptr;
    TfLiteSignatureRunner* m_runner = nullptr;  ///< the first signature's; NULL without one

    std::vector<TfLiteTensor*> m_inputs;         ///< per input slot: the tensor it fills
    std::vector<const TfLiteTensor*> m_outputs;  ///< per output slot: the tensor it reads
    std::vector<size_t> m_input_elements;
    std::vector<size_t> m_output_elements;
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
    tflite_check(has_signature() ? TfLiteSignatureRunnerAllocateTensors(m_runner)
                                 : TfLiteInterpreterAllocateTensors(m_interpreter),
                 has_signature() ? "TfLiteSignatureRunnerAllocateTensors"
                                 : "TfLiteInterpreterAllocateTensors");
    // The check of the binding rule against the allocated tensors, and the per-slot tensors.
    m_inputs.clear();
    m_outputs.clear();
    m_input_elements.clear();
    m_output_elements.clear();
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
    }
}

void Instance::warm_up(uint32_t iterations) {
    if (iterations == 0) { return; }
    for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
        const std::vector<float> zeros(m_input_elements[slot], 0.F);
        tflite_check(TfLiteTensorCopyFromBuffer(m_inputs[slot],
                                                zeros.data(),
                                                m_input_elements[slot] * sizeof(float)),
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
    for (size_t slot = 0; slot < m_inputs.size(); ++slot) {
        const float* data = host_f32_packed(ctx.inputs[slot], m_input_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": input slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_input_elements[slot]) + " elements");
        }
        tflite_check(TfLiteTensorCopyFromBuffer(m_inputs[slot],
                                                data,
                                                m_input_elements[slot] * sizeof(float)),
                     "TfLiteTensorCopyFromBuffer");
    }
    tflite_check(invoke(), "TfLiteInterpreterInvoke");
    for (size_t slot = 0; slot < m_outputs.size(); ++slot) {
        float* data = host_f32_packed(ctx.outputs[slot], m_output_elements[slot]);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  " is not a packed float32 host block of " +
                                  std::to_string(m_output_elements[slot]) + " elements");
        }
        tflite_check(TfLiteTensorCopyToBuffer(m_outputs[slot],
                                              data,
                                              m_output_elements[slot] * sizeof(float)),
                     "TfLiteTensorCopyToBuffer");
    }
}

anira_status Instance::process(const anira_engine_ctx& ctx, ChunkBuffers* /*chunk*/) noexcept {
    try {
        run(ctx);
        return ANIRA_OK;
    } catch (const StatusError& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_tflite, "%s", e.what());
        return e.status();
    } catch (const std::exception& e) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_tflite, "%s", e.what());
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_tflite,
                           "tflite threw a non-std exception out of the invoke");
        return ANIRA_ERROR_ENGINE;
    }
}

void Instance::reset(const anira_engine_ctx& /*ctx*/) noexcept {
    if (TfLiteInterpreterResetVariableTensors(m_interpreter) != kTfLiteOk) {
        ANIRA_LOG_RT_ERROR(log_group::k_backend_tflite,
                           "TfLiteInterpreterResetVariableTensors failed; the model's variable "
                           "tensors keep their values into the new stream");
    }
}

/// The loaded model: the model and the options every executor shares (SharedModel, loaded
/// once), the names of a probe interpreter over them bind the slots, the check against the
/// dims runs in bind, after the resize and the allocation, on every executor; the probe is
/// the executor of shared slot 0 or, for a record without a shared slot, the spare of the
/// first exclusive session.
class TFLiteLoaded final : public ExecutorLoaded {
public:
    std::string provider_reason() const override {
        return "the TensorFlow Lite C API of this build ships no delegate factory, so the "
               "adapter runs the default provider alone in this pre-release";
    }

protected:
    void do_load(const Model& model) override {
        require_f32(model, k_engine);
        m_shared = std::make_shared<const SharedModel>(model);
        auto probe = std::make_unique<Instance>(m_shared);
        const std::vector<std::string> in_names = probe->input_names();
        const std::vector<std::string> out_names = probe->output_names();
        m_input_bindings = bind_slots(model.m_inputs, in_names, in_names.size(), k_engine, "input");
        m_output_bindings =
            bind_slots(model.m_outputs, out_names, out_names.size(), k_engine, "output");
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

std::shared_ptr<Loaded> make_tflite_loaded() {
    return std::make_shared<TFLiteLoaded>();
}

}  // namespace anira::backend

#endif  // USE_TFLITE
