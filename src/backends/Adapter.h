#ifndef ANIRA_BACKENDS_ADAPTER_H
#define ANIRA_BACKENDS_ADAPTER_H
/*
 * The engine room's interface: what every engine anira runs looks like from the scheduler,
 * built in or registered, in the shape of the C descriptor (anira/abi/engine.h). Private to
 * src/backends and the scheduler (and the tests through the src/ include directory): nothing
 * here enters the ABI.
 *
 * One prepared model is one Adapter. prepare(model) loads it once (the core pools adapters by
 * model identity, engine id and carrier; a session-exclusive model gets its own), and
 * run(ctx, chunk, reset_first) runs one inference of one session on one instance of it: the
 * spin-and-claim loop the five 2.x processors each carried is here, once, and the claimed
 * instance's index travels in the context. The adapter rule of anira/abi/engine.h binds here
 * too: process reads every extent and every memory handle from the context's tensors on every
 * call, keeps nothing of them, and assumes no tensor to be anira's own buffer.
 */
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/system/Exports.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/Logger.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace anira::backend {

/// One tensor of a prepared model as its engine is handed it: the canonical name, the export's
/// name where the entry's tensors record names the slot (empty otherwise: the slot then binds
/// by its canonical name where the engine's side has one, else by its position; every name is
/// empty on the 2.x path), the engine's extents (the entry's layout applied to the spec's) at
/// the pinned window, the spec's dtype and the element count.
struct TensorInfo {
    std::string m_name;
    std::string m_engine_name;
    std::vector<int64_t> m_dims;
    anira_dtype m_dtype = ANIRA_DTYPE_F32;
    size_t m_num_elements = 0;

    bool operator==(const TensorInfo& other) const = default;
};

/// The record of one prepared model: everything an adapter reads at prepare, copied out of the
/// configuration it came from (a model entry and the validator's derived quantities on the C
/// path, an InferenceConfig on the 2.x path), so that a pooled adapter never aliases a
/// session's configuration (issue #76). Two records compare equal when they describe the same
/// model for the same engine with the same tensors, instances, warm-up and entry, and, for a
/// custom engine, the same variant: the pool's key, beside the carrier of a custom engine. The
/// bytes compare by identity, as the 2.x ModelData compares them; the bytes owner and the log level
/// are no part of the identity (the level is what the engine's environment is created with, and the
/// core reconciles one level for the process).
struct ANIRA_API Model {
    anira_engine m_engine = ANIRA_ENGINE_NONE;  ///< the built-in engine; NONE with m_engine_id
                                                ///< for a registered one, NONE alone for the
                                                ///< 2.x custom and roundtrip adapters
    std::string m_engine_id;                    ///< a registered engine's id, else empty
    std::string m_path;                         ///< the model file; empty for bytes
    const void* m_bytes = nullptr;              ///< the model bytes; NULL for a path
    size_t m_num_bytes = 0;
    /// Keeps the bytes alive while the adapter lives: an aliasing shared_ptr over the row's
    /// carrier on the C path; empty on the 2.x path, whose caller owns the bytes.
    std::shared_ptr<const void> m_bytes_owner;
    std::string m_entry;  ///< the method or function of the file to run; empty for its default
    /// A custom engine's variant as canonical JSON text (anira::capi::model_config_json):
    /// its prepare may read the whole model config, so its prepared models are shared over an
    /// equal one only. Empty for a built-in engine, which reads the record alone.
    std::string m_variant;
    std::vector<TensorInfo> m_inputs;   ///< in slot order, State tensors included
    std::vector<TensorInfo> m_outputs;  ///< in slot order, State tensors included
    uint32_t m_instances = 1;  ///< process calls that may run at once on the prepared model,
                               ///< each on its own instance below this count; 1 for a
                               ///< session-exclusive model
    uint32_t m_warm_up = 0;    ///< the inferences prepare runs before the first real one
    anira::LogLevel m_log_level = anira::LogLevel::Warning;  ///< the level in effect at prepare
    bool m_session_exclusive = false;  ///< prepared for its session alone, never pooled

    bool operator==(const Model& other) const;
};

/// The two 2.x buffer vectors of the chunk's struct, what the legacy adapter's 2.x virtual
/// takes; an adapter of the descriptor shape (a built-in engine of this line, a registered
/// engine) reads the context's tensors alone and never looks here. Non-const: the 2.x virtual
/// takes the inputs by non-const reference (two 2.x processors swap their memory), and the
/// legacy adapter copies a foreign input into them ahead of the call.
struct ChunkBuffers {
    std::vector<anira::BufferF>* m_inputs = nullptr;
    std::vector<anira::BufferF>* m_outputs = nullptr;
};

/// How prepare bound every slot of a prepared model to the engine's tensors
/// (anira_plan_slot.binding of the plan report): one anira_binding per slot of either side.
struct Bindings {
    std::vector<anira_binding> m_inputs;
    std::vector<anira_binding> m_outputs;
};

/// The interface: the C descriptor's lifecycle as a class. prepare once on the control thread,
/// run per inference on an inference thread, the destructor frees what prepare loaded.
class ANIRA_API Adapter {
public:
    Adapter() = default;
    virtual ~Adapter() = default;
    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;
    Adapter(Adapter&&) = delete;
    Adapter& operator=(Adapter&&) = delete;

    /// Control thread, once, under the core's lifecycle lock: keeps a copy of the record,
    /// loads the model (do_prepare) and sizes the instance claims to its instance count. A
    /// throw leaves the adapter unprepared (run refuses), and the core drops it.
    void prepare(const Model& model);

    /// Inference thread: one inference over the context's tensors. Claims a free instance of
    /// the prepared model (the first whose busy flag was clear, spinning until one is; the flag
    /// is released on every exit path), writes the instance's index into a copy of the context
    /// (the caller's stays as it is), calls reset on it when `reset_first`, then process, and
    /// returns process's status. An adapter that manages its instances itself
    /// (claims_instances() false: the legacy adapter, whose 2.x backend has its own rule) runs
    /// without the claim and reports instance 0. ANIRA_ERROR_INVALID_STATE, and no engine
    /// call, on an adapter that was never prepared.
    anira_status run(const anira_engine_ctx& ctx, ChunkBuffers* chunk, bool reset_first) noexcept;

    /// The engine's promises (ANIRA_ENGINE_FLAG_*); 0 promises nothing.
    virtual uint32_t flags() const noexcept { return 0; }

    /// The record prepare kept.
    const Model& model() const noexcept { return m_model; }

    /// Whether prepare succeeded.
    bool prepared() const noexcept { return m_num_instances > 0; }

    /// How prepare bound every slot: ANIRA_BINDING_POSITION for every slot of the record
    /// unless do_prepare said otherwise (set_bindings); the legacy adapter's 2.x backend binds
    /// by position. Sized to the record after prepare, empty before.
    const Bindings& bindings() const noexcept { return m_bindings; }

protected:
    /// Loads the model of the record: the engine's session, its environment, the warm-up.
    /// Throws anira::StatusError (MODEL_LOAD, NO_SUCH_FILE, ENGINE, CONFIG, NOT_SUPPORTED)
    /// with the message the caller reads.
    virtual void do_prepare(const Model& model) = 0;
    /// One inference on ctx.instance over the context's tensors (chunk: the struct's 2.x
    /// buffers, for the legacy adapter alone). Any status but ANIRA_OK fails the chunk in the
    /// scheduler; nothing is cleared or recorded here.
    virtual anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept = 0;
    /// Re-initialises the state the engine keeps inside itself on ctx.instance, right before
    /// the process of the first inference of a new stream; nothing by default.
    virtual void reset(const anira_engine_ctx& ctx) noexcept { static_cast<void>(ctx); }
    /// Whether run claims an instance per call (the default). The legacy adapter answers
    /// false: a 2.x BackendBase takes no instance index and is called as concurrently as the
    /// scheduler dispatches, as it always was.
    virtual bool claims_instances() const noexcept { return true; }
    /// What do_prepare reports for the slots it bound (bind_slots): one entry per slot of
    /// either side of the record.
    void set_bindings(Bindings bindings) noexcept { m_bindings = std::move(bindings); }

private:
    Model m_model;
    Bindings m_bindings;
    std::vector<std::atomic<bool>> m_busy;  ///< one busy flag per instance, sized at prepare
    uint32_t m_num_instances = 0;           ///< 0 until prepare succeeded
};

/// How one slot was bound to the engine's tensor on its side: the tensor's index in the
/// engine's order and the way (bind_slots).
struct SlotBinding {
    size_t m_index = 0;
    anira_binding m_binding = ANIRA_BINDING_POSITION;
};

/// The binding rule of every built-in engine, one side at a time. `engine_names` are the
/// engine's tensors of the side in the engine's order (the graph order of ONNX Runtime, the
/// signature's index order of LiteRT and TFLite, the method's argument order of LibTorch; all
/// empty strings for a side without names). A slot the entry's tensors record names
/// (TensorInfo::m_engine_name) binds to the engine tensor of that name, which must exist; a
/// slot without a record binds to the engine tensor of its canonical name where the side has
/// one, else to the engine tensor at the slot's own position. Afterwards every engine tensor
/// below `required` is bound exactly once and none at or above it twice (`required` is the
/// engine's count, except where trailing engine tensors may stay unbound: a LibTorch argument
/// with a default value). Throws anira::StatusError(ANIRA_ERROR_CONFIG) naming `engine`, the
/// side, the slot and the engine's names for a record name the engine lacks, a slot whose
/// position the engine has no tensor at, a duplicate or an unbound engine tensor. Logs nothing.
ANIRA_API std::vector<SlotBinding> bind_slots(const std::vector<TensorInfo>& slots,
                                              const std::vector<std::string>& engine_names,
                                              size_t required,
                                              const char* engine,
                                              const char* side);

/// One tensor of the engine's side as the engine describes it: its name (empty for a side
/// without names), its extents with a negative extent for a dynamic one, its element type as an
/// anira_dtype (0 for one anira has no code for) and the engine's own word for that type.
struct EngineTensor {
    std::string m_name;
    std::vector<int64_t> m_dims;
    anira_dtype m_dtype = 0;
    std::string m_type_word;
};

/// How check_engine_tensor reads the engine's extents.
enum class ExtentRule : uint8_t {
    /// A static extent must equal the record's; a dynamic (negative) one matches anything.
    Exact,
    /// The engine reports the planned upper bound of every axis and marks no axis dynamic
    /// (ExecuTorch): a record extent at or below the bound matches, one above it does not.
    UpperBound,
};

/// The check of the binding rule, whichever way a slot was bound: the engine tensor's element
/// type must be the record's dtype, the ranks must agree, and every extent per `rule`. Throws
/// anira::StatusError(ANIRA_ERROR_CONFIG) naming `engine`, the side, the slot and the engine
/// tensor, with both shapes.
ANIRA_API void check_engine_tensor(const TensorInfo& slot,
                                   const EngineTensor& engine_tensor,
                                   ExtentRule rule,
                                   const char* engine,
                                   const char* side);

/// The names of `engine_tensors`, in their order: what bind_slots takes.
ANIRA_API std::vector<std::string> names_of(const std::vector<EngineTensor>& engine_tensors);

/// One side at prepare, as every built-in adapter does it: bind_slots over the names of
/// `engine_tensors`, then check_engine_tensor for every slot against the engine tensor it
/// bound. Throws what they throw.
ANIRA_API std::vector<SlotBinding> bind_side(const std::vector<TensorInfo>& slots,
                                             const std::vector<EngineTensor>& engine_tensors,
                                             size_t required,
                                             ExtentRule rule,
                                             const char* engine,
                                             const char* side);

/// The report of two bound sides: what set_bindings takes.
ANIRA_API Bindings bindings_of(const std::vector<SlotBinding>& inputs,
                               const std::vector<SlotBinding>& outputs);

/// A float pointer over the memory of a descriptor a built-in adapter can hand its engine as
/// one packed block: host memory (pageable or page-locked), float32, not planar, `expected`
/// elements, and packed row-major strides (all zero, or the row-major strides themselves; the
/// stride of an axis of extent 1 is never stepped and may be anything), at byte_offset from
/// the base. NULL for anything else, a NULL base included.
ANIRA_API float* host_f32_packed(const anira_tensor& tensor, size_t expected) noexcept;

/// The float32 rule of the built-in adapters: throws anira::StatusError(ANIRA_ERROR_CONFIG)
/// naming `engine` and the first tensor of the record (inputs, then outputs) whose dtype is
/// not float32.
ANIRA_API void require_f32(const Model& model, const char* engine);

}  // namespace anira::backend

#endif  // ANIRA_BACKENDS_ADAPTER_H
