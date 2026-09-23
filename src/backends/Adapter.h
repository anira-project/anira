#ifndef ANIRA_BACKENDS_ADAPTER_H
#define ANIRA_BACKENDS_ADAPTER_H
/*
 * The engine room's interface: what every engine anira runs looks like from the scheduler,
 * built in or registered, in the shape of the C descriptor (anira/abi/engine.h) and its
 * lifecycle. Private to src/backends and the scheduler (and the tests through the src/ include
 * directory): nothing here enters the ABI. An adapter is what adapts an engine to this shape;
 * the files keep that word, the classes carry the lifecycle's.
 *
 * Three objects, the C lifecycle's levels as classes:
 *
 *   Loaded    one loaded model (the C loaded pointer): what load(model) loaded once, pooled by
 *             the core by the record of the model and the carrier of a registered engine and
 *             shared by every session whose plan describes that model, a session-exclusive
 *             session's included (exclusivity is no part of the pool's key any more). It holds
 *             the shared call slots (instances) and the spin-and-claim loop the five 2.x
 *             processors each carried, here once: a shared call claims the first slot whose
 *             busy flag was clear, and the claimed slot's index travels in the context.
 *   Prepared  one session's handle over a loaded model (the C prepared pointer): what
 *             Loaded::prepare hands back per session, held in the session's plan table beside
 *             the loaded model, and what the inference thread runs (run). For an exclusive
 *             session (a model declared stateful or with a declared State pair: its inferences
 *             run one at a time under the dispatch gate) it owns what the session keeps per
 *             stream, its own executor included, and its calls claim no shared slot; for a
 *             shared session it routes every call through the loaded model's claim loop.
 *   Executor  one call slot of a loaded model: the engine's executor (a session, an
 *             interpreter, a method), what one process call runs on, never two at once. The
 *             built-in engines make them (ExecutorLoaded); a registered engine and a 2.x backend
 *             size their own.
 *
 * The adapter rule of anira/abi/engine.h binds here too: process reads every extent and every
 * memory handle from the context's tensors on every call, keeps nothing of them, and assumes
 * no tensor to be anira's own buffer.
 */
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
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
#include <string_view>
#include <utility>
#include <vector>

namespace anira::backend {

/// One tensor of a loaded model as its engine is handed it: the canonical name, the export's
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

/// The record of one loaded model: everything an engine reads at load, copied out of the
/// configuration it came from (a model entry and the validator's derived quantities on the C
/// path, an InferenceConfig on the 2.x path), so that a pooled model never aliases a session's
/// configuration (issue #76). Two records compare equal when they describe the same model for
/// the same engine on the same provider with the same tensors, shared slots, warm-up and
/// entry, and, for a custom engine, the same variant: the pool's key, beside the carrier of a
/// custom engine (two providers of one model load twice). The bytes
/// compare by identity, as the 2.x ModelData compares them; the bytes owner and the log level
/// are no part of the identity (the level is what the engine's environment is created with, and
/// the core reconciles one level for the process). Whether a session runs the model exclusively
/// is the session's property (PrepareRequest), not the record's: a stateful model is loaded once
/// for every session that runs it, with no shared slot.
struct ANIRA_API Model {
    anira_engine m_engine = ANIRA_ENGINE_NONE;  ///< the built-in engine; NONE with m_engine_id
                                                ///< for a registered one, NONE alone for the
                                                ///< 2.x custom and roundtrip adapters
    /// A registered engine's id, else empty: what the messages name the engine by. No part of
    /// the key: the carrier beside the record is, and the id is the carrier's own.
    std::string m_engine_id;
    std::string m_path;             ///< the model file; empty for bytes
    const void* m_bytes = nullptr;  ///< the model bytes; NULL for a path
    size_t m_num_bytes = 0;
    /// Keeps the bytes alive while the loaded model lives: an aliasing shared_ptr over the
    /// row's carrier on the C path; empty on the 2.x path, whose caller owns the bytes.
    std::shared_ptr<const void> m_bytes_owner;
    std::string m_entry;  ///< the method or function of the file to run; empty for its default
    /// A custom engine's variant as canonical JSON text (anira::capi::model_config_json): its
    /// load may read the whole model config, so its loaded models are shared over an equal one
    /// only. Empty for a built-in engine, which reads the record alone.
    std::string m_variant;
    /// The provider the loaded model runs on: a provider of the enum, or ANIRA_PROVIDER_DEFAULT
    /// beside m_provider_id for a custom one, in the engine's own vocabulary (what the plan's
    /// candidate named, or the entry's pin). Part of the key.
    anira_provider m_provider = ANIRA_PROVIDER_DEFAULT;
    std::string m_provider_id;
    /// The options the engine's runtime takes for the provider, as string pairs in the
    /// runtime's own vocabulary (the "provider_options" context extension's set for this
    /// backend); part of the key: two option sets are two loaded models. Empty for none.
    std::vector<std::pair<std::string, std::string>> m_options;
    std::vector<TensorInfo> m_inputs;   ///< in slot order, State tensors included
    std::vector<TensorInfo> m_outputs;  ///< in slot order, State tensors included
    /// The shared call slots of the loaded model: the process calls that may run at once on
    /// them, each on its own instance below this count (the model's max_instances); 0 for a
    /// model whose sessions are exclusive (declared stateful, or with a declared State pair),
    /// which run on what their prepare builds and never on a shared slot.
    uint32_t m_instances = 1;
    uint32_t m_warm_up = 0;  ///< the inferences load runs on every executor before the first
                             ///< real one
    anira::LogLevel m_log_level = anira::LogLevel::Warning;  ///< the level in effect at load

    bool operator==(const Model& other) const;
};

/// The two 2.x buffer vectors of the chunk's struct, what the legacy adapter's 2.x virtual
/// takes; an executor of the descriptor shape (a built-in engine of this line, a registered
/// engine) reads the context's tensors alone and never looks here. Non-const: the 2.x virtual
/// takes the inputs by non-const reference (two 2.x processors swap their memory), and the
/// legacy adapter copies a foreign input into them ahead of the call.
struct ChunkBuffers {
    std::vector<anira::BufferF>* m_inputs = nullptr;
    std::vector<anira::BufferF>* m_outputs = nullptr;
};

/// How load bound every slot of a loaded model to the engine's tensors
/// (anira_plan_slot.binding of the plan report): one anira_binding per slot of either side.
struct Bindings {
    std::vector<anira_binding> m_inputs;
    std::vector<anira_binding> m_outputs;
};

/// What a session asks of a loaded model at its prepare (Loaded::prepare): whether the session
/// is exclusive (ANIRA_PREPARE_EXCLUSIVE: its inferences run one at a time and in order under
/// the dispatch gate, with a reset at every stream start, so its calls claim no shared slot
/// and run on an executor of the session's own), and the C record of the prepare for a
/// registered engine (NULL on the 2.x path, which registers no engine).
struct PrepareRequest {
    bool m_exclusive = false;
    const anira_prepare_info* m_info = nullptr;
};

/// One call slot of a loaded model: the engine's executor, what one process call runs on, never
/// two at once. The built-in engines' file-local Instance classes implement it over the object
/// their loaded model shares (a session over ONNX Runtime's environment, an interpreter over
/// TFLite's model, a compiled model over LiteRT's, a Module over ExecuTorch's program, a
/// TorchScript module). Made on the control thread (ExecutorLoaded::make_executor), where it
/// may load, allocate and throw; process and reset run on an inference thread.
class ANIRA_API Executor {
public:
    Executor() = default;
    virtual ~Executor() = default;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&&) = delete;
    Executor& operator=(Executor&&) = delete;

    /// One inference over the context's tensors; any status but ANIRA_OK fails the chunk in the
    /// scheduler, nothing is cleared or recorded here.
    virtual anira_status process(const anira_engine_ctx& ctx, ChunkBuffers* chunk) noexcept = 0;
    /// Re-initialises the state the engine keeps inside this executor, right before the
    /// process of the first inference of a new stream (an exclusive session's executor alone
    /// is ever reset); nothing by default.
    virtual void reset(const anira_engine_ctx& ctx) noexcept { static_cast<void>(ctx); }
    /// The FIXED warm-up of the record over this executor's own scratch, on the control thread
    /// right after it was made; throws anira::StatusError(ANIRA_ERROR_ENGINE) with the
    /// engine's text for an inference that fails. Nothing by default.
    virtual void warm_up(uint32_t iterations) { static_cast<void>(iterations); }
};

class Prepared;

/// One loaded model: the C lifecycle's loaded pointer as a class. load once on the control
/// thread under the core's lifecycle lock (the core pools loaded models by record and carrier),
/// prepare once per session that runs the model (the Prepared the session's plan slot holds),
/// the destructor frees what load loaded, when the pool lets the model go.
class ANIRA_API Loaded {
public:
    Loaded() = default;
    virtual ~Loaded() = default;
    Loaded(const Loaded&) = delete;
    Loaded& operator=(const Loaded&) = delete;
    Loaded(Loaded&&) = delete;
    Loaded& operator=(Loaded&&) = delete;

    /// Control thread, under the core's lifecycle lock, right before load: the engine's init,
    /// once per engine object, with the facts of the core in effect (a custom engine's init
    /// slot; the carrier remembers, so a second loaded model of the object calls nothing).
    /// Nothing by default: a built-in engine has no init of its own. Throws anira::StatusError
    /// with the status a refused init returned, naming the engine.
    virtual void init(const anira_init_info& info) { static_cast<void>(info); }

    /// Control thread, once, under the core's lifecycle lock: keeps a copy of the record, loads
    /// the model (do_load) and sizes the shared slots' claims to the record's instances (0 for
    /// a model run by exclusive sessions alone). A throw leaves the model unloaded (prepare and
    /// run refuse), and the core drops it.
    void load(const Model& model);

    /// Control thread, once per session that runs this model, at the session's prepare: the
    /// session's Prepared (do_prepare), which the session's plan slot holds and its inference
    /// thread runs. For an exclusive session the Prepared owns what the session keeps per
    /// stream (a built-in engine's own executor, made and warmed up here); for a shared one it
    /// routes every call through the claim loop. Throws anira::StatusError (ENGINE, CONFIG,
    /// INVALID_STATE on a model that was never loaded) with the message the caller reads.
    std::unique_ptr<Prepared> prepare(const PrepareRequest& request);

    /// The engine's promises (ANIRA_ENGINE_FLAG_*); 0 promises nothing.
    virtual uint32_t flags() const noexcept { return 0; }

    /// Whether the engine serves a provider: the default provider alone unless an adapter says
    /// otherwise (a built-in engine whose runtime takes one, the descriptor adapter, whose
    /// engine's load decides). load refuses a record on a provider the adapter does not serve
    /// with ANIRA_ERROR_NOT_SUPPORTED, before do_load.
    virtual bool serves(anira_provider provider, std::string_view provider_id) const noexcept {
        return provider == ANIRA_PROVIDER_DEFAULT && provider_id.empty();
    }
    /// Why the adapter serves what it serves, for the refusal of a provider it does not: the
    /// runtime's own list where it has one, the pre-release's limit else. Named after the
    /// engine and the provider in load's ANIRA_ERROR_NOT_SUPPORTED.
    virtual std::string provider_reason() const {
        return "this engine serves the default provider alone in this pre-release";
    }

    /// The record load kept.
    const Model& model() const noexcept { return m_model; }

    /// Whether load succeeded.
    bool loaded() const noexcept { return m_loaded; }

    /// The shared call slots: the record's instances after load, 0 before and for a model run
    /// by exclusive sessions alone.
    uint32_t num_instances() const noexcept { return m_num_instances; }

    /// How load bound every slot: ANIRA_BINDING_POSITION for every slot of the record unless
    /// do_load said otherwise (set_bindings); the legacy adapter's 2.x backend binds by
    /// position. Sized to the record after load, empty before.
    const Bindings& bindings() const noexcept { return m_bindings; }

    /// What the C descriptor's load handed back, the loaded pointer every call sees in
    /// anira_engine_ctx.loaded (DescriptorLoaded); NULL for every other engine.
    virtual void* engine_loaded() const noexcept { return nullptr; }

protected:
    /// Loads the model of the record: the engine's shared object, the binding, the shared
    /// executors and their warm-up. Throws anira::StatusError (MODEL_LOAD, NO_SUCH_FILE,
    /// ENGINE, CONFIG, NOT_SUPPORTED) with the message the caller reads.
    virtual void do_load(const Model& model) = 0;
    /// One session's Prepared over this loaded model, per the request.
    virtual std::unique_ptr<Prepared> do_prepare(const PrepareRequest& request) = 0;
    /// Whether a shared call claims a slot (the default). The legacy adapter answers false: a
    /// 2.x BackendBase takes no instance index and is called as concurrently as the scheduler
    /// dispatches, as it always was.
    virtual bool claims_instances() const noexcept { return true; }
    /// What do_load reports for the slots it bound (bind_slots): one entry per slot of either
    /// side of the record.
    void set_bindings(Bindings bindings) noexcept { m_bindings = std::move(bindings); }

private:
    friend class Prepared;
    /// Inference thread, a shared session's call: claims a free slot of the model (the first
    /// whose busy flag was clear, spinning until one is; the flag is released on every path
    /// out), writes its index into the call and runs the prepared's process on it. Without a
    /// claim (claims_instances() false) the call runs at once with instance 0.
    /// ANIRA_ERROR_INVALID_STATE, and no engine call, on a model without a shared slot.
    anira_status claim_and_run(Prepared& prepared,
                               anira_engine_ctx& call,
                               ChunkBuffers* chunk) noexcept;

    Model m_model;
    Bindings m_bindings;
    std::vector<std::atomic<bool>> m_busy;  ///< one busy flag per shared slot, sized at load
    uint32_t m_num_instances = 0;           ///< the shared slots; 0 before load
    bool m_loaded = false;
};

/// One session's handle over a loaded model: the C lifecycle's prepared pointer as a class.
/// Made by Loaded::prepare on the control thread, run per inference on an inference thread, the
/// destructor gives back what the session kept (before the loaded model is released: the
/// session's plan slot holds the two in that order).
class ANIRA_API Prepared {
public:
    virtual ~Prepared() = default;
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    Prepared(Prepared&&) = delete;
    Prepared& operator=(Prepared&&) = delete;

    /// Inference thread: one inference of this session over the context's tensors. Copies the
    /// context (the caller's stays as it is) and writes the loaded pointer into it; an
    /// exclusive session's call carries ANIRA_ENGINE_CALL_EXCLUSIVE with instance 0, claims no
    /// slot, is reset first when `reset_first` and runs on what this session keeps; a shared
    /// session's call goes through the loaded model's claim loop, which writes the claimed
    /// slot's index into the call. Returns process's status; ANIRA_ERROR_INVALID_STATE, and no
    /// engine call, over a model that was never loaded.
    anira_status run(const anira_engine_ctx& ctx, ChunkBuffers* chunk, bool reset_first) noexcept;

    /// Whether this session's calls run exclusively (ANIRA_PREPARE_EXCLUSIVE at its prepare).
    bool exclusive() const noexcept { return m_exclusive; }

    /// The loaded model this handle is over.
    Loaded& loaded() const noexcept { return *m_loaded; }

protected:
    Prepared(Loaded& loaded, bool exclusive) noexcept : m_loaded(&loaded), m_exclusive(exclusive) {}

    /// One inference of this session: on the shared slot the call names (call.instance, claimed
    /// by the loaded model) or, under ANIRA_ENGINE_CALL_EXCLUSIVE, on what this session keeps.
    /// Any status but ANIRA_OK fails the chunk in the scheduler; nothing is cleared or recorded
    /// here.
    virtual anira_status process(const anira_engine_ctx& call, ChunkBuffers* chunk) noexcept = 0;
    /// Re-initialises what this session keeps per stream, right before the process of the
    /// first inference of a new stream (exclusive sessions alone); nothing by default.
    virtual void reset(const anira_engine_ctx& call) noexcept { static_cast<void>(call); }

private:
    friend class Loaded;
    Loaded* m_loaded;
    bool m_exclusive;
};

/// A loaded model whose call slots are executors of the engine (Executor): the shape of the
/// five built-in engines. do_load loads the engine's shared object, makes one executor over it
/// (the probe, whose view of the model the binding rule runs over), binds, and hands the probe
/// to adopt, which makes the executors of the shared slots with make_executor (the probe is
/// slot 0) or keeps the probe as the spare of the first exclusive session when the record has
/// no shared slot, and warms every one of them up; make_executor makes one more for an
/// exclusive session at its prepare. The Prepared of such a model (ExecutorPrepared) runs a
/// shared call on the executor of the claimed slot and an exclusive call on the session's own.
class ANIRA_API ExecutorLoaded : public Loaded {
public:
    /// The executor of a shared slot; the slot is below num_instances().
    Executor& executor(uint32_t instance) const noexcept { return *m_shared[instance]; }

protected:
    /// One more executor over the loaded object, bound as the shared ones are, not yet warmed
    /// up (the caller warms it): what an exclusive session's prepare takes when no spare is
    /// left. Control thread; throws anira::StatusError with the message the caller reads.
    virtual std::unique_ptr<Executor> make_executor() = 0;

    /// The end of every do_load: `probe`, the one executor do_load read the binding off (bound
    /// already), becomes the executor of shared slot 0, with one of make_executor's per further
    /// slot of the record, or, for a record without a shared slot, the spare the first exclusive
    /// session takes; every one of them runs the record's FIXED warm-up here. Throws what
    /// make_executor and the warm-ups throw.
    void adopt(std::unique_ptr<Executor> probe);

    std::unique_ptr<Prepared> do_prepare(const PrepareRequest& request) override;

private:
    friend class ExecutorPrepared;
    std::vector<std::unique_ptr<Executor>> m_shared;
    std::unique_ptr<Executor> m_spare;
};

/// The Prepared of an ExecutorLoaded: an exclusive session's own executor, or the routing of a
/// shared session's calls to the executor of the claimed slot.
class ANIRA_API ExecutorPrepared final : public Prepared {
public:
    ExecutorPrepared(ExecutorLoaded& loaded, bool exclusive, std::unique_ptr<Executor> own) noexcept
        : Prepared(loaded, exclusive), m_loaded(&loaded), m_own(std::move(own)) {}

    /// This session's own executor; NULL for a shared session.
    Executor* own() const noexcept { return m_own.get(); }

protected:
    anira_status process(const anira_engine_ctx& call, ChunkBuffers* chunk) noexcept override;
    void reset(const anira_engine_ctx& call) noexcept override;

private:
    ExecutorLoaded* m_loaded;
    std::unique_ptr<Executor> m_own;
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

/// One side at load, as every built-in adapter does it: bind_slots over the names of
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
