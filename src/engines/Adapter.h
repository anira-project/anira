#ifndef ANIRA_ENGINES_ADAPTER_H
#define ANIRA_ENGINES_ADAPTER_H
/*
 * The engine room's interface: what every engine anira runs looks like from the scheduler,
 * built in or registered, in the shape of the C descriptor (anira/abi/engine.h) and its
 * lifecycle. Private to src/engines and the scheduler (and the tests through the src/ include
 * directory): nothing here enters the ABI. An adapter is what adapts an engine to this shape;
 * the files keep that word, the classes carry the lifecycle's.
 *
 * Three objects, the C lifecycle's levels as classes:
 *
 *   Loaded    one loaded model (the C loaded pointer): what load(model) loaded once, pooled by
 *             the core by the record of the model and the carrier of a registered engine and
 *             shared by every session whose plan describes that model, a session-exclusive
 *             session's included (exclusivity is no part of the pool's key any more). It holds
 *             the shared call slots (instances) and their busy flags: a shared call runs on
 *             a slot claimed for it (the first whose busy flag was clear), and the claimed
 *             slot's index travels in the context. The inference thread claims before it
 *             takes a job and never waits for a slot (try_claim, release, run_claimed); the
 *             claim loop the five 2.x processors each carried (claim_and_run, under run) is
 *             here once more for a caller that runs a shared call directly.
 *   Prepared  one session's handle over a loaded model (the C prepared pointer): what
 *             Loaded::prepare hands back per session, held in the session's plan table beside
 *             the loaded model, and what the inference thread runs (run). For an exclusive
 *             session (a model declared stateful or with a declared State pair: its inferences
 *             run one at a time under the dispatch gate) it owns what the session keeps per
 *             stream, its own executor included, and its calls claim no shared slot; for a
 *             shared session it runs every call on a claimed slot of the loaded model.
 *   Executor  one call slot of a loaded model: the engine's executor (a session, an
 *             interpreter, a method), what one process call runs on, never two at once. The
 *             built-in engines make them (ExecutorLoaded); a registered engine sizes its own.
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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace anira::engine {

/// One tensor of a loaded model as its engine is handed it: the canonical name, the export's
/// name where the entry's tensors record names the slot (empty otherwise: the slot then binds
/// by its canonical name where the engine's side has one, else by its position), the engine's
/// extents (the entry's layout applied to the spec's) at
/// the pinned window, the spec's dtype and the element count.
struct TensorInfo {
    std::string m_name;
    std::string m_export_name;
    std::vector<int64_t> m_dims;
    anira_dtype m_dtype = ANIRA_DTYPE_F32;
    size_t m_num_elements = 0;

    bool operator==(const TensorInfo& other) const = default;
};

/// The record of one loaded model: everything an engine reads at load, copied out of the
/// configuration it came from (a model entry and the validator's derived quantities), so that
/// a pooled model never aliases a session's
/// configuration (issue #76). Two records compare equal when they describe the same model for
/// the same engine on the same provider with the same tensors, shared slots, warm-up and
/// entry, and, for a custom engine, the same variant: the pool's key, beside the carrier of a
/// custom engine (two providers of one model load twice). The bytes
/// compare by identity, as the 2.x ModelData compares them; the bytes owner and the log level
/// are no part of the identity (the level is what the engine's environment is created with, and
/// the core reconciles one level for the process). Whether a session runs the model exclusively
/// is the session's property (PrepareRequest), not the record's: a stateful model is loaded once
/// for every session that runs it, with no shared slot.
struct Model {
    /// The built-in engine; ANIRA_ENGINE_CUSTOM with m_engine_id for a registered one (the
    /// shim's anira.v2.custom included), under the pair rule.
    anira_engine m_engine = ANIRA_ENGINE_NONE;
    /// A custom engine's id (m_engine ANIRA_ENGINE_CUSTOM), else empty: what the messages name
    /// the engine by. No part of the key: the carrier beside the record is, and the id is the
    /// carrier's own.
    std::string m_engine_id;
    std::string m_path;             ///< the model file; empty for bytes
    const void* m_bytes = nullptr;  ///< the model bytes; NULL for a path
    size_t m_num_bytes = 0;
    /// Keeps the bytes alive while the loaded model lives: an aliasing shared_ptr over the
    /// row's carrier.
    std::shared_ptr<const void> m_bytes_owner;
    std::string m_entry;  ///< the method or function of the file to run; empty for its default
    /// A custom engine's variant as canonical JSON text (anira::capi::model_config_json): its
    /// load may read the whole model config, so its loaded models are shared over an equal one
    /// only. Empty for a built-in engine, which reads the record alone.
    std::string m_variant;
    /// The provider the loaded model runs on: a provider of the enum, or ANIRA_PROVIDER_CUSTOM
    /// with m_provider_id for a custom one, in the engine's own vocabulary (what the plan's
    /// candidate named, or the entry's pin). Part of the key.
    anira_provider m_provider = ANIRA_PROVIDER_CPU;
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

    bool operator==(const Model& other) const;
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
/// registered engine (NULL for a built-in engine, which reads none).
struct PrepareRequest {
    bool m_exclusive = false;
    const anira_prepare_info* m_info = nullptr;
};

/// Whether the thread that runs a call is being stopped, asked by the claim loop of a shared
/// call (Loaded::claim_and_run) whenever no instance was free. The inference thread passes
/// its own stop flag (InferenceThread::should_exit), though it claims before it runs and never
/// reaches the loop for a pooled model; the default never stops.
struct StopSignal {
    bool (*m_stopping)(const void* context) noexcept = nullptr;
    const void* m_context = nullptr;

    /// True once the thread was told to stop; false without a query.
    [[nodiscard]] bool stopping() const noexcept {
        return m_stopping != nullptr && m_stopping(m_context);
    }
};

/// One call slot of a loaded model: the engine's executor, what one process call runs on, never
/// two at once. The built-in engines' file-local Instance classes implement it over the object
/// their loaded model shares (a session over ONNX Runtime's environment, an interpreter over
/// TFLite's model, a compiled model over LiteRT's, a Module over ExecuTorch's program, a
/// TorchScript module). Made on the control thread (ExecutorLoaded::make_executor), where it
/// may load, allocate and throw; process and reset run on an inference thread.
/// Whether a failing chunk's text is logged: the first failure since the last success is
/// (first_failure), the rest are not until a chunk succeeds (succeeded); the scheduler latches
/// the status itself and counts the rest. One per executor, on the inference thread, without
/// a lock.
class FailureLatch {
public:
    /// True for the first failure since the last success: the caller logs.
    bool first_failure() noexcept { return !m_failed.exchange(true, std::memory_order_relaxed); }
    /// A success: the next failure is logged again.
    void succeeded() noexcept { m_failed.store(false, std::memory_order_relaxed); }

private:
    std::atomic<bool> m_failed{false};
};

class Executor {
public:
    Executor() = default;
    virtual ~Executor() = default;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&&) = delete;
    Executor& operator=(Executor&&) = delete;

    /// One inference over the context's tensors; any status but ANIRA_OK fails the chunk in the
    /// scheduler, nothing is cleared or recorded here.
    virtual anira_status process(const anira_engine_ctx& ctx) noexcept = 0;
    /// Re-initialises the state the engine keeps inside this executor, right before the
    /// process of the first inference of a new stream (an exclusive session's executor alone
    /// is ever reset); nothing by default.
    virtual void reset(const anira_engine_ctx& ctx) noexcept { static_cast<void>(ctx); }
    /// The FIXED warm-up of the record over this executor's own scratch, on the control thread
    /// right after it was made; throws anira::StatusError(ANIRA_ERROR_ENGINE) with the
    /// engine's text for an inference that fails. Nothing by default.
    virtual void warm_up(uint32_t iterations) { static_cast<void>(iterations); }

    /// The copies of a slot's memory this executor made since it was made, per side: an engine
    /// counts one per slot in its copy branches and nothing where it binds the memory in place,
    /// so the tests see what the plan report's edge class claims (warm-up calls included).
    uint64_t copies_in() const noexcept { return m_copies_in.load(std::memory_order_relaxed); }
    uint64_t copies_out() const noexcept { return m_copies_out.load(std::memory_order_relaxed); }

protected:
    /// One copy of an input slot's memory into the engine's own, or of an output out of it.
    void count_copy_in() const noexcept { m_copies_in.fetch_add(1, std::memory_order_relaxed); }
    void count_copy_out() const noexcept { m_copies_out.fetch_add(1, std::memory_order_relaxed); }

    /// The latch of this executor's process: the engine's text goes to the log for the first
    /// failing chunk since the last success, the scheduler's latch counts the rest.
    FailureLatch m_failures;

private:
    mutable std::atomic<uint64_t> m_copies_in{0};
    mutable std::atomic<uint64_t> m_copies_out{0};
};

/// One provider a built-in engine's runtime reports usable here: a provider of the enum, or
/// ANIRA_PROVIDER_CUSTOM with a name in the runtime's own words (an ONNX Runtime execution
/// provider by its registered name, a LiteRT accelerator by its hardware: "gpu", "npu",
/// "webnn"). What the context's capabilities list per (engine, provider).
struct ProviderInfo {
    anira_provider m_provider = ANIRA_PROVIDER_CPU;
    std::string m_provider_id;

    bool operator==(const ProviderInfo& other) const = default;
};

/// One built-in engine of this build as an object, the twin of a custom engine's carrier
/// (src/capi/engine.h): what the engine keeps per process rather than per loaded model, built
/// by its init once, with the facts of the core in effect, and kept for the object's life:
/// ONNX Runtime's environment (its logger at the level in effect), LiteRT's environment (its
/// logger likewise, and the accelerators it registers), ExecuTorch's runtime initialisation,
/// LibTorch's intra-op thread count and c10 log level; TensorFlow Lite keeps nothing. The
/// core hands out one per compiled-in engine (Core::builtin_engine), initialises it right
/// before the engine's first load, on the control thread under its lifecycle lock
/// (Loaded::init), and hands it to every loaded model of the engine (make_builtin_loaded),
/// which holds it: the object lives while a loaded model of its engine (or a probe) holds it,
/// the last holder frees it on its own thread, and the next load makes a new one whose init
/// runs again.
class BuiltinEngine {
public:
    explicit BuiltinEngine(anira_engine engine) noexcept : m_engine(engine) {}
    virtual ~BuiltinEngine() = default;
    BuiltinEngine(const BuiltinEngine&) = delete;
    BuiltinEngine& operator=(const BuiltinEngine&) = delete;
    BuiltinEngine(BuiltinEngine&&) = delete;
    BuiltinEngine& operator=(BuiltinEngine&&) = delete;

    /// The built-in engine this object is. (Not named engine(): a member named like its
    /// namespace, anira::engine, mangles to a back-reference on MSVC that hides the namespace
    /// from the export check.)
    anira_engine kind() const noexcept { return m_engine; }

    /// The engine's init, once per object: the first call runs do_init with the record and
    /// remembers a success, every later call returns at once; a throw leaves the object
    /// uninitialised, so the next call runs do_init again. Control thread, under the core's
    /// lifecycle lock (the core calls it through Loaded::init). Throws anira::StatusError
    /// with the message the caller reads.
    void ensure_init(const anira_init_info& info);

    /// Whether init ran and succeeded: false until then, and after a throw.
    bool initialised() const noexcept { return m_initialised; }

    /// The providers the engine serves here: the CPU path first, then what its
    /// runtime reports usable (runtime_providers), each once. What the context's capabilities
    /// list per (engine, provider), asked at every probe. Runs on any control thread, before
    /// or after init and any number of times, without a lock: it touches nothing init builds
    /// (the runtime is asked directly; LiteRT's query creates an environment of its own).
    std::vector<ProviderInfo> providers() const;

protected:
    /// What the engine builds once per process, with the facts of the record (the level in
    /// effect above all). Nothing by default. Throws anira::StatusError (ENGINE,
    /// NOT_SUPPORTED) with the message the caller reads.
    virtual void do_init(const anira_init_info& info) { static_cast<void>(info); }

    /// What the runtime reports usable here beyond the CPU path, which the caller
    /// lists first: ONNX Runtime's available execution providers, LiteRT's registered
    /// accelerators, ExecuTorch's registered backends; nothing for TFLite and LibTorch in this
    /// pre-release. Empty, with a warning logged, when the runtime cannot be asked.
    virtual std::vector<ProviderInfo> runtime_providers() const { return {}; }

private:
    anira_engine m_engine;
    bool m_initialised = false;
};

class Prepared;

/// One loaded model: the C lifecycle's loaded pointer as a class. load once on the control
/// thread under the core's lifecycle lock (the core pools loaded models by record and carrier),
/// prepare once per session that runs the model (the Prepared the session's plan slot holds),
/// the destructor frees what load loaded, when the pool lets the model go.
class Loaded {
public:
    Loaded() = default;
    virtual ~Loaded() = default;
    Loaded(const Loaded&) = delete;
    Loaded& operator=(const Loaded&) = delete;
    Loaded(Loaded&&) = delete;
    Loaded& operator=(Loaded&&) = delete;

    /// Control thread, under the core's lifecycle lock, right before load: the engine's init,
    /// once per engine object, with the facts of the core in effect (a built-in engine's
    /// object, ExecutorLoaded; a custom engine's init slot through its carrier,
    /// DescriptorLoaded; the object remembers, so a second loaded model of it calls nothing).
    /// Nothing by default. Throws anira::StatusError with the status a refused init returned,
    /// naming the engine.
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
    /// Holds prepare_mutex() around do_prepare: the prepares of one loaded model never overlap
    /// (a custom engine's prepare runs on the handler's thread, outside the core's lifecycle
    /// lock, and a pooled loaded model is every equal handler's).
    std::unique_ptr<Prepared> prepare(const PrepareRequest& request);

    /// The lock of prepare and unprepare on this loaded model: prepare holds it around
    /// do_prepare and a custom engine's Prepared holds it around the engine's unprepare, so
    /// the prepares and unprepares of one loaded model never overlap (two handlers prepared on
    /// two threads reach the engine's loaded pointer one after the other) while those of
    /// different loaded models run at once. Taken after the core's lifecycle lock where both
    /// are held (create_session, release_session), never before it.
    std::mutex& prepare_mutex() noexcept { return m_prepare_mutex; }

    /// The engine's promises (ANIRA_ENGINE_FLAG_*); 0 promises nothing.
    virtual uint32_t flags() const noexcept { return 0; }

    /// The boundary, in bytes (a power of two), a chunk's slot buffer must start on for the
    /// engine to bind it in place: every session sizes its chunks to the largest any of its
    /// plans' engines asks for. 64 by default, what LiteRT and TFLite ask of the host memory
    /// they bind and more than the other runtimes need; an engine whose runtime maps memory on a
    /// coarser grain (a page, for a GPU that maps host memory) says so.
    virtual size_t buffer_alignment() const noexcept { return k_default_buffer_alignment; }
    /// The alignment every engine gets unless it asks for more.
    static constexpr size_t k_default_buffer_alignment = 64;

    /// How the engine passes one slot's chunk buffer to its runtime, as the plan report states
    /// it (anira_plan_slot.edge_class and reason): ANIRA_EDGE_ZERO_COPY where it binds the
    /// buffer in place, ANIRA_EDGE_HOST_COPY with the reason where it copies through memory of
    /// its own. A registered engine is handed the buffer itself: zero-copy by default.
    struct SlotEdge {
        anira_edge_class m_class = ANIRA_EDGE_ZERO_COPY;
        const char* m_reason = nullptr;  ///< static storage; NULL with nothing to say
    };
    virtual SlotEdge slot_edge(bool is_input, uint32_t slot) const noexcept {
        static_cast<void>(is_input);
        static_cast<void>(slot);
        return SlotEdge{};
    }

    /// Whether the engine serves a provider: the CPU path alone unless an adapter says
    /// otherwise (a built-in engine whose runtime takes one, the descriptor adapter, whose
    /// engine's load decides). load refuses a record on a provider the adapter does not serve
    /// with ANIRA_ERROR_NOT_SUPPORTED, before do_load.
    virtual bool serves(anira_provider provider, std::string_view provider_id) const noexcept {
        return provider == ANIRA_PROVIDER_CPU && provider_id.empty();
    }
    /// Why the adapter serves what it serves, for the refusal of a provider it does not: the
    /// runtime's own list where it has one, the pre-release's limit else. Named after the
    /// engine and the provider in load's ANIRA_ERROR_NOT_SUPPORTED.
    virtual std::string provider_reason() const {
        return "this engine serves the CPU path alone in this pre-release";
    }

    /// The record load kept.
    const Model& model() const noexcept { return m_model; }

    /// Whether load succeeded.
    bool loaded() const noexcept { return m_loaded; }

    /// The shared call slots: the record's instances after load, 0 before and for a model run
    /// by exclusive sessions alone.
    uint32_t num_instances() const noexcept { return m_num_instances; }

    /// Inference thread: claims the first shared slot whose busy flag is clear and writes its
    /// index into `instance`; false, and nothing claimed, when every slot is busy or the model
    /// has none. Never waits. A slot whose flag reads set is skipped without a write; every
    /// read and write is sequentially consistent, so a claim that failed and a release() that
    /// follows it are ordered for the dispatch of InferenceThread (a job that waits for a slot
    /// is seen by the thread that releases one). The caller runs the call on the slot
    /// (Prepared::run_claimed) and gives it back with release().
    bool try_claim(uint32_t& instance) noexcept;

    /// Gives back a slot try_claim() claimed (sequentially consistent, after the call on it).
    void release(uint32_t instance) noexcept;

    /// How load bound every slot: ANIRA_BINDING_POSITION for every slot of the record unless
    /// do_load said otherwise (set_bindings). Sized to the record after load, empty before.
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
    /// The first check of load: that the engine's init ran (ExecutorLoaded: its engine
    /// object's; DescriptorLoaded: its carrier's), since what load builds runs over what init
    /// built. Throws anira::StatusError(ANIRA_ERROR_INVALID_STATE) naming the engine when it
    /// has not; nothing by default.
    virtual void require_initialised() const {}
    /// What do_load reports for the slots it bound (bind_slots): one entry per slot of either
    /// side of the record.
    void set_bindings(Bindings bindings) noexcept { m_bindings = std::move(bindings); }

private:
    friend class Prepared;
    /// A shared session's call run directly (Prepared::run): claims a free slot of the model
    /// (try_claim; the flag is released on every path out), writes its index into the call and
    /// runs the prepared's process on it. The inference thread never gets here for a pooled
    /// model: it claims before it takes a job (InferenceThread) and runs the call with
    /// Prepared::run_claimed; this loop serves the callers that run a shared call themselves
    /// (the engine room's tests drive it from several threads). While every slot is busy it waits
    /// with the backoff of the inference thread's loop (src/utils/Backoff.h: spins, pause
    /// hints, then a yield and a 100 us sleep per pass), and gives up once `stop` says the
    /// thread is being stopped, so a pool's join never waits on a thread that waits for a
    /// slot another thread holds. ANIRA_ERROR_INVALID_STATE, and no engine call, on a model
    /// without a shared slot and when it gave up.
    anira_status claim_and_run(Prepared& prepared,
                               anira_engine_ctx& call,
                               StopSignal stop) noexcept;

    Model m_model;
    Bindings m_bindings;
    std::vector<std::atomic<bool>> m_busy;  ///< one busy flag per shared slot, sized at load
    uint32_t m_num_instances = 0;           ///< the shared slots; 0 before load
    bool m_loaded = false;
    std::mutex m_prepare_mutex;
};

/// One session's handle over a loaded model: the C lifecycle's prepared pointer as a class.
/// Made by Loaded::prepare on the control thread, run per inference on an inference thread, the
/// destructor gives back what the session kept (before the loaded model is released: the
/// session's plan slot holds the two in that order).
class Prepared {
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
    /// slot's index into the call, and gives up waiting for one once `stop` says the thread is
    /// being stopped. Returns process's status; ANIRA_ERROR_INVALID_STATE, and no engine call,
    /// over a model that was never loaded and for a shared call that gave up.
    anira_status run(const anira_engine_ctx& ctx, bool reset_first, StopSignal stop = {}) noexcept;

    /// Inference thread: one shared call of this session on a slot the caller claimed
    /// (Loaded::try_claim) and gives back (Loaded::release) itself: copies the context, writes
    /// the loaded pointer and `instance` into it and runs process on it, without the claim
    /// loop. For a session whose calls claim a slot (claims_instance()). Returns process's
    /// status; ANIRA_ERROR_INVALID_STATE, and no engine call, over a model that was never
    /// loaded.
    anira_status run_claimed(const anira_engine_ctx& ctx, uint32_t instance) noexcept;

    /// Whether this session's calls run on a claimed shared slot of the loaded model: a shared
    /// session over a model with at least one. False for an exclusive session, whose calls run
    /// on what it keeps.
    bool claims_instance() const noexcept { return !m_exclusive && m_loaded->num_instances() > 0; }

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
    virtual anira_status process(const anira_engine_ctx& call) noexcept = 0;
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
class ExecutorLoaded : public Loaded {
public:
    /// Over the engine object of the engine (the core's one per built-in engine,
    /// Core::builtin_engine), which the loaded model holds for its life: what load builds runs
    /// over what the object's init built.
    explicit ExecutorLoaded(std::shared_ptr<BuiltinEngine> engine) noexcept
        : m_engine(std::move(engine)) {}

    /// The built-in engine object; initialised by the time load ran.
    BuiltinEngine& builtin() const noexcept { return *m_engine; }

    /// The engine object's init, once per object (BuiltinEngine::ensure_init).
    void init(const anira_init_info& info) override { m_engine->ensure_init(info); }

    /// The executor of a shared slot; the slot is below num_instances().
    Executor& executor(uint32_t instance) const noexcept { return *m_shared[instance]; }

protected:
    /// ANIRA_ERROR_INVALID_STATE naming the engine when its object was never initialised.
    void require_initialised() const override;

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
    std::shared_ptr<BuiltinEngine> m_engine;  ///< declared first: outlives the executors
    std::vector<std::unique_ptr<Executor>> m_shared;
    std::unique_ptr<Executor> m_spare;
};

/// The Prepared of an ExecutorLoaded: an exclusive session's own executor, or the routing of a
/// shared session's calls to the executor of the claimed slot.
class ExecutorPrepared final : public Prepared {
public:
    ExecutorPrepared(ExecutorLoaded& loaded, bool exclusive, std::unique_ptr<Executor> own) noexcept
        : Prepared(loaded, exclusive), m_loaded(&loaded), m_own(std::move(own)) {}

    /// This session's own executor; NULL for a shared session.
    Executor* own() const noexcept { return m_own.get(); }

protected:
    anira_status process(const anira_engine_ctx& call) noexcept override;
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

/// The binding rule of every built-in engine, one side at a time. `export_names` are the
/// engine's tensors of the side in the engine's order (the graph order of ONNX Runtime, the
/// key order of the signature on LiteRT and TFLite, the method's argument order of LibTorch; all
/// empty strings for a side without names). A slot the entry's tensors record names
/// (TensorInfo::m_export_name) binds to the engine tensor of that name, which must exist; a
/// slot without a record binds to the engine tensor of its canonical name where the side has
/// one, else to the engine tensor at the slot's own position. Afterwards every engine tensor
/// below `required` is bound exactly once and none at or above it twice (`required` is the
/// engine's count, except where trailing engine tensors may stay unbound: a LibTorch argument
/// with a default value). Throws anira::StatusError(ANIRA_ERROR_CONFIG) naming `engine`, the
/// side, the slot and the engine's names for a record name the engine lacks, a slot whose
/// position the engine has no tensor at, a duplicate or an unbound engine tensor. Logs nothing.
std::vector<SlotBinding> bind_slots(const std::vector<TensorInfo>& slots,
                                    const std::vector<std::string>& export_names,
                                    size_t required,
                                    const char* engine,
                                    const char* side);

/// One tensor of the engine's side as the engine describes it: its name (empty for a side
/// without names), its extents with a negative extent for a dynamic one, its element type as an
/// anira_dtype (0 for one anira has no code for), the engine's own word for that type, and
/// whether its values are quantised (they mean scale * (q - zero point): TFLite and LiteRT
/// quantisation parameters, a quantised torch or ExecuTorch scalar type).
struct EngineTensor {
    std::string m_name;
    std::vector<int64_t> m_dims;
    anira_dtype m_dtype = 0;
    std::string m_type_word;
    bool m_quantised = false;
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
/// tensor, with both shapes. A quantised engine tensor comes first: quantised I/O is not
/// supported (ANIRA_ERROR_NOT_SUPPORTED), refused rather than handed on as plain integers.
void check_engine_tensor(const TensorInfo& slot,
                         const EngineTensor& engine_tensor,
                         ExtentRule rule,
                         const char* engine,
                         const char* side);

/// The names of `engine_tensors`, in their order: what bind_slots takes.
std::vector<std::string> names_of(const std::vector<EngineTensor>& engine_tensors);

/// One side at load, as every built-in adapter does it: bind_slots over the names of
/// `engine_tensors`, then check_engine_tensor for every slot against the engine tensor it
/// bound. Throws what they throw.
std::vector<SlotBinding> bind_side(const std::vector<TensorInfo>& slots,
                                   const std::vector<EngineTensor>& engine_tensors,
                                   size_t required,
                                   ExtentRule rule,
                                   const char* engine,
                                   const char* side);

/// The report of two bound sides: what set_bindings takes.
Bindings bindings_of(const std::vector<SlotBinding>& inputs,
                     const std::vector<SlotBinding>& outputs);

/// The memory of a descriptor a built-in adapter can hand its engine as one packed block:
/// host memory (pageable or page-locked), of `dtype`, not planar, `expected` elements, and
/// packed row-major strides (all zero, or the row-major strides themselves; the stride of an
/// axis of extent 1 is never stepped and may be anything), at byte_offset from the base. NULL
/// for anything else, a NULL base included.
void* host_packed(const anira_tensor& tensor, anira_dtype dtype, size_t expected) noexcept;

/// Whether `data` starts on an `alignment` boundary (a power of two): what an engine that binds
/// memory in place checks per call, since no tensor is assumed to be anira's own buffer.
inline bool aligned_to(const void* data, size_t alignment) noexcept {
    return (reinterpret_cast<uintptr_t>(
                data) &  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            (alignment - 1)) == 0;
}

/// The dtype rule of a built-in adapter at load: throws anira::StatusError
/// (ANIRA_ERROR_NOT_SUPPORTED) naming `engine` and the first tensor of the record (inputs,
/// then outputs) whose dtype `carries` refuses, an engine's limit and no configuration error.
void require_dtypes(const Model& model, const char* engine, bool (*carries)(anira_dtype) noexcept);

/// The float32 rule of an adapter that binds float32 alone: require_dtypes over float32.
void require_f32(const Model& model, const char* engine);

}  // namespace anira::engine

#endif  // ANIRA_ENGINES_ADAPTER_H
