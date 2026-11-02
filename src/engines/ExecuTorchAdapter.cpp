/*
 * The ExecuTorch adapter: one data loader and one program (the .pte parsed once) per loaded
 * model, shared by every executor, and one Module over that program with its own loaded entry
 * method per executor (the runtime's Module is not thread-safe; its constructor over a shared
 * program is the documented way to run one program from several), the slots bound to the
 * method's inputs and outputs by position (the method meta carries no tensor names) and
 * checked against its tensor metas at load (the meta reports the planned upper bound of every
 * axis and marks no axis dynamic, so an extent at or below the bound matches), and every slot
 * handed over by the memory plan's rule: the executor keeps one tensor per input slot and
 * re-points its data at the descriptor's memory per call, which set_input aliases for an input
 * the plan leaves out and copies into the plan's memory for a planned one (the one copy, the
 * runtime's); an output the plan leaves out is bound to the descriptor's memory per call
 * (set_output_data_ptr) where the slot holds the output's capacity, and a planned one is
 * copied out of the plan's memory into the descriptor's. The XNNPACK
 * delegate of every method is loaded with its runtime options set to no shared workspace and
 * no weight cache (the process-wide mutexes the prebuilt runtime otherwise takes around every
 * call), and it captures its thread pool at that load under NoThreadPoolGuard, so every call
 * runs inline on the inference thread that made it. File-local over anira::engine::Model:
 * nothing of ExecuTorch enters a public header.
 */
#ifdef USE_EXECUTORCH

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "../capi/words.h"
#include "../scheduler/ChunkStorage.h"
#include "../scheduler/TensorRun.h"
#include "../utils/Logger.h"
#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "Adapter.h"
#include "Adapters.h"

// IWYU pragma: begin_keep — the ExecuTorch headers are compiled as SYSTEM includes,
// where misc-include-cleaner cannot attribute the used symbols to their providers.
#include "executorch/extension/data_loader/buffer_data_loader.h"
#include "executorch/extension/data_loader/file_data_loader.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor_ptr.h"
#include "executorch/extension/tensor/tensor_ptr_maker.h"
#include "executorch/extension/threadpool/threadpool_guard.h"
#include "executorch/runtime/backend/backend_options_map.h"
#include "executorch/runtime/backend/interface.h"
#include "executorch/runtime/backend/options.h"
#include "executorch/runtime/core/data_loader.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/evalue.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/result.h"
#include "executorch/runtime/core/span.h"
#include "executorch/runtime/core/tag.h"
#include "executorch/runtime/executor/method_meta.h"
#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/platform/runtime.h"
// IWYU pragma: end_keep

namespace anira::engine {

namespace {

constexpr const char* k_engine = anira::capi::engine_word(ANIRA_ENGINE_EXECUTORCH);

// The XNNPACK delegate's backend id and its workspace sharing mode "disabled", as
// backends/xnnpack/runtime/XNNPACKBackend.h of the pinned release spells them (the prebuilt
// package ships no header of the delegate; the option keys are string literals below, since
// BackendOptions::set_option takes an array). The delegate reads both options at its init,
// from the map the method is loaded with: the sharing mode (0: a workspace per delegate
// instance; 1: one per model; 2: one per process) and whether the weights are unpacked into
// the process-wide cache. Both off, two executors never meet on the mutexes the shared
// workspace and the cache take around every call, at the price of a workspace and an unpacked
// copy of the weights per executor.
constexpr const char* k_xnnpack_backend = "XnnpackBackend";
constexpr int k_xnnpack_workspace_per_instance = 0;

// The names ExecuTorch's delegates register under (its "backends", the word quoted as the
// runtime's own below) that spell a provider of anira's enum: the delegate an export is
// lowered for, as the .pte names it; every other registered name travels as it is, the
// runtime's own word, in provider_id.
constexpr std::array<std::pair<const char*, anira_provider>, 3> k_provider_names{{
    {k_xnnpack_backend, ANIRA_PROVIDER_XNNPACK},
    {"CoreMLBackend", ANIRA_PROVIDER_COREML},
    {"VulkanBackend", ANIRA_PROVIDER_VULKAN},
}};

// The registered name of a provider: the table's for one of the enum, the provider_id itself
// (a registered name the capabilities listed) for a custom one; empty for the CPU path
// and for a provider of the enum no delegate spells.
std::string registered_name(anira_provider provider, std::string_view provider_id) {
    if (!provider_id.empty()) { return std::string(provider_id); }
    for (const auto& [name, value] : k_provider_names) {
        if (value == provider) { return name; }
    }
    return "";
}

// The delegates registered to this runtime and available (ExecuTorch's backends), by their
// registered names.
std::vector<std::string> registered_backends() {
    std::vector<std::string> names;
    const size_t count = executorch::runtime::get_num_registered_backends();
    for (size_t i = 0; i < count; ++i) {
        const auto name = executorch::runtime::get_backend_name(i);
        if (!name.ok() || name.get() == nullptr) { continue; }
        const executorch::runtime::BackendInterface* backend =
            executorch::runtime::get_backend_class(name.get());
        if (backend == nullptr || !backend->is_available()) { continue; }
        names.emplace_back(name.get());
    }
    return names;
}

std::string backends_list(const std::vector<std::string>& names) {
    std::string text;
    for (const std::string& name : names) {
        if (!text.empty()) { text += ", "; }
        text += name;
    }
    return text.empty() ? "none" : text;
}

/// The XNNPACK delegate's runtime options as its registered backend reports them.
struct XnnpackOptions {
    int m_workspace_sharing_mode = 0;
    bool m_weight_cache_enabled = false;
};

// One option of the XNNPACK delegate as the runtime's record spells it: the key copied into
// the record's array, the value in its variant.
executorch::runtime::BackendOption xnnpack_option(const char* key,
                                                  executorch::runtime::OptionValue value) {
    executorch::runtime::BackendOption option{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) the key is a C array
    std::snprintf(option.key, sizeof(option.key), "%s", key);
    option.value = value;
    return option;
}

// The two options of this adapter as one record: a workspace per delegate instance, no
// weight cache.
std::array<executorch::runtime::BackendOption, 2> xnnpack_options_of_this_adapter() {
    return {xnnpack_option("workspace_sharing_mode", int{k_xnnpack_workspace_per_instance}),
            xnnpack_option("weight_cache_enabled", false)};
}

// The delegate's options in effect for the process, read back through the registered backend
// (executorch::runtime::get_option): nothing when the backend is not registered here or does
// not answer.
std::optional<XnnpackOptions> xnnpack_options_in_effect() {
    std::array<executorch::runtime::BackendOption, 2> options = xnnpack_options_of_this_adapter();
    const executorch::runtime::Error error = executorch::runtime::get_option(
        k_xnnpack_backend,
        executorch::runtime::Span<executorch::runtime::BackendOption>(options.data(),
                                                                      options.size()));
    if (error != executorch::runtime::Error::Ok) { return std::nullopt; }
    const int* mode = std::get_if<int>(&options[0].value);
    const bool* cache = std::get_if<bool>(&options[1].value);
    if (mode == nullptr || cache == nullptr) { return std::nullopt; }
    return XnnpackOptions{.m_workspace_sharing_mode = *mode, .m_weight_cache_enabled = *cache};
}

// The delegate's options set for the process, at the engine's init: what every method of
// every loaded model runs under here (the load names them per method too, in its map). Read
// back afterwards, so that a release that spells the option keys otherwise says so at the
// engine's init, as a warning naming the values, instead of leaving the shared workspace and
// the weight cache on, and their mutexes with them, in silence. Nothing when the backend is
// not registered in this runtime.
void set_xnnpack_options_of_the_process() {
    std::array<executorch::runtime::BackendOption, 2> options = xnnpack_options_of_this_adapter();
    const executorch::runtime::Error error = executorch::runtime::set_option(
        k_xnnpack_backend,
        executorch::runtime::Span<executorch::runtime::BackendOption>(options.data(),
                                                                      options.size()));
    if (error == executorch::runtime::Error::NotFound) { return; }
    if (error != executorch::runtime::Error::Ok) {
        ANIRA_LOG_WARNING(log_group::k_engine_executorch,
                          "executorch: the XNNPACK delegate refused its runtime options "
                          "(workspace_sharing_mode %d, weight_cache_enabled 0): %s; this release "
                          "spells the option keys otherwise, and executors may meet on the "
                          "delegate's process-wide mutexes",
                          k_xnnpack_workspace_per_instance,
                          executorch::runtime::to_string(error));
        return;
    }
    const std::optional<XnnpackOptions> in_effect = xnnpack_options_in_effect();
    if (!in_effect.has_value() ||
        in_effect->m_workspace_sharing_mode != k_xnnpack_workspace_per_instance ||
        in_effect->m_weight_cache_enabled) {
        ANIRA_LOG_WARNING(log_group::k_engine_executorch,
                          "executorch: the XNNPACK delegate reports workspace_sharing_mode %d "
                          "and weight_cache_enabled %d after its init set %d and 0: this "
                          "release spells the option keys otherwise, and executors may meet on "
                          "the delegate's process-wide mutexes",
                          in_effect.has_value() ? in_effect->m_workspace_sharing_mode : -1,
                          in_effect.has_value() && in_effect->m_weight_cache_enabled ? 1 : 0,
                          k_xnnpack_workspace_per_instance);
    }
}

// Every fallible ExecuTorch call returns a runtime::Error (or a Result carrying one). A failure
// here means a setup or runtime problem, so it becomes a StatusError with the failing call and
// the error named (executorch::runtime::to_string, e.g. "InvalidProgram"); the status is
// ANIRA_ERROR_ENGINE unless the caller passes another.
void executorch_check(executorch::runtime::Error error,
                      const char* what,
                      anira_status failure = ANIRA_ERROR_ENGINE,
                      const std::string& where = "") {
    if (error != executorch::runtime::Error::Ok) {
        const std::string text = std::string(what) +
                                 " failed with Error::" + executorch::runtime::to_string(error) +
                                 " (" + std::to_string(static_cast<uint32_t>(error)) + ")";
        throw StatusError(failure,
                          model_file::message(k_engine, where.empty() ? what : where, text));
    }
}

// The anira dtype of an ExecuTorch scalar type; 0 for one anira has no code for.
anira_dtype dtype_of(executorch::aten::ScalarType type) {
    switch (type) {
        case executorch::aten::ScalarType::Float: return ANIRA_DTYPE_F32;
        case executorch::aten::ScalarType::Double: return ANIRA_DTYPE_F64;
        case executorch::aten::ScalarType::Half: return ANIRA_DTYPE_F16;
        case executorch::aten::ScalarType::BFloat16: return ANIRA_DTYPE_BF16;
        case executorch::aten::ScalarType::Char: return ANIRA_DTYPE_I8;
        case executorch::aten::ScalarType::Byte: return ANIRA_DTYPE_U8;
        case executorch::aten::ScalarType::Short: return ANIRA_DTYPE_I16;
        case executorch::aten::ScalarType::Int: return ANIRA_DTYPE_I32;
        case executorch::aten::ScalarType::Long: return ANIRA_DTYPE_I64;
        case executorch::aten::ScalarType::Bool: return ANIRA_DTYPE_BOOL8;
        default: return 0;
    }
}

// The ExecuTorch scalar type of an anira dtype: every dtype anira names has one; Undefined for
// any other, which the validator refuses before a record reaches the adapter.
executorch::aten::ScalarType executorch_type_of(anira_dtype dtype) noexcept {
    switch (dtype) {
        case ANIRA_DTYPE_F32: return executorch::aten::ScalarType::Float;
        case ANIRA_DTYPE_F64: return executorch::aten::ScalarType::Double;
        case ANIRA_DTYPE_F16: return executorch::aten::ScalarType::Half;
        case ANIRA_DTYPE_BF16: return executorch::aten::ScalarType::BFloat16;
        case ANIRA_DTYPE_I8: return executorch::aten::ScalarType::Char;
        case ANIRA_DTYPE_U8: return executorch::aten::ScalarType::Byte;
        case ANIRA_DTYPE_I16: return executorch::aten::ScalarType::Short;
        case ANIRA_DTYPE_I32: return executorch::aten::ScalarType::Int;
        case ANIRA_DTYPE_I64: return executorch::aten::ScalarType::Long;
        case ANIRA_DTYPE_BOOL8: return executorch::aten::ScalarType::Bool;
        default: return executorch::aten::ScalarType::Undefined;
    }
}

bool carries(anira_dtype dtype) noexcept {
    return executorch_type_of(dtype) != executorch::aten::ScalarType::Undefined;
}

// Whether a scalar type is quantised (its values mean scale * (q - zero point)).
bool quantised(executorch::aten::ScalarType type) noexcept {
    switch (type) {
        case executorch::aten::ScalarType::QInt8:
        case executorch::aten::ScalarType::QUInt8:
        case executorch::aten::ScalarType::QInt32:
        case executorch::aten::ScalarType::QUInt4x2:
        case executorch::aten::ScalarType::QUInt2x4: return true;
        default: return false;
    }
}

// One tensor of the method as its meta describes it: no name, the planned sizes, the type.
EngineTensor engine_tensor_of(
    const executorch::runtime::Result<executorch::runtime::TensorInfo>& info,
    bool is_tensor) {
    EngineTensor tensor;
    if (!is_tensor || !info.ok()) {
        tensor.m_type_word = "not a tensor";
        return tensor;
    }
    for (const int32_t size : info->sizes()) { tensor.m_dims.push_back(size); }
    tensor.m_dtype = dtype_of(info->scalar_type());
    tensor.m_type_word =
        "ExecuTorch scalar type " + std::to_string(static_cast<int>(info->scalar_type()));
    tensor.m_quantised = quantised(info->scalar_type());
    return tensor;
}

/// What one load shares between its executors: the data loader over the file or the bytes of
/// the record (which the record's owner keeps alive for the loaded model's life) and the
/// program parsed from it, immutable once loaded; the runtime asks that the loader outlive the
/// program, and every Module over the program (Module's constructor over a shared program)
/// dies before this does. Held by the loaded model and by every executor, freed with the last
/// of them.
struct SharedProgram {
    explicit SharedProgram(const Model& model);
    ~SharedProgram() = default;
    SharedProgram(const SharedProgram&) = delete;
    SharedProgram& operator=(const SharedProgram&) = delete;
    SharedProgram(SharedProgram&&) = delete;
    SharedProgram& operator=(SharedProgram&&) = delete;

    std::string m_where;  ///< the model path or "memory", for the failure messages
    std::unique_ptr<executorch::runtime::DataLoader> m_loader;  ///< declared first: dies last
    std::shared_ptr<executorch::runtime::Program> m_program;
};

SharedProgram::SharedProgram(const Model& model) {
    if (model.m_bytes != nullptr) {
        // BufferDataLoader keeps a pointer into the caller's buffer.
        m_where = model_file::k_memory;
        m_loader = std::make_unique<executorch::extension::BufferDataLoader>(model.m_bytes,
                                                                             model.m_num_bytes);
    } else {
        m_where = model_file::require_readable(model.m_path, k_engine);
        auto loader = executorch::extension::FileDataLoader::from(m_where.c_str());
        executorch_check(loader.error(), "FileDataLoader::from", ANIRA_ERROR_MODEL_LOAD, m_where);
        m_loader = std::make_unique<executorch::extension::FileDataLoader>(std::move(loader.get()));
    }
    // The program: parsed and verified once, shared by every Module over it.
    auto program = executorch::runtime::Program::load(m_loader.get());
    executorch_check(program.error(), "Program::load", ANIRA_ERROR_MODEL_LOAD, m_where);
    m_program = std::make_shared<executorch::runtime::Program>(std::move(program.get()));
}

/// One Module over the shared program with its own loaded entry method: what one executor of
/// the adapter runs on. Loads the method at construction (the delegates initialised with the
/// runtime options above, under the thread-pool guard; the planned memory allocated); binds
/// afterwards; runs every inference over the context's descriptors, each slot handed over by
/// the memory plan's rule (the file comment).
class Instance final : public Executor {
public:
    Instance(std::shared_ptr<const SharedProgram> shared, const Model& model);
    ~Instance() override = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    /// The method's tensors of either side as the meta describes them, in the method's order.
    std::vector<EngineTensor> inputs();
    std::vector<EngineTensor> outputs();
    /// The backends the method uses (the delegates the export was lowered to), by their
    /// registered names; empty for a portable export.
    std::vector<std::string> backends();
    /// The method executed per inference.
    const std::string& method() const noexcept { return m_method; }

    /// Builds the kept tensors once, in the method's positional order: per input slot one of
    /// the record's dtype and engine dims over an aligned block of the executor's own (zeros:
    /// the warm-up's input, until a call re-points it), and per output the plan leaves out one
    /// over a block of the output's capacity, which is the output's memory (set_output) until a
    /// call binds the slot's. Reads per slot whether the plan owns the tensor and whether an
    /// unplanned output's slot holds its capacity.
    void bind(const Model& model,
              const std::vector<SlotBinding>& inputs,
              const std::vector<SlotBinding>& outputs);

    /// The edge of a slot after bind: ZERO_COPY for an input the plan leaves out and for an
    /// unplanned output whose slot holds its capacity, HOST_COPY with the reason else.
    Loaded::SlotEdge slot_edge(bool is_input, size_t slot) const noexcept;

    /// The FIXED warm-up of the record over the executor's own blocks (zeros); a failing
    /// execute is StatusError(ANIRA_ERROR_ENGINE) with the error named.
    void warm_up(uint32_t iterations) override;

    /// One inference over the context's descriptors: ANIRA_ERROR_INVALID_ARGUMENT for a
    /// descriptor this adapter cannot hand the engine, ANIRA_ERROR_ENGINE for a failing
    /// execute or a result of another shape than the record's.
    anira_status process(const anira_engine_ctx& ctx) noexcept override;

private:
    /// Re-points every input tensor at its descriptor's memory, binds every unplanned output
    /// (the slot's memory, or the executor's block of the output's capacity), executes, and
    /// copies every result that did not land in its descriptor's memory; throws StatusError.
    void run(const anira_engine_ctx& ctx);
    /// Whether the plan owns each tensor of a side, and an output's capacity (its meta's bytes,
    /// the planned upper bound), from the method's meta.
    std::vector<bool> planned(bool is_input);
    std::vector<size_t> output_capacities();

    std::shared_ptr<const SharedProgram> m_shared;            ///< the program, shared
    std::unique_ptr<executorch::extension::Module> m_module;  ///< this executor's, over it
    std::string m_method;  ///< the method executed per inference: the record's entry, else forward
    std::string m_execute_label;  ///< what a failing execute is named after, built once
    /// The XNNPACK delegate's runtime options and the map that hands them to the method's
    /// load, keyed by the delegate's id (the map keeps a view of the options: both members).
    executorch::runtime::BackendOptions<2> m_xnnpack_options;
    executorch::runtime::LoadBackendOptionsMap m_backend_options;

    std::vector<ChunkBlock> m_input_blocks;  ///< per input slot: the executor's own memory
    /// Per input slot: kept, re-pointed per call at the descriptor's memory (TensorImpl's data,
    /// no allocation).
    std::vector<executorch::extension::TensorPtr> m_input_tensors;
    std::vector<executorch::runtime::EValue> m_input_values;  ///< the reusable argument list,
                                                              ///< in the method's order
    std::vector<size_t> m_input_elements;
    std::vector<anira_dtype> m_input_dtypes;  ///< per input slot: the record's dtype
    std::vector<bool> m_input_planned;        ///< per input slot: set_input copies into the plan
    std::vector<size_t> m_result_of_slot;     ///< per output slot: the result's index
    std::vector<size_t> m_output_elements;
    std::vector<anira_dtype> m_output_dtypes;
    std::vector<bool> m_output_planned;  ///< per output slot: the plan owns the result
    /// Per output slot the plan leaves out: whether the slot holds the output's capacity (bound
    /// in place), the executor's block of that capacity and the kept tensor set_output takes.
    std::vector<bool> m_output_bindable;
    std::vector<ChunkBlock> m_output_blocks;
    std::vector<executorch::extension::TensorPtr> m_output_tensors;
};

Instance::Instance(std::shared_ptr<const SharedProgram> shared, const Model& model)
    : m_shared(std::move(shared))
    , m_module(std::make_unique<executorch::extension::Module>(m_shared->m_program)) {
    m_method = model.m_entry.empty() ? "forward" : model.m_entry;
    m_execute_label = "Module::execute(\"" + m_method + "\")";

    // The delegate's runtime options, read at its init (the method's load below): no shared
    // workspace, no weight cache.
    executorch_check(
        m_xnnpack_options.set_option("workspace_sharing_mode", k_xnnpack_workspace_per_instance),
        "BackendOptions::set_option(\"workspace_sharing_mode\")");
    executorch_check(m_xnnpack_options.set_option("weight_cache_enabled", false),
                     "BackendOptions::set_option(\"weight_cache_enabled\")");
    executorch_check(m_backend_options.set_options(k_xnnpack_backend, m_xnnpack_options.view()),
                     "LoadBackendOptionsMap::set_options");

    // The selected method up front: this initialises the delegates (the XNNPACK runtime
    // captures the thread pool it will run on here, and under the guard it captures none, so
    // its calls run inline on the calling thread, as the other engines' do) and allocates the
    // planned memory, none of which may happen lazily on the inference thread.
    const executorch::extension::threadpool::NoThreadPoolGuard no_threadpool;
    executorch_check(m_module->load_method(m_method, nullptr, nullptr, &m_backend_options),
                     ("Module::load_method(\"" + m_method + "\")").c_str(),
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
}

std::vector<EngineTensor> Instance::inputs() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    std::vector<EngineTensor> tensors;
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
        const auto tag = meta->input_tag(i);
        tensors.push_back(engine_tensor_of(meta->input_tensor_meta(i),
                                           tag.ok() && *tag == executorch::runtime::Tag::Tensor));
    }
    return tensors;
}

std::vector<EngineTensor> Instance::outputs() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    std::vector<EngineTensor> tensors;
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
        const auto tag = meta->output_tag(i);
        tensors.push_back(engine_tensor_of(meta->output_tensor_meta(i),
                                           tag.ok() && *tag == executorch::runtime::Tag::Tensor));
    }
    return tensors;
}

std::vector<std::string> Instance::backends() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    std::vector<std::string> names;
    for (size_t i = 0; i < meta->num_backends(); ++i) {
        const auto name = meta->get_backend_name(i);
        if (name.ok() && name.get() != nullptr) { names.emplace_back(name.get()); }
    }
    return names;
}

std::vector<bool> Instance::planned(bool is_input) {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    const size_t count = is_input ? meta->num_inputs() : meta->num_outputs();
    std::vector<bool> planned(count, true);
    for (size_t i = 0; i < count; ++i) {
        const auto info = is_input ? meta->input_tensor_meta(i) : meta->output_tensor_meta(i);
        planned[i] = !info.ok() || info->is_memory_planned();
    }
    return planned;
}

std::vector<size_t> Instance::output_capacities() {
    auto meta = m_module->method_meta(m_method);
    executorch_check(meta.error(),
                     "Module::method_meta",
                     ANIRA_ERROR_MODEL_LOAD,
                     m_shared->m_where);
    std::vector<size_t> capacities(meta->num_outputs(), 0);
    for (size_t i = 0; i < capacities.size(); ++i) {
        const auto info = meta->output_tensor_meta(i);
        if (info.ok()) { capacities[i] = info->nbytes(); }
    }
    return capacities;
}

Loaded::SlotEdge Instance::slot_edge(bool is_input, size_t slot) const noexcept {
    if (is_input) {
        if (slot < m_input_planned.size() && !m_input_planned[slot]) { return {}; }
        return {.m_class = ANIRA_EDGE_HOST_COPY,
                .m_reason = "executorch: the memory plan owns the input; set_input copies into it"};
    }
    if (slot < m_output_planned.size() && m_output_planned[slot]) {
        return {.m_class = ANIRA_EDGE_HOST_COPY,
                .m_reason = "executorch: the memory plan owns the output; copied out of it"};
    }
    if (slot < m_output_bindable.size() && m_output_bindable[slot]) { return {}; }
    return {.m_class = ANIRA_EDGE_HOST_COPY,
            .m_reason =
                "executorch: the output's capacity exceeds the slot; copied through a block "
                "of the executor's own"};
}

void Instance::bind(const Model& model,
                    const std::vector<SlotBinding>& inputs,
                    const std::vector<SlotBinding>& outputs) {
    // Positional on both sides: the exactly-once rule made slot i the method's input i.
    const std::vector<bool> inputs_planned = planned(true);
    const std::vector<bool> outputs_planned = planned(false);
    const std::vector<size_t> capacities = output_capacities();
    m_input_blocks.clear();
    m_input_tensors.clear();
    m_input_values.clear();
    m_input_elements.clear();
    m_input_dtypes.clear();
    m_input_planned.clear();
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        const TensorInfo& tensor = model.m_inputs[slot];
        std::vector<executorch::aten::SizesType> sizes(tensor.m_dims.begin(), tensor.m_dims.end());
        m_input_blocks.emplace_back(host_chunk_storage(),
                                    tensor.m_num_elements * tensor_run::dtype_size(tensor.m_dtype),
                                    Loaded::k_default_buffer_alignment);
        m_input_tensors.push_back(
            executorch::extension::from_blob(m_input_blocks.back().data(),
                                             std::move(sizes),
                                             executorch_type_of(tensor.m_dtype)));
        m_input_values.emplace_back(*m_input_tensors.back());
        m_input_elements.push_back(tensor.m_num_elements);
        m_input_dtypes.push_back(tensor.m_dtype);
        m_input_planned.push_back(inputs[slot].m_index >= inputs_planned.size() ||
                                  inputs_planned[inputs[slot].m_index]);
    }
    m_result_of_slot.clear();
    m_output_elements.clear();
    m_output_dtypes.clear();
    m_output_planned.clear();
    m_output_bindable.clear();
    m_output_blocks.clear();
    m_output_tensors.clear();
    for (size_t slot = 0; slot < outputs.size(); ++slot) {
        const TensorInfo& tensor = model.m_outputs[slot];
        const size_t index = outputs[slot].m_index;
        const size_t bytes = tensor.m_num_elements * tensor_run::dtype_size(tensor.m_dtype);
        m_result_of_slot.push_back(index);
        m_output_elements.push_back(tensor.m_num_elements);
        m_output_dtypes.push_back(tensor.m_dtype);
        const bool is_planned = index >= outputs_planned.size() || outputs_planned[index];
        m_output_planned.push_back(is_planned);
        // An output the plan leaves out has no memory until set_output gives it some: a block of
        // the output's capacity (the runtime refuses a smaller one), its memory until a call
        // binds the slot's; the slot is bound in place when it holds that capacity.
        const size_t capacity = index < capacities.size() ? capacities[index] : 0;
        m_output_bindable.push_back(!is_planned && capacity <= bytes);
        if (is_planned) {
            m_output_blocks.emplace_back();
            m_output_tensors.emplace_back();
            continue;
        }
        const size_t element = tensor_run::dtype_size(tensor.m_dtype);
        m_output_blocks.emplace_back(host_chunk_storage(),
                                     capacity,
                                     Loaded::k_default_buffer_alignment);
        m_output_tensors.push_back(executorch::extension::from_blob(
            m_output_blocks.back().data(),
            {static_cast<executorch::aten::SizesType>(capacity / element)},
            executorch_type_of(tensor.m_dtype)));
        executorch_check(m_module->set_output(m_method, *m_output_tensors.back(), index),
                         "Module::set_output",
                         ANIRA_ERROR_MODEL_LOAD,
                         m_shared->m_where);
    }
}

void Instance::warm_up(uint32_t iterations) {
    const executorch::extension::threadpool::NoThreadPoolGuard no_threadpool;
    for (uint32_t i = 0; i < iterations; ++i) {
        const auto result = m_module->execute(m_method, m_input_values);
        executorch_check(result.error(), (m_execute_label + " (warm-up)").c_str());
    }
}

void Instance::run(const anira_engine_ctx& ctx) {
    if (ctx.inputs == nullptr || ctx.outputs == nullptr ||
        ctx.num_inputs != m_input_elements.size() || ctx.num_outputs != m_result_of_slot.size()) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          std::string(k_engine) + ": the context carries " +
                              std::to_string(ctx.num_inputs) + " inputs and " +
                              std::to_string(ctx.num_outputs) + " outputs for a model of " +
                              std::to_string(m_input_elements.size()) + " and " +
                              std::to_string(m_result_of_slot.size()));
    }
    const auto packed = [](const anira_tensor& tensor,
                           anira_dtype dtype,
                           size_t elements,
                           const char* side,
                           size_t slot) {
        void* data = host_packed(tensor, dtype, elements);
        if (data == nullptr) {
            throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                              std::string(k_engine) + ": " + side + " slot " +
                                  std::to_string(slot) + " is not a packed " +
                                  capi::dtype_word(dtype) + " host block of " +
                                  std::to_string(elements) + " elements");
        }
        return data;
    };
    // In: every kept input tensor re-pointed at its descriptor's memory, which set_input
    // aliases for an input the plan leaves out and copies into the plan's memory (the one copy,
    // counted) for a planned one.
    for (size_t slot = 0; slot < m_input_elements.size(); ++slot) {
        void* data =
            packed(ctx.inputs[slot], m_input_dtypes[slot], m_input_elements[slot], "input", slot);
        m_input_tensors[slot]->unsafeGetTensorImpl()->set_data(data);
        if (m_input_planned[slot]) { count_copy_in(); }
    }
    // Out: every output the plan leaves out bound to the slot's memory where it holds the
    // output's capacity, else to the executor's block.
    for (size_t slot = 0; slot < m_result_of_slot.size(); ++slot) {
        if (m_output_planned[slot]) { continue; }
        void* target = m_output_bindable[slot] ? packed(ctx.outputs[slot],
                                                        m_output_dtypes[slot],
                                                        m_output_elements[slot],
                                                        "output",
                                                        slot)
                                               : m_output_blocks[slot].data();
        m_output_tensors[slot]->unsafeGetTensorImpl()->set_data(target);
        executorch_check(
            m_module->set_output(m_method, *m_output_tensors[slot], m_result_of_slot[slot]),
            "Module::set_output");
    }

    // Inline on this thread: the guard keeps every operator that asks for the thread pool
    // per call on it (the delegate captured none at its load).
    const executorch::extension::threadpool::NoThreadPoolGuard no_threadpool;
    const auto result = m_module->execute(m_method, m_input_values);
    executorch_check(result.error(), m_execute_label.c_str());
    const std::vector<executorch::runtime::EValue>& outputs = result.get();

    // Out of the plan's memory or the executor's block, one copy each into the descriptor's
    // memory; a result in the descriptor's memory is the slot's already.
    for (size_t slot = 0; slot < m_result_of_slot.size(); ++slot) {
        const size_t index = m_result_of_slot[slot];
        if (index >= outputs.size() || !outputs[index].isTensor()) {
            throw StatusError(ANIRA_ERROR_ENGINE,
                              std::string(k_engine) +
                                  ": the method returned no tensor for output slot " +
                                  std::to_string(slot) + " (result " + std::to_string(index) +
                                  " of " + std::to_string(outputs.size()) + ")");
        }
        const executorch::aten::Tensor& tensor = outputs[index].toTensor();
        const anira_dtype dtype = m_output_dtypes[slot];
        if (tensor.scalar_type() != executorch_type_of(dtype) ||
            std::cmp_not_equal(tensor.numel(), m_output_elements[slot])) {
            throw StatusError(ANIRA_ERROR_ENGINE,
                              std::string(k_engine) + ": output slot " + std::to_string(slot) +
                                  ": the method returned " + std::to_string(tensor.numel()) +
                                  " elements of scalar type " +
                                  std::to_string(static_cast<int>(tensor.scalar_type())) +
                                  " for the spec's " + std::to_string(m_output_elements[slot]) +
                                  " " + capi::dtype_word(dtype) + " elements");
        }
        void* data = packed(ctx.outputs[slot], dtype, m_output_elements[slot], "output", slot);
        if (tensor.const_data_ptr() == data) { continue; }
        std::memcpy(data,
                    tensor.const_data_ptr(),
                    m_output_elements[slot] * tensor_run::dtype_size(dtype));
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
            ANIRA_LOG_RT_ERROR(log_group::k_engine_executorch, "%s", e.what());
        }
        return e.status();
    } catch (const std::exception& e) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_executorch, "%s", e.what());
        }
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        if (m_failures.first_failure()) {
            ANIRA_LOG_RT_ERROR(log_group::k_engine_executorch,
                               "executorch threw a non-std exception out of execute");
        }
        return ANIRA_ERROR_ENGINE;
    }
}

/// The engine object: the runtime's global initialisation (executorch::runtime::runtime_init,
/// the platform abstraction layer; the kernels and the backends register at static
/// initialisation) and the XNNPACK delegate's options for the process (a workspace per delegate
/// instance, no weight cache: set_xnnpack_options_of_the_process, read back), run once at the
/// engine's init; the thread-pool guard is per call. The core hands out one object per engine
/// while a loaded model holds it (Core::builtin_engine); the next one after the last let go
/// runs its init again.
class ExecuTorchEngine final : public BuiltinEngine {
public:
    ExecuTorchEngine() : BuiltinEngine(ANIRA_ENGINE_EXECUTORCH) {}

protected:
    void do_init(const anira_init_info& /*info*/) override {
        // The runtime's global initialisation once per process (a second call is no part of
        // its contract), the delegate's options with every object. The flag is trivially
        // destructible: nothing is registered for static teardown.
        static std::once_flag s_runtime_init;
        std::call_once(s_runtime_init, [] { executorch::runtime::runtime_init(); });
        set_xnnpack_options_of_the_process();
    }

    std::vector<ProviderInfo> runtime_providers() const override { return executorch_providers(); }
};

/// The loaded model: the program every executor shares (SharedProgram, parsed once), positional
/// on both sides (the meta carries no names), checked against a probe's metas: the planned
/// upper bound of every axis, an extent at or below it matching; the probe is the executor of
/// shared slot 0 or, for a record without a shared slot, the spare of the first exclusive
/// session.
class ExecuTorchLoaded final : public ExecutorLoaded {
public:
    explicit ExecuTorchLoaded(std::shared_ptr<ExecuTorchEngine> engine)
        : ExecutorLoaded(std::move(engine)) {}

    /// Per slot, by the memory plan's rule (the probe's bind): an input or output the plan
    /// leaves out is bound in place, a planned one costs one copy.
    SlotEdge slot_edge(bool is_input, uint32_t slot) const noexcept override {
        const std::vector<SlotEdge>& edges = is_input ? m_input_edges : m_output_edges;
        return slot < edges.size() ? edges[slot] : SlotEdge{};
    }

    /// The CPU path (a portable export, or whatever delegates the runtime has for the
    /// export's), or a backend registered to this runtime and available, by the name the
    /// enum's spelling maps to or the registered name itself.
    bool serves(anira_provider provider, std::string_view provider_id) const noexcept override {
        if (provider == ANIRA_PROVIDER_CPU && provider_id.empty()) { return true; }
        try {
            const std::string wanted = registered_name(provider, provider_id);
            if (wanted.empty()) { return false; }
            const std::vector<std::string> registered = registered_backends();
            return std::ranges::find(registered, wanted) != registered.end();
        } catch (...) {  // NOLINT(bugprone-empty-catch) an unlisted backend is the answer
            return false;
        }
    }

    std::string provider_reason() const override {
        std::string reason =
            "ExecuTorch runs the delegates compiled into its runtime, its backends (registered "
            "and available here: ";
        try {
            reason += backends_list(registered_backends());
        } catch (...) {  // NOLINT(bugprone-empty-catch) the list stays as far as it got
        }
        reason += "); an entry pinned to a provider names the delegate its export was lowered for";
        return reason;
    }

protected:
    void do_load(const Model& model) override {
        require_dtypes(model, k_engine, carries);
        m_shared = std::make_shared<const SharedProgram>(model);
        auto probe = std::make_unique<Instance>(m_shared, model);
        // A pinned entry names the delegate (ExecuTorch's backend) its export was lowered for:
        // the method must use it, or the export is mislabeled and refused here rather than at
        // its first call.
        if (model.m_provider != ANIRA_PROVIDER_CPU || !model.m_provider_id.empty()) {
            const std::string wanted = registered_name(model.m_provider, model.m_provider_id);
            const std::vector<std::string> used = probe->backends();
            if (std::ranges::find(used, wanted) == used.end()) {
                throw StatusError(
                    ANIRA_ERROR_CONFIG,
                    "executorch: " + m_shared->m_where + ": the entry is pinned to provider '" +
                        anira::capi::provider_label(model.m_provider, model.m_provider_id) +
                        "' (ExecuTorch backend '" + wanted + "'), but its method '" +
                        probe->method() + "' uses " +
                        (used.empty() ? "no delegate" : "the delegates " + backends_list(used)) +
                        "; a pin names the delegate the export was lowered for");
            }
        }
        const std::vector<EngineTensor> inputs = probe->inputs();
        const std::vector<EngineTensor> outputs = probe->outputs();
        m_input_bindings = bind_side(model.m_inputs,
                                     inputs,
                                     inputs.size(),
                                     ExtentRule::UpperBound,
                                     k_engine,
                                     "input");
        m_output_bindings = bind_side(model.m_outputs,
                                      outputs,
                                      outputs.size(),
                                      ExtentRule::UpperBound,
                                      k_engine,
                                      "output");
        probe->bind(model, m_input_bindings, m_output_bindings);
        m_input_edges.clear();
        m_output_edges.clear();
        for (size_t slot = 0; slot < model.m_inputs.size(); ++slot) {
            m_input_edges.push_back(probe->slot_edge(true, slot));
        }
        for (size_t slot = 0; slot < model.m_outputs.size(); ++slot) {
            m_output_edges.push_back(probe->slot_edge(false, slot));
        }
        set_bindings(bindings_of(m_input_bindings, m_output_bindings));
        adopt(std::move(probe));
    }

    std::unique_ptr<Executor> make_executor() override {
        auto executor = std::make_unique<Instance>(m_shared, model());
        executor->bind(model(), m_input_bindings, m_output_bindings);
        return executor;
    }

private:
    std::shared_ptr<const SharedProgram> m_shared;  ///< freed with the last executor over it
    std::vector<SlotBinding> m_input_bindings;
    std::vector<SlotBinding> m_output_bindings;
    std::vector<SlotEdge> m_input_edges;  ///< per input slot: the probe's edge
    std::vector<SlotEdge> m_output_edges;
};

}  // namespace

std::shared_ptr<BuiltinEngine> make_executorch_engine() {
    return std::make_shared<ExecuTorchEngine>();
}

std::shared_ptr<Loaded> make_executorch_loaded(const std::shared_ptr<BuiltinEngine>& engine) {
    std::shared_ptr<ExecuTorchEngine> own = std::dynamic_pointer_cast<ExecuTorchEngine>(engine);
    if (own == nullptr) {
        throw StatusError(ANIRA_ERROR_INVALID_ARGUMENT,
                          "executorch: the engine object is not this adapter's");
    }
    return std::make_shared<ExecuTorchLoaded>(std::move(own));
}

bool executorch_xnnpack_options(int& workspace_sharing_mode, bool& weight_cache_enabled) {
    const std::optional<XnnpackOptions> in_effect = xnnpack_options_in_effect();
    if (!in_effect.has_value()) { return false; }
    workspace_sharing_mode = in_effect->m_workspace_sharing_mode;
    weight_cache_enabled = in_effect->m_weight_cache_enabled;
    return true;
}

std::vector<ProviderInfo> executorch_providers() {
    std::vector<ProviderInfo> providers;
    for (const std::string& name : registered_backends()) {
        ProviderInfo info;
        info.m_provider = ANIRA_PROVIDER_CUSTOM;
        info.m_provider_id = name;
        for (const auto& [registered, value] : k_provider_names) {
            if (name == registered) {
                info.m_provider = value;
                info.m_provider_id.clear();
                break;
            }
        }
        providers.push_back(std::move(info));
    }
    return providers;
}

}  // namespace anira::engine

#endif  // USE_EXECUTORCH
