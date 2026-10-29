// anira/anira_em.h: the Wasm Audio Worklet node of a prepared handler. One render callback per
// quantum on the browser's audio thread, in C: the node's input and output buffers are the
// handler's planar host tensors (no copy), the miss rule is applied here, the Static inputs a
// node serves arrive as AudioParams (scalars) or through a seqlock block per slot (tensors), and
// a JavaScript stage of the handler runs through the trampolines on the worklet scope's
// registry. Compiled on the Wasm binary format only.
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/anira_em.h>
#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>
#include <emscripten/webaudio.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../capi/capi_internal.h"
#include "../../capi/handler.h"

using anira::capi::translate_exception;

namespace {

constexpr size_t k_worklet_stack_bytes = size_t{512} * 1024;
constexpr uint32_t k_max_nodes = 256;
constexpr uint32_t k_max_channels = 64;
constexpr uint32_t k_max_params = 32;
constexpr size_t k_align = 16;

// One Static input a node serves from the main thread (anira_em_audio_node_set_static_input):
// the writer makes the sequence word odd, writes the value and makes it even again; the render
// callback copies a value whose word was even and unchanged across the copy and has not been
// pushed yet, then hands the copy to the handler. Never torn, never waited for.
struct StaticBlock {
    uint32_t m_slot = 0;
    std::atomic<uint32_t> m_sequence{0};
    uint32_t m_pushed = 0;
    std::vector<uint8_t> m_value;
    std::vector<uint8_t> m_copy;
    std::array<int64_t, ANIRA_MAX_RANK> m_shape{};
    anira_tensor m_tensor{};
};

// One AudioParam of a node: the Static scalar input it drives and the value pushed last.
struct ParamSlot {
    uint32_t m_slot = 0;
    float m_value = 0.0F;
    bool m_pushed = false;
    int64_t m_shape = 1;
    anira_tensor m_tensor{};
};

// What the render callback reads: built on the main thread at create, never resized after.
struct NodeState {
    anira_handler* m_handler = nullptr;
    uint32_t m_input_channels = 0;
    uint32_t m_output_channels = 0;
    uint32_t m_input_slot = 0;
    uint32_t m_output_slot = 0;
    uint32_t m_quantum = 0;
    double m_wait_ms = 0.0;
    std::vector<float> m_zeros;  ///< one quantum: an input channel not connected
    std::array<void*, k_max_channels> m_input_planes{};
    std::array<void*, k_max_channels> m_output_planes{};
    std::array<int64_t, 2> m_input_shape{};
    std::array<int64_t, 2> m_output_shape{};
    anira_tensor m_input{};
    anira_tensor m_output{};
    std::vector<ParamSlot> m_params;
    std::vector<std::unique_ptr<StaticBlock>> m_blocks;
};

// The node table: a slot per node, never freed, so a render callback that runs after its node
// was destroyed reads a generation that no longer matches and does nothing. The render callback
// marks itself busy before it reads the state, the destroy marks the slot closing and waits
// until the callback is not busy (both sequentially consistent, so one of them sees the other).
struct NodeSlot {
    std::atomic<uint32_t> m_generation{0};
    std::atomic<uint32_t> m_open{0};
    std::atomic<uint32_t> m_busy{0};
    NodeState* m_state = nullptr;  ///< written on the main thread while closed
    EMSCRIPTEN_AUDIO_WORKLET_NODE_T m_node = 0;
};

std::array<NodeSlot, k_max_nodes> s_nodes;

uintptr_t encode(uint32_t index, uint32_t generation) {
    return (static_cast<uintptr_t>(generation) << 8U) | index;
}

void silence(int num_outputs, AudioSampleFrame* outputs) {
    for (int o = 0; o < num_outputs; ++o) {
        const size_t count = static_cast<size_t>(outputs[o].numberOfChannels) *
                             static_cast<size_t>(outputs[o].samplesPerChannel);
        std::memset(outputs[o].data, 0, count * sizeof(float));
    }
}

void push_params(NodeState& state, int num_params, const AudioParamFrame* params) {
    const size_t count = state.m_params.size() < static_cast<size_t>(num_params)
                             ? state.m_params.size()
                             : static_cast<size_t>(num_params);
    for (size_t i = 0; i < count; ++i) {
        ParamSlot& p = state.m_params[i];
        // k-rate: one value per quantum; an a-rate array starts on the quantum's first value.
        const float value = params[i].data[0];
        if (p.m_pushed && value == p.m_value) { continue; }
        p.m_value = value;
        p.m_pushed = true;
        static_cast<void>(anira_handler_set_static_input(state.m_handler, p.m_slot, &p.m_tensor));
    }
}

void pull_blocks(NodeState& state) {
    for (auto& block : state.m_blocks) {
        StaticBlock& b = *block;
        const uint32_t before = b.m_sequence.load(std::memory_order::acquire);
        if ((before & 1U) != 0 || before == b.m_pushed) { continue; }
        std::memcpy(b.m_copy.data(), b.m_value.data(), b.m_value.size());
        std::atomic_thread_fence(std::memory_order::acquire);
        if (b.m_sequence.load(std::memory_order::relaxed) != before) { continue; }
        b.m_pushed = before;
        static_cast<void>(anira_handler_set_static_input(state.m_handler, b.m_slot, &b.m_tensor));
    }
}

// The render callback, once per quantum on the audio thread: allocates nothing, locks nothing.
bool render(int num_inputs,
            const AudioSampleFrame* inputs,
            int num_outputs,
            AudioSampleFrame* outputs,
            int num_params,
            const AudioParamFrame* params,
            void* user_data) {
    const auto key = reinterpret_cast<uintptr_t>(user_data);
    NodeSlot& slot = s_nodes[key & 0xffU];
    slot.m_busy.store(1, std::memory_order::seq_cst);
    if (slot.m_open.load(std::memory_order::seq_cst) == 0 ||
        slot.m_generation.load(std::memory_order::seq_cst) != static_cast<uint32_t>(key >> 8U)) {
        slot.m_busy.store(0, std::memory_order::seq_cst);
        silence(num_outputs, outputs);
        return false;  // destroyed: the browser may let the node go
    }
    NodeState& state = *slot.m_state;
    if (num_outputs < 1 || outputs[0].samplesPerChannel != static_cast<int>(state.m_quantum)) {
        // Not the quantum the handler is prepared for: silence, the handler untouched.
        silence(num_outputs, outputs);
        slot.m_busy.store(0, std::memory_order::release);
        return true;
    }
    push_params(state, num_params, params);
    pull_blocks(state);

    const size_t quantum = state.m_quantum;
    const bool has_input = num_inputs > 0;
    for (uint32_t ch = 0; ch < state.m_input_channels; ++ch) {
        const bool connected = has_input && ch < static_cast<uint32_t>(inputs[0].numberOfChannels);
        state.m_input_planes[ch] = connected ? static_cast<void*>(inputs[0].data + ch * quantum)
                                             : static_cast<void*>(state.m_zeros.data());
    }
    for (uint32_t ch = 0; ch < state.m_output_channels; ++ch) {
        state.m_output_planes[ch] = outputs[0].data + ch * quantum;
    }
    anira_tensor_init_host_planar(&state.m_input,
                                  state.m_input_planes.data(),
                                  state.m_input_channels,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  state.m_input_shape.data());
    anira_tensor_init_host_planar(&state.m_output,
                                  state.m_output_planes.data(),
                                  state.m_output_channels,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  state.m_output_shape.data());
    size_t delivered = 0;
    const anira_status status = state.m_wait_ms > 0.0
                                    ? anira_handler_process_wait(state.m_handler,
                                                                 &state.m_input,
                                                                 state.m_input_slot,
                                                                 &state.m_output,
                                                                 state.m_output_slot,
                                                                 &delivered,
                                                                 state.m_wait_ms)
                                    : anira_handler_process(state.m_handler,
                                                            &state.m_input,
                                                            state.m_input_slot,
                                                            &state.m_output,
                                                            state.m_output_slot,
                                                            &delivered);
    // The miss rule: ANIRA_MISSED delivers 0 by the ABI, yet the miss policy (bypass, hold the
    // last, a callback) filled the whole block, which is the output already; a failure is
    // silence; any other success is what it delivered, the rest silence.
    if (status != ANIRA_MISSED) {
        const size_t filled =
            ANIRA_FAILED(status) ? 0 : (delivered < quantum ? delivered : quantum);
        if (filled < quantum) {
            for (uint32_t ch = 0; ch < state.m_output_channels; ++ch) {
                std::memset(outputs[0].data + ch * quantum + filled,
                            0,
                            (quantum - filled) * sizeof(float));
            }
        }
    }
    // Output channels the node has beyond the handler's (none: the node is built with as many).
    slot.m_busy.store(0, std::memory_order::release);
    return true;
}

// ---- the audio context's worklet thread and the processors, per context ----

// The JavaScript stages of the worklet scope are built over its instance before the first
// quantum reaches them: the registry's boot, posted to the audio thread at every create.
EM_JS(void, anira_em_js_boot_worklet_registry, (), {
    var registry = globalThis[Symbol.for ('anira.registry')];
    if (registry && registry.boot) {
        Promise.resolve(registry.boot(Module)).catch(function(e) {
            console.error('anira: the worklet registry failed to boot:', e);
        });
    }
});

void boot_worklet_registry() {
    anira_em_js_boot_worklet_registry();
}

struct Pending {
    uint32_t m_slot_index = 0;
    std::string m_processor;
    int32_t* m_out_status = nullptr;
    uint32_t* m_out_node = nullptr;
};

struct ContextState {
    bool m_started = false;  ///< the worklet thread runs
    bool m_failed = false;
    void* m_stack = nullptr;  ///< the worklet thread's stack: the context's for its lifetime
    std::map<std::string, int> m_processors;  ///< 0 registering, 1 registered, -1 failed
    std::vector<Pending> m_pending;
};

std::map<EMSCRIPTEN_WEBAUDIO_T, ContextState> s_contexts;
// The descriptors of the processors being registered: the registration reads them later.
std::map<std::string, std::vector<WebAudioParamDescriptor>> s_param_descriptors;

void finish(const Pending& pending, anira_status status, EMSCRIPTEN_AUDIO_WORKLET_NODE_T node) {
    *pending.m_out_node = static_cast<uint32_t>(node);
    __atomic_store_n(pending.m_out_status, static_cast<int32_t>(status), __ATOMIC_RELEASE);
}

void free_state(NodeSlot& slot) {
    delete slot.m_state;  // NOLINT(cppcoreguidelines-owning-memory)
    slot.m_state = nullptr;
}

void create_node(EMSCRIPTEN_WEBAUDIO_T context, const Pending& pending) {
    NodeSlot& slot = s_nodes[pending.m_slot_index];
    NodeState& state = *slot.m_state;
    int output_channels[1] = {static_cast<int>(state.m_output_channels)};
    EmscriptenAudioWorkletNodeCreateOptions options{};
    options.numberOfInputs = state.m_input_channels > 0 ? 1 : 0;
    options.numberOfOutputs = 1;
    options.outputChannelCounts = output_channels;
    options.channelCount = state.m_input_channels > 0 ? state.m_input_channels : 1;
    options.channelCountMode = WEBAUDIO_CHANNEL_COUNT_MODE_EXPLICIT;
    options.channelInterpretation = WEBAUDIO_CHANNEL_INTERPRETATION_DISCRETE;
    const uintptr_t key = encode(pending.m_slot_index, slot.m_generation.load());
    emscripten_audio_worklet_post_function_v(context, &boot_worklet_registry);
    const EMSCRIPTEN_AUDIO_WORKLET_NODE_T node =
        emscripten_create_wasm_audio_worklet_node(context,
                                                  pending.m_processor.c_str(),
                                                  &options,
                                                  &render,
                                                  reinterpret_cast<void*>(key));
    if (node == 0) {
        free_state(slot);
        finish(pending, ANIRA_ERROR_UNKNOWN, 0);
        return;
    }
    slot.m_node = node;
    slot.m_open.store(1, std::memory_order::seq_cst);
    finish(pending, ANIRA_OK, node);
}

void fail_pending(ContextState& ctx, const std::string& processor) {
    std::vector<Pending> keep;
    for (const Pending& p : ctx.m_pending) {
        if (!processor.empty() && p.m_processor != processor) {
            keep.push_back(p);
            continue;
        }
        free_state(s_nodes[p.m_slot_index]);
        finish(p, ANIRA_ERROR_NOT_SUPPORTED, 0);
    }
    ctx.m_pending = std::move(keep);
}

void serve_pending(EMSCRIPTEN_WEBAUDIO_T context);

void processor_created(EMSCRIPTEN_WEBAUDIO_T context, bool success, void* user_data) {
    auto* name = static_cast<std::string*>(user_data);
    ContextState& ctx = s_contexts[context];
    ctx.m_processors[*name] = success ? 1 : -1;
    if (!success) { fail_pending(ctx, *name); }
    delete name;  // NOLINT(cppcoreguidelines-owning-memory)
    serve_pending(context);
}

// Creates the nodes whose processor is registered, registers the processors still missing.
void serve_pending(EMSCRIPTEN_WEBAUDIO_T context) {
    ContextState& ctx = s_contexts[context];
    std::vector<Pending> waiting;
    for (const Pending& p : ctx.m_pending) {
        auto found = ctx.m_processors.find(p.m_processor);
        if (found != ctx.m_processors.end() && found->second == 1) {
            create_node(context, p);
            continue;
        }
        if (found == ctx.m_processors.end()) {
            ctx.m_processors[p.m_processor] = 0;
            const std::vector<WebAudioParamDescriptor>& descriptors =
                s_param_descriptors[p.m_processor];
            static std::map<std::string, WebAudioWorkletProcessorCreateOptions> s_options;
            WebAudioWorkletProcessorCreateOptions& options = s_options[p.m_processor];
            options.name = s_param_descriptors.find(p.m_processor)->first.c_str();
            options.numAudioParams = static_cast<int>(descriptors.size());
            options.audioParamDescriptors = descriptors.data();
            emscripten_create_wasm_audio_worklet_processor_async(
                context,
                &options,
                &processor_created,
                new std::string(p.m_processor));  // NOLINT(cppcoreguidelines-owning-memory)
        }
        waiting.push_back(p);
    }
    ctx.m_pending = std::move(waiting);
}

void worklet_started(EMSCRIPTEN_WEBAUDIO_T context, bool success, void* /*user_data*/) {
    ContextState& ctx = s_contexts[context];
    ctx.m_started = success;
    ctx.m_failed = !success;
    if (!success) {
        fail_pending(ctx, "");
        return;
    }
    serve_pending(context);
}

// The processor of a node with `params` AudioParams: one per parameter count and context.
std::string processor_for(uint32_t params) {
    std::string name = "anira-node-" + std::to_string(params);
    if (s_param_descriptors.find(name) == s_param_descriptors.end()) {
        WebAudioParamDescriptor descriptor{};
        descriptor.defaultValue = 0.0F;
        descriptor.minValue = -3.4028235e38F;
        descriptor.maxValue = 3.4028235e38F;
        descriptor.automationRate = WEBAUDIO_PARAM_K_RATE;
        s_param_descriptors[name] = std::vector<WebAudioParamDescriptor>(params, descriptor);
    }
    return name;
}

const anira_tensor_spec* static_spec(const anira_handler& handler, uint32_t slot) {
    if (handler.m_pipeline.m_variants.empty() || slot >= handler.m_num_inputs) { return nullptr; }
    const anira_tensor_spec* spec =
        anira_model_config_input(&handler.m_pipeline.m_variants[0], slot);
    if (spec == nullptr || anira_tensor_spec_role(spec) != ANIRA_ROLE_STATIC) { return nullptr; }
    return spec;
}

}  // namespace

anira_status ANIRA_CALL anira_em_audio_node_create(anira_handler* handler,
                                                   uint32_t audio_context,
                                                   uint32_t input_channels,
                                                   uint32_t output_channels,
                                                   uint32_t input_slot,
                                                   uint32_t output_slot,
                                                   const uint32_t* param_slots,
                                                   uint32_t num_param_slots,
                                                   const uint32_t* tensor_slots,
                                                   uint32_t num_tensor_slots,
                                                   double wait_ms,
                                                   int32_t* out_status,
                                                   uint32_t* out_node,
                                                   anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(handler != nullptr && out_status != nullptr && out_node != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: NULL handler, out_status or out_node");
    ANIRA_CAPI_REQUIRE(handler->m_runnable.load(std::memory_order::acquire),
                       err,
                       ANIRA_ERROR_INVALID_STATE,
                       "audio node: the handler is not prepared");
    ANIRA_CAPI_REQUIRE(anira_contract_get_kind(&handler->m_contract) == ANIRA_CONTRACT_HARD,
                       err,
                       ANIRA_ERROR_CONFIG,
                       "audio node: the handler is not prepared under a Hard contract");
    ANIRA_CAPI_REQUIRE(output_channels > 0 && output_channels <= k_max_channels &&
                           input_channels <= k_max_channels,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: %u input and %u output channels (1 to %u outputs, at most "
                       "%u inputs)",
                       input_channels,
                       output_channels,
                       k_max_channels,
                       k_max_channels);
    ANIRA_CAPI_REQUIRE(num_param_slots <= k_max_params && (num_param_slots == 0 || param_slots),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: %u AudioParams (at most %u, with their slots)",
                       num_param_slots,
                       k_max_params);
    ANIRA_CAPI_REQUIRE(num_tensor_slots == 0 || tensor_slots != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: tensor slots without their list");
    const int quantum = emscripten_audio_context_quantum_size(static_cast<int>(audio_context));
    double block_min = 0;
    double block_max = 0;
    double rate = 0;
    static_cast<void>(
        anira_contract_hard_geometry(&handler->m_contract, &block_min, &block_max, &rate));
    ANIRA_CAPI_REQUIRE(quantum > 0 && block_min <= quantum && quantum <= block_max,
                       err,
                       ANIRA_ERROR_CONFIG,
                       "audio node: the context renders quanta of %d frames, the handler is "
                       "prepared for blocks of %.0f to %.0f",
                       quantum,
                       block_min,
                       block_max);

    auto state = std::make_unique<NodeState>();
    state->m_handler = handler;
    state->m_input_channels = input_channels;
    state->m_output_channels = output_channels;
    state->m_input_slot = input_slot;
    state->m_output_slot = output_slot;
    state->m_quantum = static_cast<uint32_t>(quantum);
    state->m_wait_ms = wait_ms;
    state->m_zeros.assign(static_cast<size_t>(quantum), 0.0F);
    state->m_input_shape = {static_cast<int64_t>(input_channels), quantum};
    state->m_output_shape = {static_cast<int64_t>(output_channels), quantum};
    for (uint32_t i = 0; i < num_param_slots; ++i) {
        const anira_tensor_spec* spec = static_spec(*handler, param_slots[i]);
        uint64_t elements = 1;
        const uint32_t ndim = spec != nullptr ? anira_tensor_spec_ndim(spec) : 0;
        for (uint32_t a = 0; a < ndim; ++a) {
            anira_axis_tag tag{};
            int64_t extent = 0;
            static_cast<void>(anira_tensor_spec_axis(spec, a, &tag, &extent));
            elements *= extent > 0 ? static_cast<uint64_t>(extent) : 0;
        }
        ANIRA_CAPI_REQUIRE(
            spec != nullptr && anira_tensor_spec_dtype(spec) == ANIRA_DTYPE_F32 && elements == 1,
            err,
            ANIRA_ERROR_CONFIG,
            "audio node: AudioParam %u drives input slot %u, which is not a "
            "float32 Static scalar",
            i,
            param_slots[i]);
        state->m_params.emplace_back();
        state->m_params.back().m_slot = param_slots[i];
    }
    for (ParamSlot& p : state->m_params) {
        // The spec's own rank and extents (all 1): the handler checks the tensor against them.
        const anira_tensor_spec* spec = static_spec(*handler, p.m_slot);
        const uint32_t ndim = anira_tensor_spec_ndim(spec);
        anira_tensor_init_host(&p.m_tensor,
                               &p.m_value,
                               ANIRA_DTYPE_F32,
                               ndim > 0 ? 1 : 0,
                               &p.m_shape);
        if (ndim > 1) {
            // A scalar of a higher rank: every extent 1, one element; the shape array is kept
            // inside the block for each axis.
            ANIRA_CAPI_REQUIRE(false,
                               err,
                               ANIRA_ERROR_NOT_SUPPORTED,
                               "audio node: input slot %u is a scalar of rank %u; an AudioParam "
                               "drives a rank-1 scalar ([1]); serve it as a tensor slot",
                               p.m_slot,
                               ndim);
        }
    }
    for (uint32_t i = 0; i < num_tensor_slots; ++i) {
        const anira_tensor_spec* spec = static_spec(*handler, tensor_slots[i]);
        ANIRA_CAPI_REQUIRE(spec != nullptr,
                           err,
                           ANIRA_ERROR_CONFIG,
                           "audio node: tensor slot %u is not a Static input",
                           tensor_slots[i]);
        auto block = std::make_unique<StaticBlock>();
        block->m_slot = tensor_slots[i];
        const anira_dtype dtype = anira_tensor_spec_dtype(spec);
        const uint32_t ndim = anira_tensor_spec_ndim(spec);
        ANIRA_CAPI_REQUIRE(ndim <= ANIRA_MAX_RANK,
                           err,
                           ANIRA_ERROR_CONFIG,
                           "audio node: tensor slot %u has rank %u",
                           tensor_slots[i],
                           ndim);
        uint64_t elements = 1;
        for (uint32_t a = 0; a < ndim; ++a) {
            anira_axis_tag tag{};
            int64_t extent = 0;
            static_cast<void>(anira_tensor_spec_axis(spec, a, &tag, &extent));
            ANIRA_CAPI_REQUIRE(extent >= 0,
                               err,
                               ANIRA_ERROR_CONFIG,
                               "audio node: tensor slot %u has a dynamic extent",
                               tensor_slots[i]);
            block->m_shape[a] = extent;
            elements *= static_cast<uint64_t>(extent);
        }
        const size_t bytes = static_cast<size_t>(
            elements * ((ANIRA_DTYPE_BITS(dtype) * ANIRA_DTYPE_LANES(dtype) + 7U) / 8U));
        block->m_value.assign(bytes, 0);
        block->m_copy.assign(bytes, 0);
        anira_tensor_init_host(&block->m_tensor,
                               block->m_copy.data(),
                               dtype,
                               ndim,
                               block->m_shape.data());
        state->m_blocks.push_back(std::move(block));
    }

    const auto context = static_cast<EMSCRIPTEN_WEBAUDIO_T>(audio_context);
    {
        auto known = s_contexts.find(context);
        ANIRA_CAPI_REQUIRE(known == s_contexts.end() || !known->second.m_failed,
                           err,
                           ANIRA_ERROR_NOT_SUPPORTED,
                           "audio node: the audio context's worklet thread failed to start");
    }
    uint32_t index = k_max_nodes;
    for (uint32_t i = 0; i < k_max_nodes; ++i) {
        if (s_nodes[i].m_state == nullptr && s_nodes[i].m_open.load() == 0) {
            index = i;
            break;
        }
    }
    ANIRA_CAPI_REQUIRE(index < k_max_nodes,
                       err,
                       ANIRA_ERROR_OUT_OF_MEMORY,
                       "audio node: %u nodes exist already",
                       k_max_nodes);
    NodeSlot& slot = s_nodes[index];
    slot.m_generation.fetch_add(1, std::memory_order::seq_cst);
    slot.m_state = state.release();
    __atomic_store_n(out_status, static_cast<int32_t>(ANIRA_INCOMPLETE), __ATOMIC_RELEASE);
    *out_node = 0;

    const bool fresh = s_contexts.find(context) == s_contexts.end();
    ContextState& ctx = s_contexts[context];
    ctx.m_pending.push_back(Pending{index, processor_for(num_param_slots), out_status, out_node});
    if (fresh) {
        ctx.m_stack = std::aligned_alloc(k_align, k_worklet_stack_bytes);
        if (ctx.m_stack == nullptr) {
            s_contexts.erase(context);
            free_state(slot);
            ANIRA_CAPI_REQUIRE(false,
                               err,
                               ANIRA_ERROR_OUT_OF_MEMORY,
                               "audio node: no stack for the worklet thread");
        }
        emscripten_start_wasm_audio_worklet_thread_async(context,
                                                         ctx.m_stack,
                                                         k_worklet_stack_bytes,
                                                         &worklet_started,
                                                         nullptr);
    } else if (ctx.m_started) {
        serve_pending(context);
    }
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_em_audio_node_set_static_input(uint32_t node,
                                                             uint32_t slot,
                                                             const void* bytes,
                                                             size_t size,
                                                             anira_error* err) ANIRA_NOEXCEPT try {
    for (NodeSlot& s : s_nodes) {
        if (s.m_node != static_cast<int>(node) || s.m_open.load() == 0 || s.m_state == nullptr) {
            continue;
        }
        for (auto& block : s.m_state->m_blocks) {
            if (block->m_slot != slot) { continue; }
            ANIRA_CAPI_REQUIRE(bytes != nullptr && size == block->m_value.size(),
                               err,
                               ANIRA_ERROR_INVALID_ARGUMENT,
                               "audio node: input slot %u takes %zu bytes, not %zu",
                               slot,
                               block->m_value.size(),
                               size);
            block->m_sequence.fetch_add(1, std::memory_order::acq_rel);
            std::memcpy(block->m_value.data(), bytes, size);
            block->m_sequence.fetch_add(1, std::memory_order::release);
            return ANIRA_OK;
        }
        ANIRA_CAPI_REQUIRE(false,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "audio node: it serves no tensor slot %u",
                           slot);
    }
    ANIRA_CAPI_REQUIRE(false, err, ANIRA_ERROR_INVALID_ARGUMENT, "audio node: no node %u", node);
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_em_audio_node_destroy(uint32_t node) ANIRA_NOEXCEPT try {
    for (NodeSlot& s : s_nodes) {
        if (s.m_node != static_cast<int>(node) || s.m_open.load() == 0) { continue; }
        s.m_open.store(0, std::memory_order::seq_cst);
        // A quantum inside the callback finishes first (it may wait up to the node's wait_ms).
        const double start = emscripten_get_now();
        while (s.m_busy.load(std::memory_order::seq_cst) != 0) {
            if (emscripten_get_now() - start > 5000.0) { break; }
        }
        emscripten_destroy_web_audio_node(s.m_node);
        s.m_node = 0;
        if (s.m_busy.load(std::memory_order::seq_cst) == 0) { free_state(s); }
        return;
    }
} catch (...) { anira::capi::report_void_failure(__func__); }
