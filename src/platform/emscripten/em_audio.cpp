// anira/anira_em.h: the Wasm Audio Worklet node of a prepared handler. One render callback per
// quantum on the browser's audio thread, in C: the node's inputs and outputs are the handler's
// Streamed slots, their buffers its planar host tensors (no copy), one process_multi call per
// quantum over all of them, the miss rule is applied here, the Static inputs a
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
constexpr uint32_t k_max_streams = 32;
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

// One Streamed slot a node input or output serves: the slot, its channels and where its planes
// start in the node's plane array.
struct Stream {
    uint32_t m_slot = 0;
    uint32_t m_channels = 0;
    size_t m_first_plane = 0;
    std::array<int64_t, 2> m_shape{};
};

// What the render callback reads: built on the main thread at create, never resized after.
struct NodeState {
    anira_handler* m_handler = nullptr;
    uint32_t m_quantum = 0;
    double m_wait_ms = 0.0;
    std::vector<float> m_zeros;     ///< one quantum: an input channel not connected
    std::vector<Stream> m_inputs;   ///< node input k serves m_inputs[k]
    std::vector<Stream> m_outputs;  ///< node output k plays m_outputs[k]
    std::vector<void*> m_input_planes;
    std::vector<void*> m_output_planes;
    /// One tensor per slot of each side, as the multi form takes them: the handler's empty
    /// tensors, the served slots' entries rewritten every quantum.
    std::vector<anira_tensor> m_input_tensors;
    std::vector<anira_tensor> m_output_tensors;
    std::vector<size_t> m_delivered;
    std::vector<ParamSlot> m_params;
    std::vector<std::unique_ptr<StaticBlock>> m_blocks;
    bool m_counted = false;  ///< counted in the handler's m_audio_nodes (main thread)
};

// The node table: a slot per node, never freed, so a render callback that runs after its node
// was destroyed reads a generation that no longer matches and does nothing. The render callback
// marks itself busy before it reads the state, the destroy marks the slot closing and waits
// until the callback is not busy (both sequentially consistent, so one of them sees the other).
struct NodeSlot {
    std::atomic<uint32_t> m_generation{0};
    std::atomic<uint32_t> m_open{0};
    std::atomic<uint32_t> m_busy{0};
    /// 1 while anira_em_audio_node_prepare re-prepares the handler: the callback outputs silence
    /// and leaves the handler alone (the same handshake as the destroy's, with m_open kept).
    std::atomic<uint32_t> m_held{0};
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
    if (slot.m_held.load(std::memory_order::seq_cst) != 0) {
        // Re-preparing on the main thread: silence, the handler untouched.
        slot.m_busy.store(0, std::memory_order::seq_cst);
        silence(num_outputs, outputs);
        return true;
    }
    NodeState& state = *slot.m_state;
    const int frames = num_outputs > 0  ? outputs[0].samplesPerChannel
                       : num_inputs > 0 ? inputs[0].samplesPerChannel
                                        : 0;
    if (frames != static_cast<int>(state.m_quantum)) {
        // Not the quantum the handler is prepared for: silence, the handler untouched.
        silence(num_outputs, outputs);
        slot.m_busy.store(0, std::memory_order::release);
        return true;
    }
    push_params(state, num_params, params);
    pull_blocks(state);

    const size_t quantum = state.m_quantum;
    for (size_t k = 0; k < state.m_inputs.size(); ++k) {
        const Stream& stream = state.m_inputs[k];
        const bool present = k < static_cast<size_t>(num_inputs);
        for (uint32_t ch = 0; ch < stream.m_channels; ++ch) {
            const bool connected =
                present && ch < static_cast<uint32_t>(inputs[k].numberOfChannels);
            state.m_input_planes[stream.m_first_plane + ch] =
                connected ? static_cast<void*>(inputs[k].data + ch * quantum)
                          : static_cast<void*>(state.m_zeros.data());
        }
        anira_tensor_init_host_planar(&state.m_input_tensors[stream.m_slot],
                                      &state.m_input_planes[stream.m_first_plane],
                                      stream.m_channels,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      stream.m_shape.data());
    }
    for (size_t k = 0; k < state.m_outputs.size(); ++k) {
        const Stream& stream = state.m_outputs[k];
        for (uint32_t ch = 0; ch < stream.m_channels; ++ch) {
            state.m_output_planes[stream.m_first_plane + ch] = outputs[k].data + ch * quantum;
        }
        anira_tensor_init_host_planar(&state.m_output_tensors[stream.m_slot],
                                      &state.m_output_planes[stream.m_first_plane],
                                      stream.m_channels,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      stream.m_shape.data());
    }
    const auto num_in = static_cast<uint32_t>(state.m_input_tensors.size());
    const auto num_out = static_cast<uint32_t>(state.m_output_tensors.size());
    const anira_status status =
        state.m_wait_ms > 0.0 ? anira_handler_process_multi_wait(state.m_handler,
                                                                 state.m_input_tensors.data(),
                                                                 num_in,
                                                                 state.m_output_tensors.data(),
                                                                 num_out,
                                                                 state.m_delivered.data(),
                                                                 state.m_wait_ms)
                              : anira_handler_process_multi(state.m_handler,
                                                            state.m_input_tensors.data(),
                                                            num_in,
                                                            state.m_output_tensors.data(),
                                                            num_out,
                                                            state.m_delivered.data());
    // The miss rule: ANIRA_MISSED delivers 0 by the ABI, yet the miss policy (bypass, hold the
    // last, a callback) filled every output's whole block, which is the output already; a
    // failure is silence; any other success is what each output was delivered, the rest silence.
    if (status != ANIRA_MISSED) {
        for (size_t k = 0; k < state.m_outputs.size(); ++k) {
            const Stream& stream = state.m_outputs[k];
            const size_t delivered = state.m_delivered[stream.m_slot];
            const size_t filled =
                ANIRA_FAILED(status) ? 0 : (delivered < quantum ? delivered : quantum);
            if (filled == quantum) { continue; }
            for (uint32_t ch = 0; ch < stream.m_channels; ++ch) {
                std::memset(outputs[k].data + ch * quantum + filled,
                            0,
                            (quantum - filled) * sizeof(float));
            }
        }
    }
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

// The node no longer streams through its handler: anira_handler_prepare is legal again once no
// node does.
void uncount(NodeState& state) {
    if (!state.m_counted) { return; }
    state.m_counted = false;
    state.m_handler->m_audio_nodes.fetch_sub(1, std::memory_order_acq_rel);
}

void free_state(NodeSlot& slot) {
    if (slot.m_state != nullptr) { uncount(*slot.m_state); }
    delete slot.m_state;  // NOLINT(cppcoreguidelines-owning-memory)
    slot.m_state = nullptr;
}

// Waits, on the main thread, until the slot's render callback is outside its call; false after
// the bound. The callback never waits for this.
bool wait_until_idle(NodeSlot& slot, double bound_ms) {
    const double start = emscripten_get_now();
    while (slot.m_busy.load(std::memory_order::seq_cst) != 0) {
        if (emscripten_get_now() - start > bound_ms) { return false; }
    }
    return true;
}

// The handler's empty tensors at every position of the multi call, with the served slots'
// entries rewritten each quantum; rebuilt after a re-prepare, which rebuilds the handler's.
void copy_empty_tensors(NodeState& state) {
    state.m_input_tensors = state.m_handler->m_input_tensors;
    state.m_output_tensors = state.m_handler->m_output_tensors;
    state.m_delivered.assign(state.m_output_tensors.size(), 0);
}

void create_node(EMSCRIPTEN_WEBAUDIO_T context, const Pending& pending) {
    NodeSlot& slot = s_nodes[pending.m_slot_index];
    NodeState& state = *slot.m_state;
    std::array<int, k_max_streams> output_channels{};
    for (size_t k = 0; k < state.m_outputs.size(); ++k) {
        output_channels[k] = static_cast<int>(state.m_outputs[k].m_channels);
    }
    // One channelCount for every input of a node: the largest slot's; each slot reads its first.
    uint32_t input_channels = 1;
    for (const Stream& stream : state.m_inputs) {
        input_channels = stream.m_channels > input_channels ? stream.m_channels : input_channels;
    }
    EmscriptenAudioWorkletNodeCreateOptions options{};
    options.numberOfInputs = static_cast<int>(state.m_inputs.size());
    options.numberOfOutputs = static_cast<int>(state.m_outputs.size());
    options.outputChannelCounts = output_channels.data();
    options.channelCount = static_cast<int>(input_channels);
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

// The Streamed slots of one side a node serves: the listed ones, or every Streamed slot in slot
// order for NULL. Checked against the handler's ports (prepared: their channels are set).
anira_status resolve_streams(const anira_handler& handler,
                             bool input,
                             const uint32_t* slots,
                             uint32_t num_slots,
                             bool partial,
                             std::vector<Stream>& streams,
                             anira_error* err) {
    const std::vector<anira::capi::Port>& ports =
        input ? handler.m_input_ports : handler.m_output_ports;
    const char* side = input ? "input" : "output";
    std::vector<uint32_t> listed;
    if (slots == nullptr) {
        for (uint32_t s = 0; s < ports.size(); ++s) {
            if (anira::capi::stream_port(ports, s) != nullptr) { listed.push_back(s); }
        }
    } else {
        ANIRA_CAPI_REQUIRE(num_slots <= k_max_streams,
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "audio node: %u %s slots (at most %u)",
                           num_slots,
                           side,
                           k_max_streams);
        listed.assign(slots, slots + num_slots);
    }
    ANIRA_CAPI_REQUIRE(listed.size() <= k_max_streams,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: the model has %zu Streamed %s slots (a node serves at most %u)",
                       listed.size(),
                       side,
                       k_max_streams);
    std::vector<bool> served(ports.size(), false);
    size_t planes = 0;
    for (const uint32_t s : listed) {
        const anira::capi::StreamPort* port = anira::capi::stream_port(ports, s);
        ANIRA_CAPI_REQUIRE(port != nullptr,
                           err,
                           ANIRA_ERROR_CONFIG,
                           "audio node: %s slot %u is not a Streamed slot of the model",
                           side,
                           s);
        ANIRA_CAPI_REQUIRE(!served[s],
                           err,
                           ANIRA_ERROR_INVALID_ARGUMENT,
                           "audio node: %s slot %u is listed twice",
                           side,
                           s);
        ANIRA_CAPI_REQUIRE(port->m_ring_dtype == ANIRA_DTYPE_F32,
                           err,
                           ANIRA_ERROR_CONFIG,
                           "audio node: %s slot %u streams another dtype than float32",
                           side,
                           s);
        ANIRA_CAPI_REQUIRE(port->m_channels >= 1 && port->m_channels <= k_max_channels,
                           err,
                           ANIRA_ERROR_CONFIG,
                           "audio node: %s slot %u has %u channels (1 to %u)",
                           side,
                           s,
                           port->m_channels,
                           k_max_channels);
        served[s] = true;
        Stream stream;
        stream.m_slot = s;
        stream.m_channels = port->m_channels;
        stream.m_first_plane = planes;
        planes += port->m_channels;
        streams.push_back(stream);
    }
    if (!partial) {
        for (uint32_t s = 0; s < ports.size(); ++s) {
            ANIRA_CAPI_REQUIRE(anira::capi::stream_port(ports, s) == nullptr || served[s],
                               err,
                               ANIRA_ERROR_CONFIG,
                               "audio node: no %s of the node serves the Streamed %s slot %u "
                               "(every Streamed slot is served by one, or pass "
                               "ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS)",
                               side,
                               side,
                               s);
        }
    }
    return ANIRA_OK;
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
                                                   const uint32_t* input_slots,
                                                   uint32_t num_input_slots,
                                                   const uint32_t* output_slots,
                                                   uint32_t num_output_slots,
                                                   const uint32_t* param_slots,
                                                   uint32_t num_param_slots,
                                                   const uint32_t* tensor_slots,
                                                   uint32_t num_tensor_slots,
                                                   uint32_t flags,
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
    state->m_quantum = static_cast<uint32_t>(quantum);
    state->m_wait_ms = wait_ms;
    state->m_zeros.assign(static_cast<size_t>(quantum), 0.0F);
    const bool partial = (flags & ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS) != 0;
    anira_status resolved = resolve_streams(*handler,
                                            true,
                                            input_slots,
                                            num_input_slots,
                                            partial,
                                            state->m_inputs,
                                            err);
    if (resolved != ANIRA_OK) { return resolved; }
    resolved = resolve_streams(*handler,
                               false,
                               output_slots,
                               num_output_slots,
                               partial,
                               state->m_outputs,
                               err);
    if (resolved != ANIRA_OK) { return resolved; }
    ANIRA_CAPI_REQUIRE(!state->m_inputs.empty() || !state->m_outputs.empty(),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: it serves no Streamed slot (no input, no output)");
    size_t input_planes = 0;
    for (Stream& stream : state->m_inputs) {
        stream.m_shape = {static_cast<int64_t>(stream.m_channels), quantum};
        input_planes += stream.m_channels;
    }
    size_t output_planes = 0;
    for (Stream& stream : state->m_outputs) {
        stream.m_shape = {static_cast<int64_t>(stream.m_channels), quantum};
        output_planes += stream.m_channels;
    }
    state->m_input_planes.assign(input_planes, nullptr);
    state->m_output_planes.assign(output_planes, nullptr);
    // The handler's empty tensors (one per slot, built at prepare): the multi form's elements of
    // every slot the node does not serve, and of the Static ones (pushed separately).
    copy_empty_tensors(*state);
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
    slot.m_held.store(0, std::memory_order::seq_cst);
    handler->m_audio_nodes.fetch_add(1, std::memory_order_acq_rel);
    state->m_counted = true;
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

anira_status ANIRA_CALL anira_em_audio_node_prepare(uint32_t node,
                                                    const anira_contract* contract,
                                                    anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(contract != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: NULL contract");
    NodeSlot* found = nullptr;
    for (NodeSlot& s : s_nodes) {
        if (s.m_node == static_cast<int>(node) && s.m_open.load() != 0 && s.m_state != nullptr) {
            found = &s;
            break;
        }
    }
    ANIRA_CAPI_REQUIRE(found != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "audio node: no node %u",
                       node);
    NodeSlot& slot = *found;
    NodeState& state = *slot.m_state;
    ANIRA_CAPI_REQUIRE(anira_contract_get_kind(contract) == ANIRA_CONTRACT_HARD,
                       err,
                       ANIRA_ERROR_CONFIG,
                       "audio node: the contract is not Hard");
    double block_min = 0;
    double block_max = 0;
    double rate = 0;
    static_cast<void>(anira_contract_hard_geometry(contract, &block_min, &block_max, &rate));
    const auto quantum = static_cast<double>(state.m_quantum);
    ANIRA_CAPI_REQUIRE(block_min <= quantum && quantum <= block_max,
                       err,
                       ANIRA_ERROR_CONFIG,
                       "audio node: the context renders quanta of %.0f frames, the contract "
                       "declares blocks of %.0f to %.0f",
                       quantum,
                       block_min,
                       block_max);
    // Hold the callback off: from its next quantum it outputs silence; wait until the one in
    // flight returns. Then nothing on the audio thread touches the handler or this state.
    slot.m_held.store(1, std::memory_order::seq_cst);
    if (!wait_until_idle(slot, 5000.0)) {
        slot.m_held.store(0, std::memory_order::seq_cst);
        ANIRA_CAPI_REQUIRE(false,
                           err,
                           ANIRA_ERROR_INVALID_STATE,
                           "audio node: the render callback did not return within 5000 ms; "
                           "the handler is not re-prepared");
    }
    try {
        anira::capi::prepare_or_unprepare(*state.m_handler, *contract);
    } catch (...) {
        // Unprepared: the callback's process calls fail into silence until a prepare succeeds.
        copy_empty_tensors(state);
        slot.m_held.store(0, std::memory_order::seq_cst);
        throw;
    }
    // What depends on the prepare, rebuilt off the audio thread: the handler's empty tensors,
    // and the Static inputs, pushed again at the next quantum.
    copy_empty_tensors(state);
    for (ParamSlot& p : state.m_params) { p.m_pushed = false; }
    for (auto& block : state.m_blocks) {
        if (block->m_sequence.load(std::memory_order::acquire) != 0) { block->m_pushed = ~0U; }
    }
    slot.m_held.store(0, std::memory_order::seq_cst);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_em_audio_node_destroy(uint32_t node) ANIRA_NOEXCEPT try {
    for (NodeSlot& s : s_nodes) {
        if (s.m_node != static_cast<int>(node) || s.m_open.load() == 0) { continue; }
        s.m_open.store(0, std::memory_order::seq_cst);
        // A quantum inside the callback finishes first (it may wait up to the node's wait_ms).
        const bool idle = wait_until_idle(s, 5000.0);
        emscripten_destroy_web_audio_node(s.m_node);
        s.m_node = 0;
        if (s.m_state != nullptr) { uncount(*s.m_state); }
        if (idle) { free_state(s); }
        return;
    }
} catch (...) { anira::capi::report_void_failure(__func__); }
