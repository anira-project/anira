#include <anira/anira.h>
#include <emscripten/emscripten.h>

#include <atomic>

// process_prepost_js multiplexes all four PrePostProcessor JS callbacks over a
// single Module hook, keyed by `phase`:
//   0 = pre_process, 1 = post_process   -- run on the audio worklet thread at
//       submission/collection time; both buffer vectors are meaningful.
//   2 = before_inference (buffers in input_ptr, output_ptr = 0),
//   3 = after_inference  (buffers in output_ptr, input_ptr = 0)   -- run on the
//       inference worker thread, immediately around the backend; each carries a
//       single std::vector<BufferF>.
// The phase determines which JS thread's Module.processPrePost actually fires,
// so the worklet and inference-worker dispatchers only ever see their own phases.
EM_JS(void,
      process_prepost_js,
      (uintptr_t self_ptr, uintptr_t input_ptr, uintptr_t output_ptr, int backend, int phase),
      { return Module.processPrePost(self_ptr, input_ptr, output_ptr, backend, phase); });

class JSPrePostProcessor : public anira::PrePostProcessor {
public:
    explicit JSPrePostProcessor(anira::InferenceConfig& inference_config)
        : anira::PrePostProcessor(inference_config) {}

    void pre_process(std::vector<anira::RingBuffer>& input,
                     std::vector<anira::BufferF>& output,
                     anira::InferenceBackend current_inference_backend) override {
        process_prepost_js(reinterpret_cast<uintptr_t>(this),
                           reinterpret_cast<uintptr_t>(&input),
                           reinterpret_cast<uintptr_t>(&output),
                           static_cast<int>(current_inference_backend),
                           0);
    }

    void post_process(std::vector<anira::BufferF>& input,
                      std::vector<anira::RingBuffer>& output,
                      anira::InferenceBackend current_inference_backend) override {
        process_prepost_js(reinterpret_cast<uintptr_t>(this),
                           reinterpret_cast<uintptr_t>(&input),
                           reinterpret_cast<uintptr_t>(&output),
                           static_cast<int>(current_inference_backend),
                           1);
    }

    // before_inference/after_inference fire on the inference worker for every
    // inference. Their C++ default is a no-op, so unless JS has actually opted
    // in (a JSPrePostProcessor subclass registered on the inference worker via
    // AniraWeb.registerPrePostProcessor(), which flips m_js_inference_hooks),
    // skip the JS boundary crossing entirely and behave like the base class.
    void before_inference(std::vector<anira::BufferF>& input,
                          anira::InferenceBackend current_inference_backend) override {
        if (!m_js_inference_hooks.load(std::memory_order::acquire)) { return; }
        process_prepost_js(reinterpret_cast<uintptr_t>(this),
                           reinterpret_cast<uintptr_t>(&input),
                           0,
                           static_cast<int>(current_inference_backend),
                           2);
    }

    void after_inference(std::vector<anira::BufferF>& output,
                         anira::InferenceBackend current_inference_backend) override {
        if (!m_js_inference_hooks.load(std::memory_order::acquire)) { return; }
        process_prepost_js(reinterpret_cast<uintptr_t>(this),
                           0,
                           reinterpret_cast<uintptr_t>(&output),
                           static_cast<int>(current_inference_backend),
                           3);
    }

    void wasm_pre_process(std::vector<anira::RingBuffer>& input,
                          std::vector<anira::BufferF>& output,
                          anira::InferenceBackend current_inference_backend) {
        PrePostProcessor::pre_process(input, output, current_inference_backend);
    }

    void wasm_post_process(std::vector<anira::BufferF>& input,
                           std::vector<anira::RingBuffer>& output,
                           anira::InferenceBackend current_inference_backend) {
        PrePostProcessor::post_process(input, output, current_inference_backend);
    }

    void wasm_before_inference(std::vector<anira::BufferF>& input,
                               anira::InferenceBackend current_inference_backend) {
        PrePostProcessor::before_inference(input, current_inference_backend);
    }

    void wasm_after_inference(std::vector<anira::BufferF>& output,
                              anira::InferenceBackend current_inference_backend) {
        PrePostProcessor::after_inference(output, current_inference_backend);
    }

    // Toggled from the inference worker when a subclass is (de)registered there.
    // Lives in shared WASM memory and is read on every inference by every
    // inference worker, so it must be atomic.
    void set_inference_hooks(bool enabled) {
        m_js_inference_hooks.store(enabled, std::memory_order::release);
    }

private:
    std::atomic<bool> m_js_inference_hooks{false};
};

extern "C" {

EMSCRIPTEN_KEEPALIVE
uintptr_t jsprepostprocessor_create(uintptr_t inference_config_ptr) {
    auto* config = reinterpret_cast<anira::InferenceConfig*>(inference_config_ptr);
    return reinterpret_cast<uintptr_t>(new JSPrePostProcessor(*config));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_wasm_pre_process(uintptr_t self_ptr,
                                         uintptr_t input_ptr,
                                         uintptr_t output_ptr,
                                         int backend) {
    auto* processor = reinterpret_cast<JSPrePostProcessor*>(self_ptr);
    auto* input = reinterpret_cast<std::vector<anira::RingBuffer>*>(input_ptr);
    auto* output = reinterpret_cast<std::vector<anira::BufferF>*>(output_ptr);
    processor->wasm_pre_process(*input, *output, static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_wasm_post_process(uintptr_t self_ptr,
                                          uintptr_t input_ptr,
                                          uintptr_t output_ptr,
                                          int backend) {
    auto* processor = reinterpret_cast<JSPrePostProcessor*>(self_ptr);
    auto* input = reinterpret_cast<std::vector<anira::BufferF>*>(input_ptr);
    auto* output = reinterpret_cast<std::vector<anira::RingBuffer>*>(output_ptr);
    processor->wasm_post_process(*input, *output, static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_wasm_before_inference(uintptr_t self_ptr,
                                              uintptr_t buffers_ptr,
                                              int backend) {
    auto* processor = reinterpret_cast<JSPrePostProcessor*>(self_ptr);
    auto* buffers = reinterpret_cast<std::vector<anira::BufferF>*>(buffers_ptr);
    processor->wasm_before_inference(*buffers, static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_wasm_after_inference(uintptr_t self_ptr,
                                             uintptr_t buffers_ptr,
                                             int backend) {
    auto* processor = reinterpret_cast<JSPrePostProcessor*>(self_ptr);
    auto* buffers = reinterpret_cast<std::vector<anira::BufferF>*>(buffers_ptr);
    processor->wasm_after_inference(*buffers, static_cast<anira::InferenceBackend>(backend));
}

// Enable (1) or disable (0) the JS before/after_inference callbacks for this
// processor. Called by the inference worker when the matching subclass is
// registered/unregistered there; gates the boundary crossing in
// before_inference()/after_inference() above.
EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_set_inference_hooks(uintptr_t self_ptr, int enabled) {
    reinterpret_cast<JSPrePostProcessor*>(self_ptr)->set_inference_hooks(enabled != 0);
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_destroy(uintptr_t ptr) {
    delete reinterpret_cast<JSPrePostProcessor*>(ptr);
}

}  // extern "C"