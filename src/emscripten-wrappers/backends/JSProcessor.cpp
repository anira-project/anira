#include <anira/anira.h>
#include <emscripten/emscripten.h>

EM_JS(void, process_buffers_js, (uintptr_t self_ptr, uintptr_t input_ptr, uintptr_t output_ptr), {
    return Module.processBuffers(self_ptr, input_ptr, output_ptr);
});

class JSProcessor : public anira::BackendBase {
public:
    JSProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void prepare() override {}

    void wasm_process(std::vector<anira::BufferF>& input,
                      std::vector<anira::BufferF>& output,
                      [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) {
        // Call BackendBase's process to handle the simple pass-through logic
        BackendBase::process(input, output, session);
    }

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        // Call JavaScript to do the processing, passing our own pointer for dispatch
        process_buffers_js(reinterpret_cast<uintptr_t>(this),
                           reinterpret_cast<uintptr_t>(&input),
                           reinterpret_cast<uintptr_t>(&output));
    }
};

// ------ JSProcessor C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t jsprocessor_create(uintptr_t inference_config_ptr) {
    auto* config = reinterpret_cast<anira::InferenceConfig*>(inference_config_ptr);
    return reinterpret_cast<uintptr_t>(new JSProcessor(*config));
}

EMSCRIPTEN_KEEPALIVE
void jsprocessor_wasm_process(uintptr_t self_ptr, uintptr_t input_ptr, uintptr_t output_ptr) {
    auto* processor = reinterpret_cast<JSProcessor*>(self_ptr);
    auto* input = reinterpret_cast<std::vector<anira::BufferF>*>(input_ptr);
    auto* output = reinterpret_cast<std::vector<anira::BufferF>*>(output_ptr);
    processor->wasm_process(*input, *output, nullptr);
}

EMSCRIPTEN_KEEPALIVE
void jsprocessor_destroy(uintptr_t ptr) {
    delete reinterpret_cast<JSProcessor*>(ptr);
}

}  // extern "C"
