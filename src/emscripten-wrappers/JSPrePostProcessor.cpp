#include <anira/anira.h>
#include <emscripten/emscripten.h>

EM_JS(void, process_prepost_js,
      (uintptr_t self_ptr, uintptr_t input_ptr, uintptr_t output_ptr, int backend,
       int phase),
      {
        return Module.processPrePost(
          self_ptr,
          input_ptr,
          output_ptr,
          backend,
          phase
        );
      });

class JSPrePostProcessor : public anira::PrePostProcessor {
public:
  explicit JSPrePostProcessor(anira::InferenceConfig &inference_config)
      : anira::PrePostProcessor(inference_config) {
  }

  void pre_process(std::vector<anira::RingBuffer> &input,
                   std::vector<anira::BufferF> &output,
                   anira::InferenceBackend current_inference_backend) override {
    process_prepost_js(reinterpret_cast<uintptr_t>(this),
                       reinterpret_cast<uintptr_t>(&input),
                       reinterpret_cast<uintptr_t>(&output),
                       static_cast<int>(current_inference_backend), 0);
  }

  void post_process(std::vector<anira::BufferF> &input,
                    std::vector<anira::RingBuffer> &output,
                    anira::InferenceBackend current_inference_backend) override {
    process_prepost_js(reinterpret_cast<uintptr_t>(this),
                       reinterpret_cast<uintptr_t>(&input),
                       reinterpret_cast<uintptr_t>(&output),
                       static_cast<int>(current_inference_backend), 1);
  }

  void wasm_pre_process(std::vector<anira::RingBuffer> &input,
                        std::vector<anira::BufferF> &output,
                        anira::InferenceBackend current_inference_backend) {
    PrePostProcessor::pre_process(input, output, current_inference_backend);
  }

  void wasm_post_process(std::vector<anira::BufferF> &input,
                         std::vector<anira::RingBuffer> &output,
                         anira::InferenceBackend current_inference_backend) {
    PrePostProcessor::post_process(input, output, current_inference_backend);
  }
};

extern "C" {

EMSCRIPTEN_KEEPALIVE
uintptr_t jsprepostprocessor_create(uintptr_t inference_config_ptr) {
  auto *config = reinterpret_cast<anira::InferenceConfig *>(inference_config_ptr);
  return reinterpret_cast<uintptr_t>(new JSPrePostProcessor(*config));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_wasm_pre_process(uintptr_t self_ptr, uintptr_t input_ptr,
                                         uintptr_t output_ptr, int backend) {
  auto *processor = reinterpret_cast<JSPrePostProcessor *>(self_ptr);
  auto *input = reinterpret_cast<std::vector<anira::RingBuffer> *>(input_ptr);
  auto *output = reinterpret_cast<std::vector<anira::BufferF> *>(output_ptr);
  processor->wasm_pre_process(*input, *output,
                              static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_wasm_post_process(uintptr_t self_ptr, uintptr_t input_ptr,
                                          uintptr_t output_ptr, int backend) {
  auto *processor = reinterpret_cast<JSPrePostProcessor *>(self_ptr);
  auto *input = reinterpret_cast<std::vector<anira::BufferF> *>(input_ptr);
  auto *output = reinterpret_cast<std::vector<anira::RingBuffer> *>(output_ptr);
  processor->wasm_post_process(*input, *output,
                               static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
void jsprepostprocessor_destroy(uintptr_t ptr) {
  delete reinterpret_cast<JSPrePostProcessor *>(ptr);
}

} // extern "C"