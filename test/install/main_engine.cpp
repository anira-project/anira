// A consumer of the installed anira that calls an engine itself: it includes the
// ONNX Runtime header and links anira::anira together with anira::onnxruntime — the
// same ONNX Runtime anira uses. Built twice by CMakeLists.txt: as an executable (run
// by the install test) and, with CONSUMER_ENGINE_MODULE defined, as a plugin-shaped
// module with hidden visibility whose export table the install test checks for
// engine symbols. Without anira::onnxruntime the engine include below must fail —
// the negative try_compile in CMakeLists.txt asserts that.
#include <anira/anira.h>  // NOLINT(misc-include-cleaner)
#include <onnxruntime_cxx_api.h>

#include <cstdio>

namespace {

int run() {
    // Resolves through the ONNX Runtime anira links (a mismatched copy would make
    // anira's OnnxRuntimeProcessor throw at construction).
    const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "anira-install-consumer");
    const anira::InferenceConfig config{};  // NOLINT(misc-include-cleaner)
    std::printf("onnxruntime %s, anira tensors %zu\n",
                Ort::GetVersionString().c_str(),
                config.get_tensor_input_size().size());
    return 0;
}

}  // namespace

#ifdef CONSUMER_ENGINE_MODULE
#ifdef _WIN32
#define CONSUMER_ENGINE_EXPORT __declspec(dllexport)
#else
#define CONSUMER_ENGINE_EXPORT __attribute__((visibility("default")))
#endif
extern "C" CONSUMER_ENGINE_EXPORT int consumer_engine_entry() {
    return run();
}
#else
int main() {
    return run();
}
#endif
