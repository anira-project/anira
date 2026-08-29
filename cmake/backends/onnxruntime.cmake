# anira_setup_onnxruntime(<target>) — ONNX Runtime, shared or static (C API, on-demand).

macro(anira_setup_onnxruntime target)
    anira_setup_backend(onnxruntime)
    _anira_apply_backend_dirs(${target})
    target_sources(${target} PRIVATE "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/OnnxRuntimeProcessor.cpp")
    target_compile_definitions(${target} PUBLIC USE_ONNXRUNTIME)
    _anira_link_backend(${target} ONNXRUNTIME onnxruntime)
endmacro()
