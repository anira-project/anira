# ==============================================================================
# backends/onnxruntime.cmake — anira_setup_onnxruntime(<target>)
# ONNX Runtime — shared or static (on-demand) libonnxruntime; C API only.
# Included by cmake/AniraBackends.cmake, which provides anira_setup_backend() and the
# shared _anira_apply_backend_dirs / _anira_link_backend / anira_target_link_static_backend.
# ==============================================================================

macro(anira_setup_onnxruntime target)
    anira_setup_backend(onnxruntime)
    _anira_apply_backend_dirs(${target})
    target_sources(${target} PRIVATE "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/OnnxRuntimeProcessor.cpp")
    target_compile_definitions(${target} PUBLIC USE_ONNXRUNTIME)
    _anira_link_backend(${target} ONNXRUNTIME onnxruntime)
endmacro()
