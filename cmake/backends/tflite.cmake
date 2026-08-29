# ==============================================================================
# backends/tflite.cmake — anira_setup_tflite(<target>)
# Legacy TensorFlow Lite (TfLite* C API) — shared or static libtensorflowlite_c.
# Included by cmake/AniraBackends.cmake, which provides anira_setup_backend() and the
# shared _anira_apply_backend_dirs / _anira_link_backend / anira_target_link_static_backend.
# ==============================================================================

macro(anira_setup_tflite target)
    anira_setup_backend(tflite)
    _anira_apply_backend_dirs(${target})
    target_sources(${target} PRIVATE "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/TFLiteProcessor.cpp")
    target_compile_definitions(${target} PUBLIC USE_TFLITE)
    _anira_link_backend(${target} TFLITE tensorflowlite_c)
endmacro()
