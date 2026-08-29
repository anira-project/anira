# ==============================================================================
# backends/litert.cmake — anira_setup_litert(<target>)
# LiteRT (LiteRt* C API) — shared or static libLiteRt.
# Included by cmake/AniraBackends.cmake, which provides anira_setup_backend() and the
# shared _anira_apply_backend_dirs / _anira_link_backend / anira_target_link_static_backend.
# ==============================================================================

macro(anira_setup_litert target)
    anira_setup_backend(litert)
    _anira_apply_backend_dirs(${target})
    target_sources(${target} PRIVATE "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/LiteRtProcessor.cpp")
    target_compile_definitions(${target} PUBLIC USE_LITERT)
    # The desktop static archives are pre-isolated to the LiteRt* C API at
    # packaging (backends >= v2.3.0), so they coexist with ExecuTorch's XNNPACK —
    # see the note in AniraValidate.cmake.
    _anira_link_backend(${target} LITERT LiteRt)
endmacro()
