# anira_setup_litert(<target>) — LiteRT (LiteRt* C API), shared or static. The desktop
# static archives are pre-isolated to the LiteRt* API (backends >= v2.3.0) so they
# coexist with ExecuTorch's XNNPACK.

macro(anira_setup_litert target)
    anira_setup_backend(litert)
    _anira_apply_backend_dirs(${target})
    target_sources(${target} PRIVATE "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/LiteRtProcessor.cpp")
    target_compile_definitions(${target} PUBLIC USE_LITERT)
    _anira_link_backend(${target} LITERT LiteRt)
endmacro()
