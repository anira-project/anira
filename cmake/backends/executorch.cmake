# anira_setup_executorch(<target>) — ExecuTorch, static-only.
# One merged libexecutorch.a (runtime + extensions + optimized/quantized CPU kernels +
# XNNPACK) with the kernel/backend registrations pre-linked into one archive member, so
# it links on-demand like the other static backends. Windows has no partial link: the
# registration set ships as executorch_registrations.lib and is whole-archived
# (install.cmake adds the $<INSTALL_INTERFACE> side).
# The ExecuTorch headers vendor a c10 that must not shadow LibTorch's in other TUs, so
# they are given to ExecuTorchProcessor.cpp only, as plain -I (searched before the
# target's SYSTEM dirs), together with the runtime's compile definitions.
# Must be called before anira_setup_libtorch(): both bundle XNNPACK, and torch first on
# the link line binds the delegate's xnn_* symbols to libtorch's copy (runtime failure).

macro(anira_setup_executorch target)
    anira_setup_backend(executorch)
    _anira_apply_backend_dirs(${target})
    target_compile_definitions(${target} PUBLIC USE_EXECUTORCH)
    anira_target_link_static_backend(${target} "${ANIRA_EXECUTORCH_STATIC_LIB}")
    if(WIN32)
        target_link_libraries(${target} PUBLIC
            "$<BUILD_INTERFACE:${ANIRA_EXECUTORCH_REGISTRATIONS_LIB}>")
        target_link_options(${target} PUBLIC
            "$<BUILD_INTERFACE:/WHOLEARCHIVE:${ANIRA_EXECUTORCH_REGISTRATIONS_LIB}>")
    endif()
    set(_ab_et_src "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/ExecuTorchProcessor.cpp")
    target_sources(${target} PRIVATE "${_ab_et_src}")
    set_source_files_properties("${_ab_et_src}" PROPERTIES
        COMPILE_OPTIONS "-I${ANIRA_EXECUTORCH_INCLUDE_DIR};-I${ANIRA_EXECUTORCH_INCLUDE_DIR}/executorch/runtime/core/portable_type/c10"
        COMPILE_DEFINITIONS "${ANIRA_EXECUTORCH_COMPILE_DEFINITIONS}")
    unset(_ab_et_src)
endmacro()
