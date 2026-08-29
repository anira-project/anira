# ==============================================================================
# backends/executorch.cmake — anira_setup_executorch(<target>)
# ExecuTorch — static-only, one merged libexecutorch.a with pre-linked registrations.
# Included by cmake/AniraBackends.cmake, which provides anira_setup_backend() and the
# shared _anira_apply_backend_dirs / _anira_link_backend / anira_target_link_static_backend.
# ==============================================================================

macro(anira_setup_executorch target)
    anira_setup_backend(executorch)
    _anira_apply_backend_dirs(${target})
    target_compile_definitions(${target} PUBLIC USE_EXECUTORCH)
    # One merged libexecutorch.a per platform/ABI/xcframework slice (runtime +
    # extensions + optimized/quantized CPU kernels + XNNPACK delegate), built by
    # anira-project/backends with the kernel/backend registrations pre-linked into
    # a single archive member, so it links on-demand like the other static
    # backends — no force-load, no CMake package.
    anira_target_link_static_backend(${target} "${ANIRA_EXECUTORCH_STATIC_LIB}")
    if(WIN32)
        # COFF has no partial link, so the pre-linked registration member does not
        # exist there: the registration set (runtime core + kernel/backend
        # registrations + XNNPACK microkernels) ships as a second, small lib that
        # has to be whole-archived. install.cmake adds the $<INSTALL_INTERFACE> side.
        target_link_libraries(${target} PUBLIC
            "$<BUILD_INTERFACE:${ANIRA_EXECUTORCH_REGISTRATIONS_LIB}>")
        target_link_options(${target} PUBLIC
            "$<BUILD_INTERFACE:/WHOLEARCHIVE:${ANIRA_EXECUTORCH_REGISTRATIONS_LIB}>")
    endif()
    # ExecuTorchProcessor.cpp is the only TU that may see the ExecuTorch headers:
    # they vendor a copy of c10 that must never shadow LibTorch's real c10 in the
    # other backend TUs (it breaks the LibTorch backend at link time on MSVC). So
    # the ExecuTorch include dirs are not on the target (anira_setup_backend keeps
    # them out of BACKEND_BUILD_HEADER_DIRS); this one source gets them as plain
    # -I flags, which are searched before every SYSTEM (-isystem / /external:I)
    # dir of the target — so inside this TU the vendored c10 wins over LibTorch's.
    set(_ab_et_src "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/ExecuTorchProcessor.cpp")
    target_sources(${target} PRIVATE "${_ab_et_src}")
    set_source_files_properties("${_ab_et_src}" PROPERTIES
        COMPILE_OPTIONS "-I${ANIRA_EXECUTORCH_INCLUDE_DIR};-I${ANIRA_EXECUTORCH_INCLUDE_DIR}/executorch/runtime/core/portable_type/c10"
        COMPILE_DEFINITIONS "${ANIRA_EXECUTORCH_COMPILE_DEFINITIONS}")
    unset(_ab_et_src)
endmacro()
