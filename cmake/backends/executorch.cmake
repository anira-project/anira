# anira::executorch — ExecuTorch, static-only, consumed as ONE merged
# libexecutorch.a per platform (anira-project/backends >= v2.4.0): runtime +
# extensions + optimized/quantized CPU kernels + XNNPACK delegate, with the
# kernel/backend registrations pre-linked into one archive member, so it links
# on demand like the other static backends — no find_package, no force-load,
# no whole-archive microkernels. Windows has no partial link: the registration
# set ships separately as executorch_registrations.lib and is whole-archived.
#
# The headers vendor a c10 (given as an extra include dir) and need the
# runtime's compile definitions. LibTorch is shared-only and ExecuTorch
# static-only, so the vendored c10 can never shadow LibTorch's in one image.

macro(_anira_wire_executorch)
    _anira_resolve_backend_layout()
    set(_ab_extra_defs C10_USING_CUSTOM_GENERATED_MACROS ET_LOG_ENABLED=0 ET_USE_THREADPOOL)
    if(EXISTS "${_ab_rootdir}/include/executorch/runtime/core/portable_type/c10")
        set(_ab_extra_incdirs "${_ab_rootdir}/include/executorch/runtime/core/portable_type/c10")
    endif()
    if(TANH_BINARY_FORMAT STREQUAL "PE")
        set(ANIRA_EXECUTORCH_REGISTRATIONS_LIB "${_ab_libdir}/executorch_registrations.lib")
        set(ANIRA_EXECUTORCH_REGISTRATIONS_SUBPATH "executorch_registrations.lib")
        set(_ab_extra_link_libs "${ANIRA_EXECUTORCH_REGISTRATIONS_LIB}")
        set(_ab_extra_link_opts "/WHOLEARCHIVE:${ANIRA_EXECUTORCH_REGISTRATIONS_LIB}")
    endif()
    _anira_define_generic_target()
endmacro()
