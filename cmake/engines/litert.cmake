# anira::litert — the uniform prebuilt layout and the generic static/shared target (LiteRT's
# static lib ships import stubs, so not even TFLite's TFL_COMPILE_LIBRARY is needed), plus one
# fact about the package: whether its library exports LiteRT's accelerator query
# (LiteRtGetNumAccelerators, LiteRtGetAccelerator, LiteRtGetAcceleratorHardwareSupport of
# litert/c/internal/litert_accelerator.h). Every static archive carries it, and so do the shared
# libraries whose export list takes every LiteRt* name (Linux, Android, macOS); the Windows DLL's
# export list (lib/LiteRt.def) holds the public C API and the accelerator registration entries,
# not the query. ANIRA_LITERT_ACCELERATOR_QUERY tells the adapter it may ask; without it the
# capabilities list LiteRT on the default provider alone and a named accelerator is refused at
# load, the reason saying so.

macro(_anira_wire_litert)
    _anira_resolve_backend_layout()
    _anira_define_generic_target()
    if(_ab_linkage STREQUAL "static" OR NOT TANH_BINARY_FORMAT STREQUAL "PE")
        set_property(TARGET anira::litert APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS ANIRA_LITERT_ACCELERATOR_QUERY)
    endif()
endmacro()
