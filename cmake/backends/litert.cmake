# anira::litert — nothing engine-specific: the uniform prebuilt layout and the
# generic static/shared target cover it (LiteRT's static lib ships import stubs,
# so not even TFLite's TFL_COMPILE_LIBRARY is needed).

macro(_anira_wire_litert)
    _anira_resolve_backend_layout()
    _anira_define_generic_target()
endmacro()
