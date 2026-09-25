# anira::onnxruntime — nothing engine-specific: the uniform prebuilt layout and
# the generic static/shared target cover it (ONNX Runtime's C API needs no extra
# definitions; its static archive links on demand like any other).

macro(_anira_wire_onnxruntime)
    _anira_resolve_backend_layout()
    _anira_define_generic_target()
endmacro()
