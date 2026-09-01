# anira::tflite — the legacy TFLite C library.
#
# iOS ships a TensorFlowLiteC.framework xcframework: a static (pre-linked
# Mach-O) framework binary plus flat module headers, and a fat arm64+x86_64
# simulator slice. anira includes the headers by their canonical
# <tensorflow/lite/...c_api.h> paths, which the flat Headers/ don't provide, so
# a tiny generated shim tree forwards onto the framework's flat c_api.h and
# Headers/ is added for its (quote-included) siblings.

macro(_anira_wire_tflite)
    _anira_resolve_backend_layout()
    if(TANH_OPERATING_SYSTEM STREQUAL "iOS")
        if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
            set(_ab_slice "ios-arm64_x86_64-simulator")
        else()
            set(_ab_slice "ios-arm64")
        endif()
        set(_ab_fwk "${_ab_rootdir}/TensorFlowLiteC.xcframework/${_ab_slice}/TensorFlowLiteC.framework")
        set(ANIRA_${_ab_ID}_STATIC_LIB "${_ab_fwk}/TensorFlowLiteC")
        set(ANIRA_${_ab_ID}_STATIC_LIB_SUBPATH "TensorFlowLiteC.xcframework/${_ab_slice}/TensorFlowLiteC.framework/TensorFlowLiteC")
        set(ANIRA_${_ab_ID}_IOS_SLICE "${_ab_slice}")
        set(_ab_shim "${CMAKE_BINARY_DIR}/anira-tflite-ios-shim")
        set(ANIRA_${_ab_ID}_IOS_SHIM "${_ab_shim}")
        file(WRITE "${_ab_shim}/tensorflow/lite/c_api.h" "#include <c_api.h>\n")
        file(WRITE "${_ab_shim}/tensorflow/lite/core/c/c_api.h" "#include <c_api.h>\n")
        set(_ab_incdir "${_ab_shim}" "${_ab_fwk}/Headers")
        set(_ab_libdir "${_ab_fwk}")
        unset(_ab_fwk)
        unset(_ab_shim)
    endif()
    if(_ab_linkage STREQUAL "static")
        # The TFLite C API headers default to __declspec(dllimport) on Windows;
        # linking the static archive then leaves __imp_TfLite* unresolved (no
        # import stubs). TFL_COMPILE_LIBRARY switches the decoration to a direct
        # reference (a no-op elsewhere).
        set(_ab_extra_defs TFL_COMPILE_LIBRARY)
    endif()
    _anira_define_generic_target()
endmacro()
