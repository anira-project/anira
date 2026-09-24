# anira::onnxruntime — the uniform prebuilt layout and the generic static/shared
# target cover the engine itself (ONNX Runtime's C API needs no extra definitions;
# its static archive links on demand like any other).
#
# The -gpu variant (ANIRA_ONNXRUNTIME_VARIANT=gpu) additionally carries the WebGPU EP,
# built by backends against an EXTERNAL Dawn that ships inside the same archive:
# lib/libwebgpu_dawn (webgpu_dawn.dll), include/{webgpu,dawn}, and DAWN_VERSION — the
# revision ORT's own cmake/deps.txt pins, so ORT, Dawn and the DawnProcTable layout are
# one versioned triple. ORT links only the dawn_proc thunks; the process's one Dawn is
# anira's (the Machine creates or borrows the instance/device on it and hands ORT
# `dawn::native::GetProcs()` as ep.webgpuexecutionprovider.dawnProcTable). That Dawn is
# exposed here as anira::webgpu_dawn, with ANIRA_ONNXRUNTIME_HAS_WEBGPU and
# ANIRA_ONNXRUNTIME_DAWN_VERSION for the revision assertion at anira_machine_create.

macro(_anira_wire_onnxruntime)
    _anira_resolve_backend_layout()
    _anira_define_generic_target()

    set(ANIRA_ONNXRUNTIME_HAS_WEBGPU FALSE)
    set(ANIRA_ONNXRUNTIME_DAWN_VERSION "")
    set(ANIRA_WEBGPU_DAWN_SHARED_LIB_SUBPATH "")
    set(ANIRA_WEBGPU_DAWN_IMPLIB_SUBPATH "")
    # upstream's presence marker for the WebGPU EP; the packages ship it only in -gpu.
    if(EXISTS "${_ab_incdir}/webgpu_provider_factory.h")
        set(ANIRA_ONNXRUNTIME_HAS_WEBGPU TRUE)
        if(EXISTS "${_ab_rootdir}/DAWN_VERSION")
            file(READ "${_ab_rootdir}/DAWN_VERSION" ANIRA_ONNXRUNTIME_DAWN_VERSION)
            string(STRIP "${ANIRA_ONNXRUNTIME_DAWN_VERSION}" ANIRA_ONNXRUNTIME_DAWN_VERSION)
        endif()
        if(NOT EXISTS "${_ab_incdir}/dawn/native/DawnNative.h")
            message(FATAL_ERROR "anira: the onnxruntime -gpu package advertises the WebGPU EP but ships no Dawn headers (include/dawn/native/DawnNative.h)")
        endif()
        _anira_locate_shared_lib("${_ab_libdir}" "${_ab_rootdir}" "webgpu_dawn" _ab_dawn_lib _ab_dawn_implib)
        anira_define_backend_target(webgpu_dawn SHARED GLOBAL
            LOCATION "${_ab_dawn_lib}" IMPLIB "${_ab_dawn_implib}"
            INCLUDE_DIRS "${_ab_incdir}"
            # the monolithic shared Dawn: import (not inline) the dawn_native + C entry points
            DEFINITIONS DAWN_NATIVE_SHARED_LIBRARY WGPU_SHARED_LIBRARY)
        # Install-subpath facts for the installed package's twin definition (install.cmake).
        foreach(_ab_kind dawn_lib dawn_implib)
            set(_ab_sub "")
            if(NOT _ab_${_ab_kind} STREQUAL "")
                foreach(_ab_dir lib bin)
                    file(RELATIVE_PATH _ab_rel "${_ab_rootdir}/${_ab_dir}" "${_ab_${_ab_kind}}")
                    if(NOT _ab_rel MATCHES "^\\.\\.")
                        set(_ab_sub "${_ab_rel}")
                        break()
                    endif()
                endforeach()
            endif()
            if(_ab_kind STREQUAL "dawn_lib")
                set(ANIRA_WEBGPU_DAWN_SHARED_LIB_SUBPATH "${_ab_sub}")
            else()
                set(ANIRA_WEBGPU_DAWN_IMPLIB_SUBPATH "${_ab_sub}")
            endif()
        endforeach()
        unset(_ab_kind)
        unset(_ab_sub)
        unset(_ab_dir)
        unset(_ab_rel)
        unset(_ab_dawn_lib)
        unset(_ab_dawn_implib)
        message(STATUS "anira: onnxruntime WebGPU EP present — anira::webgpu_dawn (Dawn ${ANIRA_ONNXRUNTIME_DAWN_VERSION})")
    endif()
endmacro()
