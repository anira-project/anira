cmake_minimum_required(VERSION 3.24)
project(minimal-executorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-executorch
  minimal-executorch.cpp
)

# Deliberately NOT linked against anira. anira embeds its own copy of the
# ExecuTorch runtime, and mixing that copy with a second one linked here is
# fragile: libanira exports the runtime's registry symbols (register_kernels
# etc.), and depending on link order this executable's force-loaded kernel
# tables can bind to libanira's registry — which already holds the same kernels
# — and abort with a duplicate-registration error at startup. Linking only the
# static ExecuTorch libraries keeps this example's runtime fully self-contained.
target_link_libraries(minimal-executorch
  executorch executorch_extensions executorch_kernels xnnpack_backend
)
# anira strips the compile usage requirements off the imported ExecuTorch targets
# (their vendored c10 headers must not leak into anira's own TUs), so this example
# adds the include dirs and the collected compile definitions explicitly.
target_include_directories(minimal-executorch SYSTEM PRIVATE
  "${ANIRA_EXECUTORCH_ROOTDIR}/include"
  "${ANIRA_EXECUTORCH_ROOTDIR}/include/executorch/runtime/core/portable_type/c10"
)
target_compile_definitions(minimal-executorch PRIVATE ${ANIRA_EXECUTORCH_COMPILE_DEFINITIONS})
if(TARGET xnnpack-microkernels-prod)
    # See the matching comment in the top-level CMakeLists.txt.
    target_link_libraries(minimal-executorch "$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack-microkernels-prod>")
endif()

# Guard the ordering constraint above at configure time: if the anira target ever
# gets linked before the ExecuTorch archives, the registry symbols resolve from
# libanira and the example aborts at startup with a duplicate kernel registration.
get_target_property(_minimal_et_link_order minimal-executorch LINK_LIBRARIES)
if(_minimal_et_link_order)
    list(FIND _minimal_et_link_order "executorch" _minimal_et_index)
    list(FIND _minimal_et_link_order "anira::anira" _minimal_anira_index)
    if(_minimal_et_index EQUAL -1
       OR (_minimal_anira_index GREATER -1 AND _minimal_et_index GREATER _minimal_anira_index))
        message(FATAL_ERROR
            "minimal-executorch: link-order violation — the ExecuTorch targets must be "
            "linked before anira (see the comment above).")
    endif()
    unset(_minimal_et_index)
    unset(_minimal_anira_index)
endif()
unset(_minimal_et_link_order)
