# anira_setup_libtorch(<target>) — LibTorch, shared-only, via find_package(Torch).
# Fails at configure if ExecuTorch is enabled but not linked yet (XNNPACK ordering).
# TORCH_LIBRARIES holds absolute paths (libc10, kineto) that must not leak into the
# installed export: only torch / torch_library link PUBLIC, the rest PRIVATE. CMake < 3.26
# as a subproject cannot link torch_cpu publicly (cmake#24163) -> everything PRIVATE,
# TORCH_LIBRARIES_ALL_PRIVATE tells install.cmake; armv7l still needs torch_cpu PUBLIC.

macro(anira_setup_libtorch target)
    if(ANIRA_WITH_EXECUTORCH)
        get_target_property(_ab_links ${target} LINK_LIBRARIES)
        if(NOT _ab_links MATCHES "libexecutorch\\.a|executorch\\.lib")
            message(FATAL_ERROR "anira: call anira_setup_executorch() before anira_setup_libtorch() "
                                "(both bundle XNNPACK; the ExecuTorch archive must precede the torch libraries).")
        endif()
        unset(_ab_links)
    endif()
    anira_setup_backend(libtorch)
    _anira_apply_backend_dirs(${target})
    target_sources(${target} PRIVATE "${ANIRA_BACKENDS_CMAKE_DIR}/../src/backends/LibTorchProcessor.cpp")
    target_compile_definitions(${target} PUBLIC USE_LIBTORCH)
    if (CMAKE_VERSION VERSION_LESS "3.26.0" AND NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR))
        target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
        set(TORCH_LIBRARIES_ALL_PRIVATE TRUE)
        if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
            target_link_libraries(${target} PUBLIC torch_cpu)
        endif()
    else()
        foreach(TORCH_LIB ${TORCH_LIBRARIES})
            if(TORCH_LIB STREQUAL "torch" OR TORCH_LIB STREQUAL "torch_library")
                target_link_libraries(${target} PUBLIC ${TORCH_LIB})
            else()
                target_link_libraries(${target} PRIVATE ${TORCH_LIB})
            endif()
        endforeach()
    endif()
endmacro()
