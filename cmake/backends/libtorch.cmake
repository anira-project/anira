# ==============================================================================
# backends/libtorch.cmake — anira_setup_libtorch(<target>)
# LibTorch — shared-only, wired through its own CMake package (find_package(Torch)).
# Included by cmake/AniraBackends.cmake, which provides anira_setup_backend() and the
# shared _anira_apply_backend_dirs / _anira_link_backend / anira_target_link_static_backend.
# ==============================================================================

macro(anira_setup_libtorch target)
    # Must come AFTER anira_setup_executorch(): libtorch_cpu exports its own
    # (different) XNNPACK, and if the torch dylibs precede the ExecuTorch archive
    # on the link line, part of the delegate's xnn_* references bind to libtorch's
    # copy and delegate init fails at runtime. LINK_LIBRARIES preserves call order,
    # so a wrong order is caught here at configure time.
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
    # The find_package(Torch) adds the libraries libc10.so and libkineto.a as full paths to ${TORCH_LIBRARIES}. This is no problem when we add anira as a subdirectory to another project, but when we install the library, the torch libraries will be link targets of the anira library with full paths and hence not found on other systems. Therefore, we link those libs privately and only add the torch target publicly.
    # Also until cmake 3.26, there is a bug where the torch_cpu library is not found when linking publicly https://gitlab.kitware.com/cmake/cmake/-/issues/24163 and anira is added as a subdirectory to another project, see
    # But this is necessary for when we install the library since otherwise symbols are not found
    # Another problem are that on armv7l with benchmarking enabled, some symbols are not found when linking the torch_cpu library privately
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
