# anira::libtorch — the one engine with its own CMake package: wired via
# find_package(Torch), then wrapped as an INTERFACE target over torch,
# torch_library and what the package lists by absolute path.

macro(_anira_wire_libtorch)
    if(_ab_byo STREQUAL "" AND NOT _ab_arch STREQUAL "armv7l")
        list(APPEND CMAKE_PREFIX_PATH "${_ab_rootdir}")
        find_package(Torch REQUIRED)
    endif()
    # -w silences the (many) warnings from the prebuilt torch headers.
    if(TARGET torch)
        target_link_options(torch INTERFACE "-w")
    endif()
    if(TARGET torch_library)
        target_link_options(torch_library INTERFACE "-w")
    endif()
    set(LIBTORCH_ROOTDIR "${_ab_rootdir}")
    set(ANIRA_LIBTORCH_ROOTDIR "${_ab_rootdir}")
    set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${_ab_rootdir}")
    # anira::libtorch wraps the package: torch + torch_library + what
    # find_package(Torch) lists by absolute path (libc10, libkineto, ...), with the
    # two-level include layout. TORCH_CXX_FLAGS (the libstdc++ ABI switch) rides
    # on the torch target's interface; CMakeLists.txt additionally makes it a
    # PUBLIC requirement of anira, since it decides anira's own std:: ABI.
    set(_ab_torch_libs torch torch_library ${TORCH_LIBRARIES})
    list(REMOVE_DUPLICATES _ab_torch_libs)
    anira_define_backend_target(libtorch INTERFACE GLOBAL
        LINK_LIBRARIES ${_ab_torch_libs}
        INCLUDE_DIRS "${_ab_rootdir}/include" "${_ab_rootdir}/include/torch/csrc/api/include")
    unset(_ab_torch_libs)
endmacro()
