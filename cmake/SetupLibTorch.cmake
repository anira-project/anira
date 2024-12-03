# torch stopped uploading the binaries for macOS x86_64, but we use self-built binaries
if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7a")
    set(LIBTORCH_VERSION 2.5.1)
else()
    set(LIBTORCH_VERSION 2.4.1)
endif()

if (NOT WIN32)
    set(TORCH_BUILD_TYPE "")
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(TORCH_BUILD_TYPE "-debug")
    else()
        set(TORCH_BUILD_TYPE "-release")
    endif()
endif()

option(LIBTORCH_ROOTDIR "libtorch root dir")
set(LIBTORCH_DIR_NAME "libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}")
set(LIBTORCH_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/${LIBTORCH_DIR_NAME})

if(EXISTS ${LIBTORCH_ROOTDIR}/)
    message(STATUS "Libtorch-Runtime library found at ${LIBTORCH_ROOTDIR}")
else()
    file(MAKE_DIRECTORY ${LIBTORCH_ROOTDIR}/)
    message(STATUS "Libtorch library not found - downloading pre-built library.")

    if(WIN32)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "libtorch-win-shared-with-deps-debug-${LIBTORCH_VERSION}%2Bcpu")
        else()
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu")
        endif()
        set(LIB_LIBTORCH_PRE_BUILD_LIB_TYPE "zip")
    endif()

    if(UNIX AND NOT APPLE)
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "libtorch-${LIBTORCH_VERSION}-Linux-aarch64")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_TYPE "zip")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "libtorch-${LIBTORCH_VERSION}-Linux-x86_64")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_TYPE "zip")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7a")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "pytorch-v${LIBTORCH_VERSION}")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_TYPE "tar.gz")
        endif()
    endif()

    if(UNIX AND APPLE)
        if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "libtorch-${LIBTORCH_VERSION}-macOS-x86_64")
        elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(LIB_LIBTORCH_PRE_BUILD_LIB_NAME "libtorch-${LIBTORCH_VERSION}-macOS-arm64")
        endif()
        set(LIB_LIBTORCH_PRE_BUILD_LIB_TYPE "zip")
    endif()

    if (NOT DEFINED LIBTORCH_URL)
        if(WIN32)
            set(LIBTORCH_URL https://download.pytorch.org/libtorch/cpu/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}.${LIB_LIBTORCH_PRE_BUILD_LIB_TYPE})
        elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7a")
            set(LIBTORCH_URL https://github.com/pelinski/bela-torch/releases/download/v${LIBTORCH_VERSION}/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}.${LIB_LIBTORCH_PRE_BUILD_LIB_TYPE})
        else()
            set(LIBTORCH_URL https://github.com/faressc/libtorch-cpp-lib/releases/download/v${LIBTORCH_VERSION}/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}.${LIB_LIBTORCH_PRE_BUILD_LIB_TYPE})
        endif()
    endif()

    message(STATUS "Downloading: ${LIBTORCH_URL}")

    set(LIBTORCH_PATH ${CMAKE_BINARY_DIR}/import/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}.${LIB_LIBTORCH_PRE_BUILD_LIB_TYPE})

    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_PATH} STATUS LIBTORCH_DOWNLOAD_STATUS SHOW_PROGRESS)
    list(GET LIBTORCH_DOWNLOAD_STATUS 0 LIBTORCH_DOWNLOAD_STATUS_NO)

    if(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7a")
        execute_process(
            COMMAND tar -xzf ${LIBTORCH_PATH} -C ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/
        )
    else()
        file(ARCHIVE_EXTRACT
            INPUT ${CMAKE_BINARY_DIR}/import/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}.${LIB_LIBTORCH_PRE_BUILD_LIB_TYPE}
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/libtorch)
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/libtorch/ DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/)
        file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/libtorch/)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME})
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}/ DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/)
        file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/${LIB_LIBTORCH_PRE_BUILD_LIB_NAME}/)
    endif()

    if(LIBTORCH_DOWNLOAD_STATUS_NO)
        message(STATUS "Pre-built library not downloaded. Error occurred, try again and check cmake/SetupLibTorch.cmake")
        file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE})
        file(REMOVE ${LIBTORCH_PATH})
    else()
        message(STATUS "Linking downloaded LibTorch pre-built library.")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOTDIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Suppress warnings by setting -w flag as a linker option
target_link_options(torch INTERFACE "-w")
target_link_options(torch_library INTERFACE "-w")

if (MSVC)
    if (EXISTS "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64")
        list(APPEND BACKEND_BUILD_LIBRARY_DIRS "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64")
        message(STATUS "Intel MKL library found at C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64")
    endif()
endif()

set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules/libtorch-${LIBTORCH_VERSION}${TORCH_BUILD_TYPE}/")

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(ANIRA_LIBTORCH_SHARED_LIB_PATH "${ANIRA_LIBTORCH_SHARED_LIB_PATH}" PARENT_SCOPE)
endif()

# Normally the following lines are not needed, because the Torch package already includes the necessary directories
# However, since we link the Torch package as a PRIVATE target to the anira library, and aniras public headers include torch headers, we need to add the Torch include directories as public headers to the anira library via our BACKEND_BUILD_HEADER_DIRS (see main CMakeLists.txt)
list(APPEND BACKEND_BUILD_HEADER_DIRS "${LIBTORCH_ROOTDIR}/include")
list(APPEND BACKEND_BUILD_HEADER_DIRS "${LIBTORCH_ROOTDIR}/include/torch/csrc/api/include")
