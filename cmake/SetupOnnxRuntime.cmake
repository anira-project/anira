set(LIBONNXRUNTIME_VERSION 1.15.0)

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
set(ONNXRUNTIME_ROOTDIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-${LIBONNXRUNTIME_VERSION})

if(EXISTS ${ONNXRUNTIME_ROOTDIR}/)
    message(STATUS "ONNX-Runtime library found at ${ONNXRUNTIME_ROOTDIR}")
else()
    file(MAKE_DIRECTORY ${ONNXRUNTIME_ROOTDIR}/)
    message(STATUS "ONNX-Runtime library not found - downloading pre-built library.")

    if(WIN32)
        set(LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME "onnxruntime-win-x64-${LIBONNXRUNTIME_VERSION}")
        set(LIB_ONNXRUNTIME_PRE_BUILD_LIB_TYPE "zip")
    endif()

    if(UNIX AND NOT APPLE)
        set(LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME "onnxruntime-linux-x64-${LIBONNXRUNTIME_VERSION}")
        set(LIB_ONNXRUNTIME_PRE_BUILD_LIB_TYPE "tgz")
    endif()

    if(UNIX AND APPLE)
        set(LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME "onnxruntime-osx-universal2-${LIBONNXRUNTIME_VERSION}")
        set(LIB_ONNXRUNTIME_PRE_BUILD_LIB_TYPE "tgz")
    endif()

    set(LIBONNXRUNTIME_URL https://github.com/microsoft/onnxruntime/releases/download/v${LIBONNXRUNTIME_VERSION}/${LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME}.${LIB_ONNXRUNTIME_PRE_BUILD_LIB_TYPE})
    set(LIBONNXRUNTIME_PATH ${CMAKE_BINARY_DIR}/import/${LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME}.${LIB_ONNXRUNTIME_PRE_BUILD_LIB_TYPE})

    file(DOWNLOAD ${LIBONNXRUNTIME_URL} ${LIBONNXRUNTIME_PATH} STATUS LIBONNXRUNTIME_DOWNLOAD_STATUS SHOW_PROGRESS)
    list(GET LIBONNXRUNTIME_DOWNLOAD_STATUS 0 LIBONNXRUNTIME_DOWNLOAD_STATUS_NO)

    file(ARCHIVE_EXTRACT
            INPUT ${CMAKE_BINARY_DIR}/import/${LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME}.${LIB_ONNXRUNTIME_PRE_BUILD_LIB_TYPE}
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-${LIBONNXRUNTIME_VERSION}/)

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-${LIBONNXRUNTIME_VERSION}/${LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME}/ DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-${LIBONNXRUNTIME_VERSION}/)

    file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-${LIBONNXRUNTIME_VERSION}/${LIB_ONNXRUNTIME_PRE_BUILD_LIB_NAME})

    if(LIBONNXRUNTIME_DOWNLOAD_STATUS_NO)
        message(STATUS "Pre-built library not downloaded. Error occurred, try again and check cmake/SetupOnnxRuntime.cmake")
        file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-${LIBONNXRUNTIME_VERSION})
        file(REMOVE ${LIBONNXRUNTIME_PATH})
    else()
        message(STATUS "Linking downloaded ONNX-Runtime pre-built library.")
    endif()
endif()

include_directories(SYSTEM
        "${ONNXRUNTIME_ROOTDIR}/include"
        "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
)

link_directories(
        "${ONNXRUNTIME_ROOTDIR}/lib"
)