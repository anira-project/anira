if(NOT WASM)
  message(FATAL_ERROR "Cannot build WASM target without Emscripten toolchain")
endif()

# ==============================================================================
# AniraWeb WASM target
# ==============================================================================

set(ANIRA_WASM_TARGET_NAME "AniraWeb")
set(ANIRA_WASM_OUTPUT_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/web/wasm")
message(STATUS "Building AniraWeb WASM module...")

set(ANIRA_WEB_LICENSES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/web/licenses")

# Set flags if Debug
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ANIRA_WASM_DEBUG_FLAGS "-O0 -gsource-map")
else()
  set(ANIRA_WASM_DEBUG_FLAGS "")
endif()

set(ANIRA_WASM_WRAPPER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/InferenceThread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/JSPrePostProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/backends/BackendBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/backends/JSProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/utils/Buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/utils/HostConfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/utils/RingBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/utils/Vectors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/InferenceConfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/InferenceHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emscripten-wrappers/PrePostProcessor.cpp
)

# Static library of WASM wrappers — linkable by AniraWeb and external WASM targets
add_library(anira_wasm_wrappers STATIC ${ANIRA_WASM_WRAPPER_SOURCES})
target_link_libraries(anira_wasm_wrappers PUBLIC anira::anira)
target_compile_features(anira_wasm_wrappers PUBLIC cxx_std_20)
add_library(anira::wasm_wrappers ALIAS anira_wasm_wrappers)

set(ANIRA_WASM_LINK_FLAGS "\
  --no-entry \
  --emit-tsd=${ANIRA_WASM_OUTPUT_FOLDER}/${ANIRA_WASM_TARGET_NAME}.d.ts \
  -s STACK_OVERFLOW_CHECK=0 \
  -s IMPORTED_MEMORY=1 \
  -s INITIAL_MEMORY=536870912 \
  -s SHARED_MEMORY=1 \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s MALLOC=emmalloc \
  -s EXPORT_ES6=1 \
  -s MODULARIZE=1 \
  -s ENVIRONMENT=worklet,web \
  -s ASSERTIONS=1 \
  -s NO_DISABLE_EXCEPTION_CATCHING \
  -s STACK_SIZE=33554432 \
  -s EXPORTED_FUNCTIONS='[\"_free\",\"_malloc\"]' \
  -s EXPORT_KEEPALIVE=1 \
  -s EXPORTED_RUNTIME_METHODS='[\"UTF8ToString\",\"HEAPU32\",\"HEAPF32\",\"stackSave\",\"stackRestore\"]' \
  ")

# CMake requires at least one source file for add_executable.
# This is a --no-entry Emscripten module so there is no main();
# all symbols come from the linked anira_wasm_wrappers static lib.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/aniraweb_stub.cpp" "")
add_executable(${ANIRA_WASM_TARGET_NAME} "${CMAKE_CURRENT_BINARY_DIR}/aniraweb_stub.cpp")

target_link_libraries(${ANIRA_WASM_TARGET_NAME} PUBLIC
    -Wl,--whole-archive anira::wasm_wrappers -Wl,--no-whole-archive)
target_compile_features(${ANIRA_WASM_TARGET_NAME} PUBLIC cxx_std_20)

set_target_properties(${ANIRA_WASM_TARGET_NAME} PROPERTIES
  OUTPUT_NAME ${ANIRA_WASM_TARGET_NAME}
  LINK_FLAGS "${ANIRA_WASM_DEBUG_FLAGS} ${ANIRA_WASM_LINK_FLAGS}"
  RUNTIME_OUTPUT_DIRECTORY ${ANIRA_WASM_OUTPUT_FOLDER}
)

# ==============================================================================
# Bundle license files from native deps that get statically linked into the
# WASM binary, so the @anira-project/anira npm package can ship them and downstream
# consumers can satisfy their attribution obligations. Tied to the AniraWeb
# build target so it runs as part of the actual wasm build (not at configure
# time, where it would re-fire on every cmake configure regardless of whether
# wasm is being built).
# ==============================================================================
if(ANIRA_WITH_ONNXRUNTIME)
    set(_ort_license_dest "${ANIRA_WEB_LICENSES_DIR}/onnxruntime")
    set(_ort_license_inputs "")
    foreach(_f LICENSE ThirdPartyNotices.txt)
        if(EXISTS "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/${_f}")
            list(APPEND _ort_license_inputs "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/${_f}")
        else()
            message(WARNING "ONNX Runtime ${_f} not found at "
                "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH} — Anira Web will ship "
                "without it; downstream consumers won't see attribution.")
        endif()
    endforeach()

    add_custom_command(TARGET ${ANIRA_WASM_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_ort_license_dest}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_ort_license_dest}"
        COMMAND ${CMAKE_COMMAND} -E copy ${_ort_license_inputs} "${_ort_license_dest}/"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-PACKAGE.txt"
            "${_ort_license_dest}/PACKAGE.txt"
        VERBATIM
        COMMENT "Bundling ONNX Runtime ${LIBONNXRUNTIME_VERSION} license + ThirdPartyNotices into Anira Web"
    )

    # PACKAGE.txt only depends on configure-time vars (version, etc.), so
    # generating it at configure time is safe — it'll be regenerated whenever
    # the wasm preset is reconfigured. The actual copy into web/licenses/
    # happens at build time, alongside the heavy LICENSE/ThirdPartyNotices.
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-PACKAGE.txt"
        "name: onnxruntime\nversion: ${LIBONNXRUNTIME_VERSION}\nlicense: MIT\nhomepage: https://github.com/microsoft/onnxruntime\n")
endif()
