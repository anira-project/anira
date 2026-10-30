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

# Set flags if Debug. ASSERTIONS=1 only there: Emscripten's Web Audio library asserts that a
# context is a window.AudioContext (libwebaudio.js), which refuses the OfflineAudioContext the
# browser test renders through, so the shipped (Release) module is built without assertions.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ANIRA_WASM_DEBUG_FLAGS "-O0 -gsource-map -s ASSERTIONS=1")
else()
  set(ANIRA_WASM_DEBUG_FLAGS "-s ASSERTIONS=0")
endif()

# The C ABI's export list is the generated web/src/abi/exports_wasm.txt (one _-prefixed
# name per line: _malloc, _free and every promised, draft and Emscripten entry point of
# abi/anira.yml; tools/abi/gen.py --check keeps it in step with the registry, and build_web
# runs that check before this build). The list is all the module exports: the linker pulls
# every archive member that defines one of its names out of libanira.a, so no
# --whole-archive is needed, and a registry entry without a definition on Emscripten fails
# this link: the presence gate of the wasm leg. build_web compares the emitted .d.ts with
# the symbol lists in both directions.
#
# Threads: -pthread with -sWASM_WORKERS and -sAUDIO_WORKLET (Emscripten's hybrid mode). The
# module's threads are Wasm Workers and the Wasm Audio Worklet, created from C on the one
# imported shared memory; -pthread gives them the threaded libc and libc++, whose mutexes are
# real futex mutexes across them (the no-pthread libraries' are stubs that lock nothing, and
# since emsdk 5 do not link). No code of the module may start a pthread: one started from the
# browser's main thread runs only once that thread yields, so a join there never returns.
# INCOMING_MODULE_JS_API names every Module property the package passes at instantiation
# (emsdk 6 reads neither wasmMemory nor wasmBinary by default).
# Memory: one imported shared memory that grows (Anira.create makes it: 256 MiB to begin with,
# the core and a small handler; 2 GiB at most). A shared memory cannot move, so the browser
# reserves its maximum as address space when it is created, and the pages are committed as they
# are touched. MAXIMUM_MEMORY is the heap's ceiling for malloc (getHeapMax): a page's smaller
# maximum fails the growth earlier, and a larger one is capped here. 2 GiB keeps every heap
# address below 2^31, where the package's `>>> 2` indexing and the glue need no unsigned pointers.
# With -pthread the glue reads the heap through a check per access (growMemViews; Emscripten
# warns pthreads-mem-growth); the package's own code goes through heap() of web/src/abi/bindings.ts.
# heap() reads the realm's Module.wasmMemory, which anira's boot of an inference Worker and of the
# worklet assigns (src/platform/emscripten/em_worker.cpp, em_audio.cpp), so wasmMemory is a
# runtime export: an unexported runtime symbol is a getter-only property of Module under
# ASSERTIONS (a Debug build), and the glue is an ES module, whose strict mode makes that
# assignment throw (no Worker entered its loop). The ABI gates' glue is a script, where it
# fails silently, so only a Debug build of this module shows it.
# A library's runtime never exits: EXIT_RUNTIME=0 here, and the noExitRuntime symbol that makes
# keepRuntimeAlive() honour it comes with the library itself (src/platform/emscripten/
# em_runtime.cpp), for this module and every other that links anira.
set(ANIRA_WASM_LINK_FLAGS "\
  --no-entry \
  -pthread \
  -s WASM_WORKERS=1 \
  -s AUDIO_WORKLET=1 \
  -s INCOMING_MODULE_JS_API=wasmMemory,wasmBinary,locateFile,instantiateWasm,print,printErr \
  -s EXIT_RUNTIME=0 \
  --emit-tsd=${ANIRA_WASM_OUTPUT_FOLDER}/${ANIRA_WASM_TARGET_NAME}.d.ts \
  -s STACK_OVERFLOW_CHECK=0 \
  -s IMPORTED_MEMORY=1 \
  -s INITIAL_MEMORY=268435456 \
  -s SHARED_MEMORY=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MAXIMUM_MEMORY=2147483648 \
  -s MALLOC=emmalloc \
  -s EXPORT_ES6=1 \
  -s MODULARIZE=1 \
  -s ENVIRONMENT=web,worker,worklet \
  -s NO_DISABLE_EXCEPTION_CATCHING \
  -s STACK_SIZE=33554432 \
  -s EXPORTED_FUNCTIONS=@${CMAKE_CURRENT_SOURCE_DIR}/web/src/abi/exports_wasm.txt \
  -s EXPORTED_RUNTIME_METHODS='[\"UTF8ToString\",\"stringToUTF8OnStack\",\"lengthBytesUTF8\",\"HEAPU8\",\"HEAP32\",\"HEAPU32\",\"HEAPF32\",\"HEAPF64\",\"HEAP64\",\"HEAPU64\",\"stackSave\",\"stackRestore\",\"stackAlloc\",\"emscriptenRegisterAudioObject\",\"emscriptenGetAudioObject\",\"wasmMemory\"]' \
  ")

# CMake requires at least one source file for add_executable.
# This is a --no-entry Emscripten module so there is no main();
# every export is an entry point of the C ABI in libanira.a.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/aniraweb_stub.cpp" "")
add_executable(${ANIRA_WASM_TARGET_NAME} "${CMAKE_CURRENT_BINARY_DIR}/aniraweb_stub.cpp")

target_link_libraries(${ANIRA_WASM_TARGET_NAME} PRIVATE anira::anira)
target_compile_features(${ANIRA_WASM_TARGET_NAME} PRIVATE cxx_std_20)

set_target_properties(${ANIRA_WASM_TARGET_NAME} PROPERTIES
  OUTPUT_NAME ${ANIRA_WASM_TARGET_NAME}
  LINK_FLAGS "${ANIRA_WASM_DEBUG_FLAGS} ${ANIRA_WASM_LINK_FLAGS}"
  RUNTIME_OUTPUT_DIRECTORY ${ANIRA_WASM_OUTPUT_FOLDER}
  JOB_POOL_LINK anira_wasm_link
)

# The glue's one construct that has Vite's dev server inject its client (the pthread pool's
# literal worker URL), rewritten to go through locateFile: cmake/wasm-glue-rewrite.cmake.
add_custom_command(TARGET ${ANIRA_WASM_TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        "-DGLUE=${ANIRA_WASM_OUTPUT_FOLDER}/${ANIRA_WASM_TARGET_NAME}.js"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wasm-glue-rewrite.cmake"
    VERBATIM
    COMMENT "Rewriting the pthread worker URL of ${ANIRA_WASM_TARGET_NAME}.js for Vite's dev server")

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

    # The backends WASM archive currently ships only include/ + lib/, so the
    # heavy LICENSE/ThirdPartyNotices may be absent; only copy them if present
    # (an empty `cmake -E copy` would fail the build). PACKAGE.txt is always written.
    # The directory is not cleared first: web/licenses/onnxruntime/ is tracked, and an
    # archive without the two files would otherwise delete them from the checkout and from
    # the package, which still links ONNX Runtime.
    set(_ort_license_copy "")
    if(_ort_license_inputs)
        set(_ort_license_copy COMMAND ${CMAKE_COMMAND} -E copy ${_ort_license_inputs} "${_ort_license_dest}/")
    endif()

    add_custom_command(TARGET ${ANIRA_WASM_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_ort_license_dest}"
        ${_ort_license_copy}
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-PACKAGE.txt"
            "${_ort_license_dest}/PACKAGE.txt"
        VERBATIM
        COMMENT "Bundling ONNX Runtime ${ANIRA_ONNXRUNTIME_VERSION} license + ThirdPartyNotices into Anira Web"
    )

    # PACKAGE.txt only depends on configure-time vars (version, etc.), so
    # generating it at configure time is safe — it'll be regenerated whenever
    # the wasm preset is reconfigured. The actual copy into web/licenses/
    # happens at build time, alongside the heavy LICENSE/ThirdPartyNotices.
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-PACKAGE.txt"
        "name: onnxruntime\nversion: ${ANIRA_ONNXRUNTIME_VERSION}\nlicense: MIT\nhomepage: https://github.com/microsoft/onnxruntime\n")
endif()
