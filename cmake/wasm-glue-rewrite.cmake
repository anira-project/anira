# The post-link rewrite of AniraWeb.js for Vite's dev server (decision D2 of PR 11). The dev
# server transforms every module it serves, the module's glue included, and a glue it transforms
# imports /@vite/client, which the AudioWorkletGlobalScope cannot load: the context's worklet
# thread then fails to start. One construct of the -pthread glue makes it do so, the pthread
# pool's literal `new Worker(new URL("AniraWeb.js", import.meta.url), ...)`, which Vite rewrites
# into a worker import with its client. anira starts no pthread, but the pool code is linked: its
# worker URL goes through locateFile, as the Wasm Workers' does already. (The other trigger, a
# dynamic import() with a variable specifier, is written as a template literal in
# src/platform/emscripten/em_worker.cpp, which Vite leaves alone.) A glue without the construct
# fails the build, so an emsdk that changes it is noticed here.
#
# The construct is matched as a regex over either quote and any whitespace: the minified glue of
# a Release build writes `new URL("AniraWeb.js",import.meta.url)`, the -O0 glue of a Debug build
# `new URL('AniraWeb.js', import.meta.url)`, and the options object after it differs between the
# two (ASSERTIONS appends the worker's ID to its name), so only the URL is matched and replaced.
#
# usage: cmake -DGLUE=<AniraWeb.js> -P cmake/wasm-glue-rewrite.cmake
if(NOT GLUE)
  message(FATAL_ERROR "wasm-glue-rewrite: GLUE (the path of AniraWeb.js) is required")
endif()
file(READ "${GLUE}" _glue)
set(_ws "[ \t\r\n]*")
set(_pthread_worker
    "new Worker\\(${_ws}new URL\\([\"']AniraWeb\\.js[\"'],${_ws}import\\.meta\\.url\\)")
# A glue rewritten already: the pthread pool's worker (named em-pthread...) comes from locateFile.
# The Wasm Workers' (named em-ww) does from the start, so the name tells the two apart.
set(_rewritten
    "new Worker\\(${_ws}locateFile\\([\"']AniraWeb\\.js[\"']\\)${_ws},${_ws}{[^}]*em-pthread")
string(REGEX MATCH "${_pthread_worker}" _found "${_glue}")
if(NOT _found)
  string(REGEX MATCH "${_rewritten}" _done "${_glue}")
  if(NOT _done)
    message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} has no pthread worker URL to rewrite; "
                        "the emsdk changed the glue (see cmake/wasm-glue-rewrite.cmake)")
  endif()
  return()
endif()
string(REGEX REPLACE "${_pthread_worker}" "new Worker(locateFile(\"AniraWeb.js\")" _glue "${_glue}")
string(REGEX MATCH "new URL\\([\"']AniraWeb\\.js[\"']" _left "${_glue}")
if(_left)
  message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} still names AniraWeb.js by a literal URL")
endif()
file(WRITE "${GLUE}" "${_glue}")
