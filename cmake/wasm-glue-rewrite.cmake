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
# usage: cmake -DGLUE=<AniraWeb.js> -P cmake/wasm-glue-rewrite.cmake
if(NOT GLUE)
  message(FATAL_ERROR "wasm-glue-rewrite: GLUE (the path of AniraWeb.js) is required")
endif()
file(READ "${GLUE}" _glue)
set(_pthread_worker [=[new Worker(new URL("AniraWeb.js",import.meta.url),{type:"module",name:"em-pthread"})]=])
set(_rewritten [=[new Worker(locateFile("AniraWeb.js"),{type:"module",name:"em-pthread"})]=])
string(FIND "${_glue}" "${_pthread_worker}" _at)
if(_at EQUAL -1)
  string(FIND "${_glue}" "${_rewritten}" _done)
  if(_done EQUAL -1)
    message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} has no pthread worker URL to rewrite; "
                        "the emsdk changed the glue (see cmake/wasm-glue-rewrite.cmake)")
  endif()
  return()
endif()
string(REPLACE "${_pthread_worker}" "${_rewritten}" _glue "${_glue}")
string(FIND "${_glue}" "new URL(\"AniraWeb.js\",import.meta.url)" _left)
if(NOT _left EQUAL -1)
  message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} still names AniraWeb.js by a literal URL")
endif()
file(WRITE "${GLUE}" "${_glue}")
