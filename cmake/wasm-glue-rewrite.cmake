# The post-link rewrites of AniraWeb.js for Vite (decision D2 of PR 11), two of them.
#
# 1. The pthread pool's worker URL, for Vite's dev server. The dev server transforms every module
#    it serves, the module's glue included, and a glue it transforms imports /@vite/client, which
#    the AudioWorkletGlobalScope cannot load: the context's worklet thread then fails to start. One
#    construct of the -pthread glue makes it do so, the pthread pool's literal
#    `new Worker(new URL("AniraWeb.js", import.meta.url), ...)`, which Vite rewrites into a worker
#    import with its client. anira starts no pthread, but the pool code is linked: its worker URL
#    goes through locateFile, as the Wasm Workers' does already.
# 2. The Wasm Workers' import of the context's worker module, for Vite's import analysis. The
#    other trigger of the client, a dynamic import() with a variable specifier, is written as a
#    template literal in src/platform/emscripten/em_worker.cpp, `import(`${url}`)`, which the dev
#    server leaves alone but cannot analyse, so it warned ("The above dynamic import cannot be
#    analyzed by Vite"). The comment that silences it, `/* @vite-ignore */` inside the call,
#    cannot be written there: an EM_JS body is a macro argument, whose comments the preprocessor
#    strips. It is inserted here: `import(/* @vite-ignore */ `${url}`)`. Vite's specifier is the
#    template literal still, so the dev server leaves the import as it was.
#
# Each construct is matched as a regex over any whitespace (and the URL over either quote): the
# minified glue of a Release build writes `new URL("AniraWeb.js",import.meta.url)`, the -O0 glue of
# a Debug build `new URL('AniraWeb.js', import.meta.url)`, and the options object after it differs
# between the two (ASSERTIONS appends the worker's ID to its name), so only the URL is matched and
# replaced. A glue rewritten already is left as it is; a glue with neither the construct nor its
# rewrite fails the build, so an emsdk (or an em_worker.cpp) that changes the glue is noticed here.
#
# usage: cmake -DGLUE=<AniraWeb.js> -P cmake/wasm-glue-rewrite.cmake
if(NOT GLUE)
  message(FATAL_ERROR "wasm-glue-rewrite: GLUE (the path of AniraWeb.js) is required")
endif()
file(READ "${GLUE}" _glue)
set(_changed FALSE)
set(_ws "[ \t\r\n]*")

# 1. The pthread pool's worker URL.
set(_pthread_worker
    "new Worker\\(${_ws}new URL\\([\"']AniraWeb\\.js[\"'],${_ws}import\\.meta\\.url\\)")
# A glue rewritten already: the pthread pool's worker (named em-pthread...) comes from locateFile.
# The Wasm Workers' (named em-ww) does from the start, so the name tells the two apart.
set(_rewritten
    "new Worker\\(${_ws}locateFile\\([\"']AniraWeb\\.js[\"']\\)${_ws},${_ws}{[^}]*em-pthread")
string(REGEX MATCH "${_pthread_worker}" _found "${_glue}")
if(_found)
  string(REGEX REPLACE "${_pthread_worker}" "new Worker(locateFile(\"AniraWeb.js\")" _glue
                       "${_glue}")
  string(REGEX MATCH "new URL\\([\"']AniraWeb\\.js[\"']" _left "${_glue}")
  if(_left)
    message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} still names AniraWeb.js by a literal URL")
  endif()
  set(_changed TRUE)
else()
  string(REGEX MATCH "${_rewritten}" _done "${_glue}")
  if(NOT _done)
    message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} has no pthread worker URL to rewrite; "
                        "the emsdk changed the glue (see cmake/wasm-glue-rewrite.cmake)")
  endif()
endif()

# 2. The Wasm Workers' import of the worker module. Bracket arguments, since `${url}` in a quoted
# argument is a CMake variable reference; `\$` keeps the regex's `$` a character.
set(_url_literal [=[`\${url}`]=])
set(_worker_import "import\\(${_ws}${_url_literal}")
set(_ignored_import "import\\(${_ws}/\\*${_ws}@vite-ignore${_ws}\\*/${_ws}${_url_literal}")
string(REGEX MATCH "${_worker_import}" _found "${_glue}")
if(_found)
  string(REGEX REPLACE "${_worker_import}" [=[import(/* @vite-ignore */ `${url}`]=] _glue
                       "${_glue}")
  set(_changed TRUE)
else()
  string(REGEX MATCH "${_ignored_import}" _done "${_glue}")
  if(NOT _done)
    message(FATAL_ERROR "wasm-glue-rewrite: ${GLUE} has no import of the worker module to mark "
                        "@vite-ignore; the emsdk or src/platform/emscripten/em_worker.cpp changed "
                        "the glue (see cmake/wasm-glue-rewrite.cmake)")
  endif()
endif()

if(_changed)
  file(WRITE "${GLUE}" "${_glue}")
endif()
