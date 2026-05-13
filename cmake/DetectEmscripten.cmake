if("${CMAKE_CXX_COMPILER}" MATCHES "em\\+\\+")
  set(WASM true)
  execute_process(
      COMMAND ${CMAKE_C_COMPILER} --version
      OUTPUT_VARIABLE EMSDK_VERSION_OUTPUT
  )
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" EMSDK_VERSION "${EMSDK_VERSION_OUTPUT}")
  message(STATUS "Emscripten compiler detected: ${EMSDK_VERSION}")

  message(STATUS "Using Emscripten compiler: ${CMAKE_CXX_COMPILER}")
  set(CMAKE_EXECUTABLE_SUFFIX ".js")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -matomics -msimd128 -mbulk-memory -sNO_DISABLE_EXCEPTION_CATCHING")
endif()