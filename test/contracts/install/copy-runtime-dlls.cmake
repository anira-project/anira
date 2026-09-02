# Copies a target's runtime DLLs next to its binary — run as a POST_BUILD step via
#   cmake -DDLLS=<;-list from $<TARGET_RUNTIME_DLLS:...>> -DDEST=<dir> -P copy-runtime-dlls.cmake
# A no-op when the list is empty (a static anira has no runtime DLLs), which is why
# this is a script rather than a bare `cmake -E copy` (that errors on zero files).
if(DLLS)
    file(COPY ${DLLS} DESTINATION "${DEST}")
endif()
