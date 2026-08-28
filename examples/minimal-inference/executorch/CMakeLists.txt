cmake_minimum_required(VERSION 3.15)
project(minimal-executorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-executorch
  minimal-executorch.cpp
)

# Deliberately NOT linked against anira. anira embeds its own copy of the
# ExecuTorch runtime, and mixing that copy with a second one linked here is
# fragile: the kernel/backend registrations are pre-linked into the archive, so
# two copies in one image register every kernel twice and abort at startup.
# Linking only the merged ExecuTorch archive keeps this example's runtime fully
# self-contained. anira_target_link_static_backend adds the system libraries the
# archive needs on each platform.
anira_target_link_static_backend(minimal-executorch "${ANIRA_EXECUTORCH_STATIC_LIB}")
target_include_directories(minimal-executorch SYSTEM PRIVATE
  "${ANIRA_EXECUTORCH_INCLUDE_DIR}"
  "${ANIRA_EXECUTORCH_INCLUDE_DIR}/executorch/runtime/core/portable_type/c10"
)
target_compile_definitions(minimal-executorch PRIVATE ${ANIRA_EXECUTORCH_COMPILE_DEFINITIONS})
