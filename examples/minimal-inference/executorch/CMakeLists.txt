cmake_minimum_required(VERSION 3.24)
project(minimal-executorch C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(minimal-executorch
  minimal-executorch.cpp
)

# This example drives the ExecuTorch runtime directly and does not use anira, so it
# links the engine target alone. anira::executorch carries the runtime, extension,
# kernel and XNNPACK-delegate archives with their force-load options, the
# whole-archive microkernels, the include directories and the compile definitions
# the ExecuTorch headers need — everything the example used to assemble by hand.
target_link_libraries(minimal-executorch anira::executorch)
