# The root of the model tree at run time (extras/CMakeLists.txt): the examples load the
# configuration files extras/models/model_files.h names, and spell any other path relative to it.
add_compile_definitions(ANIRA_EXTRAS_MODELS_DIR="${ANIRA_EXTRAS_MODELS_DIR}")


if(ANIRA_WITH_BENCHMARK)
    add_subdirectory(benchmark)
endif()

add_subdirectory(minimal-inference)
add_subdirectory(juce-audio-plugin)
add_subdirectory(clap-audio-plugin)