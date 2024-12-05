add_compile_definitions(
    BELA_MODELS_PATH="/root/models"
)

if(ANIRA_WITH_BENCHMARK)
    add_subdirectory(bela-benchmark)
endif()

add_subdirectory(bela-inference)