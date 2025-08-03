#code adopted from https://github.com/jatinchowdhury18/RTNeural/blob/main/cmake/Sanitizers.cmake
function(anira_rtsan_configure target)
    target_compile_definitions(${target} PUBLIC ANIRA_WITH_RTSAN)
    target_compile_options(${target} PUBLIC -fsanitize=realtime)
    target_link_options(${target} PUBLIC -fsanitize=realtime)
endfunction()