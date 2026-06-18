#include <anira/anira.h>

// Exercises the installed package: the public header must resolve, anira::anira
// must link, and at runtime libanira (plus its backend libraries) must load. We
// call a member defined in libanira (not header-inline) so the link is real and
// cannot be dropped by --as-needed.
int main() {
    anira::InferenceConfig config;
    const auto& input_sizes = config.get_tensor_input_size();
    return input_sizes.empty() ? 0 : 1;
}
