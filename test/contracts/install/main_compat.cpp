// A 2.x-shaped consumer of the installed package through anira/compat/v2.hpp: the shim is
// header-only over anira.hpp and the C entries, so the install tree must carry it, anira.hpp
// and the anira/abi headers, and anira::anira must link what they call. The configuration is
// built the 2.x way, upgraded by the library and read back.
#include <anira/compat/v2.hpp>
#include <cstdio>
#include <exception>
#include <sstream>

int main() {
    try {
        const anira::v2::InferenceConfig config(
            {anira::v2::ModelData("placeholder", anira::v2::CUSTOM)},
            {anira::v2::TensorShape({{1, 1, 64}}, {{1, 1, 64}})},
            5.0F);
        std::istringstream document(R"({ "context_config": { "num_threads": 1 } })");
        anira::v2::JsonConfigLoader loader(document);
        const auto context = loader.get_context_config();
        std::printf("hop=%zu threads=%u custom=%d\n",
                    config.get_preprocess_input_size()[0],
                    context->m_num_threads,
                    static_cast<int>(anira::v2::is_available(anira::v2::CUSTOM)));
        return (config.get_preprocess_input_size()[0] == 64 && context->m_num_threads == 1 &&
                config.get_model_data(anira::v2::CUSTOM) != nullptr)
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "consumer_compat: %s\n", error.what());
        return 1;
    }
}
