// Exercises the public headers, the tanh-lib-backed containers and a
// real InferenceHandler round trip so that the installed package must carry
// anira's headers, its exported target, tanh::Core and the enabled backends.
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>

#include <array>
#include <cstdio>
#include <vector>

int main() {
    anira::RingBuffer ring;
    ring.initialize_with_positions(1, 8);
    for (int i = 0; i < 10; ++i) { ring.push_sample(0, static_cast<float>(i)); }

    anira::RingBufferT<int> tokens;
    tokens.initialize_with_positions(1, 4);
    tokens.push_sample(0, 7);

    anira::BufferF buffer(2, 16);
    buffer.set_sample(1, 3, 0.5F);

    // Bypass (CUSTOM backend without a processor) still drives the whole
    // scheduler / pre-post-processing path through the installed library.
    anira::InferenceConfig config({}, {{{{1, 1, 64}}, {{1, 1, 64}}}}, 5.0F);
    anira::PrePostProcessor pp(config);
    anira::InferenceHandler handler(pp, config);
    handler.prepare({64.0F, 48000.0});
    handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
    std::vector<float> data(64, 0.25F);
    std::array<float*, 1> channels = {data.data()};
    handler.process(channels.data(), 64);

    std::printf("oldest=%g tokens=%d sample=%g latency=%d\n",
                static_cast<double>(ring.pop_sample(0)),
                tokens.pop_sample(0),
                static_cast<double>(buffer.get_sample(1, 3)),
                handler.get_latency());
    return (ring.get_available_samples(0) == 7 && buffer.get_num_samples() == 16) ? 0 : 1;
}
