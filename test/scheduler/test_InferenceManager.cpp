#include <thread>
#include <stdint.h>
#include <chrono>

#include "gtest/gtest.h"
#include <anira/anira.h>
#include <anira/utils/helperFunctions.h>

using namespace anira;

struct InferenceManagerTestParams {
    HostAudioConfig host_config;
    InferenceConfig inference_config;
    std::vector<unsigned int> expected_latency;
};

std::ostream& operator<<(std::ostream& stream, const InferenceManagerTestParams& params)
{
   
    stream << "{ ";
    stream << "Host Config: { ";
    stream << "host_buffer_size = " << params.host_config.m_buffer_size;
    stream << ", host_sample_rate = " << params.host_config.m_sample_rate;
    stream << " }, Inference Config: { ";
    stream << "max_inference_time = " << params.inference_config.m_max_inference_time << " ms";
    stream << " }";

    return stream;
}

// // Test fixture for paramterized inference tests
class InferenceManagerTest: public ::testing::TestWithParam<InferenceManagerTestParams>{
};

TEST_P(InferenceManagerTest, Simple){

    auto test_params = GetParam();
    auto buffer_size = test_params.host_config.m_buffer_size;
    auto sample_rate = test_params.host_config.m_sample_rate;

    PrePostProcessor pp_processor(test_params.inference_config);
    BackendBase* custom_processor = nullptr; // Use default processor
    ContextConfig context_config;

    InferenceManager inference_manager(
        pp_processor,
        test_params.inference_config,
        custom_processor,
        context_config
    );

    inference_manager.prepare(test_params.host_config);

    std::vector<unsigned int> latency = inference_manager.get_latency();

    for (size_t i = 0; i < test_params.expected_latency.size(); ++i) {
        ASSERT_EQ(latency[i], test_params.expected_latency[i]) << "Latency mismatch for tensor " << i;
    }
}

std::string build_test_name(const testing::TestParamInfo<InferenceManagerTest::ParamType>& info){
    std::stringstream ss_sample_rate, ss_buffer_size, ss_max_inference_time;
    std::vector<std::stringstream> ss_tensor_input_size, ss_tensor_output_size;

    // Set precision to 4 decimal places for cleaner names
    ss_sample_rate << std::fixed << std::setprecision(4) << info.param.host_config.m_sample_rate;
    ss_buffer_size << std::fixed << std::setprecision(4) << info.param.host_config.m_buffer_size;
    ss_max_inference_time << std::fixed << std::setprecision(2) << info.param.inference_config.m_max_inference_time;

    std::stringstream ss;
    ss << "__input_size_";
    for (const auto& size : info.param.inference_config.get_tensor_input_size()) {
        ss_tensor_input_size.push_back(std::stringstream());
        ss_tensor_input_size.back() << size;
        ss << ss_tensor_input_size.back().str() << "_";
    }

    ss << "_output_size_";
    for (const auto& size : info.param.inference_config.get_tensor_output_size()) {
        ss_tensor_output_size.push_back(std::stringstream());
        ss_tensor_output_size.back() << size;
        ss << ss_tensor_output_size.back().str() << "_";
    }

    std::string sample_rate_str = ss_sample_rate.str();
    std::string buffer_size_str = ss_buffer_size.str();
    std::string max_inference_time_str = ss_max_inference_time.str();
    std::string tensor_shape_str = ss.str();

    // Replace decimal points with underscores to make valid test names
    std::replace(sample_rate_str.begin(), sample_rate_str.end(), '.', '_');
    std::replace(buffer_size_str.begin(), buffer_size_str.end(), '.', '_');
    std::replace(max_inference_time_str.begin(), max_inference_time_str.end(), '.', '_');

    return "host_config_" + buffer_size_str + "x" + sample_rate_str + tensor_shape_str + "_max_time_" + max_inference_time_str;
}

INSTANTIATE_TEST_SUITE_P(
    CalculateLatency, InferenceManagerTest, ::testing::Values(
        InferenceManagerTestParams {
            HostAudioConfig(2048, 48000, true),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                1.f
            ),
            { 4095 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1, 48000.0/2048, true),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                40.f
            ),
            { 5886 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1, 48000.0/2048),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                50.f
            ),
            { 4096 },
        },
        InferenceManagerTestParams {
            HostAudioConfig(256, 48000.0, true),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 4, 1}})},
                ProcessingSpec({1}, {4}),
                20.f
            ),
            { 2 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1./256., 48000./2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 4, 1}}, {{1, 1, 2048}})},
                ProcessingSpec({4}, {1}),
                40.f
            ),
            { 3960 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1., 48000./2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}}, {{1, 1, 2048}})},
                ProcessingSpec({16}, {1}),
                40.f
            ),
            { 2048 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1., 48000./2048., true),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}}, {{1, 1, 2048}, {2, 256}})},
                ProcessingSpec({16}, {1, 2}),
                40.f
            ),
            { 6144, 768 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(256., 48000./8, false, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                ProcessingSpec({16, 2}, {1, 3}),
                40.f
            ),
            { 2048, 128 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(128., 48000./8, true, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                ProcessingSpec({16, 2}, {1, 3}),
                40.f
            ),
            { 6144, 384 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(128., 48000./8, true, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}, {2, 256}}, {{1, 2, 2048}})},
                ProcessingSpec({16, 2}, {2}),
                40.f
            ),
            { 4924 }
        },
        // Non-power-of-two buffer size tests
        InferenceManagerTestParams {
            HostAudioConfig(100, 48000),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                13.f
            ),
            { 2744 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(300, 44100),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1024}}, {{1, 1, 1024}})},
                40.f
            ),
            { 2820 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(2.5, 48000./2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 8, 1}}, {{1, 1, 1024}})},
                ProcessingSpec({8}, {1}),
                12.f
            ),
            { 3072 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(2.5, 48000./2048., true),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 8, 1}}, {{1, 1, 1024}})},
                ProcessingSpec({8}, {1}),
                13.f
            ),
            { 3583 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1500, 44100./8., false, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 4, 1}, {2, 128}}, {{1, 1, 512}, {3, 64}})},
                ProcessingSpec({4, 2}, {1, 1}),
                50.f
            ),
            { 18944, 7104 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(1500, 44100./8., true, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 4, 1}, {2, 128}, {1, 2}}, {{1, 1, 512}, {3, 64}})},
                ProcessingSpec({4, 2, 1}, {1, 1}),
                50.f
            ),
            { 18944, 7104 }
        },
        InferenceManagerTestParams {
            HostAudioConfig(256., 48000./8, false, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 4, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                ProcessingSpec({1, 2}, {1, 1}, {0, 256}, {2048, 0}),
                40.f
            ),
            { 2048, 0 }
        }
    ),
    build_test_name
);
