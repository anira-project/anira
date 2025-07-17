#include <thread>
#include <stdint.h>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "gtest/gtest.h"
#include <anira/anira.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/helperFunctions.h>

using namespace anira;

struct SessionElementTestParams {
    HostAudioConfig host_config;
    InferenceConfig inference_config;
    size_t expected_num_structs;
};

std::ostream& operator<<(std::ostream& stream, const SessionElementTestParams& params)
{
    stream << "{ ";
    stream << "Host Config: { ";
    stream << "max_host_buffer_size = " << params.host_config.m_max_buffer_size;
    stream << ", host_sample_rate = " << params.host_config.m_sample_rate;
    stream << ", tensor_index = " << params.host_config.m_tensor_index;
    stream << " }, Inference Config: { ";
    stream << "max_inference_time = " << params.inference_config.m_max_inference_time << " ms";
    stream << " }, Expected num_structs = " << params.expected_num_structs;
    stream << " }";

    return stream;
}

// Test fixture for parameterized SessionElement tests
class SessionElementTest: public ::testing::TestWithParam<SessionElementTestParams>{
};

TEST_P(SessionElementTest, CalculateNumStructs){
    auto test_params = GetParam();
    
    PrePostProcessor pp_processor(test_params.inference_config);
    
    SessionElement session_element(
        0, // session_id
        pp_processor,
        test_params.inference_config
    );

    size_t actual_num_structs = session_element.calculate_num_structs(test_params.host_config);

    ASSERT_EQ(actual_num_structs, test_params.expected_num_structs) 
        << "Number of structs mismatch. Expected: " << test_params.expected_num_structs 
        << ", Got: " << actual_num_structs;
}

std::string build_test_name(const testing::TestParamInfo<SessionElementTest::ParamType>& info){
    std::stringstream ss_sample_rate, ss_buffer_size, ss_max_inference_time, ss_tensor_index;
    std::vector<std::stringstream> ss_tensor_input_size, ss_tensor_output_size;

    // Set precision to 4 decimal places for cleaner names
    ss_sample_rate << std::fixed << std::setprecision(4) << info.param.host_config.m_sample_rate;
    ss_buffer_size << std::fixed << std::setprecision(4) << info.param.host_config.m_max_buffer_size;
    ss_max_inference_time << std::fixed << std::setprecision(2) << info.param.inference_config.m_max_inference_time;
    ss_tensor_index << info.param.host_config.m_tensor_index;

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
    std::string tensor_index_str = ss_tensor_index.str();
    std::string tensor_shape_str = ss.str();

    // Replace decimal points with underscores to make valid test names
    std::replace(sample_rate_str.begin(), sample_rate_str.end(), '.', '_');
    std::replace(buffer_size_str.begin(), buffer_size_str.end(), '.', '_');
    std::replace(max_inference_time_str.begin(), max_inference_time_str.end(), '.', '_');

    return "host_config_" + sample_rate_str + "x" + buffer_size_str + "_tidx_" + tensor_index_str + tensor_shape_str + "_max_time_" + max_inference_time_str;
}

INSTANTIATE_TEST_SUITE_P(
    CalculateNumStructs, SessionElementTest, ::testing::Values(
        // Basic test cases similar to InferenceManager tests
        SessionElementTestParams {
            HostAudioConfig(2048, 48000),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                40.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(1, 48000.0/2048),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                40.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(1, 48000.0/2048),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1}}, {{1, 1, 2048}})},
                50.f
            ),
            3
        },
        SessionElementTestParams {
            HostAudioConfig(256, 48000.0),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 4, 1}})},
                ProcessingSpec({1}, {4}),
                40.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(1./256., 48000./2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 4, 1}}, {{1, 1, 2048}})},
                ProcessingSpec({4}, {1}),
                40.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(1., 48000./2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}}, {{1, 1, 2048}, {2, 256}})},
                ProcessingSpec({16}, {1, 2}),
                40.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(256., 48000./8, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                ProcessingSpec({16, 2}, {1, 3}),
                5.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(600., 48000./8, 1),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 16, 1}, {2, 256}}, {{1, 1, 2048}, {3, 128}})},
                ProcessingSpec({16, 2}, {1, 3}),
                50.f
            ),
            9
        },
        // Non-power-of-two buffer size tests
        SessionElementTestParams {
            HostAudioConfig(100, 48000),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 2048}}, {{1, 1, 2048}})},
                13.f
            ),
            2
        },
        SessionElementTestParams {
            HostAudioConfig(300, 44100),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1024}}, {{1, 1, 1024}})},
                40.f
            ),
            3
        },
        SessionElementTestParams {
            HostAudioConfig(2.5, 48000./2048.),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 8, 1}}, {{1, 1, 1024}})},
                ProcessingSpec({8}, {1}),
                12.f
            ),
            6
        },
        // Edge cases with very small buffer sizes
        SessionElementTestParams {
            HostAudioConfig(1, 44100),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
                30.f
            ),
            4
        },
        // Test with large buffer sizes
        SessionElementTestParams {
            HostAudioConfig(4096, 96000),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 1024}}, {{1, 1, 1024}})},
                20.f
            ),
            12
        },
        // Test with very short inference times
        SessionElementTestParams {
            HostAudioConfig(512, 48000),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 256}}, {{1, 1, 256}})},
                1.f
            ),
            4
        },
        // Test with very long inference times
        SessionElementTestParams {
            HostAudioConfig(512, 48000),
            InferenceConfig(
                std::vector<ModelData>{ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
                std::vector<TensorShape>{TensorShape({{1, 1, 256}}, {{1, 1, 256}})},
                100.f
            ),
            40
        }
    ),
    build_test_name
);
