#ifndef JSONCONFIGLOADER_H
#define JSONCONFIGLOADER_H

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace anira {

class ANIRA_API JsonConfigLoader {
public:
    JsonConfigLoader(const std::string& file_path);
    JsonConfigLoader(std::istream& stream);

    std::unique_ptr<anira::ContextConfig> get_context_config();
    std::unique_ptr<anira::InferenceConfig> get_inference_config();

private:
    struct SingleParameterStruct {
        bool m_max_inference_time_set = false;
        float m_max_inference_time = 0.f;
        unsigned int m_warm_up = anira::InferenceConfig::Defaults::k_warm_up;
        bool m_session_exclusive_processor =
            anira::InferenceConfig::Defaults::k_session_exclusive_processor;
        float m_blocking_ratio = anira::InferenceConfig::Defaults::k_blocking_ratio;
        unsigned int m_num_parallel_processors =
            anira::InferenceConfig::Defaults::m_num_parallel_processors;
    };

    void initialize_from_stream(std::istream& stream);

    void parse(const nlohmann::json& config);
    void parse_context_config(const nlohmann::json& config);
    void parse_inference_config(const nlohmann::json& config);

    static std::vector<anira::ModelData> create_model_data_from_config(
        const nlohmann::basic_json<>& config);
    static std::vector<anira::TensorShape> create_tensor_shape_from_config(
        const nlohmann::basic_json<>& config);
    static anira::TensorShapeList parse_tensor_json_shape(const nlohmann::json& shape_node);
    static anira::ProcessingSpec create_processing_spec_from_config(
        const nlohmann::basic_json<>& config,
        bool& config_required);
    static std::vector<size_t> parse_size_t_json_shape(const nlohmann::json& shape_node,
                                                       const std::string& json_key_name);
    static SingleParameterStruct create_single_parameters_from_config(
        const nlohmann::basic_json<>& config,
        bool& necessary_parameter_set);

    std::unique_ptr<anira::ContextConfig> m_context_config;
    std::unique_ptr<anira::InferenceConfig> m_inference_config;
};
}  // namespace anira

#endif  // JSONCONFIGLOADER_H
