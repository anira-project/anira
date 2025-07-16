#ifndef JSONCONFIGLOADER_H
#define JSONCONFIGLOADER_H

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "anira/utils/Logger.h"
#include "anira/ContextConfig.h"
#include "anira/InferenceConfig.h"

namespace anira {

class JsonConfigLoader {
public:
    JsonConfigLoader(const std::string& file_path) {
        std::ifstream config_file(file_path);
        if (!config_file.is_open()) {
            LOG_ERROR << "Could not open file at " + file_path << std::endl;
        }
        initialize_from_stream(config_file);
    }

    JsonConfigLoader(std::istream& stream) {
        initialize_from_stream(stream);
    }

    std::unique_ptr<anira::ContextConfig> get_context_config() {
        return std::move(m_context_config);
    }

    std::unique_ptr<anira::InferenceConfig> get_inference_config() {
        return std::move(m_inference_config);
    }

private:
    void initialize_from_stream(std::istream& stream) {
        try {
            nlohmann::json json_config;
            stream >> json_config;
            parse(json_config);
        }
        catch (const nlohmann::json::parse_error& e) {
            LOG_ERROR << "JSON parse error: " << e.what() << std::endl;
        }
    }

    void parse(const nlohmann::json& config)
    {
        parse_context_config(config);
        parse_inference_config(config);
    }

    void parse_context_config(const nlohmann::json& config)
    {
        if (config.contains("context_config")) {
            const auto& context_json = config.at("context_config");

            if (!context_json.contains("num_threads")) {
                m_context_config = std::make_unique<anira::ContextConfig>();
                return;
            }

            if (context_json.at("num_threads").is_number_unsigned()) {
                unsigned int num_threads = context_json.at("num_threads").get<unsigned int>();
                m_context_config = std::make_unique<anira::ContextConfig>(num_threads);
                return;
            } else {
                LOG_ERROR << "Invalid 'num_threads' value: expected an unsigned integer." << std::endl;
            }
        }

        m_context_config = std::make_unique<anira::ContextConfig>();
    }

    void parse_inference_config(const nlohmann::json& config)
    {
        if (!config.contains("inference_config")) {
            LOG_ERROR << "Missing 'inference_config' key." << std::endl;
            return;
        }

        const auto& inference_json = config.at("inference_config");

        std::vector<anira::ModelData> model_data;
        std::vector<anira::TensorShape> tensor_shape;
        anira::ProcessingSpec processing_spec;
        SingleParameterStruct single_parameters;

        bool processing_spec_required = false;
        bool max_inference_time_defined = false;

        if (inference_json.contains("model_data"))
        {
            const auto& model_data_json = inference_json.at("model_data");
            model_data = create_model_data_from_config(model_data_json);
        }

        if (inference_json.contains("tensor_shape"))
        {
            const auto& tensor_shape_json = inference_json.at("tensor_shape");
            tensor_shape = create_tensor_shape_from_config(tensor_shape_json);
        }

        if (inference_json.contains("processing_spec"))
        {
            const auto& processing_spec_json = inference_json.at("processing_spec");
            processing_spec = create_processing_spec_from_config(processing_spec_json, processing_spec_required);
        }

        single_parameters = create_single_parameters_from_config(inference_json, max_inference_time_defined);

        if (!model_data.empty() && !tensor_shape.empty() && max_inference_time_defined) {
            if (processing_spec_required) {
                m_inference_config = std::make_unique<anira::InferenceConfig>(
                    model_data,
                    tensor_shape,
                    processing_spec,
                    single_parameters.m_max_inference_time,
                    single_parameters.m_warm_up,
                    single_parameters.m_session_exclusive_processor,
                    single_parameters.m_num_parallel_processors
#ifdef USE_CONTROLLED_BLOCKING
                    single_parameters.m_wait_in_process_block
#endif
                );
            } else {
                m_inference_config = std::make_unique<anira::InferenceConfig>(
                    model_data,
                    tensor_shape,
                    single_parameters.m_max_inference_time,
                    single_parameters.m_warm_up,
                    single_parameters.m_session_exclusive_processor,
                    single_parameters.m_num_parallel_processors
#ifdef USE_CONTROLLED_BLOCKING
                    single_parameters.m_wait_in_process_block
#endif
                );
            }
        }
    }

    static std::vector<anira::ModelData> create_model_data_from_config(const nlohmann::basic_json<>& config)
    {
        std::vector<anira::ModelData> model_data;

        if (!config.is_array()) {
            LOG_ERROR << "Invalid 'model_data' value: expected an array." << std::endl;
            return model_data;
        }

        if (config.empty()) {
            LOG_ERROR << "Invalid 'model_data' array: empty array." << std::endl;
            return model_data;
        }

        for (const auto& item : config) {
            if (!item.contains("model_path") || !item.contains("inference_backend")) {
                LOG_ERROR << "Missing key pair 'model_path' and 'inference_backend' in 'model_data' array entry." << std::endl;
                continue;
            }

            if (!item.at("model_path").is_string()) {
                LOG_ERROR << "Invalid 'model_path' value: expected a string." << std::endl;
                continue;
            }

            if (!item.at("inference_backend").is_string()) {
                LOG_ERROR << "Invalid 'inference_backend' value: expected a string." << std::endl;
                continue;
            }

            const std::string model_path = item.at("model_path").get<std::string>();
            const std::string model_backend = item.at("inference_backend").get<std::string>();

            if (model_backend == "ONNX") {
                model_data.emplace_back(model_path, anira::InferenceBackend::ONNX);
            } else if (model_backend == "TFLITE") {
                model_data.emplace_back(model_path, anira::InferenceBackend::TFLITE);
            } else if (model_backend == "LIBTORCH") {
                model_data.emplace_back(model_path, anira::InferenceBackend::LIBTORCH);
            } else if (model_backend == "CUSTOM") {
                model_data.emplace_back(model_path, anira::InferenceBackend::CUSTOM);
            } else {
                LOG_ERROR << "Invalid 'inference_backend' value in 'model_data' array entry : expected a string of the following list ['ONNX', 'TFLITE', 'LIBTORCH', 'CUSTOM']." << std::endl;
            }
        }

        return model_data;
    }

    static std::vector<anira::TensorShape> create_tensor_shape_from_config(const nlohmann::basic_json<>& config)
    {
        std::vector<anira::TensorShape> tensor_shape;

        if (!config.is_array()) {
            LOG_ERROR << "Invalid 'tensor_shape' value: expected an array." << std::endl;
            return tensor_shape;
        }

        if (config.empty()) {
            LOG_ERROR << "Invalid 'tensor_shape' array: empty array." << std::endl;
            return tensor_shape;
        }

        for (const auto& item : config) {
            if (!item.contains("input_shape") || !item.contains("output_shape")) {
                LOG_ERROR << "Missing key pair 'input_shape' and 'output_shape' in 'tensor_shape' array entry." << std::endl;
                continue;
            }

            const auto& input_shape = item.at("input_shape");
            const auto& output_shape = item.at("output_shape");

            anira::TensorShapeList input_shape_list = parse_tensor_json_shape(input_shape);
            anira::TensorShapeList output_shape_list = parse_tensor_json_shape(output_shape);

            std::string tensor_backend = "UNIVERSAL";

            if (item.contains("inference_backend")) {
                if (item.at("inference_backend").is_string()) {
                    tensor_backend = item.at("inference_backend").get<std::string>();
                } else {
                    LOG_ERROR << "Invalid 'inference_backend' value in 'tensor_shape' array entry: expected a string." << std::endl;
                }
            }

            if (tensor_backend == "ONNX") {
                tensor_shape.emplace_back(input_shape_list, output_shape_list, anira::InferenceBackend::ONNX);
            } else if (tensor_backend == "TFLITE") {
                tensor_shape.emplace_back(input_shape_list, output_shape_list, anira::InferenceBackend::TFLITE);
            } else if (tensor_backend == "LIBTORCH") {
                tensor_shape.emplace_back(input_shape_list, output_shape_list, anira::InferenceBackend::LIBTORCH);
            } else if (tensor_backend == "CUSTOM") {
                tensor_shape.emplace_back(input_shape_list, output_shape_list, anira::InferenceBackend::CUSTOM);
            } else if (tensor_backend == "UNIVERSAL") {
                tensor_shape.emplace_back(input_shape_list, output_shape_list);
            } else {
                LOG_ERROR << "Invalid 'inference_backend' value in 'tensor_shape' array entry : expected a string of the following list ['ONNX', 'TFLITE', 'LIBTORCH']." << std::endl;
            }
        }

        return tensor_shape;
    }

    static anira::TensorShapeList parse_tensor_json_shape(const nlohmann::json& shape_node) {
        if (!shape_node.is_array()) {
            LOG_ERROR << "Invalid 'shape' value in 'tensor_shape' array entry: expected an array." << std::endl;
        }

        if (shape_node.empty()) {
            LOG_ERROR << "Invalid 'shape' value in 'tensor_shape' array entry: empty array." << std::endl;
            return {};
        }

        if (shape_node.front().is_array()) {
            return shape_node.get<anira::TensorShapeList>();
        }

        if (shape_node.front().is_number()) {
            std::vector<int64_t> flat_shape = shape_node.get<std::vector<int64_t>>();
            return {flat_shape};
        }

        LOG_ERROR << "Invalid 'shape' value inside 'tensor_shape' array entry: expected an array." << std::endl;
        return {};
    }

    static anira::ProcessingSpec create_processing_spec_from_config(const nlohmann::basic_json<>& config, bool& config_required)
    {
        anira::ProcessingSpec processing_spec;

        if (config.contains("preprocess_input_channels")) {
            const auto& preprocess_input_channels = config.at("preprocess_input_channels");
            processing_spec.m_preprocess_input_channels = parse_size_t_json_shape(preprocess_input_channels, "preprocess_input_channels");
            config_required = true;
        }

        if (config.contains("postprocess_output_channels")) {
            const auto& postprocess_output_channels = config.at("postprocess_output_channels");
            processing_spec.m_postprocess_output_channels = parse_size_t_json_shape(postprocess_output_channels, "postprocess_output_channels");
            config_required = true;
        }

        if (config.contains("preprocess_input_size")) {
            const auto& preprocess_input_size = config.at("preprocess_input_size");
            processing_spec.m_preprocess_input_size = parse_size_t_json_shape(preprocess_input_size, "preprocess_input_size");
            config_required = true;
        }

        if (config.contains("postprocess_output_size")) {
            const auto& postprocess_output_size = config.at("postprocess_output_size");
            processing_spec.m_postprocess_output_size = parse_size_t_json_shape(postprocess_output_size, "postprocess_output_size");
            config_required = true;
        }

        if (config.contains("internal_latency")) {
            const auto& internal_latency = config.at("internal_latency");
            processing_spec.m_internal_latency = parse_size_t_json_shape(internal_latency, "internal_latency");
            config_required = true;
        }

        return processing_spec;
    }

    static std::vector<size_t> parse_size_t_json_shape(const nlohmann::json& shape_node, std::string json_key_name)
    {
        if (!shape_node.is_array()) {
            LOG_ERROR << "Invalid '" << json_key_name << "' value: expected an array." << std::endl;
            return {};
        }

        if (shape_node.empty()) {
            LOG_ERROR << "Invalid '" << json_key_name << "' array: empty array." << std::endl;
            return {};
        }

        if (shape_node.front().is_number_unsigned()) {
            return shape_node.get<std::vector<size_t>>();
        }

        LOG_ERROR << "Invalid '" << json_key_name << "' array: expected an unsigned integer array." << std::endl;
        return {};
    }

    struct SingleParameterStruct {
        bool m_max_inference_time_set = false;
        float m_max_inference_time = 0.f;
        unsigned int m_warm_up = anira::InferenceConfig::Defaults::m_warm_up;
        bool m_session_exclusive_processor = anira::InferenceConfig::Defaults::m_session_exclusive_processor;
        unsigned int m_num_parallel_processors = anira::InferenceConfig::Defaults::m_num_parallel_processors;
#ifdef USE_CONTROLLED_BLOCKING
        static constexpr float m_wait_in_process_block = anira::InferenceConfig::Defaults::m_wait_in_process_block;
#endif
    };

    static SingleParameterStruct create_single_parameters_from_config(const nlohmann::basic_json<>& config, bool& necessary_parameter_set)
    {
        SingleParameterStruct single_parameters;

        if (config.contains("max_inference_time")) {
            const auto& max_inference_time_json = config.at("max_inference_time");
            if (max_inference_time_json.is_number_float()) {
                const float max_inference_time = max_inference_time_json.get<float>();
                single_parameters.m_max_inference_time = max_inference_time;
                necessary_parameter_set = true;
            } else {
                LOG_ERROR << "Invalid 'max_inference_time' value: expected a float." << std::endl;
            }
        } else {
            LOG_ERROR << "Missing 'max_inference_time' key." << std::endl;
        }

        if (config.contains("warm_up"))
        {
            const auto& warm_up_json = config.at("warm_up");
            if (warm_up_json.is_number_unsigned()) {
                const unsigned int warm_up = warm_up_json.get<unsigned int>();
                single_parameters.m_warm_up = warm_up;
            } else {
                LOG_ERROR << "Invalid 'warm_up' value: expected an unsigned integer." << std::endl;
            }
        }

        if (config.contains("session_exclusive_processor")) {
            const auto& session_exclusive_processor_json = config.at("session_exclusive_processor");
            if (session_exclusive_processor_json.is_boolean()) {
                const bool session_exclusive_processor = session_exclusive_processor_json.get<bool>();
                single_parameters.m_session_exclusive_processor = session_exclusive_processor;
            } else {
                LOG_ERROR << "Invalid 'session_exclusive_processor' value: expected a bool." << std::endl;
            }
        }

        if (config.contains("num_parallel_processors")) {
            const auto& num_parallel_processors_json = config.at("num_parallel_processors");
            if (num_parallel_processors_json.is_number_unsigned()) {
                const unsigned int num_parallel_processors = num_parallel_processors_json.get<unsigned int>();
                single_parameters.m_num_parallel_processors = num_parallel_processors;
            } else {
                LOG_ERROR << "Invalid 'num_parallel_processors' value: expected an unsigned integer." << std::endl;
            }
        }
#ifdef USE_CONTROLLED_BLOCKING
        if (config.contains("wait_in_process_block")) {
            const auto& wait_in_process_block_json = config.at("wait_in_process_block");
            if (wait_in_process_block_json.is_number_float()) {
                const float wait_in_process_block = wait_in_process_block_json.get<float>();
                single_parameters.m_wait_in_process_block = wait_in_process_block;
            } else {
                LOG_ERROR << "Invalid 'wait_in_process_block' value: expected a float." << std::endl;
            }
        }
#else
        if (config.contains("wait_in_process_block")) {
            LOG_INFO << "Invalid 'wait_in_process_block' value: anira was configured without the 'USE_CONTROLLED_BLOCKING' feature." << std::endl;
        }
#endif

        return single_parameters;
    }

    std::unique_ptr<anira::ContextConfig> m_context_config;
    std::unique_ptr<anira::InferenceConfig> m_inference_config;
};
} // namespace anira

#endif //JSONCONFIGLOADER_H
