#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/JsonConfigLoader.h>
#include <anira/utils/Logger.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>  // IWYU pragma: keep - declares the nlohmann::json type name
#include <string>
#include <utility>
#include <vector>

anira::JsonConfigLoader::JsonConfigLoader(const std::string& file_path) {
    std::ifstream config_file(file_path);
    if (!config_file.is_open()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config, "Could not open file at %s", file_path.c_str());
    }
    initialize_from_stream(config_file);
}

anira::JsonConfigLoader::JsonConfigLoader(std::istream& stream) {
    initialize_from_stream(stream);
}

std::unique_ptr<anira::ContextConfig> anira::JsonConfigLoader::get_context_config() {
    return std::move(m_context_config);
}

std::unique_ptr<anira::InferenceConfig> anira::JsonConfigLoader::get_inference_config() {
    return std::move(m_inference_config);
}

void anira::JsonConfigLoader::initialize_from_stream(std::istream& stream) {
    try {
        nlohmann::json json_config;
        stream >> json_config;
        parse(json_config);
    } catch (const nlohmann::json::parse_error& e) {
        ANIRA_LOG_ERROR(anira::log_group::k_config, "JSON parse error: %s", e.what());
    }
}

void anira::JsonConfigLoader::parse(const nlohmann::json& config) {
    parse_context_config(config);
    parse_inference_config(config);
}

void anira::JsonConfigLoader::parse_context_config(const nlohmann::json& config) {
    m_context_config = std::make_unique<anira::ContextConfig>();

    if (!config.contains("context_config")) { return; }
    const auto& context_json = config.at("context_config");

    if (context_json.contains("num_threads")) {
        if (context_json.at("num_threads").is_number_unsigned()) {
#ifdef __EMSCRIPTEN__
            // The context cannot run inference threads on WebAssembly — they are
            // always supplied externally (e.g. AniraWeb.spinUpInferenceWorker()).
            // Accept the (valid) value so shared config files keep working, but
            // coerce it.
            if (context_json.at("num_threads").get<unsigned int>() > 0) {
                ANIRA_LOG_WARNING(anira::log_group::k_config,
                                  "'num_threads' > 0 is not supported on WebAssembly builds: "
                                  "inference threads must be supplied externally (e.g. "
                                  "AniraWeb.spinUpInferenceWorker()). Using num_threads = 0.");
            }
#else
            m_context_config->m_num_threads = context_json.at("num_threads").get<unsigned int>();
#endif
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'num_threads' value: expected an unsigned integer.");
        }
    }

    if (context_json.contains("wait_strategy")) {
        const auto& strategy_json = context_json.at("wait_strategy");
        std::string const strategy =
            strategy_json.is_string() ? strategy_json.get<std::string>() : std::string();
        if (strategy == "spin_backoff") {
            m_context_config->m_wait_strategy = anira::WaitStrategy::SpinBackoff;
        } else if (strategy == "blocking") {
#ifdef __EMSCRIPTEN__
            // Blocking waits are impossible on WebAssembly: inference loops are
            // driven cooperatively by JS Workers, and there is no pthreads
            // runtime to block on. Accept the (valid) value so shared config
            // files keep working, but coerce it.
            ANIRA_LOG_WARNING(anira::log_group::k_config,
                              "wait_strategy 'blocking' is not supported on WebAssembly builds. "
                              "Using 'spin_backoff'.");
#else
            m_context_config->m_wait_strategy = anira::WaitStrategy::Blocking;
#endif
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'wait_strategy' value: expected \"spin_backoff\" or "
                            "\"blocking\". Defaulting to \"spin_backoff\".");
        }
    }

    const auto parse_level = [this](const nlohmann::json& level_json, const char* key) {
        std::string const level =
            level_json.is_string() ? level_json.get<std::string>() : std::string();
        if (level == "debug") {
            m_context_config->m_log.m_level = anira::LogLevel::Debug;
        } else if (level == "info") {
            m_context_config->m_log.m_level = anira::LogLevel::Info;
        } else if (level == "warning") {
            m_context_config->m_log.m_level = anira::LogLevel::Warning;
        } else if (level == "error") {
            m_context_config->m_log.m_level = anira::LogLevel::Error;
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid '%s' value: expected \"debug\", \"info\", \"warning\" or "
                            "\"error\". Using the default log level.",
                            key);
        }
    };

    // Legacy key (anira <= 2.2): the level alone. Superseded by the "log" block.
    if (context_json.contains("log_level")) {
        parse_level(context_json.at("log_level"), "log_level");
    }

    if (context_json.contains("log")) {
        const auto& log_json = context_json.at("log");
        if (!log_json.is_object()) {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'log' value: expected an object with the optional keys "
                            "'level', 'drain', 'queue_capacity' and 'drain_interval_ms'.");
            return;
        }
        if (log_json.contains("level")) { parse_level(log_json.at("level"), "log.level"); }
        if (log_json.contains("drain")) {
            const auto& drain_json = log_json.at("drain");
            std::string const drain =
                drain_json.is_string() ? drain_json.get<std::string>() : std::string();
            if (drain == "thread") {
#ifdef __EMSCRIPTEN__
                ANIRA_LOG_WARNING(anira::log_group::k_config,
                                  "log.drain 'thread' is not supported on WebAssembly builds: "
                                  "no thread can drain the log queue there. Using 'manual'.");
                m_context_config->m_log.m_drain = anira::LogDrain::Manual;
#else
                m_context_config->m_log.m_drain = anira::LogDrain::Thread;
#endif
            } else if (drain == "manual") {
                m_context_config->m_log.m_drain = anira::LogDrain::Manual;
            } else {
                ANIRA_LOG_ERROR(anira::log_group::k_config,
                                "Invalid 'log.drain' value: expected \"thread\" or \"manual\". "
                                "Using the default.");
            }
        }
        if (log_json.contains("queue_capacity")) {
            const auto& capacity_json = log_json.at("queue_capacity");
            if (capacity_json.is_number_unsigned()) {
                m_context_config->m_log.m_queue_capacity = capacity_json.get<size_t>();
            } else {
                ANIRA_LOG_ERROR(anira::log_group::k_config,
                                "Invalid 'log.queue_capacity' value: expected an unsigned "
                                "integer. Using the default.");
            }
        }
        if (log_json.contains("drain_interval_ms")) {
            const auto& interval_json = log_json.at("drain_interval_ms");
            if (interval_json.is_number_unsigned()) {
                m_context_config->m_log.m_drain_interval_ms = interval_json.get<uint32_t>();
            } else {
                ANIRA_LOG_ERROR(anira::log_group::k_config,
                                "Invalid 'log.drain_interval_ms' value: expected an unsigned "
                                "integer. Using the default.");
            }
        }
    }
}

void anira::JsonConfigLoader::parse_inference_config(const nlohmann::json& config) {
    if (!config.contains("inference_config")) {
        ANIRA_LOG_ERROR(anira::log_group::k_config, "Missing 'inference_config' key.");
        return;
    }

    const auto& inference_json = config.at("inference_config");

    std::vector<anira::ModelData> model_data;
    std::vector<anira::TensorShape> tensor_shape;
    anira::ProcessingSpec processing_spec;
    SingleParameterStruct single_parameters;

    bool processing_spec_required = false;
    bool max_inference_time_defined = false;

    if (inference_json.contains("model_data")) {
        const auto& model_data_json = inference_json.at("model_data");
        model_data = create_model_data_from_config(model_data_json);
    }

    if (inference_json.contains("tensor_shape")) {
        const auto& tensor_shape_json = inference_json.at("tensor_shape");
        tensor_shape = create_tensor_shape_from_config(tensor_shape_json);
    }

    if (inference_json.contains("processing_spec")) {
        const auto& processing_spec_json = inference_json.at("processing_spec");
        processing_spec =
            create_processing_spec_from_config(processing_spec_json, processing_spec_required);
    }

    single_parameters =
        create_single_parameters_from_config(inference_json, max_inference_time_defined);

    if (!model_data.empty() && !tensor_shape.empty() && max_inference_time_defined) {
        if (processing_spec_required) {
            m_inference_config = std::make_unique<anira::InferenceConfig>(
                model_data,
                tensor_shape,
                processing_spec,
                single_parameters.m_max_inference_time,
                single_parameters.m_warm_up,
                single_parameters.m_session_exclusive_processor,
                single_parameters.m_blocking_ratio,
                single_parameters.m_num_parallel_processors);
        } else {
            m_inference_config = std::make_unique<anira::InferenceConfig>(
                model_data,
                tensor_shape,
                single_parameters.m_max_inference_time,
                single_parameters.m_warm_up,
                single_parameters.m_session_exclusive_processor,
                single_parameters.m_blocking_ratio,
                single_parameters.m_num_parallel_processors);
        }
    }
}

std::vector<anira::ModelData> anira::JsonConfigLoader::create_model_data_from_config(
    const nlohmann::basic_json<>& config) {
    std::vector<anira::ModelData> model_data;

    if (!config.is_array()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config,
                        "Invalid 'model_data' value: expected an array.");
        return model_data;
    }

    if (config.empty()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config, "Invalid 'model_data' array: empty array.");
        return model_data;
    }

    for (const auto& item : config) {
        if (!item.contains("model_path") || !item.contains("inference_backend")) {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Missing key pair 'model_path' and 'inference_backend' in 'model_data' "
                            "array entry.");
            continue;
        }

        if (!item.at("model_path").is_string()) {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'model_path' value: expected a string.");
            continue;
        }

        if (!item.at("inference_backend").is_string()) {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'inference_backend' value: expected a string.");
            continue;
        }

        const std::string model_path = item.at("model_path").get<std::string>();
        const std::string model_backend = item.at("inference_backend").get<std::string>();

        if (model_backend == "ONNX") {
#if USE_ONNXRUNTIME
            model_data.emplace_back(model_path, anira::InferenceBackend::ONNX);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'model_data' array entry : ONNX "
                            "currently disabled in config.");
#endif
        } else if (model_backend == "TFLITE") {
#if USE_TFLITE
            model_data.emplace_back(model_path, anira::InferenceBackend::TFLITE);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'model_data' array entry : "
                            "TFLITE currently disabled in config.");
#endif
        } else if (model_backend == "LITERT") {
#if USE_LITERT
            model_data.emplace_back(model_path, anira::InferenceBackend::LITERT);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'model_data' array entry : "
                            "LITERT currently disabled in config.");
#endif
        } else if (model_backend == "EXECUTORCH") {
#if USE_EXECUTORCH
            if (item.contains("model_function")) {
                if (!item.at("model_function").is_string()) {
                    ANIRA_LOG_ERROR(anira::log_group::k_config,
                                    "Invalid 'model_function' value in 'model_data' array entry: "
                                    "expected a string.");
                    continue;
                }
                const std::string model_function = item.at("model_function").get<std::string>();
                model_data.emplace_back(model_path,
                                        anira::InferenceBackend::EXECUTORCH,
                                        model_function);
            } else {
                model_data.emplace_back(model_path, anira::InferenceBackend::EXECUTORCH);
            }
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'model_data' array entry : "
                            "EXECUTORCH currently disabled in config.");
#endif
        } else if (model_backend == "LIBTORCH") {
#if USE_LIBTORCH
            if (item.contains("model_function")) {
                if (!item.at("model_function").is_string()) {
                    ANIRA_LOG_ERROR(anira::log_group::k_config,
                                    "Invalid 'model_function' value in 'model_data' array entry: "
                                    "expected a string.");
                    continue;
                }
                const std::string model_function = item.at("model_function").get<std::string>();
                model_data.emplace_back(model_path,
                                        anira::InferenceBackend::LIBTORCH,
                                        model_function);
            } else {
                model_data.emplace_back(model_path, anira::InferenceBackend::LIBTORCH);
            }
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'model_data' array entry : "
                            "LIBTORCH currently disabled in config.");
#endif
        } else if (model_backend == "CUSTOM") {
            model_data.emplace_back(model_path, anira::InferenceBackend::CUSTOM);
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'inference_backend' value in 'model_data' array entry : "
                            "expected a string of the following list ['ONNX', 'TFLITE', 'LITERT', "
                            "'EXECUTORCH', 'LIBTORCH', 'CUSTOM'].");
        }
    }

    return model_data;
}

std::vector<anira::TensorShape> anira::JsonConfigLoader::create_tensor_shape_from_config(
    const nlohmann::basic_json<>& config) {
    std::vector<anira::TensorShape> tensor_shape;

    if (!config.is_array()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config,
                        "Invalid 'tensor_shape' value: expected an array.");
        return tensor_shape;
    }

    if (config.empty()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config, "Invalid 'tensor_shape' array: empty array.");
        return tensor_shape;
    }

    for (const auto& item : config) {
        if (!item.contains("input_shape") || !item.contains("output_shape")) {
            ANIRA_LOG_ERROR(
                anira::log_group::k_config,
                "Missing key pair 'input_shape' and 'output_shape' in 'tensor_shape' array entry.");
            continue;
        }

        const auto& input_shape = item.at("input_shape");
        const auto& output_shape = item.at("output_shape");

        anira::TensorShapeList const input_shape_list = parse_tensor_json_shape(input_shape);
        anira::TensorShapeList const output_shape_list = parse_tensor_json_shape(output_shape);

        std::string tensor_backend = "UNIVERSAL";

        if (item.contains("inference_backend")) {
            if (item.at("inference_backend").is_string()) {
                tensor_backend = item.at("inference_backend").get<std::string>();
            } else {
                ANIRA_LOG_ERROR(anira::log_group::k_config,
                                "Invalid 'inference_backend' value in 'tensor_shape' array entry: "
                                "expected a string.");
            }
        }

        if (tensor_backend == "ONNX") {
#if USE_ONNXRUNTIME
            tensor_shape.emplace_back(input_shape_list,
                                      output_shape_list,
                                      anira::InferenceBackend::ONNX);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'tensor_shape' array entry : "
                            "ONNX currently disabled in config.");
#endif
        } else if (tensor_backend == "TFLITE") {
#if USE_TFLITE
            tensor_shape.emplace_back(input_shape_list,
                                      output_shape_list,
                                      anira::InferenceBackend::TFLITE);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'tensor_shape' array entry : "
                            "TFLITE currently disabled in config.");
#endif
        } else if (tensor_backend == "LITERT") {
#if USE_LITERT
            tensor_shape.emplace_back(input_shape_list,
                                      output_shape_list,
                                      anira::InferenceBackend::LITERT);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'tensor_shape' array entry : "
                            "LITERT currently disabled in config.");
#endif
        } else if (tensor_backend == "EXECUTORCH") {
#if USE_EXECUTORCH
            tensor_shape.emplace_back(input_shape_list,
                                      output_shape_list,
                                      anira::InferenceBackend::EXECUTORCH);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'tensor_shape' array entry : "
                            "EXECUTORCH currently disabled in config.");
#endif
        } else if (tensor_backend == "LIBTORCH") {
#if USE_LIBTORCH
            tensor_shape.emplace_back(input_shape_list,
                                      output_shape_list,
                                      anira::InferenceBackend::LIBTORCH);
#else
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Disabled 'inference_backend' value in 'tensor_shape' array entry : "
                            "LIBTORCH currently disabled in config.");
#endif
        } else if (tensor_backend == "CUSTOM") {
            tensor_shape.emplace_back(input_shape_list,
                                      output_shape_list,
                                      anira::InferenceBackend::CUSTOM);
        } else if (tensor_backend == "UNIVERSAL") {
            tensor_shape.emplace_back(input_shape_list, output_shape_list);
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'inference_backend' value in 'tensor_shape' array entry : "
                            "expected a string of the following list ['ONNX', 'TFLITE', 'LITERT', "
                            "'EXECUTORCH', 'LIBTORCH'].");
        }
    }

    return tensor_shape;
}

anira::TensorShapeList anira::JsonConfigLoader::parse_tensor_json_shape(
    const nlohmann::json& shape_node) {
    if (!shape_node.is_array()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config,
                        "Invalid 'shape' value in 'tensor_shape' array entry: expected an array.");
    }

    if (shape_node.empty()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config,
                        "Invalid 'shape' value in 'tensor_shape' array entry: empty array.");
        return {};
    }

    if (shape_node.front().is_array()) { return shape_node.get<anira::TensorShapeList>(); }

    if (shape_node.front().is_number()) {
        std::vector<int64_t> const flat_shape = shape_node.get<std::vector<int64_t>>();
        return {flat_shape};
    }

    ANIRA_LOG_ERROR(anira::log_group::k_config,
                    "Invalid 'shape' value inside 'tensor_shape' array entry: expected an array.");
    return {};
}

anira::ProcessingSpec anira::JsonConfigLoader::create_processing_spec_from_config(
    const nlohmann::basic_json<>& config,
    bool& config_required) {
    anira::ProcessingSpec processing_spec;

    if (config.contains("preprocess_input_channels")) {
        const auto& preprocess_input_channels = config.at("preprocess_input_channels");
        processing_spec.m_preprocess_input_channels =
            parse_size_t_json_shape(preprocess_input_channels, "preprocess_input_channels");
        config_required = true;
    }

    if (config.contains("postprocess_output_channels")) {
        const auto& postprocess_output_channels = config.at("postprocess_output_channels");
        processing_spec.m_postprocess_output_channels =
            parse_size_t_json_shape(postprocess_output_channels, "postprocess_output_channels");
        config_required = true;
    }

    if (config.contains("preprocess_input_size")) {
        const auto& preprocess_input_size = config.at("preprocess_input_size");
        processing_spec.m_preprocess_input_size =
            parse_size_t_json_shape(preprocess_input_size, "preprocess_input_size");
        config_required = true;
    }

    if (config.contains("postprocess_output_size")) {
        const auto& postprocess_output_size = config.at("postprocess_output_size");
        processing_spec.m_postprocess_output_size =
            parse_size_t_json_shape(postprocess_output_size, "postprocess_output_size");
        config_required = true;
    }

    if (config.contains("internal_model_latency")) {
        const auto& internal_model_latency = config.at("internal_model_latency");
        processing_spec.m_internal_model_latency =
            parse_size_t_json_shape(internal_model_latency, "internal_model_latency");
        config_required = true;
    }

    return processing_spec;
}

std::vector<size_t> anira::JsonConfigLoader::parse_size_t_json_shape(
    const nlohmann::json& shape_node,
    const std::string& json_key_name) {
    if (!shape_node.is_array()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config,
                        "Invalid '%s' value: expected an array.",
                        json_key_name.c_str());
        return {};
    }

    if (shape_node.empty()) {
        ANIRA_LOG_ERROR(anira::log_group::k_config,
                        "Invalid '%s' array: empty array.",
                        json_key_name.c_str());
        return {};
    }

    if (shape_node.front().is_number_unsigned()) { return shape_node.get<std::vector<size_t>>(); }

    ANIRA_LOG_ERROR(anira::log_group::k_config,
                    "Invalid '%s' array: expected an unsigned integer array.",
                    json_key_name.c_str());
    return {};
}

anira::JsonConfigLoader::SingleParameterStruct
    anira::JsonConfigLoader::create_single_parameters_from_config(
        const nlohmann::basic_json<>& config,
        bool& necessary_parameter_set) {
    SingleParameterStruct single_parameters;

    if (config.contains("max_inference_time")) {
        const auto& max_inference_time_json = config.at("max_inference_time");
        if (max_inference_time_json.is_number_float()) {
            const float max_inference_time = max_inference_time_json.get<float>();
            single_parameters.m_max_inference_time = max_inference_time;
            necessary_parameter_set = true;
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'max_inference_time' value: expected a float.");
        }
    } else {
        ANIRA_LOG_ERROR(anira::log_group::k_config, "Missing 'max_inference_time' key.");
    }

    if (config.contains("warm_up")) {
        const auto& warm_up_json = config.at("warm_up");
        if (warm_up_json.is_number_unsigned()) {
            const unsigned int warm_up = warm_up_json.get<unsigned int>();
            single_parameters.m_warm_up = warm_up;
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'warm_up' value: expected an unsigned integer.");
        }
    }

    if (config.contains("session_exclusive_processor")) {
        const auto& session_exclusive_processor_json = config.at("session_exclusive_processor");
        if (session_exclusive_processor_json.is_boolean()) {
            const bool session_exclusive_processor = session_exclusive_processor_json.get<bool>();
            single_parameters.m_session_exclusive_processor = session_exclusive_processor;
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'session_exclusive_processor' value: expected a bool.");
        }
    }

    if (config.contains("blocking_ratio")) {
        const auto& blocking_ratio_json = config.at("blocking_ratio");
        if (blocking_ratio_json.is_number_float()) {
            const float blocking_ratio = blocking_ratio_json.get<float>();
            single_parameters.m_blocking_ratio = blocking_ratio;
        } else {
            ANIRA_LOG_ERROR(anira::log_group::k_config,
                            "Invalid 'blocking_ratio' value: expected a float.");
        }
    }

    if (config.contains("num_parallel_processors")) {
        const auto& num_parallel_processors_json = config.at("num_parallel_processors");
        if (num_parallel_processors_json.is_number_unsigned()) {
            const unsigned int num_parallel_processors =
                num_parallel_processors_json.get<unsigned int>();
            single_parameters.m_num_parallel_processors = num_parallel_processors;
        } else {
            ANIRA_LOG_ERROR(
                anira::log_group::k_config,
                "Invalid 'num_parallel_processors' value: expected an unsigned integer.");
        }
    }

    return single_parameters;
}
