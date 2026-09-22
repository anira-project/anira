#include "Adapters.h"

#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/backends/BackendBase.h>
#ifdef USE_EXECUTORCH
#include <anira/backends/ExecuTorchProcessor.h>
#endif
#ifdef USE_LIBTORCH
#include <anira/backends/LibTorchProcessor.h>
#endif
#ifdef USE_LITERT
#include <anira/backends/LiteRtProcessor.h>
#endif
#ifdef USE_ONNXRUNTIME
#include <anira/backends/OnnxRuntimeProcessor.h>
#endif
#ifdef USE_TFLITE
#include <anira/backends/TFLiteProcessor.h>
#endif
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Adapter.h"
#include "LegacyAdapter.h"

namespace anira::backend {

namespace {

// The element count of one tensor's extents.
size_t product_of(const std::vector<int64_t>& dims) {
    size_t elements = 1;
    for (const int64_t extent : dims) { elements *= extent > 0 ? static_cast<size_t>(extent) : 0U; }
    return elements;
}

// One request of the 2.x table: the row's backend, its model record, and its source.
PlanRequest legacy_request(const anira::InferenceConfig& config,
                           anira::InferenceBackend backend,
                           anira::BackendBase* custom,
                           bool missing_model) {
    PlanRequest request;
    request.m_legacy_backend = backend;
    request.m_model = model_of(config, backend);
    request.m_missing_model = missing_model;
    if (backend == anira::InferenceBackend::CUSTOM) {
        request.m_source = custom != nullptr ? Source::Legacy : Source::Roundtrip;
        request.m_backend = custom;
    } else {
        request.m_source = missing_model ? Source::Roundtrip : Source::BuiltIn;
    }
    return request;
}

// The 2.x backend of a built-in engine; CUSTOM for an engine this build does not carry.
anira::InferenceBackend backend_of_engine(anira_engine engine) noexcept {
    switch (engine) {
#ifdef USE_LIBTORCH
        case ANIRA_ENGINE_LIBTORCH: return anira::InferenceBackend::LIBTORCH;
#endif
#ifdef USE_ONNXRUNTIME
        case ANIRA_ENGINE_ONNXRUNTIME: return anira::InferenceBackend::ONNX;
#endif
#ifdef USE_TFLITE
        case ANIRA_ENGINE_TFLITE: return anira::InferenceBackend::TFLITE;
#endif
#ifdef USE_LITERT
        case ANIRA_ENGINE_LITERT: return anira::InferenceBackend::LITERT;
#endif
#ifdef USE_EXECUTORCH
        case ANIRA_ENGINE_EXECUTORCH: return anira::InferenceBackend::EXECUTORCH;
#endif
        default: return anira::InferenceBackend::CUSTOM;
    }
}

// The 2.x processors, one factory each: what a LegacyAdapter of a built-in engine builds at
// prepare from the record's 2.x configuration.
#ifdef USE_LIBTORCH
std::unique_ptr<anira::BackendBase> make_libtorch_processor(anira::InferenceConfig& config) {
    return std::make_unique<anira::LibtorchProcessor>(config);
}
#endif
#ifdef USE_ONNXRUNTIME
std::unique_ptr<anira::BackendBase> make_onnxruntime_processor(anira::InferenceConfig& config) {
    return std::make_unique<anira::OnnxRuntimeProcessor>(config);
}
#endif
#ifdef USE_TFLITE
std::unique_ptr<anira::BackendBase> make_tflite_processor(anira::InferenceConfig& config) {
    return std::make_unique<anira::TFLiteProcessor>(config);
}
#endif
#ifdef USE_LITERT
std::unique_ptr<anira::BackendBase> make_litert_processor(anira::InferenceConfig& config) {
    return std::make_unique<anira::LiteRtProcessor>(config);
}
#endif
#ifdef USE_EXECUTORCH
std::unique_ptr<anira::BackendBase> make_executorch_processor(anira::InferenceConfig& config) {
    return std::make_unique<anira::ExecuTorchProcessor>(config);
}
#endif

}  // namespace

anira::InferenceConfig legacy_config_of(const Model& model) {
    const anira::InferenceBackend backend = backend_of_engine(model.m_engine);
    std::vector<anira::ModelData> rows;
    if (model.m_bytes != nullptr) {
        // Borrowed, as the 2.x binary ModelData always is: the record's owner keeps the
        // bytes alive for the adapter's life.
        rows.emplace_back(
            const_cast<void*>(model.m_bytes),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
            model.m_num_bytes,
            backend,
            model.m_entry,
            /*is_binary=*/true);
    } else {
        rows.emplace_back(model.m_path, backend, model.m_entry);
    }
    anira::TensorShapeList inputs;
    anira::TensorShapeList outputs;
    for (const TensorInfo& tensor : model.m_inputs) { inputs.push_back(tensor.m_dims); }
    for (const TensorInfo& tensor : model.m_outputs) { outputs.push_back(tensor.m_dims); }
    // The budget is not read by a 2.x processor; the constructor refuses zero.
    constexpr float k_unread_budget_ms = 1.F;
    return anira::InferenceConfig(
        std::move(rows),
        std::vector<anira::TensorShape>{anira::TensorShape(std::move(inputs), std::move(outputs))},
        anira::ProcessingSpec{},
        k_unread_budget_ms,
        model.m_warm_up,
        model.m_session_exclusive,
        /*blocking_ratio=*/0.F,
        model.m_instances);
}

std::shared_ptr<Adapter> make_builtin_adapter(anira_engine engine) {
    // The five 2.x processors, unchanged, behind the legacy adapter: each is built at
    // prepare from the record's 2.x configuration and owned by its adapter.
    switch (engine) {
#ifdef USE_LIBTORCH
        case ANIRA_ENGINE_LIBTORCH:
            return std::make_shared<LegacyAdapter>(&make_libtorch_processor);
#endif
#ifdef USE_ONNXRUNTIME
        case ANIRA_ENGINE_ONNXRUNTIME:
            return std::make_shared<LegacyAdapter>(&make_onnxruntime_processor);
#endif
#ifdef USE_TFLITE
        case ANIRA_ENGINE_TFLITE: return std::make_shared<LegacyAdapter>(&make_tflite_processor);
#endif
#ifdef USE_LITERT
        case ANIRA_ENGINE_LITERT: return std::make_shared<LegacyAdapter>(&make_litert_processor);
#endif
#ifdef USE_EXECUTORCH
        case ANIRA_ENGINE_EXECUTORCH:
            return std::make_shared<LegacyAdapter>(&make_executorch_processor);
#endif
        default: return nullptr;
    }
}

anira_engine engine_of(anira::InferenceBackend backend) noexcept {
    switch (backend) {
#ifdef USE_LIBTORCH
        case anira::InferenceBackend::LIBTORCH: return ANIRA_ENGINE_LIBTORCH;
#endif
#ifdef USE_ONNXRUNTIME
        case anira::InferenceBackend::ONNX: return ANIRA_ENGINE_ONNXRUNTIME;
#endif
#ifdef USE_TFLITE
        case anira::InferenceBackend::TFLITE: return ANIRA_ENGINE_TFLITE;
#endif
#ifdef USE_LITERT
        case anira::InferenceBackend::LITERT: return ANIRA_ENGINE_LITERT;
#endif
#ifdef USE_EXECUTORCH
        case anira::InferenceBackend::EXECUTORCH: return ANIRA_ENGINE_EXECUTORCH;
#endif
        case anira::InferenceBackend::CUSTOM:
        default: return ANIRA_ENGINE_NONE;
    }
}

Model model_of(const anira::InferenceConfig& config, anira::InferenceBackend backend) {
    Model model;
    model.m_engine = engine_of(backend);
    if (const anira::ModelData* row = config.get_model_data(backend)) {
        if (row->m_is_binary) {
            model.m_bytes = row->m_data;
            model.m_num_bytes = row->m_size;
        } else {
            model.m_path.assign(static_cast<const char*>(row->m_data), row->m_size);
        }
        model.m_entry = row->m_model_function;
    }
    // The shapes the 2.x processors read: the backend-qualified row where the configuration
    // holds one, the universal row else (InferenceConfig::get_tensor_shape).
    for (const std::vector<int64_t>& dims : config.get_tensor_input_shape(backend)) {
        TensorInfo tensor;
        tensor.m_dims = dims;
        tensor.m_num_elements = product_of(dims);
        model.m_inputs.push_back(std::move(tensor));
    }
    for (const std::vector<int64_t>& dims : config.get_tensor_output_shape(backend)) {
        TensorInfo tensor;
        tensor.m_dims = dims;
        tensor.m_num_elements = product_of(dims);
        model.m_outputs.push_back(std::move(tensor));
    }
    model.m_instances = config.m_num_parallel_processors;
    model.m_warm_up = config.m_warm_up;
    model.m_log_level = anira::get_log_level();
    model.m_session_exclusive = config.m_session_exclusive_processor;
    return model;
}

std::vector<PlanRequest> legacy_plan_requests(const anira::InferenceConfig& config,
                                              anira::BackendBase* custom) {
    const std::vector<anira::InferenceBackend> every_backend = {
#ifdef USE_LIBTORCH
        anira::InferenceBackend::LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
        anira::InferenceBackend::ONNX,
#endif
#ifdef USE_TFLITE
        anira::InferenceBackend::TFLITE,
#endif
#ifdef USE_LITERT
        anira::InferenceBackend::LITERT,
#endif
#ifdef USE_EXECUTORCH
        anira::InferenceBackend::EXECUTORCH,
#endif
        anira::InferenceBackend::CUSTOM,
    };
    std::vector<PlanRequest> requests;
    requests.reserve(config.m_model_data.size() + every_backend.size());
    const auto has_row = [&requests](anira::InferenceBackend backend) {
        return std::ranges::any_of(requests, [backend](const PlanRequest& request) {
            return request.m_legacy_backend == backend;
        });
    };
    // One row per configured model, in m_model_data order (the order a 3.x handler numbers its
    // plans in).
    for (const anira::ModelData& row : config.m_model_data) {
        requests.push_back(legacy_request(config, row.m_backend, custom, /*missing_model=*/false));
    }
    // Then every other backend of the build, CUSTOM last: a row for every backend a 2.x caller
    // can name. Selecting one without a model runs the default processor, as it always did.
    for (const anira::InferenceBackend backend : every_backend) {
        if (has_row(backend)) { continue; }
        requests.push_back(
            legacy_request(config,
                           backend,
                           custom,
                           /*missing_model=*/backend != anira::InferenceBackend::CUSTOM));
    }
    return requests;
}

}  // namespace anira::backend
