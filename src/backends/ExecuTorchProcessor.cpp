#ifdef USE_EXECUTORCH

#include <anira/InferenceConfig.h>
#include <anira/abi/status.h>
#include <anira/backends/BackendBase.h>
#include <anira/backends/ExecuTorchProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>  // IWYU pragma: keep
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils/ModelFile.h"
#include "../utils/StatusError.h"
#include "ProcessingGuard.h"

// IWYU pragma: begin_keep — the ExecuTorch headers are compiled as SYSTEM includes,
// where misc-include-cleaner cannot attribute the used symbols to their providers.
#include "executorch/extension/data_loader/buffer_data_loader.h"
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor_ptr.h"
#include "executorch/extension/tensor/tensor_ptr_maker.h"
#include "executorch/extension/threadpool/threadpool.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/evalue.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
// IWYU pragma: end_keep

namespace anira {

namespace {

// Every fallible ExecuTorch call returns a runtime::Error (or a Result carrying one).
// A failure here means a setup or runtime problem, so we throw with the failing call +
// error code — this keeps a broken state from silently producing zeros.
// Control path: a failing call becomes ANIRA_ERROR_ENGINE (or the status the caller passes)
// with the call and the error named (executorch::runtime::to_string, e.g. "InvalidProgram").
inline void executorch_check(executorch::runtime::Error error,
                             const char* what,
                             anira_status failure = ANIRA_ERROR_ENGINE,
                             const std::string& where = "") {
    if (error != executorch::runtime::Error::Ok) {
        const std::string text = std::string(what) +
                                 " failed with Error::" + executorch::runtime::to_string(error) +
                                 " (" + std::to_string(static_cast<uint32_t>(error)) + ")";
        throw StatusError(failure,
                          model_file::message("executorch", where.empty() ? what : where, text));
    }
}

// Pin ExecuTorch's process-wide XNNPACK threadpool to a single thread, to match the
// other backends (anira gets its parallelism from running multiple processor
// instances, and worker-pool fan-out is unwanted on real-time audio systems). The
// threadpool is a process-global singleton, so this is done once, not per instance.
void pin_threadpool_to_one_thread() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto* threadpool = executorch::extension::threadpool::get_threadpool();
        // _unsafe_reset_threadpool is deprecated but remains the only exported way to
        // size the pool permanently: the suggested UseNThreadsThreadPoolGuard is a
        // scoped, Meta-internal API. Resizing here is safe — no inference has run yet,
        // so no threadpool pointer is held anywhere.
        // NOLINTNEXTLINE(clang-diagnostic-deprecated-declarations)
        if (threadpool == nullptr || !threadpool->_unsafe_reset_threadpool(1)) {
            ANIRA_LOG_WARNING(log_group::k_backend_executorch,
                              "could not pin the XNNPACK threadpool to "
                              "a single thread.");
        }
    });
}

}  // namespace

// Defined here, not in the header: it owns ExecuTorch runtime objects, and the
// ExecuTorch headers (with their vendored c10) must stay out of anira's public
// headers (see the note on ExecuTorchProcessor).
struct ExecuTorchProcessor::Instance {
    Instance(InferenceConfig& inference_config);
    ~Instance() = default;

    void prepare();
    void process(std::vector<BufferF>& input,
                 std::vector<BufferF>& output,
                 const std::shared_ptr<SessionElement>& session);

    std::unique_ptr<executorch::extension::Module> m_module;  ///< Loaded .pte program with
                                                              ///< its selected method

    std::string m_method;  ///< Method executed per inference: the config's model_function
    std::string m_where;   ///< The model path or "memory", for the failure messages
                           ///< (a .pte can carry several named entry points, e.g.
                           ///< encode/decode), or "forward" when none is set

    std::vector<std::vector<float>> m_input_data;  ///< Instance-owned host memory backing
                                                   ///< the input tensors
    std::vector<executorch::extension::TensorPtr> m_input_tensors;  ///< Input tensors
                                                                    ///< wrapping m_input_data
    std::vector<executorch::runtime::EValue> m_input_values;        ///< Reusable 'forward'
                                                                    ///< argument list

    InferenceConfig& m_inference_config;    ///< Reference to inference configuration
    std::atomic<bool> m_processing{false};  ///< Flag indicating if instance is currently
                                            ///< processing
};

ExecuTorchProcessor::ExecuTorchProcessor(InferenceConfig& inference_config)
    : BackendBase(inference_config) {
    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

ExecuTorchProcessor::~ExecuTorchProcessor() = default;

void ExecuTorchProcessor::prepare() {
    for (auto& instance : m_instances) { instance->prepare(); }
}

void ExecuTorchProcessor::process(std::vector<BufferF>& input,
                                  std::vector<BufferF>& output,
                                  std::shared_ptr<SessionElement> session) {
    while (true) {
        for (auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                // The flag is released on every path, a throw of a type the instance's
                // catch does not name included, so a failing inference can never leave the
                // instance busy forever.
                const detail::ProcessingGuard guard(instance->m_processing);
                instance->process(input, output, session);
                return;
            }
        }
    }
}

ExecuTorchProcessor::Instance::Instance(InferenceConfig& inference_config)
    : m_inference_config(inference_config) {
    pin_threadpool_to_one_thread();

    if (inference_config.is_model_binary(anira::InferenceBackend::EXECUTORCH)) {
        const anira::ModelData* model_data =
            m_inference_config.get_model_data(anira::InferenceBackend::EXECUTORCH);
        assert(model_data && "Model data not found for binary model!");
        // BufferDataLoader keeps a pointer into the caller's buffer; the ModelData
        // blob lives in the InferenceConfig, which outlives this instance.
        m_where = model_file::k_memory;
        m_module = std::make_unique<executorch::extension::Module>(
            std::make_unique<executorch::extension::BufferDataLoader>(model_data->m_data,
                                                                      model_data->m_size));
    } else {
        m_where = model_file::require_readable(
            m_inference_config.get_model_path(anira::InferenceBackend::EXECUTORCH),
            "executorch");
        m_module = std::make_unique<executorch::extension::Module>(m_where);
    }

    // Load the program and the selected method up front: this parses the .pte,
    // initializes the delegates and allocates the planned memory — none of which may
    // happen lazily on the real-time inference path.
    m_method = m_inference_config.get_model_function(anira::InferenceBackend::EXECUTORCH);
    if (m_method.empty()) { m_method = "forward"; }
    executorch_check(m_module->load_method(m_method),
                     ("Module::load_method(\"" + m_method + "\")").c_str(),
                     ANIRA_ERROR_MODEL_LOAD,
                     m_where);

    // Build the input tensors once, wrapping instance-owned host memory: the .pte
    // interface is positional float32 tensors of the configured shapes.
    const size_t num_inputs = m_inference_config.get_tensor_input_shape().size();
    m_input_data.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        const std::vector<int64_t>& shape =
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::EXECUTORCH)[i];
        std::vector<executorch::aten::SizesType> sizes(shape.begin(), shape.end());
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i], 0.f);
        m_input_tensors.emplace_back(
            executorch::extension::from_blob(m_input_data[i].data(),
                                             std::move(sizes),
                                             executorch::aten::ScalarType::Float));
        m_input_values.emplace_back(*m_input_tensors.back());
    }

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        const auto result = m_module->execute(m_method, m_input_values);
        executorch_check(result.error(),
                         ("Module::execute(\"" + m_method + "\") (warm-up)").c_str());
    }
}

void ExecuTorchProcessor::Instance::prepare() {
    // Reset input buffers to a known (zero) state between sessions.
    for (auto& input : m_input_data) { std::ranges::fill(input, 0.f); }
}

void ExecuTorchProcessor::Instance::process(std::vector<BufferF>& input,
                                            std::vector<BufferF>& output,
                                            const std::shared_ptr<SessionElement>& session) {
    // Catch+log like the other backends (cf. OnnxRuntimeProcessor): an ExecuTorch
    // runtime failure must not throw out onto the real-time inference thread (the
    // caller's ProcessingGuard releases m_processing either way).
    try {
        for (size_t i = 0; i < m_input_data.size(); ++i) {
            std::memcpy(m_input_data[i].data(),
                        input[i].get_memory_block().data(),
                        m_inference_config.get_tensor_input_size()[i] * sizeof(float));
        }

        const auto result = m_module->execute(m_method, m_input_values);
        executorch_check(result.error(), ("Module::execute(\"" + m_method + "\")").c_str());
        const std::vector<executorch::runtime::EValue>& outputs = result.get();

        const size_t num_outputs = m_inference_config.get_tensor_output_shape().size();
        if (outputs.size() < num_outputs) {
            throw std::runtime_error("[anira][ExecuTorch] model returned " +
                                     std::to_string(outputs.size()) + " outputs, expected " +
                                     std::to_string(num_outputs));
        }
        for (size_t i = 0; i < num_outputs; ++i) {
            if (!outputs[i].isTensor()) {
                throw std::runtime_error("[anira][ExecuTorch] model output " + std::to_string(i) +
                                         " is not a tensor");
            }
            std::memcpy(output[i].get_memory_block().data(),
                        outputs[i].toTensor().const_data_ptr<float>(),
                        m_inference_config.get_tensor_output_size()[i] * sizeof(float));
        }
    } catch (const std::exception& e) {
        // No caller waits for this task: deliver zeros, never stale output. The failure is
        // ENGINE on the session's latch (a 3.x handler's word), logged on the first failure
        // since the latch's re-arm and counted afterwards; the direct processor tests pass
        // no session and get every record.
        if (session == nullptr || session->m_rt->record(ANIRA_ERROR_ENGINE)) {
            ANIRA_LOG_RT_ERROR(log_group::k_backend_executorch, "%s", e.what());
        }
        for (auto& buffer : output) { buffer.clear(); }
    }
}

}  // namespace anira

#endif  // USE_EXECUTORCH
