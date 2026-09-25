// The compat gate: anira/compat/v2.hpp alone as C++20 under the strict flags, compiled the way a
// 2.x consumer compiles it: anira's include directories and no anira define at all. The shim is
// exercised at compile time only (the value types' traits, the conversions as constant
// expressions, every member behind a branch that never runs), so a regression is a compiler
// error; nothing runs and nothing links. The gate includes the shim on purpose, and the C
// enumerators reach a consumer through it, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/compat/v2.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace v2 = anira::v2;

// The conversions round-trip every enumerator, at compile time.
static_assert(v2::to_backend(v2::to_engine(v2::LIBTORCH)) == v2::LIBTORCH);
static_assert(v2::to_backend(v2::to_engine(v2::ONNX)) == v2::ONNX);
static_assert(v2::to_backend(v2::to_engine(v2::TFLITE)) == v2::TFLITE);
static_assert(v2::to_backend(v2::to_engine(v2::LITERT)) == v2::LITERT);
static_assert(v2::to_backend(v2::to_engine(v2::EXECUTORCH)) == v2::EXECUTORCH);
static_assert(v2::to_backend(v2::to_engine(v2::CUSTOM)) == v2::CUSTOM);
static_assert(std::is_same_v<decltype(v2::to_engine(v2::CUSTOM)), anira::EngineRef>);
static_assert(v2::to_engine(v2::CUSTOM).kind == ANIRA_ENGINE_CUSTOM);
static_assert(v2::names(v2::to_engine(v2::CUSTOM), v2::CUSTOM));
static_assert(!v2::names(anira::EngineRef{.kind = ANIRA_ENGINE_CUSTOM, .id = "org.example.x"},
                         v2::CUSTOM));
static_assert(std::is_same_v<decltype(v2::TFLITE), v2::InferenceBackend>,
              "unscoped, as in 2.x: the enumerators are in the namespace");

// The value types: copyable as in 2.x, the loader's products owned.
static_assert(std::is_copy_constructible_v<v2::InferenceConfig>);
static_assert(std::is_copy_assignable_v<v2::InferenceConfig>);
static_assert(std::is_nothrow_move_constructible_v<v2::InferenceConfig>);
static_assert(std::is_copy_constructible_v<v2::ModelData>);
static_assert(std::is_copy_constructible_v<v2::TensorShape>);
static_assert(!std::is_default_constructible_v<v2::TensorShape>);
static_assert(std::is_default_constructible_v<v2::ProcessingSpec>);
static_assert(std::is_default_constructible_v<v2::HostConfig>);
static_assert(std::is_convertible_v<unsigned, v2::ContextConfig>, "ContextConfig(2) as in 2.x");
static_assert(!std::is_convertible_v<const anira::ContextConfig&, v2::ContextConfig>);
static_assert(
    std::is_same_v<decltype(std::declval<const v2::InferenceConfig&>().get_tensor_input_shape()),
                   const v2::TensorShapeList&>);
static_assert(std::is_same_v<decltype(std::declval<const v2::InferenceConfig&>()
                                          .get_tensor_output_shape(v2::TFLITE)),
                             const v2::TensorShapeList&>);
static_assert(
    std::is_same_v<decltype(std::declval<const v2::InferenceConfig&>().get_preprocess_input_size()),
                   const std::vector<size_t>&>);
static_assert(std::is_same_v<decltype(std::declval<v2::JsonConfigLoader&>().get_context_config()),
                             std::unique_ptr<v2::ContextConfig>>);
static_assert(std::is_same_v<decltype(std::declval<v2::JsonConfigLoader&>().get_inference_config()),
                             std::unique_ptr<v2::InferenceConfig>>);
static_assert(v2::default_log_drain() == v2::LogDrain::Thread ||
              v2::default_log_drain() == v2::LogDrain::Manual);

// The runtime half: views, the processor, the stage and the engine.
static_assert(std::is_nothrow_default_constructible_v<v2::RingBuffer>);
static_assert(std::is_trivially_copyable_v<v2::RingBuffer>);
static_assert(std::is_nothrow_default_constructible_v<v2::BufferF>);
static_assert(std::is_trivially_copyable_v<v2::BufferF>);
static_assert(!std::is_default_constructible_v<v2::PrePostProcessor>);
static_assert(!std::is_copy_constructible_v<v2::PrePostProcessor>);
static_assert(std::has_virtual_destructor_v<v2::PrePostProcessor>);
static_assert(std::is_base_of_v<anira::Stage, v2::LegacyProcessorStage>);
static_assert(std::is_base_of_v<anira::Stage::Prepared, v2::LegacyProcessorStage::Prepared>);
static_assert(std::is_base_of_v<anira::Engine, v2::PassthroughEngine>);
static_assert(std::is_default_constructible_v<v2::PassthroughEngine>);
static_assert(noexcept(std::declval<v2::PrePostProcessor&>().set_input(0.F, 0, 0)));
static_assert(noexcept(std::declval<const v2::PrePostProcessor&>().get_output(0, 0)));
static_assert(noexcept(v2::Context::shutdown()));

/// A 2.x processor overriding the four virtuals, as the extras' processors do.
class ProbeProcessor : public v2::PrePostProcessor {
public:
    explicit ProbeProcessor(v2::InferenceConfig& config) : v2::PrePostProcessor(config) {}
    void pre_process(std::vector<v2::RingBuffer>& input,
                     std::vector<v2::BufferF>& output,
                     v2::InferenceBackend backend) override {
        v2::PrePostProcessor::pre_process(input, output, backend);
        pop_samples_from_buffer(input[0], output[0], 1, 149, 0);
    }
    void post_process(std::vector<v2::BufferF>& input,
                      std::vector<v2::RingBuffer>& output,
                      v2::InferenceBackend backend) override {
        push_samples_to_buffer(input[0], output[0], input[0].get_num_samples());
        static_cast<void>(backend);
    }
    void before_inference(std::vector<v2::BufferF>& input,
                          v2::InferenceBackend /*backend*/) override {
        input[0].clear();
    }
    void after_inference(std::vector<v2::BufferF>& output,
                         v2::InferenceBackend /*backend*/) override {
        output[0].set_sample(0, 0, m_inference_config.m_blocking_ratio);
    }
};

/// What a phase of the processor reaches is nonblocking: under -Werror=function-effects (clang
/// 20 or later) a view method or a helper that allocates, locks or calls an unattributed function
/// is a compiler error here.
std::size_t nonblocking_probe(anira_ring* ring,
                              float* data,
                              v2::PrePostProcessor& processor) noexcept ANIRA_NONBLOCKING {
    v2::RingBuffer view(ring);
    v2::BufferF buffer(data, 4);
    std::size_t checks = view ? 1 : 0;
    checks += view.get_num_channels() + view.get_available_samples(0) +
              view.get_available_past_samples(0);
    view.push_sample(0, 1.F);
    checks += view.pop_sample(0) > 0.F ? 1 : 0;
    view.push_block(0, buffer.get_read_pointer(0), buffer.get_num_samples());
    view.pop_block(0, buffer.get_write_pointer(0), buffer.get_num_samples());
    view.peek_past_block(0, buffer.data(), 2);
    view.push_fill(0, 0.5F, 2);
    checks += view.discard(0, 1);
    checks += view.pop_windows(0, buffer.data(), ANIRA_DTYPE_F32, 1, 1, 0, 2);
    checks += view.dtype() == ANIRA_DTYPE_F32 && view.native() == ring ? 1 : 0;
    buffer.set_sample(0, 1, buffer.get_sample(0, 0));
    checks += buffer.get_num_channels() + (buffer.get_read_pointer(0, 1) != nullptr ? 1 : 0);
    checks += buffer.get_write_pointer(0, 2) != nullptr ? 1 : 0;
    checks += buffer.get_array_of_read_pointers() != nullptr ? 1 : 0;
    checks += buffer.get_array_of_write_pointers() != nullptr ? 1 : 0;
    buffer.clear();
    processor.set_input(1.F, 0, 0);
    processor.set_output(processor.get_input(0, 0), 0, 0);
    checks += processor.get_output(0, 0) > 0.F ? 1 : 0;
    processor.pop_samples_from_buffer(view, buffer, 2);
    processor.pop_samples_from_buffer(view, buffer, 1, 1);
    processor.pop_samples_from_buffer(view, buffer, 1, 1, 0);
    processor.pop_samples_from_buffer(view, buffer, 1, 1, 0, 2);
    processor.push_samples_to_buffer(buffer, view, 2);
    return checks;
}

}  // namespace

int anira_header_compat_probe();  // NOLINT(misc-use-internal-linkage)
int anira_header_compat_probe() {
    int checks = 0;
    if (false) {  // referenced so that it compiles; never run (no file, no C call)
        checks += v2::is_available(v2::ONNX) ? 1 : 0;
        checks += v2::names(anira::EngineRef{}, v2::LITERT) ? 1 : 0;
        checks += std::string(v2::to_string(v2::WaitStrategy::Blocking)).size() > 0 ? 1 : 0;
        checks += std::string(v2::to_string(v2::LogLevel::Debug)).size() > 0 ? 1 : 0;
        checks += std::string(v2::to_string(v2::LogDrain::Manual)).size() > 0 ? 1 : 0;

        v2::ContextConfig context(2, v2::WaitStrategy::Blocking, v2::LogLevel::Warning);
        context.m_log.m_drain = v2::LogDrain::Manual;
        const anira::ContextConfig minted = context.to_context_config();
        const v2::ContextConfig read(minted);
        checks += read.m_log == context.m_log ? 1 : 0;
        const v2::HostConfig host(512.F, 48000.F, true, 0, false);
        checks += host == v2::HostConfig() ? 0 : 1;

        std::vector<float> bytes(4);
        const v2::ModelData path("model.onnx", v2::ONNX);
        const v2::ModelData binary(bytes.data(),
                                   bytes.size() * sizeof(float),
                                   v2::LIBTORCH,
                                   "forward");
        checks += path == binary ? 0 : 1;
        const v2::TensorShape universal({{1, 1, 512}}, {{1, 1, 512}});
        const v2::TensorShape tflite({{1, 512, 1}}, {{1, 512, 1}}, v2::TFLITE);
        checks += universal.is_universal() && universal != tflite ? 1 : 0;
        const v2::ProcessingSpec spec({1}, {1}, {512}, {512}, {0});
        checks += spec == v2::ProcessingSpec({1}, {1}) ? 0 : 1;

        const v2::InferenceConfig config({path, binary}, {universal, tflite}, spec, 5.F);
        const v2::InferenceConfig defaults({path}, {universal}, 5.F, 1, true, 0.5F, 2);
        v2::InferenceConfig copy = config;
        copy = defaults;
        checks += copy == defaults && copy != config ? 1 : 0;
        checks += static_cast<int>(config.get_model_path(v2::ONNX).size());
        checks += config.get_model_data(v2::LIBTORCH) != nullptr ? 1 : 0;
        checks += static_cast<int>(config.get_model_function(v2::LIBTORCH).size());
        checks += config.is_model_binary(v2::LIBTORCH) ? 1 : 0;
        checks += static_cast<int>(config.get_tensor_input_shape().size() +
                                   config.get_tensor_output_shape().size() +
                                   config.get_tensor_input_shape(v2::TFLITE).size() +
                                   config.get_tensor_output_shape(v2::TFLITE).size());
        checks += static_cast<int>(config.get_tensor_input_size().size() +
                                   config.get_tensor_output_size().size() +
                                   config.get_preprocess_input_channels().size() +
                                   config.get_postprocess_output_channels().size() +
                                   config.get_preprocess_input_size().size() +
                                   config.get_postprocess_output_size().size() +
                                   config.get_internal_model_latency().size());
        checks += static_cast<int>(config.m_model_data.size() + config.m_tensor_shape.size());
        checks += config.m_max_inference_time > 0.F ? 1 : 0;
        checks += static_cast<int>(config.m_warm_up + config.m_num_parallel_processors);
        checks += config.m_session_exclusive_processor ? 1 : 0;
        checks += config.m_blocking_ratio > 0.F ? 1 : 0;
        checks += static_cast<int>(config.model_config().model_count());
        checks += static_cast<int>(config.model_config_copy().model_count());
        checks += static_cast<int>(config.hard().warmup_iterations);
        checks += static_cast<int>(v2::InferenceConfig::Defaults::num_parallel_processors());

        v2::JsonConfigLoader from_file("config.json");
        std::istringstream text("{}");
        v2::JsonConfigLoader from_stream(text);
        const std::unique_ptr<v2::ContextConfig> loaded_context = from_file.get_context_config();
        const std::unique_ptr<v2::InferenceConfig> loaded = from_stream.get_inference_config();
        checks += loaded_context != nullptr && loaded != nullptr ? 1 : 0;

        v2::InferenceConfig mutable_config = config;
        ProbeProcessor processor(mutable_config);
        checks += static_cast<int>(nonblocking_probe(nullptr, nullptr, processor));
        checks += &processor.config() == &mutable_config ? 1 : 0;
        const std::shared_ptr<v2::LegacyProcessorStage> stage =
            std::make_shared<v2::LegacyProcessorStage>(processor);
        checks += static_cast<int>(stage->phases() + stage->flags());
        const std::shared_ptr<anira::Engine> engine = std::make_shared<v2::PassthroughEngine>();
        checks += static_cast<int>(engine->flags() + engine->id().size());
        anira::Pipeline pipeline;
        pipeline.register_engine(engine);
        pipeline.inference(config.model_config());
        pipeline.add(anira::stage::Custom(stage));
        checks += v2::Context::shutdown() == ANIRA_OK ? 1 : 0;
        checks += v2::Context::release_core_if_idle() && v2::Context::has_core() ? 1 : 0;
        checks +=
            static_cast<int>(v2::Context::get_num_inference_threads() + v2::Context::drain_log());
    }
    return checks;
}
// NOLINTEND(misc-include-cleaner)
