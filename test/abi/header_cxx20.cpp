// Gate 4 (docs/anira-v3-architecture.md, section 6a): anira.hpp alone as C++20 under the
// strict flags, compiled the way a consumer compiles it: anira's include directories and
// no anira define at all. The wrapper is exercised at compile time only (the handle and
// aggregate traits, the ext<ext::Entry> members, the from_file loaders behind a branch
// that never runs), so a regression is a compiler error; nothing runs and nothing links.
// The gate includes anira.hpp on purpose, and the C enumerators reach a consumer through
// it, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// The ext<Ext> members, instantiated for the one extension kind of 3.0.
template anira::TensorSpec& anira::TensorSpec::ext<anira::ext::Entry>(const anira::ext::Entry&);
template anira::ContractHandle& anira::ContractHandle::ext<anira::ext::Entry>(
    const anira::ext::Entry&);
template anira::ModelConfig& anira::ModelConfig::model_ext<anira::ext::Entry>(
    uint32_t,
    const anira::ext::Entry&);
template anira::ModelConfig& anira::ModelConfig::ext<anira::ext::Entry>(const anira::ext::Entry&);
template anira::ContextConfig& anira::ContextConfig::ext<anira::ext::Entry>(
    const anira::ext::Entry&);
template anira::JobOptionsHandle& anira::JobOptionsHandle::ext<anira::ext::Entry>(
    const anira::ext::Entry&);

namespace {

/// A handle owns one C object: moved (without throwing), never copied.
template <class Handle>
constexpr bool k_move_only =
    std::is_nothrow_move_constructible_v<Handle> && std::is_nothrow_move_assignable_v<Handle> &&
    !std::is_copy_constructible_v<Handle> && !std::is_copy_assignable_v<Handle>;

static_assert(k_move_only<anira::TensorSpec>);
static_assert(k_move_only<anira::ModelConfig>);
static_assert(k_move_only<anira::ContextConfig>);
static_assert(k_move_only<anira::ContractHandle>);
static_assert(k_move_only<anira::JobOptionsHandle>);
static_assert(k_move_only<anira::Context>);
static_assert(k_move_only<anira::Pipeline>);

// The plan report is a view, copied freely; the inference stage takes one model config.
static_assert(std::is_copy_constructible_v<anira::PlanReport>);
static_assert(std::is_constructible_v<anira::stage::Inference, const anira::ModelConfig&>);
static_assert(std::is_same_v<anira::Pipeline::Stage, std::variant<anira::stage::Inference>>);

// The contract and job-option values are aggregates, spelled with designated initializers.
static_assert(std::is_aggregate_v<anira::Hard>);
static_assert(std::is_aggregate_v<anira::Async>);
static_assert(std::is_aggregate_v<anira::JobOptions>);
static_assert(std::is_same_v<anira::Contract, std::variant<anira::Hard, anira::Async>>);

// The runtime tensor and its token are the C structs with names on them: same size, no member
// added, trivially copyable, so a Tensor* is an anira_tensor*. The field fills and the reads
// are noexcept; only from_dlpack and SyncToken::dup can fail, and they throw.
static_assert(std::is_base_of_v<anira_tensor, anira::Tensor>);
static_assert(sizeof(anira::Tensor) == sizeof(anira_tensor) &&
              std::is_trivially_copyable_v<anira::Tensor> &&
              std::is_standard_layout_v<anira::Tensor>);
static_assert(std::is_base_of_v<anira_sync_token, anira::SyncToken>);
static_assert(sizeof(anira::SyncToken) == sizeof(anira_sync_token) &&
              std::is_trivially_copyable_v<anira::SyncToken>);
static_assert(
    noexcept(anira::Tensor::from_host(nullptr, ANIRA_DTYPE_F32, std::span<const int64_t>{})));
static_assert(noexcept(anira::Tensor::from_host_planar<float>(std::span<float* const>{},
                                                              std::span<const int64_t>{})));
static_assert(!noexcept(anira::Tensor::from_dlpack(nullptr)));
static_assert(noexcept(std::declval<const anira::Tensor&>().plane<const float>(0)));
static_assert(anira::detail::dtype_of<const int16_t>() == ANIRA_DTYPE_I16 &&
              anira::detail::dtype_of<bool>() == ANIRA_DTYPE_BOOL8);
static_assert(noexcept(std::declval<const anira::Tensor&>().data_f32()) &&
              noexcept(std::declval<const anira::Tensor&>().data(ANIRA_DTYPE_F32)) &&
              noexcept(std::declval<const anira::Tensor&>().num_elements()) &&
              noexcept(std::declval<const anira::Tensor&>().extent(0)));
static_assert(noexcept(std::declval<anira::SyncToken&>().reset()) &&
              !noexcept(std::declval<const anira::SyncToken&>().dup()));

// Every failure is an anira::Error, which a host catches as a std::exception.
static_assert(std::is_base_of_v<std::runtime_error, anira::Error>);
static_assert(std::is_same_v<decltype(anira::Error::status), anira_status>);

}  // namespace

// Exported on purpose, so that the TU holds a symbol a linker would see; the
// internal-linkage check is off for it.
int anira_header_cxx20_probe();  // NOLINT(misc-use-internal-linkage)
int anira_header_cxx20_probe() {
    // The aggregates as a consumer spells them; the defaults stand for what is not named.
    const anira::Hard hard{.block_min = 64, .block_max = 2048, .rate = 48000.0};
    const anira::Async async{.deadline = std::chrono::milliseconds(20), .on_late = ANIRA_LATE_DROP};
    const anira::Contract contract = hard;
    const anira::JobOptions options{.head_trim = {0, 0}, .tail_flush = false};
    int checks = 0;
    checks += std::holds_alternative<anira::Hard>(contract) ? 1 : 0;
    checks += async.deadline.has_value() ? 1 : 0;
    checks += options.head_trim.size() == 2 ? 1 : 0;
    if (false) {  // referenced so that it compiles; never run (no file, no C call)
        const std::filesystem::path path = "model.json";
        anira::ModelConfig model = anira::ModelConfig::from_file(path);
        anira::ContextConfig context = anira::ContextConfig::from_file(path);
        anira::ContractHandle loaded = anira::ContractHandle::from_file(path);
        anira::TensorSpec spec("x", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        anira::JobOptionsHandle job(options);
        spec.axis(0, ANIRA_AXIS_CHANNEL, 1)
            .axis(1, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
            .ext(anira::ext::Entry{.name = "x"});
        model.model_ext(0, anira::ext::Entry{.name = "decode"}).input(spec);
        context.ext(anira::ext::Entry{.name = "forward"});
        loaded.ext(anira::ext::Entry{.name = "forward"});
        job.ext(anira::ext::Entry{.name = "forward"});
        const anira::ContractHandle minted(contract);
        checks += model.upgraded() || context.upgraded() || loaded.upgraded() ? 1 : 0;
        checks += minted.native() != nullptr ? 1 : 0;
        anira::Context running(context);
        const anira::Capabilities capabilities = running.capabilities();
        const std::vector<anira::BackendId> backends = capabilities.backends();
        const anira::BackendId first = backends.empty() ? anira::BackendId{} : backends.front();
        anira::Pipeline pipe{anira::stage::Inference(model)};
        pipe.inference(model, {first});
        const anira::stage::Inference two({std::cref(model), std::cref(model)}, {first});
        pipe.add(two);
        const anira::PlanReport report(nullptr);
        const std::size_t report_rows = report.num_plans() + report.plans().size() +
                                        report.slots(0, true).size() + report.extensions(0).size();
        checks += report_rows > 0 ? 1 : 0;
        const anira_edge_info edge = capabilities.edge(ANIRA_DOMAIN_HOST, first);
        running.probe(true);
        const std::size_t rows = capabilities.domains().size() + capabilities.ext_kinds().size() +
                                 capabilities.edges().size() + anira::enabled_backends().size() +
                                 anira::drain_log();
        const uint64_t numbers = anira::num_inference_threads() + edge.available +
                                 running.byte_image_bytes(1, ANIRA_DTYPE_F32) + anira::now_ns();
        checks += rows > 0 ? 1 : 0;
        checks += numbers > 0 ? 1 : 0;
        checks += anira::now_ms() > 0.0 ? 1 : 0;
        checks += anira::shutdown() == ANIRA_OK ? 1 : 0;
        checks += anira::has_core() || anira::release_core_if_idle() ? 1 : 0;
        // The runtime tensor: every factory of anira/abi/tensor.h, the reads, the token.
        std::array<float, 4> samples{};
        const std::array<int64_t, 1> shape{4};
        anira::SyncToken fence{};
        anira::Tensor tensor = anira::Tensor::from_host(samples.data(), ANIRA_DTYPE_F32, shape);
        const anira_tensor* const record = &tensor;
        const std::size_t elements = tensor.num_elements() + tensor.extent(0) + record->ndim;
        checks += elements > 0 && tensor.data_f32() != nullptr ? 1 : 0;
        checks += tensor.data(ANIRA_DTYPE_I16) == nullptr ? 1 : 0;
        const std::array<float*, 1> planes{samples.data()};
        const std::array<int64_t, 2> block{1, 4};
        tensor = anira::Tensor::from_host_planar<float>(planes, block);
        checks += tensor.plane<float>(0) == samples.data() ? 1 : 0;
        tensor = anira::Tensor::from_pinned(samples.data(), ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_cuda(nullptr, 0, nullptr, ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_gl_buffer(0, 0, nullptr, ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_vulkan(0, 0, 0, 0, 0, ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_opaque_fd(-1, 0, ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_wgpu_buffer(nullptr, 0, &fence, ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_dmabuf(-1, 0, 0, -1, ANIRA_DTYPE_F32, shape);
        tensor = anira::Tensor::from_dlpack(nullptr);
        fence = fence.dup();
        fence.reset();
    }
    return checks;
}
// NOLINTEND(misc-include-cleaner)
