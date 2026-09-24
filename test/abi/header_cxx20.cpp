// Gate 4 (docs/anira-v3-architecture.md, section 6a): anira.hpp alone as C++20 under the
// strict flags, compiled the way a consumer compiles it: anira's include directories and
// no anira define at all. The wrapper is exercised at compile time only (the handle and
// aggregate traits, the ext<ext::Entry> members, the from_file loaders behind a branch
// that never runs), so a regression is a compiler error; nothing runs and nothing links.
// The gate includes anira.hpp on purpose, and the C enumerators reach a consumer through
// it, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
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
// The typed RingView calls, instantiated for an int16 ring and, where a call only reads its
// span, for a span of const elements.
template std::size_t anira::RingView::pop_block<int16_t>(uint32_t, std::span<int16_t>) noexcept;
template std::size_t anira::RingView::peek_past_block<int16_t>(uint32_t,
                                                               std::span<int16_t>) const noexcept;
template std::size_t anira::RingView::push_block<int16_t>(uint32_t, std::span<int16_t>) noexcept;
template std::size_t anira::RingView::push_block<const float>(uint32_t,
                                                              std::span<const float>) noexcept;
template std::size_t anira::RingView::push_fill<int16_t>(uint32_t,
                                                         const int16_t&,
                                                         std::size_t) noexcept;
template std::size_t anira::RingView::pop_windows<int16_t>(uint32_t,
                                                           std::span<int16_t>,
                                                           std::size_t,
                                                           std::size_t,
                                                           std::size_t,
                                                           uint32_t) noexcept;

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
static_assert(std::is_same_v<anira::Pipeline::AnyStage,
                             std::variant<anira::stage::Inference, anira::stage::Custom>>);
static_assert(std::is_constructible_v<anira::stage::Custom, std::shared_ptr<anira::Stage>> &&
              std::is_copy_constructible_v<anira::stage::Custom>);

// A stage is a class a consumer subclasses: abstract (phases() and prepare() are the
// subclass's statements), never copied or moved (the pipeline's carrier shares it), destroyed
// through the base. Its Prepared, what prepare returns per handler, is concrete (every phase
// has a base body), never copied or moved (anira owns it), destroyed through the base by anira.
static_assert(std::is_abstract_v<anira::Stage> && std::has_virtual_destructor_v<anira::Stage> &&
              !std::is_copy_constructible_v<anira::Stage> &&
              !std::is_move_constructible_v<anira::Stage>);
static_assert(!std::is_abstract_v<anira::Stage::Prepared> &&
              std::has_virtual_destructor_v<anira::Stage::Prepared> &&
              !std::is_copy_constructible_v<anira::Stage::Prepared> &&
              !std::is_move_constructible_v<anira::Stage::Prepared>);
static_assert(std::is_same_v<decltype(std::declval<anira::Stage&>().prepare(
                                 std::declval<const anira::PrepareInfo&>())),
                             std::unique_ptr<anira::Stage::Prepared>>);
static_assert(std::is_same_v<
              decltype(std::declval<anira::Stage&>().init(std::declval<const anira::InitInfo&>())),
              void>);
// The two shared records (anira/abi/lifecycle.h) are views over the C records: trivially
// copyable, no default. The init record: the level, the pool, the context, all noexcept. The
// prepare record: the templates as a span of Tensor, the report as a PlanReport, the flags and
// whether the handler is exclusive.
static_assert(std::is_trivially_copyable_v<anira::InitInfo> &&
              !std::is_default_constructible_v<anira::InitInfo>);
static_assert(
    std::is_same_v<decltype(std::declval<const anira::InitInfo&>().log_level()), anira_log_level> &&
    std::is_same_v<decltype(std::declval<const anira::InitInfo&>().num_threads()), uint32_t> &&
    std::is_same_v<decltype(std::declval<const anira::InitInfo&>().context()),
                   const anira_context*> &&
    noexcept(std::declval<const anira::InitInfo&>().log_level()) &&
    noexcept(std::declval<const anira::InitInfo&>().num_threads()) &&
    noexcept(std::declval<const anira::InitInfo&>().context()));
static_assert(std::is_trivially_copyable_v<anira::PrepareInfo> &&
              !std::is_default_constructible_v<anira::PrepareInfo>);
static_assert(
    std::is_same_v<decltype(std::declval<const anira::PrepareInfo&>().inputs()),
                   std::span<const anira::Tensor>> &&
    std::is_same_v<decltype(std::declval<const anira::PrepareInfo&>().report()),
                   anira::PlanReport> &&
    std::is_same_v<decltype(std::declval<const anira::PrepareInfo&>().exclusive()), bool> &&
    noexcept(std::declval<const anira::PrepareInfo&>().num_entries()) &&
    noexcept(std::declval<const anira::PrepareInfo&>().input_names()) &&
    noexcept(std::declval<const anira::PrepareInfo&>().flags()) &&
    noexcept(std::declval<const anira::PrepareInfo&>().exclusive()));
static_assert(anira::Stage::k_pre_process == 1U && anira::Stage::k_post_process == 2U &&
              anira::Stage::k_before_inference == 4U && anira::Stage::k_after_inference == 16U &&
              anira::Stage::phase_bit(ANIRA_PHASE_AFTER_INFERENCE) ==
                  anira::Stage::k_after_inference);
// What a phase callback reaches is nonblocking: the context and the ring view are trivially
// copyable views, and every method is noexcept (a status or a count, never a throw).
static_assert(std::is_trivially_copyable_v<anira::RingView> &&
              std::is_nothrow_default_constructible_v<anira::RingView> &&
              !std::is_convertible_v<anira::RingView, bool> &&
              std::is_constructible_v<bool, anira::RingView>);
static_assert(std::is_trivially_copyable_v<anira::StageContext> &&
              !std::is_default_constructible_v<anira::StageContext>);
static_assert(noexcept(std::declval<const anira::RingView&>().dtype()) &&
              noexcept(std::declval<const anira::RingView&>().num_channels()) &&
              noexcept(std::declval<const anira::RingView&>().available(0)) &&
              noexcept(std::declval<const anira::RingView&>().available_past(0)) &&
              noexcept(std::declval<anira::RingView&>().discard(0, 0)) &&
              noexcept(std::declval<anira::RingView&>().pop_block(0, std::span<float>{})) &&
              noexcept(std::declval<anira::RingView&>().push_fill(0, 0.0F, 0)));
static_assert(noexcept(std::declval<const anira::StageContext&>().phase()) &&
              noexcept(std::declval<const anira::StageContext&>()
                           .input_role(0, std::declval<anira::Role&>())) &&
              noexcept(std::declval<const anira::StageContext&>()
                           .input_ring(0, std::declval<anira::RingView&>())) &&
              noexcept(std::declval<const anira::StageContext&>()
                           .output_ring(0, std::declval<anira::RingView&>())) &&
              noexcept(std::declval<const anira::StageContext&>()
                           .input_tensor(0, std::declval<anira::Tensor&>())) &&
              noexcept(std::declval<const anira::StageContext&>()
                           .output_tensor(0, std::declval<anira::Tensor&>())));
// Every context accessor returns the C status and fills an out-parameter; the view is
// copy-assignable, so a refused ring leaves a view of no ring.
static_assert(std::is_same_v<decltype(std::declval<const anira::StageContext&>()
                                          .input_ring(0, std::declval<anira::RingView&>())),
                             anira_status>);
static_assert(std::is_same_v<decltype(std::declval<const anira::StageContext&>()
                                          .output_role(0, std::declval<anira::Role&>())),
                             anira_status>);
static_assert(
    std::is_nothrow_copy_assignable_v<anira::RingView> &&
    std::is_same_v<decltype(std::declval<const anira::StageContext&>().ticket()), anira_ticket>);
// Every per-chunk virtual of the Prepared is noexcept, prepare of the registration is not.
static_assert(
    noexcept(std::declval<anira::Stage::Prepared&>().pre_process(
        std::declval<anira::StageContext&>())) &&
    noexcept(std::declval<anira::Stage::Prepared&>().post_process(
        std::declval<anira::StageContext&>())) &&
    noexcept(std::declval<anira::Stage::Prepared&>().before_inference(
        std::declval<anira::StageContext&>())) &&
    noexcept(std::declval<anira::Stage::Prepared&>().after_inference(
        std::declval<anira::StageContext&>())) &&
    noexcept(std::declval<anira::Stage::Prepared&>().reset(std::declval<anira::StageContext&>())) &&
    noexcept(std::declval<anira::Stage&>().release()) &&
    noexcept(std::declval<const anira::Stage&>().consumed_kinds()) &&
    noexcept(std::declval<const anira::Stage&>().flags()) &&
    noexcept(std::declval<const anira::StageContext&>().entry()) &&
    !noexcept(std::declval<anira::Stage&>().prepare(std::declval<const anira::PrepareInfo&>())) &&
    !noexcept(std::declval<anira::Stage&>().init(std::declval<const anira::InitInfo&>())));

// The enum alias of anira_engine is EngineKind; the name Engine is the class of a custom
// engine: the registration, abstract (load is the subclass's statement), never copied or
// moved (the pipeline's carrier shares it), destroyed through the base. Its Loaded, what load
// returns per loaded model, is abstract too (prepare is its statement), as is its Prepared,
// what the Loaded's prepare returns per handler (process is the engine's statement); neither
// is copied or moved (anira owns them), both are destroyed through the base by anira.
static_assert(std::is_same_v<anira::EngineKind, anira_engine>);
static_assert(std::is_abstract_v<anira::Engine> && std::has_virtual_destructor_v<anira::Engine> &&
              !std::is_copy_constructible_v<anira::Engine> &&
              !std::is_move_constructible_v<anira::Engine>);
static_assert(std::is_abstract_v<anira::Engine::Loaded> &&
              std::has_virtual_destructor_v<anira::Engine::Loaded> &&
              !std::is_copy_constructible_v<anira::Engine::Loaded> &&
              !std::is_move_constructible_v<anira::Engine::Loaded>);
static_assert(std::is_abstract_v<anira::Engine::Prepared> &&
              std::has_virtual_destructor_v<anira::Engine::Prepared> &&
              !std::is_copy_constructible_v<anira::Engine::Prepared> &&
              !std::is_move_constructible_v<anira::Engine::Prepared>);
static_assert(std::is_same_v<decltype(std::declval<anira::Engine&>().load(
                                 std::declval<const anira::EngineLoadInfo&>())),
                             std::unique_ptr<anira::Engine::Loaded>>);
static_assert(std::is_same_v<decltype(std::declval<anira::Engine::Loaded&>().prepare(
                                 std::declval<const anira::PrepareInfo&>())),
                             std::unique_ptr<anira::Engine::Prepared>>);
static_assert(std::is_same_v<
              decltype(std::declval<anira::Engine&>().init(std::declval<const anira::InitInfo&>())),
              void>);
// The load record and the context are views over the C records: trivially copyable, no
// default; the templates and the descriptors as spans of Tensor (the outputs writable), the
// row's facts through the config getters as views and an EngineKind; the context's loaded
// pointer and whether the call is an exclusive handler's.
static_assert(std::is_trivially_copyable_v<anira::EngineLoadInfo> &&
              !std::is_default_constructible_v<anira::EngineLoadInfo>);
static_assert(std::is_same_v<decltype(std::declval<const anira::EngineLoadInfo&>().inputs()),
                             std::span<const anira::Tensor>> &&
              std::is_same_v<decltype(std::declval<const anira::EngineLoadInfo&>().model_bytes(0)),
                             std::span<const std::byte>> &&
              std::is_same_v<decltype(std::declval<const anira::EngineLoadInfo&>().model_engine(0)),
                             anira::EngineKind> &&
              std::is_same_v<decltype(std::declval<const anira::EngineLoadInfo&>().model_path(0)),
                             std::string_view> &&
              std::is_same_v<decltype(std::declval<const anira::EngineLoadInfo&>().model()),
                             const anira_model_config*> &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().row()) &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().instances()) &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().model_count()) &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().model_engine_id(0)) &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().model_bytes(0)) &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().input_names()) &&
              noexcept(std::declval<const anira::EngineLoadInfo&>().output_names()));
static_assert(std::is_trivially_copyable_v<anira::EngineContext> &&
              !std::is_default_constructible_v<anira::EngineContext>);
static_assert(
    std::is_same_v<decltype(std::declval<const anira::EngineContext&>().inputs()),
                   std::span<const anira::Tensor>> &&
    std::is_same_v<decltype(std::declval<const anira::EngineContext&>().outputs()),
                   std::span<anira::Tensor>> &&
    std::is_same_v<decltype(std::declval<const anira::EngineContext&>().ticket()), anira_ticket> &&
    std::is_same_v<decltype(std::declval<const anira::EngineContext&>().loaded()), void*> &&
    std::is_same_v<decltype(std::declval<const anira::EngineContext&>().exclusive()), bool> &&
    noexcept(std::declval<const anira::EngineContext&>().instance()) &&
    noexcept(std::declval<const anira::EngineContext&>().entry()) &&
    noexcept(std::declval<const anira::EngineContext&>().ticket()) &&
    noexcept(std::declval<const anira::EngineContext&>().flags()) &&
    noexcept(std::declval<const anira::EngineContext&>().loaded()) &&
    noexcept(std::declval<const anira::EngineContext&>().exclusive()) &&
    noexcept(std::declval<const anira::EngineContext&>().inputs()) &&
    noexcept(std::declval<const anira::EngineContext&>().outputs()));
// Every per-inference virtual of the Prepared is noexcept; init and load of the registration
// and prepare of the Loaded are not.
static_assert(
    noexcept(
        std::declval<anira::Engine::Prepared&>().process(std::declval<anira::EngineContext&>())) &&
    noexcept(
        std::declval<anira::Engine::Prepared&>().reset(std::declval<anira::EngineContext&>())) &&
    noexcept(std::declval<anira::Engine&>().release()) &&
    noexcept(std::declval<const anira::Engine&>().consumed_kinds()) &&
    noexcept(std::declval<const anira::Engine&>().flags()) &&
    !noexcept(std::declval<anira::Engine&>().init(std::declval<const anira::InitInfo&>())) &&
    !noexcept(std::declval<anira::Engine&>().load(std::declval<const anira::EngineLoadInfo&>())) &&
    !noexcept(
        std::declval<anira::Engine::Loaded&>().prepare(std::declval<const anira::PrepareInfo&>())));
// The two registration spellings: on the pipeline and on the inference stage, both taking the
// shared engine object alone, whose id is its own (the constructor's, read back by id()); the
// stage lists what it brought.
static_assert(
    std::is_same_v<decltype(&anira::Pipeline::register_engine),
                   anira::Pipeline& (anira::Pipeline::*)(const std::shared_ptr<anira::Engine>&)>);
static_assert(std::is_same_v<decltype(&anira::stage::Inference::engine),
                             anira::stage::Inference& (
                                 anira::stage::Inference::*)(std::shared_ptr<anira::Engine>)>);
static_assert(std::is_same_v<decltype(std::declval<const anira::stage::Inference&>().engines()),
                             std::span<const std::shared_ptr<anira::Engine>>>);
static_assert(
    std::is_same_v<decltype(std::declval<const anira::Engine&>().id()), const std::string&>);
static_assert(noexcept(std::declval<const anira::Engine&>().id()));

// The contract and job-option values are aggregates, spelled with designated initializers.
static_assert(std::is_aggregate_v<anira::Hard>);
static_assert(std::is_aggregate_v<anira::Async>);
static_assert(std::is_aggregate_v<anira::JobOptions>);
static_assert(std::is_same_v<anira::Contract, std::variant<anira::Hard, anira::Async>>);
// The backup function of ANIRA_MISS_CALLBACK is the C typedef, a raw function pointer, on the
// aggregate and on the handle's setter.
static_assert(std::is_same_v<decltype(anira::Hard::miss_fn), anira_miss_fn>);
static_assert(std::is_same_v<decltype(anira::Hard::miss_user_data), void*>);
static_assert(
    std::is_same_v<decltype(&anira::ContractHandle::hard_miss_fn),
                   anira::ContractHandle& (anira::ContractHandle::*)(anira_miss_fn, void*)>);
// The Static entries reach a C++ consumer through anira.hpp until the handler class arrives: a
// slot and a whole anira::Tensor, which is an anira_tensor.
static_assert(noexcept(anira_handler_set_static_input(nullptr, 0, nullptr)));
static_assert(noexcept(anira_handler_get_static_output(nullptr, 0, nullptr)));
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_handler_set_static_input),
                                    anira_handler*,
                                    uint32_t,
                                    const anira::Tensor*>);
// Declared state: the fourth role, and the pairing stated once, on the input, through
// TensorSpec::state_source over the C setter.
static_assert(std::is_same_v<anira::Role, decltype(ANIRA_ROLE_STATE)>);
static_assert(ANIRA_ROLE_STATE == 3);
static_assert(noexcept(anira_tensor_spec_set_state_source(nullptr, nullptr)));
static_assert(std::is_same_v<decltype(&anira::TensorSpec::state_source),
                             anira::TensorSpec& (anira::TensorSpec::*)(std::string_view)>);

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

/// A stage as a consumer writes it: the registration states the phases it fills and its
/// promise, and prepare returns the Prepared of one handler, whose overrides are noexcept. It
/// converts an int16 ring into the float model tensor and leaves the push to anira's default.
class ProbePrepared final : public anira::Stage::Prepared {
public:
    explicit ProbePrepared(uint32_t num_entries) : m_num_entries(num_entries) {}

    anira_status pre_process(anira::StageContext& ctx) noexcept override {
        if (ctx.entry() >= m_num_entries) { return ANIRA_ERROR_INTERNAL; }
        anira::Tensor input{};
        const anira_status exposed = ctx.input_tensor(0, input);
        if (exposed != ANIRA_OK) { return exposed; }
        anira::RingView ring;
        const anira_status has_ring = ctx.input_ring(0, ring);
        if (has_ring != ANIRA_OK) { return has_ring; }
        if (ring.dtype() != ANIRA_DTYPE_I16) { return ANIRA_ERROR_CONFIG; }
        float* const samples = input.data_f32();
        const std::size_t hop = std::min(m_block.size(), input.num_elements());
        const std::size_t popped = ring.pop_block(0, std::span<int16_t>(m_block).first(hop));
        for (std::size_t n = 0; n < popped; ++n) { samples[n] = static_cast<float>(m_block[n]); }
        return ANIRA_OK;
    }
    anira_status after_inference(anira::StageContext& ctx) noexcept override {
        anira::Role role = ANIRA_ROLE_FORCE32;
        anira::Tensor output{};
        const anira_status asked = ctx.output_role(0, role);
        if (asked != ANIRA_OK) { return asked; }
        return role == ANIRA_ROLE_STREAMED ? ctx.output_tensor(0, output) : ANIRA_ERROR_CONFIG;
    }
    // The first chunk of a new stream: the block this object keeps starts over.
    void reset(anira::StageContext& /*ctx*/) noexcept override { m_block.fill(0); }

private:
    std::array<int16_t, 64> m_block{};
    uint32_t m_num_entries;
};

class ProbeStage final : public anira::Stage {
public:
    uint32_t phases() const noexcept override { return k_pre_process | k_after_inference; }
    // The real-time promise a Hard contract requires of a filled pre_process.
    uint32_t flags() const noexcept override { return ANIRA_STAGE_FLAG_REALTIME_PRE_POST; }

    void init(const anira::InitInfo& info) override {
        if (info.context() == nullptr || info.log_level() > ANIRA_LOG_ERROR) {
            throw anira::Error(ANIRA_ERROR_CONFIG, "nothing to init");
        }
        m_num_threads = info.num_threads();
    }

    std::unique_ptr<anira::Stage::Prepared> prepare(const anira::PrepareInfo& info) override {
        if (info.handler() == nullptr || info.report().num_plans() == 0 ||
            info.inputs().size() != info.input_names().size() || info.exclusive() ||
            info.flags() != 0) {
            throw anira::Error(ANIRA_ERROR_CONFIG, "nothing to prepare");
        }
        return std::make_unique<ProbePrepared>(info.num_entries());
    }

private:
    uint32_t m_num_threads = 0;
};

/// An engine as a consumer writes it: the registration states its promise, load returns the
/// Loaded of one loaded model (the row's facts come through the record), and the Loaded's
/// prepare returns the Prepared of one handler, whose process copies the first input into the
/// first output.
class ProbeEnginePrepared final : public anira::Engine::Prepared {
public:
    explicit ProbeEnginePrepared(std::size_t elements) : m_elements(elements) {}

    anira_status process(anira::EngineContext& ctx) noexcept override {
        if (ctx.inputs().empty() || ctx.outputs().empty() || ctx.loaded() == nullptr ||
            (ctx.exclusive() && ctx.instance() != 0)) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        const float* const in = ctx.inputs()[0].data_f32();
        float* const out = ctx.outputs()[0].data_f32();
        if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        const std::size_t count = std::min(m_elements, ctx.outputs()[0].num_elements());
        for (std::size_t n = 0; n < count; ++n) { out[n] = in[n]; }
        return ANIRA_OK;
    }
    void reset(anira::EngineContext& /*ctx*/) noexcept override {}

private:
    std::size_t m_elements;
};

class ProbeLoaded final : public anira::Engine::Loaded {
public:
    explicit ProbeLoaded(std::size_t elements) : m_elements(elements) {}

    std::unique_ptr<anira::Engine::Prepared> prepare(const anira::PrepareInfo& info) override {
        if (info.handler() == nullptr || info.inputs().size() != info.input_names().size()) {
            throw anira::Error(ANIRA_ERROR_CONFIG, "nothing to prepare");
        }
        // An exclusive handler runs on what this Prepared builds; a shared one on the Loaded's
        // instances. The probe's Prepared holds nothing either way.
        return std::make_unique<ProbeEnginePrepared>(m_elements);
    }

private:
    std::size_t m_elements;
};

class ProbeEngine final : public anira::Engine {
public:
    ProbeEngine() : anira::Engine("org.example.probe") {}
    uint32_t flags() const noexcept override { return ANIRA_ENGINE_FLAG_REALTIME_SAFE; }
    // The query: every declared provider but the first is usable here.
    std::uint64_t available(const anira::InitInfo& info) const override {
        return info.num_threads() > 0 ? ~std::uint64_t{1} : 0;
    }

    void init(const anira::InitInfo& info) override {
        if (info.context() == nullptr) {
            throw anira::Error(ANIRA_ERROR_CONFIG, "nothing to init");
        }
    }

    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& info) override {
        if (info.model() == nullptr || info.row() >= info.model_count() ||
            info.model_engine(info.row()) != ANIRA_ENGINE_NONE ||
            info.model_engine_id(info.row()).empty() || info.inputs().empty() ||
            info.inputs().size() != info.input_names().size() ||
            info.outputs().size() != info.output_names().size() ||
            info.option_keys().size() != info.option_values().size() ||
            (info.model_path(info.row()).empty() && info.model_bytes(info.row()).empty())) {
            throw anira::Error(ANIRA_ERROR_CONFIG, "nothing to load");
        }
        // instances() is 0 for a model whose handlers are exclusive: nothing shared to build.
        return std::make_unique<ProbeLoaded>(info.inputs()[0].num_elements());
    }
};

/// Every read an engine's process can make on the context and its tensors, from a nonblocking
/// function: what a body that keeps ANIRA_ENGINE_FLAG_REALTIME_SAFE is composed of.
std::size_t engine_probe(const anira_engine_ctx* record) noexcept ANIRA_NONBLOCKING {
    const anira::EngineContext ctx(record);
    std::size_t seen = ctx.instance() + ctx.entry() + ctx.ticket() + ctx.flags();
    seen += ctx.inputs().size() + ctx.outputs().size();
    for (const anira::Tensor& input : ctx.inputs()) {
        seen += input.num_elements() + (input.data_f32() != nullptr ? 1 : 0);
    }
    for (const anira::Tensor& output : ctx.outputs()) {
        float* const samples = output.data_f32();
        if (samples != nullptr && output.num_elements() > 0) { samples[0] = 0.0F; }
        seen += output.extent(0);
    }
    seen += ctx.native() != nullptr ? 1 : 0;
    return seen;
}

/// Every call a phase callback can make on the two views, from a nonblocking function: where
/// the compiler has the analysis (-Werror=function-effects, see CMakeLists.txt) a method that
/// allocates, locks or calls a function that is not nonblocking is a compiler error here.
std::size_t nonblocking_probe(const anira_stage_ctx* record) noexcept ANIRA_NONBLOCKING {
    const anira::StageContext ctx(record);
    anira::Tensor tensor{};
    anira::Role role = ANIRA_ROLE_FORCE32;
    std::array<float, 8> block{};
    const float fill = 0.0F;
    anira::RingView in;
    anira::RingView out;
    std::size_t moved = ctx.input_ring(0, in) == ANIRA_OK ? 1 : 0;
    moved += ctx.output_ring(0, out) == ANIRA_OK ? 1 : 0;
    moved += in.pop_block(0, std::span<float>(block)) +
             in.peek_past_block(0, std::span<float>(block)) +
             in.pop_windows(0, std::span<float>(block), 2, 2, 0, 2) + in.discard(0, 1) +
             out.push_block(0, std::span<const float>(block)) + out.push_fill(0, fill, 1) +
             in.available(0) + in.available_past(0) + in.num_channels() + in.dtype();
    moved += ctx.num_inputs() + ctx.num_outputs() + ctx.variant() + ctx.ticket() + ctx.phase() +
             ctx.engine() + ctx.provider() + ctx.entry();
    moved += ctx.input_role(0, role) == ANIRA_OK ? 1 : 0;
    moved += ctx.output_role(0, role) == ANIRA_OK ? 1 : 0;
    moved += role;
    moved += ctx.input_tensor(0, tensor) == ANIRA_OK ? 1 : 0;
    moved += ctx.output_tensor(0, tensor) == ANIRA_OK ? 1 : 0;
    moved += static_cast<bool>(in) && out.native() != nullptr && ctx.native() != nullptr ? 1 : 0;
    return moved;
}

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
        loaded.hard_on_miss(ANIRA_MISS_CALLBACK).hard_miss_fn(nullptr, nullptr);
        loaded.host_domain("x", ANIRA_DOMAIN_HOST);
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
        // A custom stage beside the inference stage, in the initializer list of one pipeline
        // and through add on another (a pipeline holds at most one).
        const anira::stage::Custom custom(std::make_shared<ProbeStage>());
        const anira::Pipeline staged{anira::stage::Inference(model), custom};
        anira::Pipeline other{anira::stage::Inference(model)};
        other.add(custom);
        checks += custom.stage() != nullptr && custom.stage()->flags() != 0 ? 1 : 0;
        checks += nonblocking_probe(nullptr) > 0 ? 1 : 0;
        // A custom engine registered on the pipeline, and brought along by the inference
        // stage in an initializer list; the enum alias where the model config takes one.
        const std::shared_ptr<anira::Engine> probe_engine = std::make_shared<ProbeEngine>();
        other.register_engine(probe_engine);
        // The pipeline's capabilities on a context: the rows, an edge.
        const anira::PipelineCapabilities caps =
            other.capabilities(static_cast<const anira_context*>(nullptr));
        checks += caps.backends().empty() ? 0 : 1;
        checks += caps.edge(ANIRA_DOMAIN_HOST, caps.backends().front()).available == 1U ? 1 : 0;
        const anira::Pipeline with_engine{anira::stage::Inference(model).engine(probe_engine)};
        const anira::stage::Inference brings(model);
        checks += brings.engines().empty() ? 1 : 0;
        const anira::EngineKind kind = model.model_engine(0);
        model.default_engine(kind).add_model_path("org.example.probe", path);
        checks += engine_probe(nullptr) > 0 ? 1 : 0;
        anira::TensorSpec state("state_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
        state.axis(0, ANIRA_AXIS_ANY, 2).state_source("state_out");
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
