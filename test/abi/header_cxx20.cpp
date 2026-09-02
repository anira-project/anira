// Gate 4 (docs/anira-v3-architecture.md, section 6a): anira.hpp alone as C++20 under the
// strict flags, compiled the way a consumer compiles it: anira's include directories and
// no anira define at all. The wrapper is exercised at compile time only (the handle and
// aggregate traits, the ext<ext::Entry> members, the from_file loaders behind a branch
// that never runs), so a regression is a compiler error; nothing runs and nothing links.
// The gate includes anira.hpp on purpose, and the C enumerators reach a consumer through
// it, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/anira.hpp>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <variant>

// The ext<Ext> members, instantiated for the one extension kind of 3.0.
template anira::TensorSpec& anira::TensorSpec::ext<anira::ext::Entry>(const anira::ext::Entry&);
template anira::ContractHandle& anira::ContractHandle::ext<anira::ext::Entry>(
    const anira::ext::Entry&);
template anira::ModelConfig& anira::ModelConfig::model_ext<anira::ext::Entry>(
    uint32_t,
    const anira::ext::Entry&);
template anira::ModelConfig& anira::ModelConfig::ext<anira::ext::Entry>(const anira::ext::Entry&);
template anira::MachineConfig& anira::MachineConfig::ext<anira::ext::Entry>(
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
static_assert(k_move_only<anira::MachineConfig>);
static_assert(k_move_only<anira::ContractHandle>);
static_assert(k_move_only<anira::JobOptionsHandle>);

// The contract and job-option values are aggregates, spelled with designated initializers.
static_assert(std::is_aggregate_v<anira::Hard>);
static_assert(std::is_aggregate_v<anira::Async>);
static_assert(std::is_aggregate_v<anira::JobOptions>);
static_assert(std::is_same_v<anira::Contract, std::variant<anira::Hard, anira::Async>>);

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
        anira::MachineConfig machine = anira::MachineConfig::from_file(path);
        anira::ContractHandle loaded = anira::ContractHandle::from_file(path);
        anira::TensorSpec spec("x", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        anira::JobOptionsHandle job(options);
        spec.axis(0, ANIRA_AXIS_CHANNEL, 1)
            .axis(1, ANIRA_AXIS_TIME, ANIRA_DYNAMIC)
            .ext(anira::ext::Entry{.name = "x"});
        model.model_ext(0, anira::ext::Entry{.name = "decode"}).input(spec);
        machine.ext(anira::ext::Entry{.name = "forward"});
        loaded.ext(anira::ext::Entry{.name = "forward"});
        job.ext(anira::ext::Entry{.name = "forward"});
        const anira::ContractHandle minted(contract);
        checks += model.upgraded() || machine.upgraded() || loaded.upgraded() ? 1 : 0;
        checks += minted.native() != nullptr ? 1 : 0;
    }
    return checks;
}
// NOLINTEND(misc-include-cleaner)
