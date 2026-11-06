#ifndef ANIRA_TEST_FILE_CONTRACTS_H
#define ANIRA_TEST_FILE_CONTRACTS_H

// The bundled contract files (extras/models/**/*.contract.json) run the library's defaults, a
// measured budget and the UNTIL_STABLE warm-up, so a latency they give depends on the machine.
// The suites that assert a latency, a count or a stream over a file pin the file's 2.x figures,
// the explicit budget and the fixed warm-up it spelled before the measured budget, from this
// one table: every asserted latency stays, and the goldens stay byte-identical.
// test_ExtrasFixtures checks the table against the 2.x header literals, so it cannot drift.

#include <anira/abi/enums.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <string_view>

#include "../../extras/models/model_files.h"

namespace anira_test {

/// One bundled contract file's 2.x figures: the explicit budget in ms and the fixed warm-up
/// count (0: none).
struct FileFigures {
    const char* m_contract_json = nullptr;
    double m_budget_ms = 0.0;
    uint32_t m_warm_up = 0;
};

inline constexpr std::array<FileFigures, 11> k_file_figures{{
    {.m_contract_json = k_cnn_contract_json, .m_budget_ms = 42.66, .m_warm_up = 2},
    {.m_contract_json = k_medium_cnn_contract_json, .m_budget_ms = 42.66, .m_warm_up = 0},
    {.m_contract_json = k_small_cnn_contract_json, .m_budget_ms = 42.66, .m_warm_up = 0},
    {.m_contract_json = k_hybridnn_contract_json, .m_budget_ms = 5.33, .m_warm_up = 3},
    {.m_contract_json = k_rnn_contract_json, .m_budget_ms = 42.66, .m_warm_up = 2},
    {.m_contract_json = k_gain_contract_json, .m_budget_ms = 5.0, .m_warm_up = 1},
    {.m_contract_json = k_stereo_gain_contract_json, .m_budget_ms = 5.0, .m_warm_up = 1},
    {.m_contract_json = k_stateful_accumulator_contract_json, .m_budget_ms = 5.0, .m_warm_up = 1},
    {.m_contract_json = k_rave_funk_drum_contract_json, .m_budget_ms = 42.66, .m_warm_up = 5},
    {.m_contract_json = k_rave_funk_drum_encoder_contract_json,
     .m_budget_ms = 42.66,
     .m_warm_up = 5},
    {.m_contract_json = k_rave_funk_drum_decoder_contract_json,
     .m_budget_ms = 42.66,
     .m_warm_up = 5},
}};

/// The table's row of a bundled contract file; a file the table lacks fails the test.
inline FileFigures file_figures(const char* contract_json) {
    for (const FileFigures& figures : k_file_figures) {
        if (std::string_view(figures.m_contract_json) == contract_json) { return figures; }
    }
    ADD_FAILURE() << "no 2.x figures for " << contract_json;
    return {};
}

/// A bundled contract file with its 2.x figures pinned: the explicit budget and the fixed
/// warm-up of k_file_figures over what the file spells (the library's defaults).
inline anira::ContractHandle pinned_file_contract(const char* contract_json) {
    anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    const FileFigures figures = file_figures(contract_json);
    contract.hard_budget(ANIRA_BUDGET_EXPLICIT,
                         std::chrono::duration<double, std::milli>(figures.m_budget_ms));
    contract.hard_warmup(figures.m_warm_up > 0 ? ANIRA_WARMUP_FIXED : ANIRA_WARMUP_NONE,
                         figures.m_warm_up);
    return contract;
}

}  // namespace anira_test

#endif  // ANIRA_TEST_FILE_CONTRACTS_H
