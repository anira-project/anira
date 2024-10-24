#ifndef ANIRA_BENCHMARK_HELPERFUNCTIONS_H
#define ANIRA_BENCHMARK_HELPERFUNCTIONS_H

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace anira{
namespace benchmark{

/* ============================================================ *
 * ===================== Helper functions ===================== *
 * ============================================================ */

static float random_sample () {
    return -1.f + (float) (std::rand()) / ((float) (RAND_MAX/2.f));
}

static double calculate_percentile(const std::vector<double>& v, double percentile) {
    // Make sure the data is not empty
    if (v.empty()) {
        throw std::invalid_argument("Input vector is empty.");
    }

    // Sort the data in ascending order
    std::vector<double> sorted_data = v;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Calculate the index for the 99th percentile
    size_t n = sorted_data.size();
    size_t percentile_index = (size_t) (percentile * (n - 1));

    // Check if the index is an integer
    if (percentile_index == static_cast<size_t>(percentile_index)) {
        // The index is an integer, return the value at that index
        return sorted_data[static_cast<size_t>(percentile_index)];
    } else {
        // Interpolate between the two nearest values
        size_t lower_index = static_cast<size_t>(percentile_index);
        size_t upper_index = lower_index + 1;
        double fraction = percentile_index - lower_index;
        return (1.0 - fraction) * sorted_data[lower_index] + fraction * sorted_data[upper_index];
    }
}

const auto calculate_min = [](const std::vector<double>& v) -> double {
    return *(std::min_element(std::begin(v), std::end(v)));
};

const auto calculate_max = [](const std::vector<double>& v) -> double {
    return *(std::max_element(std::begin(v), std::end(v)));
};

} // namespace benchmark
} // namespace anira

#endif // ANIRA_BENCHMARK_HELPERFUNCTIONS_H