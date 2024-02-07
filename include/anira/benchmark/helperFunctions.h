#ifndef ANIRA_BENMARK_HELPERFUNCTIONS_H
#define ANIRA_BENMARK_HELPERFUNCTIONS_H

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace anira{
namespace benchmark{

/* ============================================================ *
 * ===================== Helper functions ===================== *
 * ============================================================ */

static float randomSample () {
    return -1.f + (float) (std::rand()) / ((float) (RAND_MAX/2.f));
}

static double calculatePercentile(const std::vector<double>& v, double percentile) {
    // Make sure the data is not empty
    if (v.empty()) {
        throw std::invalid_argument("Input vector is empty.");
    }

    // Sort the data in ascending order
    std::vector<double> sortedData = v;
    std::sort(sortedData.begin(), sortedData.end());

    // Calculate the index for the 99th percentile
    size_t n = sortedData.size();
    size_t percentileIndex = (size_t) (percentile * (n - 1));

    // Check if the index is an integer
    if (percentileIndex == static_cast<size_t>(percentileIndex)) {
        // The index is an integer, return the value at that index
        return sortedData[static_cast<size_t>(percentileIndex)];
    } else {
        // Interpolate between the two nearest values
        size_t lowerIndex = static_cast<size_t>(percentileIndex);
        size_t upperIndex = lowerIndex + 1;
        double fraction = percentileIndex - lowerIndex;
        return (1.0 - fraction) * sortedData[lowerIndex] + fraction * sortedData[upperIndex];
    }
}

const auto calculateMin = [](const std::vector<double>& v) -> double {
    return *(std::min_element(std::begin(v), std::end(v)));
};

const auto calculateMax = [](const std::vector<double>& v) -> double {
    return *(std::max_element(std::begin(v), std::end(v)));
};

} // namespace benchmark
} // namespace anira

#endif // ANIRA_BENMARK_HELPERFUNCTIONS_H