#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace rtensor {

inline size_t normalise_dimension(size_t rank, int32_t dimension) {
  const int64_t normalised =
      dimension < 0 ? static_cast<int64_t>(rank) + dimension : dimension;
  if (normalised < 0 || normalised >= static_cast<int64_t>(rank)) {
    throw std::out_of_range("tensor dimension out of range");
  }
  return static_cast<size_t>(normalised);
}

} // namespace rtensor
