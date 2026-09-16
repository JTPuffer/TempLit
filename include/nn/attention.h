#pragma once

#include "RTensor/RTensor.h"
#include <cstdint>
#include <limits>
#include <vector>

namespace nn {
template <typename Device, typename T>
RTensor<Device, T> causal_mask(uint32_t query_length, uint32_t key_length) {

  RTensor<Device, T> out({1, 1, query_length, key_length});
  std::vector<T> mask(query_length * key_length, T{0});

  for (uint32_t x = 0; x < query_length; x++) {
    for (uint32_t y = x + 1; y < key_length; y++) {
      mask[x * key_length + y] = -std::numeric_limits<T>::infinity();
    }
  }
  out.set(mask);
  return out;
}
} // namespace nn
