#include "RTensor/infer_shape.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
  using Shape = std::vector<uint32_t>;

  assert(rtensor::infer_sum_to_shape(Shape{2, 3, 4}, Shape{1, 3, 1}) ==
         Shape({1, 3, 1}));
  assert(rtensor::infer_sum_to_shape(Shape{2, 3, 4}, Shape{4}) == Shape({4}));
  assert(rtensor::infer_sum_to_shape(Shape{1, 4, 128}, Shape{1, 128}) ==
         Shape({1, 128}));

  bool threw = false;
  try {
    (void)rtensor::infer_sum_to_shape(Shape{2, 3}, Shape{1, 2, 3});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  assert(rtensor::infer_embedding_lookup(Shape{8, 3}, Shape{2, 2}) ==
         Shape({2, 2, 3}));
  assert(rtensor::infer_scatter_index(Shape{2, 2, 3}, Shape{2, 2},
                                      Shape{8, 3}) == Shape({8, 3}));
  assert(rtensor::infer_scatter_add_rows(Shape{8, 3}, Shape{2, 2},
                                         Shape{2, 2, 3}) == Shape({8, 3}));
  assert(rtensor::infer_rms(Shape{2, 3, 4}) == Shape({2, 3, 1}));
  assert(rtensor::infer_rms_norm(Shape{2, 3, 4}) == Shape({2, 3, 4}));

  return 0;
}
