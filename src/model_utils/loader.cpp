#include "model_utils/loader.h"
#include <cstddef>
#include <stdexcept>

Loader::Loader(std::span<const uint32_t> dataset, uint32_t seed)
    : dataset_(dataset), generator_(seed) {
  if (dataset_.empty()) {
    throw std::invalid_argument("loader dataset must not be empty");
  }
}

std::span<const uint32_t> Loader::get_sample(size_t len) {
  if (len == 0 || len > dataset_.size()) {
    throw std::invalid_argument("sample length exceeds dataset size");
  }

  std::uniform_int_distribution<size_t> distribution(0,
                                                      dataset_.size() - len);
  const size_t index = distribution(generator_);
  return dataset_.subspan(index, len);
}

const Loader::Batch &Loader::get_batch(size_t batch_size,
                                      size_t sequence_size) {
  if (batch_size == 0 || sequence_size == 0) {
    throw std::invalid_argument("batch dimensions must be non-zero");
  }

  batch_.inputs.resize(batch_size * sequence_size);
  batch_.targets.resize(batch_size * sequence_size);

  for (size_t batch = 0; batch < batch_size; ++batch) {
    const auto sample = get_sample(sequence_size + 1);

    for (size_t position = 0; position < sequence_size; ++position) {
      const size_t offset = batch * sequence_size + position;
      batch_.inputs[offset] = sample[position];
      batch_.targets[offset] = sample[position + 1];
    }
  }

  return batch_;
}
