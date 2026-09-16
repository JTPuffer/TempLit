#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <vector>

class Dataset {
  std::vector<uint32_t> data_;
  size_t split_index_ = 0;

public:
  explicit Dataset(std::span<const uint32_t> tokens,
                   double training_fraction = 0.95);

  static Dataset load_text(std::istream &input,
                           double training_fraction = 0.95);
  static Dataset load_tokens(std::istream &input,
                             double training_fraction = 0.95);

  void save(std::ostream &output) const;

  std::span<const uint32_t> training_split() const;
  std::span<const uint32_t> validation_split() const;

  std::span<const uint32_t> get() const;
};
