#include "model_utils/Dataset.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <istream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>

namespace {
constexpr uint32_t eos_token = 256;
}

Dataset::Dataset(std::span<const uint32_t> tokens, double training_fraction)
    : data_(tokens.begin(), tokens.end()) {
  if (training_fraction <= 0.0 || training_fraction >= 1.0) {
    throw std::invalid_argument(
        "training fraction must be between zero and one");
  }

  if (data_.size() < 2) {
    throw std::invalid_argument("dataset must contain at least two tokens");
  }

  const size_t target = static_cast<size_t>(static_cast<double>(data_.size()) *
                                            training_fraction);
  const auto boundary =
      std::find(data_.begin() + static_cast<std::ptrdiff_t>(target),
                data_.end(), eos_token);
  split_index_ =
      boundary == data_.end()
          ? target
          : static_cast<size_t>(std::distance(data_.begin(), boundary)) + 1;

  if (split_index_ == 0 || split_index_ >= data_.size()) {
    throw std::invalid_argument("dataset split produced an empty partition");
  }
}

Dataset Dataset::load_text(std::istream &input, double training_fraction) {
  if (!input) {
    throw std::runtime_error("failed to read text dataset");
  }

  const std::string bytes{std::istreambuf_iterator<char>(input),
                          std::istreambuf_iterator<char>()};
  std::vector<uint32_t> tokens;
  tokens.reserve(bytes.size());

  for (size_t i = 0; i < bytes.size();) {
    if (i + 1 < bytes.size() && bytes[i] == '\n' && bytes[i + 1] == '\n') {
      tokens.push_back(eos_token);
      i += 2;
    } else {
      tokens.push_back(static_cast<unsigned char>(bytes[i]));
      ++i;
    }
  }

  return Dataset(tokens, training_fraction);
}

Dataset Dataset::load_tokens(std::istream &input, double training_fraction) {
  if (!input) {
    throw std::runtime_error("failed to read token dataset");
  }

  const std::vector<char> bytes{std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>()};

  if (bytes.size() % sizeof(uint32_t) != 0) {
    throw std::runtime_error("invalid token dataset size");
  }

  std::vector<uint32_t> tokens(bytes.size() / sizeof(uint32_t));
  if (!bytes.empty()) {
    std::memcpy(tokens.data(), bytes.data(), bytes.size());
  }

  return Dataset(tokens, training_fraction);
}

void Dataset::save(std::ostream &output) const {
  output.write(reinterpret_cast<const char *>(data_.data()),
               static_cast<std::streamsize>(data_.size() * sizeof(uint32_t)));
  if (!output) {
    throw std::runtime_error("failed to write token dataset");
  }
}

std::span<const uint32_t> Dataset::training_split() const {
  return std::span<const uint32_t>(data_).first(split_index_);
}

std::span<const uint32_t> Dataset::validation_split() const {
  return std::span<const uint32_t>(data_).subspan(split_index_);
}
std::span<const uint32_t> Dataset::get() const {
  return std::span<const uint32_t>(data_);
}
