#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>
class Loader {
public:
  struct Batch {
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> targets;
  };

private:
  std::span<const uint32_t> dataset_;
  std::mt19937 generator_;
  Batch batch_;

public:
  explicit Loader(std::span<const uint32_t> dataset, uint32_t seed = 1);

  std::span<const uint32_t> get_sample(size_t len);
  const Batch &get_batch(size_t batch_size, size_t sequence_size);
};
