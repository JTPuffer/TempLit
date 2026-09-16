#pragma once

#include "RTensor/storage.h"
#include "cpu.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T> class Storage<cpu::CPU, T> {
public:
  explicit Storage(size_t size) : ref_count(1), data_(size) {}

  int ref_count;

  std::vector<T> &data() { return data_; }
  const std::vector<T> &data() const { return data_; }

  void write(std::span<const T> values, size_t offset) {
    if (offset + values.size() > data_.size()) {
      throw std::out_of_range("write out of bounds");
    }
    std::copy(values.begin(), values.end(), data_.begin() + offset);
  }

  void read(std::span<T> out, size_t offset) const {
    if (offset + out.size() > data_.size()) {
      throw std::out_of_range("read out of bounds");
    }
    std::copy(data_.begin() + offset, data_.begin() + offset + out.size(),
              out.begin());
  }
  void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

  Storage *clone() const {
    Storage *other = new Storage(data_.size());
    other->data_ = data_;
    return other;
  }

  bool compare(const Storage &other) const { return data_ == other.data_; }

private:
  std::vector<T> data_;
};
