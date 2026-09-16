#pragma once

#include "RTensor/storage.h"
#include "metal.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>

template <typename T> class Storage<metal::Metal, T> {
public:
  explicit Storage(size_t size) : ref_count(1), size_(size) {
    buffer = metal::device()->newBuffer(size * sizeof(T),
                                        MTL::ResourceStorageModeShared);
  }
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  Storage(Storage &&) = delete;
  Storage &operator=(Storage &&) = delete;

  ~Storage() { buffer->release(); }
  int ref_count;
  MTL::Buffer *get_buffer() const { return buffer; }

  void write(std::span<const T> values, size_t offset) {
    if (offset + values.size() > size_) {
      throw std::out_of_range("write out of bounds");
    }
    std::memcpy(static_cast<T *>(buffer->contents()) + offset, values.data(),
                values.size_bytes());
  }

  void read(std::span<T> out, size_t offset) const {
    if (offset + out.size() > size_) {
      throw std::out_of_range("read out of bounds");
    }
    std::memcpy(out.data(), static_cast<const T *>(buffer->contents()) + offset,
                out.size_bytes());
  }
  void fill(T value) {
    std::fill_n(static_cast<T *>(buffer->contents()), size_, value);
  }

  Storage *clone() const {
    Storage *other = new Storage(size_);
    std::memcpy(static_cast<T *>(other->buffer->contents()),
                static_cast<const T *>(buffer->contents()), size_ * sizeof(T));
    return other;
  }

  bool compare(const Storage &other) const {
    if (other.size_ != size_) {
      return false;
    }

    return std::memcmp(buffer->contents(), other.buffer->contents(),
                       size_ * sizeof(T)) == 0;
  }

private:
  MTL::Buffer *buffer = nullptr;
  std::size_t size_ = 0;
};
