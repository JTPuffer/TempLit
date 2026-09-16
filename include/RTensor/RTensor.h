#pragma once

#include "model_utils/archive_concepts.h"
#include <RTensor/storage.h>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rtensor {
using index_type = uint32_t;
using shape_type = std::vector<index_type>;
} // namespace rtensor

template <typename device, typename T> class RTensor {
public:
  using device_type = device;
  using index_type = rtensor::index_type;
  using shape_type = rtensor::shape_type;
  using value_type = T;
  template <typename U> using rebind = RTensor<device, U>;

private:
  shape_type _shape;
  shape_type _strides;
  Storage<device, T> *storage;

public:
  // do not use me people
  RTensor() : storage(nullptr) {}

  const Storage<device, T> *get_storage() const { return storage; }
  Storage<device, T> *get_storage() { return storage; }
  const shape_type &shape() const { return _shape; }
  const shape_type &strides() const { return _strides; }
  size_t numel() const {
    if (_shape.empty()) {
      return 0;
    }

    return std::accumulate(_shape.begin(), _shape.end(), size_t{1},
                           std::multiplies<>{});
  }

  RTensor(const RTensor &other)
      : _shape(other._shape), _strides(other._strides) {

    storage = other.storage;
    storage->ref_count += 1;
  }
  RTensor(T val)
    requires WritableStorage<Storage<device, T>, T>
      : _shape({1, 1}), _strides({1, 1}) {
    storage = new Storage<device, T>(1);
    storage->ref_count = 1;
    storage->write(std::span<const T>(&val, 1), 0);
  }
  RTensor &operator=(const RTensor &other) {
    if (this == &other)
      return *this;

    release_storage();

    _shape = other._shape;
    _strides = other._strides;

    storage = other.storage;
    storage->ref_count += 1;

    return *this;
  }
  RTensor(RTensor &&other) noexcept
      : _shape(std::move(other._shape)), _strides(std::move(other._strides)),
        storage(other.storage) {
    other.storage = nullptr;
  }
  RTensor &operator=(RTensor &&other) noexcept {
    if (this == &other)
      return *this;

    release_storage();

    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    storage = other.storage;

    other.storage = nullptr;
    other._shape.clear();
    other._strides.clear();

    return *this;
  }
  RTensor clone() const
    requires CloneableStorage<Storage<device, T>>
  {
    RTensor out;
    out._shape = _shape;
    out._strides = _strides;
    out.storage = storage->clone();
    return out;
  }
  RTensor(std::initializer_list<index_type> dim) : _shape(dim), _strides(dim) {
    size_t stride = 1;
    for (size_t i = _shape.size(); i-- > 0;) {
      _strides[i] = static_cast<index_type>(stride);
      stride *= _shape[i];
      if (stride > std::numeric_limits<index_type>::max()) {
        throw std::length_error("RTensor exceeds 32-bit indexing capacity");
      }
    }
    storage = new Storage<device, T>(stride);
    // dont need to do this but makes the deconstructor make more sense
    storage->ref_count = 1;
  }
  RTensor(std::span<const index_type> dim)
      : _shape(dim.begin(), dim.end()), _strides(dim.size()) {
    size_t stride = 1;

    for (size_t i = _shape.size(); i-- > 0;) {
      _strides[i] = static_cast<index_type>(stride);
      stride *= _shape[i];
      if (stride > std::numeric_limits<index_type>::max()) {
        throw std::length_error("RTensor exceeds 32-bit indexing capacity");
      }
    }

    storage = new Storage<device, T>(stride);
    storage->ref_count = 1;
  }

  static RTensor
  from_values(std::initializer_list<std::initializer_list<T>> values)
    requires WritableStorage<Storage<device, T>, T>
  {

    const size_t rows = values.size();
    if (rows == 0) {
      return RTensor({0, 0});
    }

    const size_t cols = values.begin()->size();
    for (const auto &row : values) {
      if (row.size() != cols) {
        throw std::invalid_argument("Tensor initializer must be rectangular");
      }
    }
    RTensor out({static_cast<index_type>(rows), static_cast<index_type>(cols)});

    std::vector<T> flattened;
    flattened.reserve(rows * cols);

    for (const auto &row : values) {
      flattened.insert(flattened.end(), row.begin(), row.end());
    }

    out.set(flattened);
    return out;
  }
  static RTensor from_zeros(std::initializer_list<index_type> dim)
    requires FillableStorage<Storage<device, T>, T>
  {
    RTensor ret(dim);
    ret.storage->fill(T{});
    return ret;
  }

  static RTensor from_zeros(std::span<const index_type> dim)
    requires FillableStorage<Storage<device, T>, T>
  {
    RTensor ret(dim);
    ret.storage->fill(T{});
    return ret;
  }
  bool operator==(const RTensor &other) const
    requires ComparableStorage<Storage<device, T>>
  {
    return other.shape() == _shape && other.strides() == _strides &&
           storage->compare(*other.storage);
  }

  template <typename... Indices>
  T get(Indices... indices) const
    requires((std::convertible_to<Indices, size_t> && ...) &&
             ReadableStorage<Storage<device, T>, T>)
  {
    static constexpr size_t rank = sizeof...(Indices);
    assert(rank == _shape.size());
    // idk might remove this guard
    size_t offset = 0;
    size_t dim = 0;
    ((offset += static_cast<size_t>(indices) * _strides[dim++]), ...);
    T out;

    storage->read(std::span<T>(&out, 1), offset);
    return out;
  }
  template <typename... Indices>
  void set(T value, Indices... indices)
    requires((std::convertible_to<Indices, size_t> && ...) &&
             WritableStorage<Storage<device, T>, T>)
  {
    static constexpr size_t rank = sizeof...(Indices);
    assert(rank == _shape.size());

    size_t offset = 0;
    size_t dim = 0;
    ((offset += static_cast<size_t>(indices) * _strides[dim++]), ...);

    storage->write(std::span<const T>(&value, 1), offset);
  }
  void set(std::span<const T> values)
    requires(WritableStorage<Storage<device, T>, T>)
  {
    const size_t expected_size = std::accumulate(
        _shape.begin(), _shape.end(), size_t{1}, std::multiplies<>{});

    if (values.size() != expected_size) {
      throw std::invalid_argument(
          "RTensor: values size does not match tensor shape");
    }

    storage->write(values, 0);
  }
  void read(std::span<T> values) const
    requires(ReadableStorage<Storage<device, T>, T>)
  {
    if (storage == nullptr) {
      throw std::logic_error("cannot read an uninitialised tensor");
    }
    if (values.size() != numel()) {
      throw std::invalid_argument(
          "RTensor: output size does not match tensor shape");
    }

    storage->read(values, 0);
  }
  RTensor view(std::span<const index_type> shape) const {
    if (storage == nullptr) {
      throw std::logic_error("cannot create a view of an uninitialised tensor");
    }

    size_t old_size = 1;
    size_t expected_stride = 1;
    for (size_t i = _shape.size(); i-- > 0;) {
      if (_strides[i] != expected_stride) {
        throw std::invalid_argument("cannot reshape a non-contiguous tensor");
      }
      old_size *= _shape[i];
      expected_stride *= _shape[i];
    }

    size_t new_size = 1;
    for (index_type dim : shape) {
      new_size *= dim;
      if (new_size > std::numeric_limits<index_type>::max()) {
        throw std::length_error("RTensor exceeds 32-bit indexing capacity");
      }
    }
    if (old_size != new_size) {
      throw std::invalid_argument("view shape must preserve element count");
    }

    RTensor result(*this);
    result._shape.assign(shape.begin(), shape.end());
    result._strides.resize(shape.size());

    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      result._strides[i] = static_cast<index_type>(stride);
      stride *= shape[i];
    }
    return result;
  }

  friend std::ostream &operator<<(std::ostream &stream, const RTensor &tensor)
    requires(ReadableStorage<Storage<device, T>, T>)
  {
    size_t numel = 1;
    for (size_t dim : tensor._shape)
      numel *= dim;

    std::vector<T> data(numel);
    tensor.storage->read(std::span<T>(data), 0);

    size_t offset = 0;

    auto print_dim = [&](auto &&self, size_t dim) -> void {
      stream << "[";

      if (dim == tensor._shape.size() - 1) {
        for (size_t i = 0; i < tensor._shape[dim]; ++i) {
          if (i != 0)
            stream << ", ";

          stream << data[offset++];
        }
      } else {
        for (size_t i = 0; i < tensor._shape[dim]; ++i) {
          if (i != 0)
            stream << ", ";

          self(self, dim + 1);
        }
      }

      stream << "]";
    };

    if (tensor._shape.empty()) {
      stream << "[]";
      return stream;
    }

    print_dim(print_dim, 0);
    return stream;
  }

  ~RTensor() { release_storage(); }

  template <WritableArchive Arch> void save(Arch &archive) const {
    if (storage == nullptr) {
      throw std::logic_error("cannot save an uninitialised tensor");
    }

    std::vector<T> data(numel());
    storage->read(std::span<T>(data), 0);
    archive.write("size", std::as_bytes(std::span<const index_type>(_shape)));
    archive.write("data", std::as_bytes(std::span<const T>(data)));
  }

  template <ReadableArchive Arch> void load(const Arch &archive) {
    if (storage == nullptr) {
      throw std::logic_error("cannot load into an uninitialised tensor");
    }

    shape_type load_shape(_shape.size());

    archive.read("size", std::as_writable_bytes(std::span(load_shape)));
    if (load_shape != _shape) {
      throw std::runtime_error(
          "current tensor shape and loaded tensor shape do not match");
    }

    std::vector<T> data(numel());
    archive.read("data", std::as_writable_bytes(std::span(data)));
    storage->write(std::span<const T>(data), 0);
  }

private:
  void release_storage() noexcept {
    if (!storage)
      return;

    storage->ref_count -= 1;
    if (storage->ref_count == 0)
      delete storage;

    storage = nullptr;
  }
};
