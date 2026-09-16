#pragma once
#include <concepts>
#include <cstddef>
#include <span>
template <typename Device, typename T> class Storage;

template <typename S, typename T>
concept WritableStorage =
    requires(S &storage, std::span<const T> values, size_t offset) {
      storage.write(values, offset);
    };

template <typename S, typename T>
concept ReadableStorage =
    requires(const S &storage, std::span<T> out, size_t offset) {
      storage.read(out, offset);
    };

template <typename S, typename T>
concept FillableStorage =
    requires(S &storage, const T &out) { storage.fill(out); };
// means later we dont have to copy Storage to CPU for comparisons
template <typename S>
concept ComparableStorage = requires(const S &storage) {
  { storage.compare(storage) } -> std::same_as<bool>;
};
template <typename S>
concept CloneableStorage = requires(const S &storage) {
  { storage.clone() } -> std::same_as<S *>;
};
