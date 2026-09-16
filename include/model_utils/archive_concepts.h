#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <string>

template <typename T>
concept WritableArchive = requires(T &archive, const std::string &name,
                                   std::span<const std::byte> values) {
  { archive.write(name, values) } -> std::same_as<void>;
};

template <typename T>
concept ReadableArchive = requires(const T &archive, const std::string &name,
                                   std::span<std::byte> values) {
  { archive.read(name, values) } -> std::same_as<void>;
};

template <typename T>
concept ArchiveType = WritableArchive<T> && ReadableArchive<T>;
