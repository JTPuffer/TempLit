#pragma once

#include <concepts>

namespace Optimiser {

template <typename T>
concept Optimisable = requires(typename T::value_type lr) {
  { T(lr) } -> std::convertible_to<T>;
};

} // namespace Optimiser
