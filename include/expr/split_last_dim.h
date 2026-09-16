#pragma once

#include "binary_expr.h"

#include <cstdint>

namespace expr {

template <uint32_t Outer, uint32_t Inner> struct SPLIT_LAST_DIM {
  static_assert(Outer > 0);
  static_assert(Inner > 0);
};

template <typename Value, uint32_t Outer, uint32_t Inner>
using EXPR_SPLIT_LAST_DIM =
    ExprNode<SPLIT_LAST_DIM<Outer, Inner>, Value>;

template <uint32_t Outer, uint32_t Inner, EXPR_TYPE X>
auto split_last_dim(X &&value) {
  return EXPR_SPLIT_LAST_DIM<std::remove_cvref_t<X>, Outer, Inner>(
      std::forward<X>(value));
}

} // namespace expr
