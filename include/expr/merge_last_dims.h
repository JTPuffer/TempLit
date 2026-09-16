#pragma once

#include "binary_expr.h"

#include <cstdint>

namespace expr {

template <uint32_t Outer, uint32_t Inner> struct MERGE_LAST_DIMS {
  static_assert(Outer > 0);
  static_assert(Inner > 0);
};

template <typename Value, uint32_t Outer, uint32_t Inner>
using EXPR_MERGE_LAST_DIMS =
    ExprNode<MERGE_LAST_DIMS<Outer, Inner>, Value>;

template <uint32_t Outer, uint32_t Inner, EXPR_TYPE X>
auto merge_last_dims(X &&value) {
  return EXPR_MERGE_LAST_DIMS<std::remove_cvref_t<X>, Outer, Inner>(
      std::forward<X>(value));
}

} // namespace expr
