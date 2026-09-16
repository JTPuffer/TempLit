#pragma once
#include "binary_expr.h"

#include <cstdint>

namespace expr {

template <int32_t DimA = -2, int32_t DimB = -1> struct TRAN {};

template <typename Value, int32_t DimA = -2, int32_t DimB = -1>
using EXPR_TRAN = ExprNode<TRAN<DimA, DimB>, Value>;

template <int32_t DimA = -2, int32_t DimB = -1, EXPR_TYPE X>
auto transpose(X &&value) {
  static_assert(DimA != DimB, "transpose dimensions must be different");
  return EXPR_TRAN<std::remove_cvref_t<X>, DimA, DimB>(
      std::forward<X>(value));
}
} // namespace expr
