#pragma once
#include "binary_expr.h"
namespace expr {

struct SQRT {};

template <typename one> using EXPR_SQRT = ExprNode<SQRT, one>;

template <EXPR_TYPE X>

auto sqrt(X &&value) {
  return EXPR_SQRT<std::remove_cvref_t<X>>(std::forward<X>(value));
}
} // namespace expr
