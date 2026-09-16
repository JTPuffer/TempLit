#pragma once
#include "binary_expr.h"
namespace expr {

struct GELU {};
struct GELU_DERIVATIVE {};

template <typename one> using EXPR_GELU = ExprNode<GELU, one>;
template <typename Value>
using EXPR_GELU_DERIVATIVE = ExprNode<GELU_DERIVATIVE, Value>;

template <EXPR_TYPE X>

auto gelu(X &&value) {
  return EXPR_GELU<std::remove_cvref_t<X>>(std::forward<X>(value));
}
} // namespace expr
