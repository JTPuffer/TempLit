#pragma once
#include "binary_expr.h"
namespace expr {

struct SFTMX {};
struct SFTMX_DERIVATIVE {};

template <typename one> using EXPR_SFTMX = ExprNode<SFTMX, one>;
template <typename Grad, typename Output>
using EXPR_SFTMX_DERIVATIVE = ExprNode<SFTMX_DERIVATIVE, Grad, Output>;

template <EXPR_TYPE X>

auto soft_max(X &&value) {
  return EXPR_SFTMX<std::remove_cvref_t<X>>(std::forward<X>(value));
}

template <EXPR_TYPE X> auto softmax(X &&value) {
  return soft_max(std::forward<X>(value));
}
} // namespace expr
