#pragma once
#include "binary_expr.h"
namespace expr {

struct SIG {};

template <typename one> using EXPR_SIG = ExprNode<SIG, one>;

template <EXPR_TYPE X>

auto sigmoid(X &&value) {
  return EXPR_SIG<std::remove_cvref_t<X>>(std::forward<X>(value));
}
} // namespace expr
