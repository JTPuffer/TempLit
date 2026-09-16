#pragma once
#include "binary_expr.h"
#include "expr/constant.h"
namespace expr {

template <auto Exponent> struct POW {};

template <typename Value, auto Exponent>
using EXPR_POW = ExprNode<POW<Exponent>, Value>;

template <EXPR_TYPE X, auto Exponent>
auto pow(X &&value, EXPR_CONSTANT<Exponent>) {
  return EXPR_POW<std::remove_cvref_t<X>, Exponent>(
      std::forward<X>(value));
}
} // namespace expr
