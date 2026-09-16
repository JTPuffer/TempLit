#pragma once
#include "binary_expr.h"

namespace expr {

struct SUB {};

template <typename L, typename R> using EXPR_SUB = ExprNode<SUB, L, R>;

template <EXPR_TYPE L, EXPR_TYPE R> auto sub(L &&left, R &&right) {
  using LL = std::remove_cvref_t<L>;
  using RR = std::remove_cvref_t<R>;
  return EXPR_SUB<LL, RR>(std::forward<L>(left), std::forward<R>(right));
}

template <EXPR_TYPE L, EXPR_TYPE R> auto operator-(L &&left, R &&right) {
  return EXPR_SUB<std::remove_cvref_t<L>, std::remove_cvref_t<R>>(
      std::forward<L>(left), std::forward<R>(right));
}
} // namespace expr
