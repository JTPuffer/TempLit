#pragma once
#include "binary_expr.h"

namespace expr {

struct DIV {};

template <typename L, typename R> using EXPR_DIV = ExprNode<DIV, L, R>;

template <EXPR_TYPE L, EXPR_TYPE R>

auto div(L &&left, R &&right) {
  using LL = std::remove_cvref_t<L>;
  using RR = std::remove_cvref_t<R>;
  return EXPR_DIV<LL, RR>(std::forward<L>(left), std::forward<R>(right));
}

template <EXPR_TYPE L, EXPR_TYPE R> auto divide(L &&left, R &&right) {
  return div(std::forward<L>(left), std::forward<R>(right));
}

template <EXPR_TYPE L, EXPR_TYPE R> auto operator/(L &&left, R &&right) {
  return EXPR_DIV<std::remove_cvref_t<L>, std::remove_cvref_t<R>>(
      std::forward<L>(left), std::forward<R>(right));
}
} // namespace expr
