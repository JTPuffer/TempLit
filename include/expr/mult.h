#pragma once
#include "binary_expr.h"

namespace expr {

struct MULT {};
struct MULT_LHS_DERIVATIVE {};
struct MULT_RHS_DERIVATIVE {};

template <typename L, typename R> using EXPR_MULT = ExprNode<MULT, L, R>;
template <typename Grad, typename Rhs, typename Lhs>
using EXPR_MULT_LHS_DERIVATIVE =
    ExprNode<MULT_LHS_DERIVATIVE, Grad, Rhs, Lhs>;
template <typename Grad, typename Lhs, typename Rhs>
using EXPR_MULT_RHS_DERIVATIVE =
    ExprNode<MULT_RHS_DERIVATIVE, Grad, Lhs, Rhs>;

template <EXPR_TYPE L, EXPR_TYPE R>

auto mult(L &&left, R &&right) {
  using LL = std::remove_cvref_t<L>;
  using RR = std::remove_cvref_t<R>;
  return EXPR_MULT<LL, RR>(std::forward<L>(left), std::forward<R>(right));
}

template <EXPR_TYPE L, EXPR_TYPE R> auto operator*(L &&left, R &&right) {
  return EXPR_MULT<std::remove_cvref_t<L>, std::remove_cvref_t<R>>(
      std::forward<L>(left), std::forward<R>(right));
}
} // namespace expr
