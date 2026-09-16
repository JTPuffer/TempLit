#pragma once
#include "binary_expr.h"

namespace expr {

struct HADAMARD {};

template <typename L, typename R>
using EXPR_HADAMARD = ExprNode<HADAMARD, L, R>;

template <EXPR_TYPE L, EXPR_TYPE R> auto hadamard(L &&left, R &&right) {
  using LL = std::remove_cvref_t<L>;
  using RR = std::remove_cvref_t<R>;
  return EXPR_HADAMARD<LL, RR>(std::forward<L>(left), std::forward<R>(right));
}
} // namespace expr
