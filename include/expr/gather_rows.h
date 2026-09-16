#pragma once

#include "binary_expr.h"

namespace expr {

struct GATHER_ROWS {};

template <typename L, typename R>
using EXPR_GATHER_ROWS = ExprNode<GATHER_ROWS, L, R>;

template <EXPR_TYPE Values, EXPR_TENSOR Indices>
  requires(!requires_grad_v<Indices>)
auto gather_rows(Values &&values, Indices &&indices) {
  using V = std::remove_cvref_t<Values>;
  using I = std::remove_cvref_t<Indices>;
  return EXPR_GATHER_ROWS<V, I>(std::forward<Values>(values),
                                std::forward<Indices>(indices));
}

} // namespace expr
