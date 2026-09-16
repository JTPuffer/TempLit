#pragma once

#include "binary_expr.h"

namespace expr {

struct SCATTER_INDEX {};

template <typename Values, typename Indices, typename Shape>
using EXPR_SCATTER_INDEX = ExprNode<SCATTER_INDEX, Values, Indices, Shape>;

template <EXPR_TYPE Values, EXPR_TENSOR Indices, typename Shape>
  requires(!requires_grad_v<Indices>)
auto scatter_index(Values &&values, Indices &&indices, Shape &&output_shape) {
  using V = std::remove_cvref_t<Values>;
  using I = std::remove_cvref_t<Indices>;
  using S = std::remove_cvref_t<Shape>;
  return EXPR_SCATTER_INDEX<V, I, S>(
      std::forward<Values>(values), std::forward<Indices>(indices),
      std::forward<Shape>(output_shape));
}

} // namespace expr
