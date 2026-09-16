#pragma once

#include "binary_expr.h"

namespace expr {

struct SCATTER_ADD_ROWS {};

template <typename Base, typename Indices, typename Values>
using EXPR_SCATTER_ADD_ROWS =
    ExprNode<SCATTER_ADD_ROWS, Base, Indices, Values>;

template <EXPR_TYPE Base, EXPR_TENSOR Indices, EXPR_TYPE Values>
  requires(!requires_grad_v<Indices>)
auto scatter_add_rows(Base &&base, Indices &&indices, Values &&values) {
  using B = std::remove_cvref_t<Base>;
  using I = std::remove_cvref_t<Indices>;
  using V = std::remove_cvref_t<Values>;
  return EXPR_SCATTER_ADD_ROWS<B, I, V>(
      std::forward<Base>(base), std::forward<Indices>(indices),
      std::forward<Values>(values));
}

} // namespace expr
