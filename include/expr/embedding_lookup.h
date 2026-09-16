#pragma once

#include "binary_expr.h"

namespace expr {

struct EMBEDDING_LOOKUP {};

template <typename Weights, typename Indices>
using EXPR_EMBEDDING_LOOKUP = ExprNode<EMBEDDING_LOOKUP, Weights, Indices>;

template <EXPR_TYPE Weights, EXPR_TENSOR Indices>
  requires(!requires_grad_v<Indices>)
auto embedding_lookup(Weights &&weights, Indices &&indices) {
  using W = std::remove_cvref_t<Weights>;
  using I = std::remove_cvref_t<Indices>;
  return EXPR_EMBEDDING_LOOKUP<W, I>(std::forward<Weights>(weights),
                                     std::forward<Indices>(indices));
}

} // namespace expr
