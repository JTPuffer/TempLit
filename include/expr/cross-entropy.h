#pragma once
#include "binary_expr.h"
namespace expr {

struct X_ENTROPY {};
struct X_ENTROPY_DERIVATIVE {};

template <typename Pred, typename Target>
using EXPR_X_ENTROPY = ExprNode<X_ENTROPY, Pred, Target>;
template <typename Pred, typename Target>
using EXPR_X_ENTROPY_DERIVATIVE =
    ExprNode<X_ENTROPY_DERIVATIVE, Pred, Target>;

namespace loss {

template <EXPR_TYPE Pred, EXPR_TENSOR Target>
  requires(!requires_grad_v<Target>)
auto cross_entropy(Pred &&pred, Target &&target) {
  return EXPR_X_ENTROPY<std::remove_cvref_t<Pred>, std::remove_cvref_t<Target>>(
      std::forward<Pred>(pred), std::forward<Target>(target));
}
} // namespace loss
} // namespace expr
