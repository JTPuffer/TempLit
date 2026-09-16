#pragma once
#include "binary_expr.h"
namespace expr {

struct SQE {};

template <typename Pred, typename Target>
using EXPR_SQE = ExprNode<SQE, Pred, Target>;

namespace loss {

template <typename Pred, typename Target>
auto sqe(Pred &&pred, Target &&target) {
  return EXPR_SQE<std::remove_cvref_t<Pred>, std::remove_cvref_t<Target>>(
      std::forward<Pred>(pred), std::forward<Target>(target));
}
} // namespace loss
} // namespace expr
