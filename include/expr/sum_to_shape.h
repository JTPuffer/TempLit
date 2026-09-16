#pragma once
#include "binary_expr.h"

namespace expr {

struct SUM_TO_SHAPE {};

template <typename Expr> struct shape_of {};

template <typename Value, typename Shape>
using EXPR_SUM_TO_SHAPE = ExprNode<SUM_TO_SHAPE, Value, Shape>;

template <EXPR_TYPE Value, typename Shape>
auto sum_to_shape(Value &&value, Shape &&shape) {
  using V = std::remove_cvref_t<Value>;
  using S = std::remove_cvref_t<Shape>;
  return EXPR_SUM_TO_SHAPE<V, S>(std::forward<Value>(value),
                                 std::forward<Shape>(shape));
}

} // namespace expr
