#pragma once
#include "binary_expr.h"

namespace expr {

template <auto V> struct ConstantOp {
  static constexpr auto value = V;
};

template <auto V> using EXPR_CONSTANT = ExprNode<ConstantOp<V>>;

} // namespace expr
