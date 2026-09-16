#pragma once

#include "binary_expr.h"

namespace expr {

struct SUM {};

template <typename VAL> using EXPR_SUM = ExprNode<SUM, VAL>;

} // namespace expr
