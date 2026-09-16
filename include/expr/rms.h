#pragma once

#include "binary_expr.h"

namespace expr {

struct RMS {};

template <typename Value> using EXPR_RMS = ExprNode<RMS, Value>;

template <EXPR_TYPE X> auto rms(X &&value) {
  return EXPR_RMS<std::remove_cvref_t<X>>(std::forward<X>(value));
}

} // namespace expr
