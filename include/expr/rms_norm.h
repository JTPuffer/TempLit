#pragma once
#include "binary_expr.h"
namespace expr {

struct RMS_NORM {};
struct RMSN_DERIVATIVE {};

template <typename one> using EXPR_RMS_NORM = ExprNode<RMS_NORM, one>;
template <typename Grad, typename Value, typename Output>
using EXPR_RMSN_DERIVATIVE =
    ExprNode<RMSN_DERIVATIVE, Grad, Value, Output>;

template <EXPR_TYPE X>

auto rms_norm(X &&value) {
  return EXPR_RMS_NORM<std::remove_cvref_t<X>>(std::forward<X>(value));
}
} // namespace expr
