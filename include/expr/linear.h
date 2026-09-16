#pragma once
#include "binary_expr.h"

namespace expr {
struct LINEAR {};
template <typename one, typename two, typename three>
using EXPR_LINEAR = ExprNode<LINEAR, one, two, three>;

template <EXPR_TYPE ONE, EXPR_TYPE TWO, EXPR_TYPE THREE>
auto linear(ONE &&input, TWO &&weight, THREE &&bias) {
  using O = std::remove_cvref_t<ONE>;
  using T = std::remove_cvref_t<TWO>;
  using TH = std::remove_cvref_t<THREE>;

  return EXPR_LINEAR<O, T, TH>(std::forward<ONE>(input), std::forward<TWO>(weight),
                            std::forward<THREE>(bias));
}

} // namespace expr
