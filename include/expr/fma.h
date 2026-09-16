#pragma once
#include "binary_expr.h"

namespace expr {
struct FMA {};
template <typename one, typename two, typename three>
using EXPR_FMA = ExprNode<FMA, one, two, three>;

template <EXPR_TYPE ONE, EXPR_TYPE TWO, EXPR_TYPE THREE>
auto fma(ONE &&one, TWO &&two, THREE &&three) {
  using O = std::remove_cvref_t<ONE>;
  using T = std::remove_cvref_t<TWO>;
  using TH = std::remove_cvref_t<THREE>;

  return EXPR_FMA<O, T, TH>(std::forward<ONE>(one), std::forward<TWO>(two),
                            std::forward<THREE>(three));
}

} // namespace expr
