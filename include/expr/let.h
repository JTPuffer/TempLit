#pragma once
#include "binary_expr.h"

#include <functional>

namespace expr {
template <typename TAG> struct LET {};

template <typename TAG, bool RequiresGrad> struct FUNC_PARAM : EXPR {};

template <typename TAG, bool RequiresGrad>
struct requires_grad<FUNC_PARAM<TAG, RequiresGrad>>
    : std::bool_constant<RequiresGrad> {};

template <typename TAG, typename VAL, typename REF>
using EXPR_LET = ExprNode<LET<TAG>, VAL, REF>;

template <typename Function, typename Argument>
concept ExpressionBuilder = requires(
    Function &fn,
    FUNC_PARAM<std::remove_cvref_t<Function>, requires_grad_v<Argument>> &arg) {
  { std::invoke(fn, arg) } -> EXPR_TYPE;
};

template <expr::EXPR_TYPE Value, typename Builder>
  requires ExpressionBuilder<Builder, Value>
auto let(Value &&value, Builder &&builder) {
  using Tag = std::remove_cvref_t<Builder>;
  FUNC_PARAM<Tag, requires_grad_v<Value>> parameter;

  auto body = std::invoke(builder, parameter);

  return EXPR_LET<Tag, std::remove_cvref_t<Value>,
                  std::remove_cvref_t<decltype(body)>>(
      std::forward<Value>(value), std::move(body));
}

} // namespace expr
