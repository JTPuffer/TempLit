#pragma once

#include "expr.h"
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace expr {

template <typename T>

concept CanonicalExprType = std::same_as<T, std::remove_cvref_t<T>>;

template <typename OP, CanonicalExprType... Args>

struct ExprNode : EXPR {
  std::tuple<Args...> args_;

  template <typename... Values>
    requires(sizeof...(Values) == sizeof...(Args) &&
             (std::constructible_from<Args, Values &&> && ...))
  explicit ExprNode(Values &&...values)
      : args_(std::forward<Values>(values)...) {}
};

template <typename OP, CanonicalExprType... Args>
struct requires_grad<ExprNode<OP, Args...>>
    : std::bool_constant<(requires_grad_v<Args> || ...)> {};
} // namespace expr
