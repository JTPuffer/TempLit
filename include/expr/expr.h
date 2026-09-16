#pragma once

#include <concepts>
#include <type_traits>

namespace expr {

struct EXPR {};

struct TENSOR_EXPR : EXPR {};

template <class x>
concept EXPR_TYPE = std::is_base_of<EXPR, typename std::decay<x>::type>::value;

template <typename E> struct requires_grad : std::false_type {};

template <typename E>
inline constexpr bool requires_grad_v =
    requires_grad<std::remove_cvref_t<E>>::value;

template <typename X>
concept EXPR_TENSOR =
    std::derived_from<std::remove_cvref_t<X>, TENSOR_EXPR>;
} // namespace expr
