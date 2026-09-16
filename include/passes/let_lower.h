#pragma once

#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace passes {
namespace let_lower {
template <typename Expr, typename ProgPath> struct eval {
  static_assert(always_false_v<Expr>,
                "No memory-lowering rule exists for this program type");
};

template <typename Op, typename ProgPath, typename Args, typename Indices>
struct helper;

template <typename Op, typename ProgPath, typename... Args, size_t... Is>
struct helper<Op, ProgPath, type_list<Args...>, std::index_sequence<Is...>> {
  using type =
      expr::ExprNode<Op, typename eval<Args, prog_path<ProgPath, Is>>::type...>;
};

template <typename Op, typename... Args, typename ProgPath>
struct eval<expr::ExprNode<Op, Args...>, ProgPath> {
  using lowered = typename helper<Op, ProgPath, type_list<Args...>,
                                  std::index_sequence_for<Args...>>::type;

  using type = lowered;
};

template <typename ProgPath, typename Ops, typename Indices>
struct operations_helper;

template <typename ProgPath, typename... Ops, size_t... Is>
struct operations_helper<ProgPath, type_list<Ops...>,
                         std::index_sequence<Is...>> {
  using type = operations<typename eval<Ops, prog_path<ProgPath, Is>>::type...>;
};

template <typename... Ops, typename ProgPath>
struct eval<operations<Ops...>, ProgPath> {
  using type =
      typename operations_helper<ProgPath, type_list<Ops...>,
                                 std::index_sequence_for<Ops...>>::type;
};

} // namespace let_lower
} // namespace passes
