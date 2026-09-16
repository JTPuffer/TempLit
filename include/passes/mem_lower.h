#pragma once

#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace passes {
namespace mem_lower {

template <typename Expr, typename ProgPath, bool Temporary> struct eval {
  static_assert(always_false_v<Expr>,
                "No memory-lowering rule exists for this program type");
};

template <typename Op, typename ProgPath, typename Args, typename Indices>
struct helper;

template <typename Op, typename ProgPath, typename... Args, size_t... Is>
struct helper<Op, ProgPath, type_list<Args...>, std::index_sequence<Is...>> {
  using type = expr::ExprNode<
      Op, typename eval<Args, prog_path<ProgPath, Is>, true>::type...>;
};

template <typename Op, typename... Args, typename ProgPath, bool Temporary>
struct eval<expr::ExprNode<Op, Args...>, ProgPath, Temporary> {
  using lowered = typename helper<Op, ProgPath, type_list<Args...>,
                                  std::index_sequence_for<Args...>>::type;

  using type =
      std::conditional_t<Temporary, temp_ref<lowered, ProgPath>, lowered>;
};

template <typename ProgPath, typename Ops, typename Indices>
struct operations_helper;

template <typename ProgPath, typename... Ops, size_t... Is>
struct operations_helper<ProgPath, type_list<Ops...>,
                         std::index_sequence<Is...>> {
  using type =
      operations<typename eval<Ops, prog_path<ProgPath, Is>, false>::type...>;
};

template <typename... Ops, typename ProgPath, bool Temporary>
struct eval<operations<Ops...>, ProgPath, Temporary> {
  using type =
      typename operations_helper<ProgPath, type_list<Ops...>,
                                 std::index_sequence_for<Ops...>>::type;
};

template <typename PROG1, typename PROG2, typename ProgPath, bool Temporary>
struct eval<prog_link<PROG1, PROG2>, ProgPath, Temporary> {
  using prog1_path = prog_path<ProgPath, 0>;
  using prog2_path = prog_path<ProgPath, 1>;

  using lowered_1 = typename eval<PROG1, prog1_path, false>::type;
  using lowered_2 = typename eval<PROG2, prog2_path, false>::type;

  using lowered = prog_link<lowered_1, lowered_2>;

  using type =
      std::conditional_t<Temporary, temp_ref<lowered, ProgPath>, lowered>;
};

template <typename Expr, typename GraphPath, typename ProgPath, bool Temporary>
struct eval<fwd_save<Expr, GraphPath>, ProgPath, Temporary> {
  using lowered = typename eval<Expr, ProgPath, false>::type;
  using type = fwd_save<lowered, GraphPath>;
};

template <typename Grad, typename GraphPath, typename ProgPath, bool Temporary>
struct eval<accumulate_grad<Grad, GraphPath>, ProgPath, Temporary> {
  using lowered = typename eval<Grad, prog_path<ProgPath, 0>, true>::type;
  using type = accumulate_grad<lowered, GraphPath>;
};

template <typename Grad, typename GraphPath, typename ProgPath, bool Temporary>
struct eval<accumulate_local_grad<Grad, GraphPath>, ProgPath, Temporary> {
  using lowered = typename eval<Grad, prog_path<ProgPath, 0>, true>::type;
  using type = accumulate_local_grad<lowered, GraphPath>;
};

template <typename GraphPath, typename ProgPath, bool Temporary>
struct eval<input_ref<GraphPath>, ProgPath, Temporary> {
  using type = input_ref<GraphPath>;
};

template <typename GraphPath, typename ProgPath, bool Temporary>
struct eval<index_ref<GraphPath>, ProgPath, Temporary> {
  using type = index_ref<GraphPath>;
};

template <typename GraphPath, typename ProgPath, bool Temporary>
struct eval<fwd_ref<GraphPath>, ProgPath, Temporary> {
  using type = fwd_ref<GraphPath>;
};

template <typename GraphPath, typename ProgPath, bool Temporary>
struct eval<fwd_shape<GraphPath>, ProgPath, Temporary> {
  using type = fwd_shape<GraphPath>;
};

template <typename GraphPath, typename ProgPath, bool Temporary>
struct eval<fwd_last_dim<GraphPath>, ProgPath, Temporary> {
  using type = fwd_last_dim<GraphPath>;
};

template <typename GraphPath, typename ProgPath, bool Temporary>
struct eval<grad_ref<GraphPath>, ProgPath, Temporary> {
  using type = grad_ref<GraphPath>;
};

template <typename ProgPath, bool Temporary>
struct eval<grad_seed, ProgPath, Temporary> {
  using type = grad_seed;
};

template <typename ProgPath, bool Temporary>
struct eval<no_grad, ProgPath, Temporary> {
  using type = no_grad;
};

template <typename Program>
using lower_t = typename eval<Program, prog_root, false>::type;

} // namespace mem_lower
} // namespace passes
