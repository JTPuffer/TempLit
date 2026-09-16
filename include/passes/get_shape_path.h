#pragma once

#include "expr/binary_expr.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <cstddef>
#include <utility>

namespace passes {

template <typename Expr, typename ProgPath> struct get_shape_path;

template <typename ProgPath, typename Args, typename Indices>
struct get_expr_shape_paths;

template <typename ProgPath, typename... Args, size_t... Is>
struct get_expr_shape_paths<ProgPath, type_list<Args...>,
                            std::index_sequence<Is...>> {
  using type =
      concat_t<type_list<ProgPath>,
               typename get_shape_path<Args, prog_path<ProgPath, Is>>::type...>;
};

template <typename Op, typename... Args, typename ProgPath>
struct get_shape_path<expr::ExprNode<Op, Args...>, ProgPath> {
  using type =
      typename get_expr_shape_paths<ProgPath, type_list<Args...>,
                                    std::index_sequence_for<Args...>>::type;
};

template <typename ProgPath, typename Ops, typename Indices>
struct get_operation_shape_paths;

template <typename ProgPath, typename... Ops, size_t... Is>
struct get_operation_shape_paths<ProgPath, type_list<Ops...>,
                                 std::index_sequence<Is...>> {
  using type =
      concat_t<typename get_shape_path<Ops, prog_path<ProgPath, Is>>::type...>;
};

template <typename... Ops, typename ProgPath>
struct get_shape_path<operations<Ops...>, ProgPath> {
  using type =
      typename get_operation_shape_paths<ProgPath, type_list<Ops...>,
                                         std::index_sequence_for<Ops...>>::type;
};

template <typename PROG1, typename PROG2, typename ProgPath>
struct get_shape_path<prog_link<PROG1, PROG2>, ProgPath> {
  using prog1_path = prog_path<ProgPath, 0>;
  using prog2_path = prog_path<ProgPath, 1>;

  using type = concat_t<typename get_shape_path<PROG1, prog1_path>::type,
                        typename get_shape_path<PROG2, prog2_path>::type,
                        type_list<ProgPath>>;
};

template <typename Expr, typename GraphPath, typename ProgPath>
struct get_shape_path<fwd_save<Expr, GraphPath>, ProgPath>
    : get_shape_path<Expr, ProgPath> {};

template <typename Expr, typename TempPath, typename ProgPath>
struct get_shape_path<temp_ref<Expr, TempPath>, ProgPath>
    : get_shape_path<Expr, ProgPath> {};

template <typename Grad, typename GraphPath, typename ProgPath>
struct get_shape_path<accumulate_grad<Grad, GraphPath>, ProgPath>
    : get_shape_path<Grad, prog_path<ProgPath, 0>> {};

template <typename Grad, typename GraphPath, typename ProgPath>
struct get_shape_path<accumulate_local_grad<Grad, GraphPath>, ProgPath>
    : get_shape_path<Grad, prog_path<ProgPath, 0>> {};

template <typename GraphPath, typename ProgPath>
struct get_shape_path<fwd_ref<GraphPath>, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename GraphPath, typename ProgPath>
struct get_shape_path<fwd_shape<GraphPath>, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename GraphPath, typename ProgPath>
struct get_shape_path<fwd_last_dim<GraphPath>, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename GraphPath, typename ProgPath>
struct get_shape_path<grad_ref<GraphPath>, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename GraphPath, typename ProgPath>
struct get_shape_path<input_ref<GraphPath>, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename GraphPath, typename ProgPath>
struct get_shape_path<index_ref<GraphPath>, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename ProgPath> struct get_shape_path<grad_seed, ProgPath> {
  using type = type_list<ProgPath>;
};

template <typename ProgPath> struct get_shape_path<no_grad, ProgPath> {
  using type = type_list<>;
};

} // namespace passes
