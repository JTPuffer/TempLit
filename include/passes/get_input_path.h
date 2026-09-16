#pragma once

#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

namespace passes {

template <typename Expr> struct get_input_path;

template <typename... Ops> struct get_input_path<operations<Ops...>> {

  using type = concat_t<typename get_input_path<Ops>::type...>;
};

template <typename PROG1, typename PROG2>
struct get_input_path<prog_link<PROG1, PROG2>> {

  using type = concat_t<typename get_input_path<PROG1>::type,
                        typename get_input_path<PROG2>::type>;
};

template <typename Op, typename... Args>
struct get_input_path<expr::ExprNode<Op, Args...>> {
  using type = concat_t<typename get_input_path<Args>::type...>;
};

template <typename GRAD, typename PATH>
struct get_input_path<accumulate_grad<GRAD, PATH>> {

  using type = typename get_input_path<GRAD>::type;
};
template <typename GRAD, typename PATH>
struct get_input_path<accumulate_local_grad<GRAD, PATH>> {
  using type = typename get_input_path<GRAD>::type;
};
template <typename PATH> struct get_input_path<fwd_ref<PATH>> {
  using type = type_list<>;
};

template <typename PATH> struct get_input_path<fwd_shape<PATH>> {
  using type = type_list<>;
};
template <typename PATH> struct get_input_path<fwd_last_dim<PATH>> {
  using type = type_list<>;
};
template <typename PATH> struct get_input_path<grad_ref<PATH>> {
  using type = type_list<>;
};
template <typename EXPR, typename PATH>
struct get_input_path<fwd_save<EXPR, PATH>> {
  using type = typename get_input_path<EXPR>::type;
};
template <typename PATH> struct get_input_path<input_ref<PATH>> {
  using type = type_list<PATH>;
};
template <typename PATH> struct get_input_path<index_ref<PATH>> {
  using type = type_list<>;
};
template <> struct get_input_path<no_grad> {
  using type = type_list<>;
};

template <> struct get_input_path<grad_seed> {
  using type = type_list<>;
};

template <auto V> struct get_input_path<expr::EXPR_CONSTANT<V>> {
  using type = type_list<>;
};

} // namespace passes
