#pragma once

#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

namespace passes {

template <typename Expr> struct get_saved_paths;

template <typename... Ops> struct get_saved_paths<operations<Ops...>> {
  using type = concat_t<typename get_saved_paths<Ops>::type...>;
};

template <typename PROG1, typename PROG2>
struct get_saved_paths<prog_link<PROG1, PROG2>> {
  using type = concat_t<typename get_saved_paths<PROG1>::type,
                        typename get_saved_paths<PROG2>::type>;
};

template <typename Op, typename... Args>
struct get_saved_paths<expr::ExprNode<Op, Args...>> {
  using type = concat_t<typename get_saved_paths<Args>::type...>;
};

template <typename GRAD, typename PATH>
struct get_saved_paths<accumulate_grad<GRAD, PATH>> {
  using type = typename get_saved_paths<GRAD>::type;
};

template <typename GRAD, typename PATH>
struct get_saved_paths<accumulate_local_grad<GRAD, PATH>> {
  using type = typename get_saved_paths<GRAD>::type;
};

template <typename EXPR, typename PATH>
struct get_saved_paths<fwd_save<EXPR, PATH>> {
  using type = concat_t<type_list<PATH>, typename get_saved_paths<EXPR>::type>;
};

template <typename PATH> struct get_saved_paths<fwd_ref<PATH>> {
  using type = type_list<>;
};

template <typename PATH> struct get_saved_paths<fwd_shape<PATH>> {
  using type = type_list<>;
};

template <typename PATH> struct get_saved_paths<fwd_last_dim<PATH>> {
  using type = type_list<>;
};

template <typename PATH> struct get_saved_paths<grad_ref<PATH>> {
  using type = type_list<>;
};

template <typename PATH> struct get_saved_paths<input_ref<PATH>> {
  using type = type_list<>;
};

template <typename PATH> struct get_saved_paths<index_ref<PATH>> {
  using type = type_list<>;
};

template <> struct get_saved_paths<no_grad> {
  using type = type_list<>;
};

template <> struct get_saved_paths<grad_seed> {
  using type = type_list<>;
};

template <auto V> struct get_saved_paths<expr::EXPR_CONSTANT<V>> {
  using type = type_list<>;
};

} // namespace passes
