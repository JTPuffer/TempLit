#pragma once

#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

namespace passes {

template <typename Expr> struct get_required_paths;

template <typename... Ops> struct get_required_paths<operations<Ops...>> {

  using type = concat_t<typename get_required_paths<Ops>::type...>;
};

template <typename PROG1, typename PROG2>
struct get_required_paths<prog_link<PROG1, PROG2>> {
  using type = concat_t<typename get_required_paths<PROG1>::type,
                        typename get_required_paths<PROG2>::type>;
};

template <typename Op, typename... Args>
struct get_required_paths<expr::ExprNode<Op, Args...>> {
  using type = concat_t<typename get_required_paths<Args>::type...>;
};

template <typename GRAD, typename PATH>
struct get_required_paths<accumulate_grad<GRAD, PATH>> {

  using type = typename get_required_paths<GRAD>::type;
};
template <typename GRAD, typename PATH>
struct get_required_paths<accumulate_local_grad<GRAD, PATH>> {
  using type = typename get_required_paths<GRAD>::type;
};
template <typename PATH> struct get_required_paths<fwd_ref<PATH>> {
  using type = type_list<PATH>;
};

template <typename PATH> struct get_required_paths<fwd_shape<PATH>> {
  using type = type_list<PATH>;
};
template <typename PATH> struct get_required_paths<fwd_last_dim<PATH>> {
  using type = type_list<PATH>;
};
template <typename PATH> struct get_required_paths<grad_ref<PATH>> {
  using type = type_list<PATH>;
};
template <typename PATH> struct get_required_paths<index_ref<PATH>> {
  using type = type_list<>;
};
template <> struct get_required_paths<no_grad> {
  using type = type_list<>;
};

template <> struct get_required_paths<grad_seed> {
  using type = type_list<>;
};

template <auto V> struct get_required_paths<expr::EXPR_CONSTANT<V>> {
  using type = type_list<>;
};

} // namespace passes
