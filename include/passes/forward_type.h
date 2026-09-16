#pragma once

#include "RTensor/RTensor.h"
#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace passes {
namespace forward_type {

template <expr::EXPR_TYPE Expr, typename REQUIRED, typename PATH> struct eval;

template <typename Op, typename PATH, typename REQUIRED, typename Args,
          typename Indices>
struct helper;

template <typename Op, typename PATH, typename REQUIRED, typename... Args,
          size_t... Is>
struct helper<Op, PATH, REQUIRED, type_list<Args...>,
              std::index_sequence<Is...>> {
  using type = expr::ExprNode<
      Op, typename eval<Args, REQUIRED, graph_path<PATH, Is>>::type...>;
};

template <typename Op, expr::EXPR_TYPE... Args, typename REQUIRED,
          typename PATH>
struct eval<expr::ExprNode<Op, Args...>, REQUIRED, PATH> {
  using fwd = typename helper<Op, PATH, REQUIRED, type_list<Args...>,
                              std::index_sequence_for<Args...>>::type;

  using type = std::conditional_t<contains_type_v<REQUIRED, PATH>,
                                  fwd_save<fwd, PATH>, fwd>;
};

template <typename T, expr::tensor::PARAM_TYPE Policy, typename REQUIRED,
          typename PATH>
struct eval<expr::tensor::Tensor<T, Policy>, REQUIRED, PATH> {
  using type = std::conditional_t<
      std::same_as<typename T::value_type, rtensor::index_type>,
      index_ref<PATH>,
      std::conditional_t<contains_type_v<REQUIRED, PATH>,
                         fwd_save<input_ref<PATH>, PATH>, input_ref<PATH>>>;
};

template <typename TAG, typename VAL, typename BODY, typename REQUIRED,
          typename PATH>
struct eval<expr::EXPR_LET<TAG, VAL, BODY>, REQUIRED, PATH> {
  using val_path = graph_path<PATH, 0>;
  using local_path = graph_path<PATH, 1>;

  using ValueProgram = typename eval<VAL, REQUIRED, val_path>::type;

  using BodyProgram = typename eval<BODY, REQUIRED, local_path>::type;

  using fwd = prog_link<fwd_save<ValueProgram, TAG>, BodyProgram>;

  using type = std::conditional_t<contains_type_v<REQUIRED, PATH>,
                                  fwd_save<fwd, PATH>, fwd>;
  // builds save and input refs currentyl but when it sees a let it needs to
  // also do something
};
template <typename TAG, bool RequiresGrad, typename REQUIRED, typename PATH>
struct eval<expr::FUNC_PARAM<TAG, RequiresGrad>, REQUIRED, PATH> {
  using type = std::conditional_t<contains_type_v<REQUIRED, PATH>,
                                  fwd_save<fwd_ref<TAG>, PATH>, fwd_ref<TAG>>;
};

} // namespace forward_type
} // namespace passes
