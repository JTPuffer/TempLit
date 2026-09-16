#pragma once

#include "expr/api.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <cstdint>
#include <type_traits>

namespace passes {
namespace autograd {
// allowed random program sepcific shit
template <expr::EXPR_TYPE Expr, typename Grad, typename PATH> struct eval {
  static_assert(always_false_v<Expr>,
                "No autograd rule exists for this expression type");
};
template <typename TAG, typename VAL, typename BODY, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_LET<TAG, VAL, BODY>, GRAD, PATH> {
  using val_path = graph_path<PATH, 0>;
  using body_path = graph_path<PATH, 1>;

  using body_grad =
      std::conditional_t<expr::requires_grad_v<BODY>,
                         typename eval<BODY, GRAD, body_path>::type, no_grad>;

  // dont take in body grad what you want is the accumalted func param not body
  // grad
  using val_grad =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, grad_ref<TAG>, val_path>::type,
                         no_grad>;
  using type = operations<body_grad, val_grad>;
};

template <typename GRAD, typename TAG, bool RequiresGrad, typename PATH>
struct eval<expr::FUNC_PARAM<TAG, RequiresGrad>, GRAD, PATH> {
  using type = accumulate_local_grad<GRAD, TAG>;
};

template <typename T, typename GRAD, typename PATH>
struct eval<expr::tensor::Tensor<T, expr::tensor::no_grad>, GRAD, PATH> {
  using type = no_grad;
};

template <typename T, typename GRAD, typename PATH>
struct eval<expr::tensor::Tensor<T, expr::tensor::grad>, GRAD, PATH> {

  using type = accumulate_grad<GRAD, PATH>;
};

template <auto Value, typename GRAD, typename PATH>
struct eval<expr::EXPR_CONSTANT<Value>, GRAD, PATH> {
  using type = no_grad;
};

template <expr::EXPR_TYPE LHS, expr::EXPR_TYPE RHS, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_SUB<LHS, RHS>, GRAD, PATH> {

  using lhs_path = graph_path<PATH, 0>;
  using rhs_path = graph_path<PATH, 1>;

  using lhs_grad = expr::EXPR_SUM_TO_SHAPE<GRAD, fwd_shape<lhs_path>>;

  using rhs_grad =
      expr::EXPR_HADAMARD<expr::EXPR_CONSTANT<-1>,
                          expr::EXPR_SUM_TO_SHAPE<GRAD, fwd_shape<rhs_path>>>;

  using lhs_eval =
      std::conditional_t<expr::requires_grad_v<LHS>,
                         typename eval<LHS, lhs_grad, lhs_path>::type, no_grad>;

  using rhs_eval =
      std::conditional_t<expr::requires_grad_v<RHS>,
                         typename eval<RHS, rhs_grad, rhs_path>::type, no_grad>;

  using type = operations<lhs_eval, rhs_eval>;
};

template <expr::EXPR_TYPE LHS, expr::EXPR_TYPE RHS, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_ADD<LHS, RHS>, GRAD, PATH> {

  using lhs_path = graph_path<PATH, 0>;
  using rhs_path = graph_path<PATH, 1>;

  using lhs_grad = expr::EXPR_SUM_TO_SHAPE<GRAD, fwd_shape<lhs_path>>;

  using rhs_grad = expr::EXPR_SUM_TO_SHAPE<GRAD, fwd_shape<rhs_path>>;

  using lhs_eval =
      std::conditional_t<expr::requires_grad_v<LHS>,
                         typename eval<LHS, lhs_grad, lhs_path>::type, no_grad>;

  using rhs_eval =
      std::conditional_t<expr::requires_grad_v<RHS>,
                         typename eval<RHS, rhs_grad, rhs_path>::type, no_grad>;

  using type = operations<lhs_eval, rhs_eval>;
};

template <expr::EXPR_TYPE LHS, expr::EXPR_TYPE RHS, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_MULT<LHS, RHS>, GRAD, PATH> {

  using lhs_path = graph_path<PATH, 0>;
  using rhs_path = graph_path<PATH, 1>;

  using lhs_grad = expr::EXPR_MULT_LHS_DERIVATIVE<
      GRAD, fwd_ref<rhs_path>, fwd_ref<lhs_path>>;

  using rhs_grad = expr::EXPR_MULT_RHS_DERIVATIVE<
      GRAD, fwd_ref<lhs_path>, fwd_ref<rhs_path>>;

  using lhs_eval =
      std::conditional_t<expr::requires_grad_v<LHS>,
                         typename eval<LHS, lhs_grad, lhs_path>::type, no_grad>;

  using rhs_eval =
      std::conditional_t<expr::requires_grad_v<RHS>,
                         typename eval<RHS, rhs_grad, rhs_path>::type, no_grad>;

  using type = operations<lhs_eval, rhs_eval>;
};

template <expr::EXPR_TYPE LHS, expr::EXPR_TYPE RHS, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_HADAMARD<LHS, RHS>, GRAD, PATH> {

  using lhs_path = graph_path<PATH, 0>;
  using rhs_path = graph_path<PATH, 1>;

  using lhs_grad =
      expr::EXPR_SUM_TO_SHAPE<expr::EXPR_HADAMARD<GRAD, fwd_ref<rhs_path>>,
                              fwd_shape<lhs_path>>;

  using rhs_grad =
      expr::EXPR_SUM_TO_SHAPE<expr::EXPR_HADAMARD<GRAD, fwd_ref<lhs_path>>,
                              fwd_shape<rhs_path>>;

  using lhs_eval =
      std::conditional_t<expr::requires_grad_v<LHS>,
                         typename eval<LHS, lhs_grad, lhs_path>::type, no_grad>;

  using rhs_eval =
      std::conditional_t<expr::requires_grad_v<RHS>,
                         typename eval<RHS, rhs_grad, rhs_path>::type, no_grad>;

  using type = operations<lhs_eval, rhs_eval>;
};

template <expr::EXPR_TYPE VAL, typename GRAD, typename PATH>
struct eval<expr::EXPR_SIG<VAL>, GRAD, PATH> {

  using val_path = graph_path<PATH, 0>;

  using val_grad = expr::EXPR_HADAMARD<
      expr::EXPR_HADAMARD<GRAD, fwd_ref<PATH>>,
      expr::EXPR_SUB<expr::EXPR_CONSTANT<1>, fwd_ref<PATH>>>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;

  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE LHS, expr::EXPR_TYPE RHS, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_DIV<LHS, RHS>, GRAD, PATH> {

  using lhs_path = graph_path<PATH, 0>;
  using rhs_path = graph_path<PATH, 1>;

  using lhs_grad =
      expr::EXPR_SUM_TO_SHAPE<expr::EXPR_DIV<GRAD, fwd_ref<rhs_path>>,
                              fwd_shape<lhs_path>>;

  using rhs_grad = expr::EXPR_SUM_TO_SHAPE<
      expr::EXPR_HADAMARD<
          expr::EXPR_CONSTANT<-1>,
          expr::EXPR_DIV<
              expr::EXPR_HADAMARD<GRAD, fwd_ref<lhs_path>>,
              expr::EXPR_HADAMARD<fwd_ref<rhs_path>, fwd_ref<rhs_path>>>>,
      fwd_shape<rhs_path>>;

  using lhs_eval =
      std::conditional_t<expr::requires_grad_v<LHS>,
                         typename eval<LHS, lhs_grad, lhs_path>::type, no_grad>;

  using rhs_eval =
      std::conditional_t<expr::requires_grad_v<RHS>,
                         typename eval<RHS, rhs_grad, rhs_path>::type, no_grad>;

  using type = operations<lhs_eval, rhs_eval>;
};

template <expr::EXPR_TYPE VAL, typename GRAD, typename PATH>
struct eval<expr::EXPR_SFTMX<VAL>, GRAD, PATH> {

  using val_path = graph_path<PATH, 0>;

  using val_grad = expr::EXPR_SFTMX_DERIVATIVE<GRAD, fwd_ref<PATH>>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;

  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE VAL, typename GRAD, typename PATH>
struct eval<expr::EXPR_SQRT<VAL>, GRAD, PATH> {

  using val_path = graph_path<PATH, 0>;

  using val_grad = expr::EXPR_DIV<
      GRAD, expr::EXPR_HADAMARD<expr::EXPR_CONSTANT<2>, fwd_ref<PATH>>>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;

  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE VAL, int32_t DimA, int32_t DimB, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_TRAN<VAL, DimA, DimB>, GRAD, PATH> {

  using val_path = graph_path<PATH, 0>;

  using val_grad = expr::EXPR_TRAN<GRAD, DimA, DimB>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;

  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE VAL, uint32_t Outer, uint32_t Inner, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_SPLIT_LAST_DIM<VAL, Outer, Inner>, GRAD, PATH> {
  using val_path = graph_path<PATH, 0>;
  using val_grad = expr::EXPR_MERGE_LAST_DIMS<GRAD, Outer, Inner>;
  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;
  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE VAL, uint32_t Outer, uint32_t Inner, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_MERGE_LAST_DIMS<VAL, Outer, Inner>, GRAD, PATH> {
  using val_path = graph_path<PATH, 0>;
  using val_grad = expr::EXPR_SPLIT_LAST_DIM<GRAD, Outer, Inner>;
  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;
  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE VAL, expr::EXPR_TENSOR IDS, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_EMBEDDING_LOOKUP<VAL, IDS>, GRAD, PATH> {
  using val_path = graph_path<PATH, 0>;
  using ids_path = graph_path<PATH, 1>;
  using val_grad =
      expr::EXPR_SCATTER_INDEX<GRAD, index_ref<ids_path>, fwd_shape<val_path>>;
  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;
  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE BASE, expr::EXPR_TENSOR IDS, expr::EXPR_TYPE VALUES,
          typename GRAD, typename PATH>
struct eval<expr::EXPR_SCATTER_ADD_ROWS<BASE, IDS, VALUES>, GRAD, PATH> {
  using base_path = graph_path<PATH, 0>;
  using ids_path = graph_path<PATH, 1>;
  using values_path = graph_path<PATH, 2>;

  using base_eval =
      std::conditional_t<expr::requires_grad_v<BASE>,
                         typename eval<BASE, GRAD, base_path>::type, no_grad>;
  using values_grad = expr::EXPR_EMBEDDING_LOOKUP<GRAD, index_ref<ids_path>>;
  using values_eval =
      std::conditional_t<expr::requires_grad_v<VALUES>,
                         typename eval<VALUES, values_grad, values_path>::type,
                         no_grad>;

  using type = operations<base_eval, values_eval>;
};

template <expr::EXPR_TYPE VALUES, expr::EXPR_TENSOR IDS, typename SHAPE,
          typename GRAD, typename PATH>
struct eval<expr::EXPR_SCATTER_INDEX<VALUES, IDS, SHAPE>, GRAD, PATH> {
  using values_path = graph_path<PATH, 0>;
  using ids_path = graph_path<PATH, 1>;
  using values_grad = expr::EXPR_EMBEDDING_LOOKUP<GRAD, index_ref<ids_path>>;
  using values_eval =
      std::conditional_t<expr::requires_grad_v<VALUES>,
                         typename eval<VALUES, values_grad, values_path>::type,
                         no_grad>;

  using type = operations<values_eval>;
};

template <expr::EXPR_TYPE PRED, expr::EXPR_TYPE TARGET, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_SQE<PRED, TARGET>, GRAD, PATH> {

  using pred_path = graph_path<PATH, 0>;
  using target_path = graph_path<PATH, 1>;

  using base_grad = expr::EXPR_HADAMARD<
      GRAD, expr::EXPR_HADAMARD<
                expr::EXPR_CONSTANT<2>,
                expr::EXPR_SUB<fwd_ref<pred_path>, fwd_ref<target_path>>>>;

  using pred_grad = expr::EXPR_SUM_TO_SHAPE<base_grad, fwd_shape<pred_path>>;

  using target_grad = expr::EXPR_SUM_TO_SHAPE<
      expr::EXPR_HADAMARD<expr::EXPR_CONSTANT<-1>, base_grad>,
      fwd_shape<target_path>>;

  using pred_eval =
      std::conditional_t<expr::requires_grad_v<PRED>,
                         typename eval<PRED, pred_grad, pred_path>::type,
                         no_grad>;

  using target_eval =
      std::conditional_t<expr::requires_grad_v<TARGET>,
                         typename eval<TARGET, target_grad, target_path>::type,
                         no_grad>;

  using type = operations<pred_eval, target_eval>;
};

template <expr::EXPR_TYPE PRED, expr::EXPR_TENSOR TARGET, typename GRAD,
          typename PATH>
struct eval<expr::EXPR_X_ENTROPY<PRED, TARGET>, GRAD, PATH> {
  using pred_path = graph_path<PATH, 0>;
  using target_path = graph_path<PATH, 1>;

  using derivative = expr::EXPR_X_ENTROPY_DERIVATIVE<
      fwd_ref<pred_path>, index_ref<target_path>>;
  using pred_grad = expr::EXPR_HADAMARD<GRAD, derivative>;

  using pred_eval =
      std::conditional_t<expr::requires_grad_v<PRED>,
                         typename eval<PRED, pred_grad, pred_path>::type,
                         no_grad>;

  using type = operations<pred_eval>;
};

template <expr::EXPR_TYPE VAL, typename GRAD, typename PATH>
struct eval<expr::EXPR_RMS<VAL>, GRAD, PATH> {
  using val_path = graph_path<PATH, 0>;

  using derivative =
      expr::EXPR_DIV<expr::EXPR_DIV<fwd_ref<val_path>, fwd_ref<PATH>>,
                     fwd_last_dim<val_path>>;
  using val_grad = expr::EXPR_HADAMARD<GRAD, derivative>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;
  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE VAL, typename GRAD, typename PATH>
struct eval<expr::EXPR_RMS_NORM<VAL>, GRAD, PATH> {
  using val_path = graph_path<PATH, 0>;

  using val_grad = expr::EXPR_RMSN_DERIVATIVE<
      GRAD, fwd_ref<val_path>, fwd_ref<PATH>>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;

  using type = operations<val_eval>;
};
template <expr::EXPR_TYPE VAL, typename GRAD, typename PATH>
struct eval<expr::EXPR_GELU<VAL>, GRAD, PATH> {

  using val_path = graph_path<PATH, 0>;

  using derivative = expr::EXPR_GELU_DERIVATIVE<fwd_ref<val_path>>;
  using val_grad = expr::EXPR_HADAMARD<GRAD, derivative>;

  using val_eval =
      std::conditional_t<expr::requires_grad_v<VAL>,
                         typename eval<VAL, val_grad, val_path>::type, no_grad>;

  using type = operations<val_eval>;
};

template <expr::EXPR_TYPE INPUT, expr::EXPR_TYPE WEIGHT,
          expr::EXPR_TYPE BIAS, typename GRAD, typename PATH>
struct eval<expr::EXPR_LINEAR<INPUT, WEIGHT, BIAS>, GRAD, PATH> {
  using input_path = graph_path<PATH, 0>;
  using weight_path = graph_path<PATH, 1>;
  using bias_path = graph_path<PATH, 2>;

  using input_grad = expr::EXPR_MULT_LHS_DERIVATIVE<
      GRAD, fwd_ref<weight_path>, fwd_ref<input_path>>;

  using weight_grad = expr::EXPR_MULT_RHS_DERIVATIVE<
      GRAD, fwd_ref<input_path>, fwd_ref<weight_path>>;

  using bias_grad =
      expr::EXPR_SUM_TO_SHAPE<GRAD, fwd_shape<bias_path>>;

  using input_eval =
      std::conditional_t<expr::requires_grad_v<INPUT>,
                         typename eval<INPUT, input_grad, input_path>::type,
                         no_grad>;

  using weight_eval =
      std::conditional_t<expr::requires_grad_v<WEIGHT>,
                         typename eval<WEIGHT, weight_grad, weight_path>::type,
                         no_grad>;

  using bias_eval =
      std::conditional_t<expr::requires_grad_v<BIAS>,
                         typename eval<BIAS, bias_grad, bias_path>::type,
                         no_grad>;

  using type = operations<input_eval, weight_eval, bias_eval>;
};


} // namespace autograd
} // namespace passes
