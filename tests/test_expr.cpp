#include "expr/api.h"
#include <concepts>
#include <utility>

int main() {
  using Leaf = expr::tensor::Tensor<int>;

  static_assert(std::same_as<decltype(expr::sub(std::declval<Leaf>(),
                                                std::declval<Leaf>())),
                             expr::EXPR_SUB<Leaf, Leaf>>);

  static_assert(
      std::same_as<decltype(std::declval<Leaf>() - std::declval<Leaf>()),
                   expr::EXPR_SUB<Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::hadamard(std::declval<Leaf>(),
                                                     std::declval<Leaf>())),
                             expr::EXPR_HADAMARD<Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::divide(std::declval<Leaf>(),
                                                   std::declval<Leaf>())),
                             expr::EXPR_DIV<Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::softmax(std::declval<Leaf>())),
                             expr::EXPR_SFTMX<Leaf>>);
  static_assert(std::same_as<
                expr::EXPR_SFTMX_DERIVATIVE<Leaf, Leaf>,
                expr::ExprNode<expr::SFTMX_DERIVATIVE, Leaf, Leaf>>);

  static_assert(std::same_as<
                expr::EXPR_MULT_LHS_DERIVATIVE<Leaf, Leaf, Leaf>,
                expr::ExprNode<expr::MULT_LHS_DERIVATIVE, Leaf, Leaf, Leaf>>);
  static_assert(std::same_as<
                expr::EXPR_MULT_RHS_DERIVATIVE<Leaf, Leaf, Leaf>,
                expr::ExprNode<expr::MULT_RHS_DERIVATIVE, Leaf, Leaf, Leaf>>);

  static_assert(std::same_as<
                decltype(expr::pow(std::declval<Leaf>(),
                                   expr::EXPR_CONSTANT<3>{})),
                expr::EXPR_POW<Leaf, 3>>);

  static_assert(std::same_as<decltype(expr::rms(std::declval<Leaf>())),
                             expr::EXPR_RMS<Leaf>>);

  static_assert(std::same_as<decltype(expr::rms_norm(std::declval<Leaf>())),
                             expr::EXPR_RMS_NORM<Leaf>>);
  static_assert(std::same_as<
                expr::EXPR_RMSN_DERIVATIVE<Leaf, Leaf, Leaf>,
                expr::ExprNode<expr::RMSN_DERIVATIVE, Leaf, Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::gelu(std::declval<Leaf>())),
                             expr::EXPR_GELU<Leaf>>);
  static_assert(
      std::same_as<expr::EXPR_GELU_DERIVATIVE<Leaf>,
                   expr::ExprNode<expr::GELU_DERIVATIVE, Leaf>>);

  static_assert(std::same_as<
                decltype(expr::loss::cross_entropy(std::declval<Leaf>(),
                                                   std::declval<Leaf>())),
                expr::EXPR_X_ENTROPY<Leaf, Leaf>>);
  static_assert(std::same_as<
                expr::EXPR_X_ENTROPY_DERIVATIVE<Leaf, Leaf>,
                expr::ExprNode<expr::X_ENTROPY_DERIVATIVE, Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::gather_rows(
                                 std::declval<Leaf>(), std::declval<Leaf>())),
                             expr::EXPR_GATHER_ROWS<Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::gather_rows(
                                 std::declval<Leaf &>(),
                                 std::declval<Leaf &>())),
                             expr::EXPR_GATHER_ROWS<Leaf, Leaf>>);

  static_assert(std::same_as<decltype(expr::embedding_lookup(
                                 std::declval<Leaf>(), std::declval<Leaf>())),
                             expr::EXPR_EMBEDDING_LOOKUP<Leaf, Leaf>>);

  static_assert(std::same_as<
                decltype(expr::scatter_add_rows(std::declval<Leaf>(),
                                                std::declval<Leaf>(),
                                                std::declval<Leaf>())),
                expr::EXPR_SCATTER_ADD_ROWS<Leaf, Leaf, Leaf>>);

  static_assert(std::same_as<
                decltype(expr::scatter_index(std::declval<Leaf>(),
                                             std::declval<Leaf>(),
                                             expr::shape_of<Leaf>{})),
                expr::EXPR_SCATTER_INDEX<Leaf, Leaf,
                                         expr::shape_of<Leaf>>>);

  static_assert(
      std::same_as<decltype(expr::sum_to_shape(std::declval<Leaf>(),
                                               expr::shape_of<Leaf>{})),
                   expr::EXPR_SUM_TO_SHAPE<Leaf, expr::shape_of<Leaf>>>);

  return 0;
}
