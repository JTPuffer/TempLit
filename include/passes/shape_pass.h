#pragma once

#include "RTensor/infer_shape.h"
#include "expr/api.h"
#include "passes/concepts.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace passes {
namespace shape_pass {

template <typename Indices> struct BackwardContext {
  Indices &indices;
  const rtensor::shape_type &root_grad_shape;
};

template <typename DEST_PATH, typename Shapes>
  requires ContextForPath<Shapes, DEST_PATH>
const rtensor::shape_type &save_shape(Shapes &shapes,
                                      rtensor::shape_type shape) {
  auto &stored = shapes.template get_node<DEST_PATH>();
  stored = std::move(shape);
  return stored;
}

template <typename Expr, typename PROG_PATH> struct eval_expr {
  template <typename Context, typename ProgramShapes, typename SavedShapes>
  static void run(Context &, ProgramShapes &, SavedShapes &) {
    static_assert(always_false_v<Expr>,
                  "No shape inference rule exists for this program type");
  }
};

template <typename PROG_PATH, typename Ops, typename Indices>
struct eval_operations;

template <typename PROG_PATH, typename... Ops, size_t... Is>
struct eval_operations<PROG_PATH, type_list<Ops...>,
                       std::index_sequence<Is...>> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static void run(Ctx &ctx, ProgramShapes &program_shapes,
                  SavedShapes &saved_shapes) {
    (eval_expr<Ops, prog_path<PROG_PATH, Is>>::run(ctx, program_shapes,
                                                   saved_shapes),
     ...);
  }
};

template <typename... Ops, typename PROG_PATH>
struct eval_expr<operations<Ops...>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static void run(Ctx &ctx, ProgramShapes &program_shapes,
                  SavedShapes &saved_shapes) {
    eval_operations<PROG_PATH, type_list<Ops...>,
                    std::index_sequence_for<Ops...>>::run(ctx, program_shapes,
                                                          saved_shapes);
  }
};

template <typename PROG1, typename PROG2, typename PROG_PATH>
struct eval_expr<prog_link<PROG1, PROG2>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using b_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {

    eval_expr<PROG1, a_prog_path>::run(ctx, program_shapes, saved_shapes);

    const auto &prog2_shape =
        eval_expr<PROG2, b_prog_path>::run(ctx, program_shapes, saved_shapes);

    return save_shape<PROG_PATH>(program_shapes, prog2_shape);
  }
};

template <typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<fwd_ref<GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires ContextForPath<SavedShapes, GRAPH_PATH> &&
             ContextForPath<ProgramShapes, PROG_PATH>
  static const auto &run(Ctx &, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    return save_shape<PROG_PATH>(program_shapes,
                                 saved_shapes.template get_node<GRAPH_PATH>());
  }
};

template <typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<fwd_shape<GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires ContextForPath<SavedShapes, GRAPH_PATH> &&
             ContextForPath<ProgramShapes, PROG_PATH>
  static const auto &run(Ctx &, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    return save_shape<PROG_PATH>(program_shapes,
                                 saved_shapes.template get_node<GRAPH_PATH>());
  }
};

template <typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<fwd_last_dim<GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires ContextForPath<SavedShapes, GRAPH_PATH> &&
             ContextForPath<ProgramShapes, PROG_PATH>
  static const auto &run(Ctx &, ProgramShapes &program_shapes, SavedShapes &) {
    return save_shape<PROG_PATH>(program_shapes, rtensor::shape_type{1, 1});
  }
};

template <typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<grad_ref<GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires ContextForPath<SavedShapes, GRAPH_PATH> &&
             ContextForPath<ProgramShapes, PROG_PATH>
  static const auto &run(Ctx &, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    return save_shape<PROG_PATH>(program_shapes,
                                 saved_shapes.template get_node<GRAPH_PATH>());
  }
};

template <typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<index_ref<GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires IndexNodeFor<Ctx, GRAPH_PATH> &&
             ContextForPath<ProgramShapes, PROG_PATH>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &) {
    return save_shape<PROG_PATH>(
        program_shapes, ctx.indices.template get_node<GRAPH_PATH>().shape());
  }
};

template <typename EXPR, typename TEMP_PATH, typename PROG_PATH>
struct eval_expr<temp_ref<EXPR, TEMP_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    static_assert(std::same_as<TEMP_PATH, PROG_PATH>,
                  "temp_ref path does not match program traversal path");
    return eval_expr<EXPR, PROG_PATH>::run(ctx, program_shapes, saved_shapes);
  }
};

template <typename EXPR, typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<fwd_save<EXPR, GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires ContextForPath<SavedShapes, GRAPH_PATH>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<EXPR, PROG_PATH>::run(ctx, program_shapes, saved_shapes);
    save_shape<GRAPH_PATH>(saved_shapes, shape);
    return shape;
  }
};

template <typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<input_ref<GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
    requires InputNodeFor<Ctx, GRAPH_PATH>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &) {
    return save_shape<PROG_PATH>(
        program_shapes, ctx.input.template get_node<GRAPH_PATH>().shape());
  }
};

template <typename GRAD, typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<accumulate_grad<GRAD, GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    return eval_expr<GRAD, prog_path<PROG_PATH, 0>>::run(ctx, program_shapes,
                                                         saved_shapes);
  }
};

template <typename GRAD, typename GRAPH_PATH, typename PROG_PATH>
struct eval_expr<accumulate_local_grad<GRAD, GRAPH_PATH>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    return eval_expr<GRAD, prog_path<PROG_PATH, 0>>::run(ctx, program_shapes,
                                                         saved_shapes);
  }
};

template <typename PROG_PATH> struct eval_expr<grad_seed, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &) {
    return save_shape<PROG_PATH>(program_shapes, ctx.root_grad_shape);
  }
};

template <typename PROG_PATH> struct eval_expr<no_grad, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static void run(Ctx &, ProgramShapes &, SavedShapes &) {}
};

template <typename A, typename SHAPE, typename PROG_PATH>
struct eval_expr<expr::EXPR_SUM_TO_SHAPE<A, SHAPE>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using shape_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &a_shape =
        eval_expr<A, a_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &output_shape = eval_expr<SHAPE, shape_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes, rtensor::infer_sum_to_shape(a_shape, output_shape));
  }
};

template <typename A, typename B, typename PROG_PATH>
struct eval_expr<expr::EXPR_MULT<A, B>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using b_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &a_shape =
        eval_expr<A, a_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &b_shape =
        eval_expr<B, b_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_mult(a_shape, b_shape));
  }
};

template <typename GRAD, typename RHS, typename LHS, typename PROG_PATH>
struct eval_expr<expr::EXPR_MULT_LHS_DERIVATIVE<GRAD, RHS, LHS>, PROG_PATH> {
  using grad_prog_path = prog_path<PROG_PATH, 0>;
  using rhs_prog_path = prog_path<PROG_PATH, 1>;
  using lhs_prog_path = prog_path<PROG_PATH, 2>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    eval_expr<GRAD, grad_prog_path>::run(ctx, program_shapes, saved_shapes);
    eval_expr<RHS, rhs_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &lhs_shape = eval_expr<LHS, lhs_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, lhs_shape);
  }
};

template <typename GRAD, typename LHS, typename RHS, typename PROG_PATH>
struct eval_expr<expr::EXPR_MULT_RHS_DERIVATIVE<GRAD, LHS, RHS>, PROG_PATH> {
  using grad_prog_path = prog_path<PROG_PATH, 0>;
  using lhs_prog_path = prog_path<PROG_PATH, 1>;
  using rhs_prog_path = prog_path<PROG_PATH, 2>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    eval_expr<GRAD, grad_prog_path>::run(ctx, program_shapes, saved_shapes);
    eval_expr<LHS, lhs_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &rhs_shape = eval_expr<RHS, rhs_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, rhs_shape);
  }
};

template <typename A, typename B, typename PROG_PATH>
struct eval_expr<expr::EXPR_ADD<A, B>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using b_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &a_shape =
        eval_expr<A, a_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &b_shape =
        eval_expr<B, b_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_add(a_shape, b_shape));
  }
};

template <typename INPUT, typename WEIGHT, typename BIAS, typename PROG_PATH>
struct eval_expr<expr::EXPR_LINEAR<INPUT, WEIGHT, BIAS>, PROG_PATH> {
  using input_prog_path = prog_path<PROG_PATH, 0>;
  using weight_prog_path = prog_path<PROG_PATH, 1>;
  using bias_prog_path = prog_path<PROG_PATH, 2>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &input_shape = eval_expr<INPUT, input_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    const auto &weight_shape = eval_expr<WEIGHT, weight_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    const auto &bias_shape = eval_expr<BIAS, bias_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    const auto output_shape = rtensor::infer_mult(input_shape, weight_shape);
    return save_shape<PROG_PATH>(
        program_shapes, rtensor::infer_add(output_shape, bias_shape));
  }
};

template <typename A, typename B, typename PROG_PATH>
struct eval_expr<expr::EXPR_SUB<A, B>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using b_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &a_shape =
        eval_expr<A, a_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &b_shape =
        eval_expr<B, b_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_sub(a_shape, b_shape));
  }
};

template <typename A, typename B, typename PROG_PATH>
struct eval_expr<expr::EXPR_HADAMARD<A, B>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using b_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &a_shape =
        eval_expr<A, a_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &b_shape =
        eval_expr<B, b_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_hadamard(a_shape, b_shape));
  }
};

template <typename A, typename B, typename PROG_PATH>
struct eval_expr<expr::EXPR_DIV<A, B>, PROG_PATH> {
  using a_prog_path = prog_path<PROG_PATH, 0>;
  using b_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &a_shape =
        eval_expr<A, a_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &b_shape =
        eval_expr<B, b_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_divide(a_shape, b_shape));
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_SIG<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, rtensor::infer_sigmoid(shape));
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_SFTMX<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, rtensor::infer_softmax(shape));
  }
};

template <typename GRAD, typename OUTPUT, typename PROG_PATH>
struct eval_expr<expr::EXPR_SFTMX_DERIVATIVE<GRAD, OUTPUT>, PROG_PATH> {
  using grad_prog_path = prog_path<PROG_PATH, 0>;
  using output_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    eval_expr<GRAD, grad_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &output_shape = eval_expr<OUTPUT, output_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, output_shape);
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_SQRT<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, rtensor::infer_sqrt(shape));
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_RMS<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, rtensor::infer_rms(shape));
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_RMS_NORM<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_rms_norm(shape));
  }
};

template <typename GRAD, typename VAL, typename OUTPUT, typename PROG_PATH>
struct eval_expr<expr::EXPR_RMSN_DERIVATIVE<GRAD, VAL, OUTPUT>, PROG_PATH> {
  using grad_prog_path = prog_path<PROG_PATH, 0>;
  using val_prog_path = prog_path<PROG_PATH, 1>;
  using output_prog_path = prog_path<PROG_PATH, 2>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    eval_expr<GRAD, grad_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &val_shape = eval_expr<VAL, val_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    eval_expr<OUTPUT, output_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, val_shape);
  }
};

template <typename VAL, auto Exponent, typename PROG_PATH>
struct eval_expr<expr::EXPR_POW<VAL, Exponent>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, shape);
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_GELU<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, shape);
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_GELU_DERIVATIVE<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes, shape);
  }
};

template <typename VAL, typename PROG_PATH>
struct eval_expr<expr::EXPR_SUM<VAL>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_sum(shape, shape.size() - 1));
  }
};

template <typename VAL, int32_t DimA, int32_t DimB, typename PROG_PATH>
struct eval_expr<expr::EXPR_TRAN<VAL, DimA, DimB>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(program_shapes,
                                 rtensor::infer_transpose(shape, DimA, DimB));
  }
};

template <typename VAL, uint32_t Outer, uint32_t Inner, typename PROG_PATH>
struct eval_expr<expr::EXPR_SPLIT_LAST_DIM<VAL, Outer, Inner>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes, rtensor::infer_split_last_dim(shape, Outer, Inner));
  }
};

template <typename VAL, uint32_t Outer, uint32_t Inner, typename PROG_PATH>
struct eval_expr<expr::EXPR_MERGE_LAST_DIMS<VAL, Outer, Inner>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes, rtensor::infer_merge_last_dims(shape, Outer, Inner));
  }
};

template <typename VAL, typename IDS, typename PROG_PATH>
struct eval_expr<expr::EXPR_EMBEDDING_LOOKUP<VAL, IDS>, PROG_PATH> {
  using val_prog_path = prog_path<PROG_PATH, 0>;
  using ids_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &val_shape =
        eval_expr<VAL, val_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &ids_shape =
        eval_expr<IDS, ids_prog_path>::run(ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes, rtensor::infer_embedding_lookup(val_shape, ids_shape));
  }
};

template <typename BASE, typename IDS, typename VALUES, typename PROG_PATH>
struct eval_expr<expr::EXPR_SCATTER_ADD_ROWS<BASE, IDS, VALUES>, PROG_PATH> {
  using base_prog_path = prog_path<PROG_PATH, 0>;
  using ids_prog_path = prog_path<PROG_PATH, 1>;
  using values_prog_path = prog_path<PROG_PATH, 2>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &base_shape =
        eval_expr<BASE, base_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &ids_shape =
        eval_expr<IDS, ids_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &values_shape = eval_expr<VALUES, values_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes,
        rtensor::infer_scatter_add_rows(base_shape, ids_shape, values_shape));
  }
};

template <typename VALUES, typename IDS, typename SHAPE, typename PROG_PATH>
struct eval_expr<expr::EXPR_SCATTER_INDEX<VALUES, IDS, SHAPE>, PROG_PATH> {
  using values_prog_path = prog_path<PROG_PATH, 0>;
  using ids_prog_path = prog_path<PROG_PATH, 1>;
  using shape_prog_path = prog_path<PROG_PATH, 2>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &values_shape = eval_expr<VALUES, values_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    const auto &ids_shape =
        eval_expr<IDS, ids_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &output_shape = eval_expr<SHAPE, shape_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes,
        rtensor::infer_scatter_index(values_shape, ids_shape, output_shape));
  }
};

template <typename PRED, typename TARGET, typename PROG_PATH>
struct eval_expr<expr::EXPR_SQE<PRED, TARGET>, PROG_PATH> {
  using pred_prog_path = prog_path<PROG_PATH, 0>;
  using target_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &pred_shape =
        eval_expr<PRED, pred_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &target_shape = eval_expr<TARGET, target_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    auto diff_shape = rtensor::infer_sub(pred_shape, target_shape);
    return save_shape<PROG_PATH>(
        program_shapes, rtensor::infer_hadamard(diff_shape, diff_shape));
  }
};

template <typename PRED, typename TARGET, typename PROG_PATH>
struct eval_expr<expr::EXPR_X_ENTROPY<PRED, TARGET>, PROG_PATH> {
  using pred_prog_path = prog_path<PROG_PATH, 0>;
  using target_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &pred_shape =
        eval_expr<PRED, pred_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &target_shape = eval_expr<TARGET, target_prog_path>::run(
        ctx, program_shapes, saved_shapes);

    return save_shape<PROG_PATH>(
        program_shapes,
        rtensor::infer_cross_entropy(pred_shape, target_shape));
  }
};

template <typename PRED, typename TARGET, typename PROG_PATH>
struct eval_expr<expr::EXPR_X_ENTROPY_DERIVATIVE<PRED, TARGET>, PROG_PATH> {
  using pred_prog_path = prog_path<PROG_PATH, 0>;
  using target_prog_path = prog_path<PROG_PATH, 1>;

  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &ctx, ProgramShapes &program_shapes,
                         SavedShapes &saved_shapes) {
    const auto &pred_shape =
        eval_expr<PRED, pred_prog_path>::run(ctx, program_shapes, saved_shapes);
    const auto &target_shape = eval_expr<TARGET, target_prog_path>::run(
        ctx, program_shapes, saved_shapes);
    return save_shape<PROG_PATH>(
        program_shapes,
        rtensor::infer_cross_entropy_derivative(pred_shape, target_shape));
  }
};

template <auto VALUE, typename PROG_PATH>
struct eval_expr<expr::EXPR_CONSTANT<VALUE>, PROG_PATH> {
  template <typename Ctx, typename ProgramShapes, typename SavedShapes>
  static const auto &run(Ctx &, ProgramShapes &program_shapes, SavedShapes &) {
    return save_shape<PROG_PATH>(program_shapes, {1, 1});
  }
};

} // namespace shape_pass
} // namespace passes
