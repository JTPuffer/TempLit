#pragma once
#include "RTensor/infer_shape.h"
#include "expr/api.h"
#include "passes/concepts.h"
#include "passes/get_at_path.h"
#include "passes/meta.h"
#include "passes/program.h"

#include <array>
#include <type_traits>
#include <utility>

namespace passes {
namespace exec {

template <typename Expr> struct eval_expr {
  template <typename Run, typename Context, typename Destination>
  static void run(Run &, Context &, Destination &) {
    static_assert(always_false_v<Expr>,
                  "No executor rule exists for this program type");
  }
};

template <typename... Ops> struct eval_expr<operations<Ops...>> {
  template <typename Run, typename Ctx>
  static auto run(Run &runtime, Ctx &ctx) {
    (eval_expr<Ops>::run(runtime, ctx), ...);
  }
};
template <typename PROG1, typename PROG2>
struct eval_expr<prog_link<PROG1, PROG2>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    eval_expr<PROG1>::run(runtime, ctx);

    return eval_expr<PROG2>::run(runtime, ctx, destination);
  }
};
template <typename PATH> struct eval_expr<fwd_ref<PATH>> {
  template <typename Run, typename Ctx>
    requires ForwardNodeFor<Ctx, PATH>
  static auto &run(Run &, Ctx &ctx) {
    return ctx.forward.template get_node<PATH>();
  }

  template <typename Run, typename Ctx, typename Destination>
    requires ForwardNodeFor<Ctx, PATH>
  static auto &run(Run &, Ctx &ctx, Destination &destination) {
    destination = ctx.forward.template get_node<PATH>();
    return destination;
  }
};

template <typename PATH> struct eval_expr<fwd_shape<PATH>> {
  template <typename Run, typename Ctx>
    requires ForwardNodeFor<Ctx, PATH>
  static auto &run(Run &, Ctx &ctx) {
    return ctx.forward.template get_node<PATH>().shape();
  }
};

template <typename PATH> struct eval_expr<fwd_last_dim<PATH>> {
  template <typename Run, typename Ctx>
    requires ForwardNodeFor<Ctx, PATH>
  static auto run(Run &, Ctx &ctx) {
    const auto &value = ctx.forward.template get_node<PATH>();
    using Tensor = std::remove_cvref_t<decltype(value)>;
    return static_cast<typename Tensor::value_type>(value.shape().back());
  }
};

template <typename EXPR, typename PATH> struct eval_expr<fwd_save<EXPR, PATH>> {
  template <typename Run, typename Ctx>
    requires ForwardNodeFor<Ctx, PATH>
  static auto &run(Run &runtime, Ctx &ctx) {
    auto &destination = ctx.forward.template get_node<PATH>();
    eval_expr<EXPR>::run(runtime, ctx, destination);
    return destination;
  }

  template <typename Run, typename Ctx, typename Destination>
    requires ForwardNodeFor<Ctx, PATH>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    eval_expr<EXPR>::run(runtime, ctx, destination);
    ctx.forward.template get_node<PATH>() = destination;
    return destination;
  }
};

template <typename EXPR, typename PATH> struct eval_expr<temp_ref<EXPR, PATH>> {
  template <typename Run, typename Ctx>
    requires ContextForPath<decltype(std::declval<Ctx &>().temporary), PATH>
  static auto &run(Run &runtime, Ctx &ctx) {
    auto &destination = ctx.temporary.template get_node<PATH>();
    eval_expr<EXPR>::run(runtime, ctx, destination);
    return destination;
  }
};

template <typename PATH> struct eval_expr<input_ref<PATH>> {
  template <typename Run, typename Ctx>
    requires InputNodeFor<Ctx, PATH>
  static auto &run(Run &, Ctx &ctx) {
    return ctx.input.template get_node<PATH>();
  }

  template <typename Run, typename Ctx, typename Destination>
    requires InputNodeFor<Ctx, PATH>
  static auto &run(Run &, Ctx &ctx, Destination &destination) {
    destination = ctx.input.template get_node<PATH>();
    return destination;
  }
};

template <typename PATH> struct eval_expr<index_ref<PATH>> {
  template <typename Run, typename Ctx>
    requires IndexNodeFor<Ctx, PATH>
  static auto &run(Run &, Ctx &ctx) {
    return ctx.indices.template get_node<PATH>();
  }
};

template <typename GRAD, typename PATH>
struct eval_expr<accumulate_grad<GRAD, PATH>> {
  template <typename Run, typename Ctx>
    requires GradientNodeFor<Ctx, PATH, Run>
  static void run(Run &runtime, Ctx &ctx) {
    decltype(auto) grad = eval_expr<GRAD>::run(runtime, ctx);
    ctx.gradients.template accumulate<PATH>(runtime, grad);
  }
};

template <typename GRAD, typename PATH>
struct eval_expr<accumulate_local_grad<GRAD, PATH>> {
  template <typename Run, typename Ctx>
    requires LocalGradientNodeFor<Ctx, PATH, Run>
  static void run(Run &runtime, Ctx &ctx) {
    decltype(auto) grad = eval_expr<GRAD>::run(runtime, ctx);
    ctx.local_gradients.template accumulate<PATH>(runtime, grad);
  }
};

template <typename PATH> struct eval_expr<grad_ref<PATH>> {
  template <typename Run, typename Ctx>
  static const auto &run(Run &, Ctx &ctx) {
    return ctx.local_gradients.template get_node<PATH>();
  }
};

template <> struct eval_expr<grad_seed> {
  template <typename Run, typename Ctx>
    requires RootGradientContext<Ctx>
  static const auto &run(Run &, Ctx &ctx) {
    return ctx.root_grad;
  }

  template <typename Run, typename Ctx, typename Destination>
    requires RootGradientContext<Ctx>
  static auto &run(Run &, Ctx &ctx, Destination &destination) {
    destination = ctx.root_grad;
    return destination;
  }
};

template <> struct eval_expr<no_grad> {
  template <typename Run, typename Ctx> static void run(Run &, Ctx &) {}
};

template <typename A, typename SHAPE>
struct eval_expr<expr::EXPR_SUM_TO_SHAPE<A, SHAPE>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.sum_to_shape(eval_expr<A>::run(runtime, ctx),
                         eval_expr<SHAPE>::run(runtime, ctx), destination);
    return destination;
  };
};

template <typename A, typename B> struct eval_expr<expr::EXPR_MULT<A, B>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.mult(eval_expr<A>::run(runtime, ctx),
                 eval_expr<B>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename GRAD, typename RHS, typename LHS>
struct eval_expr<expr::EXPR_MULT_LHS_DERIVATIVE<GRAD, RHS, LHS>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    const auto &grad = eval_expr<GRAD>::run(runtime, ctx);
    const auto &rhs = eval_expr<RHS>::run(runtime, ctx);

    auto transposed_shape = rtensor::infer_transpose(rhs.shape());
    Destination transposed_rhs(transposed_shape);
    runtime.transpose(rhs, transposed_rhs);

    auto product_shape =
        rtensor::infer_mult(grad.shape(), transposed_rhs.shape());
    if (product_shape == destination.shape()) {
      runtime.mult(grad, transposed_rhs, destination);
    } else {
      Destination product(product_shape);
      runtime.mult(grad, transposed_rhs, product);
      runtime.sum_to_shape(product, destination.shape(), destination);
    }
    return destination;
  };
};
template <typename GRAD, typename LHS, typename RHS>
struct eval_expr<expr::EXPR_MULT_RHS_DERIVATIVE<GRAD, LHS, RHS>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    const auto &grad = eval_expr<GRAD>::run(runtime, ctx);
    const auto &lhs = eval_expr<LHS>::run(runtime, ctx);

    auto transposed_shape = rtensor::infer_transpose(lhs.shape());
    Destination transposed_lhs(transposed_shape);
    runtime.transpose(lhs, transposed_lhs);

    auto product_shape =
        rtensor::infer_mult(transposed_lhs.shape(), grad.shape());
    if (product_shape == destination.shape()) {
      runtime.mult(transposed_lhs, grad, destination);
    } else {
      Destination product(product_shape);
      runtime.mult(transposed_lhs, grad, product);
      runtime.sum_to_shape(product, destination.shape(), destination);
    }
    return destination;
  };
};
template <typename A, typename B> struct eval_expr<expr::EXPR_ADD<A, B>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.add(eval_expr<A>::run(runtime, ctx),
                eval_expr<B>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename INPUT, typename WEIGHT, typename BIAS>
struct eval_expr<expr::EXPR_LINEAR<INPUT, WEIGHT, BIAS>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    const auto &input = eval_expr<INPUT>::run(runtime, ctx);
    const auto &weight = eval_expr<WEIGHT>::run(runtime, ctx);
    const auto &bias = eval_expr<BIAS>::run(runtime, ctx);
    runtime.mult(input, weight, destination);
    runtime.add(destination, bias, destination);
    return destination;
  };
};
template <typename A, typename B> struct eval_expr<expr::EXPR_SUB<A, B>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.sub(eval_expr<A>::run(runtime, ctx),
                eval_expr<B>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename A, typename B> struct eval_expr<expr::EXPR_HADAMARD<A, B>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.hadamard(eval_expr<A>::run(runtime, ctx),
                     eval_expr<B>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename A, typename B> struct eval_expr<expr::EXPR_DIV<A, B>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.divide(eval_expr<A>::run(runtime, ctx),
                   eval_expr<B>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_SIG<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.sigmoid(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_SFTMX<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.softmax(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename GRAD, typename OUTPUT>
struct eval_expr<expr::EXPR_SFTMX_DERIVATIVE<GRAD, OUTPUT>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    const auto &grad = eval_expr<GRAD>::run(runtime, ctx);
    const auto &output = eval_expr<OUTPUT>::run(runtime, ctx);

    auto reduced_shape = output.shape();
    reduced_shape.back() = 1;
    Destination reduced(reduced_shape);

    runtime.hadamard(grad, output, destination);
    runtime.sum(destination, destination.shape().size() - 1, reduced);
    runtime.sub(grad, reduced, destination);
    runtime.hadamard(output, destination, destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_SQRT<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.sqrt(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_RMS<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.RMS(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_RMS_NORM<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.RMSNorm(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename GRAD, typename VAL, typename OUTPUT>
struct eval_expr<expr::EXPR_RMSN_DERIVATIVE<GRAD, VAL, OUTPUT>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    const auto &grad = eval_expr<GRAD>::run(runtime, ctx);
    const auto &value = eval_expr<VAL>::run(runtime, ctx);
    const auto &output = eval_expr<OUTPUT>::run(runtime, ctx);

    auto reduced_shape = value.shape();
    reduced_shape.back() = 1;
    Destination rms(reduced_shape);
    Destination mean_grad_output(reduced_shape);

    runtime.RMS(value, rms);
    runtime.hadamard(grad, output, destination);
    runtime.sum(destination, destination.shape().size() - 1,
                mean_grad_output);
    runtime.divide(mean_grad_output,
                   static_cast<typename Destination::value_type>(
                       value.shape().back()),
                   mean_grad_output);
    runtime.hadamard(output, mean_grad_output, destination);
    runtime.sub(grad, destination, destination);
    runtime.divide(destination, rms, destination);
    return destination;
  };
};
template <typename VAL, auto Exponent>
struct eval_expr<expr::EXPR_POW<VAL, Exponent>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    using T = typename std::remove_reference_t<Destination>::value_type;
    runtime.pow(eval_expr<VAL>::run(runtime, ctx), static_cast<T>(Exponent),
                destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_SUM<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    const auto &value = eval_expr<VAL>::run(runtime, ctx);
    runtime.sum(value, value.shape().size() - 1, destination);
    return destination;
  };
};
template <typename VAL, int32_t DimA, int32_t DimB>
struct eval_expr<expr::EXPR_TRAN<VAL, DimA, DimB>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.transpose(eval_expr<VAL>::run(runtime, ctx), destination, DimA,
                      DimB);
    return destination;
  };
};
template <typename VAL, uint32_t Outer, uint32_t Inner>
struct eval_expr<expr::EXPR_SPLIT_LAST_DIM<VAL, Outer, Inner>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.split_last_dim(eval_expr<VAL>::run(runtime, ctx), destination,
                           Outer, Inner);
    return destination;
  };
};
template <typename VAL, uint32_t Outer, uint32_t Inner>
struct eval_expr<expr::EXPR_MERGE_LAST_DIMS<VAL, Outer, Inner>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.merge_last_dims(eval_expr<VAL>::run(runtime, ctx), destination,
                            Outer, Inner);
    return destination;
  };
};
template <typename VAL, typename IDS>
struct eval_expr<expr::EXPR_EMBEDDING_LOOKUP<VAL, IDS>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.embedding_lookup(eval_expr<VAL>::run(runtime, ctx),
                             eval_expr<IDS>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename BASE, typename IDS, typename VALUES>
struct eval_expr<expr::EXPR_SCATTER_ADD_ROWS<BASE, IDS, VALUES>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.scatter_add_rows(eval_expr<BASE>::run(runtime, ctx),
                             eval_expr<IDS>::run(runtime, ctx),
                             eval_expr<VALUES>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VALUES, typename IDS, typename SHAPE>
struct eval_expr<expr::EXPR_SCATTER_INDEX<VALUES, IDS, SHAPE>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.scatter_index(eval_expr<VALUES>::run(runtime, ctx),
                          eval_expr<IDS>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename PRED, typename TARGET>
struct eval_expr<expr::EXPR_SQE<PRED, TARGET>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.sqe(eval_expr<PRED>::run(runtime, ctx),
                eval_expr<TARGET>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename PRED, typename TARGET>
struct eval_expr<expr::EXPR_X_ENTROPY<PRED, TARGET>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.cross_entropy(eval_expr<PRED>::run(runtime, ctx),
                          eval_expr<TARGET>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename PRED, typename TARGET>
struct eval_expr<expr::EXPR_X_ENTROPY_DERIVATIVE<PRED, TARGET>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.cross_entropy_derivative(eval_expr<PRED>::run(runtime, ctx),
                                     eval_expr<TARGET>::run(runtime, ctx),
                                     destination);
    return destination;
  };
};
template <auto VALUE> struct eval_expr<expr::EXPR_CONSTANT<VALUE>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &, Ctx &, Destination &destination) {
    using T = typename std::remove_reference_t<Destination>::value_type;
    const std::array<T, 1> value{static_cast<T>(VALUE)};
    destination.set(std::span<const T>(value));
    return destination;
  };
};

template <typename A, expr::EXPR_TENSOR B>
struct eval_expr<expr::EXPR_GATHER_ROWS<A, B>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.gather_rows(eval_expr<A>::run(runtime, ctx),
                        eval_expr<B>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_GELU<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.GELU(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};
template <typename VAL> struct eval_expr<expr::EXPR_GELU_DERIVATIVE<VAL>> {
  template <typename Run, typename Ctx, typename Destination>
  static auto &run(Run &runtime, Ctx &ctx, Destination &destination) {
    runtime.GELU_derivative(eval_expr<VAL>::run(runtime, ctx), destination);
    return destination;
  };
};

template <typename Run, typename Network, typename GradientContext,
          typename OptimiserContext, typename Shared>
void step(Run &runtime, Network &network, GradientContext &gradients,
          OptimiserContext &optimiser_ctx, Shared &shared) {
  gradients.for_each([&]<typename Path>(const auto &grad) {
    auto &tensor = get_at_path<Path>(network);
    auto &optimiser = optimiser_ctx.template get_node<Path>();
    auto &parameter = tensor.get_value();

    optimiser.update(runtime, shared, parameter, grad, parameter);
  });
}

} // namespace exec
} // namespace passes
