#pragma once

#include "RTensor/RTensor.h"
#include "expr/expr.h"
#include "model_utils/archive_concepts.h"
#include "passes/autograd.h"
#include "passes/bind_inputs.h"
#include "passes/concepts.h"
#include "passes/context.h"
#include "passes/execute.h"
#include "passes/for_each_path.h"
#include "passes/forward_type.h"
#include "passes/get_at_path.h"
#include "passes/get_gradient_path.h"
#include "passes/get_index_path.h"
#include "passes/get_input_path.h"
#include "passes/get_local_gradient_path.h"
#include "passes/get_required_path.h"
#include "passes/get_saved_path.h"
#include "passes/get_shape_path.h"
#include "passes/get_temp_path.h"
#include "passes/mem_lower.h"
#include "passes/meta.h"
#include "passes/program.h"
#include "passes/shape_pass.h"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace passes {

template <typename Mat, expr::EXPR_TYPE Network, typename Run,
          OptimiserFor<Run, Mat, Mat> Optimiser>
  requires ExecutionRuntime<Mat, Run>
class Compiled {
  using BackwardProgram =
      typename autograd::eval<Network, grad_seed, graph_root>::type;
  using BackwardRequiredPaths = get_required_paths<BackwardProgram>::type;

  using ForwardProgram =
      forward_type::eval<Network, BackwardRequiredPaths, graph_root>::type;
  using ForwardSavedPaths = get_saved_paths<ForwardProgram>::type;
  using ForwardCachePaths =
      unique_t<concat_t<BackwardRequiredPaths, ForwardSavedPaths>>;

  using InputPaths = unique_t<typename get_input_path<ForwardProgram>::type>;
  using IndexPaths =
      unique_t<concat_t<typename get_index_path<ForwardProgram>::type,
                        typename get_index_path<BackwardProgram>::type>>;
  using GradientPaths =
      unique_t<typename get_gradient_paths<BackwardProgram>::type>;
  using LocalGradientPaths =
      unique_t<typename get_local_gradient_paths<BackwardProgram>::type>;

  using BackwardProgram_temp = mem_lower::lower_t<BackwardProgram>;
  using ForwardProgram_temp = mem_lower::lower_t<ForwardProgram>;

  using BackwardTempPaths = get_temp_path<BackwardProgram_temp>::type;
  using ForwardTempPaths = get_temp_path<ForwardProgram_temp>::type;
  using BackwardShapePaths =
      unique_t<typename get_shape_path<BackwardProgram_temp, prog_root>::type>;
  using ForwardShapePaths =
      unique_t<typename get_shape_path<ForwardProgram_temp, prog_root>::type>;

  Network network_;
  PathContext<Mat, ForwardCachePaths> forward_cache_;
  PathContext<Mat, InputPaths> input_context_;
  PathContext<typename Mat::template rebind<rtensor::index_type>, IndexPaths>
      index_context_;
  GradientContext<Mat, GradientPaths> gradients_;
  GradientContext<Mat, LocalGradientPaths> local_gradients_;

  PathContext<rtensor::shape_type, ForwardShapePaths> forward_shapes_;
  PathContext<rtensor::shape_type, BackwardShapePaths> backward_shapes_;
  PathContext<rtensor::shape_type, ForwardCachePaths> saved_forward_shapes_;
  PathContext<rtensor::shape_type, InputPaths> input_shapes_;
  PathContext<rtensor::shape_type, IndexPaths> index_shapes_;

  PathContext<Mat, ForwardTempPaths> forward_temp;
  PathContext<Mat, BackwardTempPaths> backward_temp;

  PathContext<Optimiser, GradientPaths> optimisers_;
  typename Optimiser::Shared optimiser_shared_;
  Mat gradient_squared_norm_;
  rtensor::shape_type output_shape_;
  bool prepared_ = false;

  Run &runtime_;

public:
  explicit Compiled(Network network, Run &runtime,
                    typename Optimiser::Shared optimiser_shared)
      : network_(std::move(network)),
        optimiser_shared_(std::move(optimiser_shared)), runtime_(runtime) {}

  void prepare() {
    prepared_ = false;
    gradients_.clear();
    local_gradients_.clear();
    bind(network_, input_context_);
    bind(network_, index_context_);

    input_context_.for_each([&]<typename PATH>(const auto &input) {
      input_shapes_.template get_node<PATH>() = input.shape();
    });
    index_context_.for_each([&]<typename PATH>(const auto &indices) {
      index_shapes_.template get_node<PATH>() = indices.shape();
    });

    ForwardExecutionContext context{.input = input_context_,
                                    .indices = index_context_,
                                    .forward = forward_cache_,
                                    .temporary = forward_temp};

    output_shape_ = shape_pass::eval_expr<ForwardProgram_temp, prog_root>::run(
        context, forward_shapes_, saved_forward_shapes_);

    shape_pass::BackwardContext backward_shape_context{
        .indices = index_context_,
        .root_grad_shape = output_shape_,
    };

    shape_pass::eval_expr<BackwardProgram_temp, prog_root>::run(
        backward_shape_context, backward_shapes_, saved_forward_shapes_);

    forward_temp.for_each([&]<typename PATH>(auto &temp) {
      temp = Mat(forward_shapes_.template get_node<PATH>());
    });
    backward_temp.for_each([&]<typename PATH>(auto &temp) {
      temp = Mat(backward_shapes_.template get_node<PATH>());
    });
    forward_cache_.for_each([&]<typename PATH>(auto &saved) {
      saved = Mat(saved_forward_shapes_.template get_node<PATH>());
    });
    local_gradients_.for_each_slot([&]<typename PATH>(auto &saved) {
      saved = Mat(saved_forward_shapes_.template get_node<PATH>());
    });
    gradients_.for_each_slot([&]<typename PATH>(auto &saved) {
      saved = Mat(get_at_path<PATH>(network_).get_value().shape());
    });
    gradient_squared_norm_ = Mat::from_zeros({1});

    prepared_ = true;
  }

  Mat forward() {
    if (!prepared_) {
      throw std::logic_error(
          "Compiled program must be prepared before forward");
    }

    bind(network_, input_context_);
    bind(network_, index_context_);
    input_context_.for_each([&]<typename PATH>(const auto &input) {
      if (input.shape() != input_shapes_.template get_node<PATH>()) {
        throw std::invalid_argument(
            "Compiled program input shape changed; call prepare again");
      }
    });
    index_context_.for_each([&]<typename PATH>(const auto &indices) {
      if (indices.shape() != index_shapes_.template get_node<PATH>()) {
        throw std::invalid_argument(
            "Compiled program index shape changed; call prepare again");
      }
    });

    ForwardExecutionContext context{.input = input_context_,
                                    .indices = index_context_,
                                    .forward = forward_cache_,
                                    .temporary = forward_temp};

    runtime_.start();
    try {
      Mat result(output_shape_);
      exec::eval_expr<ForwardProgram_temp>::run(runtime_, context, result);
      runtime_.submit();
      runtime_.wait();
      return result;
    } catch (...) {
      runtime_.abort();
      throw;
    }
  }

  template <typename Path> Mat &saved_forward() {
    static_assert(ContextForPath<decltype(forward_cache_), Path>,
                  "Requested path is not saved by the forward program");
    return forward_cache_.template get_node<Path>();
  }

  void backward(const Mat &root_grad) {
    if (!prepared_) {
      throw std::logic_error(
          "Compiled program must be prepared before backward");
    }
    if (root_grad.shape() != output_shape_) {
      throw std::invalid_argument(
          "Backward root gradient shape does not match prepared output");
    }

    local_gradients_.clear();
    AutogradContext context{.forward = forward_cache_,
                            .gradients = gradients_,
                            .local_gradients = local_gradients_,
                            .indices = index_context_,
                            .temporary = backward_temp,
                            .root_grad = root_grad};
    runtime_.start();
    try {
      exec::eval_expr<BackwardProgram_temp>::run(runtime_, context);
      runtime_.submit();
      runtime_.wait();
    } catch (...) {
      runtime_.abort();
      throw;
    }
  }

  void zero_grad() { gradients_.clear(); }

  void step() {
    runtime_.start();
    try {
      exec::step(runtime_, network_, gradients_, optimisers_,
                 optimiser_shared_);
      runtime_.submit();
      runtime_.wait();
    } catch (...) {
      runtime_.abort();
      throw;
    }
  }

  void step(typename Mat::value_type max_grad_norm)
    requires GradientClippingRuntime<Mat, Run>
  {
    gradient_squared_norm_.set(typename Mat::value_type{}, 0);
    runtime_.start();
    try {
      gradients_.for_each([&]<typename Path>(auto &gradient) {
        runtime_.accumulate_gradient_squared_norm(gradient,
                                                  gradient_squared_norm_);
      });
      gradients_.for_each([&]<typename Path>(auto &gradient) {
        runtime_.clip_gradient(gradient, gradient_squared_norm_, max_grad_norm,
                               gradient);
      });
      exec::step(runtime_, network_, gradients_, optimisers_,
                 optimiser_shared_);
      runtime_.submit();
      runtime_.wait();
    } catch (...) {
      runtime_.abort();
      throw;
    }
  }
  template <ArchiveType Arch> void load(Arch &archive) {
    size_t index = 0;
    auto parameters = archive.scope("parameters");

    for_each_path<GradientPaths>([&]<typename Path> {
      auto &tensor = get_at_path<Path>(network_).get_value();
      auto tensor_archive = parameters.scope(std::to_string(index++));
      tensor.load(tensor_archive);
    });
  }
  template <ArchiveType Arch> void save(Arch &archive) {
    size_t index = 0;
    auto parameters = archive.scope("parameters");

    for_each_path<GradientPaths>([&]<typename Path> {
      const auto &tensor = get_at_path<Path>(network_).get_value();
      auto tensor_archive = parameters.scope(std::to_string(index++));
      tensor.save(tensor_archive);
    });
  }

  template <ArchiveType Arch> void load_optimisers(Arch &archive) {
    size_t index = 0;
    auto optimiser_archive = archive.scope("optimisers");

    optimisers_.for_each([&]<typename Path>(auto &optimiser) {
      const auto &parameter = get_at_path<Path>(network_).get_value();
      optimiser.initialise_for(parameter);
      auto state_archive = optimiser_archive.scope(std::to_string(index++));
      optimiser.load(state_archive);
    });
  }
  template <ArchiveType Arch> void save_optimisers(Arch &archive) {
    size_t index = 0;
    auto optimiser_archive = archive.scope("optimisers");

    optimisers_.for_each([&]<typename Path>(const auto &optimiser) {
      auto state_archive = optimiser_archive.scope(std::to_string(index++));
      optimiser.save(state_archive);
    });
  }

  template <typename Value> void set_lr(Value lr) {
    optimiser_shared_.set_lr(std::forward<Value>(lr));
  }
};

template <typename Mat, OptimiserType Optimiser, expr::EXPR_TYPE Network,
          typename Run>
  requires ExecutionRuntime<Mat, Run> &&
           OptimiserFor<Optimiser, Run, Mat, Mat>
auto compile(Network &&network, Run &runtime,
             typename Optimiser::Shared optimiser_shared) {
  using NetworkType = std::remove_cvref_t<Network>;
  return Compiled<Mat, NetworkType, Run, Optimiser>(
      std::forward<Network>(network), runtime, std::move(optimiser_shared));
}

} // namespace passes
