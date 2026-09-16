#pragma once

#include "passes/meta.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace passes {

template <typename T, typename Run>
concept GradientRuntime = requires(Run &run, T output, const T &a,
                                   const T &b) {
  { run.copy(a, output) } -> std::same_as<void>;
  { run.add(a, b, output) } -> std::same_as<void>;
};

template <typename T, typename Run>
concept ExecutionRuntime = GradientRuntime<T, Run> &&
    requires(Run &run, T output, const T &a, const T &b) {
  { run.sub(a, b, output) } -> std::same_as<void>;
  { run.mult(a, b, output) } -> std::same_as<void>;
  { run.transpose(a, output) } -> std::same_as<void>;
  { run.transpose(a, output, int32_t{}, int32_t{}) } -> std::same_as<void>;
  { run.split_last_dim(a, output, uint32_t{}, uint32_t{}) } ->
      std::same_as<void>;
  { run.merge_last_dims(a, output, uint32_t{}, uint32_t{}) } ->
      std::same_as<void>;
  { run.sigmoid(a, output) } -> std::same_as<void>;
  { run.hadamard(a, b, output) } -> std::same_as<void>;
  { run.divide(a, b, output) } -> std::same_as<void>;
  { run.sqrt(a, output) } -> std::same_as<void>;
  { run.sum(a, output) } -> std::same_as<void>;
  { run.sum(a, std::size_t{}, output) } -> std::same_as<void>;
  { run.sum_to_shape(a, a.shape(), output) } -> std::same_as<void>;
  { run.softmax(a, output) } -> std::same_as<void>;
  { run.sqe(a, b, output) } -> std::same_as<void>;
  { run.start() } -> std::same_as<void>;
  { run.submit() } -> std::same_as<void>;
  { run.wait() } -> std::same_as<void>;
  { run.abort() } noexcept -> std::same_as<void>;
};

template <typename T, typename Run>
concept NeuralRuntime = requires(Run &run, T output, const T &a,
                                 typename T::value_type scalar) {
  { run.divide(a, scalar, output) } -> std::same_as<void>;
  { run.pow(a, scalar, output) } -> std::same_as<void>;
  { run.RMS(a, output) } -> std::same_as<void>;
  { run.RMSNorm(a, output) } -> std::same_as<void>;
  { run.GELU(a, output) } -> std::same_as<void>;
  { run.GELU_derivative(a, output) } -> std::same_as<void>;
};

template <typename T, typename Run>
concept IndexedRuntime = requires(
    Run &run, T output, const T &a, const T &b,
    const typename T::template rebind<uint32_t> &indices) {
  { run.embedding_lookup(a, indices, output) } -> std::same_as<void>;
  { run.gather_rows(a, indices, output) } -> std::same_as<void>;
  { run.scatter_add_rows(a, indices, b, output) } -> std::same_as<void>;
  { run.scatter_index(a, indices, output) } -> std::same_as<void>;
  { run.cross_entropy(a, indices, output) } -> std::same_as<void>;
  { run.cross_entropy_derivative(a, indices, output) } -> std::same_as<void>;
};

template <typename T, typename Run>
concept GradientClippingRuntime = requires(
    Run &run, T output, const T &gradient, const T &squared_norm,
    typename T::value_type max_norm) {
  { run.accumulate_gradient_squared_norm(gradient, output) } ->
      std::same_as<void>;
  { run.clip_gradient(gradient, squared_norm, max_norm, output) } ->
      std::same_as<void>;
};

template <typename T, typename Run>
concept AdamWRuntime = requires(
    Run &run, T output, T momentum1, T momentum2, const T &parameter,
    const T &gradient, const T &beta1, const T &beta2, const T &decay,
    const T &learning_rate, typename T::value_type epsilon) {
  { run.adamW(parameter, gradient, output, momentum1, momentum2, beta1, beta2,
              decay, learning_rate, epsilon, uint32_t{}) } ->
      std::same_as<void>;
};

template <typename T, typename Run>
concept Runnable = ExecutionRuntime<T, Run> && NeuralRuntime<T, Run> &&
                   IndexedRuntime<T, Run> &&
                   GradientClippingRuntime<T, Run> && AdamWRuntime<T, Run>;

template <typename Context> using context_type_t = std::remove_cvref_t<Context>;

template <typename Context>
using context_value_t = typename context_type_t<Context>::value_type;

template <typename Context>
concept NodeContext = requires {
  typename context_type_t<Context>::value_type;
  typename context_type_t<Context>::paths_type;
  requires TypeList<typename context_type_t<Context>::paths_type>;
};

template <typename Context, typename Path>
concept ContextForPath =
    NodeContext<Context> &&
    contains_type_v<typename context_type_t<Context>::paths_type, Path>;

template <typename Context, typename Path, typename Run>
concept GradientContextForPath =
    ContextForPath<Context, Path> &&
    GradientRuntime<context_value_t<Context>, Run> &&
    requires(Context &ctx, Run &run, const context_value_t<Context> &value) {
      { ctx.template has_gradient<Path>() } -> std::same_as<bool>;
      ctx.template accumulate<Path>(run, value);
    };

template <typename Context, typename Path>
concept ForwardNodeFor = requires(Context &ctx) {
  requires ContextForPath<decltype(ctx.forward), Path>;
};

template <typename Context, typename Path>
concept InputNodeFor = requires(Context &ctx) {
  requires ContextForPath<decltype(ctx.input), Path>;
};

template <typename Context, typename Path>
concept IndexNodeFor = requires(Context &ctx) {
  requires ContextForPath<decltype(ctx.indices), Path>;
};

template <typename Context, typename Path, typename Run>
concept GradientNodeFor = requires(Context &ctx) {
  requires GradientContextForPath<decltype(ctx.gradients), Path, Run>;
};

template <typename Context, typename Path, typename Run>
concept LocalGradientNodeFor = requires(Context &ctx) {
  requires GradientContextForPath<decltype(ctx.local_gradients), Path, Run>;
};

template <typename Context>
concept RootGradientContext = requires(Context &ctx) {
  typename context_type_t<Context>::value_type;
  { ctx.root_grad } -> std::same_as<const context_value_t<Context> &>;
};

template <typename T>
concept OptimiserType = std::default_initializable<T> && requires {
  typename T::Shared;
};

template <typename Optimiser, typename Run, typename Parameter,
          typename Gradient>
concept OptimiserFor =
    OptimiserType<Optimiser> &&
    requires(Optimiser &optimiser, typename Optimiser::Shared &shared,
             Run &runtime, const Parameter &parameter,
             const Gradient &gradient, Parameter &destination) {
      {
        optimiser.update(runtime, shared, parameter, gradient, destination)
      } -> std::same_as<void>;
    };

} // namespace passes
