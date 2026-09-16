#include <cassert>
#include <cmath>
#include <expr/api.h>
#include <nn/layer.h>
#include <optimiser/sgd.h>
#include <passes/compile.h>
#include <passes/get_at_path.h>
#include <random>
#include <stdexcept>
#include <test_backend.h>
#include <vector>

using test_backend::Mat;
using test_backend::Runtime;
using SGD = Optimiser::SGD<Mat>;

struct CountingRuntime : Runtime {
  size_t hadamard_calls = 0;

  void hadamard(const Mat &a, const Mat &b, Mat &destination) {
    ++hadamard_calls;
    Runtime::hadamard(a, b, destination);
  }
};

using TestPath = passes::graph_path<passes::graph_root, 0>;
using TestPaths = passes::type_list<TestPath>;
static_assert(
    passes::ContextForPath<passes::PathContext<Mat, TestPaths>, TestPath>);
static_assert(passes::GradientContextForPath<
              passes::GradientContext<Mat, TestPaths>, TestPath, Runtime>);
static_assert(passes::OptimiserFor<Optimiser::SGD<Mat>, Runtime, Mat, Mat>);
static_assert(std::same_as<
              typename passes::autograd::eval<expr::EXPR_CONSTANT<2>,
                                              passes::grad_seed,
                                              passes::graph_root>::type,
              passes::no_grad>);

void test_gradient_context_accumulates_and_clears() {
  using GradPath = passes::graph_path<passes::graph_root, 0>;
  using GradPaths = passes::type_list<GradPath>;

  passes::GradientContext<Mat, GradPaths> gradients;
  Runtime runtime;
  assert(!gradients.has_gradient<GradPath>());

  gradients.for_each_slot(
      []<typename Path>(auto &gradient) { gradient = Mat({1, 2}); });

  runtime.start();
  gradients.accumulate<GradPath>(runtime, Mat::from_values({{1.0f, 2.0f}}));
  gradients.accumulate<GradPath>(runtime, Mat::from_values({{3.0f, 4.0f}}));
  runtime.submit();
  runtime.wait();

  assert(gradients.has_gradient<GradPath>());
  assert((gradients.get_node<GradPath>() == Mat::from_values({{4.0f, 6.0f}})));

  gradients.clear();
  assert(!gradients.has_gradient<GradPath>());
}

void test_compile_rebinds_inputs_on_forward() {
  Runtime runtime;
  expr::tensor::GradTensor<Mat> lhs(Mat::from_values({{1.0f}}));
  expr::tensor::NoGradTensor<Mat> rhs(Mat::from_values({{2.0f}}));

  auto expression = expr::add(lhs, rhs);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(0.0f));
  compiled.prepare();

  assert(compiled.forward().get(0, 0) == 3.0f);

  using LhsPath = passes::graph_path<passes::graph_root, 0>;
  passes::get_at_path<LhsPath>(expression)
      .set_value(Mat::from_values({{4.0f}}));
  assert(compiled.forward().get(0, 0) == 6.0f);
}

void test_compiled_recovers_after_execution_failure() {
  Runtime runtime;
  expr::tensor::NoGradTensor<Mat> lhs(Mat::from_values({{1.0f, 2.0f}}));
  expr::tensor::NoGradTensor<Mat> rhs(Mat::from_values({{1.0f, 2.0f, 3.0f}}));

  auto expression = expr::add(lhs, rhs);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(0.0f));

  bool threw = false;
  try {
    compiled.prepare();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  rhs.set_value(Mat::from_values({{3.0f, 4.0f}}));
  compiled.prepare();
  assert((compiled.forward() == Mat::from_values({{4.0f, 6.0f}})));
}

void test_let_evaluates_value_once() {
  CountingRuntime runtime;
  expr::tensor::NoGradTensor<Mat> lhs(
      Mat::from_values({{1.0f, 2.0f}, {3.0f, 4.0f}}));
  expr::tensor::NoGradTensor<Mat> rhs(
      Mat::from_values({{5.0f, 6.0f}, {7.0f, 8.0f}}));

  auto value = expr::hadamard(lhs, rhs);
  auto reused =
      expr::let(value, [](auto input) { return expr::add(input, input); });
  auto expression = expr::add(reused, rhs);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(0.0f));

  compiled.prepare();
  runtime.hadamard_calls = 0;
  const Mat result = compiled.forward();

  assert(runtime.hadamard_calls == 1);
  assert((result == Mat::from_values({{15.0f, 30.0f}, {49.0f, 72.0f}})));
}

void test_let_accumulates_local_gradient() {
  Runtime runtime;
  expr::tensor::GradTensor<Mat> input(Mat::from_values({{2.0f}}));

  auto value = expr::hadamard(input, input);
  auto expression =
      expr::let(value, [](auto local) { return expr::add(local, local); });
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(1.0f));
  compiled.prepare();

  assert(compiled.forward().get(0, 0) == 8.0f);

  compiled.backward(Mat(1.0f));
  compiled.step();

  assert(input.get_value().get(0, 0) == -6.0f);
}

void test_compiled_sgd_updates_network() {
  Runtime runtime;
  expr::tensor::GradTensor<Mat> parameter(Mat::from_values({{1.0f}}));
  expr::tensor::NoGradTensor<Mat> constant(Mat::from_values({{2.0f}}));

  auto expression = expr::add(parameter, constant);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(0.5f));
  compiled.prepare();

  assert(compiled.forward().get(0, 0) == 3.0f);

  compiled.backward(Mat(1.0f));
  const auto *parameter_storage = parameter.get_value().get_storage();
  compiled.step();

  assert(parameter.get_value().get_storage() == parameter_storage);
  assert(compiled.forward().get(0, 0) == 2.5f);
}

void test_attention_shape_operations() {
  Runtime runtime;
  Mat input_value({1, 3, 4});
  input_value.set(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                     12.0f});
  expr::tensor::NoGradTensor<Mat> input(std::move(input_value));

  auto split = expr::split_last_dim<2, 2>(input);
  auto heads = expr::transpose<1, 2>(split);
  auto compiled_heads = passes::compile<Mat, SGD>(
      heads, runtime, SGD::generate_params(0.0f));
  compiled_heads.prepare();

  const auto heads_result = compiled_heads.forward();
  assert((heads_result.shape() == std::vector<uint32_t>{1, 2, 3, 2}));
  Mat expected_heads({1, 2, 3, 2});
  expected_heads.set(std::vector<float>{1.0f, 2.0f, 5.0f, 6.0f, 9.0f, 10.0f,
                                        3.0f, 4.0f, 7.0f, 8.0f, 11.0f,
                                        12.0f});
  assert((heads_result == expected_heads));

  auto ordered = expr::transpose<1, 2>(std::move(heads));
  auto merged = expr::merge_last_dims<2, 2>(std::move(ordered));
  auto compiled_merged = passes::compile<Mat, SGD>(
      merged, runtime, SGD::generate_params(0.0f));
  compiled_merged.prepare();

  const auto merged_result = compiled_merged.forward();
  assert((merged_result.shape() == std::vector<uint32_t>{1, 3, 4}));
  assert((merged_result == input.get_value()));
}

#if defined(GRAPH_TEST_BACKEND_METAL)
void test_compiled_rms_expressions() {
  Runtime runtime;
  Mat input_value({2, 2});
  input_value.set(std::vector<float>{3.0f, 4.0f, 0.0f, 0.0f});
  expr::tensor::NoGradTensor<Mat> input(std::move(input_value));

  auto rms_expression = expr::rms(input);
  auto compiled_rms = passes::compile<Mat, SGD>(
      rms_expression, runtime, SGD::generate_params(0.0f));
  compiled_rms.prepare();

  const float first_rms = std::sqrt(12.5f + 1e-6f);
  Mat expected_rms({2, 1});
  expected_rms.set(std::vector<float>{first_rms, 1e-3f});
  Mat rms_result = compiled_rms.forward();
  assert(std::abs(rms_result.get(0, 0) - expected_rms.get(0, 0)) < 1e-5f);
  assert(std::abs(rms_result.get(1, 0) - expected_rms.get(1, 0)) < 1e-5f);

  auto norm_expression = expr::rms_norm(input);
  auto compiled_norm = passes::compile<Mat, SGD>(
      norm_expression, runtime, SGD::generate_params(0.0f));
  compiled_norm.prepare();

  Mat expected_norm({2, 2});
  expected_norm.set(
      std::vector<float>{3.0f / first_rms, 4.0f / first_rms, 0.0f, 0.0f});
  Mat norm_result = compiled_norm.forward();
  assert(std::abs(norm_result.get(0, 0) - expected_norm.get(0, 0)) < 1e-5f);
  assert(std::abs(norm_result.get(0, 1) - expected_norm.get(0, 1)) < 1e-5f);
  assert(std::abs(norm_result.get(1, 0) - expected_norm.get(1, 0)) < 1e-5f);
  assert(std::abs(norm_result.get(1, 1) - expected_norm.get(1, 1)) < 1e-5f);

  expr::tensor::GradTensor<Mat> grad_input(
      Mat::from_values({{3.0f, 4.0f}}));
  auto grad_expression = expr::rms(grad_input);
  auto compiled_grad = passes::compile<Mat, SGD>(
      grad_expression, runtime, SGD::generate_params(1.0f));
  compiled_grad.prepare();
  compiled_grad.forward();
  compiled_grad.backward(Mat(1.0f));

  compiled_grad.step();

  const float rms = std::sqrt(12.5f + 1e-6f);
  assert(std::abs(grad_input.get_value().get(0, 0) -
                  (3.0f - 3.0f / (2.0f * rms))) <
         1e-5f);
  assert(std::abs(grad_input.get_value().get(0, 1) -
                  (4.0f - 4.0f / (2.0f * rms))) <
         1e-5f);

  expr::tensor::GradTensor<Mat> norm_grad_input(
      Mat::from_values({{3.0f, 4.0f}}));
  auto norm_grad_expression = expr::rms_norm(norm_grad_input);
  auto compiled_norm_grad = passes::compile<Mat, SGD>(
      norm_grad_expression, runtime, SGD::generate_params(1.0f));
  compiled_norm_grad.prepare();
  Mat norm_output = compiled_norm_grad.forward();
  compiled_norm_grad.backward(Mat::from_values({{1.0f, 2.0f}}));
  compiled_norm_grad.step();

  const float mean_grad_y =
      (norm_output.get(0, 0) + 2.0f * norm_output.get(0, 1)) / 2.0f;
  const float first_gradient =
      (1.0f - norm_output.get(0, 0) * mean_grad_y) / rms;
  const float second_gradient =
      (2.0f - norm_output.get(0, 1) * mean_grad_y) / rms;
  assert(std::abs(norm_grad_input.get_value().get(0, 0) -
                  (3.0f - first_gradient)) <
         1e-5f);
  assert(std::abs(norm_grad_input.get_value().get(0, 1) -
                  (4.0f - second_gradient)) <
         1e-5f);
}

void test_compiled_embedding_uses_index_context() {
  using IndexMat = typename Mat::template rebind<uint32_t>;

  Runtime runtime;
  expr::tensor::GradTensor<Mat> weights(
      Mat::from_values({{1.0f, 2.0f, 3.0f},
                        {4.0f, 5.0f, 6.0f},
                        {7.0f, 8.0f, 9.0f},
                        {10.0f, 11.0f, 12.0f}}));
  IndexMat ids_value({2, 2});
  ids_value.set(std::vector<uint32_t>{2, 0, 2, 1});
  expr::tensor::NoGradTensor<IndexMat> ids(std::move(ids_value));

  auto expression = expr::embedding_lookup(weights, ids);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(0.5f));
  compiled.prepare();

  Mat expected({2, 2, 3});
  expected.set(std::vector<float>{7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f,
                                  7.0f, 8.0f, 9.0f, 4.0f, 5.0f, 6.0f});
  assert(compiled.forward() == expected);

  Mat root_grad({2, 2, 3});
  root_grad.set(std::vector<float>(12, 1.0f));
  compiled.backward(root_grad);

  compiled.step();
  assert(weights.get_value() ==
         Mat::from_values({{0.5f, 1.5f, 2.5f},
                           {3.5f, 4.5f, 5.5f},
                           {6.0f, 7.0f, 8.0f},
                           {10.0f, 11.0f, 12.0f}}));
}

void test_compiled_gelu_backward() {
  Runtime runtime;
  expr::tensor::GradTensor<Mat> input(
      Mat::from_values({{-1.0f, 0.0f, 1.0f}}));
  auto expression = expr::gelu(input);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(1.0f));
  compiled.prepare();
  compiled.forward();
  compiled.backward(Mat::from_values({{1.0f, 1.0f, 1.0f}}));

  compiled.step();

  const auto derivative = [](float x) {
    constexpr float a = 0.7978845608f;
    constexpr float b = 0.044715f;
    const float x_squared = x * x;
    const float tanh_value = std::tanh(a * (x + b * x_squared * x));
    return 0.5f * (1.0f + tanh_value) +
           0.5f * x * (1.0f - tanh_value * tanh_value) * a *
               (1.0f + 3.0f * b * x_squared);
  };

  assert(std::abs(input.get_value().get(0, 0) -
                  (-1.0f - derivative(-1.0f))) <
         1e-5f);
  assert(std::abs(input.get_value().get(0, 1) - (0.0f - derivative(0.0f))) <
         1e-5f);
  assert(std::abs(input.get_value().get(0, 2) - (1.0f - derivative(1.0f))) <
         1e-5f);
}

void test_compiled_cross_entropy_backward() {
  using IndexMat = typename Mat::template rebind<uint32_t>;

  Runtime runtime;
  expr::tensor::GradTensor<Mat> logits(
      Mat::from_values({{1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f}}));

  IndexMat target_values({2});
  target_values.set(std::vector<uint32_t>{2, 0});
  expr::tensor::NoGradTensor<IndexMat> targets(std::move(target_values));

  auto expression = expr::loss::cross_entropy(logits, targets);
  auto compiled = passes::compile<Mat, SGD>(
      expression, runtime, SGD::generate_params(1.0f));
  compiled.prepare();

  Mat losses = compiled.forward();
  assert((losses.shape() == std::vector<uint32_t>{2, 1}));

  const float denominator = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
  const float low = std::exp(-2.0f) / denominator;
  const float middle = std::exp(-1.0f) / denominator;
  const float high = 1.0f / denominator;
  assert(std::abs(losses.get(0, 0) - std::log(denominator)) < 1e-4f);
  assert(std::abs(losses.get(1, 0) - std::log(3.0f)) < 1e-4f);

  Mat root_grad({2, 1});
  root_grad.set(std::vector<float>{2.0f, 0.5f});
  compiled.backward(root_grad);

  compiled.step();

  Mat &updated = logits.get_value();
  assert(std::abs(updated.get(0, 0) - (1.0f - 2.0f * low)) < 1e-4f);
  assert(std::abs(updated.get(0, 1) - (2.0f - 2.0f * middle)) < 1e-4f);
  assert(std::abs(updated.get(0, 2) - (3.0f - 2.0f * (high - 1.0f))) <
         1e-4f);
  assert(std::abs(updated.get(1, 0) - (1.0f + 1.0f / 3.0f)) < 1e-4f);
  assert(std::abs(updated.get(1, 1) - (1.0f - 1.0f / 6.0f)) < 1e-4f);
  assert(std::abs(updated.get(1, 2) - (1.0f - 1.0f / 6.0f)) < 1e-4f);
}
#endif

int main() {
  test_backend::init();
  test_gradient_context_accumulates_and_clears();
  test_compile_rebinds_inputs_on_forward();
  test_compiled_recovers_after_execution_failure();
  test_let_evaluates_value_once();
  test_let_accumulates_local_gradient();
  test_compiled_sgd_updates_network();
  test_attention_shape_operations();
#if defined(GRAPH_TEST_BACKEND_METAL)
  test_compiled_rms_expressions();
  test_compiled_embedding_uses_index_context();
  test_compiled_gelu_backward();
  test_compiled_cross_entropy_backward();
#endif
  Runtime runtime;

  std::mt19937 gen(1);
  nn::Layer<Mat> query(128, 64, gen);

  Mat input_mat({1, 128});
  Mat input_mat_1({64, 1});

  expr::tensor::GradTensor input(std::move(input_mat));
  expr::tensor::GradTensor input_1(std::move(input_mat_1));

  auto scores = expr::mult(query(input), input_1);
  auto activated = expr::sigmoid(scores);
  auto rooted = expr::sqrt(activated);
  auto probabilities = expr::softmax(rooted);
  auto scaled = expr::hadamard(probabilities, probabilities);
  auto normalised = expr::divide(scaled, probabilities);
  auto loss = expr::loss::sqe(normalised, probabilities);

  auto compiled = passes::compile<Mat, SGD>(
      loss, runtime, SGD::generate_params(0.0f));
  compiled.prepare();
  auto forward_result = compiled.forward();
  assert((forward_result.shape() == std::vector<uint32_t>{1, 1}));

  Mat root_grad(1.0f);
  compiled.zero_grad();
  compiled.backward(root_grad);
  compiled.backward(root_grad);
  compiled.zero_grad();
  test_backend::shutdown();
}
