#include "expr/tensor.h"
#include "optimiser/sgd.h"
#include "passes/compile.h"
#include "test_backend.h"

#include <cassert>
#include <cmath>
#include <span>
#include <utility>
#include <vector>
using namespace test_backend;
using namespace expr::tensor;
using SGD = Optimiser::SGD<Mat>;
int main() {
  test_backend::init();
  Runtime runtime;
  using Tensor = Tensor<Mat>;

  Mat input_mat = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat weight_mat = Mat::from_values({
      {2, 0},
      {1, 2},
  });

  Mat bias_mat = Mat::from_values({
      {1, 1},
      {1, 1},
  });

  GradTensor input(std::move(input_mat));
  GradTensor weight(std::move(weight_mat));
  GradTensor bias(std::move(bias_mat));

  // layer = sigmoid(input * weight + bias)
  auto output = expr::sigmoid(expr::add(expr::mult(input, weight), bias));

  /*
      Forward:

      input * weight

      [1 2] [2 0]   [ 4 4]
      [3 4] [1 2] = [10 8]

      + bias

      [ 5 5]
      [11 9]

      sigmoid(...)

      [sig(5)  sig(5)]
      [sig(11) sig(9)]
  */

  auto optimiser_shared = SGD::generate_params(0.5f);
  auto compiled =
      passes::compile<Mat, SGD>(output, runtime, optimiser_shared);
  compiled.prepare();
  Mat forward_result = compiled.forward();

  const auto sig = [](kernel_type x) {
    return kernel_type{1} / (kernel_type{1} + std::exp(-x));
  };

  const auto close = [](kernel_type a, kernel_type b) {
    return std::abs(a - b) < 1e-5f;
  };

  const auto check_tensor = [&](const Mat &actual, const Mat &expected) {
    assert(actual.shape() == expected.shape());

    size_t size = 1;
    for (const size_t dim : actual.shape()) {
      size *= dim;
    }

    std::vector<kernel_type> actual_data(size);
    std::vector<kernel_type> expected_data(size);
    actual.get_storage()->read(std::span<kernel_type>(actual_data), 0);
    expected.get_storage()->read(std::span<kernel_type>(expected_data), 0);

    for (size_t i = 0; i < size; ++i) {
      assert(close(actual_data[i], expected_data[i]));
    }
  };
  Mat expected_forward = Mat::from_values({
      {sig(5.0f), sig(5.0f)},
      {sig(11.0f), sig(9.0f)},
  });

  check_tensor(forward_result, expected_forward);

  /*
      Backward

      upstream gradient = all ones
  */

  Mat grad = Mat::from_values({
      {1, 1},
      {1, 1},
  });

  compiled.zero_grad();
  compiled.backward(grad);
  compiled.backward(grad);

  const kernel_type d5 = sig(5.0f) * (1.0f - sig(5.0f));

  const kernel_type d11 = sig(11.0f) * (1.0f - sig(11.0f));

  const kernel_type d9 = sig(9.0f) * (1.0f - sig(9.0f));

  /*
      gradient after sigmoid:

      [d5  d5 ]
      [d11 d9 ]
  */

  Mat expected_bias_grad = Mat::from_values({
      {d5, d5},
      {d11, d9},
  });

  /*
      dInput = grad_after_sigmoid * weight^T

      weight^T:

      [2 1]
      [0 2]
  */

  Mat expected_input_grad = Mat::from_values({
      {
          2.0f * d5,
          d5 + 2.0f * d5,
      },
      {
          2.0f * d11,
          d11 + 2.0f * d9,
      },
  });

  /*
      dWeight = input^T * grad_after_sigmoid

      input^T:

      [1 3]
      [2 4]
  */

  Mat expected_weight_grad = Mat::from_values({
      {
          d5 + 3.0f * d11,
          d5 + 3.0f * d9,
      },
      {
          2.0f * d5 + 4.0f * d11,
          2.0f * d5 + 4.0f * d9,
      },
  });

  compiled.step();

  check_tensor(input.get_value(),
               sub(Mat::from_values({{1, 2}, {3, 4}}), expected_input_grad));
  check_tensor(weight.get_value(),
               sub(Mat::from_values({{2, 0}, {1, 2}}), expected_weight_grad));
  check_tensor(bias.get_value(),
               sub(Mat::from_values({{1, 1}, {1, 1}}), expected_bias_grad));

  test_backend::shutdown();
  return 0;
}
