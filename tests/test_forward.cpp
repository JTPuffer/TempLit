#include "expr/tensor.h"
#include "optimiser/sgd.h"
#include "passes/compile.h"
#include "test_backend.h"
#include <cassert>
#include <span>
#include <utility>
#include <vector>
using namespace test_backend;
using namespace expr::tensor;
using SGD = Optimiser::SGD<Mat>;
int main() {
  test_backend::init();
  Runtime runtime;
  auto optimiser_shared = SGD::generate_params(0.0f);

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

  Mat expected = Mat::from_values({
      {5, 5},
      {11, 9},
  });

  Tensor input(std::move(input_mat));
  Tensor weight(std::move(weight_mat));
  Tensor bias(std::move(bias_mat));

  auto output = expr::add(expr::mult(input, weight), bias);
  auto compiled_output =
      passes::compile<Mat, SGD>(output, runtime, optimiser_shared);
  compiled_output.prepare();
  auto actual = compiled_output.forward();

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
    assert(actual_data[i] == expected_data[i]);
  }

  auto sub_output = expr::sub(input, bias);
  Mat expected_sub = Mat::from_values({
      {0, 1},
      {2, 3},
  });
  auto compiled_sub =
      passes::compile<Mat, SGD>(sub_output, runtime, optimiser_shared);
  compiled_sub.prepare();
  assert(compiled_sub.forward() == expected_sub);

  auto hadamard_output = expr::hadamard(input, weight);
  Mat expected_hadamard = Mat::from_values({
      {2, 0},
      {3, 8},
  });
  auto compiled_hadamard =
      passes::compile<Mat, SGD>(hadamard_output, runtime, optimiser_shared);
  compiled_hadamard.prepare();
  assert(compiled_hadamard.forward() == expected_hadamard);

  auto divide_output = expr::divide(input, bias);
  auto compiled_divide =
      passes::compile<Mat, SGD>(divide_output, runtime, optimiser_shared);
  compiled_divide.prepare();
  assert(compiled_divide.forward() == input.get_value());

  test_backend::shutdown();
}
