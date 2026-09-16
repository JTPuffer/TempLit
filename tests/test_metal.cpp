#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "CPU/cpu.h"
#include "CPU/storage.h"
#include "metal_runtime_adapter.h"

using Tensor = RTensor<metal::Metal, float>;
using CpuTensor = RTensor<cpu::CPU, float>;

constexpr float EPS = 1e-5f;

bool approx(float a, float b, float eps = EPS) { return std::abs(a - b) < eps; }

template <typename TensorT> size_t tensor_size(const TensorT &tensor) {
  return std::accumulate(tensor.shape().begin(), tensor.shape().end(),
                         size_t{1}, std::multiplies<>{});
}

template <typename TensorT>
void fill_tensor(TensorT &tensor, std::initializer_list<float> values) {
  size_t count = tensor_size(tensor);

  if (values.size() != count) {
    std::cerr << "fill_tensor size mismatch: tensor has " << count
              << " elements but got " << values.size() << " values\n";
    assert(false);
  }

  tensor.get_storage()->write(
      std::span<const float>(values.begin(), values.size()), 0);
}

std::vector<float> read_tensor(const Tensor &tensor) {
  std::vector<float> data(tensor_size(tensor));
  tensor.get_storage()->read(std::span<float>(data), 0);
  return data;
}

std::vector<float> read_tensor(const CpuTensor &tensor) {
  std::vector<float> data(tensor_size(tensor));
  tensor.get_storage()->read(std::span<float>(data), 0);
  return data;
}

void print_tensor(const Tensor &tensor) {
  std::vector<float> data = read_tensor(tensor);

  std::cerr << "[";

  for (size_t i = 0; i < data.size(); ++i) {
    if (i != 0) {
      std::cerr << ", ";
    }

    std::cerr << data[i];
  }

  std::cerr << "]\n";
}

void assert_backend_parity(const CpuTensor &cpu_tensor,
                           const Tensor &metal_tensor,
                           std::string_view test_name, float eps = EPS) {
  if (cpu_tensor.shape() != metal_tensor.shape()) {
    std::cerr << "\nFAILED: " << test_name << '\n';
    std::cerr << "Shape mismatch between CPU and Metal\n";
    std::cerr << "CPU rank:   " << cpu_tensor.shape().size() << '\n';
    std::cerr << "Metal rank: " << metal_tensor.shape().size() << '\n';
    assert(false);
  }

  const std::vector<float> cpu_data = read_tensor(cpu_tensor);
  const std::vector<float> metal_data = read_tensor(metal_tensor);

  if (cpu_data.size() != metal_data.size()) {
    std::cerr << "\nFAILED: " << test_name << '\n';
    std::cerr << "Flat size mismatch between CPU and Metal\n";
    assert(false);
  }

  for (size_t i = 0; i < cpu_data.size(); ++i) {
    if (!approx(cpu_data[i], metal_data[i], eps)) {
      std::cerr << "\nFAILED: " << test_name << '\n';
      std::cerr << "Mismatch at flat index " << i << '\n';
      std::cerr << "CPU:   " << cpu_data[i] << '\n';
      std::cerr << "Metal: " << metal_data[i] << '\n';
      assert(false);
    }
  }
}

void assert_tensor_values(const Tensor &tensor,
                          std::initializer_list<float> expected,
                          std::string_view test_name, float eps = EPS) {
  size_t count = tensor_size(tensor);
  std::vector<float> data(count);
  tensor.get_storage()->read(std::span<float>(data), 0);

  if (count != expected.size()) {
    std::cerr << "\nFAILED: " << test_name << '\n';
    std::cerr << "Tensor size mismatch: got " << count << ", expected "
              << expected.size() << '\n';
    assert(false);
  }

  size_t i = 0;

  for (float expected_value : expected) {
    if (!approx(data[i], expected_value, eps)) {
      std::cerr << "\nFAILED: " << test_name << '\n';
      std::cerr << "Mismatch at flat index " << i << '\n';
      std::cerr << "Got:      " << data[i] << '\n';
      std::cerr << "Expected: " << expected_value << '\n';
      std::cerr << "Diff:     " << std::abs(data[i] - expected_value) << '\n';

      std::cerr << "Actual tensor:   ";
      print_tensor(tensor);

      std::cerr << "Expected tensor: [";

      size_t j = 0;
      for (float value : expected) {
        if (j++ != 0) {
          std::cerr << ", ";
        }

        std::cerr << value;
      }

      std::cerr << "]\n";

      assert(false);
    }

    ++i;
  }
}

void assert_tensor_shape(const Tensor &tensor,
                         std::initializer_list<size_t> expected,
                         std::string_view test_name) {
  if (tensor.shape().size() != expected.size()) {
    std::cerr << "\nFAILED: " << test_name << '\n';
    std::cerr << "Rank mismatch: got " << tensor.shape().size() << ", expected "
              << expected.size() << '\n';
    assert(false);
  }

  size_t i = 0;
  for (size_t dim : expected) {
    if (tensor.shape()[i] != dim) {
      std::cerr << "\nFAILED: " << test_name << '\n';
      std::cerr << "Shape mismatch at dim " << i << ": got "
                << tensor.shape()[i] << ", expected " << dim << '\n';
      assert(false);
    }

    ++i;
  }
}

void test_metal_broadcast_singleton() {
  Tensor a({2, 3});
  Tensor b({2, 1});

  fill_tensor(a, {1, 2, 3, 4, 5, 6});
  fill_tensor(b, {10, 20});

  Tensor out = metal_test_backend::add(a, b);

  assert_tensor_values(out, {11, 12, 13, 24, 25, 26},
                       "test_metal_broadcast_singleton");
}

void test_cpu_metal_elementwise_parity() {
  CpuTensor cpu_a({2, 3});
  CpuTensor cpu_b({2, 3});
  Tensor metal_a({2, 3});
  Tensor metal_b({2, 3});

  fill_tensor(cpu_a, {2, 4, 6, 8, 10, 12});
  fill_tensor(cpu_b, {1, 2, 3, 4, 5, 6});
  fill_tensor(metal_a, {2, 4, 6, 8, 10, 12});
  fill_tensor(metal_b, {1, 2, 3, 4, 5, 6});

  assert_backend_parity(cpu::add(cpu_a, cpu_b),
                        metal_test_backend::add(metal_a, metal_b),
                        "test_cpu_metal_elementwise_parity add");
  assert_backend_parity(cpu::sub(cpu_a, cpu_b),
                        metal_test_backend::sub(metal_a, metal_b),
                        "test_cpu_metal_elementwise_parity sub");
  assert_backend_parity(cpu::hadamard(cpu_a, cpu_b),
                        metal_test_backend::hadamard(metal_a, metal_b),
                        "test_cpu_metal_elementwise_parity hadamard");
  assert_backend_parity(cpu::divide(cpu_a, cpu_b),
                        metal_test_backend::divide(metal_a, metal_b),
                        "test_cpu_metal_elementwise_parity divide");
  assert_backend_parity(cpu::sqrt(cpu_a), metal_test_backend::sqrt(metal_a),
                        "test_cpu_metal_elementwise_parity sqrt");
  assert_backend_parity(cpu::sigmoid(cpu_a),
                        metal_test_backend::sigmoid(metal_a),
                        "test_cpu_metal_elementwise_parity sigmoid");
}

void test_cpu_metal_matmul_parity() {
  CpuTensor cpu_a({2, 3});
  CpuTensor cpu_b({3, 2});
  Tensor metal_a({2, 3});
  Tensor metal_b({3, 2});

  fill_tensor(cpu_a, {1, 2, 3, 4, 5, 6});
  fill_tensor(cpu_b, {7, 8, 9, 10, 11, 12});
  fill_tensor(metal_a, {1, 2, 3, 4, 5, 6});
  fill_tensor(metal_b, {7, 8, 9, 10, 11, 12});

  assert_backend_parity(cpu::mult(cpu_a, cpu_b),
                        metal_test_backend::mult(metal_a, metal_b),
                        "test_cpu_metal_matmul_parity");
}

void test_cpu_metal_transpose_parity() {
  CpuTensor cpu_a({2, 3});
  Tensor metal_a({2, 3});

  fill_tensor(cpu_a, {1, 2, 3, 4, 5, 6});
  fill_tensor(metal_a, {1, 2, 3, 4, 5, 6});

  assert_backend_parity(cpu::transpose(cpu_a),
                        metal_test_backend::transpose(metal_a),
                        "test_cpu_metal_transpose_parity");
}

void test_cpu_metal_argmax_parity() {
  CpuTensor cpu_a({2, 3});
  Tensor metal_a({2, 3});

  fill_tensor(cpu_a, {1, 4, 2, 9, 3, 5});
  fill_tensor(metal_a, {1, 4, 2, 9, 3, 5});

  assert_backend_parity(cpu::argmax(cpu_a),
                        metal_test_backend::argmax(metal_a),
                        "test_cpu_metal_argmax_parity");
}

void test_cpu_metal_sum_parity() {
  CpuTensor cpu_a({2, 3});
  Tensor metal_a({2, 3});

  fill_tensor(cpu_a, {1, 2, 3, 4, 5, 6});
  fill_tensor(metal_a, {1, 2, 3, 4, 5, 6});

  CpuTensor cpu_out = cpu::sum(cpu_a);
  Tensor metal_out = metal_test_backend::sum(metal_a);

  assert_backend_parity(cpu_out, metal_out, "test_cpu_metal_sum_parity");
}

void test_cpu_metal_sum_to_shape_parity() {
  CpuTensor cpu_a({2, 3});
  Tensor metal_a({2, 3});

  fill_tensor(cpu_a, {1, 2, 3, 4, 5, 6});
  fill_tensor(metal_a, {1, 2, 3, 4, 5, 6});

  std::vector<uint32_t> shape = {2, 1};
  CpuTensor cpu_out = cpu::sum_to_shape(cpu_a, shape);
  Tensor metal_out = metal_test_backend::sum_to_shape(metal_a, shape);

  assert_backend_parity(cpu_out, metal_out,
                        "test_cpu_metal_sum_to_shape_parity");
}

void test_metal_matmul_broadcast_rhs() {
  Tensor a({2, 2, 3});
  Tensor b({3, 2});

  fill_tensor(a, {// batch 0
                  1, 2, 3, 4, 5, 6,
                  // batch 1
                  2, 0, 1, 1, 3, 2});

  fill_tensor(b, {7, 8, 9, 10, 11, 12});

  Tensor out = metal_test_backend::mult(a, b);

  assert(out.shape().size() == 3);
  assert(out.shape()[0] == 2);
  assert(out.shape()[1] == 2);
  assert(out.shape()[2] == 2);

  assert_tensor_values(out,
                       {// batch 0
                        58, 64, 139, 154,

                        // batch 1
                        25, 28, 56, 62},
                       "test_metal_matmul_broadcast_rhs");
}

void test_metal_matmul_broadcast_batch_dims() {
  Tensor a({2, 1, 2, 3});
  Tensor b({1, 2, 3, 2});

  fill_tensor(a, {// A batch [0,0]
                  1, 2, 3, 4, 5, 6,
                  // A batch [1,0]
                  2, 0, 1, 1, 3, 2});

  fill_tensor(b, {// B batch [0,0]
                  7, 8, 9, 10, 11, 12,
                  // B batch [0,1]
                  1, 2, 3, 4, 5, 6});

  Tensor out = metal_test_backend::mult(a, b);

  assert(out.shape().size() == 4);
  assert(out.shape()[0] == 2);
  assert(out.shape()[1] == 2);
  assert(out.shape()[2] == 2);
  assert(out.shape()[3] == 2);

  assert_tensor_values(out,
                       {// out batch [0,0]
                        58, 64, 139, 154,
                        // out batch [0,1]
                        22, 28, 49, 64,
                        // out batch [1,0]
                        25, 28, 56, 62,
                        // out batch [1,1]
                        7, 10, 20, 26},
                       "test_metal_matmul_broadcast_batch_dims");
}

void test_metal_transpose_batched() {
  Tensor a({2, 2, 3});
  fill_tensor(a, {// batch 0
                  1, 2, 3, 4, 5, 6,
                  // batch 1
                  7, 8, 9, 10, 11, 12});

  Tensor out = metal_test_backend::transpose(a);

  assert_tensor_shape(out, {2, 3, 2}, "test_metal_transpose_batched");
  assert_tensor_values(out,
                       {// batch 0
                        1, 4, 2, 5, 3, 6,
                        // batch 1
                        7, 10, 8, 11, 9, 12},
                       "test_metal_transpose_batched");
}

void test_metal_transpose_rank_one_rejected() {
  Tensor a({3});
  fill_tensor(a, {1, 2, 3});

  bool threw = false;

  try {
    Tensor out = metal_test_backend::transpose(a);
    (void)out;
  } catch (const std::exception &) {
    threw = true;
  }

  if (!threw) {
    std::cerr << "\nFAILED: test_metal_transpose_rank_one_rejected\n";
    std::cerr << "Expected transpose to reject rank-one tensor\n";
    assert(false);
  }
}

void test_metal_sum_to_same_shape() {
  Tensor a({2, 3});
  fill_tensor(a, {1, 2, 3, 4, 5, 6});

  std::vector<uint32_t> shape = {2, 3};
  Tensor out = metal_test_backend::sum_to_shape(a, shape);

  assert_tensor_shape(out, {2, 3}, "test_metal_sum_to_same_shape");
  assert_tensor_values(out, {1, 2, 3, 4, 5, 6}, "test_metal_sum_to_same_shape");
}

void test_metal_sum_to_invalid_shape() {
  Tensor a({2, 3});
  fill_tensor(a, {1, 2, 3, 4, 5, 6});

  std::vector<uint32_t> shape = {3, 2};
  bool threw = false;

  try {
    Tensor out = metal_test_backend::sum_to_shape(a, shape);
    (void)out;
  } catch (const std::exception &) {
    threw = true;
  }

  if (!threw) {
    std::cerr << "\nFAILED: test_metal_sum_to_invalid_shape\n";
    std::cerr << "Expected sum_to_shape to reject incompatible shape\n";
    assert(false);
  }
}

void test_metal_max_to_same_shape() {
  Tensor a({2, 3});
  fill_tensor(a, {1, 2, 3, 4, 5, 6});

  std::vector<uint32_t> shape = {2, 3};
  Tensor out = metal_test_backend::max_to_shape(a, shape);

  assert_tensor_shape(out, {2, 3}, "test_metal_max_to_same_shape");
  assert_tensor_values(out, {1, 2, 3, 4, 5, 6}, "test_metal_max_to_same_shape");
}

void test_metal_max_to_row_shape() {
  Tensor a({2, 3});
  fill_tensor(a, {1, 9, 3, 4, 5, 8});

  std::vector<uint32_t> shape = {1, 3};
  Tensor out = metal_test_backend::max_to_shape(a, shape);

  assert_tensor_shape(out, {1, 3}, "test_metal_max_to_row_shape");
  assert_tensor_values(out, {4, 9, 8}, "test_metal_max_to_row_shape");
}

void test_metal_max_to_shape_negative_values() {
  Tensor a({2, 3});
  fill_tensor(a, {-9, -2, -7, -4, -6, -8});

  std::vector<uint32_t> shape = {2, 1};
  Tensor out = metal_test_backend::max_to_shape(a, shape);

  assert_tensor_shape(out, {2, 1}, "test_metal_max_to_shape_negative_values");
  assert_tensor_values(out, {-2, -4},
                       "test_metal_max_to_shape_negative_values");
}

void test_metal_max_to_invalid_shape() {
  Tensor a({2, 3});
  fill_tensor(a, {1, 2, 3, 4, 5, 6});

  std::vector<uint32_t> shape = {3, 2};
  bool threw = false;

  try {
    Tensor out = metal_test_backend::max_to_shape(a, shape);
    (void)out;
  } catch (const std::exception &) {
    threw = true;
  }

  if (!threw) {
    std::cerr << "\nFAILED: test_metal_max_to_invalid_shape\n";
    std::cerr << "Expected max_to_shape to reject incompatible shape\n";
    assert(false);
  }
}

void run_test(std::string_view name, void (*test)()) {
  std::cout << "Running " << name << "... " << std::flush;

  test();

  std::cout << "PASSED\n";
}

int main() {
  metal_test_backend::init();

  run_test("test_metal_broadcast_singleton", test_metal_broadcast_singleton);
  run_test("test_cpu_metal_elementwise_parity",
           test_cpu_metal_elementwise_parity);
  run_test("test_cpu_metal_matmul_parity", test_cpu_metal_matmul_parity);
  run_test("test_cpu_metal_transpose_parity", test_cpu_metal_transpose_parity);
  run_test("test_cpu_metal_argmax_parity", test_cpu_metal_argmax_parity);
  run_test("test_cpu_metal_sum_parity", test_cpu_metal_sum_parity);
  run_test("test_cpu_metal_sum_to_shape_parity",
           test_cpu_metal_sum_to_shape_parity);
  run_test("test_metal_matmul_broadcast_rhs", test_metal_matmul_broadcast_rhs);
  run_test("test_metal_matmul_broadcast_batch_dims",
           test_metal_matmul_broadcast_batch_dims);
  run_test("test_metal_transpose_batched", test_metal_transpose_batched);
  run_test("test_metal_transpose_rank_one_rejected",
           test_metal_transpose_rank_one_rejected);
  run_test("test_metal_sum_to_same_shape", test_metal_sum_to_same_shape);
  run_test("test_metal_sum_to_invalid_shape", test_metal_sum_to_invalid_shape);
  run_test("test_metal_max_to_same_shape", test_metal_max_to_same_shape);
  run_test("test_metal_max_to_row_shape", test_metal_max_to_row_shape);
  run_test("test_metal_max_to_shape_negative_values",
           test_metal_max_to_shape_negative_values);
  run_test("test_metal_max_to_invalid_shape", test_metal_max_to_invalid_shape);
  metal_test_backend::shutdown();

  std::cout << "\nAll Metal tests passed\n";
}
