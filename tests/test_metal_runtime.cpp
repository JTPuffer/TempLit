#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "my_metal/runtime.h"
#include "my_metal/storage.h"
#include "optimiser/AdamW.h"

using Tensor = RTensor<metal::Metal, metal::kernel_type>;
using IndexTensor = RTensor<metal::Metal, metal::index_type>;

static_assert(passes::Runnable<Tensor, metal::Runtime>);

constexpr float EPS = 1e-5f;

template <typename TensorT> size_t tensor_size(const TensorT &tensor) {
  return std::accumulate(tensor.shape().begin(), tensor.shape().end(),
                         size_t{1}, std::multiplies<>{});
}

void fill_tensor(Tensor &tensor, std::initializer_list<float> values) {
  assert(values.size() == tensor_size(tensor));
  tensor.get_storage()->write(
      std::span<const float>(values.begin(), values.size()), 0);
}

template <typename TensorT>
std::vector<typename TensorT::value_type> read_tensor(const TensorT &tensor) {
  std::vector<typename TensorT::value_type> values(tensor_size(tensor));
  tensor.get_storage()->read(std::span<typename TensorT::value_type>(values),
                             0);
  return values;
}

template <typename TensorT>
void assert_shape(const TensorT &tensor,
                  std::initializer_list<uint32_t> expected,
                  std::string_view test_name) {
  const std::vector<uint32_t> expected_shape(expected);
  if (tensor.shape() != expected_shape) {
    std::cerr << "\nFAILED: " << test_name << "\nUnexpected tensor shape\n";
    assert(false);
  }
}

void assert_values(const Tensor &tensor, std::initializer_list<float> expected,
                   std::string_view test_name, float eps = EPS) {
  const std::vector<float> actual = read_tensor(tensor);
  assert(actual.size() == expected.size());

  size_t index = 0;
  for (float expected_value : expected) {
    if (std::abs(actual[index] - expected_value) >= eps) {
      std::cerr << "\nFAILED: " << test_name << "\nMismatch at index " << index
                << ": got " << actual[index] << ", expected " << expected_value
                << '\n';
      assert(false);
    }
    ++index;
  }
}

void assert_index(const IndexTensor &tensor, metal::index_type expected,
                  std::string_view test_name) {
  assert_shape(tensor, {1, 1}, test_name);
  const std::vector<metal::index_type> actual = read_tensor(tensor);

  if (actual != std::vector<metal::index_type>{expected}) {
    std::cerr << "\nFAILED: " << test_name << "\nGot index " << actual.front()
              << ", expected " << expected << '\n';
    assert(false);
  }
}

void test_runtime_lifecycle(metal::Runtime &runtime) {
  bool threw = false;
  try {
    runtime.submit();
  } catch (const std::logic_error &) {
    threw = true;
  }
  assert(threw);

  runtime.start();

  threw = false;
  try {
    runtime.start();
  } catch (const std::logic_error &) {
    threw = true;
  }
  assert(threw);

  runtime.abort();
  runtime.start();
  runtime.submit();

  threw = false;
  try {
    runtime.submit();
  } catch (const std::logic_error &) {
    threw = true;
  }
  assert(threw);

  runtime.wait();

  threw = false;
  try {
    runtime.wait();
  } catch (const std::logic_error &) {
    threw = true;
  }
  assert(threw);
}

void test_runtime_cleanup_is_idempotent() {
  metal::Runtime runtime;
  runtime.cleanup();
  runtime.cleanup();
}

void test_runtime_binary_and_chained_operations(metal::Runtime &runtime) {
  Tensor a({2, 2});
  Tensor b({2, 2});
  fill_tensor(a, {1, 2, 3, 4});
  fill_tensor(b, {5, 6, 7, 8});

  Tensor added({2, 2});
  Tensor subtracted({2, 2});
  Tensor multiplied({2, 2});
  Tensor divided({2, 2});
  Tensor chained({2, 2});

  runtime.start();
  runtime.add(a, b, added);
  runtime.sub(a, b, subtracted);
  runtime.hadamard(a, b, multiplied);
  runtime.divide(a, b, divided);
  runtime.hadamard(added, a, chained);
  runtime.submit();
  runtime.wait();

  assert_values(added, {6, 8, 10, 12}, "runtime add");
  assert_values(subtracted, {-4, -4, -4, -4}, "runtime sub");
  assert_values(multiplied, {5, 12, 21, 32}, "runtime hadamard");
  assert_values(divided, {0.2f, 1.0f / 3.0f, 3.0f / 7.0f, 0.5f},
                "runtime divide");
  assert_values(chained, {6, 16, 30, 48}, "runtime chained operations");
}

void test_runtime_broadcasting(metal::Runtime &runtime) {
  Tensor matrix({2, 3});
  Tensor row({3});
  fill_tensor(matrix, {1, 2, 3, 4, 5, 6});
  fill_tensor(row, {10, 20, 30});
  Tensor out({2, 3});

  runtime.start();
  runtime.add(matrix, row, out);
  runtime.submit();
  runtime.wait();

  assert_shape(out, {2, 3}, "runtime broadcasting");
  assert_values(out, {11, 22, 33, 14, 25, 36}, "runtime broadcasting");
}

void test_runtime_scalar_operations(metal::Runtime &runtime) {
  Tensor input({2, 2});
  Tensor added({2, 2});
  Tensor divided({2, 2});
  Tensor cubed({2, 2});
  Tensor in_place({2, 2});
  fill_tensor(input, {2, 4, 6, 8});
  fill_tensor(in_place, {2, 4, 6, 8});

  runtime.start();
  runtime.add(input, 3.0f, added);
  runtime.divide(input, 2.0f, divided);
  runtime.pow(input, 3.0f, cubed);
  runtime.divide(in_place, 2.0f, in_place);
  runtime.pow(in_place, 2.0f, in_place);
  runtime.submit();
  runtime.wait();

  assert_values(added, {5, 7, 9, 11}, "runtime scalar add");
  assert_values(divided, {1, 2, 3, 4}, "runtime scalar divide");
  assert_values(cubed, {8, 64, 216, 512}, "runtime scalar power", 1e-3f);
  assert_values(in_place, {1, 4, 9, 16}, "runtime in-place scalar power",
                1e-3f);
}

void test_runtime_unary_operations(metal::Runtime &runtime) {
  Tensor input({4});
  Tensor squares({4});
  fill_tensor(input, {0, 1, -1, 2});
  fill_tensor(squares, {0, 1, 4, 9});
  Tensor sigmoid({4});
  Tensor exponential({4});
  Tensor square_root({4});

  runtime.start();
  runtime.sigmoid(input, sigmoid);
  runtime.exp(input, exponential);
  runtime.sqrt(squares, square_root);
  runtime.submit();
  runtime.wait();

  assert_values(sigmoid,
                {0.5f, 1.0f / (1.0f + std::exp(-1.0f)),
                 1.0f / (1.0f + std::exp(1.0f)),
                 1.0f / (1.0f + std::exp(-2.0f))},
                "runtime sigmoid");
  assert_values(exponential,
                {1.0f, std::exp(1.0f), std::exp(-1.0f), std::exp(2.0f)},
                "runtime exp");
  assert_values(square_root, {0, 1, 2, 3}, "runtime sqrt");
}

void test_runtime_rms_and_rms_norm(metal::Runtime &runtime) {
  Tensor input({2, 2});
  Tensor rms({2, 1});
  Tensor normalised({2, 2});
  fill_tensor(input, {3, 4, 0, 0});

  runtime.start();
  runtime.RMS(input, rms);
  runtime.RMSNorm(input, normalised);
  runtime.submit();
  runtime.wait();

  const float first_rms = std::sqrt(12.5f + 1e-6f);
  assert_values(rms, {first_rms, 1e-3f}, "runtime RMS");
  assert_values(normalised, {3.0f / first_rms, 4.0f / first_rms, 0, 0},
                "runtime RMSNorm");
}

void test_runtime_gelu(metal::Runtime &runtime) {
  Tensor input({5});
  Tensor output({5});
  Tensor derivative({5});
  fill_tensor(input, {-2, -1, 0, 1, 2});

  runtime.start();
  runtime.GELU(input, output);
  runtime.GELU_derivative(input, derivative);
  runtime.submit();
  runtime.wait();

  const auto gelu = [](float x) {
    constexpr float a = 0.7978845608f;
    return 0.5f * x *
           (1.0f + std::tanh(a * (x + 0.044715f * x * x * x)));
  };
  const auto gelu_derivative = [](float x) {
    constexpr float a = 0.7978845608f;
    constexpr float b = 0.044715f;
    const float x_squared = x * x;
    const float tanh_value = std::tanh(a * (x + b * x_squared * x));
    return 0.5f * (1.0f + tanh_value) +
           0.5f * x * (1.0f - tanh_value * tanh_value) * a *
               (1.0f + 3.0f * b * x_squared);
  };

  assert_values(output, {gelu(-2), gelu(-1), gelu(0), gelu(1), gelu(2)},
                "runtime GELU");
  assert_values(derivative,
                {gelu_derivative(-2), gelu_derivative(-1),
                 gelu_derivative(0), gelu_derivative(1), gelu_derivative(2)},
                "runtime GELU derivative");
}

void test_runtime_matmul(metal::Runtime &runtime) {
  Tensor a({2, 3});
  Tensor b({3, 2});
  fill_tensor(a, {1, 2, 3, 4, 5, 6});
  fill_tensor(b, {7, 8, 9, 10, 11, 12});

  Tensor batch_a({2, 2, 2});
  Tensor batch_b({2, 2, 2});
  fill_tensor(batch_a, {1, 2, 3, 4, 2, 0, 1, 2});
  fill_tensor(batch_b, {5, 6, 7, 8, 1, 3, 4, 2});
  Tensor out({2, 2});
  Tensor batch_out({2, 2, 2});

  runtime.start();
  runtime.mult(a, b, out);
  runtime.mult(batch_a, batch_b, batch_out);
  runtime.submit();
  runtime.wait();

  assert_shape(out, {2, 2}, "runtime matmul");
  assert_values(out, {58, 64, 139, 154}, "runtime matmul");
  assert_shape(batch_out, {2, 2, 2}, "runtime batched matmul");
  assert_values(batch_out, {19, 22, 43, 50, 2, 6, 9, 7},
                "runtime batched matmul");
}

void test_runtime_tiled_contiguous_matmul(metal::Runtime &runtime) {
  constexpr uint32_t size = 64;
  constexpr uint32_t batches = 2;
  Tensor identity({batches, size, size});
  Tensor values({batches, size, size});
  Tensor output({batches, size, size});

  std::vector<float> identity_values(batches * size * size, 0.0f);
  std::vector<float> input_values(batches * size * size);
  for (uint32_t batch = 0; batch < batches; ++batch) {
    for (uint32_t row = 0; row < size; ++row) {
      identity_values[batch * size * size + row * size + row] = 1.0f;
    }
  }
  for (uint32_t index = 0; index < input_values.size(); ++index) {
    input_values[index] = static_cast<float>(index % 31) - 15.0f;
  }

  identity.set(identity_values);
  values.set(input_values);

  runtime.start();
  runtime.mult(identity, values, output);
  runtime.submit();
  runtime.wait();

  const auto actual = read_tensor(output);
  for (uint32_t index = 0; index < actual.size(); ++index) {
    if (std::abs(actual[index] - input_values[index]) >= EPS) {
      std::cerr << "\nFAILED: runtime tiled contiguous matmul\nMismatch at "
                << index << ": got " << actual[index] << ", expected "
                << input_values[index] << '\n';
      assert(false);
    }
  }
}

void test_runtime_reductions_and_transpose(metal::Runtime &runtime) {
  Tensor input({2, 3});
  fill_tensor(input, {1, 2, 3, 4, 5, 6});
  const std::vector<uint32_t> column_shape = {2, 1};
  const std::vector<uint32_t> row_shape = {1, 3};
  Tensor total({1});
  Tensor row_sums({2, 1});
  Tensor column_sums(row_shape);
  Tensor row_maxima(column_shape);
  Tensor transposed({3, 2});

  runtime.start();
  runtime.sum(input, total);
  runtime.sum(input, 1, row_sums);
  runtime.sum_to_shape(input, row_shape, column_sums);
  runtime.max_to_shape(input, column_shape, row_maxima);
  runtime.transpose(input, transposed);
  runtime.submit();
  runtime.wait();

  assert_shape(total, {1}, "runtime sum");
  assert_values(total, {21}, "runtime sum");
  assert_shape(row_sums, {2, 1}, "runtime axis sum");
  assert_values(row_sums, {6, 15}, "runtime axis sum");
  assert_shape(column_sums, {1, 3}, "runtime sum_to_shape");
  assert_values(column_sums, {5, 7, 9}, "runtime sum_to_shape");
  assert_shape(row_maxima, {2, 1}, "runtime max_to_shape");
  assert_values(row_maxima, {3, 6}, "runtime max_to_shape");
  assert_shape(transposed, {3, 2}, "runtime transpose");
  assert_values(transposed, {1, 4, 2, 5, 3, 6}, "runtime transpose");
}

void test_runtime_sum_to_shape_reduces_multiple_dimensions(
    metal::Runtime &runtime) {
  Tensor input({2, 3, 4});
  fill_tensor(input, {1,  2,  3,  4,  5,  6,  7,  8,
                      9,  10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24});

  const std::vector<uint32_t> non_adjacent_shape = {1, 3, 1};
  const std::vector<uint32_t> adjacent_shape = {2, 1, 1};
  const std::vector<uint32_t> scalar_shape = {1, 1, 1};
  const std::vector<uint32_t> trailing_shape = {4};
  Tensor non_adjacent(non_adjacent_shape);
  Tensor adjacent(adjacent_shape);
  Tensor scalar(scalar_shape);
  Tensor trailing(trailing_shape);

  runtime.start();
  runtime.sum_to_shape(input, non_adjacent_shape, non_adjacent);
  runtime.sum_to_shape(input, adjacent_shape, adjacent);
  runtime.sum_to_shape(input, scalar_shape, scalar);
  runtime.sum_to_shape(input, trailing_shape, trailing);
  runtime.submit();
  runtime.wait();

  assert_shape(non_adjacent, {1, 3, 1},
               "runtime sum_to_shape non-adjacent dimensions");
  assert_values(non_adjacent, {68, 100, 132},
                "runtime sum_to_shape non-adjacent dimensions");
  assert_shape(adjacent, {2, 1, 1},
               "runtime sum_to_shape adjacent dimensions");
  assert_values(adjacent, {78, 222},
                "runtime sum_to_shape adjacent dimensions");
  assert_shape(scalar, {1, 1, 1}, "runtime sum_to_shape all dimensions");
  assert_values(scalar, {300}, "runtime sum_to_shape all dimensions");
  assert_shape(trailing, {4}, "runtime sum_to_shape lower rank");
  assert_values(trailing, {66, 72, 78, 84},
                "runtime sum_to_shape lower rank");
}

void test_runtime_sum_to_shape_removes_leading_singletons(
    metal::Runtime &runtime) {
  Tensor input({1, 4});
  fill_tensor(input, {1, 2, 3, 4});
  const std::vector<uint32_t> output_shape = {4};
  Tensor output(output_shape);

  runtime.start();
  runtime.sum_to_shape(input, output_shape, output);
  runtime.submit();
  runtime.wait();

  assert_shape(output, {4}, "runtime sum_to_shape removes leading singleton");
  assert_values(output, {1, 2, 3, 4},
                "runtime sum_to_shape removes leading singleton");
}

void test_runtime_softmax(metal::Runtime &runtime) {
  Tensor input({2, 3});
  fill_tensor(input, {1, 2, 3, 1, 1, 1});
  Tensor out({2, 3});

  Tensor rank_one({3});
  fill_tensor(rank_one, {1, 2, 3});
  Tensor rank_one_out({3});

  Tensor rank_four({2, 2, 1, 3});
  fill_tensor(rank_four,
              {1, 2, 3, 3, 2, 1, 1000, 1001, 1002, -1000, -1000, -1000});
  Tensor rank_four_out({2, 2, 1, 3});

  Tensor singleton({2, 1});
  fill_tensor(singleton, {4, -7});
  Tensor singleton_out({2, 1});

  runtime.start();
  runtime.softmax(input, out);
  runtime.softmax(rank_one, rank_one_out);
  runtime.softmax(rank_four, rank_four_out);
  runtime.softmax(singleton, singleton_out);
  runtime.submit();
  runtime.wait();

  const float denominator = std::exp(-2.0f) + std::exp(-1.0f) + std::exp(0.0f);
  const float low = std::exp(-2.0f) / denominator;
  const float middle = std::exp(-1.0f) / denominator;
  const float high = 1.0f / denominator;
  assert_values(out,
                {low, middle, high, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f},
                "runtime softmax");
  assert_values(rank_one_out, {low, middle, high},
                "runtime softmax rank one");
  assert_values(rank_four_out,
                {low, middle, high, high, middle, low, low, middle, high,
                 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f},
                "runtime softmax rank four");
  assert_values(singleton_out, {1, 1}, "runtime softmax singleton rows");
}

void test_runtime_cross_entropy(metal::Runtime &runtime) {
  Tensor logits({2, 2, 3});
  fill_tensor(logits,
              {1, 2, 3, 1, 1, 1, 1000, 1001, 1002, -1, -2, -3});

  IndexTensor targets({2, 2});
  const std::vector<metal::index_type> target_values{2, 0, 1, 2};
  targets.get_storage()->write(
      std::span<const metal::index_type>(target_values), 0);

  Tensor losses({2, 2, 1});
  Tensor gradients({2, 2, 3});
  runtime.start();
  runtime.cross_entropy(logits, targets, losses);
  runtime.cross_entropy_derivative(logits, targets, gradients);
  runtime.submit();
  runtime.wait();

  const float denominator = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
  const float low = std::exp(-2.0f) / denominator;
  const float middle = std::exp(-1.0f) / denominator;
  const float high = 1.0f / denominator;
  const float log_denominator =
      std::log(denominator);
  assert_values(losses,
                {log_denominator, std::log(3.0f), 1.0f + log_denominator,
                 2.0f + log_denominator},
                "runtime cross entropy", 1e-4f);
  assert_values(gradients,
                {low, middle, high - 1.0f, -2.0f / 3.0f, 1.0f / 3.0f,
                 1.0f / 3.0f, low, middle - 1.0f, high, high, middle,
                 low - 1.0f},
                "runtime cross entropy derivative", 1e-4f);
}

void test_runtime_argmax(metal::Runtime &runtime) {
  Tensor small({2, 3});
  fill_tensor(small, {1, 4, 2, 9, 3, 5});

  constexpr size_t element_count = 5000;
  constexpr metal::index_type expected_index = 4097;
  Tensor large({element_count});
  std::vector<float> values(element_count, 0.0f);
  values[expected_index] = 100.0f;
  large.get_storage()->write(std::span<const float>(values), 0);
  IndexTensor small_out({1, 1});
  IndexTensor large_out({1, 1});

  runtime.start();
  runtime.argmax(small, small_out);
  runtime.argmax(large, large_out);
  runtime.submit();
  runtime.wait();

  static_assert(std::is_same_v<decltype(small_out), IndexTensor>);
  assert_index(small_out, 3, "runtime argmax small tensor");
  assert_index(large_out, expected_index, "runtime argmax multiple reductions");
}

void test_runtime_embedding_lookup(metal::Runtime &runtime) {
  Tensor weights({4, 3});
  fill_tensor(weights, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  IndexTensor ids({2, 2});
  const std::vector<metal::index_type> index_values{2, 0, 3, 1};
  ids.get_storage()->write(std::span<const metal::index_type>(index_values), 0);

  Tensor output({2, 2, 3});
  runtime.start();
  runtime.embedding_lookup(weights, ids, output);
  runtime.submit();
  runtime.wait();

  assert_values(output, {7, 8, 9, 1, 2, 3, 10, 11, 12, 4, 5, 6},
                "runtime embedding lookup");
}

void test_runtime_scatter_index(metal::Runtime &runtime) {
  Tensor values({2, 2, 3});
  fill_tensor(values, {1, 2, 3, 4, 5, 6, 10, 20, 30, 7, 8, 9});

  IndexTensor ids({2, 2});
  const std::vector<metal::index_type> index_values{2, 0, 2, 1};
  ids.get_storage()->write(std::span<const metal::index_type>(index_values), 0);

  Tensor output({4, 3});
  fill_tensor(output, {99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99});

  runtime.start();
  runtime.scatter_index(values, ids, output);
  runtime.submit();
  runtime.wait();

  assert_values(output, {4, 5, 6, 7, 8, 9, 11, 22, 33, 0, 0, 0},
                "runtime scatter index");
}

void test_runtime_scatter_add_rows(metal::Runtime &runtime) {
  Tensor base({4, 3});
  fill_tensor(base,
              {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111});

  Tensor values({2, 2, 3});
  fill_tensor(values, {1, 2, 3, 4, 5, 6, 10, 20, 30, 7, 8, 9});

  IndexTensor ids({2, 2});
  const std::vector<metal::index_type> index_values{2, 0, 2, 1};
  ids.get_storage()->write(std::span<const metal::index_type>(index_values), 0);

  Tensor output({4, 3});
  runtime.start();
  runtime.scatter_add_rows(base, ids, values, output);
  runtime.submit();
  runtime.wait();

  assert_values(output,
                {104, 106, 108, 110, 112, 114, 117, 129, 141, 109, 110,
                 111},
                "runtime scatter add rows");
}

void test_runtime_gradient_clipping(metal::Runtime &runtime) {
  Tensor first({3});
  Tensor second({1});
  Tensor squared_norm = Tensor::from_zeros({1});
  fill_tensor(first, {3.0f, 4.0f, 0.0f});
  fill_tensor(second, {12.0f});

  runtime.start();
  runtime.accumulate_gradient_squared_norm(first, squared_norm);
  runtime.accumulate_gradient_squared_norm(second, squared_norm);
  runtime.clip_gradient(first, squared_norm, 1.0f, first);
  runtime.clip_gradient(second, squared_norm, 1.0f, second);
  runtime.submit();
  runtime.wait();

  assert_values(squared_norm, {169.0f}, "runtime gradient squared norm");
  assert_values(first, {3.0f / 13.0f, 4.0f / 13.0f, 0.0f},
                "runtime clipped first gradient");
  assert_values(second, {12.0f / 13.0f},
                "runtime clipped second gradient");
}

void test_runtime_adamw(metal::Runtime &runtime) {
  using AdamW = Optimiser::AdamW<Tensor>;

  Tensor parameter({4});
  Tensor gradient({4});
  fill_tensor(parameter, {1.0f, -2.0f, 3.0f, -4.0f});
  fill_tensor(gradient, {0.5f, -0.25f, 1.0f, -0.75f});

  AdamW optimiser;
  auto params = AdamW::generate_params(0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

  std::vector<float> expected_parameter{1.0f, -2.0f, 3.0f, -4.0f};
  const std::vector<float> gradients{0.5f, -0.25f, 1.0f, -0.75f};
  std::vector<float> momentum1(4, 0.0f);
  std::vector<float> momentum2(4, 0.0f);

  for (uint32_t step = 1; step <= 2; ++step) {
    runtime.start();
    optimiser.update(runtime, params, parameter, gradient, parameter);
    runtime.submit();
    runtime.wait();

    const float correction1 = 1.0f - std::pow(0.9f, static_cast<float>(step));
    const float correction2 =
        1.0f - std::pow(0.999f, static_cast<float>(step));
    for (size_t index = 0; index < expected_parameter.size(); ++index) {
      momentum1[index] =
          0.9f * momentum1[index] + 0.1f * gradients[index];
      momentum2[index] = 0.999f * momentum2[index] +
                         0.001f * gradients[index] * gradients[index];
      expected_parameter[index] =
          expected_parameter[index] * 0.9999f -
          0.01f * (momentum1[index] / correction1) /
              (std::sqrt(momentum2[index] / correction2) + 1e-8f);
    }

    assert_values(parameter,
                  {expected_parameter[0], expected_parameter[1],
                   expected_parameter[2], expected_parameter[3]},
                  "runtime AdamW", 1e-5f);
  }
}

template <typename Test> void run_test(std::string_view name, Test &&test) {
  std::cout << "Running " << name << "... " << std::flush;
  test();
  std::cout << "PASSED\n";
}

int main() {
  metal::init();

  {
    metal::Runtime runtime;

    run_test("test_runtime_lifecycle",
             [&] { test_runtime_lifecycle(runtime); });
    run_test("test_runtime_binary_and_chained_operations",
             [&] { test_runtime_binary_and_chained_operations(runtime); });
    run_test("test_runtime_broadcasting",
             [&] { test_runtime_broadcasting(runtime); });
    run_test("test_runtime_scalar_operations",
             [&] { test_runtime_scalar_operations(runtime); });
    run_test("test_runtime_unary_operations",
             [&] { test_runtime_unary_operations(runtime); });
    run_test("test_runtime_rms_and_rms_norm",
             [&] { test_runtime_rms_and_rms_norm(runtime); });
    run_test("test_runtime_gelu", [&] { test_runtime_gelu(runtime); });
    run_test("test_runtime_matmul", [&] { test_runtime_matmul(runtime); });
    run_test("test_runtime_tiled_contiguous_matmul",
             [&] { test_runtime_tiled_contiguous_matmul(runtime); });
    run_test("test_runtime_reductions_and_transpose",
             [&] { test_runtime_reductions_and_transpose(runtime); });
    run_test("test_runtime_sum_to_shape_reduces_multiple_dimensions", [&] {
      test_runtime_sum_to_shape_reduces_multiple_dimensions(runtime);
    });
    run_test("test_runtime_sum_to_shape_removes_leading_singletons", [&] {
      test_runtime_sum_to_shape_removes_leading_singletons(runtime);
    });
    run_test("test_runtime_softmax", [&] { test_runtime_softmax(runtime); });
    run_test("test_runtime_cross_entropy",
             [&] { test_runtime_cross_entropy(runtime); });
    run_test("test_runtime_argmax", [&] { test_runtime_argmax(runtime); });
    run_test("test_runtime_embedding_lookup",
             [&] { test_runtime_embedding_lookup(runtime); });
    run_test("test_runtime_scatter_index",
             [&] { test_runtime_scatter_index(runtime); });
    run_test("test_runtime_scatter_add_rows",
             [&] { test_runtime_scatter_add_rows(runtime); });
    run_test("test_runtime_gradient_clipping",
             [&] { test_runtime_gradient_clipping(runtime); });
    run_test("test_runtime_adamw", [&] { test_runtime_adamw(runtime); });
  }

  run_test("test_runtime_cleanup_is_idempotent",
           test_runtime_cleanup_is_idempotent);

  metal::shutdown();
  std::cout << "\nAll Metal runtime tests passed\n";
}
