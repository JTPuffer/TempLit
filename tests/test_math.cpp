#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "test_backend.h"

using namespace test_backend;

constexpr kernel_type EPS = 1e-5f;

bool approx(kernel_type a, kernel_type b, kernel_type eps = EPS) {
  return std::abs(a - b) < eps;
}

Mat scalar(kernel_type value) { return Mat(value); }

void assert_matrix_approx(const Mat &a, const Mat &b, kernel_type eps = EPS) {
  assert(a.shape() == b.shape());

  size_t size = 1;
  for (const size_t dim : a.shape()) {
    size *= dim;
  }

  std::vector<kernel_type> a_data(size);
  std::vector<kernel_type> b_data(size);
  a.get_storage()->read(std::span<kernel_type>(a_data), 0);
  b.get_storage()->read(std::span<kernel_type>(b_data), 0);

  for (size_t i = 0; i < size; ++i) {
    assert(approx(a_data[i], b_data[i], eps));
  }
}

void test_scalar_left_addition() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = add(scalar(10.0f), a);

  Mat expected = Mat::from_values({
      {11, 12},
      {13, 14},
  });

  assert(result == expected);
}

void test_scalar_left_multiplication() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = hadamard(scalar(3.0f), a);

  Mat expected = Mat::from_values({
      {3, 6},
      {9, 12},
  });

  assert(result == expected);
}

void test_scalar_left_subtraction() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = sub(scalar(10.0f), a);

  Mat expected = Mat::from_values({
      {9, 8},
      {7, 6},
  });

  assert(result == expected);
}

void test_scalar_left_division() {
  Mat a = Mat::from_values({
      {1, 2},
      {4, 5},
  });

  Mat result = divide(scalar(20.0f), a);

  Mat expected = Mat::from_values({
      {20, 10},
      {5, 4},
  });

  assert(result == expected);
}

void test_sqrt() {
  Mat a = Mat::from_values({
      {1, 4},
      {9, 16},
  });

  Mat result = sqrt(a);

  Mat expected = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  assert_matrix_approx(result, expected);

  // lvalue copied
  assert(a == Mat::from_values({{1, 4}, {9, 16}}));
}

void test_sqrt_rvalue() {
  Mat result = sqrt(Mat::from_values({
      {25, 36},
      {49, 64},
  }));

  Mat expected = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  assert_matrix_approx(result, expected);
}

void test_sigmoid_zero() {
  Mat a = Mat::from_values({
      {0, 0},
      {0, 0},
  });

  Mat result = sigmoid(a);

  Mat expected = Mat::from_values({
      {0.5f, 0.5f},
      {0.5f, 0.5f},
  });

  assert_matrix_approx(result, expected);
}

void test_sigmoid_values() {
  Mat a = Mat::from_values({
      {-1, 0, 1},
  });

  Mat result = sigmoid(a);

  assert(approx(result.get(0, 0), 1.0f / (1.0f + std::exp(1.0f))));
  assert(approx(result.get(0, 1), 0.5f));
  assert(approx(result.get(0, 2), 1.0f / (1.0f + std::exp(-1.0f))));
}

void test_softmax_sums_to_one() {
  Mat a = Mat::from_values({
      {1, 2, 3, 4},
  });

  Mat result = softmax(a);

  kernel_type total = 0.0f;

  for (size_t j = 0; j < result.shape()[1]; ++j) {
    total += result.get(0, j);
  }

  assert(approx(total, 1.0f));
}

void test_softmax_equal_values() {
  Mat a = Mat::from_values({
      {5, 5, 5, 5},
  });

  Mat result = softmax(a);

  Mat expected = Mat::from_values({
      {0.25f, 0.25f, 0.25f, 0.25f},
  });

  assert_matrix_approx(result, expected);
}

void test_softmax_large_values_stable() {
  Mat a = Mat::from_values({
      {1000, 1000},
  });

  Mat result = softmax(a);

  assert(approx(result.get(0, 0), 0.5f));
  assert(approx(result.get(0, 1), 0.5f));
}

void test_argmax() {
  Mat a = Mat::from_values({
      {1, 9, 3},
      {4, 5, 6},
  });

  size_t index = argmax(a).get(0, 0);

  // Row-major flattened index:
  // [1, 9, 3, 4, 5, 6]
  assert(index == 1);
}

void test_argmax_last_element() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 100},
  });

  assert(argmax(a) == 3);
}

void test_row_sum() {
  Mat a = Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  });

  Mat result = sum(a, 1);

  Mat expected = Mat::from_values({
      {6},
      {15},
  });

  assert(result == expected);
}

void test_row_sum_single_column() {
  Mat a = Mat::from_values({
      {1},
      {2},
      {3},
  });

  Mat result = sum(a, 1);

  assert(result == a);
}

void test_hadamard_same_shape() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  Mat result = hadamard(a, b);

  Mat expected = Mat::from_values({
      {5, 12},
      {21, 32},
  });

  assert(result == expected);
}

void test_hadamard_rvalue_left() {
  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  Mat result = hadamard(Mat::from_values({
                            {1, 2},
                            {3, 4},
                        }),
                        b);

  Mat expected = Mat::from_values({
      {5, 12},
      {21, 32},
  });

  assert(result == expected);
}

void test_hadamard_rvalue_right() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = hadamard(a, Mat::from_values({
                               {5, 6},
                               {7, 8},
                           }));

  Mat expected = Mat::from_values({
      {5, 12},
      {21, 32},
  });

  assert(result == expected);
}

void test_divide_same_shape() {
  Mat a = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  Mat b = Mat::from_values({
      {2, 4},
      {5, 8},
  });

  Mat result = divide(a, b);

  Mat expected = Mat::from_values({
      {5, 5},
      {6, 5},
  });

  assert_matrix_approx(result, expected);
}

void test_divide_rvalue_right_preserves_operand_order() {
  Mat a = Mat::from_values({
      {12, 18},
      {20, 30},
  });

  Mat result = divide(a, Mat::from_values({
                             {3, 6},
                             {4, 5},
                         }));

  Mat expected = Mat::from_values({
      {4, 3},
      {5, 6},
  });

  assert_matrix_approx(result, expected);
}

void test_broadcast_column_vector_hadamard() {
  Mat a = Mat::from_values({
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16},
  });

  Mat b = Mat::from_values({
      {1},
      {2},
      {3},
      {4},
  });

  Mat result = hadamard(a, b);

  Mat expected = Mat::from_values({
      {1, 2, 3, 4},
      {10, 12, 14, 16},
      {27, 30, 33, 36},
      {52, 56, 60, 64},
  });

  assert(result == expected);
}

void test_broadcast_row_vector_hadamard() {
  Mat a = Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  });

  Mat b = Mat::from_values({
      {10, 20, 30},
  });

  Mat result = hadamard(a, b);

  Mat expected = Mat::from_values({
      {10, 40, 90},
      {40, 100, 180},
  });

  assert(result == expected);
}

void test_broadcast_1x1_hadamard() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {10},
  });

  Mat result = hadamard(a, b);

  Mat expected = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  assert(result == expected);
}

void test_broadcast_column_vector_divide() {
  Mat a = Mat::from_values({
      {10, 20, 30},
      {40, 50, 60},
  });

  Mat b = Mat::from_values({
      {10},
      {20},
  });

  Mat result = divide(a, b);

  Mat expected = Mat::from_values({
      {1, 2, 3},
      {2, 2.5f, 3},
  });

  assert_matrix_approx(result, expected);
}

void test_broadcast_row_vector_divide() {
  Mat a = Mat::from_values({
      {10, 20, 30},
      {40, 50, 60},
  });

  Mat b = Mat::from_values({
      {10, 5, 2},
  });

  Mat result = divide(a, b);

  Mat expected = Mat::from_values({
      {1, 4, 15},
      {4, 10, 30},
  });

  assert_matrix_approx(result, expected);
}

void test_broadcast_reverse_column_vector_rvalue_rhs() {
  Mat a = Mat::from_values({
      {10},
      {20},
  });

  Mat result = hadamard(a, Mat::from_values({
                               {1, 2, 3},
                               {4, 5, 6},
                           }));

  Mat expected = Mat::from_values({
      {10, 20, 30},
      {80, 100, 120},
  });

  assert(result == expected);
}

void test_broadcast_reverse_row_vector_rvalue_rhs() {
  Mat a = Mat::from_values({
      {10, 20, 30},
  });

  Mat result = hadamard(a, Mat::from_values({
                               {1, 2, 3},
                               {4, 5, 6},
                           }));

  Mat expected = Mat::from_values({
      {10, 40, 90},
      {40, 100, 180},
  });

  assert(result == expected);
}

void test_broadcast_both_rvalues_reuse_left_shape() {
  Mat result = hadamard(Mat::from_values({
                            {1, 2, 3},
                            {4, 5, 6},
                        }),
                        Mat::from_values({
                            {10},
                            {20},
                        }));

  Mat expected = Mat::from_values({
      {10, 20, 30},
      {80, 100, 120},
  });

  assert(result == expected);
}

void test_broadcast_both_rvalues_reuse_right_shape() {
  Mat result = hadamard(Mat::from_values({
                            {10},
                            {20},
                        }),
                        Mat::from_values({
                            {1, 2, 3},
                            {4, 5, 6},
                        }));

  Mat expected = Mat::from_values({
      {10, 20, 30},
      {80, 100, 120},
  });

  assert(result == expected);
}

void test_broadcast_outer_shape() {
  Mat a = Mat::from_values({
      {1},
      {2},
      {3},
  });

  Mat b = Mat::from_values({
      {10, 20, 30, 40},
  });

  Mat result = hadamard(std::move(a), std::move(b));

  Mat expected = Mat::from_values({
      {10, 20, 30, 40},
      {20, 40, 60, 80},
      {30, 60, 90, 120},
  });

  assert(result == expected);
}

void test_broadcast_invalid_shape() {
  Mat a({2, 3});
  Mat b({4, 2});

  bool threw = false;

  try {
    auto result = hadamard(a, b);
    (void)result;
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  assert(threw);
}

void test_broadcast_divide_operand_order_reverse_shape() {
  Mat a = Mat::from_values({
      {10},
      {20},
  });

  Mat b = Mat::from_values({
      {2, 4, 5},
      {1, 2, 4},
  });

  Mat result = divide(a, std::move(b));

  Mat expected = Mat::from_values({
      {5, 2.5f, 2},
      {20, 10, 5},
  });

  assert_matrix_approx(result, expected);
}

void test_construction() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  assert(a.shape()[0] == 2);
  assert(a.shape()[1] == 2);

  assert(a.get(0, 0) == 1);
  assert(a.get(0, 1) == 2);
  assert(a.get(1, 0) == 3);
  assert(a.get(1, 1) == 4);
}

void test_rectangular_construction() {
  Mat a = Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  });

  assert(a.shape()[0] == 2);
  assert(a.shape()[1] == 3);

  assert(a.get(0, 0) == 1);
  assert(a.get(0, 2) == 3);
  assert(a.get(1, 0) == 4);
  assert(a.get(1, 2) == 6);
}

void test_invalid_construction() {
  bool threw = false;

  try {
    Mat a = Mat::from_values({
        {1, 2, 3},
        {4, 5},
    });
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  assert(threw);
}

void test_addition() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  Mat result = add(a, b);

  Mat expected = Mat::from_values({
      {6, 8},
      {10, 12},
  });

  assert(result == expected);

  // Make sure lvalue operands were not modified.
  assert(a == Mat::from_values({{1, 2}, {3, 4}}));
  assert(b == Mat::from_values({{5, 6}, {7, 8}}));
}

void test_addition_lvalue_plus_rvalue() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = add(a, Mat::from_values({
                          {5, 6},
                          {7, 8},
                      }));

  Mat expected = Mat::from_values({
      {6, 8},
      {10, 12},
  });

  assert(result == expected);

  // lhs must still be unchanged
  assert(a == Mat::from_values({{1, 2}, {3, 4}}));
}

void test_addition_rvalue_plus_lvalue() {
  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  Mat result = add(Mat::from_values({
                       {1, 2},
                       {3, 4},
                   }),
                   b);

  Mat expected = Mat::from_values({
      {6, 8},
      {10, 12},
  });

  assert(result == expected);

  assert(b == Mat::from_values({{5, 6}, {7, 8}}));
}

void test_addition_rvalue_plus_rvalue() {
  Mat result = add(Mat::from_values({
                       {1, 2},
                       {3, 4},
                   }),
                   Mat::from_values({
                       {5, 6},
                       {7, 8},
                   }));

  Mat expected = Mat::from_values({
      {6, 8},
      {10, 12},
  });

  assert(result == expected);
}

void test_addition_invalid_shape() {
  Mat a({2, 3});
  Mat b({3, 2});

  bool threw = false;

  try {
    auto result = add(a, b);
    (void)result;
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  assert(threw);
}

void test_addition_new_output_value() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  auto output = add(a, b);

  Mat expected = Mat::from_values({
      {6, 8},
      {10, 12},
  });

  assert(output == expected);
  assert(a == Mat::from_values({{1, 2}, {3, 4}}));
  assert(b == Mat::from_values({{5, 6}, {7, 8}}));
}

void test_addition_new_output_invalid_shape() {
  Mat a({2, 3});
  Mat b({3, 2});

  bool threw = false;

  try {
    auto output = add(a, b);
    (void)output;
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  assert(threw);
}

void test_subtraction() {
  Mat a = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  Mat b = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = sub(a, b);

  Mat expected = Mat::from_values({
      {9, 18},
      {27, 36},
  });

  assert(result == expected);
}

void test_subtraction_lvalue_minus_rvalue() {
  Mat a = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  Mat result = sub(a, Mat::from_values({
                          {1, 2},
                          {3, 4},
                      }));

  Mat expected = Mat::from_values({
      {9, 18},
      {27, 36},
  });

  assert(result == expected);

  // Important because the RHS-reuse implementation has reversed arithmetic
  // internally.
  assert(a == Mat::from_values({{10, 20}, {30, 40}}));
}

void test_subtraction_rvalue_minus_lvalue() {
  Mat b = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = sub(Mat::from_values({
                       {10, 20},
                       {30, 40},
                   }),
                   b);

  Mat expected = Mat::from_values({
      {9, 18},
      {27, 36},
  });

  assert(result == expected);
}

void test_subtraction_rvalue_minus_rvalue() {
  Mat result = sub(Mat::from_values({
                       {10, 20},
                       {30, 40},
                   }),
                   Mat::from_values({
                       {1, 2},
                       {3, 4},
                   }));

  Mat expected = Mat::from_values({
      {9, 18},
      {27, 36},
  });

  assert(result == expected);
}

void test_subtraction_invalid_shape() {
  Mat a({2, 3});
  Mat b({3, 2});

  bool threw = false;

  try {
    auto result = sub(a, b);
    (void)result;
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  assert(threw);
}

void test_scalar_addition() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = add(a, scalar(10.0f));

  Mat expected = Mat::from_values({
      {11, 12},
      {13, 14},
  });

  assert(result == expected);

  assert(a == Mat::from_values({{1, 2}, {3, 4}}));
}

void test_scalar_addition_rvalue() {
  Mat result = add(Mat::from_values({
                       {1, 2},
                       {3, 4},
                   }),
                   scalar(10.0f));

  Mat expected = Mat::from_values({
      {11, 12},
      {13, 14},
  });

  assert(result == expected);
}

void test_scalar_subtraction() {
  Mat a = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  Mat result = sub(a, scalar(5.0f));

  Mat expected = Mat::from_values({
      {5, 15},
      {25, 35},
  });

  assert(result == expected);
}

void test_scalar_subtraction_rvalue() {
  Mat result = sub(Mat::from_values({
                       {10, 20},
                       {30, 40},
                   }),
                   scalar(5.0f));

  Mat expected = Mat::from_values({
      {5, 15},
      {25, 35},
  });

  assert(result == expected);
}

void test_scalar_multiplication() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = hadamard(a, scalar(2.0f));

  Mat expected = Mat::from_values({
      {2, 4},
      {6, 8},
  });

  assert(result == expected);
}

void test_scalar_multiplication_rvalue() {
  Mat result = hadamard(Mat::from_values({
                            {1, 2},
                            {3, 4},
                        }),
                        scalar(2.0f));

  Mat expected = Mat::from_values({
      {2, 4},
      {6, 8},
  });

  assert(result == expected);
}

void test_scalar_division() {
  Mat a = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  Mat result = divide(a, scalar(10.0f));

  Mat expected = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  assert(result == expected);
}

void test_scalar_division_rvalue() {
  Mat result = divide(Mat::from_values({
                          {10, 20},
                          {30, 40},
                      }),
                      scalar(10.0f));

  Mat expected = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  assert(result == expected);
}

void test_matrix_multiplication() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  Mat result = mult(a, b);

  Mat expected = Mat::from_values({
      {19, 22},
      {43, 50},
  });

  assert(result == expected);
}

void test_rectangular_matrix_multiplication() {
  // 2x3
  Mat a = Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  });

  // 3x2
  Mat b = Mat::from_values({
      {7, 8},
      {9, 10},
      {11, 12},
  });

  // 2x2
  Mat result = mult(a, b);

  Mat expected = Mat::from_values({
      {58, 64},
      {139, 154},
  });

  assert(result == expected);

  assert(result.shape()[0] == 2);
  assert(result.shape()[1] == 2);
}

void test_matrix_multiplication_different_result_shape() {
  // 2x2
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  // 2x3
  Mat b = Mat::from_values({
      {5, 6, 7},
      {8, 9, 10},
  });

  // result = 2x3
  Mat result = mult(a, b);

  Mat expected = Mat::from_values({
      {21, 24, 27},
      {47, 54, 61},
  });

  assert(result == expected);

  assert(result.shape()[0] == 2);
  assert(result.shape()[1] == 3);
}

void test_matrix_multiplication_invalid_shape() {
  Mat a({2, 3});
  Mat b({2, 3});

  bool threw = false;

  try {
    auto result = mult(a, b);
    (void)result;
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  assert(threw);
}

void test_square_transpose_lvalue() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat result = transpose(a);

  Mat expected = Mat::from_values({
      {1, 3},
      {2, 4},
  });

  assert(result == expected);

  // lvalue transpose must leave original untouched
  assert(a == Mat::from_values({{1, 2}, {3, 4}}));
}

void test_square_transpose_rvalue() {
  Mat result = transpose(Mat::from_values({
      {1, 2},
      {3, 4},
  }));

  Mat expected = Mat::from_values({
      {1, 3},
      {2, 4},
  });

  assert(result == expected);
}

void test_rectangular_transpose_lvalue() {
  Mat a = Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  });

  Mat result = transpose(a);

  Mat expected = Mat::from_values({
      {1, 4},
      {2, 5},
      {3, 6},
  });

  assert(result == expected);

  assert(result.shape()[0] == 3);
  assert(result.shape()[1] == 2);

  assert(a.shape()[0] == 2);
  assert(a.shape()[1] == 3);

  assert(a == Mat::from_values({{1, 2, 3}, {4, 5, 6}}));
}

void test_rectangular_transpose_rvalue() {
  Mat result = transpose(Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  }));

  Mat expected = Mat::from_values({
      {1, 4},
      {2, 5},
      {3, 6},
  });

  assert(result == expected);

  assert(result.shape()[0] == 3);
  assert(result.shape()[1] == 2);
}

void test_double_transpose() {
  Mat a = Mat::from_values({
      {1, 2, 3},
      {4, 5, 6},
  });

  Mat result = transpose(transpose(a));

  assert(result == a);
}

void test_single_element() {
  Mat a = Mat::from_values({
      {42},
  });

  assert(a.shape()[0] == 1);
  assert(a.shape()[1] == 1);
  assert(a.get(0, 0) == 42);

  assert(transpose(a) == a);
  assert(add(a, scalar(1.0f)) == Mat::from_values({{43}}));
  assert(sub(a, scalar(2.0f)) == Mat::from_values({{40}}));
  assert(hadamard(a, scalar(2.0f)) == Mat::from_values({{84}}));
  assert(divide(a, scalar(2.0f)) == Mat::from_values({{21}}));
}

void test_zero_initialised_matrix() {
  Mat a({2, 3});

  assert(a.shape()[0] == 2);
  assert(a.shape()[1] == 3);

  for (size_t i = 0; i < a.shape()[0]; ++i) {
    for (size_t j = 0; j < a.shape()[1]; ++j) {
      assert(a.get(i, j) == 0.0f);
    }
  }
}

void test_expression_with_multiple_temporaries() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  Mat c = Mat::from_values({
      {10, 20},
      {30, 40},
  });

  // Exercises temporary reuse:
  //
  // a + b -> temporary
  // temporary * 2 -> reuse temporary
  // temporary - c -> reuse temporary
  Mat result = sub(hadamard(add(a, b), scalar(2.0f)), c);

  Mat expected = Mat::from_values({
      {2, -4},
      {-10, -16},
  });

  assert(result == expected);
}

void test_move_construction() {
  Mat original = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat moved = std::move(original);

  Mat expected = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  assert(moved == expected);

  // Do not make assumptions about the contents of original after move.
  // It is valid but its exact state is unspecified.
}

void test_copy_construction() {
  Mat original = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat copy = original;

  assert(copy == original);

  copy.set(999, 0, 0);

  // Copy construction is a shared storage view.
  assert(original.get(0, 0) == 999);
  assert(copy.get(0, 0) == 999);
}

void test_clone_construction() {
  Mat original = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat copy = original.clone();

  assert(copy == original);

  copy.set(999, 0, 0);

  assert(original.get(0, 0) == 1);
  assert(copy.get(0, 0) == 999);
}

void test_copy_assignment() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  b = a;

  assert(b == a);

  b.set(999, 0, 0);

  // Copy assignment is a shared storage view.
  assert(a.get(0, 0) == 999);
  assert(b.get(0, 0) == 999);
}

void test_clone_assignment() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  b = a.clone();

  assert(b == a);

  b.set(999, 0, 0);

  assert(a.get(0, 0) == 1);
  assert(b.get(0, 0) == 999);
}

void test_move_assignment() {
  Mat a = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  Mat b = Mat::from_values({
      {5, 6},
      {7, 8},
  });

  b = std::move(a);

  Mat expected = Mat::from_values({
      {1, 2},
      {3, 4},
  });

  assert(b == expected);
}

#define RUN_TEST(test)                                                         \
  do {                                                                         \
    std::cout << #test << "...\n";                                             \
    test();                                                                    \
  } while (false)

int main() {
  test_backend::init();
  RUN_TEST(test_construction);
  RUN_TEST(test_rectangular_construction);
  RUN_TEST(test_invalid_construction);

  RUN_TEST(test_addition);
  RUN_TEST(test_addition_lvalue_plus_rvalue);
  RUN_TEST(test_addition_rvalue_plus_lvalue);
  RUN_TEST(test_addition_rvalue_plus_rvalue);
  RUN_TEST(test_addition_invalid_shape);
  RUN_TEST(test_addition_new_output_value);
  RUN_TEST(test_addition_new_output_invalid_shape);

  RUN_TEST(test_subtraction);
  RUN_TEST(test_subtraction_lvalue_minus_rvalue);
  RUN_TEST(test_subtraction_rvalue_minus_lvalue);
  RUN_TEST(test_subtraction_rvalue_minus_rvalue);
  RUN_TEST(test_subtraction_invalid_shape);

  RUN_TEST(test_scalar_addition);
  RUN_TEST(test_scalar_addition_rvalue);
  RUN_TEST(test_scalar_subtraction);
  RUN_TEST(test_scalar_subtraction_rvalue);
  RUN_TEST(test_scalar_multiplication);
  RUN_TEST(test_scalar_multiplication_rvalue);
  RUN_TEST(test_scalar_division);
  RUN_TEST(test_scalar_division_rvalue);

  RUN_TEST(test_matrix_multiplication);
  RUN_TEST(test_rectangular_matrix_multiplication);
  RUN_TEST(test_matrix_multiplication_different_result_shape);
  RUN_TEST(test_matrix_multiplication_invalid_shape);

  RUN_TEST(test_square_transpose_lvalue);
  RUN_TEST(test_square_transpose_rvalue);
  RUN_TEST(test_rectangular_transpose_lvalue);
  RUN_TEST(test_rectangular_transpose_rvalue);
  RUN_TEST(test_double_transpose);

  RUN_TEST(test_single_element);
  RUN_TEST(test_zero_initialised_matrix);

  RUN_TEST(test_expression_with_multiple_temporaries);

  RUN_TEST(test_copy_construction);
  RUN_TEST(test_clone_construction);
  RUN_TEST(test_move_construction);
  RUN_TEST(test_copy_assignment);
  RUN_TEST(test_clone_assignment);
  RUN_TEST(test_move_assignment);

  RUN_TEST(test_scalar_left_addition);
  RUN_TEST(test_scalar_left_multiplication);
  RUN_TEST(test_scalar_left_subtraction);
  RUN_TEST(test_scalar_left_division);

  RUN_TEST(test_sqrt);
  RUN_TEST(test_sqrt_rvalue);

  RUN_TEST(test_sigmoid_zero);
  RUN_TEST(test_sigmoid_values);

  RUN_TEST(test_softmax_sums_to_one);
  RUN_TEST(test_softmax_equal_values);
  RUN_TEST(test_softmax_large_values_stable);

  RUN_TEST(test_argmax);
  RUN_TEST(test_argmax_last_element);

  RUN_TEST(test_row_sum);
  RUN_TEST(test_row_sum_single_column);

  RUN_TEST(test_hadamard_same_shape);
  RUN_TEST(test_hadamard_rvalue_left);
  RUN_TEST(test_hadamard_rvalue_right);

  RUN_TEST(test_divide_same_shape);
  RUN_TEST(test_divide_rvalue_right_preserves_operand_order);

  RUN_TEST(test_broadcast_column_vector_hadamard);
  RUN_TEST(test_broadcast_row_vector_hadamard);
  RUN_TEST(test_broadcast_1x1_hadamard);

  RUN_TEST(test_broadcast_column_vector_divide);
  RUN_TEST(test_broadcast_row_vector_divide);

  RUN_TEST(test_broadcast_reverse_column_vector_rvalue_rhs);
  RUN_TEST(test_broadcast_reverse_row_vector_rvalue_rhs);

  RUN_TEST(test_broadcast_both_rvalues_reuse_left_shape);
  RUN_TEST(test_broadcast_both_rvalues_reuse_right_shape);

  RUN_TEST(test_broadcast_outer_shape);
  RUN_TEST(test_broadcast_invalid_shape);
  RUN_TEST(test_broadcast_divide_operand_order_reverse_shape);

  std::cout << "All Matrix tests passed.\n";
  test_backend::shutdown();
  return 0;
}
