#include <CPU/cpu.h>
#include <CPU/storage.h>
#include <RTensor/dimension.h>
#include <RTensor/infer_shape.h>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cpu {

template <typename F>
void transform(const RTensor<CPU, kernel_type> &x,
               RTensor<CPU, kernel_type> &result, F fn) {
  const auto &input = x.get_storage()->data();
  auto &output = result.get_storage()->data();

  const size_t size = std::accumulate(x.shape().begin(), x.shape().end(),
                                      size_t{1}, std::multiplies<size_t>());

  for (size_t i = 0; i < size; ++i) {
    output[i] = fn(input[i]);
  }
}

template <typename F>
RTensor<CPU, kernel_type> transform(const RTensor<CPU, kernel_type> &x, F fn) {
  RTensor<CPU, kernel_type> result(x.shape());
  transform(x, result, fn);
  return result;
}

template <typename F>
void binary_transform(const RTensor<CPU, kernel_type> &a,
                      const RTensor<CPU, kernel_type> &b,
                      RTensor<CPU, kernel_type> &result, F fn) {
  const uint32_t a_rows = a.shape()[0];
  const uint32_t a_cols = a.shape()[1];

  const uint32_t b_rows = b.shape()[0];
  const uint32_t b_cols = b.shape()[1];

  const uint32_t out_rows = std::max(a_rows, b_rows);
  const uint32_t out_cols = std::max(a_cols, b_cols);

  const auto &a_data = a.get_storage()->data();
  const auto &b_data = b.get_storage()->data();
  auto &result_data = result.get_storage()->data();

  if (a_rows == b_rows && a_cols == b_cols) {
    for (size_t i = 0; i < a_rows * a_cols; ++i) {
      result_data[i] = fn(a_data[i], b_data[i]);
    }
  }

  else if (b_rows == a_rows && b_cols == 1) {
    for (size_t i = 0; i < a_rows; ++i) {
      const kernel_type bv = b_data[i];
      const size_t base = i * a_cols;

      for (size_t j = 0; j < a_cols; ++j) {
        result_data[base + j] = fn(a_data[base + j], bv);
      }
    }
  }

  else if (b_rows == 1 && b_cols == a_cols) {
    for (size_t i = 0; i < a_rows; ++i) {
      const size_t base = i * a_cols;

      for (size_t j = 0; j < a_cols; ++j) {
        result_data[base + j] = fn(a_data[base + j], b_data[j]);
      }
    }
  }

  else if (b_rows == 1 && b_cols == 1) {
    const kernel_type bv = b_data[0];

    for (size_t i = 0; i < a_rows * a_cols; ++i) {
      result_data[i] = fn(a_data[i], bv);
    }
  }

  else if (a_cols == 1 && b_rows == 1) {
    for (size_t i = 0; i < out_rows; ++i) {
      const kernel_type av = a_data[i];
      const size_t base = i * out_cols;

      for (size_t j = 0; j < out_cols; ++j) {
        result_data[base + j] = fn(av, b_data[j]);
      }
    }
  }

  else if (a_rows == 1 && b_cols == 1) {
    for (size_t i = 0; i < out_rows; ++i) {
      const kernel_type bv = b_data[i];
      const size_t base = i * out_cols;

      for (size_t j = 0; j < out_cols; ++j) {
        result_data[base + j] = fn(a_data[j], bv);
      }
    }
  }

  else if (a_rows == b_rows && a_cols == 1) {
    for (size_t i = 0; i < b_rows; ++i) {
      const kernel_type av = a_data[i];
      const size_t base = i * b_cols;

      for (size_t j = 0; j < b_cols; ++j) {
        result_data[base + j] = fn(av, b_data[base + j]);
      }
    }
  }

  else if (a_rows == 1 && a_cols == b_cols) {
    for (size_t i = 0; i < b_rows; ++i) {
      const size_t base = i * b_cols;

      for (size_t j = 0; j < b_cols; ++j) {
        result_data[base + j] = fn(a_data[j], b_data[base + j]);
      }
    }
  }

  else if (a_rows == 1 && a_cols == 1) {
    const kernel_type av = a_data[0];

    for (size_t i = 0; i < b_rows * b_cols; ++i) {
      result_data[i] = fn(av, b_data[i]);
    }
  }

  else {
    throw std::invalid_argument("at least 1 tensor dimension must match");
  }
}

template <typename F>
RTensor<CPU, kernel_type> binary_transform(const RTensor<CPU, kernel_type> &a,
                                           const RTensor<CPU, kernel_type> &b,
                                           F fn) {
  const uint32_t out_rows = std::max(a.shape()[0], b.shape()[0]);
  const uint32_t out_cols = std::max(a.shape()[1], b.shape()[1]);
  RTensor<CPU, kernel_type> result({out_rows, out_cols});
  binary_transform(a, b, result, fn);
  return result;
}
template <typename F>
kernel_type reduce(const RTensor<CPU, kernel_type> &x, kernel_type init, F fn) {
  const auto &data = x.get_storage()->data();

  for (const kernel_type value : data) {
    init = fn(std::move(init), value);
  }

  return init;
}

template <typename F>
void reduce(const RTensor<CPU, kernel_type> &x, kernel_type init, F fn,
            size_t axis, RTensor<CPU, kernel_type> &result) {

  const uint32_t rows = x.shape()[0];
  const uint32_t cols = x.shape()[1];

  const auto &data = x.get_storage()->data();

  if (axis == 1) {
    auto &result_data = result.get_storage()->data();

    for (size_t i = 0; i < rows; ++i) {
      result_data[i] = init;

      const kernel_type *row = data.data() + i * cols;

      for (size_t j = 0; j < cols; ++j) {
        result_data[i] = fn(std::move(result_data[i]), row[j]);
      }
    }

    return;
  }

  auto &result_data = result.get_storage()->data();

  for (size_t j = 0; j < cols; ++j) {
    result_data[j] = init;
  }

  for (size_t i = 0; i < rows; ++i) {
    const kernel_type *row = data.data() + i * cols;

    for (size_t j = 0; j < cols; ++j) {
      result_data[j] = fn(std::move(result_data[j]), row[j]);
    }
  }
}

template <typename F>
RTensor<CPU, kernel_type> reduce(const RTensor<CPU, kernel_type> &x,
                                 kernel_type init, F fn, size_t axis) {
  RTensor<CPU, kernel_type> result(axis == 1
                                       ? rtensor::shape_type{x.shape()[0], 1}
                                       : rtensor::shape_type{1, x.shape()[1]});
  reduce(x, init, fn, axis, result);
  return result;
}
RTensor<CPU, kernel_type> add(const RTensor<CPU, kernel_type> &a,
                              const RTensor<CPU, kernel_type> &b) {
  return binary_transform(a, b,
                          [](kernel_type x, kernel_type y) { return x + y; });
}

void add(const RTensor<CPU, kernel_type> &a, const RTensor<CPU, kernel_type> &b,
         RTensor<CPU, kernel_type> &result) {
  binary_transform(a, b, result,
                   [](kernel_type x, kernel_type y) { return x + y; });
}

RTensor<CPU, kernel_type> sub(const RTensor<CPU, kernel_type> &a,
                              const RTensor<CPU, kernel_type> &b) {
  return binary_transform(a, b,
                          [](kernel_type x, kernel_type y) { return x - y; });
}

void sub(const RTensor<CPU, kernel_type> &a, const RTensor<CPU, kernel_type> &b,
         RTensor<CPU, kernel_type> &result) {
  binary_transform(a, b, result,
                   [](kernel_type x, kernel_type y) { return x - y; });
}

RTensor<CPU, kernel_type> mult(const RTensor<CPU, kernel_type> &a,
                               const RTensor<CPU, kernel_type> &b) {

  RTensor<CPU, kernel_type> result({a.shape()[0], b.shape()[1]});
  mult(a, b, result);
  return result;
}

void mult(const RTensor<CPU, kernel_type> &a,
          const RTensor<CPU, kernel_type> &b,
          RTensor<CPU, kernel_type> &result) {

  const uint32_t a_rows = a.shape()[0];
  const uint32_t a_cols = a.shape()[1];

  const uint32_t b_rows = b.shape()[0];
  const uint32_t b_cols = b.shape()[1];

  if (a_cols != b_rows) {
    throw std::invalid_argument(
        "Matrices must have compatible dimensions for multiplication");
  }

  const auto &a_data = a.get_storage()->data();
  const auto &b_data = b.get_storage()->data();
  auto &result_data = result.get_storage()->data();
  std::fill(result_data.begin(), result_data.end(), kernel_type{});

  for (size_t i = 0; i < a_rows; ++i) {
    for (size_t k = 0; k < a_cols; ++k) {
      const kernel_type value = a_data[i * a_cols + k];

      for (size_t j = 0; j < b_cols; ++j) {
        result_data[i * b_cols + j] += value * b_data[k * b_cols + j];
      }
    }
  }
}

RTensor<CPU, kernel_type> hadamard(const RTensor<CPU, kernel_type> &a,
                                   const RTensor<CPU, kernel_type> &b) {
  return binary_transform(a, b,
                          [](kernel_type x, kernel_type y) { return x * y; });
}

void hadamard(const RTensor<CPU, kernel_type> &a,
              const RTensor<CPU, kernel_type> &b,
              RTensor<CPU, kernel_type> &result) {
  binary_transform(a, b, result,
                   [](kernel_type x, kernel_type y) { return x * y; });
}

RTensor<CPU, kernel_type> divide(const RTensor<CPU, kernel_type> &a,
                                 const RTensor<CPU, kernel_type> &b) {
  return binary_transform(a, b,
                          [](kernel_type x, kernel_type y) { return x / y; });
}

void divide(const RTensor<CPU, kernel_type> &a,
            const RTensor<CPU, kernel_type> &b,
            RTensor<CPU, kernel_type> &result) {
  binary_transform(a, b, result,
                   [](kernel_type x, kernel_type y) { return x / y; });
}

RTensor<CPU, kernel_type> sigmoid(const RTensor<CPU, kernel_type> &a) {
  return transform(a, [](kernel_type x) {
    return kernel_type{1} / (kernel_type{1} + std::exp(-x));
  });
}

void sigmoid(const RTensor<CPU, kernel_type> &a,
             RTensor<CPU, kernel_type> &result) {
  transform(a, result, [](kernel_type x) {
    return kernel_type{1} / (kernel_type{1} + std::exp(-x));
  });
}

RTensor<CPU, kernel_type> sqrt(const RTensor<CPU, kernel_type> &a) {
  return transform(a, [](kernel_type x) { return std::sqrt(x); });
}

void sqrt(const RTensor<CPU, kernel_type> &a,
          RTensor<CPU, kernel_type> &result) {
  transform(a, result, [](kernel_type x) { return std::sqrt(x); });
}

RTensor<CPU, kernel_type> sum(const RTensor<CPU, kernel_type> &a) {
  RTensor<CPU, kernel_type> result({1});
  result.set(reduce(a, kernel_type{},
                    [](kernel_type current, kernel_type value) {
                      return current + value;
                    }),
             0);

  return result;
}

void sum(const RTensor<CPU, kernel_type> &a,
         RTensor<CPU, kernel_type> &result) {
  result.set(reduce(a, kernel_type{},
                    [](kernel_type current, kernel_type value) {
                      return current + value;
                    }),
             0);
}

RTensor<CPU, kernel_type> sum(const RTensor<CPU, kernel_type> &a, size_t axis) {
  if (axis >= a.shape().size()) {
    throw std::out_of_range("sum axis out of range");
  }

  std::vector<uint32_t> output_shape = a.shape();
  output_shape[axis] = 1;

  return sum_to_shape(a, output_shape);
}

void sum(const RTensor<CPU, kernel_type> &a, size_t axis,
         RTensor<CPU, kernel_type> &result) {
  if (axis >= a.shape().size()) {
    throw std::out_of_range("sum axis out of range");
  }

  sum_to_shape(a, result.shape(), result);
}

RTensor<CPU, kernel_type> transpose(const RTensor<CPU, kernel_type> &data,
                                    int32_t dim_a, int32_t dim_b) {
  RTensor<CPU, kernel_type> result(
      rtensor::infer_transpose(data.shape(), dim_a, dim_b));
  transpose(data, result, dim_a, dim_b);
  return result;
}

void transpose(const RTensor<CPU, kernel_type> &data,
               RTensor<CPU, kernel_type> &result, int32_t dim_a,
               int32_t dim_b) {
  const size_t axis_a =
      rtensor::normalise_dimension(data.shape().size(), dim_a);
  const size_t axis_b =
      rtensor::normalise_dimension(data.shape().size(), dim_b);
  if (axis_a == axis_b) {
    throw std::invalid_argument("transpose dimensions must be different");
  }

  const auto expected_shape =
      rtensor::infer_transpose(data.shape(), dim_a, dim_b);
  if (result.shape() != expected_shape) {
    throw std::invalid_argument("transpose destination has incorrect shape");
  }

  const auto &input = data.get_storage()->data();
  auto &output = result.get_storage()->data();
  auto input_strides = data.strides();
  std::swap(input_strides[axis_a], input_strides[axis_b]);

  const size_t output_size =
      std::accumulate(result.shape().begin(), result.shape().end(), size_t{1},
                      std::multiplies<>{});
  for (size_t index = 0; index < output_size; ++index) {
    size_t remainder = index;
    size_t input_offset = 0;

    for (size_t dim = 0; dim < result.shape().size(); ++dim) {
      const size_t coordinate = remainder / result.strides()[dim];
      remainder %= result.strides()[dim];
      input_offset += coordinate * input_strides[dim];
    }

    output[index] = input[input_offset];
  }
}
RTensor<CPU, kernel_type> sum_to_shape(const RTensor<CPU, kernel_type> &a,
                                       std::span<const uint32_t> shape) {

  const uint32_t rows = shape[0];
  const uint32_t cols = shape[1];

  const uint32_t a_rows = a.shape()[0];
  const uint32_t a_cols = a.shape()[1];

  if (a_rows == rows && a_cols == cols) {
    return a;
  }

  else if (a_rows == rows && cols == 1) {
    return reduce(
        a, kernel_type{},
        [](kernel_type current, kernel_type value) { return current + value; },
        1);
  }

  else if (a_cols == cols && rows == 1) {
    return reduce(
        a, kernel_type{},
        [](kernel_type current, kernel_type value) { return current + value; },
        0);
  }

  else if (rows == 1 && cols == 1) {
    return RTensor<CPU, kernel_type>(
        reduce(a, kernel_type{}, [](kernel_type current, kernel_type value) {
          return current + value;
        }));
  }

  else {
    throw std::invalid_argument("Cannot sum tensor to requested shape");
  }
}

void sum_to_shape(const RTensor<CPU, kernel_type> &a,
                  std::span<const uint32_t> shape,
                  RTensor<CPU, kernel_type> &result) {
  const uint32_t rows = shape[0];
  const uint32_t cols = shape[1];
  const uint32_t a_rows = a.shape()[0];
  const uint32_t a_cols = a.shape()[1];

  if (a_rows == rows && a_cols == cols) {
    result.get_storage()->data() = a.get_storage()->data();
  } else if (a_rows == rows && cols == 1) {
    reduce(
        a, kernel_type{},
        [](kernel_type current, kernel_type value) { return current + value; },
        1, result);
  } else if (a_cols == cols && rows == 1) {
    reduce(
        a, kernel_type{},
        [](kernel_type current, kernel_type value) { return current + value; },
        0, result);
  } else if (rows == 1 && cols == 1) {
    result.set(reduce(a, kernel_type{},
                      [](kernel_type current, kernel_type value) {
                        return current + value;
                      }),
               0, 0);
  } else {
    throw std::invalid_argument("Cannot sum tensor to requested shape");
  }
}

RTensor<CPU, kernel_type> softmax(const RTensor<CPU, kernel_type> &x) {
  auto max_vals = reduce(
      x, std::numeric_limits<kernel_type>::lowest(),
      [](kernel_type current, kernel_type value) {
        return std::max(current, value);
      },
      1);

  auto shifted = sub(x, std::move(max_vals));

  auto exps = transform(std::move(shifted),
                        [](kernel_type value) { return std::exp(value); });

  auto denominators = reduce(
      exps, kernel_type{},
      [](kernel_type current, kernel_type value) { return current + value; },
      1);

  return divide(std::move(exps), std::move(denominators));
}

void softmax(const RTensor<CPU, kernel_type> &x,
             RTensor<CPU, kernel_type> &result) {
  auto max_vals = reduce(
      x, std::numeric_limits<kernel_type>::lowest(),
      [](kernel_type current, kernel_type value) {
        return std::max(current, value);
      },
      1);

  auto shifted = sub(x, max_vals);
  auto exps =
      transform(shifted, [](kernel_type value) { return std::exp(value); });
  auto denominators = reduce(
      exps, kernel_type{},
      [](kernel_type current, kernel_type value) { return current + value; },
      1);

  divide(exps, denominators, result);
}

void sqe(const RTensor<CPU, kernel_type> &pred,
         const RTensor<CPU, kernel_type> &target,
         RTensor<CPU, kernel_type> &result) {
  auto diff = sub(pred, target);
  hadamard(diff, diff, result);
}

RTensor<CPU, kernel_type> argmax(const RTensor<CPU, kernel_type> &x) {
  const auto &data = x.get_storage()->data();
  auto it = std::ranges::max_element(data);
  return static_cast<size_t>(std::distance(data.begin(), it));
}

} // namespace cpu
