#pragma once
#include <cmath>
#include <cstddef>
#include <ios>
#include <numeric>
#include <ostream>
#include <ranges>
#include <span>
namespace math {

template <typename T> class Matrix {
private:
  size_t rows{}, cols{};
  std::vector<T> data;

  template <typename U, typename F>
  friend Matrix<U> transform(Matrix<U> x, F fn);

  template <typename U, typename F>
  friend Matrix<U> binary_transform(const Matrix<U> &a, const Matrix<U> &b,
                                    F fn);
  template <typename U, typename F>
  friend Matrix<U> binary_transform(Matrix<U> &&a, const Matrix<U> &b, F fn);
  template <typename U, typename F>
  friend Matrix<U> binary_transform(const Matrix<U> &a, Matrix<U> &&b, F fn);
  template <typename U, typename F>
  friend Matrix<U> binary_transform(Matrix<U> &&a, Matrix<U> &&b, F fn);

  template <typename U, typename F>
  friend U reduce(const Matrix<U> &x, std::type_identity_t<U> init, F fn);

  template <typename U, typename F>
  friend Matrix<U> reduce(const Matrix<U> &x, F fn, size_t axis, U init);

  template <typename U> friend size_t argmax(const Matrix<U> &matrix);

public:
  Matrix() = delete;
  Matrix(T value) : rows(1), cols(1), data({value}) {}
  template <typename S>
    requires std::convertible_to<S, T>
  Matrix(S value) : rows(1), cols(1), data({static_cast<T>(value)}) {}
  Matrix(size_t rows_t, size_t cols_t)
      : rows(rows_t), cols(cols_t), data(rows * cols) {}
  Matrix(size_t rows_t, size_t cols_t, const T &val)
      : rows(rows_t), cols(cols_t), data(rows * cols, val) {}
  Matrix(std::initializer_list<std::initializer_list<T>> values)
      : rows(values.size()),
        cols(values.size() == 0 ? 0 : values.begin()->size()) {

    data.reserve(rows * cols);

    for (const auto &row : values) {
      if (row.size() != cols) {
        throw std::invalid_argument("Matrix rows must have equal length");
      }

      data.insert(data.end(), row.begin(), row.end());
    }
  }
  template <typename R>
    requires std::ranges::forward_range<R> &&
             std::ranges::forward_range<std::ranges::range_reference_t<R>>
  explicit Matrix(const R &values) {
    rows = std::ranges::distance(values);

    if (rows == 0) {
      cols = 0;
      return;
    }

    cols = std::ranges::distance(*std::ranges::begin(values));

    data.reserve(rows * cols);

    for (const auto &row : values) {
      if (std::ranges::distance(row) != cols) {
        throw std::invalid_argument("Matrix rows must have equal length");
      }

      for (const auto &value : row) {
        data.push_back(value);
      }
    }
  }

  inline size_t get_rows() const noexcept { return rows; }
  inline size_t get_cols() const noexcept { return cols; }
  inline std::vector<size_t> shape() const { return {rows, cols}; }
  inline T &operator()(size_t i, size_t j) noexcept {
    return data[i * cols + j];
  }
  inline const T &operator()(size_t i, size_t j) const noexcept {
    return data[i * cols + j];
  }

  Matrix operator*(const Matrix &other) const {
    if (cols != other.rows) {
      throw std::invalid_argument(
          "Matrices must have compatible dimensions for multiplication");
    }
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = 0; k < cols; ++k) {
        const T &a = (*this)(i, k);

        for (size_t j = 0; j < other.cols; ++j) {
          result(i, j) += a * other(k, j);
        }
      }
    }
    return result;
  }
  Matrix &operator+=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument(
          "Matrices must have the same dimensions for addition");
    }
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] += other.data[i];
    }
    return *this;
  }
  Matrix &operator-=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument(
          "Matrices must have the same dimensions for subtraction");
    }
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] -= other.data[i];
    }
    return *this;
  }
  Matrix operator/(const T &scalar) const & {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
      result.data[i] = data[i] / scalar;
    }

    return result;
  }

  Matrix operator/(const T &scalar) && {
    for (auto &value : data) {
      value /= scalar;
    }

    return std::move(*this);
  }
  Matrix operator+(const T &scalar) const & {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
      result.data[i] = data[i] + scalar;
    }

    return result;
  }

  Matrix operator+(const T &scalar) && {
    for (auto &value : data) {
      value += scalar;
    }

    return std::move(*this);
  }
  Matrix operator-(const T &scalar) const & {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
      result.data[i] = data[i] - scalar;
    }

    return result;
  }

  Matrix operator-(const T &scalar) && {
    for (auto &value : data) {
      value -= scalar;
    }

    return std::move(*this);
  }
  Matrix operator*(const T &scalar) const & {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
      result.data[i] = data[i] * scalar;
    }

    return result;
  }

  Matrix operator*(const T &scalar) && {
    for (auto &value : data) {
      value *= scalar;
    }

    return std::move(*this);
  }

  friend std::ostream &operator<<(std::ostream &stream, const Matrix &matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
      for (size_t j = 0; j < matrix.cols; ++j) {
        stream << matrix(i, j) << " ";
      }
      stream << std::endl;
    }
    return stream;
  }
  bool operator==(const Matrix &other) const = default;
  bool operator!=(const Matrix &other) const = default;
};

template <typename U, typename F> Matrix<U> transform(Matrix<U> x, F fn) {
  for (auto &v : x.data) {
    v = fn(v);
  }
  return x;
}
template <typename T, typename F>
Matrix<T> binary_transform(const Matrix<T> &a, const Matrix<T> &b, F fn) {
  const size_t out_rows = std::max(a.rows, b.rows);
  const size_t out_cols = std::max(a.cols, b.cols);

  Matrix<T> result(out_rows, out_cols);

  if (a.rows == b.rows && a.cols == b.cols) {
    for (size_t i = 0; i < a.data.size(); ++i) {
      result.data[i] = fn(a.data[i], b.data[i]);
    }
  }

  else if (b.rows == a.rows && b.cols == 1) {
    for (size_t i = 0; i < a.rows; ++i) {
      const T bv = b.data[i];
      const size_t base = i * a.cols;

      for (size_t j = 0; j < a.cols; ++j) {
        result.data[base + j] = fn(a.data[base + j], bv);
      }
    }
  }

  else if (b.rows == 1 && b.cols == a.cols) {
    for (size_t i = 0; i < a.rows; ++i) {
      const size_t base = i * a.cols;

      for (size_t j = 0; j < a.cols; ++j) {
        result.data[base + j] = fn(a.data[base + j], b.data[j]);
      }
    }
  }

  else if (b.rows == 1 && b.cols == 1) {
    const T bv = b.data[0];
    for (size_t i = 0; i < a.data.size(); ++i) {
      result.data[i] = fn(a.data[i], bv);
    }
  }

  else if (a.cols == 1 && b.rows == 1) {
    for (size_t i = 0; i < out_rows; ++i) {
      const T av = a.data[i];
      const size_t base = i * out_cols;

      for (size_t j = 0; j < out_cols; ++j) {
        result.data[base + j] = fn(av, b.data[j]);
      }
    }
  }

  else if (a.rows == 1 && b.cols == 1) {
    for (size_t i = 0; i < out_rows; ++i) {
      const T bv = b.data[i];
      const size_t base = i * out_cols;

      for (size_t j = 0; j < out_cols; ++j) {
        result.data[base + j] = fn(a.data[j], bv);
      }
    }
  }

  else if (a.rows == b.rows && a.cols == 1) {
    for (size_t i = 0; i < b.rows; ++i) {
      const T av = a.data[i];
      const size_t base = i * b.cols;
      for (size_t j = 0; j < b.cols; ++j) {
        result.data[base + j] = fn(av, b.data[base + j]);
      }
    }
  }

  else if (a.rows == 1 && a.cols == b.cols) {
    for (size_t i = 0; i < b.rows; ++i) {
      const size_t base = i * b.cols;
      for (size_t j = 0; j < b.cols; ++j) {
        result.data[base + j] = fn(a.data[j], b.data[base + j]);
      }
    }
  }

  else if (a.rows == 1 && a.cols == 1) {
    const T av = a.data[0];
    for (size_t i = 0; i < b.data.size(); ++i) {
      result.data[i] = fn(av, b.data[i]);
    }
  }

  else {
    throw std::invalid_argument("atleast 1 Matrix dimensions must match");
  }

  return result;
}

template <typename T, typename F>
Matrix<T> binary_transform(Matrix<T> &&a, const Matrix<T> &b, F fn) {
  if (a.rows == b.rows && a.cols == b.cols) {
    for (size_t i = 0; i < a.data.size(); ++i) {
      a.data[i] = fn(a.data[i], b.data[i]);
    }

    return std::move(a);
  }

  if (b.rows == a.rows && b.cols == 1) {
    for (size_t i = 0; i < a.rows; ++i) {
      const T bv = b.data[i];
      const size_t base = i * a.cols;

      for (size_t j = 0; j < a.cols; ++j) {
        a.data[base + j] = fn(a.data[base + j], bv);
      }
    }

    return std::move(a);
  }

  if (b.rows == 1 && b.cols == a.cols) {
    for (size_t i = 0; i < a.rows; ++i) {
      const size_t base = i * a.cols;

      for (size_t j = 0; j < a.cols; ++j) {
        a.data[base + j] = fn(a.data[base + j], b.data[j]);
      }
    }

    return std::move(a);
  }

  if (b.rows == 1 && b.cols == 1) {
    const T bv = b.data[0];

    for (size_t i = 0; i < a.data.size(); ++i) {
      a.data[i] = fn(a.data[i], bv);
    }

    return std::move(a);
  }

  return binary_transform(static_cast<const Matrix<T> &>(a), b, fn);
}

template <typename T, typename F>
Matrix<T> binary_transform(const Matrix<T> &a, Matrix<T> &&b, F fn) {
  if (a.rows == b.rows && a.cols == b.cols) {
    for (size_t i = 0; i < b.data.size(); ++i) {
      b.data[i] = fn(a.data[i], b.data[i]);
    }

    return std::move(b);
  }

  if (a.rows == b.rows && a.cols == 1) {
    for (size_t i = 0; i < b.rows; ++i) {
      const T av = a.data[i];
      const size_t base = i * b.cols;

      for (size_t j = 0; j < b.cols; ++j) {
        b.data[base + j] = fn(av, b.data[base + j]);
      }
    }

    return std::move(b);
  }

  if (a.rows == 1 && a.cols == b.cols) {
    for (size_t i = 0; i < b.rows; ++i) {
      const size_t base = i * b.cols;

      for (size_t j = 0; j < b.cols; ++j) {
        b.data[base + j] = fn(a.data[j], b.data[base + j]);
      }
    }

    return std::move(b);
  }

  if (a.rows == 1 && a.cols == 1) {
    const T av = a.data[0];

    for (size_t i = 0; i < b.data.size(); ++i) {
      b.data[i] = fn(av, b.data[i]);
    }

    return std::move(b);
  }

  return binary_transform(a, static_cast<const Matrix<T> &>(b), fn);
}

template <typename T, typename F>
Matrix<T> binary_transform(Matrix<T> &&a, Matrix<T> &&b, F fn) {
  if ((b.rows == a.rows || b.rows == 1) && (b.cols == a.cols || b.cols == 1)) {
    return binary_transform(std::move(a), static_cast<const Matrix<T> &>(b),
                            fn);
  }

  if ((a.rows == b.rows || a.rows == 1) && (a.cols == b.cols || a.cols == 1)) {
    return binary_transform(static_cast<const Matrix<T> &>(a), std::move(b),
                            fn);
  }

  return binary_transform(static_cast<const Matrix<T> &>(a),
                          static_cast<const Matrix<T> &>(b), fn);
}

template <typename T, typename F>
T reduce(const Matrix<T> &x, std::type_identity_t<T> init, F fn) {
  for (const auto &value : x.data) {
    init = fn(std::move(init), value);
  }

  return init;
}
template <typename U, typename F>
Matrix<U> reduce(const Matrix<U> &x, F fn, size_t axis, U init = U{}) {

  if (axis == 1) {
    Matrix<U> result(x.rows, 1, init);
    for (size_t i = 0; i < x.rows; i++) {
      const U *row = x.data.data() + i * x.cols;

      for (size_t j = 0; j < x.cols; j++) {
        result.data[i] = fn(std::move(result.data[i]), row[j]);
      }
    }
    return result;
  } else {
    Matrix<U> result(1, x.cols, init);

    for (size_t i = 0; i < x.rows; i++) {
      const U *row = x.data.data() + i * x.cols;
      for (size_t j = 0; j < x.cols; j++) {
        result.data[j] = fn(std::move(result.data[j]), row[j]);
      }
    }
    return result;
  }
}
template <typename T, typename M>
  requires std::same_as<std::remove_cvref_t<M>, Matrix<T>>
Matrix<T> operator+(const T &scalar, M &&matrix) {
  return std::forward<M>(matrix) + scalar;
}

template <typename T, typename M>
  requires std::same_as<std::remove_cvref_t<M>, Matrix<T>>
Matrix<T> operator*(const T &scalar, M &&matrix) {
  return std::forward<M>(matrix) * scalar;
}
template <typename T, typename S>
  requires std::convertible_to<S, T>
Matrix<T> operator-(const S &scalar, Matrix<T> matrix) {
  const T converted = static_cast<T>(scalar);

  return transform(std::move(matrix),
                   [converted](T value) { return converted - value; });
}

template <typename T, typename S>
  requires std::convertible_to<S, T>
Matrix<T> operator/(const S &scalar, Matrix<T> matrix) {
  const T converted = static_cast<T>(scalar);

  return transform(std::move(matrix),
                   [converted](T value) { return converted / value; });
}
template <typename T> Matrix<T> transpose(Matrix<T> &&data) {
  const size_t rows = data.get_rows();
  const size_t cols = data.get_cols();

  if (rows == cols) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = i + 1; j < cols; ++j) {
        std::swap(data(i, j), data(j, i));
      }
    }
    return std::move(data);
  }

  Matrix<T> result(cols, rows);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result(j, i) = data(i, j);
    }
  }

  return result;
}

template <typename T> Matrix<T> transpose(const Matrix<T> &data) {
  const size_t rows = data.get_rows();
  const size_t cols = data.get_cols();

  Matrix<T> result(cols, rows);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result(j, i) = data(i, j);
    }
  }

  return result;
}
template <typename T> Matrix<T> sigmoid(Matrix<T> x) {
  return transform(std::move(x),
                   [](T value) { return T{1} / (T{1} + std::exp(-value)); });
}

template <typename T> Matrix<T> global_softmax(Matrix<T> x) {
  const T max_val =
      reduce(x, std::numeric_limits<T>::lowest(),
             [](T current, T value) { return std::max(current, value); });

  x = transform(std::move(x),
                [max_val](T value) { return std::exp(value - max_val); });

  const T denominator =
      reduce(x, T{}, [](T sum, T value) { return sum + value; });

  return transform(std::move(x),
                   [denominator](T value) { return value / denominator; });
}
template <typename T> Matrix<T> softmax(Matrix<T> x) {
  auto max_vals = reduce(
      x, [](T current, T value) { return std::max(current, value); }, 1,
      std::numeric_limits<T>::lowest());

  x = x - max_vals;

  x = transform(std::move(x), [](T value) { return std::exp(value); });

  auto denominators =
      reduce(x, [](T current, T value) { return current + value; }, 1);

  return divide(std::move(x), denominators);
}
template <typename T> Matrix<T> sqrt(Matrix<T> x) {
  return transform(std::move(x), [](T value) { return std::sqrt(value); });
}

template <typename T>
Matrix<T> hadamard(const Matrix<T> &a, const Matrix<T> &b) {
  return binary_transform(a, b, [](T x, T y) { return x * y; });
}

template <typename T> Matrix<T> hadamard(Matrix<T> &&a, const Matrix<T> &b) {
  return binary_transform(std::move(a), b, [](T x, T y) { return x * y; });
}

template <typename T> Matrix<T> hadamard(const Matrix<T> &a, Matrix<T> &&b) {
  return binary_transform(a, std::move(b), [](T x, T y) { return x * y; });
}

template <typename T> Matrix<T> hadamard(Matrix<T> &&a, Matrix<T> &&b) {
  return binary_transform(std::move(a), std::move(b),
                          [](T x, T y) { return x * y; });
}

template <typename T> Matrix<T> divide(const Matrix<T> &a, const Matrix<T> &b) {
  return binary_transform(a, b, [](T x, T y) { return x / y; });
}

template <typename T> Matrix<T> divide(Matrix<T> &&a, const Matrix<T> &b) {
  return binary_transform(std::move(a), b, [](T x, T y) { return x / y; });
}

template <typename T> Matrix<T> divide(const Matrix<T> &a, Matrix<T> &&b) {
  return binary_transform(a, std::move(b), [](T x, T y) { return x / y; });
}

template <typename T> Matrix<T> divide(Matrix<T> &&a, Matrix<T> &&b) {
  return binary_transform(std::move(a), std::move(b),
                          [](T x, T y) { return x / y; });
}
template <typename T> size_t argmax(const Matrix<T> &mat) {
  const auto &data = mat.data;
  auto it = std::ranges::max_element(data);
  return static_cast<size_t>(std::distance(data.begin(), it));
}
template <typename T> Matrix<T> sum(const Matrix<T> &mat) {
  return reduce(mat, [](T current, T value) { return current + value; }, 1);
}

template <typename T>
Matrix<T> sum_to_shape(const Matrix<T> &mat, std::span<const size_t> shape) {
  size_t rows = shape[0];
  size_t cols = shape[1];
  if (mat.get_rows() == rows && mat.get_cols() == cols) {
    return mat;
  }

  else if (mat.get_rows() == rows && cols == 1) {
    return reduce(mat, [](T current, T value) { return current + value; }, 1);
  } else if (mat.get_cols() == cols && rows == 1) {
    return reduce(mat, [](T current, T value) { return current + value; }, 0);
  } else if (rows == 1 && cols == 1) {
    return Matrix<T>(1, 1, reduce(mat, T{}, [](T current, T value) {
                       return current + value;
                     }));
  } else {
    throw std::invalid_argument("Cannot sum matrix to requested shape");
  }
}

template <typename T>
Matrix<T> sum_to_shape(Matrix<T> &&mat, std::span<const size_t> shape) {
  size_t rows = shape[0];
  size_t cols = shape[1];
  if (mat.get_rows() == rows && mat.get_cols() == cols) {
    return std::move(mat);
  }

  else if (mat.get_rows() == rows && cols == 1) {
    return reduce(mat, [](T current, T value) { return current + value; }, 1);
  } else if (mat.get_cols() == cols && rows == 1) {
    return reduce(mat, [](T current, T value) { return current + value; }, 0);
  } else if (rows == 1 && cols == 1) {
    return Matrix<T>(1, 1, reduce(mat, T{}, [](T current, T value) {
                       return current + value;
                     }));
  } else {
    throw std::invalid_argument("Cannot sum matrix to requested shape");
  }
}
template <typename T>
Matrix<T> operator+(const Matrix<T> &a, const Matrix<T> &b) {
  return binary_transform(a, b, [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> operator+(Matrix<T> &&a, const Matrix<T> &b) {
  return binary_transform(std::move(a), b, [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> operator+(const Matrix<T> &a, Matrix<T> &&b) {
  return binary_transform(a, std::move(b), [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> operator+(Matrix<T> &&a, Matrix<T> &&b) {
  return binary_transform(std::move(a), std::move(b),
                          [](T x, T y) { return x + y; });
}
template <typename T>
Matrix<T> operator-(const Matrix<T> &a, const Matrix<T> &b) {
  return binary_transform(a, b, [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> operator-(Matrix<T> &&a, const Matrix<T> &b) {
  return binary_transform(std::move(a), b, [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> operator-(const Matrix<T> &a, Matrix<T> &&b) {
  return binary_transform(a, std::move(b), [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> operator-(Matrix<T> &&a, Matrix<T> &&b) {
  return binary_transform(std::move(a), std::move(b),
                          [](T x, T y) { return x - y; });
}
template <typename T> Matrix<T> add(const Matrix<T> &a, const Matrix<T> &b) {
  return binary_transform(a, b, [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> add(Matrix<T> &&a, const Matrix<T> &b) {
  return binary_transform(std::move(a), b, [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> add(const Matrix<T> &a, Matrix<T> &&b) {
  return binary_transform(a, std::move(b), [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> add(Matrix<T> &&a, Matrix<T> &&b) {
  return binary_transform(std::move(a), std::move(b),
                          [](T x, T y) { return x + y; });
}

template <typename T> Matrix<T> sub(const Matrix<T> &a, const Matrix<T> &b) {
  return binary_transform(a, b, [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> sub(Matrix<T> &&a, const Matrix<T> &b) {
  return binary_transform(std::move(a), b, [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> sub(const Matrix<T> &a, Matrix<T> &&b) {
  return binary_transform(a, std::move(b), [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> sub(Matrix<T> &&a, Matrix<T> &&b) {
  return binary_transform(std::move(a), std::move(b),
                          [](T x, T y) { return x - y; });
}

template <typename T> Matrix<T> mult(const Matrix<T> &a, const Matrix<T> &b) {
  if (a.get_cols() != b.get_rows()) {
    throw std::invalid_argument(
        "Matrices must have compatible dimensions for multiplication");
  }

  Matrix<T> result(a.get_rows(), b.get_cols());

  for (size_t i = 0; i < a.get_rows(); ++i) {
    for (size_t k = 0; k < a.get_cols(); ++k) {
      const T value = a(i, k);

      for (size_t j = 0; j < b.get_cols(); ++j) {
        result(i, j) += value * b(k, j);
      }
    }
  }

  return result;
}
} // namespace math
