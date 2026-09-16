#pragma once

#include "expr/expr.h"
#include <memory>
#include <type_traits>
#include <utility>
namespace expr {

namespace tensor {
struct TRole {};
struct grad : TRole {};
struct no_grad : TRole {};

template <class x>
concept PARAM_TYPE =
    std::is_base_of<TRole, typename std::decay<x>::type>::value;

template <typename T, PARAM_TYPE Policy = no_grad> struct Tensor : TENSOR_EXPR {
private:
  std::shared_ptr<T> value;

public:
  Tensor(T tens_val) : value(std::make_shared<T>(std::move(tens_val))) {}
  Tensor() : value(std::make_shared<T>()) {}
  const T &operator()() noexcept { return *value; }
  const T &get_value() const noexcept { return *value; }
  T &get_value() noexcept { return *value; }
  void set_value(const T &data) { *value = data; }
  void set_value(T &&data) { *value = std::move(data); }
};
template <typename T> using GradTensor = Tensor<T, grad>;

template <typename T> using NoGradTensor = Tensor<T, no_grad>;

} // namespace tensor

template <typename T>
struct requires_grad<expr::tensor::Tensor<T, expr::tensor::grad>>
    : std::true_type {};

template <typename T>
struct requires_grad<expr::tensor::Tensor<T, expr::tensor::no_grad>>
    : std::false_type {};
} // namespace expr
