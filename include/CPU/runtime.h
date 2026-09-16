#pragma once

#include "CPU/cpu.h"
#include "CPU/storage.h"
#include "RTensor/infer_shape.h"

#include <algorithm>

namespace cpu {

class Runtime {
public:
  void start() {}
  void submit() {}
  void wait() {}
  void abort() noexcept {}
  void cleanup() {}

  void copy(const RTensor<CPU, kernel_type> &a,
            RTensor<CPU, kernel_type> &destination) {
    if (a.shape() != destination.shape()) {
      throw std::invalid_argument("copy destination has incorrect shape");
    }

    std::copy(a.get_storage()->data().begin(), a.get_storage()->data().end(),
              destination.get_storage()->data().begin());
  }

  void add(const RTensor<CPU, kernel_type> &a,
           const RTensor<CPU, kernel_type> &b,
           RTensor<CPU, kernel_type> &destination) {
    cpu::add(a, b, destination);
  }

  void sub(const RTensor<CPU, kernel_type> &a,
           const RTensor<CPU, kernel_type> &b,
           RTensor<CPU, kernel_type> &destination) {
    cpu::sub(a, b, destination);
  }

  void mult(const RTensor<CPU, kernel_type> &a,
            const RTensor<CPU, kernel_type> &b,
            RTensor<CPU, kernel_type> &destination) {
    cpu::mult(a, b, destination);
  }

  void transpose(const RTensor<CPU, kernel_type> &a,
                 RTensor<CPU, kernel_type> &destination, int32_t dim_a = -2,
                 int32_t dim_b = -1) {
    cpu::transpose(a, destination, dim_a, dim_b);
  }

  void split_last_dim(const RTensor<CPU, kernel_type> &a,
                      RTensor<CPU, kernel_type> &destination, uint32_t outer,
                      uint32_t inner) {
    if (destination.shape() !=
        rtensor::infer_split_last_dim(a.shape(), outer, inner)) {
      throw std::invalid_argument(
          "split_last_dim destination has incorrect shape");
    }
    const auto shape = destination.shape();
    destination = a.view(shape);
  }

  void merge_last_dims(const RTensor<CPU, kernel_type> &a,
                       RTensor<CPU, kernel_type> &destination, uint32_t outer,
                       uint32_t inner) {
    if (destination.shape() !=
        rtensor::infer_merge_last_dims(a.shape(), outer, inner)) {
      throw std::invalid_argument(
          "merge_last_dims destination has incorrect shape");
    }
    const auto shape = destination.shape();
    destination = a.view(shape);
  }

  void sigmoid(const RTensor<CPU, kernel_type> &a,
               RTensor<CPU, kernel_type> &destination) {
    cpu::sigmoid(a, destination);
  }

  void hadamard(const RTensor<CPU, kernel_type> &a,
                const RTensor<CPU, kernel_type> &b,
                RTensor<CPU, kernel_type> &destination) {
    cpu::hadamard(a, b, destination);
  }

  void divide(const RTensor<CPU, kernel_type> &a,
              const RTensor<CPU, kernel_type> &b,
              RTensor<CPU, kernel_type> &destination) {
    cpu::divide(a, b, destination);
  }

  void sqrt(const RTensor<CPU, kernel_type> &a,
            RTensor<CPU, kernel_type> &destination) {
    cpu::sqrt(a, destination);
  }

  void sum(const RTensor<CPU, kernel_type> &a,
           RTensor<CPU, kernel_type> &destination) {
    cpu::sum(a, destination);
  }

  void sum(const RTensor<CPU, kernel_type> &a, size_t axis,
           RTensor<CPU, kernel_type> &destination) {
    cpu::sum(a, axis, destination);
  }

  void sum_to_shape(const RTensor<CPU, kernel_type> &a,
                    std::span<const uint32_t> shape,
                    RTensor<CPU, kernel_type> &destination) {
    cpu::sum_to_shape(a, shape, destination);
  }

  void softmax(const RTensor<CPU, kernel_type> &a,
               RTensor<CPU, kernel_type> &destination) {
    cpu::softmax(a, destination);
  }

  void sqe(const RTensor<CPU, kernel_type> &pred,
           const RTensor<CPU, kernel_type> &target,
           RTensor<CPU, kernel_type> &destination) {
    cpu::sqe(pred, target, destination);
  }
};

} // namespace cpu
