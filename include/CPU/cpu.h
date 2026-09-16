#pragma once
#include "RTensor/RTensor.h"

#include <cstddef>
#include <span>

namespace cpu {

struct CPU {};

using kernel_type = float;

RTensor<CPU, kernel_type> add(const RTensor<CPU, kernel_type> &,
                              const RTensor<CPU, kernel_type> &);
void add(const RTensor<CPU, kernel_type> &, const RTensor<CPU, kernel_type> &,
         RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> sub(const RTensor<CPU, kernel_type> &,
                              const RTensor<CPU, kernel_type> &);
void sub(const RTensor<CPU, kernel_type> &, const RTensor<CPU, kernel_type> &,
         RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> mult(const RTensor<CPU, kernel_type> &,
                               const RTensor<CPU, kernel_type> &);
void mult(const RTensor<CPU, kernel_type> &, const RTensor<CPU, kernel_type> &,
          RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> hadamard(const RTensor<CPU, kernel_type> &,
                                   const RTensor<CPU, kernel_type> &);
void hadamard(const RTensor<CPU, kernel_type> &,
              const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> divide(const RTensor<CPU, kernel_type> &,
                                 const RTensor<CPU, kernel_type> &);
void divide(const RTensor<CPU, kernel_type> &,
            const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> sigmoid(const RTensor<CPU, kernel_type> &);
void sigmoid(const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> sqrt(const RTensor<CPU, kernel_type> &);
void sqrt(const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> sum(const RTensor<CPU, kernel_type> &);
void sum(const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> sum(const RTensor<CPU, kernel_type> &, size_t axis);
void sum(const RTensor<CPU, kernel_type> &, size_t axis,
         RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> transpose(const RTensor<CPU, kernel_type> &,
                                    int32_t dim_a = -2,
                                    int32_t dim_b = -1);
void transpose(const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &,
               int32_t dim_a = -2, int32_t dim_b = -1);

RTensor<CPU, kernel_type> softmax(const RTensor<CPU, kernel_type> &);
void softmax(const RTensor<CPU, kernel_type> &, RTensor<CPU, kernel_type> &);

void sqe(const RTensor<CPU, kernel_type> &, const RTensor<CPU, kernel_type> &,
         RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> argmax(const RTensor<CPU, kernel_type> &);

RTensor<CPU, kernel_type> sum_to_shape(const RTensor<CPU, kernel_type> &,
                                       std::span<const uint32_t>);
void sum_to_shape(const RTensor<CPU, kernel_type> &, std::span<const uint32_t>,
                  RTensor<CPU, kernel_type> &);

} // namespace cpu
