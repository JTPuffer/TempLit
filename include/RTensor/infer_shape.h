#pragma once

#include "RTensor.h"
#include "dimension.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>

namespace rtensor {

inline shape_type create_out(std::span<const uint32_t> a_shape,
                             std::span<const uint32_t> b_shape) {

  size_t rank = std::max(a_shape.size(), b_shape.size());
  std::vector<uint32_t> dims(rank);

  for (size_t i = 0; i < rank; ++i) {
    uint32_t a_dim = 1;
    uint32_t b_dim = 1;

    if (i < a_shape.size()) {
      a_dim = a_shape[a_shape.size() - 1 - i];
    }

    if (i < b_shape.size()) {
      b_dim = b_shape[b_shape.size() - 1 - i];
    }

    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      throw std::invalid_argument("shapes cannot be broadcast");
    }
    dims[rank - 1 - i] = std::max(a_dim, b_dim);
  }
  return dims;
}

inline shape_type infer_add(std::span<const index_type> a_shape,
                            std::span<const index_type> b_shape) {

  return create_out(a_shape, b_shape);
}

inline shape_type infer_sub(std::span<const index_type> a_shape,
                            std::span<const index_type> b_shape) {

  return create_out(a_shape, b_shape);
}

inline shape_type infer_hadamard(std::span<const index_type> a_shape,
                                 std::span<const index_type> b_shape) {

  return create_out(a_shape, b_shape);
}

inline shape_type infer_divide(std::span<const index_type> a_shape,
                               std::span<const index_type> b_shape) {

  return create_out(a_shape, b_shape);
}

inline shape_type infer_mult(std::span<const index_type> a,
                             std::span<const index_type> b) {
  if (a.size() < 2 || b.size() < 2) {
    throw std::runtime_error("matmul requires rank >= 2");
  }

  uint32_t M = a[a.size() - 2];
  uint32_t K = a[a.size() - 1];
  uint32_t N = b[b.size() - 1];
  uint32_t Kb = b[b.size() - 2];

  if (K != Kb) {
    throw std::invalid_argument("matmul inner dimensions do not match");
  }

  // creating output matrix size after broadcasting
  const auto &a_shape = a;
  const auto &b_shape = b;

  size_t a_batch_rank = a_shape.size() - 2;
  size_t b_batch_rank = b_shape.size() - 2;
  size_t rank = std::max(a_batch_rank, b_batch_rank);

  std::vector<uint32_t> dims(rank);

  for (size_t i = 0; i < rank; ++i) {
    uint32_t a_dim = 1;
    uint32_t b_dim = 1;

    if (i < a_batch_rank) {
      a_dim = a_shape[a_batch_rank - 1 - i];
    }

    if (i < b_batch_rank) {
      b_dim = b_shape[b_batch_rank - 1 - i];
    }

    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      throw std::runtime_error("shapes cannot be broadcast");
    }

    dims[rank - 1 - i] = std::max(a_dim, b_dim);
  }

  dims.push_back(M);
  dims.push_back(N);
  return dims;
}

inline shape_type infer_unary(std::span<const index_type> shape) {
  return shape_type(shape.begin(), shape.end());
}

inline shape_type infer_sigmoid(std::span<const index_type> shape) {
  return infer_unary(shape);
}

inline shape_type infer_sqrt(std::span<const index_type> shape) {
  return infer_unary(shape);
}

inline shape_type infer_rms(std::span<const index_type> shape) {
  if (shape.empty()) {
    throw std::invalid_argument("RMS requires rank >= 1");
  }

  shape_type output_shape(shape.begin(), shape.end());
  output_shape.back() = 1;
  return output_shape;
}

inline shape_type infer_rms_norm(std::span<const index_type> shape) {
  if (shape.empty()) {
    throw std::invalid_argument("RMSNorm requires rank >= 1");
  }

  return infer_unary(shape);
}

inline shape_type infer_exp(std::span<const index_type> shape) {
  return infer_unary(shape);
}

inline shape_type infer_sum(std::span<const index_type>) { return {1}; }

inline shape_type infer_transpose(std::span<const index_type> shape,
                                  int32_t dim_a = -2,
                                  int32_t dim_b = -1) {
  if (shape.size() < 2) {
    throw std::invalid_argument("transpose requires rank >= 2");
  }

  const size_t axis_a = normalise_dimension(shape.size(), dim_a);
  const size_t axis_b = normalise_dimension(shape.size(), dim_b);
  if (axis_a == axis_b) {
    throw std::invalid_argument("transpose dimensions must be different");
  }

  std::vector<uint32_t> output_shape(shape.begin(), shape.end());
  std::swap(output_shape[axis_a], output_shape[axis_b]);

  return output_shape;
}

inline shape_type infer_split_last_dim(std::span<const index_type> shape,
                                       index_type outer, index_type inner) {
  if (shape.empty()) {
    throw std::invalid_argument("split_last_dim requires rank >= 1");
  }
  if (outer == 0 || inner == 0 ||
      static_cast<uint64_t>(outer) * inner != shape.back()) {
    throw std::invalid_argument(
        "split_last_dim factors must match the final dimension");
  }

  shape_type output_shape(shape.begin(), shape.end() - 1);
  output_shape.push_back(outer);
  output_shape.push_back(inner);
  return output_shape;
}

inline shape_type infer_merge_last_dims(std::span<const index_type> shape,
                                        index_type outer, index_type inner) {
  if (shape.size() < 2) {
    throw std::invalid_argument("merge_last_dims requires rank >= 2");
  }
  if (shape[shape.size() - 2] != outer || shape.back() != inner) {
    throw std::invalid_argument(
        "merge_last_dims factors do not match the final dimensions");
  }

  const uint64_t merged = static_cast<uint64_t>(outer) * inner;
  if (merged > std::numeric_limits<index_type>::max()) {
    throw std::length_error("merged tensor dimension exceeds index capacity");
  }

  shape_type output_shape(shape.begin(), shape.end() - 2);
  output_shape.push_back(static_cast<index_type>(merged));
  return output_shape;
}

inline shape_type
infer_embedding_lookup(std::span<const index_type> weights_shape,
                       std::span<const index_type> ids_shape) {
  if (weights_shape.size() != 2) {
    throw std::invalid_argument("embedding weights must have rank 2");
  }

  shape_type output_shape(ids_shape.begin(), ids_shape.end());
  output_shape.push_back(weights_shape.back());
  return output_shape;
}

inline shape_type
infer_cross_entropy(std::span<const index_type> logits_shape,
                    std::span<const index_type> targets_shape) {
  if (logits_shape.size() < 2 || logits_shape.back() == 0) {
    throw std::invalid_argument(
        "cross_entropy requires logits with a non-empty final dimension");
  }
  if (targets_shape.size() + 1 != logits_shape.size()) {
    throw std::invalid_argument(
        "cross_entropy target rank must be one less than logits rank");
  }
  for (size_t i = 0; i < targets_shape.size(); ++i) {
    if (targets_shape[i] != logits_shape[i]) {
      throw std::invalid_argument(
          "cross_entropy target shape must match logits leading dimensions");
    }
  }

  shape_type output_shape(targets_shape.begin(), targets_shape.end());
  output_shape.push_back(1);
  return output_shape;
}

inline shape_type
infer_cross_entropy_derivative(std::span<const index_type> logits_shape,
                               std::span<const index_type> targets_shape) {
  (void)infer_cross_entropy(logits_shape, targets_shape);
  return shape_type(logits_shape.begin(), logits_shape.end());
}

inline shape_type
infer_scatter_index(std::span<const index_type> values_shape,
                    std::span<const index_type> ids_shape,
                    std::span<const index_type> output_shape) {
  if (output_shape.size() != 2) {
    throw std::invalid_argument("scatter output must have rank 2");
  }

  shape_type expected_values_shape(ids_shape.begin(), ids_shape.end());
  expected_values_shape.push_back(output_shape.back());
  const shape_type values(values_shape.begin(), values_shape.end());
  if (values != expected_values_shape) {
    throw std::invalid_argument(
        "scatter values shape does not match IDs and output");
  }

  return shape_type(output_shape.begin(), output_shape.end());
}

inline shape_type
infer_scatter_add_rows(std::span<const index_type> base_shape,
                       std::span<const index_type> ids_shape,
                       std::span<const index_type> values_shape) {
  return infer_scatter_index(values_shape, ids_shape, base_shape);
}

inline shape_type
infer_reduce_to_shape(std::span<const index_type> a_shape,
                      std::span<const index_type> output_shape) {
  if (output_shape.size() > a_shape.size()) {
    throw std::invalid_argument(
        "cannot reduce to a shape with a higher rank");
  }

  const size_t rank_offset = a_shape.size() - output_shape.size();

  for (size_t i = 0; i < a_shape.size(); ++i) {
    const index_type output_dim =
        i < rank_offset ? index_type{1} : output_shape[i - rank_offset];
    if (a_shape[i] != output_dim && output_dim != 1) {
      throw std::invalid_argument("invalid reduce_to_shape");
    }
  }

  return shape_type(output_shape.begin(), output_shape.end());
}

inline shape_type infer_sum_to_shape(std::span<const index_type> a_shape,
                                     std::span<const index_type> output_shape) {
  return infer_reduce_to_shape(a_shape, output_shape);
}

inline shape_type infer_max_to_shape(std::span<const index_type> a_shape,
                                     std::span<const index_type> output_shape) {
  return infer_reduce_to_shape(a_shape, output_shape);
}

inline shape_type infer_sum(std::span<const index_type> input_shape,
                            size_t axis) {
  if (axis >= input_shape.size()) {
    throw std::out_of_range("sum axis out of range");
  }

  std::vector<uint32_t> output_shape(input_shape.begin(), input_shape.end());
  output_shape[axis] = 1;

  return infer_sum_to_shape(input_shape, output_shape);
}

inline shape_type infer_softmax(std::span<const index_type> x_shape) {
  auto reduced_shape = shape_type(x_shape.begin(), x_shape.end());
  reduced_shape.back() = 1;

  auto max_vals = infer_max_to_shape(x_shape, reduced_shape);
  auto shifted = infer_sub(x_shape, max_vals);
  auto exps = infer_exp(shifted);
  auto sums = infer_sum_to_shape(exps, reduced_shape);

  return infer_divide(exps, sums);
}

inline shape_type infer_argmax(std::span<const index_type> shape) {
  const size_t element_count = std::accumulate(shape.begin(), shape.end(),
                                               size_t{1}, std::multiplies<>{});

  if (element_count == 0) {
    throw std::invalid_argument("argmax requires a non-empty tensor");
  }

  return {1, 1};
}

} // namespace rtensor
