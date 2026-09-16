#pragma once

#include "RTensor/RTensor.h"
#include "metal.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
namespace metal {
class Runtime {
  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *command_queue_ = nullptr;
  MTL::Library *library_ = nullptr;
  MTL::CaptureManager *manager_ = nullptr;

  MTL::ComputePipelineState *add_pipeline_ = nullptr;
  MTL::ComputePipelineState *add_scalar_pipeline_ = nullptr;
  MTL::ComputePipelineState *subtract_pipeline_ = nullptr;
  MTL::ComputePipelineState *hadamard_pipeline_ = nullptr;
  MTL::ComputePipelineState *divide_pipeline_ = nullptr;
  MTL::ComputePipelineState *divide_scalar_pipeline_ = nullptr;
  MTL::ComputePipelineState *matmul_contiguous_pipeline_ = nullptr;
  MTL::ComputePipelineState *matmul_strided_pipeline_ = nullptr;
  MTL::ComputePipelineState *matmul_generic_pipeline_ = nullptr;
  MTL::ComputePipelineState *sigmoid_pipeline_ = nullptr;
  MTL::ComputePipelineState *sqrt_pipeline_ = nullptr;
  MTL::ComputePipelineState *pow_pipeline_ = nullptr;
  MTL::ComputePipelineState *exp_pipeline_ = nullptr;
  MTL::ComputePipelineState *sum_pipeline_ = nullptr;
  MTL::ComputePipelineState *transpose_pipeline_ = nullptr;
  MTL::ComputePipelineState *sum_to_shape_pipeline_ = nullptr;
  MTL::ComputePipelineState *max_to_shape_pipeline_ = nullptr;
  MTL::ComputePipelineState *argmax_pipeline_ = nullptr;
  MTL::ComputePipelineState *argmax_reduce_pipeline_ = nullptr;
  MTL::ComputePipelineState *embedding_lookup_pipeline_ = nullptr;
  MTL::ComputePipelineState *scatter_add_rows_pipeline_ = nullptr;
  MTL::ComputePipelineState *gelu_pipeline_ = nullptr;
  MTL::ComputePipelineState *gelu_derivative_pipeline_ = nullptr;
  MTL::ComputePipelineState *softmax_pipeline_ = nullptr;
  MTL::ComputePipelineState *cross_entropy_pipeline_ = nullptr;
  MTL::ComputePipelineState *cross_entropy_derivative_pipeline_ = nullptr;
  MTL::ComputePipelineState *accumulate_gradient_norm_pipeline_ = nullptr;
  MTL::ComputePipelineState *clip_gradient_pipeline_ = nullptr;
  MTL::ComputePipelineState *adamw_pipeline_ = nullptr;
  MTL::ComputePipelineState *copy_pipeline_ = nullptr;

  MTL::CommandBuffer *cmdBuff_ = nullptr;
  MTL::ComputeCommandEncoder *encoder_ = nullptr;

  MTL::ComputePipelineState *load_kernel(std::string_view kernel_name);

  // templated helpers
  template <typename T>
  void call_binary_op(const RTensor<Metal, T> &a, const RTensor<Metal, T> &b,
                      RTensor<Metal, T> &output,
                      MTL::ComputePipelineState *pipe);

  template <typename T>
  void call_unary_op(const RTensor<Metal, T> &a, RTensor<Metal, T> &output,
                     MTL::ComputePipelineState *pipe);

  template <typename T>
  void call_scalar_op(const RTensor<Metal, T> &a, T scalar,
                      RTensor<Metal, T> &output,
                      MTL::ComputePipelineState *pipe);

  void dispatch_scatter_add_rows(const RTensor<Metal, kernel_type> &,
                                 const RTensor<Metal, index_type> &,
                                 RTensor<Metal, kernel_type> &);

  void RMS(const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &,
           RTensor<Metal, kernel_type> &);

public:
  Runtime();
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  ~Runtime();
  void start();
  void submit();
  void wait();
  void abort() noexcept;
  void cleanup() noexcept;

  void start_profile(std::string_view output_path);
  void end_profile();
  void reduce_to_shape(const RTensor<Metal, kernel_type> &a,
                       std::span<const uint32_t> output_shape,
                       RTensor<Metal, kernel_type> &output,
                       MTL::ComputePipelineState *pipe);
  void add(const RTensor<Metal, kernel_type> &,
           const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void add(const RTensor<Metal, kernel_type> &, kernel_type,
           RTensor<Metal, kernel_type> &);
  void sub(const RTensor<Metal, kernel_type> &,
           const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void mult(const RTensor<Metal, kernel_type> &,
            const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void hadamard(const RTensor<Metal, kernel_type> &,
                const RTensor<Metal, kernel_type> &,
                RTensor<Metal, kernel_type> &);
  void divide(const RTensor<Metal, kernel_type> &,
              const RTensor<Metal, kernel_type> &,
              RTensor<Metal, kernel_type> &);
  void divide(const RTensor<Metal, kernel_type> &, kernel_type,
              RTensor<Metal, kernel_type> &);
  void sigmoid(const RTensor<Metal, kernel_type> &,
               RTensor<Metal, kernel_type> &);
  void sqrt(const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void pow(const RTensor<Metal, kernel_type> &, kernel_type,
           RTensor<Metal, kernel_type> &);
  void exp(const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void sum(const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void sum(const RTensor<Metal, kernel_type> &, size_t axis,
           RTensor<Metal, kernel_type> &);
  void transpose(const RTensor<Metal, kernel_type> &,
                 RTensor<Metal, kernel_type> &, int32_t dim_a = -2,
                 int32_t dim_b = -1);
  void split_last_dim(const RTensor<Metal, kernel_type> &,
                      RTensor<Metal, kernel_type> &, uint32_t outer,
                      uint32_t inner);
  void merge_last_dims(const RTensor<Metal, kernel_type> &,
                       RTensor<Metal, kernel_type> &, uint32_t outer,
                       uint32_t inner);
  void sum_to_shape(const RTensor<Metal, kernel_type> &,
                    std::span<const uint32_t>, RTensor<Metal, kernel_type> &);
  void max_to_shape(const RTensor<Metal, kernel_type> &,
                    std::span<const uint32_t>, RTensor<Metal, kernel_type> &);
  void softmax(const RTensor<Metal, kernel_type> &,
               RTensor<Metal, kernel_type> &);
  void cross_entropy(const RTensor<Metal, kernel_type> &,
                     const RTensor<Metal, index_type> &,
                     RTensor<Metal, kernel_type> &);
  void cross_entropy_derivative(const RTensor<Metal, kernel_type> &,
                                const RTensor<Metal, index_type> &,
                                RTensor<Metal, kernel_type> &);
  void sqe(const RTensor<Metal, kernel_type> &,
           const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void argmax(const RTensor<Metal, kernel_type> &,
              RTensor<Metal, index_type> &);
  void embedding_lookup(const RTensor<Metal, kernel_type> &,
                        const RTensor<Metal, index_type> &,
                        RTensor<Metal, kernel_type> &);
  void gather_rows(const RTensor<Metal, kernel_type> &,
                   const RTensor<Metal, index_type> &,
                   RTensor<Metal, kernel_type> &);
  void scatter_add_rows(const RTensor<Metal, kernel_type> &,
                        const RTensor<Metal, index_type> &,
                        const RTensor<Metal, kernel_type> &,
                        RTensor<Metal, kernel_type> &);
  void scatter_index(const RTensor<Metal, kernel_type> &,
                     const RTensor<Metal, index_type> &,
                     RTensor<Metal, kernel_type> &);
  void RMS(const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void RMSNorm(const RTensor<Metal, kernel_type> &,
               RTensor<Metal, kernel_type> &);
  void GELU(const RTensor<Metal, kernel_type> &, RTensor<Metal, kernel_type> &);
  void GELU_derivative(const RTensor<Metal, kernel_type> &,
                       RTensor<Metal, kernel_type> &);
  void accumulate_gradient_squared_norm(
      const RTensor<Metal, kernel_type> &gradient,
      RTensor<Metal, kernel_type> &squared_norm);
  void clip_gradient(const RTensor<Metal, kernel_type> &gradient,
                     const RTensor<Metal, kernel_type> &squared_norm,
                     kernel_type max_norm,
                     RTensor<Metal, kernel_type> &destination);
  void adamW(const RTensor<Metal, kernel_type> &parameter,
             const RTensor<Metal, kernel_type> &gradient,
             RTensor<Metal, kernel_type> &destination,
             RTensor<Metal, kernel_type> &momentum1,
             RTensor<Metal, kernel_type> &momentum2,
             const RTensor<Metal, kernel_type> &beta1,
             const RTensor<Metal, kernel_type> &beta2,
             const RTensor<Metal, kernel_type> &decay,
             const RTensor<Metal, kernel_type> &learning_rate,
             kernel_type epsilon, uint32_t step);

  void copy(const RTensor<Metal, kernel_type> &a,
            RTensor<Metal, kernel_type> &output);
};
} // namespace metal
