#include "my_metal/runtime.h"
#include "RTensor/dimension.h"
#include "RTensor/infer_shape.h"
#include "my_metal/storage.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace metal {

namespace {
std::vector<uint32_t> broadcast_strides(std::span<const uint32_t> shape,
                                        std::span<const uint32_t> strides,
                                        std::span<const uint32_t> out_shape) {

  std::vector<uint32_t> result(out_shape.size(), 0);
  //[3,4] - [1,4]
  size_t offset = out_shape.size() - shape.size();

  for (size_t i = 0; i < shape.size(); ++i) {
    size_t out_i = offset + i;

    if (shape[i] == out_shape[out_i]) {
      result[out_i] = strides[i];
    } else if (shape[i] == 1) {
      result[out_i] = 0;
    } else {
      throw std::runtime_error("invalid broadcast");
    }
  }

  return result;
}
} // namespace

void Runtime::start_profile(std::string_view output_path) {

  auto *descriptor = MTL::CaptureDescriptor::alloc()->init();
  descriptor->setCaptureObject(command_queue_);
  descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);

  auto *path = NS::String::string(std::string(output_path).c_str(),
                                  NS::UTF8StringEncoding);

  descriptor->setOutputURL(NS::URL::fileURLWithPath(path));

  NS::Error *error = nullptr;
  if (!manager_->startCapture(descriptor, &error)) {
    const char *message = error ? error->localizedDescription()->utf8String()
                                : "unknown capture error";

    descriptor->release();
    throw std::runtime_error(message);
  }
  descriptor->release();
}
void Runtime::end_profile() { manager_->stopCapture(); }

template <typename T>
void Runtime::call_binary_op(const RTensor<Metal, T> &a,
                             const RTensor<Metal, T> &b,
                             RTensor<Metal, T> &output,
                             MTL::ComputePipelineState *pipe) {
  encoder_->setComputePipelineState(pipe);

  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(b.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 2);

  auto stride_a = broadcast_strides(a.shape(), a.strides(), output.shape());
  auto stride_b = broadcast_strides(b.shape(), b.strides(), output.shape());

  encoder_->setBytes(stride_a.data(), stride_a.size() * sizeof(uint32_t), 3);
  encoder_->setBytes(stride_b.data(), stride_b.size() * sizeof(uint32_t), 4);
  encoder_->setBytes(output.strides().data(),
                     output.strides().size() * sizeof(uint32_t), 5);

  uint32_t nDim = static_cast<uint32_t>(output.shape().size());
  encoder_->setBytes(&nDim, sizeof(nDim), 6);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);

  MTL::Size threadgroupSize(256, 1, 1);

  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

template <typename T>
void Runtime::call_unary_op(const RTensor<Metal, T> &a,
                            RTensor<Metal, T> &output,
                            MTL::ComputePipelineState *pipe) {
  encoder_->setComputePipelineState(pipe);

  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);

  MTL::Size threadgroupSize(256, 1, 1);

  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

template <typename T>
void Runtime::call_scalar_op(const RTensor<Metal, T> &a, T scalar,
                             RTensor<Metal, T> &output,
                             MTL::ComputePipelineState *pipe) {
  if (a.shape() != output.shape()) {
    throw std::invalid_argument(
        "scalar operation input and output shapes must match");
  }

  encoder_->setComputePipelineState(pipe);
  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBytes(&scalar, sizeof(scalar), 1);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 2);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);
  MTL::Size threadgroupSize(256, 1, 1);
  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

MTL::ComputePipelineState *Runtime::load_kernel(std::string_view kernel_name) {
  NS::Error *error = nullptr;
  std::string name(kernel_name);

  MTL::Function *fun = library_->newFunction(
      NS::String::string(name.c_str(), NS::UTF8StringEncoding));

  if (!fun) {
    throw std::runtime_error("Failed to load Metal kernel '" + name + "'");
  }

  MTL::ComputePipelineState *pipe =
      device_->newComputePipelineState(fun, &error);

  fun->release();

  if (!pipe) {
    if (error) {
      std::cerr << "Failed to create Metal pipeline for kernel '" << name
                << "': " << error->localizedDescription()->utf8String() << '\n';
    } else {
      std::cerr << "Failed to create Metal pipeline for kernel '" << name
                << "'\n";
    }

    throw std::runtime_error("Failed to create Metal pipeline for kernel '" +
                             name + "'");
  }

  return pipe;
}

void Runtime::abort() noexcept {
  if (encoder_) {
    encoder_->endEncoding();
    encoder_ = nullptr;
  }

  cmdBuff_ = nullptr;
}

void Runtime::cleanup() noexcept {
  if (encoder_) {
    abort();
  }

  if (cmdBuff_) {
    cmdBuff_->waitUntilCompleted();
    cmdBuff_ = nullptr;
  }

  auto release = [](auto *&object) {
    if (object) {
      object->release();
      object = nullptr;
    }
  };

  release(copy_pipeline_);
  release(adamw_pipeline_);
  release(clip_gradient_pipeline_);
  release(accumulate_gradient_norm_pipeline_);
  release(gelu_derivative_pipeline_);
  release(gelu_pipeline_);
  release(softmax_pipeline_);
  release(cross_entropy_pipeline_);
  release(cross_entropy_derivative_pipeline_);
  release(scatter_add_rows_pipeline_);
  release(embedding_lookup_pipeline_);
  release(argmax_reduce_pipeline_);
  release(argmax_pipeline_);
  release(max_to_shape_pipeline_);
  release(sum_to_shape_pipeline_);
  release(transpose_pipeline_);
  release(sum_pipeline_);
  release(exp_pipeline_);
  release(pow_pipeline_);
  release(sqrt_pipeline_);
  release(sigmoid_pipeline_);
  release(matmul_generic_pipeline_);
  release(matmul_strided_pipeline_);
  release(matmul_contiguous_pipeline_);
  release(divide_pipeline_);
  release(divide_scalar_pipeline_);
  release(hadamard_pipeline_);
  release(subtract_pipeline_);
  release(add_pipeline_);
  release(add_scalar_pipeline_);
  release(library_);
  release(command_queue_);
  manager_ = nullptr;
  device_ = nullptr;
}

Runtime::Runtime() {
  try {
    device_ = metal::device();
    if (!device_) {
      throw std::logic_error("Metal must be initialised before Runtime");
    }

    command_queue_ = device_->newCommandQueue();

    NS::Error *error = nullptr;
    library_ = device_->newLibrary(
        NS::String::string("./Metal_Math.metallib", NS::ASCIIStringEncoding),
        &error);

    if (library_ == nullptr) {
      throw std::runtime_error("Failed to load Metal library");
    }
    manager_ = MTL::CaptureManager::sharedCaptureManager();

    add_pipeline_ = load_kernel("add");
    add_scalar_pipeline_ = load_kernel("add_scalar");
    subtract_pipeline_ = load_kernel("sub");
    hadamard_pipeline_ = load_kernel("hadamard");
    divide_pipeline_ = load_kernel("divide");
    divide_scalar_pipeline_ = load_kernel("divide_scalar");
    matmul_contiguous_pipeline_ = load_kernel("matmul_contiguous");
    matmul_strided_pipeline_ = load_kernel("matmul_strided");
    matmul_generic_pipeline_ = load_kernel("matmul_generic");
    sigmoid_pipeline_ = load_kernel("sigmoid");
    sqrt_pipeline_ = load_kernel("sqrt");
    pow_pipeline_ = load_kernel("pow_scalar");
    exp_pipeline_ = load_kernel("exp");
    sum_pipeline_ = load_kernel("sum");
    transpose_pipeline_ = load_kernel("transpose");
    sum_to_shape_pipeline_ = load_kernel("sum_to_shape");
    argmax_pipeline_ = load_kernel("argmax");
    argmax_reduce_pipeline_ = load_kernel("argmax_reduce");
    embedding_lookup_pipeline_ = load_kernel("embedding_lookup");
    scatter_add_rows_pipeline_ = load_kernel("scatter_add_rows");
    max_to_shape_pipeline_ = load_kernel("max_to_shape");
    gelu_pipeline_ = load_kernel("GELU");
    gelu_derivative_pipeline_ = load_kernel("GELU_derivative");
    softmax_pipeline_ = load_kernel("softmax");
    cross_entropy_pipeline_ = load_kernel("cross_entropy_rows");
    cross_entropy_derivative_pipeline_ =
        load_kernel("cross_entropy_rows_backward");
    accumulate_gradient_norm_pipeline_ =
        load_kernel("accumulate_gradient_squared_norm");
    clip_gradient_pipeline_ = load_kernel("clip_gradient");
    adamw_pipeline_ = load_kernel("adamW");
    copy_pipeline_ = load_kernel("copy");
  } catch (...) {
    cleanup();
    throw;
  }
}

Runtime::~Runtime() { cleanup(); }
void Runtime::start() {
  if (cmdBuff_ || encoder_) {
    throw std::logic_error("Metal runtime execution already started");
  }

  cmdBuff_ = command_queue_->commandBuffer();
  encoder_ = cmdBuff_->computeCommandEncoder();
}

void Runtime::submit() {
  if (!cmdBuff_ || !encoder_) {
    throw std::logic_error("Metal runtime execution has not started");
  }

  // need to add something to keep track of params possibly a memory pool or
  // somethin!!
  encoder_->endEncoding();
  encoder_ = nullptr;
  cmdBuff_->commit();
}
void Runtime::wait() {
  if (!cmdBuff_ || encoder_) {
    throw std::logic_error("Metal runtime execution has not been submitted");
  }

  cmdBuff_->waitUntilCompleted();
  cmdBuff_ = nullptr;
}

using kernel_type = float;

void Runtime::add(const RTensor<Metal, kernel_type> &a,
                  const RTensor<Metal, kernel_type> &b,
                  RTensor<Metal, kernel_type> &output) {
  call_binary_op(a, b, output, add_pipeline_);
}

void Runtime::add(const RTensor<Metal, kernel_type> &a, kernel_type scalar,
                  RTensor<Metal, kernel_type> &output) {
  call_scalar_op(a, scalar, output, add_scalar_pipeline_);
}

void Runtime::sub(const RTensor<Metal, kernel_type> &a,
                  const RTensor<Metal, kernel_type> &b,
                  RTensor<Metal, kernel_type> &output) {
  call_binary_op(a, b, output, subtract_pipeline_);
}

void Runtime::hadamard(const RTensor<Metal, kernel_type> &a,
                       const RTensor<Metal, kernel_type> &b,
                       RTensor<Metal, kernel_type> &output) {
  call_binary_op(a, b, output, hadamard_pipeline_);
}

void Runtime::divide(const RTensor<Metal, kernel_type> &a,
                     const RTensor<Metal, kernel_type> &b,
                     RTensor<Metal, kernel_type> &output) {
  call_binary_op(a, b, output, divide_pipeline_);
}

void Runtime::divide(const RTensor<Metal, kernel_type> &a, kernel_type scalar,
                     RTensor<Metal, kernel_type> &output) {
  call_scalar_op(a, scalar, output, divide_scalar_pipeline_);
}

void Runtime::sigmoid(const RTensor<Metal, kernel_type> &input,
                      RTensor<Metal, kernel_type> &output) {
  call_unary_op(input, output, sigmoid_pipeline_);
}

void Runtime::sqrt(const RTensor<Metal, kernel_type> &input,
                   RTensor<Metal, kernel_type> &output) {
  call_unary_op(input, output, sqrt_pipeline_);
}

void Runtime::pow(const RTensor<Metal, kernel_type> &input,
                  kernel_type exponent, RTensor<Metal, kernel_type> &output) {
  call_scalar_op(input, exponent, output, pow_pipeline_);
}

void Runtime::exp(const RTensor<Metal, kernel_type> &input,
                  RTensor<Metal, kernel_type> &output) {
  call_unary_op(input, output, exp_pipeline_);
}

void Runtime::mult(const RTensor<Metal, kernel_type> &a,
                   const RTensor<Metal, kernel_type> &b,
                   RTensor<Metal, kernel_type> &output) {

  if (a.shape().size() < 2 || b.shape().size() < 2) {
    throw std::runtime_error("matmul requires rank >= 2");
  }

  uint32_t M = a.shape()[a.shape().size() - 2];
  uint32_t K = a.shape()[a.shape().size() - 1];
  uint32_t N = b.shape()[b.shape().size() - 1];
  uint32_t Kb = b.shape()[b.shape().size() - 2];

  if (K != Kb) {
    throw std::invalid_argument("matmul inner dimensions do not match");
  }

  const auto &a_shape = a.shape();
  const auto &b_shape = b.shape();

  size_t a_batch_rank = a_shape.size() - 2;
  size_t b_batch_rank = b_shape.size() - 2;
  size_t rank = std::max(a_batch_rank, b_batch_rank);

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
  }

  // batch strides
  std::vector<uint32_t> batch_decode_strides(output.shape().size() - 2);

  uint32_t stride = 1;
  for (size_t i = output.shape().size() - 2; i-- > 0;) {
    batch_decode_strides[i] = stride;
    stride *= output.shape()[i];
  }

  auto a_batch_shape =
      std::span<const uint32_t>(a.shape()).first(a.shape().size() - 2);
  auto a_batch_strides =
      std::span<const uint32_t>(a.strides()).first(a.strides().size() - 2);

  auto b_batch_shape =
      std::span<const uint32_t>(b.shape()).first(b.shape().size() - 2);
  auto b_batch_strides =
      std::span<const uint32_t>(b.strides()).first(b.strides().size() - 2);

  auto out_batch_shape = std::span<const uint32_t>(output.shape())
                             .first(output.shape().size() - 2);

  auto batch_stride_a =
      broadcast_strides(a_batch_shape, a_batch_strides, out_batch_shape);
  auto batch_stride_b =
      broadcast_strides(b_batch_shape, b_batch_strides, out_batch_shape);
  auto batch_stride_c = std::span<const uint32_t>(output.strides())
                            .first(output.strides().size() - 2);

  uint32_t nDim = static_cast<uint32_t>(batch_decode_strides.size());

  uint32_t batch_count = static_cast<uint32_t>(
      std::accumulate(output.shape().begin(), output.shape().end() - 2,
                      size_t{1}, std::multiplies<>{}));

  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(b.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 2);

  encoder_->setBytes(&M, sizeof(M), 3);
  encoder_->setBytes(&K, sizeof(K), 4);
  encoder_->setBytes(&N, sizeof(N), 5);

  uint32_t stride_a0 = a.strides()[a.strides().size() - 2];
  uint32_t stride_a1 = a.strides()[a.strides().size() - 1];
  uint32_t stride_b0 = b.strides()[b.strides().size() - 2];
  uint32_t stride_b1 = b.strides()[b.strides().size() - 1];
  uint32_t stride_c0 = output.strides()[output.strides().size() - 2];
  uint32_t stride_c1 = output.strides()[output.strides().size() - 1];

  const bool aligned = M % 64 == 0 && K % 8 == 0 && N % 64 == 0;
  const bool contiguous = stride_a0 == K && stride_a1 == 1 && stride_b0 == N &&
                          stride_b1 == 1 && stride_c0 == N && stride_c1 == 1;

  if (aligned) {
    encoder_->setComputePipelineState(contiguous ? matmul_contiguous_pipeline_
                                                 : matmul_strided_pipeline_);
  } else {
    encoder_->setComputePipelineState(matmul_generic_pipeline_);
  }

  encoder_->setBytes(&stride_a0, sizeof(stride_a0), 6);
  encoder_->setBytes(&stride_a1, sizeof(stride_a1), 7);
  encoder_->setBytes(&stride_b0, sizeof(stride_b0), 8);
  encoder_->setBytes(&stride_b1, sizeof(stride_b1), 9);
  encoder_->setBytes(&stride_c0, sizeof(stride_c0), 10);
  encoder_->setBytes(&stride_c1, sizeof(stride_c1), 11);

  auto set_array = [this](const auto &values, NS::UInteger index) {
    if (!values.empty()) {
      encoder_->setBytes(values.data(), values.size() * sizeof(uint32_t),
                         index);
    }
  };

  set_array(batch_stride_a, 12);
  set_array(batch_stride_b, 13);
  set_array(batch_stride_c, 14);
  set_array(batch_decode_strides, 15);

  encoder_->setBytes(&nDim, sizeof(nDim), 16);

  if (aligned) {
    encoder_->dispatchThreadgroups(MTL::Size(N / 64, M / 64, batch_count),
                                   MTL::Size(16, 8, 1));
  } else {
    encoder_->dispatchThreads(MTL::Size(N, M, batch_count),
                              MTL::Size(16, 16, 1));
  }
}
void Runtime::sum(const RTensor<Metal, kernel_type> &input,
                  RTensor<Metal, kernel_type> &output) {
  encoder_->setComputePipelineState(sum_pipeline_);
  encoder_->setBuffer(input.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);

  uint32_t count = static_cast<uint32_t>(
      std::accumulate(input.shape().begin(), input.shape().end(), size_t{1},
                      std::multiplies<>{}));

  encoder_->setBytes(&count, sizeof(count), 2);
  encoder_->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void Runtime::sum(const RTensor<Metal, kernel_type> &input, size_t axis,
                  RTensor<Metal, kernel_type> &output) {
  if (axis >= input.shape().size()) {
    throw std::out_of_range("sum axis out of range");
  }

  sum_to_shape(input, output.shape(), output);
}

void Runtime::transpose(const RTensor<Metal, kernel_type> &a,
                        RTensor<Metal, kernel_type> &output, int32_t dim_a,
                        int32_t dim_b) {
  if (a.shape().size() < 2) {
    throw std::invalid_argument("transpose requires rank >= 2");
  }

  const size_t axis_a = rtensor::normalise_dimension(a.shape().size(), dim_a);
  const size_t axis_b = rtensor::normalise_dimension(a.shape().size(), dim_b);
  if (axis_a == axis_b) {
    throw std::invalid_argument("transpose dimensions must be different");
  }
  if (output.shape() != rtensor::infer_transpose(a.shape(), dim_a, dim_b)) {
    throw std::invalid_argument("transpose destination has incorrect shape");
  }

  encoder_->setComputePipelineState(transpose_pipeline_);

  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);

  auto stride_a = a.strides();
  auto stride_out = output.strides();
  std::swap(stride_a[axis_a], stride_a[axis_b]);

  encoder_->setBytes(stride_a.data(), stride_a.size() * sizeof(uint32_t), 2);
  encoder_->setBytes(stride_out.data(), stride_out.size() * sizeof(uint32_t),
                     3);

  uint32_t nDim = static_cast<uint32_t>(output.shape().size());
  encoder_->setBytes(&nDim, sizeof(nDim), 4);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);

  MTL::Size threadgroupSize(256, 1, 1);

  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

void Runtime::split_last_dim(const RTensor<Metal, kernel_type> &a,
                             RTensor<Metal, kernel_type> &output,
                             uint32_t outer, uint32_t inner) {
  if (output.shape() !=
      rtensor::infer_split_last_dim(a.shape(), outer, inner)) {
    throw std::invalid_argument(
        "split_last_dim destination has incorrect shape");
  }
  const auto input = a.view(output.shape());
  add(input, kernel_type{0}, output);
}

void Runtime::merge_last_dims(const RTensor<Metal, kernel_type> &a,
                              RTensor<Metal, kernel_type> &output,
                              uint32_t outer, uint32_t inner) {
  if (output.shape() !=
      rtensor::infer_merge_last_dims(a.shape(), outer, inner)) {
    throw std::invalid_argument(
        "merge_last_dims destination has incorrect shape");
  }
  const auto input = a.view(output.shape());
  add(input, kernel_type{0}, output);
}

void Runtime::reduce_to_shape(const RTensor<Metal, kernel_type> &a,
                              std::span<const uint32_t> output_shape,
                              RTensor<Metal, kernel_type> &output,
                              MTL::ComputePipelineState *pipe) {
  // i want to compare the input dim and output dims and then work out what i
  // need to do to get it there can input dims be converted to output dims
  // 1st pad output.shaoe with ones
  if (output_shape.size() != a.shape().size()) {
    throw std::invalid_argument("cannot reduce to shape if ranks differ");
  }

  const auto &a_shape = a.shape();

  uint32_t shrink_dim = 0;
  uint32_t shrink_size = 0;
  size_t shrink_count = 0;

  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (a_shape[i] != output_shape[i]) {
      if (output_shape[i] != 1) {
        throw std::invalid_argument("invalid reduce_to_shape");
      }

      if (++shrink_count > 1) {
        throw std::invalid_argument(
            "reduce_to_shape currently supports only one shrinking dimension");
      }

      shrink_dim = static_cast<uint32_t>(i);
      shrink_size = a_shape[i];
      shrink_count += 1;
    }
  }

  if (shrink_count == 0) {
    output = a;
    return;
  }

  encoder_->setComputePipelineState(pipe);

  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);

  auto stride_a = a.strides();
  auto stride_out = output.strides();

  encoder_->setBytes(stride_a.data(), stride_a.size() * sizeof(uint32_t), 2);
  encoder_->setBytes(stride_out.data(), stride_out.size() * sizeof(uint32_t),
                     3);

  uint32_t nDim = static_cast<uint32_t>(output.shape().size());
  encoder_->setBytes(&nDim, sizeof(nDim), 4);
  encoder_->setBytes(&shrink_dim, sizeof(shrink_dim), 5);
  encoder_->setBytes(&shrink_size, sizeof(shrink_size), 6);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);

  MTL::Size threadgroupSize(256, 1, 1);

  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

void Runtime::sum_to_shape(const RTensor<Metal, kernel_type> &a,
                           std::span<const uint32_t> output_shape,
                           RTensor<Metal, kernel_type> &output) {
  const auto &a_shape = a.shape();
  if (output_shape.size() > a_shape.size()) {
    throw std::invalid_argument("cannot reduce to a shape with a higher rank");
  }
  if (output.shape().size() != output_shape.size() ||
      !std::equal(output.shape().begin(), output.shape().end(),
                  output_shape.begin())) {
    throw std::invalid_argument("sum_to_shape destination has incorrect shape");
  }

  const size_t rank_offset = a_shape.size() - output_shape.size();
  std::vector<uint32_t> aligned_output_shape(a_shape.size(), 1);
  for (size_t i = 0; i < output_shape.size(); ++i) {
    aligned_output_shape[rank_offset + i] = output_shape[i];
  }

  std::vector<uint32_t> reduced_dims;
  std::vector<uint32_t> reduced_sizes;
  for (size_t i = 0; i < a_shape.size(); ++i) {
    if (a_shape[i] != aligned_output_shape[i]) {
      if (aligned_output_shape[i] != 1) {
        throw std::invalid_argument("invalid reduce_to_shape");
      }
      reduced_dims.push_back(static_cast<uint32_t>(i));
      reduced_sizes.push_back(a_shape[i]);
    }
  }

  if (reduced_dims.empty()) {
    const auto input =
        a.shape().size() == output_shape.size() ? a : a.view(output_shape);

    add(input, kernel_type{0}, output);
    return;
  }

  encoder_->setComputePipelineState(sum_to_shape_pipeline_);

  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);

  auto stride_a = a.strides();
  std::vector<uint32_t> stride_out(aligned_output_shape.size());
  uint32_t stride = 1;
  for (size_t i = aligned_output_shape.size(); i-- > 0;) {
    stride_out[i] = stride;
    stride *= aligned_output_shape[i];
  }

  encoder_->setBytes(stride_a.data(), stride_a.size() * sizeof(uint32_t), 2);
  encoder_->setBytes(stride_out.data(), stride_out.size() * sizeof(uint32_t),
                     3);
  // which dimensions need reducing ?
  encoder_->setBytes(reduced_dims.data(),
                     reduced_dims.size() * sizeof(uint32_t), 4);
  encoder_->setBytes(reduced_sizes.data(),
                     reduced_sizes.size() * sizeof(uint32_t), 5);

  uint32_t nDim = static_cast<uint32_t>(a_shape.size());
  uint32_t num_reduce_dims = static_cast<uint32_t>(reduced_dims.size());
  uint32_t reduced_count =
      std::accumulate(reduced_sizes.begin(), reduced_sizes.end(), uint32_t{1},
                      std::multiplies<>{});
  encoder_->setBytes(&nDim, sizeof(nDim), 6);
  encoder_->setBytes(&num_reduce_dims, sizeof(num_reduce_dims), 7);
  encoder_->setBytes(&reduced_count, sizeof(reduced_count), 8);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);

  MTL::Size threadgroupSize(256, 1, 1);

  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

void Runtime::max_to_shape(const RTensor<Metal, kernel_type> &a,
                           std::span<const uint32_t> output_shape,
                           RTensor<Metal, kernel_type> &output) {
  reduce_to_shape(a, output_shape, output, max_to_shape_pipeline_);
}

void Runtime::softmax(const RTensor<Metal, kernel_type> &x,
                      RTensor<Metal, kernel_type> &output) {
  if (x.shape().empty() || x.shape().back() == 0) {
    throw std::invalid_argument("softmax requires a non-empty final dimension");
  }
  if (output.shape() != x.shape()) {
    throw std::invalid_argument("softmax destination has incorrect shape");
  }

  const uint32_t last_dim = x.shape().back();
  const size_t element_count = std::accumulate(
      x.shape().begin(), x.shape().end(), size_t{1}, std::multiplies<>{});
  const size_t row_count = element_count / last_dim;

  encoder_->setComputePipelineState(softmax_pipeline_);
  encoder_->setBuffer(x.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);
  encoder_->setBytes(&last_dim, sizeof(last_dim), 2);
  encoder_->dispatchThreads(MTL::Size(row_count, 1, 1), MTL::Size(256, 1, 1));
}

void Runtime::cross_entropy(const RTensor<Metal, kernel_type> &logits,
                            const RTensor<Metal, index_type> &targets,
                            RTensor<Metal, kernel_type> &output) {
  if (logits.shape().size() < 2 || logits.shape().back() == 0) {
    throw std::invalid_argument(
        "cross_entropy requires logits with a non-empty final dimension");
  }
  if (targets.shape().size() + 1 != logits.shape().size()) {
    throw std::invalid_argument(
        "cross_entropy target rank must be one less than logits rank");
  }
  for (size_t i = 0; i < targets.shape().size(); ++i) {
    if (targets.shape()[i] != logits.shape()[i]) {
      throw std::invalid_argument(
          "cross_entropy target shape must match logits leading dimensions");
    }
  }
  auto output_shape = targets.shape();
  output_shape.push_back(1);
  if (output.shape() != output_shape) {
    throw std::invalid_argument(
        "cross_entropy destination must match target shape with a trailing "
        "singleton dimension");
  }

  const uint32_t vocab_size = logits.shape().back();
  const size_t row_count =
      std::accumulate(targets.shape().begin(), targets.shape().end(), size_t{1},
                      std::multiplies<>{});

  encoder_->setComputePipelineState(cross_entropy_pipeline_);
  encoder_->setBuffer(logits.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(targets.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 2);
  encoder_->setBytes(&vocab_size, sizeof(vocab_size), 3);
  encoder_->dispatchThreads(MTL::Size(row_count, 1, 1), MTL::Size(256, 1, 1));
}

void Runtime::cross_entropy_derivative(
    const RTensor<Metal, kernel_type> &logits,
    const RTensor<Metal, index_type> &targets,
    RTensor<Metal, kernel_type> &output) {
  if (logits.shape().size() < 2 || logits.shape().back() == 0) {
    throw std::invalid_argument("cross_entropy_derivative requires logits "
                                "with a non-empty final dimension");
  }
  if (targets.shape().size() + 1 != logits.shape().size()) {
    throw std::invalid_argument("cross_entropy_derivative target rank must be "
                                "one less than logits rank");
  }
  for (size_t i = 0; i < targets.shape().size(); ++i) {
    if (targets.shape()[i] != logits.shape()[i]) {
      throw std::invalid_argument(
          "cross_entropy_derivative target shape must match logits leading "
          "dimensions");
    }
  }
  if (output.shape() != logits.shape()) {
    throw std::invalid_argument(
        "cross_entropy_derivative destination must match logits shape");
  }

  const uint32_t vocab_size = logits.shape().back();
  const size_t row_count =
      std::accumulate(targets.shape().begin(), targets.shape().end(), size_t{1},
                      std::multiplies<>{});

  encoder_->setComputePipelineState(cross_entropy_derivative_pipeline_);
  encoder_->setBuffer(logits.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(targets.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 2);
  encoder_->setBytes(&vocab_size, sizeof(vocab_size), 3);
  encoder_->dispatchThreads(MTL::Size(row_count, 1, 1), MTL::Size(256, 1, 1));
}

void Runtime::sqe(const RTensor<Metal, kernel_type> &pred,
                  const RTensor<Metal, kernel_type> &target,
                  RTensor<Metal, kernel_type> &output) {
  sub(pred, target, output);
  hadamard(output, output, output);
}

void Runtime::argmax(const RTensor<Metal, kernel_type> &a,
                     RTensor<Metal, index_type> &output) {
  const size_t element_count = std::accumulate(
      a.shape().begin(), a.shape().end(), size_t{1}, std::multiplies<>{});

  if (element_count == 0) {
    throw std::invalid_argument("argmax requires a non-empty tensor");
  }

  constexpr uint32_t items_per_thread = 16;
  uint32_t current_count = static_cast<uint32_t>(element_count);
  uint32_t first_count =
      (current_count + items_per_thread - 1) / items_per_thread;

  MTL::Buffer *current_values = device_->newBuffer(
      first_count * sizeof(kernel_type), MTL::ResourceStorageModeShared);
  MTL::Buffer *current_indices =
      first_count == 1 ? output.get_storage()->get_buffer()
                       : device_->newBuffer(first_count * sizeof(index_type),
                                            MTL::ResourceStorageModeShared);

  encoder_->setComputePipelineState(argmax_pipeline_);
  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(current_values, 0, 1);
  encoder_->setBuffer(current_indices, 0, 2);
  encoder_->setBytes(&current_count, sizeof(current_count), 3);

  MTL::Size threadgroup_size(256, 1, 1);
  encoder_->dispatchThreads(MTL::Size(first_count, 1, 1), threadgroup_size);

  if (first_count == 1) {
    return;
  }

  current_count = first_count;
  uint32_t scratch_count =
      (current_count + items_per_thread - 1) / items_per_thread;

  MTL::Buffer *scratch_values = device_->newBuffer(
      scratch_count * sizeof(kernel_type), MTL::ResourceStorageModeShared);
  MTL::Buffer *scratch_indices =
      scratch_count == 1
          ? nullptr
          : device_->newBuffer(scratch_count * sizeof(index_type),
                               MTL::ResourceStorageModeShared);

  while (current_count > 1) {
    uint32_t next_count =
        (current_count + items_per_thread - 1) / items_per_thread;
    MTL::Buffer *result_indices =
        next_count == 1 ? output.get_storage()->get_buffer() : scratch_indices;

    encoder_->setComputePipelineState(argmax_reduce_pipeline_);
    encoder_->setBuffer(current_values, 0, 0);
    encoder_->setBuffer(current_indices, 0, 1);
    encoder_->setBuffer(scratch_values, 0, 2);
    encoder_->setBuffer(result_indices, 0, 3);
    encoder_->setBytes(&current_count, sizeof(current_count), 4);
    encoder_->dispatchThreads(MTL::Size(next_count, 1, 1), threadgroup_size);

    if (next_count == 1) {
      break;
    }

    current_count = next_count;
    std::swap(current_values, scratch_values);
    std::swap(current_indices, scratch_indices);
  }
}

void Runtime::embedding_lookup(const RTensor<Metal, kernel_type> &weights,
                               const RTensor<Metal, index_type> &ids,
                               RTensor<Metal, kernel_type> &output) {
  if (weights.shape().size() != 2) {
    throw std::invalid_argument("embedding weights must have rank 2");
  }

  const uint32_t embedding_size = weights.shape().back();
  auto expected_shape = ids.shape();
  expected_shape.push_back(embedding_size);
  if (output.shape() != expected_shape) {
    throw std::invalid_argument("embedding output shape does not match IDs");
  }

  encoder_->setComputePipelineState(embedding_lookup_pipeline_);
  encoder_->setBuffer(weights.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(ids.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 2);
  encoder_->setBytes(&embedding_size, sizeof(embedding_size), 3);

  const size_t output_size =
      std::accumulate(output.shape().begin(), output.shape().end(), size_t{1},
                      std::multiplies<>{});
  encoder_->dispatchThreads(MTL::Size(output_size, 1, 1), MTL::Size(256, 1, 1));
}

void Runtime::gather_rows(const RTensor<Metal, kernel_type> &values,
                          const RTensor<Metal, index_type> &indices,
                          RTensor<Metal, kernel_type> &output) {
  embedding_lookup(values, indices, output);
}

void Runtime::dispatch_scatter_add_rows(
    const RTensor<Metal, kernel_type> &values,
    const RTensor<Metal, index_type> &ids,
    RTensor<Metal, kernel_type> &destination) {
  if (destination.shape().size() != 2) {
    throw std::invalid_argument("scatter destination must have rank 2");
  }

  const uint32_t embedding_size = destination.shape().back();
  auto expected_shape = ids.shape();
  expected_shape.push_back(embedding_size);
  if (values.shape() != expected_shape) {
    throw std::invalid_argument(
        "scatter input shape does not match IDs and embedding size");
  }

  const size_t input_size =
      std::accumulate(values.shape().begin(), values.shape().end(), size_t{1},
                      std::multiplies<>{});
  if (input_size == 0) {
    return;
  }

  encoder_->setComputePipelineState(scatter_add_rows_pipeline_);
  encoder_->setBuffer(values.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(ids.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(destination.get_storage()->get_buffer(), 0, 2);
  encoder_->setBytes(&embedding_size, sizeof(embedding_size), 3);
  encoder_->dispatchThreads(MTL::Size(input_size, 1, 1), MTL::Size(256, 1, 1));
}

void Runtime::scatter_add_rows(const RTensor<Metal, kernel_type> &base,
                               const RTensor<Metal, index_type> &ids,
                               const RTensor<Metal, kernel_type> &values,
                               RTensor<Metal, kernel_type> &destination) {
  if (base.shape() != destination.shape()) {
    throw std::invalid_argument(
        "scatter base and destination shapes must match");
  }

  MTL::Buffer *base_buffer = base.get_storage()->get_buffer();
  MTL::Buffer *destination_buffer = destination.get_storage()->get_buffer();
  if (base_buffer != destination_buffer) {
    const size_t destination_size =
        std::accumulate(destination.shape().begin(), destination.shape().end(),
                        size_t{1}, std::multiplies<>{});

    encoder_->endEncoding();
    encoder_ = nullptr;
    MTL::BlitCommandEncoder *blit = cmdBuff_->blitCommandEncoder();
    blit->copyFromBuffer(base_buffer, 0, destination_buffer, 0,
                         destination_size * sizeof(kernel_type));
    blit->endEncoding();
    encoder_ = cmdBuff_->computeCommandEncoder();
  }

  dispatch_scatter_add_rows(values, ids, destination);
}

void Runtime::scatter_index(const RTensor<Metal, kernel_type> &values,
                            const RTensor<Metal, index_type> &ids,
                            RTensor<Metal, kernel_type> &destination) {
  const size_t destination_size =
      std::accumulate(destination.shape().begin(), destination.shape().end(),
                      size_t{1}, std::multiplies<>{});

  if (destination_size > 0) {
    encoder_->endEncoding();
    encoder_ = nullptr;
    MTL::BlitCommandEncoder *blit = cmdBuff_->blitCommandEncoder();
    blit->fillBuffer(destination.get_storage()->get_buffer(),
                     NS::Range::Make(0, destination_size * sizeof(kernel_type)),
                     0);
    blit->endEncoding();
    encoder_ = cmdBuff_->computeCommandEncoder();
  }

  dispatch_scatter_add_rows(values, ids, destination);
}
void Runtime::RMS(const RTensor<Metal, kernel_type> &values,
                  RTensor<Metal, kernel_type> &destination,
                  RTensor<Metal, kernel_type> &workspace) {
  auto rms_shape = values.shape();
  rms_shape.back() = 1;

  if (destination.shape() != rms_shape) {
    throw std::invalid_argument("RMS destination has incorrect shape");
  }
  if (workspace.shape() != values.shape()) {
    throw std::invalid_argument("RMS workspace has incorrect shape");
  }

  constexpr kernel_type epsilon = 1e-6f;
  const kernel_type hidden_size =
      static_cast<kernel_type>(values.shape().back());
  hadamard(values, values, workspace);

  sum(workspace, workspace.shape().size() - 1, destination);
  divide(destination, hidden_size, destination);
  add(destination, epsilon, destination);
  sqrt(destination, destination);
}

void Runtime::RMS(const RTensor<Metal, kernel_type> &values,
                  RTensor<Metal, kernel_type> &destination) {
  RTensor<Metal, kernel_type> workspace(values.shape());
  RMS(values, destination, workspace);
}

void Runtime::RMSNorm(const RTensor<Metal, kernel_type> &values,
                      RTensor<Metal, kernel_type> &destination) {
  if (destination.shape() != values.shape()) {
    throw std::invalid_argument("RMSNorm destination has incorrect shape");
  }

  auto rms_shape = values.shape();
  rms_shape.back() = 1;
  RTensor<Metal, kernel_type> rms(rms_shape);
  RMS(values, rms, destination);
  divide(values, rms, destination);
}
void Runtime::GELU(const RTensor<Metal, kernel_type> &input,
                   RTensor<Metal, kernel_type> &output) {
  call_unary_op(input, output, gelu_pipeline_);
}

void Runtime::GELU_derivative(const RTensor<Metal, kernel_type> &input,
                              RTensor<Metal, kernel_type> &output) {
  call_unary_op(input, output, gelu_derivative_pipeline_);
}

void Runtime::accumulate_gradient_squared_norm(
    const RTensor<Metal, kernel_type> &gradient,
    RTensor<Metal, kernel_type> &squared_norm) {
  if (squared_norm.numel() != 1) {
    throw std::invalid_argument("gradient norm destination must be a scalar");
  }

  const uint32_t count = static_cast<uint32_t>(gradient.numel());
  constexpr uint32_t threadgroup_size = 256;
  const uint32_t threadgroup_count =
      (count + threadgroup_size - 1) / threadgroup_size;

  encoder_->setComputePipelineState(accumulate_gradient_norm_pipeline_);
  encoder_->setBuffer(gradient.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(squared_norm.get_storage()->get_buffer(), 0, 1);
  encoder_->setBytes(&count, sizeof(count), 2);
  encoder_->dispatchThreadgroups(MTL::Size(threadgroup_count, 1, 1),
                                 MTL::Size(threadgroup_size, 1, 1));
}

void Runtime::clip_gradient(const RTensor<Metal, kernel_type> &gradient,
                            const RTensor<Metal, kernel_type> &squared_norm,
                            kernel_type max_norm,
                            RTensor<Metal, kernel_type> &destination) {
  if (squared_norm.numel() != 1) {
    throw std::invalid_argument("gradient norm input must be a scalar");
  }
  if (gradient.shape() != destination.shape()) {
    throw std::invalid_argument(
        "gradient clipping input and destination shapes must match");
  }
  if (!(max_norm > 0.0f) || !std::isfinite(max_norm)) {
    throw std::invalid_argument(
        "maximum gradient norm must be finite and positive");
  }

  const uint32_t count = static_cast<uint32_t>(gradient.numel());
  encoder_->setComputePipelineState(clip_gradient_pipeline_);
  encoder_->setBuffer(gradient.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(squared_norm.get_storage()->get_buffer(), 0, 1);
  encoder_->setBytes(&max_norm, sizeof(max_norm), 2);
  encoder_->setBuffer(destination.get_storage()->get_buffer(), 0, 3);
  encoder_->setBytes(&count, sizeof(count), 4);
  encoder_->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
}

void Runtime::adamW(const RTensor<Metal, kernel_type> &parameter,
                    const RTensor<Metal, kernel_type> &gradient,
                    RTensor<Metal, kernel_type> &destination,
                    RTensor<Metal, kernel_type> &momentum1,
                    RTensor<Metal, kernel_type> &momentum2,
                    const RTensor<Metal, kernel_type> &beta1,
                    const RTensor<Metal, kernel_type> &beta2,
                    const RTensor<Metal, kernel_type> &decay,
                    const RTensor<Metal, kernel_type> &learning_rate,
                    kernel_type epsilon, uint32_t step) {
  if (gradient.shape() != parameter.shape() ||
      destination.shape() != parameter.shape() ||
      momentum1.shape() != parameter.shape() ||
      momentum2.shape() != parameter.shape()) {
    throw std::invalid_argument("AdamW tensor shapes must match");
  }
  if (beta1.numel() != 1 || beta2.numel() != 1 || decay.numel() != 1 ||
      learning_rate.numel() != 1) {
    throw std::invalid_argument("AdamW parameters must be scalars");
  }
  if (step == 0) {
    throw std::invalid_argument("AdamW step must be greater than zero");
  }

  encoder_->setComputePipelineState(adamw_pipeline_);
  encoder_->setBuffer(parameter.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(gradient.get_storage()->get_buffer(), 0, 1);
  encoder_->setBuffer(destination.get_storage()->get_buffer(), 0, 2);
  encoder_->setBuffer(momentum1.get_storage()->get_buffer(), 0, 3);
  encoder_->setBuffer(momentum2.get_storage()->get_buffer(), 0, 4);
  encoder_->setBuffer(beta1.get_storage()->get_buffer(), 0, 5);
  encoder_->setBuffer(beta2.get_storage()->get_buffer(), 0, 6);
  encoder_->setBuffer(decay.get_storage()->get_buffer(), 0, 7);
  encoder_->setBuffer(learning_rate.get_storage()->get_buffer(), 0, 8);
  encoder_->setBytes(&epsilon, sizeof(epsilon), 9);
  encoder_->setBytes(&step, sizeof(step), 10);

  encoder_->dispatchThreads(MTL::Size(parameter.numel(), 1, 1),
                            MTL::Size(256, 1, 1));
}

void Runtime::copy(const RTensor<Metal, kernel_type> &a,
                   RTensor<Metal, kernel_type> &output) {
  if (a.shape() != output.shape()) {
    throw std::invalid_argument("copy destination has incorrect shape");
  }

  encoder_->setComputePipelineState(copy_pipeline_);
  encoder_->setBuffer(a.get_storage()->get_buffer(), 0, 0);
  encoder_->setBuffer(output.get_storage()->get_buffer(), 0, 1);

  MTL::Size gridSize =
      MTL::Size(std::accumulate(output.shape().begin(), output.shape().end(),
                                size_t{1}, std::multiplies<>{}),
                1, 1);
  MTL::Size threadgroupSize(256, 1, 1);
  encoder_->dispatchThreads(gridSize, threadgroupSize);
}

} // namespace metal
