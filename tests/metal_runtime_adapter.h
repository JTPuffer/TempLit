#pragma once

#include "RTensor/infer_shape.h"
#include "my_metal/runtime.h"
#include "my_metal/storage.h"

#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace metal_test_backend {
using Device = metal::Metal;
using kernel_type = metal::kernel_type;
using Mat = RTensor<Device, kernel_type>;
using Runtime = metal::Runtime;

inline std::unique_ptr<Runtime> &runtime_storage() {
  static std::unique_ptr<Runtime> runtime;
  return runtime;
}

inline Runtime &runtime() {
  if (!runtime_storage()) {
    throw std::logic_error("Metal test backend is not initialised");
  }
  return *runtime_storage();
}

inline void init() {
  metal::init();
  runtime_storage() = std::make_unique<Runtime>();
}

inline void shutdown() {
  runtime_storage().reset();
  metal::shutdown();
}

template <typename T, typename Function>
RTensor<Device, T> evaluate(std::span<const uint32_t> shape,
                            Function &&function) {
  RTensor<Device, T> output(shape);
  Runtime &active_runtime = runtime();
  active_runtime.start();
  try {
    std::forward<Function>(function)(active_runtime, output);
    active_runtime.submit();
    active_runtime.wait();
  } catch (...) {
    active_runtime.abort();
    throw;
  }
  return output;
}

inline Mat add(const Mat &a, const Mat &b) {
  const auto shape = rtensor::infer_add(a.shape(), b.shape());
  return evaluate<kernel_type>(shape,
                               [&](Runtime &run, Mat &out) { run.add(a, b, out); });
}

inline Mat sub(const Mat &a, const Mat &b) {
  const auto shape = rtensor::infer_sub(a.shape(), b.shape());
  return evaluate<kernel_type>(shape,
                               [&](Runtime &run, Mat &out) { run.sub(a, b, out); });
}

inline Mat mult(const Mat &a, const Mat &b) {
  const auto shape = rtensor::infer_mult(a.shape(), b.shape());
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.mult(a, b, out);
  });
}

inline Mat hadamard(const Mat &a, const Mat &b) {
  const auto shape = rtensor::infer_hadamard(a.shape(), b.shape());
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.hadamard(a, b, out);
  });
}

inline Mat divide(const Mat &a, const Mat &b) {
  const auto shape = rtensor::infer_divide(a.shape(), b.shape());
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.divide(a, b, out);
  });
}

inline Mat sigmoid(const Mat &value) {
  return evaluate<kernel_type>(value.shape(), [&](Runtime &run, Mat &out) {
    run.sigmoid(value, out);
  });
}

inline Mat sqrt(const Mat &value) {
  return evaluate<kernel_type>(value.shape(), [&](Runtime &run, Mat &out) {
    run.sqrt(value, out);
  });
}

inline Mat softmax(const Mat &value) {
  return evaluate<kernel_type>(value.shape(), [&](Runtime &run, Mat &out) {
    run.softmax(value, out);
  });
}

inline Mat sum(const Mat &value) {
  const rtensor::shape_type shape{1};
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.sum(value, out);
  });
}

inline Mat sum(const Mat &value, size_t axis) {
  if (axis >= value.shape().size()) {
    throw std::out_of_range("sum axis out of range");
  }
  auto shape = value.shape();
  shape[axis] = 1;
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.sum(value, axis, out);
  });
}

inline Mat transpose(const Mat &value) {
  const auto shape = rtensor::infer_transpose(value.shape());
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.transpose(value, out);
  });
}

inline Mat sum_to_shape(const Mat &value,
                        std::span<const uint32_t> shape) {
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.sum_to_shape(value, shape, out);
  });
}

inline Mat max_to_shape(const Mat &value,
                        std::span<const uint32_t> shape) {
  return evaluate<kernel_type>(shape, [&](Runtime &run, Mat &out) {
    run.max_to_shape(value, shape, out);
  });
}

inline Mat argmax(const Mat &value) {
  const rtensor::shape_type shape{1, 1};
  auto index = evaluate<metal::index_type>(
      shape, [&](Runtime &run, RTensor<Device, metal::index_type> &out) {
        run.argmax(value, out);
      });
  return Mat(static_cast<kernel_type>(index.get(0, 0)));
}
} // namespace metal_test_backend
