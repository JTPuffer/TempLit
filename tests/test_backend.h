#pragma once

#if defined(GRAPH_TEST_BACKEND_METAL)

#include "metal_runtime_adapter.h"

namespace test_backend {
using metal_test_backend::Device;
using metal_test_backend::kernel_type;
using metal_test_backend::Mat;
using metal_test_backend::Runtime;

using metal_test_backend::add;
using metal_test_backend::argmax;
using metal_test_backend::divide;
using metal_test_backend::hadamard;
using metal_test_backend::init;
using metal_test_backend::mult;
using metal_test_backend::shutdown;
using metal_test_backend::sigmoid;
using metal_test_backend::softmax;
using metal_test_backend::sqrt;
using metal_test_backend::sub;
using metal_test_backend::sum;
using metal_test_backend::sum_to_shape;
using metal_test_backend::transpose;
} // namespace test_backend

#elif defined(GRAPH_TEST_BACKEND_CPU)

#include "CPU/runtime.h"

namespace test_backend {
using Device = cpu::CPU;
using kernel_type = cpu::kernel_type;
using Mat = RTensor<Device, kernel_type>;
using Runtime = cpu::Runtime;

using cpu::add;
using cpu::argmax;
using cpu::divide;
using cpu::hadamard;
using cpu::mult;
using cpu::sigmoid;
using cpu::softmax;
using cpu::sqrt;
using cpu::sub;
using cpu::sum;
using cpu::sum_to_shape;
using cpu::transpose;

inline void init() {}
inline void shutdown() {}
} // namespace test_backend

#else
#error "Define GRAPH_TEST_BACKEND_CPU or GRAPH_TEST_BACKEND_METAL"
#endif

using Mat = test_backend::Mat;
using test_backend::kernel_type;
