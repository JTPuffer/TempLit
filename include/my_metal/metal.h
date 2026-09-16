#pragma once
#include <Metal/Metal.hpp>
#include <cstdint>

namespace metal {

struct Metal {};
MTL::Device *device();

void init();
void shutdown();

using kernel_type = float;
using index_type = uint32_t;
} // namespace metal
