#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "my_metal/metal.h"

#include <stdexcept>

namespace metal {
namespace {
MTL::Device *g_device = nullptr;
}

void init() {
  if (g_device) {
    return;
  }

  g_device = MTL::CreateSystemDefaultDevice();
  if (!g_device) {
    throw std::runtime_error("Failed to create Metal device");
  }
}

void shutdown() {
  if (!g_device) {
    return;
  }

  g_device->release();
  g_device = nullptr;
}

MTL::Device *device() { return g_device; }
} // namespace metal
