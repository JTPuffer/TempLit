#pragma once
#include "passes/get_at_path.h"

namespace passes {

template <typename Graph, typename Context>
void bind(Graph &graph, Context &ctx) {
  ctx.for_each([&]<typename Path>(auto &destination) {
    destination = get_at_path<Path>(graph).get_value();
  });
}

} // namespace passes
