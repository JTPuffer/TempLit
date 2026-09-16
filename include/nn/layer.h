#pragma once

#include "expr/api.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace nn {

template <typename T> class Layer {
public:
  Layer(uint32_t input, uint32_t output, std::mt19937 &gen)
      : weight(T({input, output})), bias(T::from_zeros({1, output})) {
    using value_type = typename T::value_type;

    // should probably make that some kinda param
    // create 2 ranges that have been randomly created and then pass it to T
    value_type xavier_scale =
        std::sqrt(value_type{2} / static_cast<value_type>(input + output));
    std::normal_distribution<value_type> dis(value_type{0}, xavier_scale);
    std::vector<value_type> buff(static_cast<size_t>(input) * output);

    for (auto &ele : buff) {
      ele = dis(gen);
    }
    weight.get_value().set(buff);
  }

  expr::tensor::GradTensor<T> weight;
  expr::tensor::GradTensor<T> bias;

  template <expr::EXPR_TYPE Input> auto operator()(Input &&input) const {
    // this is so sexy
    // can reuse temparary input value and use it as output no need for aextra
    // mem
    return linear(std::forward<Input>(input), weight, bias);
  }
};

} // namespace nn
