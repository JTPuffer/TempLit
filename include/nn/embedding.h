#pragma once

#include "expr/api.h"
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace nn {

template <typename T, uint32_t NumEmbeddings, uint32_t EmbeddingSize>
class Embedding {
  expr::tensor::GradTensor<T> weights{T({NumEmbeddings, EmbeddingSize})};
  using IndexRTensor = typename T::template rebind<uint32_t>;
  using Ids = expr::tensor::NoGradTensor<IndexRTensor>;

public:
  Embedding(std::mt19937 &gen) {
    using value_type = typename T::value_type;

    // should probably make that some kinda param
    // create 2 ranges that have been randomly created and then pass it to T
    value_type xavier_scale = std::sqrt(
        value_type{2} / static_cast<value_type>(NumEmbeddings + EmbeddingSize));
    std::normal_distribution<value_type> dis(value_type{0}, xavier_scale);
    std::vector<value_type> buff(static_cast<size_t>(NumEmbeddings) *
                                 EmbeddingSize);

    for (auto &ele : buff) {
      ele = dis(gen);
    }
    weights.get_value().set(buff);
  }
  auto operator()(Ids ids) const {
    return expr::embedding_lookup(weights, std::move(ids));
  }
};

} // namespace nn
