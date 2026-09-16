#pragma once

#include "expr/api.h"
#include "nn/layer.h"
#include "nn/multi_head.h"
#include <cstdint>
#include <random>
#include <utility>

namespace nn {

template <typename T, uint32_t EmbeddingSize, uint32_t Heads,
          uint32_t FFNSize = 4 * EmbeddingSize>
class DecoderBlock {

  MultiHeadAttention<T, EmbeddingSize, Heads> self_attention;

  Layer<T> mlp_1;
  Layer<T> mlp_2;

public:
  DecoderBlock(std::mt19937 &gen)
      : self_attention(gen), mlp_1(EmbeddingSize, FFNSize, gen),
        mlp_2(FFNSize, EmbeddingSize, gen) {}

  template <expr::EXPR_TYPE IN, expr::EXPR_TYPE MASK>
  auto operator()(IN &&input, MASK &&mask) const {
    return expr::let(std::forward<IN>(input), [&](auto &block_input) {
      auto norm1 = expr::rms_norm(block_input);

      auto attention = expr::let(norm1, [&](auto &norm) {
        return self_attention(norm, norm, norm, std::forward<MASK>(mask));
      });

      auto residual_t = expr::add(block_input, std::move(attention));

      return expr::let(residual_t, [&](auto &residual) {
        auto norm2 = expr::rms_norm(residual);
        auto ffn = mlp_2(expr::gelu(mlp_1(std::move(norm2))));
        return expr::add(residual, std::move(ffn));
      });
    });
  }
};

} // namespace nn
