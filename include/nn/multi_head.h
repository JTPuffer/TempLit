#pragma once

#include "expr/api.h"
#include "nn/layer.h"
#include <cstdint>
#include <random>
#include <utility>

namespace nn {

template <typename T, uint32_t EmbeddingSize, uint32_t Heads>
class MultiHeadAttention {
  static_assert(Heads > 0);
  static_assert(EmbeddingSize % Heads == 0);

  static constexpr uint32_t HeadSize = EmbeddingSize / Heads;

  Layer<T> w_q;
  Layer<T> w_k;
  Layer<T> w_v;
  Layer<T> w_o;

public:
  MultiHeadAttention(std::mt19937 &gen)
      : w_q(EmbeddingSize, EmbeddingSize, gen),
        w_k(EmbeddingSize, EmbeddingSize, gen),
        w_v(EmbeddingSize, EmbeddingSize, gen),
        w_o(EmbeddingSize, EmbeddingSize, gen) {}

  template <expr::EXPR_TYPE Query, expr::EXPR_TYPE Key, expr::EXPR_TYPE Value,
            expr::EXPR_TYPE MASK>
  auto operator()(Query &&Q, Key &&K, Value &&V, MASK &&mask) const {
    auto q = w_q(std::forward<Query>(Q));
    auto k = w_k(std::forward<Key>(K));
    auto v = w_v(std::forward<Value>(V));

    auto re_q = expr::transpose<1, 2>(
        expr::split_last_dim<Heads, HeadSize>(std::move(q)));
    auto re_k = expr::transpose<1, 2>(
        expr::split_last_dim<Heads, HeadSize>(std::move(k)));
    auto re_v = expr::transpose<1, 2>(
        expr::split_last_dim<Heads, HeadSize>(std::move(v)));

    auto scores =
        expr::add(expr::divide(expr::mult(std::move(re_q),
                                          expr::transpose(std::move(re_k))),
                               expr::sqrt(expr::EXPR_CONSTANT<HeadSize>{})),
                  std::forward<MASK>(mask));
    auto attention =
        expr::mult(expr::soft_max(std::move(scores)), std::move(re_v));
    auto ordered = expr::transpose<1, 2>(std::move(attention));
    auto merged = expr::merge_last_dims<Heads, HeadSize>(std::move(ordered));

    return w_o(std::move(merged));
  }
  template <expr::EXPR_TYPE Query, expr::EXPR_TYPE Key, expr::EXPR_TYPE Value>
  auto operator()(Query &&Q, Key &&K, Value &&V) const {
    auto q = w_q(std::forward<Query>(Q));
    auto k = w_k(std::forward<Key>(K));
    auto v = w_v(std::forward<Value>(V));

    auto re_q = expr::transpose<1, 2>(
        expr::split_last_dim<Heads, HeadSize>(std::move(q)));
    auto re_k = expr::transpose<1, 2>(
        expr::split_last_dim<Heads, HeadSize>(std::move(k)));
    auto re_v = expr::transpose<1, 2>(
        expr::split_last_dim<Heads, HeadSize>(std::move(v)));

    auto scores = expr::divide(
        expr::mult(std::move(re_q), expr::transpose(std::move(re_k))),
        expr::sqrt(expr::EXPR_CONSTANT<HeadSize>{}));
    auto attention =
        expr::mult(expr::soft_max(std::move(scores)), std::move(re_v));
    auto ordered = expr::transpose<1, 2>(std::move(attention));
    auto merged = expr::merge_last_dims<Heads, HeadSize>(std::move(ordered));

    return w_o(std::move(merged));
  }
};

} // namespace nn
