#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace inference {

uint32_t sample_top_k(const auto &logits, uint32_t position, uint32_t top_k,
                      float temperature, std::mt19937 &rng) {
  const uint32_t vocabulary_size = logits.shape().back();
  top_k = std::clamp(top_k, 1u, vocabulary_size);

  std::vector<std::pair<float, uint32_t>> candidates;
  candidates.reserve(vocabulary_size);

  for (uint32_t token = 0; token < vocabulary_size; ++token) {
    candidates.emplace_back(logits.get(0, position, token), token);
  }

  std::partial_sort(
      candidates.begin(), candidates.begin() + top_k, candidates.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  candidates.resize(top_k);

  const float maximum = candidates.front().first;
  std::vector<float> weights;
  weights.reserve(top_k);

  for (const auto &[logit, token] : candidates) {
    weights.push_back(std::exp((logit - maximum) / temperature));
  }

  std::discrete_distribution<size_t> distribution(weights.begin(),
                                                  weights.end());
  return candidates[distribution(rng)].second;
}

template <typename CompiledModel, typename Input, typename Emit>
void autoregression(CompiledModel &compiled_model, Input &input,
                    std::span<const uint32_t> prompt, uint32_t eos_token,
                    std::mt19937 rng, uint32_t top_k, float temperature,
                    uint32_t minimum_length, Emit &&emit) {
  const auto &input_shape = input.get_value().shape();

  if (input_shape.size() != 2 || input_shape[0] != 1) {
    throw std::invalid_argument(
        "autoregression currently requires batch size one");
  }

  const uint32_t context_length = input_shape[1];

  if (prompt.empty() || prompt.size() > context_length) {
    throw std::invalid_argument("invalid prompt length");
  }

  std::vector<uint32_t> context(context_length, eos_token);
  std::copy(prompt.begin(), prompt.end(), context.begin());

  uint32_t valid_length = static_cast<uint32_t>(prompt.size());

  minimum_length = std::min(minimum_length, context_length);

  for (uint32_t generated = 0; generated < context_length; ++generated) {
    input.get_value().set(context);
    const auto logits = compiled_model.forward();

    if (logits.shape().size() != 3) {
      throw std::invalid_argument("model output must contain token logits");
    }

    const uint32_t prediction_position = valid_length - 1;

    uint32_t best_token;
    do {
      best_token =
          sample_top_k(logits, prediction_position, top_k, temperature, rng);
    } while (best_token == eos_token && generated < minimum_length);

    emit(best_token);

    if (best_token == eos_token) {
      break;
    }

    if (valid_length < context_length) {
      context[valid_length++] = best_token;
    } else {
      std::move(context.begin() + 1, context.end(), context.begin());
      context.back() = best_token;
    }
  }
}

template <typename Tokeniser>
void print_tokens_inclusive(const Tokeniser &tokeniser,
                            std::span<uint32_t> tokens) {
  const auto decoded_prompt = tokeniser.decode(tokens);
  for (const uint32_t token : decoded_prompt) {
    if (token == tokeniser.eos_token()) {
      std::cout << "<EOS>";
    } else {
      std::cout.put(static_cast<char>(token));
    }
  }
}
template <typename Tokeniser>
void print_tokens(const Tokeniser &tokeniser, std::span<uint32_t> tokens) {
  const auto decoded_prompt = tokeniser.decode(tokens);
  for (const uint32_t token : decoded_prompt) {
    if (token == tokeniser.eos_token()) {
      break;
    } else {
      std::cout.put(static_cast<char>(token));
    }
  }
}
} // namespace inference
