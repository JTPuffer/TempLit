#pragma once

#include "expr/api.h"
#include "nn/attention.h"
#include "nn/decoder_block.h"
#include "nn/embedding.h"
#include "nn/layer.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nn {

template <typename Mat, uint32_t VocabSize, uint32_t MaxSequenceSize,
          uint32_t EmbeddingSize, uint32_t Heads>
class GPT {
  using IndexMat = typename Mat::template rebind<uint32_t>;
  using Indices = expr::tensor::NoGradTensor<IndexMat>;

  static IndexMat make_positions(uint32_t batch_size,
                                 uint32_t sequence_size) {
    if (batch_size == 0 || sequence_size == 0 ||
        sequence_size > MaxSequenceSize) {
      throw std::invalid_argument("invalid GPT input dimensions");
    }

    IndexMat values({batch_size, sequence_size});
    std::vector<uint32_t> tokens(batch_size * sequence_size);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      for (uint32_t position = 0; position < sequence_size; ++position) {
        tokens[batch * sequence_size + position] = position;
      }
    }

    values.set(tokens);
    return values;
  }

  Indices positions;
  expr::tensor::NoGradTensor<Mat> mask;
  Embedding<Mat, VocabSize, EmbeddingSize> token_embedding;
  Embedding<Mat, MaxSequenceSize, EmbeddingSize> position_embedding;
  DecoderBlock<Mat, EmbeddingSize, Heads> block_1;
  DecoderBlock<Mat, EmbeddingSize, Heads> block_2;
  DecoderBlock<Mat, EmbeddingSize, Heads> block_3;
  DecoderBlock<Mat, EmbeddingSize, Heads> block_4;
  Layer<Mat> projection_layer;

public:
  GPT(uint32_t batch_size, uint32_t sequence_size, std::mt19937 &gen)
      : positions(make_positions(batch_size, sequence_size)),
        mask(causal_mask<typename Mat::device_type, typename Mat::value_type>(
            sequence_size, sequence_size)),
        token_embedding(gen), position_embedding(gen), block_1(gen),
        block_2(gen), block_3(gen), block_4(gen),
        projection_layer(EmbeddingSize, VocabSize, gen) {}

  auto operator()(Indices input) const {
    auto hidden = expr::add(token_embedding(std::move(input)),
                            position_embedding(positions));

    return projection_layer(expr::rms_norm(block_4(
        block_3(block_2(block_1(std::move(hidden), mask), mask), mask), mask)));
  }
};

} // namespace nn
