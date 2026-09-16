#include "expr/api.h"
#include "my_metal/runtime.h"
#include "my_metal/storage.h"
#include "nn/attention.h"
#include "nn/decoder_block.h"
#include "nn/multi_head.h"
#include "optimiser/sgd.h"
#include "passes/compile.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Mat = RTensor<metal::Metal, metal::kernel_type>;
using SGD = Optimiser::SGD<Mat>;

template <typename T> class GradientRecorder {
public:
  struct Shared {};

  inline static std::vector<std::vector<typename T::value_type>> gradients;

  template <typename Run>
  void update(Run &, const Shared &, const T &, const T &gradient, T &) {
    auto &values = gradients.emplace_back(gradient.numel());
    gradient.read(std::span(values));
  }

  static void clear() { gradients.clear(); }
};

constexpr size_t batch_size = 1;
constexpr size_t sequence_size = 4;
constexpr size_t embedding_size = 128;
constexpr size_t sample_size = batch_size * sequence_size * embedding_size;

std::array<size_t, sequence_size> fill_sample(
    Mat &sample, std::mt19937::result_type seed,
    const std::vector<float> &empty_sample) {
  sample.set(empty_sample);

  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> distribution(0, embedding_size - 1);
  std::array<size_t, sequence_size> expected_indices{};

  for (size_t row = 0; row < sequence_size; ++row) {
    expected_indices[row] = distribution(rng);
    sample.set(1.0f, 0, row, expected_indices[row]);
  }

  return expected_indices;
}

void test_attention_learns_to_favour_input_tokens() {
  metal::Runtime runtime;
  Mat input_value({batch_size, sequence_size, embedding_size});
  expr::tensor::Tensor input(std::move(input_value));
  expr::tensor::Tensor mask(
      nn::causal_mask<metal::Metal, metal::kernel_type>(sequence_size,
                                                        sequence_size));

  std::mt19937 gen(1);
  nn::MultiHeadAttention<Mat, embedding_size, 1> attention_layer(gen);
  auto attention = attention_layer(input, input, input, mask);
  expr::tensor::Tensor target(
      Mat({batch_size, sequence_size, embedding_size}));
  auto loss = expr::loss::sqe(attention, target);

  Mat root_grad({batch_size, sequence_size, embedding_size});
  root_grad.set(std::vector<float>(sample_size, 1.0f));

  auto optimiser_shared = SGD::generate_params(0.001f);
  auto compiled_loss =
      passes::compile<Mat, SGD>(loss, runtime, optimiser_shared);
  auto compiled_attention =
      passes::compile<Mat, SGD>(attention, runtime, optimiser_shared);
  compiled_loss.prepare();
  compiled_attention.prepare();
  const std::vector<float> empty_sample(sample_size, 0.0f);
  for (size_t sample = 0; sample < 10000; ++sample) {
    Mat training_sample({batch_size, sequence_size, embedding_size});
    fill_sample(training_sample,
                static_cast<std::mt19937::result_type>(sample), empty_sample);

    input.set_value(training_sample);
    target.set_value(training_sample);
    compiled_loss.zero_grad();
    (void)compiled_loss.forward();
    compiled_loss.backward(root_grad);
    compiled_loss.step();
  }

  for (size_t sample = 0; sample < 2; ++sample) {
    Mat test_sample({batch_size, sequence_size, embedding_size});
    const auto expected = fill_sample(
        test_sample, static_cast<std::mt19937::result_type>(10000 + sample),
        empty_sample);

    input.set_value(test_sample);
    Mat output = compiled_attention.forward();

    for (size_t row = 0; row < sequence_size; ++row) {
      float row_total = 0.0f;
      for (size_t column = 0; column < embedding_size; ++column) {
        row_total += output.get(0, row, column);
      }

      const float row_average =
          row_total / static_cast<float>(embedding_size);
      const float expected_value = output.get(0, row, expected[row]);
      if (!std::isfinite(expected_value) || expected_value <= row_average) {
        throw std::runtime_error(
            "attention output did not favour the input token");
      }
    }
  }
}

void test_decoder_block_let_gradients_are_distinct() {
  constexpr uint32_t decoder_batch_size = 1;
  constexpr uint32_t decoder_sequence_size = 3;
  constexpr uint32_t decoder_embedding_size = 8;
  constexpr size_t decoder_sample_size =
      decoder_batch_size * decoder_sequence_size * decoder_embedding_size;

  using Recorder = GradientRecorder<Mat>;

  metal::Runtime runtime;
  expr::tensor::NoGradTensor<Mat> input(
      Mat({decoder_batch_size, decoder_sequence_size,
           decoder_embedding_size}));
  expr::tensor::NoGradTensor<Mat> mask(
      nn::causal_mask<metal::Metal, metal::kernel_type>(
          decoder_sequence_size, decoder_sequence_size));

  std::mt19937 gen(7);
  nn::DecoderBlock<Mat, decoder_embedding_size, 2> block(gen);
  auto output = block(input, mask);
  auto compiled =
      passes::compile<Mat, Recorder>(output, runtime, Recorder::Shared{});
  compiled.prepare();

  size_t identical_gradient_count = 0;
  for (size_t sample = 0; sample < 3; ++sample) {
    std::vector<float> input_values(decoder_sample_size);
    std::vector<float> root_gradient_values(decoder_sample_size);
    for (size_t index = 0; index < decoder_sample_size; ++index) {
      input_values[index] =
          static_cast<float>((sample + 1) * (index + 1)) / 31.0f;
      root_gradient_values[index] =
          static_cast<float>((sample + 2) * (index + 3)) / 29.0f;
    }

    Mat input_value(
        {decoder_batch_size, decoder_sequence_size, decoder_embedding_size});
    input_value.set(input_values);
    input.set_value(std::move(input_value));

    Mat root_gradient(
        {decoder_batch_size, decoder_sequence_size, decoder_embedding_size});
    root_gradient.set(root_gradient_values);

    compiled.zero_grad();
    (void)compiled.forward();
    compiled.backward(root_gradient);

    Recorder::clear();
    compiled.step();

    if (Recorder::gradients.size() != 12) {
      throw std::runtime_error(
          "decoder block did not produce all parameter gradients");
    }

    const auto &mlp_2_bias_gradient = Recorder::gradients[3];
    const auto &attention_output_bias_gradient = Recorder::gradients[11];
    if (mlp_2_bias_gradient == attention_output_bias_gradient) {
      ++identical_gradient_count;
    }
  }

  if (identical_gradient_count != 0) {
    throw std::runtime_error(
        "decoder block produced identical mlp_2 and attention output bias "
        "gradients for " +
        std::to_string(identical_gradient_count) + " of 3 inputs");
  }
}

} // namespace

int main() {
  metal::init();
  test_decoder_block_let_gradients_are_distinct();
  test_attention_learns_to_favour_input_tokens();
  metal::shutdown();
}
