#include "model_utils/archive.h"
#include "model_utils/bpe_tokeniser.h"
#include "my_metal/runtime.h"
#include "my_metal/storage.h"
#include "nn/GPT.h"
#include "optimiser/AdamW.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <inference.h>
#include <passes/compile.h>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
constexpr uint32_t batch_size = 1;
constexpr uint32_t sequence_size = 192;
constexpr uint32_t vocab_size = 2048;
constexpr uint32_t embedding_size = 384;
constexpr uint32_t heads = 6;

using Mat = RTensor<metal::Metal, metal::kernel_type>;
using SGD = Optimiser::AdamW<Mat>;

int main(int argc, char **argv) {
  const std::string prompt = argc > 1 ? argv[1] : "Once upon a time";
  const std::string tokeniser_path =
      argc > 2 ? argv[2] : "../dataset/tokenise.bpe";
  const std::string model_path = argc > 3 ? argv[3] : "tiny_stories_model.bin";
  const uint32_t top_k =
      argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 40;
  const float temperature = argc > 5 ? std::stof(argv[5]) : 0.6f;
  const uint32_t minimum_length =
      argc > 6 ? static_cast<uint32_t>(std::stoul(argv[6])) : 64;
  const uint32_t seed =
      argc > 7 ? static_cast<uint32_t>(std::stoul(argv[7])) : 1;

  metal::init();

  metal::Runtime runtime;

  std::ifstream tokeniser_input(tokeniser_path);
  auto tokeniser = BPETokenizer::load(tokeniser_input);
  if (tokeniser.vocabulary_size() != vocab_size) {
    throw std::runtime_error("tokeniser vocabulary does not match the model");
  }

  std::ifstream checkpoint(model_path, std::ios::binary);
  Archive archive;
  archive.restore(checkpoint);

  // creating expr graph/ model ////////////////////
  using IndexMat = typename Mat::template rebind<uint32_t>;

  IndexMat input_values({batch_size, sequence_size});

  expr::tensor::Tensor input(std::move(input_values));

  std::mt19937 gen(1);
  nn::GPT<Mat, vocab_size, sequence_size, embedding_size, heads> model(
      batch_size, sequence_size, gen);
  auto graph = model(input);

  auto optimiser_shared = SGD::generate_params(0.0001f);
  auto compiled_graph =
      passes::compile<Mat, SGD>(graph, runtime, optimiser_shared);

  compiled_graph.prepare();
  compiled_graph.load(archive);

  /////// inference /////////////////
  std::vector<uint32_t> u_prompt(prompt.begin(), prompt.end());
  std::vector<uint32_t> en_prompt = tokeniser.encode(u_prompt);

  std::mt19937 rng(seed);

  inference::autoregression(
      compiled_graph, input, en_prompt, tokeniser.eos_token(), rng, top_k,
      temperature, minimum_length, [&](uint32_t token) {
        if (token == tokeniser.eos_token()) {
          return;
        }

        const std::array<uint32_t, 1> encoded{token};
        const std::vector<uint32_t> decoded = tokeniser.decode(encoded);

        std::cout << std::string(decoded.begin(), decoded.end());

        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      });

  metal::shutdown();
}
