#include "model_utils/Dataset.h"
#include "model_utils/bpe_tokeniser.h"
#include <cstdint>
#include <fstream>
#include <iostream>

constexpr uint32_t eos_token = 256;
constexpr uint32_t vocabulary_size = 2048;

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: train_bpe <text-dataset> <tokeniser>\n";
    return 1;
  }

  std::ifstream text_input(argv[1], std::ios::binary);
  Dataset dataset = Dataset::load_text(text_input);

  auto tokeniser =
      BPETokenizer::train(dataset.get(), vocabulary_size, eos_token);

  std::ofstream tokeniser_output(argv[2]);
  tokeniser.save(tokeniser_output);
}
