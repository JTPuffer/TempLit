#include "model_utils/Dataset.h"
#include "model_utils/bpe_tokeniser.h"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: tokenise_dataset <text-dataset> <tokeniser> "
                 "<token-dataset>\n";
    return 1;
  }

  std::ifstream text_input(argv[1], std::ios::binary);
  Dataset dataset = Dataset::load_text(text_input);

  std::ifstream bpe_file(argv[2]);
  auto tokeniser = BPETokenizer::load(bpe_file);

  auto encoded = tokeniser.encode(dataset.get());

  std::ofstream token_output(argv[3], std::ios::binary);
  Dataset(encoded).save(token_output);

  std::cout << "wrote " << encoded.size() << " tokens with vocabulary size "
            << tokeniser.vocabulary_size() << '\n';
}
