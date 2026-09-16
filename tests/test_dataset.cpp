#include "model_utils/Dataset.h"
#include "model_utils/loader.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ios>
#include <sstream>
#include <vector>

int main() {
  std::istringstream text_input("abc\n\ndef\n\nghi");

  Dataset dataset = Dataset::load_text(text_input, 0.5);
  const auto training = dataset.training_split();
  const auto validation = dataset.validation_split();

  assert(!training.empty());
  assert(!validation.empty());
  assert(training.back() == 256);
  assert(validation.front() == static_cast<uint32_t>('g'));
  assert(training.data() + training.size() == validation.data());

  Loader loader(validation);
  const auto &batch = loader.get_batch(1, 2);
  assert(batch.inputs.size() == 2);
  assert(batch.targets.size() == 2);
  assert(batch.inputs[0] == static_cast<uint32_t>('g'));
  assert(batch.inputs[1] == static_cast<uint32_t>('h'));
  assert(batch.targets[0] == static_cast<uint32_t>('h'));
  assert(batch.targets[1] == static_cast<uint32_t>('i'));

  const std::vector<uint32_t> tokens = {1, 2, 256, 3, 4};
  Dataset span_dataset(tokens, 0.4);
  assert(span_dataset.get().size() == tokens.size());
  assert(span_dataset.training_split().back() == 256);
  assert(span_dataset.validation_split().front() == 3);

  std::stringstream token_stream(std::ios::in | std::ios::out |
                                 std::ios::binary);
  span_dataset.save(token_stream);
  token_stream.seekg(0);

  Dataset token_dataset = Dataset::load_tokens(token_stream, 0.4);
  assert(std::equal(token_dataset.get().begin(), token_dataset.get().end(),
                    tokens.begin(), tokens.end()));
  assert(token_dataset.training_split().back() == 256);
  assert(token_dataset.validation_split().front() == 3);
}
