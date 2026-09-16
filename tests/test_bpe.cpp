#include "model_utils/bpe_tokeniser.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

constexpr uint32_t EOS = 0;

uint32_t vocabulary_size_for(std::span<const uint32_t> tokens,
                             uint32_t merge_count) {
  return *std::max_element(tokens.begin(), tokens.end()) + 1 + merge_count;
}

void print_tokens(const char *name, const std::vector<uint32_t> &tokens) {
  std::cout << name << ": ";
  for (auto x : tokens)
    std::cout << x << " ";
  std::cout << "\n";
}

void test_round_trip(const std::string &text, uint32_t merge_count) {
  std::vector<uint32_t> tokens(text.begin(), text.end());
  tokens.push_back(EOS);

  auto lan =
      BPETokenizer::train(tokens, vocabulary_size_for(tokens, merge_count), EOS);

  auto encoded = lan.encode(tokens);
  auto decoded = lan.decode(encoded);

  print_tokens("encoded", encoded);
  print_tokens("decoded", decoded);

  assert(decoded == tokens);
}

void test_aaa() {
  std::vector<uint32_t> tokens = {'a', 'a', 'a', EOS};

  auto lan = BPETokenizer::train(tokens, vocabulary_size_for(tokens, 3), EOS);

  auto encoded = lan.encode(tokens);
  auto decoded = lan.decode(encoded);

  print_tokens("aaa encoded", encoded);
  print_tokens("aaa decoded", decoded);

  assert(decoded == tokens);
}

void test_aa() {
  std::vector<uint32_t> tokens = {'a', 'a', EOS};

  auto lan = BPETokenizer::train(tokens, vocabulary_size_for(tokens, 1), EOS);

  auto encoded = lan.encode(tokens);
  auto decoded = lan.decode(encoded);

  print_tokens("aa encoded", encoded);

  assert(decoded == tokens);

  // "aa" should merge into one token plus EOS.
  assert(encoded.size() == 2);
  assert(encoded.back() == EOS);
}

void test_no_merge() {
  std::vector<uint32_t> tokens = {'a', 'b', 'c', EOS};

  auto lan = BPETokenizer::train(tokens, vocabulary_size_for(tokens, 0), EOS);

  auto encoded = lan.encode(tokens);
  auto decoded = lan.decode(encoded);

  assert(encoded == tokens);
  assert(decoded == tokens);
}

void test_single_character() {
  std::vector<uint32_t> tokens = {'a', EOS};

  auto lan = BPETokenizer::train(tokens, vocabulary_size_for(tokens, 10), EOS);

  auto encoded = lan.encode(tokens);
  auto decoded = lan.decode(encoded);

  assert(encoded == tokens);
  assert(decoded == tokens);
}

void test_repeated_pattern() { test_round_trip("abababab", 8); }

void test_repeated_word() { test_round_trip("the the the the", 16); }

void test_spaces() { test_round_trip("a a a a", 8); }

void test_sentence() {
  test_round_trip("the short mouse ran to the hills", 16);
}
void test_save_load() {
  std::string text = "the short mouse ran to the hills";

  std::vector<uint32_t> tokens(text.begin(), text.end());
  tokens.push_back(EOS);

  auto tokenizer =
      BPETokenizer::train(tokens, vocabulary_size_for(tokens, 16), EOS);

  std::stringstream buffer;
  tokenizer.save(buffer);

  auto loaded = BPETokenizer::load(buffer);
  assert(loaded.eos_token() == EOS);
  assert(loaded.vocabulary_size() == tokenizer.vocabulary_size());
  auto encoded_original = tokenizer.encode(tokens);
  auto encoded_loaded = loaded.encode(tokens);

  assert(encoded_original == encoded_loaded);
  auto decoded = loaded.decode(encoded_loaded);
  assert(decoded == tokens);

  std::cout << "save/load test passed\n";
}

int main() {
  test_aa();
  test_aaa();
  test_no_merge();
  test_single_character();
  test_repeated_pattern();
  test_repeated_word();
  test_spaces();
  test_sentence();
  test_save_load();

  std::cout << "all BPE tests passed\n";
}
