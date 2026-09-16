#include "model_utils/bpe_tokeniser.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <queue>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef NDEBUG

#define BPE_DEBUG(x)                                                           \
  do {                                                                         \
    std::cerr << "[BPE DEBUG] " << x << '\n';                                  \
  } while (0)
#else
#define BPE_DEBUG(x)                                                           \
  do {                                                                         \
  } while (0)
#endif

namespace {

struct TrainingCorpus {
  std::vector<uint32_t> token;
  std::vector<uint32_t> previous;
  std::vector<uint32_t> next;
};
struct Candidate {
  uint32_t rank;
  uint32_t index;

  bool operator>(const Candidate &other) const {
    if (rank != other.rank)
      return rank > other.rank;
    return index > other.index;
  }
};
} // namespace
BPETokenizer BPETokenizer::train(std::span<const uint32_t> tokens,
                                 uint32_t vocabulary_size, uint32_t eos_token) {

  if (tokens.empty()) {
    throw std::invalid_argument("cannot train a tokenizer on an empty corpus");
  }

  std::unordered_map<Pair, uint64_t, PairHash> counts;
  std::unordered_map<Pair, std::vector<uint32_t>, PairHash> occurrences;

  TrainingCorpus corpus;
  corpus.token.assign(tokens.begin(), tokens.end());
  corpus.previous.resize(tokens.size());
  corpus.next.resize(tokens.size());

  for (uint32_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == eos_token) {
      corpus.previous[i] = end;
      corpus.next[i] = end;
      continue;
    }

    if (i == 0 || tokens[i - 1] == eos_token)
      corpus.previous[i] = end;
    else
      corpus.previous[i] = i - 1;

    if (i + 1 == tokens.size() || tokens[i + 1] == eos_token)
      corpus.next[i] = end;
    else
      corpus.next[i] = i + 1;
  }

  for (uint32_t i = 0; i + 1 < tokens.size(); ++i) {
    if (tokens[i] == eos_token || tokens[i + 1] == eos_token)
      continue;

    Pair pair{tokens[i], tokens[i + 1]};

    ++counts[pair];
    occurrences[pair].push_back(i);
  }

  uint32_t new_token = *std::max_element(tokens.begin(), tokens.end()) + 1;
  std::unordered_map<Pair, uint32_t, PairHash> merges;

  while (new_token < vocabulary_size && !counts.empty()) {
    std::pair<Token, Token> best_pair;
    uint64_t best_count = 0;

    for (const auto &[pair, count] : counts) {
      if (count > best_count) {
        best_count = count;
        best_pair = pair;
      }
    }

    if (best_count == 0)
      break;

    auto best_occurrences = occurrences[best_pair];

    for (const auto index : best_occurrences) {
      uint32_t right = corpus.next[index];

      if (right == end)
        continue;
      if (corpus.previous[right] != index)
        continue;
      // handles case where we remove something and dont derefernce it
      if (corpus.token[index] != best_pair.first ||
          corpus.token[right] != best_pair.second)
        continue;

      uint32_t left = corpus.previous[index];
      uint32_t after_right = corpus.next[right];

      // Remove old neighboring pairs BEFORE mutation.
      if (left != end) {
        Pair old_left{corpus.token[left], corpus.token[index]};
        --counts[old_left];
      }

      if (after_right != end) {
        Pair old_right{corpus.token[right], corpus.token[after_right]};
        --counts[old_right];
      }

      // Perform merge.
      corpus.token[index] = new_token;
      corpus.next[index] = after_right;

      if (after_right != end) {
        corpus.previous[after_right] = index;
      }

      // Add new neighboring pairs.
      if (left != end) {
        Pair new_left{corpus.token[left], new_token};
        ++counts[new_left];
        occurrences[new_left].push_back(left);
      }

      if (after_right != end) {
        Pair new_right{new_token, corpus.token[after_right]};
        ++counts[new_right];
        occurrences[new_right].push_back(index);
      }

      corpus.previous[right] = end;
      corpus.next[right] = end;
    }

    counts.erase(best_pair);
    occurrences.erase(best_pair);
    merges[best_pair] = new_token;
    new_token += 1;
  }

  // flattern hashmap
  return BPETokenizer(std::move(merges), eos_token);
}

BPETokenizer::BPETokenizer(std::unordered_map<Pair, uint32_t, PairHash> merges,
                           uint32_t eos_token)
    : merges(std::move(merges)), eos_token_(eos_token),
      vocabulary_size_(eos_token + 1) {

  for (const auto &[pair, token] : this->merges) {
    decode_merges[token] = pair;
    vocabulary_size_ = std::max(vocabulary_size_, token + 1);
  }
}
std::vector<uint32_t>
BPETokenizer::encode(std::span<const uint32_t> text) const {
  std::priority_queue<Candidate, std::vector<Candidate>,
                      std::greater<Candidate>>
      heap;

  TrainingCorpus corpus;
  corpus.token.assign(text.begin(), text.end());
  corpus.previous.resize(text.size());
  corpus.next.resize(text.size());

  for (uint32_t i = 0; i < text.size(); ++i) {
    if (text[i] == eos_token_) {
      corpus.previous[i] = end;
      corpus.next[i] = end;
      continue;
    }

    if (i == 0 || text[i - 1] == eos_token_)
      corpus.previous[i] = end;
    else
      corpus.previous[i] = i - 1;

    if (i + 1 == text.size() || text[i + 1] == eos_token_)
      corpus.next[i] = end;
    else
      corpus.next[i] = i + 1;
  }

  for (uint32_t i = 0; i + 1 < text.size(); ++i) {
    if (text[i] == eos_token_ || text[i + 1] == eos_token_)
      continue;

    Pair pair{text[i], text[i + 1]};

    auto it = merges.find(pair);
    if (it != merges.end()) {
      heap.push(Candidate{.rank = it->second, .index = i});
    }
  }

  while (!heap.empty()) {
    Candidate best = heap.top();
    heap.pop();

    uint32_t index = best.index;
    uint32_t token = best.rank;

    uint32_t right = corpus.next[index];
    if (right == end || right == removed)
      continue;

    Pair current_pair{corpus.token[index], corpus.token[right]};

    auto current = merges.find(current_pair);
    if (current == merges.end() || current->second != token)
      continue;

    uint32_t left = corpus.previous[index];
    uint32_t after_right = corpus.next[right];

    corpus.token[index] = token;
    corpus.next[index] = after_right;

    if (after_right != end) {
      corpus.previous[after_right] = index;
    }

    corpus.previous[right] = removed;
    corpus.next[right] = removed;

    if (left != end) {
      Pair new_left{corpus.token[left], token};

      auto it = merges.find(new_left);
      if (it != merges.end()) {
        heap.push(Candidate{
            .rank = it->second,
            .index = left,
        });
      }
    }

    if (after_right != end) {
      Pair new_right{token, corpus.token[after_right]};

      auto it = merges.find(new_right);
      if (it != merges.end()) {
        heap.push(Candidate{
            .rank = it->second,
            .index = index,
        });
      }
    }
  }
  // construct the new list
  std::vector<uint32_t> output;

  for (uint32_t i = 0; i < text.size(); ++i) {
    if (text[i] == eos_token_) {
      output.push_back(eos_token_);
      continue;
    }

    if (corpus.previous[i] != end)
      continue;
    if (corpus.previous[i] == removed)
      continue;

    uint32_t next_token = i;

    while (next_token != end) {
      output.push_back(corpus.token[next_token]);
      next_token = corpus.next[next_token];
    }
  }
  return output;
}
void BPETokenizer::decode_token(uint32_t token,
                                std::vector<uint32_t> &output) const {
  auto it = decode_merges.find(token);

  if (it == decode_merges.end()) {
    output.push_back(token);
    return;
  }

  decode_token(it->second.first, output);
  decode_token(it->second.second, output);
}

std::vector<uint32_t>
BPETokenizer::decode(std::span<const uint32_t> encoded) const {
  std::vector<uint32_t> output;

  for (uint32_t token : encoded) {
    decode_token(token, output);
  }

  return output;
}

void BPETokenizer::save(std::ostream &out) const {
  out << eos_token_ << '\n';
  out << merges.size() << '\n';

  for (const auto &[pair, token] : merges) {
    out << pair.first << ' ' << pair.second << ' ' << token << '\n';
  }

  if (!out) {
    throw std::runtime_error("failed to write tokeniser");
  }
}

BPETokenizer BPETokenizer::load(std::istream &in) {
  uint32_t eos_token;
  size_t merge_count;

  if (!(in >> eos_token >> merge_count)) {
    throw std::runtime_error("failed to read tokeniser header");
  }

  std::unordered_map<Pair, uint32_t, PairHash> merges;

  for (size_t i = 0; i < merge_count; ++i) {
    uint32_t left;
    uint32_t right;
    uint32_t token;
    if (!(in >> left >> right >> token)) {
      throw std::runtime_error("failed to read tokeniser merge");
    }
    merges[{left, right}] = token;
  }
  return BPETokenizer(std::move(merges), eos_token);
}
