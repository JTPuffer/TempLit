#pragma once
#include <cstdint>
#include <iosfwd>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

class BPETokenizer {
  struct PairHash {
    template <typename A, typename B>
    std::size_t operator()(const std::pair<A, B> &p) const {
      auto h1 = std::hash<A>{}(p.first);
      auto h2 = std::hash<B>{}(p.second);

      return h1 ^ (h2 << 1);
    }
  };
  using Token = uint32_t;

  using Pair = std::pair<Token, Token>;
  std::unordered_map<Pair, uint32_t, PairHash> merges;
  std::unordered_map<uint32_t, Pair> decode_merges;

  explicit BPETokenizer(std::unordered_map<Pair, uint32_t, PairHash> merges,
                        uint32_t eos_token);
  uint32_t eos_token_;
  uint32_t vocabulary_size_;

  const static uint32_t end = UINT32_MAX;
  const static uint32_t removed = UINT32_MAX - 1;

  void decode_token(uint32_t token, std::vector<uint32_t> &output) const;

public:
  static BPETokenizer train(std::span<const uint32_t> tokens,
                            uint32_t vocabulary_size, uint32_t eos_token);

  std::vector<uint32_t> encode(std::span<const uint32_t> text) const;
  std::vector<uint32_t> decode(std::span<const uint32_t> tokens) const;

  uint32_t eos_token() const { return eos_token_; }
  uint32_t vocabulary_size() const { return vocabulary_size_; }

  void save(std::ostream &out) const;

  static BPETokenizer load(std::istream &in);
};
