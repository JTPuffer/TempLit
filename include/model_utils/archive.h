#pragma once

#include "model_utils/archive_concepts.h"

#include <cstddef>
#include <istream>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

class ArchiveView;

class Archive {

  std::unordered_map<std::string, std::vector<std::byte>> storage;

public:
  void write(const std::string &name, std::span<const std::byte> data);
  void read(const std::string &name, std::span<std::byte> data) const;

  ArchiveView scope(const std::string &name);

  void commit(std::ostream &) const;
  void restore(std::istream &);
};

class ArchiveView {
  Archive *archive_;
  std::string prefix_;
  ArchiveView(Archive *archive, std::string_view name);

  friend class Archive;

public:
  void write(const std::string &name, std::span<const std::byte> data);
  void read(const std::string &name, std::span<std::byte> data) const;

  ArchiveView scope(const std::string &name) const;
};

static_assert(ArchiveType<Archive>);
static_assert(ArchiveType<ArchiveView>);
