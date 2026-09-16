#include "model_utils/archive.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

// Archive::Archive() {}

void Archive::write(const std::string &name, std::span<const std::byte> blob) {
  if (storage.find(name) != storage.end()) {
    throw std::runtime_error(
        "cannot write to that name already exsists in archive");
  }
  storage.emplace(name, std::vector<std::byte>(blob.begin(), blob.end()));
}

void Archive::read(const std::string &name, std::span<std::byte> dst) const {

  auto it = storage.find(name);

  if (it == storage.end())
    throw std::runtime_error("name does not exist in archive");

  const auto &blob = it->second;

  if (dst.size() != blob.size())
    throw std::runtime_error("destination size does not match blob size");

  std::copy(blob.begin(), blob.end(), dst.begin());
}

void Archive::commit(std::ostream &out) const {

  for (auto &&[name, blob] : storage) {

    uint32_t name_len = static_cast<uint32_t>(name.size());

    uint32_t blob_len = static_cast<uint32_t>(blob.size());

    out.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
    out.write(name.data(), name.size());

    out.write(reinterpret_cast<const char *>(&blob_len), sizeof(blob_len));
    out.write(reinterpret_cast<const char *>(blob.data()), blob.size());
  }
}

void Archive::restore(std::istream &file) {
  uint32_t nameLen;
  while (true) {

    if (!file.read(reinterpret_cast<char *>(&nameLen), sizeof(nameLen))) {
      break;
    }
    std::string name(nameLen, '\0');

    if (!file.read(name.data(), nameLen))
      break;

    uint32_t blobLen;

    if (!file.read(reinterpret_cast<char *>(&blobLen), sizeof(blobLen)))
      break;

    std::vector<std::byte> blob(blobLen);

    if (!file.read(reinterpret_cast<char *>(blob.data()), blobLen))
      break;
    storage[name] = std::move(blob);
  }
}

ArchiveView Archive::scope(const std::string &name) {
  return ArchiveView(this, name);
}

ArchiveView::ArchiveView(Archive *archive, std::string_view name)
    : archive_(archive), prefix_(name) {}

void ArchiveView::write(const std::string &name,
                        std::span<const std::byte> data) {
  archive_->write(prefix_ + "/" + name, data);
}

void ArchiveView::read(const std::string &name,
                       std::span<std::byte> data) const {
  archive_->read(prefix_ + "/" + name, data);
}

ArchiveView ArchiveView::scope(const std::string &name) const {
  return ArchiveView(archive_, prefix_ + "/" + name);
}
