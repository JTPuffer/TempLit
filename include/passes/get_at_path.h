#pragma once

#include "passes/program.h"

#include <tuple>

namespace passes {

template <typename Path> struct path_access;

template <> struct path_access<graph_root> {
  template <typename Graph> static Graph &get(Graph &graph) { return graph; }

  template <typename Graph> static const Graph &get(const Graph &graph) {
    return graph;
  }
};

template <typename Parent, std::size_t Index>
struct path_access<graph_path<Parent, Index>> {
  template <typename Graph> static decltype(auto) get(Graph &graph) {
    auto &parent = path_access<Parent>::get(graph);
    return std::get<Index>(parent.args_);
  }

  template <typename Graph> static decltype(auto) get(const Graph &graph) {
    const auto &parent = path_access<Parent>::get(graph);
    return std::get<Index>(parent.args_);
  }
};

template <typename Path, typename Graph>
decltype(auto) get_at_path(Graph &graph) {
  return path_access<Path>::get(graph);
}

template <typename Path, typename Graph>
decltype(auto) get_at_path(const Graph &graph) {
  return path_access<Path>::get(graph);
}

} // namespace passes
