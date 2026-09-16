#pragma once

#include "passes/meta.h"

#include <utility>

template <typename Paths> struct for_each_path_impl;

template <typename... Paths>
struct for_each_path_impl<passes::type_list<Paths...>> {
  template <typename Fn> static void run(Fn &&fn) {
    (fn.template operator()<Paths>(), ...);
  }
};

template <passes::TypeList Paths, typename Fn> void for_each_path(Fn &&fn) {
  for_each_path_impl<Paths>::run(std::forward<Fn>(fn));
}
