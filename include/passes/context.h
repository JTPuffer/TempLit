#pragma once

#include "passes/concepts.h"
#include "passes/meta.h"

#include <utility>

namespace passes {

template <typename T, typename Path> struct node_slot {
  T node{};
};

template <typename T, typename Paths> struct PathContext {
  static_assert(TypeList<Paths>,
                "PathContext paths must be passes::type_list<...>");
};

template <typename T, typename... Paths>
struct PathContext<T, type_list<Paths...>> : node_slot<T, Paths>... {
  using value_type = T;
  using paths_type = type_list<Paths...>;

  template <typename Path> T &get_node() {
    if constexpr (contains_type_v<paths_type, Path>) {
      return static_cast<node_slot<T, Path> &>(*this).node;
    } else {
      static_assert(always_false_v<Path>,
                    "PathContext does not contain the requested path");
    }
  }

  template <typename Path> const T &get_node() const {
    if constexpr (contains_type_v<paths_type, Path>) {
      return static_cast<const node_slot<T, Path> &>(*this).node;
    } else {
      static_assert(always_false_v<Path>,
                    "PathContext does not contain the requested path");
    }
  }

  template <typename Fn> void for_each(Fn &&fn) {
    (fn.template operator()<Paths>(
         static_cast<node_slot<T, Paths> &>(*this).node),
     ...);
  }

  template <typename Fn> void for_each(Fn &&fn) const {
    (fn.template operator()<Paths>(
         static_cast<const node_slot<T, Paths> &>(*this).node),
     ...);
  }
};

template <typename T, typename Path> struct gradient_slot {
  T gradient{};
  bool initialised = false;

  void clear() { initialised = false; }
};

template <typename T, typename Paths> struct GradientContext {
  static_assert(TypeList<Paths>,
                "GradientContext paths must be passes::type_list<...>");
};

template <typename T, typename... Paths>
struct GradientContext<T, type_list<Paths...>> : gradient_slot<T, Paths>... {
  using value_type = T;
  using paths_type = type_list<Paths...>;

  template <typename Path, typename Run>
    requires GradientRuntime<T, Run>
  void accumulate(Run &run, const T &grad) {
    if constexpr (contains_type_v<paths_type, Path>) {
      auto &slot = static_cast<gradient_slot<T, Path> &>(*this);

      if (slot.initialised) {
        run.add(slot.gradient, grad, slot.gradient);
        return;
      }

      run.copy(grad, slot.gradient);
      slot.initialised = true;
    } else {
      static_assert(always_false_v<Path>,
                    "GradientContext does not contain the requested path");
    }
  }

  template <typename Path> bool has_gradient() const {
    if constexpr (contains_type_v<paths_type, Path>) {
      return static_cast<const gradient_slot<T, Path> &>(*this).initialised;
    } else {
      static_assert(always_false_v<Path>,
                    "GradientContext does not contain the requested path");
    }
  }

  template <typename Path> T &get_node() {
    if constexpr (contains_type_v<paths_type, Path>) {
      return static_cast<gradient_slot<T, Path> &>(*this).gradient;
    } else {
      static_assert(always_false_v<Path>,
                    "GradientContext does not contain the requested path");
    }
  }

  template <typename Path> const T &get_node() const {
    if constexpr (contains_type_v<paths_type, Path>) {
      return static_cast<const gradient_slot<T, Path> &>(*this).gradient;
    } else {
      static_assert(always_false_v<Path>,
                    "GradientContext does not contain the requested path");
    }
  }

  void clear() { (static_cast<gradient_slot<T, Paths> &>(*this).clear(), ...); }

  template <typename Fn> void for_each_slot(Fn &&fn) {
    (fn.template operator()<Paths>(
         static_cast<gradient_slot<T, Paths> &>(*this).gradient),
     ...);
  }

  template <typename Fn> void for_each(Fn &&fn) {
    (
        [&] {
          auto &slot = static_cast<gradient_slot<T, Paths> &>(*this);
          if (slot.initialised)
            fn.template operator()<Paths>(slot.gradient);
        }(),
        ...);
  }

  template <typename Fn> void for_each(Fn &&fn) const {
    (
        [&] {
          const auto &slot =
              static_cast<const gradient_slot<T, Paths> &>(*this);
          if (slot.initialised)
            fn.template operator()<Paths>(slot.gradient);
        }(),
        ...);
  }
};

template <typename Forward, typename Gradients, typename LocalGradients,
          typename Indices, typename Temporary, typename T>
struct AutogradContext {
  using value_type = T;

  Forward &forward;
  Gradients &gradients;
  LocalGradients &local_gradients;
  Indices &indices;
  Temporary &temporary;
  const T &root_grad;
};

template <typename Forward, typename Input, typename Indices,
          typename Temporary>
struct ForwardExecutionContext {
  using value_type = typename std::remove_reference_t<Forward>::value_type;
  static_assert(
      std::same_as<value_type,
                   typename std::remove_reference_t<Input>::value_type>);
  static_assert(
      std::same_as<value_type,
                   typename std::remove_reference_t<Temporary>::value_type>);

  Input &input;
  Indices &indices;
  Forward &forward;
  Temporary &temporary;
};

} // namespace passes
