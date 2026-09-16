#pragma once

#include "optimiser/concepts.h"

#include <stdexcept>

namespace Optimiser {

template <Optimisable T> class SGD {
  T scaled_gradient_;
  bool initialised_ = false;

public:
  struct Shared {
  private:
    T learning_rate_;

  public:
    explicit Shared(typename T::value_type learning_rate)
        : learning_rate_(learning_rate) {}

    friend class SGD;
    void set_lr(typename T::value_type lr) { learning_rate_.set(lr, 0); }
    typename T::value_type get_lr() { return learning_rate_.get(0); }
  };

  static Shared generate_params(typename T::value_type learning_rate) {
    return Shared(learning_rate);
  }

  template <typename Run>
  void update(Run &runtime, const Shared &shared, const T &parameter,
              const T &gradient, T &destination) {
    if (!initialised_) {
      scaled_gradient_ = T(gradient.shape());
      initialised_ = true;
    } else if (scaled_gradient_.shape() != gradient.shape()) {
      throw std::invalid_argument("SGD parameter shape changed");
    }

    runtime.hadamard(gradient, shared.learning_rate_, scaled_gradient_);
    runtime.sub(parameter, scaled_gradient_, destination);
  }
};

} // namespace Optimiser
