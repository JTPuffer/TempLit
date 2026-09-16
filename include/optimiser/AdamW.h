#pragma once

#include <cstdint>
#include <model_utils/archive_concepts.h>
#include <optimiser/concepts.h>
#include <passes/concepts.h>
#include <span>
#include <stdexcept>
namespace Optimiser {

template <Optimisable T> class AdamW {
  using value_type = typename T::value_type;

public:
  struct Shared {
  private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T decay_;

    value_type epsilon_value_;
    value_type weight_decay_value_;

  public:
    Shared(value_type learning_rate, value_type beta1 = 0.9f,
           value_type beta2 = 0.999f, value_type epsilon = 1e-8f,
           value_type weight_decay = 0.01f)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2),
          decay_(value_type{1} - learning_rate * weight_decay),
          epsilon_value_(epsilon), weight_decay_value_(weight_decay) {}

    friend class AdamW;
    void set_lr(value_type lr) {
      learning_rate_.set(lr, 0);
      decay_.set(value_type{1} - lr * weight_decay_value_, 0);
    }
    value_type get_lr() { return learning_rate_.get(0); }
  };

private:
  uint32_t step_ = 0;

  T momentum1_;
  T momentum2_;

  bool initialised_ = false;

public:
  static Shared generate_params(value_type learning_rate,
                                value_type beta1 = 0.9f,
                                value_type beta2 = 0.999f,
                                value_type epsilon = 1e-8f,
                                value_type weight_decay = 0.01f) {
    return Shared(learning_rate, beta1, beta2, epsilon, weight_decay);
  }

  void initialise_for(const T &parameter) {
    if (initialised_) {
      if (momentum1_.shape() != parameter.shape()) {
        throw std::invalid_argument("AdamW parameter shape changed");
      }

      return;
    }

    momentum1_ = T::from_zeros(parameter.shape());
    momentum2_ = T::from_zeros(parameter.shape());

    initialised_ = true;
  }

  template <typename Run>
    requires passes::AdamWRuntime<T, Run>
  void update(Run &runtime, const Shared &params, const T &parameter,
              const T &gradient, T &destination) {
    initialise_for(parameter);
    ++step_;
    runtime.adamW(parameter, gradient, destination, momentum1_, momentum2_,
                  params.beta1_, params.beta2_, params.decay_,
                  params.learning_rate_, params.epsilon_value_, step_);
  }

  template <WritableArchive Arch> void save(Arch &archive) const {
    if (!initialised_) {
      throw std::logic_error("cannot save uninitialised AdamW state");
    }

    auto momentum1_archive = archive.scope("momentum1");
    auto momentum2_archive = archive.scope("momentum2");
    momentum1_.save(momentum1_archive);
    momentum2_.save(momentum2_archive);
    archive.write("step", std::as_bytes(std::span(&step_, 1)));
  }

  template <ReadableArchive Arch> void load(Arch &archive) {
    if (!initialised_) {
      throw std::logic_error("AdamW must be initialised before loading state");
    }

    auto momentum1_archive = archive.scope("momentum1");
    auto momentum2_archive = archive.scope("momentum2");
    momentum1_.load(momentum1_archive);
    momentum2_.load(momentum2_archive);
    archive.read("step", std::as_writable_bytes(std::span(&step_, 1)));
  }
};

} // namespace Optimiser
