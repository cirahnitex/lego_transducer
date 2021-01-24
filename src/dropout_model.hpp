//
// Created by Dekai WU and YAN Yuchen on 20200507.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_DROPOUT_MODEL_HPP
#define LEGO_DROPOUT_MODEL_HPP

#include "backprop_trainable_parameter.hpp"
#include "include/transducer_typed_value.hpp"


namespace tg {
  class dropout_model {
    float dropout_rate_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dropout_rate_m);
    }

    dropout_model() = default;

    dropout_model(const dropout_model&) = default;

    dropout_model(dropout_model&&) noexcept = default;

    dropout_model& operator=(const dropout_model&) = default;

    dropout_model& operator=(dropout_model&&) noexcept = default;

    explicit dropout_model(float dropout_rate);

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class axis_synchronized_dropout_model {
    float dropout_rate{};
    std::unordered_set<unsigned long> synchronized_axes;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dropout_rate, synchronized_axes);
    }

    axis_synchronized_dropout_model() = default;

    axis_synchronized_dropout_model(const axis_synchronized_dropout_model&) = default;

    axis_synchronized_dropout_model(axis_synchronized_dropout_model&&) noexcept = default;

    axis_synchronized_dropout_model& operator=(const axis_synchronized_dropout_model&) = default;

    axis_synchronized_dropout_model& operator=(axis_synchronized_dropout_model&&) noexcept = default;

    axis_synchronized_dropout_model(float dropout_rate, std::unordered_set<unsigned long> synchronized_axes);

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };
}

#endif
