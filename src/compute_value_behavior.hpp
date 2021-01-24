//
// Created by Dekai WU and YAN Yuchen on 20201126.
//

#ifndef LEGO_COMPUTE_VALUE_BEHAVIOR_HPP
#define LEGO_COMPUTE_VALUE_BEHAVIOR_HPP

#include "include/value_placeholder.hpp"
#include <memory>
#include <variant>
#include "include/generate_type_consistent_tuple.hpp"
namespace tg {

  class transducer_variant;

  // Represents a value that is computed from a constant value.
  using compute_value_const_impl = value_t;

  // Represents a value that is computed by invoking transducers
  // the transducer may be held weakly.
  // This will happen when a transducer is used before its definition, usually when defining recursion.
  template<unsigned long N>
  struct compute_value_transduce_impl {
    static constexpr unsigned long ARITY = N;

    std::variant<std::shared_ptr<transducer_variant>, std::weak_ptr<transducer_variant>> transducer;
    type_consistent_tuple_t<N, value_placeholder> inputs;
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(transducer, inputs);
    }
  };

  // Represents a value that is supposed to be supplied as input
  // So there is actually no instruction to compute it.
  struct compute_value_input_impl {
    template<typename Archive>
    void serialize(Archive& ar) {}
  };

  // Represents a value that is computed by simply wrapping its inputs into a list
  // This is a special treatment for the make list operation because it may often exceeds the arity 8 limit
  struct compute_value_make_list_impl {
    std::vector<value_placeholder> inputs;
    compute_value_make_list_impl() = default;
    compute_value_make_list_impl(const compute_value_make_list_impl&) = default;
    compute_value_make_list_impl(compute_value_make_list_impl&&) noexcept = default;
    compute_value_make_list_impl& operator=(const compute_value_make_list_impl&) = default;
    compute_value_make_list_impl& operator=(compute_value_make_list_impl&&) noexcept = default;
    explicit compute_value_make_list_impl(std::vector<value_placeholder> inputs):inputs(std::move(inputs)){}
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(inputs);
    }
  };

  /**
   * \brief Specifies how a value_placeholder should be computed.
   */
  struct compute_value_behavior {
    std::variant<compute_value_input_impl, compute_value_const_impl, compute_value_transduce_impl<0>, compute_value_transduce_impl<1>, compute_value_transduce_impl<2>, compute_value_transduce_impl<3>, compute_value_transduce_impl<4>, compute_value_transduce_impl<5>, compute_value_transduce_impl<6>, compute_value_transduce_impl<7>, compute_value_transduce_impl<8>, compute_value_make_list_impl> impl;
    compute_value_behavior() = default;
    compute_value_behavior(const compute_value_behavior&) = default;
    compute_value_behavior(compute_value_behavior&&) noexcept = default;
    compute_value_behavior& operator=(const compute_value_behavior&) = default;
    compute_value_behavior& operator=(compute_value_behavior&&) noexcept = default;
    template<typename T>
    explicit compute_value_behavior(T impl):impl(std::move(impl)) {}
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(impl);
    }
  };
}



#endif
