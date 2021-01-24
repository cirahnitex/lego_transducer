//
// Created by Dekai WU and YAN Yuchen on 20200507.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_BILINEAR_MODEL_HPP
#define LEGO_BILINEAR_MODEL_HPP

#include "backprop_trainable_parameter.hpp"
#include "include/transducer_typed_value.hpp"

namespace tg {
  /**
   * A scalar-output bilinear layer performs
   * $$ f(\mathbf{x},\mathbf{y})=\mathbf{x}^T\mathbf{W}\mathbf{x}+b $$
   * where $ \mathbf{W} $ and $ b $ are parameters.
   *
   * If we apply multiple scalar-output bilinear layers in parallel, then we get a vector-output bilinear layer.
   */
  class bilinear_model {
    unsigned long input_0_size_m{};
    unsigned long input_1_size_m{};
    unsigned long output_size_m{};
    backprop_trainable_parameter W;
    backprop_trainable_bias_parameter b;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(input_0_size_m, input_1_size_m, output_size_m, W, b);
    }

    bilinear_model() = default;
    bilinear_model(const bilinear_model&) = default;
    bilinear_model(bilinear_model&&) noexcept = default;
    bilinear_model& operator=(const bilinear_model&) = default;
    bilinear_model& operator=(bilinear_model&&) noexcept = default;

    bilinear_model(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size, bool with_bias = true);

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;

  };

  class biaffine_model  {
    unsigned long input_0_size_m{};
    unsigned long input_1_size_m{};
    unsigned long output_size_m{};
    backprop_trainable_parameter W_bilinear_m, W_linear0_m, W_linear1_m;
    backprop_trainable_bias_parameter b_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(input_0_size_m, input_1_size_m, output_size_m, W_bilinear_m, W_linear0_m, W_linear1_m, b_m);
    }

    biaffine_model() = default;
    biaffine_model(const biaffine_model&) = default;
    biaffine_model(biaffine_model&&) noexcept = default;
    biaffine_model& operator=(const biaffine_model&) = default;
    biaffine_model& operator=(biaffine_model&&) noexcept = default;

    biaffine_model(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size, bool with_bias = true);

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;

  };

}


#endif
