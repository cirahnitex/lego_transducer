//
// Created by Dekai WU and YAN Yuchen on 20200422.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_DENSE_MODEL_HPP
#define LEGO_DENSE_MODEL_HPP

#include "backprop_trainable_parameter.hpp"
#include "include/transducer_typed_value.hpp"

namespace tg {
  class dense_model {
    unsigned long dim_in_m{};
    unsigned long dim_out_m{};
    backprop_trainable_parameter W_m;
    backprop_trainable_bias_parameter b_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dim_in_m, dim_out_m, W_m, b_m);
    }

    dense_model() = default;

    dense_model(const dense_model&) = default;

    dense_model(dense_model&&) noexcept = default;

    dense_model& operator=(const dense_model&) = default;

    dense_model& operator=(dense_model&&) noexcept = default;

    dense_model(unsigned long input_size, unsigned long output_size, bool has_bias = true);

    std::string default_name() const;

    value_t transduce(const value_t& in0);

    symbolic_tensor_t transduce_impl(const symbolic_tensor_t& x);

    unsigned long input_size() const;

    unsigned long output_size() const;

  };

  /**
   * Similar to a dense layer except that it takes multiple inputs.
   * Mathematically identical to first concatenating all inputs and then applying a dense layer.
   */
  class n_ary_dense_model {
    std::vector<unsigned long> dim_ins_m{};
    unsigned long dim_out_m{};
    std::vector<backprop_trainable_parameter> Ws_m;
    backprop_trainable_bias_parameter b_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dim_ins_m, dim_out_m, Ws_m, b_m);
    }
    n_ary_dense_model() = default;
    n_ary_dense_model(const n_ary_dense_model&) = default;
    n_ary_dense_model(n_ary_dense_model&&) noexcept = default;
    n_ary_dense_model& operator=(const n_ary_dense_model&) = default;
    n_ary_dense_model& operator=(n_ary_dense_model&&) noexcept = default;
    n_ary_dense_model(std::vector<unsigned long> input_sizes, unsigned long output_size, bool has_bias = true);

    bool is_arity(unsigned long arity) const;

    value_t apply(const std::vector<value_t>& ins);

    template<typename ...Args>
    value_t transduce(Args... args) {
      return value_t(apply_impl({args.as_symbolic_tensor()...}));
    }

    symbolic_tensor_t apply_impl(const std::vector<symbolic_tensor_t>& ins);

    unsigned output_size() const;

    std::string default_name() const;
  };
}



#endif //LEGO_DENSE_MODEL_HPP
