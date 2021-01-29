//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_TENSOR_OPERATIONS_IMPL_HPP
#define LEGO_TENSOR_OPERATIONS_IMPL_HPP

#include "include/transducer_typed_value.hpp"

namespace tg {

  class tensor_concat_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& tensors, const value_t& axis);

    std::string default_name() const;
  };

  class tensor_concat_op_static_axis {
    unsigned long axis_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axis_m);
    }

    tensor_concat_op_static_axis() = default;

    tensor_concat_op_static_axis(const tensor_concat_op_static_axis&) = default;

    tensor_concat_op_static_axis(tensor_concat_op_static_axis&&) noexcept = default;

    tensor_concat_op_static_axis& operator=(const tensor_concat_op_static_axis&) = default;

    tensor_concat_op_static_axis& operator=(tensor_concat_op_static_axis&&) noexcept = default;

    explicit tensor_concat_op_static_axis(unsigned long axis);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class n_ary_concat_op_static_axis {
    unsigned long axis_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axis_m);
    }
    n_ary_concat_op_static_axis() = default;
    n_ary_concat_op_static_axis(const n_ary_concat_op_static_axis&) = default;
    n_ary_concat_op_static_axis(n_ary_concat_op_static_axis&&) noexcept = default;
    n_ary_concat_op_static_axis& operator=(const n_ary_concat_op_static_axis&) = default;
    n_ary_concat_op_static_axis& operator=(n_ary_concat_op_static_axis&&) noexcept = default;
    explicit n_ary_concat_op_static_axis(unsigned long axis);


    template<typename ...Args>
    value_t transduce(const Args& ...args) {
      return apply_impl(std::vector<value_t>{args...}, axis_m);
    }

    static value_t apply_impl(const std::vector<value_t>& ins, unsigned long axis);

    std::string default_name() const;
  };

  class split_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class max_index_of_tensor1d_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class softmax_op {
    unsigned long axis_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axis_m);
    }
    softmax_op() = default;
    softmax_op(const softmax_op&) = default;
    softmax_op(softmax_op&&) noexcept = default;
    softmax_op& operator=(const softmax_op&) = default;
    softmax_op& operator=(softmax_op&&) noexcept = default;
    explicit softmax_op(unsigned long axis): axis_m(axis) {}

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class log_softmax_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class neg_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class minus_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };


  template<typename scalar_T>
  class minus_scalar_op {
    scalar_T operand_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(operand_m);
    }

    minus_scalar_op() = default;

    minus_scalar_op(const minus_scalar_op&) = default;

    minus_scalar_op(minus_scalar_op&&) noexcept = default;

    minus_scalar_op& operator=(const minus_scalar_op&) = default;

    minus_scalar_op& operator=(minus_scalar_op&&) noexcept = default;

    minus_scalar_op(scalar_T operand) : operand_m(operand) {};

    value_t transduce(const value_t& in0) {
      return in0.visit([&](auto&& x) -> value_t {
        using X = std::decay_t<decltype(x)>;
        constexpr auto traits = value_t::static_type_info<X>();
        if constexpr (traits.is_any_tensor) {
          return value_t(in0.as_symbolic_tensor() - (float) operand_m);
        } else if constexpr (traits.is_any_scalar) {
          return value_t(x - operand_m);
        }
        std::stringstream ss;
        ss << "Cannot invoke " << default_name() << " on input of type " << in0.type_name();
        throw_with_nested(std::runtime_error(ss.str()));
      });
    }


    std::string default_name() const {
      return "operator-";
    }
  };

  template<typename scalar_T>
  class scalar_minus_op {
    scalar_T operand_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(operand_m);
    }

    scalar_minus_op() = default;

    scalar_minus_op(const scalar_minus_op&) = default;

    scalar_minus_op(scalar_minus_op&&) noexcept = default;

    scalar_minus_op& operator=(const scalar_minus_op&) = default;

    scalar_minus_op& operator=(scalar_minus_op&&) noexcept = default;

    scalar_minus_op(scalar_T operand) : operand_m(operand) {};

    value_t transduce(const value_t& in0) {
      return in0.visit([&](auto&& x) -> value_t {
        using X = std::decay_t<decltype(x)>;
        constexpr auto traits = value_t::static_type_info<X>();
        if constexpr (traits.is_any_tensor) {
          return value_t((float) operand_m - in0.as_symbolic_tensor());
        } else if constexpr (traits.is_any_scalar) {
          return value_t(operand_m - x);
        }
        std::stringstream ss;
        ss << "Cannot invoke " << default_name() << " on input of type " << in0.type_name();
        throw_with_nested(std::runtime_error(ss.str()));
      });
    }


    std::string default_name() const {
      return "operator-";
    }
  };

  class add_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  template<typename scalar_T>
  class add_scalar_op {
    scalar_T operand_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(operand_m);
    }

    add_scalar_op() = default;

    add_scalar_op(const add_scalar_op&) = default;

    add_scalar_op(add_scalar_op&&) noexcept = default;

    add_scalar_op& operator=(const add_scalar_op&) = default;

    add_scalar_op& operator=(add_scalar_op&&) noexcept = default;

    add_scalar_op(scalar_T operand) : operand_m(operand) {};

    value_t transduce(const value_t& in0) {
      return in0.visit([&](auto&& x) -> value_t {
        using X = std::decay_t<decltype(x)>;
        constexpr auto traits = value_t::static_type_info<X>();
        if constexpr (traits.is_any_tensor) {
          return value_t((float) operand_m + in0.as_symbolic_tensor());
        } else if constexpr (traits.is_any_scalar) {
          return value_t(operand_m + x);
        }
        std::stringstream ss;
        ss << "Cannot invoke " << default_name() << " on input of type " << in0.type_name();
        throw_with_nested(std::runtime_error(ss.str()));
      });
    }


    std::string default_name() const {
      return "operator+";
    }
  };

  class cmult_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  template<typename scalar_T>
  class cmult_scalar_op {
    scalar_T operand_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(operand_m);
    }

    cmult_scalar_op() = default;

    cmult_scalar_op(const cmult_scalar_op&) = default;

    cmult_scalar_op(cmult_scalar_op&&) noexcept = default;

    cmult_scalar_op& operator=(const cmult_scalar_op&) = default;

    cmult_scalar_op& operator=(cmult_scalar_op&&) noexcept = default;

    cmult_scalar_op(scalar_T operand) : operand_m(operand) {};

    value_t transduce(const value_t& in0) {
      return in0.visit([&](auto&& x) -> value_t {
        using X = std::decay_t<decltype(x)>;
        constexpr auto traits = value_t::static_type_info<X>();
        if constexpr (traits.is_any_tensor) {
          return value_t(in0.as_symbolic_tensor() * (float) operand_m);
        } else if constexpr (traits.is_any_scalar) {
          return value_t(x * operand_m);
        }
        std::stringstream ss;
        ss << "Cannot invoke " << default_name() << " on input of type " << in0.type_name();
        throw_with_nested(std::runtime_error(ss.str()));
      });
    }


    std::string default_name() const {
      return "operator*";
    }
  };

  class matmult_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  class divide_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);

    std::string default_name() const;
  };

  template<typename scalar_T>
  class divide_scalar_op {
    scalar_T operand_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(operand_m);
    }

    divide_scalar_op() = default;

    divide_scalar_op(const divide_scalar_op&) = default;

    divide_scalar_op(divide_scalar_op&&) noexcept = default;

    divide_scalar_op& operator=(const divide_scalar_op&) = default;

    divide_scalar_op& operator=(divide_scalar_op&&) noexcept = default;

    divide_scalar_op(scalar_T operand) : operand_m(operand) {};

    value_t transduce(const value_t& in0) {
      return in0.visit([&](auto&& x) -> value_t {
        using X = std::decay_t<decltype(x)>;
        constexpr auto traits = value_t::static_type_info<X>();
        if constexpr (traits.is_any_tensor) {
          return value_t(in0.as_symbolic_tensor() / (float) operand_m);
        } else if constexpr (traits.is_any_scalar) {
          return value_t(x / operand_m);
        }
        std::stringstream ss;
        ss << "Cannot invoke " << default_name() << " on input of type " << in0.type_name();
        throw_with_nested(std::runtime_error(ss.str()));
      });
    }


    std::string default_name() const {
      return "operator/";
    }
  };

  class tensor_pick_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& tensor, const value_t& index, const value_t& axis);

    std::string default_name() const;
  };

  class tensor_slice_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t
    transduce(const value_t& in0, const value_t& in1, const value_t& in2, const value_t& in3);


    std::string default_name() const;
  };

  class tensor_reshape_op {
    tensor_shape_t output_shape_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(output_shape_m);
    }
    tensor_reshape_op() = default;
    tensor_reshape_op(const tensor_reshape_op&) = default;
    tensor_reshape_op(tensor_reshape_op&&) noexcept = default;
    tensor_reshape_op& operator=(const tensor_reshape_op&) = default;
    tensor_reshape_op& operator=(tensor_reshape_op&&) noexcept = default;
    explicit tensor_reshape_op(tensor_shape_t output_shape): output_shape_m(std::move(output_shape)){}
    value_t transduce(const value_t& x);
    std::string default_name() const;
  };

  class tensor_transpose_op {
    std::vector<unsigned int> axes_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axes_m);
    }
    tensor_transpose_op() = default;
    tensor_transpose_op(const tensor_transpose_op&) = default;
    tensor_transpose_op(tensor_transpose_op&&) noexcept = default;
    tensor_transpose_op& operator=(const tensor_transpose_op&) = default;
    tensor_transpose_op& operator=(tensor_transpose_op&&) noexcept = default;
    explicit tensor_transpose_op(std::vector<unsigned int> axes):axes_m(std::move(axes)) {}
    value_t transduce(const value_t& x);
    std::string default_name() const;
  };

  class tanh_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class relu_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class leaky_relu_op {
    float alpha_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(alpha_m);
    }

    leaky_relu_op() = default;

    leaky_relu_op(const leaky_relu_op&) = default;

    leaky_relu_op(leaky_relu_op&&) noexcept = default;

    leaky_relu_op& operator=(const leaky_relu_op&) = default;

    leaky_relu_op& operator=(leaky_relu_op&&) noexcept = default;

    explicit leaky_relu_op(float alpha);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class elu_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class selu_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class gelu_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class sigmoid_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

  class sqrt_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class pow_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class list_sum_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class list_max_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class binary_max_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class binary_min_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class list_min_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class pickneglogsoftmax_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class tensor_sum_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_average_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_std_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_axis_sum_op {
    std::vector<unsigned> axes_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axes_m);
    }

    tensor_axis_sum_op() = default;

    tensor_axis_sum_op(const tensor_axis_sum_op&) = default;

    tensor_axis_sum_op(tensor_axis_sum_op&&) noexcept = default;

    tensor_axis_sum_op& operator=(const tensor_axis_sum_op&) = default;

    tensor_axis_sum_op& operator=(tensor_axis_sum_op&&) noexcept = default;

    explicit tensor_axis_sum_op(const std::vector<unsigned long>& axes);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_axis_average_op {
    std::vector<unsigned> axes_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axes_m);
    }

    tensor_axis_average_op() = default;

    tensor_axis_average_op(const tensor_axis_average_op&) = default;

    tensor_axis_average_op(tensor_axis_average_op&&) noexcept = default;

    tensor_axis_average_op& operator=(const tensor_axis_average_op&) = default;

    tensor_axis_average_op& operator=(tensor_axis_average_op&&) noexcept = default;

    explicit tensor_axis_average_op(const std::vector<unsigned long>& axes);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_axis_std_op {
    std::vector<unsigned> axes_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axes_m);
    }

    tensor_axis_std_op() = default;

    tensor_axis_std_op(const tensor_axis_std_op&) = default;

    tensor_axis_std_op(tensor_axis_std_op&&) noexcept = default;

    tensor_axis_std_op& operator=(const tensor_axis_std_op&) = default;

    tensor_axis_std_op& operator=(tensor_axis_std_op&&) noexcept = default;

    explicit tensor_axis_std_op(const std::vector<unsigned long>& axes);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_axis_max_op {
    unsigned axis_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axis_m);
    }

    tensor_axis_max_op() = default;

    tensor_axis_max_op(const tensor_axis_max_op&) = default;

    tensor_axis_max_op(tensor_axis_max_op&&) noexcept = default;

    tensor_axis_max_op& operator=(const tensor_axis_max_op&) = default;

    tensor_axis_max_op& operator=(tensor_axis_max_op&&) noexcept = default;

    explicit tensor_axis_max_op(unsigned long axis);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_axis_min_op {
    unsigned axis_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(axis_m);
    }

    tensor_axis_min_op() = default;

    tensor_axis_min_op(const tensor_axis_min_op&) = default;

    tensor_axis_min_op(tensor_axis_min_op&&) noexcept = default;

    tensor_axis_min_op& operator=(const tensor_axis_min_op&) = default;

    tensor_axis_min_op& operator=(tensor_axis_min_op&&) noexcept = default;

    explicit tensor_axis_min_op(unsigned long axis);

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class tensor_l2_norm_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }
    tensor_l2_norm_op() = default;
    tensor_l2_norm_op(const tensor_l2_norm_op&) = default;
    tensor_l2_norm_op(tensor_l2_norm_op&&) noexcept = default;
    tensor_l2_norm_op& operator=(const tensor_l2_norm_op&) = default;
    tensor_l2_norm_op& operator=(tensor_l2_norm_op&&) noexcept = default;
    value_t transduce(const value_t& in0);
    std::string default_name() const;
  };

  class random_normal_op {
    dynet::Dim dim_m;
    float mean_m{};
    float stddev_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dim_m, mean_m, stddev_m);
    }

    random_normal_op() = default;

    random_normal_op(const random_normal_op&) = default;

    random_normal_op(random_normal_op&&) noexcept = default;

    random_normal_op& operator=(const random_normal_op&) = default;

    random_normal_op& operator=(random_normal_op&&) noexcept = default;

    random_normal_op(const tensor_shape_t& shape, float mean, float stddev);

    value_t transduce();


    std::string default_name() const;
  };

  class random_uniform_op {
    dynet::Dim dim_m;
    float min_m{};
    float max_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dim_m, min_m, max_m);
    }

    random_uniform_op() = default;

    random_uniform_op(const random_uniform_op&) = default;

    random_uniform_op(random_uniform_op&&) noexcept = default;

    random_uniform_op& operator=(const random_uniform_op&) = default;

    random_uniform_op& operator=(random_uniform_op&&) noexcept = default;

    random_uniform_op(const tensor_shape_t& shape, float min_val, float max_val);

    value_t transduce();


    std::string default_name() const;
  };

  class random_bernoulli_op {
    dynet::Dim dim_m;
    float p_m{};
    float scale_m{};
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(dim_m, p_m, scale_m);
    }

    random_bernoulli_op() = default;

    random_bernoulli_op(const random_bernoulli_op&) = default;

    random_bernoulli_op(random_bernoulli_op&&) noexcept = default;

    random_bernoulli_op& operator=(const random_bernoulli_op&) = default;

    random_bernoulli_op& operator=(random_bernoulli_op&&) noexcept = default;

    random_bernoulli_op(const tensor_shape_t& shape, float p, float scale);

    value_t transduce();


    std::string default_name() const;
  };

  class pickneglogsigmoid_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0, const value_t& in1);


    std::string default_name() const;
  };

  class log_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {

    }

    value_t transduce(const value_t& in0);


    std::string default_name() const;
  };

  class exp_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }

    value_t transduce(const value_t& x);

    std::string default_name() const;
  };

  class squared_distance_op {
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
    }

    value_t transduce(const value_t& x, const value_t& y);

    std::string default_name() const;
  };
}


#endif //LEGO_TENSOR_OPERATIONS_IMPL_HPP
