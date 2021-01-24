//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

#include "tensor_operations_impl.hpp"
#include "include/transducer_typed_value.hpp"
#include <dynet/expr.h>
#include <dynet/nodes-minmax.h>
#include "dynet_computation_graph.hpp"
#include <cmath>

using namespace tg;
using namespace std;

tg::value_t tg::tensor_concat_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return n_ary_concat_op_static_axis::apply_impl(in0.as_list(), in1.as_integer());
}

string tg::tensor_concat_op::default_name() const {
  return "list_concat";
}

tg::tensor_concat_op_static_axis::tensor_concat_op_static_axis(unsigned long axis): axis_m(axis) {}

value_t tg::tensor_concat_op_static_axis::transduce(const tg::value_t& in0) {
  return n_ary_concat_op_static_axis::apply_impl(in0.as_list(), axis_m);
}

string tensor_concat_op_static_axis::default_name() const {
  return "list_concat";
}

n_ary_concat_op_static_axis::n_ary_concat_op_static_axis(unsigned long axis):axis_m(axis) {}


value_t n_ary_concat_op_static_axis::apply_impl(const std::vector<value_t>& ins, unsigned long axis) {
  if (ins.empty()) {
    stringstream ss;
    ss << "Cannot invoke concatenate on an empty list";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  auto rank = ins.front().tensor_shape().size();
  vector<symbolic_tensor_t> tensors;
  for (auto&& in : ins) {
    if (rank != in.tensor_rank()) {
      stringstream ss;
      ss << "Cannot concatenate tensor of different ranks. Got " << rank << " and " << in.tensor_shape().size();
      throw_with_nested(std::runtime_error(ss.str()));
    }
    tensors.push_back(in.as_symbolic_tensor());
  }
  return (tg::value_t) dynet::concatenate(tensors, axis);
}

std::string n_ary_concat_op_static_axis::default_name() const {
  return "concat";
}

tg::value_t tg::split_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  auto shape = in0.tensor_shape();
  auto&& x = in0.as_symbolic_tensor();
  auto axis = (unsigned long) in1.as_integer();
  if (axis >= shape.size()) {
    stringstream ss;
    ss << "Cannot split tensor of rank " << shape.size() << " by axis #" << axis;
    throw_with_nested(std::runtime_error(ss.str()));
  }
  auto axis_length = shape[axis];
  vector<value_t> ret;
  ret.reserve(axis_length);
  for (unsigned long i = 0; i < axis_length; ++i) {
    ret.emplace_back(dynet::pick(x, i, axis));
  }
  return value_t(move(ret));
}

string tg::split_op::default_name() const {
  return "split";
}

tg::value_t tg::max_index_of_tensor1d_op::transduce(const tg::value_t& in0) {
  auto logits_value = dynet::as_vector(dynet_computation_graph::p()->incremental_forward(in0.as_symbolic_tensor()));
  float max_value = logits_value[0];
  unsigned long max_index = 0;
  for (unsigned i = 1; i < logits_value.size(); ++i) {
    float val = logits_value[i];
    if (val > max_value) {
      max_value = val;
      max_index = i;
    }
  }
  return value_t(max_index);
}

string tg::max_index_of_tensor1d_op::default_name() const {
  return "max_index_of";
}

value_t tg::softmax_op::transduce(const value_t& in0) {
  return (tg::value_t) dynet::softmax(in0.as_symbolic_tensor(), axis_m);
}

std::string tg::softmax_op::default_name() const {
  return "softmax";
}

tg::value_t tg::log_softmax_op::transduce(const tg::value_t& in0) {
  return (tg::value_t) dynet::log_softmax(in0.as_symbolic_tensor());
}

string tg::log_softmax_op::default_name() const {
  return "log_softmax";
}

tg::value_t
tg::tensor_pick_op::transduce(const tg::value_t& tensor, const tg::value_t& index,
                              const tg::value_t& axis) {
  return tg::value_t(dynet::pick(tensor.as_symbolic_tensor(), index.as_integer(), axis.as_integer()));
}

string tg::tensor_pick_op::default_name() const {
  return "pick";
}

tg::value_t
tg::tensor_slice_op::transduce(const tg::value_t& in0, const tg::value_t& in1, const tg::value_t& in2,
                               const tg::value_t& in3) {
  auto&& x = in0.as_symbolic_tensor();
  auto begin = (unsigned) in1.as_integer();
  auto end = (unsigned) in2.as_integer();
  auto axis = (unsigned) in3.as_integer();
  return value_t(dynet::pick_range(x, begin, end, axis));
}

string tg::tensor_slice_op::default_name() const {
  return "tensor_slice";
}


tg::value_t tg::tanh_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(tanh((float)v));
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::tanh(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::tanh_op::default_name() const {
  return "tanh";
}

tg::value_t tg::relu_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(v > 0 ? (float)v : (float)0);
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::rectify(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::relu_op::default_name() const {
  return "relu";
}

tg::value_t tg::sigmoid_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(0.5 + 0.5*tanh(0.5*v));
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::logistic(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::sigmoid_op::default_name() const {
  return "sigmoid";
}

tg::value_t tg::list_sum_op::transduce(const tg::value_t& in0) {
  auto&& ins = in0.as_list();
  if(ins.empty()) {
    return value_t((long)0);
  }

  {
    bool is_all_integer = true;
    for(auto&& in:ins) {
      if(!in.is_integer()) {
        is_all_integer = false;
        break;
      }
    }

    if(is_all_integer) {
      long ret = 0;
      for (auto&& in:ins) {
        ret += in.as_integer();
      }
      return value_t(ret);
    }
  }

  {
    bool is_all_scalar = true;
    for(auto&& in:ins) {
      if(!in.is_any_scalar()) is_all_scalar = false;
      break;
    }

    if(is_all_scalar) {
      float ret = 0;
      for (auto&& in:ins) {
        ret += in.as_float();
      }
      return value_t(ret);
    }
  }

  vector<symbolic_tensor_t> tensors;
  tensors.reserve(ins.size());
  for (auto&& in:ins) {
    tensors.push_back(in.as_symbolic_tensor());
  }
  return value_t(dynet::sum(tensors));
}

string tg::list_sum_op::default_name() const {
  return "list_sum";
}

tg::value_t tg::list_max_op::transduce(const tg::value_t& in0) {
  auto&& ins = in0.as_list();
  if(ins.empty()) throw_with_nested(std::runtime_error("Failed to apply " + default_name() + " on empty list"));
  {
    bool is_all_integer = true;
    for(auto&& in:ins) {
      if(!in.is_integer()) {
        is_all_integer = false;
        break;
      }
    }

    if(is_all_integer) {
      long ret = -std::numeric_limits<long>::infinity();
      for (auto&& in:ins) {
        auto&& i = in.as_integer();
        if(i > ret) ret = i;
      }
      return value_t(ret);
    }
  }

  {
    bool is_all_scalar = true;
    for(auto&& in:ins) {
      if(!in.is_any_scalar()) is_all_scalar = false;
      break;
    }

    if(is_all_scalar) {
      float ret = -std::numeric_limits<float>::infinity();
      for (auto&& in:ins) {
        auto&& i = in.as_float();
        if(i > ret) ret = i;
      }
      return value_t(ret);
    }
  }

  switch (ins.size()) {
    case 1:
      return ins[0];
    case 2:
      return value_t(dynet::max(ins[0].as_symbolic_tensor(), ins[1].as_symbolic_tensor()));
    default:
      vector<symbolic_tensor_t> tensors;
      tensors.reserve(ins.size());
      for (auto&& in:ins) {
        tensors.push_back(in.as_symbolic_tensor());
      }
      unsigned tmp_axis = tensors[0].dim().nd;
      return value_t(dynet::max_dim(dynet::concatenate(tensors, tmp_axis), tmp_axis));
  }
}

string tg::list_max_op::default_name() const {
  return "list_cmax";
}


tg::value_t tg::list_min_op::transduce(const tg::value_t& in0) {
  auto&& ins = in0.as_list();
  if(ins.empty()) throw_with_nested(std::runtime_error("Failed to apply " + default_name() + " on empty list"));
  {
    bool is_all_integer = true;
    for(auto&& in:ins) {
      if(!in.is_integer()) {
        is_all_integer = false;
        break;
      }
    }

    if(is_all_integer) {
      long ret = std::numeric_limits<long>::infinity();
      for (auto&& in:ins) {
        auto&& i = in.as_integer();
        if(i < ret) ret = i;
      }
      return value_t(ret);
    }
  }

  {
    bool is_all_scalar = true;
    for(auto&& in:ins) {
      if(!in.is_any_scalar()) is_all_scalar = false;
      break;
    }

    if(is_all_scalar) {
      float ret = std::numeric_limits<float>::infinity();
      for (auto&& in:ins) {
        auto&& i = in.as_float();
        if(i < ret) ret = i;
      }
      return value_t(ret);
    }
  }

  switch (ins.size()) {
    case 1:
      return ins[0];
    case 2:
      return value_t(dynet::min(ins[0].as_symbolic_tensor(), ins[1].as_symbolic_tensor()));
    default:
      vector<symbolic_tensor_t> tensors;
      tensors.reserve(ins.size());
      for (auto&& in:ins) {
        tensors.push_back(in.as_symbolic_tensor());
      }
      unsigned tmp_axis = tensors[0].dim().nd;
      return value_t(dynet::min_dim(dynet::concatenate(tensors, tmp_axis), tmp_axis));
  }
}

string tg::list_min_op::default_name() const {
  return "list_cmin";
}

tg::value_t tg::pickneglogsoftmax_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t(dynet::pickneglogsoftmax(in0.as_symbolic_tensor(), in1.as_integer()));
}

string tg::pickneglogsoftmax_op::default_name() const {
  return "pickneglogsoftmax";
}

tg::value_t tg::tensor_sum_op::transduce(const tg::value_t& in0) {
  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::sum_elems(x));
}

string tg::tensor_sum_op::default_name() const {
  return "tensor_sum";
}

tg::value_t tg::tensor_average_op::transduce(const tg::value_t& in0) {
  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::mean_elems(x));
}

string tg::tensor_average_op::default_name() const {
  return "tensor_mean";
}

tg::value_t tg::tensor_std_op::transduce(const tg::value_t& in0) {
  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::std_elems(x));
}

string tg::tensor_std_op::default_name() const {
  return "tensor_std";
}

tg::tensor_axis_sum_op::tensor_axis_sum_op(const std::vector<unsigned long>& axes) : axes_m(axes.begin(), axes.end()) {

}

tg::value_t tg::tensor_axis_sum_op::transduce(const tg::value_t& in0) {
  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::sum_dim(x, axes_m));
}

string tg::tensor_axis_sum_op::default_name() const {

  stringstream ss;
  ss << "tensor_sum[axis=";
  ss << axes_m[0];
  for (unsigned long i = 1; i < axes_m.size(); ++i) {
    ss << "," << axes_m[i];
  }
  ss << "]";

  return ss.str();
}

tg::value_t tg::tensor_axis_average_op::transduce(const tg::value_t& in0) {
  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::mean_dim(x, axes_m));
}

string tg::tensor_axis_average_op::default_name() const {
  stringstream ss;
  ss << "tensor_average[axis=";
  ss << axes_m[0];
  for (unsigned long i = 1; i < axes_m.size(); ++i) {
    ss << "," << axes_m[i];
  }
  ss << "]";

  return ss.str();
}

tg::tensor_axis_average_op::tensor_axis_average_op(const std::vector<unsigned long>& axes) : axes_m(axes.begin(),
                                                                                                    axes.end()) {

}

tg::value_t tg::tensor_axis_std_op::transduce(const tg::value_t& in0) {
  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::std_dim(x, axes_m));
}

string tg::tensor_axis_std_op::default_name() const {
  stringstream ss;
  ss << "tensor_std[axis=";
  ss << axes_m[0];
  for (unsigned long i = 1; i < axes_m.size(); ++i) {
    ss << "," << axes_m[i];
  }
  ss << "]";

  return ss.str();
}

tg::tensor_axis_std_op::tensor_axis_std_op(const std::vector<unsigned long>& axes) : axes_m(axes.begin(), axes.end()) {

}


string tg::random_normal_op::default_name() const {
  return "random_normal";
}


tg::value_t tg::random_normal_op::transduce() {
  return value_t(dynet::random_uniform(*dynet_computation_graph::p(), dim_m, mean_m, stddev_m));
}

tg::random_normal_op::random_normal_op(const tg::tensor_shape_t& shape, float mean, float stddev)
: dim_m(to_dynet_dim(shape)), mean_m(mean), stddev_m(stddev){

}

tg::value_t tg::random_uniform_op::transduce() {
  return value_t(dynet::random_uniform(*dynet_computation_graph::p(), dim_m, min_m, max_m));
}

string tg::random_uniform_op::default_name() const {
  return "random_uniform";
}

tg::random_uniform_op::random_uniform_op(const tg::tensor_shape_t& shape, float min_val, float max_val):dim_m(to_dynet_dim(shape)), min_m(min_val), max_m(max_val) {

}

string tg::random_bernoulli_op::default_name() const {
  return "random_bernoulli";
}

tg::random_bernoulli_op::random_bernoulli_op(const tg::tensor_shape_t& shape, float p, float scale)
: dim_m(to_dynet_dim(shape)), p_m(p), scale_m(scale) {}

tg::value_t tg::random_bernoulli_op::transduce() {
  return value_t(dynet::random_bernoulli(*dynet_computation_graph::p(), dim_m, p_m, scale_m));
}

tg::value_t tg::tensor_axis_max_op::transduce(const tg::value_t& in0) {
  return value_t(dynet::max_dim(in0.as_symbolic_tensor(), axis_m));
}

string tg::tensor_axis_max_op::default_name() const {
  return "axis_max[axis=" + std::to_string(axis_m) + "]";
}

tg::tensor_axis_max_op::tensor_axis_max_op(unsigned long axis) : axis_m(axis) {

}

tg::value_t tg::tensor_axis_min_op::transduce(const tg::value_t& in0) {
  return value_t(dynet::min_dim(in0.as_symbolic_tensor(), axis_m));
}

string tg::tensor_axis_min_op::default_name() const {
  return "axis_min[axis=" + std::to_string(axis_m) + "]";
}

tg::tensor_axis_min_op::tensor_axis_min_op(unsigned long axis) : axis_m(axis) {

}

tg::value_t tg::leaky_relu_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(v > 0 ? (float)v : (float)v * alpha_m);
    }
    else if constexpr (t.is_any_tensor) {
      auto&& x = in0.as_symbolic_tensor();
      return value_t(dynet::max(x * alpha_m, x));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });

}

string tg::leaky_relu_op::default_name() const {
  return "leaky_relu";
}

tg::leaky_relu_op::leaky_relu_op(float alpha): alpha_m(alpha) {
  if(alpha >= 1) throw_with_nested(std::runtime_error("Cannot create leaky ReLU with alpha >= 1. Got " + std::to_string(alpha_m)));
}

tg::value_t tg::elu_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(v > 0 ? (float)v : exp(v) - 1);
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::elu(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });

}

string tg::elu_op::default_name() const {
  return "elu";
}

tg::value_t tg::selu_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      constexpr float alpha = 1.67326324;
      constexpr float lambda = 1.05070099;
      constexpr float lambda_alpha = lambda * alpha;
      return value_t(v > 0 ? lambda * v : lambda_alpha * (exp(v) - 1));
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::selu(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::selu_op::default_name() const {
  return "selu";
}

tg::value_t tg::gelu_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(0.5 * (float)v * (1 + tanh((float)0.79788456 * ((float)v + (float)0.044715 * pow((float)v, 3)))));
    }
    else if constexpr (t.is_any_tensor) {
      auto&& x = in0.as_symbolic_tensor();
      return value_t(dynet::cmult(0.5 * x, (1.0 + dynet::tanh(0.79788456 * (x + 0.044715 * dynet::cube(x))))));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::gelu_op::default_name() const {
  return "gelu";
}


// todo: finish the rest of the CPU implementation of tensor arithmetic
tg::value_t tg::binary_max_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {

  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor || y_t.is_any_tensor) {
      return value_t(dynet::max(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      if constexpr (x_t.is_integer && y_t.is_integer) {
        return value_t(x > y ? x : y);
      }
      else {
        auto _x = (float)x;
        auto _y = (float)y;
        return value_t(_x > _y ? _x : _y);
      }
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);

}

string tg::binary_max_op::default_name() const {
  return "cmax";
}

tg::value_t tg::binary_min_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor || y_t.is_any_tensor) {
      return value_t(dynet::min(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      if constexpr (x_t.is_integer && y_t.is_integer) {
        return value_t(x < y ? x : y);
      }
      else {
        auto _x = (float)x;
        auto _y = (float)y;
        return value_t(_x < _y ? _x : _y);
      }
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);
}

string tg::binary_min_op::default_name() const {
  return "cmin";
}

tg::value_t tg::sqrt_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(sqrt((float)v));
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::sqrt(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::sqrt_op::default_name() const {
  return "sqrt";
}

value_t tensor_l2_norm_op::transduce(const value_t& in0) {
  return value_t(dynet::l2_norm(in0.as_symbolic_tensor()));
}

std::string tensor_l2_norm_op::default_name() const {
  return "l2_norm";
}


tg::value_t tg::pow_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor || y_t.is_any_tensor) {
      return value_t(dynet::pow(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      return value_t(pow(x, y));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);

}

string tg::pow_op::default_name() const {
  return "pow";
}

tg::value_t tg::pickneglogsigmoid_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  auto&& x = in0.as_symbolic_tensor();
  auto&& oracle = in1.as_symbolic_tensor();
  return value_t(dynet::soft_ifelse(oracle, -dynet::log_sigmoid(x), -dynet::log_sigmoid(-x)));
}

string tg::pickneglogsigmoid_op::default_name() const {
  return "pickneglogsigmoid";
}

tg::value_t tg::log_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    constexpr auto t = value_t::static_type_info<decltype(v)>();

    if constexpr (t.is_any_scalar) {
      return value_t(log((float)v));
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::log(in0.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });

}

string tg::log_op::default_name() const {
  return "log";
}

value_t tg::exp_op::transduce(const value_t& x) {
  return x.visit([&](auto&& v)->value_t {
    constexpr auto t = value_t::static_type_info<decltype(v)>();

    if constexpr (t.is_any_scalar) {
      return value_t(std::exp((float)v));
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(dynet::exp(x.as_symbolic_tensor()));
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< x.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

std::string tg::exp_op::default_name() const {
  return "exp";
}


tg::value_t tg::neg_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v)->value_t {
    using T = decay_t<decltype(v)>;
    constexpr auto t = value_t::static_type_info<T>();
    if constexpr (t.is_any_scalar) {
      return value_t(-v);
    }
    else if constexpr (t.is_any_tensor) {
      return value_t(-in0.as_symbolic_tensor());
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  });
}

string tg::neg_op::default_name() const {
  return "operator-";
}

tg::value_t tg::minus_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor) {
      if constexpr (y_t.is_any_tensor) {
        return tg::value_t(in0.as_symbolic_tensor() - in1.as_symbolic_tensor());
      }
      else if constexpr (y_t.is_any_scalar) {
        return tg::value_t(in0.as_symbolic_tensor() - (float)y);
      }
    }
    else if constexpr (y_t.is_any_tensor) {
      if constexpr (x_t.is_any_scalar) {
        return tg::value_t((float)x - in1.as_symbolic_tensor());
      }
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      return tg::value_t(x - y);
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);
}

string tg::minus_op::default_name() const {
  return "operator-";
}


tg::value_t tg::add_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor) {
      if constexpr (y_t.is_any_tensor) {
        return tg::value_t(in0.as_symbolic_tensor() + in1.as_symbolic_tensor());
      }
      else if constexpr (y_t.is_any_scalar) {
        return tg::value_t(in0.as_symbolic_tensor() + (float)y);
      }
    }
    else if constexpr (y_t.is_any_tensor) {
      if constexpr (x_t.is_any_scalar) {
        return tg::value_t((float)x + in1.as_symbolic_tensor());
      }
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      return tg::value_t(x + y);
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);
}

string tg::add_op::default_name() const {
  return "operator+";
}

tg::value_t tg::matmult_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t(in0.as_symbolic_tensor() * in1.as_symbolic_tensor());
}

string tg::matmult_op::default_name() const {
  return "matmult";
}

tg::value_t tg::divide_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor) {
      if constexpr (y_t.is_any_tensor) {
        return tg::value_t(dynet::cdiv(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
      }
      else if constexpr (y_t.is_any_scalar) {
        return tg::value_t(in0.as_symbolic_tensor() / (float)y);
      }
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      return tg::value_t(x / y);
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);
}

string tg::divide_op::default_name() const {
  return "operator/";
}

tg::value_t tg::cmult_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y)->value_t {
    using X = decay_t<decltype(x)>;
    using Y = decay_t<decltype(y)>;
    constexpr auto x_t = value_t::static_type_info<X>();
    constexpr auto y_t = value_t::static_type_info<Y>();

    if constexpr (x_t.is_any_tensor) {
      if constexpr (y_t.is_any_tensor) {
        return tg::value_t(dynet::cmult(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
      }
      else if constexpr (y_t.is_any_scalar) {
        return tg::value_t(in0.as_symbolic_tensor() * (float)y);
      }
    }
    else if constexpr (y_t.is_any_tensor) {
      if constexpr (x_t.is_any_scalar) {
        return tg::value_t((float)x * in1.as_symbolic_tensor());
      }
    }
    else if constexpr (x_t.is_any_scalar && y_t.is_any_scalar) {
      return tg::value_t(x * y);
    }
    stringstream ss;
    ss << "Cannot invoke "<< default_name() << " on input of type "<< in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(std::runtime_error(ss.str()));
  }, in0, in1);
}

string tg::cmult_op::default_name() const {
  return "operator*";
}

value_t tensor_reshape_op::transduce(const value_t& x) {
  return (value_t)dynet::reshape(x.as_symbolic_tensor(), to_dynet_dim(output_shape_m));
}

std::string tensor_reshape_op::default_name() const {
  return "reshape";
}

value_t tensor_transpose_op::transduce(const value_t& x) {
  return (value_t)dynet::transpose(x.as_symbolic_tensor(), axes_m);
}

std::string tensor_transpose_op::default_name() const {
  return "transpose";
}
