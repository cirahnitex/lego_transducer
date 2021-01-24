//
// Created by Dekai WU and YAN Yuchen on 20200507.
//

#include "bilinear_model.hpp"
#include "dynet_computation_graph.hpp"
using namespace tg;
using namespace std;

bilinear_model::bilinear_model(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size, bool with_bias)
: input_0_size_m(input_0_size), input_1_size_m(input_1_size), output_size_m(output_size),
  W({input_0_size * output_size, input_1_size}){
  if(with_bias) b = backprop_trainable_bias_parameter({output_size});
}

value_t bilinear_model::transduce(const value_t& in0, const value_t& in1) {
  auto&& x = in0.as_symbolic_tensor();
  auto&& y = in1.as_symbolic_tensor();
  if(in0.tensor_shape() != tensor_shape_t({input_0_size_m}) || in1.tensor_shape() != tensor_shape_t({input_1_size_m})) {
    stringstream ss;
    ss << "Failed to apply "<<default_name()<<": Expected input shapes "
       << print_tensor_shape(tensor_shape_t({input_0_size_m})) << " and " << print_tensor_shape(tensor_shape_t({input_1_size_m}))
       << ", got "
       << print_tensor_shape(in0.tensor_shape()) << " and "<< print_tensor_shape(in1.tensor_shape()) << ".";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  auto t = dynet::reshape(W.as_symbolic_tensor() * y, {(unsigned)input_0_size_m, (unsigned)output_size_m});
  auto ret = dynet::reshape(dynet::reshape(x, {1, (unsigned)input_0_size_m}) * t, {(unsigned)output_size_m});
  if(b) ret = ret + b.as_symbolic_tensor();
  return value_t(ret);
}

std::string bilinear_model::default_name() const {
  return "bilinear[" + std::to_string(input_0_size_m) + "," + std::to_string(input_1_size_m) + "=>" + std::to_string(output_size_m) + "]";
}


biaffine_model::biaffine_model(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size,
                               bool with_bias)
 : input_0_size_m(input_0_size), input_1_size_m(input_1_size), output_size_m(output_size), W_bilinear_m({input_0_size * output_size, input_1_size}), W_linear0_m({output_size, input_0_size}), W_linear1_m({output_size, input_1_size}) {
  if(with_bias) b_m = backprop_trainable_bias_parameter({output_size});
}

value_t biaffine_model::transduce(const value_t& in0, const value_t& in1) {
  auto&& x = in0.as_symbolic_tensor();
  auto&& y = in1.as_symbolic_tensor();
  if(in0.tensor_shape() != tensor_shape_t({input_0_size_m}) || in1.tensor_shape() != tensor_shape_t({input_1_size_m})) {
    stringstream ss;
    ss << "Failed to apply "<<default_name()<<": Expected input shapes "
       << print_tensor_shape(tensor_shape_t({input_0_size_m})) << " and " << print_tensor_shape(tensor_shape_t({input_1_size_m}))
       << ", got "
       << print_tensor_shape(in0.tensor_shape()) << " and "<< print_tensor_shape(in1.tensor_shape()) << ".";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  auto t = dynet::reshape(W_bilinear_m.as_symbolic_tensor() * y, {(unsigned)input_0_size_m, (unsigned)output_size_m});

  auto biaffine_term = dynet::reshape(dynet::reshape(x, {1, (unsigned)input_0_size_m}) * t, {(unsigned)output_size_m});

  auto b = b_m ? b_m.as_symbolic_tensor() : dynet::zeros(*dynet_computation_graph::p(), {(unsigned) output_size_m});

  return value_t(biaffine_term + dynet::affine_transform({b, W_linear0_m.as_symbolic_tensor(), x, W_linear1_m.as_symbolic_tensor(), y}));
}

std::string biaffine_model::default_name() const {
  return "biaffine[" + std::to_string(input_0_size_m) + "," + std::to_string(input_1_size_m) + "=>" + std::to_string(output_size_m) + "]";
}
