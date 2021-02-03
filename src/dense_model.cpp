//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#include "dense_model.hpp"
#include "include/transducer_typed_value.hpp"
#include "dynet_computation_graph.hpp"

using namespace std;
using namespace tg;

tg::symbolic_tensor_t tg::dense_model::transduce_impl(const tg::symbolic_tensor_t& x) {
  if(b_m) {
    return  dynet::affine_transform({b_m.as_symbolic_tensor(), W_m.as_symbolic_tensor(), x});
  }
  else {
    return W_m.as_symbolic_tensor() * x;
  }
}

tg::value_t tg::dense_model::transduce(const tg::value_t& _x) {

  auto&& x = _x.as_symbolic_tensor();

  auto dims = _x.tensor_shape();

  if (dims.size() == 1) {
    if (dims[0] != dim_in_m) {
      stringstream ss;
      ss << "Failed to apply " << default_name() << ": expected input dimension " << dim_in_m << ", got "
         << _x.print_tensor_shape() << ".";
      std::throw_with_nested(std::runtime_error(ss.str()));
    }
  } else if (dims.size() == 2) {
    if (dims[1] != dim_in_m) {
      stringstream ss;
      ss << "Failed to apply " << default_name() << ": expected input dimension " << dim_in_m << ", got "
         << _x.print_tensor_shape() << ".";
      std::throw_with_nested(std::runtime_error(ss.str()));
    }
  } else {
    stringstream ss;
    ss << "Failed to apply " << default_name() << " on input tensor of shape " << _x.print_tensor_shape() << ".";
    std::throw_with_nested(std::runtime_error(ss.str()));
  }

  return value_t(transduce_impl(x));
}

string tg::dense_model::default_name() const {
  return "dense[" + std::to_string(dim_in_m) + "=>" + std::to_string(dim_out_m) + "]";
}


tg::dense_model::dense_model(unsigned long input_size, unsigned long output_size, bool has_bias) :
  dim_in_m(input_size), dim_out_m(output_size), W_m({(long)output_size, (long)input_size}), b_m() {
  if (has_bias) b_m = backprop_trainable_bias_parameter({(long)output_size});
}


unsigned long dense_model::input_size() const {
  return dim_in_m;
}

unsigned long dense_model::output_size() const {
  return dim_out_m;
}


bool n_ary_dense_model::is_arity(unsigned long arity) const {
  return dim_ins_m.size() == arity;
}

value_t n_ary_dense_model::apply(const std::vector<value_t>& ins) {
  std::vector<symbolic_tensor_t> _ins;
  _ins.reserve(ins.size());
  for(auto&& in:ins) {
    _ins.push_back(in.as_symbolic_tensor());
  }
  return value_t(apply_impl(_ins));
}

symbolic_tensor_t n_ary_dense_model::apply_impl(const std::vector<symbolic_tensor_t>& ins) {
  vector<dynet::Expression> args;
  auto arity = dim_ins_m.size();
  args.reserve(1 + 2 * arity);
  args.push_back(b_m ? b_m.as_symbolic_tensor() : dynet::zeros(*dynet_computation_graph::p(), {(unsigned) dim_out_m}));
  for (unsigned long i = 0; i < arity; ++i) {
    args.push_back(Ws_m[i].as_symbolic_tensor());
    args.push_back(ins[i]);
  }
  return dynet::affine_transform(args);
}

n_ary_dense_model::n_ary_dense_model(std::vector<unsigned long> input_sizes, unsigned long output_size, bool has_bias):
  dim_ins_m(move(input_sizes)), dim_out_m{output_size}, Ws_m(), b_m(){
  Ws_m.reserve(dim_ins_m.size());
  for(auto&& input_size:dim_ins_m) {
    Ws_m.push_back(backprop_trainable_parameter(make_tensor_shape(output_size, input_size)));
  }
  if (has_bias) b_m = backprop_trainable_bias_parameter(make_tensor_shape(output_size));
}

string n_ary_dense_model::default_name() const {
  return "dense";
}

unsigned n_ary_dense_model::output_size() const {
  return dim_out_m;
}
