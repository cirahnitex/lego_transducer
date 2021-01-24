//
// Created by Dekai WU and YAN Yuchen on 20200507.
//

#include "dropout_model.hpp"
#include "include/lego_guard.hpp"
#include <dynet/expr.h>
#include "dynet_computation_graph.hpp"
using namespace tg;
using namespace std;
tg::value_t tg::dropout_model::transduce(const tg::value_t& in0) {
  if(!lego_training_guard::is_guarded()) return in0;

  auto&& x = in0.as_symbolic_tensor();
  return value_t(dynet::dropout(x, dropout_rate_m));
}

string dropout_model::default_name() const {
  return "dropout[p=" + std::to_string(dropout_rate_m) + "]";
}

dropout_model::dropout_model(float dropout_rate): dropout_rate_m(dropout_rate) {

}

axis_synchronized_dropout_model::axis_synchronized_dropout_model(float dropout_rate, std::unordered_set<unsigned long> synchronized_axes): dropout_rate(dropout_rate), synchronized_axes(move(synchronized_axes)) {

}

value_t axis_synchronized_dropout_model::transduce(const value_t& in0) {
  if(!lego_training_guard::is_guarded()) return in0;

  auto&& x = in0.as_symbolic_tensor();
  auto shape = in0.tensor_shape();
  vector<long> shrunken_sizes(shape.begin(), shape.end());
  for(auto&& axis:synchronized_axes) {
    shrunken_sizes[axis] = 1;
  }
  auto dropout_mask = dynet::ones(*dynet_computation_graph::p(), shrunken_sizes);
  dropout_mask = dynet::dropout(dropout_mask, dropout_rate);

  for(auto&& axis:synchronized_axes) {
    dropout_mask = dynet::concatenate(vector<symbolic_tensor_t>(shape[axis], dropout_mask), axis);
  }
  return value_t(dynet::cmult(x, dropout_mask));
}

string axis_synchronized_dropout_model::default_name() const {
  return "dropout[p=" + std::to_string(dropout_rate) + "]";
}
