//
// Created by Dekai WU and YAN Yuchen on 20200526.
//

#include "rnn_cells.hpp"
#include "dynet_computation_graph.hpp"
using namespace tg;
using namespace std;

std::string tg::naive_rnn_cell::default_name() const {
  return "naive_rnn_cell";
}

std::pair<value_t, value_t>
tg::naive_rnn_cell::transduce_impl(const tg::value_t& prev_state, const tg::value_t& x) {
  auto out = value_t(dynet::rectify(dense_m.apply_impl({prev_state.as_symbolic_tensor(), x.as_symbolic_tensor()})));
  return make_pair(out, out);
}

value_t naive_rnn_cell::default_initial_state() const {
  return value_t(dynet::zeros(*dynet_computation_graph::p(), {(unsigned)dense_m.output_size()}));
}


naive_rnn_cell::naive_rnn_cell(unsigned long input_size, unsigned long output_size)
: dense_m({output_size, input_size}, output_size){

}

value_t rnn_cell_base::null_state_to_default_state(const tg::value_t& state) {
  return state.is_null() ? default_initial_state() : state;
}

tg::value_t tg::rnn_cell_base::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  auto[out, next_state] = transduce_impl(null_state_to_default_state(in0), in1);
  return value_t({out, next_state});
}

pair<value_t, value_t> vanilla_lstm_cell::transduce_impl(const value_t& prev_state, const value_t& x) {
  auto&&[cell_state, hidden_state] = prev_state.as_tuple<2>();

  auto cat_input = dynet::concatenate({hidden_state.as_symbolic_tensor(), x.as_symbolic_tensor()});
  auto forget_mask = dynet::logistic(forget_gate.transduce_impl(cat_input));
  auto input_mask = dynet::logistic(input_gate.transduce_impl(cat_input));
  auto output_mask = dynet::logistic(output_gate.transduce_impl(cat_input));
  auto input_candidate = dynet::tanh(input_layer.transduce_impl(cat_input));
  auto next_cell_state =
    dynet::cmult(cell_state.as_symbolic_tensor(), forget_mask) + dynet::cmult(input_candidate, input_mask);
  auto next_hidden_state =
    value_t(dynet::cmult(dynet::tanh(next_cell_state), output_mask));

  return make_pair(
    next_hidden_state,
    value_t({value_t(next_cell_state), next_hidden_state})
  );
}

vanilla_lstm_cell::vanilla_lstm_cell(unsigned long input_size, unsigned long output_size)
  : input_size_m(input_size), output_size_m(output_size),
    forget_gate(input_size + output_size, output_size),
    input_gate(input_size + output_size, output_size),
    output_gate(input_size + output_size, output_size),
    input_layer(input_size + output_size, output_size){

}

value_t vanilla_lstm_cell::default_initial_state() const {
  auto zeros = value_t(dynet::zeros(*dynet_computation_graph::p(), {(unsigned)output_size_m}));
  return value_t({zeros, zeros});
}


string vanilla_lstm_cell::default_name() const {
  return "vanilla_lstm_cell";
}

coupled_lstm_cell::coupled_lstm_cell(unsigned long input_size, unsigned long output_size)
  : input_size_m(input_size), output_size_m(output_size),
    forget_gate(input_size + output_size, output_size),
    output_gate(input_size + output_size, output_size),
    input_layer(input_size + output_size, output_size){

}

value_t coupled_lstm_cell::default_initial_state() const {
  auto zeros = value_t(dynet::zeros(*dynet_computation_graph::p(), {(unsigned)output_size_m}));
  return value_t({zeros, zeros});
}



string coupled_lstm_cell::default_name() const {
  return "coupled_lstm_cell";
}

std::pair<value_t, value_t>
coupled_lstm_cell::transduce_impl(const value_t& prev_state, const value_t& x) {
  auto&&[cell_state, hidden_state] = prev_state.as_tuple<2>();

  auto cat_input = dynet::concatenate({hidden_state.as_symbolic_tensor(), x.as_symbolic_tensor()});
  auto forget_mask = dynet::logistic(forget_gate.transduce_impl(cat_input));
  auto input_mask = 1.0f - forget_mask;
  auto output_mask = dynet::logistic(output_gate.transduce_impl(cat_input));
  auto input_candidate = dynet::tanh(input_layer.transduce_impl(cat_input));
  auto next_cell_state =
    dynet::cmult(cell_state.as_symbolic_tensor(), forget_mask) + dynet::cmult(input_candidate, input_mask);
  auto next_hidden_state =
    value_t(dynet::cmult(dynet::tanh(next_cell_state), output_mask));

  return make_pair(
    next_hidden_state,
    value_t({value_t(next_cell_state), next_hidden_state})
  );
}

pair<value_t, value_t> gru_cell::transduce_impl(const value_t& prev_state, const value_t& x) {
  auto input_for_gates = dynet::concatenate({prev_state.as_symbolic_tensor(), x.as_symbolic_tensor()});
  auto pre_input_gate_coef = dynet::logistic(pre_input_gate_m.transduce_impl(input_for_gates));
  auto output_gate_coef = dynet::logistic(output_gate_m.transduce_impl(input_for_gates));
  auto gated_concat = dynet::concatenate(
    {dynet::cmult(prev_state.as_symbolic_tensor(), pre_input_gate_coef), x.as_symbolic_tensor()});
  auto output_candidate = dynet::tanh(input_fc_m.transduce_impl(gated_concat));
  auto after_forget = dynet::cmult(prev_state.as_symbolic_tensor(), (float) 1.0 - output_gate_coef);
  auto output_hidden = value_t(after_forget + dynet::cmult(output_gate_coef, output_candidate));
  return std::make_pair(output_hidden, output_hidden);
}

value_t gru_cell::default_initial_state() const {
  return value_t(dynet::zeros(*dynet_computation_graph::p(), {(unsigned)output_size_m}));
}

string gru_cell::default_name() const {
  return "gru_cell";
}

gru_cell::gru_cell(unsigned long input_size, unsigned long output_size)
  : input_size_m(input_size), output_size_m(output_size), pre_input_gate_m(output_size + input_size, output_size),
    input_fc_m(output_size + input_size, output_size), output_gate_m(output_size + input_size, output_size) {

}
