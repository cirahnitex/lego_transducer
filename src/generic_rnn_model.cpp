//
// Created by Dekai WU and YAN Yuchen on 20200525.
//

#include "generic_rnn_model.hpp"
#include "rnn_cells.hpp"
using namespace tg;
using namespace std;

std::pair<std::vector<value_t>, value_t>
generic_rnn_model::transduce_impl(const value_t& init_state, const std::vector<value_t>& xs) {
  auto state = init_state;
  vector<value_t> ys;
  for (auto&& x:xs) {
    auto [y, next_state] = rnn_cell_m->transduce_impl(rnn_cell_m->null_state_to_default_state(state), x);
    ys.push_back(y);
    state = next_state;
  }
  return make_pair(ys, state);
}


generic_rnn_model::generic_rnn_model(std::shared_ptr<rnn_cell_base> rnn_cell) : rnn_cell_m(move(rnn_cell)) {

}

string generic_rnn_model::default_name() const {
  return "RNN";
}

value_t generic_rnn_model::transduce(const value_t& init_state, const value_t& xs) {
  auto[ys, state] = transduce_impl(init_state, xs.as_list());
  return value_t({value_t(move(ys)), state});
}


std::pair<std::vector<value_t>, std::vector<value_t>>
generic_stacked_rnn_model::transduce_impl(const std::vector<value_t>& init_state,
                                          const std::vector<value_t>& xs) {
  vector<value_t> final_state(rnns_m.size());
  auto ys = xs;
  for (unsigned long i = 0; i < rnns_m.size(); ++i) {

    // apply dropout between layers
    if (dropout_m && i > 0) {
      for (auto& y:ys) {
        y = dropout_m->transduce(y);
      }
    }

    auto&& i_init_state = init_state[i];
    auto&& rnn = rnns_m[i];
    tie(ys, final_state[i]) = rnn.transduce_impl(i_init_state, ys);
  }
  return make_pair(ys, final_state);
}

string generic_stacked_rnn_model::default_name() const {
  return "RNN[" + std::to_string(rnns_m.size()) + "-stacks]";
}

generic_stacked_rnn_model::generic_stacked_rnn_model(const std::vector<std::shared_ptr<rnn_cell_base>>& rnn_cells,
                                                     float dropout_rate)
  : rnns_m(), dropout_m(dropout_rate > 0 ? make_shared<dropout_model>(dropout_rate) : nullptr) {
  rnns_m.reserve(rnn_cells.size());
  for (auto&& rnn_cell:rnn_cells) {
    rnns_m.emplace_back(rnn_cell);
  }
}

unsigned long generic_stacked_rnn_model::num_stacks() const {
  return rnns_m.size();
}


value_t generic_stacked_rnn_model::transduce(const value_t& init_state, const value_t& xs) {
  bool use_default_state = init_state.is_null();
  auto[ys, final_state] = transduce_impl(use_default_state ? vector<value_t>(rnns_m.size()) : init_state.as_list(),
                                         xs.as_list());
  return value_t({value_t(ys), value_t(final_state)});
}



string generic_bidirectional_rnn_model::default_name() const {
  return "BiRNN";
}


std::pair<std::vector<value_t>, std::pair<value_t, value_t>>
generic_bidirectional_rnn_model::transduce_impl(const std::pair<value_t, value_t>& init_state,
                                                const std::vector<value_t>& xs) {
  vector<value_t> forward_ys;
  value_t forward_state = init_state.first;
  {

    for (auto&& x:xs) {
      auto [y, next_state] = forward_cell_m->transduce_impl(forward_cell_m->null_state_to_default_state(forward_state), x);
      forward_ys.push_back(y);
      forward_state = next_state;
    }
  }
  vector<value_t> backward_ys_reversed;
  value_t backward_state = init_state.second;
  {

    for (auto itr = xs.rbegin(); itr != xs.rend(); ++itr) {
      auto [y, next_state] = backward_cell_m->transduce_impl(forward_cell_m->null_state_to_default_state(backward_state), *itr);
      backward_ys_reversed.push_back(y);
      backward_state = next_state;
    }
  }
  vector<value_t> concat_ys;
  {
    auto i_forward = forward_ys.begin();
    auto i_backward = backward_ys_reversed.rbegin();
    while (i_forward != forward_ys.end()) {
      concat_ys.emplace_back(dynet::concatenate({i_forward->as_symbolic_tensor(), i_backward->as_symbolic_tensor()}));
      ++i_forward;
      ++i_backward;
    }
  }

  return make_pair(concat_ys, make_pair(forward_state, backward_state));
}


generic_bidirectional_rnn_model::generic_bidirectional_rnn_model(std::shared_ptr<rnn_cell_base> forward_cell,
                                                                 std::shared_ptr<rnn_cell_base> backward_cell)
  : forward_cell_m(move(forward_cell)),
    backward_cell_m(move(backward_cell)) {

}



value_t generic_bidirectional_rnn_model::transduce(const value_t& _init_state, const value_t& xs) {
  std::pair<value_t, value_t> init_state = _init_state.is_null() ? make_pair(value_t(), value_t())
                                                            : _init_state.as_pair();

  auto[outputs, final_state] = transduce_impl(init_state, xs.as_list());
  return value_t({value_t(outputs), value_t({final_state.first, final_state.second})});
}


std::pair<std::vector<value_t>, std::vector<std::pair<value_t, value_t>>>
generic_stacked_bidirectional_rnn_model::transduce_impl(
  const std::vector<std::pair<value_t, value_t>>& init_state, const std::vector<value_t>& xs) {
  auto ys = xs;
  std::vector<std::pair<value_t, value_t>> states(birnns_m.size());
  for (unsigned long i = 0; i < birnns_m.size(); ++i) {
    if (dropout_m && i > 0) {
      for (auto& y:ys) {
        y = dropout_m->transduce(y);
      }
    }
    auto&& birnn = birnns_m[i];
    tie(ys, states[i]) = birnn.transduce_impl(init_state[i], ys);
  }
  return make_pair(ys, states);
}


generic_stacked_bidirectional_rnn_model::generic_stacked_bidirectional_rnn_model(
  const std::vector<std::pair<std::shared_ptr<rnn_cell_base>, std::shared_ptr<rnn_cell_base>>>& rnn_cell_pairs,
  float dropout_rate)
  : birnns_m(),
    dropout_m(dropout_rate > 0 ? make_shared<dropout_model>(dropout_rate) : nullptr) {
  birnns_m.reserve(rnn_cell_pairs.size());
  for (auto&&[forward_cell, backward_cell]:rnn_cell_pairs) {
    birnns_m.emplace_back(forward_cell, backward_cell);
  }
}

string generic_stacked_bidirectional_rnn_model::default_name() const {
  return "BiRNN[" + std::to_string(birnns_m.size()) + "-stacks]";
}


unsigned long generic_stacked_bidirectional_rnn_model::num_stacks() const {
  return birnns_m.size();
}


value_t generic_stacked_bidirectional_rnn_model::transduce(const value_t& _init_state, const value_t& xs) {
  std::vector<std::pair<value_t, value_t>> init_state;
  if (_init_state.is_null()) {
    init_state.resize(num_stacks(), make_pair(value_t(), value_t()));
  } else {
    for (auto&& state_for_stack : _init_state.as_list()) {
      init_state.push_back(state_for_stack.as_pair());
    }
  }

  auto[outputs, final_state] = transduce_impl(init_state, xs.as_list());

  vector<value_t> final_state_ret;
  for (auto&&[forward_state, backward_state]:final_state) {
    final_state_ret.push_back(value_t({forward_state, backward_state}));
  }

  return value_t({value_t(outputs), value_t({final_state_ret})});
}


