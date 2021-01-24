#include "../include/lego_transducer.hpp"
using namespace std;
using namespace tg;

int main() {
  lego_initialize();




constexpr unsigned long D = 16;  // D is hidden dimension
transducer_model naive_rnn_cell([&](const value_placeholder& prev_state, const value_placeholder& x)->value_placeholder {
  transducer_model tanh_dense = compose(tg::tanh, dense_structure.initialize(D + D, D));
  value_placeholder dense_out = tanh_dense(tensor_concat({prev_state, x}));

  // An RNN cell returns (1) the output at current timestep and (2) the state for next timestep.
  // Also give this transducer a name for a more meaningful error reporting.
  return make_list(dense_out, dense_out);
});
naive_rnn_cell.rename("RNN cell");

transducer_model naive_rnn([&](const value_placeholder& ins)->value_placeholder {

  // The initial state is zero vector
  value_placeholder init_state = value_placeholder::zeros({D});

  transducer_model reducer([&](const value_placeholder& accumulator, const value_placeholder& in)->value_placeholder {
    value_placeholder prev_ys = accumulator[0];
    value_placeholder prev_state = accumulator[1];
    value_placeholder t = naive_rnn_cell(prev_state, in);
    value_placeholder y = t[0];
    value_placeholder next_state = t[1];

    // push_back appends a value to a list
    return make_list(push_back(prev_ys, y), next_state);
  });

  return list_reduce(reducer, ins, make_list(value_placeholder::empty_list(), init_state))[0];
});





  vector<string> token_vocab{"take", "put", "the", "on", "block", "square", "circle", "cube", "cone", "red", "green", "blue"};
  transducer_model symbolic_embedding_table = symbolic_embedding_table_structure.initialize(D, token_vocab);

  vector<string> postag_vocab{"VB", "DT", "IN", "NN", "JJ"};
  transducer_model readout_dense = dense_structure.initialize(D, postag_vocab.size());
  transducer_model classify, classification_loss;
  std::tie(classify, classification_loss) = symbolic_classify_from_logits_structure.initialize(postag_vocab);


transducer_model compute_output_logits([&](const value_placeholder& input_tokens) -> value_placeholder {
  // Lookup the sentence into a list of token embeddings
  value_placeholder input_token_embs = list_map(symbolic_embedding_table, input_tokens);

  // Apply the RNN on the list of token embeddings
  value_placeholder rnn_outputs = naive_rnn(input_token_embs);

  // Transform the RNN outputs into logits
  return list_map(readout_dense, rnn_outputs);
});


transducer_model naive_rnn_postagger([&](const value_placeholder& input_tokens) -> value_placeholder {

  value_placeholder output_logits = compute_output_logits(input_tokens);

  // Readout the POS tags from logits
  value_placeholder output_postags = list_map(classify, output_logits);

  return output_postags;
});


transducer_model naive_rnn_postagger_loss_fn([&](const value_placeholder& input_tokens, const value_placeholder& oracle_postags) -> value_placeholder {

  value_placeholder output_logits = compute_output_logits(input_tokens);

  value_placeholder losses = list_map(classification_loss, output_logits, oracle_postags);

  // Aggregate the loss on every predicted POS tag.
  return list_sum(losses);
});




  vector<value_t> values;
  for(unsigned long i=0; i<20; ++i) {
    values.emplace_back(value_t("take"));
    values.emplace_back(value_t("the"));
    values.emplace_back(value_t("block"));
  }
  cout << naive_rnn_postagger(values) << endl;


}
