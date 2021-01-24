//
// Created by Dekai WU and YAN Yuchen on 20200428.
//

#include "include/transducer_structure.hpp"
#include "dense_model.hpp"
#include "embedding_table_model.hpp"
#include "symbol_id_converter_model.hpp"
#include "bilinear_model.hpp"
#include "generic_rnn_model.hpp"
#include "rnn_cells.hpp"
#include "dropout_model.hpp"
#include "include/lego_tensor_operations.hpp"
#include "transducer_variant.hpp"

using namespace tg;
using namespace std;

tg::transducer_model
dense_structure_t::initialize(unsigned long input_dimension, unsigned long output_dimension, bool use_bias) {
  return transducer_model(std::make_shared<transducer_variant>(dense_model(input_dimension, output_dimension, use_bias)));
}

dense_structure_t tg::dense_structure;

transducer_model n_ary_dense_structure_t::initialize(std::vector<unsigned long> input_dimensions,
                                                     unsigned long output_dimension, bool use_bias) {
  return transducer_model(std::make_shared<transducer_variant>(n_ary_dense_model(move(input_dimensions), output_dimension, use_bias)));
}

n_ary_dense_structure_t tg::n_ary_dense_structure;

tg::transducer_model embedding_table_structure_t::initialize(unsigned long embedding_size, unsigned long capacity) {
  return tg::transducer_model(std::make_shared<transducer_variant>(tg::embedding_table_model(embedding_size, capacity)));
}

transducer_model embedding_table_structure_t::initialize(unsigned long embedding_size,
                                                         const std::vector<std::vector<float>>& pretrained_embeddings) {
  return tg::transducer_model(std::make_shared<transducer_variant>(tg::embedding_table_model(embedding_size, pretrained_embeddings)));
}

embedding_table_structure_t tg::embedding_table_structure;

tg::transducer_model dict_structure_t::initialize(const std::vector<symbol_t>& vocab) {
  return transducer_model(std::make_shared<transducer_variant>(tg::dict_model(vocab)));
}

dict_structure_t tg::dict_structure;

transducer_model
bilinear_structure_t::initialize(unsigned long input_0_size, unsigned long input_1_size, unsigned long output_size, bool with_bias) {
  return transducer_model(std::make_shared<transducer_variant>(bilinear_model(input_0_size, input_1_size, output_size, with_bias)));
}

bilinear_structure_t tg::bilinear_structure;

transducer_model biaffine_structure_t::initialize(unsigned long input_0_size, unsigned long input_1_size,
                                                  unsigned long output_size, bool with_bias) {
  return transducer_model(std::make_shared<transducer_variant>(biaffine_model(input_0_size, input_1_size, output_size, with_bias)));
}

biaffine_structure_t tg::biaffine_structure;

transducer_model
rnn_structure_t::initialize(unsigned long input_size, unsigned long output_size, RNN_CELL_TYPE cell_type,
                            unsigned long num_stacks, float dropout_rate) {
  std::function<shared_ptr<rnn_cell_base>()> make_cell;
  std::string name = "[" + to_string(input_size) + "=>" + to_string(output_size) + "]";
  switch (cell_type) {
    case NAIVE_RNN:
      make_cell = [&]() { return make_shared<naive_rnn_cell>(input_size, output_size); };
      name = "naive_rnn" + name;
      break;
    case VANILLA_LSTM:
      make_cell = [&]() { return make_shared<vanilla_lstm_cell>(input_size, output_size); };
      name = "vanilla_lstm" + name;
      break;
    case COUPLED_LSTM:
      make_cell = [&]() { return make_shared<coupled_lstm_cell>(input_size, output_size); };
      name = "coupled_lstm" + name;
      break;
    case GRU:
      make_cell = [&]() { return make_shared<gru_cell>(input_size, output_size); };
      name = "gru" + name;
      break;
    default:
      throw_with_nested(std::runtime_error("Unknown RNN cell type"));
  }

  if (num_stacks <= 1) {
    if (dropout_rate > 0) throw_with_nested(std::runtime_error("Dropout only applies to multi-stack RNN."));
    transducer_model ret(make_shared<transducer_variant>(generic_rnn_model(make_cell())));
    ret.rename(name);
    return ret;
  }

  vector<shared_ptr<rnn_cell_base>> cells;
  cells.reserve(num_stacks);
  for (unsigned long i = 0; i < num_stacks; ++i) {
    cells.push_back(make_cell());
  }
  transducer_model ret(make_shared<transducer_variant>(generic_stacked_rnn_model(cells, dropout_rate)));
  ret.rename(name);
  return ret;
}

rnn_structure_t tg::rnn_structure;

transducer_model
bidirectional_rnn_structure_t::initialize(unsigned long input_size, unsigned long output_size, RNN_CELL_TYPE cell_type,
                                          unsigned long num_stacks, float dropout_rate) {
  if (output_size % 2 != 0) throw_with_nested(std::runtime_error(
      "The output size of a bidirectional RNN must be a multiple of 2. Got " + std::to_string(output_size)));

  std::function<shared_ptr<rnn_cell_base>(unsigned long i, unsigned long o)> make_cell;
  std::string name = "[" + to_string(input_size) + "=>" + to_string(output_size) + "]";
  switch (cell_type) {
    case NAIVE_RNN:
      make_cell = [&](unsigned long i, unsigned long o) { return make_shared<naive_rnn_cell>(i, o / 2); };
      name = "bi_naive_rnn" + name;
      break;
    case VANILLA_LSTM:
      make_cell = [&](unsigned long i, unsigned long o) { return make_shared<vanilla_lstm_cell>(i, o / 2); };
      name = "bi_vanilla_lstm" + name;
      break;
    case COUPLED_LSTM:
      make_cell = [&](unsigned long i, unsigned long o) { return make_shared<coupled_lstm_cell>(i, o / 2); };
      name = "bi_coupled_lstm" + name;
      break;
    case GRU:
      make_cell = [&](unsigned long i, unsigned long o) { return make_shared<gru_cell>(i, o / 2); };
      name = "bi_gru" + name;
      break;
    default:
      throw_with_nested(std::runtime_error("Unknown RNN cell type"));
  }

  if (num_stacks <= 1) {
    if (dropout_rate > 0) throw_with_nested(std::runtime_error("Dropout only applies to multi-stack RNN."));
    transducer_model ret(make_shared<transducer_variant>(generic_bidirectional_rnn_model(make_cell(input_size, output_size), make_cell(input_size, output_size))));
    ret.rename(name);
    return ret;
  }

  vector<pair<shared_ptr<rnn_cell_base>, shared_ptr<rnn_cell_base>>> cells;
  cells.reserve(num_stacks);
  for (unsigned long i = 0; i < num_stacks; ++i) {
    unsigned long _input_size = i == 0?input_size:output_size;
    cells.emplace_back(make_cell(_input_size, output_size), make_cell(_input_size, output_size));
  }
  transducer_model ret(make_shared<transducer_variant>(generic_stacked_bidirectional_rnn_model(cells, dropout_rate)));
  ret.rename(name);
  return ret;
}

bidirectional_rnn_structure_t tg::bidirectional_rnn_structure;

transducer_model dropout_structure_t::initialize(float dropout_rate) {
  return (transducer_model)std::make_shared<transducer_variant>(dropout_model(dropout_rate));
}

dropout_structure_t tg::dropout_structure;

transducer_model axis_synchronized_dropout_structure_t::initialize(float dropout_rate,
                                                                   std::unordered_set<unsigned long> synchronized_axes) {
  return (transducer_model)std::make_shared<transducer_variant>(axis_synchronized_dropout_model(dropout_rate, move(synchronized_axes)));
}

axis_synchronized_dropout_structure_t tg::axis_synchronized_dropout_structure;


transducer_model
symbolic_embedding_table_structure_t::initialize(unsigned long embedding_size, const std::vector<symbol_t>& vocab) {
  return transducer_model([&](const value_placeholder& x) {
    auto dict = dict_structure.initialize(vocab);
    auto embedding_table = embedding_table_structure.initialize(embedding_size, vocab.size() + 1);
    return embedding_table(dict(x));
  }).rename("symbolic_embedding_table");
}

transducer_model
symbolic_embedding_table_structure_t::initialize(unsigned long embedding_size, const std::unordered_map<symbol_t,
  std::vector<float>>& pretrained_embeddings) {
  std::vector<symbol_t> vocab;
  std::vector<std::vector<float>> pretrained_embeddings_vec;
  vocab.reserve(pretrained_embeddings.size());
  pretrained_embeddings_vec.reserve(pretrained_embeddings.size());
  for(auto&& [symbol, embedding]:pretrained_embeddings) {
    vocab.push_back(symbol);
    pretrained_embeddings_vec.push_back(embedding);
  }

  pretrained_embeddings_vec.emplace_back(); // additional entry for <unk> token

  return transducer_model([&](const value_placeholder& x) {
    auto dict = dict_structure.initialize(vocab);
    auto embedding_table = embedding_table_structure.initialize(embedding_size, pretrained_embeddings_vec);
    return embedding_table(dict(x));
  }).rename("symbolic_embedding_table");

}

symbolic_embedding_table_structure_t tg::symbolic_embedding_table_structure;

pair<transducer_model, transducer_model> classify_from_logits_structure_t::initialize(unsigned long num_labels) {
  return make_pair(transducer_model([&](const value_placeholder& logits) {
    return max_index_of_tensor1d(logits);
  }).rename("readout"), transducer_model([&](const value_placeholder& logits, const value_placeholder& oracle) {
    return pickneglogsoftmax(logits, oracle);
  }).rename("readout_loss"));
}

classify_from_logits_structure_t tg::classify_from_logits_structure;

std::pair<transducer_model, transducer_model>
symbolic_classify_from_logits_structure_t::initialize(const std::vector<symbol_t>& labels) {
  auto dict = dict_structure.initialize(labels);
  auto&& [readout, readout_loss_fn] = classify_from_logits_structure.initialize(labels.size());

  return make_pair(transducer_model([&, readout(readout)](const value_placeholder& logits) {
    return dict(readout(logits));
  }).rename("symbolic_readout"), transducer_model([&, readout_loss_fn(readout_loss_fn)](const value_placeholder& logits, const value_placeholder& oracle) {
    return readout_loss_fn(logits, dict(oracle));
  }).rename("symbolic_readout_loss"));
}

symbolic_classify_from_logits_structure_t tg::symbolic_classify_from_logits_structure;
