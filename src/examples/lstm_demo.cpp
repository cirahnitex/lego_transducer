#include "../include/lego_transducer.hpp"
#include "../transducer_variant.hpp"
using namespace std;
using namespace tg;

int main() {
  lego_initialize();

  transducer_model emb = symbolic_embedding_table_structure.initialize(4, {"a","b","c"});
  transducer_model bilstm = bidirectional_rnn_structure.initialize(4,4,VANILLA_LSTM, 2);
  transducer_model biaffine = biaffine_structure.initialize(4,4,4);
  auto [ro, ro_loss] = symbolic_classify_from_logits_structure.initialize({"A", "B", "C", "D"});

  transducer_model compute_logits([&](const value_placeholder& xs) {
    auto embs = list_map(emb, xs);
    auto outs = bilstm(value_placeholder::constant(nullptr), embs)[0];
    auto first = embs[0];
    auto dropout = dropout_structure.initialize(0.2);
    auto logits = list_map(transducer_model([&](const value_placeholder& x) {
      return biaffine(dropout(x), first);
    }), outs);
    return logits;
  });

  transducer_model performance_component([&, ro(ro)](const value_placeholder& xs) {
    auto outputs = list_map(ro, compute_logits(xs));
    return outputs;
  });

  transducer_model loss_fn([&, ro_loss(ro_loss)](const value_placeholder& xs, const value_placeholder& ys) {
    auto losses = list_map(ro_loss, compute_logits(xs), ys);
    return list_sum(losses);
  });

  auto dataset = create_transducer_dataset(2);
  dataset->emplace_back(value_t::make_list("a","a"), value_t::make_list("A","A"));
  dataset->emplace_back(value_t::make_list("a","b"), value_t::make_list("A","B"));
  dataset->emplace_back(value_t::make_list("c","b", "a"), value_t::make_list("C","B", "A"));

  dataset->emplace_back(value_t::make_list("b","b", "c"), value_t::make_list("B","B", "C"));

  auto validation_set = create_transducer_dataset(2);
  validation_set->emplace_back(value_t::make_list("c","b", "b"), value_t::make_list("C","B", "B"));

  validation_set->emplace_back(value_t::make_list("a","b", "c"), value_t::make_list("A","B", "C"));


  adam_optimizer optimizer(0.01);
  optimizer.set_weight_decay(0.001);
  training_pipeline pipeline(&optimizer);
  pipeline.set_num_epochs(100);
  pipeline.set_num_workers(2);
  pipeline.add_new_best_listener([&]() {
    transducer_model::save_to_file("model.bin", performance_component, loss_fn);
  });
  pipeline.train_and_validate(loss_fn, dataset, validation_set);


  {
    auto test_dataset = create_transducer_dataset(1);
    for(auto&& datum:*dataset) {
      test_dataset->emplace_back(datum[0]);
    }
    cout << "predicting" << endl;
    for(auto&& pred:pipeline.transduce_many(performance_component, test_dataset)) {
      cout << pred << endl;
    }

    cout << "predicting without using pipeline" << endl;
    for(auto&& datum:*dataset) {
      cout << performance_component(datum[0]) << endl;
    }
  }


  {
    cout << "reload test" << endl;
    transducer_model perf, loss_fn;
    transducer_model::load_from_file("model.bin", perf,loss_fn);
    float sum_loss = 0;
    for(auto&& datum:*validation_set) {
      auto pred = loss_fn(datum[0], datum[1]);
      sum_loss += loss_fn(datum[0], datum[1]).as_float();
    }
    cout << "average loss = " << sum_loss / validation_set->size() << endl;
  }


  return 0;
}

