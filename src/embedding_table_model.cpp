//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

#include "embedding_table_model.hpp"
#include "include/transducer_typed_value.hpp"
#include "backprop_trainable_parameter.hpp"

using namespace std;
using namespace tg;

tg::value_t tg::embedding_table_model::transduce(const tg::value_t& in0) {
  unsigned long idx = in0.as_integer();
  if (idx >= num_entries_m) {
    stringstream ss;
    ss << "Failed to transduce in transducer " << default_name() << ": supplied index (" << idx
       << ") is greater than embedding table capacity (" << num_entries_m << ")";
    throw_with_nested(std::runtime_error(ss.str()));
  }
  return (tg::value_t) lookup_table_m.lookup_as_symbolic_tensor(in0.as_integer());
}

tg::embedding_table_model::embedding_table_model(unsigned long embedding_size, unsigned long num_entries) :
  num_entries_m(num_entries), embedding_size_m(embedding_size), lookup_table_m(num_entries, {embedding_size}) {

}

string tg::embedding_table_model::default_name() const {
  return "embedding_table";
}

embedding_table_model::embedding_table_model(unsigned long embedding_size,
                                             const std::vector<std::vector<float>>& pretrained_embeddings) :
                                             num_entries_m(pretrained_embeddings.size()), embedding_size_m(embedding_size),
                                             lookup_table_m(num_entries_m, {embedding_size}){
  for(unsigned long i=0; i<pretrained_embeddings.size(); ++i) {
    auto&& pretrained_embedding = pretrained_embeddings[i];
    if(pretrained_embedding.empty()) continue;
    if(pretrained_embedding.size() != embedding_size) {
      stringstream ss;
      ss << "Cannot initialize embedding table with pre-trained values. Expected embedding size "<<embedding_size<<", got "<< pretrained_embedding.size() << " at entry #"<< i;
      throw_with_nested(std::runtime_error(ss.str()));
    }
    lookup_table_m.set_value(i, pretrained_embedding);
  }
}

