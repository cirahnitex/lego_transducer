//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_EMBEDDING_TABLE_MODEL_HPP
#define LEGO_EMBEDDING_TABLE_MODEL_HPP

#include "include/transducer_typed_value.hpp"
#include "backprop_trainable_parameter.hpp"

namespace tg {
  class embedding_table_model {
    unsigned long num_entries_m{};
    unsigned long embedding_size_m{};
    backprop_trainable_lookup_parameter lookup_table_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(num_entries_m, embedding_size_m, lookup_table_m);
    }
    embedding_table_model() = default;
    embedding_table_model(const embedding_table_model&) = default;
    embedding_table_model(embedding_table_model&&) noexcept = default;
    embedding_table_model& operator=(const embedding_table_model&) = default;
    embedding_table_model& operator=(embedding_table_model&&) noexcept = default;
    embedding_table_model(unsigned long embedding_size, unsigned long num_entries);

    /**
     * Initialize the embedding table with pre-trained embeddings.
     * The pre-trained embeddings supplied can have missing entries
     * in which case will fallback to random initialization for those missing entries.
     *
     * \param embedding_size the embedding size
     * \param pretrained_embeddings the pre-trained embedding table. Pass an empty vector for a missing entry.
     */
    embedding_table_model(unsigned long embedding_size, const std::vector<std::vector<float>>& pretrained_embeddings);

    value_t transduce(const value_t& in0);

    std::string default_name() const;


  };
}


#endif
