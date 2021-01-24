//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_SYMBOL_ID_CONVERTER_MODEL_HPP
#define LEGO_SYMBOL_ID_CONVERTER_MODEL_HPP

#include "include/transducer_typed_value.hpp"

namespace tg {
  class dict_model {
    std::string unk_token_m;
    std::vector<std::string> vocab_m;
    std::unordered_map<std::string, unsigned long> reverse_index_m;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(unk_token_m, vocab_m, reverse_index_m);
    }
    dict_model() = default;
    dict_model(const dict_model&) = default;
    dict_model(dict_model&&) noexcept = default;
    dict_model& operator=(const dict_model&) = default;
    dict_model& operator=(dict_model&&) noexcept = default;
    explicit dict_model(std::vector<std::string> vocab, std::string unk_token="<unk>");

    value_t transduce(const value_t& in0);

    std::string default_name() const;
  };

}


#endif //LEGO_SYMBOL_ID_CONVERTER_MODEL_HPP
