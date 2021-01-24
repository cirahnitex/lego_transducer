//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

#include "symbol_id_converter_model.hpp"
#include "include/transducer_typed_value.hpp"

using namespace std;

tg::value_t tg::dict_model::transduce(const tg::value_t& in0) {
  if (in0.is_symbol()) {
    // convert symbol to ID if input is of symbol type
    auto ret = reverse_index_m.find(in0.as_symbol());
    if (ret == reverse_index_m.end()) return value_t(vocab_m.size());
    return value_t(ret->second);
  } else {
    // convert ID to symbol if input is of type scalar
    unsigned long index = in0.as_integer();
    if (index >= vocab_m.size()) return value_t(unk_token_m);
    return value_t(vocab_m[index]);
  }
}

string tg::dict_model::default_name() const {
  return "symbol_id_converter[size=" + std::to_string(vocab_m.size()) + "]";
}

tg::dict_model::dict_model(std::vector<std::string> _vocab, std::string unk_token) : unk_token_m(move(unk_token)),
                                                                                     vocab_m(move(_vocab)) {
  for (unsigned long i = 0; i < vocab_m.size(); ++i) {
    auto&& token = vocab_m[i];
    if (reverse_index_m.find(token) != reverse_index_m.end()) throw std::runtime_error(
        "failed to construct symbol_id_converter: Supplied vocabulary contains duplicated tokens. Duplicated token: " +
        token);
    reverse_index_m[token] = i;
  }
}
