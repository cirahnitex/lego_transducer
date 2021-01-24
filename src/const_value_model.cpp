//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

#include "const_value_model.hpp"


tg::value_t tg::const_value_model::transduce() {
  return v;
}

tg::const_value_model::const_value_model(tg::value_t v) : v(std::move(v)) {}

std::string tg::const_value_model::default_name() const {
  std::stringstream ss;
  ss << v;
  return ss.str();
}
