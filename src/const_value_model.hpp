//
// Created by Dekai WU and YAN Yuchen on 20200426.
//

/// \cond SHOW_INTERNAL_IMPL

#ifndef LEGO_CONST_VALUE_MODEL_HPP
#define LEGO_CONST_VALUE_MODEL_HPP

#include "include/transducer_typed_value.hpp"

namespace tg {
  class const_value_model {
    value_t v;
  public:
    template<typename Archive>
    void serialize(Archive& ar) {
      ar(v);
    }

    const_value_model() = default;
    const_value_model(const const_value_model&) = default;
    const_value_model(const_value_model&&) noexcept = default;
    const_value_model& operator=(const const_value_model&) = default;
    const_value_model& operator=(const_value_model&&) noexcept = default;
    const_value_model(value_t v);

    std::string default_name() const;

    value_t transduce();

  };
}


#endif
