//
// Created by Dekai WU and YAN Yuchen on 20201124.
//

#include "include/value_placeholder.hpp"
#include "include/transducer_model.hpp"
#include "include/transducer_instance.hpp"
#include "include/transducer_dataset.hpp"
#include "list_operations_impl.hpp"
#include "const_value_model.hpp"
#include "lambda_transducer_model.hpp"
#include "transducer_variant.hpp"
#include <dynet/dynet.h>
#include "dynet_computation_graph.hpp"
using namespace tg;
using namespace std;


value_placeholder value_placeholder::operator[](const value_placeholder& idx) const {
  static auto model = transducer_model(make_shared<transducer_variant>(list_select_op()));
  return model(*this, idx);
}

value_placeholder value_placeholder::operator[](long idx) const {
  return operator[](value_placeholder::constant(idx));
}


value_placeholder value_placeholder::zeros(const tensor_shape_t& shape) {
  return value_placeholder::constant(tensor_t::zeros(shape));
}

value_placeholder value_placeholder::ones(const tg::tensor_shape_t& shape) {
  return value_placeholder::constant(tensor_t::ones(shape));
}

value_placeholder value_placeholder::constant(value_t x) {
  return lambda_transducer_model_construction_guard::top()->make_value_placeholder_from_constant(std::move(x));
}

bool tg::value_placeholder::valid() const {
  return owner_nesting_depth != -1;
}

namespace tg {
  std::ostream& operator<<(std::ostream& os, const value_placeholder& vp) {
    return os << "[value placeholder] idx="<<vp.value_idx<< ", depth="<<vp.owner_nesting_depth;
  }
}
