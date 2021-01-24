//
// Created by Dekai WU and YAN Yuchen on 20200511.
//

#include "include/lego_io_operations.hpp"
#include "io_operations_impl.hpp"
#include "transducer_variant.hpp"
#include "include/transducer_model.hpp"

using namespace tg;
using namespace std;

value_placeholder tg::trace(const value_placeholder& x, const string& label) {
  transducer_model model(std::make_shared<transducer_variant>(trace_op(label)));
  return model(x);
}

namespace tg {
  transducer_model identity(std::make_shared<transducer_variant>(identity_op()));
}
