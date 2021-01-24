//
// Created by Dekai WU and YAN Yuchen on 20200511.
//

#include "io_operations_impl.hpp"
using namespace std;
tg::value_t tg::trace_op::transduce(const tg::value_t& in0) {
  if(prefix_m.empty()) {
    cout << in0 << endl;
  }
  else {
    cout << prefix_m << " " << in0 << endl;
  }
  return in0;
}

string tg::trace_op::default_name() const {
  return "print";
}

tg::trace_op::trace_op(std::string prefix): prefix_m(move(prefix)) {

}

tg::value_t tg::identity_op::transduce(const tg::value_t& in0) {
  return in0;
}

std::string tg::identity_op::default_name() const {
  return "identity";
}
