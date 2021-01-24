//
// Created by Dekai WU and YAN Yuchen on 20200627.
//

#include "boolean_operations_impl.hpp"
#include "dynet_computation_graph.hpp"
#include <dynet/dynet.h>

using namespace std;
using namespace tg;

std::string tg::lazy_ifelse_op::default_name() const {
  return "if_";
}


value_t lazy_ifelse_op::lazy_transduce(const std::function<value_t()>& in0,
                                  const std::function<value_t()>& in1,
                                  const std::function<value_t()>& in2) {
  auto&& in0_val = in0();
  return in0_val.visit([&](auto&& cond)->value_t {
    constexpr auto Cond = value_t::static_type_info<decltype(cond)>();
    if constexpr (Cond.is_any_scalar) {
      if(cond) {
        return in1();
      }
      else {
        return in2();
      }
    }
    stringstream ss;
    ss << "Cannot apply lazy_ifelse on condition of type " << in0_val.type_name();
    throw_with_nested(runtime_error(ss.str()));
  });
}

tg::value_t tg::lt_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y) -> value_t {
    constexpr auto X = value_t::static_type_info<decltype(x)>();
    constexpr auto Y = value_t::static_type_info<decltype(y)>();

    if constexpr (X.is_any_tensor || Y.is_any_tensor) {
      return value_t(in0.as_symbolic_tensor() < in1.as_symbolic_tensor());
    }
    if constexpr (X.is_any_scalar && Y.is_any_scalar) {
      return value_t(x < y);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(runtime_error(ss.str()));
  }, in0, in1);
}

std::string tg::lt_op::default_name() const {
  return "operator<";
}

tg::value_t tg::gt_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y) -> value_t {
    constexpr auto X = value_t::static_type_info<decltype(x)>();
    constexpr auto Y = value_t::static_type_info<decltype(y)>();
    if constexpr (X.is_any_tensor || Y.is_any_tensor) {
      return value_t(in0.as_symbolic_tensor() > in1.as_symbolic_tensor());
    }
    if constexpr (X.is_any_scalar && Y.is_any_scalar) {
      return value_t(x > y);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(runtime_error(ss.str()));
  }, in0, in1);
}

std::string tg::gt_op::default_name() const {
  return "operator>";
}


tg::value_t tg::eq_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y) -> value_t {
    constexpr auto X = value_t::static_type_info<decltype(x)>();
    constexpr auto Y = value_t::static_type_info<decltype(y)>();
    if constexpr (X.is_any_tensor && Y.is_any_tensor) {
      return value_t(dynet::cwiseEq(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
    }
    if constexpr (X.is_any_scalar && Y.is_any_scalar) {
      return value_t(x == y);
    }
    if constexpr (X.is_symbol && Y.is_symbol) {
      return value_t(x == y);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(runtime_error(ss.str()));
  }, in0, in1);
}

std::string tg::eq_op::default_name() const {
  return "operator==";
}

tg::value_t tg::ne_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y) -> value_t {
    constexpr auto X = value_t::static_type_info<decltype(x)>();
    constexpr auto Y = value_t::static_type_info<decltype(y)>();
    if constexpr (X.is_any_tensor || Y.is_any_tensor) {
      return value_t(dynet::cwiseNe(in0.as_symbolic_tensor(), in1.as_symbolic_tensor()));
    }
    if constexpr (X.is_any_scalar && Y.is_any_scalar) {
      return value_t(x != y);
    }
    if constexpr (X.is_symbol && Y.is_symbol) {
      return value_t(x != y);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(runtime_error(ss.str()));
  }, in0, in1);
}

std::string tg::ne_op::default_name() const {
  return "operator!=";
}


std::string tg::logical_and_op::default_name() const {
  return "operator&&";
}

tg::value_t tg::logical_and_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y) -> value_t {
    constexpr auto X = value_t::static_type_info<decltype(x)>();
    constexpr auto Y = value_t::static_type_info<decltype(y)>();
    if constexpr (X.is_any_tensor || Y.is_any_tensor) {
      return value_t(in0.as_symbolic_tensor() && in1.as_symbolic_tensor());
    }
    if constexpr (X.is_any_scalar && Y.is_any_scalar) {
      return value_t(x && y);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(runtime_error(ss.str()));
  }, in0, in1);
}


std::string tg::logical_or_op::default_name() const {
  return "operator||";
}

tg::value_t tg::logical_or_op::transduce(const tg::value_t& in0, const tg::value_t& in1) {
  return value_t::visit_many([&](auto&& x, auto&& y) -> value_t {
    constexpr auto X = value_t::static_type_info<decltype(x)>();
    constexpr auto Y = value_t::static_type_info<decltype(y)>();
    if constexpr (X.is_any_tensor && Y.is_any_tensor) {
      return value_t(in0.as_symbolic_tensor() || in1.as_symbolic_tensor());
    }
    if constexpr (X.is_any_scalar && Y.is_any_scalar) {
      return value_t(x || y);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name() << " and "<<in1.type_name();
    throw_with_nested(runtime_error(ss.str()));
  }, in0, in1);
}

tg::value_t tg::to_boolean_op::transduce(const tg::value_t& in0) {
  return in0.visit([&](auto&& v) -> value_t {
    constexpr auto V = value_t::static_type_info<decltype(v)>();
    if constexpr (V.is_any_tensor) {
      return value_t(dynet::to_boolean(in0.as_symbolic_tensor()));
    }
    else if constexpr (V.is_any_scalar) {
      return value_t(v > 0);
    }
    stringstream ss;
    ss << "Cannot apply " << default_name() << " on operand of type " << in0.type_name();
    throw_with_nested(runtime_error(ss.str()));
  });
}

std::string tg::to_boolean_op::default_name() const {
  return "to_boolean";
}

tg::value_t tg::logical_not_op::transduce(const tg::value_t& in0) {
  return value_t(!in0.as_symbolic_tensor());
}

std::string tg::logical_not_op::default_name() const {
  return "operator!";
}

value_t eager_ifelse_op::transduce(const value_t& in0, const value_t& in1, const value_t& in2) {
  return in0.visit([&](auto&& cond)->value_t {
    constexpr auto Cond = value_t::static_type_info<decltype(cond)>();
    if constexpr (Cond.is_any_scalar) {
      return cond ? in1 : in2;
    }
    stringstream ss;
    ss << "Cannot apply eager_ifelse on condition of type " << in0.type_name();
    throw_with_nested(runtime_error(ss.str()));
  });
}

string eager_ifelse_op::default_name() const {
  return "eager_ifelse";
}

value_t soft_ifelse_op::transduce(const value_t& in0, const value_t& in1, const value_t& in2) {
  return in0.visit([&](auto&& cond)->value_t {
    constexpr auto Cond = value_t::static_type_info<decltype(cond)>();
    if constexpr (Cond.is_any_tensor) {
      return value_t(dynet::soft_ifelse(in0.as_symbolic_tensor(), in1.as_symbolic_tensor(), in2.as_symbolic_tensor()));
    }
    else if constexpr (Cond.is_any_scalar) {
      return cond ? in1 : in2;
    }
    stringstream ss;
    ss << "Cannot apply soft_ifelse on condition of type " << in0.type_name();
    throw_with_nested(runtime_error(ss.str()));
  });
}

string soft_ifelse_op::default_name() const {
  return "soft_ifelse";
}
