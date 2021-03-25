//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#include "include/transducer_typed_value.hpp"
#include "dynet_computation_graph.hpp"
#include <chrono>
#include "include/lego_guard.hpp"

using namespace std;
using namespace tg;

tg::value_t::value_t() : v() {

}

tg::value_t::value_t(float x) : v(make_shared<varient_t>(x)) {

}

tg::value_t::value_t(double x) : v(make_shared<varient_t>((scalar_t)x)) {

}

tg::value_t::value_t(const std::string& x) : v(make_shared<varient_t>(x)) {

}

void tg::block_nan_or_inf(float value) {
  switch(fpclassify(value)) {
    case FP_INFINITE:
      throw nan_or_inf_exception("Value Inf encountered");
    case FP_NAN:
      throw nan_or_inf_exception("Value NaN encountered");
    default:
      break;
  }
}

void tg::block_nan_or_inf(const std::vector<float>& values) {
  for(auto&& v:values) {
    block_nan_or_inf(v);
  }
}

tg::value_t::value_t(const symbolic_tensor_t& x) : v(make_shared<varient_t>(x)) {
  if(immediate_computation_guard::is_guarded()) {
    block_nan_or_inf(dynet::as_vector(dynet_computation_graph::p()->forward(x)));
  }
}

tg::value_t::value_t(const std::vector<value_t>& xs) : v(make_shared<varient_t>(xs)) {

}

value_t::value_t(std::vector<value_t>&& xs) : v(make_shared<varient_t>(move(xs))) {

}

value_t::value_t(const tensor_t& x) : v(make_shared<varient_t>(x)) {

}

value_t::value_t(tensor_t&& x) : v(make_shared<varient_t>(move(x))) {

}

long tg::value_t::as_integer() const {
  if(!v) throw_with_nested(std::runtime_error("Type mismatch. Expected integer, got " + type_name()));
  return std::visit([&](auto&& r)->long {
    using T = std::decay_t<decltype(r)>;
    if constexpr (is_same_v<T, long>) {
      return r;
    }
    else if constexpr (is_same_v<T, float>) {
      return (long)r;
    }
    else if constexpr (is_same_v<T, tensor_t>) {
      if (r.values.size() > 1)
        throw_with_nested(std::runtime_error("Cannot get integer from a tensor containing multiple values"));
      return (long)r.values[0];
    }
    else if constexpr (is_same_v<T, symbolic_tensor_t>) {
      if (r.is_stale()) throw_with_nested(std::runtime_error("Cannot get value from stale symbolic tensor"));
      if (r.dim().sum_dims() > 1) {
        throw_with_nested(std::runtime_error("Cannot get integer from a tensor containing multiple values"));
      }
      return (long)dynet::as_scalar(dynet_computation_graph::p()->forward(r));
    }
    else {
      throw_with_nested(std::runtime_error("Type mismatch. Expected integer, got " + type_name()));
    }
  }, *v);

}

float tg::value_t::as_float() const {
  if(!v) throw_with_nested(std::runtime_error("Type mismatch. Expected scalar, got " + type_name()));
  return std::visit([&](auto&& r)->float {
    using T = std::decay_t<decltype(r)>;
    if constexpr (is_same_v<T, long>) {
      return r;
    }
    else if constexpr (is_same_v<T, float>) {
      return r;
    }
    else if constexpr (is_same_v<T, tensor_t>) {
      if (r.values.size() > 1)
        throw_with_nested(std::runtime_error("Cannot get scalar from a tensor containing multiple values"));
      return r.values[0];
    }
    else if constexpr (is_same_v<T, symbolic_tensor_t>) {
      if (r.is_stale()) throw_with_nested(std::runtime_error("Cannot get value from stale symbolic tensor"));
      if (r.dim().sum_dims() > 1) {
        throw_with_nested(std::runtime_error("Cannot get scalar from a tensor containing multiple values"));
      }
      return dynet::as_scalar(dynet_computation_graph::p()->forward(r));
    }
    else {
      throw_with_nested(std::runtime_error("Type mismatch. Expected scalar, got " + type_name()));
    }
  }, *v);
}


const std::string& tg::value_t::as_symbol() const {
  if (is_symbol()) return get<string>(*v);
  throw_with_nested(std::runtime_error("Type mismatch. Expected symbol, got " + type_name()));
}


tg::symbolic_tensor_t tg::value_t::as_symbolic_tensor() const {
  if(!v) throw_with_nested(std::runtime_error("Type mismatch. Expected tensor, got " + type_name()));
  return std::visit([&](auto&& r)->symbolic_tensor_t {

    constexpr auto V = static_type_info<decltype(r)>();

    if constexpr (V.is_symbolic_tensor) {
      return r;
    }
    if constexpr (V.is_any_scalar) {
      return dynet::input(*dynet_computation_graph::p(), (float)r);
    }
    if constexpr (V.is_tensor) {

      return dynet::input(*dynet_computation_graph::p(), to_dynet_dim(r.shape), &r.values);
    }

    throw_with_nested(std::runtime_error("Type mismatch. Expected tensor, got " + type_name()));
  }, *v);
}

tensor_t value_t::as_tensor() const {
  if(!v) throw_with_nested(std::runtime_error("Type mismatch. Expected tensor, got " + type_name()));
  return std::visit([&](auto&& r)->tensor_t {
    constexpr auto V = static_type_info<decltype(r)>();
    if constexpr (V.is_any_scalar) {
      return tensor_t({(float)r}, {1});
    }
    else if constexpr (V.is_tensor) {
      return r;
    }
    else {
      throw_with_nested(std::runtime_error("Type mismatch. Expected tensor, got " + type_name()));
    }

  }, *v);
}

const std::vector<tg::value_t>& tg::value_t::as_list() const & {
  if (is_list()) return get<std::vector<value_t>>(*v);
  throw_with_nested(std::runtime_error("Type mismatch. Expected list, got " + type_name()));
}

list_t tg::value_t::as_list() const && {
  if (is_list()) return move(get<std::vector<value_t>>(*v));
  throw_with_nested(std::runtime_error("Type mismatch. Expected list, got " + type_name()));
}


bool tg::value_t::is_integer() const {
  return v && holds_alternative<long>(*v);
}

bool tg::value_t::is_float() const {
  return v && holds_alternative<float>(*v);
}

bool value_t::is_any_scalar() const {
  if(!v) return false;
  return this->visit([&](auto&& r)->bool{
    using T = decay_t<decltype(r)>;
    return static_type_info<T>().is_any_scalar;
  });
}

bool value_t::is_any_tensor() const {
  if(!v) return false;
  return this->visit([&](auto&& r)->bool{
    using T = decay_t<decltype(r)>;
    return static_type_info<T>().is_any_tensor;
  });
}

bool tg::value_t::is_symbol() const {
  return v && holds_alternative<string>(*v);
}

bool tg::value_t::is_tensor() const {
  return v && holds_alternative<tensor_t>(*v);
}

bool value_t::is_symbolic_tensor() const {
  return v && holds_alternative<symbolic_tensor_t>(*v);
}

bool tg::value_t::is_list() const {
  return v && holds_alternative<std::vector<value_t>>(*v);
}

bool tg::value_t::is_null() const {
  return !v;
}

std::ostream& tg::operator<<(std::ostream& os, const value_t& x) {
  if(x.is_null()) {
    return os << "#null";
  }
  return x.visit([&](auto&& v)->std::ostream& {
    constexpr auto traits = value_t::static_type_info<decltype(v)>();
    if constexpr (traits.is_symbol) {
      return os << "\"" << v << "\"";
    }
    else if constexpr (traits.is_symbolic_tensor) {
      if(v.is_stale()) return os << "<stale>";
      if (x.tensor_num_elements() > tensor_t::MAX_TENSOR_ELEMS_TO_PRINT)
        return os << "tensor(" << x.print_tensor_shape() << ")";
      auto dynet_tensor = dynet_computation_graph::p()->forward(v);

      if(dynet_tensor.d.batch_size() == 1) return os << dynet_tensor.v[0];

      return os << tensor_t::from_dynet_tensor(dynet_tensor);
    }
    else if constexpr (traits.is_tensor) {
      if (x.tensor_num_elements() > tensor_t::MAX_TENSOR_ELEMS_TO_PRINT)
        return os << "tensor(" << x.print_tensor_shape() << ")";

      if(x.tensor_num_elements() == 1) return os << v.values[0];

      return os << v;
    }
    else if constexpr (traits.is_list) {
      os << "(";
      bool first = true;
      for (auto&& item:v) {
        if (first) {
          first = false;
        } else {
          os << " ";
        }
        os << item;
      }
      return os << ")";
    }
    else if constexpr (traits.is_any_scalar) {
      return os << v;
    }
    throw_with_nested(std::runtime_error("unrecognized value_t type"));
  });

}

tensor_shape_t value_t::tensor_shape() const {

  if(!v) throw_with_nested(std::runtime_error("Cannot get tensor dim from value of type " + type_name()));

  return std::visit([&](auto&& r)->tensor_shape_t {
    using T = std::decay_t<decltype(r)>;

    if constexpr (is_same_v<T, symbolic_tensor_t>) {
      return from_dynet_dim(r.dim());
    }
    if constexpr (is_same_v<T, long>) {
      return {1};
    }
    else if constexpr (is_same_v<T, float>) {
      return {1};
    }
    else if constexpr (is_same_v<T, tensor_t>) {
      return r.shape;
    }
    else {
      throw_with_nested(std::runtime_error("Type mismatch. Expected tensor, got " + type_name()));
    }

  }, *v);
}

unsigned long value_t::tensor_rank() const {
  if(!v) throw_with_nested(std::runtime_error("Cannot get tensor dim from value of type " + type_name()));

  return std::visit([&](auto&& r)->unsigned long {
    using T = std::decay_t<decltype(r)>;

    if constexpr (is_same_v<T, symbolic_tensor_t>) {
      return r.dim().nd;
    }
    if constexpr (is_same_v<T, long>) {
      return 1;
    }
    else if constexpr (is_same_v<T, float>) {
      return 1;
    }
    else if constexpr (is_same_v<T, tensor_t>) {
      return r.shape.size();
    }
    else {
      throw_with_nested(std::runtime_error("Type mismatch. Expected tensor, got " + type_name()));
    }

  }, *v);
}

unsigned long value_t::tensor_num_elements() const {
  unsigned long ret = 1;
  for (auto&& dim:tensor_shape()) {
    ret *= dim;
  }
  return ret;
}

std::string value_t::print_tensor_shape() const {
  return tg::print_tensor_shape(tensor_shape());
}

std::string value_t::type_name() const {
  if (is_null()) {
    return "null";
  }
  if (is_symbol()) {
    return "symbol";
  }
  if (is_tensor()) {
    return "tensor";
  }
  if (is_symbolic_tensor()) {
    return "symbolic tensor";
  }
  if (is_list()) {
    return "list";
  }

  if (is_integer()) {
    return "integer";
  }

  if (is_float()) {
    return "float";
  }
  throw_with_nested(std::runtime_error("unrecognized value_t type"));
}


std::pair<value_t, value_t> value_t::as_pair() const {
  auto&& list = as_list();
  if (list.size() < 2)
    throw_with_nested(std::runtime_error("Cannot convert list of size " + std::to_string(list.size()) + " into  pair"));
  return make_pair(list[0], list[1]);
}

value_t value_t::select(long i) const {
  auto&& list = as_list();
  if (i >= (long)list.size() || i + (long)list.size() < 0) {
    stringstream ss;
    ss << "Out of range: cannot select index " << i << " from a list of size " << list.size();
    throw_with_nested(std::runtime_error(ss.str()));
  }
  return i >= 0 ? list.at(i) : list.at(list.size() + i);
}

value_t value_t::select(unsigned long i, unsigned long axis) const {
  if (is_symbolic_tensor()) {
    return value_t(dynet::pick(as_symbolic_tensor(), i, axis));
  }

  stringstream ss;
  ss << "Cannot invoke tensor select from value of type: " << type_name();
  throw_with_nested(std::runtime_error(ss.str()));
}

value_t value_t::select_many(const std::vector<unsigned long>& indices) const {
  vector<value_t> ret;
  ret.reserve(indices.size());
  for (auto&& i : indices) {
    ret.push_back(select((long)i));
  }
  return value_t(ret);
}

value_t value_t::select_many(const std::vector<long>& indices) const {
  vector<value_t> ret;
  ret.reserve(indices.size());
  for (auto&& i : indices) {
    ret.push_back(select(i));
  }
  return value_t(ret);
}

value_t value_t::select_many(const std::vector<unsigned long>& indices, unsigned long axis) const {
  if (is_symbolic_tensor()) {
    auto&& self = as_symbolic_tensor();
    vector<symbolic_tensor_t> pieces;
    pieces.reserve(indices.size());

    for (auto&& i:indices) {
      pieces.push_back(dynet::pick_range(self, i, i + 1, axis));
    }

    return value_t(dynet::concatenate(pieces, axis));
  }

  stringstream ss;
  ss << "Cannot invoke tensor select from value of type: " << type_name();
  throw_with_nested(std::runtime_error(ss.str()));
}

value_t value_t::slice(unsigned long start, unsigned long end) const {
  auto&& list = as_list();
  auto end_itr = end >= list.size() ? list.end() : (list.begin() + end);
  return value_t(vector<value_t>(list.begin() + start, end_itr));
}

value_t value_t::slice(unsigned long start, unsigned long end, unsigned long axis) const {
  if (is_symbolic_tensor()) {
    return value_t(dynet::pick_range(as_symbolic_tensor(), start, end, axis));
  }

  stringstream ss;
  ss << "Cannot invoke tensor select from value of type: " << type_name();
  throw_with_nested(std::runtime_error(ss.str()));
}

value_t value_t::operator[](long i) const {
  return select(i);
}




value_t::value_t(bool x): value_t(x ? (long)1.0 : (long)0.0) {

}

value_t::value_t(int x): value_t((long) x) {}

value_t::value_t(long x): v(make_shared<varient_t>((long)x)) {}

value_t::value_t(unsigned int x): value_t((long) x) {}

value_t::value_t(unsigned long x): value_t((long) x) {}

value_t::value_t(const char *x): value_t(string(x)) {}

value_t::value_t(nullptr_t):value_t() {

}


void value_t::evaluate() {

  // collect all nested symbolic tensors
  std::unordered_set<varient_t *> expr_holders;
  collect_nested_symbolic_tensors(expr_holders);

  if(expr_holders.empty()) return;

  // if there is only one symbolic tensor, evaluate it and we are done
  if(expr_holders.size() == 1) {
    for(auto&& var:expr_holders) {
      auto& expr = get<symbolic_tensor_t>(*var);
      auto dynet_tensor = dynet_computation_graph::p()->incremental_forward(expr);
      auto vec = dynet::as_vector(dynet_tensor);
      block_nan_or_inf(vec);
      *var = tensor_t(vec, from_dynet_dim(expr.dim()));
    }
    return;
  }

  // concatenate them into a big tensor
  vector<symbolic_tensor_t> exprs;
  exprs.reserve(expr_holders.size());
  for(auto&& var:expr_holders) {
    auto& expr = get<symbolic_tensor_t>(*var);
    exprs.push_back(dynet::reshape(expr, {expr.dim().size()}));
  }
  auto concatenated = dynet::concatenate(exprs, 0);

  // evaluate the concatenated tensor
  auto dynet_tensor = dynet_computation_graph::p()->incremental_forward(concatenated);
  auto concatenated_val = dynet::as_vector(dynet_tensor);
  block_nan_or_inf(concatenated_val);

  // reconstruct back each individual tensors by slicing
  // and replace the symbolic tensor with the reconstructed tensor
  auto offset = concatenated_val.begin();
  for(auto&& var:expr_holders) {
    auto& expr = get<symbolic_tensor_t>(*var);
    auto end =  offset + expr.dim().size();
    *var = tensor_t(vector<float>(offset, end), from_dynet_dim(expr.dim()));
    offset = end;
  }
}

void value_t::collect_nested_symbolic_tensors(std::unordered_set<varient_t *>& ret) {
  if(!v) return;
  visit([&](auto&& x) {
    constexpr auto traits = static_type_info<decltype(x)>();
    if constexpr (traits.is_symbolic_tensor) {
      ret.insert(&*v);
    }
    else if constexpr (traits.is_list) {
      for(auto&& child:x) {
        child.collect_nested_symbolic_tensors(ret);
      }
    }
  });
}

value_t::~value_t() {
}
