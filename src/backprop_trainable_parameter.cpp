//
// Created by Dekai WU and YAN Yuchen on 20200503.
//

#include "backprop_trainable_parameter.hpp"
#include "dynet_computation_graph.hpp"

using namespace tg;
using namespace std;

thread_local std::unordered_map<const backprop_trainable_parameter*, dynet::Expression> backprop_trainable_parameter::symbolic_tensor_cache;
thread_local std::unordered_map<const backprop_trainable_lookup_parameter*, std::vector<dynet::Expression>> backprop_trainable_lookup_parameter::symbolic_tensors_cache;
dynet::Expression backprop_trainable_parameter::as_symbolic_tensor() const {
  auto& ret = symbolic_tensor_cache[this];
  if (!ret.is_stale()) return ret;
  return ret = dynet::parameter(*dynet_computation_graph::p(), v);
}

tensor_t backprop_trainable_parameter::as_tensor() const {
  return tensor_t::from_dynet_tensor(*v.values());
}

backprop_trainable_parameter::backprop_trainable_parameter(const tensor_shape_t& dim) : backprop_trainable_parameter_base(), internal_pc_m(
  make_unique<dynet::ParameterCollection>()), v(internal_pc_m->add_parameters(to_dynet_dim(dim))) {

}

bool backprop_trainable_parameter::is_using_internal_pc() const {
  return (bool) internal_pc_m;
}

bool backprop_trainable_parameter::is_using_external_pc() const {
  return !internal_pc_m;
}

void backprop_trainable_parameter::use_external_pc(dynet::ParameterCollection& pc) {
  if(!operator bool()) return;
  if (is_using_external_pc()) {
    throw_with_nested(runtime_error("Cannot transfer trainable parameters to the trainer, because this parameter has already been managed by a trainer."));
  }
  auto new_v = pc.add_parameters(v.dim());
  new_v.set_value(dynet::as_vector(*v.values()));
  v = new_v;
  internal_pc_m = nullptr;
}

void backprop_trainable_parameter::use_internal_pc() {
  if(!operator bool()) return;
  if (is_using_internal_pc()) return;
  internal_pc_m = make_unique<dynet::ParameterCollection>();
  auto new_v = internal_pc_m->add_parameters(v.dim());
  new_v.set_value(dynet::as_vector(*v.values()));
  v = new_v;
}

backprop_trainable_parameter& backprop_trainable_parameter::operator=(const backprop_trainable_parameter& x) {
  internal_pc_m = make_unique<dynet::ParameterCollection>();
  v = internal_pc_m->add_parameters(x.v.dim());
  v.set_value(dynet::as_vector(*x.v.values()));
  return *this;
}

backprop_trainable_parameter::backprop_trainable_parameter(const backprop_trainable_parameter& x)
  : internal_pc_m(make_unique<dynet::ParameterCollection>()),
    v(internal_pc_m->add_parameters(x.v.dim())) {
  v.set_value(dynet::as_vector(*x.v.values()));
}

backprop_trainable_parameter::operator bool() const {
  return (bool) v.p;
}

backprop_trainable_parameter::~backprop_trainable_parameter() {
  symbolic_tensor_cache.erase(this);
}

dynet::Expression backprop_trainable_lookup_parameter::lookup_as_symbolic_tensor(unsigned long index) const {
  auto& table_cache = symbolic_tensors_cache[this];
  if(table_cache.empty()) table_cache.resize(num_entries());
  auto& expr_cache = table_cache[index];
  if (!expr_cache.is_stale()) return expr_cache;
  return expr_cache = dynet::lookup(*dynet_computation_graph::p(), v, index);
}

tensor_t backprop_trainable_lookup_parameter::lookup_as_tensor(unsigned long index) const {
  return tensor_t::from_dynet_tensor(v.values()->at(index));
}

bool backprop_trainable_lookup_parameter::is_using_internal_pc() const {
  return (bool) internal_pc_m;
}

bool backprop_trainable_lookup_parameter::is_using_external_pc() const {
  return !internal_pc_m;
}

void backprop_trainable_lookup_parameter::use_internal_pc() {
  if (is_using_internal_pc()) return;
  internal_pc_m = make_unique<dynet::ParameterCollection>();
  auto new_v = internal_pc_m->add_lookup_parameters(v.values()->size(), v.dim());
  for (unsigned long i = 0; i < v.values()->size(); ++i) {
    new_v.initialize(i, dynet::as_vector(v.values()->at(i)));
  }
  v = new_v;
}

void backprop_trainable_lookup_parameter::use_external_pc(dynet::ParameterCollection& pc) {
  auto new_v = pc.add_lookup_parameters(v.values()->size(), v.dim());
  for (unsigned long i = 0; i < v.values()->size(); ++i) {
    new_v.initialize(i, dynet::as_vector(v.values()->at(i)));
  }
  v = new_v;
  internal_pc_m = nullptr;
}

backprop_trainable_lookup_parameter::backprop_trainable_lookup_parameter(
  unsigned num_entries, const tg::tensor_shape_t& embedding_dim)
  : backprop_trainable_parameter_base(), internal_pc_m(make_unique<dynet::ParameterCollection>()),
    v(internal_pc_m->add_lookup_parameters(num_entries, to_dynet_dim(embedding_dim))) {
}

backprop_trainable_lookup_parameter::backprop_trainable_lookup_parameter(const backprop_trainable_lookup_parameter& x)
  : internal_pc_m(make_unique<dynet::ParameterCollection>()),
    v(internal_pc_m->add_lookup_parameters(x.num_entries(), x.v.dim())) {
  unsigned long num_entries = x.num_entries();
  for (unsigned long i = 0; i < num_entries; ++i) {
    v.initialize(i, dynet::as_vector(x.v.values()->at(i)));
  }
}

backprop_trainable_lookup_parameter&
backprop_trainable_lookup_parameter::operator=(const backprop_trainable_lookup_parameter& x) {
  unsigned long num_entries = x.num_entries();
  internal_pc_m = make_unique<dynet::ParameterCollection>();
  v = internal_pc_m->add_lookup_parameters(num_entries, x.v.dim());
  for (unsigned long i = 0; i < num_entries; ++i) {
    v.initialize(i, dynet::as_vector(x.v.values()->at(i)));
  }
  return *this;
}

unsigned long backprop_trainable_lookup_parameter::num_entries() const {
  return v.values()->size();
}

tensor_shape_t backprop_trainable_lookup_parameter::embedding_dim() const {
  return from_dynet_dim(v.dim());
}

backprop_trainable_lookup_parameter::operator bool() const {
  return (bool) v.p;
}

void backprop_trainable_lookup_parameter::set_value(unsigned long index, const std::vector<float>& value) {
  v.initialize(index, value);
}

backprop_trainable_lookup_parameter::~backprop_trainable_lookup_parameter() {
  symbolic_tensors_cache.erase(this);
}

thread_local std::unordered_set<backprop_trainable_parameter_base*> backprop_trainable_parameter_base::all_parameters{};
backprop_trainable_parameter_base::backprop_trainable_parameter_base():path(lego_param_naming_guard::v) {
  all_parameters.insert(this);

}

backprop_trainable_parameter_base::~backprop_trainable_parameter_base() {
  all_parameters.erase(this);
}

backprop_trainable_parameter_base::backprop_trainable_parameter_base(const backprop_trainable_parameter_base& x):path(x.path) {
  all_parameters.insert(this);
}

backprop_trainable_parameter_base::backprop_trainable_parameter_base(backprop_trainable_parameter_base&& x) noexcept :path(std::move(x.path)) {
  all_parameters.insert(this);
}
