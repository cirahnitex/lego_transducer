//
// Created by Dekai WU and YAN Yuchen on 20200423.
//

#include <fstream>
#include <utility>
#include "include/transducer_model.hpp"
#include "include/transducer_instance.hpp"
#include "include/transducer_dataset.hpp"
#include "include/lego_guard.hpp"
#include "list_operations_impl.hpp"
#include "const_value_model.hpp"
#include "dynet_computation_graph.hpp"
#include "tensor_operations_impl.hpp"
#include "lambda_transducer_model.hpp"
#include "tbd_transducer.hpp"
#include "transducer_variant.hpp"
#include "composed_transducer_model.hpp"

using namespace std;
using namespace tg;

tg::transducer_model::transducer_model(std::shared_ptr<transducer_variant> impl) : impl(move(impl)) {

}

tg::transducer_model::transducer_model(tg::value_t v) : impl(
  make_shared<transducer_variant>(const_value_model(move(v)))) {}


bool tg::transducer_model::valid() const {
  return (bool) impl;
}


transducer_instance tg::transducer_model::instantiate() const {
  return transducer_instance(impl);
}

std::string transducer_model::name() const {
  return impl->name();
}


transducer_model& transducer_model::rename(const std::string& name) {
  impl->rename(name);
  return *this;
}

transducer_model transducer_model::find_transducer_by_name(const std::string& name) {
  auto ret = impl->find_transducer_by_name(name);
  if(!ret) throw std::runtime_error("Transducer not found given name: " + name);
  return transducer_model(ret);
}

value_placeholder transducer_model::apply(const std::vector<value_placeholder>& input_placeholders) const {
  if(std::holds_alternative<list_op>(impl->v)) {
    return lambda_transducer_model_construction_guard::top()->make_value_placeholder_from_make_list(input_placeholders);
  }
  switch(input_placeholders.size()) {
    case 0:
      return lambda_transducer_model_construction_guard::top()->make_value_placeholder_from_transducing(impl);
    case 1:
      return operator()(input_placeholders[0]);
    case 2:
      return operator()(input_placeholders[0], input_placeholders[1]);
    case 3:
      return operator()(input_placeholders[0], input_placeholders[1], input_placeholders[2]);
    case 4:
      return operator()(input_placeholders[0], input_placeholders[1], input_placeholders[2], input_placeholders[3]);
    case 5:
      return operator()(input_placeholders[0], input_placeholders[1], input_placeholders[2], input_placeholders[3], input_placeholders[4]);
    case 6:
      return operator()(input_placeholders[0], input_placeholders[1], input_placeholders[2], input_placeholders[3], input_placeholders[4], input_placeholders[5]);
    case 7:
      return operator()(input_placeholders[0], input_placeholders[1], input_placeholders[2], input_placeholders[3], input_placeholders[4], input_placeholders[5], input_placeholders[6]);
    case 8:
      return operator()(input_placeholders[0], input_placeholders[1], input_placeholders[2], input_placeholders[3], input_placeholders[4], input_placeholders[5], input_placeholders[6], input_placeholders[7]);
    default:
      throw std::runtime_error("Arity > 8 is not supported");
  }
}

template<typename... T>
value_placeholder transducer_model::operator()(const value_placeholder& in0, const T& ...ins) const {
  constexpr auto argc = sizeof...(ins) + 1;
  if (!impl) {
    impl = make_shared<transducer_variant>(tbd_transducer());
  }
  else if(std::holds_alternative<list_op>(impl->v)) {
    return lambda_transducer_model_construction_guard::top()->make_value_placeholder_from_make_list(std::vector<value_placeholder>{in0, ins...});
  }
  else if (!impl->is_arity(argc)) {
    stringstream ss;
    ss << "Transducer "<<impl->name()<<" does not take "<<argc <<" arguments.";
    throw_with_nested(std::runtime_error(ss.str()));
  }

  return lambda_transducer_model_construction_guard::top()->make_value_placeholder_from_transducing(impl, in0, ins...);
}

template value_placeholder transducer_model::operator()(const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&, const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&) const;
template value_placeholder transducer_model::operator()(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&) const;

const std::shared_ptr<transducer_variant>& transducer_model::_get_impl() const {
  if (!impl) impl = make_shared<transducer_variant>(tbd_transducer());
  return impl;
}


value_t transducer_model::apply(const std::vector<value_t>& args) const {
  return instantiate().apply(args);
}

transducer_model::transducer_model(const std::function<value_placeholder(const value_placeholder&)>& fn_unary) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<1>(fn_unary))) {

}

transducer_model::transducer_model(
  const std::function<value_placeholder(const value_placeholder&, const value_placeholder&)>& binary_fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<2>(binary_fn))) {

}

transducer_model::transducer_model(
  const std::function<value_placeholder(const value_placeholder&, const value_placeholder&,
                                       const value_placeholder&)>& ternary_fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<3>(ternary_fn))) {

}

transducer_model::transducer_model(const std::function<value_placeholder(
  const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<4>(fn))) {

}

transducer_model::transducer_model(const std::function<value_placeholder(
  const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&,
  const value_placeholder&)>& fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<5>(fn))) {

}

transducer_model::transducer_model(const std::function<value_placeholder(
  const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&,
  const value_placeholder&, const value_placeholder&)>& fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<6>(fn))) {

}

transducer_model::transducer_model(const std::function<value_placeholder(
  const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&,
  const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<7>(fn))) {

}

transducer_model::transducer_model(const std::function<value_placeholder(
  const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&,
  const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&)>& fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<8>(fn))) {

}

transducer_model::transducer_model(const std::function<value_placeholder()>& fn) :
  transducer_model(std::make_shared<transducer_variant>(lambda_transducer_model::from_lambda_fn<0>(fn))) {

}


bool transducer_model::is_arity(unsigned long arity) const {
  if(!impl) {
    throw std::runtime_error("Cannot call is_arity() on an empty transducer");
  }
  return impl->is_arity(arity);
}

void transducer_model::save_to_stream_impl(std::ostream& os, const std::vector<transducer_model>& models) {
  cereal::BinaryOutputArchive oa(os);
  oa << models;
}

void transducer_model::load_from_stream_impl(std::istream& is, std::vector<transducer_model>& models) {
  cereal::BinaryInputArchive ia(is);
  ia >> models;
}

transducer_model& transducer_model::operator=(const tg::transducer_model& x) {
  if (!impl) {
    impl = x.impl;
    return *this;
  }
  impl->visit([&](auto&& v) {
    using T = decay_t<decltype(v)>;
    if constexpr (is_same_v<T, tbd_transducer>) {
      *impl = *x.impl;
    } else {
      impl = x.impl;
    }
  });
  return *this;
}

transducer_model& transducer_model::operator=(tg::transducer_model&& x) noexcept {
  if (!impl) {
    impl = move(x.impl);
    return *this;
  }
  impl->visit([&](auto&& v) {
    using T = decay_t<decltype(v)>;
    if constexpr (is_same_v<T, tbd_transducer>) {
      *impl = move(*x.impl);
    } else {
      impl = move(x.impl);
    }
  });
  return *this;
}

std::vector<value_t> transducer_model::batch_transduce(const std::shared_ptr<const transducer_dataset>& batch) const {
  return instantiate().batch_apply(batch);
}

template<typename Archive>
void transducer_model::serialize(Archive& ar) {
  ar(impl);
}


template void transducer_model::serialize<cereal::BinaryOutputArchive>(cereal::BinaryOutputArchive&);

template void transducer_model::serialize<cereal::BinaryInputArchive>(cereal::BinaryInputArchive&);


transducer_model tg::compose_impl(std::initializer_list<std::shared_ptr<transducer_variant>> pieces) {
  return transducer_model(std::make_shared<transducer_variant>(composed_transducer_model::compose_from(pieces)));
}
