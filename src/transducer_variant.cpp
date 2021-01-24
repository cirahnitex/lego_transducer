//
// Created by Dekai WU and YAN Yuchen on 20200716.
//

#include "transducer_variant.hpp"
#include "lambda_transducer_model.hpp"
using namespace tg;
using namespace std;


std::shared_ptr<transducer_variant> transducer_variant::find_transducer_by_name(const string& name) {
  if(this->name() == name) return shared_from_this();
  return visit([&name](auto&& t)->std::shared_ptr<transducer_variant> {
    if constexpr (has_nested_transducers_v<decltype(t)>) {
      for(auto&& child: t.nested_transducers()) {
        auto ret = child->find_transducer_by_name(name);
        if(ret) return ret;
      }
    }
    return nullptr;
  });
}

std::vector<std::shared_ptr<transducer_variant>> transducer_variant::nested_transducers() {
  return visit([&](auto&& t)->vector<shared_ptr<transducer_variant>> {
    using T = std::decay_t<decltype(t)>;
    if constexpr (has_nested_transducers_v<T>) {
      return t.nested_transducers();
    }
    return {};
  });
}


std::string transducer_variant::name() const {
  if(user_defined_display_name.empty()) return this->visit([&](auto&& v)->std::string {
    return v.default_name();
  });
  return user_defined_display_name;
}

void transducer_variant::rename(const string& name) {
  user_defined_display_name = name;
}

template<typename ...Args>
value_t transducer_variant::transduce(const Args& ...args) {
  constexpr auto argc = sizeof...(args);
  try {
    return std::visit([&](auto&& t)->value_t {
      constexpr auto traits = transducer_traits<decltype(t)>();
      if constexpr (!get<argc>(traits.static_arities)) {
        throw std::runtime_error("Argument count error");
      }
      else if constexpr (traits.is_lazy) {
        auto make_getter = [](auto&& x){
          return [&](){return x;};
        };
        return t.lazy_transduce(make_getter(args)...);
      }
      else {
        return t.transduce(args...);
      }
    }, v);
  }
  catch(...) {
    std::throw_with_nested(std::runtime_error("In " + name()));
  }
}

template value_t transducer_variant::transduce();
template value_t transducer_variant::transduce(const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&, const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&, const value_t&, const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);
template value_t transducer_variant::transduce(const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&, const value_t&);

template<typename ...Args>
value_t transducer_variant::transduce_placeholder(const Args& ...args) {
  constexpr auto argc = sizeof...(args);

  try {
    return std::visit([&](auto&& t)->value_t {
      constexpr auto traits = transducer_traits<decltype(t)>();
      if constexpr (!get<argc>(traits.static_arities)) {
        throw std::runtime_error("Argument count error");
      }
      else if constexpr (traits.is_lazy) {
        auto make_getter = [](auto&& x) {
          return [&](){return lambda_transducer_model::evaluate_value_placeholder(x);};
        };
        return t.lazy_transduce(make_getter(args)...);
      }
      else {
        return t.transduce(lambda_transducer_model::evaluate_value_placeholder(args)...);
      }
    }, v);
  }
  catch(...) {
    std::throw_with_nested(std::runtime_error("In " + name()));
  }
}

template value_t transducer_variant::transduce_placeholder();
template value_t transducer_variant::transduce_placeholder(const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&);
template value_t transducer_variant::transduce_placeholder(const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&, const value_placeholder&);

value_t transducer_variant::_apply(const std::vector<value_t>& ins) {
  if(std::holds_alternative<list_op>(v)) {
    return value_t(ins);
  }
  switch(ins.size()) {
    case 0:
      return transduce();
    case 1:
      return transduce(ins[0]);
    case 2:
      return transduce(ins[0], ins[1]);
    case 3:
      return transduce(ins[0], ins[1], ins[2]);
    case 4:
      return transduce(ins[0], ins[1], ins[2], ins[3]);
    case 5:
      return transduce(ins[0], ins[1], ins[2], ins[3], ins[4]);
    case 6:
      return transduce(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5]);
    case 7:
      return transduce(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6]);
    case 8:
      return transduce(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7]);
    default:
      throw std::runtime_error("arity > 8 is not supported");
  }
}

bool transducer_variant::is_arity(unsigned long arity) const {
  return std::visit([&](auto&& t)->bool {
    using transducer_T = std::decay_t<decltype(t)>;
    constexpr auto traits = transducer_traits<transducer_T>();
    if constexpr (traits.has_dynamic_arity) {
      return t.is_arity(arity);
    }
    else if constexpr (std::is_same_v<transducer_T, list_op>) {
      return true;
    }
    else {
      switch(arity) {
        case 0:
          return get<0>(traits.static_arities);
        case 1:
          return get<1>(traits.static_arities);
        case 2:
          return get<2>(traits.static_arities);
        case 3:
          return get<3>(traits.static_arities);
        case 4:
          return get<4>(traits.static_arities);
        case 5:
          return get<5>(traits.static_arities);
        case 6:
          return get<6>(traits.static_arities);
        case 7:
          return get<7>(traits.static_arities);
        case 8:
          return get<8>(traits.static_arities);
        default:
          return false;
      }
    }
  }, v);
}
